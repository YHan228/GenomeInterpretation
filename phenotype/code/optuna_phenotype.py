"""
Optuna HPO for the sporulation CNN.

- Reuses dataset utilities from `sporulation/code/training.py` (no code duplication)
- Explores a compact but expressive CNN hyperparameter space
- Uses balanced accuracy on a fixed validation sampler as the objective
- Supports pruning per-epoch via Optuna pruners
- Designed to run on long genomic sequences with aggressive downsampling

Usage (local):
  python -u sporulation/optuna_sporulation.py \
    --tune \
    --study-name sporo_cnn_v1 \
    --n-trials 50 \
    --max-epochs 25 \
    --storage sqlite:///./optuna.db

Usage (SLURM, see slurm_scripts/run_optuna_sporulation.sh):
  sbatch --array=1-8 slurm_scripts/run_optuna_sporulation.sh
"""

import os
import sys
import argparse
import subprocess
import json as _json
import shutil
import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import balanced_accuracy_score

# Prefer local copies in phenotype/code; fallback to original sporulation package
_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE_DIR = os.path.join(_HERE)
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)
try:
    from training import FastaDataset, one_hot_encode, parse_fasta  # type: ignore
    from model import SporulationModel, load_best_hparams_from_summary  # type: ignore
except Exception:
    from sporulation.code.training import FastaDataset, one_hot_encode, parse_fasta  # type: ignore
    from sporulation.code.model import SporulationModel, load_best_hparams_from_summary  # type: ignore

try:
    from phenotype_utils import build_labels_map_and_classes, DATA_ROOT
except ImportError:  # pragma: no cover
    from .phenotype_utils import build_labels_map_and_classes, DATA_ROOT  # type: ignore
import pandas as pd  # noqa: E402

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import HyperbandPruner, MedianPruner
except Exception:  # pragma: no cover
    optuna = None


# ----------------------------
# Data/config defaults (phenotype-agnostic)
# ----------------------------
BASE_DIR = str(DATA_ROOT)
# Read labels from metadata Excel and a chosen phenotype column
METADATA_XLSX = os.path.join('sporulation', 'microbe.cards table S1.xlsx')
PHENOTYPE_COL = 'Spore formation'
FILE_COL = 'Fasta file'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Robust direct-hotflip epsilon (max_flip_fraction) search defaults
ROBUST_EPS_MIN_DEFAULT = 1e-6
ROBUST_EPS_MAX_DEFAULT = 1e-1

# Default path for fixed best HPs summary (overridable via CLI)
SUMMARY_PATH = os.path.join('spore_optuna', 'sporo_full_std_v2_cont_exp_sporulation', 'summary.txt')

# ----------------------------
# Metadata/labels utilities
# ----------------------------
def _norm_basename(val: object) -> str:
    s = str(val) if not pd.isna(val) else ""
    s = os.path.basename(s)
    return s.strip().lower()

def read_metadata_table(xlsx_path: str) -> pd.DataFrame:
    md = pd.read_excel(xlsx_path)
    if 'Fasta file' not in md.columns:
        raise ValueError("Expected column 'Fasta file' in metadata Excel")
    md['Fasta file_norm'] = md['Fasta file'].map(_norm_basename)
    return md

def _count_labeled_fastas_in_dir(dir_path: str, labels_map: dict) -> int:
    if not os.path.isdir(dir_path):
        return 0
    exts = ('.fasta', '.fa', '.fna')
    count = 0
    try:
        for name in os.listdir(dir_path):
            if name.endswith(exts):
                if labels_map.get(str(name).strip().lower()) in (0, 1):
                    count += 1
    except Exception:
        return 0
    return count

def ensure_data_quality(metadata_df: pd.DataFrame, base_dir: str, phenotype_col: str, file_col: str,
                        min_train: int = 500, min_val: int = 100, min_test: int = 100) -> None:
    train_dir = os.path.join(base_dir, 'train')
    val_dir = os.path.join(base_dir, 'validation')
    test_dir = os.path.join(base_dir, 'test')
    labels_map, _classes = build_labels_map_and_classes(
        metadata_df,
        phenotype_col=phenotype_col,
        file_col=file_col,
        train_dirs=[train_dir],
    )
    n_train = _count_labeled_fastas_in_dir(train_dir, labels_map)
    n_val = _count_labeled_fastas_in_dir(val_dir, labels_map)
    n_test = _count_labeled_fastas_in_dir(test_dir, labels_map)
    if (n_train < min_train) or (n_val < min_val) or (n_test < min_test):
        print(f"[data-quality] Insufficient labeled FASTA counts: train={n_train}, val={n_val}, test={n_test}. "
              f"Require train>={min_train}, val>={min_val}, test>={min_test}. Aborting.", flush=True)
        raise SystemExit(1)


def set_seeds(seed: int = 42, deterministic: bool = False) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


# ----------------------------
# Model definition (parametric)
# ----------------------------
class SporoCNN(nn.Module):
    """Flexible 1D CNN for long DNA sequences.

    Architecture pattern:
      [Conv1 (large-k, stride s1) + BN + act1 + Drop + Pool1] ->
      N small conv blocks: [Conv + BN + ReLU + Drop + (optional Pool)] x N ->
      Global Pool -> FC -> 2-class logits
    """

    def __init__(
        self,
        k1: int = 201,
        c1: int = 64,
        stride1: int = 10,
        pool1_k: int = 50,
        pool1_s: int = 25,
        n_blocks: int = 2,
        k_small: int = 5,
        c2: int = 128,
        c3: int = 256,
        use_pool2: bool = True,
        pool2_k: int = 5,
        pool2_s: int = 5,
        use_pool3: bool = False,
        pool3_k: int = 3,
        pool3_s: int = 3,
        drop1: float = 0.1,
        drop2: float = 0.1,
        drop3: float = 0.1,
        drop_fc: float = 0.3,
        fc_hidden: int = 256,
        act1: str = 'relu',  # {'exp','relu','gelu','softplus'}
        global_pool: str = 'avg',  # {'avg','max'}
        num_classes: int = 2,
    ):
        super().__init__()
        assert n_blocks in (1, 2, 3), "n_blocks must be 1, 2, or 3"

        self.act1_name = act1
        self.global_pool = global_pool

        # Conv1: large kernel, stride to aggressively reduce length
        p1 = (k1 - 1) // 2
        self.conv1 = nn.Conv1d(4, c1, kernel_size=k1, stride=stride1, padding=p1, bias=True)
        self.bn1 = nn.BatchNorm1d(c1)
        self.drop1 = nn.Dropout(drop1)
        self.pool1 = nn.MaxPool1d(kernel_size=pool1_k, stride=pool1_s)

        # Small conv blocks
        ch = c1
        blocks = []
        out_dims = [c2, c3, max(c3, 256)]  # ensure growth for 3rd block if used
        pool_cfg = [
            (use_pool2, pool2_k, pool2_s),
            (use_pool3, pool3_k, pool3_s),
            (False, 1, 1),  # placeholder for block3 if n_blocks==3
        ]
        drops = [drop2, drop3, drop3]

        for i in range(n_blocks):
            co = out_dims[i]
            p = (k_small - 1) // 2
            blocks.append(nn.Conv1d(ch, co, kernel_size=k_small, padding=p, bias=True))
            blocks.append(nn.BatchNorm1d(co))
            blocks.append(nn.ReLU(inplace=True))
            blocks.append(nn.Dropout(drops[i]))
            use_pool, pk, ps = pool_cfg[i]
            if use_pool:
                blocks.append(nn.MaxPool1d(kernel_size=pk, stride=ps))
            ch = co

        self.blocks = nn.Sequential(*blocks)
        self.gpool = nn.AdaptiveAvgPool1d(1) if global_pool == 'avg' else nn.AdaptiveMaxPool1d(1)

        self.fc_drop = nn.Dropout(drop_fc)
        self.fc1 = nn.Linear(ch, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, num_classes)

    def _act1(self, x: torch.Tensor) -> torch.Tensor:
        if self.act1_name == 'exp':
            return torch.exp(x)
        if self.act1_name == 'gelu':
            return F.gelu(x)
        if self.act1_name == 'softplus':
            return F.softplus(x)
        return F.relu(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self._act1(x)
        x = self.drop1(x)
        x = self.pool1(x)

        x = self.blocks(x)
        x = self.gpool(x).squeeze(-1)
        x = self.fc_drop(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


# ----------------------------
# Training / evaluation helpers
# ----------------------------
# Cache sequences across Optuna trials to avoid repeated disk I/O
_TRAIN_SEQ_CACHE = None  # type: Optional[list]
_VAL_SEQ_CACHE = None    # type: Optional[list]


def _load_sequences_list(data_dir: str, metadata_df: pd.DataFrame, seq_len: int, phenotype_col: str, file_col: str, verbose: bool = True):
    labels_map, _classes = build_labels_map_and_classes(
        metadata_df,
        phenotype_col=phenotype_col,
        file_col=file_col,
        train_dirs=[os.path.join(BASE_DIR, 'train')],
    )
    sequences = []
    file_list = [f for f in os.listdir(data_dir) if f.endswith(('.fasta', '.fa', '.fna'))]
    total_files = len(file_list)
    if verbose:
        print(f"Preloading sequences (cached) from {data_dir}...")
    for i, file_name in enumerate(file_list):
        if verbose and (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{total_files} files...")
        label = labels_map.get(str(file_name).strip().lower())
        if label is None:
            continue
        fp = os.path.join(data_dir, file_name)
        seq = parse_fasta(fp)
        if not seq:
            continue
        if len(seq) < seq_len:
            seq = seq.ljust(seq_len, 'N')
        sequences.append({'sequence': seq, 'label': label})
    if verbose:
        print(f"Preloading complete. Total genomes: {len(sequences)}")
    return sequences


class CachedFastaDataset(Dataset):
    def __init__(self, sequences: list, seq_len: int, epoch_budget: Optional[int] = None, is_validation: bool = False):
        self.sequences = sequences  # list of {'sequence': str, 'label': int}
        self.seq_len = seq_len
        self.is_validation = is_validation
        if epoch_budget is not None:
            self.epoch_len = int(epoch_budget)
        else:
            total_len = sum(len(s['sequence']) for s in self.sequences)
            self.epoch_len = total_len // self.seq_len
        if self.is_validation:
            self._generate_deterministic_samples()

    def _generate_deterministic_samples(self):
        self.deterministic_samples = []
        rng = np.random.RandomState(42)
        for i in range(self.epoch_len):
            genome_idx = i % len(self.sequences)
            seq_str = self.sequences[genome_idx]['sequence']
            start = 0
            if len(seq_str) > self.seq_len:
                start = rng.randint(0, len(seq_str) - self.seq_len + 1)
            self.deterministic_samples.append({'genome_idx': genome_idx, 'start': start})

    def __len__(self):
        return self.epoch_len

    def __getitem__(self, idx):
        if self.is_validation:
            info = self.deterministic_samples[idx]
            genome_idx = info['genome_idx']
            start = info['start']
            sample = self.sequences[genome_idx]
            seq_str = sample['sequence'][start:start + self.seq_len]
            label = sample['label']
        else:
            genome_idx = idx % len(self.sequences)
            sample = self.sequences[genome_idx]
            seq_str = sample['sequence']
            label = sample['label']
            if len(seq_str) > self.seq_len:
                start = np.random.randint(0, len(seq_str) - self.seq_len + 1)
                seq_str = seq_str[start:start + self.seq_len]
        one_hot_seq = one_hot_encode(seq_str)
        return one_hot_seq, torch.tensor(label, dtype=torch.long)


def make_datasets(seq_len: int, epoch_budget: int, val_epoch_budget: int):
    global _TRAIN_SEQ_CACHE, _VAL_SEQ_CACHE
    metadata_df = read_metadata_table(METADATA_XLSX)
    train_dir = os.path.join(BASE_DIR, 'train')
    val_dir = os.path.join(BASE_DIR, 'validation')
    if _TRAIN_SEQ_CACHE is None:
        _TRAIN_SEQ_CACHE = _load_sequences_list(train_dir, metadata_df, seq_len, phenotype_col=PHENOTYPE_COL, file_col=FILE_COL, verbose=True)
    if _VAL_SEQ_CACHE is None:
        _VAL_SEQ_CACHE = _load_sequences_list(val_dir, metadata_df, seq_len, phenotype_col=PHENOTYPE_COL, file_col=FILE_COL, verbose=True)
    train_ds = CachedFastaDataset(_TRAIN_SEQ_CACHE, seq_len, epoch_budget=epoch_budget, is_validation=False)
    val_ds = CachedFastaDataset(_VAL_SEQ_CACHE, seq_len, epoch_budget=val_epoch_budget, is_validation=True)
    return train_ds, val_ds


def class_weights_from_dataset(train_ds: FastaDataset) -> torch.Tensor:
    labels = [sample['label'] for sample in train_ds.sequences]
    vc = pd.Series(labels).value_counts().sort_index()
    weights = (len(labels) / (2 * vc)).values
    return torch.tensor(weights, dtype=torch.float32, device=DEVICE)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, max_batches: Optional[int] = None) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    preds_all = []
    labels_all = []
    for bi, (xb, yb) in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)
        logits = model(xb)
        loss = loss_fn(logits, yb)
        total_loss += loss.item()
        n_batches += 1
        preds = torch.argmax(logits, dim=1)
        preds_all.extend(preds.detach().cpu().tolist())
        labels_all.extend(yb.detach().cpu().tolist())
    avg_loss = total_loss / max(1, n_batches)
    bal_acc = balanced_accuracy_score(labels_all, preds_all) if labels_all else 0.0
    return avg_loss, bal_acc


# ----------------------------
# Direct HotFlip for Sporulation (multiclass CE, vectorized)
# ----------------------------
def _generate_direct_hotflip_examples_sporo(
    model: nn.Module,
    xb: torch.Tensor,
    yb: torch.Tensor,
    loss_fn: nn.Module,
    flip_fraction: float,
) -> torch.Tensor:
    """Single-pass direct HotFlip generating adversarial batch for SporulationModel.

    Args:
        model: classifier producing (B, 2) logits
        xb: (B, 4, L) one-hot
        yb: (B,) class indices
        loss_fn: CrossEntropyLoss (with optional class weights)
        flip_fraction: fraction of positions to flip in each sequence

    Returns:
        adv_xb: (B, 4, L) adversarial examples
    """
    seq_len = xb.shape[2]
    batch_size = xb.shape[0]
    k_flips = int(max(0, min(1.0, float(flip_fraction))) * seq_len)
    if k_flips <= 0:
        return xb

    adv_xb = xb.clone().detach().requires_grad_(True)
    model.zero_grad(set_to_none=True)
    with autocast():
        logits = model(adv_xb)
        loss = loss_fn(logits, yb)
    loss.backward()
    grad = adv_xb.grad.data

    # Compute saliency for switching away from current base
    current_bases_onehot = (adv_xb > 0.5).float()
    grad_at_current = (grad * current_bases_onehot).sum(dim=1, keepdim=True)
    saliency = grad - grad_at_current
    saliency.masked_fill_(current_bases_onehot.bool(), float('-inf'))

    # Flatten saliency across channels and positions; pick top-k flips
    saliency_flat = saliency.reshape(batch_size, -1)
    k = min(int(k_flips), saliency_flat.shape[1])
    topk_values, topk_indices = torch.topk(saliency_flat, k, dim=1)

    adv_xb = adv_xb.detach()
    device = xb.device
    for flip_idx in range(k):
        batch_indices = torch.arange(batch_size, device=device)
        new_bases = topk_indices[:, flip_idx] // seq_len
        positions = topk_indices[:, flip_idx] % seq_len
        current_bases = adv_xb[batch_indices, :, positions].argmax(dim=1)
        adv_xb[batch_indices, current_bases, positions] = 0.0
        adv_xb[batch_indices, new_bases, positions] = 1.0

    return adv_xb

def train_one_trial(
    trial: 'optuna.trial.Trial',
    args: argparse.Namespace,
) -> float:
    phenotype_col = getattr(args, 'phenotype_col', PHENOTYPE_COL)
    phenotype_slug = phenotype_col.strip().lower().replace(' ', '_')
    standard_model_dir = os.path.join(args.outdir_root, phenotype_slug)
    os.makedirs(standard_model_dir, exist_ok=True)

    # Robust epsilon-only branch for Sporulation dataset with direct hotflip training
    if getattr(args, 'robust_epsilon_only', False):
        set_seeds(args.seed, deterministic=False)

        # Sample only epsilon (max_flip_fraction)
        eps_min = float(getattr(args, 'eps_min', ROBUST_EPS_MIN_DEFAULT))
        eps_max = float(getattr(args, 'eps_max', ROBUST_EPS_MAX_DEFAULT))
        max_flip_fraction = trial.suggest_float('max_flip_fraction', eps_min, eps_max, log=True)

        # Data
        seq_len = int(args.seq_len)
        robust_epoch_budget = int(getattr(args, 'robust_epoch_budget', 0) or args.epoch_budget)
        robust_val_epoch_budget = int(getattr(args, 'robust_val_epoch_budget', 0) or args.val_epoch_budget)
        train_ds, val_ds = make_datasets(seq_len, epoch_budget=robust_epoch_budget, val_epoch_budget=robust_val_epoch_budget)
        train_bs = int(getattr(args, 'batch_size', 20) or 20)
        val_bs = min(train_bs, 32)
        train_loader = DataLoader(train_ds, batch_size=train_bs, shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(val_ds, batch_size=val_bs, shuffle=False, num_workers=args.num_workers)

        # Model
        model = SporulationModel(summary_path=SUMMARY_PATH).to(DEVICE)
        # Optional compile for speed (PyTorch 2.x)
        if hasattr(torch, "compile"):
            try:
                model = torch.compile(model)  # type: ignore[attr-defined]
                print("[robust] Model compiled for speed", flush=True)
            except Exception:
                pass

        # Loss with class weights
        class_w = class_weights_from_dataset(train_ds)
        loss_fn = nn.CrossEntropyLoss(weight=class_w)

        # Optimizer & scheduler from best summary
        _hp = load_best_hparams_from_summary(SUMMARY_PATH)
        tuned_lr = float(_hp.get('lr', 0.001))
        tuned_wd = float(_hp.get('weight_decay', 1e-6))
        tuned_bs = int(_hp.get('batch_size', 20))
        tuned_grad_clip = float(_hp.get('grad_clip', 6.0))
        # Use tuned batch size for loaders
        train_bs = int(getattr(args, 'batch_size', tuned_bs) or tuned_bs)
        val_bs = min(train_bs, 32)
        # Rebuild loaders with tuned batch size
        train_loader = DataLoader(train_ds, batch_size=train_bs, shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(val_ds, batch_size=val_bs, shuffle=False, num_workers=args.num_workers)

        optimizer = torch.optim.AdamW(model.parameters(), lr=tuned_lr, weight_decay=tuned_wd)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
        scaler = GradScaler()

        # Training loop with direct hotflip
        best_val_loss = float('inf')
        es_counter = 0
        early_stopping_patience = int(getattr(args, 'early_stop_patience', 7))
        early_min_delta = float(getattr(args, 'early_stop_min_delta', 1e-4))
        max_epochs = int(getattr(args, 'max_epochs', 25))
        grad_clip = tuned_grad_clip

        # Scheduled direct hotflip: linearly ramp flips from 1 to max over epochs
        seq_len_int = int(seq_len)
        max_flips = max(1, int(float(max_flip_fraction) * seq_len_int))

        print(f"[robust] Trial {trial.number} | eps_max={max_flip_fraction:.3g} | schedule=ON | epochs={max_epochs} | train_bs={train_bs} | val_bs={val_bs}", flush=True)
        for epoch in range(max_epochs):
            model.train()
            total_loss = 0.0
            n_batches = 0
            # Compute scheduled flip fraction for this epoch
            if max_epochs > 1:
                num_flips = 1 + int(np.floor((max_flips - 1) * (epoch / (max_epochs - 1))))
            else:
                num_flips = max_flips
            current_flip_fraction = float(num_flips) / float(seq_len_int)
            for bi, (xb, yb) in enumerate(train_loader):
                if args.train_steps_per_epoch is not None and bi >= args.train_steps_per_epoch:
                    break
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)

                # Generate adversarial batch via direct hotflip (single backward pass)
                adv_xb = _generate_direct_hotflip_examples_sporo(
                    model, xb, yb, loss_fn,
                    flip_fraction=float(current_flip_fraction),
                )

                optimizer.zero_grad(set_to_none=True)
                with autocast():
                    logits = model(adv_xb)
                    loss = loss_fn(logits, yb)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                scaler.step(optimizer)
                scaler.update()
                total_loss += float(loss.item())
                n_batches += 1

            avg_train_loss = total_loss / max(1, n_batches)
            val_loss, val_bal_acc = evaluate(model, val_loader, loss_fn, max_batches=args.val_steps)
            scheduler.step(val_loss)

            # Report progress to Optuna and allow pruning
            try:
                trial.report(float(val_bal_acc), step=epoch)
                if 'optuna' in globals() and optuna is not None and trial.should_prune():
                    print(f"[robust] Trial {trial.number} pruned at epoch {epoch+1} (bal_acc={val_bal_acc:.4f})", flush=True)
                    raise optuna.TrialPruned()  # type: ignore[attr-defined]
            except Exception:
                pass

            # Early stopping on val loss
            improved = (best_val_loss - val_loss) > early_min_delta
            if improved:
                best_val_loss = val_loss
                es_counter = 0
            else:
                es_counter += 1
            if epoch >= int(getattr(args, 'early_stop_warmup', 3)) and es_counter >= early_stopping_patience:
                print(f"[robust] Early stopping at epoch {epoch+1}: val_loss={val_loss:.4f}", flush=True)
                break

            if getattr(args, 'verbose', False) or ((epoch + 1) % max(1, int(getattr(args, 'log_every', 1))) == 0):
                lr = optimizer.param_groups[0]['lr']
                print(f"[robust] Epoch {epoch+1}/{max_epochs} | train={avg_train_loss:.4f} | val={val_loss:.4f} | balAcc={val_bal_acc:.4f} | lr={lr:.2e} | eps={current_flip_fraction:.4g}", flush=True)

        # Save trial model
        out_root = os.path.join('slurm_results', 'sporo_robust_eps', f"{args.study_name}", f"trial_{trial.number}")
        os.makedirs(out_root, exist_ok=True)
        model_path = os.path.join(out_root, 'model.pth')
        torch.save(model.state_dict(), model_path)

        # Full evaluation (once) via subprocess to existing evaluation script
        eval_script = os.path.join('phenotype', 'code', 'evaluation.py')
        phenotype_col = getattr(args, 'phenotype_col', PHENOTYPE_COL)
        phenotype_slug = phenotype_col.strip().lower().replace(' ', '_')
        eval_dir_root = getattr(args, 'eval_dir', os.path.join('phenotype', 'data', 'eval'))
        eval_dir_root_norm = os.path.normpath(str(eval_dir_root))
        if os.path.basename(eval_dir_root_norm) == phenotype_slug:
            eval_dir = eval_dir_root_norm
        else:
            eval_dir = os.path.join(eval_dir_root_norm, phenotype_slug)
        test_dir = getattr(args, 'test_dir', str(DATA_ROOT / 'test'))
        # Efficiency knobs for evaluation
        robust_acc_bs = int(getattr(args, 'robust_acc_batch_size', 24))
        robust_ig_steps = int(getattr(args, 'robust_eval_ig_steps', 16))
        robust_pgd_steps = int(getattr(args, 'robust_eval_pgd_steps', 10))
        robust_sauc_sub = int(getattr(args, 'robust_eval_sauc_subsample', 50000))

        cmd = [sys.executable, '-u', eval_script,
               '--model_path', model_path,
               '--eval_dir', str(eval_dir),
               '--test_dir', str(test_dir),
               '--pgd_epsilon', str(max_flip_fraction),
               '--acc_batch_size', str(robust_acc_bs),
               '--ig_steps', str(robust_ig_steps),
               '--pgd_steps', str(robust_pgd_steps),
               '--sauc_outside_subsample', str(robust_sauc_sub),
               '--out_dir', out_root,
               '--phenotype', str(phenotype_col)]
        try:
            print("[robust] Running evaluation...", flush=True)
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError:
            # If evaluation fails, return poor score to allow Optuna to continue
            return 0.0

        # Read metrics.json (includes SaAUC, accuracy, SaSNR, and new Wilcoxon/CI fields)
        metrics_path = os.path.join(out_root, 'metrics.json')
        try:
            with open(metrics_path, 'r') as f:
                metrics = _json.load(f)
            sauc = float(metrics.get('mean_sauc', 0.0) or 0.0)
            acc = float(metrics.get('accuracy', 0.0) or 0.0)
            sasnr_raw = metrics.get('mean_sasnr')
            sasnr_exp_raw = metrics.get('mean_sasnr_expected')
            sasnr = float(sasnr_raw) if sasnr_raw is not None else float('nan')
            sasnr_expected = float(sasnr_exp_raw) if sasnr_exp_raw is not None else float('nan')
            # New fields from evaluation.py
            wilcoxon_p = metrics.get('sasnr_delta_wilcoxon_p_greater', None)
            delta_median = metrics.get('sasnr_delta_median', None)
            delta_ci_low = metrics.get('sasnr_delta_median_ci_low', None)
            delta_ci_high = metrics.get('sasnr_delta_median_ci_high', None)
            delta_n = int(metrics.get('sasnr_delta_n', 0) or 0)
        except Exception:
            sauc, acc = 0.0, 0.0
            sasnr = float('nan')
            sasnr_expected = float('nan')
            wilcoxon_p = None
            delta_median = None
            delta_ci_low = None
            delta_ci_high = None
            delta_n = 0

        # Log concise summary including new Wilcoxon/CI if available
        if wilcoxon_p is not None and delta_median is not None and delta_ci_low is not None and delta_ci_high is not None:
            print(f"[robust] Eval done. mean_sauc={sauc:.4f} acc={acc:.4f} sasnr={sasnr:.4f} exp={sasnr_expected:.4f} | Wilcoxon p={wilcoxon_p:.3g} medianΔ={delta_median:.4f} CI[{delta_ci_low:.4f},{delta_ci_high:.4f}] n={delta_n}", flush=True)
        else:
            print(f"[robust] Eval done. mean_sauc={sauc:.4f} acc={acc:.4f} sasnr={sasnr:.4f} exp={sasnr_expected:.4f}", flush=True)
        obj = 0.9 * sauc + 0.1 * acc

        # Persist as current-best checkpoint if improved (atomic replace)
        try:
            robust_dir = os.path.join('phenotype', 'robust_model', f"{args.study_name}")
            os.makedirs(robust_dir, exist_ok=True)
            best_model_path = os.path.join(robust_dir, 'best_model.pth')
            best_meta_path = os.path.join(robust_dir, 'best_meta.json')
            prev_best_val = float('-inf')
            if os.path.exists(best_meta_path):
                try:
                    with open(best_meta_path, 'r') as f:
                        prev = _json.load(f)
                    prev_best_val = float(prev.get('best_value', float('-inf')))
                except Exception:
                    prev_best_val = float('-inf')
            if float(obj) > prev_best_val:
                tmp_model = best_model_path + '.tmp'
                tmp_meta = best_meta_path + '.tmp'
                shutil.copy2(model_path, tmp_model)
                def _safe_float(val: float) -> Optional[float]:
                    return float(val) if math.isfinite(val) else None
                meta = {
                    'study': str(args.study_name),
                    'trial': int(trial.number),
                    'epsilon': float(max_flip_fraction),
                    'sauc': float(sauc),
                    'accuracy': float(acc),
                    'sasnr': _safe_float(sasnr),
                    'sasnr_expected': _safe_float(sasnr_expected),
                    'wilcoxon_p_greater': float(wilcoxon_p) if isinstance(wilcoxon_p, (int, float)) and math.isfinite(float(wilcoxon_p)) else None,
                    'sasnr_delta_median': float(delta_median) if isinstance(delta_median, (int, float)) and math.isfinite(float(delta_median)) else None,
                    'sasnr_delta_ci_low': float(delta_ci_low) if isinstance(delta_ci_low, (int, float)) and math.isfinite(float(delta_ci_low)) else None,
                    'sasnr_delta_ci_high': float(delta_ci_high) if isinstance(delta_ci_high, (int, float)) and math.isfinite(float(delta_ci_high)) else None,
                    'sasnr_delta_n': int(delta_n),
                    'best_value': float(obj),
                    'source_model_path': model_path,
                }
                with open(tmp_meta, 'w') as f:
                    _json.dump(meta, f, indent=2)
                os.replace(tmp_model, best_model_path)
                os.replace(tmp_meta, best_meta_path)
                print(f"[robust] New best saved: {best_model_path}", flush=True)
        except Exception:
            pass
        trial.set_user_attr('epsilon', float(max_flip_fraction))
        trial.set_user_attr('sauc', sauc)
        trial.set_user_attr('val_acc_eval', acc)
        trial.set_user_attr('sasnr', float(sasnr) if math.isfinite(sasnr) else None)
        trial.set_user_attr('sasnr_expected', float(sasnr_expected) if math.isfinite(sasnr_expected) else None)
        # New attributes for downstream analysis/triage
        try:
            trial.set_user_attr('wilcoxon_p_greater', float(wilcoxon_p) if wilcoxon_p is not None else None)
            trial.set_user_attr('sasnr_delta_median', float(delta_median) if delta_median is not None else None)
            trial.set_user_attr('sasnr_delta_ci_low', float(delta_ci_low) if delta_ci_low is not None else None)
            trial.set_user_attr('sasnr_delta_ci_high', float(delta_ci_high) if delta_ci_high is not None else None)
            trial.set_user_attr('sasnr_delta_n', int(delta_n))
        except Exception:
            pass
        # Persist best-so-far model under phenotype/robust_model/<study_name>
        try:
            robust_dir = os.path.join('phenotype', 'robust_model', f"{args.study_name}")
            os.makedirs(robust_dir, exist_ok=True)
            best_model_path = os.path.join(robust_dir, 'best_model.pth')
            best_meta_path = os.path.join(robust_dir, 'best_meta.json')
            prev_best = None
            if os.path.exists(best_meta_path):
                try:
                    with open(best_meta_path, 'r') as f:
                        prev_best = _json.load(f)
                except Exception:
                    prev_best = None
            prev_val = float(prev_best.get('best_value', float('-inf'))) if isinstance(prev_best, dict) else float('-inf')
            if float(obj) > prev_val:
                tmp_model = best_model_path + '.tmp'
                shutil.copy2(model_path, tmp_model)
                os.replace(tmp_model, best_model_path)
                def _safe_float(val: float) -> Optional[float]:
                    return float(val) if math.isfinite(val) else None
                meta = {
                    'study': str(args.study_name),
                    'trial': int(trial.number),
                    'epsilon': float(max_flip_fraction),
                    'sauc': float(sauc),
                    'accuracy': float(acc),
                    'sasnr': _safe_float(sasnr),
                    'sasnr_expected': _safe_float(sasnr_expected),
                    'best_value': float(obj),
                    'source_model_path': model_path,
                }
                tmp_meta = best_meta_path + '.tmp'
                with open(tmp_meta, 'w') as f:
                    _json.dump(meta, f, indent=2)
                os.replace(tmp_meta, best_meta_path)
        except Exception:
            pass
        return float(obj)
    set_seeds(args.seed, deterministic=False)

    # Suggest hyperparameters
    # Architectural (quasi-continuous where possible; rounded to valid ints)
    k1 = 2 * trial.suggest_int('k1_idx', 25, 200) + 1
    c1_cont = trial.suggest_float('c1_cont', 32, 128, log=True)
    c1 = int(max(16, min(128, 16 * round(c1_cont / 16.0))))
    stride1 = max(1, int(round(trial.suggest_float('stride1_cont', 5, 50, log=True))))
    pool1_k = max(2, int(round(trial.suggest_float('pool1_k_cont', 10, 128, log=True))))
    pool1_s = max(1, int(round(trial.suggest_float('pool1_s_cont', 5, 50, log=True))))
    n_blocks = trial.suggest_int('n_blocks', 1, 3)
    k_small = 2 * trial.suggest_int('k_small_idx', 1, 6) + 1
    c2_cont = trial.suggest_float('c2_cont', 64, 192, log=True)
    c2 = int(max(32, min(256, 32 * round(c2_cont / 32.0))))
    c3_cont = trial.suggest_float('c3_cont', 128, 384, log=True)
    c3 = int(max(64, min(512, 32 * round(c3_cont / 32.0))))
    use_pool2 = trial.suggest_categorical('use_pool2', [True, False])
    pool2_k = (2 * trial.suggest_int('pool2_k_idx', 1, 5) + 1) if use_pool2 else 3
    pool2_s = int(trial.suggest_int('pool2_s_int', 2, 7)) if use_pool2 else 2
    use_pool3 = trial.suggest_categorical('use_pool3', [True, False]) if n_blocks >= 2 else False
    pool3_k = (2 * trial.suggest_int('pool3_k_idx', 1, 4) + 1) if n_blocks >= 3 and use_pool3 else 3
    pool3_s = int(trial.suggest_int('pool3_s_int', 2, 7)) if n_blocks >= 3 and use_pool3 else 2
    drop1 = trial.suggest_float('drop1', 0.0, 0.4)
    drop2 = trial.suggest_float('drop2', 0.0, 0.4)
    drop3 = trial.suggest_float('drop3', 0.0, 0.4)
    drop_fc = trial.suggest_float('drop_fc', 0.2, 0.7)
    fc_hidden_cont = trial.suggest_float('fc_hidden_cont', 128, 512, log=True)
    fc_hidden = int(max(64, min(1024, 32 * round(fc_hidden_cont / 32.0))))
    act1 = trial.suggest_categorical('act1', ['exp'])
    global_pool = trial.suggest_categorical('global_pool', ['avg', 'max'])

    # Optimization
    lr = trial.suggest_float('lr', 1e-5, 3e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-8, 1e-2, log=True)
    grad_clip = trial.suggest_float('grad_clip', 0.5, 10.0)
    batch_size = trial.suggest_categorical('batch_size', [4, 8, 12, 16, 20, 24, 32])

    # Data and loader
    seq_len = args.seq_len
    train_ds, val_ds = make_datasets(seq_len, epoch_budget=args.epoch_budget, val_epoch_budget=args.val_epoch_budget)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=min(batch_size, 32), shuffle=False, num_workers=args.num_workers)

    # Determine number of classes from cached dataset labels
    train_labels = [sample['label'] for sample in train_ds.sequences]
    num_classes = int(pd.Series(train_labels).nunique()) if len(train_labels) > 0 else 2

    # Model & training setup
    model = SporoCNN(
        k1=k1, c1=c1, stride1=stride1, pool1_k=pool1_k, pool1_s=pool1_s,
        n_blocks=n_blocks, k_small=k_small, c2=c2, c3=c3,
        use_pool2=use_pool2, pool2_k=pool2_k, pool2_s=pool2_s,
        use_pool3=use_pool3, pool3_k=pool3_k, pool3_s=pool3_s,
        drop1=drop1, drop2=drop2, drop3=drop3, drop_fc=drop_fc,
        fc_hidden=fc_hidden, act1=act1, global_pool=global_pool,
        num_classes=num_classes,
    ).to(DEVICE)

    class_w = class_weights_from_dataset(train_ds)
    loss_fn = nn.CrossEntropyLoss(weight=class_w)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
    scaler = GradScaler()

    best_bal_acc = 0.0
    best_val_loss = float('inf')
    early_stopping_patience = args.early_stop_patience
    early_min_delta = args.early_stop_min_delta
    es_counter = 0

    best_checkpoint_acc = float('-inf')
    best_checkpoint_state: Optional[Dict[str, torch.Tensor]] = None
    best_checkpoint_epoch = -1
    best_checkpoint_loss = None

    for epoch in range(args.max_epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for bi, (xb, yb) in enumerate(train_loader):
            if args.train_steps_per_epoch is not None and bi >= args.train_steps_per_epoch:
                break
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                logits = model(xb)
                loss = loss_fn(logits, yb)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            n_batches += 1

        avg_train_loss = total_loss / max(1, n_batches)
        val_loss, val_bal_acc = evaluate(model, val_loader, loss_fn, max_batches=args.val_steps)
        scheduler.step(val_loss)

        # Track best & early stopping on val loss
        improved = (best_val_loss - val_loss) > early_min_delta
        if improved:
            best_val_loss = val_loss
            es_counter = 0
        else:
            es_counter += 1

        best_bal_acc = max(best_bal_acc, val_bal_acc)

        if float(val_bal_acc) >= best_checkpoint_acc:
            best_checkpoint_acc = float(val_bal_acc)
            best_checkpoint_loss = float(val_loss)
            best_checkpoint_epoch = epoch
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_checkpoint_state = state

        # Report to Optuna and prune if needed
        trial.report(val_bal_acc, step=epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        if args.verbose and (epoch % max(1, args.log_every) == 0 or epoch == args.max_epochs - 1):
            print(f"Epoch {epoch+1}/{args.max_epochs} | Train {avg_train_loss:.4f} | Val {val_loss:.4f} | BalAcc {val_bal_acc:.4f} | LR {optimizer.param_groups[0]['lr']:.2e}")

        if epoch >= args.early_stop_warmup and es_counter >= early_stopping_patience:
            if args.verbose:
                print(f"Early stopping at epoch {epoch+1}")
            break

    if best_checkpoint_state is not None:
        best_model_path = os.path.join(standard_model_dir, 'best_model.pth')
        best_meta_path = os.path.join(standard_model_dir, 'best_meta.json')
        prev_best_val = float('-inf')
        if os.path.exists(best_meta_path):
            try:
                with open(best_meta_path, 'r') as f:
                    prev = _json.load(f)
                prev_best_val = float(prev.get('best_value', float('-inf')))
            except Exception:
                prev_best_val = float('-inf')
        if float(best_bal_acc) > prev_best_val:
            tmp_model = best_model_path + '.tmp'
            torch.save(best_checkpoint_state, tmp_model)
            tmp_meta = best_meta_path + '.tmp'
            meta = {
                'study': str(args.study_name),
                'trial': int(trial.number),
                'best_value': float(best_bal_acc),
                'val_loss': float(best_checkpoint_loss) if best_checkpoint_loss is not None else None,
                'epoch': int(best_checkpoint_epoch),
                'phenotype': phenotype_col,
                'params': trial.params,
            }
            with open(tmp_meta, 'w') as f:
                _json.dump(meta, f, indent=2)
            os.replace(tmp_model, best_model_path)
            os.replace(tmp_meta, best_meta_path)
            print(f"[standard] New best saved: {best_model_path}", flush=True)

    return best_bal_acc


def build_study(args: argparse.Namespace) -> 'optuna.Study':
    assert optuna is not None, "Optuna is not installed. Please install optuna."
    # Use constant_liar to better support asynchronous parallel optimization
    sampler = TPESampler(seed=args.seed, constant_liar=True)  # single-objective Bayesian
    if args.sampler == 'tpe':
        sampler = TPESampler(seed=args.seed, constant_liar=True)
    # Future: add other samplers if needed

    if args.pruner == 'hyperband':
        pruner = HyperbandPruner()
    elif args.pruner == 'median':
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=3)
    else:
        pruner = None

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction='maximize',
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )
    return study


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Optuna HPO for phenotype-agnostic CNN')
    # HPO controls
    p.add_argument('--tune', action='store_true', help='Run Optuna tuning')
    p.add_argument('--study-name', type=str, default='phenotype_cnn')
    p.add_argument('--storage', type=str, default=os.environ.get('OPTUNA_STORAGE', 'sqlite:///./optuna.db'))
    p.add_argument('--sampler', type=str, default='tpe', choices=['tpe'])
    p.add_argument('--pruner', type=str, default='hyperband', choices=['hyperband', 'median', 'none'])
    p.add_argument('--n-trials', type=int, default=50)
    p.add_argument('--max-epochs', type=int, default=25)

    # Data/loader controls
    p.add_argument('--seq-len', type=int, default=1_000_000, help='Window length sampled from each genome')
    p.add_argument('--epoch-budget', type=int, default=4096, help='Number of training samples per epoch')
    p.add_argument('--val-epoch-budget', type=int, default=2048, help='Number of validation samples per epoch')
    p.add_argument('--train-steps-per-epoch', type=int, default=None, help='Max training batches per epoch (None=all)')
    p.add_argument('--val-steps', type=int, default=64, help='Max validation batches per epoch')
    p.add_argument('--num-workers', type=int, default=2)

    # Early stopping & logs
    p.add_argument('--early-stop-patience', type=int, default=7)
    p.add_argument('--early-stop-min-delta', type=float, default=1e-4)
    p.add_argument('--early-stop-warmup', type=int, default=3)
    p.add_argument('--log-every', type=int, default=1)
    p.add_argument('--verbose', action='store_true')

    # Misc
    p.add_argument('--seed', type=int, default=42)

    # Phenotype/labels (from metadata Excel)
    p.add_argument('--metadata-xlsx', type=str, default=METADATA_XLSX, help='Path to metadata Excel with phenotype labels')
    p.add_argument('--phenotype-col', type=str, default=PHENOTYPE_COL, help='Phenotype column name to use as binary label')
    p.add_argument('--file-col', type=str, default=FILE_COL, help='Column containing FASTA filenames (e.g., "Fasta file")')
    p.add_argument('--summary-path', type=str, default=SUMMARY_PATH, help='Optuna summary.txt path for loading tuned HPs')
    p.add_argument('--outdir-root', type=str, default=os.path.join('phenotype', 'model'), help='Root directory to save tuning outputs by phenotype')

    # Robust epsilon-only (sporulation, direct hotflip)
    p.add_argument('--robust-epsilon-only', action='store_true', help='Enable robust tuning for Sporulation with direct hotflip; tune epsilon only')
    p.add_argument('--eps-min', type=float, default=ROBUST_EPS_MIN_DEFAULT, help='Min epsilon (max_flip_fraction) for direct hotflip (log-uniform)')
    p.add_argument('--eps-max', type=float, default=ROBUST_EPS_MAX_DEFAULT, help='Max epsilon (max_flip_fraction) for direct hotflip (log-uniform)')
    p.add_argument('--eval-dir', type=str, default=os.path.join('phenotype', 'data', 'eval'), help='Base evaluation directory (phenotype subfolder appended automatically)')
    p.add_argument('--test-dir', type=str, default=str(DATA_ROOT / 'test'), help='Test FASTA root for sporulation evaluation.py')
    # Robust training efficiency/verbosity controls
    p.add_argument('--robust-epoch-budget', type=int, default=None, help='Override epoch budget for robust mode (defaults to epoch_budget)')
    p.add_argument('--robust-val-epoch-budget', type=int, default=None, help='Override val epoch budget for robust mode (defaults to val_epoch_budget)')
    # removed: hotflip cap/stride; using full top-k over (channel,position)
    p.add_argument('--robust-acc-batch-size', type=int, default=24, help='Batch size for evaluation accuracy pass')
    p.add_argument('--robust-eval-ig-steps', type=int, default=16, help='Fewer IG steps for faster robust eval')
    p.add_argument('--robust-eval-pgd-steps', type=int, default=10, help='Fewer PGD steps for faster robust eval')
    p.add_argument('--robust-eval-sauc-subsample', type=int, default=50000, help='Subsample negatives for SaAUC outside set during robust eval')

    args = p.parse_args()
    # Normalize names
    args.max_epochs = int(args.max_epochs)
    return args


def main():
    args = parse_args()
    if not args.tune:
        print('Nothing to do: pass --tune to run Optuna.'); return

    # Ensure result export root exists
    os.makedirs('spore_optuna', exist_ok=True)

    # Apply phenotype/label configuration
    global METADATA_XLSX, PHENOTYPE_COL, FILE_COL, SUMMARY_PATH
    METADATA_XLSX = args.metadata_xlsx
    PHENOTYPE_COL = args.phenotype_col
    FILE_COL = args.file_col
    SUMMARY_PATH = args.summary_path

    # Data quality gate
    md = read_metadata_table(METADATA_XLSX)
    ensure_data_quality(md, BASE_DIR, PHENOTYPE_COL, FILE_COL, min_train=500, min_val=100, min_test=100)

    study = build_study(args)

    def _obj(trial: 'optuna.trial.Trial') -> float:
        return train_one_trial(trial, args)

    study.optimize(_obj, n_trials=args.n_trials, gc_after_trial=True)

    # Minimal summary
    print(f"Study {study.study_name} finished. Best value: {study.best_value:.5f}")
    print("Best params:")
    for k, v in study.best_trial.params.items():
        print(f"  {k}: {v}")

    # Export simple CSV of trials under phenotype-specific subfolder
    try:
        df = study.trials_dataframe()
        # phenotype slug with underscores
        pheno_slug = str(args.phenotype_col).strip().lower().replace(' ', '_')
        outdir = os.path.join(args.outdir_root, pheno_slug)
        os.makedirs(outdir, exist_ok=True)
        df.to_csv(os.path.join(outdir, 'trials.csv'), index=False)
        with open(os.path.join(outdir, 'summary.txt'), 'w') as f:
            f.write(f"best_value={study.best_value}\n")
            for k, v in study.best_trial.params.items():
                f.write(f"{k}={v}\n")
        print(f"Trials exported to {outdir}")
    except Exception as e:
        print(f"Warning: could not export trials: {e}")


if __name__ == '__main__':
    main()
