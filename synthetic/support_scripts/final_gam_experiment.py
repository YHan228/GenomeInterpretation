"""
Final experiment to explore the interplay between signal strength (motif conservation)
and a GC-content confounder on model performance and interpretability.

The script includes a sophisticated data generator that creates positive examples
with multiple causal blocks and a promoter, alongside a mixed population of
negative examples (decoys, promoter-only, and pure background).

The analysis pipeline uses Expectile GAMs to model the full distribution of
performance metrics, not just the mean. Key features include:
  - Systematic exploration of a (GC Gap, Conservation) hyperparameter grid.
  - Automatic model selection to test for and visualize interaction effects.
  - Generation of "fan plots" to show how effects vary across expectiles.
  - Optional bootstrapping for robust confidence intervals on median effects.
  - Rich visualizations combining IQR heatmaps with median performance contours.

Execution is flexible, supporting single jobs on a SLURM cluster via --array_idx,
full local sweeps, and post-hoc analysis of existing results via --aggregate_only.
"""

import argparse
import os
import random
import sys
from typing import Tuple
import itertools
import glob
import json
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from captum.attr import IntegratedGradients
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score

# Check for pygam installation
try:
    from pygam import GAM, s, te, f, ExpectileGAM
except ImportError:
    print("Error: pygam is not installed. Please install it using 'pip install pygam'")
    sys.exit(1)

# ------------------------------------------------------------------ #
# 0.  Global settings
# ------------------------------------------------------------------ #
SEQ_LEN                = 1000
BLOCKS_RANGE           = (3, 4)
BLOCK_LEN_MEAN         = 55
BLOCK_LEN_SD           = 15
BLOCK_LEN_MIN, BLOCK_LEN_MAX = 40, 70
PROMOTER_HEX_1, PROMOTER_HEX_2 = "TTGACA", "TATAAT"
PROMOTER_SPACER        = 17
MIN_GAP_BETWEEN_BLOCKS = 30
DEFAULT_MOTIF_REPERTOIRE = 30  # new default size of ancestral motif library
N_ANCESTORS            = DEFAULT_MOTIF_REPERTOIRE  # may be overridden by CLI
N_TOTAL                = 5000  # Per HP combination
DEFAULT_BATCH_SIZE     = 512
DEFAULT_EPOCHS         = 50

# --- Hyperparameter Grid Definition ---
CONS_SPACE = np.linspace(0.55, 0.95, 10)
GC_GAP_SPACE = np.linspace(0.0, 0.20, 10)
TARGET_SIGNAL_FRAC = 0.20
HP_COMBOS = list(itertools.product(GC_GAP_SPACE, CONS_SPACE))

# --------------------------------------------------------------------------- #
# 1. Utilities
# --------------------------------------------------------------------------- #

def set_seeds(seed_value: int = 42) -> None:
    """Sets random seeds for reproducibility."""
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

ALPH = np.array(list("ACGT"), dtype="U1")
to_ix = {b: i for i, b in enumerate(ALPH)}

def sample_background(length: int, gc: float) -> np.ndarray:
    p = np.array([(1 - gc) / 2, gc / 2, gc / 2, (1 - gc) / 2])
    return np.random.choice(ALPH, size=length, p=p)

def mutate(chunk: np.ndarray, conservation: float) -> np.ndarray:
    mutated_chunk = chunk.copy()
    n_to_mutate = int(len(chunk) * (1.0 - conservation))
    pos_to_mutate = np.random.choice(len(chunk), n_to_mutate, replace=False)
    for pos in pos_to_mutate:
        original_base = mutated_chunk[pos]
        mutated_chunk[pos] = np.random.choice(np.setdiff1d(ALPH, [original_base]))
    return mutated_chunk

def one_hot(seq: np.ndarray) -> np.ndarray:
    arr = np.zeros((4, len(seq)), dtype=np.float32)
    for i, b in enumerate(seq):
        arr[to_ix[b], i] = 1.0
    return arr

def _trunc_norm(mu, sd, lo, hi, size=None):
    while True:
        v = np.random.normal(mu, sd, size)
        if np.all((v >= lo) & (v <= hi)):
            return v.astype(int)

def _nonoverlap_positions(seq_len, lens):
    tries = 0
    while tries < 1000:
        starts = sorted(np.random.randint(0, seq_len - sum(lens) - (len(lens) - 1) * MIN_GAP_BETWEEN_BLOCKS + 1, size=len(lens)))
        offsets = np.array([0] + [l + MIN_GAP_BETWEEN_BLOCKS for l in lens[:-1]])
        starts = np.array(starts) + np.cumsum(offsets)
        if starts[-1] + lens[-1] <= seq_len:
            if all(starts[i] + lens[i] + MIN_GAP_BETWEEN_BLOCKS <= starts[i+1] for i in range(len(starts)-1)):
                return starts.tolist()
        tries += 1
    raise RuntimeError("Could not place blocks without overlap.")

def build_promoter(gc):
    spacer = sample_background(PROMOTER_SPACER, gc)
    return np.array(list(PROMOTER_HEX_1 + ''.join(spacer) + PROMOTER_HEX_2), dtype="U1")

# --------------------------------------------------------------------------- #
# 2. Dataset Generation
# --------------------------------------------------------------------------- #

class SeqDS(Dataset):
    def __init__(self, xs, ys, ms):
        self.x, self.y, self.m = xs, ys, ms
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.m[idx]

def generate_dataset(gc_gap: float, conservation: float, target_signal_frac: float,
                     motif_repertoire: int = DEFAULT_MOTIF_REPERTOIRE,
                     include_partial_negatives: bool = True) -> Tuple[SeqDS, dict]:

    """Create dataset with possible partial-segment (decoy) negatives.

    Returns
    -------
    SeqDS
        Dataset containing one-hot encoded sequences, labels and causal masks.
    dict
        Book-keeping summary for reproducibility.
    """

    # --- decide sample budget to guarantee >=30 examples per (motif, hp-combo) ---
    avg_blocks = max((target_signal_frac * SEQ_LEN) / BLOCK_LEN_MEAN, 1.0)
    required_pos = int(np.ceil(30 * motif_repertoire / avg_blocks))
    pos_n_default = N_TOTAL // 2
    POS_N = max(required_pos, pos_n_default)
    NEG_N = POS_N  # keep 50/50 balance

    decoy_neg_n = int(0.2 * NEG_N) if include_partial_negatives else 0
    std_neg_n   = NEG_N - decoy_neg_n
    promoter_only_neg_n = int(0.2 * NEG_N) # 20% of negatives are promoter-only
    background_only_neg_n = std_neg_n - promoter_only_neg_n

    # build ancestral pool
    ancestral_pool = []
    for _ in range(motif_repertoire):
        ancestor_gc = np.clip(np.random.normal(0.50 + gc_gap, 0.02), 0.25, 0.75)
        ancestor_seq = sample_background(BLOCK_LEN_MAX, gc=ancestor_gc)
        ancestral_pool.append(ancestor_seq)

    X, y, masks = [], [], []

    # --- Positive examples (must have ≥3 blocks + promoter) ---
    n_pos_generated = 0
    realised_fracs_pos = []
    while n_pos_generated < POS_N:
        current_gc_pos = np.clip(np.random.normal(0.50 + gc_gap, 0.04), 0.25, 0.75)
        bg = sample_background(SEQ_LEN, gc=current_gc_pos)

        n_blocks_mean = (target_signal_frac * SEQ_LEN) / BLOCK_LEN_MEAN
        n_blocks_low  = int(np.floor(n_blocks_mean))
        n_blocks_high = int(np.ceil(n_blocks_mean))
        if n_blocks_low == n_blocks_high:
            n_blocks_high += 1
        n_blocks      = np.random.randint(max(3, n_blocks_low), max(4, n_blocks_high))

        blk_lens   = _trunc_norm(BLOCK_LEN_MEAN, BLOCK_LEN_SD, BLOCK_LEN_MIN, BLOCK_LEN_MAX, n_blocks)
        blk_starts = _nonoverlap_positions(SEQ_LEN, blk_lens)

        mask = np.zeros(SEQ_LEN, dtype=bool)
        for blen, start in zip(blk_lens, blk_starts):
            ancestor = random.choice(ancestral_pool)
            master   = ancestor[:blen]
            chunk    = mutate(master, conservation)
            bg[start:start+blen] = chunk
            mask[start:start+blen] = True

        # mandatory σ70-like promoter
        promoter_full_len = len(PROMOTER_HEX_1) + PROMOTER_SPACER + len(PROMOTER_HEX_2)
        first_start = min(blk_starts)
        if first_start < (promoter_full_len + MIN_GAP_BETWEEN_BLOCKS):
            continue  # cannot fit promoter – try again

        prom_seq = build_promoter(current_gc_pos)
        prom_pos = first_start - promoter_full_len - MIN_GAP_BETWEEN_BLOCKS
        if prom_pos < 0:
            continue

        bg[prom_pos: prom_pos + promoter_full_len] = prom_seq
        mask[prom_pos: prom_pos + promoter_full_len] = True

        # add finished positive
        X.append(one_hot(bg))
        y.append(1)
        masks.append(mask)
        realised_fracs_pos.append(mask.sum() / SEQ_LEN)
        n_pos_generated += 1

    # --- Decoy negatives (partial-segment) ---
    for _ in range(decoy_neg_n):
        current_gc = np.clip(np.random.normal(0.50, 0.04), 0.25, 0.75)
        bg = sample_background(SEQ_LEN, gc=current_gc)

        n_blocks = np.random.randint(1, 3)  # 1–2 blocks
        blk_lens   = _trunc_norm(BLOCK_LEN_MEAN, BLOCK_LEN_SD, BLOCK_LEN_MIN, BLOCK_LEN_MAX, n_blocks)
        blk_starts = _nonoverlap_positions(SEQ_LEN, blk_lens)

        mask = np.zeros(SEQ_LEN, dtype=bool)
        for blen, start in zip(blk_lens, blk_starts):
            ancestor = random.choice(ancestral_pool)
            master   = ancestor[:blen]
            chunk    = mutate(master, conservation)
            bg[start:start+blen] = chunk
            mask[start:start+blen] = True

        X.append(one_hot(bg))
        y.append(0)
        masks.append(mask)

    # --- Promoter-only negatives ---
    for _ in range(promoter_only_neg_n):
        current_gc = np.clip(np.random.normal(0.50, 0.04), 0.25, 0.75)
        bg = sample_background(SEQ_LEN, gc=current_gc)

        prom_seq = build_promoter(current_gc)
        promoter_full_len = len(prom_seq)
        
        if SEQ_LEN >= promoter_full_len:
            prom_pos = np.random.randint(0, SEQ_LEN - promoter_full_len + 1)
            bg[prom_pos : prom_pos + promoter_full_len] = prom_seq

        X.append(one_hot(bg))
        y.append(0)
        masks.append(np.zeros(SEQ_LEN, dtype=bool))

    # --- Standard negatives (no blocks / no promoter) ---
    for _ in range(background_only_neg_n):
        bg_gc = np.clip(np.random.normal(0.50, 0.04), 0.25, 0.75)
        bg    = sample_background(SEQ_LEN, gc=bg_gc)
        X.append(one_hot(bg))
        y.append(0)
        masks.append(np.zeros(SEQ_LEN, dtype=bool))

    X = torch.from_numpy(np.stack(X)).float()
    y = torch.from_numpy(np.array(y)).float()
    masks = np.stack(masks)

    avg_realised_frac = np.mean(realised_fracs_pos) if realised_fracs_pos else 0.0

    summary = {
        "n_sequences": len(X),
        "n_positive": POS_N,
        "n_decoy_negative": decoy_neg_n,
        "n_promoter_only_negative": promoter_only_neg_n,
        "motif_repertoire": motif_repertoire,
        "seed": np.random.get_state()[1][0].item(),
        "target_signal_frac": target_signal_frac,
        "avg_realised_frac": avg_realised_frac,
    }

    return SeqDS(X, y, masks), summary

# --------------------------------------------------------------------------- #
# 3. Model Definition
# --------------------------------------------------------------------------- #

class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(4, 32, 13, padding=6)
        self.conv2 = nn.Conv1d(32, 64, 7, padding=3)
        self.conv3 = nn.Conv1d(64, 128, 7, padding=3)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc   = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x)); x = F.max_pool1d(x, 2)
        x = F.relu(self.conv2(x)); x = F.max_pool1d(x, 2)
        x = F.relu(self.conv3(x)); x = F.max_pool1d(x, 2)
        x = self.pool(x).squeeze(-1)
        logits = self.fc(x)
        return logits.squeeze(-1), None

class LogisticRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
    def forward(self, x):
        return self.linear(x).squeeze(-1)

# --------------------------------------------------------------------------- #
# 4. Training & Evaluation
# --------------------------------------------------------------------------- #

def validate_epoch(model, loader, loss_fn, dev):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for xb, yb, _ in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            with autocast():
                logits, _ = model(xb)
                loss = loss_fn(logits, yb)
            total_loss += loss.item()
    return total_loss / len(loader) if len(loader) > 0 else 0

def train_standard(model, train_loader, val_loader, epochs, dev, early_stopping_patience: int = 10, early_stopping_min_delta: float = 1e-4):
    print(f"  Standard training for {epochs} epochs (patience={early_stopping_patience}, min_delta={early_stopping_min_delta})...")
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=4, verbose=False)
    scaler = GradScaler()
    
    best_val_loss = float('inf')
    early_stopping_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss, num_batches = 0.0, 0
        for xb, yb, _ in train_loader:
            xb, yb = xb.to(dev), yb.to(dev)
            optimizer.zero_grad()
            with autocast():
                logits, _ = model(xb)
                loss = loss_fn(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            num_batches += 1
        
        avg_train_loss = total_loss / num_batches if num_batches > 0 else 0
        avg_val_loss = validate_epoch(model, val_loader, loss_fn, dev)
        scheduler.step(avg_val_loss)
        print(f"    Epoch {epoch + 1}/{epochs}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        if (best_val_loss - avg_val_loss) > early_stopping_min_delta:
            best_val_loss = avg_val_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
        
        if early_stopping_counter >= early_stopping_patience:
            print(f"  -> Early stopping at epoch {epoch + 1}.")
            break

def evaluate_model(model, test_dl, dev):
    print("  Evaluating model...")
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb, _ in test_dl:
            xb, yb = xb.to(dev), yb.to(dev)
            with autocast():
                logits, _ = model(xb)
            
            # For AUROC
            all_preds.extend(torch.sigmoid(logits).cpu().numpy())
            all_labels.extend(yb.cpu().numpy())

            # For Accuracy
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == yb).sum().item()
            total += len(yb)
            
    accuracy = correct / total if total else 0
    model_auroc = roc_auc_score(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else 0.5

    def model_for_captum(x):
        return model(x)[0].unsqueeze(-1)

    ig = IntegratedGradients(model_for_captum)
    test_ds = test_dl.dataset
    positive_indices = [i for i, (_, y, _) in enumerate(test_ds) if y == 1]
    
    if not positive_indices:
        return accuracy, 0.0, 0.0, model_auroc

    sample_n = min(50, len(positive_indices))
    idxs = random.sample(positive_indices, sample_n)

    results = []
    for idx in idxs:
        xb, yb_scalar, mask = test_ds[idx]
        xb = xb.unsqueeze(0).to(dev)
        yb = torch.tensor([yb_scalar], device=dev, dtype=torch.float)
        
        proportions = xb.mean(dim=2, keepdim=True)
        baseline = proportions.expand_as(xb)
        attributions = ig.attribute(xb, baselines=baseline, target=0).abs().sum(1).squeeze(0).cpu().numpy()
        
        inside_scores = attributions[mask]
        outside_scores = attributions[~mask]
        saliency_auc = (inside_scores[:, None] > outside_scores[None, :]).mean() if len(inside_scores) > 0 and len(outside_scores) > 0 else 0.5
        sum_sq_inside = np.sum(inside_scores**2)
        sum_sq_total = np.sum(attributions**2)
        saliency_snr = sum_sq_inside / (sum_sq_total + 1e-9)
        results.append({'auc': saliency_auc, 'snr': saliency_snr})

    mean_auc = np.mean([r['auc'] for r in results]) if results else 0.0
    mean_snr = np.mean([r['snr'] for r in results]) if results else 0.0
    
    print(f"    Accuracy: {accuracy:.3f}, ModelAUROC: {model_auroc:.3f}, SaliencyAUC: {mean_auc:.3f}, SaliencySNR: {mean_snr:.3f}")
    return accuracy, mean_auc, mean_snr, model_auroc

def run_gc_logit_regression(train_ds, val_ds, test_ds, dev):
    print("\n--- Running GC-Content-Only Logistic Regression Baseline ---")
    
    def extract_gc(dataset):
        gcs, labels = [], []
        source_dataset = dataset.dataset if isinstance(dataset, torch.utils.data.Subset) else dataset
        indices = dataset.indices if isinstance(dataset, torch.utils.data.Subset) else range(len(source_dataset))
        for i in indices:
            x, y, _ = source_dataset[i]
            gc_content = (x[1].sum() + x[2].sum()) / SEQ_LEN
            gcs.append(gc_content.item())
            labels.append(y.item())
        return torch.tensor(gcs).float().unsqueeze(1), torch.tensor(labels).float()

    X_train_gc, y_train = extract_gc(train_ds)
    X_test_gc, y_test = extract_gc(test_ds)
    train_gc_ds = TensorDataset(X_train_gc, y_train)
    test_gc_ds = TensorDataset(X_test_gc, y_test)
    train_gc_dl = DataLoader(train_gc_ds, batch_size=DEFAULT_BATCH_SIZE, shuffle=True)
    test_gc_dl = DataLoader(test_gc_ds, batch_size=DEFAULT_BATCH_SIZE * 2)

    model = LogisticRegression().to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.BCEWithLogitsLoss()

    for _ in range(20):
        model.train()
        for x_gc, y_gc in train_gc_dl:
            x_gc, y_gc = x_gc.to(dev), y_gc.to(dev)
            optimizer.zero_grad()
            logits = model(x_gc)
            loss = loss_fn(logits, y_gc)
            loss.backward()
            optimizer.step()

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x_gc, y_gc in test_gc_dl:
            x_gc, y_gc = x_gc.to(dev), y_gc.to(dev)
            logits = model(x_gc)
            preds = torch.sigmoid(logits)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_gc.cpu().numpy())
            
    auroc = roc_auc_score(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else 0.5
    print(f"  GC-Only LogReg Test AUROC: {auroc:.4f}")
    return auroc

# --------------------------------------------------------------------------- #
# 5. Experiment Runner & Analysis
# --------------------------------------------------------------------------- #

def run_single_experiment(gc_gap: float, conservation: float, target_signal_frac: float, args: argparse.Namespace):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- Running: GC Gap={gc_gap:.3f}, Cons={conservation:.3f}, TargetSignalFrac={target_signal_frac:.2f} ---")
    
    # Use a unique seed for each data generation process for reproducibility
    # The combination of gc_gap, conservation, and a potential replicate ID will make this unique
    seed_base = int((gc_gap * 1000) + (conservation * 1000) + (target_signal_frac * 100))
    # If this is part of a SLURM array, use the array index to ensure seed uniqueness across replicates
    rep_id = args.array_idx if hasattr(args, 'array_idx') and args.array_idx is not None else 0
    seed = seed_base + rep_id
    
    set_seeds(seed)
    main_ds, gen_summary = generate_dataset(
        gc_gap=gc_gap,
        conservation=conservation,
        target_signal_frac=target_signal_frac,
        motif_repertoire=args.motif_repertoire,
        include_partial_negatives=args.include_partial_negatives,
    )
    avg_realised_frac = gen_summary['avg_realised_frac']

    print("[DGP]", json.dumps(gen_summary, separators=(",", ":")))
    
    train_size = int(0.7 * len(main_ds))
    val_size = int(0.15 * len(main_ds))
    test_size = len(main_ds) - train_size - val_size
    train_ds, val_ds, test_ds = random_split(main_ds, [train_size, val_size, test_size])

    gc_auroc = run_gc_logit_regression(train_ds, val_ds, test_ds, dev)

    persistent = args.num_workers > 0
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, persistent_workers=persistent)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size * 2, num_workers=args.num_workers, pin_memory=True, persistent_workers=persistent)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size * 2, num_workers=args.num_workers, pin_memory=True, persistent_workers=persistent)

    set_seeds(42)
    model = TinyCNN().to(dev)
    if hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"  torch.compile() failed, proceeding without it: {e}")
    train_standard(model, train_dl, val_dl, args.epochs, dev)

    acc, auc, snr, model_auroc = evaluate_model(model, test_dl, dev)
    return acc, auc, snr, gc_auroc, avg_realised_frac, model_auroc

def run_experiment_suite(args):
    print(f"Exploring {len(HP_COMBOS)} combinations of (GC Gap, Conservation) with target_signal_frac={TARGET_SIGNAL_FRAC:.2f}...")

    results_data = []
    for i, (gc_gap, conservation) in enumerate(HP_COMBOS):
        acc, auc, snr, gc_auroc, realised_frac, model_auroc = run_single_experiment(gc_gap, conservation, TARGET_SIGNAL_FRAC, args)
        results_data.append({
            'gc_gap': gc_gap,
            'conservation': conservation,
            'target_signal_frac': TARGET_SIGNAL_FRAC,
            'realised_frac': realised_frac,
            'accuracy': acc,
            'saliency_auc': auc,
            'saliency_snr': snr,
            'gc_auroc': gc_auroc,
            'model_auroc': model_auroc,
        })
    return pd.DataFrame(results_data)

def run_single_combo_for_slurm(args):
    array_idx = args.array_idx
    if array_idx < 0 or array_idx >= len(HP_COMBOS):
        print(f"Error: --array_idx {array_idx} is out of range for {len(HP_COMBOS)} total jobs.")
        sys.exit(1)
    
    gc_gap, conservation = HP_COMBOS[array_idx]
    
    acc, auc, snr, gc_auroc, realised_frac, model_auroc = run_single_experiment(gc_gap, conservation, TARGET_SIGNAL_FRAC, args)

    results_dir = os.path.join(args.output_dir, "npz_results")
    os.makedirs(results_dir, exist_ok=True)
    result_path = os.path.join(results_dir, f"result_{array_idx}.npz")
    
    np.savez(
        result_path,
        gc_gap=gc_gap,
        conservation=conservation,
        target_signal_frac=TARGET_SIGNAL_FRAC,
        realised_frac=realised_frac,
        accuracy=acc,
        saliency_auc=auc,
        saliency_snr=snr,
        gc_auroc=gc_auroc,
        model_auroc=model_auroc,
    )
    print(f"\nSaved single result to {result_path}")

def aggregate_and_analyze_results(args, expectiles):
    print("--- Aggregating results and running GAM analysis ---")
    npz_dir = os.path.join(args.output_dir, "npz_results")
    if not os.path.isdir(npz_dir):
        print(f"Error: The results directory '{npz_dir}' could not be found.")
        sys.exit(1)

    npz_files = glob.glob(os.path.join(npz_dir, "result_*.npz"))
    if not npz_files:
        print(f"No .npz result files found in {npz_dir}; nothing to aggregate.")
        sys.exit(1)
    
    print(f"Found {len(npz_files)} result files, building master dataframe...")
    all_data = []
    for f_path in npz_files:
        try:
            match = re.search(r'result_(\d+)', os.path.basename(f_path))
            rep_id = int(match.group(1)) if match else -1
            
            data = np.load(f_path)
            all_data.append({
                'rep_id': rep_id,
                'gc_gap': data['gc_gap'].item(),
                'conservation': data['conservation'].item(),
                'target_signal_frac': data['target_signal_frac'].item(),
                'realised_frac': data['realised_frac'].item(),
                'accuracy': data['accuracy'].item(),
                'saliency_auc': data['saliency_auc'].item(),
                'saliency_snr': data['saliency_snr'].item(),
                'gc_auroc': data['gc_auroc'].item(),
                'model_auroc': data['model_auroc'].item() if 'model_auroc' in data else np.nan,
            })
        except Exception as e:
            print(f"Could not process file {f_path}: {e}")

    df_results = pd.DataFrame(all_data)
    
    master_csv_path = os.path.join(args.output_dir, 'master_results.csv')
    df_results.to_csv(master_csv_path, index=False)
    print(f"\nSaved master results table to {master_csv_path}")

    analyze_and_plot_gams(df_results, args.output_dir, expectiles, args.n_boot)

def fit_expectile_gams(X, y, expectiles, lam, n_splines, n_splines_interact):
    """Fits multiple ExpectileGAMs for a given list of expectiles."""
    models = {}
    
    # Build model terms based on whether an interaction is requested
    gam_terms = s(0, n_splines=n_splines, lam=lam) + s(1, n_splines=n_splines, lam=lam)
    if n_splines_interact is not None:
        gam_terms += te(0, 1, n_splines=n_splines_interact, lam=lam)

    for q in expectiles:
        print(f"    Fitting ExpectileGAM for expectile q={q:.2f}...")
        gam = ExpectileGAM(gam_terms, expectile=q).fit(X, y)
        models[q] = gam
    return models

def bootstrap_partial_dependence(df, metric, term_index, n_boot, expectile, gam_terms):
    """Computes bootstrap confidence intervals for a partial dependence curve."""
    X = df[['gc_gap', 'conservation']].values
    y = df[metric].values
    
    # Fit on full data to generate grid and the main curve
    gam_full = ExpectileGAM(gam_terms, expectile=expectile).fit(X, y)
    grid = gam_full.generate_X_grid(term=term_index)
    pdep_full = gam_full.partial_dependence(term=term_index, X=grid)

    print(f"    Bootstrapping with {n_boot} samples for term {term_index}...")
    boot_pdeps = []
    for _ in range(n_boot):
        sample_indices = df.sample(n=len(df), replace=True).index
        X_boot, y_boot = X[sample_indices], y[sample_indices]
        try:
            gam_boot = ExpectileGAM(gam_terms, expectile=expectile).fit(X_boot, y_boot)
            boot_pdeps.append(gam_boot.partial_dependence(term=term_index, X=grid))
        except Exception as e:
            print(f"      Bootstrap fit failed: {e}. Skipping sample.")
            continue
            
    if not boot_pdeps:
        return grid, pdep_full, None, None

    lower = np.percentile(boot_pdeps, 2.5, axis=0)
    upper = np.percentile(boot_pdeps, 97.5, axis=0)
    
    return grid, pdep_full, lower, upper

def analyze_and_plot_gams(df_results, output_dir, expectiles, n_boot):
    print("\n--- Fitting Generalized Additive Models (GAMs) ---")
    os.makedirs(output_dir, exist_ok=True)
    
    # For bounded metrics, BetaGAM would be better, but it's not in stable pygam yet.
    # When available, one could do:
    # from pygam import BetaGAM
    # if metric in ['accuracy', 'saliency_auc', 'model_auroc']:
    #   gam = BetaGAM(link='logit', ...).fit(X,y)
    
    if n_boot > 0:
        print(f"--- Using bootstrap with n_boot={n_boot} to estimate CIs for the median (τ=0.5) ---")
        expectiles = [0.5] # focus on median when bootstrapping
    else:
        print(f"--- Plotting expectile fan for τ={expectiles} ---")

    low_q, mid_q, high_q = expectiles[0], expectiles[len(expectiles)//2], expectiles[-1]
    
    X = df_results[['gc_gap', 'conservation']].values
    metrics = ['accuracy', 'saliency_auc', 'saliency_snr', 'gc_auroc', 'model_auroc']
    
    # Pre-fit median expectile GAM for model_auroc to use in gc_auroc plot
    gam_model_auroc_median = None
    if 'model_auroc' in df_results.columns and not df_results['model_auroc'].isnull().all():
        y_model_auroc = df_results['model_auroc'].values
        lam, n_splines = 0.6, 8
        n_splines_interact = (min(n_splines, len(df_results['gc_gap'].unique())), 
                              min(n_splines, len(df_results['conservation'].unique())))
        
        gam_model_auroc_median = ExpectileGAM(s(0, n_splines=n_splines, lam=lam) +
                                              s(1, n_splines=n_splines, lam=lam) +
                                              te(0, 1, n_splines=n_splines_interact, lam=lam),
                                              expectile=mid_q).fit(X, y_model_auroc)

    for metric in metrics:
        if metric not in df_results.columns or df_results[metric].isnull().all():
            print(f"Skipping metric '{metric}' as it is not available in the results.")
            continue
            
        y = df_results[metric].values
        print(f"\n--- GAM for metric: {metric} ---")
        
        lam, n_splines = 0.6, 8
        n_splines_interact = (min(n_splines, len(df_results['gc_gap'].unique())), 
                              min(n_splines, len(df_results['conservation'].unique())))
        
        # --- Model Complexity Test (AICc comparison) ---
        print("  Running model complexity test (comparing with/without interaction term)...")
        gam_terms_simple = s(0, n_splines=n_splines, lam=lam) + s(1, n_splines=n_splines, lam=lam)
        gam_terms_complex = gam_terms_simple + te(0, 1, n_splines=n_splines_interact, lam=lam)
        
        gam_simple = ExpectileGAM(gam_terms_simple, expectile=0.5).fit(X, y)
        gam_complex = ExpectileGAM(gam_terms_complex, expectile=0.5).fit(X, y)
        
        aicc_simple = gam_simple.statistics_['AICc']
        aicc_complex = gam_complex.statistics_['AICc']
        
        print(f"    Simple Model (s(0)+s(1)) AICc: {aicc_simple:.2f}")
        print(f"    Complex Model (s(0)+s(1)+te(0,1)) AICc: {aicc_complex:.2f}")
        
        use_complex_model = aicc_complex < aicc_simple - 2
        if use_complex_model:
            print("  -> Conclusion: Complex model is substantially better. The interaction term is justified.")
            gam_terms = gam_terms_complex
        else:
            if aicc_simple < aicc_complex - 2:
                print("  -> Conclusion: Simple model is substantially better. Interaction term may not be needed.")
            else:
                print("  -> Conclusion: Models are very similar. The simpler model is preferred by parsimony.")
            gam_terms = gam_terms_simple
        print("    (Proceeding with the selected model for plotting.)\n")
        
        if n_boot == 0:
            # Re-fit expectile models with the chosen terms
            models = fit_expectile_gams(X, y, expectiles, lam, n_splines, n_splines_interact if use_complex_model else None)
            gam_mid = models[mid_q]
            print(gam_mid.summary())
        else:
            # For bootstrapping, we only need the median model
            gam_mid = ExpectileGAM(gam_terms, expectile=mid_q).fit(X, y)

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        if n_boot > 0:
            plot_title = f'Median GAM (τ=0.5) with {n_boot} Bootstrap CIs for {metric.replace("_", " ").title()}'
        else:
            if len(expectiles) > 3:
                plot_title = f'Expectile GAM Fan Plot for {metric.replace("_", " ").title()}'
            else:
                plot_title = f'Expectile GAM (τ={",".join(map(str, expectiles))}) for {metric.replace("_", " ").title()}'
        fig.suptitle(plot_title, fontsize=16)


        # Plot for gc_gap
        ax = axes[0, 0]
        if n_boot > 0:
            grid, pdep, lower, upper = bootstrap_partial_dependence(df_results, metric, 0, n_boot, mid_q, gam_terms)
            ax.plot(grid[:, 0], pdep, label=f's(GC Gap) (τ={mid_q:.2f})')
            if lower is not None and upper is not None:
                ax.fill_between(grid[:, 0], lower, upper, alpha=0.2, label='95% CI')
        else:
            XX_gc = models[mid_q].generate_X_grid(term=0)
            for q, gam in models.items():
                pdep = gam.partial_dependence(term=0, X=XX_gc)
                ax.plot(XX_gc[:, 0], pdep, label=f'τ={q:.2f}', ls='--' if q != mid_q else '-')
        
        ax.set_title('Partial Effect of GC Gap'); ax.set_xlabel('GC Gap (μ_pos - μ_neg)'); ax.set_ylabel(f's(GC Gap)')
        ax.legend()
        
        # Plot for conservation
        ax = axes[0, 1]
        if n_boot > 0:
            grid, pdep, lower, upper = bootstrap_partial_dependence(df_results, metric, 1, n_boot, mid_q, gam_terms)
            ax.plot(grid[:, 1], pdep, label=f's(Conservation) (τ={mid_q:.2f})')
            if lower is not None and upper is not None:
                ax.fill_between(grid[:, 1], lower, upper, alpha=0.2, label='95% CI')
        else:
            XX_cons = models[mid_q].generate_X_grid(term=1)
            for q, gam in models.items():
                pdep = gam.partial_dependence(term=1, X=XX_cons)
                ax.plot(XX_cons[:, 1], pdep, label=f'τ={q:.2f}', ls='--' if q != mid_q else '-')

        ax.set_title('Partial Effect of Conservation'); ax.set_xlabel('Causal Motif Conservation'); ax.set_ylabel('s(Conservation)')
        ax.legend()
        
        # Plot for interaction term or residuals
        ax = axes[1, 0]
        if use_complex_model:
            XX_grid = gam_mid.generate_X_grid(term=2, n=50) # te is term 2 in complex model
            gc_grid = XX_grid[:,0].reshape((50, 50)); cons_grid = XX_grid[:,1].reshape((50, 50))
            pdep_inter = gam_mid.partial_dependence(term=2, X=XX_grid).reshape((50, 50))
            im = ax.pcolormesh(gc_grid, cons_grid, pdep_inter, cmap='viridis', shading='auto')
            ax.set_title(f'Interaction Effect (τ={mid_q:.2f}): te(GC Gap, Conservation)'); ax.set_xlabel('GC Gap'); ax.set_ylabel('Conservation')
            fig.colorbar(im, ax=ax, label='Interaction Effect')
        else:
            # If no interaction, show a residuals vs. fitted plot
            fitted_values = gam_mid.predict(X)
            residuals = y - fitted_values
            ax.scatter(fitted_values, residuals, alpha=0.5, s=10)
            ax.axhline(0, ls='--', color='red')
            ax.set_title(f'Residuals vs. Fitted (τ={mid_q:.2f})'); ax.set_xlabel('Fitted Values'); ax.set_ylabel('Residuals')
            ax.grid(True, linestyle='--', alpha=0.6)

        # Plot for full model prediction or IQR
        ax = axes[1, 1]
        
        # The grid for prediction needs to have the correct number of features
        if use_complex_model:
            # Use term 2 (the interaction) to generate a grid for both features
            grid_source_term = 2
        else:
            # Use term 0 (gc_gap) just to get the base grid generation logic
            grid_source_term = 0
        XX_pred_grid = gam_mid.generate_X_grid(term=grid_source_term, n=50)

        if use_complex_model:
             gc_grid_pred = XX_pred_grid[:,0].reshape((50, 50))
             cons_grid_pred = XX_pred_grid[:,1].reshape((50, 50))
        else:
            # For simple model, must build grid manually for 2 features from the template
            gc_space = np.linspace(df_results['gc_gap'].min(), df_results['gc_gap'].max(), 50)
            cons_space = np.linspace(df_results['conservation'].min(), df_results['conservation'].max(), 50)
            gc_grid_pred, cons_grid_pred = np.meshgrid(gc_space, cons_space)
            XX_pred_grid = np.c_[gc_grid_pred.ravel(), cons_grid_pred.ravel()]

        if n_boot > 0:
            # With bootstrap, just show the median prediction surface
            pred_grid = gam_mid.predict(XX_pred_grid).reshape((50, 50))
            im_title = f'Full Model Prediction (τ={mid_q:.2f})'
            im_label = f'Predicted {metric}'

            # Clip for bounded metrics and set color bar limits
            if metric in ['accuracy', 'saliency_auc', 'model_auroc', 'gc_auroc', 'saliency_snr']:
                pred_grid = np.clip(pred_grid, 0, 1)
                im = ax.pcolormesh(gc_grid_pred, cons_grid_pred, pred_grid, cmap='viridis', shading='auto', vmin=0, vmax=1)
            else:
                im = ax.pcolormesh(gc_grid_pred, cons_grid_pred, pred_grid, cmap='viridis', shading='auto')
        else:
            # With expectile fan, show the inter-quantile range with median contours
            gam_low = models[low_q]
            gam_high = models[high_q]
            pred_low = gam_low.predict(XX_pred_grid)
            pred_high = gam_high.predict(XX_pred_grid)
            pred_mid = gam_mid.predict(XX_pred_grid)

            # Clip predictions for bounded metrics to make plots interpretable
            if metric in ['accuracy', 'saliency_auc', 'model_auroc', 'gc_auroc', 'saliency_snr']:
                pred_low, pred_high, pred_mid = [np.clip(p, 0, 1) for p in [pred_low, pred_high, pred_mid]]
            
            iqr_grid = (pred_high - pred_low).reshape((50, 50))
            
            # The heatmap color represents the spread (IQR)
            im = ax.pcolormesh(gc_grid_pred, cons_grid_pred, iqr_grid, cmap='viridis_r', shading='auto')
            
            # Overlay contour lines for the median prediction
            pred_mid_grid = pred_mid.reshape((50,50))
            contour_levels = np.arange(0, 1.01, 0.1) if metric in ['accuracy', 'saliency_auc', 'model_auroc', 'gc_auroc', 'saliency_snr'] else 10
            contour_set = ax.contour(gc_grid_pred, cons_grid_pred, pred_mid_grid, levels=contour_levels, colors='white', linewidths=0.7)
            ax.clabel(contour_set, inline=True, fontsize=8, fmt='%.2f')

            im_title = f'IQR of {metric.replace("_", " ")} with Median Contours'
            im_label = f'Prediction Spread (τ={high_q:.2f} - τ={low_q:.2f})'

        ax.set_title(im_title); ax.set_xlabel('GC Gap'); ax.set_ylabel('Conservation')
        fig.colorbar(im, ax=ax, label=im_label)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plot_path = os.path.join(output_dir, f'gam_plot_{metric}.png')
        plt.savefig(plot_path, dpi=200)
        plt.close(fig)
        print(f"Saved GAM plot to {plot_path}")

# --------------------------------------------------------------------------- #
# 6. Main Entry-point
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    try:
        torch.multiprocessing.set_start_method('forkserver', force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(description="Run Signal vs Confounder GAM analysis.")
    parser.add_argument("--output_dir", type=str, default="gam_final_results", help="Where to save results and plots.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Max training epochs per model.")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Training batch size.")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers.")
    parser.add_argument("--motif_repertoire", type=int, default=DEFAULT_MOTIF_REPERTOIRE, help="Number of ancestral motifs to sample from.")
    parser.add_argument("--include_partial_negatives", type=lambda x: str(x).lower() not in ['false','0','no'], default=True,
                    help="Include 20% partial-segment decoy negatives (default: True). Use '--include_partial_negatives False' to disable")
    parser.add_argument("--array_idx", type=int, default=None, help="SLURM_ARRAY_TASK_ID for single experiment.")
    parser.add_argument("--aggregate_only", action="store_true", help="Skip training, only aggregate npz results.")
    parser.add_argument("--expectiles", type=str, default=None, help="Comma-separated list of expectiles. If not provided when not bootstrapping, a fine-grained default fan is used.")
    parser.add_argument("--n_boot", type=int, default=0, help="Number of bootstrap samples for confidence intervals. If > 0, this overrides expectile fan plotting to show the median with a 95% CI.")
    
    args = parser.parse_args()

    # Smartly set expectiles based on run mode
    if args.n_boot > 0:
        final_expectiles = [0.5]
    else:
        if args.expectiles is None:
            # Default to a fine-grained fan plot if not bootstrapping and no specific expectiles are given
            final_expectiles = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
        else:
            final_expectiles = sorted([float(e.strip()) for e in args.expectiles.split(',')])

    if args.aggregate_only:
        aggregate_and_analyze_results(args, final_expectiles)
    elif args.array_idx is not None:
        run_single_combo_for_slurm(args)
    else:
        print("--- Running full experiment suite locally ---")
        df_results = run_experiment_suite(args)
        os.makedirs(args.output_dir, exist_ok=True)
        results_csv_path = os.path.join(args.output_dir, 'master_results.csv')
        df_results.to_csv(results_csv_path, index=False)
        print(f"\nSaved master results table to {results_csv_path}")
        analyze_and_plot_gams(df_results, args.output_dir, final_expectiles, args.n_boot)

    print("\nAnalysis complete.") 