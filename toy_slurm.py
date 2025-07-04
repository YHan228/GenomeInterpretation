"""
Synthetic 1-kbp phenotype dataset (SLURM-compatible version)
    positives: high-GC background + one 60-bp causal block
    negatives: low-GC background, no causal block
CNN training + Integrated Gradients attribution quality
Author: Yichen Han, 2025-06-29

--------------------------------------------------------------------------------
High-level overview of the experimental design
--------------------------------------------------------------------------------
This script implements a controlled synthetic experiment to evaluate the effect
of adversarial training on model interpretability in the presence of a feature
confounder. The key components are:

a. Synthetic Data Generation:
   - binary classification task of 1 kbp DNA sequences.
   - Negative examples: random background with a GC content of 50%.
   - Positive examples: embedded with a conserved 60-bp "causal
     motif" into a background with a higher GC content (a confounding feature). The
     strength of the confounder is controlled by the `gc_pos` parameter, and the
     strength of the true signal is controlled by the `conservation` parameter.

b. Adversarial Attack (HotFlip-style):
   - Adversarially-trained models are generated using an iterative,
     gradient-based attack inspired by HotFlip.
   - For each sequence in a batch, the process is as follows:
     1. The gradient of the loss is computed with respect to the one-hot
        encoded input sequence.
     2. This gradient is used to calculate a saliency score for flipping each
        nucleotide at each position to one of the other three bases.
     3. The single best flip (i.e., the position and new base that most
        increases the loss) is identified.
     4. The sequence is modified with this single flip.
     5. This process is repeated for *k* iterations, where *k* is the number of
        nucleotides to flip (k = epsilon * sequence length). This iterative design creates
        progressively more challenging adversarial examples.
   - The model is then trained on this final, "attacked" batch.
   
c. Interpretability Metrics:
   - We use Integrated Gradients with an all-zero baseline to generate
     attribution maps for each sequence. The quality of these maps is
     quantified using three metrics (all ranging from 0 to 1, except SNR):
     1. wIoU (weighted Intersection-over-Union): Measures how well the
        attribution map *locates* the true causal motif. It is calculated as
        the sum of attribution scores inside the true motif divided by the sum
        of scores in the union of the true motif and the predicted region
        (defined as the 60bp window with the highest total attribution). A
        score of 1 indicates a perfect match.
     2. SaliencyAUC: Measures the *purity* of attributions. It is the
        probability that a randomly chosen position inside the motif has a
        higher attribution score than a randomly chosen position outside. A
        score of 1 means all attributions are correctly concentrated within
        the motif.
     3. SaliencySNR (Signal-to-Noise Ratio): Measures the *sharpness* of
        attributions. It is the ratio of the mean attribution score inside the
        motif to the mean score outside. Its range is 0 to infinity.

d. Model Architecture (TinyCNN):
   - A simple, 3-layer convolutional neural network (CNN) is used for the
     classification task.
   - It consists of three Conv1d layers with ReLU activations and max-pooling,
     followed by a final fully-connected layer.
   - This simple architecture is chosen to be representative of common models
     in genomics while remaining fast to train.
--------------------------------------------------------------------------------
"""

import itertools
import math
import os
import random
import string
import argparse
import sys
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from captum.attr import IntegratedGradients
from torch.utils.data import DataLoader, Dataset, random_split
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
import glob
import pandas as pd
import seaborn as sns
import matplotlib.ticker as ticker


# --------------------------------------------------------------------------- //
# 1. Configuration & Utilities
# --------------------------------------------------------------------------- #

WITH_CONFOUNDER = True # Global switch for GC-content difference

# --- Hyperparameter Search Space ---
GC_HPARAMS = [0.525, 0.535, 0.55, 0.575, 0.6, 0.625, 0.65]
CONS_HPARAMS = [0.6, 0.7, 0.8]
# ---

SEEDS = [0, 1, 2, 3, 4]
EPSILONS = [0.001, 0.003, 0.005, 0.01, 0.025, 0.05]

# Directory to cache synthetic datasets so we do not regenerate the same
# (gc, conservation) combination multiple times across different SLURM tasks.
DATASET_CACHE_DIR = "dataset_cache"
os.makedirs(DATASET_CACHE_DIR, exist_ok=True)

# Default training parameters – can be overriden via CLI.
DEFAULT_BATCH_SIZE = 512  # fits comfortably on V100 / T4

def set_seeds(seed_value: int = 42) -> None:
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    # Enable fastest cuDNN kernels; reproducibility is ensured by fixed seeds.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


set_seeds(42)  # initial seed for consistency

ALPH = np.array(list("ACGT"), dtype="U1")
to_ix = {b: i for i, b in enumerate(ALPH)}


def sample_background(length: int, gc: float) -> np.ndarray:
    """iid sampling with given GC content, returns char array"""
    p = np.array([(1 - gc) / 2, gc / 2, gc / 2, (1 - gc) / 2])  # A,C,G,T
    return np.random.choice(ALPH, size=length, p=p)


def random_chunk(length: int) -> np.ndarray:
    """60-bp random chunk with balanced GC ≈ 50 %"""
    return sample_background(length, 0.50)


def mutate(chunk: np.ndarray, conservation: float) -> np.ndarray:
    """Return a new chunk with given conservation level (≈ %identity)"""
    mutated_chunk = chunk.copy()
    n_to_mutate = int(len(chunk) * (1.0 - conservation))
    pos_to_mutate = np.random.choice(len(chunk), n_to_mutate, replace=False)
    for pos in pos_to_mutate:
        original_base = mutated_chunk[pos]
        mutated_chunk[pos] = np.random.choice(np.setdiff1d(ALPH, [original_base]))
    return mutated_chunk


def embed(seq: np.ndarray, chunk: np.ndarray) -> Tuple[np.ndarray, int]:
    """Insert chunk at random non-overlapping position; return new seq and start idx"""
    L, l = len(seq), len(chunk)
    start = np.random.randint(0, L - l + 1)
    seq[start:start + l] = chunk
    return seq, start


def one_hot(seq: np.ndarray) -> np.ndarray:
    """(1000,) char → (4,1000) float32 one-hot"""
    arr = np.zeros((4, len(seq)), dtype=np.float32)
    for i, b in enumerate(seq):
        arr[to_ix[b], i] = 1.0
    return arr


def one_hot_to_seq(one_hot_tensor: torch.Tensor) -> str:
    """ (4, L) float tensor -> (L,) string """
    indices = torch.argmax(one_hot_tensor, dim=0).cpu().numpy()
    return "".join(ALPH[indices])


# --------------------------------------------------------------------------- #
# 1.a  Dataset caching helpers
# --------------------------------------------------------------------------- #


def _dataset_cache_path(gc_pos: float, conservation: float) -> str:
    """Return cache file path for a given (gc_pos, conservation)."""
    return os.path.join(
        DATASET_CACHE_DIR,
        f"gc_{gc_pos:.3f}_cons_{conservation:.3f}.npz",
    )


def load_or_generate_dataset(gc_pos: float, conservation: float):
    """Load dataset from cache if present, otherwise generate and cache it."""
    cache_path = _dataset_cache_path(gc_pos, conservation)
    if os.path.exists(cache_path):
        data = np.load(cache_path)
        X = torch.tensor(data["X"], dtype=torch.float32)
        y = torch.tensor(data["y"], dtype=torch.float)
        masks = data["masks"]
        return SeqDS(X, y, masks)

    ds = generate_dataset(gc_pos=gc_pos, conservation=conservation)
    # Save to cache (ensure tensors are on CPU and numpy format)
    np.savez(
        cache_path,
        X=ds.x.cpu().numpy(),
        y=ds.y.cpu().numpy(),
        masks=ds.m,
    )
    return ds


# --------------------------------------------------------------------------- #
# 2. Dataset generation
# --------------------------------------------------------------------------- #

SEQ_LEN = 1000
CHUNK_LEN = 60
N_TOTAL = 10000
POS_N = N_TOTAL // 2
NEG_N = N_TOTAL - POS_N

def generate_dataset(gc_pos: float, conservation: float):
    """Generates the main dataset based on global config."""
    print(f"Generating dataset with GC_POS={gc_pos:.2f} and conservation={conservation:.2f}...")
    GC_NEG = 0.50

    X, y, masks = [], [], []
    master_chunk = random_chunk(CHUNK_LEN)

    for _ in range(POS_N):
        bg = sample_background(SEQ_LEN, gc=gc_pos)
        chunk = mutate(master_chunk, conservation)
        seq, start = embed(bg, chunk)
        X.append(one_hot(seq))
        y.append(1)
        mask = np.zeros(SEQ_LEN, dtype=bool)
        mask[start:start + CHUNK_LEN] = True
        masks.append(mask)

    for _ in range(NEG_N):
        bg = sample_background(SEQ_LEN, gc=GC_NEG)
        X.append(one_hot(bg))
        y.append(0)
        masks.append(np.zeros(SEQ_LEN, dtype=bool))

    X = torch.tensor(np.stack(X))
    y = torch.tensor(y, dtype=torch.float)
    masks = np.stack(masks)
    
    return SeqDS(X, y, masks)


# --------------------------------------------------------------------------- #
# 3. Model and Dataset Classes
# --------------------------------------------------------------------------- #

class SeqDS(Dataset):
    def __init__(self, xs, ys, ms):
        self.x, self.y, self.m = xs, ys, ms

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.m[idx]

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
        return logits.squeeze(-1)


# --------------------------------------------------------------------------- #
# 4. Training, Scheduling, and Evaluation
# --------------------------------------------------------------------------- #

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=5, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

def validate_epoch(model, loader, loss_fn, dev):
    """Calculates the loss on a validation set."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        for xb, yb, _ in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            with autocast():
                logits = model(xb)
                loss = loss_fn(logits, yb)
            total_loss += loss.item()
            num_batches += 1
    return total_loss / num_batches if num_batches > 0 else 0


def train_standard(model, train_loader, val_loader, loss_fn, optimizer, dev, scaler, writer, scheduler, early_stopper, epochs: int = 10) -> None:
    print("Starting standard training...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0
        for xb, yb, _ in train_loader:
            xb, yb = xb.to(dev), yb.to(dev)
            optimizer.zero_grad()
            with autocast():
                loss = loss_fn(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            num_batches += 1
        
        avg_train_loss = total_loss / num_batches if num_batches > 0 else 0
        avg_val_loss = validate_epoch(model, val_loader, loss_fn, dev)

        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/validation', avg_val_loss, epoch)
        writer.add_scalar('LR/train', scheduler.get_last_lr()[0], epoch)
        print(f"  Epoch {epoch + 1}/{epochs}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        scheduler.step()
        early_stopper(avg_val_loss)
        if early_stopper.early_stop:
            print("Early stopping triggered.")
            break

def generate_hotflip_examples(model, xb, yb, loss_fn, flip_fraction: float, 
                              neighborhood_size: int = 20, penalize_nearby: bool = False):
    seq_len = xb.shape[2]
    k_flips = int(flip_fraction * seq_len)
    adv_xb = xb.clone()
    forbidden_regions = torch.zeros_like(adv_xb[:, 0, :], dtype=torch.bool, device=xb.device)

    for _ in range(k_flips):
        adv_xb.requires_grad = True
        model.zero_grad()
        with autocast():
            logits = model(adv_xb)
            loss = loss_fn(logits, yb)
        loss.backward()
        grad = adv_xb.grad.data
        current_bases_onehot = (adv_xb > 0.5).float()
        grad_at_current_bases = (grad * current_bases_onehot).sum(dim=1, keepdim=True)
        saliency_scores = grad - grad_at_current_bases
        saliency_scores.masked_fill_(current_bases_onehot.bool(), -1e9)
        best_flip_scores_per_pos, _ = saliency_scores.max(dim=1)
        if penalize_nearby:
            best_flip_scores_per_pos.masked_fill_(forbidden_regions, -1e9)
        best_pos_to_flip = best_flip_scores_per_pos.argmax(dim=1)
        best_new_base_idx = saliency_scores[range(len(xb)), :, best_pos_to_flip].argmax(dim=1)
        old_base_idx = adv_xb[range(len(xb)), :, best_pos_to_flip].argmax(dim=1)
        adv_xb = adv_xb.detach()
        adv_xb[range(len(xb)), old_base_idx, best_pos_to_flip] = 0.0
        adv_xb[range(len(xb)), best_new_base_idx, best_pos_to_flip] = 1.0
        if penalize_nearby:
            pos = best_pos_to_flip
            start = torch.clamp(pos - neighborhood_size, 0)
            end = torch.clamp(pos + neighborhood_size + 1, max=seq_len)
            indices = torch.arange(seq_len, device=xb.device).unsqueeze(0)
            newly_forbidden = (indices >= start.unsqueeze(1)) & (indices < end.unsqueeze(1))
            forbidden_regions |= newly_forbidden
    return adv_xb

def train_hotflip(model, train_loader, val_loader, loss_fn, optimizer, dev, scaler, writer, scheduler, early_stopper,
                  max_flip_fraction: float, epochs: int = 10, use_scheduling: bool = True) -> None:
    
    scheduling_str = "ON" if use_scheduling else "OFF"
    print(f"Starting HotFlip training with max_flip_fraction = {max_flip_fraction:.4f}, Scheduling: {scheduling_str}...")
    
    for epoch in range(epochs):
        
        max_flips = int(max_flip_fraction * SEQ_LEN)
        
        if use_scheduling:
            # Smart scheduling: Linearly ramp up from 1 flip to max_flips.
            # Ensures at least one flip occurs, even for small epsilons.
            num_flips = 1 + int(np.floor((max_flips - 1) * (epoch / (epochs - 1)))) if epochs > 1 else max_flips
            current_flip_fraction = num_flips / SEQ_LEN
        else:
            current_flip_fraction = max_flip_fraction
        
        model.train()
        total_loss = 0.0
        num_batches = 0
        for xb, yb, _ in train_loader:
            xb, yb = xb.to(dev), yb.to(dev)
            adv_xb = generate_hotflip_examples(model, xb, yb, loss_fn, current_flip_fraction)
            optimizer.zero_grad()
            with autocast():
                logits_adv = model(adv_xb)
                loss_adv = loss_fn(logits_adv, yb)
            scaler.scale(loss_adv).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss_adv.item()
            num_batches += 1

        avg_train_loss = total_loss / num_batches if num_batches > 0 else 0
        avg_val_loss = validate_epoch(model, val_loader, loss_fn, dev)
        
        writer.add_scalar('Loss/train_adversarial', avg_train_loss, epoch)
        writer.add_scalar('Loss/validation_adversarial', avg_val_loss, epoch)
        writer.add_scalar('LR/train_adversarial', scheduler.get_last_lr()[0], epoch)
        writer.add_scalar('Epsilon/train_adversarial', current_flip_fraction, epoch)
        print(f"  Epoch {epoch + 1}/{epochs}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Epsilon: {current_flip_fraction:.4f}")

        scheduler.step()
        early_stopper(avg_val_loss)
        if early_stopper.early_stop:
            print("Early stopping triggered.")
            break

def evaluate_model(model, test_dl, dev):
    print(f"Evaluating model...")
    SAMPLE_N = 100  # Reduced from 300 for faster evaluation
    ANALYSIS_CHUNK_LEN = 60

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb, _ in test_dl:
            xb, yb = xb.to(dev), yb.to(dev)
            with autocast():
                logits = model(xb)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == yb).sum().item()
            total += len(yb)
    accuracy = correct / total if total else 0
    print(f"  Test accuracy: {accuracy:.3f}")

    def model_for_captum(x):
        with autocast():
            return model(x).unsqueeze(-1)

    ig = IntegratedGradients(model_for_captum)
    test_ds = test_dl.dataset
    positive_subset_indices = [
        i for i, original_idx in enumerate(test_ds.indices)
        if test_ds.dataset.y[original_idx] == 1
    ]

    rng = np.random.default_rng(0)
    sample_n_actual = min(SAMPLE_N, len(positive_subset_indices))
    if sample_n_actual == 0:
        print("Warning: No positive samples in test set for evaluation.")
        return 0.0, accuracy, 0.0, 0.0
        
    idxs = rng.choice(positive_subset_indices, size=sample_n_actual, replace=False)

    results = []
    for idx in idxs:
        xb, _, mask = test_ds[idx]
        xb = xb.unsqueeze(0).to(dev)
        attributions = ig.attribute(xb, target=0).abs().sum(1).squeeze(0).cpu().numpy()
        window_sums = np.convolve(attributions, np.ones(ANALYSIS_CHUNK_LEN), mode='valid')
        best_window_start = np.argmax(window_sums)
        pred_mask_cont = np.zeros(SEQ_LEN, dtype=bool)
        pred_mask_cont[best_window_start:best_window_start + ANALYSIS_CHUNK_LEN] = True
        inter_cont = (pred_mask_cont & mask).sum()
        union_cont = (pred_mask_cont | mask).sum()
        iou_cont = inter_cont / union_cont if union_cont else 0
        inside_scores = attributions[mask]
        outside_scores = attributions[~mask]
        saliency_auc = (inside_scores[:, None] > outside_scores[None, :]).mean()
        
        # New metric: Saliency Signal-to-Noise Ratio
        mean_inside = inside_scores.mean()
        mean_outside = outside_scores.mean()
        saliency_snr = mean_inside / (mean_outside + 1e-9)

        results.append(dict(iou_cont=iou_cont, saliency_auc=saliency_auc, saliency_snr=saliency_snr))

    mean_iou_cont = np.mean([r['iou_cont'] for r in results])
    mean_saliency_auc = np.mean([r['saliency_auc'] for r in results])
    mean_saliency_snr = np.mean([r['saliency_snr'] for r in results])
    print(f"  Mean wIoU: {mean_iou_cont:.3f}")
    print(f"  Mean Saliency AUC: {mean_saliency_auc:.3f}")
    print(f"  Mean Saliency SNR: {mean_saliency_snr:.3f}")

    return mean_iou_cont, accuracy, mean_saliency_auc, mean_saliency_snr

# --------------------------------------------------------------------------- #
# 5. Experiment Runner
# --------------------------------------------------------------------------- #

def run_single_experiment(seed: int, epsilons_to_test: List[float], main_ds, tb_run_dir: str, npz_run_dir: str, epochs: int, use_scheduling: bool):
    print(f"\n{'=' * 20}  SEED {seed}  {'=' * 20}")
    
    # Create main training and test splits
    train_val_ds, test_ds = random_split(
        main_ds,
        [int(0.8 * N_TOTAL), N_TOTAL - int(0.8 * N_TOTAL)],
        generator=torch.Generator().manual_seed(seed) # Use seed for split
    )
    
    # Further split training data into training and validation
    train_size = int(0.9 * len(train_val_ds))
    val_size = len(train_val_ds) - train_size
    train_ds, val_ds = random_split(train_val_ds, [train_size, val_size], generator=torch.Generator().manual_seed(seed))

    # ------------------------------------------------------------------ #
    # Efficient DataLoaders
    # ------------------------------------------------------------------ #

    num_cpu_available = os.cpu_count() or 4
    effective_num_workers = (
        int(os.environ.get("SLURM_CPUS_PER_TASK", NUM_WORKERS))
        if "SLURM_CPUS_PER_TASK" in os.environ
        else NUM_WORKERS
    )
    effective_num_workers = min(num_cpu_available, int(effective_num_workers))

    train_dl = DataLoader(
        train_ds,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=effective_num_workers,
        pin_memory=True,
        persistent_workers=effective_num_workers > 0,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=TRAIN_BATCH_SIZE * 2,
        num_workers=effective_num_workers,
        pin_memory=True,
        persistent_workers=effective_num_workers > 0,
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=TRAIN_BATCH_SIZE * 2,
        num_workers=effective_num_workers,
        pin_memory=True,
        persistent_workers=effective_num_workers > 0,
    )

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bce = nn.BCEWithLogitsLoss()
    scaler = GradScaler()

    # --- Standard Model ---
    set_seeds(seed)
    standard_model = TinyCNN().to(dev)
    # Optional: use PyTorch 2.x dynamic compilation for ~10-20% speed-up
    if hasattr(torch, "compile"):
        try:
            standard_model = torch.compile(standard_model)
        except Exception:
            pass  # compile not supported on this environment
    opt_standard = torch.optim.Adam(standard_model.parameters(), lr=1e-3)
    std_writer = SummaryWriter(log_dir=os.path.join(tb_run_dir, "seed_{seed}", "standard"))
    std_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt_standard, T_max=epochs)
    std_early_stopper = EarlyStopping(patience=5, verbose=True)

    train_standard(standard_model, train_dl, val_dl, bce, opt_standard, dev, scaler, std_writer, std_scheduler, std_early_stopper, epochs=epochs)
    std_wiou, std_acc, std_auc, std_snr = evaluate_model(standard_model, test_dl, dev)
    std_writer.add_hparams(
        {'model': 'standard', 'epsilon': 0, 'seed': seed},
        {'hparam/accuracy': std_acc, 'hparam/wIoU': std_wiou, 'hparam/saliency_auc': std_auc, 'hparam/saliency_snr': std_snr}
    )
    std_writer.close()

    robust_wious, robust_accs, robust_aucs, robust_snrs = [], [], [], []
    for eps in epsilons_to_test:
        set_seeds(seed)
        mdl = TinyCNN().to(dev)
        if hasattr(torch, "compile"):
            try:
                mdl = torch.compile(mdl)
            except Exception:
                pass
        opt = torch.optim.Adam(mdl.parameters(), lr=1e-3)
        rob_writer = SummaryWriter(log_dir=os.path.join(tb_run_dir, f"seed_{seed}", f"eps_{eps:.4f}"))
        rob_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        rob_early_stopper = EarlyStopping(patience=5, verbose=True)
        
        if eps > 0:
            train_hotflip(mdl, train_dl, val_dl, bce, opt, dev, scaler, rob_writer, rob_scheduler, rob_early_stopper, 
                          max_flip_fraction=eps, epochs=epochs, use_scheduling=use_scheduling)
            wio, acc, auc, snr = evaluate_model(mdl, test_dl, dev)
            rob_writer.add_hparams(
                {'model': 'robust', 'epsilon': eps, 'seed': seed},
                {'hparam/accuracy': acc, 'hparam/wIoU': wio, 'hparam/saliency_auc': auc, 'hparam/saliency_snr': snr}
            )
            robust_wious.append(wio); robust_accs.append(acc); robust_aucs.append(auc); robust_snrs.append(snr)
        else: # Handle eps=0 case
            rob_writer.add_hparams(
                {'model': 'standard', 'epsilon': 0, 'seed': seed},
                {'hparam/accuracy': std_acc, 'hparam/wIoU': std_wiou, 'hparam/saliency_auc': std_auc, 'hparam/saliency_snr': std_snr}
            )
            robust_wious.append(std_wiou); robust_accs.append(std_acc); robust_aucs.append(std_auc); robust_snrs.append(std_snr)
        rob_writer.close()

    return std_wiou, std_acc, std_auc, std_snr, robust_wious, robust_accs, robust_aucs, robust_snrs

# --------------------------------------------------------------------------- #
# 6. Main entry-point
# --------------------------------------------------------------------------- #

def main(args):
    # Set the multiprocessing start method to be safer for shared file systems
    try:
        torch.multiprocessing.set_start_method('forkserver', force=True)
        print("Multiprocessing start method set to 'forkserver'.")
    except RuntimeError:
        print("Multiprocessing start method already set.")

    # Determine scheduling mode from SLURM task ID
    if args.task_id is None:
        print("Warning: --task_id not provided. Running BOTH scheduled and non-scheduled modes sequentially.")
        schedule_modes = [True, False]
    else:
        # Task 0 will be scheduled, Task 1 will be non-scheduled
        schedule_modes = [args.task_id == 0]

    for use_scheduling in schedule_modes:
        schedule_mode_str = "scheduled" if use_scheduling else "no_schedule"
        print(f"\n\n{'#'*80}")
        print(f"## RUNNING EXPERIMENT SET: Scheduling = {schedule_mode_str.upper()}")
        print(f"{'#'*80}\n")
        
        base_schedule_dir = os.path.join(args.output_dir, schedule_mode_str)

        # --- Outer loop for Hyperparameter Search ---
        for gc_hparam in GC_HPARAMS:
            for cons_hparam in CONS_HPARAMS:
                
                run_output_dir = os.path.join(base_schedule_dir, f"gc_{gc_hparam:.2f}_cons_{cons_hparam:.2f}")
                os.makedirs(run_output_dir, exist_ok=True)
                
                print(f"\n{'#'*60}")
                print(f"## Starting HP experiment: GC={gc_hparam:.2f}, Conservation={cons_hparam:.2f}")
                print(f"## Results will be saved to: {run_output_dir}")
                print(f"{'#'*60}\n")
                
                main_dataset = load_or_generate_dataset(gc_pos=gc_hparam, conservation=cons_hparam)
                
                all_std_wious, all_std_accs, all_std_aucs, all_std_snrs = [], [], [], []
                all_rob_wious, all_rob_accs, all_rob_aucs, all_rob_snrs = [], [], [], []

                for sd in SEEDS:
                    sw, sa, sa_auc, s_snr, rw, ra, ra_auc, r_snr = run_single_experiment(
                        sd, EPSILONS, main_dataset, run_output_dir, run_output_dir, args.epochs, use_scheduling
                    )
                    all_std_wious.append(sw); all_std_accs.append(sa); all_std_aucs.append(sa_auc); all_std_snrs.append(s_snr)
                    all_rob_wious.append(rw); all_rob_accs.append(ra); all_rob_aucs.append(ra_auc); all_rob_snrs.append(r_snr)

                # --- Save Raw Results ---
                np.savez(
                    os.path.join(run_output_dir, 'multi_seed_results.npz'),
                    epsilons=EPSILONS, seeds=SEEDS,
                    gc_pos=gc_hparam, conservation=cons_hparam,
                    std_wious=all_std_wious, std_accs=all_std_accs, std_aucs=all_std_aucs, std_snrs=all_std_snrs,
                    rob_wious=all_rob_wious, rob_accs=all_rob_accs, rob_aucs=all_rob_aucs, rob_snrs=all_rob_snrs
                )
                print(f"\nSaved raw results to {os.path.join(run_output_dir, 'multi_seed_results.npz')}")

    # --- Final Aggregation Across All Runs ---
    # This part is now handled exclusively by the --aggregate_only flag
    # which provides much better visualizations.
    print("\n\nFull sweep finished. To generate plots, run with --aggregate_only.")
    
    return


    ############################################################################
#                                Array-mode path                               #
    ############################################################################


# If --array_idx is supplied we execute **one** (schedule, gc, conservation)
# combination.  This allows efficient SLURM array jobs while limiting the
# number of concurrent GPUs via the %N syntax in the submission script.


def main_single_combo(args, array_idx: int):
    """Run experiments for a single combination determined by array_idx."""

    # Build mapping list once
    combos = []
    for schedule in [True, False]:
        for gc_val in GC_HPARAMS:
            for cons_val in CONS_HPARAMS:
                combos.append((schedule, gc_val, cons_val))

    if array_idx < 0 or array_idx >= len(combos):
        raise ValueError(
            f"array_idx {array_idx} is out of range (0-{len(combos) - 1})")

    use_scheduling, gc_hparam, cons_hparam = combos[array_idx]

    schedule_mode_str = "scheduled" if use_scheduling else "no_schedule"
    
    # Define distinct output paths for tensorboard and npz files
    tb_run_dir = os.path.join(args.output_dir, "tensorboard", schedule_mode_str, f"gc_{gc_hparam:.3f}_cons_{cons_hparam:.2f}")
    npz_run_dir = os.path.join(args.output_dir, "npz_results", schedule_mode_str, f"gc_{gc_hparam:.3f}_cons_{cons_hparam:.2f}")
    os.makedirs(tb_run_dir, exist_ok=True)
    os.makedirs(npz_run_dir, exist_ok=True)

    print(f"Running single-combo job: schedule={schedule_mode_str}, "
          f"GC={gc_hparam:.3f}, CONS={cons_hparam:.2f}")
    print(f"  - Tensorboard logs will be saved to: {tb_run_dir}")
    print(f"  - NPZ results will be saved to: {npz_run_dir}")

    main_dataset = load_or_generate_dataset(gc_pos=gc_hparam, conservation=cons_hparam)

    all_std_wious, all_std_accs, all_std_aucs, all_std_snrs = [], [], [], []
    all_rob_wious, all_rob_accs, all_rob_aucs, all_rob_snrs = [], [], [], []

    # Training parameters derived from CLI / environment
    global TRAIN_BATCH_SIZE, NUM_WORKERS
    TRAIN_BATCH_SIZE = args.batch_size if args.batch_size else DEFAULT_BATCH_SIZE
    NUM_WORKERS = args.num_workers

    for sd in SEEDS:
        sw, sa, sa_auc, s_snr, rw, ra, ra_auc, r_snr = run_single_experiment(
            sd,
            EPSILONS,
            main_dataset,
            tb_run_dir, # Pass the specific tensorboard directory
            npz_run_dir, # Pass the specific npz directory
            args.epochs,
            use_scheduling,
        )
        all_std_wious.append(sw)
        all_std_accs.append(sa)
        all_std_aucs.append(sa_auc)
        all_std_snrs.append(s_snr)
        all_rob_wious.append(rw)
        all_rob_accs.append(ra)
        all_rob_aucs.append(ra_auc)
        all_rob_snrs.append(r_snr)

    # Save raw results just like in the sweep mode
    np.savez(
        os.path.join(npz_run_dir, "multi_seed_results.npz"),
        epsilons=EPSILONS,
        seeds=SEEDS,
        gc_pos=gc_hparam,
        conservation=cons_hparam,
        std_wious=all_std_wious,
        std_accs=all_std_accs,
        std_aucs=all_std_aucs,
        std_snrs=all_std_snrs,
        rob_wious=all_rob_wious,
        rob_accs=all_rob_accs,
        rob_aucs=all_rob_aucs,
        rob_snrs=all_rob_snrs,
    )

    print("Single-combo job finished and results saved.")

    # We do **not** run the heavyweight plotting routines in array mode –
    # collect plots in a downstream aggregation step to save GPU time.

# --------------------------------------------------------------------------- #
# 7. Data-loading defaults (set after CLI parsing)
# --------------------------------------------------------------------------- #

TRAIN_BATCH_SIZE = DEFAULT_BATCH_SIZE
NUM_WORKERS = 4  # sensible default; overridden later

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run robustness experiment or aggregate results.")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to save results and plots.")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs.")
    parser.add_argument(
        "--task_id",
        type=int,
        default=None,
        help="LEGACY: Only used to pick schedule=True/False when not using --array_idx.",
    )
    parser.add_argument(
        "--array_idx",
        type=int,
        default=None,
        help="Index from SLURM_ARRAY_TASK_ID that selects (schedule, gc, cons) combo.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Training batch size (default 512).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="DataLoader workers (overrides SLURM_CPUS_PER_TASK if provided).",
    )
    parser.add_argument(
        "--aggregate_only",
        action="store_true",
        help="Skip training entirely and only generate plots from existing npz results.",
    )

    args = parser.parse_args()

    # Update global DataLoader defaults
    TRAIN_BATCH_SIZE = args.batch_size
    NUM_WORKERS = args.num_workers

    # ------------------------------------------------------------------ #
    # Aggregate-only fast path
    # ------------------------------------------------------------------ #

    if args.aggregate_only:
        print("Aggregate-only mode: generating combo plots and summary from existing results …")

        # All plots will be stored here
        plots_dir = os.path.join(args.output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        # --- Scan and plot per-combo ---
        npz_path = os.path.join(args.output_dir, "npz_results")
        npz_files = glob.glob(os.path.join(npz_path, '**', 'multi_seed_results.npz'), recursive=True)
        if not npz_files:
            print(f"No .npz result files found in {npz_path}; nothing to aggregate.")
            sys.exit(1)
        
        print(f"Found {len(npz_files)} result files, building master dataframe...")

        all_data = []
        for f_path in npz_files:
            try:
                data = np.load(f_path)
                # Extract HPs from path or data
                scheduling_mode = "scheduled" if "scheduled" in f_path else "no_schedule"
                gc_pos = float(data['gc_pos'])
                cons = float(data['conservation'])
                
                seeds = data['seeds']
                epsilons = data['epsilons']
                
                # metrics for std model
                std_wious = data['std_wious']
                std_accs = data['std_accs']
                std_aucs = data['std_aucs']
                std_snrs = data['std_snrs']
                
                # metrics for robust models
                rob_wious = data['rob_wious']
                rob_accs = data['rob_accs']
                rob_aucs = data['rob_aucs']
                rob_snrs = data['rob_snrs']
                
                for i, seed in enumerate(seeds):
                    # Add standard model data (epsilon = 0)
                    all_data.append({
                        'scheduling_mode': scheduling_mode, 'gc_pos': gc_pos, 'conservation': cons,
                        'seed': seed, 'epsilon': 0, 'wIoU': std_wious[i], 'Accuracy': std_accs[i],
                        'SaliencyAUC': std_aucs[i], 'SaliencySNR': std_snrs[i],
                    })
                    
                    for j, eps in enumerate(epsilons):
                        # Add robust model data
                        all_data.append({
                            'scheduling_mode': scheduling_mode, 'gc_pos': gc_pos, 'conservation': cons,
                            'seed': seed, 'epsilon': eps, 'wIoU': rob_wious[i, j], 'Accuracy': rob_accs[i, j],
                            'SaliencyAUC': rob_aucs[i, j], 'SaliencySNR': rob_snrs[i, j],
                        })
            except Exception as e:
                print(f"Could not process file {f_path}: {e}")

        df = pd.DataFrame(all_data)
        master_csv_path = os.path.join(plots_dir, 'full_results_long_format.csv')
        df.to_csv(master_csv_path, index=False)
        print(f"Saved master data table to {master_csv_path}")

        print("Generating summary distribution plots...")
        metrics = ['wIoU', 'Accuracy', 'SaliencyAUC', 'SaliencySNR']
        
        # --- Create Delta Plots ---
        print("Calculating delta metrics for focused plots...")
        baseline_metrics = df[df['epsilon'] == 0].set_index(
            ['scheduling_mode', 'gc_pos', 'conservation', 'seed']
        )[metrics].rename(columns=lambda c: f"{c}_base")
        
        df_merged = df.join(baseline_metrics, on=['scheduling_mode', 'gc_pos', 'conservation', 'seed'])
        
        for metric in metrics:
            df_merged[f'delta_{metric}'] = df_merged[metric] - df_merged[f'{metric}_base']
            
        df_deltas = df_merged[df_merged['epsilon'] > 0].copy()
        
        # --- Create Detailed Box Plots ---
        print("Generating detailed distribution plots...")
        metrics_to_plot = ['wIoU', 'Accuracy', 'SaliencyAUC', 'SaliencySNR']
        
        baseline_stats = df_merged.groupby(['scheduling_mode', 'gc_pos'])[[f'{m}_base' for m in metrics_to_plot]].agg(['mean', 'sem'])

        for metric in metrics_to_plot:
            delta_metric = f'delta_{metric}'
            base_metric = f'{metric}_base'
            print(f"  - Plotting {delta_metric} distributions...")

            is_snr = delta_metric == 'delta_SaliencySNR'
            
            g = sns.catplot(
                data=df_deltas,
                x="epsilon", y=delta_metric, hue="conservation",
                col="scheduling_mode", row="gc_pos",
                kind="box", height=2.5, aspect=2.2,
                palette='viridis', legend_out=True,
                fliersize=2.5,
                linewidth=1.0,
                sharey=False,
                showfliers=True
            )
            g.set_axis_labels("Epsilon", f"Improvement ({delta_metric})")
            g.fig.suptitle(f"Improvement in {metric} vs. Adversarial Epsilon", y=0.98, fontsize=16)

            mid_row_idx = g.axes.shape[0] // 2
            for i, ax in enumerate(g.axes[:, 0]):
                if i != mid_row_idx:
                    ax.set_ylabel("")

            for (gc, mode), ax in g.axes_dict.items():
                ax.axhline(0, ls='--', color='red', zorder=0)
                ax.tick_params(axis='x', labelrotation=45)
                
                mean_val = baseline_stats.loc[(mode, gc)][(f'{base_metric}', 'mean')]
                sem_val = baseline_stats.loc[(mode, gc)][(f'{base_metric}', 'sem')]
                ax.set_title(f"GC: {gc} | {mode}\n(Baseline {metric} ≈ {mean_val:.2f} ± {sem_val:.2f})", fontsize=10)
                
                ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6, prune='both'))

                if is_snr:
                    ax.set_yscale('symlog', linthresh=1.0)
                    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())

            g.fig.tight_layout(rect=[0, 0, 1, 0.93])
            sns.move_legend(g, "lower center", bbox_to_anchor=(.5, 0.93), ncol=3, title="Conservation", frameon=False)
            
            plot_path = os.path.join(plots_dir, f"summary_boxplot_{delta_metric}.png")
            g.savefig(plot_path, dpi=150)
            plt.close(g.fig)
            print(f"    Saved to {plot_path}")

        # --- Create Final Summary Bar Chart ---
        print("\nCreating final summary bar chart...")
        
        # Melt the dataframe to get a 'metric' column
        df_long = df_deltas.melt(
            id_vars=['scheduling_mode', 'gc_pos', 'conservation', 'epsilon', 'seed'],
            value_vars=[f'delta_{m}' for m in metrics],
            var_name='metric',
            value_name='delta_value'
        )

        # Calculate the mean improvement across all epsilons > 0
        mean_improvement_df = df_long.groupby(
            ['scheduling_mode', 'gc_pos', 'conservation', 'metric'],
            as_index=False
        )['delta_value'].mean()
        
        mean_improvement_df['metric'] = mean_improvement_df['metric'].str.replace('delta_', 'Δ')
        
        g = sns.catplot(
            data=mean_improvement_df,
            x='gc_pos',
            y='delta_value',
            hue='conservation',
            col='scheduling_mode',
            row='metric',
            kind='bar',
            height=3,
            aspect=2.5,
            legend_out=True,
            sharey=False, 
            palette='viridis'
        )
        
        g.set_axis_labels("GC Content (Confounder Strength)", "Mean Improvement")
        g.set_titles(row_template="{row_name}", col_template="{col_name}")
        g.fig.suptitle("Mean Interpretability Improvement from Adversarial Training", y=1.03, fontsize=16)

        for ax in g.axes.flat:
            ax.axhline(0, ls='--', color='gray', zorder=0)

        sns.move_legend(g, "upper right", bbox_to_anchor=(0.95, 0.9), title="Conservation")
        
        g.fig.tight_layout(rect=[0, 0, 1, 0.96])
        
        overview_path = os.path.join(plots_dir, "summary_barchart_mean_improvement.png")
        g.savefig(overview_path, dpi=150)
        plt.close(g.fig)
        print(f"Saved final summary plot to {overview_path}")

        sys.exit(0)

    if args.array_idx is not None:
        main_single_combo(args, args.array_idx)
    else:
        main(args) 