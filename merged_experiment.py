"""
Merged experiment: Complex multi-block data generation with advanced training infrastructure
Combines clean_real.py's sophisticated data scheme with toy_slurm.py's modern experimental setup
Author: Merged version, 2025

Key changes from toy_slurm.py:
- Adopted multi-block data generation from clean_real.py
- Replaced gc_pos with gc_gap parameter space
- Removed wIoU metric (not applicable to multi-signal data)
- Updated hyperparameter grids to match clean_real.py
- Kept GPU optimizations and advanced training methods
"""

import itertools
import math
import os
import random
import string
import argparse
import sys
from typing import List, Tuple, Dict

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

try:
    import logomaker
except ImportError:
    print("Warning: logomaker is not installed. Motif visualization will be disabled.")
    logomaker = None

# --------------------------------------------------------------------------- #
# 1. Configuration & Utilities
# --------------------------------------------------------------------------- #

# --- Data Generation Settings (from clean_real.py) ---
SEQ_LEN                = 1000
BLOCKS_RANGE           = (3, 4)
BLOCK_LEN_MEAN         = 55
BLOCK_LEN_SD           = 15
BLOCK_LEN_MIN, BLOCK_LEN_MAX = 40, 70
PROMOTER_HEX_1, PROMOTER_HEX_2 = "TTGACA", "TATAAT"
PROMOTER_SPACER        = 17
MIN_GAP_BETWEEN_BLOCKS = 30
DEFAULT_MOTIF_REPERTOIRE = 30
N_ANCESTORS            = DEFAULT_MOTIF_REPERTOIRE
N_TOTAL                = 10000 
TARGET_SIGNAL_FRAC     = 0.20
DEFAULT_BATCH_SIZE     = 512
DEFAULT_EPOCHS         = 50

# --- Hyperparameter Search Space (from clean_real.py) ---
# New 3x3 grid based on gc_gap
GC_GAP_HPARAMS = [0.0, 0.03, 0.05, 0.1, 0.15, 0.2]
CONS_HPARAMS = [0.6, 0.7, 0.8, 0.9]

# Removed randomized smoothing - now only comparing hotflip variants

# Training parameters
SEEDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
EPSILONS = [0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.10, 0.15]

# Directory to cache synthetic datasets
DATASET_CACHE_DIR = "dataset_cache_merged"

# GPU Prefetch settings
PREFETCH_FACTOR = 4

# --------------------------------------------------------------------------- #
# 1.a. Basic utilities
# --------------------------------------------------------------------------- #

class GPUPrefetchDataLoader:
    """GPU prefetching wrapper around a standard DataLoader"""
    def __init__(self, dataloader, device):
        self.dataloader = dataloader
        self.device = device
        
    def __iter__(self):
        stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        first = True
        
        for next_batch in self.dataloader:
            if stream is not None:
                with torch.cuda.stream(stream):
                    # Transfer next batch to GPU asynchronously
                    next_batch = [item.to(self.device, non_blocking=True) if isinstance(item, torch.Tensor) else item 
                                  for item in next_batch]
            else:
                next_batch = [item.to(self.device) if isinstance(item, torch.Tensor) else item 
                              for item in next_batch]
            
            if not first:
                # Return the previous batch while next is being prepared
                yield current_batch
            else:
                first = False
                
            if stream is not None:
                # Ensure transfer completes before reassigning
                torch.cuda.current_stream().wait_stream(stream)
            current_batch = next_batch
            
        # Don't forget the last batch
        if not first:
            yield current_batch
            
    def __len__(self):
        return len(self.dataloader)

    def __getattr__(self, name):
        """Forward attribute lookups to the wrapped dataloader."""
        return getattr(self.dataloader, name)

def set_seeds(seed_value: int = 42, deterministic: bool = False) -> None:
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

def log_gpu_stats(prefix=""):
    """Log current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"{prefix} GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

set_seeds(42)  # initial seed for consistency

ALPH = np.array(list("ACGT"), dtype="U1")
to_ix = {b: i for i, b in enumerate(ALPH)}

def sample_background(length: int, gc: float) -> np.ndarray:
    p = np.array([(1 - gc) / 2, gc / 2, gc / 2, (1 - gc) / 2])
    return np.random.choice(ALPH, size=length, p=p)

def mutate(chunk: np.ndarray, conservation: float, gc_target: float) -> np.ndarray:
    """Return a new chunk with given conservation level, with mutations
    sampled to match the target GC content."""
    mutated_chunk = chunk.copy()
    n_to_mutate = int(len(chunk) * (1.0 - conservation))
    pos_to_mutate = np.random.choice(len(chunk), n_to_mutate, replace=False)
    
    # Distribution for sampling new bases
    p = np.array([(1 - gc_target) / 2, gc_target / 2, gc_target / 2, (1 - gc_target) / 2])
    
    for pos in pos_to_mutate:
        original_base = mutated_chunk[pos]
        
        # Create a temporary probability distribution for the other 3 bases
        temp_p = p.copy()
        temp_p[to_ix[original_base]] = 0
        
        # If the sum is zero (shouldn't happen with 4 bases), fall back to uniform
        if temp_p.sum() == 0:
            mutated_chunk[pos] = np.random.choice(np.setdiff1d(ALPH, [original_base]))
        else:
            temp_p /= temp_p.sum()
            mutated_chunk[pos] = np.random.choice(ALPH, p=temp_p)

    return mutated_chunk

def one_hot(seq: np.ndarray) -> np.ndarray:
    """(L,) char → (4,L) float32 one-hot"""
    arr = np.zeros((4, len(seq)), dtype=np.float32)
    for i, b in enumerate(seq):
        arr[to_ix[b], i] = 1.0
    return arr

def one_hot_to_seq(one_hot_tensor: torch.Tensor) -> str:
    """ (4, L) float tensor -> (L,) string """
    indices = torch.argmax(one_hot_tensor, dim=0).cpu().numpy()
    return "".join(ALPH[indices])

# Removed concentration_from_epsilon - only used for randomized smoothing

# --------------------------------------------------------------------------- #
# 1.b. Data generation helpers (from clean_real.py)
# --------------------------------------------------------------------------- #

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
# 1.c. Visualization helpers
# --------------------------------------------------------------------------- #

def log_conv1_motifs(
    model: nn.Module,
    writer: SummaryWriter,
    epoch: int,
    output_dir: str = None,
    save_to_disk: bool = False,
    log_to_tb: bool = True,
):
    """
    Generates sequence logos from the first conv layer's weights
    """
    if logomaker is None:
        return
        
    # Ensure model is on CPU for weight processing
    if hasattr(model, '_orig_mod'):
        weights = model._orig_mod.conv1.weight.detach().cpu()
    else:
        weights = model.conv1.weight.detach().cpu()

    pwms = F.softmax(weights, dim=1)
    
    # Create a list of pandas DataFrames for logomaker
    pwms_dfs = []
    for i in range(pwms.shape[0]):
        pwm_numpy = pwms[i].numpy().T
        df = pd.DataFrame(pwm_numpy, columns=list("ACGT"))
        pwms_dfs.append(df)

    # Save to disk if requested
    if save_to_disk and output_dir:
        logo_dir = os.path.join(output_dir, "motif_logos")
        os.makedirs(logo_dir, exist_ok=True)
        print(f"  Saving motif logos to {logo_dir}...")
        for i, df in enumerate(pwms_dfs):
            fig, ax = plt.subplots(1, 1, figsize=(max(df.shape[0] * 0.4, 2), 2.5))
            logo = logomaker.Logo(df, ax=ax)
            logo.style_spines(visible=False)
            logo.style_spines(spines=['left', 'bottom'], visible=True)
            ax.set_ylabel('Probability')
            ax.set_yticks([0, 0.5, 1.0])
            ax.set_ylim(0, 1)
            ax.set_title(f'Filter {i}')
            fig.tight_layout()
            plt.savefig(os.path.join(logo_dir, f"filter_{i}.png"), dpi=300)
            plt.close(fig)

    # Log to TensorBoard if requested
    if log_to_tb:
        num_filters = len(pwms_dfs)
        cols = 6
        rows = math.ceil(num_filters / cols)
        
        fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 2.5 * rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        if cols == 1:
            axes = axes.reshape(-1, 1)
            
        axes = axes.flatten()

        for i in range(num_filters):
            plt.sca(axes[i])
            logo = logomaker.Logo(pwms_dfs[i], ax=axes[i])
            logo.style_spines(visible=False)
            logo.style_spines(spines=['left', 'bottom'], visible=True)
            axes[i].set_title(f"Filter {i}")
            axes[i].set_xticks([])
            axes[i].set_yticks([0, 0.5, 1.0])
            axes[i].set_ylim(0, 1)
            if i % cols == 0:
                axes[i].set_ylabel("Probability")
        
        # Hide unused subplots
        for i in range(num_filters, len(axes)):
            axes[i].axis('off')
            
        fig.suptitle("First Convolutional Layer Motifs", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        writer.add_figure("Conv1_motifs", fig, global_step=epoch)
        writer.flush()
        plt.close(fig)

# --------------------------------------------------------------------------- #
# 1.d. Dataset caching helpers
# --------------------------------------------------------------------------- #

def _dataset_cache_path(gc_gap: float, conservation: float) -> str:
    """Return cache file path for a given (gc_gap, conservation)."""
    return os.path.join(
        DATASET_CACHE_DIR,
        f"v3_gc-gap_{gc_gap:.3f}_cons_{conservation:.3f}.npz",
    )

def load_or_generate_dataset(gc_gap: float, conservation: float):
    """Load dataset from cache if present, otherwise generate and cache it."""
    os.makedirs(DATASET_CACHE_DIR, exist_ok=True)

    cache_path = _dataset_cache_path(gc_gap, conservation)
    if os.path.exists(cache_path):
        print(f"Loading cached dataset from {cache_path}")
        data = np.load(cache_path)
        X = torch.tensor(data["X"], dtype=torch.float32)
        y = torch.tensor(data["y"], dtype=torch.float)
        masks = data["masks"]
        return SeqDS(X, y, masks)

    print(f"Generating dataset with GC_gap={gc_gap:.3f} and conservation={conservation:.2f}...")

    ds, _ = generate_dataset(
        gc_gap=gc_gap,
        conservation=conservation,
        target_signal_frac=TARGET_SIGNAL_FRAC,
        motif_repertoire=DEFAULT_MOTIF_REPERTOIRE,
        include_partial_negatives=True,
    )

    # Atomic saving to prevent race conditions
    temp_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=10))
    temp_path = f"{cache_path}.{temp_id}.tmp"
    
    try:
        np.savez(
            temp_path,
            X=ds.x.cpu().numpy(),
            y=ds.y.cpu().numpy(),
            masks=ds.m,
        )
        os.rename(f"{temp_path}.npz", cache_path)
    except Exception as e:
        print(f"Error caching dataset: {e}. Cleaning up temp file...")
        final_temp_path = f"{temp_path}.npz"
        if os.path.exists(final_temp_path):
            os.remove(final_temp_path)
        raise
        
    return ds

# --------------------------------------------------------------------------- #
# 2. Dataset generation (from clean_real.py)
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
    
    This function is migrated from clean_real.py with multi-block design.
    """
    
    # Sample budget
    POS_N = N_TOTAL // 2
    NEG_N = N_TOTAL - POS_N

    decoy_neg_n = int(0.2 * NEG_N) if include_partial_negatives else 0
    promoter_only_neg_n = int(0.2 * NEG_N)
    std_neg_n = NEG_N - decoy_neg_n
    background_only_neg_n = std_neg_n - promoter_only_neg_n

    # Build ancestral pool
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
        n_blocks_low = int(np.floor(n_blocks_mean))
        n_blocks_high = int(np.ceil(n_blocks_mean))
        if n_blocks_low == n_blocks_high:
            n_blocks_high += 1
        n_blocks = np.random.randint(max(3, n_blocks_low), max(4, n_blocks_high))

        blk_lens = _trunc_norm(BLOCK_LEN_MEAN, BLOCK_LEN_SD, BLOCK_LEN_MIN, BLOCK_LEN_MAX, n_blocks)
        try:
            blk_starts = _nonoverlap_positions(SEQ_LEN, blk_lens)
        except RuntimeError:
            continue

        mask = np.zeros(SEQ_LEN, dtype=bool)
        for blen, start in zip(blk_lens, blk_starts):
            ancestor = random.choice(ancestral_pool)
            master = ancestor[:blen]
            chunk = mutate(master, conservation, gc_target=current_gc_pos)
            bg[start:start+blen] = chunk
            mask[start:start+blen] = True

        # Mandatory σ70-like promoter
        promoter_full_len = len(PROMOTER_HEX_1) + PROMOTER_SPACER + len(PROMOTER_HEX_2)
        first_start = min(blk_starts)
        if first_start < (promoter_full_len + MIN_GAP_BETWEEN_BLOCKS):
            continue

        prom_seq = build_promoter(current_gc_pos)
        prom_pos = first_start - promoter_full_len - MIN_GAP_BETWEEN_BLOCKS
        if prom_pos < 0:
            continue

        bg[prom_pos: prom_pos + promoter_full_len] = prom_seq
        mask[prom_pos: prom_pos + promoter_full_len] = True

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
        blk_lens = _trunc_norm(BLOCK_LEN_MEAN, BLOCK_LEN_SD, BLOCK_LEN_MIN, BLOCK_LEN_MAX, n_blocks)
        try:
            blk_starts = _nonoverlap_positions(SEQ_LEN, blk_lens)
        except RuntimeError:
            X.append(one_hot(sample_background(SEQ_LEN, gc=current_gc)))
            y.append(0)
            masks.append(np.zeros(SEQ_LEN, dtype=bool))
            continue

        mask = np.zeros(SEQ_LEN, dtype=bool)
        for blen, start in zip(blk_lens, blk_starts):
            ancestor = random.choice(ancestral_pool)
            master = ancestor[:blen]
            chunk = mutate(master, conservation, gc_target=current_gc)
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
        bg = sample_background(SEQ_LEN, gc=bg_gc)
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
        "n_background_only_negative": background_only_neg_n,
        "motif_repertoire": motif_repertoire,
        "seed": np.random.get_state()[1][0].item() if len(np.random.get_state()[1]) > 0 else -1,
        "target_signal_frac": target_signal_frac,
        "avg_realised_frac": avg_realised_frac,
    }

    return SeqDS(X, y, masks), summary 

# --------------------------------------------------------------------------- #
# 3. Model Architectures
# --------------------------------------------------------------------------- #

class TinyCNNv0(nn.Module):
    """Original simple architecture for backwards compatibility"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(4, 32, 13, padding=6)
        self.conv2 = nn.Conv1d(32, 64, 7, padding=3)
        self.conv3 = nn.Conv1d(64, 128, 7, padding=3)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x)); x = F.max_pool1d(x, 2)
        x = F.relu(self.conv2(x)); x = F.max_pool1d(x, 2)
        x = F.relu(self.conv3(x)); x = F.max_pool1d(x, 2)
        x = self.pool(x).squeeze(-1)
        logits = self.fc(x)
        return logits.squeeze(-1), x

class TinyCNN(nn.Module):
    """Modern architecture with batch norm and localist pooling"""
    def __init__(self):
        super().__init__()
        # User-specified kernel sizes
        self.k1, self.k2, self.k3 = 30, 3, 3

        # Calculate padding to keep sequence length constant
        p1 = (self.k1 - 1) // 2
        p2 = (self.k2 - 1) // 2
        p3 = (self.k3 - 1) // 2

        # Conv Block 1
        self.conv1 = nn.Conv1d(4, 32, kernel_size=self.k1, padding=p1)
        self.bn1 = nn.BatchNorm1d(32)
        self.dropout1 = nn.Dropout(0.1)

        # Conv Block 2
        self.conv2 = nn.Conv1d(32, 64, kernel_size=self.k2, padding=p2)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.1)

        # Conv Block 3
        self.conv3 = nn.Conv1d(64, 128, kernel_size=self.k3, padding=p3)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(0.1)

        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc_dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        # Conv Block 1: Motif scanning
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.exp(x)  # Exponential activation
        x = self.dropout1(x)
        
        # Localist pooling
        x = F.max_pool1d(x, 50)

        # Conv Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        # Conv Block 3
        conv3_out = self.conv3(x)
        x = self.bn3(conv3_out)
        x = F.relu(x)
        x = self.dropout3(x)

        # FC Layer
        x = self.pool(x).squeeze(-1)
        x = self.fc_dropout(x)
        logits = self.fc(x)
        return logits.squeeze(-1), conv3_out
    
    def receptive_field(self) -> int:
        """Return the receptive field size"""
        return self.k1

# --------------------------------------------------------------------------- #
# 4. Training Utilities
# --------------------------------------------------------------------------- #

def validate_epoch(model, loader, loss_fn, dev):
    """Calculates the loss on a validation set."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        for xb, yb, _ in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            with autocast():
                logits, _ = model(xb)
                loss = loss_fn(logits, yb)
            total_loss += loss.item()
            num_batches += 1
    return total_loss / num_batches if num_batches > 0 else 0

def train_standard(model, train_loader, val_loader, loss_fn, optimizer, dev, scaler, writer, scheduler, 
                   epochs: int = 10, early_stopping_patience: int = 15, early_stopping_min_delta: float = 1e-4) -> None:
    """Standard training loop with early stopping"""
    print(f"Starting standard training with early stopping (patience={early_stopping_patience}, min_delta={early_stopping_min_delta})...")
    best_val_loss = float('inf')
    early_stopping_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0
        for xb, yb, _ in train_loader:
            xb, yb = xb.to(dev), yb.to(dev)
            optimizer.zero_grad()
            with autocast():
                logits, _ = model(xb)
                loss = loss_fn(logits, yb)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            num_batches += 1
        
        avg_train_loss = total_loss / num_batches if num_batches > 0 else 0
        avg_val_loss = validate_epoch(model, val_loader, loss_fn, dev)

        scheduler.step(avg_val_loss)
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/validation', avg_val_loss, epoch)
        writer.add_scalar('LR/train', scheduler.optimizer.param_groups[0]['lr'], epoch)
        print(f"  Epoch {epoch + 1}/{epochs}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Check for improvement
        if (best_val_loss - avg_val_loss) > early_stopping_min_delta:
            best_val_loss = avg_val_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
        
        if early_stopping_counter >= early_stopping_patience:
            print(f"  -> Early stopping at epoch {epoch + 1}")
            break

# Removed randomized smoothing functions - only hotflip variants remain

# --------------------------------------------------------------------------- #
# 5. Adversarial Training (Advanced HotFlip from toy_slurm.py)
# --------------------------------------------------------------------------- #

def generate_hotflip_examples_optimized(model, xb, yb, loss_fn, flip_fraction: float, 
                                       neighborhood_size: int = 20, penalize_nearby: bool = False):
    """Optimized HotFlip with optional neighborhood penalty"""
    seq_len = xb.shape[2]
    k_flips = int(flip_fraction * seq_len)
    adv_xb = xb.clone()
    
    flipped_positions = set()
    
    for flip_idx in range(k_flips):
        adv_xb.requires_grad = True
        model.zero_grad()
        
        with autocast():
            logits, _ = model(adv_xb)
            loss = loss_fn(logits, yb)
        
        loss.backward()
        grad = adv_xb.grad.data
        
        # Compute saliency scores
        current_bases_onehot = (adv_xb > 0.5).float()
        grad_at_current_bases = (grad * current_bases_onehot).sum(dim=1, keepdim=True)
        saliency_scores = grad - grad_at_current_bases
        saliency_scores.masked_fill_(current_bases_onehot.bool(), -1e9)
        
        # Apply neighborhood penalty if requested
        if penalize_nearby and flipped_positions:
            penalty_mask = torch.zeros(xb.shape[0], seq_len, device=xb.device)
            for pos in flipped_positions:
                start = max(0, pos - neighborhood_size)
                end = min(seq_len, pos + neighborhood_size + 1)
                penalty_mask[:, start:end] = 1.0
            
            penalty_strength = 0.5 * (flip_idx / k_flips)
            saliency_scores -= penalty_strength * penalty_mask.unsqueeze(1) * saliency_scores.abs().max()
        
        # Find best flip
        best_flip_scores_per_pos, _ = saliency_scores.max(dim=1)
        best_pos_to_flip = best_flip_scores_per_pos.argmax(dim=1)
        best_new_base_idx = saliency_scores[range(len(xb)), :, best_pos_to_flip].argmax(dim=1)
        
        # Apply flip
        old_base_idx = adv_xb[range(len(xb)), :, best_pos_to_flip].argmax(dim=1)
        adv_xb = adv_xb.detach()
        adv_xb[range(len(xb)), old_base_idx, best_pos_to_flip] = 0.0
        adv_xb[range(len(xb)), best_new_base_idx, best_pos_to_flip] = 1.0
        
        # Track flipped positions
        for pos in best_pos_to_flip.cpu().numpy():
            flipped_positions.add(int(pos))
    
    return adv_xb

def generate_direct_hotflip_examples_optimized(model, xb, yb, loss_fn, flip_fraction: float):
    """Direct HotFlip implementation optimized for batch processing"""
    seq_len = xb.shape[2]
    k_flips = int(flip_fraction * seq_len)
    adv_xb = xb.clone()
    
    # Pre-allocate tensors
    batch_size = xb.shape[0]
    batch_indices = torch.arange(batch_size, device=xb.device)
    
    for _ in range(k_flips):
        adv_xb.requires_grad = True
        model.zero_grad()
        
        with autocast():
            logits, _ = model(adv_xb)
            loss = loss_fn(logits, yb)
        
        loss.backward()
        grad = adv_xb.grad.data
        
        # Vectorized computation of saliency scores
        current_bases_mask = (adv_xb > 0.5)
        grad_masked = grad.masked_fill(~current_bases_mask, 0)
        grad_at_current = grad_masked.sum(dim=1, keepdim=True)
        
        saliency_scores = grad - grad_at_current
        saliency_scores.masked_fill_(current_bases_mask, -1e9)
        
        # Batch-wise argmax
        saliency_flat = saliency_scores.view(batch_size, -1)
        best_flip_flat = saliency_flat.argmax(dim=1)
        
        # Convert flat indices to (base, position)
        best_new_base_idx = best_flip_flat // seq_len
        best_pos_to_flip = best_flip_flat % seq_len
        
        # Vectorized flip application
        old_base_idx = adv_xb[batch_indices, :, best_pos_to_flip].argmax(dim=1)
        
        adv_xb = adv_xb.detach()
        adv_xb[batch_indices, old_base_idx, best_pos_to_flip] = 0.0
        adv_xb[batch_indices, best_new_base_idx, best_pos_to_flip] = 1.0
    
    return adv_xb 

def train_hotflip(model, train_loader, val_loader, loss_fn, optimizer, dev, scaler, writer, scheduler,
                  max_flip_fraction: float, epochs: int = 10, use_scheduling: bool = True, 
                  early_stopping_patience: int = 25, early_stopping_min_delta: float = 1e-4, 
                  gc_gap: float = 0.1) -> None:
    """HotFlip adversarial training with optional scheduling"""
    scheduling_str = "ON" if use_scheduling else "OFF"
    print(f"Starting HotFlip training with max_flip_fraction = {max_flip_fraction:.4f}, Scheduling: {scheduling_str}")
    
    previous_val_loss = float('inf')
    early_stopping_counter = 0

    for epoch in range(epochs):
        max_flips = int(max_flip_fraction * SEQ_LEN)
        
        if use_scheduling:
            # Linearly ramp up from 1 flip to max_flips
            num_flips = 1 + int(np.floor((max_flips - 1) * (epoch / (epochs - 1)))) if epochs > 1 else max_flips
            current_flip_fraction = num_flips / SEQ_LEN
        else:
            current_flip_fraction = max_flip_fraction
        
        model.train()
        total_loss = 0.0
        num_batches = 0
        for xb, yb, _ in train_loader:
            xb, yb = xb.to(dev), yb.to(dev)
            adv_xb = generate_hotflip_examples_optimized(model, xb, yb, loss_fn, current_flip_fraction)
            optimizer.zero_grad()
            with autocast():
                logits_adv, _ = model(adv_xb)
                loss_adv = loss_fn(logits_adv, yb)

            scaler.scale(loss_adv).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss_adv.item()
            num_batches += 1

        avg_train_loss = total_loss / num_batches if num_batches > 0 else 0
        avg_val_loss = validate_epoch(model, val_loader, loss_fn, dev)
        
        if use_scheduling:
            scheduler.best = previous_val_loss

        scheduler.step(avg_val_loss)
        writer.add_scalar('Loss/train_adversarial', avg_train_loss, epoch)
        writer.add_scalar('Loss/validation_adversarial', avg_val_loss, epoch)
        writer.add_scalar('LR/train_adversarial', scheduler.optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('Epsilon/train_adversarial', current_flip_fraction, epoch)
        print(f"  Epoch {epoch + 1}/{epochs}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Epsilon: {current_flip_fraction:.4f}")

        # Early stopping
        if (previous_val_loss - avg_val_loss) > early_stopping_min_delta:
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
        
        previous_val_loss = avg_val_loss
        
        if early_stopping_counter >= early_stopping_patience:
            print(f"  -> Early stopping at epoch {epoch + 1}")
            break

def train_direct_hotflip(model, train_loader, val_loader, loss_fn, optimizer, dev, scaler, writer, scheduler,
                         max_flip_fraction: float, epochs: int = 10, use_scheduling: bool = True, 
                         early_stopping_patience: int = 25, early_stopping_min_delta: float = 1e-4, 
                         gc_gap: float = 0.1) -> None:
    """Direct HotFlip training (optimized version)"""
    scheduling_str = "ON" if use_scheduling else "OFF"
    print(f"Starting Direct HotFlip training with max_flip_fraction = {max_flip_fraction:.4f}, Scheduling: {scheduling_str}")
    
    previous_val_loss = float('inf')
    early_stopping_counter = 0

    for epoch in range(epochs):
        max_flips = int(max_flip_fraction * SEQ_LEN)
        
        if use_scheduling:
            num_flips = 1 + int(np.floor((max_flips - 1) * (epoch / (epochs - 1)))) if epochs > 1 else max_flips
            current_flip_fraction = num_flips / SEQ_LEN
        else:
            current_flip_fraction = max_flip_fraction
        
        model.train()
        total_loss = 0.0
        num_batches = 0
        for xb, yb, _ in train_loader:
            xb, yb = xb.to(dev), yb.to(dev)
            adv_xb = generate_direct_hotflip_examples_optimized(model, xb, yb, loss_fn, current_flip_fraction)
            optimizer.zero_grad()
            with autocast():
                logits_adv, _ = model(adv_xb)
                loss_adv = loss_fn(logits_adv, yb)

            scaler.scale(loss_adv).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss_adv.item()
            num_batches += 1

        avg_train_loss = total_loss / num_batches if num_batches > 0 else 0
        avg_val_loss = validate_epoch(model, val_loader, loss_fn, dev)
        
        if use_scheduling:
            scheduler.best = previous_val_loss

        scheduler.step(avg_val_loss)
        writer.add_scalar('Loss/train_direct_adversarial', avg_train_loss, epoch)
        writer.add_scalar('Loss/validation_direct_adversarial', avg_val_loss, epoch)
        writer.add_scalar('LR/train_direct_adversarial', scheduler.optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('Epsilon/train_direct_adversarial', current_flip_fraction, epoch)
        print(f"  Epoch {epoch + 1}/{epochs}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Epsilon: {current_flip_fraction:.4f}")

        # Early stopping
        if (previous_val_loss - avg_val_loss) > early_stopping_min_delta:
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
        
        previous_val_loss = avg_val_loss
        
        if early_stopping_counter >= early_stopping_patience:
            print(f"  -> Early stopping at epoch {epoch + 1}")
            break

# --------------------------------------------------------------------------- #
# 6. Evaluation Functions  
# --------------------------------------------------------------------------- #

def find_adversarial_baseline_pgd(model, xb: torch.Tensor, yb: torch.Tensor, dev: torch.device,
                                  num_iter: int = 20, epsilon: float = 0.1, step_size: float = 0.01):
    """Find adversarial baseline using PGD for Integrated Gradients"""
    adv_xb = xb.clone().detach()
    stats = {
        'success': False,
        'initial_logit': 0.0,
        'final_logit': 0.0,
        'found_at_iter': num_iter,
        'initial_prediction_correct': False
    }

    with torch.no_grad(), autocast():
        initial_logits, _ = model(adv_xb)
        initial_pred_class = (initial_logits > 0).float()
        stats['initial_logit'] = initial_logits.item()
        stats['final_logit'] = initial_logits.item()

    is_correct = initial_pred_class.item() == yb.item()
    stats['initial_prediction_correct'] = is_correct

    # Only run PGD for correct positive predictions
    if not is_correct or yb.item() == 0:
        return torch.zeros_like(xb, device=dev), stats

    loss_fn = nn.BCEWithLogitsLoss()
    step_size = epsilon / 10.0

    for i in range(num_iter):
        adv_xb.requires_grad = True
        with autocast():
            logits, _ = model(adv_xb)
            loss = loss_fn(logits, yb.expand_as(logits))
        
        model.zero_grad()
        loss.backward()
        
        with torch.no_grad():
            grad = adv_xb.grad.data
            adv_xb = adv_xb + step_size * grad.sign()
            
            delta = adv_xb - xb
            delta = torch.clamp(delta, -epsilon, epsilon)
            adv_xb = torch.clamp(xb + delta, 0, 1)

            current_logits, _ = model(adv_xb)
            current_pred_class = (current_logits > 0).float()
            
            if current_pred_class.item() != initial_pred_class.item():
                stats['success'] = True
                stats['final_logit'] = current_logits.item()
                stats['found_at_iter'] = i + 1
                return adv_xb.detach(), stats
    
    with torch.no_grad():
        final_logits, _ = model(adv_xb)
        stats['final_logit'] = final_logits.item()

    return torch.zeros_like(xb, device=dev), stats

def find_adversarial_baseline_pgd_batch_optimized(model, xb_batch: torch.Tensor, yb_batch: torch.Tensor, dev: torch.device,
                                                  num_iter: int = 20, epsilon: float = 0.1):
    """GPU-optimized batched PGD that minimizes synchronization."""
    batch_size = xb_batch.shape[0]
    adv_xb_batch = xb_batch.clone().detach()
    
    # Initialize stats - we'll update them at the end to minimize synchronization
    with torch.no_grad(), autocast():
        initial_logits, _ = model(adv_xb_batch)
        initial_pred_classes = (initial_logits > 0).float()
    
    # Determine which samples to attack (positive samples with correct predictions)
    is_correct = (initial_pred_classes == yb_batch)
    is_positive = (yb_batch == 1)
    active_mask = is_correct & is_positive
    
    # Early return if no samples need attacking
    if not active_mask.any():
        # Create stats with minimal synchronization
        stats_list = []
        for i in range(batch_size):
            stats_list.append({
                'success': False,
                'initial_logit': initial_logits[i].item(),
                'final_logit': initial_logits[i].item(),
                'found_at_iter': num_iter,
                'initial_prediction_correct': is_correct[i].item()
            })
        return torch.zeros_like(xb_batch, device=dev), stats_list
    
    loss_fn = nn.BCEWithLogitsLoss(reduction='none')
    step_size = epsilon / 10.0
    
    # Track success without synchronization
    success_mask = torch.zeros(batch_size, dtype=torch.bool, device=dev)
    success_iter = torch.full((batch_size,), num_iter, dtype=torch.long, device=dev)
    final_baselines = torch.zeros_like(xb_batch, device=dev)
    
    for iter_idx in range(num_iter):
        if not active_mask.any():
            break
        
        # Compute gradients only for active samples
        active_xb = adv_xb_batch[active_mask].detach().requires_grad_(True)
        
        with autocast():
            active_logits, _ = model(active_xb)
            active_labels = yb_batch[active_mask]
            losses = loss_fn(active_logits, active_labels)
            loss = losses.mean()
        
        model.zero_grad()
        loss.backward()
        
        # Update adversarial examples
        with torch.no_grad():
            grad_sign = active_xb.grad.sign()
            active_xb_new = active_xb + step_size * grad_sign
            
            # Get indices of active samples
            active_indices = torch.where(active_mask)[0]
            
            # Vectorized projection back to epsilon ball
            for j, idx in enumerate(active_indices):
                delta = active_xb_new[j] - xb_batch[idx]
                delta = torch.clamp(delta, -epsilon, epsilon)
                adv_xb_batch[idx] = torch.clamp(xb_batch[idx] + delta, 0, 1)
            
            # Check for successful flips
            current_logits, _ = model(adv_xb_batch[active_mask])
            current_pred_classes = (current_logits > 0).float()
            
            # Find newly successful attacks
            flip_occurred = (current_pred_classes != initial_pred_classes[active_mask])
            
            # Update success tracking
            for j, idx in enumerate(active_indices):
                if flip_occurred[j] and not success_mask[idx]:
                    success_mask[idx] = True
                    success_iter[idx] = iter_idx + 1
                    final_baselines[idx] = adv_xb_batch[idx].clone()
                    active_mask[idx] = False
    
    # Compute final logits
    with torch.no_grad():
        final_logits, _ = model(adv_xb_batch)
    
    # Create stats with single synchronization at the end
    stats_list = []
    for i in range(batch_size):
        stats_list.append({
            'success': success_mask[i].item(),
            'initial_logit': initial_logits[i].item(),
            'final_logit': final_logits[i].item() if success_mask[i] else final_logits[i].item(),
            'found_at_iter': success_iter[i].item(),
            'initial_prediction_correct': is_correct[i].item()
        })
    
    return final_baselines, stats_list

def evaluate_model(model, test_dl, dev, pgd_cache=None):
    """Evaluate model with batched saliency metrics"""
    print(f"Evaluating model...")
    SAMPLE_N = 50
    PGD_BATCH_SIZE = 25  # Batch size for PGD
    IG_BATCH_SIZE = 10   # Batch size for Integrated Gradients

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb, _ in test_dl:
            xb, yb = xb.to(dev), yb.to(dev)
            with autocast():
                logits, _ = model(xb)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == yb).sum().item()
            total += len(yb)
    accuracy = correct / total if total else 0
    print(f"  Test accuracy: {accuracy:.3f}")

    def model_for_captum(x):
        with autocast():
            return model(x)[0].unsqueeze(-1)

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
        return accuracy, 0.0, 0.0, {'pgd_success_rate': 0, 'pgd_mean_iters_to_flip': 0}
        
    idxs = rng.choice(positive_subset_indices, size=sample_n_actual, replace=False)

    # --- PGD Caching ---
    if pgd_cache is None:
        pgd_cache = {}
    
    with torch.no_grad():
        if hasattr(model, '_orig_mod'):
            fingerprint = model._orig_mod.conv1.weight.detach().cpu().numpy().tobytes()[:64]
        else:
            fingerprint = model.conv1.weight.detach().cpu().numpy().tobytes()[:64]

    # --- Batched PGD Processing ---
    all_pgd_baselines = []
    all_pgd_stats = []
    
    if fingerprint in pgd_cache:
        print(f"  Using cached PGD results...")
        cached_data = pgd_cache[fingerprint]
        all_pgd_baselines = cached_data['baselines']
        all_pgd_stats = cached_data['stats']
    else:
        print(f"  Computing PGD baselines in batches of {PGD_BATCH_SIZE}...")
        for batch_start in range(0, len(idxs), PGD_BATCH_SIZE):
            batch_end = min(batch_start + PGD_BATCH_SIZE, len(idxs))
            batch_idxs = idxs[batch_start:batch_end]
            
            xb_list = [test_ds[i][0] for i in batch_idxs]
            yb_list = [test_ds[i][1] for i in batch_idxs]
            
            xb_batch = torch.stack(xb_list).to(dev)
            yb_batch = torch.tensor(yb_list, device=dev, dtype=torch.float)
            
            pgd_baselines_batch, pgd_stats_batch = find_adversarial_baseline_pgd_batch_optimized(
                model, xb_batch, yb_batch, dev
            )
            
            all_pgd_baselines.extend([pgd_baselines_batch[i] for i in range(len(batch_idxs))])
            all_pgd_stats.extend(pgd_stats_batch)
        
        pgd_cache[fingerprint] = {
            'baselines': all_pgd_baselines,
            'stats': all_pgd_stats
        }

    # --- Batched IG Processing ---
    results = []
    print(f"  Computing Integrated Gradients in batches of {IG_BATCH_SIZE}...")
    
    for batch_start in range(0, len(idxs), IG_BATCH_SIZE):
        batch_end = min(batch_start + IG_BATCH_SIZE, len(idxs))
        batch_range = range(batch_start, batch_end)
        
        xb_list, mask_list, baseline_list = [], [], []
        
        for i in batch_range:
            idx = idxs[i]
            xb, _, mask = test_ds[idx]
            xb_list.append(xb)
            mask_list.append(mask)
            
            if all_pgd_stats[i]['success']:
                baseline_list.append(all_pgd_baselines[i].squeeze(0).cpu())
            else:
                proportions = xb.mean(dim=1, keepdim=True)
                baseline_list.append(proportions.expand_as(xb))
        
        xb_batch = torch.stack(xb_list).to(dev)
        baseline_batch = torch.stack(baseline_list).to(dev)
        
        raw_attributions_batch = ig.attribute(xb_batch, baselines=baseline_batch, target=0)
        
        for j, i in enumerate(batch_range):
            raw_attr = raw_attributions_batch[j]
            xb = xb_batch[j]
            mask = mask_list[j]
            
            corrected_attr = raw_attr - raw_attr.mean(dim=0, keepdim=True)
            attributions = np.abs((corrected_attr * xb).sum(0).cpu().numpy())

            inside_scores = attributions[mask]
            outside_scores = attributions[~mask]
            
            saliency_auc = (inside_scores[:, None] > outside_scores[None, :]).mean() if len(inside_scores) > 0 and len(outside_scores) > 0 else 0.5
            
            sum_sq_inside = np.sum(inside_scores**2)
            sum_sq_total = np.sum(attributions**2)
            saliency_snr = sum_sq_inside / (sum_sq_total + 1e-9)

            results.append(dict(saliency_auc=saliency_auc, saliency_snr=saliency_snr))

    # --- Aggregate Stats ---
    attackable_samples = [r for r in all_pgd_stats if r['initial_prediction_correct']]
    pgd_success_count = sum(1 for r in attackable_samples if r['success'])
    pgd_success_rate = pgd_success_count / len(attackable_samples) if attackable_samples else 0
    
    iters_to_flip = [r['found_at_iter'] for r in attackable_samples if r['success']]
    pgd_mean_iters_to_flip = np.mean(iters_to_flip) if iters_to_flip else 0

    pgd_stats = {
        "pgd_success_rate": pgd_success_rate,
        "pgd_mean_iters_to_flip": pgd_mean_iters_to_flip,
    }
    
    print(f"  PGD baseline success rate: {pgd_success_rate:.3f}")

    mean_saliency_auc = np.mean([r['saliency_auc'] for r in results]) if results else 0.0
    mean_saliency_snr = np.mean([r['saliency_snr'] for r in results]) if results else 0.0
    print(f"  Mean Saliency AUC: {mean_saliency_auc:.3f}")
    print(f"  Mean Saliency SNR: {mean_saliency_snr:.3f}")

    return accuracy, mean_saliency_auc, mean_saliency_snr, pgd_stats

# --------------------------------------------------------------------------- #
# 7. Experiment Runners
# --------------------------------------------------------------------------- #

def run_single_experiment(args, seed: int, main_ds, tb_run_dir: str, npz_run_dir: str, 
                         epochs: int, use_scheduling: bool, gc_gap: float = 0.1):
    """Run a single experiment with given seed"""
    print(f"\n{'=' * 20}  SEED {seed} | Schedule: {use_scheduling}  {'=' * 20}")
    
    # Create train/val/test splits
    current_n_total = len(main_ds)
    train_size = int(0.70 * current_n_total)
    val_size = int(0.15 * current_n_total)
    test_size = current_n_total - train_size - val_size
    train_ds, val_ds, test_ds = random_split(
        main_ds,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )

    # Efficient DataLoaders
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
        prefetch_factor=PREFETCH_FACTOR if effective_num_workers > 0 else None,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=TRAIN_BATCH_SIZE * 2,
        num_workers=effective_num_workers,
        pin_memory=True,
        persistent_workers=effective_num_workers > 0,
        prefetch_factor=PREFETCH_FACTOR if effective_num_workers > 0 else None,
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=TRAIN_BATCH_SIZE * 2,
        num_workers=effective_num_workers,
        pin_memory=True,
        persistent_workers=effective_num_workers > 0,
        prefetch_factor=PREFETCH_FACTOR if effective_num_workers > 0 else None,
    )

    # Use GPU prefetching if available
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        train_dl = GPUPrefetchDataLoader(train_dl, dev)
        val_dl = GPUPrefetchDataLoader(val_dl, dev)
        test_dl = GPUPrefetchDataLoader(test_dl, dev)
    
    bce = nn.BCEWithLogitsLoss()
    scaler = GradScaler()

    # --- PGD Cache for this seed ---
    pgd_cache = {}

    # Train standard model
    set_seeds(seed, deterministic=args.deterministic)
    standard_model = TinyCNN().to(dev)
    # Optional: use PyTorch 2.x compilation
    if hasattr(torch, "compile") and not args.no_compile:
        try:
            standard_model = torch.compile(standard_model)
            print("  ✓ Standard model compilation successful")
        except Exception as e:
            print(f"  ✗ Standard model compilation failed: {type(e).__name__} - {e}")
    
    opt_standard = torch.optim.AdamW(standard_model.parameters(), lr=3e-4, weight_decay=1e-6)
    std_writer = SummaryWriter(log_dir=os.path.join(tb_run_dir, f"seed_{seed}", "standard"))
    std_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_standard, 'min', factor=0.5, patience=8, verbose=True)

    train_standard(standard_model, train_dl, val_dl, bce, opt_standard, dev, scaler, std_writer, std_scheduler, epochs=epochs, early_stopping_patience=15)
    
    std_acc, std_auc, std_snr, std_pgd_stats = evaluate_model(standard_model, test_dl, dev, pgd_cache=pgd_cache)
    std_writer.add_hparams(
        {'model': 'standard', 'epsilon': 0, 'seed': seed},
        {'hparam/accuracy': std_acc, 'hparam/saliency_auc': std_auc, 'hparam/saliency_snr': std_snr}
    )
    std_writer.close()

    robust_accs, robust_aucs, robust_snrs, robust_pgd_stats_list = [], [], [], []
    
    # Adversarial training loop
    if args.experiment_mode == 'adv_vs_std':
        param_iterator = EPSILONS
        param_name = 'epsilon'
    elif args.experiment_mode == 'direct_hotflip':
        param_iterator = EPSILONS
        param_name = 'epsilon'
    else:
        param_iterator = []
        param_name = 'param'

    for param_val in param_iterator:
        set_seeds(seed, deterministic=args.deterministic)
        mdl = TinyCNN().to(dev)
        if hasattr(torch, "compile") and not args.no_compile:
            try:
                mdl = torch.compile(mdl)
                print(f"  ✓ Robust model (param={param_val:.4f}) compilation successful")
            except Exception as e:
                print(f"  ✗ Robust model (param={param_val:.4f}) compilation failed: {type(e).__name__} - {e}")
        
        opt = torch.optim.AdamW(mdl.parameters(), lr=3e-4, weight_decay=1e-6)
        rob_writer = SummaryWriter(log_dir=os.path.join(tb_run_dir, f"seed_{seed}", f"{param_name}_{param_val:.4f}"))
        rob_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', factor=0.5, patience=8, verbose=True)
        
        # Choose training method
        if args.experiment_mode == 'adv_vs_std':
            if param_val > 0:
                patience = 25 if use_scheduling else 15
                train_hotflip(mdl, train_dl, val_dl, bce, opt, dev, scaler, rob_writer, rob_scheduler, 
                             max_flip_fraction=param_val, epochs=epochs, use_scheduling=use_scheduling, 
                             early_stopping_patience=patience, gc_gap=gc_gap)
                acc, auc, snr, pgd_stats = evaluate_model(mdl, test_dl, dev, pgd_cache=pgd_cache)
                hparams = {'model': 'robust', 'epsilon': param_val, 'seed': seed}
            else:
                acc, auc, snr, pgd_stats = std_acc, std_auc, std_snr, std_pgd_stats
                hparams = {'model': 'standard', 'epsilon': 0, 'seed': seed}
        
        elif args.experiment_mode == 'direct_hotflip':
            if param_val > 0:
                patience = 25 if use_scheduling else 15
                train_direct_hotflip(mdl, train_dl, val_dl, bce, opt, dev, scaler, rob_writer, rob_scheduler,
                                    max_flip_fraction=param_val, epochs=epochs, use_scheduling=use_scheduling,
                                    early_stopping_patience=patience, gc_gap=gc_gap)
                acc, auc, snr, pgd_stats = evaluate_model(mdl, test_dl, dev, pgd_cache=pgd_cache)
                hparams = {'model': 'robust_direct', 'epsilon': param_val, 'seed': seed}
            else:
                acc, auc, snr, pgd_stats = std_acc, std_auc, std_snr, std_pgd_stats
                hparams = {'model': 'standard', 'epsilon': 0, 'seed': seed}

        rob_writer.add_hparams(
            hparams,
            {'hparam/accuracy': acc, 'hparam/saliency_auc': auc, 'hparam/saliency_snr': snr}
        )
        robust_accs.append(acc); robust_aucs.append(auc); robust_snrs.append(snr); robust_pgd_stats_list.append(pgd_stats)
        rob_writer.close()

    return std_acc, std_auc, std_snr, std_pgd_stats, robust_accs, robust_aucs, robust_snrs, robust_pgd_stats_list 

# --------------------------------------------------------------------------- #
# 8. Experiment Management and Array Job Support
# --------------------------------------------------------------------------- #

def get_experiment_info(experiment_mode: str, array_idx: int) -> dict:
    """
    Maps array indices to specific experiment configurations.
    
    For 'adv_vs_std': 2 scheduling modes × 3 GC-gap × 3 conservation = 18 jobs
    For 'direct_hotflip': Same as adv_vs_std = 18 jobs
    """
    if experiment_mode in ['adv_vs_std', 'direct_hotflip']:
        # Total 18 combinations: schedule × gc_gap × conservation
        schedules = [True, False]
        n_per_schedule = len(GC_GAP_HPARAMS) * len(CONS_HPARAMS)
        
        schedule_idx = array_idx // n_per_schedule
        remainder = array_idx % n_per_schedule
        gc_idx = remainder // len(CONS_HPARAMS)
        cons_idx = remainder % len(CONS_HPARAMS)
        
        if schedule_idx >= len(schedules):
            raise ValueError(f"Array index {array_idx} out of range for {experiment_mode} (max: 17)")
            
        return {
            'schedule': schedules[schedule_idx],
            'gc_gap': GC_GAP_HPARAMS[gc_idx],
            'cons': CONS_HPARAMS[cons_idx],
            'description': f"schedule={'scheduled' if schedules[schedule_idx] else 'no_schedule'}, "
                          f"gc_gap={GC_GAP_HPARAMS[gc_idx]:.3f}, cons={CONS_HPARAMS[cons_idx]:.2f}"
        }
    else:
        raise ValueError(f"Unknown experiment mode: {experiment_mode}")

def main_single_combo(args, array_idx: int):
    """Run experiments for a single combination determined by array_idx."""
    
    # Get the specific configuration for this array index
    try:
        config = get_experiment_info(args.experiment_mode, array_idx)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    print(f"Running experiment: {config['description']}")
    
    # Extract parameters from config
    gc_gap_hparam = config['gc_gap']
    cons_hparam = config['cons']
    use_scheduling = config.get('schedule', False)
    single_param_val = config.get('epsilon', None)
    
    # Set up paths
    if args.experiment_mode in ['adv_vs_std', 'direct_hotflip']:
        schedule_mode_str = "scheduled" if use_scheduling else "no_schedule"
        run_spec_str = f"gc-gap_{gc_gap_hparam:.3f}_cons_{cons_hparam:.2f}"
        tb_run_dir = os.path.join(args.output_dir, "tensorboard", schedule_mode_str, run_spec_str)
        npz_run_dir = os.path.join(args.output_dir, "npz_results", schedule_mode_str, run_spec_str)
        
    os.makedirs(tb_run_dir, exist_ok=True)
    os.makedirs(npz_run_dir, exist_ok=True)

    print(f"  - Tensorboard logs will be saved to: {tb_run_dir}")
    print(f"  - NPZ results will be saved to: {npz_run_dir}")

    main_dataset = load_or_generate_dataset(gc_gap=gc_gap_hparam, conservation=cons_hparam)

    all_std_accs, all_std_aucs, all_std_snrs, all_std_pgd_stats = [], [], [], []
    all_rob_accs, all_rob_aucs, all_rob_snrs, all_rob_pgd_stats = [], [], [], []

    # Training parameters
    global TRAIN_BATCH_SIZE, NUM_WORKERS
    TRAIN_BATCH_SIZE = args.batch_size if args.batch_size else DEFAULT_BATCH_SIZE
    NUM_WORKERS = args.num_workers

    for sd in SEEDS:
        sa, sa_auc, s_snr, s_pgd, ra, ra_auc, r_snr, r_pgd = run_single_experiment(
            args, sd, main_dataset, tb_run_dir, npz_run_dir, args.epochs, use_scheduling, gc_gap_hparam
        )
        all_std_accs.append(sa); all_std_aucs.append(sa_auc); all_std_snrs.append(s_snr); all_std_pgd_stats.append(s_pgd)
        all_rob_accs.append(ra); all_rob_aucs.append(ra_auc); all_rob_snrs.append(r_snr); all_rob_pgd_stats.append(r_pgd)

    # Save raw results
    save_payload = {
        'experiment_mode': args.experiment_mode,
        'seeds': SEEDS,
        'gc_gap': gc_gap_hparam,
        'gc_pos': 0.5 + gc_gap_hparam,  # For backward compatibility
        'conservation': cons_hparam,
        'std_accs': all_std_accs,
        'std_aucs': all_std_aucs, 
        'std_snrs': all_std_snrs,
        'std_pgd_stats': all_std_pgd_stats,
        'rob_accs': all_rob_accs,
        'rob_aucs': all_rob_aucs,
        'rob_snrs': all_rob_snrs,
        'rob_pgd_stats': all_rob_pgd_stats
    }
    
    # Add experiment-specific parameters
    if args.experiment_mode in ['adv_vs_std', 'direct_hotflip']:
        save_payload['param_name'] = 'epsilon'
        save_payload['param_values'] = EPSILONS
        save_payload['scheduling'] = use_scheduling

    np.savez(os.path.join(npz_run_dir, "multi_seed_results.npz"), **save_payload)
    print("Single-combo job finished and results saved.")

def main(args):
    """Main entry point for sweep mode"""
    
    # Set multiprocessing start method
    try:
        torch.multiprocessing.set_start_method('forkserver', force=True)
        print("Multiprocessing start method set to 'forkserver'.")
    except RuntimeError:
        print("Multiprocessing start method already set.")

    # Define param_iterator
    if args.experiment_mode in ['adv_vs_std', 'direct_hotflip']:
        param_iterator = EPSILONS
    else:
        param_iterator = []

    # Determine scheduling mode from task ID
    if args.task_id is None:
        print("Warning: --task_id not provided. Running BOTH scheduled and non-scheduled modes sequentially.")
        schedule_modes = [True, False]
    else:
        schedule_modes = [args.task_id == 0]

    for use_scheduling in schedule_modes:
        schedule_mode_str = "scheduled" if use_scheduling else "no_schedule"
        print(f"\n\n{'#'*80}")
        print(f"## RUNNING EXPERIMENT SET: Scheduling = {schedule_mode_str.upper()}")
        print(f"{'#'*80}\n")
        
        base_schedule_dir = os.path.join(args.output_dir, schedule_mode_str)

        # Outer loop for Hyperparameter Search
        for gc_gap_hparam in GC_GAP_HPARAMS:
            for cons_hparam in CONS_HPARAMS:
                
                run_output_dir = os.path.join(base_schedule_dir, f"gc-gap_{gc_gap_hparam:.3f}_cons_{cons_hparam:.2f}")
                os.makedirs(run_output_dir, exist_ok=True)
                
                print(f"\n{'#'*60}")
                print(f"## Starting HP experiment: GC-Gap={gc_gap_hparam:.3f}, Conservation={cons_hparam:.2f}")
                print(f"## Results will be saved to: {run_output_dir}")
                print(f"{'#'*60}\n")
                
                main_dataset = load_or_generate_dataset(gc_gap=gc_gap_hparam, conservation=cons_hparam)
                
                all_std_accs, all_std_aucs, all_std_snrs, all_std_pgd_stats = [], [], [], []
                all_rob_accs, all_rob_aucs, all_rob_snrs, all_rob_pgd_stats = [], [], [], []

                for sd in SEEDS:
                    sa, sa_auc, s_snr, s_pgd, ra, ra_auc, r_snr, r_pgd = run_single_experiment(
                        args, sd, main_dataset, run_output_dir, run_output_dir, args.epochs, use_scheduling, gc_gap_hparam
                    )
                    all_std_accs.append(sa); all_std_aucs.append(sa_auc); all_std_snrs.append(s_snr); all_std_pgd_stats.append(s_pgd)
                    all_rob_accs.append(ra); all_rob_aucs.append(ra_auc); all_rob_snrs.append(r_snr); all_rob_pgd_stats.append(r_pgd)

                # Save Raw Results
                save_payload = {
                    'experiment_mode': args.experiment_mode,
                    'seeds': SEEDS,
                    'gc_gap': gc_gap_hparam,
                    'gc_pos': 0.5 + gc_gap_hparam,  # For backward compatibility
                    'conservation': cons_hparam,
                    'std_accs': all_std_accs,
                    'std_aucs': all_std_aucs,
                    'std_snrs': all_std_snrs,
                    'std_pgd_stats': all_std_pgd_stats,
                    'rob_accs': [[float(x) for x in seed_results] for seed_results in all_rob_accs],
                    'rob_aucs': all_rob_aucs,
                    'rob_snrs': all_rob_snrs,
                    'rob_pgd_stats': all_rob_pgd_stats
                }
                
                # Add experiment-specific parameters
                if args.experiment_mode in ['adv_vs_std', 'direct_hotflip']:
                    save_payload['param_name'] = 'epsilon'
                    save_payload['param_values'] = param_iterator
                    save_payload['scheduling'] = use_scheduling
                
                np.savez(
                    os.path.join(run_output_dir, 'multi_seed_results.npz'),
                    **save_payload
                )
                print(f"\nSaved raw results to {os.path.join(run_output_dir, 'multi_seed_results.npz')}")

    print("\n\nFull sweep finished. To generate plots, run with --aggregate_only.")

# --------------------------------------------------------------------------- #
# 9. Command Line Interface
# --------------------------------------------------------------------------- #

# Default values
TRAIN_BATCH_SIZE = DEFAULT_BATCH_SIZE
NUM_WORKERS = 4

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merged experiment with complex data and advanced training.")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to save results and plots.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Training epochs.")
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
        help="Index from SLURM_ARRAY_TASK_ID that selects experiment combo.",
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
    parser.add_argument(
        "--experiment_mode",
        type=str,
        default="adv_vs_std",
        choices=['adv_vs_std', 'direct_hotflip'],
        help="Which set of experiments to run (iterative vs direct hotflip).",
    )
    parser.add_argument(
        "--test_mapping",
        action="store_true",
        help="Test the array index mapping without running experiments.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic cuDNN kernels for full reproducibility.",
    )
    parser.add_argument(
        "--no_compile",
        action="store_true",
        help="Disable torch.compile for debugging.",
    )

    args = parser.parse_args()

    # Update global DataLoader defaults
    TRAIN_BATCH_SIZE = args.batch_size
    NUM_WORKERS = args.num_workers

    # Test mapping mode
    if args.test_mapping:
        print("Testing array index mapping...")
        print("\n=== Iterative HotFlip (adv_vs_std) ===")
        print("Total jobs: 18 (indices 0-17)")
        for i in range(18):
            config = get_experiment_info('adv_vs_std', i)
            print(f"  Index {i:2d}: {config['description']}")
        
        print("\n=== Direct HotFlip (direct_hotflip) ===")
        print("Total jobs: 18 (indices 0-17)")
        for i in range(18):
            config = get_experiment_info('direct_hotflip', i)
            print(f"  Index {i:2d}: {config['description']}")
        
        sys.exit(0)

    # Aggregate-only mode
    if args.aggregate_only:
        print("Aggregate-only mode: generating plots from existing results...")
        print("Note: This feature is simplified in the merged version.")
        print("For full visualization, please use a separate analysis script.")
        sys.exit(0)

    # Run experiments
    if args.array_idx is not None:
        main_single_combo(args, args.array_idx)
    else:
        main(args) 