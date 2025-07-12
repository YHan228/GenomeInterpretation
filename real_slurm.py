"""
This script has been updated to use the more complex data generation scheme from
the `final_gam_experiment.py` script.

Synthetic 1-kbp phenotype dataset (SLURM-compatible version)
    positives: high-GC background + multiple causal blocks and a promoter
    negatives: low-GC background, decoys, or promoter-only sequences
CNN training + Integrated Gradients attribution quality
Author: Yichen Han, 2025-06-29

--------------------------------------------------------------------------------
High-level overview of the experimental design
--------------------------------------------------------------------------------
This script implements a controlled synthetic experiment to evaluate the effect
of adversarial training on model interpretability in the presence of a feature
confounder. The key components are:

a. Synthetic Data Generation:
   - The data generator creates positive examples with multiple causal blocks
     and a promoter, alongside a mixed population of negative examples (decoys,
     promoter-only, and pure background).
   - The strength of the confounder is controlled by the `gc_pos` parameter
     (which sets a `gc_gap`), and the strength of the true signal is controlled
     by the `conservation` parameter.

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
     1. Windowed IoU (wIoU): Measures how well the attribution map *locates*
        the true causal signal. It is the standard Intersection-over-Union
        between the true causal mask and a predicted mask (defined as the
        window with the highest total attribution score). A score of 1 indicates
        a perfect match.
     2. SaliencyAUC: Measures the *purity* of attributions. It is the
        probability that a randomly chosen position inside the motif has a
        higher attribution score than a randomly chosen position outside. A
        score of 1 means all attributions are correctly concentrated within
        the motif.
     3. SaliencySNR (Signal-to-Noise Ratio): Measures the *cleanness* of
        attributions, defined as an R-squared-like metric. It is the fraction
        of attribution "energy" (sum of squared scores) within the true causal
        motif relative to the total energy across the entire sequence.
        Calculated as `sum(inside_scores^2) / sum(all_scores^2)`, it ranges
        from 0 (no signal in motif) to 1 (all signal in motif).

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

try:
    import logomaker
except ImportError:
    print("Error: logomaker is not installed. Please install it using 'pip install logomaker'")
    sys.exit(1)

# --------------------------------------------------------------------------- //
# 1. Configuration & Utilities
# --------------------------------------------------------------------------- #

# --- Data Generation Settings (from final_gam_experiment.py) ---
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
N_TOTAL                = 10000 # Keep the original number of samples for this experiment
TARGET_SIGNAL_FRAC     = 0.20
DEFAULT_BATCH_SIZE     = 512 # Default training parameters
DEFAULT_EPOCHS         = 50

# --- Hyperparameter Search Space (from toy_slurm.py) ---
# GC_HPARAMS = [0.525, 0.535, 0.55, 0.575, 0.6, 0.625, 0.65] # OLD: gc_pos values
# CONS_HPARAMS = [0.6, 0.7, 0.8]

# New 3x3 grid based on gc_gap, as requested.
# GC_GAP_HPARAMS will be [0.0, 0.1, 0.2]
GC_GAP_HPARAMS = np.linspace(0.0, 0.2, 3).tolist()
# CONS_HPARAMS will be [0.55, 0.75, 0.95]
CONS_HPARAMS = np.linspace(0.55, 0.95, 3).tolist()

# The target "effective epsilon" for randomized smoothing. Epsilon is defined
# as 1 - E[P(original base)], where the expectation is over the Dirichlet noise.
RS_EPSILON_HPARAMS = [0.01, 0.05, 0.10, 0.25, 0.40]
# ---

SEEDS = [0, 1, 2, 3, 4]
EPSILONS = [0.001, 0.005, 0.01, 0.05, 0.10, 0.15]

# Directory to cache synthetic datasets so we do not regenerate the same
# (gc, conservation) combination multiple times across different SLURM tasks.
DATASET_CACHE_DIR = "dataset_cache_real"

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
    """(L,) char → (4,L) float32 one-hot"""
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


def one_hot_to_seq(one_hot_tensor: torch.Tensor) -> str:
    """ (4, L) float tensor -> (L,) string """
    indices = torch.argmax(one_hot_tensor, dim=0).cpu().numpy()
    return "".join(ALPH[indices])


def concentration_from_epsilon(epsilon: float) -> float:
    """
    Calculates the required major concentration for an asymmetric 4D Dirichlet
    distribution to achieve a target expected epsilon.
    Epsilon = 1 - E[P(original base)] = 3 / (concentration + 3)
    """
    if not (0 < epsilon < 1):
        raise ValueError("Epsilon must be between 0 and 1.")
    return (3.0 * (1.0 - epsilon)) / epsilon


# --------------------------------------------------------------------------- #
# 1.b Visualization Helpers
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
    Generates sequence logos from the first conv layer's weights, logs them to
    TensorBoard, and/or saves them to disk.
    """
    # Ensure model is on CPU for weight processing, especially if compiled
    if hasattr(model, '_orig_mod'): # it's a compiled model
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

    # --- Save to disk if requested ---
    if save_to_disk:
        if not output_dir:
            print("Warning: output_dir not provided for saving motif logos.")
            return
        
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

    # --- Log to TensorBoard if requested ---
    if log_to_tb:
        num_filters = len(pwms_dfs)
        cols = 6
        rows = math.ceil(num_filters / cols)
        
        fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 2.5 * rows))
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
# 1.a  Dataset caching helpers
# --------------------------------------------------------------------------- #


def _dataset_cache_path(gc_gap: float, conservation: float) -> str:
    """Return cache file path for a given (gc_gap, conservation)."""
    # Version 2 cache to avoid conflicts with old data format
    return os.path.join(
        DATASET_CACHE_DIR,
        f"v2_gc-gap_{gc_gap:.3f}_cons_{conservation:.3f}.npz",
    )


def load_or_generate_dataset(gc_gap: float, conservation: float):
    """Load dataset from cache if present, otherwise generate and cache it."""
    # Ensure the cache directory exists right before we use it. This is a more
    # robust pattern for cluster environments.
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

    # We ignore the summary dict for now as it's not used in this script.
    ds, _ = generate_dataset(
        gc_gap=gc_gap,
        conservation=conservation,
        target_signal_frac=TARGET_SIGNAL_FRAC,
        motif_repertoire=DEFAULT_MOTIF_REPERTOIRE,
        include_partial_negatives=True,
    )

    # --- Atomic saving to prevent race conditions in cluster environments ---
    temp_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=10))
    temp_path = f"{cache_path}.{temp_id}.tmp"
    
    try:
        np.savez(
            temp_path,
            X=ds.x.cpu().numpy(),
            y=ds.y.cpu().numpy(),
            masks=ds.m,
        )
        # np.savez automatically appends '.npz', so the file to rename is temp_path + '.npz'
        os.rename(f"{temp_path}.npz", cache_path)
    except Exception as e:
        print(f"Error caching dataset: {e}. Cleaning up temp file...")
        # Also need to check for the correct temp file name during cleanup
        final_temp_path = f"{temp_path}.npz"
        if os.path.exists(final_temp_path):
            os.remove(final_temp_path)
        raise # Re-raise the exception after cleanup
        
    return ds


# --------------------------------------------------------------------------- #
# 2. Dataset generation
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

    This function is migrated from final_gam_experiment.py.

    Returns
    -------
    SeqDS
        Dataset containing one-hot encoded sequences, labels and causal masks.
    dict
        Book-keeping summary for reproducibility.
    """

    # --- decide sample budget ---
    # In this script, we use a fixed N_TOTAL from the global config.
    POS_N = N_TOTAL // 2
    NEG_N = N_TOTAL - POS_N

    decoy_neg_n = int(0.2 * NEG_N) if include_partial_negatives else 0
    promoter_only_neg_n = int(0.2 * NEG_N) # 20% of negatives are promoter-only
    std_neg_n   = NEG_N - decoy_neg_n
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
        try:
            blk_starts = _nonoverlap_positions(SEQ_LEN, blk_lens)
        except RuntimeError:
            continue # Try again if placement fails

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
        try:
            blk_starts = _nonoverlap_positions(SEQ_LEN, blk_lens)
        except RuntimeError:
            # If we fail, just generate a simple background sequence
            X.append(one_hot(sample_background(SEQ_LEN, gc=current_gc)))
            y.append(0)
            masks.append(np.zeros(SEQ_LEN, dtype=bool))
            continue

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
        "n_background_only_negative": background_only_neg_n,
        "motif_repertoire": motif_repertoire,
        "seed": np.random.get_state()[1][0].item() if len(np.random.get_state()[1]) > 0 else -1,
        "target_signal_frac": target_signal_frac,
        "avg_realised_frac": avg_realised_frac,
    }

    return SeqDS(X, y, masks), summary


# --------------------------------------------------------------------------- #
# 3. Model and Dataset Classes
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
        return logits.squeeze(-1)


# --------------------------------------------------------------------------- #
# 4. Training, Scheduling, and Evaluation
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
                logits = model(xb)
                loss = loss_fn(logits, yb)
            total_loss += loss.item()
            num_batches += 1
    return total_loss / num_batches if num_batches > 0 else 0


def train_standard(model, train_loader, val_loader, loss_fn, optimizer, dev, scaler, writer, scheduler, epochs: int = 10, early_stopping_patience: int = 10, early_stopping_min_delta: float = 1e-4) -> None:
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
                logits = model(xb)
                loss = loss_fn(logits, yb)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            num_batches += 1
        
        avg_train_loss = total_loss / num_batches if num_batches > 0 else 0
        avg_val_loss = validate_epoch(model, val_loader, loss_fn, dev)

        # Step scheduler first, then log the potentially new LR
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
            print(f"  -> Early stopping at epoch {epoch + 1} due to no improvement in val_loss for {early_stopping_patience} epochs.")
            break

def generate_smoothed_batch(xb: torch.Tensor, concentration_major: float, dev: torch.device) -> torch.Tensor:
    """
    Replaces one-hot vectors with samples from an asymmetric Dirichlet distribution.
    For a given one-hot vector, the corresponding position in the alpha tensor for
    the Dirichlet distribution is set to `concentration_major`, while all other
    positions are set to a fixed `concentration_minor` of 1.0. This creates a
    "softened" one-hot vector that still sums to 1.
    """
    concentration_minor = 1.0  # Fixed minor concentration
    
    # Get original base indices, shape (B, L)
    original_bases_idx = xb.argmax(dim=1)
    
    # Create the alpha tensor for the Dirichlet. Shape (B, L, 4)
    alphas = torch.full((xb.shape[0], xb.shape[2], 4), concentration_minor, device=dev, dtype=torch.float32)
    
    # Use scatter_ to place `concentration_major` at the correct indices
    alphas.scatter_(2, original_bases_idx.unsqueeze(2), concentration_major)
    
    # Create Dirichlet distribution with these alphas
    dist = torch.distributions.Dirichlet(alphas)
    
    # Sample and permute to match xb shape (B, 4, L)
    smoothed_xb = dist.sample().permute(0, 2, 1)
    
    return smoothed_xb

def train_random_smoothing(model, train_loader, val_loader, loss_fn, optimizer, dev, scaler, writer, scheduler,
                           target_epsilon: float, epochs: int = 10, early_stopping_patience: int = 10, early_stopping_min_delta: float = 1e-4) -> None:
    
    dirichlet_concentration = concentration_from_epsilon(target_epsilon)
    print(f"Starting randomized smoothing training with target epsilon = {target_epsilon:.4f} (Dirichlet conc = {dirichlet_concentration:.2f}) and early stopping (patience={early_stopping_patience}, min_delta={early_stopping_min_delta})...")
    best_val_loss = float('inf')
    early_stopping_counter = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0
        for xb, yb, _ in train_loader:
            xb, yb = xb.to(dev), yb.to(dev)
            
            # Replace one-hot batch with a smoothed version from Dirichlet distribution
            adv_xb = generate_smoothed_batch(xb, dirichlet_concentration, dev)

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
        
        # Step scheduler first, then log the potentially new LR
        scheduler.step(avg_val_loss)
        writer.add_scalar('Loss/train_random_smoothing', avg_train_loss, epoch)
        writer.add_scalar('Loss/validation_random_smoothing', avg_val_loss, epoch)
        writer.add_scalar('LR/train_random_smoothing', scheduler.optimizer.param_groups[0]['lr'], epoch)
        print(f"  Epoch {epoch + 1}/{epochs}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Target Epsilon: {target_epsilon:.4f}")

        # Check for improvement
        if (best_val_loss - avg_val_loss) > early_stopping_min_delta:
            best_val_loss = avg_val_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
        
        if early_stopping_counter >= early_stopping_patience:
            print(f"  -> Early stopping at epoch {epoch + 1} due to no improvement in val_loss for {early_stopping_patience} epochs.")
            break

def generate_hotflip_examples(model, xb, yb, loss_fn, flip_fraction: float):
    seq_len = xb.shape[2]
    k_flips = int(flip_fraction * seq_len)
    adv_xb = xb.clone()

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
        best_pos_to_flip = best_flip_scores_per_pos.argmax(dim=1)
        best_new_base_idx = saliency_scores[range(len(xb)), :, best_pos_to_flip].argmax(dim=1)
        old_base_idx = adv_xb[range(len(xb)), :, best_pos_to_flip].argmax(dim=1)
        adv_xb = adv_xb.detach()
        adv_xb[range(len(xb)), old_base_idx, best_pos_to_flip] = 0.0
        adv_xb[range(len(xb)), best_new_base_idx, best_pos_to_flip] = 1.0
    return adv_xb

def train_hotflip(model, train_loader, val_loader, loss_fn, optimizer, dev, scaler, writer, scheduler,
                  max_flip_fraction: float, epochs: int = 10, use_scheduling: bool = True, early_stopping_patience: int = 25, early_stopping_min_delta: float = 1e-4) -> None:
    
    scheduling_str = "ON" if use_scheduling else "OFF"
    early_stop_str = "ON" if use_scheduling else f"ON (patience={early_stopping_patience}, min_delta={early_stopping_min_delta})"
    print(f"Starting HotFlip training with max_flip_fraction = {max_flip_fraction:.4f}, Scheduling: {scheduling_str}, Early Stopping: {early_stop_str}...")
    
    # We now track the previous epoch's loss for our custom early stopping logic.
    previous_val_loss = float('inf')
    early_stopping_counter = 0

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
        
        # For scheduled training, we help the LR scheduler by telling it to compare
        # against the previous epoch's loss, not the global best, to account for
        # the increasing difficulty. We do this by manually setting its internal state.
        if use_scheduling:
            scheduler.best = previous_val_loss

        # The LR scheduler (`ReduceLROnPlateau`) will now use the value we just set
        # (or its default global best if not in scheduled mode).
        scheduler.step(avg_val_loss)
        writer.add_scalar('Loss/train_adversarial', avg_train_loss, epoch)
        writer.add_scalar('Loss/validation_adversarial', avg_val_loss, epoch)
        writer.add_scalar('LR/train_adversarial', scheduler.optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('Epsilon/train_adversarial', current_flip_fraction, epoch)
        print(f"  Epoch {epoch + 1}/{epochs}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Epsilon: {current_flip_fraction:.4f}")

        # Early stopping logic now compares with the *previous* epoch's val_loss,
        # which is more robust for scheduled adversarial training where val_loss
        # may not be monotonically decreasing.
        if (previous_val_loss - avg_val_loss) > early_stopping_min_delta:
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
        
        # Update previous_val_loss for the next iteration.
        previous_val_loss = avg_val_loss
        
        if early_stopping_counter >= early_stopping_patience:
            print(f"  -> Early stopping at epoch {epoch + 1} due to no improvement in val_loss for {early_stopping_patience} consecutive epochs.")
            break

def find_adversarial_baseline_pgd(model, xb: torch.Tensor, yb: torch.Tensor, dev: torch.device,
                                  num_iter: int = 20, epsilon: float = 0.1, step_size: float = 0.01):
    """
    Finds a baseline for IG using PGD to find a nearby adversarial example
    that flips the model's prediction. Returns the baseline and a stats dict.
    """
    adv_xb = xb.clone().detach()
    stats = {
        'success': False,
        'initial_logit': 0.0,
        'final_logit': 0.0,
        'found_at_iter': num_iter,
        'initial_prediction_correct': False
    }

    with torch.no_grad(), autocast():
        initial_logits = model(adv_xb)
        initial_pred_class = (initial_logits > 0).float()
        stats['initial_logit'] = initial_logits.item()
        stats['final_logit'] = initial_logits.item() # Default final logit

    is_correct = initial_pred_class.item() == yb.item()
    stats['initial_prediction_correct'] = is_correct

    # We only run PGD if the initial prediction for a positive example is correct.
    if not is_correct or yb.item() == 0:
        return torch.zeros_like(xb, device=dev), stats

    loss_fn = nn.BCEWithLogitsLoss()
    
    # Use a more stable step size for sign-based PGD
    step_size = epsilon / 10.0

    for i in range(num_iter):
        adv_xb.requires_grad = True
        with autocast():
            logits = model(adv_xb)
            loss = loss_fn(logits, yb.expand_as(logits))
        
        model.zero_grad()
        loss.backward()
        
        with torch.no_grad():
            grad = adv_xb.grad.data
            # Use sign() for more stable PGD updates
            adv_xb = adv_xb + step_size * grad.sign()
            
            delta = adv_xb - xb
            delta = torch.clamp(delta, -epsilon, epsilon)
            adv_xb = torch.clamp(xb + delta, 0, 1)

            current_logits = model(adv_xb)
            current_pred_class = (current_logits > 0).float()
            
            if current_pred_class.item() != initial_pred_class.item():
                stats['success'] = True
                stats['final_logit'] = current_logits.item()
                stats['found_at_iter'] = i + 1
                return adv_xb.detach(), stats
    
    # If no flip was found, final logit is the last one computed
    with torch.no_grad():
        final_logits = model(adv_xb)
        stats['final_logit'] = final_logits.item()

    return torch.zeros_like(xb, device=dev), stats

def evaluate_model(model, test_dl, dev):
    print(f"Evaluating model...")
    SAMPLE_N = 50  # Reduced from 100 for faster evaluation
    ANALYSIS_CHUNK_LEN = BLOCK_LEN_MEAN # Use mean block length for wIoU

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
        # Return default PGD stats for empty case
        default_pgd_stats = {'pgd_success_rate': 0, 'pgd_mean_iters_to_flip': 0}
        return accuracy, 0.0, 0.0, default_pgd_stats
        
    idxs = rng.choice(positive_subset_indices, size=sample_n_actual, replace=False)

    results = []
    pgd_results = []
    for idx in idxs:
        xb, y_scalar, mask = test_ds[idx]
        xb = xb.unsqueeze(0).to(dev)
        yb = torch.tensor([y_scalar], device=dev, dtype=torch.float)

        # Two-stage baseline strategy:
        # 1. Try to find a decision-boundary point using PGD.
        pgd_baseline, pgd_stat = find_adversarial_baseline_pgd(model, xb, yb, dev)
        pgd_results.append(pgd_stat)

        # 2. If PGD succeeds, use the adversarial example as the baseline.
        #    If it fails, fall back to the sequence's own nucleotide composition.
        if pgd_stat['success']:
            baseline = pgd_baseline
        else:
            proportions = xb.mean(dim=2, keepdim=True)
            baseline = proportions.expand_as(xb)

        attributions = ig.attribute(xb, baselines=baseline, target=0).abs().sum(1).squeeze(0).cpu().numpy()
        
        inside_scores = attributions[mask]
        outside_scores = attributions[~mask]
        saliency_auc = (inside_scores[:, None] > outside_scores[None, :]).mean() if len(inside_scores) > 0 and len(outside_scores) > 0 else 0.5
        
        # New metric: Saliency "R-squared" (fraction of energy in motif)
        sum_sq_inside = np.sum(inside_scores**2)
        sum_sq_total = np.sum(attributions**2)
        saliency_snr = sum_sq_inside / (sum_sq_total + 1e-9)

        results.append(dict(saliency_auc=saliency_auc, saliency_snr=saliency_snr))

    # --- Aggregate PGD Stats ---
    # Only consider attacks on initially correct predictions for success rate
    attackable_samples = [r for r in pgd_results if r['initial_prediction_correct']]
    pgd_success_count = sum(1 for r in attackable_samples if r['success'])
    pgd_success_rate = pgd_success_count / len(attackable_samples) if attackable_samples else 0
    
    iters_to_flip = [r['found_at_iter'] for r in attackable_samples if r['success']]
    pgd_mean_iters_to_flip = np.mean(iters_to_flip) if iters_to_flip else 0

    pgd_stats = {
        "pgd_success_rate": pgd_success_rate,
        "pgd_mean_iters_to_flip": pgd_mean_iters_to_flip,
    }
    
    print(f"  PGD baseline success rate: {pgd_success_rate:.3f}")

    mean_saliency_auc = np.mean([r['saliency_auc'] for r in results])
    mean_saliency_snr = np.mean([r['saliency_snr'] for r in results])
    print(f"  Mean Saliency AUC: {mean_saliency_auc:.3f}")
    print(f"  Mean Saliency SNR: {mean_saliency_snr:.3f}")

    return accuracy, mean_saliency_auc, mean_saliency_snr, pgd_stats

# --------------------------------------------------------------------------- #
# 5. Experiment Runner
# --------------------------------------------------------------------------- #

def run_single_experiment(args, seed: int, main_ds, tb_run_dir: str, npz_run_dir: str, epochs: int, use_scheduling: bool, single_param_val: float = None):
    print(f"\n{'=' * 20}  SEED {seed} | Schedule: {use_scheduling}  {'=' * 20}")
    
    # Create main training, validation, and test splits (70-15-15)
    # The total number of samples is now determined by the dataset object itself.
    current_n_total = len(main_ds)
    train_size = int(0.70 * current_n_total)
    val_size = int(0.15 * current_n_total)
    test_size = current_n_total - train_size - val_size
    train_ds, val_ds, test_ds = random_split(
        main_ds,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed) # Use seed for split
    )

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
    std_writer = SummaryWriter(log_dir=os.path.join(tb_run_dir, f"seed_{seed}", "standard"))
    std_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_standard, 'min', factor=0.5, patience=8, verbose=True)

    train_standard(standard_model, train_dl, val_dl, bce, opt_standard, dev, scaler, std_writer, std_scheduler, epochs=epochs, early_stopping_patience=15)
    
    # print("Standard model training complete. Saving and logging final motifs...")
    final_epoch_count = epochs # In future could get this from train function
    # log_conv1_motifs(standard_model, std_writer, final_epoch_count, output_dir=os.path.join(npz_run_dir, "standard"), save_to_disk=True, log_to_tb=True)
    
    std_acc, std_auc, std_snr, std_pgd_stats = evaluate_model(standard_model, test_dl, dev)
    std_writer.add_hparams(
        {'model': 'standard', 'epsilon': 0, 'seed': seed},
        {'hparam/accuracy': std_acc, 'hparam/saliency_auc': std_auc, 'hparam/saliency_snr': std_snr}
    )
    std_writer.close()

    robust_accs, robust_aucs, robust_snrs, robust_pgd_stats = [], [], [], []
    
    # --- Adversarial / Robustness Training Loop ---
    # Determine the parameters to iterate over
    if args.experiment_mode == 'adv_vs_std':
        param_iterator = EPSILONS
        param_name = 'epsilon'
    elif args.experiment_mode == 'random_smoothing':
        param_iterator = RS_EPSILON_HPARAMS
        param_name = 'target_epsilon'
    else:
        param_iterator = []
        param_name = 'param' # fallback

    # If a single parameter value is provided (e.g., from a SLURM array job
    # for random_smoothing), we override the iterator to run only for that value.
    if single_param_val is not None:
        param_iterator = [single_param_val]

    for param_val in param_iterator:
        set_seeds(seed)
        mdl = TinyCNN().to(dev)
        if hasattr(torch, "compile"):
            try:
                mdl = torch.compile(mdl)
            except Exception:
                pass
        opt = torch.optim.Adam(mdl.parameters(), lr=1e-3)
        rob_writer = SummaryWriter(log_dir=os.path.join(tb_run_dir, f"seed_{seed}", f"{param_name}_{param_val:.4f}"))
        rob_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', factor=0.5, patience=8, verbose=True)
        
        # Branch for training function and its specific arguments
        if args.experiment_mode == 'adv_vs_std':
            if param_val > 0:
                patience = 25 if use_scheduling else 15
                train_hotflip(mdl, train_dl, val_dl, bce, opt, dev, scaler, rob_writer, rob_scheduler, 
                              max_flip_fraction=param_val, epochs=epochs, use_scheduling=use_scheduling, early_stopping_patience=patience)
                
                motif_out_dir = os.path.join(npz_run_dir, f"hotflip_{param_name}_{param_val:.4f}")
                # print(f"Robust model (param={param_val:.4f}) training complete. Saving and logging final motifs...")
                # log_conv1_motifs(mdl, rob_writer, epochs, output_dir=motif_out_dir, save_to_disk=True, log_to_tb=True)

                acc, auc, snr, pgd_stats = evaluate_model(mdl, test_dl, dev)
                hparams = {'model': 'robust', 'epsilon': param_val, 'seed': seed}
            else: # eps = 0 is just the standard model
                acc, auc, snr, pgd_stats = std_acc, std_auc, std_snr, std_pgd_stats
                hparams = {'model': 'standard', 'epsilon': 0, 'seed': seed}
        
        elif args.experiment_mode == 'random_smoothing':
            train_random_smoothing(mdl, train_dl, val_dl, bce, opt, dev, scaler, rob_writer, rob_scheduler,
                                   target_epsilon=param_val, epochs=epochs, early_stopping_patience=15)

            motif_out_dir = os.path.join(npz_run_dir, f"rs_{param_name}_{param_val:.4f}")
            # print(f"Robust model (param={param_val:.4f}) training complete. Saving and logging final motifs...")
            # log_conv1_motifs(mdl, rob_writer, epochs, output_dir=motif_out_dir, save_to_disk=True, log_to_tb=True)

            acc, auc, snr, pgd_stats = evaluate_model(mdl, test_dl, dev)
            hparams = {'model': 'robust_rs', 'target_epsilon': param_val, 'seed': seed}

        else: # Should not happen
            continue

        rob_writer.add_hparams(
            hparams,
            {'hparam/accuracy': acc, 'hparam/saliency_auc': auc, 'hparam/saliency_snr': snr}
        )
        robust_accs.append(acc); robust_aucs.append(auc); robust_snrs.append(snr); robust_pgd_stats.append(pgd_stats)
        rob_writer.close()

    return std_acc, std_auc, std_snr, std_pgd_stats, robust_accs, robust_aucs, robust_snrs, robust_pgd_stats


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

    # Define param_iterator in this scope to make it available for np.savez
    if args.experiment_mode == 'adv_vs_std':
        param_iterator = EPSILONS
    elif args.experiment_mode == 'random_smoothing':
        param_iterator = RS_EPSILON_HPARAMS
    else:
        # This case should not be hit due to argparse choices, but for safety:
        param_iterator = []

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
                        args, sd, main_dataset, run_output_dir, run_output_dir, args.epochs, use_scheduling
                    )
                    all_std_accs.append(sa); all_std_aucs.append(sa_auc); all_std_snrs.append(s_snr); all_std_pgd_stats.append(s_pgd)
                    all_rob_accs.append(ra); all_rob_aucs.append(ra_auc); all_rob_snrs.append(r_snr); all_rob_pgd_stats.append(r_pgd)

                # --- Save Raw Results ---
                # Note: The structure of robust metrics has changed slightly to a list of lists.
                np.savez(
                    os.path.join(run_output_dir, 'multi_seed_results.npz'),
                    experiment_mode=args.experiment_mode,
                    epsilons=np.array(param_iterator), seeds=np.array(SEEDS),
                    gc_pos=0.5 + gc_gap_hparam, conservation=cons_hparam,
                    std_accs=np.array(all_std_accs), std_aucs=np.array(all_std_aucs), std_snrs=np.array(all_std_snrs),
                    rob_accs=np.array(all_rob_accs), rob_aucs=np.array(all_rob_aucs), rob_snrs=np.array(all_rob_snrs),
                    std_pgd_stats=np.array(all_std_pgd_stats, dtype=object), rob_pgd_stats=np.array(all_rob_pgd_stats, dtype=object)
                )
                print(f"\nSaved raw results to {os.path.join(run_output_dir, 'multi_seed_results.npz')}")

    # --- Final Aggregation Across All Runs ---
    # This part is now handled exclusively by the --aggregate_only flag
    # which provides much better visualizations.
    print("\n\nFull sweep finished. To generate plots, run with --aggregate_only.")
    # python -u toy_slurm.py \      [genome] ~/GenomeInterpretation at bioinflogin01
    #   --output_dir slurm_results/job_8727062_consolidated \
    #   --aggregate_only
    
    return


    ############################################################################
#                                Array-mode path                               #
    ############################################################################


# If --array_idx is supplied we execute **one** (schedule, gc, conservation)
# combination.  This allows efficient SLURM array jobs while limiting the
# number of concurrent GPUs via the %N syntax in the submission script.


def main_single_combo(args, array_idx: int):
    """Run experiments for a single combination determined by array_idx."""

    # Build mapping lists for each experimental mode
    if args.experiment_mode == 'adv_vs_std':
        combos = []
        for schedule in [True, False]:
            for gc_gap_val in GC_GAP_HPARAMS:
                for cons_val in CONS_HPARAMS:
                    combos.append({'schedule': schedule, 'gc_gap': gc_gap_val, 'cons': cons_val})
    elif args.experiment_mode == 'random_smoothing':
        combos = []
        for gc_gap_val in GC_GAP_HPARAMS:
            for cons_val in CONS_HPARAMS:
                for eps_val in RS_EPSILON_HPARAMS:
                    combos.append({'gc_gap': gc_gap_val, 'cons': cons_val, 'epsilon': eps_val})
    else:
        raise ValueError(f"Unknown experiment mode: {args.experiment_mode}")

    num_jobs = len(combos)
    if array_idx < 0 or array_idx >= num_jobs:
        raise ValueError(
            f"array_idx {array_idx} is out of range for mode '{args.experiment_mode}' (0-{num_jobs - 1})")

    combo = combos[array_idx]
    gc_gap_hparam, cons_hparam = combo['gc_gap'], combo['cons']
    
    # Determine the specific parameters for this single run from the combo dict.
    # This is the key to fixing the bug where RS jobs did 5x the work.
    use_scheduling = combo.get('schedule', False)
    single_param_val = combo.get('epsilon', None)

    # --- Set up paths and print info ---
    if args.experiment_mode == 'adv_vs_std':
        schedule_mode_str = "scheduled" if use_scheduling else "no_schedule"
        run_spec_str = f"gc-gap_{gc_gap_hparam:.3f}_cons_{cons_hparam:.2f}"
        tb_run_dir = os.path.join(args.output_dir, "tensorboard", schedule_mode_str, run_spec_str)
        npz_run_dir = os.path.join(args.output_dir, "npz_results", schedule_mode_str, run_spec_str)
        print(f"Running single-combo job: schedule={schedule_mode_str}, GC-Gap={gc_gap_hparam:.3f}, CONS={cons_hparam:.2f}")
    
    elif args.experiment_mode == 'random_smoothing':
        eps_hparam = combo['epsilon']
        mode_str = "random_smoothing"
        run_spec_str = f"gc-gap_{gc_gap_hparam:.3f}_cons_{cons_hparam:.2f}_eps_{eps_hparam:.4f}"
        tb_run_dir = os.path.join(args.output_dir, "tensorboard", mode_str, run_spec_str)
        npz_run_dir = os.path.join(args.output_dir, "npz_results", mode_str, run_spec_str)
        print(f"Running single-combo job: mode={mode_str}, GC-Gap={gc_gap_hparam:.3f}, CONS={cons_hparam:.2f}, Epsilon={eps_hparam:.4f}")
        
    os.makedirs(tb_run_dir, exist_ok=True)
    os.makedirs(npz_run_dir, exist_ok=True)

    print(f"  - Tensorboard logs will be saved to: {tb_run_dir}")
    print(f"  - NPZ results will be saved to: {npz_run_dir}")

    main_dataset = load_or_generate_dataset(gc_gap=gc_gap_hparam, conservation=cons_hparam)

    all_std_accs, all_std_aucs, all_std_snrs, all_std_pgd_stats = [], [], [], []
    all_rob_accs, all_rob_aucs, all_rob_snrs, all_rob_pgd_stats = [], [], [], []

    # Training parameters derived from CLI / environment
    global TRAIN_BATCH_SIZE, NUM_WORKERS
    TRAIN_BATCH_SIZE = args.batch_size if args.batch_size else DEFAULT_BATCH_SIZE
    NUM_WORKERS = args.num_workers

    for sd in SEEDS:
        # Pass args to the runner function so it knows the mode
        sa, sa_auc, s_snr, s_pgd, ra, ra_auc, r_snr, r_pgd = run_single_experiment(
            args, sd, main_dataset, tb_run_dir, npz_run_dir, args.epochs, use_scheduling, single_param_val
        )
        all_std_accs.append(sa); all_std_aucs.append(sa_auc); all_std_snrs.append(s_snr); all_std_pgd_stats.append(s_pgd)
        all_rob_accs.append(ra); all_rob_aucs.append(ra_auc); all_rob_snrs.append(r_snr); all_rob_pgd_stats.append(r_pgd)

    # Save raw results just like in the sweep mode
    save_payload = {
        'experiment_mode': args.experiment_mode,
        'seeds': SEEDS,
        'gc_pos': 0.5 + gc_gap_hparam,
        'conservation': cons_hparam,
        'std_accs': all_std_accs, 'std_aucs': all_std_aucs, 'std_snrs': all_std_snrs,
        'rob_accs': all_rob_accs, 'rob_aucs': all_rob_aucs, 'rob_snrs': all_rob_snrs,
        'std_pgd_stats': all_std_pgd_stats, 'rob_pgd_stats': all_rob_pgd_stats
    }
    if args.experiment_mode == 'adv_vs_std':
        save_payload['epsilons'] = EPSILONS
        save_payload['scheduling'] = use_scheduling
    elif args.experiment_mode == 'random_smoothing':
        save_payload['epsilons'] = [eps_hparam]

    np.savez(os.path.join(npz_run_dir, "multi_seed_results.npz"), **save_payload)

    print("Single-combo job finished and results saved.")

    # We do **not** run the heavyweight plotting routines in array mode –
    # collect plots in a downstream aggregation step to save GPU time.

# --------------------------------------------------------------------------- #
# 7. Data-loading defaults (set after CLI parsing)
# --------------------------------------------------------------------------- #

TRAIN_BATCH_SIZE = DEFAULT_BATCH_SIZE
NUM_WORKERS = 4  # sensible default; overridden later

if __name__ == "__main__":
    # Set the multiprocessing start method to be safer for shared file systems
    # This must be done at the top-level, before any DataLoader is initialized.
    try:
        torch.multiprocessing.set_start_method('forkserver', force=True)
        print("Multiprocessing start method set to 'forkserver'.")
    except RuntimeError:
        print("Multiprocessing start method already set.")

    parser = argparse.ArgumentParser(description="Run robustness experiment or aggregate results.")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to save results and plots.")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
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
        choices=['adv_vs_std', 'random_smoothing'],
        help="Which set of experiments to run (training only).",
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
        plots_dir = os.path.join(args.output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        npz_path = os.path.join(args.output_dir, "npz_results")
        if not os.path.isdir(npz_path):
            print(f"Error: The results directory '{npz_path}' could not be found.")
            sys.exit(1)

        npz_files = glob.glob(os.path.join(npz_path, '**', 'multi_seed_results.npz'), recursive=True)
        if not npz_files:
            print(f"No .npz result files found in {npz_path}; nothing to aggregate.")
            sys.exit(1)
        
        print(f"Found {len(npz_files)} result files, building master dataframe...")
        all_data = []
        
        # Group files by experiment type for debugging
        rs_files = [f for f in npz_files if 'random_smoothing' in f]
        adv_files = [f for f in npz_files if 'random_smoothing' not in f]
        print(f"  - RandomSmoothing files: {len(rs_files)}")
        print(f"  - Adversarial training files: {len(adv_files)}")
        
        for f_path in npz_files:
            try:
                data = np.load(f_path, allow_pickle=True)
                
                # --- Determine experiment mode from file path ---
                if 'random_smoothing' in f_path:
                    mode = 'random_smoothing'
                    scheduling_mode = 'n/a'
                else:
                    mode = 'adv_vs_std'
                    is_scheduled = data.get('scheduling') and data['scheduling'].item()
                    scheduling_mode = "scheduled" if is_scheduled else "no_schedule"

                gc_pos = float(data['gc_pos'].item())
                cons = float(data['conservation'].item())
                seeds = data['seeds']
                
                std_metrics = {
                    'Accuracy': data['std_accs'],
                    'SaliencyAUC': data['std_aucs'], 'SaliencySNR': data['std_snrs']
                }
                # PGD stats are arrays of dicts, so we handle them carefully.
                std_pgd_stats = data.get('std_pgd_stats', [{} for _ in seeds])

                rob_metrics = {
                    'Accuracy': data['rob_accs'],
                    'SaliencyAUC': data['rob_aucs'], 'SaliencySNR': data['rob_snrs']
                }
                # PGD stats for robust models are arrays of lists of dicts.
                rob_pgd_stats = data.get('rob_pgd_stats', [[] for _ in seeds])

                # The key for the parameter list is now standardized to 'epsilons'
                params = data['epsilons']
                if mode == 'adv_vs_std':
                    param_name = 'epsilon'
                else: # random_smoothing
                    param_name = 'target_epsilon'
                    # For RS files, check if there's an epsilon_used field
                    # If so, use only that value instead of the full array
                    if 'epsilon_used' in data:
                        params = [data['epsilon_used'].item()]
                
                for i, seed in enumerate(seeds):
                    all_data.append({
                        'scheduling_mode': scheduling_mode, 'mode': mode,
                        'gc_pos': gc_pos, 'conservation': cons, 'seed': seed,
                        'param_name': param_name, 'param_val': 0,
                        'Accuracy': float(std_metrics['Accuracy'][i]),
                        'SaliencyAUC': float(std_metrics['SaliencyAUC'][i]), 
                        'SaliencySNR': float(std_metrics['SaliencySNR'][i]),
                        'pgd_success_rate': std_pgd_stats[i].get('pgd_success_rate', 0),
                        'pgd_mean_iters_to_flip': std_pgd_stats[i].get('pgd_mean_iters_to_flip', 0)
                    })
                    
                    # For RandomSmoothing with single epsilon, handle differently
                    if mode == 'random_smoothing' and len(params) == 1:
                        # Get the single epsilon value
                        p_val = params[0]
                        # For RS files, rob_metrics arrays have shape (5,1), so we need [i][0] to get scalar
                        rob_acc = float(rob_metrics['Accuracy'][i][0])
                        rob_auc = float(rob_metrics['SaliencyAUC'][i][0])
                        rob_snr = float(rob_metrics['SaliencySNR'][i][0])
                        # For single-epsilon RS files, pgd_stats is a list with one element per seed
                        current_pgd_stats = rob_pgd_stats[i][0] if rob_pgd_stats[i] else {}
                        
                        all_data.append({
                            'scheduling_mode': scheduling_mode, 'mode': mode,
                            'gc_pos': gc_pos, 'conservation': cons, 'seed': seed,
                            'param_name': param_name, 'param_val': p_val,
                            'Accuracy': rob_acc,
                            'SaliencyAUC': rob_auc, 'SaliencySNR': rob_snr,
                            'pgd_success_rate': current_pgd_stats.get('pgd_success_rate', 0),
                            'pgd_mean_iters_to_flip': current_pgd_stats.get('pgd_mean_iters_to_flip', 0),
                        })
                    else:
                        # For adversarial training or old-style RS files with multiple epsilons
                        for j, p_val in enumerate(params):
                            # Handle potentially missing PGD stats in older files
                            pgd_stats_list = rob_pgd_stats[i] if rob_pgd_stats is not None and i < len(rob_pgd_stats) else []
                            current_pgd_stats = pgd_stats_list[j] if pgd_stats_list is not None and j < len(pgd_stats_list) else {}

                            rob_acc = rob_metrics['Accuracy'][i][j]
                            rob_auc = rob_metrics['SaliencyAUC'][i][j]
                            rob_snr = rob_metrics['SaliencySNR'][i][j]

                            all_data.append({
                                'scheduling_mode': scheduling_mode, 'mode': mode,
                                'gc_pos': gc_pos, 'conservation': cons, 'seed': seed,
                                'param_name': param_name, 'param_val': p_val,
                                'Accuracy': rob_acc,
                                'SaliencyAUC': rob_auc, 'SaliencySNR': rob_snr,
                                'pgd_success_rate': current_pgd_stats.get('pgd_success_rate', 0),
                                'pgd_mean_iters_to_flip': current_pgd_stats.get('pgd_mean_iters_to_flip', 0),
                            })
            except Exception as e:
                print(f"Could not process file {f_path}: {e}")
                import traceback
                traceback.print_exc()

        df = pd.DataFrame(all_data)
        master_csv_path = os.path.join(plots_dir, 'full_results_long_format.csv')
        df.to_csv(master_csv_path, index=False)
        print(f"Saved master data table to {master_csv_path}")

        def get_model_type(row):
            if row['param_val'] == 0: return 'Standard'
            if row['mode'] == 'random_smoothing': return 'RandomSmoothing'
            if row['mode'] == 'adv_vs_std' and row['scheduling_mode'] == 'scheduled': return 'Adversarial (Scheduled)'
            if row['mode'] == 'adv_vs_std' and row['scheduling_mode'] == 'no_schedule': return 'Adversarial (No Schedule)'
            return 'Unknown'

        df['model_type'] = df.apply(get_model_type, axis=1)

        baseline_metrics = df[df['model_type'] == 'Standard'].groupby(
            ['gc_pos', 'conservation', 'seed']
        )[['Accuracy', 'SaliencyAUC', 'SaliencySNR']].mean().reset_index()

        df_robust = df[df['model_type'] != 'Standard'].copy()
        
        df_robust = pd.merge(
            df_robust,
            baseline_metrics[['gc_pos', 'conservation', 'seed', 'Accuracy', 'SaliencyAUC', 'SaliencySNR']],
            on=['gc_pos', 'conservation', 'seed'],
            suffixes=('', '_base')
        )

        metrics_to_plot = ['Accuracy', 'SaliencyAUC', 'SaliencySNR']
        for metric in metrics_to_plot:
            df_robust[f'delta_{metric}'] = df_robust[metric] - df_robust[f'{metric}_base']

        # --- Generate Combined Boxplots for all strategies ---
        print("\n--- Generating Combined Boxplots ---")
        baseline_stats_df = df[df['model_type'] == 'Standard'].groupby(['gc_pos', 'scheduling_mode']).agg(
            **{f'{m}_mean': (m, 'mean') for m in metrics_to_plot},
            **{f'{m}_std': (m, 'std') for m in metrics_to_plot}
        ).reset_index()

        df_plot_box = df_robust[df_robust['model_type'].isin(['Adversarial (No Schedule)', 'Adversarial (Scheduled)', 'RandomSmoothing'])]

        # Map model types to their corresponding schedule setting for baseline lookup
        model_type_to_sched = {
            'Adversarial (No Schedule)': 'no_schedule',
            'Adversarial (Scheduled)': 'scheduled',
        }
        
        for metric in metrics_to_plot:
            print(f"  - Plotting combined boxplot for {metric}...")
            if df_plot_box.empty:
                print(f"    Skipping {metric}, no data.")
                continue

            g = sns.catplot(
                data=df_plot_box, x="param_val", y=f"delta_{metric}",
                hue="conservation", col="model_type", row="gc_pos",
                kind="box", height=3, aspect=1.2, palette='viridis',
                fliersize=0, linewidth=1.0, showfliers=False, sharey=False, sharex=False,
                margin_titles=True,
                col_order=['Adversarial (No Schedule)', 'Adversarial (Scheduled)', 'RandomSmoothing'],
                whis=1.5  # Standard IQR whiskers
            )
            
            g.fig.suptitle(f"Improvement in {metric} vs. Standard, by Training Strategy", y=1.05, fontsize=16)

            for (row_idx, col_idx), ax in np.ndenumerate(g.axes):
                if row_idx >= len(g.row_names) or col_idx >= len(g.col_names): continue
                
                gc_val = g.row_names[row_idx]
                model_type = g.col_names[col_idx]
                
                # For adversarial models, we find the baseline from the corresponding
                # scheduled/non-scheduled standard run. RS is compared to the 'no_schedule' std run.
                sched_val = model_type_to_sched.get(model_type, 'no_schedule')
                stats = baseline_stats_df[
                    (baseline_stats_df['gc_pos'] == gc_val) & 
                    (baseline_stats_df['scheduling_mode'] == sched_val)
                ]
                
                title = f"GC={gc_val} | {model_type}"
                if not stats.empty:
                    mean_val = stats[f'{metric}_mean'].iloc[0]
                    std_val = stats[f'{metric}_std'].iloc[0]
                    title += f"\nBaseline: {mean_val:.2f} ± {std_val:.2f}"
                
                ax.set_title(title, fontsize=9)
                ax.axhline(0, ls='--', color='red', zorder=0)

                # Set custom x-axis labels
                if 'Adversarial' in model_type:
                    ax.set_xlabel("Epsilon")
                else:
                    ax.set_xlabel("Target Epsilon")
                ax.tick_params(axis='x', rotation=45, labelsize=8)

                # Unify y-axis labels
                if col_idx == 0:
                    ax.set_ylabel(f"Improvement in {metric}")
                else:
                    ax.set_ylabel("")

            sns.move_legend(g, "upper center", bbox_to_anchor=(.5, 0.99), ncol=len(df_plot_box['conservation'].unique()), title="Conservation", frameon=False)
            g.tight_layout(rect=[0, 0, 1, 0.95])
            plot_path = os.path.join(plots_dir, f"combined_delta_{metric}_boxplot.png")
            g.savefig(plot_path, dpi=200)
            plt.close(g.fig)
            print(f"    Saved to {plot_path}")

        # --- Generate Combined Summary Bar Chart ---
        print("\n--- Generating Combined Summary Bar Chart ---")
        
        id_vars = ['scheduling_mode', 'mode', 'model_type', 'gc_pos', 'conservation', 'seed']
        value_vars = [f'delta_{m}' for m in metrics_to_plot]
        df_deltas = pd.melt(df_robust, id_vars=id_vars, value_vars=value_vars, var_name='metric', value_name='delta_value')
        df_deltas['metric'] = df_deltas['metric'].str.replace('delta_', '')

        df_plot_bar = df_deltas[df_deltas['model_type'].isin(['Adversarial (No Schedule)', 'Adversarial (Scheduled)', 'RandomSmoothing'])]
        
        if not df_plot_bar.empty:
            print("  - Plotting combined summary bar chart...")
            g = sns.catplot(
                data=df_plot_bar, kind='bar', x='gc_pos', y='delta_value',
                hue='conservation', col='model_type', row='metric',
                palette='viridis', height=3, aspect=1.4, sharey=False, margin_titles=True,
                col_order=['Adversarial (No Schedule)', 'Adversarial (Scheduled)', 'RandomSmoothing'],
                row_order=['Accuracy', 'SaliencyAUC', 'SaliencySNR']
            )
            g.set_axis_labels("GC Content (Confounder Strength)", "") # Set X label, but leave Y blank for custom labels
            g.set_titles(col_template="{col_name}", row_template="{row_name}")
            g.fig.suptitle("Comparison of Training Strategies: Mean Improvement vs. Standard Model", y=1.03, fontsize=16)
            
            # Set a specific Y-axis label for each row
            for i, ax_row in enumerate(g.axes):
                metric_name = g.row_names[i]
                ax_row[0].set_ylabel(f"Mean Δ in {metric_name}")

            for ax in g.axes.flat: 
                ax.axhline(0, ls='--', color='gray', zorder=0)
                ax.tick_params(axis='x', rotation=25)
            
            sns.move_legend(g, "upper center", bbox_to_anchor=(.5, 0.98), ncol=len(df_plot_bar['conservation'].unique()), title="Conservation", frameon=False)
            g.tight_layout(rect=[0, 0, 1, 0.95])
            plot_path = os.path.join(plots_dir, f"summary_bar_combined.png")
            g.savefig(plot_path, dpi=200)
            plt.close(g.fig)
            print(f"    Saved to {plot_path}")
        else:
            print("  - Skipping summary bar chart: No data found for comparison.")

        print("\nAll plotting complete.")
        sys.exit(0)

    if args.array_idx is not None:
        main_single_combo(args, args.array_idx)
    else:
        main(args) 