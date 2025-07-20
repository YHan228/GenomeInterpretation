"""
This script runs a synthetic experiment to evaluate the effect of adversarial
training on model interpretability. For a detailed overview, see the file
`docu_toy_slurm.md`.

Author: Yichen Han, 2025-06-29

--------------------------------------------------------------------------------
CHANGELOG (2025-07-16):
  - Model Architecture (`TinyCNN`):
    - Unified architecture to a single "localist" design.
    - The model now uses a single `MaxPool1d(50)` after the first conv block.
  - Experiment Design:
    - Added "Direct HotFlip" as a new adversarial training mode.
    - The experiment now compares: Standard vs. Iterative HotFlip vs. Direct HotFlip.
  - Hyperparameters & Evaluation:
    - Added `GC_pos = 0.5` for a no-confounder baseline.
    - Increased evaluation sample size from 50 to 100 for more stable metrics.
  - Monitoring:
    - Added TensorBoard logging for NaN losses and early stopping.
    - Added lightweight, periodic monitoring of model-level "effect sizes"
      (logit-perturbation based) for both the GC confounder and the causal motif,
      allowing direct observation of how training shifts model reliance.
  - Interpretability Metric (SaliencyAUC): Refined the primary interpretability
    metric to be `Δ(RootSkill)`, a concave and orientation-aware metric that better
    rewards the "escape from randomness" and penalizes misoriented saliency maps.
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
from scipy.stats import norm

try:
    import logomaker
except ImportError:
    print("Error: logomaker is not installed. Please install it using 'pip install logomaker'")
    sys.exit(1)

# --------------------------------------------------------------------------- //
# 1. Configuration & Utilities
# --------------------------------------------------------------------------- #

WITH_CONFOUNDER = True # Global switch for GC-content difference

# --- Hyperparameter Search Space ---
GC_HPARAMS = [0.5, 0.53, 0.55, 0.575, 0.6, 0.625, 0.65, 0.675, 0.7] # n=9
CONS_HPARAMS = [0.6, 0.7, 0.8] # n=3
# ---

SEEDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] # n=10
EPSILONS = [0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.10, 0.15] # n=8

# Directory to cache synthetic datasets so we do not regenerate the same
# (gc, conservation) combination multiple times across different SLURM tasks.
DATASET_CACHE_DIR = "dataset_cache"
os.makedirs(DATASET_CACHE_DIR, exist_ok=True)

# Default training parameters – can be overriden via CLI.
DEFAULT_BATCH_SIZE = 512  # fits comfortably on V100 / T4
DEFAULT_EVAL_BATCH_SIZE = 1024  # Larger batch for evaluation

class GPUPrefetchDataLoader:
    """Wrapper to prefetch data to GPU asynchronously."""
    def __init__(self, dataloader, device):
        self.dataloader = dataloader
        self.device = device
        
    def __iter__(self):
        stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        first = True
        
        for next_batch in self.dataloader:
            if stream:
                with torch.cuda.stream(stream):
                    # Move next batch to GPU asynchronously
                    next_batch_gpu = []
                    for item in next_batch:
                        if torch.is_tensor(item):
                            next_batch_gpu.append(item.to(self.device, non_blocking=True))
                        else:
                            next_batch_gpu.append(item)
            else:
                # CPU fallback
                next_batch_gpu = []
                for item in next_batch:
                    if torch.is_tensor(item):
                        next_batch_gpu.append(item.to(self.device))
                    else:
                        next_batch_gpu.append(item)
            
            if not first:
                yield batch_gpu
            else:
                first = False
                
            if stream:
                # Synchronize the stream
                torch.cuda.current_stream().wait_stream(stream)
            batch_gpu = next_batch_gpu
        
        # Yield the last batch
        if not first:
            yield batch_gpu
    
    def __len__(self):
        return len(self.dataloader)

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
        # Enable fastest cuDNN kernels; reproducibility is ensured by fixed seeds.
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


set_seeds(42)  # initial seed for consistency

ALPH = np.array(list("ACGT"), dtype="U1")
to_ix = {b: i for i, b in enumerate(ALPH)}


def log_gpu_stats(prefix=""):
    """Log GPU utilization and memory usage."""
    if torch.cuda.is_available():
        # Get memory stats
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        
        # Try to get utilization if pynvml is available
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_util = util.gpu
            print(f"{prefix}GPU: {gpu_util}% util, {allocated:.2f}GB/{reserved:.2f}GB mem")
        except:
            print(f"{prefix}GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    return


def sample_background(length: int, gc: float) -> np.ndarray:
    """iid sampling with given GC content, returns char array"""
    p = np.array([(1 - gc) / 2, gc / 2, gc / 2, (1 - gc) / 2])  # A,C,G,T
    return np.random.choice(ALPH, size=length, p=p)


def random_chunk(length: int) -> np.ndarray:
    """60-bp random chunk with balanced GC ≈ 50 %"""
    return sample_background(length, 0.50)


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
            temp_p /= temp_p.sum() # Normalize to make it a probability distribution
            mutated_chunk[pos] = np.random.choice(ALPH, p=temp_p)

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
    
    # --- Atomic saving to prevent race conditions in cluster environments ---
    temp_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=10))
    # Explicitly create a temporary path that ends in .npz
    temp_path = f"{cache_path}.{temp_id}.tmp.npz"
    
    try:
        np.savez(
            temp_path,  # Save directly to the final temp path
            X=ds.x.cpu().numpy(),
            y=ds.y.cpu().numpy(),
            masks=ds.m,
        )
        os.rename(temp_path, cache_path)
    except Exception as e:
        print(f"Error caching dataset: {e}. Cleaning up temp file...")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise # Re-raise the exception after cleanup
        
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
    # The master chunk should be a consistent motif pattern, not random sequence
    # Generate it ONCE with a fixed seed to ensure consistency across all positive examples
    # Use gc_pos to match the background GC content
    rng_state = np.random.get_state()
    np.random.seed(42)  # Fixed seed for consistent motif
    master_chunk = sample_background(CHUNK_LEN, gc_pos)
    np.random.set_state(rng_state)  # Restore random state

    for _ in range(POS_N):
        bg = sample_background(SEQ_LEN, gc=gc_pos)
        chunk = mutate(master_chunk, conservation, gc_target=gc_pos)
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

class TinyCNNv0(nn.Module):
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
        conv3_out = F.relu(self.conv3(x)); x = F.max_pool1d(conv3_out, 2)
        x = self.pool(x).squeeze(-1)
        logits = self.fc(x)
        return logits.squeeze(-1), conv3_out


class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # User-specified k1, with sensible defaults for subsequent layers
        self.k1, self.k2, self.k3 = 30, 3, 3

        # Calculate padding to keep sequence length constant *before* pooling
        p1 = (self.k1 - 1) // 2
        # Padding for subsequent layers operating on pooled output
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
        x = torch.exp(x)
        x = self.dropout1(x)
        
        # Localist pooling: Drastically downsample to get motif presence features
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
        """
        For the localist architecture, the receptive field is conceptually the
        size of the initial motif scanners (k1), as subsequent layers operate
        on a heavily downsampled representation of motif presence.
        """
        return self.k1

# Backwards alias so downstream imports remain valid
TinyCNNv0 = TinyCNNv0

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
                logits, _ = model(xb)
                loss = loss_fn(logits, yb)
            total_loss += loss.item()
            num_batches += 1
    return total_loss / num_batches if num_batches > 0 else 0


def train_standard(model, train_loader, val_loader, loss_fn, optimizer, dev, scaler, writer, scheduler, epochs: int = 10, early_stopping_patience: int = 15, early_stopping_min_delta: float = 1e-4) -> None:
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
                logits, conv_out = model(xb)
                loss = loss_fn(logits, yb)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            num_batches += 1
        
        avg_train_loss = total_loss / num_batches if num_batches > 0 else 0
        if not np.isfinite(avg_train_loss):
            print(f"  WARNING: NaN or Inf average train loss at epoch {epoch + 1}. Stopping training for this model.")
            writer.add_scalar('AbnormalEvents/nan_loss_epoch', epoch + 1, 0)
            writer.flush()
            break
            
        avg_val_loss = validate_epoch(model, val_loader, loss_fn, dev)

        # Step scheduler first, then log the potentially new LR
        scheduler.step(avg_val_loss)
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/validation', avg_val_loss, epoch)
        writer.add_scalar('LR/train', scheduler.optimizer.param_groups[0]['lr'], epoch)
        
        # Compute and log effect sizes every 5 epochs (to avoid bottlenecks)
        if epoch % 5 == 0:
            gc_eff, motif_eff, ratio = compute_effect_sizes_fast(model, val_loader, dev, n_samples=50)
            writer.add_scalar('EffectSize/gc_effect', gc_eff, epoch)
            writer.add_scalar('EffectSize/motif_effect', motif_eff, epoch)
            writer.add_scalar('EffectSize/effect_ratio', ratio, epoch)
            print(f"  Epoch {epoch + 1}/{epochs}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, GC Effect: {gc_eff:.3f}, Motif Effect: {motif_eff:.3f}, Ratio: {ratio:.2f}")
            log_gpu_stats("    ")
        else:
            print(f"  Epoch {epoch + 1}/{epochs}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Check for improvement
        if (best_val_loss - avg_val_loss) > early_stopping_min_delta:
            best_val_loss = avg_val_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
        
        if early_stopping_counter >= early_stopping_patience:
            print(f"  -> Early stopping at epoch {epoch + 1} due to no improvement in val_loss for {early_stopping_patience} epochs.")
            writer.add_scalar('AbnormalEvents/early_stopped_at_epoch', epoch + 1, 0)
            writer.flush()
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
            writer.add_scalar('AbnormalEvents/early_stopped_at_epoch', epoch + 1, 0)
            writer.flush()
            break



def generate_hotflip_examples_optimized(model, xb, yb, loss_fn, flip_fraction: float, 
                                       neighborhood_size: int = 20, penalize_nearby: bool = False):
    """GPU-optimized iterative HotFlip that minimizes CPU-GPU synchronization."""
    seq_len = xb.shape[2]
    batch_size = xb.shape[0]
    k_flips = int(flip_fraction * seq_len)
    adv_xb = xb.clone()
    
    if penalize_nearby:
        forbidden_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=xb.device)
    
    for flip_idx in range(k_flips):
        adv_xb.requires_grad = True
        model.zero_grad()
        
        with autocast():
            logits, _ = model(adv_xb)
            loss = loss_fn(logits, yb)
        loss.backward()
        
        grad = adv_xb.grad.data
        
        # Vectorized saliency computation
        current_bases_onehot = (adv_xb > 0.5).float()
        grad_at_current = (grad * current_bases_onehot).sum(dim=1, keepdim=True)
        saliency = grad - grad_at_current
        saliency.masked_fill_(current_bases_onehot.bool(), -1e9)
        
        # Reshape for easier manipulation: (batch, 4, seq_len) -> (batch, 4*seq_len)
        saliency_flat = saliency.reshape(batch_size, -1)
        
        if penalize_nearby:
            # Create a flat forbidden mask
            forbidden_flat = forbidden_mask.unsqueeze(1).expand(-1, 4, -1).reshape(batch_size, -1)
            saliency_flat.masked_fill_(forbidden_flat, -1e9)
        
        # Find best flip for each sequence in batch
        best_flip_flat_idx = saliency_flat.argmax(dim=1)
        
        # Convert flat indices to (base, position)
        best_new_base = best_flip_flat_idx // seq_len
        best_position = best_flip_flat_idx % seq_len
        
        # Detach for the update
        adv_xb = adv_xb.detach()
        
        # Vectorized update using scatter operations
        batch_indices = torch.arange(batch_size, device=xb.device)
        
        # Find current base at each position
        current_base_at_pos = adv_xb[batch_indices, :, best_position].argmax(dim=1)
        
        # Zero out old bases
        adv_xb[batch_indices, current_base_at_pos, best_position] = 0.0
        
        # Set new bases
        adv_xb[batch_indices, best_new_base, best_position] = 1.0
        
        # Update forbidden regions if needed
        if penalize_nearby:
            # Vectorized forbidden region update
            pos_expanded = best_position.unsqueeze(1)
            seq_positions = torch.arange(seq_len, device=xb.device).unsqueeze(0)
            
            # Mark positions within neighborhood as forbidden
            within_neighborhood = (torch.abs(seq_positions - pos_expanded) <= neighborhood_size)
            forbidden_mask |= within_neighborhood
    
    return adv_xb

def generate_direct_hotflip_examples_optimized(model, xb, yb, loss_fn, flip_fraction: float):
    """GPU-optimized Direct HotFlip using fully vectorized operations."""
    seq_len = xb.shape[2]
    batch_size = xb.shape[0]
    k_flips = int(flip_fraction * seq_len)
    
    adv_xb = xb.clone()
    adv_xb.requires_grad = True
    model.zero_grad()
    
    with autocast():
        logits, _ = model(adv_xb)
        loss = loss_fn(logits, yb)
    loss.backward()
    
    grad = adv_xb.grad.data
    
    # Vectorized saliency computation
    current_bases_onehot = (adv_xb > 0.5).float()
    grad_at_current = (grad * current_bases_onehot).sum(dim=1, keepdim=True)
    saliency = grad - grad_at_current
    saliency.masked_fill_(current_bases_onehot.bool(), -1e9)
    
    # Detach for updates
    adv_xb = adv_xb.detach()
    
    # Reshape for top-k: (batch, 4, seq_len) -> (batch, 4*seq_len)
    saliency_flat = saliency.reshape(batch_size, -1)
    
    # Get top-k flips for each sequence
    topk_values, topk_indices = torch.topk(saliency_flat, k_flips, dim=1)
    
    # Convert to base and position indices
    topk_bases = topk_indices // seq_len  # (batch, k_flips)
    topk_positions = topk_indices % seq_len  # (batch, k_flips)
    
    # Apply all flips using advanced indexing
    for flip_idx in range(k_flips):
        batch_indices = torch.arange(batch_size, device=xb.device)
        positions = topk_positions[:, flip_idx]
        new_bases = topk_bases[:, flip_idx]
        
        # Find and zero out current bases
        current_bases = adv_xb[batch_indices, :, positions].argmax(dim=1)
        adv_xb[batch_indices, current_bases, positions] = 0.0
        
        # Set new bases
        adv_xb[batch_indices, new_bases, positions] = 1.0
    
    return adv_xb

def train_hotflip(model, train_loader, val_loader, loss_fn, optimizer, dev, scaler, writer, scheduler,
                  max_flip_fraction: float, epochs: int = 10, use_scheduling: bool = True, 
                  early_stopping_patience: int = 25, early_stopping_min_delta: float = 1e-4, gc_pos: float = 0.5) -> None:
    
    scheduling_str = "ON" if use_scheduling else "OFF"
    early_stop_str = f"ON (patience={early_stopping_patience}, min_delta={early_stopping_min_delta})"
    print(f"Starting HotFlip training with max_flip_fraction = {max_flip_fraction:.4f}, Scheduling: {scheduling_str}, Early Stopping: {early_stop_str}...")
    
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
        
        # Track adversarial changes for monitoring
        adv_changes_accumulator = {'optimal_acc_change': [], 'ic_change': []}
        
        for xb, yb, mb in train_loader:
            xb, yb, mb = xb.to(dev), yb.to(dev), mb.to(dev)
            adv_xb = generate_hotflip_examples_optimized(model, xb, yb, loss_fn, current_flip_fraction)
            
            # On monitoring epochs, compute adversarial changes for positive samples
            if epoch % 5 == 0 and (yb == 1).any():
                pos_mask = yb == 1
                xb_pos = xb[pos_mask]
                adv_xb_pos = adv_xb[pos_mask]
                mb_pos = mb[pos_mask]
                
                with torch.no_grad():
                    adv_changes = compute_adversarial_changes(xb_pos, adv_xb_pos, mb_pos, gc_pos)
                    if 'optimal_acc_change' in adv_changes:
                        adv_changes_accumulator['optimal_acc_change'].append(adv_changes['optimal_acc_change'])
                    if 'ic_change' in adv_changes:
                        adv_changes_accumulator['ic_change'].append(adv_changes['ic_change'])
            
            optimizer.zero_grad()
            with autocast():
                logits_adv, conv_out_adv = model(adv_xb)
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
        
        # For scheduled training, we help the LR scheduler by telling it to compare
        # against the previous epoch's loss, not the global best, to account for
        # the increasing difficulty. We do this by manually setting its internal state.
        if use_scheduling:
            scheduler.best = previous_val_loss

        # Step scheduler first, then log the potentially new LR
        scheduler.step(avg_val_loss)
        writer.add_scalar('Loss/train_adversarial', avg_train_loss, epoch)
        writer.add_scalar('Loss/validation_adversarial', avg_val_loss, epoch)
        writer.add_scalar('LR/train_adversarial', scheduler.optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('Epsilon/train_adversarial', current_flip_fraction, epoch)
        
        # Compute and log effect sizes every 5 epochs
        if epoch % 5 == 0:
            gc_eff, motif_eff, ratio = compute_effect_sizes_fast(model, val_loader, dev, n_samples=50)
            writer.add_scalar('EffectSize/gc_effect', gc_eff, epoch)
            writer.add_scalar('EffectSize/motif_effect', motif_eff, epoch)
            writer.add_scalar('EffectSize/effect_ratio', ratio, epoch)
            
            # Log adversarial changes
            if adv_changes_accumulator['optimal_acc_change']:
                mean_acc_change = np.mean(adv_changes_accumulator['optimal_acc_change'])
                writer.add_scalar('AdversarialChanges/optimal_acc_change', mean_acc_change, epoch)
            if adv_changes_accumulator['ic_change']:
                mean_ic_change = np.mean(adv_changes_accumulator['ic_change'])
                writer.add_scalar('AdversarialChanges/ic_change', mean_ic_change, epoch)
            
            print(f"  Epoch {epoch + 1}/{epochs}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Epsilon: {current_flip_fraction:.4f}, GC Effect: {gc_eff:.3f}, Motif Effect: {motif_eff:.3f}, Ratio: {ratio:.2f}")
            if adv_changes_accumulator['optimal_acc_change']:
                print(f"    Adversarial Changes: ΔOptAcc: {mean_acc_change:.3f}, ΔIC: {mean_ic_change:.3f} bits")
            log_gpu_stats("    ")
        else:
            print(f"  Epoch {epoch + 1}/{epochs}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Epsilon: {current_flip_fraction:.4f}")

        if not np.isfinite(avg_train_loss):
            print(f"  WARNING: NaN or Inf average train loss at epoch {epoch + 1}. Stopping training for this model.")
            writer.add_scalar('AbnormalEvents/nan_loss_epoch', epoch + 1, 0)
            writer.flush()
            break

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
            writer.add_scalar('AbnormalEvents/early_stopped_at_epoch', epoch + 1, 0)
            writer.flush()
            break

def train_direct_hotflip(model, train_loader, val_loader, loss_fn, optimizer, dev, scaler, writer, scheduler,
                         max_flip_fraction: float, epochs: int = 10, use_scheduling: bool = True, 
                         early_stopping_patience: int = 25, early_stopping_min_delta: float = 1e-4, gc_pos: float = 0.5) -> None:
    
    scheduling_str = "ON" if use_scheduling else "OFF"
    early_stop_str = f"ON (patience={early_stopping_patience}, min_delta={early_stopping_min_delta})"
    print(f"Starting Direct HotFlip training with max_flip_fraction = {max_flip_fraction:.4f}, Scheduling: {scheduling_str}, Early Stopping: {early_stop_str}...")
    
    previous_val_loss = float('inf')
    early_stopping_counter = 0

    for epoch in range(epochs):
        
        max_flips = int(max_flip_fraction * SEQ_LEN)
        
        if use_scheduling:
            # Smart scheduling: Linearly ramp up from 1 flip to max_flips.
            num_flips = 1 + int(np.floor((max_flips - 1) * (epoch / (epochs - 1)))) if epochs > 1 else max_flips
            current_flip_fraction = num_flips / SEQ_LEN
        else:
            current_flip_fraction = max_flip_fraction
        
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Track adversarial changes for monitoring
        adv_changes_accumulator = {'optimal_acc_change': [], 'ic_change': []}
        
        for xb, yb, mb in train_loader:
            xb, yb, mb = xb.to(dev), yb.to(dev), mb.to(dev)
            adv_xb = generate_direct_hotflip_examples_optimized(model, xb, yb, loss_fn, current_flip_fraction)
            
            # On monitoring epochs, compute adversarial changes for positive samples
            if epoch % 5 == 0 and (yb == 1).any():
                pos_mask = yb == 1
                xb_pos = xb[pos_mask]
                adv_xb_pos = adv_xb[pos_mask]
                mb_pos = mb[pos_mask]
                
                with torch.no_grad():
                    adv_changes = compute_adversarial_changes(xb_pos, adv_xb_pos, mb_pos, gc_pos)
                    if 'optimal_acc_change' in adv_changes:
                        adv_changes_accumulator['optimal_acc_change'].append(adv_changes['optimal_acc_change'])
                    if 'ic_change' in adv_changes:
                        adv_changes_accumulator['ic_change'].append(adv_changes['ic_change'])
            
            optimizer.zero_grad()
            with autocast():
                logits_adv, conv_out_adv = model(adv_xb)
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
        
        # For scheduled training, we help the LR scheduler by telling it to compare
        # against the previous epoch's loss, not the global best
        if use_scheduling:
            scheduler.best = previous_val_loss

        # Step scheduler first, then log the potentially new LR
        scheduler.step(avg_val_loss)
        writer.add_scalar('Loss/train_direct_hotflip', avg_train_loss, epoch)
        writer.add_scalar('Loss/validation_direct_hotflip', avg_val_loss, epoch)
        writer.add_scalar('LR/train_direct_hotflip', scheduler.optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('Epsilon/train_direct_hotflip', current_flip_fraction, epoch)
        
        # Compute and log effect sizes every 5 epochs
        if epoch % 5 == 0:
            gc_eff, motif_eff, ratio = compute_effect_sizes_fast(model, val_loader, dev, n_samples=50)
            writer.add_scalar('EffectSize/gc_effect', gc_eff, epoch)
            writer.add_scalar('EffectSize/motif_effect', motif_eff, epoch)
            writer.add_scalar('EffectSize/effect_ratio', ratio, epoch)
            
            # Log adversarial changes
            if adv_changes_accumulator['optimal_acc_change']:
                mean_acc_change = np.mean(adv_changes_accumulator['optimal_acc_change'])
                writer.add_scalar('AdversarialChanges/optimal_acc_change', mean_acc_change, epoch)
            if adv_changes_accumulator['ic_change']:
                mean_ic_change = np.mean(adv_changes_accumulator['ic_change'])
                writer.add_scalar('AdversarialChanges/ic_change', mean_ic_change, epoch)
            
            print(f"  Epoch {epoch + 1}/{epochs}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Epsilon: {current_flip_fraction:.4f}, GC Effect: {gc_eff:.3f}, Motif Effect: {motif_eff:.3f}, Ratio: {ratio:.2f}")
            if adv_changes_accumulator['optimal_acc_change']:
                print(f"    Adversarial Changes: ΔOptAcc: {mean_acc_change:.3f}, ΔIC: {mean_ic_change:.3f} bits")
            log_gpu_stats("    ")
        else:
            print(f"  Epoch {epoch + 1}/{epochs}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Epsilon: {current_flip_fraction:.4f}")

        if not np.isfinite(avg_train_loss):
            print(f"  WARNING: NaN or Inf average train loss at epoch {epoch + 1}. Stopping training for this model.")
            writer.add_scalar('AbnormalEvents/nan_loss_epoch', epoch + 1, 0)
            writer.flush()
            break

        # Early stopping logic
        if (previous_val_loss - avg_val_loss) > early_stopping_min_delta:
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
        
        previous_val_loss = avg_val_loss

        if early_stopping_counter >= early_stopping_patience:
            print(f"  -> Early stopping at epoch {epoch + 1} due to no improvement in val_loss for {early_stopping_patience} consecutive epochs.")
            writer.add_scalar('AbnormalEvents/early_stopped_at_epoch', epoch + 1, 0)
            writer.flush()
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
        initial_logits, _ = model(adv_xb)
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
            logits, _ = model(adv_xb)
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

            current_logits, _ = model(adv_xb)
            current_pred_class = (current_logits > 0).float()
            
            if current_pred_class.item() != initial_pred_class.item():
                stats['success'] = True
                stats['final_logit'] = current_logits.item()
                stats['found_at_iter'] = i + 1
                return adv_xb.detach(), stats
    
    # If no flip was found, final logit is the last one computed
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
    print(f"Evaluating model...")
    SAMPLE_N = 100  # Increased to 100 as requested
    ANALYSIS_CHUNK_LEN = 60
    PGD_BATCH_SIZE = 50  # Increased for better GPU utilization
    IG_BATCH_SIZE = 25   # Increased for better GPU utilization

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
        # Return default PGD stats for empty case
        default_pgd_stats = {'pgd_success_rate': 0, 'pgd_mean_iters_to_flip': 0}
        return 0.0, accuracy, 0.0, 0.0, default_pgd_stats
        
    idxs = rng.choice(positive_subset_indices, size=sample_n_actual, replace=False)

    # --- Compute model fingerprint for caching ---
    if pgd_cache is None:
        pgd_cache = {}
    
    # Simple fingerprint based on model weights
    with torch.no_grad():
        # Use first conv layer weights as fingerprint
        if hasattr(model, '_orig_mod'):  # Compiled model
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
        print(f"  Computing PGD baselines in batches...")
        for batch_start in range(0, len(idxs), PGD_BATCH_SIZE):
            batch_end = min(batch_start + PGD_BATCH_SIZE, len(idxs))
            batch_idxs = idxs[batch_start:batch_end]
            
            # Prepare batch
            xb_list, yb_list, mask_list = [], [], []
            for idx in batch_idxs:
                xb, y_scalar, mask = test_ds[idx]
                xb_list.append(xb)
                yb_list.append(y_scalar)
                mask_list.append(mask)
            
            xb_batch = torch.stack(xb_list).to(dev)
            yb_batch = torch.tensor(yb_list, device=dev, dtype=torch.float)
            
            # Batch PGD computation
            pgd_baselines_batch, pgd_stats_batch = find_adversarial_baseline_pgd_batch_optimized(
                model, xb_batch, yb_batch, dev
            )
            
            all_pgd_baselines.extend([pgd_baselines_batch[i] for i in range(len(batch_idxs))])
            all_pgd_stats.extend(pgd_stats_batch)
        
        # Cache the results
        pgd_cache[fingerprint] = {
            'baselines': all_pgd_baselines,
            'stats': all_pgd_stats
        }

    # --- Batched IG Processing ---
    results = []
    print(f"  Computing Integrated Gradients in batches...")
    
    for batch_start in range(0, len(idxs), IG_BATCH_SIZE):
        batch_end = min(batch_start + IG_BATCH_SIZE, len(idxs))
        batch_range = range(batch_start, batch_end)
        
        # Prepare batch
        xb_list, mask_list, baseline_list = [], [], []
        
        for i in batch_range:
            idx = idxs[i]
            xb, y_scalar, mask = test_ds[idx]
            xb_list.append(xb)
            mask_list.append(mask)
            
            # Use PGD baseline if successful, else compositional
            if all_pgd_stats[i]['success']:
                baseline_list.append(all_pgd_baselines[i].squeeze(0).cpu())
            else:
                proportions = xb.mean(dim=1, keepdim=True)
                baseline_list.append(proportions.expand_as(xb))
        
        # Stack for batch processing
        xb_batch = torch.stack(xb_list).to(dev)
        baseline_batch = torch.stack(baseline_list).to(dev)
        
        # Batch IG computation
        raw_attributions_batch = ig.attribute(xb_batch, baselines=baseline_batch, target=0)
        
        # Process each sample in the batch
        for j, i in enumerate(batch_range):
            raw_attr = raw_attributions_batch[j]
            xb = xb_batch[j]
            mask = mask_list[j]
            
            # Apply gradient correction
            corrected_attr = raw_attr - raw_attr.mean(dim=0, keepdim=True)
            
            # Calculate final contribution scores
            attributions = np.abs((corrected_attr * xb).sum(0).cpu().numpy())
            
            # Compute metrics
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
            
            sum_sq_inside = np.sum(inside_scores**2)
            sum_sq_total = np.sum(attributions**2)
            saliency_snr = sum_sq_inside / (sum_sq_total + 1e-9)
            
            results.append(dict(iou_cont=iou_cont, saliency_auc=saliency_auc, saliency_snr=saliency_snr))

    # --- Aggregate PGD Stats ---
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

    # Calculate and print mean values
    mean_iou_cont = np.mean([r['iou_cont'] for r in results]) if results else 0.0
    mean_saliency_auc = np.mean([r['saliency_auc'] for r in results]) if results else 0.0
    mean_saliency_snr = np.mean([r['saliency_snr'] for r in results]) if results else 0.0

    print(f"  Mean wIoU: {mean_iou_cont:.3f}")
    print(f"  Mean Saliency AUC: {mean_saliency_auc:.3f}")
    print(f"  Mean Saliency SNR: {mean_saliency_snr:.3f}")

    return mean_iou_cont, accuracy, mean_saliency_auc, mean_saliency_snr, pgd_stats

# --------------------------------------------------------------------------- #
# 4.b Effect Size Monitoring
# --------------------------------------------------------------------------- #

def compute_effect_sizes_fast(model, val_loader, dev, n_samples=50):
    """
    Lightweight effect size computation for monitoring during training.
    Uses logit perturbation to measure model reliance on GC vs motif.
    
    Returns:
        gc_effect: Average |Δlogit| when neutralizing GC content
        motif_effect: Average |Δlogit| when ablating the motif
        effect_ratio: motif_effect / (gc_effect + 1e-9)
    """
    model.eval()
    gc_deltas, motif_deltas = [], []
    
    with torch.no_grad():
        sample_count = 0
        for xb, yb, masks in val_loader:
            if sample_count >= n_samples:
                break
                
            xb, yb, masks = xb.to(dev), yb.to(dev), masks
            # Only evaluate on positive samples
            pos_mask = yb == 1
            if not pos_mask.any():
                continue
                
            xb_pos = xb[pos_mask]
            masks_pos = masks[pos_mask.cpu().numpy()]
            
            # Original logits
            with autocast():
                logits_orig, _ = model(xb_pos)
            
            # 1. GC neutralization: Set everything outside motif to uniform (0.25)
            xb_gc_neutral = xb_pos.clone()
            for i in range(len(xb_pos)):
                mask = masks_pos[i].to(dev)
                xb_gc_neutral[i, :, ~mask] = 0.25
            
            with autocast():
                logits_gc_neutral, _ = model(xb_gc_neutral)
            gc_delta = (logits_orig - logits_gc_neutral).abs().mean().item()
            gc_deltas.append(gc_delta)
            
            # 2. Motif ablation: Set motif region to uniform
            xb_motif_ablated = xb_pos.clone()
            for i in range(len(xb_pos)):
                mask = masks_pos[i].to(dev)
                xb_motif_ablated[i, :, mask] = 0.25
            
            with autocast():
                logits_motif_ablated, _ = model(xb_motif_ablated)
            motif_delta = (logits_orig - logits_motif_ablated).abs().mean().item()
            motif_deltas.append(motif_delta)
            
            sample_count += len(xb_pos)
    
    gc_effect = np.mean(gc_deltas) if gc_deltas else 0.0
    motif_effect = np.mean(motif_deltas) if motif_deltas else 0.0
    effect_ratio = motif_effect / (gc_effect + 1e-9)
    
    return gc_effect, motif_effect, effect_ratio


def compute_sequence_properties_gpu(xb: torch.Tensor, masks: torch.Tensor = None) -> dict:
    """
    Compute GC content and conservation metrics directly on GPU.
    
    Args:
        xb: One-hot encoded sequences (B, 4, L)
        masks: Binary masks indicating motif positions (B, L)
    
    Returns:
        Dictionary with mean GC%, and if masks provided, conservation within motifs
    """
    # GC content: positions 1 and 2 in the one-hot encoding are C and G
    gc_positions = xb[:, 1:3, :].sum(dim=1)  # Sum C and G channels
    total_positions = xb.sum(dim=1)  # Should be all 1s
    gc_content = (gc_positions.sum(dim=1) / total_positions.sum(dim=1)).mean().item()
    
    results = {'gc_content': gc_content}
    
    if masks is not None:
        # For conservation, we'd need the original motif pattern
        # For now, return the "purity" of the motif regions
        # (how much they deviate from uniform distribution)
        motif_positions = masks.sum().item()
        if motif_positions > 0:
            # Calculate entropy within motif regions as a proxy
            motif_one_hot = xb * masks.unsqueeze(1)
            base_freqs = motif_one_hot.sum(dim=(0, 2)) / motif_positions
            # Avoid log(0)
            epsilon = 1e-10
            entropy = -(base_freqs * torch.log2(base_freqs + epsilon)).sum().item()
            max_entropy = 2.0  # log2(4) for uniform distribution
            # Convert entropy to a "conservation-like" score (0=uniform, 1=perfect)
            conservation_score = 1.0 - (entropy / max_entropy)
            results['motif_conservation'] = conservation_score
    
    return results


def compute_adversarial_changes(xb_orig: torch.Tensor, xb_adv: torch.Tensor, 
                               masks: torch.Tensor = None, gc_pos: float = 0.5) -> dict:
    """
    Compute the changes in sequence properties after adversarial perturbation.
    Reports metrics in terms of optimal classifier accuracy and IC.
    
    Args:
        xb_orig: Original one-hot sequences (B, 4, L)
        xb_adv: Adversarial one-hot sequences (B, 4, L)
        masks: Binary masks for motif positions (B, L)
        gc_pos: Expected GC content for positive class
    
    Returns:
        Dictionary with changes in optimal classifier accuracy and IC
    """
    # Compute GC content for both
    gc_orig = (xb_orig[:, 1:3, :].sum(dim=1).sum(dim=1) / xb_orig.shape[2]).mean().item()
    gc_adv = (xb_adv[:, 1:3, :].sum(dim=1).sum(dim=1) / xb_adv.shape[2]).mean().item()
    
    # Convert to optimal classifier accuracy using Cohen's d
    # Assuming negative class has GC=0.5, and std dev of ~0.015 for 1000bp sequences
    gc_neg_mean, gc_neg_std = 0.5, 0.015
    gc_pos_std = 0.015  # Approximate, could compute from actual distribution
    
    def gc_to_optimal_acc(gc_content, gc_expected=gc_pos):
        """Convert GC content to optimal classifier accuracy via Cohen's d"""
        # For actual positive samples, compute how well they can be distinguished from negatives
        pooled_std = np.sqrt((gc_pos_std**2 + gc_neg_std**2) / 2)
        cohen_d = abs(gc_content - gc_neg_mean) / pooled_std
        # Convert to accuracy using normal CDF
        from scipy.stats import norm
        accuracy = norm.cdf(cohen_d / 2)
        return accuracy
    
    acc_orig = gc_to_optimal_acc(gc_orig)
    acc_adv = gc_to_optimal_acc(gc_adv)
    
    results = {
        'gc_orig': gc_orig,
        'gc_adv': gc_adv,
        'optimal_acc_orig': acc_orig,
        'optimal_acc_adv': acc_adv,
        'optimal_acc_change': acc_adv - acc_orig
    }
    
    # If we have masks, compute IC changes in motif regions
    if masks is not None and masks.sum() > 0:
        # Extract motif regions
        motif_orig = (xb_orig * masks.unsqueeze(1))
        motif_adv = (xb_adv * masks.unsqueeze(1))
        
        # Compute base frequencies in motif
        motif_positions = masks.sum()
        base_freqs_orig = motif_orig.sum(dim=(0, 2)) / motif_positions
        base_freqs_adv = motif_adv.sum(dim=(0, 2)) / motif_positions
        
        # Compute IC (KL divergence from uniform)
        uniform_freq = 0.25
        epsilon = 1e-10
        
        def compute_ic(freqs):
            """Compute information content in bits"""
            kl_div = (freqs * torch.log2((freqs + epsilon) / uniform_freq)).sum()
            return kl_div.item()
        
        ic_orig = compute_ic(base_freqs_orig)
        ic_adv = compute_ic(base_freqs_adv)
        
        results.update({
            'ic_orig': ic_orig,
            'ic_adv': ic_adv,
            'ic_change': ic_adv - ic_orig
        })
    
    return results

# --------------------------------------------------------------------------- #
# 5. Experiment Runner
# --------------------------------------------------------------------------- #

def run_single_experiment(args, seed: int, main_ds, tb_run_dir: str, npz_run_dir: str, epochs: int, use_scheduling: bool, gc_pos: float = 0.5):
    print(f"\n{'=' * 20}  SEED {seed} | Schedule: {use_scheduling}  {'=' * 20}")
    
    # Create main training, validation, and test splits (70-15-15)
    train_size = int(0.70 * N_TOTAL)
    val_size = int(0.15 * N_TOTAL)
    test_size = N_TOTAL - train_size - val_size
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
        prefetch_factor=2 if effective_num_workers > 0 else None,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=DEFAULT_EVAL_BATCH_SIZE,
        num_workers=effective_num_workers,
        pin_memory=True,
        persistent_workers=effective_num_workers > 0,
        prefetch_factor=2 if effective_num_workers > 0 else None,
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=DEFAULT_EVAL_BATCH_SIZE,
        num_workers=effective_num_workers,
        pin_memory=True,
        persistent_workers=effective_num_workers > 0,
        prefetch_factor=2 if effective_num_workers > 0 else None,
    )

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bce = nn.BCEWithLogitsLoss()
    scaler = GradScaler()
    
    # --- PGD Cache for this seed ---
    # This cache persists across all models in this seed, avoiding redundant PGD computations
    pgd_cache = {}

    # --- Standard Model ---
    set_seeds(seed, deterministic=args.deterministic)
    standard_model = TinyCNN().to(dev)
    # Optional: use PyTorch 2.x dynamic compilation for ~10-20% speed-up
    if hasattr(torch, "compile"):
        try:
            standard_model = torch.compile(standard_model)
            print("  ✓ Standard model compilation successful")
        except Exception as e:
            print(f"  ✗ Standard model compilation failed: {type(e).__name__}")
    opt_standard = torch.optim.AdamW(standard_model.parameters(), lr=3e-4, weight_decay=1e-6)
    std_writer = SummaryWriter(log_dir=os.path.join(tb_run_dir, f"seed_{seed}", "standard"))
    std_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_standard, 'min', factor=0.5, patience=8, verbose=True)

    train_standard(standard_model, train_dl, val_dl, bce, opt_standard, dev, scaler, std_writer, std_scheduler, epochs=epochs, early_stopping_patience=15)
    
    # print("Standard model training complete. Saving and logging final motifs...")
    final_epoch_count = epochs # In future could get this from train function
    # log_conv1_motifs(standard_model, std_writer, final_epoch_count, output_dir=os.path.join(npz_run_dir, "standard"), save_to_disk=True, log_to_tb=True)
    
    std_wiou, std_acc, std_auc, std_snr, std_pgd_stats = evaluate_model(standard_model, test_dl, dev, pgd_cache)
    std_writer.add_hparams(
        {'model': 'standard', 'epsilon': 0, 'seed': seed},
        {'hparam/accuracy': std_acc, 'hparam/wIoU': std_wiou, 'hparam/saliency_auc': std_auc, 'hparam/saliency_snr': std_snr}
    )
    std_writer.close()

    robust_wious, robust_accs, robust_aucs, robust_snrs, robust_pgd_stats_list = [], [], [], [], []
    
    # --- Adversarial / Robustness Training Loop ---
    training_params = []
    if args.experiment_mode == 'adv_vs_std':
        param_iterator = EPSILONS
        training_func = train_hotflip
        param_name = 'epsilon'
    # elif args.experiment_mode == 'random_smoothing':  # DEACTIVATED
    #     param_iterator = RS_EPSILON_HPARAMS
    #     training_func = train_random_smoothing
    #     param_name = 'target_epsilon'
    elif args.experiment_mode == 'direct_hotflip':
        param_iterator = EPSILONS
        training_func = train_direct_hotflip
        param_name = 'epsilon'
    else:
        param_iterator = []

    for param_val in param_iterator:
        set_seeds(seed, deterministic=args.deterministic)
        mdl = TinyCNN().to(dev)
        if hasattr(torch, "compile"):
            try:
                mdl = torch.compile(mdl)
                print(f"  ✓ Robust model (ε={param_val:.4f}) compilation successful")
            except Exception as e:
                print(f"  ✗ Robust model (ε={param_val:.4f}) compilation failed: {type(e).__name__}")
        opt = torch.optim.AdamW(mdl.parameters(), lr=3e-4, weight_decay=1e-6)
        rob_writer = SummaryWriter(log_dir=os.path.join(tb_run_dir, f"seed_{seed}", f"{param_name}_{param_val:.4f}"))
        rob_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', factor=0.5, patience=8, verbose=True)
        
        # Branch for training function and its specific arguments
        if args.experiment_mode == 'adv_vs_std':
            if param_val > 0:
                patience = 25 if use_scheduling else 15
                train_hotflip(mdl, train_dl, val_dl, bce, opt, dev, scaler, rob_writer, rob_scheduler, 
                              max_flip_fraction=param_val, epochs=epochs, use_scheduling=use_scheduling, early_stopping_patience=patience, gc_pos=gc_pos)
                
                motif_out_dir = os.path.join(npz_run_dir, f"hotflip_{param_name}_{param_val:.4f}")
                # print(f"Robust model (param={param_val:.4f}) training complete. Saving and logging final motifs...")
                # log_conv1_motifs(mdl, rob_writer, epochs, output_dir=motif_out_dir, save_to_disk=True, log_to_tb=True)

                wio, acc, auc, snr, pgd_stats = evaluate_model(mdl, test_dl, dev, pgd_cache)
                hparams = {'model': 'robust', 'epsilon': param_val, 'seed': seed}
            else: # eps = 0 is just the standard model
                wio, acc, auc, snr, pgd_stats = std_wiou, std_acc, std_auc, std_snr, std_pgd_stats
                hparams = {'model': 'standard', 'epsilon': 0, 'seed': seed}
        
        # elif args.experiment_mode == 'random_smoothing':  # DEACTIVATED
        #     train_random_smoothing(mdl, train_dl, val_dl, bce, opt, dev, scaler, rob_writer, rob_scheduler,
        #                            target_epsilon=param_val, epochs=epochs, early_stopping_patience=15)

        #     motif_out_dir = os.path.join(npz_run_dir, f"rs_{param_name}_{param_val:.4f}")
        #     # print(f"Robust model (param={param_val:.4f}) training complete. Saving and logging final motifs...")
        #     # log_conv1_motifs(mdl, rob_writer, epochs, output_dir=motif_out_dir, save_to_disk=True, log_to_tb=True)

        #     wio, acc, auc, snr, pgd_stats = evaluate_model(mdl, test_dl, dev, pgd_cache)
        #     hparams = {'model': 'robust_rs', 'target_epsilon': param_val, 'seed': seed}

        elif args.experiment_mode == 'direct_hotflip':
            train_direct_hotflip(mdl, train_dl, val_dl, bce, opt, dev, scaler, rob_writer, rob_scheduler,
                                 max_flip_fraction=param_val, epochs=epochs, use_scheduling=use_scheduling, early_stopping_patience=25, gc_pos=gc_pos)

            motif_out_dir = os.path.join(npz_run_dir, f"direct_hotflip_{param_name}_{param_val:.4f}")
            wio, acc, auc, snr, pgd_stats = evaluate_model(mdl, test_dl, dev, pgd_cache)
            hparams = {'model': 'robust_direct_hotflip', 'epsilon': param_val, 'seed': seed}

        else: # Should not happen
            continue

        rob_writer.add_hparams(
            hparams,
            {'hparam/accuracy': acc, 'hparam/wIoU': wio, 'hparam/saliency_auc': auc, 'hparam/saliency_snr': snr}
        )
        robust_wious.append(wio); robust_accs.append(acc); robust_aucs.append(auc); robust_snrs.append(snr); robust_pgd_stats_list.append(pgd_stats)
        rob_writer.close()

    return std_wiou, std_acc, std_auc, std_snr, std_pgd_stats, robust_wious, robust_accs, robust_aucs, robust_snrs, robust_pgd_stats_list


# --------------------------------------------------------------------------- #
# 6. Main entry-point
# --------------------------------------------------------------------------- #

def run_experiment_combo(args, combo: dict):
    """
    Runs the full experiment (all seeds) for a single hyperparameter combination.
    This function is the core execution unit called by both the sweep-runner
    and the single-job runner.
    """
    gc_hparam, cons_hparam = combo['gc'], combo['cons']

    # --- Set up paths and print info ---
    if args.experiment_mode in ['adv_vs_std', 'direct_hotflip']:
        use_scheduling = combo['schedule']
        schedule_mode_str = "scheduled" if use_scheduling else "no_schedule"
        mode_str = "iterative_hotflip" if args.experiment_mode == 'adv_vs_std' else "direct_hotflip"
        run_spec_str = f"gc_{gc_hparam:.3f}_cons_{cons_hparam:.2f}"
        tb_run_dir = os.path.join(args.output_dir, "tensorboard", mode_str, schedule_mode_str, run_spec_str)
        npz_run_dir = os.path.join(args.output_dir, "npz_results", mode_str, schedule_mode_str, run_spec_str)
        print(f"Running combo: mode={mode_str}, schedule={schedule_mode_str}, GC={gc_hparam:.3f}, CONS={cons_hparam:.2f}")
    
    else:
        raise ValueError(f"Unknown experiment mode: {args.experiment_mode}")

    os.makedirs(tb_run_dir, exist_ok=True)
    os.makedirs(npz_run_dir, exist_ok=True)

    print(f"  - Tensorboard logs will be saved to: {tb_run_dir}")
    print(f"  - NPZ results will be saved to: {npz_run_dir}")

    main_dataset = load_or_generate_dataset(gc_pos=gc_hparam, conservation=cons_hparam)

    all_std_wious, all_std_accs, all_std_aucs, all_std_snrs, all_std_pgd_stats = [], [], [], [], []
    all_rob_wious, all_rob_accs, all_rob_aucs, all_rob_snrs, all_rob_pgd_stats = [], [], [], [], []

    for sd in SEEDS:
        sw, sa, sa_auc, s_snr, s_pgd, rw, ra, ra_auc, r_snr, r_pgd = run_single_experiment(
            args, sd, main_dataset, tb_run_dir, npz_run_dir, args.epochs, use_scheduling, gc_pos=gc_hparam
        )
        all_std_wious.append(sw); all_std_accs.append(sa); all_std_aucs.append(sa_auc); all_std_snrs.append(s_snr); all_std_pgd_stats.append(s_pgd)
        all_rob_wious.append(rw); all_rob_accs.append(ra); all_rob_aucs.append(ra_auc); all_rob_snrs.append(r_snr); all_rob_pgd_stats.append(r_pgd)

    # Save raw results
    save_payload = {
        'seeds': SEEDS, 'gc_pos': gc_hparam, 'conservation': cons_hparam,
        'std_wious': all_std_wious, 'std_accs': all_std_accs, 'std_aucs': all_std_aucs, 'std_snrs': all_std_snrs, 'std_pgd_stats': all_std_pgd_stats,
        'rob_wious': all_rob_wious, 'rob_accs': all_rob_accs, 'rob_aucs': all_rob_aucs, 'rob_snrs': all_rob_snrs, 'rob_pgd_stats': all_rob_pgd_stats,
    }
    if args.experiment_mode == 'adv_vs_std':
        save_payload['epsilons'] = EPSILONS
        save_payload['scheduling'] = use_scheduling
    # elif args.experiment_mode == 'random_smoothing':  # DEACTIVATED
    #     save_payload['rs_epsilons'] = RS_EPSILON_HPARAMS
    #     save_payload['epsilon_used'] = combo['epsilon']
    elif args.experiment_mode == 'direct_hotflip':
        save_payload['direct_hotflip_epsilons'] = EPSILONS
        save_payload['scheduling'] = use_scheduling  # Added missing scheduling field

    np.savez(os.path.join(npz_run_dir, "multi_seed_results.npz"), **save_payload)
    print("Combo job finished and results saved.")

def get_experiment_combos(experiment_mode: str) -> List[dict]:
    """Generates the list of all hyperparameter combinations for a given mode."""
    combos = []
    if experiment_mode in ['adv_vs_std', 'direct_hotflip']:
        for schedule in [True, False]:
            for gc_val in GC_HPARAMS:
                for cons_val in CONS_HPARAMS:
                    combos.append({'schedule': schedule, 'gc': gc_val, 'cons': cons_val})
    # elif experiment_mode == 'random_smoothing':  # DEACTIVATED
    #     for gc_val in GC_HPARAMS:
    #         for cons_val in CONS_HPARAMS:
    #             for eps_val in RS_EPSILON_HPARAMS:
    #                 combos.append({'gc': gc_val, 'cons': cons_val, 'epsilon': eps_val})
    else:
        raise ValueError(f"Unknown experiment mode: {experiment_mode}")
    return combos

def main(args):
    """
    Main entry point for running a full sweep of experiments.
    This function iterates through all defined hyperparameter combinations
    for the selected experiment mode and executes them sequentially.
    """
    print(f"Starting full experiment sweep for mode: '{args.experiment_mode}'")
    combos = get_experiment_combos(args.experiment_mode)
    print(f"Total combinations to run: {len(combos)}")

    for i, combo in enumerate(combos):
        print(f"\n{'='*80}")
        print(f"RUNNING COMBO {i+1} / {len(combos)}")
        print(f"{'='*80}")
        run_experiment_combo(args, combo)

    print("\n\nFull sweep finished. To generate plots, run with --aggregate_only.")
    return


def main_single_combo(args, array_idx: int):
    """
    Run experiments for a single combination determined by a SLURM array_idx.
    """
    combos = get_experiment_combos(args.experiment_mode)
    num_jobs = len(combos)
    if array_idx < 0 or array_idx >= num_jobs:
        raise ValueError(
            f"array_idx {array_idx} is out of range for mode '{args.experiment_mode}' (0-{num_jobs - 1})")

    combo = combos[array_idx]
    run_experiment_combo(args, combo)

    # We do **not** run the heavyweight plotting routines in array mode –
    # collect plots in a downstream aggregation step to save GPU time.
    
# --------------------------------------------------------------------------- #
# 7. Data-loading defaults (set after CLI parsing)
# --------------------------------------------------------------------------- #

TRAIN_BATCH_SIZE = DEFAULT_BATCH_SIZE
NUM_WORKERS = 2  # reduced to avoid DataLoader warnings on compute nodes

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
        default=2,
        help="DataLoader workers (overrides SLURM_CPUS_PER_TASK if provided).",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic cuDNN kernels for full reproducibility.",
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
        for f_path in npz_files:
            try:
                data = np.load(f_path, allow_pickle=True)
                
                # Infer experiment type from path
                if 'iterative_hotflip' in f_path:
                    mode = 'iterative_hotflip'
                elif 'direct_hotflip' in f_path:
                    mode = 'direct_hotflip'
                else: # Fallback for older runs
                    mode = 'iterative_hotflip'

                is_scheduled = data.get('scheduling') and data['scheduling'].item()
                scheduling_mode = "scheduled" if is_scheduled else "no_schedule"
                
                gc_pos = float(data['gc_pos'].item())
                cons = float(data['conservation'].item())
                seeds = data['seeds']
                
                std_metrics = {
                    'wIoU': data.get('std_wious', []), 'Accuracy': data.get('std_accs', []),
                    'SaliencyAUC': data.get('std_aucs', []), 'SaliencySNR': data.get('std_snrs', [])
                }
                std_pgd_stats = data.get('std_pgd_stats', [{} for _ in seeds])

                rob_metrics = {
                    'wIoU': data.get('rob_wious', []), 'Accuracy': data.get('rob_accs', []),
                    'SaliencyAUC': data.get('rob_aucs', []), 'SaliencySNR': data.get('rob_snrs', [])
                }
                rob_pgd_stats = data.get('rob_pgd_stats', [[] for _ in seeds])

                if mode == 'iterative_hotflip':
                    params = data['epsilons']
                    param_name = 'epsilon'
                elif mode == 'direct_hotflip':
                    # Use the correct key for direct_hotflip, with a fallback for older file formats.
                    params = data['direct_hotflip_epsilons'] if 'direct_hotflip_epsilons' in data else data['epsilons']
                    param_name = 'epsilon'
                else: # Should not happen with current script
                    params = []
                    param_name = 'param'
                
                for i, seed in enumerate(seeds):
                    # Add standard model results (param_val = 0)
                    all_data.append({
                        'scheduling_mode': scheduling_mode, 'mode': mode,
                        'gc_pos': gc_pos, 'conservation': cons, 'seed': seed,
                        'param_name': param_name, 'param_val': 0,
                        'wIoU': std_metrics['wIoU'][i], 'Accuracy': std_metrics['Accuracy'][i],
                        'SaliencyAUC': std_metrics['SaliencyAUC'][i], 'SaliencySNR': std_metrics['SaliencySNR'][i],
                        'pgd_success_rate': std_pgd_stats[i].get('pgd_success_rate', 0),
                        'pgd_mean_iters_to_flip': std_pgd_stats[i].get('pgd_mean_iters_to_flip', 0)
                    })
                    
                    # Add robust model results
                    for j, p_val in enumerate(params):
                        pgd_stats_list = rob_pgd_stats[i] if rob_pgd_stats is not None and i < len(rob_pgd_stats) else []
                        current_pgd_stats = pgd_stats_list[j] if pgd_stats_list is not None and j < len(pgd_stats_list) else {}
                        
                        all_data.append({
                            'scheduling_mode': scheduling_mode, 'mode': mode,
                            'gc_pos': gc_pos, 'conservation': cons, 'seed': seed,
                            'param_name': param_name, 'param_val': p_val,
                            'wIoU': rob_metrics['wIoU'][i][j], 'Accuracy': rob_metrics['Accuracy'][i][j],
                            'SaliencyAUC': rob_metrics['SaliencyAUC'][i][j], 'SaliencySNR': rob_metrics['SaliencySNR'][i][j],
                            'pgd_success_rate': current_pgd_stats.get('pgd_success_rate', 0),
                            'pgd_mean_iters_to_flip': current_pgd_stats.get('pgd_mean_iters_to_flip', 0),
                        })
            except Exception as e:
                print(f"Could not process file {f_path}: {e}")

        df = pd.DataFrame(all_data)
        # Ensure we have data before proceeding
        if df.empty:
            print("Master dataframe is empty, cannot generate plots.")
            sys.exit(1)

        master_csv_path = os.path.join(plots_dir, 'full_results_long_format.csv')
        df.to_csv(master_csv_path, index=False)
        print(f"Saved master data table to {master_csv_path}")

        def get_model_type(row):
            if row['param_val'] == 0: return 'Standard'
            
            # Use the mode inferred from the file path
            if row['mode'] == 'iterative_hotflip':
                return 'Iterative HotFlip (Scheduled)' if row['scheduling_mode'] == 'scheduled' else 'Iterative HotFlip (No Schedule)'
            if row['mode'] == 'direct_hotflip':
                return 'Direct HotFlip (Scheduled)' if row['scheduling_mode'] == 'scheduled' else 'Direct HotFlip (No Schedule)'
            
            # Fallback for old adv_vs_std runs if they exist
            if row['mode'] == 'adv_vs_std' and row['scheduling_mode'] == 'scheduled': return 'Adversarial (Scheduled)'
            if row['mode'] == 'adv_vs_std' and row['scheduling_mode'] == 'no_schedule': return 'Adversarial (No Schedule)'
            return 'Unknown'

        df['model_type'] = df.apply(get_model_type, axis=1)

        baseline_metrics = df[df['model_type'] == 'Standard'].groupby(
            ['gc_pos', 'conservation', 'seed']
        ).mean(numeric_only=True).reset_index()

        # Create a separate baseline for per-model metrics like Accuracy
        baseline_model_metrics = df[df['model_type'] == 'Standard'][['gc_pos', 'conservation', 'seed', 'Accuracy']].drop_duplicates()

        df_robust = df[df['model_type'] != 'Standard'].copy()
        
        # Merge sample-level metrics
        df_robust = pd.merge(
            df_robust,
            baseline_metrics[['gc_pos', 'conservation', 'seed', 'wIoU', 'SaliencyAUC', 'SaliencySNR']],
            on=['gc_pos', 'conservation', 'seed'],
            suffixes=('', '_base')
        )
        # Merge model-level accuracy
        df_robust = pd.merge(
            df_robust,
            baseline_model_metrics.rename(columns={'Accuracy': 'Accuracy_base'}),
            on=['gc_pos', 'conservation', 'seed']
        )

        metrics_to_plot = ['wIoU', 'Accuracy', 'SaliencyAUC', 'SaliencySNR']
        
        # Calculate Root Skill and Linear Skill for SaliencyAUC
        def calculate_skills(auc_series):
            p = auc_series
            s = 2 * p - 1
            root_skill = np.sign(s) * np.sqrt(np.abs(s))
            return s, root_skill

        skill_robust, root_skill_robust = calculate_skills(df_robust['SaliencyAUC'])
        skill_base, root_skill_base = calculate_skills(df_robust['SaliencyAUC_base'])
        
        # Normalize the delta by dividing by the maximum possible range (2)
        df_robust['delta_Skill'] = (skill_robust - skill_base) / 2.0

        for metric in metrics_to_plot:
            if metric == 'SaliencyAUC':
                # The primary metric for selection remains RootSkill
                df_robust[f'delta_{metric}'] = (root_skill_robust - root_skill_base) / 2.0
            else:
                df_robust[f'delta_{metric}'] = df_robust[metric] - df_robust[f'{metric}_base']

        # --- Generate Combined Boxplots for all strategies ---
        print("\n--- Generating Combined Boxplots ---")

        df_plot_box = df_robust[df_robust['model_type'].isin([
            'Iterative HotFlip (No Schedule)', 'Iterative HotFlip (Scheduled)',
            'Direct HotFlip (No Schedule)', 'Direct HotFlip (Scheduled)'
        ])]
        
        for metric in metrics_to_plot:
            print(f"  - Plotting combined boxplot for {metric}...")
            if df_plot_box.empty:
                print(f"    Skipping {metric}, no data.")
                continue

            g = sns.catplot(
                data=df_plot_box, x="param_val", y=f"delta_{metric}",
                hue="conservation", col="model_type", row="gc_pos",
                kind="box", height=3, aspect=1.2, palette='Blues_d',
                fliersize=0, linewidth=1.0, showfliers=False, sharey=False, sharex=False,
                margin_titles=True,
                col_order=['Iterative HotFlip (No Schedule)', 'Iterative HotFlip (Scheduled)', 'Direct HotFlip (No Schedule)', 'Direct HotFlip (Scheduled)']
            )
            
            if metric == 'SaliencyAUC':
                g.fig.suptitle(f"Change in Saliency Root Skill (Normalized ΔRootSkill) vs. Standard", y=1.05, fontsize=16)
            else:
                g.fig.suptitle(f"Improvement in {metric} vs. Standard, by Training Strategy", y=1.05, fontsize=16)

            for (row_idx, col_idx), ax in np.ndenumerate(g.axes):
                if row_idx >= len(g.row_names) or col_idx >= len(g.col_names): continue
                
                gc_val = g.row_names[row_idx]
                model_type = g.col_names[col_idx]
                
                title = f"GC={gc_val} | {model_type}"
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
                    if metric == 'SaliencyAUC':
                        ax.set_ylabel("Normalized ΔRootSkill")
                    else:
                        ax.set_ylabel(f"Improvement in {metric}")
                else:
                    ax.set_ylabel("")

            sns.move_legend(g, "upper center", bbox_to_anchor=(.5, 0.99), ncol=len(df_plot_box['conservation'].unique()), title="Conservation", frameon=False)
            g.tight_layout(rect=[0, 0, 1, 0.95])
            plot_path = os.path.join(plots_dir, f"combined_delta_{metric}_boxplot.pdf")
            g.savefig(plot_path, dpi=300, format='pdf')
            plt.close(g.fig)
            print(f"    Saved to {plot_path}")

        # --- Prepare data for absolute value plots ---
        # For each robust model type, we want to include the corresponding standard model
        # as the baseline (param_val = 0) in the same facet.
        print("\n--- Preparing data for absolute value plots ---")
        df_abs_plot_list = []
        df_standard_models = df[df['model_type'] == 'Standard'].copy()
        
        model_type_to_sched = {
            'Iterative HotFlip (No Schedule)': 'no_schedule',
            'Iterative HotFlip (Scheduled)': 'scheduled',
            'Direct HotFlip (No Schedule)': 'no_schedule',
            'Direct HotFlip (Scheduled)': 'scheduled',
        }

        for mtype, sched_mode in model_type_to_sched.items():
            df_robust_subset = df[df['model_type'] == mtype]
            df_std_subset = df_standard_models[df_standard_models['scheduling_mode'] == sched_mode].copy()
            df_std_subset['model_type'] = mtype
            df_combined = pd.concat([df_robust_subset, df_std_subset])
            df_abs_plot_list.append(df_combined)
            
        df_plot_abs = pd.concat(df_abs_plot_list, ignore_index=True)


        # --- Generate Combined Boxplots for Absolute Values ---
        print("\n--- Generating Combined Absolute Value Boxplots ---")
        for metric in metrics_to_plot:
            print(f"  - Plotting combined boxplot for absolute {metric}...")
            if df_plot_abs.empty:
                print(f"    Skipping {metric}, no data.")
                continue

            g = sns.catplot(
                data=df_plot_abs, x="param_val", y=metric,
                hue="conservation", col="model_type", row="gc_pos",
                kind="box", height=3, aspect=1.2, palette='Blues_d',
                fliersize=0, linewidth=1.0, showfliers=False, sharey=False, sharex=False,
                margin_titles=True,
                col_order=['Iterative HotFlip (No Schedule)', 'Iterative HotFlip (Scheduled)', 'Direct HotFlip (No Schedule)', 'Direct HotFlip (Scheduled)']
            )
            
            g.fig.suptitle(f"Absolute {metric} by Training Strategy", y=1.05, fontsize=16)

            for (row_idx, col_idx), ax in np.ndenumerate(g.axes):
                if row_idx >= len(g.row_names) or col_idx >= len(g.col_names): continue
                
                gc_val = g.row_names[row_idx]
                model_type = g.col_names[col_idx]
                
                title = f"GC={gc_val} | {model_type}"
                ax.set_title(title, fontsize=9)
                # Removed ax.axhline(0, ls='--', color='gray', zorder=0)

                # Set custom x-axis labels
                if 'Adversarial' in model_type:
                    ax.set_xlabel("Epsilon")
                else:
                    ax.set_xlabel("Target Epsilon")
                ax.tick_params(axis='x', rotation=45, labelsize=8)

                # Unify y-axis labels
                if col_idx == 0:
                    ax.set_ylabel(f"Absolute {metric}")
                else:
                    ax.set_ylabel("")

            sns.move_legend(g, "upper center", bbox_to_anchor=(.5, 0.99), ncol=len(df_plot_abs['conservation'].unique()), title="Conservation", frameon=False)
            g.tight_layout(rect=[0, 0, 1, 0.95])
            plot_path = os.path.join(plots_dir, f"combined_absolute_{metric}_boxplot.pdf")
            g.savefig(plot_path, dpi=300, format='pdf')
            plt.close(g.fig)
            print(f"    Saved to {plot_path}")
            
        # --- Generate Combined Summary Bar Chart ---
        print("\n--- Generating Combined Summary Bar Chart ---")
        
        # First find the best epsilon for each combination based on mean ΔRootSkill
        if not df_robust.empty:
            # Group by all factors except seed to get mean delta_SaliencyAUC (ΔRootSkill) per epsilon
            best_eps_stats = df_robust.groupby(['model_type', 'gc_pos', 'conservation', 'param_val']).agg({
                'delta_SaliencyAUC': 'mean'
            }).reset_index()
            
            # Find the param_val that maximizes mean ΔRootSkill for each combination
            idx_best = best_eps_stats.groupby(['model_type', 'gc_pos', 'conservation'])['delta_SaliencyAUC'].idxmax()
            best_eps_per_combo = best_eps_stats.loc[idx_best][['model_type', 'gc_pos', 'conservation', 'param_val']]
            
            # Now filter df_robust to only include the best epsilon for each combination
            df_robust_best = pd.merge(
                df_robust,
                best_eps_per_combo,
                on=['model_type', 'gc_pos', 'conservation', 'param_val']
            )
        else:
            df_robust_best = df_robust
        
        id_vars = ['scheduling_mode', 'mode', 'model_type', 'gc_pos', 'conservation', 'seed']
        value_vars = [f'delta_{m}' for m in metrics_to_plot] + ['delta_Skill']
        df_deltas = pd.melt(df_robust_best, id_vars=id_vars, value_vars=value_vars, var_name='metric', value_name='delta_value')
        df_deltas['metric'] = df_deltas['metric'].str.replace('delta_', '')
        # Map SaliencyAUC to its display name for the plot
        df_deltas['metric'] = df_deltas['metric'].replace({'SaliencyAUC': 'NormΔRootSkill', 'Skill': 'NormΔSkill'})


        df_plot_bar = df_deltas[df_deltas['model_type'].isin([
            'Iterative HotFlip (No Schedule)', 'Iterative HotFlip (Scheduled)',
            'Direct HotFlip (No Schedule)', 'Direct HotFlip (Scheduled)'
        ])]
        
        if not df_plot_bar.empty:
            print("  - Plotting combined summary bar chart...")
            g = sns.catplot(
                data=df_plot_bar, kind='bar', x='gc_pos', y='delta_value',
                hue="conservation", col="model_type", row="metric",
                palette='Blues_d', height=3, aspect=1.4, sharey=False, margin_titles=True,
                col_order=['Iterative HotFlip (No Schedule)', 'Iterative HotFlip (Scheduled)', 'Direct HotFlip (No Schedule)', 'Direct HotFlip (Scheduled)'],
                row_order=['Accuracy', 'NormΔSkill', 'NormΔRootSkill', 'wIoU', 'SaliencySNR']
            )
            g.set_axis_labels("GC Content (Confounder Strength)", "") # Set X label, but leave Y blank for custom labels
            g.set_titles(col_template="{col_name}", row_template="{row_name}")
            g.fig.suptitle("Comparison of Training Strategies: Mean Improvement vs. Standard Model\n(Best ε selected by ΔRootSkill; error bars show SE across 10 seeds)", y=1.03, fontsize=14)
            
            # Set a specific Y-axis label for each row
            for i, ax_row in enumerate(g.axes):
                metric_name = g.row_names[i]
                if 'NormΔ' in metric_name:
                    ax_row[0].set_ylabel(f"Mean {metric_name}")
                else:
                    ax_row[0].set_ylabel(f"Mean Δ in {metric_name}")

            for ax in g.axes.flat: 
                ax.axhline(0, ls='--', color='gray', zorder=0)
                ax.tick_params(axis='x', rotation=25)
            
            sns.move_legend(g, "upper center", bbox_to_anchor=(.5, 0.98), ncol=len(df_plot_bar['conservation'].unique()), title="Conservation", frameon=False)
            g.tight_layout(rect=[0, 0, 1, 0.95])
            plot_path = os.path.join(plots_dir, f"summary_bar_combined.pdf")
            g.savefig(plot_path, dpi=300, format='pdf')
            plt.close(g.fig)
            print(f"    Saved to {plot_path}")
        else:
            print("  - Skipping summary bar chart: No data found for comparison.")

        # --- Select Best Epsilon and Generate 2x2 Heatmap Grid ---
        print("\n--- Generating 2x2 Heatmap Grid ---")
        
        # First, we need to find the best epsilon for each combination based on SaliencyAUC
        if not df_robust.empty:
            # Group by all factors except param_val and seed, find best param_val by mean SaliencyAUC
            best_eps_df = df_robust.groupby(['model_type', 'gc_pos', 'conservation', 'param_val']).agg({
                'SaliencyAUC': 'mean'
            }).reset_index()
            
            # Find the param_val that maximizes SaliencyAUC for each combination
            idx_best = best_eps_df.groupby(['model_type', 'gc_pos', 'conservation'])['SaliencyAUC'].idxmax()
            best_eps_df = best_eps_df.loc[idx_best]
            
            # Now get all the data for these best epsilons
            df_best_eps = pd.merge(
                df_robust,
                best_eps_df[['model_type', 'gc_pos', 'conservation', 'param_val']],
                on=['model_type', 'gc_pos', 'conservation', 'param_val']
            )
            
            # Calculate mean improvements for the best epsilon across seeds
            df_heatmap = df_best_eps.groupby(['model_type', 'gc_pos', 'conservation']).agg({
                'delta_SaliencyAUC': 'mean'
            }).reset_index()
            
            # Create 2x2 grid: one heatmap for each model type
            model_types_for_heatmap = [
                'Iterative HotFlip (No Schedule)', 'Iterative HotFlip (Scheduled)',
                'Direct HotFlip (No Schedule)', 'Direct HotFlip (Scheduled)'
            ]
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('Change in Saliency Root Skill (Normalized ΔRootSkill) by Signal and Confounder Strength\n(Best ε selected per combination)', fontsize=14)
            
            for idx, model_type in enumerate(model_types_for_heatmap):
                row = idx // 2
                col = idx % 2
                ax = axes[row, col]
                
                # Filter data for this model type
                df_model = df_heatmap[df_heatmap['model_type'] == model_type]
                
                if not df_model.empty:
                    # Pivot for heatmap
                    pivot_df = df_model.pivot(index='conservation', columns='gc_pos', values='delta_SaliencyAUC')
                    
                    # Create heatmap
                    sns.heatmap(pivot_df, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                               cbar_kws={'label': 'Normalized ΔRootSkill'},
                               ax=ax, vmin=-1, vmax=1)
                    
                    ax.set_title(model_type)
                    ax.set_xlabel('GC Content (Confounder Strength)')
                    ax.set_ylabel('Conservation (Signal Strength)')
                    ax.invert_yaxis()  # Higher conservation at top
                else:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(model_type)
            
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plot_path = os.path.join(plots_dir, "heatmap_grid_saliency_auc_improvement.pdf")
            plt.savefig(plot_path, dpi=300, format='pdf')
            plt.close(fig)
            print(f"  Saved heatmap grid to {plot_path}")
            
            # Also create a combined heatmap showing the best performing method for each gc/conservation combination
            print("\n--- Generating Best Method Heatmap ---")
            
            # Find which method performs best for each gc_pos/conservation combination
            df_best_method = df_heatmap.loc[df_heatmap.groupby(['gc_pos', 'conservation'])['delta_SaliencyAUC'].idxmax()]
            
            # Create a mapping for shorter names and a list for ordering
            method_map = {
                'Iterative HotFlip (No Schedule)': 'Iter-NoSched',
                'Iterative HotFlip (Scheduled)': 'Iter-Sched',
                'Direct HotFlip (No Schedule)': 'Direct-NoSched',
                'Direct HotFlip (Scheduled)': 'Direct-Sched'
            }
            method_order = list(method_map.values())
            
            df_best_method['method_short'] = df_best_method['model_type'].map(method_map)
            
            # Create pivot for the improvement values (to be used as text annotation)
            pivot_values = df_best_method.pivot(index='conservation', columns='gc_pos', values='delta_SaliencyAUC')
            
            # Create a pivot for the method names and map them to integers for coloring
            pivot_methods_categorical = df_best_method.pivot(index='conservation', columns='gc_pos', values='method_short')
            pivot_methods_int = pivot_methods_categorical.applymap(lambda x: method_order.index(x) if pd.notna(x) else -1)
            
            # Create a discrete colormap
            cmap = plt.get_cmap('tab10', len(method_order))
            
            fig, ax = plt.subplots(figsize=(10, 7))
            sns.heatmap(pivot_methods_int,
                        annot=pivot_values,
                        fmt='.2f',
                        cmap=cmap,
                        ax=ax,
                        linewidths=.5,
                        cbar=False,  # Disable the default colorbar, we create a manual one
                        annot_kws={'fontsize': 9})
            
            # Manually create a colorbar that acts as a discrete legend
            norm = plt.cm.colors.BoundaryNorm(np.arange(len(method_order) + 1) - 0.5, cmap.N)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([]) # You must set the array for the scalar mappable
            cbar = fig.colorbar(sm, ax=ax, ticks=np.arange(len(method_order)))
            cbar.set_ticklabels(method_order)
            cbar.set_label('Best Performing Method', rotation=270, labelpad=20)
            
            ax.set_title('Best Performing Method and Improvement (Normalized ΔRootSkill)\nby Signal and Confounder Strength', fontsize=14)
            ax.set_xlabel('GC Content (Confounder Strength)')
            ax.set_ylabel('Conservation (Signal Strength)')
            ax.invert_yaxis()
            
            plt.tight_layout()
            plot_path = os.path.join(plots_dir, "best_method_heatmap.pdf")
            plt.savefig(plot_path, dpi=300, format='pdf')
            plt.close(fig)
            print(f"  Saved best method heatmap to {plot_path}")

            # --- Generate Linear Skill Heatmap Grid ---
            print("\n--- Generating Linear Skill (ΔSkill) Heatmap Grid ---")
            
            # Calculate mean improvements for the best epsilon across seeds
            df_heatmap_skill = df_best_eps.groupby(['model_type', 'gc_pos', 'conservation']).agg({
                'delta_Skill': 'mean'
            }).reset_index()
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('Change in Linear Saliency Skill (Normalized ΔSkill) by Signal and Confounder Strength\n(Models selected by best ΔRootSkill)', fontsize=14)
            
            for idx, model_type in enumerate(model_types_for_heatmap):
                row = idx // 2
                col = idx % 2
                ax = axes[row, col]
                
                df_model = df_heatmap_skill[df_heatmap_skill['model_type'] == model_type]
                
                if not df_model.empty:
                    pivot_df = df_model.pivot(index='conservation', columns='gc_pos', values='delta_Skill')
                    sns.heatmap(pivot_df, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                               cbar_kws={'label': 'Normalized ΔSkill (Linear)'},
                               ax=ax, vmin=-1, vmax=1)
                    ax.set_title(model_type)
                    ax.set_xlabel('GC Content (Confounder Strength)')
                    ax.set_ylabel('Conservation (Signal Strength)')
                    ax.invert_yaxis()
                else:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(model_type)
            
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plot_path = os.path.join(plots_dir, "heatmap_grid_linear_skill.pdf")
            plt.savefig(plot_path, dpi=300, format='pdf')
            plt.close(fig)
            print(f"  Saved linear skill heatmap to {plot_path}")

        print("\nAll plotting complete.")
        sys.exit(0)

    if args.array_idx is not None:
        main_single_combo(args, args.array_idx)
    else:
        main(args) 