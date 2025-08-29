"""
Common utilities for sequence manipulation, GPU optimization, and other helper functions.
Used by both toy_slurm.py and merged_experiment.py experiments.
"""

import numpy as np
import torch
import torch.nn as nn
import random
import os
from typing import Tuple, Optional
import matplotlib.pyplot as plt
import pandas as pd
import math

try:
    import logomaker
except ImportError:
    print("Warning: logomaker is not installed. Motif visualization will be disabled.")
    logomaker = None


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

ALPH = np.array(list("ACGT"), dtype="U1")
to_ix = {b: i for i, b in enumerate(ALPH)}


# --------------------------------------------------------------------------- #
# Random Seed Management
# --------------------------------------------------------------------------- #

def set_seeds(seed_value: int = 42, deterministic: bool = False) -> None:
    """Set random seeds for reproducibility."""
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


# --------------------------------------------------------------------------- #
# GPU Utilities
# --------------------------------------------------------------------------- #

def log_gpu_stats(prefix=""):
    """Log current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"{prefix} GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")


class GPUPrefetchDataLoader:
    """GPU prefetching wrapper around a standard DataLoader for improved performance."""
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


# --------------------------------------------------------------------------- #
# Sequence Manipulation
# --------------------------------------------------------------------------- #

def sample_background(length: int, gc: float) -> np.ndarray:
    """Sample random DNA sequence with specified GC content."""
    p = np.array([(1 - gc) / 2, gc / 2, gc / 2, (1 - gc) / 2])
    return np.random.choice(ALPH, size=length, p=p)


def random_chunk(length: int) -> np.ndarray:
    """Generates a random DNA chunk of given length."""
    return np.random.choice(ALPH, size=length)


def one_hot(seq: np.ndarray) -> np.ndarray:
    """Convert sequence to one-hot encoding. (L,) char → (4,L) float32"""
    arr = np.zeros((4, len(seq)), dtype=np.float32)
    for i, b in enumerate(seq):
        arr[to_ix[b], i] = 1.0
    return arr


def one_hot_to_seq(one_hot_tensor: torch.Tensor) -> str:
    """Convert one-hot tensor back to sequence string. (4, L) → string"""
    indices = torch.argmax(one_hot_tensor, dim=0).cpu().numpy()
    return "".join(ALPH[indices])


def mutate(chunk: np.ndarray, conservation: float, gc_target: float) -> np.ndarray:
    """
    Return a new chunk with given conservation level, with mutations
    sampled to match the target GC content.
    """
    mutated_chunk = chunk.copy()
    n_to_mutate = np.random.binomial(len(chunk), 1.0 - conservation)
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


def embed(seq: np.ndarray, chunk: np.ndarray) -> Tuple[np.ndarray, int]:
    """Embed a chunk at a random position within a sequence."""
    start = np.random.randint(0, len(seq) - len(chunk) + 1)
    seq[start:start + len(chunk)] = chunk
    return seq, start


# --------------------------------------------------------------------------- #
# Visualization Utilities
# --------------------------------------------------------------------------- #

def log_conv1_motifs(
    model: nn.Module,
    writer,  # SummaryWriter
    epoch: int,
    output_dir: str = None,
    save_to_disk: bool = False,
    log_to_tb: bool = True,
):
    """
    Generates sequence logos from the first conv layer's weights.
    Requires logomaker to be installed.
    """
    if logomaker is None:
        return
        
    # Ensure model is on CPU for weight processing
    if hasattr(model, '_orig_mod'):
        weights = model._orig_mod.conv1.weight.detach().cpu()
    else:
        weights = model.conv1.weight.detach().cpu()

    pwms = torch.nn.functional.softmax(weights, dim=1)
    
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
# Randomized Smoothing Utilities (from toy_slurm.py)
# --------------------------------------------------------------------------- #

def concentration_from_epsilon(epsilon: float) -> float:
    """
    Convert epsilon (expected fraction of changed bases) to concentration for randomized smoothing.
    Higher concentration means less noise (fewer changes).
    """
    n_symbols = 4
    
    if epsilon == 0:
        return float('inf')
    elif epsilon >= (n_symbols - 1) / n_symbols:
        return 0.0
    else:
        # Binary search for concentration
        low, high = 0.01, 1000.0
        for _ in range(50):
            mid = (low + high) / 2.0
            expected_eps = (n_symbols - 1) / n_symbols * (1 - np.exp(mid) / (np.exp(mid) + n_symbols - 1))
            if expected_eps < epsilon:
                high = mid
            else:
                low = mid
        return mid 