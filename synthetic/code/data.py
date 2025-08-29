"""
Data generation module for synthetic sequence experiments.
Contains both vanilla (toy_slurm.py) and complex (merged_experiment.py) data generation processes.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import random
import os
import string
from typing import Tuple, Dict, Optional
from .utils import ALPH, to_ix, sample_background, one_hot, mutate, embed


# --------------------------------------------------------------------------- #
# Constants for Vanilla Dataset (toy_slurm.py)
# --------------------------------------------------------------------------- #

VANILLA_SEQ_LEN = 1000
VANILLA_CHUNK_MIN = 50
VANILLA_CHUNK_MAX = 120
VANILLA_N_TOTAL = 50000
VANILLA_DATASET_CACHE_DIR = "dataset_cache"


# --------------------------------------------------------------------------- #
# Constants for Complex Dataset (merged_experiment.py)
# --------------------------------------------------------------------------- #

COMPLEX_SEQ_LEN = 5000
COMPLEX_BLOCK_LEN_MEAN = 55
COMPLEX_BLOCK_LEN_SD = 15
COMPLEX_BLOCK_LEN_MIN = 40
COMPLEX_BLOCK_LEN_MAX = 70
COMPLEX_PROMOTER_HEX_1 = "TTGACA"
COMPLEX_PROMOTER_HEX_2 = "TATAAT"
COMPLEX_PROMOTER_SPACER_MIN = 20
COMPLEX_PROMOTER_SPACER_MAX = 30
COMPLEX_MIN_GAP_BETWEEN_BLOCKS = 30
COMPLEX_N_REPERTOIRES = 3
COMPLEX_GENES_PER_REPERTOIRE = 10
COMPLEX_DEFAULT_MOTIF_REPERTOIRE = COMPLEX_N_REPERTOIRES * COMPLEX_GENES_PER_REPERTOIRE
COMPLEX_N_ANCESTORS = COMPLEX_DEFAULT_MOTIF_REPERTOIRE
COMPLEX_N_TOTAL = 20000
COMPLEX_TARGET_SIGNAL_FRAC = 0.20
COMPLEX_DATASET_CACHE_DIR = "dataset_cache_merged_v4"


# --------------------------------------------------------------------------- #
# Dataset Caching Utilities
# --------------------------------------------------------------------------- #

def _vanilla_dataset_cache_path(gc_pos: float, conservation: float) -> str:
    """Return cache file path for a vanilla dataset with given parameters."""
    return os.path.join(
        VANILLA_DATASET_CACHE_DIR,
        f"synthetic_gcpos_{gc_pos:.2f}_cons_{conservation:.2f}_n_{VANILLA_N_TOTAL}.npz",
    )


def _complex_dataset_cache_path(gc_gap: float, conservation: float) -> str:
    """Return cache file path for a complex dataset with given parameters."""
    return os.path.join(
        COMPLEX_DATASET_CACHE_DIR,
        f"v7_20k_5kb_2kboperon_boostpos_gc-gap_{gc_gap:.3f}_cons_{conservation:.3f}.npz",
    )


# --------------------------------------------------------------------------- #
# Vanilla Dataset (from toy_slurm.py)
# --------------------------------------------------------------------------- #

class VanillaSeqDS(Dataset):
    def __init__(self, xs, ys, ms):
        self.x, self.y, self.m = xs, ys, ms

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.m[idx]


def mutate_vanilla(chunk: np.ndarray, conservation: float, gc_target: float) -> np.ndarray:
    """Vanilla mutation function for toy_slurm experiments."""
    mutated_chunk = chunk.copy()
    n_to_mutate = int(len(chunk) * (1.0 - conservation))
    pos_to_mutate = np.random.choice(len(chunk), n_to_mutate, replace=False)
    
    p = np.array([(1 - gc_target) / 2, gc_target / 2, gc_target / 2, (1 - gc_target) / 2])
    
    for pos in pos_to_mutate:
        original_base = mutated_chunk[pos]
        temp_p = p.copy()
        temp_p[to_ix[original_base]] = 0
        if temp_p.sum() == 0:
            mutated_chunk[pos] = np.random.choice(np.setdiff1d(ALPH, [original_base]))
        else:
            temp_p /= temp_p.sum()
            mutated_chunk[pos] = np.random.choice(ALPH, p=temp_p)

    return mutated_chunk


def embed_vanilla(seq: np.ndarray, chunk: np.ndarray) -> Tuple[np.ndarray, int]:
    """Embed a chunk at a random position within a sequence."""
    start = np.random.randint(0, len(seq) - len(chunk) + 1)
    seq[start:start + len(chunk)] = chunk
    return seq, start


def generate_vanilla_dataset(gc_pos: float, conservation: float):
    """
    Generate vanilla dataset as used in toy_slurm.py experiments.
    
    Parameters:
    - gc_pos: Target GC content for sequences and motifs
    - conservation: Conservation level for motifs
    
    Returns:
    - VanillaSeqDS: Dataset object
    - dict: Dataset summary statistics
    """
    pos_n = VANILLA_N_TOTAL // 2
    neg_n = VANILLA_N_TOTAL - pos_n

    # Create ancestor chunk with specified GC content
    chunk_len = np.random.randint(VANILLA_CHUNK_MIN, VANILLA_CHUNK_MAX + 1)
    ancestor = sample_background(chunk_len, gc=gc_pos)

    X, y, masks = [], [], []

    # Generate positive samples
    for _ in range(pos_n):
        seq = sample_background(VANILLA_SEQ_LEN, gc=gc_pos)
        chunk = mutate_vanilla(ancestor, conservation, gc_target=gc_pos)
        seq, start = embed_vanilla(seq, chunk)
        
        mask = np.zeros(VANILLA_SEQ_LEN, dtype=bool)
        mask[start:start + len(chunk)] = True
        
        X.append(one_hot(seq))
        y.append(1)
        masks.append(mask)

    # Generate negative samples
    for _ in range(neg_n):
        seq = sample_background(VANILLA_SEQ_LEN, gc=gc_pos)
        mask = np.zeros(VANILLA_SEQ_LEN, dtype=bool)
        
        X.append(one_hot(seq))
        y.append(0)
        masks.append(mask)

    X = torch.from_numpy(np.stack(X)).float()
    y = torch.from_numpy(np.array(y)).float()
    masks = np.stack(masks)

    summary = {
        "n_sequences": len(X),
        "n_positive": pos_n,
        "n_negative": neg_n,
        "ancestor_length": chunk_len,
        "seq_length": VANILLA_SEQ_LEN,
        "gc_pos": gc_pos,
        "conservation": conservation,
    }

    return VanillaSeqDS(X, y, masks), summary


# --------------------------------------------------------------------------- #
# Complex Dataset (from merged_experiment.py)
# --------------------------------------------------------------------------- #

class ComplexSeqDS(Dataset):
    def __init__(self, xs, ys, ms, sample_types=None):
        self.x, self.y, self.m = xs, ys, ms
        self.sample_types = sample_types  # Track sample type for stratification
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.m[idx]


def mutate_complex(chunk: np.ndarray, conservation: float, gc_target: float) -> np.ndarray:
    """Complex mutation function with binomial sampling for merged experiments."""
    return mutate(chunk, conservation, gc_target)  # Use the common mutate function


def _trunc_norm_complex(mu, sd, lo, hi, size=None):
    """Generate truncated normal values within bounds."""
    while True:
        v = np.random.normal(mu, sd, size)
        if np.all((v >= lo) & (v <= hi)):
            return v.astype(int)


def _nonoverlap_positions_complex(seq_len, lens):
    """Find non-overlapping positions for blocks with minimum gap."""
    tries = 0
    while tries < 1000:
        starts = sorted(np.random.randint(0, seq_len - sum(lens) - (len(lens) - 1) * COMPLEX_MIN_GAP_BETWEEN_BLOCKS + 1, size=len(lens)))
        offsets = np.array([0] + [l + COMPLEX_MIN_GAP_BETWEEN_BLOCKS for l in lens[:-1]])
        starts = np.array(starts) + np.cumsum(offsets)
        if starts[-1] + lens[-1] <= seq_len:
            if all(starts[i] + lens[i] + COMPLEX_MIN_GAP_BETWEEN_BLOCKS <= starts[i+1] for i in range(len(starts)-1)):
                return starts.tolist()
        tries += 1
    raise RuntimeError("Could not place blocks without overlap.")


def build_promoter_complex(gc):
    """Build promoter with variable spacer and mutated hexamers."""
    # Variable spacer length
    spacer_len = np.random.randint(COMPLEX_PROMOTER_SPACER_MIN, COMPLEX_PROMOTER_SPACER_MAX + 1)
    spacer = sample_background(spacer_len, gc)
    
    # Mutate hexamers slightly (1-2 mutations each)
    hex1 = np.array(list(COMPLEX_PROMOTER_HEX_1), dtype="U1")
    hex2 = np.array(list(COMPLEX_PROMOTER_HEX_2), dtype="U1")
    
    # Mutate 1-2 positions in each hexamer
    for hex_seq in [hex1, hex2]:
        n_mutations = np.random.randint(1, 3)  # 1 or 2 mutations
        positions = np.random.choice(len(hex_seq), n_mutations, replace=False)
        for pos in positions:
            old_base = hex_seq[pos]
            # Sample new base according to target GC
            p = np.array([(1 - gc) / 2, gc / 2, gc / 2, (1 - gc) / 2])
            p[to_ix[old_base]] = 0
            if p.sum() > 0:
                p /= p.sum()
                hex_seq[pos] = np.random.choice(ALPH, p=p)
    
    return np.concatenate([hex1, spacer, hex2])


def generate_complex_dataset(gc_gap: float, conservation: float,
                             target_signal_frac: float = COMPLEX_TARGET_SIGNAL_FRAC,
                             motif_repertoire: int = COMPLEX_DEFAULT_MOTIF_REPERTOIRE,
                             include_partial_negatives: bool = True) -> Tuple[ComplexSeqDS, dict]:
    """
    Generate complex dataset with multi-block design as used in merged_experiment.py.
    
    Features:
    - 3 repertoires with 10 genes each
    - Positive examples: exactly 3 motifs (one from each repertoire) + promoter
    - Negative decoys: 1-2 motifs from 1-2 repertoires (never all 3)
    
    Parameters:
    - gc_gap: Difference from baseline GC content (0.5)
    - conservation: Conservation level for positive motifs
    - target_signal_frac: Target fraction of sequence that is signal
    - motif_repertoire: Total number of motifs across all repertoires
    - include_partial_negatives: Whether to include decoy negatives
    
    Returns:
    - ComplexSeqDS: Dataset object
    - dict: Dataset summary statistics
    """
    
    # Sample budget
    POS_N = COMPLEX_N_TOTAL // 2
    NEG_N = COMPLEX_N_TOTAL - POS_N

    # Updated distribution: 30% motif decoys, 10% promoter-only, 60% pure background
    decoy_neg_n = int(0.3 * NEG_N) if include_partial_negatives else 0
    # 1/2 of decoys will have promoters
    decoy_with_promoter_n = decoy_neg_n // 2
    decoy_without_promoter_n = decoy_neg_n - decoy_with_promoter_n
    
    promoter_only_neg_n = int(0.1 * NEG_N)  # Reduced to 10%
    background_only_neg_n = NEG_N - decoy_neg_n - promoter_only_neg_n  # Should be 60%

    # Build 3 ancestral pools (repertoires), each with 10 genes
    ancestral_pools = []
    for rep_idx in range(COMPLEX_N_REPERTOIRES):
        repertoire = []
        for _ in range(COMPLEX_GENES_PER_REPERTOIRE):
            ancestor_gc = np.clip(np.random.normal(0.50 + gc_gap, 0.02), 0.25, 0.75)
            ancestor_seq = sample_background(COMPLEX_BLOCK_LEN_MAX, gc=ancestor_gc)
            repertoire.append(ancestor_seq)
        ancestral_pools.append(repertoire)

    X, y, masks = [], [], []
    sample_types = []  # Track sample type: 0=positive, 1=decoy_neg, 2=promoter_only_neg, 3=background_only_neg

    # --- Positive examples (must have exactly 3 blocks, one from each repertoire + promoter) ---
    n_pos_generated = 0
    realised_fracs_pos = []
    MAX_OPERON_LENGTH = int(COMPLEX_SEQ_LEN * target_signal_frac)  # Constrain total operon length
    
    while n_pos_generated < POS_N:
        current_gc_pos = np.clip(np.random.normal(0.50 + gc_gap, 0.04), 0.25, 0.75)
        bg = sample_background(COMPLEX_SEQ_LEN, gc=current_gc_pos)

        # Exactly 3 blocks (one from each repertoire)
        n_blocks = 3
        blk_lens = _trunc_norm_complex(COMPLEX_BLOCK_LEN_MEAN, COMPLEX_BLOCK_LEN_SD, 
                                      COMPLEX_BLOCK_LEN_MIN, COMPLEX_BLOCK_LEN_MAX, n_blocks)
        
        # Constraint: total operon length including gaps
        prom_seq = build_promoter_complex(current_gc_pos)
        promoter_full_len = len(prom_seq)
        total_motif_len = sum(blk_lens)
        total_gaps = (n_blocks + 1) * COMPLEX_MIN_GAP_BETWEEN_BLOCKS  # gaps between promoter-motifs and between motifs
        operon_length = promoter_full_len + total_motif_len + total_gaps
        
        if operon_length > MAX_OPERON_LENGTH:
            continue  # Skip if operon would be too long
            
        try:
            # Place motifs closer together to fit within MAX_OPERON_LENGTH
            # Start placing after where promoter will go
            operon_start = np.random.randint(promoter_full_len + COMPLEX_MIN_GAP_BETWEEN_BLOCKS, 
                                           COMPLEX_SEQ_LEN - (total_motif_len + (n_blocks-1)*COMPLEX_MIN_GAP_BETWEEN_BLOCKS))
            
            # Manually calculate positions to ensure they fit within operon length constraint
            blk_starts = []
            current_pos = operon_start
            for i, blen in enumerate(blk_lens):
                blk_starts.append(current_pos)
                current_pos += blen + COMPLEX_MIN_GAP_BETWEEN_BLOCKS
                
            # Verify operon fits
            last_motif_end = blk_starts[-1] + blk_lens[-1]
            actual_operon_length = last_motif_end - (operon_start - promoter_full_len - COMPLEX_MIN_GAP_BETWEEN_BLOCKS)
            if actual_operon_length > MAX_OPERON_LENGTH:
                continue
                
        except:
            continue

        mask = np.zeros(COMPLEX_SEQ_LEN, dtype=bool)
        # Place one motif from each repertoire
        for i, (blen, start) in enumerate(zip(blk_lens, blk_starts)):
            # Select from repertoire i (0, 1, or 2)
            ancestor = random.choice(ancestral_pools[i])
            master = ancestor[:blen]
            chunk = mutate_complex(master, conservation, gc_target=current_gc_pos)
            bg[start:start+blen] = chunk
            mask[start:start+blen] = True

        # Place promoter before first motif
        first_start = min(blk_starts)
        prom_pos = first_start - promoter_full_len - COMPLEX_MIN_GAP_BETWEEN_BLOCKS
        if prom_pos < 0:
            continue

        bg[prom_pos: prom_pos + promoter_full_len] = prom_seq
        mask[prom_pos: prom_pos + promoter_full_len] = True

        X.append(one_hot(bg))
        y.append(1)
        masks.append(mask)
        sample_types.append(0)  # Positive sample
        realised_fracs_pos.append(mask.sum() / COMPLEX_SEQ_LEN)
        n_pos_generated += 1

    # --- Decoy negatives WITH promoter ---
    for _ in range(decoy_with_promoter_n):
        current_gc = np.clip(np.random.normal(0.50, 0.04), 0.25, 0.75)
        bg = sample_background(COMPLEX_SEQ_LEN, gc=current_gc)

        n_blocks = np.random.randint(1, 3)  # 1–2 blocks
        blk_lens = _trunc_norm_complex(COMPLEX_BLOCK_LEN_MEAN, COMPLEX_BLOCK_LEN_SD, 
                                      COMPLEX_BLOCK_LEN_MIN, COMPLEX_BLOCK_LEN_MAX, n_blocks)
        try:
            blk_starts = _nonoverlap_positions_complex(COMPLEX_SEQ_LEN, blk_lens)
        except RuntimeError:
            X.append(one_hot(sample_background(COMPLEX_SEQ_LEN, gc=current_gc)))
            y.append(0)
            masks.append(np.zeros(COMPLEX_SEQ_LEN, dtype=bool))
            sample_types.append(1)  # Decoy negative sample
            continue

        # Select 1-2 repertoires (but never all 3)
        n_repertoires_to_use = np.random.randint(1, min(n_blocks + 1, 3))  # 1 or 2 repertoires
        selected_repertoires = np.random.choice(COMPLEX_N_REPERTOIRES, n_repertoires_to_use, replace=False)
        
        # Set conservation for negative decoys to be low
        decoy_conservation = np.random.uniform(0.5, 0.6)
        
        mask = np.zeros(COMPLEX_SEQ_LEN, dtype=bool)
        for i, (blen, start) in enumerate(zip(blk_lens, blk_starts)):
            # Cycle through selected repertoires
            repertoire_idx = selected_repertoires[i % len(selected_repertoires)]
            ancestor = random.choice(ancestral_pools[repertoire_idx])
            master = ancestor[:blen]
            chunk = mutate_complex(master, decoy_conservation, gc_target=current_gc)
            bg[start:start+blen] = chunk
            mask[start:start+blen] = True
        
        # Add promoter (but don't include in mask since it's not discriminative)
        prom_seq = build_promoter_complex(current_gc)
        promoter_full_len = len(prom_seq)
        first_start = min(blk_starts)
        if first_start >= (promoter_full_len + COMPLEX_MIN_GAP_BETWEEN_BLOCKS):
            prom_pos = first_start - promoter_full_len - COMPLEX_MIN_GAP_BETWEEN_BLOCKS
            bg[prom_pos: prom_pos + promoter_full_len] = prom_seq
            # Note: NOT adding promoter to mask for negatives

        X.append(one_hot(bg))
        y.append(0)
        masks.append(mask)
        sample_types.append(1)  # Decoy negative sample

    # --- Decoy negatives WITHOUT promoter ---
    for _ in range(decoy_without_promoter_n):
        current_gc = np.clip(np.random.normal(0.50, 0.04), 0.25, 0.75)
        bg = sample_background(COMPLEX_SEQ_LEN, gc=current_gc)

        n_blocks = np.random.randint(1, 3)  # 1–2 blocks
        blk_lens = _trunc_norm_complex(COMPLEX_BLOCK_LEN_MEAN, COMPLEX_BLOCK_LEN_SD,
                                      COMPLEX_BLOCK_LEN_MIN, COMPLEX_BLOCK_LEN_MAX, n_blocks)
        try:
            blk_starts = _nonoverlap_positions_complex(COMPLEX_SEQ_LEN, blk_lens)
        except RuntimeError:
            X.append(one_hot(sample_background(COMPLEX_SEQ_LEN, gc=current_gc)))
            y.append(0)
            masks.append(np.zeros(COMPLEX_SEQ_LEN, dtype=bool))
            sample_types.append(1)  # Decoy negative sample
            continue

        # Select 1-2 repertoires (but never all 3)
        n_repertoires_to_use = np.random.randint(1, min(n_blocks + 1, 3))  # 1 or 2 repertoires
        selected_repertoires = np.random.choice(COMPLEX_N_REPERTOIRES, n_repertoires_to_use, replace=False)
        
        # Set conservation for negative decoys to be low
        decoy_conservation = np.random.uniform(0.5, 0.6)
        
        mask = np.zeros(COMPLEX_SEQ_LEN, dtype=bool)
        for i, (blen, start) in enumerate(zip(blk_lens, blk_starts)):
            # Cycle through selected repertoires
            repertoire_idx = selected_repertoires[i % len(selected_repertoires)]
            ancestor = random.choice(ancestral_pools[repertoire_idx])
            master = ancestor[:blen]
            chunk = mutate_complex(master, decoy_conservation, gc_target=current_gc)
            bg[start:start+blen] = chunk
            mask[start:start+blen] = True

        X.append(one_hot(bg))
        y.append(0)
        masks.append(mask)
        sample_types.append(1)  # Decoy negative sample

    # --- Promoter-only negatives ---
    for _ in range(promoter_only_neg_n):
        current_gc = np.clip(np.random.normal(0.50, 0.04), 0.25, 0.75)
        bg = sample_background(COMPLEX_SEQ_LEN, gc=current_gc)

        prom_seq = build_promoter_complex(current_gc)
        promoter_full_len = len(prom_seq)

        if COMPLEX_SEQ_LEN >= promoter_full_len:
            prom_pos = np.random.randint(0, COMPLEX_SEQ_LEN - promoter_full_len + 1)
            bg[prom_pos : prom_pos + promoter_full_len] = prom_seq

        X.append(one_hot(bg))
        y.append(0)
        masks.append(np.zeros(COMPLEX_SEQ_LEN, dtype=bool))
        sample_types.append(2)  # Promoter-only negative

    # --- Standard negatives (no blocks / no promoter) ---
    for _ in range(background_only_neg_n):
        bg_gc = np.clip(np.random.normal(0.50, 0.04), 0.25, 0.75)
        bg = sample_background(COMPLEX_SEQ_LEN, gc=bg_gc)
        X.append(one_hot(bg))
        y.append(0)
        masks.append(np.zeros(COMPLEX_SEQ_LEN, dtype=bool))
        sample_types.append(3)  # Background-only negative

    X = torch.from_numpy(np.stack(X)).float()
    y = torch.from_numpy(np.array(y)).float()
    masks = np.stack(masks)
    sample_types = np.array(sample_types)

    avg_realised_frac = np.mean(realised_fracs_pos) if realised_fracs_pos else 0.0

    summary = {
        "n_sequences": len(X),
        "n_positive": POS_N,
        "n_decoy_negative": decoy_neg_n,
        "n_decoy_with_promoter": decoy_with_promoter_n,
        "n_decoy_without_promoter": decoy_without_promoter_n,
        "n_promoter_only_negative": promoter_only_neg_n,
        "n_background_only_negative": background_only_neg_n,
        "n_repertoires": COMPLEX_N_REPERTOIRES,
        "genes_per_repertoire": COMPLEX_GENES_PER_REPERTOIRE,
        "total_motif_repertoire": motif_repertoire,
        "seed": np.random.get_state()[1][0].item() if len(np.random.get_state()[1]) > 0 else -1,
        "target_signal_frac": target_signal_frac,
        "avg_realised_frac": avg_realised_frac,
    }

    return ComplexSeqDS(X, y, masks, sample_types), summary


# --------------------------------------------------------------------------- #
# Dataset Loading with Caching
# --------------------------------------------------------------------------- #

def load_or_generate_vanilla_dataset(gc_pos: float, conservation: float):
    """Load vanilla dataset from cache if present, otherwise generate and cache it."""
    os.makedirs(VANILLA_DATASET_CACHE_DIR, exist_ok=True)
    
    cache_path = _vanilla_dataset_cache_path(gc_pos, conservation)
    if os.path.exists(cache_path):
        print(f"Loading cached vanilla dataset from {cache_path}")
        data = np.load(cache_path)
        X = torch.tensor(data["X"], dtype=torch.float32)
        y = torch.tensor(data["y"], dtype=torch.float)
        masks = data["masks"]
        return VanillaSeqDS(X, y, masks)
    
    print(f"Generating vanilla dataset with GC_pos={gc_pos:.2f} and conservation={conservation:.2f}...")
    ds, _ = generate_vanilla_dataset(gc_pos, conservation)
    
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


def load_or_generate_complex_dataset(gc_gap: float, conservation: float):
    """Load complex dataset from cache if present, otherwise generate and cache it."""
    os.makedirs(COMPLEX_DATASET_CACHE_DIR, exist_ok=True)

    cache_path = _complex_dataset_cache_path(gc_gap, conservation)
    if os.path.exists(cache_path):
        print(f"Loading cached complex dataset from {cache_path}")
        data = np.load(cache_path)
        X = torch.tensor(data["X"], dtype=torch.float32)
        y = torch.tensor(data["y"], dtype=torch.float)
        masks = data["masks"]
        sample_types = data.get("sample_types", None)  # Backward compatibility
        return ComplexSeqDS(X, y, masks, sample_types)

    print(f"Generating complex dataset with GC_gap={gc_gap:.3f} and conservation={conservation:.2f}...")

    ds, _ = generate_complex_dataset(
        gc_gap=gc_gap,
        conservation=conservation,
        target_signal_frac=COMPLEX_TARGET_SIGNAL_FRAC,
        motif_repertoire=COMPLEX_DEFAULT_MOTIF_REPERTOIRE,
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
            sample_types=ds.sample_types,
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
# Stratified Splitting
# --------------------------------------------------------------------------- #

def stratified_split(dataset, split_sizes, seed=42):
    """Perform stratified splitting of dataset based on sample types."""
    if hasattr(dataset, 'sample_types') and dataset.sample_types is not None:
        # Complex dataset with sample types
        sample_types = dataset.sample_types
        type_indices = {
            0: np.where(sample_types == 0)[0],  # Positive
            1: np.where(sample_types == 1)[0],  # Decoy negative
            2: np.where(sample_types == 2)[0],  # Promoter-only negative
            3: np.where(sample_types == 3)[0],  # Background-only negative
        }
        
        # Calculate split proportions
        total_size = len(dataset)
        train_ratio = split_sizes[0] / total_size
        val_ratio = split_sizes[1] / total_size
        
        rng = np.random.default_rng(seed)
        train_indices, val_indices, test_indices = [], [], []
        
        for sample_type, indices in type_indices.items():
            if len(indices) == 0:
                continue
                
            # Shuffle indices for this type
            indices = rng.permutation(indices)
            
            # Calculate split points
            n_train = int(len(indices) * train_ratio)
            n_val = int(len(indices) * val_ratio)
            
            # Split indices
            train_indices.extend(indices[:n_train])
            val_indices.extend(indices[n_train:n_train + n_val])
            test_indices.extend(indices[n_train + n_val:])
        
        # Create subset datasets
        from torch.utils.data import Subset
        train_ds = Subset(dataset, train_indices)
        val_ds = Subset(dataset, val_indices)
        test_ds = Subset(dataset, test_indices)
        
        return train_ds, val_ds, test_ds
    else:
        # Vanilla dataset or backward compatibility
        from torch.utils.data import random_split
        return random_split(dataset, split_sizes, generator=torch.Generator().manual_seed(seed)) 