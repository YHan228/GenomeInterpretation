#!/usr/bin/env python3
"""Rashomon Set Analysis using 1-layer CNN filter activations.

Fits many single-layer CNNs with different initializations and examines which
filter motifs consistently appear with high activation across models in the
ε-set, identifying "necessary sequence patterns" analogous to gene-level analysis.

Architecture:
    Conv1D(filters=K, kernel_size=W, activation) -> GlobalMaxPool -> Dense(1, sigmoid)

Usage:
    python sporulation/code/rashomon_cnn.py \
        --phenotype "Spore formation" \
        --n_models 100 \
        --epsilon 0.02 \
        --kernel_size 9 \
        --n_filters 64
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import tempfile

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
import joblib
from joblib import Parallel, delayed
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import balanced_accuracy_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Import utilities from phenotype code
_HERE = Path(__file__).resolve().parent
_CODE_DIR = str(_HERE)
_PHENOTYPE_CODE_DIR = str(_HERE.parent.parent / "phenotype" / "code")
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)
if _PHENOTYPE_CODE_DIR not in sys.path:
    sys.path.insert(0, _PHENOTYPE_CODE_DIR)

try:
    from training import one_hot_encode, parse_fasta
except ImportError:
    from sporulation.code.training import one_hot_encode, parse_fasta

try:
    from phenotype_utils import (
        build_labels_map_and_classes,
        DATA_ROOT,
        read_metadata_table,
        PHENOTYPE_COLUMNS,
    )
except ImportError:
    from phenotype.code.phenotype_utils import (
        build_labels_map_and_classes,
        DATA_ROOT,
        read_metadata_table,
        PHENOTYPE_COLUMNS,
    )

# Defaults
METADATA_XLSX = Path("sporulation/microbe.cards table S1.xlsx")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_SEQ_LEN = 100_000  # Shorter for faster iteration
DEFAULT_N_FILTERS = 48
DEFAULT_KERNEL_SIZE = 9
DEFAULT_EPSILON = 0.02

# Multi-architecture Rashomon: different kernel sizes capture different motif lengths
HP_CONFIGS = [
    {"kernel_size": 6, "n_filters": 48},   # Short: TATA, -10/-35 boxes
    {"kernel_size": 9, "n_filters": 48},   # Medium: typical TF binding
    {"kernel_size": 12, "n_filters": 48},  # Medium-long
    {"kernel_size": 15, "n_filters": 48},  # Longer structured motifs
    {"kernel_size": 21, "n_filters": 48},  # Long: riboswitches, etc.
]
SEEDS_PER_CONFIG = 40  # 40 seeds × 5 configs = 200 models per activation


# ---------------------------------------------------------------------------
# Fast Vectorized Encoding
# ---------------------------------------------------------------------------

def one_hot_encode_fast(seq: str) -> torch.Tensor:
    """Vectorized one-hot encoding - 100-1000x faster than loop-based version.

    Args:
        seq: DNA sequence string (A/C/G/T/N, case-insensitive)

    Returns:
        Tensor of shape (4, len(seq)) with one-hot encoding
        Non-ACGT characters (N, etc.) become all zeros.
    """
    seq_upper = seq.upper()
    seq_array = np.frombuffer(seq_upper.encode('ascii'), dtype=np.uint8)

    one_hot = np.zeros((4, len(seq)), dtype=np.float32)
    one_hot[0] = (seq_array == ord('A'))
    one_hot[1] = (seq_array == ord('C'))
    one_hot[2] = (seq_array == ord('G'))
    one_hot[3] = (seq_array == ord('T'))

    return torch.from_numpy(one_hot)


# ---------------------------------------------------------------------------
# Data Utilities
# ---------------------------------------------------------------------------

def _load_sequences_list(
    data_dir: Path,
    metadata_df: pd.DataFrame,
    seq_len: int,
    phenotype_col: str,
    file_col: str = "Fasta file",
    verbose: bool = True,
) -> List[Dict]:
    """Load sequences from FASTA files with labels."""
    train_dirs = [str(DATA_ROOT / "train")]
    labels_map, classes = build_labels_map_and_classes(
        metadata_df,
        phenotype_col=phenotype_col,
        file_col=file_col,
        train_dirs=train_dirs,
    )

    sequences = []
    exts = (".fasta", ".fa", ".fna")
    file_list = [f for f in os.listdir(data_dir) if f.endswith(exts)]

    if verbose:
        print(f"Loading sequences from {data_dir}...")

    for file_name in file_list:
        label = labels_map.get(file_name.strip().lower())
        if label is None:
            continue
        fp = data_dir / file_name
        seq = parse_fasta(str(fp))
        if not seq:
            continue
        if len(seq) < seq_len:
            seq = seq.ljust(seq_len, "N")
        sequences.append({"sequence": seq, "label": label, "filename": file_name})

    if verbose:
        print(f"Loaded {len(sequences)} sequences")
    return sequences


def precompute_windows(
    sequences: List[Dict],
    seq_len: int,
    n_windows: int,
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pre-compute one-hot encoded windows from sequences.

    Returns:
        X: (n_windows, 4, seq_len) tensor of one-hot encoded windows
        y: (n_windows,) tensor of labels
    """
    rng = np.random.RandomState(seed)

    if verbose:
        print(f"Pre-encoding {n_windows} windows of length {seq_len}...", flush=True)

    X_list = []
    y_list = []

    for i in tqdm(range(n_windows), desc="Encoding", disable=not verbose, miniters=1000):
        genome_idx = i % len(sequences)
        sample = sequences[genome_idx]
        seq_str = sample["sequence"]
        label = sample["label"]

        if len(seq_str) > seq_len:
            start = rng.randint(0, len(seq_str) - seq_len + 1)
            seq_str = seq_str[start : start + seq_len]
        elif len(seq_str) < seq_len:
            seq_str = seq_str.ljust(seq_len, "N")

        one_hot = one_hot_encode_fast(seq_str)  # Use vectorized encoding
        X_list.append(one_hot)
        y_list.append(label)

    X = torch.stack(X_list)
    y = torch.tensor(y_list, dtype=torch.long)

    if verbose:
        print(f"Encoded X: {X.shape}, y: {y.shape}", flush=True)

    return X, y


class PrecomputedDataset(Dataset):
    """Dataset with pre-computed one-hot encoded windows."""

    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class SequenceDataset(Dataset):
    """Dataset for DNA sequences with deterministic validation sampling."""

    def __init__(
        self,
        sequences: List[Dict],
        seq_len: int,
        epoch_budget: Optional[int] = None,
        is_validation: bool = False,
        seed: int = 42,
    ):
        self.sequences = sequences
        self.seq_len = seq_len
        self.is_validation = is_validation

        if epoch_budget is not None:
            self.epoch_len = int(epoch_budget)
        else:
            total_len = sum(len(s["sequence"]) for s in self.sequences)
            self.epoch_len = max(len(sequences), total_len // self.seq_len)

        if self.is_validation:
            self._generate_deterministic_samples(seed)

    def _generate_deterministic_samples(self, seed: int):
        rng = np.random.RandomState(seed)
        self.deterministic_samples = []
        for i in range(self.epoch_len):
            genome_idx = i % len(self.sequences)
            seq_str = self.sequences[genome_idx]["sequence"]
            start = 0
            if len(seq_str) > self.seq_len:
                start = rng.randint(0, len(seq_str) - self.seq_len + 1)
            self.deterministic_samples.append({"genome_idx": genome_idx, "start": start})

    def __len__(self):
        return self.epoch_len

    def __getitem__(self, idx):
        if self.is_validation:
            info = self.deterministic_samples[idx]
            genome_idx = info["genome_idx"]
            start = info["start"]
            sample = self.sequences[genome_idx]
            seq_str = sample["sequence"][start : start + self.seq_len]
            label = sample["label"]
        else:
            genome_idx = idx % len(self.sequences)
            sample = self.sequences[genome_idx]
            seq_str = sample["sequence"]
            label = sample["label"]
            if len(seq_str) > self.seq_len:
                start = np.random.randint(0, len(seq_str) - self.seq_len + 1)
                seq_str = seq_str[start : start + self.seq_len]

        one_hot = one_hot_encode(seq_str)
        return one_hot, torch.tensor(label, dtype=torch.long)


# ---------------------------------------------------------------------------
# Model Definition
# ---------------------------------------------------------------------------

class SingleLayerCNN(nn.Module):
    """Single-layer CNN for sequence classification.

    Architecture:
        Conv1D(4 -> n_filters, kernel_size) -> activation -> GlobalMaxPool -> FC(1)
    """

    def __init__(
        self,
        n_filters: int = 64,
        kernel_size: int = 9,
        activation: str = "relu",  # "relu" or "exp"
        num_classes: int = 2,
    ):
        super().__init__()
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.activation_name = activation
        self.num_classes = num_classes

        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(4, n_filters, kernel_size=kernel_size, padding=padding, bias=True)
        self.bn = nn.BatchNorm1d(n_filters)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(n_filters, num_classes)

    def _activation(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation_name == "exp":
            return torch.exp(x)
        return F.relu(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 4, L)
        x = self.conv(x)  # (B, n_filters, L)
        x = self.bn(x)
        x = self._activation(x)
        x = self.pool(x).squeeze(-1)  # (B, n_filters)
        x = self.fc(x)  # (B, num_classes)
        return x

    def get_filter_activations(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return filter activations and max positions.

        Returns:
            activations: (B, n_filters) max activation per filter
            positions: (B, n_filters) position of max activation
        """
        with torch.no_grad():
            conv_out = self.conv(x)  # (B, n_filters, L)
            conv_out = self.bn(conv_out)
            conv_out = self._activation(conv_out)
            max_vals, max_pos = conv_out.max(dim=2)  # (B, n_filters)
        return max_vals, max_pos

    def get_filter_weights(self) -> np.ndarray:
        """Extract filter weights as numpy array of shape (n_filters, kernel_size, 4)."""
        weights = self.conv.weight.detach().cpu().numpy()  # (n_filters, 4, kernel_size)
        return weights.transpose(0, 2, 1)  # (n_filters, kernel_size, 4)


# ---------------------------------------------------------------------------
# Rashomon Data Structures
# ---------------------------------------------------------------------------

@dataclass
class RashomonCNNModel:
    """A single CNN model in the Rashomon set."""
    seed: int
    activation: str
    performance: float
    filter_weights: np.ndarray  # (n_filters, kernel_size, 4)
    filter_importance: np.ndarray  # (n_filters,) mean activation magnitude


@dataclass
class RashomonCNNSet:
    """Collection of CNN models within epsilon of optimal."""
    activation: str
    epsilon: float
    best_performance: float
    threshold: float
    models: List[RashomonCNNModel] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.models)


# ---------------------------------------------------------------------------
# Training and Evaluation
# ---------------------------------------------------------------------------

def set_seeds(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def class_weights_from_sequences(sequences: List[Dict]) -> torch.Tensor:
    labels = [s["label"] for s in sequences]
    vc = pd.Series(labels).value_counts().sort_index()
    weights = (len(labels) / (len(vc) * vc)).values
    return torch.tensor(weights, dtype=torch.float32, device=DEVICE)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
) -> Tuple[float, float]:
    """Evaluate model and return (loss, balanced_accuracy)."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    preds_all, labels_all = [], []

    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        logits = model(xb)
        loss = loss_fn(logits, yb)
        total_loss += loss.item()
        n_batches += 1
        preds_all.extend(logits.argmax(dim=1).cpu().tolist())
        labels_all.extend(yb.cpu().tolist())

    avg_loss = total_loss / max(1, n_batches)
    bal_acc = balanced_accuracy_score(labels_all, preds_all) if labels_all else 0.0
    return avg_loss, bal_acc


@torch.no_grad()
def compute_filter_importance(
    model: nn.Module,
    loader: DataLoader,
) -> np.ndarray:
    """Compute mean max-activation per filter across dataset."""
    model.eval()
    all_activations = []

    for xb, _ in loader:
        xb = xb.to(DEVICE)
        activations, _ = model.get_filter_activations(xb)
        all_activations.append(activations.cpu().numpy())

    if not all_activations:
        return np.zeros(model.n_filters)

    all_activations = np.vstack(all_activations)
    return all_activations.mean(axis=0)


def fit_single_cnn(
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_filters: int,
    kernel_size: int,
    activation: str,
    seed: int,
    num_classes: int,
    class_weights: torch.Tensor,
    max_epochs: int = 15,
    lr: float = 1e-3,
    early_stop_patience: int = 5,
) -> RashomonCNNModel:
    """Fit a single 1-layer CNN and return Rashomon model object."""
    set_seeds(seed)

    # Model
    model = SingleLayerCNN(
        n_filters=n_filters,
        kernel_size=kernel_size,
        activation=activation,
        num_classes=num_classes,
    ).to(DEVICE)

    # Loss and optimizer
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)
    scaler = GradScaler()

    best_val_loss = float("inf")
    best_bal_acc = 0.0
    best_state = None
    es_counter = 0

    for epoch in range(max_epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                logits = model(xb)
                loss = loss_fn(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        val_loss, val_bal_acc = evaluate_model(model, val_loader, loss_fn)
        scheduler.step(val_loss)

        if val_bal_acc > best_bal_acc:
            best_bal_acc = val_bal_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            es_counter = 0
        else:
            es_counter += 1

        if es_counter >= early_stop_patience:
            break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(DEVICE)

    # Extract filter info
    filter_weights = model.get_filter_weights()
    filter_importance = compute_filter_importance(model, val_loader)

    return RashomonCNNModel(
        seed=seed,
        activation=activation,
        performance=best_bal_acc,
        filter_weights=filter_weights,
        filter_importance=filter_importance,
    )


# ---------------------------------------------------------------------------
# Filter Analysis (Improved)
# ---------------------------------------------------------------------------

def filter_to_pwm(weights: np.ndarray) -> np.ndarray:
    """Convert filter weights to PWM (position weight matrix).

    Args:
        weights: (kernel_size, 4) filter weights

    Returns:
        pwm: (kernel_size, 4) normalized PWM
    """
    # Softmax normalization to get probabilities
    exp_weights = np.exp(weights - weights.max(axis=1, keepdims=True))
    pwm = exp_weights / exp_weights.sum(axis=1, keepdims=True)
    return pwm


def pwm_to_ic(pwm: np.ndarray, pseudocount: float = 0.01) -> np.ndarray:
    """Convert PWM to information content per position.

    Args:
        pwm: (kernel_size, 4) PWM matrix
        pseudocount: small value to avoid log(0)

    Returns:
        ic: (kernel_size,) information content in bits (max 2 bits per position)
    """
    pwm_safe = np.clip(pwm, pseudocount, 1.0 - pseudocount)
    pwm_safe = pwm_safe / pwm_safe.sum(axis=1, keepdims=True)
    # IC = 2 + sum(p * log2(p)) for each position
    ic = 2.0 + np.sum(pwm_safe * np.log2(pwm_safe), axis=1)
    return np.clip(ic, 0, 2)


def pwm_reverse_complement(pwm: np.ndarray) -> np.ndarray:
    """Get reverse complement of PWM.

    Args:
        pwm: (kernel_size, 4) PWM with columns [A, C, G, T]

    Returns:
        rc_pwm: reverse complement PWM
    """
    # Reverse positions and swap A<->T, C<->G
    # Column order: A=0, C=1, G=2, T=3
    # RC: A->T (0->3), C->G (1->2), G->C (2->1), T->A (3->0)
    rc = pwm[::-1, [3, 2, 1, 0]]
    return rc


def pwm_to_consensus(pwm: np.ndarray, threshold: float = 0.5) -> str:
    """Convert PWM to consensus sequence.

    Args:
        pwm: (kernel_size, 4) PWM matrix
        threshold: minimum probability for uppercase letter

    Returns:
        consensus string with A/C/G/T (uppercase if prob > threshold, else lowercase)
    """
    bases = "ACGT"
    consensus = []
    for pos in range(pwm.shape[0]):
        idx = pwm[pos].argmax()
        prob = pwm[pos, idx]
        base = bases[idx]
        if prob < threshold:
            base = base.lower()
        consensus.append(base)
    return "".join(consensus)


def compute_pwm_similarity_ic_weighted(
    pwm1: np.ndarray,
    pwm2: np.ndarray,
    check_revcomp: bool = True,
) -> Tuple[float, bool]:
    """Compute IC-weighted Pearson correlation between PWMs.

    Args:
        pwm1, pwm2: (kernel_size, 4) PWM matrices
        check_revcomp: also check reverse complement and return best

    Returns:
        similarity: IC-weighted correlation (0 to 1)
        is_revcomp: True if best match is reverse complement
    """
    if pwm1.shape != pwm2.shape:
        return 0.0, False

    def _weighted_corr(p1: np.ndarray, p2: np.ndarray) -> float:
        # Compute IC weights (average of both PWMs)
        ic1 = pwm_to_ic(p1)
        ic2 = pwm_to_ic(p2)
        weights = (ic1 + ic2) / 2.0  # (kernel_size,)

        # Flatten and weight
        flat1 = p1.flatten()
        flat2 = p2.flatten()
        # Expand weights to match flattened shape (kernel_size * 4)
        weights_expanded = np.repeat(weights, 4)

        # Weighted Pearson correlation
        w_sum = weights_expanded.sum()
        if w_sum < 1e-8:
            return 0.0

        mean1 = np.average(flat1, weights=weights_expanded)
        mean2 = np.average(flat2, weights=weights_expanded)

        cov = np.sum(weights_expanded * (flat1 - mean1) * (flat2 - mean2))
        std1 = np.sqrt(np.sum(weights_expanded * (flat1 - mean1) ** 2))
        std2 = np.sqrt(np.sum(weights_expanded * (flat2 - mean2) ** 2))

        if std1 < 1e-8 or std2 < 1e-8:
            return 0.0

        return max(0.0, cov / (std1 * std2))

    sim_fwd = _weighted_corr(pwm1, pwm2)

    if check_revcomp:
        pwm2_rc = pwm_reverse_complement(pwm2)
        sim_rc = _weighted_corr(pwm1, pwm2_rc)
        if sim_rc > sim_fwd:
            return sim_rc, True

    return sim_fwd, False


def compute_pairwise_similarities_vectorized(
    pwms: np.ndarray,
    check_revcomp: bool = True,
) -> np.ndarray:
    """Compute pairwise similarities efficiently using vectorization.

    Args:
        pwms: (n_pwms, kernel_size, 4) array of PWMs
        check_revcomp: also check reverse complement

    Returns:
        sim_matrix: (n_pwms, n_pwms) similarity matrix
    """
    n = len(pwms)
    sim_matrix = np.eye(n)

    # Precompute ICs for all PWMs
    all_ics = np.array([pwm_to_ic(p) for p in pwms])  # (n, kernel_size)

    # Precompute reverse complements if needed
    if check_revcomp:
        pwms_rc = np.array([pwm_reverse_complement(p) for p in pwms])
        all_ics_rc = np.array([pwm_to_ic(p) for p in pwms_rc])

    for i in tqdm(range(n), desc="Computing similarities", leave=False, miniters=1000):
        for j in range(i + 1, n):
            # Forward comparison
            weights = (all_ics[i] + all_ics[j]) / 2.0
            weights_exp = np.repeat(weights, 4)
            w_sum = weights_exp.sum()

            if w_sum < 1e-8:
                sim_fwd = 0.0
            else:
                flat1 = pwms[i].flatten()
                flat2 = pwms[j].flatten()
                mean1 = np.average(flat1, weights=weights_exp)
                mean2 = np.average(flat2, weights=weights_exp)
                cov = np.sum(weights_exp * (flat1 - mean1) * (flat2 - mean2))
                std1 = np.sqrt(np.sum(weights_exp * (flat1 - mean1) ** 2))
                std2 = np.sqrt(np.sum(weights_exp * (flat2 - mean2) ** 2))
                sim_fwd = max(0.0, cov / (std1 * std2)) if std1 > 1e-8 and std2 > 1e-8 else 0.0

            best_sim = sim_fwd

            # Reverse complement comparison
            if check_revcomp:
                weights_rc = (all_ics[i] + all_ics_rc[j]) / 2.0
                weights_rc_exp = np.repeat(weights_rc, 4)
                w_sum_rc = weights_rc_exp.sum()

                if w_sum_rc >= 1e-8:
                    flat2_rc = pwms_rc[j].flatten()
                    mean1 = np.average(flat1, weights=weights_rc_exp)
                    mean2_rc = np.average(flat2_rc, weights=weights_rc_exp)
                    cov_rc = np.sum(weights_rc_exp * (flat1 - mean1) * (flat2_rc - mean2_rc))
                    std1_rc = np.sqrt(np.sum(weights_rc_exp * (flat1 - mean1) ** 2))
                    std2_rc = np.sqrt(np.sum(weights_rc_exp * (flat2_rc - mean2_rc) ** 2))
                    sim_rc = max(0.0, cov_rc / (std1_rc * std2_rc)) if std1_rc > 1e-8 and std2_rc > 1e-8 else 0.0
                    best_sim = max(best_sim, sim_rc)

            sim_matrix[i, j] = best_sim
            sim_matrix[j, i] = best_sim

    return sim_matrix


def cluster_filters_across_models(
    models: List[RashomonCNNModel],
    similarity_threshold: float = 0.7,
    check_revcomp: bool = True,
) -> Tuple[np.ndarray, List[np.ndarray], List[Tuple[int, int]]]:
    """Cluster similar filters across all Rashomon models.

    Uses IC-weighted Pearson correlation with reverse complement matching.

    Returns:
        cluster_ids: cluster assignment for each (model, filter) pair
        all_pwms: list of all PWMs
        filter_indices: list of (model_idx, filter_idx) for each PWM
    """
    all_pwms = []
    filter_indices = []

    for model_idx, model in enumerate(models):
        for filter_idx in range(model.filter_weights.shape[0]):
            pwm = filter_to_pwm(model.filter_weights[filter_idx])
            all_pwms.append(pwm)
            filter_indices.append((model_idx, filter_idx))

    n = len(all_pwms)
    if n <= 1:
        return np.zeros(n, dtype=int), all_pwms, filter_indices

    # Stack PWMs for vectorized computation
    pwm_array = np.array(all_pwms)  # (n, kernel_size, 4)

    # Compute IC-weighted pairwise similarities with revcomp matching
    print(f"Clustering {n} filters (IC-weighted, revcomp={check_revcomp})...", flush=True)
    sim_matrix = compute_pairwise_similarities_vectorized(pwm_array, check_revcomp=check_revcomp)

    # Convert to distance and cluster
    dist_matrix = 1.0 - sim_matrix
    np.fill_diagonal(dist_matrix, 0)

    condensed = pdist(dist_matrix)
    if np.any(np.isnan(condensed)):
        condensed = np.nan_to_num(condensed, nan=1.0)

    Z = linkage(condensed, method="average")
    cluster_ids = fcluster(Z, t=1.0 - similarity_threshold, criterion="distance")

    n_clusters = len(set(cluster_ids))
    print(f"Found {n_clusters} clusters from {n} filters", flush=True)

    return cluster_ids, all_pwms, filter_indices


def compute_cluster_frequencies(
    cluster_ids: np.ndarray,
    filter_indices: List[Tuple[int, int]],
    n_models: int,
) -> Dict[int, float]:
    """Compute frequency of each filter cluster across Rashomon models."""
    cluster_model_presence = {}

    for cluster_id, (model_idx, _) in zip(cluster_ids, filter_indices):
        if cluster_id not in cluster_model_presence:
            cluster_model_presence[cluster_id] = set()
        cluster_model_presence[cluster_id].add(model_idx)

    frequencies = {}
    for cluster_id, model_set in cluster_model_presence.items():
        frequencies[cluster_id] = len(model_set) / n_models

    return frequencies


def get_cluster_representative(
    cluster_id: int,
    cluster_ids: np.ndarray,
    all_pwms: List[np.ndarray],
    filter_indices: List[Tuple[int, int]],
    models: List[RashomonCNNModel],
) -> Tuple[np.ndarray, float, str, float]:
    """Get representative PWM for a cluster (highest importance).

    Returns:
        pwm: representative PWM
        importance: importance score
        consensus: consensus sequence
        avg_ic: average information content
    """
    mask = cluster_ids == cluster_id
    indices = np.where(mask)[0]

    best_importance = -1.0
    best_pwm = None

    for idx in indices:
        model_idx, filter_idx = filter_indices[idx]
        importance = models[model_idx].filter_importance[filter_idx]
        if importance > best_importance:
            best_importance = importance
            best_pwm = all_pwms[idx]

    consensus = pwm_to_consensus(best_pwm) if best_pwm is not None else ""
    avg_ic = float(pwm_to_ic(best_pwm).mean()) if best_pwm is not None else 0.0

    return best_pwm, best_importance, consensus, avg_ic


def save_filter_weights(
    models: List[RashomonCNNModel],
    out_path: Path,
    activation: str,
):
    """Save raw filter weights for all models in Rashomon set.

    Saves:
        - filter_weights: (n_models, n_filters, kernel_size, 4)
        - filter_importance: (n_models, n_filters)
        - model_seeds: (n_models,)
        - model_performance: (n_models,)
    """
    n_models = len(models)
    if n_models == 0:
        return

    n_filters = models[0].filter_weights.shape[0]
    kernel_size = models[0].filter_weights.shape[1]

    weights = np.zeros((n_models, n_filters, kernel_size, 4), dtype=np.float32)
    importance = np.zeros((n_models, n_filters), dtype=np.float32)
    seeds = np.zeros(n_models, dtype=np.int64)
    performance = np.zeros(n_models, dtype=np.float32)

    for i, model in enumerate(models):
        weights[i] = model.filter_weights
        importance[i] = model.filter_importance
        seeds[i] = model.seed
        performance[i] = model.performance

    np.savez_compressed(
        out_path / f"filter_weights_{activation}.npz",
        filter_weights=weights,
        filter_importance=importance,
        model_seeds=seeds,
        model_performance=performance,
        activation=activation,
    )
    print(f"Saved filter weights: {out_path / f'filter_weights_{activation}.npz'}", flush=True)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def save_fig_formats(fig: plt.Figure, base_path: Path, dpi: int = 150):
    for ext in (".png", ".pdf", ".svg"):
        fig.savefig(Path(base_path).with_suffix(ext), dpi=dpi, bbox_inches="tight")


def plot_pwm_logo(pwm: np.ndarray, ax: plt.Axes, title: str = ""):
    """Plot a simple PWM logo on the given axes."""
    bases = ["A", "C", "G", "T"]
    colors = {"A": "#00A000", "C": "#0000FF", "G": "#FFA500", "T": "#FF0000"}

    n_pos = pwm.shape[0]
    for pos in range(n_pos):
        sorted_idx = np.argsort(pwm[pos])
        y = 0
        for idx in sorted_idx:
            height = pwm[pos, idx]
            if height > 0.05:
                ax.text(
                    pos + 0.5, y + height / 2, bases[idx],
                    ha="center", va="center",
                    fontsize=min(12, 120 / n_pos),
                    fontweight="bold",
                    color=colors[bases[idx]],
                )
            y += height

    ax.set_xlim(0, n_pos)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Position")
    ax.set_ylabel("Probability")
    if title:
        ax.set_title(title, fontsize=10)


def plot_cluster_frequency(
    frequencies: Dict[int, float],
    cluster_consensuses: Dict[int, str],
    out_dir: Path,
    activation: str,
    topn: int = 30,
):
    """Plot filter cluster frequency in Rashomon set."""
    sorted_clusters = sorted(frequencies.items(), key=lambda x: -x[1])[:topn]

    cluster_ids = [c[0] for c in sorted_clusters]
    freqs = [c[1] for c in sorted_clusters]
    labels = [cluster_consensuses.get(c, f"C{c}") for c in cluster_ids]

    colors = [
        "#2ca02c" if f >= 1.0 else "#1f78b4" if f >= 0.5 else "#ff7f0e" if f > 0 else "#d62728"
        for f in freqs
    ]

    fig, ax = plt.subplots(figsize=(max(10, 0.3 * len(labels)), 5))
    x = np.arange(len(labels))
    ax.bar(x, freqs, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.axhline(1.0, color="green", linestyle="--", linewidth=1, alpha=0.7)
    ax.axhline(0.5, color="blue", linestyle=":", linewidth=1, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylabel("Frequency in Rashomon set")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"{activation.upper()}: Filter Motif Frequency (Top {topn})")

    patches = [
        mpatches.Patch(color="#2ca02c", label="Necessary (=1.0)"),
        mpatches.Patch(color="#1f78b4", label="Common (>=0.5)"),
        mpatches.Patch(color="#ff7f0e", label="Rare (<0.5)"),
    ]
    ax.legend(handles=patches, loc="upper right", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    save_fig_formats(fig, out_dir / f"filter_frequency_{activation}")
    plt.close(fig)


def plot_top_motifs(
    frequencies: Dict[int, float],
    cluster_ids: np.ndarray,
    all_pwms: List[np.ndarray],
    filter_indices: List[Tuple[int, int]],
    models: List[RashomonCNNModel],
    out_dir: Path,
    activation: str,
    topn: int = 12,
):
    """Plot top motif logos."""
    sorted_clusters = sorted(frequencies.items(), key=lambda x: -x[1])[:topn]

    n_cols = 4
    n_rows = (len(sorted_clusters) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows))
    axes = np.atleast_2d(axes)

    for i, (cluster_id, freq) in enumerate(sorted_clusters):
        row, col = divmod(i, n_cols)
        ax = axes[row, col]
        pwm, importance, consensus, avg_ic = get_cluster_representative(
            cluster_id, cluster_ids, all_pwms, filter_indices, models
        )
        if pwm is not None:
            plot_pwm_logo(pwm, ax, title=f"{consensus}\nfreq={freq:.2f}, IC={avg_ic:.2f}")

    # Hide empty axes
    for i in range(len(sorted_clusters), n_rows * n_cols):
        row, col = divmod(i, n_cols)
        axes[row, col].axis("off")

    fig.suptitle(f"{activation.upper()}: Top {topn} Motifs in Rashomon Set", fontsize=12, fontweight="bold")
    fig.tight_layout()
    save_fig_formats(fig, out_dir / f"top_motifs_{activation}")
    plt.close(fig)


def plot_performance_distribution(
    all_models: List[RashomonCNNModel],
    rset: RashomonCNNSet,
    out_dir: Path,
    activation: str,
):
    """Plot distribution of model performances."""
    perfs = [m.performance for m in all_models]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(perfs, bins=30, color="#1f78b4", alpha=0.7, edgecolor="black")
    ax.axvline(rset.threshold, color="red", linestyle="--", linewidth=2, label=f"Thresh ({rset.threshold:.3f})")
    ax.axvline(rset.best_performance, color="green", linestyle="-", linewidth=2, label=f"Best ({rset.best_performance:.3f})")
    ax.set_xlabel("Balanced Accuracy")
    ax.set_ylabel("Count")
    ax.set_title(f"{activation.upper()}: {rset.size}/{len(all_models)} models in Rashomon set")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    save_fig_formats(fig, out_dir / f"performance_distribution_{activation}")
    plt.close(fig)


def plot_comparison(
    freq_relu: Dict[int, float],
    freq_exp: Dict[int, float],
    cons_relu: Dict[int, str],
    cons_exp: Dict[int, str],
    out_dir: Path,
):
    """Compare necessary/frequent filters between ReLU and exp activations."""
    # Count necessary and frequent
    nec_relu = sum(1 for f in freq_relu.values() if f >= 1.0)
    com_relu = sum(1 for f in freq_relu.values() if f >= 0.5)
    nec_exp = sum(1 for f in freq_exp.values() if f >= 1.0)
    com_exp = sum(1 for f in freq_exp.values() if f >= 0.5)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(2)
    width = 0.35

    ax.bar(x - width/2, [nec_relu, nec_exp], width, label="Necessary (freq=1.0)", color="#2ca02c", alpha=0.8)
    ax.bar(x + width/2, [com_relu, com_exp], width, label="Common (freq>=0.5)", color="#1f78b4", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(["ReLU", "Exp"])
    ax.set_ylabel("Number of filter clusters")
    ax.set_title("Filter Clusters by Activation Type")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    # Annotate
    for i, (nec, com) in enumerate([(nec_relu, com_relu), (nec_exp, com_exp)]):
        ax.text(i - width/2, nec + 0.5, str(nec), ha="center", va="bottom", fontweight="bold")
        ax.text(i + width/2, com + 0.5, str(com), ha="center", va="bottom", fontweight="bold")

    fig.tight_layout()
    save_fig_formats(fig, out_dir / "activation_comparison")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_hps_from_json(json_path: Path) -> Dict:
    """Load first-layer hyperparameters from tuned model's best_meta.json.

    Decodes Optuna's continuous/indexed parameters to actual values.
    """
    with open(json_path) as f:
        meta = json.load(f)

    params = meta.get("params", {})

    # Decode kernel size: k1 = 2 * k1_idx + 1
    k1_idx = params.get("k1_idx", 4)  # default ~9
    kernel_size = 2 * k1_idx + 1

    # Decode number of filters (round continuous value)
    n_filters = int(round(params.get("c1_cont", 64)))

    # Learning rate
    lr = params.get("lr", 1e-3)

    # Batch size
    batch_size = params.get("batch_size", 16)

    return {
        "kernel_size": kernel_size,
        "n_filters": n_filters,
        "lr": lr,
        "batch_size": batch_size,
        "source": str(json_path),
    }


def main():
    parser = argparse.ArgumentParser(description="Rashomon CNN Filter Analysis (Multi-Architecture)")
    parser.add_argument("--phenotype", type=str, default="Spore formation")
    parser.add_argument("--output_dir", type=Path, default=Path("sporulation/results/rashomon_cnn"))
    parser.add_argument("--metadata_xlsx", type=Path, default=METADATA_XLSX)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds_per_config", type=int, default=SEEDS_PER_CONFIG, help="Seeds per architecture config")
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON, help="Performance tolerance")
    parser.add_argument("--seq_len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--max_epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--similarity_threshold", type=float, default=0.7, help="PWM similarity threshold for clustering")
    # Parallel mode: run single config+activation (for SLURM array jobs)
    parser.add_argument("--config_idx", type=int, default=None, help="Run only this config index (0-4)")
    parser.add_argument("--activation", type=str, default=None, choices=["relu", "exp"], help="Run only this activation")
    args = parser.parse_args()

    phen_name = args.phenotype.strip()
    phen_safe = phen_name.replace(" ", "_")
    out_dir = args.output_dir / phen_safe
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine which configs and activations to run
    if args.config_idx is not None and args.activation is not None:
        # Single config+activation mode (parallel array job)
        configs_to_run = [HP_CONFIGS[args.config_idx]]
        activations_to_run = [args.activation]
        print(f"=== Rashomon CNN (parallel mode): {phen_name} ===", flush=True)
        print(f"Config {args.config_idx}: kernel={configs_to_run[0]['kernel_size']}, filters={configs_to_run[0]['n_filters']}", flush=True)
        print(f"Activation: {args.activation}", flush=True)
    else:
        # Full mode: all configs × all activations
        configs_to_run = HP_CONFIGS
        activations_to_run = ["relu", "exp"]
        n_total_models = len(HP_CONFIGS) * args.seeds_per_config
        print(f"=== Multi-Architecture Rashomon CNN: {phen_name} ===", flush=True)
        print(f"Configs: {len(HP_CONFIGS)} architectures × {args.seeds_per_config} seeds = {n_total_models} models per activation", flush=True)
        for cfg in HP_CONFIGS:
            print(f"  - kernel={cfg['kernel_size']:2d}, filters={cfg['n_filters']}", flush=True)

    print(f"Device: {DEVICE}", flush=True)
    print(f"Epsilon: {args.epsilon} (models within {args.epsilon*100:.1f}% bal-acc of best)", flush=True)

    # Load data
    metadata_df = read_metadata_table(args.metadata_xlsx)
    train_dir = DATA_ROOT / "train"
    val_dir = DATA_ROOT / "validation"

    train_sequences = _load_sequences_list(
        train_dir, metadata_df, args.seq_len, phen_name, verbose=True
    )
    val_sequences = _load_sequences_list(
        val_dir, metadata_df, args.seq_len, phen_name, verbose=True
    )

    if len(train_sequences) < 50 or len(val_sequences) < 20:
        print(f"[Abort] Insufficient sequences: train={len(train_sequences)}, val={len(val_sequences)}")
        return

    print(f"Data: train={len(train_sequences)}, val={len(val_sequences)}", flush=True)

    # Pre-compute one-hot encoded windows ONCE (this is the bottleneck fix)
    # Cache to DATA_ROOT to avoid recomputing across runs
    n_train_windows = 2048
    n_val_windows = 512

    cache_dir = DATA_ROOT / ".cache" / "rashomon_cnn"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{phen_safe}_seqlen{args.seq_len}_nwin{n_train_windows}_seed{args.seed}.pt"

    if cache_file.exists():
        print(f"\n--- Loading cached data from {cache_file} ---", flush=True)
        cached = torch.load(cache_file, weights_only=False)
        X_train, y_train = cached["X_train"], cached["y_train"]
        X_val, y_val = cached["X_val"], cached["y_val"]
        print(f"Loaded Train: {X_train.shape}, Val: {X_val.shape}", flush=True)
    else:
        print("\n--- Pre-encoding data (one-time cost) ---", flush=True)
        X_train, y_train = precompute_windows(
            train_sequences, args.seq_len, n_train_windows, seed=args.seed, verbose=True
        )
        X_val, y_val = precompute_windows(
            val_sequences, args.seq_len, n_val_windows, seed=args.seed + 1, verbose=True
        )
        # Save cache to DATA_ROOT (not working directory)
        print(f"Saving cache to {cache_file}...", flush=True)
        torch.save({"X_train": X_train, "y_train": y_train, "X_val": X_val, "y_val": y_val}, cache_file)
        print("Cache saved.", flush=True)

    # Create datasets and dataloaders (reused across all models)
    # num_workers=0 because data is already in memory, no need for parallel loading
    train_ds = PrecomputedDataset(X_train, y_train)
    val_ds = PrecomputedDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # Compute class weights once
    num_classes = len(set(s["label"] for s in train_sequences))
    class_weights = class_weights_from_sequences(train_sequences)

    print(f"Pre-encoding complete. Train: {X_train.shape}, Val: {X_val.shape}", flush=True)

    # Results structure: results[activation][kernel_size] = {...}
    results = {"relu": {}, "exp": {}}
    all_models_by_activation = {"relu": [], "exp": []}

    # Train models for each architecture config
    # In parallel mode, offset seed by config_idx to ensure different seeds across jobs
    base_model_idx = (args.config_idx or 0) * len(["relu", "exp"]) * args.seeds_per_config
    if args.activation == "exp":
        base_model_idx += args.seeds_per_config
    model_idx = base_model_idx

    for cfg_idx, cfg in enumerate(configs_to_run):
        kernel_size = cfg["kernel_size"]
        n_filters = cfg["n_filters"]

        print(f"\n{'='*60}", flush=True)
        print(f"Config: kernel={kernel_size}, filters={n_filters}", flush=True)
        print(f"{'='*60}", flush=True)

        for activation in activations_to_run:
            print(f"\n  [{activation.upper()}] Training {args.seeds_per_config} models...", flush=True)

            config_models = []
            for i in tqdm(range(args.seeds_per_config), desc=f"k{kernel_size}_{activation}",
                          miniters=max(1, args.seeds_per_config // 10)):
                model_seed = args.seed + model_idx * 7919
                model_idx += 1
                model = fit_single_cnn(
                    train_loader=train_loader,
                    val_loader=val_loader,
                    n_filters=n_filters,
                    kernel_size=kernel_size,
                    activation=activation,
                    seed=model_seed,
                    num_classes=num_classes,
                    class_weights=class_weights,
                    max_epochs=args.max_epochs,
                    lr=args.lr,
                )
                config_models.append(model)
                all_models_by_activation[activation].append(model)

            # Build Rashomon set for this config
            best_perf = max(m.performance for m in config_models)
            threshold = best_perf - args.epsilon
            rset = RashomonCNNSet(
                activation=activation,
                epsilon=args.epsilon,
                best_performance=best_perf,
                threshold=threshold,
                models=[m for m in config_models if m.performance >= threshold],
            )

            print(f"  [{activation.upper()}] best={best_perf:.4f}, Rashomon={rset.size}/{len(config_models)}", flush=True)

            if rset.size == 0:
                print(f"  [Warning] Empty Rashomon set for {activation} k={kernel_size}")
                continue

            # Cluster filters within this config (same kernel_size = comparable PWMs)
            cluster_ids, all_pwms, filter_indices = cluster_filters_across_models(
                rset.models, similarity_threshold=args.similarity_threshold
            )

            # Compute frequencies
            frequencies = compute_cluster_frequencies(cluster_ids, filter_indices, rset.size)

            # Get consensus sequences and IC values
            cluster_consensuses = {}
            cluster_ics = {}
            for cluster_id in frequencies.keys():
                _, _, consensus, avg_ic = get_cluster_representative(
                    cluster_id, cluster_ids, all_pwms, filter_indices, rset.models
                )
                cluster_consensuses[cluster_id] = consensus
                cluster_ics[cluster_id] = avg_ic

            # Count necessary and common
            n_necessary = sum(1 for f in frequencies.values() if f >= 1.0)
            n_common = sum(1 for f in frequencies.values() if f >= 0.5)
            print(f"  [{activation.upper()}] {n_necessary} necessary, {n_common} common clusters", flush=True)

            # Create config-specific output dir
            cfg_out_dir = out_dir / f"k{kernel_size}"
            cfg_out_dir.mkdir(parents=True, exist_ok=True)

            # Save filter weights
            save_filter_weights(rset.models, cfg_out_dir, activation)

            # Plots per config
            plot_performance_distribution(config_models, rset, cfg_out_dir, activation)
            plot_cluster_frequency(frequencies, cluster_consensuses, cfg_out_dir, activation)
            plot_top_motifs(frequencies, cluster_ids, all_pwms, filter_indices, rset.models, cfg_out_dir, activation)

            # Store results
            results[activation][kernel_size] = {
                "config_models": config_models,
                "rset": rset,
                "frequencies": frequencies,
                "cluster_consensuses": cluster_consensuses,
                "cluster_ics": cluster_ics,
                "n_necessary": n_necessary,
                "n_common": n_common,
                "cluster_ids": cluster_ids,
                "all_pwms": all_pwms,
                "filter_indices": filter_indices,
            }

    # Cross-architecture summary
    print(f"\n{'='*60}", flush=True)
    print("Summary", flush=True)
    print(f"{'='*60}", flush=True)

    for activation in activations_to_run:
        if not results[activation]:
            continue
        print(f"\n{activation.upper()}:", flush=True)
        for ks in sorted(results[activation].keys()):
            r = results[activation][ks]
            print(f"  k={ks:2d}: best={r['rset'].best_performance:.4f}, "
                  f"rashomon={r['rset'].size}, necessary={r['n_necessary']}, common={r['n_common']}", flush=True)

    # In parallel mode, save per-config summary; full mode saves combined summary
    is_parallel = args.config_idx is not None and args.activation is not None

    # Save summary
    summary = {
        "phenotype": phen_name,
        "epsilon": args.epsilon,
        "parallel_mode": is_parallel,
        "config_idx": args.config_idx,
        "activation_filter": args.activation,
        "hp_configs": configs_to_run,
        "seeds_per_config": args.seeds_per_config,
        "seq_len": args.seq_len,
        "data": {
            "n_train": len(train_sequences),
            "n_val": len(val_sequences),
        },
    }

    for activation in activations_to_run:
        if not results[activation]:
            continue
        summary[activation] = {}
        for ks, r in results[activation].items():
            summary[activation][f"k{ks}"] = {
                "kernel_size": ks,
                "n_filters": next(c["n_filters"] for c in configs_to_run if c["kernel_size"] == ks),
                "best_performance": float(r["rset"].best_performance),
                "rashomon_size": r["rset"].size,
                "n_filter_clusters": len(r["frequencies"]),
                "n_necessary": r["n_necessary"],
                "n_common": r["n_common"],
                "top_motifs": [
                    {
                        "consensus": r["cluster_consensuses"][cid],
                        "frequency": float(freq),
                        "avg_ic": float(r["cluster_ics"].get(cid, 0.0)),
                    }
                    for cid, freq in sorted(r["frequencies"].items(), key=lambda x: -x[1])[:5]
                ],
            }

    # In parallel mode, save with config/activation suffix to avoid conflicts
    if is_parallel:
        cfg = configs_to_run[0]
        summary_file = out_dir / f"summary_k{cfg['kernel_size']}_{args.activation}.json"
    else:
        summary_file = out_dir / "rashomon_cnn_summary.json"

    with summary_file.open("w") as f:
        json.dump(summary, f, indent=2)

    # Save motif frequencies to CSV per config (with IC values)
    for activation in activations_to_run:
        if not results[activation]:
            continue
        for ks, r in results[activation].items():
            rows = []
            for cid, freq in r["frequencies"].items():
                rows.append({
                    "kernel_size": ks,
                    "cluster_id": cid,
                    "consensus": r["cluster_consensuses"].get(cid, ""),
                    "avg_ic": r["cluster_ics"].get(cid, 0.0),
                    "frequency": freq,
                    "is_necessary": freq >= 1.0,
                    "is_common": freq >= 0.5,
                })
            df = pd.DataFrame(rows).sort_values("frequency", ascending=False)
            cfg_out_dir = out_dir / f"k{ks}"
            df.to_csv(cfg_out_dir / f"filter_clusters_{activation}.csv", index=False)

    # Create combined CSV across all configs (only in full mode)
    if not is_parallel:
        for activation in activations_to_run:
            if not results[activation]:
                continue
            all_rows = []
            for ks, r in results[activation].items():
                for cid, freq in r["frequencies"].items():
                    all_rows.append({
                        "kernel_size": ks,
                        "cluster_id": cid,
                        "consensus": r["cluster_consensuses"].get(cid, ""),
                        "avg_ic": r["cluster_ics"].get(cid, 0.0),
                        "frequency": freq,
                        "is_necessary": freq >= 1.0,
                        "is_common": freq >= 0.5,
                    })
            df = pd.DataFrame(all_rows).sort_values(["kernel_size", "frequency"], ascending=[True, False])
            df.to_csv(out_dir / f"all_filter_clusters_{activation}.csv", index=False)

    print(f"\nOutputs saved to: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
