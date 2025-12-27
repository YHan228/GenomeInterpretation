#!/usr/bin/env python3
"""Ablation experiments to understand what CNN is learning.

Tests:
1. Full model (baseline)
2. Random frozen filters + trained dense layer only
3. No dense layer (direct sigmoid on pooled activations)
4. K-mer frequency baseline (explicit feature extraction)

If (2) performs as well as (1), filters don't matter - it's all in the dense layer.
If (4) performs similarly, CNN is just counting k-mers.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import balanced_accuracy_score
from sklearn.linear_model import LogisticRegression

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import utilities
_HERE = Path(__file__).resolve().parent
_PHENOTYPE_CODE_DIR = str(_HERE.parent.parent / "phenotype" / "code")
if _PHENOTYPE_CODE_DIR not in sys.path:
    sys.path.insert(0, _PHENOTYPE_CODE_DIR)

from phenotype_utils import (
    build_labels_map_and_classes,
    DATA_ROOT,
    read_metadata_table,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
METADATA_XLSX = Path("sporulation/microbe.cards table S1.xlsx")


def one_hot_encode_fast(seq: str) -> torch.Tensor:
    """Vectorized one-hot encoding."""
    seq_upper = seq.upper()
    seq_array = np.frombuffer(seq_upper.encode('ascii'), dtype=np.uint8)
    one_hot = np.zeros((4, len(seq)), dtype=np.float32)
    one_hot[0] = (seq_array == ord('A'))
    one_hot[1] = (seq_array == ord('C'))
    one_hot[2] = (seq_array == ord('G'))
    one_hot[3] = (seq_array == ord('T'))
    return torch.from_numpy(one_hot)


class GenomeDataset(Dataset):
    def __init__(self, tensors: List[torch.Tensor], labels: List[int]):
        self.tensors = tensors
        self.labels = labels

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, idx):
        return self.tensors[idx], self.labels[idx]


# ---------------------------------------------------------------------------
# Model Variants
# ---------------------------------------------------------------------------

class FullCNN(nn.Module):
    """Standard 1-layer CNN (baseline). Returns logits for BCEWithLogitsLoss."""
    def __init__(self, n_filters: int = 48, kernel_size: int = 9, activation: str = "relu"):
        super().__init__()
        self.conv = nn.Conv1d(4, n_filters, kernel_size, padding=0)
        self.bn = nn.BatchNorm1d(n_filters)
        self.activation = nn.ReLU() if activation == "relu" else lambda x: torch.exp(x)
        self.fc = nn.Linear(n_filters, 1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x) if callable(self.activation) else torch.exp(x)
        x = x.max(dim=2)[0]  # GlobalMaxPool
        return self.fc(x)  # Return logits, not probabilities


class FrozenFilterCNN(nn.Module):
    """Random frozen filters, only train dense layer. Returns logits."""
    def __init__(self, n_filters: int = 48, kernel_size: int = 9, activation: str = "relu"):
        super().__init__()
        self.conv = nn.Conv1d(4, n_filters, kernel_size, padding=0)
        self.bn = nn.BatchNorm1d(n_filters)
        self.activation = nn.ReLU() if activation == "relu" else lambda x: torch.exp(x)
        self.fc = nn.Linear(n_filters, 1)

        # Freeze conv and bn
        for param in self.conv.parameters():
            param.requires_grad = False
        for param in self.bn.parameters():
            param.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            x = self.conv(x)
            x = self.bn(x)
            x = self.activation(x) if callable(self.activation) else torch.exp(x)
            x = x.max(dim=2)[0]
        return self.fc(x)  # Return logits


class NoDenseCNN(nn.Module):
    """No dense layer - direct logit on mean of pooled activations."""
    def __init__(self, n_filters: int = 48, kernel_size: int = 9, activation: str = "relu"):
        super().__init__()
        self.conv = nn.Conv1d(4, n_filters, kernel_size, padding=0)
        self.bn = nn.BatchNorm1d(n_filters)
        self.activation = nn.ReLU() if activation == "relu" else lambda x: torch.exp(x)
        # Single learnable scalar + bias for combining filter outputs
        self.scale = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x) if callable(self.activation) else torch.exp(x)
        x = x.max(dim=2)[0]  # (batch, n_filters)
        x = x.mean(dim=1, keepdim=True)  # Average all filters
        return self.scale * x + self.bias  # Return logits


# ---------------------------------------------------------------------------
# K-mer Baseline (Vectorized)
# ---------------------------------------------------------------------------

def extract_kmer_features_fast(sequences: List[str], k: int = 6) -> np.ndarray:
    """Extract k-mer frequency features using vectorized numpy operations.

    ~100x faster than naive Python implementation.
    """
    n_kmers = 4 ** k
    features = np.zeros((len(sequences), n_kmers), dtype=np.float32)

    # Base encoding: A=0, C=1, G=2, T=3
    base_map = np.zeros(256, dtype=np.int8)
    base_map[ord('A')] = 0
    base_map[ord('C')] = 1
    base_map[ord('G')] = 2
    base_map[ord('T')] = 3
    base_map[ord('a')] = 0
    base_map[ord('c')] = 1
    base_map[ord('g')] = 2
    base_map[ord('t')] = 3
    # Invalid bases get -1 (will be filtered)
    base_map[:] = np.where(base_map == 0, -1, base_map)
    base_map[ord('A')] = 0
    base_map[ord('a')] = 0

    # Powers for k-mer index computation
    powers = 4 ** np.arange(k - 1, -1, -1)

    for i, seq in enumerate(tqdm(sequences, desc=f"Extracting {k}-mers")):
        # Convert to byte array and encode
        seq_bytes = np.frombuffer(seq.upper().encode(), dtype=np.uint8)
        encoded = base_map[seq_bytes].astype(np.int32)  # Cast to int32 to avoid overflow

        # Find valid positions (no N or other invalid bases)
        valid = encoded >= 0

        # Use sliding window to compute k-mer indices
        # Only compute if we have enough valid bases
        if len(seq_bytes) >= k:
            # Create view of k consecutive positions
            # This is a strided array trick for efficiency
            n_pos = len(seq_bytes) - k + 1

            # Compute k-mer indices using convolution-like approach
            kmer_indices = np.zeros(n_pos, dtype=np.int32)
            all_valid = np.ones(n_pos, dtype=bool)

            for j in range(k):
                pos_encoded = encoded[j:j+n_pos]
                all_valid &= (pos_encoded >= 0)
                # Use max(0, x) to avoid negative contributions
                kmer_indices += np.maximum(pos_encoded, 0) * powers[j]

            # Count only valid k-mers
            valid_indices = kmer_indices[all_valid]
            if len(valid_indices) > 0:
                counts = np.bincount(valid_indices, minlength=n_kmers)
                total = counts.sum()
                if total > 0:
                    features[i] = counts / total

    return features


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_cnn(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    max_epochs: int = 15,
    lr: float = 0.001,
) -> Tuple[float, List[float]]:
    """Train CNN model. Models return logits, use BCEWithLogitsLoss."""
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)
    criterion = nn.BCEWithLogitsLoss()  # AMP-safe
    scaler = GradScaler()

    best_acc = 0.0
    history = []

    for epoch in range(max_epochs):
        model.train()
        for x, y in train_loader:
            x = x.to(DEVICE)
            y = y.float().to(DEVICE).unsqueeze(1)

            optimizer.zero_grad()
            with autocast():
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # Validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(DEVICE)
                with autocast():
                    logits = model(x)
                probs = torch.sigmoid(logits)  # Convert logits to probabilities
                all_preds.extend((probs.cpu().numpy() > 0.5).astype(int).flatten())
                all_labels.extend(y.numpy())

        acc = balanced_accuracy_score(all_labels, all_preds)
        history.append(acc)
        scheduler.step(1 - acc)

        if acc > best_acc:
            best_acc = acc

    return best_acc, history


def train_kmer_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> float:
    """Train logistic regression on k-mer features."""
    clf = LogisticRegression(max_iter=1000, class_weight='balanced', n_jobs=-1)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_val)
    return balanced_accuracy_score(y_val, preds)


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def parse_fasta_simple(filepath: str) -> str:
    """Parse first sequence from FASTA file."""
    seq_parts = []
    with open(filepath) as f:
        for line in f:
            if line.startswith(">"):
                if seq_parts:
                    break
            else:
                seq_parts.append(line.strip())
    return "".join(seq_parts)


def load_data(phenotype: str, seq_len: int = 100000) -> Tuple[List, List, List, List]:
    """Load sequences and labels."""
    metadata_df = read_metadata_table(METADATA_XLSX)
    phenotype_col = phenotype  # Keep original name with spaces

    train_dirs = [str(DATA_ROOT / "train")]
    labels_map, classes = build_labels_map_and_classes(
        metadata_df, phenotype_col=phenotype_col, file_col="Fasta file", train_dirs=train_dirs
    )

    data_dir = DATA_ROOT / "train"
    sequences_train, labels_train = [], []
    sequences_val, labels_val = [], []

    exts = (".fasta", ".fa", ".fna")
    files = [f for f in os.listdir(data_dir) if f.endswith(exts)]

    for fname in tqdm(files, desc="Loading sequences"):
        # Labels map uses lowercase keys
        label = labels_map.get(fname.strip().lower())
        if label is None:
            continue

        fpath = data_dir / fname
        seq = parse_fasta_simple(str(fpath))

        if len(seq) < seq_len:
            seq = seq.ljust(seq_len, "N")  # Pad if needed

        seq = seq[:seq_len]

        # Simple split
        if np.random.random() < 0.18:
            sequences_val.append(seq)
            labels_val.append(label)
        else:
            sequences_train.append(seq)
            labels_train.append(label)

    return sequences_train, labels_train, sequences_val, labels_val


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phenotype", default="Spore formation")
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--kernel_size", type=int, default=9)
    parser.add_argument("--n_filters", type=int, default=48)
    parser.add_argument("--seq_len", type=int, default=100000)
    parser.add_argument("--output_dir", type=str, default="sporulation/results/ablation")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"Phenotype: {args.phenotype}")

    # Load data
    np.random.seed(42)
    seqs_train, labels_train, seqs_val, labels_val = load_data(args.phenotype, args.seq_len)

    print(f"\nData: {len(seqs_train)} train, {len(seqs_val)} val")
    print(f"Class balance train: {np.mean(labels_train):.2%} positive")
    print(f"Class balance val: {np.mean(labels_val):.2%} positive")

    # Prepare tensors for CNN
    print("\nEncoding sequences...")
    tensors_train = [one_hot_encode_fast(s) for s in tqdm(seqs_train, desc="Train")]
    tensors_val = [one_hot_encode_fast(s) for s in tqdm(seqs_val, desc="Val")]

    train_dataset = GenomeDataset(tensors_train, labels_train)
    val_dataset = GenomeDataset(tensors_val, labels_val)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)

    results = {
        "full_cnn": [],
        "frozen_filters": [],
        "no_dense": [],
        "kmer_6": None,
        "kmer_4": None,
    }

    # Run CNN ablations
    for seed in range(args.n_seeds):
        print(f"\n=== Seed {seed} ===")
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Full CNN
        print("Training Full CNN...")
        model = FullCNN(args.n_filters, args.kernel_size, "relu")
        acc, _ = train_cnn(model, train_loader, val_loader)
        results["full_cnn"].append(acc)
        print(f"  Full CNN: {acc:.4f}")

        # Frozen filters
        print("Training Frozen Filters + Dense...")
        model = FrozenFilterCNN(args.n_filters, args.kernel_size, "relu")
        acc, _ = train_cnn(model, train_loader, val_loader)
        results["frozen_filters"].append(acc)
        print(f"  Frozen Filters: {acc:.4f}")

        # No dense
        print("Training No Dense Layer...")
        model = NoDenseCNN(args.n_filters, args.kernel_size, "relu")
        acc, _ = train_cnn(model, train_loader, val_loader)
        results["no_dense"].append(acc)
        print(f"  No Dense: {acc:.4f}")

    # K-mer baselines (no seeds needed - deterministic)
    print("\n=== K-mer Baselines ===")

    for k in [4, 6]:
        print(f"\nExtracting {k}-mer features...")
        X_train = extract_kmer_features_fast(seqs_train, k)
        X_val = extract_kmer_features_fast(seqs_val, k)

        print(f"Training {k}-mer logistic regression...")
        acc = train_kmer_baseline(X_train, np.array(labels_train), X_val, np.array(labels_val))
        results[f"kmer_{k}"] = acc
        print(f"  {k}-mer LR: {acc:.4f}")

    # Summary
    print("\n" + "=" * 60)
    print("ABLATION RESULTS SUMMARY")
    print("=" * 60)

    print(f"\nFull CNN:        {np.mean(results['full_cnn']):.4f} ± {np.std(results['full_cnn']):.4f}")
    print(f"Frozen Filters:  {np.mean(results['frozen_filters']):.4f} ± {np.std(results['frozen_filters']):.4f}")
    print(f"No Dense Layer:  {np.mean(results['no_dense']):.4f} ± {np.std(results['no_dense']):.4f}")
    print(f"4-mer LR:        {results['kmer_4']:.4f}")
    print(f"6-mer LR:        {results['kmer_6']:.4f}")

    # Interpretation
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)

    full_mean = np.mean(results['full_cnn'])
    frozen_mean = np.mean(results['frozen_filters'])

    if abs(full_mean - frozen_mean) < 0.02:
        print("✓ Frozen filters ≈ Full CNN: Filters don't matter!")
        print("  The dense layer learns to combine random projections.")
    else:
        print("✗ Frozen filters < Full CNN: Filters do learn something.")

    if results['kmer_6'] is not None and abs(full_mean - results['kmer_6']) < 0.03:
        print("✓ K-mer baseline ≈ Full CNN: CNN is just counting k-mers!")
    else:
        print("✗ K-mer baseline ≠ Full CNN: CNN learns more than k-mer counts.")

    # Save results
    with open(output_dir / "ablation_results.json", "w") as f:
        json.dump({k: v if not isinstance(v, list) else {"mean": np.mean(v), "std": np.std(v), "values": v}
                   for k, v in results.items()}, f, indent=2)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    methods = ["Full CNN", "Frozen Filters", "No Dense", "4-mer LR", "6-mer LR"]
    means = [
        np.mean(results['full_cnn']),
        np.mean(results['frozen_filters']),
        np.mean(results['no_dense']),
        results['kmer_4'],
        results['kmer_6'],
    ]
    stds = [
        np.std(results['full_cnn']),
        np.std(results['frozen_filters']),
        np.std(results['no_dense']),
        0,
        0,
    ]

    colors = ['#1f78b4', '#33a02c', '#e31a1c', '#ff7f00', '#6a3d9a']
    bars = ax.bar(methods, means, yerr=stds, capsize=5, color=colors, alpha=0.8)

    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Random')
    ax.set_ylabel("Balanced Accuracy", fontsize=12)
    ax.set_title(f"Ablation Study: {args.phenotype}\nWhat does the CNN actually learn?", fontsize=13)
    ax.set_ylim(0.4, 1.0)
    ax.grid(True, axis='y', alpha=0.3)

    # Add value labels
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{mean:.3f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    fig.savefig(output_dir / "ablation_results.png", dpi=150, bbox_inches="tight")
    fig.savefig(output_dir / "ablation_results.pdf", bbox_inches="tight")
    plt.close()

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
