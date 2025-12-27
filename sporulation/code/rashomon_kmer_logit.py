#!/usr/bin/env python3
"""
Rashomon analysis for k-mer logistic regression models.

Given that k-mer LR outperforms CNN for sporulation prediction, we apply the
Rashomon framework to identify which k-mers are consistently important across
models in the ε-Rashomon set.

Usage:
    python rashomon_kmer_logit.py "Spore formation" --k 3 4 6 --n_models 100
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
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

METADATA_XLSX = Path("sporulation/microbe.cards table S1.xlsx")


# ---------------------------------------------------------------------------
# K-mer utilities
# ---------------------------------------------------------------------------

def generate_all_kmers(k: int) -> List[str]:
    """Generate all possible k-mers."""
    bases = 'ACGT'
    return [''.join(p) for p in itertools.product(bases, repeat=k)]


def extract_kmer_features_fast(sequences: List[str], k: int) -> np.ndarray:
    """Extract k-mer frequency features using vectorized numpy operations."""
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

    for i, seq in enumerate(tqdm(sequences, desc=f"Extracting {k}-mers", leave=False)):
        # Convert to byte array and encode
        seq_bytes = np.frombuffer(seq.upper().encode(), dtype=np.uint8)
        encoded = base_map[seq_bytes].astype(np.int32)

        # Find valid positions (no N or other invalid bases)
        valid = encoded >= 0

        if len(seq_bytes) >= k:
            n_pos = len(seq_bytes) - k + 1
            kmer_indices = np.zeros(n_pos, dtype=np.int32)
            all_valid = np.ones(n_pos, dtype=bool)

            for j in range(k):
                pos_encoded = encoded[j:j+n_pos]
                all_valid &= (pos_encoded >= 0)
                kmer_indices += np.maximum(pos_encoded, 0) * powers[j]

            valid_indices = kmer_indices[all_valid]
            if len(valid_indices) > 0:
                counts = np.bincount(valid_indices, minlength=n_kmers)
                total = counts.sum()
                if total > 0:
                    features[i] = counts / total

    return features


def kmer_index_to_sequence(idx: int, k: int) -> str:
    """Convert k-mer index back to sequence."""
    bases = 'ACGT'
    seq = ''
    for _ in range(k):
        seq = bases[idx % 4] + seq
        idx //= 4
    return seq


def reverse_complement(seq: str) -> str:
    """Get reverse complement of a DNA sequence."""
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    return ''.join(complement[b] for b in reversed(seq))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(phenotype: str, seq_len: int = 100000) -> Tuple[List[str], List[int], List[str]]:
    """Load sequences and labels."""
    metadata_df = read_metadata_table(METADATA_XLSX)
    phenotype_col = phenotype

    train_dirs = [str(DATA_ROOT / "train")]
    exts = ('.fna', '.fasta', '.fa')

    labels_map, class_names = build_labels_map_and_classes(
        metadata_df, phenotype_col, file_col="Fasta file", train_dirs=train_dirs
    )

    sequences = []
    labels = []
    filenames = []

    data_dir = DATA_ROOT / "train"
    files = [f for f in os.listdir(data_dir) if f.endswith(exts)]

    for fname in tqdm(files, desc="Loading sequences"):
        # Labels map uses lowercase filename (with extension)
        label = labels_map.get(fname.strip().lower())
        if label is None:
            continue

        fpath = data_dir / fname
        with open(fpath) as f:
            seq = ''.join(line.strip() for line in f if not line.startswith('>'))
        seq = seq[:seq_len]

        sequences.append(seq)
        labels.append(label)
        filenames.append(fname)

    print(f"Loaded {len(sequences)} sequences for phenotype '{phenotype}'")
    print(f"Class distribution: {Counter(labels)}")

    return sequences, labels, filenames


# ---------------------------------------------------------------------------
# Rashomon analysis
# ---------------------------------------------------------------------------

def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    C: float = 1.0,
    penalty: str = 'l2',
    seed: int = 42,
) -> Tuple[float, np.ndarray]:
    """Train logistic regression and return balanced accuracy and coefficients."""
    model = LogisticRegression(
        C=C,
        penalty=penalty,
        solver='saga' if penalty == 'l1' else 'lbfgs',
        max_iter=1000,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    acc = balanced_accuracy_score(y_val, y_pred)
    coef = model.coef_.flatten()
    return acc, coef


def run_rashomon_analysis(
    sequences: List[str],
    labels: List[int],
    k: int,
    n_models: int = 100,
    epsilon: float = 0.05,
    C_values: List[float] = [2.0, 5.0, 10.0, 20.0],  # Higher C for less regularization
    penalties: List[str] = ['l1', 'l2'],
) -> Dict:
    """Run Rashomon analysis for k-mer logistic regression."""

    print(f"\n{'='*60}")
    print(f"K-MER RASHOMON ANALYSIS: k={k}")
    print(f"{'='*60}")

    # Extract k-mer features
    print(f"\nExtracting {k}-mer features...")
    X = extract_kmer_features_fast(sequences, k)
    y = np.array(labels)
    print(f"Feature matrix shape: {X.shape}")

    # Generate k-mer names
    kmer_names = generate_all_kmers(k)

    # Store all model results
    all_results = []

    # Train models with different configurations
    print(f"\nTraining {n_models} models per configuration...")

    for C in C_values:
        for penalty in penalties:
            if penalty == 'l1' and C > 10:
                continue  # Skip L1 with very high C (numerical issues)

            for seed in tqdm(range(n_models), desc=f"C={C}, {penalty}"):
                # Different train/val splits per seed
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=0.18, random_state=seed, stratify=y
                )

                acc, coef = train_logistic_regression(
                    X_train, y_train, X_val, y_val,
                    C=C, penalty=penalty, seed=seed
                )

                all_results.append({
                    'seed': seed,
                    'C': C,
                    'penalty': penalty,
                    'accuracy': acc,
                    'coefficients': coef,
                })

    # Find best accuracy and Rashomon set
    accuracies = [r['accuracy'] for r in all_results]
    best_acc = max(accuracies)
    threshold = best_acc - epsilon

    rashomon_set = [r for r in all_results if r['accuracy'] >= threshold]

    print(f"\nBest accuracy: {best_acc:.4f}")
    print(f"Rashomon threshold (ε={epsilon}): {threshold:.4f}")
    print(f"Models in Rashomon set: {len(rashomon_set)} / {len(all_results)}")

    # Analyze coefficient importance across Rashomon set
    coef_matrix = np.array([r['coefficients'] for r in rashomon_set])

    # Compute importance metrics
    mean_coef = np.mean(coef_matrix, axis=0)
    std_coef = np.std(coef_matrix, axis=0)
    mean_abs_coef = np.mean(np.abs(coef_matrix), axis=0)

    # Sign consistency: fraction of models with same sign
    sign_consistency = np.mean(np.sign(coef_matrix) == np.sign(mean_coef), axis=0)

    # Nonzero frequency (for L1 models)
    nonzero_freq = np.mean(np.abs(coef_matrix) > 1e-6, axis=0)

    # Rank k-mers by importance (mean absolute coefficient)
    importance_order = np.argsort(-mean_abs_coef)

    # Identify "necessary" k-mers (nonzero in 100% of models) and "common" (≥50%)
    necessary_kmers = []
    common_kmers = []

    for idx in importance_order:
        kmer = kmer_names[idx]
        freq = nonzero_freq[idx]
        if freq >= 0.99:  # Allow small tolerance
            necessary_kmers.append((kmer, mean_coef[idx], freq))
        if freq >= 0.50:
            common_kmers.append((kmer, mean_coef[idx], freq))

    print(f"\nNecessary k-mers (nonzero in ≥99% of models): {len(necessary_kmers)}")
    print(f"Common k-mers (nonzero in ≥50% of models): {len(common_kmers)}")

    # Top 20 most important k-mers
    print(f"\nTop 20 most important {k}-mers (by mean |coef|):")
    print(f"{'Rank':<6}{'K-mer':<10}{'Mean Coef':<12}{'Std':<10}{'Sign Cons.':<12}{'Nonzero %':<10}")
    print("-" * 60)

    for rank, idx in enumerate(importance_order[:20], 1):
        kmer = kmer_names[idx]
        rc = reverse_complement(kmer)
        kmer_display = f"{kmer}" if kmer == rc else f"{kmer}/{rc}"
        print(f"{rank:<6}{kmer_display:<10}{mean_coef[idx]:>+10.4f}{std_coef[idx]:>10.4f}{sign_consistency[idx]:>10.1%}{nonzero_freq[idx]:>10.1%}")

    # Separate positive and negative predictors
    positive_predictors = [(kmer_names[i], mean_coef[i]) for i in importance_order if mean_coef[i] > 0][:10]
    negative_predictors = [(kmer_names[i], mean_coef[i]) for i in importance_order if mean_coef[i] < 0][:10]

    print(f"\nTop 10 positive predictors (sporulating):")
    for kmer, coef in positive_predictors:
        print(f"  {kmer}: {coef:+.4f}")

    print(f"\nTop 10 negative predictors (non-sporulating):")
    for kmer, coef in negative_predictors:
        print(f"  {kmer}: {coef:+.4f}")

    # Compute GC content of top k-mers
    def gc_content(kmer):
        return sum(1 for b in kmer if b in 'GC') / len(kmer)

    top_positive_gc = np.mean([gc_content(kmer) for kmer, _ in positive_predictors])
    top_negative_gc = np.mean([gc_content(kmer) for kmer, _ in negative_predictors])

    print(f"\nGC content of top positive predictors: {top_positive_gc:.1%}")
    print(f"GC content of top negative predictors: {top_negative_gc:.1%}")

    # Prepare results
    results = {
        'k': k,
        'n_models_total': len(all_results),
        'n_models_rashomon': len(rashomon_set),
        'best_accuracy': best_acc,
        'rashomon_threshold': threshold,
        'epsilon': epsilon,
        'n_necessary_kmers': len(necessary_kmers),
        'n_common_kmers': len(common_kmers),
        'top_20_kmers': [
            {
                'kmer': kmer_names[idx],
                'mean_coef': float(mean_coef[idx]),
                'std_coef': float(std_coef[idx]),
                'sign_consistency': float(sign_consistency[idx]),
                'nonzero_freq': float(nonzero_freq[idx]),
            }
            for idx in importance_order[:20]
        ],
        'necessary_kmers': [
            {'kmer': kmer, 'mean_coef': float(coef), 'freq': float(freq)}
            for kmer, coef, freq in necessary_kmers[:50]  # Limit output
        ],
        'common_kmers': [
            {'kmer': kmer, 'mean_coef': float(coef), 'freq': float(freq)}
            for kmer, coef, freq in common_kmers[:100]
        ],
        'positive_predictors': [
            {'kmer': kmer, 'mean_coef': float(coef)}
            for kmer, coef in positive_predictors
        ],
        'negative_predictors': [
            {'kmer': kmer, 'mean_coef': float(coef)}
            for kmer, coef in negative_predictors
        ],
        'gc_content_positive': top_positive_gc,
        'gc_content_negative': top_negative_gc,
        'accuracy_distribution': {
            'mean': float(np.mean(accuracies)),
            'std': float(np.std(accuracies)),
            'min': float(np.min(accuracies)),
            'max': float(np.max(accuracies)),
        },
    }

    return results, coef_matrix, kmer_names


def plot_coefficient_heatmap(
    coef_matrix: np.ndarray,
    kmer_names: List[str],
    k: int,
    output_path: Path,
    top_n: int = 50,
    min_coef_threshold: float = 1e-4,
):
    """Plot heatmap of top k-mer coefficients across Rashomon models.

    Filters out k-mers with near-zero coefficients to prevent overflow in
    high-k analyses where most k-mers have negligible importance.
    """
    # Compute mean absolute coefficient for each k-mer
    mean_abs = np.mean(np.abs(coef_matrix), axis=0)

    # Filter out k-mers with effectively zero coefficients
    nonzero_mask = mean_abs > min_coef_threshold
    n_nonzero = nonzero_mask.sum()

    if n_nonzero == 0:
        print(f"Warning: No k-mers with |coef| > {min_coef_threshold}, skipping heatmap")
        return

    # Select top k-mers from those with nonzero coefficients
    valid_indices = np.where(nonzero_mask)[0]
    valid_mean_abs = mean_abs[nonzero_mask]
    n_to_show = min(top_n, len(valid_indices))
    top_valid_idx = np.argsort(-valid_mean_abs)[:n_to_show]
    top_indices = valid_indices[top_valid_idx]

    coef_subset = coef_matrix[:, top_indices]
    kmer_subset = [kmer_names[i] for i in top_indices]

    fig, ax = plt.subplots(figsize=(14, 10))

    # Limit to first 50 models for visibility
    n_models_show = min(50, coef_subset.shape[0])

    # Use robust percentile for color limits
    coef_flat = coef_subset[:n_models_show].flatten()
    vmax = np.percentile(np.abs(coef_flat), 95)
    if vmax < 1e-6:
        vmax = np.max(np.abs(coef_flat)) + 1e-6  # Fallback for very small values

    im = ax.imshow(
        coef_subset[:n_models_show].T,
        aspect='auto',
        cmap='RdBu_r',
        vmin=-vmax,
        vmax=vmax,
    )

    ax.set_xlabel('Model index')
    ax.set_ylabel(f'{k}-mer')
    ax.set_yticks(range(len(kmer_subset)))
    ax.set_yticklabels(kmer_subset, fontsize=8)
    ax.set_title(f'Top {n_to_show} {k}-mer coefficients across Rashomon set\n(first {n_models_show} models, {n_nonzero}/{len(mean_abs)} k-mers with |coef| > {min_coef_threshold})')

    plt.colorbar(im, ax=ax, label='Coefficient')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved coefficient heatmap to {output_path}")


def plot_accuracy_distribution(
    all_results: List[Dict],
    k: int,
    epsilon: float,
    output_path: Path,
):
    """Plot accuracy distribution with Rashomon threshold."""
    accuracies = [r['accuracy'] for r in all_results]
    best_acc = max(accuracies)
    threshold = best_acc - epsilon

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(accuracies, bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Rashomon threshold (ε={epsilon})')
    ax.axvline(best_acc, color='green', linestyle='-', linewidth=2, label=f'Best accuracy')

    ax.set_xlabel('Balanced Accuracy')
    ax.set_ylabel('Count')
    ax.set_title(f'{k}-mer Logistic Regression: Accuracy Distribution')
    ax.legend()

    # Add text annotation
    n_rashomon = sum(1 for a in accuracies if a >= threshold)
    ax.text(0.05, 0.95, f'Total models: {len(accuracies)}\nRashomon set: {n_rashomon}',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved accuracy distribution to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="K-mer Logistic Regression Rashomon Analysis")
    parser.add_argument("phenotype", type=str, help="Phenotype name (e.g., 'Spore formation')")
    parser.add_argument("--k", type=int, nargs='+', default=[3, 4, 6], help="K-mer sizes")
    parser.add_argument("--n_models", type=int, default=100, help="Number of models per configuration")
    parser.add_argument("--epsilon", type=float, default=0.05, help="Rashomon epsilon threshold")
    parser.add_argument("--output_dir", type=str, default="sporulation/reports", help="Output directory")
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    sequences, labels, filenames = load_data(args.phenotype)

    # Phenotype slug for filenames
    phenotype_slug = args.phenotype.lower().replace(" ", "_")

    # Run analysis for each k
    all_k_results = {}

    for k in args.k:
        results, coef_matrix, kmer_names = run_rashomon_analysis(
            sequences, labels, k,
            n_models=args.n_models,
            epsilon=args.epsilon,
        )
        all_k_results[k] = results

        # Plot coefficient heatmap
        plot_coefficient_heatmap(
            coef_matrix, kmer_names, k,
            output_dir / f"rashomon_kmer_{phenotype_slug}_k{k}_coefficients.png",
        )

    # Save results - per-k files for parallel safety, plus combined file
    for k in args.k:
        per_k_file = output_dir / f"rashomon_kmer_{phenotype_slug}_k{k}_results.json"
        with open(per_k_file, 'w') as f:
            json.dump({str(k): all_k_results[k]}, f, indent=2)
        print(f"Saved k={k} results to {per_k_file}")

    # Also save combined file (safe when running single k, or sequentially)
    output_file = output_dir / f"rashomon_kmer_{phenotype_slug}_results.json"
    with open(output_file, 'w') as f:
        json.dump({str(k): v for k, v in all_k_results.items()}, f, indent=2)
    print(f"Saved combined results to {output_file}")

    # Summary comparison
    print("\n" + "=" * 60)
    print("SUMMARY ACROSS K VALUES")
    print("=" * 60)
    print(f"\n{'k':<5}{'Best Acc':<12}{'Rashomon Size':<15}{'Necessary':<12}{'Common':<10}")
    print("-" * 54)
    for k in args.k:
        r = all_k_results[k]
        print(f"{k:<5}{r['best_accuracy']:<12.4f}{r['n_models_rashomon']:<15}{r['n_necessary_kmers']:<12}{r['n_common_kmers']:<10}")

    print("\nDone!")


if __name__ == "__main__":
    main()
