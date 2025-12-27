#!/usr/bin/env python3
"""Post-hoc re-clustering of Rashomon CNN filters with different similarity thresholds.

Re-uses saved filter weights (NPZ files) to explore how clustering threshold
affects the analysis without retraining models.

Usage:
    python sporulation/code/recluster_rashomon_cnn.py --threshold 0.5
    python sporulation/code/recluster_rashomon_cnn.py --threshold 0.5 0.6 0.7 0.8
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# PWM Utilities (copied from rashomon_cnn.py for standalone operation)
# ---------------------------------------------------------------------------

def filter_to_pwm(weights: np.ndarray) -> np.ndarray:
    """Convert filter weights to PWM (position weight matrix)."""
    exp_weights = np.exp(weights - weights.max(axis=1, keepdims=True))
    pwm = exp_weights / exp_weights.sum(axis=1, keepdims=True)
    return pwm


def pwm_to_ic(pwm: np.ndarray, pseudocount: float = 0.01) -> np.ndarray:
    """Convert PWM to information content per position."""
    pwm_safe = np.clip(pwm, pseudocount, 1.0 - pseudocount)
    pwm_safe = pwm_safe / pwm_safe.sum(axis=1, keepdims=True)
    ic = 2.0 + np.sum(pwm_safe * np.log2(pwm_safe), axis=1)
    return np.clip(ic, 0, 2)


def pwm_reverse_complement(pwm: np.ndarray) -> np.ndarray:
    """Reverse complement a PWM. Assumes columns are A, C, G, T."""
    return pwm[::-1, ::-1].copy()


def pwm_to_consensus(pwm: np.ndarray, threshold: float = 0.5) -> str:
    """Convert PWM to consensus sequence."""
    bases = "ACGT"
    consensus = []
    for pos in range(pwm.shape[0]):
        idx = pwm[pos].argmax()
        prob = pwm[pos, idx]
        base = bases[idx]
        consensus.append(base if prob >= threshold else base.lower())
    return "".join(consensus)


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def compute_pairwise_similarities(pwms: np.ndarray, check_revcomp: bool = True) -> np.ndarray:
    """Compute IC-weighted pairwise similarities between PWMs."""
    n = len(pwms)
    sim_matrix = np.eye(n, dtype=np.float32)

    # Pre-compute ICs
    all_ics = np.array([pwm_to_ic(p) for p in pwms])

    if check_revcomp:
        pwms_rc = np.array([pwm_reverse_complement(p) for p in pwms])
        all_ics_rc = np.array([pwm_to_ic(p) for p in pwms_rc])

    for i in tqdm(range(n), desc="Computing similarities", leave=False, miniters=max(1, n//100)):
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
                    flat1 = pwms[i].flatten()
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


def cluster_with_threshold(
    sim_matrix: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Cluster using precomputed similarity matrix with given threshold."""
    dist_matrix = 1.0 - sim_matrix
    np.fill_diagonal(dist_matrix, 0)

    condensed = pdist(dist_matrix)
    if np.any(np.isnan(condensed)):
        condensed = np.nan_to_num(condensed, nan=1.0)

    Z = linkage(condensed, method="average")
    cluster_ids = fcluster(Z, t=1.0 - threshold, criterion="distance")
    return cluster_ids


def compute_cluster_frequencies(
    cluster_ids: np.ndarray,
    n_models: int,
    n_filters: int,
) -> Dict[int, float]:
    """Compute frequency of each filter cluster across Rashomon models."""
    # filter_indices = (model_idx, filter_idx) for each element
    cluster_model_presence = {}

    for idx, cluster_id in enumerate(cluster_ids):
        model_idx = idx // n_filters
        if cluster_id not in cluster_model_presence:
            cluster_model_presence[cluster_id] = set()
        cluster_model_presence[cluster_id].add(model_idx)

    frequencies = {}
    for cluster_id, model_set in cluster_model_presence.items():
        frequencies[cluster_id] = len(model_set) / n_models

    return frequencies


def get_cluster_stats(
    cluster_ids: np.ndarray,
    all_pwms: np.ndarray,
    filter_importance: np.ndarray,
    frequencies: Dict[int, float],
    n_models: int,
    n_filters: int,
) -> pd.DataFrame:
    """Get statistics for each cluster."""
    records = []

    for cluster_id in sorted(set(cluster_ids)):
        mask = cluster_ids == cluster_id
        indices = np.where(mask)[0]

        # Find best representative (highest importance)
        best_importance = -1.0
        best_pwm = None

        for idx in indices:
            model_idx = idx // n_filters
            filter_idx = idx % n_filters
            importance = filter_importance[model_idx, filter_idx]
            if importance > best_importance:
                best_importance = importance
                best_pwm = all_pwms[idx]

        if best_pwm is not None:
            consensus = pwm_to_consensus(best_pwm)
            avg_ic = float(pwm_to_ic(best_pwm).mean())
        else:
            consensus = ""
            avg_ic = 0.0

        records.append({
            "cluster_id": cluster_id,
            "frequency": frequencies.get(cluster_id, 0.0),
            "n_filters": int(mask.sum()),
            "consensus": consensus,
            "avg_ic": avg_ic,
            "importance": best_importance,
        })

    df = pd.DataFrame(records)
    return df.sort_values("frequency", ascending=False)


# ---------------------------------------------------------------------------
# Main Analysis
# ---------------------------------------------------------------------------

@dataclass
class RecluseterResult:
    """Results from re-clustering with a specific threshold."""
    threshold: float
    n_clusters: int
    n_necessary: int  # freq = 100%
    n_common: int     # freq >= 50%
    max_frequency: float
    top_motifs: List[Dict]
    cluster_df: pd.DataFrame


def recluster_npz(
    npz_path: Path,
    thresholds: List[float],
    check_revcomp: bool = True,
) -> Dict[float, RecluseterResult]:
    """Re-cluster filter weights from NPZ with multiple thresholds."""

    print(f"\nLoading: {npz_path}")
    data = np.load(npz_path)

    filter_weights = data["filter_weights"]  # (n_models, n_filters, kernel_size, 4)
    filter_importance = data["filter_importance"]  # (n_models, n_filters)
    activation = str(data["activation"])

    n_models, n_filters, kernel_size, _ = filter_weights.shape
    print(f"  Models: {n_models}, Filters/model: {n_filters}, Kernel: {kernel_size}, Activation: {activation}")

    # Convert all filters to PWMs
    all_pwms = []
    for model_idx in range(n_models):
        for filter_idx in range(n_filters):
            pwm = filter_to_pwm(filter_weights[model_idx, filter_idx])
            all_pwms.append(pwm)
    all_pwms = np.array(all_pwms)

    n_total = len(all_pwms)
    print(f"  Total filters: {n_total}")

    # Compute similarity matrix once (expensive)
    print("  Computing pairwise similarities...")
    sim_matrix = compute_pairwise_similarities(all_pwms, check_revcomp=check_revcomp)

    # Cluster at each threshold
    results = {}
    for thresh in thresholds:
        print(f"\n  Threshold {thresh}:")
        cluster_ids = cluster_with_threshold(sim_matrix, thresh)
        frequencies = compute_cluster_frequencies(cluster_ids, n_models, n_filters)

        cluster_df = get_cluster_stats(
            cluster_ids, all_pwms, filter_importance, frequencies, n_models, n_filters
        )

        n_clusters = len(set(cluster_ids))
        n_necessary = int((cluster_df["frequency"] >= 1.0).sum())
        n_common = int((cluster_df["frequency"] >= 0.5).sum())
        max_freq = float(cluster_df["frequency"].max()) if len(cluster_df) > 0 else 0.0

        # Top motifs
        top_motifs = []
        for _, row in cluster_df.head(5).iterrows():
            top_motifs.append({
                "consensus": row["consensus"],
                "frequency": row["frequency"],
                "avg_ic": row["avg_ic"],
            })

        print(f"    Clusters: {n_clusters}")
        print(f"    Necessary (100%): {n_necessary}")
        print(f"    Common (≥50%): {n_common}")
        print(f"    Max frequency: {max_freq*100:.1f}%")

        results[thresh] = RecluseterResult(
            threshold=thresh,
            n_clusters=n_clusters,
            n_necessary=n_necessary,
            n_common=n_common,
            max_frequency=max_freq,
            top_motifs=top_motifs,
            cluster_df=cluster_df,
        )

    return results


def plot_threshold_comparison(
    all_results: Dict[str, Dict[float, RecluseterResult]],
    out_dir: Path,
    phenotype: str,
):
    """Create comparison plot across thresholds."""

    configs = list(all_results.keys())
    thresholds = sorted(list(all_results[configs[0]].keys()))

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    colors = plt.cm.tab10(np.linspace(0, 1, len(configs)))

    # Panel A: Number of clusters
    ax = axes[0, 0]
    for i, config in enumerate(configs):
        vals = [all_results[config][t].n_clusters for t in thresholds]
        ax.plot(thresholds, vals, 'o-', label=config, color=colors[i], linewidth=2, markersize=6)
    ax.set_xlabel("Similarity Threshold", fontsize=10)
    ax.set_ylabel("Number of Clusters", fontsize=10)
    ax.set_title("A. Cluster Count vs Threshold", fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, frameon=False, loc='best')
    ax.grid(True, alpha=0.3)

    # Panel B: Max frequency
    ax = axes[0, 1]
    for i, config in enumerate(configs):
        vals = [all_results[config][t].max_frequency * 100 for t in thresholds]
        ax.plot(thresholds, vals, 'o-', label=config, color=colors[i], linewidth=2, markersize=6)
    ax.axhline(50, color="red", linestyle="--", alpha=0.7, linewidth=1, label="Common threshold")
    ax.axhline(100, color="darkred", linestyle="--", alpha=0.5, linewidth=1)
    ax.set_xlabel("Similarity Threshold", fontsize=10)
    ax.set_ylabel("Max Motif Frequency (%)", fontsize=10)
    ax.set_title("B. Max Frequency vs Threshold", fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, frameon=False, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)

    # Panel C: Common motifs (>=50%)
    ax = axes[1, 0]
    for i, config in enumerate(configs):
        vals = [all_results[config][t].n_common for t in thresholds]
        ax.plot(thresholds, vals, 'o-', label=config, color=colors[i], linewidth=2, markersize=6)
    ax.set_xlabel("Similarity Threshold", fontsize=10)
    ax.set_ylabel("Number of Common Motifs (≥50%)", fontsize=10)
    ax.set_title("C. Common Motifs vs Threshold", fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, frameon=False, loc='best')
    ax.grid(True, alpha=0.3)

    # Panel D: Necessary motifs (100%)
    ax = axes[1, 1]
    for i, config in enumerate(configs):
        vals = [all_results[config][t].n_necessary for t in thresholds]
        ax.plot(thresholds, vals, 'o-', label=config, color=colors[i], linewidth=2, markersize=6)
    ax.set_xlabel("Similarity Threshold", fontsize=10)
    ax.set_ylabel("Number of Necessary Motifs (100%)", fontsize=10)
    ax.set_title("D. Necessary Motifs vs Threshold", fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, frameon=False, loc='best')
    ax.grid(True, alpha=0.3)

    plt.suptitle(f"Re-clustering Analysis: {phenotype}\n(Effect of Similarity Threshold)",
                 fontsize=12, fontweight='bold')
    plt.tight_layout()

    for ext in (".png", ".pdf", ".svg"):
        fig.savefig(out_dir / f"recluster_threshold_comparison{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved threshold comparison to {out_dir}")


def print_summary_table(all_results: Dict[str, Dict[float, RecluseterResult]]):
    """Print summary table."""

    print("\n" + "=" * 80)
    print("RE-CLUSTERING SUMMARY")
    print("=" * 80)

    configs = list(all_results.keys())
    thresholds = sorted(list(all_results[configs[0]].keys()))

    for config in configs:
        print(f"\n{config}:")
        print("-" * 60)
        print(f"{'Threshold':>10} {'Clusters':>10} {'MaxFreq%':>10} {'Common':>10} {'Necessary':>10}")
        print("-" * 60)

        for thresh in thresholds:
            r = all_results[config][thresh]
            print(f"{thresh:>10.2f} {r.n_clusters:>10d} {r.max_frequency*100:>10.1f} "
                  f"{r.n_common:>10d} {r.n_necessary:>10d}")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Re-cluster Rashomon CNN filters")
    parser.add_argument("--results_dir", type=str,
                        default="/home/yhan/GenomeInterpretation/sporulation/results/rashomon_cnn_multiarch",
                        help="Directory with Rashomon CNN results")
    parser.add_argument("--threshold", type=float, nargs="+",
                        default=[0.4, 0.5, 0.6, 0.7, 0.8],
                        help="Similarity thresholds to try")
    parser.add_argument("--phenotype", type=str, default="Spore_formation",
                        help="Phenotype subdirectory to analyze")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    phen_dir = results_dir / args.phenotype

    if not phen_dir.exists():
        print(f"Directory not found: {phen_dir}")
        return

    thresholds = sorted(args.threshold)
    print(f"Testing thresholds: {thresholds}")

    # Find all NPZ files (one per kernel size per activation)
    all_results = {}

    for k_dir in sorted(phen_dir.glob("k*")):
        if not k_dir.is_dir():
            continue

        for npz_file in k_dir.glob("filter_weights_*.npz"):
            activation = npz_file.stem.split("_")[-1]
            kernel = k_dir.name
            config_name = f"{kernel}_{activation}"

            results = recluster_npz(npz_file, thresholds, check_revcomp=True)
            all_results[config_name] = results

            # Save detailed CSV for each threshold
            for thresh, r in results.items():
                csv_path = k_dir / f"recluster_{activation}_t{thresh:.2f}.csv"
                r.cluster_df.to_csv(csv_path, index=False)

    if not all_results:
        print("No filter weight files found!")
        return

    # Print summary
    print_summary_table(all_results)

    # Plot comparison
    plot_threshold_comparison(all_results, phen_dir, args.phenotype.replace("_", " "))

    # Save summary JSON
    summary = {}
    for config, results in all_results.items():
        summary[config] = {}
        for thresh, r in results.items():
            summary[config][str(thresh)] = {
                "n_clusters": r.n_clusters,
                "n_necessary": r.n_necessary,
                "n_common": r.n_common,
                "max_frequency": r.max_frequency,
                "top_motifs": r.top_motifs,
            }

    with open(phen_dir / "recluster_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved summary to {phen_dir / 'recluster_summary.json'}")


if __name__ == "__main__":
    main()
