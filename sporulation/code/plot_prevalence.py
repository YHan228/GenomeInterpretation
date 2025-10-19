#!/usr/bin/env python3
"""Plot gene prevalence distribution on the training split.

Creates a single PNG with two panels:
- Left: density-style histogram of per-gene prevalence on training samples
- Right: empirical CDF (ECDF) of the same prevalence values

Prevalence is computed identically to the filtering step used by training scripts:
mean presence across training samples per feature.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

# Reuse dataset utilities for consistent split construction
from train_lasso import _read_table, build_dataset


def compute_training_prevalence(X_train: np.ndarray) -> np.ndarray:
    """Return per-feature prevalence on training data (fraction of samples with presence).

    Assumes X_train is a binary (0/1) or fractional presence matrix with shape [n_samples, n_features].
    """
    if X_train.size == 0:
        return np.array([], dtype=float)
    # Mean across samples gives prevalence per feature
    return np.asarray(X_train.mean(axis=0), dtype=float)


def plot_density_and_ecdf(
    prevalence: np.ndarray,
    min_prev: float,
    out_path: Path,
    phenotype: str,
    bins: int,
) -> None:
    """Save a two-panel figure: density histogram and ECDF of prevalence.

    - prevalence: array of per-feature prevalence values in [0,1]
    - min_prev: threshold used by filtering (annotated as vertical line)
    - out_path: file path to save the figure (PNG)
    - phenotype: used for title annotation
    - bins: number of histogram bins
    """
    values = np.asarray(prevalence, dtype=float)
    n_features = int(values.size)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    # Panel A: density-style histogram
    ax = axes[0]
    if n_features > 0:
        ax.hist(values, bins=int(bins), range=(0.0, 1.0), color="#2c7fb8", alpha=0.9, edgecolor="black", density=True)
    else:
        ax.text(0.5, 0.5, "No features", ha="center", va="center", transform=ax.transAxes)
    ax.axvline(float(min_prev), color="red", linestyle="--", linewidth=1.2, label=f"min_prev={float(min_prev):.3f}")
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Training prevalence (fraction)")
    ax.set_ylabel("Density")
    ax.set_title("Prevalence density")
    if n_features > 0:
        frac_below = float(np.mean(values < float(min_prev)))
        ax.legend(title=f"< min_prev: {frac_below*100:.1f}%")
    ax.grid(True, axis="y", alpha=0.3)

    # Panel B: ECDF
    ax = axes[1]
    if n_features > 0:
        xs = np.sort(values)
        ys = np.arange(1, n_features + 1, dtype=float) / float(n_features)
        ax.step(xs, ys, where="post", color="#1b9e77", linewidth=1.6)
    else:
        ax.text(0.5, 0.5, "No features", ha="center", va="center", transform=ax.transAxes)
    ax.axvline(float(min_prev), color="red", linestyle="--", linewidth=1.2)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Training prevalence (fraction)")
    ax.set_ylabel("ECDF")
    ax.set_title("Prevalence ECDF")
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Gene prevalence on training split — {phenotype}")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot prevalence density and ECDF for training features")
    parser.add_argument("--input_dir", type=Path, default=Path("/vol/projects/BIFO/genomenet/yichen/phenotype/data/rfdata"))
    parser.add_argument("--output_dir", type=Path, default=Path("sporulation/results/prevalence"))
    parser.add_argument("--phenotype", type=str, default="Spore formation")
    parser.add_argument("--min_prev", type=float, default=0.02)
    parser.add_argument("--bins", type=int, default=50)
    args = parser.parse_args()

    input_dir: Path = args.input_dir
    out_root: Path = args.output_dir
    phen_name = str(args.phenotype).strip()
    phen_safe = phen_name.replace(" ", "_")
    out_dir = out_root / phen_safe
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    pa_path = input_dir / "rf_presence_absence.parquet"
    long_path = input_dir / "rf_dataset.parquet"
    pa_df = _read_table(pa_path)
    long_df = _read_table(long_path)

    # Build dataset (no prevalence filtering here; we want raw train prevalence)
    splits = build_dataset(pa_df, long_df, phenotype=phen_name)
    prevalence = compute_training_prevalence(splits.X_train)

    out_path = out_dir / "prevalence_density_ecdf.png"
    plot_density_and_ecdf(prevalence, float(args.min_prev), out_path, phen_name, int(args.bins))

    print("Saved figure:", str(out_path), flush=True)


if __name__ == "__main__":
    main()

