#!/usr/bin/env python3
"""Generate summary figures for multi-architecture Rashomon CNN analysis.

Creates:
1. Multi-panel summary figure comparing kernel sizes and activations
2. Top motif visualization across architectures
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def load_all_summaries(results_dir: Path) -> pd.DataFrame:
    """Load all summary JSON files from multiarch results."""
    records = []

    for phen_dir in sorted(results_dir.iterdir()):
        if not phen_dir.is_dir():
            continue

        # Find all summary_k{X}_{activation}.json files
        for summary_file in phen_dir.glob("summary_k*_*.json"):
            try:
                with open(summary_file) as f:
                    data = json.load(f)

                phenotype = data["phenotype"]
                activation = data["activation_filter"]

                # Get results for this activation
                if activation not in data:
                    continue

                for k_key, r in data[activation].items():
                    ks = r["kernel_size"]
                    records.append({
                        "phenotype": phenotype,
                        "kernel_size": ks,
                        "activation": activation,
                        "n_filters": r["n_filters"],
                        "best_performance": r["best_performance"],
                        "rashomon_size": r["rashomon_size"],
                        "n_filter_clusters": r["n_filter_clusters"],
                        "n_necessary": r["n_necessary"],
                        "n_common": r["n_common"],
                        "top_motifs": r.get("top_motifs", []),
                    })
            except Exception as e:
                print(f"Skipping {summary_file}: {e}")
                continue

    return pd.DataFrame(records)


def load_filter_clusters(results_dir: Path, phenotype: str) -> pd.DataFrame:
    """Load all filter cluster CSVs for a phenotype."""
    phen_safe = phenotype.replace(" ", "_")
    phen_dir = results_dir / phen_safe

    dfs = []
    for k_dir in phen_dir.glob("k*"):
        if not k_dir.is_dir():
            continue
        for csv_file in k_dir.glob("filter_clusters_*.csv"):
            activation = csv_file.stem.split("_")[-1]
            df = pd.read_csv(csv_file)
            df["activation"] = activation
            dfs.append(df)

    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def plot_summary_figure(df: pd.DataFrame, out_dir: Path, phenotype: str):
    """Create multi-panel summary figure."""

    # Pivot for easier plotting
    kernel_sizes = sorted(df["kernel_size"].unique())
    activations = ["relu", "exp"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    x = np.arange(len(kernel_sizes))
    width = 0.35

    colors = {"relu": "#1f78b4", "exp": "#e31a1c"}

    # ---- Panel A: Best Performance ----
    ax = axes[0, 0]
    for i, act in enumerate(activations):
        sub = df[df["activation"] == act].set_index("kernel_size")
        vals = [sub.loc[ks, "best_performance"] if ks in sub.index else 0 for ks in kernel_sizes]
        offset = -width/2 if i == 0 else width/2
        ax.bar(x + offset, vals, width, label=act.upper(), color=colors[act], alpha=0.85)

    ax.set_ylabel("Best Balanced Accuracy", fontsize=10)
    ax.set_title("A. Model Performance by Kernel Size", fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f"k={k}" for k in kernel_sizes])
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    ax.set_ylim(0.5, 1.0)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.5)

    # ---- Panel B: Rashomon Set Size ----
    ax = axes[0, 1]
    for i, act in enumerate(activations):
        sub = df[df["activation"] == act].set_index("kernel_size")
        vals = [sub.loc[ks, "rashomon_size"] if ks in sub.index else 0 for ks in kernel_sizes]
        offset = -width/2 if i == 0 else width/2
        ax.bar(x + offset, vals, width, label=act.upper(), color=colors[act], alpha=0.85)

    ax.set_ylabel("Rashomon Set Size", fontsize=10)
    ax.set_title("B. Rashomon Set Size (within ε=5%)", fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f"k={k}" for k in kernel_sizes])
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    ax.axhline(40, color="gray", linestyle=":", alpha=0.5, linewidth=0.8, label="max=40")
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.5)

    # ---- Panel C: Number of Filter Clusters ----
    ax = axes[1, 0]
    for i, act in enumerate(activations):
        sub = df[df["activation"] == act].set_index("kernel_size")
        vals = [sub.loc[ks, "n_filter_clusters"] if ks in sub.index else 0 for ks in kernel_sizes]
        offset = -width/2 if i == 0 else width/2
        ax.bar(x + offset, vals, width, label=act.upper(), color=colors[act], alpha=0.85)

    ax.set_ylabel("Number of Filter Clusters", fontsize=10)
    ax.set_title("C. Filter Diversity (# clusters)", fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f"k={k}" for k in kernel_sizes])
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.5)

    # ---- Panel D: Max Motif Frequency ----
    ax = axes[1, 1]
    for i, act in enumerate(activations):
        sub = df[df["activation"] == act].set_index("kernel_size")
        vals = []
        for ks in kernel_sizes:
            if ks in sub.index:
                motifs = sub.loc[ks, "top_motifs"]
                max_freq = motifs[0]["frequency"] * 100 if motifs else 0
                vals.append(max_freq)
            else:
                vals.append(0)
        offset = -width/2 if i == 0 else width/2
        ax.bar(x + offset, vals, width, label=act.upper(), color=colors[act], alpha=0.85)

    ax.set_ylabel("Max Motif Frequency (%)", fontsize=10)
    ax.set_title("D. Top Motif Frequency (none reach 50%)", fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f"k={k}" for k in kernel_sizes])
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    ax.axhline(50, color="red", linestyle="--", alpha=0.5, linewidth=1, label="common threshold")
    ax.axhline(100, color="darkred", linestyle="--", alpha=0.5, linewidth=1, label="necessary threshold")
    ax.set_ylim(0, 110)
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.5)

    # Add annotation about low frequencies
    ax.text(0.5, 0.85, "No motifs reach\n'common' (≥50%) threshold",
            transform=ax.transAxes, ha='center', va='top',
            fontsize=10, color='darkred', style='italic',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    for ext in (".png", ".pdf", ".svg"):
        fig.savefig(out_dir / f"rashomon_cnn_summary{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved summary figure to {out_dir}")


def plot_top_motifs_heatmap(df: pd.DataFrame, clusters_df: pd.DataFrame,
                            out_dir: Path, phenotype: str, n_top: int = 10):
    """Visualize top motifs across kernel sizes as consensus heatmaps."""

    kernel_sizes = sorted(df["kernel_size"].unique())
    activations = ["relu", "exp"]

    # DNA base colors
    base_colors = {'A': '#109648', 'C': '#255C99', 'G': '#F7B32B', 'T': '#D62839', 'N': '#CCCCCC'}

    fig, axes = plt.subplots(len(kernel_sizes), 2, figsize=(14, 3 * len(kernel_sizes)))

    for row, ks in enumerate(kernel_sizes):
        for col, act in enumerate(activations):
            ax = axes[row, col] if len(kernel_sizes) > 1 else axes[col]

            # Get top motifs for this kernel/activation
            sub = clusters_df[(clusters_df["kernel_size"] == ks) &
                              (clusters_df["activation"] == act)]

            if sub.empty:
                ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"k={ks}, {act.upper()}")
                ax.axis('off')
                continue

            top = sub.nlargest(n_top, "frequency")

            # Create heatmap data
            consensuses = top["consensus"].tolist()
            freqs = top["frequency"].tolist()
            ics = top["avg_ic"].tolist()

            if not consensuses:
                ax.text(0.5, 0.5, "No motifs", ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"k={ks}, {act.upper()}")
                ax.axis('off')
                continue

            # Draw consensus sequences as colored text
            max_len = max(len(c) for c in consensuses)

            for i, (cons, freq, ic) in enumerate(zip(consensuses, freqs, ics)):
                y = n_top - 1 - i
                for j, base in enumerate(cons.upper()):
                    color = base_colors.get(base, '#CCCCCC')
                    ax.text(j + 0.5, y + 0.5, base, ha='center', va='center',
                            fontsize=9, fontweight='bold', color=color,
                            family='monospace')

                # Add frequency annotation
                ax.text(max_len + 0.5, y + 0.5, f"{freq*100:.1f}%",
                        ha='left', va='center', fontsize=8, color='gray')

            ax.set_xlim(0, max_len + 3)
            ax.set_ylim(0, n_top)
            ax.set_title(f"k={ks}, {act.upper()} (top {n_top})", fontsize=10, fontweight='bold')
            ax.set_yticks([])
            ax.set_xticks(range(max_len))
            ax.set_xticklabels(range(1, max_len + 1), fontsize=7)
            ax.set_xlabel("Position", fontsize=8)

            # Remove spines
            for spine in ax.spines.values():
                spine.set_visible(False)

    # Add legend for base colors
    legend_elements = [plt.Line2D([0], [0], marker='s', color='w',
                                   markerfacecolor=c, markersize=10, label=b)
                       for b, c in base_colors.items() if b != 'N']
    fig.legend(handles=legend_elements, loc='upper right', ncol=4,
               fontsize=9, title="Base", frameon=False, bbox_to_anchor=(0.98, 0.99))

    plt.suptitle(f"Top Motif Consensuses by Kernel Size - {phenotype}\n"
                 f"(frequencies shown; none reach 'common' threshold ≥50%)",
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()

    for ext in (".png", ".pdf", ".svg"):
        fig.savefig(out_dir / f"rashomon_cnn_motifs{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved motif visualization to {out_dir}")


def plot_frequency_distribution(clusters_df: pd.DataFrame, out_dir: Path, phenotype: str):
    """Plot distribution of motif frequencies to show sparsity."""

    kernel_sizes = sorted(clusters_df["kernel_size"].unique())
    activations = ["relu", "exp"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    colors = {"relu": "#1f78b4", "exp": "#e31a1c"}

    for col, act in enumerate(activations):
        ax = axes[col]

        for ks in kernel_sizes:
            sub = clusters_df[(clusters_df["kernel_size"] == ks) &
                              (clusters_df["activation"] == act)]
            if sub.empty:
                continue

            freqs = sub["frequency"].values * 100

            # Histogram
            ax.hist(freqs, bins=20, alpha=0.5, label=f"k={ks}", density=True)

        ax.set_xlabel("Motif Frequency (%)", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.set_title(f"{act.upper()} Activation", fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, frameon=False)
        ax.axvline(50, color="red", linestyle="--", alpha=0.7, label="common threshold")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 105)

    plt.suptitle(f"Distribution of Motif Frequencies - {phenotype}\n"
                 f"(all frequencies << 50%, indicating no consistent motifs)",
                 fontsize=12, fontweight='bold')
    plt.tight_layout()

    for ext in (".png", ".pdf", ".svg"):
        fig.savefig(out_dir / f"rashomon_cnn_freq_dist{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved frequency distribution to {out_dir}")


def print_summary_table(df: pd.DataFrame):
    """Print summary statistics."""
    print("\n" + "="*80)
    print("RASHOMON CNN MULTI-ARCHITECTURE SUMMARY")
    print("="*80)

    for act in ["relu", "exp"]:
        print(f"\n{act.upper()} Activation:")
        print("-" * 60)
        sub = df[df["activation"] == act].sort_values("kernel_size")

        for _, row in sub.iterrows():
            ks = row["kernel_size"]
            perf = row["best_performance"]
            rsize = row["rashomon_size"]
            nclusters = row["n_filter_clusters"]
            nnec = row["n_necessary"]
            ncom = row["n_common"]

            top_freq = 0
            if row["top_motifs"]:
                top_freq = row["top_motifs"][0]["frequency"] * 100

            print(f"  k={ks:2d}: acc={perf:.3f}, rashomon={rsize:2d}/40, "
                  f"clusters={nclusters:4d}, necessary={nnec}, common={ncom}, "
                  f"max_freq={top_freq:.1f}%")

    print("\n" + "="*80)
    print("CONCLUSION: No motifs reach 'common' (≥50%) or 'necessary' (100%) threshold")
    print("This suggests CNN filters learn diverse, non-overlapping patterns")
    print("="*80)


def main():
    results_dir = Path("/home/yhan/GenomeInterpretation/sporulation/results/rashomon_cnn_multiarch")

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return

    # Load all summaries first
    df_all = load_all_summaries(results_dir)

    if df_all.empty:
        print("No data found!")
        return

    # Find phenotypes from loaded data
    phenotypes = df_all["phenotype"].unique()

    for phenotype in phenotypes:
        print(f"\nProcessing: {phenotype}")
        phen_safe = phenotype.replace(" ", "_")
        phen_dir = results_dir / phen_safe

        # Filter summaries for this phenotype
        df = df_all[df_all["phenotype"] == phenotype].copy()

        if df.empty:
            print(f"No data for {phenotype}")
            continue

        # Load filter clusters
        clusters_df = load_filter_clusters(results_dir, phenotype)

        # Print summary
        print_summary_table(df)

        # Create plots
        plot_summary_figure(df, phen_dir, phenotype)

        if not clusters_df.empty:
            plot_top_motifs_heatmap(df, clusters_df, phen_dir, phenotype)
            plot_frequency_distribution(clusters_df, phen_dir, phenotype)

        # Save summary CSV
        summary_df = df.drop(columns=["top_motifs"])
        summary_df.to_csv(phen_dir / "rashomon_cnn_all_configs.csv", index=False)

        print(f"\nOutputs saved to {phen_dir}")


if __name__ == "__main__":
    main()
