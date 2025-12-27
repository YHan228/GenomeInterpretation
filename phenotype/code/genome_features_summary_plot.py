#!/usr/bin/env python3
"""Generate summary figure comparing genome feature analysis across all phenotypes."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_all_summaries(results_dir: Path) -> pd.DataFrame:
    """Load all summary.json files from genome_stats subfolders."""
    records = []
    for phen_dir in sorted(results_dir.iterdir()):
        if not phen_dir.is_dir():
            continue
        summary_file = phen_dir / "summary.json"
        if not summary_file.exists():
            continue

        try:
            with open(summary_file) as f:
                data = json.load(f)

            # Skip custom/subset analyses
            if "custom" in phen_dir.name or "bacillales" in phen_dir.name:
                continue

            phenotype = phen_dir.name.replace("_", " ").title()
            n_classes = len(data.get("classes", []))
            total_genomes = data.get("total_genomes", 0)

            if total_genomes == 0:
                continue

            # Extract GC stats
            gc_stats = data.get("gc_content_stats", {})
            gc_means = [gc_stats.get(c, {}).get("mean", np.nan) for c in data.get("classes", [])]
            gc_diff = abs(gc_means[0] - gc_means[1]) if len(gc_means) >= 2 else np.nan

            # Extract length stats
            len_stats = data.get("length_stats", {})
            len_means = [len_stats.get(c, {}).get("mean", np.nan) for c in data.get("classes", [])]
            len_ratio = max(len_means) / min(len_means) if len(len_means) >= 2 and min(len_means) > 0 else np.nan

            # Extract logistic model metrics (balanced accuracy and ROC AUC)
            logistic = data.get("logistic_models", {})
            accuracies = {}
            aucs = {}
            for k_key, k_data in logistic.items():
                if isinstance(k_data, dict):
                    if "balanced_accuracy" in k_data and k_data["balanced_accuracy"] is not None:
                        accuracies[k_key] = k_data["balanced_accuracy"]
                    if "roc_auc" in k_data and k_data["roc_auc"] is not None:
                        aucs[k_key] = k_data["roc_auc"]

            # Best metrics across all k values
            best_acc = max(accuracies.values()) if accuracies else np.nan
            best_auc = max(aucs.values()) if aucs else np.nan
            best_k_acc = max(accuracies, key=accuracies.get) if accuracies else None
            best_k_auc = max(aucs, key=aucs.get) if aucs else None

            # Compute 3-mer profile divergence (mean |log2FC| across top k-mers)
            top_kmers = data.get("top_kmers", {}).get("k3", {})
            log2fc_values = []
            for class_data in top_kmers.values():
                if isinstance(class_data, list):
                    for kmer_info in class_data:
                        if isinstance(kmer_info, dict) and "log2fc" in kmer_info:
                            log2fc_values.append(abs(kmer_info["log2fc"]))
            kmer3_divergence = np.mean(log2fc_values) if log2fc_values else np.nan

            records.append({
                "phenotype": phenotype,
                "slug": phen_dir.name,
                "n_classes": n_classes,
                "total_genomes": total_genomes,
                "gc_diff": gc_diff,
                "len_ratio": len_ratio,
                "kmer3_divergence": kmer3_divergence,
                # Balanced accuracy
                "acc_k3": accuracies.get("k3", np.nan),
                "acc_k4": accuracies.get("k4", np.nan),
                "acc_k6": accuracies.get("k6", np.nan),
                "acc_k8": accuracies.get("k8", np.nan),
                "best_acc": best_acc,
                "best_k_acc": best_k_acc,
                # ROC AUC
                "auc_k3": aucs.get("k3", np.nan),
                "auc_k4": aucs.get("k4", np.nan),
                "auc_k6": aucs.get("k6", np.nan),
                "auc_k8": aucs.get("k8", np.nan),
                "best_auc": best_auc,
                "best_k_auc": best_k_auc,
            })
        except Exception as e:
            print(f"Skipping {phen_dir.name}: {e}")
            continue

    return pd.DataFrame(records)


def plot_summary(df: pd.DataFrame, out_dir: Path):
    """Create multi-panel summary figure."""
    # Filter to binary classification only (ROC AUC available) with sufficient data
    df = df[(df["n_classes"] == 2) & (df["best_auc"].notna())].copy()
    if df.empty:
        print("No binary classification phenotypes with AUC found!")
        return df

    # Sort by best AUC
    df = df.sort_values("best_auc", ascending=False).reset_index(drop=True)
    n = len(df)
    phenotypes = df["phenotype"].tolist()
    x = np.arange(n)

    # Create figure with shared x-axis columns
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex='col',
                              gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.08, 'wspace': 0.25})

    width = 0.18
    colors = {"k3": "#1f78b4", "k4": "#33a02c", "k6": "#ff7f00", "k8": "#e31a1c"}

    # ---- Panel A: ROC AUC by k-mer size ----
    ax = axes[0, 0]
    for i, k in enumerate(["k3", "k4", "k6", "k8"]):
        col = f"auc_{k}"
        if col in df.columns:
            vals = df[col].fillna(0)
            ax.bar(x + (i - 1.5) * width, vals, width, label=k, color=colors[k], alpha=0.85)

    ax.set_ylabel("ROC AUC", fontsize=10)
    ax.set_title("A. ROC AUC by k-mer size", fontsize=11, fontweight='bold')
    ax.legend(loc="lower left", title="k", fontsize=8, title_fontsize=8, ncol=2)
    ax.set_ylim(0.45, 1.02)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.5)
    ax.tick_params(axis='x', labelbottom=False)

    # ---- Panel B: Balanced Accuracy by k-mer size ----
    ax = axes[0, 1]
    for i, k in enumerate(["k3", "k4", "k6", "k8"]):
        col = f"acc_{k}"
        if col in df.columns:
            vals = df[col].fillna(0)
            ax.bar(x + (i - 1.5) * width, vals, width, label=k, color=colors[k], alpha=0.85)

    ax.set_ylabel("Balanced Accuracy", fontsize=10)
    ax.set_title("B. Balanced Accuracy by k-mer size", fontsize=11, fontweight='bold')
    ax.legend(loc="lower left", title="k", fontsize=8, title_fontsize=8, ncol=2)
    ax.set_ylim(0.45, 1.02)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.5)
    ax.tick_params(axis='x', labelbottom=False)

    # ---- Panel C: GC Content Difference ----
    ax = axes[1, 0]
    bars_c = ax.bar(x, df["gc_diff"].fillna(0) * 100, color="#984ea3", alpha=0.85, width=0.6)
    ax.set_ylabel("GC Difference (%)", fontsize=10)
    ax.set_title("C. GC Content Difference", fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(phenotypes, rotation=40, ha="right", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.5)

    # ---- Panel D: 3-mer Profile Divergence ----
    ax = axes[1, 1]
    bars_d = ax.bar(x, df["kmer3_divergence"].fillna(0), color="#e31a1c", alpha=0.85, width=0.6)
    ax.set_ylabel("Mean |log₂FC|", fontsize=10)
    ax.set_title("D. 3-mer Profile Divergence", fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(phenotypes, rotation=40, ha="right", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.5)

    # Adjust layout
    fig.align_ylabels(axes[:, 0])
    fig.align_ylabels(axes[:, 1])
    plt.tight_layout()

    for ext in (".png", ".pdf", ".svg"):
        fig.savefig(out_dir / f"genome_features_phenotype_summary{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- Second figure: AUC vs GC difference (correlation check) ----
    fig2, ax = plt.subplots(figsize=(7, 6))
    valid = df.dropna(subset=["best_auc", "gc_diff"])
    scatter = ax.scatter(valid["gc_diff"] * 100, valid["best_auc"],
                        s=valid["total_genomes"] / 20, alpha=0.7, c="#1f78b4", edgecolors="black")
    for _, row in valid.iterrows():
        ax.annotate(row["phenotype"], (row["gc_diff"] * 100, row["best_auc"]),
                   fontsize=7, alpha=0.8, xytext=(3, 3), textcoords="offset points")

    ax.set_xlabel("GC Content Difference Between Classes (%)")
    ax.set_ylabel("Best ROC AUC")
    ax.set_title("AUC vs. GC Difference (bubble size = dataset size)")
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3)

    fig2.tight_layout()
    for ext in (".png", ".pdf", ".svg"):
        fig2.savefig(out_dir / f"genome_features_auc_vs_gc{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig2)

    return df


def main():
    results_dir = Path("/home/yhan/GenomeInterpretation/phenotype/plots/genome_stats")
    out_dir = results_dir

    df = load_all_summaries(results_dir)
    print(f"Loaded {len(df)} phenotypes with results")

    if len(df) == 0:
        print("No results found!")
        return

    df = plot_summary(df, out_dir)

    # df is already filtered to binary with AUC in plot_summary
    df_binary = df.sort_values("best_auc", ascending=False)

    # Print summary table
    print("\n" + "="*90)
    print("GENOME FEATURES ANALYSIS SUMMARY (binary phenotypes only, sorted by Best AUC)")
    print("="*90)

    cols = ["phenotype", "total_genomes", "gc_diff", "best_auc", "best_k_auc",
            "auc_k3", "auc_k4", "auc_k6", "auc_k8"]
    summary = df_binary[cols].copy()
    summary.columns = ["Phenotype", "N", "GC Δ", "Best AUC", "Best k",
                       "k=3", "k=4", "k=6", "k=8"]

    # Format
    summary["GC Δ"] = summary["GC Δ"].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "N/A")
    for col in ["Best AUC", "k=3", "k=4", "k=6", "k=8"]:
        summary[col] = summary[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")

    print(summary.to_string(index=False))

    # Save CSV
    df.to_csv(out_dir / "genome_features_all_phenotypes.csv", index=False)
    print(f"\nOutputs saved to {out_dir}")


if __name__ == "__main__":
    main()
