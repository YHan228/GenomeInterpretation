#!/usr/bin/env python3
"""Four-panel figure for sporulation analysis.

Panels:
  A. Rashomon common overlap genes vs. spore-related (by regex) genes
  B. NT conservation col-wise vs. pairwise scatter (from codon MSA)
  C. Spore-related genes prevalence in Spore+ vs. Spore- (violin/box)
  D. Window spore fraction: zero vs. non-zero (stacked bar)

Output: PDF + SVG under sporulation/reports/
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    sns.set_theme(context="paper", style="whitegrid", font_scale=1.1)
except ImportError:
    sns = None


# Sporulation keywords regex (consistent with rashomon.py / analyze_gff.py)
SPORULATION_KEYWORDS_RE = re.compile(
    r"(?i)\b(spo|ssp|cot|sigma|sig|ger|sleb|cwlj|dpa|spov|spoii|spo0)"
)


def gene_is_sporulation_related(gene: str) -> bool:
    """Check if gene name matches sporulation-related regex."""
    if gene is None or not isinstance(gene, str):
        return False
    return SPORULATION_KEYWORDS_RE.search(gene) is not None


def _phenotype_labels(s: pd.Series) -> pd.Series:
    """Map boolean 'Spore formation' to string labels."""
    s_bool = s.astype("boolean")
    return s_bool.map({True: "Spore+", False: "Spore-"}).fillna("Unknown")


def load_rashomon_features(path: Path) -> pd.DataFrame:
    """Load rashomon features CSV."""
    df = pd.read_csv(path)
    return df


def load_conservation_summary(path: Path) -> pd.DataFrame:
    """Load codon MSA conservation summary CSV."""
    df = pd.read_csv(path)
    return df


def load_per_genome_metrics(path: Path) -> pd.DataFrame:
    """Load per-genome metrics CSV from analyze_gff."""
    df = pd.read_csv(path)
    return df


def load_window_samples(path: Path, per_genome: pd.DataFrame) -> pd.DataFrame:
    """Load window samples CSV and join with phenotype."""
    df = pd.read_csv(path)
    pheno = per_genome[["gff_filename", "Spore formation"]].copy()
    pheno["Phenotype"] = _phenotype_labels(pheno["Spore formation"])
    df = df.merge(pheno[["gff_filename", "Phenotype"]], on="gff_filename", how="left")
    return df


def panel_a_rashomon_overlap(ax, rashomon_df: pd.DataFrame) -> None:
    """Panel A: Rashomon frequent overlap vs. spore-related genes.

    Shows a stacked bar comparing:
    - Rashomon frequent genes (freq >= 0.5 in both Logistic and RF)
    - Spore-related genes (by regex)
    - Overlap between the two sets
    """
    # Frequent in both methods: freq >= 0.5 for both logistic and RF
    frequent_both_mask = (rashomon_df["freq_logistic"] >= 0.5) & (rashomon_df["freq_rf"] >= 0.5)
    frequent_both_genes = set(rashomon_df.loc[frequent_both_mask, "gene"].tolist())

    # All genes in rashomon set
    all_genes = rashomon_df["gene"].tolist()

    # Spore-related by regex
    spore_related_genes = set(g for g in all_genes if gene_is_sporulation_related(g))

    # Compute overlaps
    only_frequent = frequent_both_genes - spore_related_genes
    only_spore = spore_related_genes - frequent_both_genes
    both = frequent_both_genes & spore_related_genes

    # Grouped bar chart showing overlap
    categories = ["Rashomon\nfrequent", "Spore-regex", "Overlap"]
    values = [len(frequent_both_genes), len(spore_related_genes), len(both)]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    bars = ax.bar(categories, values, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)

    # Annotate bar values
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            str(val),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_ylabel("Number of genes")
    ax.set_title("A. Rashomon frequent vs. spore-regex", fontweight="bold", loc="left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Compute overlap percentage
    if len(frequent_both_genes) > 0:
        pct_overlap = len(both) / len(frequent_both_genes) * 100
        ax.text(
            0.95, 0.95,
            f"Overlap: {pct_overlap:.1f}% of frequent",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            style="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )


def panel_b_nt_conservation(ax, conservation_df: pd.DataFrame) -> None:
    """Panel B: NT conservation col-wise vs. pairwise scatter."""
    df = conservation_df.dropna(subset=["nt_mean_col_conservation", "nt_mean_pairwise_identity"])

    ax.scatter(
        df["nt_mean_col_conservation"],
        df["nt_mean_pairwise_identity"],
        s=25,
        alpha=0.7,
        c="#1f77b4",
        edgecolors="none",
    )

    # Add diagonal reference line
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, lw=1)

    ax.set_xlabel("NT mean column conservation")
    ax.set_ylabel("NT mean pairwise identity")
    ax.set_xlim(0.6, 1.0)
    ax.set_ylim(0, 0.85)
    ax.set_title("B. Codon MSA conservation", fontweight="bold", loc="left")

    # Annotate a few outlier genes
    df_sorted = df.sort_values("nt_mean_pairwise_identity", ascending=False)
    for _, row in df_sorted.head(3).iterrows():
        ax.annotate(
            row["gene"],
            (row["nt_mean_col_conservation"], row["nt_mean_pairwise_identity"]),
            fontsize=7,
            alpha=0.8,
            xytext=(3, 3),
            textcoords="offset points",
        )


def panel_c_spore_prevalence(ax, per_genome_df: pd.DataFrame) -> None:
    """Panel C: Spore-related genes prevalence by phenotype."""
    df = per_genome_df.copy()
    df["Phenotype"] = _phenotype_labels(df["Spore formation"])
    df = df[df["Phenotype"].isin(["Spore+", "Spore-"])]

    if sns is not None:
        sns.violinplot(
            data=df,
            x="Phenotype",
            y="n_spore_loci",
            hue="Phenotype",
            inner=None,
            cut=0,
            linewidth=0.8,
            ax=ax,
            order=["Spore+", "Spore-"],
            hue_order=["Spore+", "Spore-"],
            palette={"Spore+": "#2ca02c", "Spore-": "#d62728"},
            legend=False,
        )
        sns.boxplot(
            data=df,
            x="Phenotype",
            y="n_spore_loci",
            whis=1.5,
            width=0.2,
            showcaps=True,
            boxprops={"facecolor": "white"},
            ax=ax,
            order=["Spore+", "Spore-"],
        )
        sns.stripplot(
            data=df,
            x="Phenotype",
            y="n_spore_loci",
            color="k",
            alpha=0.3,
            size=2,
            jitter=0.12,
            ax=ax,
            order=["Spore+", "Spore-"],
        )
    else:
        df.boxplot(column="n_spore_loci", by="Phenotype", ax=ax)
        ax.get_figure().suptitle("")

    ax.set_xlabel("")
    ax.set_ylabel("# spore loci")
    ax.set_title("C. Spore loci count by phenotype", fontweight="bold", loc="left")


def panel_d_window_stacked(ax, window_df: pd.DataFrame) -> None:
    """Panel D: Window spore fraction zero vs. non-zero (stacked bar)."""
    df = window_df.copy()
    df = df[df["Phenotype"].isin(["Spore+", "Spore-"])]

    df["is_zero"] = (df["spore_frac"] <= 0).astype(int)
    counts = df.groupby(["Phenotype", "is_zero"]).size().reset_index(name="n")
    totals = counts.groupby("Phenotype")["n"].transform("sum")
    counts["prop"] = counts["n"] / totals

    # Pivot for stacked bar
    pivot = counts.pivot(index="Phenotype", columns="is_zero", values="prop").fillna(0)
    pivot = pivot.reindex(["Spore+", "Spore-"])
    pivot.columns = ["Non-zero", "Zero"]

    # Stacked bar
    x = np.arange(len(pivot))
    width = 0.6

    bars_zero = ax.bar(x, pivot["Zero"], width, label="Zero", color="#d62728", alpha=0.7)
    bars_nonzero = ax.bar(x, pivot["Non-zero"], width, bottom=pivot["Zero"], label="Non-zero", color="#2ca02c", alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index)
    ax.set_ylabel("Proportion of windows")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("D. Window spore fraction", fontweight="bold", loc="left")

    # Annotate percentages
    for i, (z, nz) in enumerate(zip(pivot["Zero"], pivot["Non-zero"])):
        ax.text(i, z / 2, f"{z * 100:.1f}%", ha="center", va="center", fontsize=9, color="white", fontweight="bold")
        ax.text(i, z + nz / 2, f"{nz * 100:.1f}%", ha="center", va="center", fontsize=9, color="white", fontweight="bold")


def create_four_panel_figure(
    rashomon_path: Path,
    conservation_path: Path,
    per_genome_path: Path,
    window_path: Path,
    output_dir: Path,
    output_name: str = "four_panel_figure",
    label: str = "",
) -> None:
    """Create a single four-panel figure with given data sources."""
    output_dir.mkdir(parents=True, exist_ok=True)

    header = f"Loading data{' (' + label + ')' if label else ''}..."
    print(header, flush=True)
    rashomon_df = load_rashomon_features(rashomon_path)
    conservation_df = load_conservation_summary(conservation_path)
    per_genome_df = load_per_genome_metrics(per_genome_path)
    window_df = load_window_samples(window_path, per_genome_df)

    print(f"  Rashomon features: {len(rashomon_df)} genes", flush=True)
    print(f"  Conservation summary: {len(conservation_df)} genes", flush=True)
    print(f"  Per-genome metrics: {len(per_genome_df)} genomes", flush=True)
    print(f"  Window samples: {len(window_df)} windows", flush=True)

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    ax_a, ax_b = axes[0]
    ax_c, ax_d = axes[1]

    print("Creating panels...", flush=True)
    panel_a_rashomon_overlap(ax_a, rashomon_df)
    panel_b_nt_conservation(ax_b, conservation_df)
    panel_c_spore_prevalence(ax_c, per_genome_df)
    panel_d_window_stacked(ax_d, window_df)

    # Add overall title if label provided
    if label:
        fig.suptitle(f"Sporulation Analysis ({label})", fontsize=12, fontweight="bold", y=1.02)

    fig.tight_layout()

    # Save in multiple formats
    out_base = output_dir / output_name
    for ext in [".pdf", ".svg", ".png"]:
        out_path = out_base.with_suffix(ext)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {out_path}", flush=True)

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Create four-panel sporulation figure")
    parser.add_argument(
        "--rashomon_csv",
        type=Path,
        default=Path("sporulation/results/rashomon/Spore_formation/rashomon_features.csv"),
    )
    parser.add_argument(
        "--conservation_csv",
        type=Path,
        default=Path("sporulation/analysis_out/codon_msa/conservation_summary.csv"),
    )
    parser.add_argument(
        "--per_genome_csv",
        type=Path,
        default=Path("sporulation/analysis_out/per_genome_metrics.csv"),
    )
    parser.add_argument(
        "--window_csv",
        type=Path,
        default=Path("sporulation/analysis_out/window_samples.csv"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("sporulation/reports"),
    )
    parser.add_argument(
        "--run_blocked",
        action="store_true",
        help="Also generate figure for rashomon_blocked results",
    )
    args = parser.parse_args()

    # Resolve paths relative to script location or cwd
    base_dir = Path(__file__).resolve().parent.parent.parent
    rashomon_path = base_dir / args.rashomon_csv if not args.rashomon_csv.is_absolute() else args.rashomon_csv
    conservation_path = base_dir / args.conservation_csv if not args.conservation_csv.is_absolute() else args.conservation_csv
    per_genome_path = base_dir / args.per_genome_csv if not args.per_genome_csv.is_absolute() else args.per_genome_csv
    window_path = base_dir / args.window_csv if not args.window_csv.is_absolute() else args.window_csv
    output_dir = base_dir / args.output_dir if not args.output_dir.is_absolute() else args.output_dir

    # Generate standard figure
    create_four_panel_figure(
        rashomon_path=rashomon_path,
        conservation_path=conservation_path,
        per_genome_path=per_genome_path,
        window_path=window_path,
        output_dir=output_dir,
        output_name="four_panel_figure",
        label="",
    )

    # Generate blocked figure if requested or if blocked path exists
    rashomon_blocked_path = base_dir / "sporulation/results/rashomon_blocked/Spore_formation_by_Order/rashomon_features.csv"
    if args.run_blocked or rashomon_blocked_path.exists():
        if rashomon_blocked_path.exists():
            print("\n" + "=" * 60, flush=True)
            print("Generating blocked features version...", flush=True)
            print("=" * 60, flush=True)
            create_four_panel_figure(
                rashomon_path=rashomon_blocked_path,
                conservation_path=conservation_path,
                per_genome_path=per_genome_path,
                window_path=window_path,
                output_dir=output_dir,
                output_name="four_panel_figure_blocked",
                label="Blocked",
            )
        else:
            print(f"\nBlocked rashomon file not found: {rashomon_blocked_path}", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
