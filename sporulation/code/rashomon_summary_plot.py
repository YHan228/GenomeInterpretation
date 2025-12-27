#!/usr/bin/env python3
"""Generate summary figure comparing Rashomon analysis across all phenotypes."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load_all_summaries(results_dir: Path) -> pd.DataFrame:
    """Load all rashomon_summary.json files."""
    records = []
    for phen_dir in sorted(results_dir.iterdir()):
        summary_file = phen_dir / "rashomon_summary.json"
        if not summary_file.exists():
            continue
        try:
            with open(summary_file) as f:
                data = json.load(f)

            # Skip if no Rashomon set (insufficient data)
            if data["logistic"]["rashomon_size"] == 0 and data["rf"]["rashomon_size"] == 0:
                continue

            p = data["data"]["n_features"]
            records.append({
                "phenotype": data["phenotype"],
                "n_train": data["data"]["n_train"],
                "n_val": data["data"]["n_val"],
                "n_features": p,
                # Logistic
                "log_best": data["logistic"]["best"],
                "log_rashomon_size": data["logistic"]["rashomon_size"],
                "log_rashomon_frac": data["logistic"]["rashomon_size"] / data["logistic"]["n_total"],
                "log_n_necessary": data["logistic"]["n_necessary"],
                "log_n_frequent": data["logistic"]["n_common"],  # freq≥0.5 (includes necessary)
                "log_necessary_frac": data["logistic"]["n_necessary"] / p * 100,
                "log_frequent_frac": data["logistic"]["n_common"] / p * 100,
                # RF
                "rf_best": data["rf"]["best"],
                "rf_rashomon_size": data["rf"]["rashomon_size"],
                "rf_rashomon_frac": data["rf"]["rashomon_size"] / data["rf"]["n_total"],
                "rf_n_necessary": data["rf"]["n_necessary"],
                "rf_n_frequent": data["rf"]["n_common"],  # freq≥0.5 (includes necessary)
                "rf_necessary_frac": data["rf"]["n_necessary"] / p * 100,
                "rf_frequent_frac": data["rf"]["n_common"] / p * 100,
                # Overlap
                "necessary_both": data["overlap"]["necessary_both"],
                "frequent_both": data["overlap"]["common_both"],
            })
        except Exception as e:
            print(f"Skipping {phen_dir.name}: {e}")
            continue

    return pd.DataFrame(records)


def plot_summary(df: pd.DataFrame, out_dir: Path):
    """Create unified multi-panel summary figure with method agreement."""

    # Sort by logistic best performance
    df = df.sort_values("log_best", ascending=False).reset_index(drop=True)
    n = len(df)
    phenotypes = df["phenotype"].tolist()
    x = np.arange(n)
    width = 0.7

    # Create 2x2 figure with shared x-axis
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex='col',
                              gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.08, 'wspace': 0.25})

    # ---- Panel A: Best Performance (grouped bars) ----
    ax = axes[0, 0]
    w = 0.35
    ax.bar(x - w/2, df["log_best"], w, label="Logistic", color="#1f78b4", alpha=0.85)
    ax.bar(x + w/2, df["rf_best"], w, label="RF", color="#e31a1c", alpha=0.85)
    ax.set_ylabel("Best Balanced Accuracy", fontsize=10)
    ax.set_title("A. Model Performance", fontsize=11, fontweight='bold')
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.set_ylim(0.5, 1.02)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.5)
    ax.tick_params(axis='x', labelbottom=False)

    # ---- Panel B: Rashomon Set Size (grouped bars) ----
    ax = axes[0, 1]
    ax.bar(x - w/2, df["log_rashomon_frac"] * 100, w, label="Logistic", color="#1f78b4", alpha=0.85)
    ax.bar(x + w/2, df["rf_rashomon_frac"] * 100, w, label="RF", color="#e31a1c", alpha=0.85)
    ax.set_ylabel("Rashomon Set Size (%)", fontsize=10)
    ax.set_title("B. Rashomon Set Size (% within ε=2%)", fontsize=11, fontweight='bold')
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.set_ylim(0, 105)
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.5)
    ax.tick_params(axis='x', labelbottom=False)

    # ---- Panel C: Necessary Features with Method Agreement (stacked) ----
    # Necessary = freq=1.0 (appears in ALL Rashomon models)
    ax = axes[1, 0]
    # Stacked: Both methods (bottom), Logistic-only (middle), RF-only (top)
    log_only = df["log_n_necessary"] - df["necessary_both"]
    rf_only = df["rf_n_necessary"] - df["necessary_both"]
    bars_both = ax.bar(x, df["necessary_both"], width, label="Both", color="#2ca02c", alpha=0.9)
    bars_log = ax.bar(x, log_only, width, bottom=df["necessary_both"], label="Logistic only", color="#1f78b4", alpha=0.85)
    bars_rf = ax.bar(x, rf_only, width, bottom=df["necessary_both"] + log_only, label="RF only", color="#e31a1c", alpha=0.85)
    ax.set_ylabel("Necessary Features (count)", fontsize=10)
    ax.set_title("C. Necessary Features (freq=1.0) by Method", fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(phenotypes, rotation=40, ha="right", fontsize=9)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.5)
    # Add total count annotations
    totals = df["log_n_necessary"] + df["rf_n_necessary"] - df["necessary_both"]
    y_max_c = totals.max()
    ax.set_ylim(0, y_max_c * 1.12)
    for i, (total, both) in enumerate(zip(totals, df["necessary_both"])):
        if int(total) > 0:
            label = f"{int(total)}" if both == 0 else f"{int(total)} ({int(both)})"
            ax.text(x[i], total + y_max_c * 0.02, label,
                    ha='center', va='bottom', fontsize=7, fontweight='bold')

    # ---- Panel D: Frequent Features with Method Agreement (stacked) ----
    # Frequent = freq≥0.5 (appears in ≥50% of Rashomon models); includes Necessary
    ax = axes[1, 1]
    log_only_d = df["log_n_frequent"] - df["frequent_both"]
    rf_only_d = df["rf_n_frequent"] - df["frequent_both"]
    bars_both_d = ax.bar(x, df["frequent_both"], width, label="Both", color="#2ca02c", alpha=0.9)
    bars_log_d = ax.bar(x, log_only_d, width, bottom=df["frequent_both"], label="Logistic only", color="#1f78b4", alpha=0.85)
    bars_rf_d = ax.bar(x, rf_only_d, width, bottom=df["frequent_both"] + log_only_d, label="RF only", color="#e31a1c", alpha=0.85)
    ax.set_ylabel("Frequent Features (count)", fontsize=10)
    ax.set_title("D. Frequent Features (freq≥0.5) by Method", fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(phenotypes, rotation=40, ha="right", fontsize=9)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.5)
    # Add total count annotations
    totals_d = df["log_n_frequent"] + df["rf_n_frequent"] - df["frequent_both"]
    y_max_d = totals_d.max()
    ax.set_ylim(0, y_max_d * 1.12)
    for i, (total, both) in enumerate(zip(totals_d, df["frequent_both"])):
        if int(total) > 0:
            label = f"{int(total)}" if both == 0 else f"{int(total)} ({int(both)})"
            ax.text(x[i], total + y_max_d * 0.02, label,
                    ha='center', va='bottom', fontsize=7, fontweight='bold')

    # Align y-labels
    fig.align_ylabels(axes[:, 0])
    fig.align_ylabels(axes[:, 1])
    plt.tight_layout()

    for ext in (".png", ".pdf", ".svg"):
        fig.savefig(out_dir / f"rashomon_phenotype_summary{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return df


def process_results_dir(results_dir: Path, label: str = "") -> None:
    """Process a single results directory and generate summary plots."""
    out_dir = results_dir

    df = load_all_summaries(results_dir)
    print(f"Loaded {len(df)} phenotypes with results from {results_dir}")

    if len(df) == 0:
        print("No results found!")
        return

    df = plot_summary(df, out_dir)

    # Print summary table
    header = f"RASHOMON ANALYSIS SUMMARY{' - ' + label if label else ''}"
    print("\n" + "="*90)
    print(header)
    print("="*90)

    cols = ["phenotype", "log_best", "log_rashomon_frac", "log_n_necessary", "necessary_both",
            "rf_best", "rf_rashomon_frac", "rf_n_necessary"]
    summary = df[cols].copy()
    summary.columns = ["Phenotype", "Log Acc", "Log Rash%", "Log Nec", "Both",
                       "RF Acc", "RF Rash%", "RF Nec"]

    # Format
    for col in ["Log Acc", "RF Acc"]:
        summary[col] = summary[col].apply(lambda x: f"{x:.3f}")
    for col in ["Log Rash%", "RF Rash%"]:
        summary[col] = summary[col].apply(lambda x: f"{x*100:.0f}%")
    for col in ["Log Nec", "RF Nec", "Both"]:
        summary[col] = summary[col].apply(lambda x: f"{int(x)}")

    print(summary.to_string(index=False))
    print("\nNec = Necessary features (freq=1.0); Both = overlap between methods")

    # Save CSV
    df.to_csv(out_dir / "rashomon_all_phenotypes.csv", index=False)
    print(f"\nOutputs saved to {out_dir}")


def main():
    base_dir = Path("/home/yhan/GenomeInterpretation/sporulation/results")

    # Process standard rashomon results
    rashomon_dir = base_dir / "rashomon"
    if rashomon_dir.exists():
        process_results_dir(rashomon_dir, label="Standard")
    else:
        print(f"Directory not found: {rashomon_dir}")

    # Process blocked rashomon results
    rashomon_blocked_dir = base_dir / "rashomon_blocked"
    if rashomon_blocked_dir.exists():
        print("\n" + "="*90)
        print("Processing blocked features results...")
        print("="*90)
        process_results_dir(rashomon_blocked_dir, label="Blocked")
    else:
        print(f"\nDirectory not found: {rashomon_blocked_dir}")


if __name__ == "__main__":
    main()
