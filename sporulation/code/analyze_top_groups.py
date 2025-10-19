#!/usr/bin/env python3
"""Analyze the top grouped-PI groups: enrichment and correlation (train+val only).

Inputs (defaults):
- Presence/absence: /vol/projects/BIFO/genomenet/yichen/phenotype/data/rfdata/rf_presence_absence.parquet (or .csv)
- Long table: /vol/projects/BIFO/genomenet/yichen/phenotype/data/rfdata/rf_dataset.parquet (or .csv)
- Analysis dir for grouped PI CSVs:
  - rf_grouped_permutation_groups_top{K}.csv
  - rf_grouped_permutation_membership_top{K}.csv

Behavior:
- Picks the largest available K automatically (max K) unless user points to explicit CSVs
- Selects the top-N groups by rank (default N=2)
- For each group:
  - Builds train+val subset of PA and labels
  - Computes per-gene enrichment (Fisher exact) between sporulation-positive and negative samples
  - Volcano plot: log2(odds ratio) vs -log10(p)
  - Correlation matrix heatmap among genes in the group
  - Saves per-group enrichment CSV and correlation CSV

Outputs (in analysis_dir):
- top_group_{gid}_enrichment.csv
- top_group_{gid}_corr.csv
- figures/top_group_{gid}_volcano.png
- figures/top_group_{gid}_corr.png
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import fisher_exact


DATA_ROOT = Path("/vol/projects/BIFO/genomenet/yichen/phenotype/data")
DEFAULT_INPUT_DIR = DATA_ROOT / "rfdata"
DEFAULT_ANALYSIS_DIR = DATA_ROOT / "rfdata" / "analysis"


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    parquet_path = path.with_suffix(".parquet")
    csv_path = path.with_suffix(".csv")
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(f"Could not find {path} (.parquet or .csv)")


def aggregate_labels_per_sample(long_df: pd.DataFrame) -> pd.Series:
    if "sample_id" not in long_df.columns:
        raise ValueError("long_df must contain column 'sample_id'")
    if "Spore formation" not in long_df.columns:
        raise ValueError("long_df must contain column 'Spore formation'")
    def _map(val: object) -> Optional[int]:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return None
        s = str(val).strip().lower()
        if s in {"1", "yes", "y", "true", "t", "spore-forming", "spore forming", "sporeformer", "spore former", "sporulating", "sporulation", "positive", "pos"}:
            return 1
        if s in {"0", "no", "n", "false", "f", "non-spore-forming", "non spore forming", "nonspore", "non-spore", "nonsporulating", "negative", "neg"}:
            return 0
        try:
            f = float(s)
            if f == 0.0:
                return 0
            if f == 1.0:
                return 1
        except Exception:
            pass
        return None
    def resolve(series: pd.Series) -> Optional[int]:
        mapped = series.map(_map).dropna()
        if mapped.empty:
            return None
        return int(float(mapped.mean()) >= 0.5)
    return long_df.groupby("sample_id")["Spore formation"].apply(resolve)


def discover_group_csvs(analysis_dir: Path) -> Tuple[Path, Path]:
    group_files = list(analysis_dir.glob("rf_grouped_permutation_groups_top*.csv"))
    member_files = list(analysis_dir.glob("rf_grouped_permutation_membership_top*.csv"))
    if not group_files or not member_files:
        raise FileNotFoundError("Grouped PI CSVs not found in analysis_dir")
    def extract_k(p: Path) -> Optional[int]:
        m = re.search(r"_top(\d+)\.csv$", p.name)
        return int(m.group(1)) if m else None
    kg = {extract_k(p): p for p in group_files if extract_k(p) is not None}
    km = {extract_k(p): p for p in member_files if extract_k(p) is not None}
    common = sorted(set(kg.keys()) & set(km.keys()))
    if not common:
        # Fallback to latest by mtime
        g = max(group_files, key=lambda p: p.stat().st_mtime)
        m = max(member_files, key=lambda p: p.stat().st_mtime)
        return g, m
    best_k = max(common)
    return kg[best_k], km[best_k]


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    n = len(pvals)
    if n == 0:
        return np.array([])
    order = np.argsort(pvals)
    ranked = pvals[order]
    q = ranked * n / (np.arange(1, n + 1))
    # enforce monotonicity
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = q
    return np.minimum(out, 1.0)


def plot_volcano(
    df: pd.DataFrame,
    fig_path: Path,
    title: str,
    alpha: float = 0.05,
    abs_log2_or_thr: float = 0.0,
    label_top: int = 10,
    dpi: int = 200,
    y_cap: Optional[float] = None,
) -> None:
    x = df["log2_or"].to_numpy()
    y = -np.log10(df["pval"].clip(lower=1e-300)).to_numpy()
    sig = (df["fdr_bh"].to_numpy() < float(alpha))
    up = (x > float(abs_log2_or_thr)) & sig
    down = (x < -float(abs_log2_or_thr)) & sig
    colors = np.where(up, "tab:red", np.where(down, "tab:blue", "lightgray"))
    plt.figure(figsize=(7, 5))
    plt.scatter(x, y, s=12, alpha=0.7, c=colors, edgecolors="none")
    # guide lines
    plt.axhline(-np.log10(max(alpha, 1e-300)), color="gray", linestyle="--", linewidth=0.8)
    plt.axvline(0.0, color="black", linewidth=0.8)
    if abs_log2_or_thr > 0:
        plt.axvline(float(abs_log2_or_thr), color="gray", linestyle=":", linewidth=0.8)
        plt.axvline(-float(abs_log2_or_thr), color="gray", linestyle=":", linewidth=0.8)
    # annotate top by smallest FDR then pval
    if label_top > 0:
        top = df.sort_values(["fdr_bh", "pval"]).head(label_top)
        for _, row in top.iterrows():
            plt.text(row["log2_or"], -np.log10(max(row["pval"], 1e-300)), row["gene_norm"], fontsize=7)
    if y_cap is None:
        y_cap = float(np.percentile(y, 99.5) + 2.0)
    plt.ylim(0, y_cap)
    plt.xlabel("log2(odds ratio) (presence ↔ sporulation)")
    plt.ylabel("-log10(p)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=dpi)
    plt.close()


def plot_corr_heatmap(mat: np.ndarray, genes: List[str], fig_path: Path, title: str, dpi: int = 200) -> None:
    if mat.size == 0:
        return
    plt.figure(figsize=(max(6, 0.25 * len(genes)), max(5, 0.25 * len(genes))))
    plt.imshow(mat, vmin=-1, vmax=1, cmap="coolwarm", interpolation="nearest")
    plt.colorbar(label="Pearson r")
    plt.title(title)
    # Optional: skip tick labels if too many
    if len(genes) <= 50:
        plt.xticks(range(len(genes)), genes, rotation=90, fontsize=6)
        plt.yticks(range(len(genes)), genes, fontsize=6)
    else:
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    plt.savefig(fig_path, dpi=dpi)
    plt.close()


def plot_combined_presence_bar(
    X_pos: pd.DataFrame,
    fig_path: Path,
    title: str,
    highlight_thr: float = 1.0,
    dpi: int = 200,
) -> pd.DataFrame:
    """Bar plot of per-gene prevalence across sporulating (positive) train+val samples.

    Returns the prevalence table (gene_norm, prevalence) and saves a figure.
    """
    if X_pos.empty:
        return pd.DataFrame(columns=["gene_norm", "prevalence"])  # empty
    prev = X_pos.astype(float).mean(axis=0).sort_values(ascending=False)
    tbl = prev.rename("prevalence").reset_index().rename(columns={"index": "gene_norm"})
    colors = ["tab:red" if p >= float(highlight_thr) else "tab:gray" for p in tbl["prevalence"]]
    width = max(8, min(0.25 * len(tbl), 40))
    plt.figure(figsize=(width, 5))
    plt.bar(range(len(tbl)), tbl["prevalence"], color=colors)
    plt.axhline(float(highlight_thr), color="black", linestyle=":", linewidth=1.0)
    plt.axhline(0.95, color="gray", linestyle="--", linewidth=0.8)
    plt.ylabel("Fraction of sporulating (train+val) with gene present")
    plt.title(title)
    # tick labels sparsely to avoid clutter
    if len(tbl) <= 80:
        plt.xticks(range(len(tbl)), tbl["gene_norm"], rotation=90, fontsize=6)
    else:
        step = max(1, len(tbl) // 50)
        idxs = list(range(0, len(tbl), step))
        plt.xticks(idxs, tbl["gene_norm"].iloc[idxs], rotation=90, fontsize=6)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=dpi)
    plt.close()
    return tbl


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze top grouped-PI groups (enrichment and correlation)")
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT_DIR, help="Directory with rf_presence_absence and rf_dataset")
    parser.add_argument("--analysis_dir", type=Path, default=DEFAULT_ANALYSIS_DIR, help="Directory with grouped PI CSVs")
    parser.add_argument("--groups_csv", type=Path, default=None, help="Path to rf_grouped_permutation_groups_top*.csv")
    parser.add_argument("--membership_csv", type=Path, default=None, help="Path to rf_grouped_permutation_membership_top*.csv")
    parser.add_argument("--top_groups", type=int, default=2, help="Number of groups to analyze (default=2)")
    parser.add_argument("--label_top", type=int, default=10, help="Annotate top-N genes on volcano")
    args = parser.parse_args()

    input_dir: Path = args.input_dir
    analysis_dir: Path = args.analysis_dir
    (analysis_dir / "figures").mkdir(parents=True, exist_ok=True)

    # Discover CSVs
    groups_csv = args.groups_csv
    members_csv = args.membership_csv
    if groups_csv is None or members_csv is None:
        g, m = discover_group_csvs(analysis_dir)
        groups_csv = groups_csv or g
        members_csv = members_csv or m

    groups_df = pd.read_csv(groups_csv)
    members_df = pd.read_csv(members_csv)
    if "rank" not in groups_df.columns:
        groups_df = groups_df.sort_values(["gpi_mean", "group_id"], ascending=[False, True]).reset_index(drop=True)
        groups_df["rank"] = np.arange(1, len(groups_df) + 1)
    groups_df = groups_df.sort_values("rank", ascending=True)
    top_groups = groups_df.head(int(args.top_groups))["group_id"].astype(int).tolist()
    print(f"Analyzing groups: {top_groups}", flush=True)

    # Load datasets
    pa_df = _read_table(input_dir / "rf_presence_absence.parquet")
    if pa_df.index.name != "sample_id":
        if "sample_id" in pa_df.columns:
            pa_df = pa_df.set_index("sample_id")
        else:
            pa_df.index.name = "sample_id"
    long_df = _read_table(input_dir / "rf_dataset.parquet")
    y_series = aggregate_labels_per_sample(long_df)

    # Train+val mask
    split = pa_df["split"].astype("string")
    mask_tv = (split == "train") | (split == "val")

    for gid in top_groups:
        genes = members_df.loc[members_df["group_id"] == gid, "gene_norm"].astype(str).unique().tolist()
        genes = [g for g in genes if g in pa_df.columns]
        if not genes:
            print(f"Group {gid}: no genes present in PA", flush=True)
            continue
        print(f"Group {gid}: {len(genes)} genes", flush=True)

        # Build X and y for train+val
        X_df = pa_df.loc[mask_tv, genes].astype(float)
        y = y_series.loc[X_df.index].astype(float).dropna()
        X_df = X_df.loc[y.index]

        # Enrichment per gene (Fisher exact), with Haldane-Anscombe correction for OR
        rows = []
        y_bin = y.astype(int).to_numpy()
        for gene in genes:
            x = X_df[gene].fillna(0).astype(int).to_numpy()
            a = int(((y_bin == 1) & (x == 1)).sum())
            b = int(((y_bin == 0) & (x == 1)).sum())
            c = int(((y_bin == 1) & (x == 0)).sum())
            d = int(((y_bin == 0) & (x == 0)).sum())
            # fisher exact p-value
            try:
                _, p = fisher_exact([[a, b], [c, d]], alternative="two-sided")
            except Exception:
                p = 1.0
            # Haldane-Anscombe for log2 OR
            a2, b2, c2, d2 = a + 0.5, b + 0.5, c + 0.5, d + 0.5
            or_val = (a2 * d2) / (b2 * c2)
            log2_or = float(np.log2(or_val))
            rows.append({
                "gene_norm": gene,
                "pos_present": a,
                "neg_present": b,
                "pos_absent": c,
                "neg_absent": d,
                "odds_ratio": float(or_val),
                "log2_or": log2_or,
                "pval": float(p),
            })
        enr_df = pd.DataFrame(rows)
        enr_df["fdr_bh"] = bh_fdr(enr_df["pval"].to_numpy())
        enr_df = enr_df.sort_values(["fdr_bh", "pval"]).reset_index(drop=True)
        enr_path = analysis_dir / f"top_group_{gid}_enrichment.csv"
        enr_df.to_csv(enr_path, index=False)

        # Volcano plot
        plot_volcano(
            enr_df,
            analysis_dir / "figures" / f"top_group_{gid}_volcano.png",
            title=f"Group {gid}: enrichment (train+val)",
            alpha=0.05,
            abs_log2_or_thr=0.0,
            label_top=int(args.label_top),
        )

        # Correlation matrix
        try:
            corr = np.corrcoef(X_df.to_numpy(dtype=float), rowvar=False)
        except Exception:
            corr = np.array([[]])
        corr_df = pd.DataFrame(corr, index=genes, columns=genes)
        corr_df.to_csv(analysis_dir / f"top_group_{gid}_corr.csv")
        plot_corr_heatmap(corr, genes, analysis_dir / "figures" / f"top_group_{gid}_corr.png", title=f"Group {gid}: gene correlation (train+val)")

    # Combined plot: are combined-group genes universally present across sporulating samples?
    try:
        all_genes = []
        for gid in top_groups:
            genes = members_df.loc[members_df["group_id"] == gid, "gene_norm"].astype(str).unique().tolist()
            all_genes.extend(genes)
        all_genes = [g for g in sorted(set(all_genes)) if g in pa_df.columns]
        if all_genes:
            X_all = pa_df.loc[mask_tv, all_genes].astype(float)
            y_all = aggregate_labels_per_sample(long_df).loc[X_all.index].astype(float).dropna()
            X_all = X_all.loc[y_all.index]
            X_pos = X_all.loc[y_all == 1]
            tbl = plot_combined_presence_bar(
                X_pos,
                analysis_dir / "figures" / "combined_pos_presence.png",
                title="Combined groups: presence in sporulating (train+val)",
            )
            tbl.to_csv(analysis_dir / "combined_pos_gene_prevalence.csv", index=False)
            print(f"Combined presence: {len(all_genes)} genes; perfect-coverage genes = {(tbl['prevalence']>=1.0).sum()}", flush=True)
        else:
            print("Combined presence plot skipped: no genes found in PA for selected groups", flush=True)
    except Exception as e:
        print(f"Combined presence plot failed: {e}", flush=True)

    # ---------------- LASSO-selected genes: enrichment, correlation, presence ----------------
    try:
        lasso_imp_path = input_dir / "lasso_feature_importance.csv"
        if lasso_imp_path.exists():
            lasso_imp_df = pd.read_csv(lasso_imp_path)
            if "coef" in lasso_imp_df.columns:
                lasso_genes = (
                    lasso_imp_df.loc[lasso_imp_df["coef"] != 0, "gene_norm"].astype(str).unique().tolist()
                )
            elif "importance" in lasso_imp_df.columns:
                lasso_genes = (
                    lasso_imp_df.loc[lasso_imp_df["importance"] > 0, "gene_norm"].astype(str).unique().tolist()
                )
            else:
                lasso_genes = []

            # Keep only genes available in PA matrix
            lasso_genes = [g for g in lasso_genes if g in pa_df.columns]

            if not lasso_genes:
                print("LASSO-selected: no selected genes found in presence/absence matrix; skipping", flush=True)
            else:
                print(f"LASSO-selected: {len(lasso_genes)} genes", flush=True)

                # Build X and y for train+val
                X_ls = pa_df.loc[mask_tv, lasso_genes].astype(float)
                y_ls = y_series.loc[X_ls.index].astype(float).dropna()
                X_ls = X_ls.loc[y_ls.index]

                # Enrichment per gene (Fisher exact) with Haldane-Anscombe correction
                rows = []
                y_bin = y_ls.astype(int).to_numpy()
                for gene in lasso_genes:
                    x = X_ls[gene].fillna(0).astype(int).to_numpy()
                    a = int(((y_bin == 1) & (x == 1)).sum())
                    b = int(((y_bin == 0) & (x == 1)).sum())
                    c = int(((y_bin == 1) & (x == 0)).sum())
                    d = int(((y_bin == 0) & (x == 0)).sum())
                    try:
                        _, p = fisher_exact([[a, b], [c, d]], alternative="two-sided")
                    except Exception:
                        p = 1.0
                    a2, b2, c2, d2 = a + 0.5, b + 0.5, c + 0.5, d + 0.5
                    or_val = (a2 * d2) / (b2 * c2)
                    log2_or = float(np.log2(or_val))
                    rows.append({
                        "gene_norm": gene,
                        "pos_present": a,
                        "neg_present": b,
                        "pos_absent": c,
                        "neg_absent": d,
                        "odds_ratio": float(or_val),
                        "log2_or": log2_or,
                        "pval": float(p),
                    })
                ls_enr_df = pd.DataFrame(rows)
                ls_enr_df["fdr_bh"] = bh_fdr(ls_enr_df["pval"].to_numpy())
                ls_enr_df = ls_enr_df.sort_values(["fdr_bh", "pval"]).reset_index(drop=True)
                ls_enr_path = analysis_dir / "lasso_selected_enrichment.csv"
                ls_enr_df.to_csv(ls_enr_path, index=False)

                # Volcano plot
                plot_volcano(
                    ls_enr_df,
                    analysis_dir / "figures" / "lasso_selected_volcano.png",
                    title="LASSO-selected genes: enrichment (train+val)",
                    alpha=0.05,
                    abs_log2_or_thr=0.0,
                    label_top=int(args.label_top),
                )

                # Correlation matrix and heatmap
                try:
                    corr_ls = np.corrcoef(X_ls.to_numpy(dtype=float), rowvar=False)
                except Exception:
                    corr_ls = np.array([[]])
                pd.DataFrame(corr_ls, index=lasso_genes, columns=lasso_genes).to_csv(
                    analysis_dir / "lasso_selected_corr.csv"
                )
                plot_corr_heatmap(
                    corr_ls,
                    lasso_genes,
                    analysis_dir / "figures" / "lasso_selected_corr.png",
                    title="LASSO-selected genes: gene correlation (train+val)",
                )

                # Presence plot among sporulating (train+val) samples
                X_pos_ls = X_ls.loc[y_ls == 1]
                tbl_ls = plot_combined_presence_bar(
                    X_pos_ls,
                    analysis_dir / "figures" / "lasso_selected_pos_presence.png",
                    title="LASSO-selected genes: presence in sporulating (train+val)",
                )
                tbl_ls.to_csv(analysis_dir / "lasso_selected_gene_prevalence.csv", index=False)
                try:
                    print(
                        f"LASSO-selected presence: perfect-coverage genes = {(tbl_ls['prevalence']>=1.0).sum()}",
                        flush=True,
                    )
                except Exception:
                    pass
        else:
            print("LASSO feature importance not found; skipping LASSO-selected analysis", flush=True)
    except Exception as e:
        print(f"LASSO-selected analysis failed: {e}", flush=True)

    print("Top groups analysis complete.")


if __name__ == "__main__":
    main()
