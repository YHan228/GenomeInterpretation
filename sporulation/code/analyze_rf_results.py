#!/usr/bin/env python3
"""Analyze and visualize Random Forest training outputs.

Reads model artifacts and datasets from rfdata/, computes additional stats,
and writes figures and summaries into rfdata/analysis/.

Inputs (default paths):
- /vol/projects/BIFO/genomenet/yichen/phenotype/data/rfdata/rf_feature_importance.{csv}
- /vol/projects/BIFO/genomenet/yichen/phenotype/data/rfdata/rf_gene_spore_stats.{csv}
- /vol/projects/BIFO/genomenet/yichen/phenotype/data/rfdata/rf_metrics.json
- /vol/projects/BIFO/genomenet/yichen/phenotype/data/rfdata/rf_hpo_best_params.json
- /vol/projects/BIFO/genomenet/yichen/phenotype/data/rfdata/rf_test_predictions.{csv}
- /vol/projects/BIFO/genomenet/yichen/phenotype/data/rfdata/rf_presence_absence.{parquet|csv}
- /vol/projects/BIFO/genomenet/yichen/phenotype/data/rfdata/rf_dataset.{parquet|csv}

Outputs:
- rf_summary.json
- figures/*.png (feature importance, correlation, ROC, PR, distributions)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for batch jobs
import matplotlib.pyplot as plt


DATA_ROOT = Path("/vol/projects/BIFO/genomenet/yichen/phenotype/data")
DEFAULT_INPUT_DIR = DATA_ROOT / "rfdata"
DEFAULT_OUTPUT_DIR = DATA_ROOT / "rfdata" / "analysis"


def _read_table(path: Path, columns: Optional[list[str]] = None) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path, columns=columns) if columns else pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, usecols=columns) if columns else pd.read_csv(path)
    # Try both
    parquet_path = path.with_suffix(".parquet")
    csv_path = path.with_suffix(".csv")
    if parquet_path.exists():
        return pd.read_parquet(parquet_path, columns=columns) if columns else pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path, usecols=columns) if columns else pd.read_csv(csv_path)
    raise FileNotFoundError(f"Could not find {path} (.parquet or .csv)")


def _safe_read_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    with path.open("r") as f:
        return json.load(f)


def ensure_dirs(out_dir: Path) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    return out_dir, fig_dir


def _normalize_gene_name(val: object) -> str:
    if pd.isna(val):
        return ""
    return str(val).strip().lower()


def plot_top_importances(
    imp_df: pd.DataFrame,
    spore_df: Optional[pd.DataFrame],
    top_k: int,
    fig_path: Path,
    dpi: int = 200,
    value_col: str = "importance",
    rank_by_col: Optional[str] = None,
    title: Optional[str] = None,
    color_mode: str = "spore",  # "spore" or "sign"
) -> None:
    df = imp_df.copy()
    if spore_df is not None and not spore_df.empty:
        df = df.merge(spore_df[["gene_norm", "any_spore_related", "frac_spore_related"]], on="gene_norm", how="left")
    else:
        df["any_spore_related"] = False
        df["frac_spore_related"] = np.nan

    order_col = rank_by_col if rank_by_col is not None else value_col
    top = df.sort_values(order_col, ascending=False).head(top_k)

    plt.figure(figsize=(10, max(4, top_k * 0.25)))
    # Coloring mode
    if color_mode == "sign":
        colors = ["tab:red" if float(v) >= 0 else "tab:purple" for v in top[value_col]]
    else:  # default: color by spore-related
        colors = ["tab:red" if bool(x) else "tab:blue" for x in top["any_spore_related"].fillna(False)]

    plt.barh(top["gene_norm"][::-1], top[value_col][::-1], color=colors[::-1])
    plt.xlabel(value_col)
    plt.ylabel("Gene (normalized)")
    ttl = title if title is not None else f"Top-{top_k} by {order_col}"
    plt.title(ttl)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=dpi)
    plt.close()


def plot_importance_vs_spore_fraction(imp_df: pd.DataFrame, spore_df: Optional[pd.DataFrame], fig_path: Path, dpi: int = 200) -> None:
    if spore_df is None or spore_df.empty:
        return
    df = imp_df.merge(spore_df[["gene_norm", "frac_spore_related"]], on="gene_norm", how="left")
    df["frac_spore_related"] = df["frac_spore_related"].fillna(0.0)
    plt.figure(figsize=(7, 5))
    plt.scatter(df["frac_spore_related"], df["importance"], s=6, alpha=0.6)
    plt.xlabel("Fraction of annotations spore-related (per gene)")
    plt.ylabel("Feature importance")
    plt.title("Importance vs. spore-related fraction")
    # Spearman correlation
    try:
        from scipy import stats
        rho, p = stats.spearmanr(df["frac_spore_related"], df["importance"], nan_policy="omit")
        plt.annotate(f"Spearman rho={rho:.3f}, p={p:.1e}", xy=(0.02, 0.98), xycoords="axes fraction", va="top")
    except Exception:
        pass
    plt.tight_layout()
    plt.savefig(fig_path, dpi=dpi)
    plt.close()


def plot_importance_boxplot_spore(
    imp_df: pd.DataFrame,
    spore_df: Optional[pd.DataFrame],
    fig_path: Path,
    dpi: int = 200,
    value_col: str = "importance",
    title: Optional[str] = None,
    nonzero_only: bool = False,
    zero_thresh: float = 0.0,
) -> None:
    if spore_df is None or spore_df.empty:
        return
    df = imp_df.merge(spore_df[["gene_norm", "any_spore_related"]], on="gene_norm", how="left")
    df["any_spore_related"] = df["any_spore_related"].fillna(False)
    vals_spore = df.loc[df["any_spore_related"], value_col].to_numpy()
    vals_non = df.loc[~df["any_spore_related"], value_col].to_numpy()
    if nonzero_only:
        vals_spore = vals_spore[np.abs(vals_spore) > zero_thresh]
        vals_non = vals_non[np.abs(vals_non) > zero_thresh]
    if len(vals_spore) == 0 or len(vals_non) == 0:
        return
    plt.figure(figsize=(6, 5))
    plt.boxplot(
        [vals_spore, vals_non],
        labels=["Spore-related", "Non-spore"],
        showfliers=False,
    )
    plt.ylabel(value_col)
    ttl = title if title is not None else f"Distribution of {value_col}: spore-related vs non-spore"
    plt.title(ttl)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=dpi)
    plt.close()


def plot_coef_density_spore(
    imp_df: pd.DataFrame,
    spore_df: Optional[pd.DataFrame],
    fig_path: Path,
    value_col: str = "coef",
    exclude_zero: bool = True,
    zero_thresh: float = 0.0,
    dpi: int = 200,
) -> None:
    if spore_df is None or spore_df.empty:
        return
    if value_col not in imp_df.columns:
        return
    df = imp_df.merge(spore_df[["gene_norm", "any_spore_related"]], on="gene_norm", how="left")
    df["any_spore_related"] = df["any_spore_related"].fillna(False)
    vals_spore = df.loc[df["any_spore_related"], value_col].astype(float).to_numpy()
    vals_non = df.loc[~df["any_spore_related"], value_col].astype(float).to_numpy()
    if exclude_zero:
        vals_spore = vals_spore[np.abs(vals_spore) > zero_thresh]
        vals_non = vals_non[np.abs(vals_non) > zero_thresh]
    if len(vals_spore) == 0 or len(vals_non) == 0:
        return
    vmin = float(min(vals_spore.min(), vals_non.min()))
    vmax = float(max(vals_spore.max(), vals_non.max()))
    span = vmax - vmin
    if span <= 0:
        return
    xs = np.linspace(vmin - 0.05 * span, vmax + 0.05 * span, 512)
    use_kde = True
    try:
        from scipy.stats import gaussian_kde
        kde_spore = gaussian_kde(vals_spore)
        kde_non = gaussian_kde(vals_non)
        ys_spore = kde_spore(xs)
        ys_non = kde_non(xs)
    except Exception:
        use_kde = False
    plt.figure(figsize=(7, 5))
    if use_kde:
        plt.plot(xs, ys_spore, label="Spore-related", color="tab:red")
        plt.plot(xs, ys_non, label="Non-spore", color="tab:blue")
    else:
        plt.hist(vals_spore, bins=60, density=True, alpha=0.5, color="tab:red", label="Spore-related")
        plt.hist(vals_non, bins=60, density=True, alpha=0.5, color="tab:blue", label="Non-spore")
    plt.xlabel(value_col)
    plt.ylabel("Density")
    plt.title("LASSO coefficients density by spore-related status")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=dpi)
    plt.close()


def compute_lasso_rf_rank_correlation(
    lasso_imp_df: pd.DataFrame,
    rf_imp_df: pd.DataFrame,
    rf_top_k: Optional[int] = None,
) -> Dict[str, object]:
    """Compute rank correlation between LASSO nonzero genes and top-N RF genes (N=#LASSO nonzero).

    - LASSO ranks are taken from the 'rank' column (based on |coef|) among nonzero coefficients.
    - RF ranks are taken from 'rank' among the top N features.
    """
    result: Dict[str, object] = {
        "n_lasso_nonzero": 0,
        "n_rf_top": 0,
        "rf_top_k_requested": rf_top_k,
        "n_overlap": 0,
        "spearman_rho": None,
        "spearman_p": None,
    }
    if lasso_imp_df.empty or rf_imp_df.empty:
        return result

    # Identify nonzero LASSO genes
    if "coef" in lasso_imp_df.columns:
        lasso_nz = lasso_imp_df[lasso_imp_df["coef"] != 0].copy()
    else:
        # Fallback: use importance>0
        if "importance" not in lasso_imp_df.columns:
            return result
        lasso_nz = lasso_imp_df[lasso_imp_df["importance"] > 0].copy()

    if lasso_nz.empty:
        return result
    n_lasso = int(lasso_nz.shape[0])
    result["n_lasso_nonzero"] = n_lasso

    # Ensure ranks exist
    if "rank" not in lasso_nz.columns:
        # Build rank by descending |coef| or importance
        if "coef" in lasso_nz.columns:
            lasso_nz = lasso_nz.assign(_abs=np.abs(lasso_nz["coef"]))
            lasso_nz = lasso_nz.sort_values("_abs", ascending=False).reset_index(drop=True)
        else:
            lasso_nz = lasso_nz.sort_values("importance", ascending=False).reset_index(drop=True)
        lasso_nz["rank"] = np.arange(1, len(lasso_nz) + 1)

    if rf_top_k is None or int(rf_top_k) <= 0:
        rf_top_k_eff = n_lasso
    else:
        rf_top_k_eff = int(rf_top_k)
    rf_top = rf_imp_df.sort_values("importance", ascending=False).head(rf_top_k_eff).copy()
    result["n_rf_top"] = int(rf_top.shape[0])

    # Prepare ranks for merging
    lasso_rank = lasso_nz[["gene_norm", "rank"]].rename(columns={"rank": "lasso_rank"})
    rf_rank = rf_top[["gene_norm", "rank"]].rename(columns={"rank": "rf_rank"})
    merged = lasso_rank.merge(rf_rank, on="gene_norm", how="inner")
    result["n_overlap"] = int(merged.shape[0])

    if merged.shape[0] >= 2:
        try:
            from scipy import stats
            rho, p = stats.spearmanr(merged["lasso_rank"], merged["rf_rank"])
            result["spearman_rho"] = float(rho)
            result["spearman_p"] = float(p)
        except Exception:
            result["spearman_rho"] = None
            result["spearman_p"] = None

    return result


def build_corr_groups(X: np.ndarray, feature_names: List[str], corr_thresh: float) -> List[List[int]]:
    """Build correlation-based feature groups via connectivity on |corr| >= corr_thresh.

    Returns list of groups, each as a list of column indices in X.
    """
    if X.size == 0 or X.shape[1] == 0:
        return []
    # Compute column-wise Pearson correlation matrix (binary features OK)
    with np.errstate(invalid="ignore"):
        corr = np.corrcoef(X, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    n = corr.shape[0]
    adj = (corr >= float(corr_thresh))
    # Ensure symmetry and diagonal True
    np.fill_diagonal(adj, True)
    visited = np.zeros(n, dtype=bool)
    groups: List[List[int]] = []
    for i in range(n):
        if visited[i]:
            continue
        # BFS/DFS to collect connected component
        stack = [i]
        comp: List[int] = []
        visited[i] = True
        while stack:
            u = stack.pop()
            comp.append(u)
            neighbors = np.where(adj[u])[0]
            for v in neighbors:
                if not visited[v]:
                    visited[v] = True
                    stack.append(v)
        groups.append(sorted(comp))
    return groups


def grouped_permutation_importance(
    model,
    X_te: np.ndarray,
    y_te: np.ndarray,
    groups: List[List[int]],
    repeats: int,
    scoring: str = "roc_auc",
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    from sklearn.metrics import roc_auc_score, average_precision_score
    rng = np.random.RandomState(random_state)
    # Baseline score
    try:
        proba = model.predict_proba(X_te)[:, 1]
    except Exception:
        proba = model.decision_function(X_te)
    if scoring == "average_precision":
        base = float(average_precision_score(y_te, proba))
    else:
        base = float(roc_auc_score(y_te, proba))
    n_groups = len(groups)
    drops = np.zeros((repeats, n_groups), dtype=float)
    for r in range(repeats):
        perm = rng.permutation(X_te.shape[0])
        for gi, idxs in enumerate(groups):
            Xp = X_te.copy()
            Xp[:, idxs] = Xp[perm][:, idxs]
            try:
                proba_p = model.predict_proba(Xp)[:, 1]
            except Exception:
                proba_p = model.decision_function(Xp)
            if scoring == "average_precision":
                score_p = float(average_precision_score(y_te, proba_p))
            else:
                score_p = float(roc_auc_score(y_te, proba_p))
            drops[r, gi] = base - score_p
    return drops.mean(axis=0), drops.std(axis=0)


def compute_lasso_vs_group_rank_unique(
    lasso_imp_df: pd.DataFrame,
    membership_df: pd.DataFrame,
    group_summary_df: pd.DataFrame,
) -> Dict[str, object]:
    """Spearman between LASSO nonzero ranks and grouped-PI group ranks (unique per group).

    Maps each gene to its group rank; overlaps are genes present in membership_df.
    """
    out: Dict[str, object] = {"n_overlap": 0, "spearman_rho": None, "spearman_p": None}
    if lasso_imp_df.empty or membership_df.empty or group_summary_df.empty:
        return out
    # LASSO nonzero with ranks
    if "coef" in lasso_imp_df.columns:
        lasso_nz = lasso_imp_df[lasso_imp_df["coef"] != 0].copy()
    else:
        if "importance" not in lasso_imp_df.columns:
            return out
        lasso_nz = lasso_imp_df[lasso_imp_df["importance"] > 0].copy()
    if lasso_nz.empty:
        return out
    if "rank" not in lasso_nz.columns:
        if "coef" in lasso_nz.columns:
            lasso_nz = lasso_nz.assign(_abs=np.abs(lasso_nz["coef"]))
            lasso_nz = lasso_nz.sort_values("_abs", ascending=False).reset_index(drop=True)
        else:
            lasso_nz = lasso_nz.sort_values("importance", ascending=False).reset_index(drop=True)
        lasso_nz["rank"] = np.arange(1, len(lasso_nz) + 1)

    lasso_rank = lasso_nz[["gene_norm", "rank"]].rename(columns={"rank": "lasso_rank"})
    group_rank = group_summary_df[["group_id", "rank"]].rename(columns={"rank": "gpi_group_rank"})
    merged = lasso_rank.merge(membership_df, on="gene_norm", how="inner").merge(group_rank, on="group_id", how="inner")
    out["n_overlap"] = int(merged.shape[0])
    if merged.shape[0] >= 2:
        try:
            from scipy import stats
            rho, p = stats.spearmanr(merged["lasso_rank"], merged["gpi_group_rank"])
            out["spearman_rho"] = float(rho)
            out["spearman_p"] = float(p)
        except Exception:
            pass
    return out

def build_top_gene_product_table(input_dir: Path, imp_df: pd.DataFrame, spore_df: Optional[pd.DataFrame], top_k: int, mode: str = "light") -> pd.DataFrame:
    """Create a table of top-K genes with importance, rank, spore flags, and product annotations.

    Reads rf_dataset in a streaming manner to avoid high memory, aggregating product annotations
    for genes that appear in the top-K by importance.
    """
    if imp_df.empty:
        return pd.DataFrame(columns=["gene_norm", "importance", "rank", "any_spore_related", "frac_spore_related", "top_product", "product_examples", "n_product_rows"])  # empty

    top = imp_df.sort_values("importance", ascending=False).head(top_k).copy()
    top_genes = list(top["gene_norm"].astype(str))
    top_set = set(g for g in top_genes if g)

    # Join spore flags if available
    if spore_df is not None and not spore_df.empty:
        top = top.merge(spore_df[["gene_norm", "any_spore_related", "frac_spore_related"]], on="gene_norm", how="left")
    else:
        top["any_spore_related"] = False
        top["frac_spore_related"] = np.nan

    # Aggregate product annotations from rf_dataset.parquet
    ds_path = input_dir / "rf_dataset.parquet"
    product_counts: Dict[str, Dict[str, int]] = {g: {} for g in top_set}

    try:
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(ds_path)
        for batch in pf.iter_batches(columns=["gene", "product"], batch_size=50000):
            pdf = batch.to_pandas(types_mapper=pd.ArrowDtype)
            # Normalize gene names
            gnorm = pdf["gene"].astype("string").str.strip().str.lower().fillna("")
            pdf = pdf.assign(gene_norm=gnorm)
            pdf = pdf[pdf["gene_norm"].isin(top_set) & pdf["product"].notna()]
            if pdf.empty:
                continue
            for g, prod in zip(pdf["gene_norm"], pdf["product"].astype("string")):
                d = product_counts[g]
                p = str(prod)
                d[p] = d.get(p, 0) + 1
    except Exception:
        # Fallback: try reading selectively (may be heavy)
        try:
            slim = _read_table(ds_path, columns=["gene", "product"]).copy()
            slim["gene_norm"] = slim["gene"].map(_normalize_gene_name)
            slim = slim[slim["gene_norm"].isin(top_set) & slim["product"].notna()]
            for g, prod in zip(slim["gene_norm"], slim["product"].astype("string")):
                d = product_counts[g]
                p = str(prod)
                d[p] = d.get(p, 0) + 1
        except Exception:
            pass

    # Build product summary columns
    top_products: list[str] = []
    product_examples: list[str] = []
    n_rows_list: list[int] = []
    for g in top["gene_norm"].astype(str):
        counts = product_counts.get(g, {})
        if not counts:
            top_products.append("")
            product_examples.append("")
            n_rows_list.append(0)
            continue
        # Sort products by frequency desc, then name asc for stability
        sorted_items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        top_products.append(sorted_items[0][0])
        product_examples.append("; ".join([p for p, _ in sorted_items[:3]]))
        n_rows_list.append(sum(counts.values()))

    top["top_product"] = top_products
    top["product_examples"] = product_examples
    top["n_product_rows"] = n_rows_list

    # If coef is available (LASSO), include it for interpretation; importance is for ranking only
    if "coef" in top.columns:
        cols = ["gene_norm", "coef", "importance", "rank", "any_spore_related", "frac_spore_related", "top_product", "product_examples", "n_product_rows"]
    else:
        cols = ["gene_norm", "importance", "rank", "any_spore_related", "frac_spore_related", "top_product", "product_examples", "n_product_rows"]
    return top[cols]


def plot_split_counts(pa_df: pd.DataFrame, fig_path: Path, dpi: int = 200) -> None:
    if "split" not in pa_df.columns:
        return
    counts = pa_df["split"].value_counts(dropna=False).sort_index()
    plt.figure(figsize=(6, 4))
    counts.plot(kind="bar")
    plt.ylabel("Num samples")
    plt.title("Samples per split")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=dpi)
    plt.close()


def plot_sample_gene_counts(pa_df: pd.DataFrame, fig_path: Path, dpi: int = 200) -> None:
    # Sum features per sample (exclude split column)
    feat_cols = [c for c in pa_df.columns if c != "split"]
    if not feat_cols:
        return
    try:
        gene_counts = pa_df[feat_cols].sum(axis=1)
    except Exception:
        # If non-numeric slips in
        gene_counts = pa_df[feat_cols].select_dtypes(include=[np.number]).sum(axis=1)
    plt.figure(figsize=(7, 5))
    plt.hist(gene_counts, bins=50, color="tab:gray")
    plt.xlabel("Num present genes per sample")
    plt.ylabel("Num samples")
    plt.title("Distribution of present genes per sample")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=dpi)
    plt.close()


def plot_roc_pr_curves(preds_df: pd.DataFrame, fig_dir: Path, prefix: str, dpi: int = 200) -> Dict[str, float]:
    from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
    metrics: Dict[str, float] = {}
    if {"y_true", "y_proba"}.issubset(preds_df.columns):
        y_true = preds_df["y_true"].to_numpy()
        y_proba = preds_df["y_proba"].to_numpy()
        # ROC
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = float(auc(fpr, tpr))
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f"ROC AUC={roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], "--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC curve (test)")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(fig_dir / f"{prefix}_roc_curve.png", dpi=dpi)
        plt.close()
        metrics["roc_auc"] = roc_auc
        # PR
        prec, rec, _ = precision_recall_curve(y_true, y_proba)
        ap = float(average_precision_score(y_true, y_proba))
        plt.figure(figsize=(6, 5))
        plt.plot(rec, prec, label=f"AP={ap:.3f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall curve (test)")
        plt.legend(loc="lower left")
        plt.tight_layout()
        plt.savefig(fig_dir / f"{prefix}_pr_curve.png", dpi=dpi)
        plt.close()
        metrics["average_precision"] = ap
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze RF outputs and visualize results")
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT_DIR, help="Directory with RF outputs and datasets")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory to write analysis outputs")
    parser.add_argument("--top_k", type=int, default=30, help="Top-K features to plot in the bar chart")
    parser.add_argument("--dpi", type=int, default=200, help="Figure DPI")
    parser.add_argument("--mode", type=str, choices=["light", "full"], default="light", help="light: minimal memory; full: read large matrices for extra plots")
    # Permutation importance (RF) on refit model using top-K MDI features
    parser.add_argument("--pi_top_k", type=int, default=1000, help="Top-K MDI features to compute permutation importance on")
    parser.add_argument("--pi_repeats", type=int, default=3, help="Permutation importance repeats")
    parser.add_argument("--pi_n_jobs", type=int, default=-1, help="Threads for permutation importance and refit")
    parser.add_argument("--pi_group_corr", type=float, default=0.8, help="Absolute correlation threshold to define groups for grouped PI")
    args = parser.parse_args()

    input_dir: Path = args.input_dir
    output_dir, fig_dir = ensure_dirs(args.output_dir)

    # Load artifacts (best-effort)
    # RF artifacts
    try:
        imp_df = pd.read_csv(input_dir / "rf_feature_importance.csv")
    except Exception:
        imp_df = pd.DataFrame(columns=["gene_norm", "importance", "rank"])  # empty
    try:
        rf_preds_df = pd.read_csv(input_dir / "rf_test_predictions.csv")
    except Exception:
        rf_preds_df = pd.DataFrame(columns=["sample_id", "y_true", "y_pred", "y_proba"])  # empty

    try:
        spore_df = pd.read_csv(input_dir / "rf_gene_spore_stats.csv")
    except Exception:
        spore_df = pd.DataFrame(columns=["gene_norm", "any_spore_related", "frac_spore_related"])  # empty

    metrics_json = _safe_read_json(input_dir / "rf_metrics.json")
    best_params_json = _safe_read_json(input_dir / "rf_hpo_best_params.json")

    # LASSO artifacts
    try:
        lasso_imp_df = pd.read_csv(input_dir / "lasso_feature_importance.csv")
    except Exception:
        lasso_imp_df = pd.DataFrame(columns=["gene_norm", "importance", "rank"])  # empty
    try:
        lasso_preds_df = pd.read_csv(input_dir / "lasso_test_predictions.csv")
    except Exception:
        lasso_preds_df = pd.DataFrame(columns=["sample_id", "y_true", "y_pred", "y_proba"])  # empty

    # Datasets for context
    # Read only the 'split' column from PA in light mode to avoid huge memory
    pa_path = input_dir / "rf_presence_absence.parquet"
    if args.mode == "light":
        try:
            pa_df = _read_table(pa_path, columns=["split"])
        except Exception:
            pa_df = pd.DataFrame()
    else:
        # Full mode may still be memory heavy; wrap to degrade gracefully
        try:
            pa_df = _read_table(pa_path)
        except MemoryError:
            print("MemoryError reading presence/absence; falling back to split-only", flush=True)
            try:
                pa_df = _read_table(pa_path, columns=["split"])
            except Exception:
                pa_df = pd.DataFrame()
        except Exception:
            pa_df = pd.DataFrame()

    # For long_df, avoid loading full table in light mode; try to get row count via parquet metadata
    long_path = input_dir / "rf_dataset.parquet"
    long_df = pd.DataFrame()
    num_genes_long: Optional[int] = None
    if args.mode == "full":
        try:
            long_df = _read_table(long_path)
        except MemoryError:
            print("MemoryError reading rf_dataset; skipping load", flush=True)
            long_df = pd.DataFrame()
        except Exception:
            long_df = pd.DataFrame()
    if num_genes_long is None:
        try:
            import pyarrow.parquet as pq
            if long_path.exists():
                pf = pq.ParquetFile(long_path)
                num_genes_long = int(pf.metadata.num_rows)
        except Exception:
            num_genes_long = None

    summary: Dict[str, object] = {
        "metrics": metrics_json,
        "best_params": best_params_json,
        "num_features": int(imp_df.shape[0]) if not imp_df.empty else 0,
        "num_samples_pa": int(pa_df.shape[0]) if not pa_df.empty else 0,
        "num_genes_long": (int(long_df.shape[0]) if not long_df.empty else (num_genes_long if num_genes_long is not None else 0)),
    }
    # Add per-model availability
    summary["has_rf"] = bool(not imp_df.empty)
    summary["has_lasso"] = bool(not lasso_imp_df.empty)

    # Plots
    if not imp_df.empty:
        plot_top_importances(imp_df, spore_df if not spore_df.empty else None, args.top_k, fig_dir / "rf_top_importances.png", dpi=args.dpi, value_col="importance", rank_by_col="importance", title="RF: Top feature importances")
        plot_importance_vs_spore_fraction(imp_df, spore_df if not spore_df.empty else None, fig_dir / "rf_importance_vs_spore_fraction.png", dpi=args.dpi)
        plot_importance_boxplot_spore(imp_df, spore_df if not spore_df.empty else None, fig_dir / "rf_importance_spore_boxplot.png", dpi=args.dpi, value_col="importance", title="RF: Importance distribution")

    if not lasso_imp_df.empty:
        # For LASSO, rank by |coef| (importance), but plot signed coef values colored by spore-related
        if "coef" in lasso_imp_df.columns:
            rank_col = "importance" if "importance" in lasso_imp_df.columns else None
            plot_top_importances(
                lasso_imp_df,
                spore_df if not spore_df.empty else None,
                args.top_k,
                fig_dir / "lasso_top_coefficients.png",
                dpi=args.dpi,
                value_col="coef",
                rank_by_col=rank_col if rank_col is not None else "importance",
                title="LASSO: Top coefficients (signed)",
                color_mode="spore",
            )
            # Boxplot on nonzero coefficients only to avoid degenerate zeros
            plot_importance_boxplot_spore(
                lasso_imp_df,
                spore_df if not spore_df.empty else None,
                fig_dir / "lasso_coef_spore_boxplot.png",
                dpi=args.dpi,
                value_col="coef",
                title="LASSO: Coefficient distribution (nonzero)",
                nonzero_only=True,
                zero_thresh=0.0,
            )
            # Overlaid density for signed coefficients
            plot_coef_density_spore(
                lasso_imp_df,
                spore_df if not spore_df.empty else None,
                fig_dir / "lasso_coef_density.png",
                value_col="coef",
                exclude_zero=True,
                zero_thresh=0.0,
                dpi=args.dpi,
            )
        else:
            plot_top_importances(lasso_imp_df, spore_df if not spore_df.empty else None, args.top_k, fig_dir / "lasso_top_importances.png", dpi=args.dpi, value_col="importance", rank_by_col="importance", title="LASSO: Top |coef|", color_mode="spore")
            plot_importance_boxplot_spore(lasso_imp_df, spore_df if not spore_df.empty else None, fig_dir / "lasso_importance_spore_boxplot.png", dpi=args.dpi, value_col="importance", title="LASSO: |coef| distribution (nonzero)", nonzero_only=True, zero_thresh=0.0)

    # Top gene + product tables (RF and LASSO)
    top_table = build_top_gene_product_table(input_dir, imp_df, spore_df if not spore_df.empty else None, args.top_k, mode=args.mode)
    if not top_table.empty:
        top_table_path = output_dir / "rf_top_genes_with_products.csv"
        top_table.to_csv(top_table_path, index=False)
    if not lasso_imp_df.empty:
        lasso_table = build_top_gene_product_table(input_dir, lasso_imp_df, spore_df if not spore_df.empty else None, args.top_k, mode=args.mode)
        if not lasso_table.empty:
            lasso_table_path = output_dir / "lasso_top_genes_with_products.csv"
            lasso_table.to_csv(lasso_table_path, index=False)

    if not pa_df.empty:
        plot_split_counts(pa_df, fig_dir / "split_counts.png", dpi=args.dpi)
        if args.mode == "full":
            try:
                plot_sample_gene_counts(pa_df, fig_dir / "sample_gene_counts.png", dpi=args.dpi)
            except MemoryError:
                print("MemoryError computing sample gene counts; skipping", flush=True)
        try:
            summary["split_counts"] = pa_df["split"].value_counts(dropna=False).to_dict()
        except Exception:
            pass

    # Curves based on test predictions
    if not rf_preds_df.empty:
        curve_metrics_rf = plot_roc_pr_curves(rf_preds_df, fig_dir, prefix="rf", dpi=args.dpi)
        summary.update({f"rf_{k}": v for k, v in curve_metrics_rf.items()})
    if not lasso_preds_df.empty:
        curve_metrics_ls = plot_roc_pr_curves(lasso_preds_df, fig_dir, prefix="lasso", dpi=args.dpi)
        summary.update({f"lasso_{k}": v for k, v in curve_metrics_ls.items()})

    # Count nonzero coefficients by group for LASSO
    if not lasso_imp_df.empty:
        if "coef" in lasso_imp_df.columns:
            lasso_imp_df["nonzero"] = lasso_imp_df["coef"] != 0
            if not spore_df.empty:
                nz = lasso_imp_df.merge(spore_df[["gene_norm", "any_spore_related"]], on="gene_norm", how="left")
                nz["any_spore_related"] = nz["any_spore_related"].fillna(False)
                summary["lasso_nonzero_spore"] = int(nz.loc[nz["any_spore_related"] & nz["nonzero"], "gene_norm"].nunique())
                summary["lasso_nonzero_nonspore"] = int(nz.loc[~nz["any_spore_related"] & nz["nonzero"], "gene_norm"].nunique())
                summary["lasso_total_spore"] = int(nz.loc[nz["any_spore_related"], "gene_norm"].nunique())
                summary["lasso_total_nonspore"] = int(nz.loc[~nz["any_spore_related"], "gene_norm"].nunique())

    # LASSO vs RF MDI rank correlation (not limited to top-K)
    if not lasso_imp_df.empty and not imp_df.empty:
        corr = compute_lasso_rf_rank_correlation(lasso_imp_df, imp_df, rf_top_k=args.pi_top_k)
        summary["lasso_rf_rank_correlation"] = corr

    # ---------------- Permutation importance on top-K MDI features (refit RF) ----------------
    # Refit RF on train+val using top-K MDI features and compute permutation importance on test
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.inspection import permutation_importance
    except Exception:
        RandomForestClassifier = None  # type: ignore[assignment]
        permutation_importance = None  # type: ignore[assignment]

    if (RandomForestClassifier is not None) and (permutation_importance is not None) and (not imp_df.empty):
        # Determine top-K features by MDI
        top_k = int(args.pi_top_k)
        mdi_sorted = imp_df.sort_values("importance", ascending=False)
        top_features: List[str] = list(mdi_sorted["gene_norm"].head(top_k))
        if len(top_features) > 0:
            # Read only split + top features from PA
            try:
                pa_subset = _read_table(pa_path, columns=["split"] + top_features)
            except Exception:
                # Fallback: read PA fully and then subset
                try:
                    pa_all = _read_table(pa_path)
                    keep_cols = [c for c in ["split"] + top_features if c in pa_all.columns]
                    pa_subset = pa_all[keep_cols]
                except Exception:
                    pa_subset = pd.DataFrame()

            # Ensure index is sample_id
            if pa_subset.index.name != "sample_id":
                if "sample_id" in pa_subset.columns:
                    pa_subset = pa_subset.set_index("sample_id")
                else:
                    pa_subset.index.name = "sample_id"

            # Build labels from long table (read only needed columns)
            try:
                long_labels = _read_table(long_path, columns=["sample_id", "Spore formation"])  # type: ignore[arg-type]
            except Exception:
                long_labels = pd.DataFrame()
            if not long_labels.empty:
                # Aggregate labels per sample (majority)
                def _map_lbl(val: object) -> Optional[int]:
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

                def _resolve(series: pd.Series) -> Optional[int]:
                    mapped = series.map(_map_lbl).dropna()
                    if mapped.empty:
                        return None
                    return int(float(mapped.mean()) >= 0.5)

                y_series = long_labels.groupby("sample_id")["Spore formation"].apply(_resolve)

                # Build X/y splits using PA subset
                splits_col = pa_subset["split"].astype("string")
                X_df = pa_subset.drop(columns=["split"]).astype(float)
                common_ids = X_df.index.intersection(y_series.index)
                X_df = X_df.loc[common_ids]
                y = y_series.loc[common_ids].astype(float).dropna()
                X_df = X_df.loc[y.index]
                split_vals = splits_col.loc[X_df.index].fillna("unspecified").astype(str).replace({"unspecified": "train"})
                train_mask = (split_vals == "train") | (split_vals == "val")
                test_mask = split_vals == "test"
                X_tr = X_df.loc[train_mask].to_numpy(dtype=np.float32)
                y_tr = y.loc[train_mask].to_numpy(dtype=np.int32)
                X_te = X_df.loc[test_mask].to_numpy(dtype=np.float32)
                y_te = y.loc[test_mask].to_numpy(dtype=np.int32)

                # Refit RF with saved best params
                rf_params = _safe_read_json(input_dir / "rf_hpo_best_params.json")
                if rf_params:
                    model = RandomForestClassifier(random_state=42, n_jobs=int(args.pi_n_jobs), **rf_params)
                    model.fit(X_tr, y_tr)

                    # Permutation importance on test set
                    try:
                        pi = permutation_importance(
                            model, X_te, y_te,
                            scoring="roc_auc",
                            n_repeats=int(args.pi_repeats),
                            n_jobs=int(args.pi_n_jobs),
                            random_state=42,
                        )
                        pi_df = pd.DataFrame({
                            "gene_norm": list(X_df.columns),
                            "pi_mean": pi.importances_mean,
                            "pi_std": pi.importances_std,
                        })
                        pi_df = pi_df.sort_values("pi_mean", ascending=False).reset_index(drop=True)
                        pi_df["rank"] = np.arange(1, len(pi_df) + 1)

                        # Save PI table
                        pi_path = output_dir / f"rf_permutation_importance_top{len(X_df.columns)}.csv"
                        pi_df.to_csv(pi_path, index=False)

                        # Plots using PI (feature-level)
                        pi_plot_df = pi_df.rename(columns={"pi_mean": "importance"})
                        plot_top_importances(pi_plot_df, spore_df if not spore_df.empty else None, args.top_k, fig_dir / "rf_pi_top_importances.png", dpi=args.dpi, value_col="importance", rank_by_col="importance", title="RF Permutation: Top importances")
                        plot_importance_vs_spore_fraction(pi_plot_df, spore_df if not spore_df.empty else None, fig_dir / "rf_pi_importance_vs_spore_fraction.png", dpi=args.dpi)
                        plot_importance_boxplot_spore(pi_plot_df, spore_df if not spore_df.empty else None, fig_dir / "rf_pi_importance_spore_boxplot.png", dpi=args.dpi, value_col="importance", title="RF Permutation: Importance distribution")

                        # LASSO vs PI rank correlation (use all LASSO nonzero vs top-K PI)
                        corr_pi = compute_lasso_rf_rank_correlation(lasso_imp_df, pi_plot_df, rf_top_k=args.pi_top_k)
                        summary["lasso_pi_rank_correlation"] = corr_pi

                        # ---------------- Grouped Permutation Importance ----------------
                        try:
                            # Build correlation groups on train+val for top features (restricted to top-K MDI)
                            groups = build_corr_groups(X_tr, feature_names=list(X_df.columns), corr_thresh=float(args.pi_group_corr))
                            try:
                                sizes = [len(g) for g in groups]
                                med_sz = int(np.median(sizes)) if sizes else 0
                                max_sz = int(max(sizes)) if sizes else 0
                            except Exception:
                                med_sz, max_sz = 0, 0
                            print(
                                f"Grouped PI: |corr|>={float(args.pi_group_corr):.2f} compressed {len(X_df.columns)} features into {len(groups)} groups (median size={med_sz}, max={max_sz})",
                                flush=True,
                            )

                            # Grouped permutation importance on test (restricted)
                            g_mean, g_std = grouped_permutation_importance(
                                model,
                                X_te,
                                y_te,
                                groups,
                                repeats=int(args.pi_repeats),
                                scoring="roc_auc",
                                random_state=42,
                            )

                            # Build two dataframes: membership (gene->group) and group summary (one row per group)
                            membership_rows = []
                            for gi, idxs in enumerate(groups):
                                for j in idxs:
                                    membership_rows.append({
                                        "gene_norm": X_df.columns[j],
                                        "group_id": int(gi),
                                        "group_size": int(len(idxs)),
                                    })
                            membership_df = pd.DataFrame(membership_rows)

                            group_summary = pd.DataFrame({
                                "group_id": np.arange(len(groups), dtype=int),
                                "gpi_mean": g_mean.astype(float),
                                "gpi_std": g_std.astype(float),
                                "group_size": [len(g) for g in groups],
                            })
                            group_summary = group_summary.sort_values(["gpi_mean", "group_id"], ascending=[False, True]).reset_index(drop=True)
                            group_summary["rank"] = np.arange(1, len(group_summary) + 1)

                            # Save per-group and per-member CSVs (restricted)
                            gpi_groups_path = output_dir / f"rf_grouped_permutation_groups_top{len(X_df.columns)}.csv"
                            group_summary.to_csv(gpi_groups_path, index=False)
                            gpi_membership_path = output_dir / f"rf_grouped_permutation_membership_top{len(X_df.columns)}.csv"
                            membership_df.to_csv(gpi_membership_path, index=False)

                            # For plotting top importances, explode group score to members (so we can color by spore)
                            gpi_df = membership_df.merge(group_summary[["group_id", "gpi_mean"]], on="group_id", how="left").rename(columns={"gpi_mean": "importance"})
                            gpi_df = gpi_df.sort_values(["importance", "group_id"], ascending=[False, True]).reset_index(drop=True)
                            gpi_df["rank"] = np.arange(1, len(gpi_df) + 1)

                            plot_top_importances(gpi_df, spore_df if not spore_df.empty else None, args.top_k, fig_dir / "rf_gpi_top_importances.png", dpi=args.dpi, value_col="importance", rank_by_col="importance", title="RF Grouped Permutation: Top importances")
                            plot_importance_vs_spore_fraction(gpi_df, spore_df if not spore_df.empty else None, fig_dir / "rf_gpi_importance_vs_spore_fraction.png", dpi=args.dpi)
                            plot_importance_boxplot_spore(gpi_df, spore_df if not spore_df.empty else None, fig_dir / "rf_gpi_importance_spore_boxplot.png", dpi=args.dpi, value_col="importance", title="RF Grouped Permutation: Importance distribution")

                            # Overlap of LASSO nonzero with top-1 group and union of top-2 groups (restricted)
                            if not lasso_imp_df.empty:
                                if "coef" in lasso_imp_df.columns:
                                    lasso_set = set(lasso_imp_df.loc[lasso_imp_df["coef"] != 0, "gene_norm"].astype(str))
                                else:
                                    lasso_set = set(lasso_imp_df.loc[lasso_imp_df.get("importance", 0) > 0, "gene_norm"].astype(str))
                                if len(group_summary) >= 1:
                                    gid1 = int(group_summary.iloc[0]["group_id"])
                                    top1_genes = set(membership_df.loc[membership_df["group_id"] == gid1, "gene_norm"].astype(str))
                                    summary["gpi_top1_group_id"] = gid1
                                    summary["gpi_top1_group_size"] = int(len(top1_genes))
                                    summary["lasso_overlap_gpi_top1"] = int(len(lasso_set & top1_genes))
                                if len(group_summary) >= 2:
                                    gid2 = int(group_summary.iloc[1]["group_id"])
                                    top2_genes = set(membership_df.loc[membership_df["group_id"] == gid2, "gene_norm"].astype(str))
                                    union12 = top1_genes | top2_genes
                                    summary["gpi_top2_group_id"] = gid2
                                    summary["gpi_top12_union_size"] = int(len(union12))
                                    summary["lasso_overlap_gpi_top2_union"] = int(len(lasso_set & union12))
                        except Exception as e:
                            print(f"Grouped permutation importance failed: {e}", flush=True)
                    except Exception as e:
                        print(f"Permutation importance failed: {e}", flush=True)

    # Save summary json
    with (output_dir / "rf_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"Analysis complete. Wrote summary to {output_dir / 'rf_summary.json'} and figures to {fig_dir}")


if __name__ == "__main__":
    main()
