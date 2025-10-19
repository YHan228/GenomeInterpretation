#!/usr/bin/env python3
"""Train an L1-penalized Logistic Regression (LASSO) on gene presence/absence.

Inputs (default paths):
- Presence/absence wide matrix: /vol/projects/BIFO/genomenet/yichen/phenotype/data/rfdata/rf_presence_absence.parquet (or .csv)
- Gene-level long table: /vol/projects/BIFO/genomenet/yichen/phenotype/data/rfdata/rf_dataset.parquet (or .csv)

Behavior:
- Builds X/y and splits identically to train_rf.
- HPO (RandomizedSearchCV) over C and class_weight with stratified CV.
- Refit best model on train+val, evaluate on test.
- Computes coefficient-based feature importances (|coef|), saves artifacts.
- Correlates importances with spore-related stats from the long table.

Outputs (written to output dir):
- lasso_hpo_best_params.json
- lasso_metrics.json
- lasso_feature_importance.csv
- lasso_gene_spore_stats.csv
- lasso_importance_spore_correlation.json
- lasso_test_predictions.csv
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold


DATA_ROOT = Path("/vol/projects/BIFO/genomenet/yichen/phenotype/data")

DEFAULT_INPUT_DIR = DATA_ROOT / "rfdata"
DEFAULT_OUTPUT_DIR = DATA_ROOT / "rfdata"


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


def _normalize_gene_name(name: object) -> str:
    if pd.isna(name):
        return ""
    return str(name).strip().lower()


def to_json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def _map_spore_label(val: object) -> Optional[int]:
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


@dataclass
class DataSplits:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: List[str]
    sample_ids: Dict[str, List[str]]


def _aggregate_labels_per_sample(long_df: pd.DataFrame, phenotype: str) -> pd.Series:
    if "sample_id" not in long_df.columns:
        raise ValueError("long_df must contain column 'sample_id'")
    if phenotype not in long_df.columns:
        raise ValueError(f"long_df must contain column '{phenotype}'")

    col = long_df[phenotype]

    def _is_binary_col(series: pd.Series) -> bool:
        vals = series.dropna().map(lambda v: str(v).strip().lower())
        if vals.empty:
            return True
        return set(vals.unique()).issubset({"true", "false"})

    is_binary = _is_binary_col(col)

    def _map_binary_label(val: object) -> Optional[int]:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return None
        s = str(val).strip().lower()
        if s in {"true", "t", "1", "yes"}:
            return 1
        if s in {"false", "f", "0", "no"}:
            return 0
        if s in {"na", "nan", ""}:
            return None
        try:
            f = float(s)
            if f == 0.0:
                return 0
            if f == 1.0:
                return 1
        except Exception:
            pass
        return None

    def resolve(series: pd.Series):
        if is_binary:
            mapped = series.map(_map_binary_label).dropna()
            if mapped.empty:
                return None
            m = float(mapped.mean())
            return int(m >= 0.5)
        vals = series.dropna().astype(str)
        if vals.empty:
            return None
        try:
            return vals.mode().iloc[0]
        except Exception:
            return vals.iloc[0]

    return long_df.groupby("sample_id")[phenotype].apply(resolve)


def build_dataset(pa_df: pd.DataFrame, long_df: pd.DataFrame, phenotype: str = "Spore formation") -> DataSplits:
    if "split" not in pa_df.columns:
        raise ValueError("Presence/absence matrix must include a 'split' column")

    if pa_df.index.name != "sample_id":
        if "sample_id" in pa_df.columns:
            pa_df = pa_df.set_index("sample_id")
        else:
            pa_df.index.name = "sample_id"

    y_series = _aggregate_labels_per_sample(long_df, phenotype)

    splits = pa_df["split"].astype("string")
    X_df = pa_df.drop(columns=["split"])

    common_ids = X_df.index.intersection(y_series.index)
    X_df = X_df.loc[common_ids]
    y_raw = y_series.loc[common_ids]
    y_raw = y_raw.dropna()
    X_df = X_df.loc[y_raw.index]

    # Encode labels to integers: binary stays 0/1; multiclass gets 0..K-1
    def _is_already_binary_int(series: pd.Series) -> bool:
        try:
            u = set(pd.unique(series.astype("int64")))
            return u.issubset({0, 1})
        except Exception:
            return False

    if _is_already_binary_int(y_raw):
        y = y_raw.astype("int32")
    else:
        y_str = y_raw.astype(str)
        classes = sorted(list(pd.unique(y_str)))
        class_to_int = {c: i for i, c in enumerate(classes)}
        y = y_str.map(class_to_int).astype("int32")

    split_vals = splits.loc[X_df.index].fillna("unspecified").astype(str)
    split_vals = split_vals.replace({"unspecified": "train"})

    feature_names = list(X_df.columns)

    train_mask = split_vals == "train"
    val_mask = split_vals == "val"
    test_mask = split_vals == "test"

    X_train = X_df.loc[train_mask].to_numpy(dtype=np.float32)
    y_train = y.loc[train_mask].to_numpy(dtype=np.int32)
    X_val = X_df.loc[val_mask].to_numpy(dtype=np.float32)
    y_val = y.loc[val_mask].to_numpy(dtype=np.int32)
    X_test = X_df.loc[test_mask].to_numpy(dtype=np.float32)
    y_test = y.loc[test_mask].to_numpy(dtype=np.int32)

    return DataSplits(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_names,
        sample_ids={
            "train": list(X_df.index[train_mask]),
            "val": list(X_df.index[val_mask]),
            "test": list(X_df.index[test_mask]),
        },
    )


def filter_low_prevalence_features(splits: DataSplits, min_prevalence: float) -> DataSplits:
    if splits.X_train.size == 0:
        return splits
    prevalence = splits.X_train.mean(axis=0)
    keep_mask = prevalence >= float(min_prevalence)
    num_kept = int(keep_mask.sum())
    if num_kept == 0:
        print("Warning: prevalence filter would remove all features; skipping filter.", flush=True)
        return splits
    before = len(splits.feature_names)
    after = num_kept
    print(f"Feature prevalence filter: kept {after}/{before} ({after/before:.1%}) with min_prev={min_prevalence}", flush=True)

    def apply_mask(X: np.ndarray) -> np.ndarray:
        if X.size == 0:
            return X
        return X[:, keep_mask]

    return DataSplits(
        X_train=apply_mask(splits.X_train),
        y_train=splits.y_train,
        X_val=apply_mask(splits.X_val),
        y_val=splits.y_val,
        X_test=apply_mask(splits.X_test),
        y_test=splits.y_test,
        feature_names=[f for f, k in zip(splits.feature_names, keep_mask) if k],
        sample_ids=splits.sample_ids,
    )


def run_hpo(
    X: np.ndarray,
    y: np.ndarray,
    n_iter: int,
    cv_splits: int,
    seed: int,
    verbose: int,
    c_min: float,
    c_max: float,
) -> Tuple[Dict[str, object], LogisticRegression]:
    base = LogisticRegression(
        solver="saga", penalty="l1", max_iter=5000, tol=1e-3, n_jobs=None,
    )

    # Log-uniform C
    c_values = np.logspace(np.log10(c_min), np.log10(c_max), num=50)
    param_distributions = {
        "C": c_values,
        "class_weight": [None, "balanced"],
        "fit_intercept": [True, False],
    }

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed)

    total_fits = int(n_iter) * int(cv_splits)
    print(
        f"Starting HPO (LASSO): n_iter={n_iter}, cv_splits={cv_splits}, total_fits={total_fits}, "
        f"X.shape={X.shape}, y_pos_rate={(float(y.mean()) if len(y)>0 else float('nan')):.3f}",
        flush=True,
    )

    search = RandomizedSearchCV(
        estimator=base,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="roc_auc",
        n_jobs=-1,
        cv=cv,
        verbose=verbose,
        random_state=seed,
        refit=True,
    )
    search.fit(X, y)
    best_params = dict(search.best_params_)
    best_est: LogisticRegression = search.best_estimator_
    try:
        print(
            f"HPO complete. best_cv_score={search.best_score_:.4f} using params={best_params}",
            flush=True,
        )
    except Exception:
        print("HPO complete.", flush=True)
    return best_params, best_est


def refit_and_evaluate(
    best_params: Dict[str, object],
    splits: DataSplits,
    seed: int,
) -> Tuple[LogisticRegression, Dict[str, object], pd.DataFrame]:
    X_tr = np.concatenate([splits.X_train, splits.X_val], axis=0)
    y_tr = np.concatenate([splits.y_train, splits.y_val], axis=0)

    model = LogisticRegression(
        solver="saga", penalty="l1", max_iter=5000, tol=1e-3, n_jobs=None,
        random_state=seed, **best_params,
    )
    model.fit(X_tr, y_tr)

    metrics: Dict[str, object] = {}
    if len(splits.y_test) > 0:
        y_true = splits.y_test
        y_pred = model.predict(splits.X_test)
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
        metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
        metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
        try:
            y_proba = model.predict_proba(splits.X_test)[:, 1]
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
            metrics["average_precision"] = float(average_precision_score(y_true, y_proba))
        except Exception:
            metrics["roc_auc"] = None
            metrics["average_precision"] = None

        preds_df = pd.DataFrame({
            "sample_id": splits.sample_ids["test"],
            "y_true": y_true,
            "y_pred": y_pred,
        })
        try:
            preds_df["y_proba"] = model.predict_proba(splits.X_test)[:, 1]
        except Exception:
            preds_df["y_proba"] = np.nan
    else:
        preds_df = pd.DataFrame(columns=["sample_id", "y_true", "y_pred", "y_proba"])

    return model, metrics, preds_df


def compute_linear_importance(model: LogisticRegression, feature_names: List[str]) -> pd.DataFrame:
    coef = getattr(model, "coef_", None)
    if coef is None:
        raise ValueError("Model does not expose coef_")
    coef = coef.reshape(-1)
    imp = pd.DataFrame({
        "gene_norm": feature_names,
        "coef": coef,
    })
    imp["importance"] = np.abs(imp["coef"])  # LASSO importance
    imp = imp.sort_values("importance", ascending=False, kind="mergesort").reset_index(drop=True)
    imp["rank"] = np.arange(1, len(imp) + 1)
    return imp


def gene_spore_stats(long_df: pd.DataFrame) -> pd.DataFrame:
    if "gene" not in long_df.columns:
        raise ValueError("long_df must contain column 'gene'")
    if "spore_related" not in long_df.columns:
        raise ValueError("long_df must contain column 'spore_related'")
    df = long_df[["gene", "spore_related"]].copy()
    df["gene_norm"] = df["gene"].map(_normalize_gene_name)
    df = df[df["gene_norm"].str.len() > 0]
    grp = df.groupby("gene_norm")["spore_related"]
    stats_df = pd.DataFrame({
        "n_rows": grp.size(),
        "n_spore_related": grp.sum().astype(int),
    })
    stats_df["frac_spore_related"] = stats_df["n_spore_related"] / stats_df["n_rows"].replace(0, np.nan)
    stats_df["any_spore_related"] = stats_df["n_spore_related"] > 0
    stats_df = stats_df.reset_index()
    return stats_df


def importance_spore_correlation(imp_df: pd.DataFrame, spore_df: pd.DataFrame) -> Dict[str, object]:
    merged = imp_df.merge(spore_df, on="gene_norm", how="left")
    merged["any_spore_related"] = merged["any_spore_related"].fillna(False)
    merged["frac_spore_related"] = merged["frac_spore_related"].fillna(0.0)
    try:
        rho, rho_p = stats.spearmanr(merged["importance"], merged["frac_spore_related"])
    except Exception:
        rho, rho_p = np.nan, np.nan
    try:
        x = merged.loc[merged["any_spore_related"], "importance"].to_numpy()
        y = merged.loc[~merged["any_spore_related"], "importance"].to_numpy()
        if len(x) > 0 and len(y) > 0:
            u, u_p = stats.mannwhitneyu(x, y, alternative="greater")
        else:
            u, u_p = np.nan, np.nan
    except Exception:
        u, u_p = np.nan, np.nan
    topk_results: Dict[str, Dict[str, float]] = {}
    for k in [50, 100, 200, 500]:
        top = merged.nsmallest(k, columns=["rank"]) if "rank" in merged.columns else merged.nlargest(k, columns=["importance"])  # rank if present
        if len(top) > 0:
            prec = float(top["any_spore_related"].mean())
        else:
            prec = float("nan")
        topk_results[str(k)] = {"spore_related_fraction": prec}
    return {
        "spearman_rho": float(rho) if rho == rho else None,
        "spearman_p": float(rho_p) if rho_p == rho_p else None,
        "mannwhitney_u": float(u) if u == u else None,
        "mannwhitney_p": float(u_p) if u_p == u_p else None,
        "topk_spore_enrichment": topk_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LASSO (logistic L1) on gene presence/absence and analyze importances")
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT_DIR, help="Directory containing rf_presence_absence and rf_dataset files")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory to write artifacts")
    parser.add_argument("--clustered", action="store_true", help="Flag accepted for consistency; training remains gene-level")
    parser.add_argument("--cluster_threshold", type=float, default=0.7, help="Accepted for consistency; not used during training")
    parser.add_argument("--n_iter", type=int, default=50, help="RandomizedSearchCV iterations")
    parser.add_argument("--cv", type=int, default=5, help="Inner CV splits for HPO")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--hpo_verbose", type=int, default=3, help="Verbosity for RandomizedSearchCV (0-3)")
    parser.add_argument("--min_prev", type=float, default=0.02, help="Drop features with training prevalence below this fraction")
    parser.add_argument("--c_min", type=float, default=1e-3, help="Lower bound for C in HPO (log-uniform grid)")
    parser.add_argument("--c_max", type=float, default=1e+2, help="Upper bound for C in HPO (log-uniform grid)")
    args = parser.parse_args()

    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load datasets
    pa_path = input_dir / "rf_presence_absence.parquet"
    long_path = input_dir / "rf_dataset.parquet"
    pa_df = _read_table(pa_path)
    long_df = _read_table(long_path)

    # Build dataset
    splits = build_dataset(pa_df, long_df)
    print(
        f"Dataset: X_train={splits.X_train.shape}, X_val={splits.X_val.shape}, X_test={splits.X_test.shape}; "
        f"y_train_pos={(float(splits.y_train.mean()) if len(splits.y_train)>0 else float('nan')):.3f}",
        flush=True,
    )

    # Prevalence-based feature filtering
    splits = filter_low_prevalence_features(splits, min_prevalence=args.min_prev)

    # HPO
    print(
        f"HPO space (LASSO): C∈[ {args.c_min:g} , {args.c_max:g} ], class_weight∈{[None, 'balanced']}",
        flush=True,
    )
    best_params, best_cv_est = run_hpo(
        splits.X_train,
        splits.y_train,
        n_iter=args.n_iter,
        cv_splits=args.cv,
        seed=args.seed,
        verbose=args.hpo_verbose,
        c_min=float(args.c_min),
        c_max=float(args.c_max),
    )

    # Refit on train+val, evaluate on test
    final_model, metrics, preds_df = refit_and_evaluate(
        best_params,
        splits,
        seed=args.seed,
    )

    # Feature importance
    imp_df = compute_linear_importance(final_model, splits.feature_names)

    # Spore-related stats and correlation
    spore_df = gene_spore_stats(long_df)
    imp_df_with_rank = imp_df.copy()
    corr_summary = importance_spore_correlation(imp_df_with_rank, spore_df)

    # Save artifacts
    with (output_dir / "lasso_hpo_best_params.json").open("w") as f:
        json.dump(to_json_safe(best_params), f, indent=2)

    with (output_dir / "lasso_metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    imp_df.to_csv(output_dir / "lasso_feature_importance.csv", index=False)
    spore_df.to_csv(output_dir / "lasso_gene_spore_stats.csv", index=False)

    with (output_dir / "lasso_importance_spore_correlation.json").open("w") as f:
        json.dump(corr_summary, f, indent=2)

    preds_df.to_csv(output_dir / "lasso_test_predictions.csv", index=False)

    print("Saved LASSO artifacts to:", str(output_dir))


if __name__ == "__main__":
    main()

