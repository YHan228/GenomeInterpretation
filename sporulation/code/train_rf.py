#!/usr/bin/env python3
"""Train a Random Forest on gene presence/absence with HPO and analyze importances.

Inputs (default paths):
- Presence/absence wide matrix: /vol/projects/BIFO/genomenet/yichen/phenotype/data/rfdata/rf_presence_absence.parquet (or .csv)
- Gene-level long table: /vol/projects/BIFO/genomenet/yichen/phenotype/data/rfdata/rf_dataset.parquet (or .csv)

Behavior:
- Uses the wide matrix for model training; columns are normalized gene names (binary)
- Uses 'split' column to define train/val/test. 'unspecified' is assigned to train by default
- Label y is built from the long table column 'Spore formation' aggregated per sample
- Runs RandomizedSearchCV for HPO on the training set with stratified CV
- Refits the best params on train+val, evaluates on test
- Computes feature importances and correlates them with spore-related genes (from long table)

Outputs (written to output dir):
- rf_hpo_best_params.json: best hyperparameters
- rf_metrics.json: evaluation metrics on test set
- rf_feature_importance.csv: ranked importance per gene feature
- rf_gene_spore_stats.csv: per-gene spore-related occurrence stats
- rf_importance_spore_correlation.json: correlation/enrichment summaries
- rf_test_predictions.csv: per-sample predictions for the test split
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
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
    # Try both
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
    """Convert numpy/pandas types to JSON-serializable Python natives recursively."""
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
        # Handle pandas NA
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
        # Try numeric
        f = float(s)
        if f == 0.0:
            return 0
        if f == 1.0:
            return 1
    except Exception:
        pass
    return None


def _aggregate_labels_per_sample(long_df: pd.DataFrame) -> pd.Series:
    # Expect columns: sample_id, Spore formation
    if "sample_id" not in long_df.columns:
        raise ValueError("long_df must contain column 'sample_id'")
    if "Spore formation" not in long_df.columns:
        raise ValueError("long_df must contain column 'Spore formation'")

    def resolve(series: pd.Series) -> Optional[int]:
        mapped = series.map(_map_spore_label).dropna()
        if mapped.empty:
            return None
        # Use majority; tie-break to 1 if mean >= 0.5
        m = float(mapped.mean())
        return int(m >= 0.5)

    by_sample = long_df.groupby("sample_id")["Spore formation"].apply(resolve)
    return by_sample


@dataclass
class DataSplits:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: List[str]
    sample_ids: Dict[str, List[str]]  # keys: train/val/test


def build_dataset(pa_df: pd.DataFrame, long_df: pd.DataFrame) -> DataSplits:
    if "split" not in pa_df.columns:
        raise ValueError("Presence/absence matrix must include a 'split' column")

    # Ensure index is sample_id
    if pa_df.index.name != "sample_id":
        # If sample_id is a column, set it; otherwise, keep integer index
        if "sample_id" in pa_df.columns:
            pa_df = pa_df.set_index("sample_id")
        else:
            pa_df.index.name = "sample_id"

    # y labels from long_df
    y_series = _aggregate_labels_per_sample(long_df)

    # Align and build X, removing split column from features
    splits = pa_df["split"].astype("string")
    X_df = pa_df.drop(columns=["split"])  # features are gene presence columns

    # Intersect samples with labels
    common_ids = X_df.index.intersection(y_series.index)
    X_df = X_df.loc[common_ids]
    y = y_series.loc[common_ids].astype("float").dropna()
    X_df = X_df.loc[y.index]

    # Handle unspecified: treat as train
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
    """Drop features whose prevalence (mean across training samples) is < min_prevalence.

    This uses only the training set to avoid test leakage.
    """
    if splits.X_train.size == 0:
        return splits
    # Compute prevalence on training (features are 0/1 floats)
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
    hpo_n_estimators_max: int,
    hpo_max_depth_cap: int,
    include_max_features_none: bool,
) -> Tuple[Dict[str, object], RandomForestClassifier]:
    # Avoid nested parallelism: let CV parallelize; keep estimator single-threaded
    base = RandomForestClassifier(random_state=seed, n_jobs=1)

    max_features_choices: List[object] = ["sqrt", "log2", 0.2, 0.5]
    if include_max_features_none:
        max_features_choices.append(None)

    param_distributions = {
        "n_estimators": np.arange(100, int(hpo_n_estimators_max) + 1, 50),
        "max_depth": list(np.arange(5, int(hpo_max_depth_cap) + 1, 5)),
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 4, 8],
        "max_features": max_features_choices,
        "bootstrap": [True, False],
        "class_weight": [None, "balanced"],
    }

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed)

    total_fits = int(n_iter) * int(cv_splits)
    print(
        f"Starting HPO: n_iter={n_iter}, cv_splits={cv_splits}, total_fits={total_fits}, "
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
    best_est: RandomForestClassifier = search.best_estimator_
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
    final_n_estimators: int,
    refit_n_jobs: int,
) -> Tuple[RandomForestClassifier, Dict[str, object], pd.DataFrame]:
    # Train on train+val
    X_tr = np.concatenate([splits.X_train, splits.X_val], axis=0)
    y_tr = np.concatenate([splits.y_train, splits.y_val], axis=0)

    refit_params = dict(best_params)
    if int(final_n_estimators) > 0:
        refit_params["n_estimators"] = int(final_n_estimators)
    model = RandomForestClassifier(random_state=seed, n_jobs=int(refit_n_jobs), **refit_params)
    model.fit(X_tr, y_tr)

    metrics: Dict[str, object] = {}
    # Evaluate on test if any samples
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

        # Predictions frame
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


def compute_feature_importance(model: RandomForestClassifier, feature_names: List[str]) -> pd.DataFrame:
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        raise ValueError("Model does not expose feature_importances_")
    imp = pd.DataFrame({
        "gene_norm": feature_names,
        "importance": importances,
    })
    imp = imp.sort_values("importance", ascending=False, kind="mergesort").reset_index(drop=True)
    imp["rank"] = np.arange(1, len(imp) + 1)
    return imp


def gene_spore_stats(long_df: pd.DataFrame) -> pd.DataFrame:
    # Expect columns: gene (raw), spore_related (bool), and sample_id
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

    # Spearman correlation between importance and fraction spore-related
    try:
        rho, rho_p = stats.spearmanr(merged["importance"], merged["frac_spore_related"])
    except Exception:
        rho, rho_p = np.nan, np.nan

    # Mann-Whitney U: importances in spore-related vs others
    try:
        x = merged.loc[merged["any_spore_related"], "importance"].to_numpy()
        y = merged.loc[~merged["any_spore_related"], "importance"].to_numpy()
        if len(x) > 0 and len(y) > 0:
            u, u_p = stats.mannwhitneyu(x, y, alternative="greater")
        else:
            u, u_p = np.nan, np.nan
    except Exception:
        u, u_p = np.nan, np.nan

    # Top-K enrichment
    topk_results: Dict[str, Dict[str, float]] = {}
    for k in [50, 100, 200, 500]:
        top = merged.nsmallest(k, columns=["rank"]) if "rank" in merged.columns else merged.nlargest(k, columns=["importance"])  # rank if present
        if len(top) > 0:
            prec = float(top["any_spore_related"].mean())
        else:
            prec = float("nan")
        topk_results[str(k)] = {"spore_related_fraction": prec}

    summary: Dict[str, object] = {
        "spearman_rho": float(rho) if rho == rho else None,
        "spearman_p": float(rho_p) if rho_p == rho_p else None,
        "mannwhitney_u": float(u) if u == u else None,
        "mannwhitney_p": float(u_p) if u_p == u_p else None,
        "topk_spore_enrichment": topk_results,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Random Forest with HPO on gene presence/absence and analyze importances")
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT_DIR, help="Directory containing rf_presence_absence and rf_dataset files")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory to write artifacts")
    parser.add_argument("--n_iter", type=int, default=50, help="RandomizedSearchCV iterations")
    parser.add_argument("--cv", type=int, default=5, help="Inner CV splits for HPO")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--hpo_verbose", type=int, default=3, help="Verbosity for RandomizedSearchCV (0-3)")
    parser.add_argument("--min_prev", type=float, default=0.02, help="Drop features with training prevalence below this fraction")
    parser.add_argument("--hpo_n_estimators_max", type=int, default=300, help="Upper bound for n_estimators during HPO (speed-up)")
    parser.add_argument("--hpo_max_depth_cap", type=int, default=30, help="Cap for max_depth during HPO (speed-up)")
    parser.add_argument("--hpo_include_max_features_none", action="store_true", help="Allow max_features=None during HPO (slower)")
    parser.add_argument("--final_n_estimators", type=int, default=600, help="n_estimators for final refit (train+val)")
    parser.add_argument("--refit_n_jobs", type=int, default=-1, help="Threads for final refit (-1 uses all)")
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

    # Prevalence-based feature filtering (on training set)
    splits = filter_low_prevalence_features(splits, min_prevalence=args.min_prev)

    # HPO on training set
    print(
        f"HPO space: n_estimators<=%d, max_depth<=%d, max_features choices=%s (include None=%s)" % (
            args.hpo_n_estimators_max,
            args.hpo_max_depth_cap,
            ["sqrt", "log2", 0.2, 0.5] + ([None] if args.hpo_include_max_features_none else []),
            str(bool(args.hpo_include_max_features_none)),
        ),
        flush=True,
    )
    best_params, best_cv_est = run_hpo(
        splits.X_train,
        splits.y_train,
        n_iter=args.n_iter,
        cv_splits=args.cv,
        seed=args.seed,
        verbose=args.hpo_verbose,
        hpo_n_estimators_max=args.hpo_n_estimators_max,
        hpo_max_depth_cap=args.hpo_max_depth_cap,
        include_max_features_none=bool(args.hpo_include_max_features_none),
    )

    # Refit on train+val, evaluate on test
    final_model, metrics, preds_df = refit_and_evaluate(
        best_params,
        splits,
        seed=args.seed,
        final_n_estimators=args.final_n_estimators,
        refit_n_jobs=args.refit_n_jobs,
    )

    # Feature importance
    imp_df = compute_feature_importance(final_model, splits.feature_names)

    # Spore-related stats and correlation
    spore_df = gene_spore_stats(long_df)
    # Merge rank before correlation
    imp_df_with_rank = imp_df.copy()
    corr_summary = importance_spore_correlation(imp_df_with_rank, spore_df)

    # Save artifacts
    with (output_dir / "rf_hpo_best_params.json").open("w") as f:
        json.dump(to_json_safe(best_params), f, indent=2)

    with (output_dir / "rf_metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    imp_df.to_csv(output_dir / "rf_feature_importance.csv", index=False)
    spore_df.to_csv(output_dir / "rf_gene_spore_stats.csv", index=False)

    with (output_dir / "rf_importance_spore_correlation.json").open("w") as f:
        json.dump(corr_summary, f, indent=2)

    preds_df.to_csv(output_dir / "rf_test_predictions.csv", index=False)

    print("Saved artifacts to:", str(output_dir))


if __name__ == "__main__":
    main()
