#!/usr/bin/env python3
"""Train a Random Forest using only genes from the top-N grouped-PI groups.

Inputs (defaults):
- Presence/absence: /vol/projects/BIFO/genomenet/yichen/phenotype/data/rfdata/rf_presence_absence.parquet (or .csv)
- Long table: /vol/projects/BIFO/genomenet/yichen/phenotype/data/rfdata/rf_dataset.parquet (or .csv)
- Analysis dir (grouped PI CSVs): /vol/projects/BIFO/genomenet/yichen/phenotype/data/rfdata/analysis
  - rf_grouped_permutation_groups_top*.csv (group_id, gpi_mean, gpi_std, group_size, rank)
  - rf_grouped_permutation_membership_top*.csv (gene_norm, group_id, group_size)
- RF best params: /vol/projects/BIFO/genomenet/yichen/phenotype/data/rfdata/rf_hpo_best_params.json (optional)

Behavior:
- Auto-discovers the latest grouped-PI CSVs in analysis_dir unless provided explicitly
- Selects top-N groups by rank (default N=2), gets their genes
- Builds dataset restricted to those genes, trains RF on train+val, evaluates on test
- Saves metrics, feature importances (for the selected genes), and test predictions

Outputs (written to output_dir):
- rf_top2groups_metrics.json
- rf_top2groups_feature_importance.csv
- rf_top2groups_test_predictions.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)


DATA_ROOT = Path("/vol/projects/BIFO/genomenet/yichen/phenotype/data")
DEFAULT_INPUT_DIR = DATA_ROOT / "rfdata"
DEFAULT_ANALYSIS_DIR = DATA_ROOT / "rfdata" / "analysis"
DEFAULT_OUTPUT_DIR = DATA_ROOT / "rfdata" / "analysis"


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


def _safe_read_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    with path.open("r") as f:
        return json.load(f)


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


def aggregate_labels_per_sample(long_df: pd.DataFrame) -> pd.Series:
    if "sample_id" not in long_df.columns:
        raise ValueError("long_df must contain column 'sample_id'")
    if "Spore formation" not in long_df.columns:
        raise ValueError("long_df must contain column 'Spore formation'")
    def resolve(series: pd.Series) -> Optional[int]:
        mapped = series.map(_map_spore_label).dropna()
        if mapped.empty:
            return None
        return int(float(mapped.mean()) >= 0.5)
    return long_df.groupby("sample_id")["Spore formation"].apply(resolve)


def discover_group_csvs(analysis_dir: Path) -> Tuple[Path, Path]:
    import re
    group_files = list(analysis_dir.glob("rf_grouped_permutation_groups_top*.csv"))
    member_files = list(analysis_dir.glob("rf_grouped_permutation_membership_top*.csv"))
    if not group_files or not member_files:
        raise FileNotFoundError("Grouped PI CSVs not found in analysis_dir")

    def extract_top_k(p: Path) -> Optional[int]:
        m = re.search(r"_top(\d+)\.csv$", p.name)
        return int(m.group(1)) if m else None

    k_to_group: Dict[int, Path] = {}
    for p in group_files:
        k = extract_top_k(p)
        if k is not None:
            k_to_group[k] = p

    k_to_member: Dict[int, Path] = {}
    for p in member_files:
        k = extract_top_k(p)
        if k is not None:
            k_to_member[k] = p

    common_ks = sorted(set(k_to_group.keys()) & set(k_to_member.keys()))
    if not common_ks:
        # Fallback to latest by mtime if no matching Ks
        g = max(group_files, key=lambda p: p.stat().st_mtime)
        m = max(member_files, key=lambda p: p.stat().st_mtime)
        return g, m

    best_k = max(common_ks)
    return k_to_group[best_k], k_to_member[best_k]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RF on genes from top grouped-PI groups")
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT_DIR, help="Directory with rf_presence_absence and rf_dataset files")
    parser.add_argument("--analysis_dir", type=Path, default=DEFAULT_ANALYSIS_DIR, help="Directory with grouped PI CSVs")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory to write outputs")
    parser.add_argument("--groups_csv", type=Path, default=None, help="Path to rf_grouped_permutation_groups_top*.csv")
    parser.add_argument("--membership_csv", type=Path, default=None, help="Path to rf_grouped_permutation_membership_top*.csv")
    parser.add_argument("--top_groups", type=int, default=2, help="Number of top groups to use (default=2)")
    parser.add_argument("--refit_n_jobs", type=int, default=-1, help="Threads for RF refit")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    input_dir: Path = args.input_dir
    analysis_dir: Path = args.analysis_dir
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover grouped PI CSVs
    groups_csv = args.groups_csv if args.groups_csv is not None else None
    members_csv = args.membership_csv if args.membership_csv is not None else None
    if groups_csv is None or members_csv is None:
        g_path, m_path = discover_group_csvs(analysis_dir)
        groups_csv = groups_csv or g_path
        members_csv = members_csv or m_path

    groups_df = pd.read_csv(groups_csv)
    members_df = pd.read_csv(members_csv)
    if "rank" not in groups_df.columns:
        # Rank by gpi_mean if rank missing
        groups_df = groups_df.sort_values(["gpi_mean", "group_id"], ascending=[False, True]).reset_index(drop=True)
        groups_df["rank"] = np.arange(1, len(groups_df) + 1)
    groups_df = groups_df.sort_values("rank", ascending=True)
    top_groups = groups_df.head(int(args.top_groups))["group_id"].astype(int).tolist()

    # Genes from top groups
    genes = members_df[members_df["group_id"].isin(top_groups)]["gene_norm"].astype(str).unique().tolist()
    print(f"Top groups={top_groups} contribute {len(genes)} unique genes", flush=True)
    if len(genes) == 0:
        raise SystemExit("No genes found for selected groups")

    # Load PA subset (split + selected genes)
    pa_path = input_dir / "rf_presence_absence.parquet"
    try:
        pa_df = _read_table(pa_path)
    except Exception:
        raise SystemExit(f"Cannot read presence/absence matrix at {pa_path}")
    if pa_df.index.name != "sample_id":
        if "sample_id" in pa_df.columns:
            pa_df = pa_df.set_index("sample_id")
        else:
            pa_df.index.name = "sample_id"
    keep_cols = ["split"] + [g for g in genes if g in pa_df.columns]
    missing = set(genes) - set(keep_cols)
    if missing:
        print(f"Warning: {len(missing)} genes missing from PA; using {len(keep_cols)-1} genes", flush=True)
    pa_sub = pa_df[keep_cols].copy()

    # Labels from long table
    long_path = input_dir / "rf_dataset.parquet"
    try:
        long_df = _read_table(long_path)
    except Exception:
        raise SystemExit(f"Cannot read long dataset at {long_path}")
    y_series = aggregate_labels_per_sample(long_df)

    # Align and split
    splits = pa_sub["split"].astype("string")
    X_df = pa_sub.drop(columns=["split"]).astype(float)
    common_ids = X_df.index.intersection(y_series.index)
    X_df = X_df.loc[common_ids]
    y = y_series.loc[common_ids].astype(float).dropna()
    X_df = X_df.loc[y.index]
    split_vals = splits.loc[X_df.index].fillna("unspecified").astype(str).replace({"unspecified": "train"})
    train_mask = (split_vals == "train") | (split_vals == "val")
    test_mask = split_vals == "test"
    X_tr = X_df.loc[train_mask].to_numpy(dtype=np.float32)
    y_tr = y.loc[train_mask].to_numpy(dtype=np.int32)
    X_te = X_df.loc[test_mask].to_numpy(dtype=np.float32)
    y_te = y.loc[test_mask].to_numpy(dtype=np.int32)

    print(
        f"Dataset (top{len(top_groups)} groups): X_tr={X_tr.shape}, X_te={X_te.shape}; y_tr_pos={(float(y_tr.mean()) if len(y_tr)>0 else float('nan')):.3f}",
        flush=True,
    )

    # Load RF params or use defaults
    best_params = _safe_read_json(input_dir / "rf_hpo_best_params.json")
    if not best_params:
        best_params = {
            "n_estimators": 600,
            "max_depth": 30,
            "class_weight": "balanced",
            "max_features": "sqrt",
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "bootstrap": True,
        }

    # Refit RF on train+val
    model = RandomForestClassifier(random_state=int(args.seed), n_jobs=int(args.refit_n_jobs), **best_params)
    model.fit(X_tr, y_tr)

    # Evaluate on test
    metrics: Dict[str, object] = {}
    preds_df = pd.DataFrame(columns=["sample_id", "y_true", "y_pred", "y_proba"]) if len(y_te) == 0 else None
    if len(y_te) > 0:
        y_pred = model.predict(X_te)
        metrics["accuracy"] = float(accuracy_score(y_te, y_pred))
        metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_te, y_pred))
        metrics["f1"] = float(f1_score(y_te, y_pred, zero_division=0))
        try:
            y_proba = model.predict_proba(X_te)[:, 1]
            metrics["roc_auc"] = float(roc_auc_score(y_te, y_proba))
            metrics["average_precision"] = float(average_precision_score(y_te, y_proba))
        except Exception:
            y_proba = np.full_like(y_te, fill_value=np.nan, dtype=float)
            metrics["roc_auc"] = None
            metrics["average_precision"] = None
        preds_df = pd.DataFrame({
            "sample_id": list(X_df.index[test_mask]),
            "y_true": y_te,
            "y_pred": y_pred,
            "y_proba": y_proba,
        })

    # Feature importance for selected genes
    importances = getattr(model, "feature_importances_", None)
    if importances is not None:
        imp_df = pd.DataFrame({
            "gene_norm": list(X_df.columns),
            "importance": importances,
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        imp_df["rank"] = np.arange(1, len(imp_df) + 1)
    else:
        imp_df = pd.DataFrame(columns=["gene_norm", "importance", "rank"])

    # Save artifacts
    (output_dir / "rf_top2groups_metrics.json").write_text(json.dumps(metrics, indent=2))
    imp_df.to_csv(output_dir / "rf_top2groups_feature_importance.csv", index=False)
    if preds_df is not None:
        preds_df.to_csv(output_dir / "rf_top2groups_test_predictions.csv", index=False)

    print(f"Saved RF (top groups) artifacts to: {output_dir}")


if __name__ == "__main__":
    main()
