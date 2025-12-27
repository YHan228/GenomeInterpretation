#!/usr/bin/env python3
"""Rashomon Set Feature Analysis with Taxonomic Holdout (blocked split).

Implements Rashomon set analysis with phylogenetically-aware validation:
- Entire taxonomic groups (Order/Family) are assigned exclusively to train/val/test
- Prevents phylogenetic leakage and provides honest generalization estimates

Models:
- L1-logistic regression (varying C and seeds)
- Random Forest (varying hyperparameters and seeds)

Identifies features that are:
- Necessary: appear in ALL models within ε of optimal
- Common: appear in majority of models within ε of optimal
- Analyzes variable importance clouds across the Rashomon set

Outputs (PNG/PDF/SVG/JSON/CSV) are written to output_dir.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import joblib
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy import stats
import tempfile

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from train_lasso import _read_table, filter_low_prevalence_features, DataSplits


# ----------------------------- Taxonomic Holdout Split -----------------------------


def create_taxonomic_holdout_split(
    pa_df: pd.DataFrame,
    long_df: pd.DataFrame,
    metadata_path: Path,
    phenotype: str,
    taxon_level: str = "Order",
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    min_samples_per_group: int = 2,
) -> DataSplits:
    """Create train/val/test splits where entire taxonomic groups are held out.

    This prevents phylogenetic leakage by ensuring no taxonomic group appears
    in multiple splits.

    Args:
        pa_df: Presence/absence matrix with sample_id index
        long_df: Long-format gene table with phenotype labels
        metadata_path: Path to metadata Excel with taxonomy columns
        phenotype: Phenotype column name
        taxon_level: Taxonomic level for blocking (Order, Family, Genus)
        val_ratio: Fraction of samples for validation
        test_ratio: Fraction of samples for test
        seed: Random seed for reproducibility
        min_samples_per_group: Minimum samples per taxon group to include

    Returns:
        DataSplits with taxonomically-blocked train/val/test
    """
    rng = np.random.default_rng(seed)

    # Load metadata
    meta_df = pd.read_excel(metadata_path)
    # Normalize binomial name for matching (sample_id in parquet = species name)
    meta_df["name_norm"] = meta_df["Binomial name"].apply(
        lambda x: str(x).strip().lower() if pd.notna(x) else ""
    )

    # Prepare PA matrix
    if pa_df.index.name != "sample_id":
        if "sample_id" in pa_df.columns:
            pa_df = pa_df.set_index("sample_id")
        else:
            pa_df.index.name = "sample_id"

    # Get phenotype labels
    def _map_label(val) -> Optional[int]:
        if pd.isna(val):
            return None
        s = str(val).strip().lower()
        if s in {"true", "t", "1", "yes", "y"}:
            return 1
        if s in {"false", "f", "0", "no", "n"}:
            return 0
        return None

    # Aggregate labels per sample from long_df
    if phenotype in long_df.columns:
        label_agg = long_df.groupby("sample_id")[phenotype].apply(
            lambda x: int(x.map(_map_label).dropna().mean() >= 0.5) if x.map(_map_label).dropna().size > 0 else None
        )
    else:
        raise ValueError(f"Phenotype '{phenotype}' not in long_df")

    # Get splits column if exists, drop it
    if "split" in pa_df.columns:
        pa_df = pa_df.drop(columns=["split"])

    # Match samples to taxonomy via Binomial name (sample_id = species name)
    sample_ids = list(pa_df.index)
    sample_to_taxon = {}
    sample_to_label = {}

    for sid in sample_ids:
        sid_norm = str(sid).strip().lower()
        match = meta_df[meta_df["name_norm"] == sid_norm]

        if len(match) > 0:
            taxon = match.iloc[0].get(taxon_level)
            if pd.notna(taxon):
                sample_to_taxon[sid] = str(taxon)

        if sid in label_agg.index and pd.notna(label_agg[sid]):
            sample_to_label[sid] = int(label_agg[sid])

    # Filter to samples with both taxon and label
    valid_samples = [s for s in sample_ids if s in sample_to_taxon and s in sample_to_label]
    if len(valid_samples) < 50:
        raise ValueError(f"Only {len(valid_samples)} samples with both taxonomy and labels")

    # Group samples by taxon
    taxon_to_samples: Dict[str, List[str]] = {}
    for s in valid_samples:
        t = sample_to_taxon[s]
        taxon_to_samples.setdefault(t, []).append(s)

    # Filter taxa with minimum samples
    taxon_to_samples = {t: samples for t, samples in taxon_to_samples.items()
                        if len(samples) >= min_samples_per_group}

    # Calculate target sizes
    all_valid = [s for samples in taxon_to_samples.values() for s in samples]
    n_total = len(all_valid)
    n_val = int(n_total * val_ratio)
    n_test = int(n_total * test_ratio)

    # Shuffle taxa and assign to splits
    taxa = list(taxon_to_samples.keys())
    rng.shuffle(taxa)

    val_samples, test_samples, train_samples = [], [], []
    val_taxa, test_taxa, train_taxa = [], [], []

    for taxon in taxa:
        samples = taxon_to_samples[taxon]
        if len(val_samples) < n_val:
            val_samples.extend(samples)
            val_taxa.append(taxon)
        elif len(test_samples) < n_test:
            test_samples.extend(samples)
            test_taxa.append(taxon)
        else:
            train_samples.extend(samples)
            train_taxa.append(taxon)

    print(f"[Taxonomic Holdout] Level={taxon_level}")
    print(f"  Train: {len(train_samples)} samples from {len(train_taxa)} {taxon_level}s")
    print(f"  Val:   {len(val_samples)} samples from {len(val_taxa)} {taxon_level}s")
    print(f"  Test:  {len(test_samples)} samples from {len(test_taxa)} {taxon_level}s")
    print(f"  Train taxa: {train_taxa[:5]}{'...' if len(train_taxa) > 5 else ''}")
    print(f"  Val taxa:   {val_taxa}")
    print(f"  Test taxa:  {test_taxa}")

    # Build arrays
    feature_names = [c for c in pa_df.columns if c != "split"]

    def get_arrays(sample_list):
        if not sample_list:
            return np.zeros((0, len(feature_names)), dtype=np.float32), np.zeros(0, dtype=np.int32)
        X = pa_df.loc[sample_list, feature_names].to_numpy(dtype=np.float32)
        y = np.array([sample_to_label[s] for s in sample_list], dtype=np.int32)
        return X, y

    X_train, y_train = get_arrays(train_samples)
    X_val, y_val = get_arrays(val_samples)
    X_test, y_test = get_arrays(test_samples)

    return DataSplits(
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        feature_names=feature_names,
        sample_ids={"train": train_samples, "val": val_samples, "test": test_samples},
    )


# ----------------------------- Utilities -----------------------------


from contextlib import contextmanager


@contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar."""
    class TqdmBatchCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_cb = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_cb
        tqdm_object.close()


def effective_joblib_n_jobs() -> int:
    env_n = os.environ.get("SLURM_CPUS_PER_TASK") or os.environ.get("LOKY_MAX_CPU_COUNT")
    try:
        n = int(env_n) if env_n else (os.cpu_count() or 1)
    except Exception:
        n = os.cpu_count() or 1
    return max(1, int(n))


def save_fig_formats(fig: plt.Figure, base_path: Path, dpi: int = 150) -> None:
    """Save figure in PNG, PDF, and SVG formats."""
    for ext in (".png", ".pdf", ".svg"):
        fig.savefig(Path(base_path).with_suffix(ext), dpi=dpi, bbox_inches="tight")


def atomic_savez_compressed(path: Path, **arrays) -> None:
    """Write NPZ atomically."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=str(target.parent))
    try:
        with os.fdopen(fd, "wb") as f:
            np.savez_compressed(f, **arrays)
        os.replace(tmp_name, target)
    except Exception:
        try:
            os.unlink(tmp_name)
        except Exception:
            pass
        raise


# ----------------------------- Data Structures -----------------------------


@dataclass
class RashomonModel:
    """A single model in the Rashomon set."""
    method: str
    seed: int
    hyperparams: Dict[str, float]
    performance: float
    feature_mask: np.ndarray
    feature_importance: np.ndarray


@dataclass
class RashomonSet:
    """Collection of models within epsilon of optimal."""
    method: str
    epsilon: float
    best_performance: float
    threshold: float
    models: List[RashomonModel] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.models)

    def feature_frequency(self, p: int) -> np.ndarray:
        if not self.models:
            return np.zeros(p)
        return np.vstack([m.feature_mask for m in self.models]).mean(axis=0)


# ----------------------------- Model Fitting -----------------------------


def get_logistic_coef(model: LogisticRegression, p: int) -> np.ndarray:
    """Extract coefficient magnitudes from logistic regression."""
    coef = getattr(model, "coef_", None)
    if coef is None:
        return np.zeros(p)
    coef = np.asarray(coef)
    if coef.ndim == 1:
        coef = coef.reshape(1, -1)
    if coef.shape[1] != p:
        return np.zeros(p)
    return np.abs(coef).max(axis=0)


def fit_logistic_model(X_train, y_train, X_val, y_val, C: float, seed: int) -> RashomonModel:
    """Fit a single L1-logistic model and evaluate."""
    p = X_train.shape[1]
    mc = "multinomial" if len(np.unique(y_train)) > 2 else "auto"
    model = LogisticRegression(
        solver="saga", penalty="l1", C=C, max_iter=5000, tol=1e-4,
        random_state=seed, class_weight="balanced", multi_class=mc,
    )
    try:
        model.fit(X_train, y_train)
        perf = balanced_accuracy_score(y_val, model.predict(X_val))
        coef = get_logistic_coef(model, p)
        mask = coef > 0
    except Exception:
        perf, coef, mask = 0.0, np.zeros(p), np.zeros(p, dtype=bool)

    return RashomonModel("logistic", seed, {"C": C}, perf, mask, coef)


def compute_split_counts(rf: RandomForestClassifier, p: int) -> np.ndarray:
    """Count how many times each feature is used in splits across all trees."""
    counts = np.zeros(p, dtype=int)
    for tree in rf.estimators_:
        features_used = tree.tree_.feature
        # feature == -2 means leaf node (no split)
        for f in features_used:
            if 0 <= f < p:
                counts[f] += 1
    return counts


def fit_rf_model(X_train, y_train, X_val, y_val, n_estimators: int,
                 max_depth: int, min_samples_leaf: int, seed: int, top_k: int) -> RashomonModel:
    """Fit a single RF model and evaluate.

    Feature selection uses split counts (how many times a feature is used across
    all trees) rather than MDI, which is biased toward high-cardinality features.
    A feature is "selected" if it's in the top-k by split count.
    """
    p = X_train.shape[1]
    rf = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
        n_jobs=1, random_state=seed, class_weight="balanced",
    )
    try:
        rf.fit(X_train, y_train)
        perf = balanced_accuracy_score(y_val, rf.predict(X_val))

        # Use split counts for feature selection (less biased than MDI)
        split_counts = compute_split_counts(rf, p)
        mask = np.zeros(p, dtype=bool)
        n_used = (split_counts > 0).sum()
        if n_used > 0:
            k = min(top_k, n_used)
            mask[np.argpartition(split_counts, -k)[-k:]] = True

        # Still store MDI as importance for visualization (standard metric)
        imp = np.asarray(rf.feature_importances_, dtype=float)
    except Exception:
        perf, imp, mask = 0.0, np.zeros(p), np.zeros(p, dtype=bool)

    return RashomonModel("rf", seed, {"n_estimators": n_estimators, "max_depth": max_depth,
                                       "min_samples_leaf": min_samples_leaf}, perf, mask, imp)


# ----------------------------- Plotting -----------------------------


def plot_rashomon_size_vs_epsilon(models_log: List[RashomonModel], models_rf: List[RashomonModel], out_dir: Path):
    """Plot Rashomon set size as a function of epsilon."""
    epsilons = np.linspace(0.001, 0.10, 50)
    best_log = max(m.performance for m in models_log) if models_log else 0
    best_rf = max(m.performance for m in models_rf) if models_rf else 0

    sizes_log = [sum(1 for m in models_log if m.performance >= best_log - e) for e in epsilons]
    sizes_rf = [sum(1 for m in models_rf if m.performance >= best_rf - e) for e in epsilons]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(epsilons * 100, sizes_log, 'o-', color="#1f78b4", label="Logistic", markersize=4)
    ax.plot(epsilons * 100, sizes_rf, 's-', color="#e31a1c", label="RF", markersize=4)
    ax.set_xlabel("ε (% balanced accuracy drop from best)")
    ax.set_ylabel("Rashomon set size")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_fig_formats(fig, out_dir / "rashomon_size_vs_epsilon")
    plt.close(fig)


def plot_feature_frequency(freq: np.ndarray, feature_names: Sequence[str], method: str, out_dir: Path, topn: int = 50):
    """Plot feature frequency in Rashomon set."""
    order = np.argsort(-freq)
    k = min(topn, len(feature_names))

    fig, ax = plt.subplots(figsize=(max(8, 0.22 * k), 4.5))
    colors = ["#2ca02c" if freq[order[i]] >= 1.0 else "#1f78b4" if freq[order[i]] >= 0.5
              else "#ff7f0e" if freq[order[i]] > 0 else "#d62728" for i in range(k)]

    ax.bar(np.arange(k), freq[order][:k], color=colors, alpha=0.85)
    ax.axhline(1.0, color="green", linestyle="--", linewidth=1, alpha=0.7)
    ax.axhline(0.5, color="blue", linestyle=":", linewidth=1, alpha=0.7)
    ax.set_xticks(np.arange(k))
    ax.set_xticklabels([feature_names[order[i]] for i in range(k)], rotation=90, fontsize=7)
    ax.set_ylabel("Frequency in Rashomon set")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"{method}: Feature Selection Frequency")
    patches = [mpatches.Patch(color="#2ca02c", label="Necessary (=1.0)"),
               mpatches.Patch(color="#1f78b4", label="Common (≥0.5)"),
               mpatches.Patch(color="#ff7f0e", label="Rare (<0.5)")]
    ax.legend(handles=patches, loc="upper right", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    save_fig_formats(fig, out_dir / f"{method.lower()}_feature_frequency")
    plt.close(fig)


def plot_importance_cloud(cloud: Dict[str, np.ndarray], feature_names: Sequence[str],
                          method: str, out_dir: Path, topn: int = 30):
    """Plot importance cloud (range of importances across Rashomon set)."""
    order = np.argsort(-cloud["mean"])
    k = min(topn, len(feature_names))

    fig, ax = plt.subplots(figsize=(max(8, 0.25 * k), 5))
    x = np.arange(k)

    for i in range(k):
        ax.plot([i, i], [cloud["min"][order[i]], cloud["max"][order[i]]], color="#888888", linewidth=1, alpha=0.5)
    ax.bar(x, cloud["q75"][order][:k] - cloud["q25"][order][:k], bottom=cloud["q25"][order][:k],
           width=0.4, color="#1f78b4", alpha=0.7)
    ax.scatter(x, cloud["mean"][order][:k], color="#e31a1c", s=20, zorder=5, label="Mean")

    ax.set_xticks(x)
    ax.set_xticklabels([feature_names[order[i]] for i in range(k)], rotation=90, fontsize=7)
    ax.set_ylabel("Feature Importance")
    ax.set_title(f"{method}: Importance Cloud (IQR + range)")
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    save_fig_formats(fig, out_dir / f"{method.lower()}_importance_cloud")
    plt.close(fig)


def plot_frequency_comparison(freq_log: np.ndarray, freq_rf: np.ndarray, feature_names: Sequence[str], out_dir: Path):
    """Scatter plot comparing feature frequency between methods."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(freq_log, freq_rf, s=15, alpha=0.5, color="#33a02c")
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)

    diff = np.abs(freq_log - freq_rf)
    for idx in np.argsort(-diff)[:5]:
        if diff[idx] > 0.3:
            ax.annotate(feature_names[idx], (freq_log[idx], freq_rf[idx]), fontsize=7, alpha=0.8)

    rho, _ = stats.spearmanr(freq_log, freq_rf)
    ax.set_xlabel("Logistic frequency")
    ax.set_ylabel("RF frequency")
    ax.set_title(f"Feature Frequency: Logistic vs RF (ρ = {rho:.3f})")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_fig_formats(fig, out_dir / "frequency_comparison")
    plt.close(fig)


def plot_performance_distribution(models_log, models_rf, rset_log, rset_rf, out_dir: Path):
    """Plot distribution of model performances."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, models, rset, color, name in [(axes[0], models_log, rset_log, "#1f78b4", "Logistic"),
                                           (axes[1], models_rf, rset_rf, "#e31a1c", "RF")]:
        perfs = [m.performance for m in models]
        ax.hist(perfs, bins=30, color=color, alpha=0.7, edgecolor="black")
        ax.axvline(rset.threshold, color="red", linestyle="--", linewidth=2, label=f"Thresh ({rset.threshold:.3f})")
        ax.axvline(rset.best_performance, color="green", linestyle="-", linewidth=2, label=f"Best ({rset.best_performance:.3f})")
        ax.set_xlabel("Balanced Accuracy")
        ax.set_ylabel("Count")
        ax.set_title(f"{name}: {rset.size}/{len(models)} in Rashomon")
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    save_fig_formats(fig, out_dir / "performance_distribution")
    plt.close(fig)


def plot_overlap_bar(nec_log, nec_rf, com_log, com_rf, out_dir: Path) -> Dict[str, int]:
    """Bar chart showing overlap of necessary/common features."""
    stats_out = {}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, mask_log, mask_rf, title, prefix in [(axes[0], nec_log, nec_rf, "Necessary (freq=1.0)", "necessary"),
                                                   (axes[1], com_log, com_rf, "Common (freq≥0.5)", "common")]:
        only_log = int((mask_log & ~mask_rf).sum())
        only_rf = int((mask_rf & ~mask_log).sum())
        both = int((mask_log & mask_rf).sum())
        stats_out[f"{prefix}_logistic_only"] = only_log
        stats_out[f"{prefix}_rf_only"] = only_rf
        stats_out[f"{prefix}_both"] = both

        bars = ax.barh([0, 1, 2], [only_log, both, only_rf], color=["#1f78b4", "#984ea3", "#e31a1c"], alpha=0.8)
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(["Logistic only", "Both", "RF only"])
        ax.set_xlabel("Number of features")
        ax.set_title(title)
        for bar, val in zip(bars, [only_log, both, only_rf]):
            ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2, str(val), va='center')
        ax.grid(True, axis='x', alpha=0.3)

    fig.tight_layout()
    save_fig_formats(fig, out_dir / "feature_overlap")
    plt.close(fig)
    return stats_out


# ----------------------------- Main -----------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Rashomon Set Feature Analysis with Taxonomic Holdout")
    parser.add_argument("--input_dir", type=Path, default=Path("/vol/projects/BIFO/genomenet/yichen/phenotype/data/rfdata"))
    parser.add_argument("--output_dir", type=Path, default=Path("sporulation/results/rashomon_blocked"))
    parser.add_argument("--metadata", type=Path, default=Path("sporulation/microbe.cards table S1.xlsx"),
                        help="Path to metadata Excel with taxonomy columns")
    parser.add_argument("--phenotype", type=str, default="Spore formation")
    parser.add_argument("--taxon_level", type=str, default="Order", choices=["Order", "Family", "Genus"],
                        help="Taxonomic level for blocking (entire groups held out)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_prev", type=float, default=0.02)
    parser.add_argument("--epsilon", type=float, default=0.02, help="Performance tolerance (absolute bal-acc drop)")
    parser.add_argument("--n_models", type=int, default=200, help="Number of models to fit per method")
    parser.add_argument("--rf_top_k", type=int, default=100, help="Top-K features per RF model")
    parser.add_argument("--topn", type=int, default=50, help="Top N features to plot")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Fraction of samples for validation")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="Fraction of samples for test")
    args = parser.parse_args()

    phen_name = str(args.phenotype).strip()
    phen_safe = phen_name.replace(" ", "_")
    out_dir: Path = args.output_dir / f"{phen_safe}_by_{args.taxon_level}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    seed, epsilon = int(args.seed), float(args.epsilon)

    print(f"=== Rashomon Set Analysis (Taxonomic Holdout): {phen_name} ===", flush=True)
    print(f"Blocking level: {args.taxon_level} (entire groups held out)", flush=True)
    print(f"Epsilon: {epsilon} (models within {epsilon*100:.1f}% bal-acc of best)", flush=True)

    # Load data with taxonomic holdout split
    pa_df = _read_table(args.input_dir / "rf_presence_absence.parquet")
    long_df = _read_table(args.input_dir / "rf_dataset.parquet")
    splits = create_taxonomic_holdout_split(
        pa_df, long_df,
        metadata_path=args.metadata,
        phenotype=phen_name,
        taxon_level=args.taxon_level,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=seed,
    )
    splits = filter_low_prevalence_features(splits, min_prevalence=float(args.min_prev))
    print(f"Data: train={splits.X_train.shape}, val={splits.X_val.shape}, p={splits.X_train.shape[1]}", flush=True)

    if splits.X_train.shape[0] < 100 or splits.X_val.shape[0] < 50:
        print(f"[Abort] Insufficient samples", flush=True)
        return

    X_train, y_train = splits.X_train, splits.y_train
    X_val, y_val = splits.X_val, splits.y_val
    feature_names = list(splits.feature_names)
    p = len(feature_names)

    # -------------------- Fit Logistic Models --------------------
    logistic_cache = cache_dir / "rashomon_logistic.npz"
    if logistic_cache.exists():
        try:
            data = np.load(str(logistic_cache), allow_pickle=True)
            all_models_log = [RashomonModel("logistic", int(data["seeds"][i]), {"C": float(data["Cs"][i])},
                              float(data["performances"][i]), data["masks"][i].astype(bool), data["importances"][i])
                              for i in range(len(data["performances"]))]
            print(f"[Cache] Loaded {len(all_models_log)} logistic models", flush=True)
        except Exception:
            all_models_log = None
    else:
        all_models_log = None

    if all_models_log is None:
        C_values = np.logspace(-3, 2, 20)
        configs = [(C, seed + i * 7919) for C in C_values for i in range(max(1, args.n_models // 20))][:args.n_models]
        print(f"Fitting {len(configs)} logistic models...", flush=True)

        with tqdm_joblib(tqdm(total=len(configs), desc="Logistic", unit="model")):
            all_models_log = Parallel(n_jobs=effective_joblib_n_jobs(), backend="loky")(
                delayed(fit_logistic_model)(X_train, y_train, X_val, y_val, C, s) for C, s in configs)

        atomic_savez_compressed(logistic_cache,
            performances=np.array([m.performance for m in all_models_log]),
            masks=np.vstack([m.feature_mask for m in all_models_log]),
            importances=np.vstack([m.feature_importance for m in all_models_log]),
            seeds=np.array([m.seed for m in all_models_log]),
            Cs=np.array([m.hyperparams["C"] for m in all_models_log]))

    best_log = max(m.performance for m in all_models_log)
    rset_log = RashomonSet("logistic", epsilon, best_log, best_log - epsilon,
                           [m for m in all_models_log if m.performance >= best_log - epsilon])
    print(f"Logistic: best={best_log:.4f}, Rashomon={rset_log.size}/{len(all_models_log)}", flush=True)

    # -------------------- Fit RF Models --------------------
    rf_cache = cache_dir / "rashomon_rf.npz"
    if rf_cache.exists():
        try:
            data = np.load(str(rf_cache), allow_pickle=True)
            hparams = json.loads(str(data["hyperparams_json"]))
            all_models_rf = [RashomonModel("rf", int(data["seeds"][i]), hparams[i],
                             float(data["performances"][i]), data["masks"][i].astype(bool), data["importances"][i])
                             for i in range(len(data["performances"]))]
            print(f"[Cache] Loaded {len(all_models_rf)} RF models", flush=True)
        except Exception:
            all_models_rf = None
    else:
        all_models_rf = None

    if all_models_rf is None:
        configs = [(n, d, l, seed + i * 12347)
                   for n in [100, 300, 500, 800] for d in [10, 20, 30, None]
                   for l in [1, 2, 5] for i in range(max(1, args.n_models // 48))][:args.n_models]
        print(f"Fitting {len(configs)} RF models...", flush=True)

        with tqdm_joblib(tqdm(total=len(configs), desc="RF", unit="model")):
            all_models_rf = Parallel(n_jobs=effective_joblib_n_jobs(), backend="loky")(
                delayed(fit_rf_model)(X_train, y_train, X_val, y_val, n, d, l, s, args.rf_top_k)
                for n, d, l, s in configs)

        atomic_savez_compressed(rf_cache,
            performances=np.array([m.performance for m in all_models_rf]),
            masks=np.vstack([m.feature_mask for m in all_models_rf]),
            importances=np.vstack([m.feature_importance for m in all_models_rf]),
            seeds=np.array([m.seed for m in all_models_rf]),
            hyperparams_json=np.array(json.dumps([m.hyperparams for m in all_models_rf])))

    best_rf = max(m.performance for m in all_models_rf)
    rset_rf = RashomonSet("rf", epsilon, best_rf, best_rf - epsilon,
                          [m for m in all_models_rf if m.performance >= best_rf - epsilon])
    print(f"RF: best={best_rf:.4f}, Rashomon={rset_rf.size}/{len(all_models_rf)}", flush=True)

    # -------------------- Analysis --------------------
    freq_log, freq_rf = rset_log.feature_frequency(p), rset_rf.feature_frequency(p)

    def compute_cloud(rset):
        if not rset.models:
            return {k: np.zeros(p) for k in ["mean", "std", "min", "max", "q25", "q75"]}
        imps = np.vstack([m.feature_importance for m in rset.models])
        return {"mean": imps.mean(0), "std": imps.std(0), "min": imps.min(0),
                "max": imps.max(0), "q25": np.percentile(imps, 25, 0), "q75": np.percentile(imps, 75, 0)}

    cloud_log, cloud_rf = compute_cloud(rset_log), compute_cloud(rset_rf)

    nec_log, com_log = freq_log >= 1.0, freq_log >= 0.5
    nec_rf, com_rf = freq_rf >= 1.0, freq_rf >= 0.5

    print(f"\nLogistic: necessary={nec_log.sum()}, common={com_log.sum()}", flush=True)
    print(f"RF: necessary={nec_rf.sum()}, common={com_rf.sum()}", flush=True)

    # -------------------- Plots --------------------
    plot_rashomon_size_vs_epsilon(all_models_log, all_models_rf, out_dir)
    plot_performance_distribution(all_models_log, all_models_rf, rset_log, rset_rf, out_dir)
    plot_feature_frequency(freq_log, feature_names, "Logistic", out_dir, args.topn)
    plot_feature_frequency(freq_rf, feature_names, "RF", out_dir, args.topn)
    plot_importance_cloud(cloud_log, feature_names, "Logistic", out_dir, args.topn)
    plot_importance_cloud(cloud_rf, feature_names, "RF", out_dir, args.topn)
    plot_frequency_comparison(freq_log, freq_rf, feature_names, out_dir)
    overlap_stats = plot_overlap_bar(nec_log, nec_rf, com_log, com_rf, out_dir)

    # -------------------- Save --------------------
    summary = {
        "phenotype": phen_name, "epsilon": epsilon,
        "split_method": "taxonomic_holdout",
        "taxon_level": args.taxon_level,
        "data": {"n_train": int(X_train.shape[0]), "n_val": int(X_val.shape[0]), "n_features": p},
        "logistic": {"n_total": len(all_models_log), "rashomon_size": rset_log.size,
                     "best": float(best_log), "n_necessary": int(nec_log.sum()), "n_common": int(com_log.sum())},
        "rf": {"n_total": len(all_models_rf), "rashomon_size": rset_rf.size,
               "best": float(best_rf), "n_necessary": int(nec_rf.sum()), "n_common": int(com_rf.sum())},
        "overlap": overlap_stats,
    }
    with (out_dir / "rashomon_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    pd.DataFrame({
        "gene": feature_names, "freq_logistic": freq_log, "freq_rf": freq_rf,
        "imp_mean_logistic": cloud_log["mean"], "imp_mean_rf": cloud_rf["mean"],
        "necessary_logistic": nec_log, "necessary_rf": nec_rf,
    }).sort_values("freq_logistic", ascending=False).to_csv(out_dir / "rashomon_features.csv", index=False)

    print(f"\nOutputs: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
