#!/usr/bin/env python3
"""Empirical demonstration of predictive multiplicity (H1) for sporulation.

Implements:
- (A) Complementary-Pairs Stability Selection (CPSS) with L1-logistic
- (B) Boruta (all-relevant RF wrapper) + permutation VI (validation-based)
- (C) Conditional Permutation Importance (CPI; within-leaf permutation)

Data policy:
- Load presence/absence and long tables via _read_table
- Build splits with build_dataset; apply filter_low_prevalence_features on train only
- Use train+val only for all fitting/importance; never access test

Outputs (PNG/JSON/CSV) are written to output_dir.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import joblib
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy import stats
import tempfile
import warnings

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.manifold import MDS

import matplotlib
matplotlib.use("Agg")  # non-interactive
import matplotlib.pyplot as plt

from boruta import BorutaPy

# Required imports from local codebase (no alternatives)
from train_lasso import _read_table, build_dataset, filter_low_prevalence_features, DataSplits
from train_rf import compute_feature_importance  # for MDI ranking


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


def set_seed_numpy(seed: int) -> np.random.RandomState:
    return np.random.RandomState(int(seed))


def effective_joblib_n_jobs() -> int:
    env_n = os.environ.get("SLURM_CPUS_PER_TASK") or os.environ.get("LOKY_MAX_CPU_COUNT")
    try:
        n = int(env_n) if env_n else (os.cpu_count() or 1)
    except Exception:
        n = os.cpu_count() or 1
    return max(1, int(n))


def nonzero_mask_from_logistic(model: LogisticRegression, p: int) -> np.ndarray:
    coef = getattr(model, "coef_", None)
    if coef is None:
        return np.zeros(p, dtype=bool)
    coef = np.asarray(coef)
    if coef.ndim == 1:
        coef = coef.reshape(1, -1)
    if coef.shape[1] != p:
        try:
            flat = coef.reshape(-1)
            if flat.size == p:
                return np.abs(flat) > 0
        except Exception:
            pass
        return np.zeros(p, dtype=bool)
    return (np.abs(coef) > 0).any(axis=0)


def save_fig_formats(fig: plt.Figure, base_path: Path, dpi: int = 150) -> None:
    """Save figure in PNG, PDF, and SVG formats.

    Args:
        fig: Matplotlib figure to save
        base_path: Path without extension (e.g., out_dir / "plot_name")
        dpi: Resolution for PNG (PDF/SVG are vector formats)
    """
    base = Path(base_path)
    for ext in (".png", ".pdf", ".svg"):
        fig.savefig(base.with_suffix(ext), dpi=dpi, bbox_inches="tight")


def atomic_savez_compressed(path: Path, **arrays: object) -> None:
    """Write NPZ atomically by saving to a temp file then replacing the target.

    Ensures the temp file is created in the same directory (so os.replace is atomic),
    fsynced before replacement, and cleaned up on error. Accepts arbitrary arrays.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=str(target.parent))
    try:
        with os.fdopen(fd, "wb") as f:
            np.savez_compressed(f, **arrays)
            try:
                f.flush()
                os.fsync(f.fileno())
            except Exception:
                # Best-effort; not all FS support fsync
                pass
        os.replace(tmp_name, target)
    except Exception:
        try:
            os.unlink(tmp_name)
        except Exception:
            pass
        raise


def pilot_choose_C(
    X: np.ndarray,
    y: np.ndarray,
    target_q: int,
    c_min: float,
    c_max: float,
    num_grid: int,
    seed: int,
) -> Tuple[float, List[Tuple[float, int]]]:
    """Pick C* whose selected count on a single stratified half-sample is closest to target_q.

    Returns (C_star, grid_results[(C, k_selected), ...]).
    """
    rng = set_seed_numpy(seed)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    # Use one stratified half-sample for pilot
    for tr_idx, _ in sss.split(X, y):
        X_half, y_half = X[tr_idx], y[tr_idx]
        break

    grid = np.logspace(np.log10(float(c_min)), np.log10(float(c_max)), num=int(num_grid))
    results: List[Tuple[float, int]] = []
    p = X.shape[1]
    for C in grid:
        try:
            model = LogisticRegression(
                solver="saga",
                penalty="l1",
                C=float(C),
                max_iter=5000,
                tol=1e-3,
                random_state=seed,
                class_weight=None,
                multi_class=("multinomial" if len(np.unique(y_half)) > 2 else "auto"),
            )
            model.fit(X_half, y_half)
            k = int(nonzero_mask_from_logistic(model, p).sum())
        except Exception:
            k = 0
        results.append((float(C), k))

    # choose C* closest to target_q (tie-break: choose with k minimal absolute diff then smaller C)
    diffs = [abs(k - int(target_q)) for (_, k) in results]
    best_idx = int(np.argmin(diffs))
    C_star = float(results[best_idx][0])
    return C_star, results


def generate_complementary_halves(
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return complementary stratified halves (indices) with attempts to ensure both classes.
    """
    max_attempts = 20
    for attempt in range(max_attempts):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed + attempt)
        for h_idx, hc_idx in sss.split(X, y):
            y_h = y[h_idx]
            y_hc = y[hc_idx]
            if len(np.unique(y_h)) >= 2 and len(np.unique(y_hc)) >= 2:
                return h_idx, hc_idx
    # Fallback: return first split regardless
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    for h_idx, hc_idx in sss.split(X, y):
        return h_idx, hc_idx
    raise RuntimeError("Failed to generate complementary halves")


def cpsi_nogueira(pi: np.ndarray) -> float:
    """Nogueira stability index based on selection probabilities pi (length p)."""
    p = float(len(pi))
    if p == 0:
        return float("nan")
    p_hat = np.clip(np.asarray(pi, dtype=float), 0.0, 1.0)
    k_bar = float(p_hat.sum())
    denom = k_bar * (1.0 - (k_bar / p))
    if denom <= 0:
        return float("nan")
    numer = float(np.sum(p_hat * (1.0 - p_hat)))
    stab = 1.0 - (numer / denom)
    return float(stab)


def cp_ss(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    seed: int,
    num_pairs: int,
    target_q: int,
    tau: float,
    c_min: float,
    c_max: float,
) -> Tuple[Dict[str, float], Dict[str, object], np.ndarray]:
    """Run CPSS with L1-logistic (saga) and return (pi_by_gene, summary).

    Summary fields: C_star, q_median, p, tau, expected_false_positives_bound, nogueira_stability
    """
    n, p = X.shape
    n_classes = int(len(np.unique(y)))
    mc = "multinomial" if n_classes > 2 else "auto"
    # Pilot choose C*
    C_star, grid_info = pilot_choose_C(
        X, y, target_q=target_q, c_min=c_min, c_max=c_max, num_grid=40, seed=seed
    )

    print(
        f"CPSS pilot: chose C*={C_star:g} with target_q={target_q}; grid snapshot (C,k) head: {grid_info[:5]}",
        flush=True,
    )

    def _fit_half_selected(half_indices: np.ndarray, model_seed: int) -> np.ndarray:
        Xh = X[half_indices]
        yh = y[half_indices]
        model = LogisticRegression(
            solver="saga",
            penalty="l1",
            C=float(C_star),
            max_iter=5000,
            tol=1e-3,
            random_state=model_seed,
            class_weight=None,
            multi_class=mc,
        )
        model.fit(Xh, yh)
        return nonzero_mask_from_logistic(model, p)

    # Prepare pairs and run fits in parallel
    print(f"CPSS: running {num_pairs} complementary pairs => total fits={2 * int(num_pairs)}", flush=True)

    def _run_one_pair(b: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        h_idx, hc_idx = generate_complementary_halves(X, y, seed=seed + b)
        sel_h = _fit_half_selected(h_idx, model_seed=seed + 7919 * (b + 1) + 1)
        sel_hc = _fit_half_selected(hc_idx, model_seed=seed + 7919 * (b + 1) + 2)
        return sel_h, sel_hc, h_idx, hc_idx

    with tqdm_joblib(tqdm(total=int(num_pairs), desc="CPSS pairs", unit="pair")):
        pair_results: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = Parallel(n_jobs=effective_joblib_n_jobs(), backend="loky")(
            delayed(_run_one_pair)(b) for b in range(int(num_pairs))
        )

    # Aggregate selection
    all_selected_masks: List[np.ndarray] = []
    half_indices_list: List[np.ndarray] = []
    for sel_h, sel_hc, h_idx, hc_idx in pair_results:
        all_selected_masks.append(sel_h)
        all_selected_masks.append(sel_hc)
        half_indices_list.append(h_idx)
        half_indices_list.append(hc_idx)
    if len(all_selected_masks) == 0:
        sel_matrix = np.zeros((0, p), dtype=float)
        pi = np.zeros(p, dtype=float)
        k_sizes: List[int] = []
    else:
        sel_matrix = np.vstack([m.astype(np.uint8) for m in all_selected_masks])  # F x p, compact
        pi = sel_matrix.mean(axis=0)
        k_sizes = [int(m.sum()) for m in all_selected_masks]

    q_median = float(np.median(k_sizes)) if len(k_sizes) > 0 else 0.0
    nog_stab = cpsi_nogueira(pi)
    tau = float(tau)
    p_float = float(p)
    ev_bound = float((q_median ** 2) / ((2.0 * tau - 1.0) * p_float)) if tau > 0.5 and p > 0 else None

    # Map to genes
    pi_by_gene = {str(g): float(v) for g, v in zip(feature_names, pi)}

    summary: Dict[str, object] = {
        "C_star": float(C_star),
        "pairs": int(num_pairs),
        "total_fits": int(len(all_selected_masks)),
        "q_median": float(q_median),
        "p": int(p),
        "tau": float(tau),
        "nogueira_stability": float(nog_stab) if nog_stab == nog_stab else None,
        "cpss_false_positives_bound": float(ev_bound) if ev_bound is not None and ev_bound == ev_bound else None,
        "grid_C_k": [(float(C), int(k)) for (C, k) in grid_info],
        "half_indices": [idx.astype(int).tolist() for idx in half_indices_list],
    }
    return pi_by_gene, summary, sel_matrix


def plot_cpss(pi_by_gene: Dict[str, float], tau: float, out_dir: Path, topn: int, ev_bound: Optional[float]) -> None:
    genes = np.array(list(pi_by_gene.keys()))
    pis = np.array(list(pi_by_gene.values()))
    order = np.argsort(-pis)
    genes_sorted = genes[order]
    pis_sorted = pis[order]

    # Top-N bar
    k = int(min(topn, len(genes_sorted)))
    fig, ax = plt.subplots(figsize=(max(6, 0.25 * k), 4))
    ax.bar(np.arange(k), pis_sorted[:k], color="#2c7fb8")
    ax.set_xticks(np.arange(k))
    ax.set_xticklabels(genes_sorted[:k], rotation=90, fontsize=8)
    ax.set_ylabel("Selection probability $\\hat{\\pi}_j$")
    ax.set_ylim(0, 1)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    save_fig_formats(fig, out_dir / "cpss_selection_probs")
    plt.close(fig)

    # Histogram of all pi
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(pis, bins=30, color="#74a9cf", edgecolor="black", alpha=0.9)
    ax.set_xlabel("Selection probability $\\hat{\\pi}_j$")
    ax.set_ylabel("Count")
    ax.set_xlim(0, 1)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    save_fig_formats(fig, out_dir / "cpss_hist")
    plt.close(fig)

    # Size vs tau
    taus = np.linspace(0.5, 0.95, 10)
    sizes = [(pis >= t).sum() for t in taus]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(taus, sizes, marker="o", color="#1b9e77")
    ax.axvline(float(tau), color="red", linestyle="--", label=f"tau={float(tau):.2f}")
    ax.set_xlabel("Threshold $\\tau$")
    ax.set_ylabel(r"$|S_\tau|$")
    if ev_bound is not None and ev_bound == ev_bound:
        ax.annotate(f"E[V] bound ≈ {ev_bound:.2f}", xy=(float(tau), max(sizes) * 0.8), color="red")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    save_fig_formats(fig, out_dir / "cpss_size_vs_tau")
    plt.close(fig)


# ----------------------------- Boruta + Permutation VI -----------------------------


def run_boruta(
    X: np.ndarray,
    y: np.ndarray,
    rf_trees: int,
    rf_max_depth: int,
    runs: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run Boruta multiple times and return confirmation rates per feature (length p)."""
    n, p = X.shape

    def _one_run(r: int) -> np.ndarray:
        rf = RandomForestClassifier(
            n_estimators=int(rf_trees),
            max_depth=int(rf_max_depth),
            n_jobs=1,  # avoid nested parallelism; outer loop is parallelized
            random_state=seed + r,
            class_weight="balanced",
        )
        bor = BorutaPy(
            estimator=rf,
            n_estimators="auto",
            verbose=0,
            random_state=seed + r,
            two_step=True,
        )
        bor.fit(X, y)
        support = np.asarray(bor.support_, dtype=bool)
        if support.size != p:
            supp = np.zeros(p, dtype=bool)
            supp[: min(p, support.size)] = support[: min(p, support.size)]
            return supp
        return support

    print(f"Running Boruta for {runs} runs...", flush=True)
    with tqdm_joblib(tqdm(total=int(runs), desc="Boruta runs", unit="run")):
        supports: List[np.ndarray] = Parallel(n_jobs=effective_joblib_n_jobs(), backend="loky")(
            delayed(_one_run)(r) for r in range(int(runs))
        )
    if len(supports) == 0:
        return np.zeros(p, dtype=float), np.zeros((0, p), dtype=bool)
    supports_matrix = np.vstack([s.astype(bool) for s in supports])  # runs x p
    confirm_counts = np.sum(supports_matrix.astype(int), axis=0)
    confirm_rate = confirm_counts / float(max(1, int(runs)))
    return confirm_rate.astype(float), supports_matrix


def plot_boruta(confirm_rate: np.ndarray, feature_names: Sequence[str], out_dir: Path, topn: int) -> None:
    order = np.argsort(-confirm_rate)
    k = int(min(topn, len(order)))
    fig, ax = plt.subplots(figsize=(max(6, 0.25 * k), 4))
    ax.bar(np.arange(k), confirm_rate[order][:k], color="#4daf4a")
    ax.set_xticks(np.arange(k))
    ax.set_xticklabels(np.array(feature_names)[order][:k], rotation=90, fontsize=8)
    ax.set_ylabel("Boruta confirm rate $c_j$")
    ax.set_ylim(0, 1)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    save_fig_formats(fig, out_dir / "boruta_confirm_rate")
    plt.close(fig)


def train_rf_and_perm_vi(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: Sequence[str],
    rf_trees: int,
    rf_max_depth: int,
    perm_reps: int,
    perm_topk: int,
    seed: int,
) -> Tuple[RandomForestClassifier, pd.DataFrame, np.ndarray, float]:
    """Train RF, get MDI df, compute validation permutation VI for top-k by MDI.

    Returns (rf_model, mdi_df, vi_perm (length p, NaN for non-computed), base_val_auc)
    """
    print("Training RandomForest (train-only) for permutation VI...", flush=True)
    rf = RandomForestClassifier(
        n_estimators=int(rf_trees),
        max_depth=int(rf_max_depth),
        n_jobs=1,  # avoid nested parallelism; prediction will be called inside joblib loops
        random_state=seed,
        class_weight="balanced",
    )
    rf.fit(X_train, y_train)

    mdi_df = compute_feature_importance(rf, list(feature_names))
    mdi_order = mdi_df.sort_values("importance", ascending=False, kind="mergesort").reset_index(drop=True)
    topk = int(min(int(perm_topk), len(mdi_order)))
    topk_genes = mdi_order.loc[: topk - 1, "gene_norm"].tolist()
    gene_to_idx = {g: i for i, g in enumerate(feature_names)}
    topk_indices = [gene_to_idx[g] for g in topk_genes if g in gene_to_idx]

    # Baseline AUC on validation (binary or multiclass)
    base_auc: float
    try:
        proba = rf.predict_proba(X_val)
        classes = np.array(getattr(rf, "classes_", []))
        if isinstance(proba, list):
            proba = np.column_stack(proba)
        if proba.ndim == 2 and proba.shape[1] > 2:
            base_auc = float(roc_auc_score(y_val, proba, multi_class="ovr"))
        else:
            if classes.size > 0:
                try:
                    idx1 = int(np.where(classes == 1)[0][0])
                except Exception:
                    idx1 = int(np.argmax(classes))
                base_auc = float(roc_auc_score(y_val, proba[:, idx1]))
            else:
                base_auc = float("nan")
    except Exception:
        base_auc = float("nan")

    print(
        f"Permutation VI: base val AUC={base_auc if base_auc == base_auc else 'nan'}; topk for permutation={len(topk_indices)}",
        flush=True,
    )

    def _vi_for_feature(j: int, feat_seed: int) -> Tuple[int, float, float]:
        if not (base_auc == base_auc):
            return j, float("nan"), float("nan")
        deltas: List[float] = []
        for r in range(int(perm_reps)):
            rng = set_seed_numpy(feat_seed + r)
            Xp = X_val.copy()
            perm = rng.permutation(Xp.shape[0])
            Xp[:, j] = Xp[perm, j]
            try:
                proba_p = rf.predict_proba(Xp)
                classes_p = np.array(getattr(rf, "classes_", []))
                if isinstance(proba_p, list):
                    proba_p = np.column_stack(proba_p)
                if proba_p.ndim == 2 and proba_p.shape[1] > 2:
                    auc_p = float(roc_auc_score(y_val, proba_p, multi_class="ovr"))
                else:
                    if classes_p.size > 0:
                        try:
                            idx1 = int(np.where(classes_p == 1)[0][0])
                        except Exception:
                            idx1 = int(np.argmax(classes_p))
                        auc_p = float(roc_auc_score(y_val, proba_p[:, idx1]))
                    else:
                        auc_p = float("nan")
                deltas.append(float(base_auc - auc_p))
            except Exception:
                deltas.append(float("nan"))
        arr = np.array(deltas, dtype=float)
        mean = float(np.nanmean(arr)) if np.any(arr == arr) else float("nan")
        se = float(np.nanstd(arr, ddof=1) / np.sqrt(max(1, int(perm_reps)))) if np.any(arr == arr) else float("nan")
        return j, mean, se

    vi_perm = np.full(len(feature_names), np.nan, dtype=float)
    with tqdm_joblib(tqdm(total=len(topk_indices), desc="Permutation VI", unit="feat")):
        results = Parallel(n_jobs=effective_joblib_n_jobs(), backend="loky")(
            delayed(_vi_for_feature)(j, seed + 21701 * (idx + 1)) for idx, j in enumerate(topk_indices)
        )
    for j, mean, se in results:
        vi_perm[j] = mean

    return rf, mdi_order, vi_perm, base_auc


# ----------------------------- CPI (within-leaf permutation) -----------------------------


def within_leaf_permute_column(
    X_val: np.ndarray,
    col_index: int,
    leaf_ids: np.ndarray,
    rng: np.random.RandomState,
) -> np.ndarray:
    Xp = X_val.copy()
    # Permute within each leaf block
    for leaf in np.unique(leaf_ids):
        idx = np.where(leaf_ids == leaf)[0]
        if idx.size <= 1:
            continue
        block = Xp[idx, col_index].copy()
        rng.shuffle(block)
        Xp[idx, col_index] = block
    return Xp


def compute_cpi(
    rf: RandomForestClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    candidate_indices: Sequence[int],
    leaf_max_depth: int,
    leaf_min_samples: int,
    cpi_reps: int,
    seed: int,
    base_val_auc: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute CPI and SE for candidate feature indices."""

    def _cpi_one(j: int, job_seed: int) -> Tuple[int, float, float]:
        if not (base_val_auc == base_val_auc):
            return j, float("nan"), float("nan")
        # Fit leaf model to predict feature j from others
        mask = np.ones(X_train.shape[1], dtype=bool)
        mask[j] = False
        Xo_tr = X_train[:, mask]
        Xo_val = X_val[:, mask]
        xj_tr = X_train[:, j].astype(int)
        dt = DecisionTreeClassifier(
            max_depth=int(leaf_max_depth),
            min_samples_leaf=int(leaf_min_samples),
            random_state=job_seed,
        )
        try:
            dt.fit(Xo_tr, xj_tr)
            leaf_ids = dt.apply(Xo_val)
        except Exception:
            # If fitting fails, fall back to global permutation
            leaf_ids = np.zeros(X_val.shape[0], dtype=int)

        deltas: List[float] = []
        for r in range(int(cpi_reps)):
            rng = set_seed_numpy(job_seed + r)
            Xp = within_leaf_permute_column(X_val, j, leaf_ids, rng)
            try:
                proba_p = rf.predict_proba(Xp)
                classes_p = np.array(getattr(rf, "classes_", []))
                if isinstance(proba_p, list):
                    proba_p = np.column_stack(proba_p)
                if proba_p.ndim == 2 and proba_p.shape[1] > 2:
                    auc_p = float(roc_auc_score(y_val, proba_p, multi_class="ovr"))
                else:
                    if classes_p.size > 0:
                        try:
                            idx1 = int(np.where(classes_p == 1)[0][0])
                        except Exception:
                            idx1 = int(np.argmax(classes_p))
                        auc_p = float(roc_auc_score(y_val, proba_p[:, idx1]))
                    else:
                        auc_p = float("nan")
                deltas.append(float(base_val_auc - auc_p))
            except Exception:
                deltas.append(float("nan"))
        arr = np.array(deltas, dtype=float)
        mean = float(np.nanmean(arr)) if np.any(arr == arr) else float("nan")
        se = float(np.nanstd(arr, ddof=1) / np.sqrt(max(1, int(cpi_reps)))) if np.any(arr == arr) else float("nan")
        return j, mean, se

    with tqdm_joblib(tqdm(total=len(candidate_indices), desc="CPI", unit="feat")):
        results = Parallel(n_jobs=effective_joblib_n_jobs(), backend="loky")(
            delayed(_cpi_one)(int(j), seed + 30011 * (idx + 1)) for idx, j in enumerate(candidate_indices)
        )
    cpi = np.full(X_train.shape[1], np.nan, dtype=float)
    cpi_se = np.full(X_train.shape[1], np.nan, dtype=float)
    for j, mean, se in results:
        cpi[int(j)] = mean
        cpi_se[int(j)] = se
    return cpi, cpi_se


def plot_vi_cpi_scatter(
    vi_perm: np.ndarray,
    cpi: np.ndarray,
    feature_names: Sequence[str],
    candidates: Sequence[int],
    out_dir: Path,
) -> None:
    xs = np.array([vi_perm[j] if j < len(vi_perm) else np.nan for j in candidates], dtype=float)
    ys = np.array([cpi[j] if j < len(cpi) else np.nan for j in candidates], dtype=float)
    names = np.array([feature_names[j] for j in candidates])

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(xs, ys, s=15, alpha=0.7, color="#377eb8")
    ax.set_xlabel("Permutation VI (ΔAUC)")
    ax.set_ylabel("CPI (ΔAUC within leaf)")
    ax.grid(True, alpha=0.3)

    # annotate top-10 by CPI
    if ys.size > 0:
        order = np.argsort(-(np.nan_to_num(ys, nan=-np.inf)))
        top = order[: min(10, len(order))]
        for i in top:
            ax.annotate(names[i], (xs[i], ys[i]), fontsize=7, alpha=0.8)

    fig.tight_layout()
    save_fig_formats(fig, out_dir / "vi_perm_vs_cpi_scatter")
    plt.close(fig)


def plot_cpi_bar(cpi: np.ndarray, feature_names: Sequence[str], out_dir: Path, topn: int) -> None:
    vals = np.array(cpi, dtype=float)
    order = np.argsort(-(np.nan_to_num(vals, nan=-np.inf)))
    k = int(min(topn, len(order)))
    idxs = order[:k]
    fig, ax = plt.subplots(figsize=(max(6, 0.25 * k), 4))
    ax.bar(np.arange(k), vals[idxs], color="#e41a1c")
    ax.set_xticks(np.arange(k))
    ax.set_xticklabels(np.array(feature_names)[idxs], rotation=90, fontsize=8)
    ax.set_ylabel("CPI (ΔAUC)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    save_fig_formats(fig, out_dir / "cpi_bar")
    plt.close(fig)


def compute_rf_val_balaccs(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    rf_trees: int,
    rf_max_depth: int,
    runs: int,
    seed: int,
) -> np.ndarray:
    """Train multiple RFs on train-only with varying seeds and return validation balanced accuracies."""
    print(f"Computing RF validation balanced accuracies for {runs} seeds...", flush=True)

    def _one(r: int) -> float:
        rf = RandomForestClassifier(
            n_estimators=int(rf_trees),
            max_depth=int(rf_max_depth),
            n_jobs=1,
            random_state=seed + r,
            class_weight="balanced",
        )
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)
        return float(balanced_accuracy_score(y_val, y_pred))

    with tqdm_joblib(tqdm(total=int(runs), desc="RF val BA", unit="seed")):
        vals = Parallel(n_jobs=effective_joblib_n_jobs(), backend="loky")(delayed(_one)(r) for r in range(int(runs)))
    return np.array(vals, dtype=float)


# ----------------------------- RF half-sample bags + bag-level BA -----------------------------


def rf_half_bags(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    seed: int,
    num_pairs: int,
    rf_trees: int,
    rf_max_depth: int,
    top_k: int,
    min_size: int,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Run RF on complementary half-samples and return selection masks (top_k MDI) for each half.

    Select top_k features among those with positive importance; drop fits with < min_size selected.
    Returns (sel_matrix bool [<=2*num_pairs x p], half_indices_list [same length]).
    """
    p = int(len(feature_names))

    def _run_one_pair(b: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray, np.ndarray]:
        h_idx, hc_idx = generate_complementary_halves(X, y, seed=seed + b)
        def _fit_mask(indices: np.ndarray, model_seed: int) -> Optional[np.ndarray]:
            rf = RandomForestClassifier(
                n_estimators=int(rf_trees),
                max_depth=int(rf_max_depth),
                n_jobs=1,
                random_state=model_seed,
                class_weight="balanced",
            )
            rf.fit(X[indices], y[indices])
            imp = np.asarray(getattr(rf, "feature_importances_", np.zeros(p, dtype=float)), dtype=float)
            # Select top_k among features with positive importance; invalidate if fewer than min_size
            pos_idx = np.where(imp > 0.0)[0]
            if pos_idx.size == 0:
                return None
            k = int(min(int(top_k), int(pos_idx.size)))
            if k <= 0:
                return None
            # obtain indices of top-k within pos_idx by importance
            if k < pos_idx.size:
                sel_local = np.argpartition(imp[pos_idx], -k)[-k:]
                top_idx = pos_idx[sel_local]
            else:
                top_idx = pos_idx
            if top_idx.size < int(min_size):
                return None
            mask = np.zeros(p, dtype=bool)
            mask[top_idx] = True
            return mask
        sel_h = _fit_mask(h_idx, seed + 12347 * (b + 1) + 1)
        sel_hc = _fit_mask(hc_idx, seed + 12347 * (b + 1) + 2)
        return sel_h, sel_hc, h_idx, hc_idx

    print(f"RF half-bags: running {num_pairs} complementary pairs => total fits={2 * int(num_pairs)}", flush=True)
    with tqdm_joblib(tqdm(total=int(num_pairs), desc="RF half-bags", unit="pair")):
        pair_results: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = Parallel(
            n_jobs=effective_joblib_n_jobs(), backend="loky"
        )(delayed(_run_one_pair)(b) for b in range(int(num_pairs)))

    all_masks: List[np.ndarray] = []
    half_indices_list: List[np.ndarray] = []
    for sel_h, sel_hc, h_idx, hc_idx in pair_results:
        if sel_h is not None:
            all_masks.append(sel_h)
            half_indices_list.append(h_idx)
        if sel_hc is not None:
            all_masks.append(sel_hc)
            half_indices_list.append(hc_idx)

    if len(all_masks) == 0:
        sel_matrix = np.zeros((0, p), dtype=bool)
    else:
        sel_matrix = np.vstack([m.astype(bool) for m in all_masks])
    return sel_matrix, half_indices_list


def compute_bag_val_balaccs(
    sel_matrix: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
) -> np.ndarray:
    """For each selection mask (bag), refit a simple logistic classifier on train-only and
    evaluate balanced accuracy on validation. Returns an array of length = num_bags.
    """
    from sklearn.linear_model import LogisticRegression as LR

    def _one(mask: np.ndarray, rseed: int) -> float:
        mask = np.asarray(mask, dtype=bool)
        if mask.size == 0 or not np.any(mask):
            return float("nan")
        Xtr = X_train[:, mask]
        Xva = X_val[:, mask]
        # Try unregularized logistic; fallback to strong L2 if unavailable or fails
        try:
            model = LR(penalty="none", solver="lbfgs", max_iter=2000, random_state=rseed)
            model.fit(Xtr, y_train)
        except Exception:
            try:
                model = LR(penalty="l2", C=1e6, solver="lbfgs", max_iter=4000, random_state=rseed)
                model.fit(Xtr, y_train)
            except Exception:
                return float("nan")
        try:
            y_pred = model.predict(Xva)
            return float(balanced_accuracy_score(y_val, y_pred))
        except Exception:
            return float("nan")

    with tqdm_joblib(tqdm(total=int(sel_matrix.shape[0]), desc="Bag-level BA", unit="bag")):
        vals = Parallel(n_jobs=effective_joblib_n_jobs(), backend="loky")(
            delayed(_one)(sel_matrix[i], seed + 991 * (i + 1)) for i in range(int(sel_matrix.shape[0]))
        )
    return np.array(vals, dtype=float)


# ----------------------------- Main -----------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Predictive multiplicity H1: CPSS, Boruta, VI, and CPI")
    parser.add_argument("--input_dir", type=Path, default=Path("/vol/projects/BIFO/genomenet/yichen/phenotype/data/rfdata"))
    parser.add_argument("--output_dir", type=Path, default=Path("sporulation/results/h1_multiplicity"))
    parser.add_argument("--phenotype", type=str, default="Spore formation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_prev", type=float, default=0.02)
    # Clustering-aware analysis
    parser.add_argument("--clustered", action="store_true", help="Use cluster-aware downstream analysis and save under results/clustering")
    parser.add_argument("--cluster_threshold", type=float, default=0.7, help="Similarity threshold |r| for clustering NPZ selection; distance=1-threshold")
    parser.add_argument("--cluster_metric", type=str, choices=["ochiai", "phi"], default="ochiai", help="Metric used to build cluster NPZ (must match precomputed mapping)")
    # CPSS
    parser.add_argument("--cpss_pairs", type=int, default=100)
    parser.add_argument("--cpss_tau", type=float, default=0.7)
    parser.add_argument("--cpss_q", type=int, default=30)
    parser.add_argument("--cpss_cv", type=int, default=5)  # reserved, not used directly
    # Boruta
    parser.add_argument("--boruta_runs", type=int, default=50)
    parser.add_argument("--rf_trees", type=int, default=600)
    parser.add_argument("--rf_max_depth", type=int, default=30)
    # Permutation VI
    parser.add_argument("--perm_reps", type=int, default=20)
    parser.add_argument("--perm_topk", type=int, default=2000)
    # CPI
    parser.add_argument("--cpi_candidates", type=int, default=300)
    parser.add_argument("--cpi_reps", type=int, default=20)
    parser.add_argument("--leaf_max_depth", type=int, default=5)
    parser.add_argument("--leaf_min_samples", type=int, default=30)
    # Plots
    parser.add_argument("--topn", type=int, default=40)
    # Overlap network
    parser.add_argument("--net_edge_tau", type=float, default=0.3)

    args = parser.parse_args()

    input_dir: Path = args.input_dir
    phen_name = str(args.phenotype).strip()
    phen_safe = phen_name.replace(" ", "_")
    base_out_dir: Path = (Path("sporulation/results/clustering") if bool(args.clustered) else args.output_dir)
    out_dir: Path = base_out_dir / phen_safe
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir: Path = out_dir / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    seed = int(args.seed)

    # Load data
    pa_path = input_dir / "rf_presence_absence.parquet"
    long_path = input_dir / "rf_dataset.parquet"
    pa_df = _read_table(pa_path)
    long_df = _read_table(long_path)

    # Build dataset and filter (train-only prevalence)
    splits = build_dataset(pa_df, long_df, phenotype=str(args.phenotype))
    print(
        f"Dataset (pre-filter): X_train={splits.X_train.shape}, X_val={splits.X_val.shape}, X_test={splits.X_test.shape}",
        flush=True,
    )
    # Log non-NA labeled sample counts per split (species-level)
    try:
        n_tr = int(len(splits.y_train))
        n_va = int(len(splits.y_val))
        n_te = int(len(splits.y_test))
        print(
            f"Phenotype '{args.phenotype}': non-NA labeled species — train={n_tr}, val={n_va}, test={n_te}, total={n_tr + n_va + n_te}",
            flush=True,
        )
    except Exception:
        pass
    splits = filter_low_prevalence_features(splits, min_prevalence=float(args.min_prev))
    print(
        f"After prevalence filter (min_prev={args.min_prev}): X_train={splits.X_train.shape}, X_val={splits.X_val.shape}",
        flush=True,
    )
    # Quality check: require sufficient labeled samples in train and val
    min_required = 100
    if splits.X_train.shape[0] < min_required or splits.X_val.shape[0] < min_required:
        print(
            f"[Abort] Phenotype '{args.phenotype}': insufficient non-NA labeled samples after filtering. "
            f"train={splits.X_train.shape[0]}, val={splits.X_val.shape[0]} (need >= {min_required}).",
            flush=True,
        )
        return

    # Use train+val for model fitting and importance; never use test
    assert splits.X_val.shape[0] > 0, "Validation set must be non-empty"
    X_trval = np.concatenate([splits.X_train, splits.X_val], axis=0)
    y_trval = np.concatenate([splits.y_train, splits.y_val], axis=0)
    X_val = splits.X_val
    y_val = splits.y_val
    feature_names = list(splits.feature_names)
    p = len(feature_names)

    # Optional: load cluster mapping for tolerant bag similarity
    feature_to_cluster: Optional[np.ndarray] = None
    num_clusters: Optional[int] = None
    if bool(args.clustered):
        try:
            cluster_dist = float(max(0.0, min(1.0, 1.0 - float(args.cluster_threshold))))
            cluster_dir = Path("/vol/projects/BIFO/genomenet/yichen")
            npz_path = cluster_dir / f"gene_clusters_all_samples_minprev{float(args.min_prev):.3f}_metric-{str(args.cluster_metric)}_mode-abs_link-average_thr-{cluster_dist:.2f}.npz"
            with np.load(str(npz_path), allow_pickle=False) as cl:
                genes_npz = cl["genes"].astype(str).tolist()
                cl_ids = cl["cluster_ids"].astype(int)
            # Map to current feature order; unseen genes become singleton clusters
            gene_to_cluster: Dict[str, int] = {}
            for g, cid in zip(genes_npz, cl_ids):
                gene_to_cluster[str(g)] = int(cid)
            # normalize cluster ids to 0..K-1
            unique_ids = sorted(set(int(v) for v in gene_to_cluster.values()))
            id_remap = {old: i for i, old in enumerate(unique_ids)}
            mapped = np.full(p, -1, dtype=int)
            for i, g in enumerate(feature_names):
                cid = gene_to_cluster.get(str(g))
                if cid is not None:
                    mapped[i] = id_remap[int(cid)]
            # Assign singleton clusters for any missing features
            max_id = int(max(id_remap.values())) if len(id_remap) > 0 else -1
            next_id = max_id + 1
            for i in range(p):
                if mapped[i] < 0:
                    mapped[i] = next_id
                    next_id += 1
            feature_to_cluster = mapped
            num_clusters = int(mapped.max() + 1)
            print(f"[Cluster] Loaded mapping from {npz_path.name}: clusters={num_clusters}", flush=True)
        except Exception as e:
            print(f"[Cluster] Warning: Failed to load cluster mapping ({e}); proceeding without clustered analysis.", flush=True)
            feature_to_cluster = None
            num_clusters = None

    print("[Stage] Starting CPSS (L1-logistic stability selection)...", flush=True)
    # ---------------- A) CPSS (with cache) ----------------
    cpss_cache_path = cache_dir / "cpss_cache.npz"
    pi_by_gene: Dict[str, float]
    cpss_summary: Dict[str, object]
    use_cpss_cache = False
    if cpss_cache_path.exists():
        try:
            with np.load(str(cpss_cache_path), allow_pickle=False) as data:
                if not ("genes" in data.files and "pi" in data.files and "summary_json" in data.files and "sel_matrix" in data.files):
                    raise RuntimeError("Missing required fields in CPSS cache")
                cached_genes = data["genes"].astype(str).tolist()
                if cached_genes == feature_names:
                    summary_json_arr = data["summary_json"]
                    summary_json = str(summary_json_arr) if getattr(summary_json_arr, "shape", ()) == () else str(summary_json_arr.item())
                    cached_summary = json.loads(summary_json)
                    # Validate key parameters
                    if int(cached_summary.get("pairs", -1)) == int(args.cpss_pairs) and int(cached_summary.get("p", -1)) == len(feature_names):
                        cached_pi = data["pi"].astype(float)
                        pi_by_gene = {g: float(v) for g, v in zip(feature_names, cached_pi)}
                        cpss_summary = cached_summary
                        use_cpss_cache = True
                        print("[Cache] Loaded CPSS results from cache.", flush=True)
        except Exception:
            use_cpss_cache = False
    if not use_cpss_cache:
        pi_by_gene, cpss_summary, sel_matrix = cp_ss(
            X_trval,
            y_trval,
            feature_names,
            seed=seed,
            num_pairs=int(args.cpss_pairs),
            target_q=int(args.cpss_q),
            tau=float(args.cpss_tau),
            c_min=1e-3,
            c_max=1e2,
        )
        try:
            atomic_savez_compressed(
                cpss_cache_path,
                genes=np.array(feature_names, dtype=str),
                pi=np.array([pi_by_gene[g] for g in feature_names], dtype=np.float32),
                summary_json=np.array(json.dumps(cpss_summary)),
                sel_matrix=np.asarray(sel_matrix, dtype=np.uint8),
            )
            print("[Cache] Saved CPSS results to cache.", flush=True)
        except Exception as e:
            print(f"[Cache] Warning: Failed to save CPSS cache: {e}", flush=True)
    plot_cpss(pi_by_gene, tau=float(args.cpss_tau), out_dir=out_dir, topn=int(args.topn), ev_bound=cpss_summary.get("cpss_false_positives_bound"))
    print("[Stage] CPSS complete.", flush=True)

    # Compute or load CPSS validation balanced accuracy distribution
    def _compute_cpss_val_balaccs(C_star: float) -> np.ndarray:
        print("[Stage] Computing CPSS validation balanced accuracy distribution (one-time to populate cache)...", flush=True)
        def _eval_one_pair(b: int) -> Tuple[float, float]:
            # Local import to ensure availability in joblib subprocess
            from sklearn.metrics import balanced_accuracy_score as _balacc
            # reconstruct halves deterministically
            h_idx, hc_idx = generate_complementary_halves(X_trval, y_trval, seed=seed + b)
            model_h = LogisticRegression(
                solver="saga", penalty="l1", C=float(C_star), max_iter=5000, tol=1e-3, random_state=seed + 7919 * (b + 1) + 1
            )
            model_h.fit(X_trval[h_idx], y_trval[h_idx])
            y_pred_h = model_h.predict(X_val)
            bal_h = float(_balacc(y_val, y_pred_h))

            model_hc = LogisticRegression(
                solver="saga", penalty="l1", C=float(C_star), max_iter=5000, tol=1e-3, random_state=seed + 7919 * (b + 1) + 2
            )
            model_hc.fit(X_trval[hc_idx], y_trval[hc_idx])
            y_pred_hc = model_hc.predict(X_val)
            bal_hc = float(_balacc(y_val, y_pred_hc))
            return bal_h, bal_hc

        n_pairs = int(args.cpss_pairs)
        with tqdm_joblib(tqdm(total=n_pairs, desc="CPSS val BA", unit="pair")):
            vals = Parallel(n_jobs=effective_joblib_n_jobs(), backend="loky")(delayed(_eval_one_pair)(b) for b in range(n_pairs))
        arr = np.array(vals, dtype=float).reshape(-1)
        return arr

    # Try to load from cache; compute if missing
    cpss_val_balaccs: Optional[np.ndarray] = None
    try:
        with np.load(str(cpss_cache_path), allow_pickle=False) as data:
            if "val_bal_accs" in data.files and len(data["val_bal_accs"]) == 2 * int(args.cpss_pairs):
                cpss_val_balaccs = data["val_bal_accs"].astype(float)
                print("[Cache] Loaded CPSS validation BA distribution from cache.", flush=True)
    except Exception:
        pass
    if cpss_val_balaccs is None:
        C_star = float(cpss_summary.get("C_star", 1.0))
        cpss_val_balaccs = _compute_cpss_val_balaccs(C_star)
        # rewrite cache with new field
        try:
            # Append field while preserving existing arrays; write atomically
            payload = {
                "genes": np.array(feature_names, dtype=str),
                "pi": np.array([pi_by_gene[g] for g in feature_names], dtype=np.float32),
                "summary_json": np.array(json.dumps(cpss_summary)),
                "val_bal_accs": np.asarray(cpss_val_balaccs, dtype=np.float32),
            }
            try:
                with np.load(str(cpss_cache_path), allow_pickle=False) as data:
                    if "sel_matrix" in data.files:
                        payload["sel_matrix"] = np.asarray(data["sel_matrix"], dtype=np.uint8)
            except Exception:
                pass
            atomic_savez_compressed(cpss_cache_path, **payload)
            print("[Cache] Updated CPSS cache with val BA distribution.", flush=True)
        except Exception as e:
            print(f"[Cache] Warning: Failed to update CPSS cache with val BA: {e}", flush=True)

    # Stable set S_tau
    pis = np.array([pi_by_gene[g] for g in feature_names], dtype=float)
    S_tau_mask = pis >= float(args.cpss_tau)
    S_tau_indices = np.where(S_tau_mask)[0].tolist()
    S_tau_genes = [feature_names[i] for i in S_tau_indices]

    # ---------------- RF half-sample bags (with cache; used for downstream bag analyses) ----------------
    rf_halves_cache = cache_dir / "rf_halves_cache.npz"
    rf_halves_sel_matrix: Optional[np.ndarray] = None
    try:
        if rf_halves_cache.exists():
            with np.load(str(rf_halves_cache), allow_pickle=False) as rfh:
                if "genes" in rfh.files and "sel_matrix" in rfh.files:
                    cached_genes = rfh["genes"].astype(str).tolist()
                    if cached_genes == feature_names:
                        rf_halves_sel_matrix = np.asarray(rfh["sel_matrix"], dtype=np.uint8).astype(bool)
                        print("[Cache] Loaded RF half-bags from cache.", flush=True)
    except Exception:
        rf_halves_sel_matrix = None
    if rf_halves_sel_matrix is None:
        print("[Stage] Building RF half-sample bags (top-k MDI masks)...", flush=True)
        rf_halves_sel_matrix, _ = rf_half_bags(
            X_trval,
            y_trval,
            feature_names,
            seed=seed,
            num_pairs=int(args.cpss_pairs),
            rf_trees=int(args.rf_trees),
            rf_max_depth=int(args.rf_max_depth),
            top_k=100,
            min_size=50,
        )
        try:
            atomic_savez_compressed(
                rf_halves_cache,
                genes=np.array(feature_names, dtype=str),
                sel_matrix=np.asarray(rf_halves_sel_matrix, dtype=np.uint8),
                rf_pairs=np.array(int(args.cpss_pairs)),
                rf_trees=np.array(int(args.rf_trees)),
                rf_max_depth=np.array(int(args.rf_max_depth)),
                rf_top_k=np.array(int(100)),
                rf_min_size=np.array(int(50)),
            )
            print("[Cache] Saved RF half-bags to cache.", flush=True)
        except Exception as e:
            print(f"[Cache] Warning: Failed to save RF half-bags cache: {e}", flush=True)

    print("[Stage] Starting Boruta (all-relevant) and Permutation VI...", flush=True)
    # ---------------- B) Boruta + Permutation VI (with cache) ----------------
    boruta_perm_cache = cache_dir / "boruta_permvi_cache.npz"
    rf_model_cache = cache_dir / "rf_model.joblib"
    use_boruta_perm_cache = False
    confirm_rate: np.ndarray
    boruta_supports_matrix: np.ndarray
    vi_perm_arr: np.ndarray
    base_val_auc: float
    rf_model: RandomForestClassifier
    if boruta_perm_cache.exists() and rf_model_cache.exists():
        try:
            with np.load(str(boruta_perm_cache), allow_pickle=False) as data:
                required = {"genes", "boruta_runs", "rf_trees", "rf_max_depth", "perm_reps", "perm_topk", "confirm_rate", "vi_perm", "base_auc", "boruta_supports_matrix"}
                if not required.issubset(set(data.files)):
                    raise RuntimeError("Missing required fields in Boruta cache")
                cached_genes = data["genes"].astype(str).tolist()
                params_ok = (
                    int(data["boruta_runs"]) == int(args.boruta_runs)
                    and int(data["rf_trees"]) == int(args.rf_trees)
                    and int(data["rf_max_depth"]) == int(args.rf_max_depth)
                    and int(data["perm_reps"]) == int(args.perm_reps)
                    and int(data["perm_topk"]) == int(args.perm_topk)
                )
                if cached_genes == feature_names and params_ok:
                    confirm_rate = data["confirm_rate"].astype(float)
                    vi_perm_arr = data["vi_perm"].astype(float)
                    base_arr = data["base_auc"]
                    base_val_auc = float(base_arr) if getattr(base_arr, "shape", ()) == () else float(base_arr.item())
                    rf_model = joblib.load(rf_model_cache)
                    use_boruta_perm_cache = True
                    print("[Cache] Loaded Boruta + Perm VI from cache.", flush=True)
        except Exception:
            use_boruta_perm_cache = False
    if not use_boruta_perm_cache:
        confirm_rate, boruta_supports_matrix = run_boruta(
            X_trval,
            y_trval,
            rf_trees=int(args.rf_trees),
            rf_max_depth=int(args.rf_max_depth),
            runs=int(args.boruta_runs),
            seed=seed,
        )
        plot_boruta(confirm_rate, feature_names, out_dir, topn=int(args.topn))
        print("[Stage] Boruta complete. Training RF (train-only) and computing permutation VI on validation...", flush=True)

        rf_model, mdi_order_df, vi_perm_arr, base_val_auc = train_rf_and_perm_vi(
            splits.X_train,
            splits.y_train,
            X_val,
            y_val,
            feature_names,
            rf_trees=int(args.rf_trees),
            rf_max_depth=int(args.rf_max_depth),
            perm_reps=int(args.perm_reps),
            perm_topk=int(args.perm_topk),
            seed=seed,
        )
        try:
            atomic_savez_compressed(
                boruta_perm_cache,
                genes=np.array(feature_names, dtype=str),
                confirm_rate=np.asarray(confirm_rate, dtype=np.float32),
                vi_perm=np.asarray(vi_perm_arr, dtype=np.float32),
                base_auc=np.array(float(base_val_auc)),
                boruta_runs=np.array(int(args.boruta_runs)),
                rf_trees=np.array(int(args.rf_trees)),
                rf_max_depth=np.array(int(args.rf_max_depth)),
                perm_reps=np.array(int(args.perm_reps)),
                perm_topk=np.array(int(args.perm_topk)),
                boruta_supports_matrix=boruta_supports_matrix.astype(np.uint8),
            )
            joblib.dump(rf_model, rf_model_cache)
            print("[Cache] Saved Boruta + Perm VI and RF model to cache.", flush=True)
        except Exception as e:
            print(f"[Cache] Warning: Failed to save Boruta/PermVI cache: {e}", flush=True)
    else:
        # Even when loaded from cache, still create the Boruta plot (top-N)
        plot_boruta(confirm_rate, feature_names, out_dir, topn=int(args.topn))
    print("[Stage] Permutation VI ready.", flush=True)

    # ---------------- C) CPI ----------------
    # Candidate set C: union of S_tau and top-K by vi_perm (capped by --cpi_candidates by vi rank)
    vi_rank_order = np.argsort(-(np.nan_to_num(vi_perm_arr, nan=-np.inf)))
    topK_by_vi = vi_rank_order[: int(min(int(args.cpi_candidates), len(vi_rank_order)))]
    candidate_set = set(S_tau_indices).union(set(topK_by_vi.tolist()))
    # Cap by rank if needed
    if len(candidate_set) > int(args.cpi_candidates):
        # sort candidates by vi_perm descending (NaN -> -inf)
        cand_list = list(candidate_set)
        cand_list.sort(key=lambda j: (np.nan_to_num(vi_perm_arr[j], nan=-np.inf)), reverse=True)
        cand_list = cand_list[: int(args.cpi_candidates)]
        candidate_indices = cand_list
    else:
        candidate_indices = sorted(candidate_set)

    # ---------------- C) CPI (with cache) ----------------
    cpi_cache = cache_dir / "cpi_cache.npz"
    use_cpi_cache = False
    cpi_vals: np.ndarray
    cpi_se: np.ndarray
    if cpi_cache.exists():
        try:
            with np.load(str(cpi_cache), allow_pickle=False) as data:
                required = {"genes", "candidate_indices", "cpi", "cpi_se", "cpi_candidates", "cpi_reps", "leaf_max_depth", "leaf_min_samples"}
                if not required.issubset(set(data.files)):
                    raise RuntimeError("Missing required fields in CPI cache")
                cached_genes = data["genes"].astype(str).tolist()
                params_ok = (
                    int(data["cpi_candidates"]) == int(args.cpi_candidates)
                    and int(data["cpi_reps"]) == int(args.cpi_reps)
                    and int(data["leaf_max_depth"]) == int(args.leaf_max_depth)
                    and int(data["leaf_min_samples"]) == int(args.leaf_min_samples)
                )
                if cached_genes == feature_names and params_ok:
                    cached_candidate_indices = data["candidate_indices"].astype(int).tolist()
                    cpi_vals = data["cpi"].astype(float)
                    cpi_se = data["cpi_se"].astype(float)
                    candidate_indices = cached_candidate_indices
                    use_cpi_cache = True
                    print("[Cache] Loaded CPI results from cache.", flush=True)
        except Exception:
            use_cpi_cache = False
    if not use_cpi_cache:
        print(f"[Stage] Starting CPI with within-leaf permutations on |C|={len(candidate_indices)} candidates...", flush=True)
        cpi_vals, cpi_se = compute_cpi(
            rf_model,
            splits.X_train,
            splits.y_train,
            X_val,
            y_val,
            candidate_indices,
            leaf_max_depth=int(args.leaf_max_depth),
            leaf_min_samples=int(args.leaf_min_samples),
            cpi_reps=int(args.cpi_reps),
            seed=seed,
            base_val_auc=base_val_auc,
        )
        try:
            atomic_savez_compressed(
                cpi_cache,
                genes=np.array(feature_names, dtype=str),
                candidate_indices=np.array(candidate_indices, dtype=int),
                cpi=np.asarray(cpi_vals, dtype=np.float32),
                cpi_se=np.asarray(cpi_se, dtype=np.float32),
                cpi_candidates=np.array(int(args.cpi_candidates)),
                cpi_reps=np.array(int(args.cpi_reps)),
                leaf_max_depth=np.array(int(args.leaf_max_depth)),
                leaf_min_samples=np.array(int(args.leaf_min_samples)),
            )
            print("[Cache] Saved CPI results to cache.", flush=True)
        except Exception as e:
            print(f"[Cache] Warning: Failed to save CPI cache: {e}", flush=True)
    plot_vi_cpi_scatter(vi_perm_arr, cpi_vals, feature_names, candidate_indices, out_dir)
    plot_cpi_bar(cpi_vals, feature_names, out_dir, topn=int(args.topn))
    print("[Stage] CPI complete.", flush=True)

    # ---------------- Additional Plots ----------------
    # 1) Balanced accuracy distribution on validation for bag-level refits (LASSO and RF half-bags)
    try:
        # Load LASSO sel_matrix from cache
        lasso_sel_matrix_bool: Optional[np.ndarray] = None
        try:
            with np.load(str(cpss_cache_path), allow_pickle=False) as data:
                if "sel_matrix" in data.files:
                    lasso_sel_matrix_bool = np.asarray(data["sel_matrix"], dtype=np.uint8).astype(bool)
        except Exception:
            lasso_sel_matrix_bool = None

        # Plot histogram of LASSO actual bag sizes (|S| per half-fit)
        try:
            if lasso_sel_matrix_bool is not None and lasso_sel_matrix_bool.size > 0:
                sizes = np.sum(lasso_sel_matrix_bool, axis=1).astype(int)
                fig_bs, ax_bs = plt.subplots(figsize=(6, 4))
                ax_bs.hist(sizes, bins=30, color="#1f78b4", edgecolor="black", alpha=0.8)
                ax_bs.set_xlabel("LASSO bag size |S|")
                ax_bs.set_ylabel("Count")
                try:
                    ax_bs.axvline(float(np.median(sizes)), color="red", linestyle="--", linewidth=1.0, label=f"median={np.median(sizes):.0f}")
                    ax_bs.legend()
                except Exception:
                    pass
                ax_bs.grid(True, axis="y", alpha=0.3)
                fig_bs.tight_layout()
                save_fig_formats(fig_bs, out_dir / "lasso_bag_size_hist")
                plt.close(fig_bs)
                print("[Plot] Saved lasso_bag_size_hist.png/pdf/svg", flush=True)
        except Exception:
            print("[Plot] Warning: Failed to save lasso_bag_size_hist.png", flush=True)

        lasso_bag_bal_accs: Optional[np.ndarray] = None
        rf_bag_bal_accs: Optional[np.ndarray] = None

        # Try caches
        try:
            with np.load(str(cpss_cache_path), allow_pickle=False) as data:
                if "bag_val_bal_accs" in data.files:
                    arr = data["bag_val_bal_accs"]
                    lasso_bag_bal_accs = arr.astype(float)
                    print("[Cache] Loaded LASSO bag-level BA distribution from cache.", flush=True)
        except Exception:
            lasso_bag_bal_accs = None

        try:
            with np.load(str(rf_halves_cache), allow_pickle=False) as data:
                if "bag_val_bal_accs" in data.files:
                    arr = data["bag_val_bal_accs"]
                    rf_bag_bal_accs = arr.astype(float)
                    print("[Cache] Loaded RF half-bag BA distribution from cache.", flush=True)
        except Exception:
            rf_bag_bal_accs = None

        # Compute if missing
        if lasso_bag_bal_accs is None and lasso_sel_matrix_bool is not None and lasso_sel_matrix_bool.size > 0:
            lasso_bag_bal_accs = compute_bag_val_balaccs(
                lasso_sel_matrix_bool,
                splits.X_train,
                splits.y_train,
                X_val,
                y_val,
                seed=seed,
            )
            # Update CPSS cache with new field
            try:
                payload = {
                    "genes": np.array(feature_names, dtype=str),
                    "pi": np.array([pi_by_gene[g] for g in feature_names], dtype=np.float32),
                    "summary_json": np.array(json.dumps(cpss_summary)),
                    "bag_val_bal_accs": np.asarray(lasso_bag_bal_accs, dtype=np.float32),
                }
                try:
                    with np.load(str(cpss_cache_path), allow_pickle=False) as data:
                        if "sel_matrix" in data.files:
                            payload["sel_matrix"] = np.asarray(data["sel_matrix"], dtype=np.uint8)
                        if "val_bal_accs" in data.files:
                            payload["val_bal_accs"] = np.asarray(data["val_bal_accs"], dtype=np.float32)
                except Exception:
                    pass
                atomic_savez_compressed(cpss_cache_path, **payload)
                print("[Cache] Updated CPSS cache with LASSO bag-level BA.", flush=True)
            except Exception as e:
                print(f"[Cache] Warning: Failed to update CPSS cache with LASSO bag BA: {e}", flush=True)

        if rf_bag_bal_accs is None and rf_halves_sel_matrix is not None and rf_halves_sel_matrix.size > 0:
            rf_bag_bal_accs = compute_bag_val_balaccs(
                rf_halves_sel_matrix,
                splits.X_train,
                splits.y_train,
                X_val,
                y_val,
                seed=seed,
            )
            # Update RF halves cache with new field
            try:
                atomic_savez_compressed(
                    rf_halves_cache,
                    genes=np.array(feature_names, dtype=str),
                    sel_matrix=np.asarray(rf_halves_sel_matrix, dtype=np.uint8),
                    rf_pairs=np.array(int(args.cpss_pairs)),
                    rf_trees=np.array(int(args.rf_trees)),
                    rf_max_depth=np.array(int(args.rf_max_depth)),
                    rf_top_k=np.array(int(args.perm_topk)),
                    rf_min_size=np.array(int(50)),
                    bag_val_bal_accs=np.asarray(rf_bag_bal_accs, dtype=np.float32),
                )
                print("[Cache] Updated RF half-bags cache with bag-level BA.", flush=True)
            except Exception as e:
                print(f"[Cache] Warning: Failed to update RF half-bags cache with BA: {e}", flush=True)

        # Plot if we have at least one
        if lasso_bag_bal_accs is not None or rf_bag_bal_accs is not None:
            fig, ax = plt.subplots(figsize=(6, 4))
            if lasso_bag_bal_accs is not None:
                ax.hist(np.asarray(lasso_bag_bal_accs, dtype=float), bins=30, color="#a6cee3", edgecolor="black", alpha=0.8, label="LASSO bags (refit)")
            if rf_bag_bal_accs is not None:
                ax.hist(np.asarray(rf_bag_bal_accs, dtype=float), bins=30, color="#fb9a99", edgecolor="black", alpha=0.6, label="RF half-bags (refit)")
            ax.set_xlabel("Balanced accuracy (validation)")
            ax.set_ylabel("Count")
            ax.legend()
            ax.grid(True, axis="y", alpha=0.3)
            # Smart x-axis range based on observed balanced accuracies
            try:
                all_vals: List[np.ndarray] = []
                if lasso_bag_bal_accs is not None:
                    all_vals.append(np.asarray(lasso_bag_bal_accs, dtype=float))
                if rf_bag_bal_accs is not None:
                    all_vals.append(np.asarray(rf_bag_bal_accs, dtype=float))
                if len(all_vals) > 0:
                    arr = np.concatenate(all_vals)
                    arr = arr[np.isfinite(arr)]
                    if arr.size > 0:
                        vmin = float(np.min(arr))
                        vmax = float(np.max(arr))
                        span = max(1e-3, vmax - vmin)
                        pad = 0.15 * span
                        xmin = max(0.0, vmin - pad)
                        xmax = min(1.0, vmax + pad)
                        if (xmax - xmin) < 0.05:
                            mid = 0.5 * (xmin + xmax)
                            xmin = max(0.0, mid - 0.025)
                            xmax = min(1.0, mid + 0.025)
                        ax.set_xlim(xmin, xmax)
            except Exception:
                pass
            fig.tight_layout()
            save_fig_formats(fig, out_dir / "val_balanced_accuracy_hist")
            plt.close(fig)
            print("[Plot] Saved val_balanced_accuracy_hist.png/pdf/svg", flush=True)
        else:
            print("[Plot] Warning: No bag-level BA data available to plot.", flush=True)
    except Exception:
        print("[Plot] Warning: Failed to save val_balanced_accuracy_hist.png", flush=True)

    # 2) Rank comparison of LASSO (selection prob rank; only genes with any selection) vs RF (MDI rank)
    try:
        mdi_df_current = compute_feature_importance(rf_model, feature_names)
        # Build rank arrays (1=best)
        lasso_scores = np.array([pi_by_gene[g] for g in feature_names], dtype=float)
        any_selected_mask = lasso_scores > 0
        if not np.any(any_selected_mask):
            raise RuntimeError("No genes ever selected by LASSO; cannot build rank comparison plot.")
        lasso_rank = stats.rankdata(-lasso_scores[any_selected_mask], method="average")  # smaller is better
        rf_scores = mdi_df_current.set_index("gene_norm").loc[np.array(feature_names)[any_selected_mask], "importance"].to_numpy()
        rf_rank = stats.rankdata(-rf_scores, method="average")
        # Spearman correlation
        rho, pval = stats.spearmanr(lasso_rank, rf_rank)

        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        ax.scatter(rf_rank, lasso_rank, s=8, alpha=0.5, color="#33a02c")
        ax.set_xlabel("RF rank (MDI, 1=highest)")
        ax.set_ylabel("LASSO rank (selection prob, 1=highest; filtered)")
        ax.grid(True, alpha=0.3)
        ax.annotate(f"Spearman rho={rho:.3f}, p={pval:.2e}", xy=(0.05, 0.95), xycoords='axes fraction', va='top', ha='left')
        fig.tight_layout()
        save_fig_formats(fig, out_dir / "rank_compare_lasso_vs_rf")
        plt.close(fig)
        print("[Plot] Saved rank_compare_lasso_vs_rf.png/pdf/svg", flush=True)
    except Exception:
        print("[Plot] Warning: Failed to save rank_compare_lasso_vs_rf.png", flush=True)

    # 3) Rank comparison of LASSO vs RF Permutation Importance (PI)
    try:
        lasso_scores = np.array([pi_by_gene[g] for g in feature_names], dtype=float)
        vi_scores = np.asarray(vi_perm_arr, dtype=float)
        mask = np.isfinite(vi_scores) & (lasso_scores > 0)
        if int(mask.sum()) >= 3:
            lasso_rank_masked = stats.rankdata(-lasso_scores[mask], method="average")
            vi_rank = stats.rankdata(-vi_scores[mask], method="average")
            rho, pval = stats.spearmanr(lasso_rank_masked, vi_rank)

            fig, ax = plt.subplots(figsize=(5.5, 5.5))
            ax.scatter(vi_rank, lasso_rank_masked, s=8, alpha=0.5, color="#ff7f00")
            ax.set_xlabel("RF rank (Permutation VI, 1=highest)")
            ax.set_ylabel("LASSO rank (selection prob, 1=highest; filtered)")
            ax.grid(True, alpha=0.3)
            ax.annotate(f"Spearman rho={rho:.3f}, p={pval:.2e}", xy=(0.05, 0.95), xycoords='axes fraction', va='top', ha='left')
            fig.tight_layout()
            save_fig_formats(fig, out_dir / "rank_compare_lasso_vs_rf_perm")
            plt.close(fig)
            print("[Plot] Saved rank_compare_lasso_vs_rf_perm.png/pdf/svg", flush=True)
        else:
            print("[Plot] Skipped rank_compare_lasso_vs_rf_perm.png (not enough finite PI entries)", flush=True)
    except Exception:
        print("[Plot] Warning: Failed to save rank_compare_lasso_vs_rf_perm.png", flush=True)

    # 4) Rank comparison of LASSO vs CPI
    try:
        cpi_scores = np.asarray(cpi_vals, dtype=float)
        mask = np.isfinite(cpi_scores) & (lasso_scores > 0)
        if int(mask.sum()) >= 3:
            lasso_rank_masked = stats.rankdata(-lasso_scores[mask], method="average")
            cpi_rank = stats.rankdata(-cpi_scores[mask], method="average")
            rho, pval = stats.spearmanr(lasso_rank_masked, cpi_rank)

            fig, ax = plt.subplots(figsize=(5.5, 5.5))
            ax.scatter(cpi_rank, lasso_rank_masked, s=8, alpha=0.5, color="#6a3d9a")
            ax.set_xlabel("RF rank (CPI, 1=highest)")
            ax.set_ylabel("LASSO rank (selection prob, 1=highest; filtered)")
            ax.grid(True, alpha=0.3)
            ax.annotate(f"Spearman rho={rho:.3f}, p={pval:.2e}", xy=(0.05, 0.95), xycoords='axes fraction', va='top', ha='left')
            fig.tight_layout()
            save_fig_formats(fig, out_dir / "rank_compare_lasso_vs_cpi")
            plt.close(fig)
            print("[Plot] Saved rank_compare_lasso_vs_cpi.png/pdf/svg", flush=True)
        else:
            print("[Plot] Skipped rank_compare_lasso_vs_cpi.png (not enough finite CPI entries)", flush=True)
    except Exception:
        print("[Plot] Warning: Failed to save rank_compare_lasso_vs_cpi.png", flush=True)

    # ---------------- Overlap Network (combined bags: LASSO + RF) ----------------
    # Goal: one plot showing both methods' bags and their overlaps, readable with sparse edges.
    # Adds: dedup identical bags (size by frequency), report Dice and Overlap, and E[Jaccard] under random bags.
    # Uses only cached CPSS sel_matrix and Boruta supports; no recomputation.
    try:
        # Load LASSO selection matrix (F x p) and convert to boolean
        with np.load(str(cpss_cache_path), allow_pickle=False) as data:
            if "sel_matrix" not in data.files:
                print("[Plot] Missing sel_matrix in CPSS cache; skipping combined overlap network.", flush=True)
                raise RuntimeError("sel_matrix missing")
            sel_matrix_cached = np.asarray(data["sel_matrix"], dtype=np.uint8)
        lasso_bin_full = sel_matrix_cached.astype(bool)
        num_lasso_full = int(lasso_bin_full.shape[0])

        # RF half-bags (2*num_pairs x p) boolean matrix
        rf_bin_full: Optional[np.ndarray] = None
        num_rf_full = 0
        try:
            if rf_halves_sel_matrix is not None and rf_halves_sel_matrix.size > 0:
                rf_bin_full = rf_halves_sel_matrix.astype(bool)
                num_rf_full = int(rf_bin_full.shape[0])
            else:
                with np.load(str(rf_halves_cache), allow_pickle=False) as rfh:
                    if "sel_matrix" in rfh.files:
                        rf_bin_full = np.asarray(rfh["sel_matrix"], dtype=np.uint8).astype(bool)
                        num_rf_full = int(rf_bin_full.shape[0])
        except Exception:
            rf_bin_full = None

        # Optional: project feature-level bags to cluster-level bags
        def _project_bins_to_clusters(bin_mat: Optional[np.ndarray], mapping: Optional[np.ndarray], k: Optional[int]) -> Optional[np.ndarray]:
            if bin_mat is None or mapping is None or k is None:
                return bin_mat
            if bin_mat.size == 0:
                return bin_mat
            n_runs = int(bin_mat.shape[0])
            out = np.zeros((n_runs, int(k)), dtype=bool)
            for i in range(n_runs):
                row = bin_mat[i]
                idx = np.where(row)[0]
                if idx.size == 0:
                    continue
                cl_ids = mapping[idx]
                # unique cluster ids selected in this bag
                out[i, np.unique(cl_ids)] = True
            return out

        if feature_to_cluster is not None and num_clusters is not None:
            lasso_bin_full = _project_bins_to_clusters(lasso_bin_full, feature_to_cluster, num_clusters)
            if rf_bin_full is not None:
                rf_bin_full = _project_bins_to_clusters(rf_bin_full, feature_to_cluster, num_clusters)

        # Deduplicate identical bags within each method and count frequencies
        def _dedup_rows(bin_mat: Optional[np.ndarray], p_dim: int) -> Tuple[np.ndarray, np.ndarray]:
            if bin_mat is None or bin_mat.size == 0:
                return np.zeros((0, p_dim), dtype=bool), np.zeros(0, dtype=int)
            seen: Dict[bytes, int] = {}
            uniques: List[np.ndarray] = []
            counts: List[int] = []
            for r in bin_mat:
                key = r.tobytes()
                idx = seen.get(key)
                if idx is None:
                    seen[key] = len(uniques)
                    uniques.append(r)
                    counts.append(1)
                else:
                    counts[idx] += 1
            if len(uniques) == 0:
                return np.zeros((0, p_dim), dtype=bool), np.zeros(0, dtype=int)
            return np.vstack(uniques).astype(bool), np.array(counts, dtype=int)

        p_dim = int(num_clusters if (feature_to_cluster is not None and num_clusters is not None) else sel_matrix_cached.shape[1])
        lasso_bin, lasso_freq = _dedup_rows(lasso_bin_full, p_dim)
        if rf_bin_full is not None:
            rf_bin, rf_freq = _dedup_rows(rf_bin_full, p_dim)
        else:
            rf_bin, rf_freq = np.zeros((0, p_dim), dtype=bool), np.zeros(0, dtype=int)

        num_lasso = int(lasso_bin.shape[0])
        num_rf = int(rf_bin.shape[0])

        # Build combined binary matrix (N x p) where rows are deduplicated bags and record frequencies
        if num_rf > 0:
            bags_bin = np.vstack([lasso_bin, rf_bin])
            bag_freq = np.concatenate([lasso_freq, rf_freq])
        else:
            bags_bin = lasso_bin
            bag_freq = lasso_freq
        num_bags = int(bags_bin.shape[0])
        if num_bags <= 1:
            print("[Plot] Not enough bags to build overlap network.", flush=True)
            raise RuntimeError("insufficient bags")

        # Vectorized pairwise Jaccard similarity using dot-products
        # intersections[i,j] = |Bi ∩ Bj|; jaccard = |∩| / (|A ∪ B|)
        row_sums = bags_bin.sum(axis=1).astype(np.int64)
        # Use int32/64 accumulation to avoid uint8 overflow wrap-around
        intersections = (bags_bin.astype(np.int32) @ bags_bin.astype(np.int32).T).astype(np.int64)
        union = row_sums[:, None] + row_sums[None, :] - intersections
        S = np.zeros((num_bags, num_bags), dtype=float)
        with np.errstate(divide='ignore', invalid='ignore'):
            valid = union > 0
            S[valid] = intersections[valid] / union[valid]
        both_empty = union == 0
        if np.any(both_empty):
            S[both_empty] = 1.0
        np.fill_diagonal(S, 1.0)

        # Overlap coefficient (Szymkiewicz–Simpson)
        mins = np.minimum(row_sums[:, None], row_sums[None, :])
        Ovl = np.zeros((num_bags, num_bags), dtype=float)
        with np.errstate(divide='ignore', invalid='ignore'):
            valid2 = mins > 0
            Ovl[valid2] = intersections[valid2] / mins[valid2]
        both_empty2 = mins == 0
        if np.any(both_empty2):
            mask_both0 = (row_sums[:, None] == 0) & (row_sums[None, :] == 0)
            Ovl[both_empty2 & mask_both0] = 1.0
            Ovl[both_empty2 & ~mask_both0] = 0.0
        np.fill_diagonal(Ovl, 1.0)

        # Compute summary overlaps (means)
        idx_lasso = np.arange(num_lasso)
        idx_rf = np.arange(num_lasso, num_lasso + num_rf)
        def tri_mean(mat: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> float:
            if rows.size == 0 or cols.size == 0:
                return float('nan')
            sub = mat[np.ix_(rows, cols)].astype(float)
            if rows is cols or (rows.size == cols.size and np.all(rows == cols)):
                m = sub[np.triu_indices_from(sub, k=1)]
            else:
                m = sub.reshape(-1)
            return float(np.nanmean(m)) if m.size > 0 else float('nan')
        mean_within_lasso = tri_mean(S, idx_lasso, idx_lasso)
        mean_within_rf = tri_mean(S, idx_rf, idx_rf) if num_rf > 0 else float('nan')
        mean_cross = tri_mean(S, idx_lasso, idx_rf) if num_rf > 0 else float('nan')

        ovl_within_lasso = tri_mean(Ovl, idx_lasso, idx_lasso)
        ovl_within_rf = tri_mean(Ovl, idx_rf, idx_rf) if num_rf > 0 else float('nan')
        ovl_cross = tri_mean(Ovl, idx_lasso, idx_rf) if num_rf > 0 else float('nan')

        # Expected Jaccard under random bags with observed size distributions (based on original runs)
        def _expected_j_mean_within(bin_full: Optional[np.ndarray]) -> float:
            if bin_full is None or bin_full.shape[0] <= 1:
                return float('nan')
            sizes = bin_full.sum(axis=1).astype(float)
            p_feat = float(bags_bin.shape[1])
            A = sizes[:, None]
            B = sizes[None, :]
            inter_exp = (A * B) / p_feat
            union_exp = A + B - inter_exp
            with np.errstate(divide='ignore', invalid='ignore'):
                Jexp = inter_exp / union_exp
            iu = np.triu_indices_from(Jexp, k=1)
            vals = Jexp[iu]
            vals = vals[np.isfinite(vals)]
            return float(np.mean(vals)) if vals.size > 0 else float('nan')

        def _expected_j_mean_cross(bin_a: Optional[np.ndarray], bin_b: Optional[np.ndarray]) -> float:
            if bin_a is None or bin_b is None or bin_a.size == 0 or bin_b.size == 0:
                return float('nan')
            sizes_a = bin_a.sum(axis=1).astype(float)
            sizes_b = bin_b.sum(axis=1).astype(float)
            p_feat = float(bags_bin.shape[1])
            A = sizes_a[:, None]
            B = sizes_b[None, :]
            inter_exp = (A * B) / p_feat
            union_exp = A + B - inter_exp
            with np.errstate(divide='ignore', invalid='ignore'):
                Jexp = inter_exp / union_exp
            vals = Jexp.reshape(-1)
            vals = vals[np.isfinite(vals)]
            return float(np.mean(vals)) if vals.size > 0 else float('nan')

        ej_within_lasso = _expected_j_mean_within(lasso_bin_full)
        ej_within_rf = _expected_j_mean_within(rf_bin_full)
        ej_cross = _expected_j_mean_cross(lasso_bin_full, rf_bin_full)

        # 2D embedding via MDS on dissimilarity (1 - Jaccard)
        dis = 1.0 - np.clip(S, 0.0, 1.0)
        try:
            mds = MDS(n_components=2, dissimilarity="precomputed", random_state=seed)
            coords = mds.fit_transform(dis)
        except Exception:
            t = np.linspace(0, 2 * np.pi, num_bags, endpoint=False)
            coords = np.stack([np.cos(t), np.sin(t)], axis=1)

        # Build sparse edges: for each node connect to top-k neighbors over a threshold.
        edge_tau = float(args.net_edge_tau)
        edge_topk_per_node = 5
        xs, ys = coords[:, 0], coords[:, 1]
        edges_drawn = set()

        fig, ax = plt.subplots(figsize=(7.5, 7.0))
        for i in range(num_bags):
            sims = S[i].copy()
            sims[i] = -np.inf
            if not np.isfinite(sims).any():
                continue
            nn_order = np.argsort(-sims)
            count = 0
            for j in nn_order:
                if count >= edge_topk_per_node:
                    break
                if sims[j] < edge_tau:
                    continue
                key = (i, j) if i < j else (j, i)
                if key in edges_drawn:
                    continue
                w = float(np.clip(sims[j], 0.0, 1.0))
                ax.plot([xs[i], xs[j]], [ys[i], ys[j]], color="#9e9e9e", alpha=min(0.4, 0.04 + 0.36 * (w - edge_tau) / max(1e-6, 1 - edge_tau)), linewidth=0.15 + 0.8 * w)
                edges_drawn.add(key)
                count += 1

        # Node sizes by frequency of identical bags
        def _scale_sizes_by_freq(freq: np.ndarray, smin: float, smax: float) -> np.ndarray:
            if freq.size == 0:
                return np.array([])
            fmin, fmax = int(np.min(freq)), int(np.max(freq))
            if fmax <= fmin:
                return np.full_like(freq, (smin + smax) / 2.0, dtype=float)
            return smin + (smax - smin) * (freq - fmin) / float(fmax - fmin)

        lasso_sizes = _scale_sizes_by_freq(lasso_freq, 14.0, 46.0)
        rf_sizes = _scale_sizes_by_freq(rf_freq, 16.0, 50.0)

        if num_lasso > 0:
            ax.scatter(xs[:num_lasso], ys[:num_lasso], s=lasso_sizes, color="#1f78b4", alpha=0.95, edgecolors="white", linewidths=0.35, label=f"LASSO (n={num_lasso_full} runs, {num_lasso} uniq)")
        if num_rf > 0:
            ax.scatter(xs[num_lasso:num_lasso + num_rf], ys[num_lasso:num_lasso + num_rf], s=rf_sizes, color="#e31a1c", alpha=0.9, marker='^', edgecolors="white", linewidths=0.35, label=f"RF half-bags (n={num_rf_full} runs, {num_rf} uniq)")

        # Cosmetics
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("equal")
        ax.legend(loc="best", fontsize=8, frameon=True)
        ax.set_title(f"Bag overlap network (Jaccard; edge ≥ {edge_tau:.2f}; top-{edge_topk_per_node} NN)")
        # Heavy textual annotations removed to avoid masking the network

        fig.tight_layout()
        save_fig_formats(fig, out_dir / "overlap_network_combined")
        plt.close(fig)
        print("[Plot] Saved overlap_network_combined.png/pdf/svg", flush=True)

        # Supplementary: UMAP layout to check for embedding artifacts (Guttman effect)
        try:
            try:
                import umap  # type: ignore
                UMAPClass = umap.UMAP
            except Exception:
                # Fallback for older umap-learn
                import umap.umap_ as umap_alt  # type: ignore
                UMAPClass = umap_alt.UMAP
            # Ensure n_neighbors is valid after deduplication
            n_neighbors = int(max(2, min(15, num_bags - 1)))
            # Use the same distance as the main plot: Jaccard-based dissimilarity (precomputed)
            reducer = UMAPClass(n_components=2, n_neighbors=n_neighbors, min_dist=0.1, metric='precomputed', random_state=seed)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*n_jobs value.*overridden.*")
                coords_u = reducer.fit_transform(dis)
            xs_u, ys_u = coords_u[:, 0], coords_u[:, 1]
            fig2, ax2 = plt.subplots(figsize=(6.6, 5.8))
            edges_drawn = set()
            for i in range(num_bags):
                sims = S[i].copy()
                sims[i] = -np.inf
                if not np.isfinite(sims).any():
                    continue
                nn_order = np.argsort(-sims)
                count = 0
                for j in nn_order:
                    if count >= edge_topk_per_node:
                        break
                    if sims[j] < edge_tau:
                        continue
                    key = (i, j) if i < j else (j, i)
                    if key in edges_drawn:
                        continue
                    w = float(np.clip(sims[j], 0.0, 1.0))
                    ax2.plot([xs_u[i], xs_u[j]], [ys_u[i], ys_u[j]], color="#9e9e9e", alpha=min(0.4, 0.04 + 0.36 * (w - edge_tau) / max(1e-6, 1 - edge_tau)), linewidth=0.15 + 0.8 * w)
                    edges_drawn.add(key)
                    count += 1
            if num_lasso > 0:
                ax2.scatter(xs_u[:num_lasso], ys_u[:num_lasso], s=lasso_sizes, color="#1f78b4", alpha=0.95, edgecolors="white", linewidths=0.35, label=f"LASSO (uniq={num_lasso})")
            if num_rf > 0:
                ax2.scatter(xs_u[num_lasso:num_lasso + num_rf], ys_u[num_lasso:num_lasso + num_rf], s=rf_sizes, color="#e31a1c", alpha=0.9, marker='^', edgecolors="white", linewidths=0.35, label=f"RF half-bags (uniq={num_rf})")
            ax2.set_xticks([])
            ax2.set_yticks([])
            # reduce blank space and keep aspect equal
            ax2.set_aspect('equal', adjustable='box')
            xmin, xmax = float(np.min(xs_u)), float(np.max(xs_u))
            ymin, ymax = float(np.min(ys_u)), float(np.max(ys_u))
            xr = xmax - xmin
            yr = ymax - ymin
            pad_x = 0.05 * xr if xr > 0 else 0.1
            pad_y = 0.05 * yr if yr > 0 else 0.1
            ax2.set_xlim(xmin - pad_x, xmax + pad_x)
            ax2.set_ylim(ymin - pad_y, ymax + pad_y)
            ax2.legend(loc="best", fontsize=8, frameon=True)
            ax2.set_title("Bag overlap network (UMAP layout; Jaccard distance)")
            # Heavy textual annotations removed to avoid masking the network
            fig2.tight_layout()
            save_fig_formats(fig2, out_dir / "overlap_network_combined_umap")
            save_fig_formats(fig2, out_dir / "panel_A_umap_network")
            plt.close(fig2)
            print("[Plot] Saved overlap_network_combined_umap.png/pdf/svg", flush=True)
        except Exception as e:
            print(f"[Plot] Skipped UMAP supplementary figure: {e}", flush=True)
        # Panel B: Dice distributions (within LASSO, within RF, cross), computed on original runs
        try:
            def _jacc_within(bin_full: Optional[np.ndarray]) -> np.ndarray:
                if bin_full is None or bin_full.shape[0] <= 1:
                    return np.array([])
                B = bin_full.astype(np.int32)
                rs = B.sum(axis=1).astype(np.int64)
                inter = (B @ B.T).astype(np.int64)
                union = rs[:, None] + rs[None, :] - inter
                with np.errstate(divide='ignore', invalid='ignore'):
                    D = inter / union
                iu = np.triu_indices_from(D, k=1)
                vals = D[iu]
                return vals[np.isfinite(vals)]

            def _jacc_cross(A_full: Optional[np.ndarray], B_full: Optional[np.ndarray]) -> np.ndarray:
                if A_full is None or B_full is None or A_full.size == 0 or B_full.size == 0:
                    return np.array([])
                A = A_full.astype(np.int32)
                B = B_full.astype(np.int32)
                rsA = A.sum(axis=1).astype(np.int64)
                rsB = B.sum(axis=1).astype(np.int64)
                inter = (A @ B.T).astype(np.int64)
                union = rsA[:, None] + rsB[None, :] - inter
                with np.errstate(divide='ignore', invalid='ignore'):
                    D = inter / union
                vals = D.reshape(-1)
                return vals[np.isfinite(vals)]

            jacc_lasso = _jacc_within(lasso_bin_full)
            jacc_rf = _jacc_within(rf_bin_full)
            jacc_cross = _jacc_cross(lasso_bin_full, rf_bin_full)

            figB, axB = plt.subplots(figsize=(6.8, 4.2))
            parts = axB.violinplot([jacc_lasso, jacc_rf, jacc_cross], showmeans=True, showmedians=False, showextrema=False)
            for pc in parts['bodies']:
                pc.set_facecolor('#999999')
                pc.set_edgecolor('#666666')
                pc.set_alpha(0.5)
            parts['cmeans'].set_color('#333333')
            # Means for labels
            mu_l = float(np.nanmean(jacc_lasso)) if jacc_lasso.size > 0 else float('nan')
            mu_r = float(np.nanmean(jacc_rf)) if jacc_rf.size > 0 else float('nan')
            mu_c = float(np.nanmean(jacc_cross)) if jacc_cross.size > 0 else float('nan')
            axB.set_xticks([1, 2, 3])
            axB.set_xticklabels([
                f"Within LASSO (J μ={mu_l:.2f})",
                f"Within RF (J μ={mu_r:.2f})",
                f"Cross-method (J μ={mu_c:.2f})",
            ], rotation=0)
            axB.set_ylabel("Jaccard")
            axB.set_ylim(0, 1)
            axB.grid(True, axis='y', alpha=0.3)
            # Include expected random Overlap (Szymkiewicz–Simpson) as subtle footnote as well
            try:
                def _expected_j_mean_within(bin_full: Optional[np.ndarray]) -> float:
                    if bin_full is None or bin_full.shape[0] <= 1:
                        return float('nan')
                    sizes = bin_full.sum(axis=1).astype(float)
                    p_feat = float(bin_full.shape[1])
                    A = sizes[:, None]
                    B = sizes[None, :]
                    inter_exp = (A * B) / p_feat
                    union_exp = A + B - inter_exp
                    with np.errstate(divide='ignore', invalid='ignore'):
                        Jexp = inter_exp / union_exp
                    iu = np.triu_indices_from(Jexp, k=1)
                    vals = Jexp[iu]
                    vals = vals[np.isfinite(vals)]
                    return float(np.mean(vals)) if vals.size > 0 else float('nan')

                def _expected_j_mean_cross(bin_a: Optional[np.ndarray], bin_b: Optional[np.ndarray]) -> float:
                    if bin_a is None or bin_b is None or bin_a.size == 0 or bin_b.size == 0:
                        return float('nan')
                    sizes_a = bin_a.sum(axis=1).astype(float)
                    sizes_b = bin_b.sum(axis=1).astype(float)
                    p_feat = float(bin_a.shape[1])
                    A = sizes_a[:, None]
                    B = sizes_b[None, :]
                    inter_exp = (A * B) / p_feat
                    union_exp = A + B - inter_exp
                    with np.errstate(divide='ignore', invalid='ignore'):
                        Jexp = inter_exp / union_exp
                    vals = Jexp.reshape(-1)
                    vals = vals[np.isfinite(vals)]
                    return float(np.mean(vals)) if vals.size > 0 else float('nan')

                ej_l = _expected_j_mean_within(lasso_bin_full)
                ej_r = _expected_j_mean_within(rf_bin_full)
                ej_c = _expected_j_mean_cross(lasso_bin_full, rf_bin_full)
                # Also compute mean Overlap coefficient
                def _ovl_within(bin_full: Optional[np.ndarray]) -> float:
                    if bin_full is None or bin_full.shape[0] <= 1:
                        return float('nan')
                    B = bin_full.astype(np.int32)
                    rs = B.sum(axis=1).astype(np.int64)
                    inter = (B @ B.T).astype(np.int64)
                    mins = np.minimum(rs[:, None], rs[None, :])
                    with np.errstate(divide='ignore', invalid='ignore'):
                        O = inter / mins
                    iu = np.triu_indices_from(O, k=1)
                    vals = O[iu]
                    vals = vals[np.isfinite(vals)]
                    return float(np.mean(vals)) if vals.size > 0 else float('nan')
                def _ovl_cross(A_full: Optional[np.ndarray], B_full: Optional[np.ndarray]) -> float:
                    if A_full is None or B_full is None or A_full.size == 0 or B_full.size == 0:
                        return float('nan')
                    A = A_full.astype(np.int32)
                    B = B_full.astype(np.int32)
                    rsA = A.sum(axis=1).astype(np.int64)
                    rsB = B.sum(axis=1).astype(np.int64)
                    inter = (A @ B.T).astype(np.int64)
                    mins = np.minimum(rsA[:, None], rsB[None, :])
                    with np.errstate(divide='ignore', invalid='ignore'):
                        O = inter / mins
                    vals = O.reshape(-1)
                    vals = vals[np.isfinite(vals)]
                    return float(np.mean(vals)) if vals.size > 0 else float('nan')

                ovl_l = _ovl_within(lasso_bin_full)
                ovl_r = _ovl_within(rf_bin_full)
                ovl_c = _ovl_cross(lasso_bin_full, rf_bin_full)
                axB.text(0.98, 0.02, f"E[Jaccard] random: L={ej_l:.2f}; RF={ej_r:.2f}; cross={ej_c:.2f} | Overlap μ: L={ovl_l:.2f}, RF={ovl_r:.2f}, cross={ovl_c:.2f}",
                         transform=axB.transAxes, ha='right', va='bottom', fontsize=8, color="#555555")
            except Exception:
                pass
            figB.tight_layout()
            save_fig_formats(figB, out_dir / "panel_B_dice_violins")
            plt.close(figB)
            print("[Plot] Saved panel_B_dice_violins.png/pdf/svg", flush=True)
        except Exception:
            print("[Plot] Warning: Failed to build panel_B_dice_violins.png", flush=True)

        # Panel C: barcode heatmap (bags × top-50 by bag frequency)
        try:
            # frequency across original runs
            freq_full = lasso_bin_full.sum(axis=0).astype(int)
            if rf_bin_full is not None and rf_bin_full.size > 0:
                freq_full = freq_full + rf_bin_full.sum(axis=0).astype(int)
            top = int(min(50, freq_full.size))
            top_idx = np.argsort(-freq_full)[:top]
            H = bags_bin[:, top_idx].astype(int)
            # Build figure
            height = max(4.0, min(9.0, 0.08 * num_bags + 1.5))
            width = max(6.0, 0.12 * top + 2.0)
            figC, axC = plt.subplots(figsize=(width, height))
            im = axC.imshow(H, aspect='auto', interpolation='nearest', cmap='Greys')
            # separator between methods
            if num_rf > 0 and num_lasso > 0:
                axC.hlines(num_lasso - 0.5, -0.5, top - 0.5, colors="#aaaaaa", linewidth=0.6)
                # Annotate which half corresponds to LASSO vs RF half-bags
                try:
                    y_top = (num_lasso - 1) / 2.0
                    y_bot = num_lasso + (num_rf - 1) / 2.0
                    x_left = -0.48  # just inside the heatmap border
                    axC.text(x_left, y_top, "LASSO", ha='right', va='center', fontsize=8, color="#1f78b4",
                            bbox=dict(boxstyle='round,pad=0.18', fc='white', ec='none', alpha=0.65))
                    axC.text(x_left, y_bot, "RF half-bags", ha='right', va='center', fontsize=8, color="#e31a1c",
                            bbox=dict(boxstyle='round,pad=0.18', fc='white', ec='none', alpha=0.65))
                except Exception:
                    pass
            axC.set_yticks([])
            axC.set_xticks(np.arange(top))
            if feature_to_cluster is not None and num_clusters is not None:
                # Cluster mode: label as C<id> or hide labels to avoid clutter
                try:
                    top_labels = [f"C{int(i)}" for i in top_idx]
                    axC.set_xticklabels(top_labels, rotation=90, fontsize=7)
                except Exception:
                    axC.set_xticklabels([])
                axC.set_xlabel("Top-50 clusters by bag frequency")
            else:
                top_gene_names = np.array(feature_names)[top_idx]
                axC.set_xticklabels(top_gene_names, rotation=90, fontsize=7)
                axC.set_xlabel("Top-50 genes by bag frequency")
            axC.set_ylabel("Bags (deduplicated)")
            figC.tight_layout()
            save_fig_formats(figC, out_dir / "panel_C_bag_barcode_top50")
            plt.close(figC)
            print("[Plot] Saved panel_C_bag_barcode_top50.png/pdf/svg", flush=True)
            # Extra: co-occurrence correlation heatmap over training+validation data for the same top-50 genes
            try:
                if feature_to_cluster is not None and num_clusters is not None:
                    # Build cluster-level presence for selected clusters
                    X_cols: List[np.ndarray] = []
                    ftc = np.asarray(feature_to_cluster, dtype=int)
                    for cid in list(top_idx):
                        mask_c = ftc == int(cid)
                        if np.any(mask_c):
                            col = (X_trval[:, mask_c] > 0).any(axis=1).astype(np.float32)
                        else:
                            col = np.zeros(X_trval.shape[0], dtype=np.float32)
                        X_cols.append(col)
                    X_co = np.column_stack(X_cols).astype(np.float32) if len(X_cols) > 0 else np.zeros((X_trval.shape[0], 0), dtype=np.float32)
                else:
                    X_co = (X_trval[:, top_idx] > 0).astype(np.float32)
                # Handle constant columns to avoid NaNs
                if X_co.shape[0] >= 2:
                    corr = np.corrcoef(X_co, rowvar=False)
                else:
                    corr = np.eye(top, dtype=float)
                corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
                np.fill_diagonal(corr, 1.0)

                # Version 1 (default): preserve barcode order so it is directly overlayable
                if feature_to_cluster is not None and num_clusters is not None:
                    names_same = np.array([f"C{int(i)}" for i in top_idx])
                    title_label = 'Top-50 cluster co-occurrence correlation (train+val; barcode order)'
                else:
                    names_same = np.array(feature_names)[top_idx]
                    title_label = 'Top-50 gene co-occurrence correlation (train+val; barcode order)'
                fH1, aH1 = plt.subplots(figsize=(max(6.5, 0.18 * top + 2.0), max(6.0, 0.18 * top + 2.0)))
                imH1 = aH1.imshow(corr, cmap='coolwarm', vmin=-1.0, vmax=1.0, interpolation='nearest')
                aH1.set_xticks(np.arange(top))
                aH1.set_yticks(np.arange(top))
                aH1.set_xticklabels(names_same, rotation=90, fontsize=7)
                aH1.set_yticklabels(names_same, fontsize=7)
                aH1.set_title(title_label)
                fH1.colorbar(imH1, ax=aH1, fraction=0.046, pad=0.04, label='Pearson r')
                fH1.tight_layout()
                save_fig_formats(fH1, out_dir / 'cooccurrence_corr_top50')
                plt.close(fH1)
                print('[Plot] Saved cooccurrence_corr_top50.png/pdf/svg', flush=True)

                # Version 2 (clustered): reveal correlation blocks for interpretation
                try:
                    from scipy.cluster.hierarchy import linkage, leaves_list
                    from scipy.spatial.distance import squareform
                    dist = np.clip(1.0 - corr, 0.0, 2.0)
                    Z = linkage(squareform(dist, checks=False), method='average')
                    order = leaves_list(Z)
                except Exception:
                    order = np.arange(top)
                corr_ord = corr[np.ix_(order, order)]
                names_ord = names_same[order]
                fH2, aH2 = plt.subplots(figsize=(max(6.5, 0.18 * top + 2.0), max(6.0, 0.18 * top + 2.0)))
                imH2 = aH2.imshow(corr_ord, cmap='coolwarm', vmin=-1.0, vmax=1.0, interpolation='nearest')
                aH2.set_xticks(np.arange(top))
                aH2.set_yticks(np.arange(top))
                aH2.set_xticklabels(names_ord, rotation=90, fontsize=7)
                aH2.set_yticklabels(names_ord, fontsize=7)
                aH2.set_title('Top-50 cluster co-occurrence correlation (train+val; clustered)' if (feature_to_cluster is not None and num_clusters is not None) else 'Top-50 gene co-occurrence correlation (train+val; clustered)')
                fH2.colorbar(imH2, ax=aH2, fraction=0.046, pad=0.04, label='Pearson r')
                fH2.tight_layout()
                save_fig_formats(fH2, out_dir / 'cooccurrence_corr_top50_clustered')
                plt.close(fH2)
                print('[Plot] Saved cooccurrence_corr_top50_clustered.png/pdf/svg', flush=True)
            except Exception:
                print('[Plot] Warning: Failed to save cooccurrence_corr_top50.png', flush=True)
        except Exception:
            print("[Plot] Warning: Failed to build panel_C_bag_barcode_top50.png", flush=True)

        # Combined 4-panel figure: Layout with A and C larger (same size), B and D smaller stacked
        try:
            import matplotlib.image as mpimg
            imgA = mpimg.imread(out_dir / "panel_A_umap_network.png")
            imgB = mpimg.imread(out_dir / "panel_B_dice_violins.png")
            imgC = mpimg.imread(out_dir / "panel_C_bag_barcode_top50.png")
            imgD = mpimg.imread(out_dir / "val_balanced_accuracy_hist.png")
            from matplotlib.gridspec import GridSpec
            figG = plt.figure(figsize=(13, 9.5))
            # 2 columns: left = A (top) and C (bottom) same height; right = B (top) and D (bottom) same height, smaller width
            gs = GridSpec(2, 2, width_ratios=[1.3, 1.0], height_ratios=[1.0, 1.0], wspace=0.06, hspace=0.08)
            axA = figG.add_subplot(gs[0, 0])
            axC = figG.add_subplot(gs[1, 0])
            axB = figG.add_subplot(gs[0, 1])
            axD = figG.add_subplot(gs[1, 1])
            # Draw images with consistent tight borders
            for axg, img, label in [
                (axA, imgA, 'A'), (axC, imgC, 'C'), (axB, imgB, 'B'), (axD, imgD, 'D')
            ]:
                axg.imshow(img)
                axg.axis('off')
                axg.text(0.02, 0.98, label, transform=axg.transAxes, fontsize=14, fontweight='bold', va='top', ha='left', bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.6))
            figG.tight_layout(rect=[0, 0, 1, 1])
            save_fig_formats(figG, out_dir / "multiplicity_panels")
            plt.close(figG)
            print("[Plot] Saved multiplicity_panels.png/pdf/svg", flush=True)
        except Exception:
            print("[Plot] Warning: Failed to build multiplicity_panels.png", flush=True)
    except Exception:
        print("[Plot] Warning: Failed to build combined overlap network.", flush=True)

    # ---------------- Summaries and files ----------------
    # h1_summary.json
    stable_count = int(np.sum(S_tau_mask))
    boruta_half_or_more = int(np.sum(confirm_rate >= 0.5))
    stable_confirm_rates = confirm_rate[S_tau_mask] if stable_count > 0 else np.array([])
    mean_c_stable = float(np.mean(stable_confirm_rates)) if stable_confirm_rates.size > 0 else None
    median_c_stable = float(np.median(stable_confirm_rates)) if stable_confirm_rates.size > 0 else None

    # Top-10 CPI
    cpi_order = np.argsort(-(np.nan_to_num(cpi_vals, nan=-np.inf)))
    top10 = []
    for idx in cpi_order[: min(10, len(cpi_order))]:
        top10.append({
            "gene": feature_names[idx],
            "CPI": float(cpi_vals[idx]) if cpi_vals[idx] == cpi_vals[idx] else None,
            "SE": float(cpi_se[idx]) if cpi_se[idx] == cpi_se[idx] else None,
        })

    summary: Dict[str, object] = {
        "data": {
            "X_train": list(map(int, splits.X_train.shape)),
            "X_val": list(map(int, splits.X_val.shape)),
            "features_kept": int(p),
            "min_prev": float(args.min_prev),
        },
        "cpss": {
            "pairs": cpss_summary.get("pairs"),
            "total_fits": cpss_summary.get("total_fits"),
            "q_median": cpss_summary.get("q_median"),
            "p": cpss_summary.get("p"),
            "tau": cpss_summary.get("tau"),
            "|S_tau|": stable_count,
            "nogueira_stability": cpss_summary.get("nogueira_stability"),
            "E[V]_bound": cpss_summary.get("cpss_false_positives_bound"),
        },
        "boruta": {
            "runs": int(args.boruta_runs),
            "num_c_ge_0.5": boruta_half_or_more,
            "mean_c_over_S_tau": mean_c_stable,
            "median_c_over_S_tau": median_c_stable,
        },
        "cpi": {
            "|C|": int(len(candidate_indices)),
            "top10": top10,
        },
    }
    with (out_dir / "h1_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    # stable_genes.csv (<=200 rows)
    stable_genes_df = pd.DataFrame({
        "gene": [feature_names[i] for i in S_tau_indices],
        "pi_hat": [pis[i] for i in S_tau_indices],
        "boruta_confirm_rate": [confirm_rate[i] if i < len(confirm_rate) else np.nan for i in S_tau_indices],
        "perm_VI": [vi_perm_arr[i] if i < len(vi_perm_arr) else np.nan for i in S_tau_indices],
        "CPI": [cpi_vals[i] if i < len(cpi_vals) else np.nan for i in S_tau_indices],
    })
    stable_genes_df = stable_genes_df.sort_values("pi_hat", ascending=False, kind="mergesort").head(200)
    stable_genes_df.to_csv(out_dir / "stable_genes.csv", index=False)

    print("Saved figures (PNG/PDF/SVG): cpss_selection_probs, cpss_hist, cpss_size_vs_tau, boruta_confirm_rate, vi_perm_vs_cpi_scatter, cpi_bar, etc.", flush=True)
    print("Saved tables: h1_summary.json, stable_genes.csv", flush=True)
    print("Wrote outputs to:", str(out_dir), flush=True)


if __name__ == "__main__":
    main()

