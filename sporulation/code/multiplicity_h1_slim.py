#!/usr/bin/env python3
"""Predictive Multiplicity (H1) demonstration using CPSS with two base learners.

Implements Complementary-Pairs Stability Selection (CPSS) with:
- L1-logistic regression (sparse linear model)
- Random Forest (non-linear ensemble)

Shows that both methods exhibit multiplicity (different splits → different features)
and that the two methods select different feature sets.

Outputs (PNG/PDF/SVG/JSON/CSV) are written to output_dir.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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
from sklearn.metrics import balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import MDS

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train_lasso import _read_table, build_dataset, filter_low_prevalence_features


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


def save_fig_formats(fig: plt.Figure, base_path: Path, dpi: int = 150) -> None:
    """Save figure in PNG, PDF, and SVG formats."""
    base = Path(base_path)
    for ext in (".png", ".pdf", ".svg"):
        fig.savefig(base.with_suffix(ext), dpi=dpi, bbox_inches="tight")


def atomic_savez_compressed(path: Path, **arrays: object) -> None:
    """Write NPZ atomically."""
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
                pass
        os.replace(tmp_name, target)
    except Exception:
        try:
            os.unlink(tmp_name)
        except Exception:
            pass
        raise


def nonzero_mask_from_logistic(model: LogisticRegression, p: int) -> np.ndarray:
    """Get boolean mask of non-zero coefficients from L1 logistic."""
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


def generate_complementary_halves(
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return complementary stratified halves (indices)."""
    max_attempts = 20
    for attempt in range(max_attempts):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed + attempt)
        for h_idx, hc_idx in sss.split(X, y):
            y_h = y[h_idx]
            y_hc = y[hc_idx]
            if len(np.unique(y_h)) >= 2 and len(np.unique(y_hc)) >= 2:
                return h_idx, hc_idx
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    for h_idx, hc_idx in sss.split(X, y):
        return h_idx, hc_idx
    raise RuntimeError("Failed to generate complementary halves")


def nogueira_stability(pi: np.ndarray) -> float:
    """Nogueira stability index from selection probabilities."""
    p = float(len(pi))
    if p == 0:
        return float("nan")
    p_hat = np.clip(np.asarray(pi, dtype=float), 0.0, 1.0)
    k_bar = float(p_hat.sum())
    denom = k_bar * (1.0 - (k_bar / p))
    if denom <= 0:
        return float("nan")
    numer = float(np.sum(p_hat * (1.0 - p_hat)))
    return 1.0 - (numer / denom)


# ----------------------------- CPSS L1-Logistic -----------------------------


def pilot_choose_C(
    X: np.ndarray,
    y: np.ndarray,
    target_q: int,
    c_min: float,
    c_max: float,
    num_grid: int,
    seed: int,
) -> Tuple[float, List[Tuple[float, int]]]:
    """Pick C* whose selected count is closest to target_q."""
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    for tr_idx, _ in sss.split(X, y):
        X_half, y_half = X[tr_idx], y[tr_idx]
        break

    grid = np.logspace(np.log10(float(c_min)), np.log10(float(c_max)), num=int(num_grid))
    results: List[Tuple[float, int]] = []
    p = X.shape[1]
    for C in grid:
        try:
            model = LogisticRegression(
                solver="saga", penalty="l1", C=float(C), max_iter=5000, tol=1e-3,
                random_state=seed, class_weight=None,
                multi_class=("multinomial" if len(np.unique(y_half)) > 2 else "auto"),
            )
            model.fit(X_half, y_half)
            k = int(nonzero_mask_from_logistic(model, p).sum())
        except Exception:
            k = 0
        results.append((float(C), k))

    diffs = [abs(k - int(target_q)) for (_, k) in results]
    best_idx = int(np.argmin(diffs))
    return float(results[best_idx][0]), results


def cpss_logistic(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    seed: int,
    num_pairs: int,
    target_q: int,
    tau: float,
) -> Tuple[np.ndarray, Dict[str, object], np.ndarray]:
    """Run CPSS with L1-logistic. Returns (pi, summary, sel_matrix)."""
    n, p = X.shape
    n_classes = int(len(np.unique(y)))
    mc = "multinomial" if n_classes > 2 else "auto"

    C_star, grid_info = pilot_choose_C(X, y, target_q=target_q, c_min=1e-3, c_max=1e2, num_grid=40, seed=seed)
    print(f"CPSS-Logistic: C*={C_star:g}, target_q={target_q}", flush=True)

    def _fit_half(half_indices: np.ndarray, model_seed: int) -> np.ndarray:
        model = LogisticRegression(
            solver="saga", penalty="l1", C=float(C_star), max_iter=5000, tol=1e-3,
            random_state=model_seed, class_weight=None, multi_class=mc,
        )
        model.fit(X[half_indices], y[half_indices])
        return nonzero_mask_from_logistic(model, p)

    def _run_one_pair(b: int) -> Tuple[np.ndarray, np.ndarray]:
        h_idx, hc_idx = generate_complementary_halves(X, y, seed=seed + b)
        sel_h = _fit_half(h_idx, model_seed=seed + 7919 * (b + 1) + 1)
        sel_hc = _fit_half(hc_idx, model_seed=seed + 7919 * (b + 1) + 2)
        return sel_h, sel_hc

    print(f"CPSS-Logistic: running {num_pairs} pairs ({2 * num_pairs} fits)...", flush=True)
    with tqdm_joblib(tqdm(total=num_pairs, desc="CPSS-Logistic", unit="pair")):
        pair_results = Parallel(n_jobs=effective_joblib_n_jobs(), backend="loky")(
            delayed(_run_one_pair)(b) for b in range(num_pairs)
        )

    all_masks = []
    for sel_h, sel_hc in pair_results:
        all_masks.append(sel_h)
        all_masks.append(sel_hc)

    sel_matrix = np.vstack([m.astype(np.uint8) for m in all_masks])
    pi = sel_matrix.mean(axis=0)
    k_sizes = [int(m.sum()) for m in all_masks]

    summary = {
        "method": "L1-logistic",
        "C_star": float(C_star),
        "pairs": int(num_pairs),
        "total_fits": len(all_masks),
        "q_median": float(np.median(k_sizes)),
        "p": int(p),
        "tau": float(tau),
        "stable_count": int((pi >= tau).sum()),
        "nogueira_stability": float(nogueira_stability(pi)),
    }
    return pi, summary, sel_matrix


# ----------------------------- CPSS RF -----------------------------


def cpss_rf(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    seed: int,
    num_pairs: int,
    rf_trees: int,
    rf_max_depth: int,
    top_k: int,
    tau: float,
) -> Tuple[np.ndarray, Dict[str, object], np.ndarray]:
    """Run CPSS with Random Forest (top-K by MDI). Returns (pi, summary, sel_matrix)."""
    p = len(feature_names)

    def _fit_half(indices: np.ndarray, model_seed: int) -> Optional[np.ndarray]:
        rf = RandomForestClassifier(
            n_estimators=rf_trees, max_depth=rf_max_depth, n_jobs=1,
            random_state=model_seed, class_weight="balanced",
        )
        rf.fit(X[indices], y[indices])
        imp = np.asarray(rf.feature_importances_, dtype=float)
        pos_idx = np.where(imp > 0.0)[0]
        if pos_idx.size == 0:
            return None
        k = min(top_k, pos_idx.size)
        if k < pos_idx.size:
            sel_local = np.argpartition(imp[pos_idx], -k)[-k:]
            top_idx = pos_idx[sel_local]
        else:
            top_idx = pos_idx
        mask = np.zeros(p, dtype=bool)
        mask[top_idx] = True
        return mask

    def _run_one_pair(b: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        h_idx, hc_idx = generate_complementary_halves(X, y, seed=seed + b)
        sel_h = _fit_half(h_idx, seed + 12347 * (b + 1) + 1)
        sel_hc = _fit_half(hc_idx, seed + 12347 * (b + 1) + 2)
        return sel_h, sel_hc

    print(f"CPSS-RF: running {num_pairs} pairs ({2 * num_pairs} fits), top_k={top_k}...", flush=True)
    with tqdm_joblib(tqdm(total=num_pairs, desc="CPSS-RF", unit="pair")):
        pair_results = Parallel(n_jobs=effective_joblib_n_jobs(), backend="loky")(
            delayed(_run_one_pair)(b) for b in range(num_pairs)
        )

    all_masks = []
    for sel_h, sel_hc in pair_results:
        if sel_h is not None:
            all_masks.append(sel_h)
        if sel_hc is not None:
            all_masks.append(sel_hc)

    if len(all_masks) == 0:
        sel_matrix = np.zeros((0, p), dtype=np.uint8)
        pi = np.zeros(p, dtype=float)
    else:
        sel_matrix = np.vstack([m.astype(np.uint8) for m in all_masks])
        pi = sel_matrix.mean(axis=0)

    k_sizes = [int(m.sum()) for m in all_masks] if all_masks else []

    summary = {
        "method": "RandomForest",
        "rf_trees": int(rf_trees),
        "rf_max_depth": int(rf_max_depth),
        "top_k": int(top_k),
        "pairs": int(num_pairs),
        "total_fits": len(all_masks),
        "q_median": float(np.median(k_sizes)) if k_sizes else 0.0,
        "p": int(p),
        "tau": float(tau),
        "stable_count": int((pi >= tau).sum()),
        "nogueira_stability": float(nogueira_stability(pi)),
    }
    return pi, summary, sel_matrix


# ----------------------------- Plotting -----------------------------


def plot_selection_probs(pi: np.ndarray, feature_names: Sequence[str], tau: float,
                         out_dir: Path, prefix: str, topn: int = 50) -> None:
    """Bar chart of top-N selection probabilities and histogram."""
    genes = np.array(feature_names)
    order = np.argsort(-pi)

    # Top-N bar
    k = min(topn, len(genes))
    fig, ax = plt.subplots(figsize=(max(8, 0.2 * k), 4))
    colors = ["#2c7fb8" if pi[order[i]] >= tau else "#a6bddb" for i in range(k)]
    ax.bar(np.arange(k), pi[order][:k], color=colors)
    ax.axhline(tau, color="red", linestyle="--", linewidth=1, label=f"τ={tau}")
    ax.set_xticks(np.arange(k))
    ax.set_xticklabels(genes[order][:k], rotation=90, fontsize=7)
    ax.set_ylabel("Selection probability π̂")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    save_fig_formats(fig, out_dir / f"{prefix}_selection_probs")
    plt.close(fig)

    # Histogram
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(pi, bins=30, color="#74a9cf", edgecolor="black", alpha=0.9)
    ax.axvline(tau, color="red", linestyle="--", linewidth=1.5, label=f"τ={tau}")
    ax.set_xlabel("Selection probability π̂")
    ax.set_ylabel("Count")
    ax.set_xlim(0, 1)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    save_fig_formats(fig, out_dir / f"{prefix}_hist")
    plt.close(fig)


def plot_size_vs_tau(pi: np.ndarray, out_dir: Path, prefix: str) -> None:
    """Plot stable set size vs threshold τ."""
    taus = np.linspace(0.5, 0.95, 10)
    sizes = [(pi >= t).sum() for t in taus]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(taus, sizes, marker="o", color="#1b9e77", linewidth=2)
    ax.set_xlabel("Threshold τ")
    ax.set_ylabel("|S_τ| (stable set size)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_fig_formats(fig, out_dir / f"{prefix}_size_vs_tau")
    plt.close(fig)


def plot_rank_comparison(pi_lasso: np.ndarray, pi_rf: np.ndarray,
                         feature_names: Sequence[str], out_dir: Path) -> None:
    """Scatter plot comparing ranks from LASSO vs RF."""
    # Only include features selected at least once by both
    mask = (pi_lasso > 0) & (pi_rf > 0)
    if mask.sum() < 3:
        print("[Plot] Skipping rank comparison (not enough shared features)", flush=True)
        return

    lasso_rank = stats.rankdata(-pi_lasso[mask], method="average")
    rf_rank = stats.rankdata(-pi_rf[mask], method="average")
    rho, pval = stats.spearmanr(lasso_rank, rf_rank)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(rf_rank, lasso_rank, s=12, alpha=0.6, color="#33a02c")
    ax.set_xlabel("RF rank (π̂, 1=highest)")
    ax.set_ylabel("LASSO rank (π̂, 1=highest)")
    ax.set_title(f"Spearman ρ = {rho:.3f} (p = {pval:.2e})")
    ax.grid(True, alpha=0.3)

    # Add diagonal reference
    lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([0, lim], [0, lim], 'k--', alpha=0.3, linewidth=1)

    fig.tight_layout()
    save_fig_formats(fig, out_dir / "rank_comparison_lasso_vs_rf")
    plt.close(fig)
    print(f"[Plot] Rank comparison: Spearman ρ = {rho:.3f}", flush=True)


def plot_overlap_network(sel_lasso: np.ndarray, sel_rf: np.ndarray,
                         out_dir: Path, edge_tau: float = 0.2) -> Dict[str, float]:
    """Plot overlap network and compute Jaccard statistics."""
    n_lasso = sel_lasso.shape[0]
    n_rf = sel_rf.shape[0]

    # Combine
    if n_rf > 0:
        bags_bin = np.vstack([sel_lasso, sel_rf])
    else:
        bags_bin = sel_lasso
    num_bags = bags_bin.shape[0]

    if num_bags <= 1:
        print("[Plot] Not enough bags for overlap network", flush=True)
        return {}

    # Pairwise Jaccard
    row_sums = bags_bin.sum(axis=1).astype(np.int64)
    intersections = (bags_bin.astype(np.int32) @ bags_bin.astype(np.int32).T).astype(np.int64)
    union = row_sums[:, None] + row_sums[None, :] - intersections
    S = np.zeros((num_bags, num_bags), dtype=float)
    valid = union > 0
    S[valid] = intersections[valid] / union[valid]
    np.fill_diagonal(S, 1.0)

    # Compute stats
    def tri_mean(mat, rows, cols):
        if len(rows) <= 1:
            return float('nan')
        sub = mat[np.ix_(rows, cols)]
        m = sub[np.triu_indices_from(sub, k=1)]
        return float(np.nanmean(m)) if m.size > 0 else float('nan')

    def cross_mean(mat, rows, cols):
        if len(rows) == 0 or len(cols) == 0:
            return float('nan')
        return float(np.nanmean(mat[np.ix_(rows, cols)]))

    idx_lasso = np.arange(n_lasso)
    idx_rf = np.arange(n_lasso, n_lasso + n_rf)

    stats_dict = {
        "jaccard_within_lasso": tri_mean(S, idx_lasso, idx_lasso),
        "jaccard_within_rf": tri_mean(S, idx_rf, idx_rf) if n_rf > 0 else float('nan'),
        "jaccard_cross": cross_mean(S, idx_lasso, idx_rf) if n_rf > 0 else float('nan'),
    }

    # MDS embedding
    dis = 1.0 - np.clip(S, 0.0, 1.0)
    try:
        mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42, n_init=4)
        coords = mds.fit_transform(dis)
    except Exception:
        t = np.linspace(0, 2 * np.pi, num_bags, endpoint=False)
        coords = np.stack([np.cos(t), np.sin(t)], axis=1)

    xs, ys = coords[:, 0], coords[:, 1]

    # Plot
    fig, ax = plt.subplots(figsize=(7, 6))

    # Draw edges (top-k neighbors above threshold)
    edge_topk = 5
    edges_drawn = set()
    for i in range(num_bags):
        sims = S[i].copy()
        sims[i] = -np.inf
        nn_order = np.argsort(-sims)
        count = 0
        for j in nn_order:
            if count >= edge_topk or sims[j] < edge_tau:
                break
            key = (min(i, j), max(i, j))
            if key in edges_drawn:
                continue
            w = float(sims[j])
            ax.plot([xs[i], xs[j]], [ys[i], ys[j]], color="#9e9e9e",
                   alpha=0.1 + 0.4 * w, linewidth=0.3 + 0.8 * w)
            edges_drawn.add(key)
            count += 1

    # Draw nodes
    ax.scatter(xs[:n_lasso], ys[:n_lasso], s=25, color="#1f78b4", alpha=0.9,
               edgecolors="white", linewidths=0.4, label=f"LASSO (n={n_lasso})")
    if n_rf > 0:
        ax.scatter(xs[n_lasso:], ys[n_lasso:], s=30, color="#e31a1c", alpha=0.85,
                   marker='^', edgecolors="white", linewidths=0.4, label=f"RF (n={n_rf})")

    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="best", fontsize=9)
    ax.set_title(f"Bag Overlap Network (Jaccard; edges ≥ {edge_tau})")

    fig.tight_layout()
    save_fig_formats(fig, out_dir / "overlap_network")
    plt.close(fig)

    # Violin plot of Jaccard distributions
    def get_jaccard_within(sel_mat):
        if sel_mat.shape[0] <= 1:
            return np.array([])
        B = sel_mat.astype(np.int32)
        rs = B.sum(axis=1)
        inter = (B @ B.T)
        union = rs[:, None] + rs[None, :] - inter
        with np.errstate(divide='ignore', invalid='ignore'):
            J = inter / union
        return J[np.triu_indices_from(J, k=1)]

    def get_jaccard_cross(sel_a, sel_b):
        if sel_a.shape[0] == 0 or sel_b.shape[0] == 0:
            return np.array([])
        A, B = sel_a.astype(np.int32), sel_b.astype(np.int32)
        rsA, rsB = A.sum(axis=1), B.sum(axis=1)
        inter = A @ B.T
        union = rsA[:, None] + rsB[None, :] - inter
        with np.errstate(divide='ignore', invalid='ignore'):
            J = inter / union
        return J.ravel()

    jacc_lasso = get_jaccard_within(sel_lasso)
    jacc_rf = get_jaccard_within(sel_rf) if n_rf > 0 else np.array([])
    jacc_cross = get_jaccard_cross(sel_lasso, sel_rf) if n_rf > 0 else np.array([])

    fig, ax = plt.subplots(figsize=(6, 4.5))
    data = [jacc_lasso[np.isfinite(jacc_lasso)],
            jacc_rf[np.isfinite(jacc_rf)] if jacc_rf.size > 0 else np.array([0]),
            jacc_cross[np.isfinite(jacc_cross)] if jacc_cross.size > 0 else np.array([0])]

    parts = ax.violinplot(data, showmeans=True, showmedians=False, showextrema=False)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(['#1f78b4', '#e31a1c', '#984ea3'][i])
        pc.set_alpha(0.6)
    parts['cmeans'].set_color('#333333')

    means = [np.nanmean(d) if len(d) > 0 else 0 for d in data]
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels([
        f"Within LASSO\n(μ={means[0]:.3f})",
        f"Within RF\n(μ={means[1]:.3f})",
        f"Cross-method\n(μ={means[2]:.3f})",
    ])
    ax.set_ylabel("Jaccard Similarity")
    ax.set_ylim(0, 1)
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_title("Pairwise Bag Similarity")

    fig.tight_layout()
    save_fig_formats(fig, out_dir / "jaccard_violins")
    plt.close(fig)

    return stats_dict


def plot_venn_stable(pi_lasso: np.ndarray, pi_rf: np.ndarray, tau: float,
                     feature_names: Sequence[str], out_dir: Path) -> Dict[str, int]:
    """Simple Venn-style bar showing overlap of stable sets."""
    stable_lasso = set(np.where(pi_lasso >= tau)[0])
    stable_rf = set(np.where(pi_rf >= tau)[0])

    only_lasso = len(stable_lasso - stable_rf)
    only_rf = len(stable_rf - stable_lasso)
    both = len(stable_lasso & stable_rf)

    fig, ax = plt.subplots(figsize=(6, 3))
    bars = ax.barh([0, 1, 2], [only_lasso, both, only_rf],
                   color=["#1f78b4", "#984ea3", "#e31a1c"], alpha=0.8)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["LASSO only", "Both", "RF only"])
    ax.set_xlabel(f"Number of stable features (τ ≥ {tau})")

    for bar, val in zip(bars, [only_lasso, both, only_rf]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                str(val), va='center', fontsize=10)

    ax.set_xlim(0, max(only_lasso, only_rf, both) * 1.15 + 1)
    ax.grid(True, axis='x', alpha=0.3)
    fig.tight_layout()
    save_fig_formats(fig, out_dir / "stable_set_overlap")
    plt.close(fig)

    return {"lasso_only": only_lasso, "rf_only": only_rf, "both": both}


# ----------------------------- Main -----------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Predictive Multiplicity H1: CPSS with L1-Logistic and RF")
    parser.add_argument("--input_dir", type=Path, default=Path("/vol/projects/BIFO/genomenet/yichen/phenotype/data/rfdata"))
    parser.add_argument("--output_dir", type=Path, default=Path("sporulation/results/h1_multiplicity"))
    parser.add_argument("--phenotype", type=str, default="Spore formation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_prev", type=float, default=0.02)
    # CPSS
    parser.add_argument("--cpss_pairs", type=int, default=100)
    parser.add_argument("--cpss_tau", type=float, default=0.7)
    parser.add_argument("--cpss_q", type=int, default=100, help="Target features for L1-logistic")
    # RF
    parser.add_argument("--rf_trees", type=int, default=600)
    parser.add_argument("--rf_max_depth", type=int, default=30)
    parser.add_argument("--rf_top_k", type=int, default=100, help="Top-K features per RF fit")
    # Plots
    parser.add_argument("--topn", type=int, default=50)
    parser.add_argument("--net_edge_tau", type=float, default=0.2)

    args = parser.parse_args()

    # Setup
    phen_name = str(args.phenotype).strip()
    phen_safe = phen_name.replace(" ", "_")
    out_dir: Path = args.output_dir / phen_safe
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    seed = int(args.seed)

    print(f"=== Predictive Multiplicity H1: {phen_name} ===", flush=True)

    # Load data
    pa_df = _read_table(args.input_dir / "rf_presence_absence.parquet")
    long_df = _read_table(args.input_dir / "rf_dataset.parquet")
    splits = build_dataset(pa_df, long_df, phenotype=phen_name)
    print(f"Dataset: train={splits.X_train.shape}, val={splits.X_val.shape}, test={splits.X_test.shape}", flush=True)

    splits = filter_low_prevalence_features(splits, min_prevalence=float(args.min_prev))
    print(f"After filter (min_prev={args.min_prev}): {splits.X_train.shape[1]} features", flush=True)

    # Quality check
    if splits.X_train.shape[0] < 100 or splits.X_val.shape[0] < 50:
        print(f"[Abort] Insufficient samples: train={splits.X_train.shape[0]}, val={splits.X_val.shape[0]}", flush=True)
        return

    # Use train+val for CPSS
    X = np.concatenate([splits.X_train, splits.X_val], axis=0)
    y = np.concatenate([splits.y_train, splits.y_val], axis=0)
    feature_names = list(splits.feature_names)
    p = len(feature_names)
    tau = float(args.cpss_tau)

    print(f"Combined data: n={X.shape[0]}, p={p}", flush=True)

    # -------------------- CPSS L1-Logistic --------------------
    lasso_cache = cache_dir / "cpss_lasso.npz"
    if lasso_cache.exists():
        try:
            with np.load(str(lasso_cache), allow_pickle=True) as data:
                pi_lasso = data["pi"].astype(float)
                sel_lasso = data["sel_matrix"].astype(np.uint8)
                summary_lasso = json.loads(str(data["summary_json"]))
                print("[Cache] Loaded CPSS-Logistic from cache", flush=True)
        except Exception:
            pi_lasso, summary_lasso, sel_lasso = cpss_logistic(
                X, y, feature_names, seed, args.cpss_pairs, args.cpss_q, tau)
    else:
        pi_lasso, summary_lasso, sel_lasso = cpss_logistic(
            X, y, feature_names, seed, args.cpss_pairs, args.cpss_q, tau)
        atomic_savez_compressed(lasso_cache,
            pi=pi_lasso.astype(np.float32),
            sel_matrix=sel_lasso,
            summary_json=np.array(json.dumps(summary_lasso)))
        print("[Cache] Saved CPSS-Logistic", flush=True)

    plot_selection_probs(pi_lasso, feature_names, tau, out_dir, "lasso", args.topn)
    plot_size_vs_tau(pi_lasso, out_dir, "lasso")
    print(f"CPSS-Logistic: |S_τ|={summary_lasso['stable_count']}, stability={summary_lasso['nogueira_stability']:.3f}", flush=True)

    # -------------------- CPSS RF --------------------
    rf_cache = cache_dir / "cpss_rf.npz"
    if rf_cache.exists():
        try:
            with np.load(str(rf_cache), allow_pickle=True) as data:
                pi_rf = data["pi"].astype(float)
                sel_rf = data["sel_matrix"].astype(np.uint8)
                summary_rf = json.loads(str(data["summary_json"]))
                print("[Cache] Loaded CPSS-RF from cache", flush=True)
        except Exception:
            pi_rf, summary_rf, sel_rf = cpss_rf(
                X, y, feature_names, seed, args.cpss_pairs,
                args.rf_trees, args.rf_max_depth, args.rf_top_k, tau)
    else:
        pi_rf, summary_rf, sel_rf = cpss_rf(
            X, y, feature_names, seed, args.cpss_pairs,
            args.rf_trees, args.rf_max_depth, args.rf_top_k, tau)
        atomic_savez_compressed(rf_cache,
            pi=pi_rf.astype(np.float32),
            sel_matrix=sel_rf,
            summary_json=np.array(json.dumps(summary_rf)))
        print("[Cache] Saved CPSS-RF", flush=True)

    plot_selection_probs(pi_rf, feature_names, tau, out_dir, "rf", args.topn)
    plot_size_vs_tau(pi_rf, out_dir, "rf")
    print(f"CPSS-RF: |S_τ|={summary_rf['stable_count']}, stability={summary_rf['nogueira_stability']:.3f}", flush=True)

    # -------------------- Cross-Method Comparison --------------------
    plot_rank_comparison(pi_lasso, pi_rf, feature_names, out_dir)
    overlap_stats = plot_overlap_network(sel_lasso, sel_rf, out_dir, args.net_edge_tau)
    venn_stats = plot_venn_stable(pi_lasso, pi_rf, tau, feature_names, out_dir)

    print(f"Jaccard: within-LASSO={overlap_stats.get('jaccard_within_lasso', 0):.3f}, "
          f"within-RF={overlap_stats.get('jaccard_within_rf', 0):.3f}, "
          f"cross={overlap_stats.get('jaccard_cross', 0):.3f}", flush=True)
    print(f"Stable overlap: LASSO-only={venn_stats['lasso_only']}, "
          f"RF-only={venn_stats['rf_only']}, both={venn_stats['both']}", flush=True)

    # -------------------- Save Summary --------------------
    summary = {
        "phenotype": phen_name,
        "data": {
            "n_samples": int(X.shape[0]),
            "n_features": int(p),
            "min_prev": float(args.min_prev),
        },
        "cpss_lasso": summary_lasso,
        "cpss_rf": summary_rf,
        "overlap": {
            **overlap_stats,
            **venn_stats,
        },
    }
    with (out_dir / "h1_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    # Stable genes CSV
    stable_mask = (pi_lasso >= tau) | (pi_rf >= tau)
    stable_idx = np.where(stable_mask)[0]
    stable_df = pd.DataFrame({
        "gene": [feature_names[i] for i in stable_idx],
        "pi_lasso": pi_lasso[stable_idx],
        "pi_rf": pi_rf[stable_idx],
        "stable_lasso": pi_lasso[stable_idx] >= tau,
        "stable_rf": pi_rf[stable_idx] >= tau,
        "stable_both": (pi_lasso[stable_idx] >= tau) & (pi_rf[stable_idx] >= tau),
    }).sort_values("pi_lasso", ascending=False)
    stable_df.to_csv(out_dir / "stable_genes.csv", index=False)

    print(f"\nOutputs written to: {out_dir}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
