#!/usr/bin/env python3
"""Compare Panel B/C outputs between clustered and non-clustered runs using cached data."""

from __future__ import annotations

import argparse
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import FormatStrFormatter, MaxNLocator


REQUIRED_CACHE_FILES = ("cpss_cache.npz", "rf_halves_cache.npz")


@dataclass
class ModeData:
    lasso: np.ndarray
    rf: np.ndarray
    labels: Sequence[str]
    total_lasso: int
    total_rf: int
    level: str
    cluster_mapping: Optional[np.ndarray] = None
    cluster_labels: Optional[Sequence[str]] = None


@dataclass
class PanelBStats:
    jacc_lasso: np.ndarray
    jacc_rf: np.ndarray
    jacc_cross: np.ndarray
    exp_jacc_lasso: float
    exp_jacc_rf: float
    exp_jacc_cross: float
    ovl_lasso: float
    ovl_rf: float
    ovl_cross: float


def _load_selection_matrices(cache_root: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    cpss_path = cache_root / "cpss_cache.npz"
    rf_path = cache_root / "rf_halves_cache.npz"
    if not cpss_path.exists():
        raise FileNotFoundError(f"Missing CPSS cache: {cpss_path}")
    if not rf_path.exists():
        raise FileNotFoundError(f"Missing RF cache: {rf_path}")
    with np.load(str(cpss_path), allow_pickle=False) as data:
        if "sel_matrix" not in data or "genes" not in data:
            raise RuntimeError(f"CPSS cache missing fields: {list(data.files)}")
        lasso_sel = np.asarray(data["sel_matrix"], dtype=bool)
        genes = np.asarray(data["genes"], dtype=str)
    with np.load(str(rf_path), allow_pickle=False) as data:
        if "sel_matrix" not in data or "genes" not in data:
            raise RuntimeError(f"RF cache missing fields: {list(data.files)}")
        rf_sel = np.asarray(data["sel_matrix"], dtype=bool)
        rf_genes = np.asarray(data["genes"], dtype=str)
    if genes.shape != rf_genes.shape or not np.all(genes == rf_genes):
        raise RuntimeError("Gene ordering mismatch between CPSS and RF caches")
    return lasso_sel, rf_sel, genes


def _load_cluster_mapping(
    feature_names: Sequence[str],
    min_prev: float,
    cluster_threshold: float,
    cluster_metric: str,
    cluster_root: Path,
) -> Tuple[np.ndarray, int]:
    dist = max(0.0, min(1.0, 1.0 - float(cluster_threshold)))
    metric_tag = f"_metric-{cluster_metric}" if cluster_metric else ""
    npz_name = (
        f"gene_clusters_all_samples_minprev{float(min_prev):.3f}{metric_tag}"
        "_mode-abs_link-average_thr-"
        f"{dist:.2f}.npz"
    )
    npz_path = cluster_root / npz_name
    if not npz_path.exists() and metric_tag:
        alt_name = (
            f"gene_clusters_all_samples_minprev{float(min_prev):.3f}"
            "_mode-abs_link-average_thr-"
            f"{dist:.2f}.npz"
        )
        alt_path = cluster_root / alt_name
        if alt_path.exists():
            npz_path = alt_path
    if not npz_path.exists():
        raise FileNotFoundError(f"Cluster mapping not found: {npz_path}")
    with np.load(str(npz_path), allow_pickle=False) as data:
        if "genes" not in data or "cluster_ids" not in data:
            raise RuntimeError(f"Cluster NPZ missing fields: {list(data.files)}")
        mapped_genes = data["genes"].astype(str)
        cluster_ids = data["cluster_ids"].astype(int)
    gene_to_cluster: Dict[str, int] = {g: int(cid) for g, cid in zip(mapped_genes, cluster_ids)}
    unique_ids = sorted(set(gene_to_cluster.values()))
    id_remap = {old: idx for idx, old in enumerate(unique_ids)}
    mapping = np.full(len(feature_names), -1, dtype=int)
    for i, g in enumerate(feature_names):
        cid = gene_to_cluster.get(str(g))
        if cid is not None:
            mapping[i] = id_remap[int(cid)]
    next_id = (max(id_remap.values()) + 1) if id_remap else 0
    for i in range(len(feature_names)):
        if mapping[i] < 0:
            mapping[i] = next_id
            next_id += 1
    return mapping, int(mapping.max() + 1)


def _project_bins_to_clusters(bin_mat: np.ndarray, mapping: np.ndarray, k: int) -> np.ndarray:
    if bin_mat.size == 0:
        return np.zeros((0, int(k)), dtype=bool)
    out = np.zeros((bin_mat.shape[0], int(k)), dtype=bool)
    for idx, row in enumerate(bin_mat):
        cols = np.where(row)[0]
        if cols.size == 0:
            continue
        clusters = np.unique(mapping[cols])
        out[idx, clusters] = True
    return out


def _jacc_within(bin_full: Optional[np.ndarray]) -> np.ndarray:
    if bin_full is None or bin_full.size == 0 or bin_full.shape[0] <= 1:
        return np.array([])
    B = bin_full.astype(np.int32)
    row_sums = B.sum(axis=1).astype(np.int64)
    inter = (B @ B.T).astype(np.int64)
    union = row_sums[:, None] + row_sums[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        D = inter / union
    iu = np.triu_indices_from(D, k=1)
    vals = D[iu]
    return vals[np.isfinite(vals)]


def _jacc_cross(A_full: Optional[np.ndarray], B_full: Optional[np.ndarray]) -> np.ndarray:
    if (
        A_full is None
        or B_full is None
        or A_full.size == 0
        or B_full.size == 0
    ):
        return np.array([])
    A = A_full.astype(np.int32)
    B = B_full.astype(np.int32)
    rows_a = A.sum(axis=1).astype(np.int64)
    rows_b = B.sum(axis=1).astype(np.int64)
    inter = (A @ B.T).astype(np.int64)
    union = rows_a[:, None] + rows_b[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        D = inter / union
    vals = D.reshape(-1)
    return vals[np.isfinite(vals)]


def _expected_j_mean_within(bin_full: Optional[np.ndarray]) -> float:
    if bin_full is None or bin_full.size == 0 or bin_full.shape[0] <= 1:
        return float("nan")
    sizes = bin_full.sum(axis=1).astype(float)
    p_feat = float(bin_full.shape[1])
    A = sizes[:, None]
    B = sizes[None, :]
    inter_exp = (A * B) / p_feat
    union_exp = A + B - inter_exp
    with np.errstate(divide="ignore", invalid="ignore"):
        J = inter_exp / union_exp
    iu = np.triu_indices_from(J, k=1)
    vals = J[iu]
    vals = vals[np.isfinite(vals)]
    return float(np.mean(vals)) if vals.size > 0 else float("nan")


def _expected_j_mean_cross(A_full: Optional[np.ndarray], B_full: Optional[np.ndarray]) -> float:
    if (
        A_full is None
        or B_full is None
        or A_full.size == 0
        or B_full.size == 0
    ):
        return float("nan")
    sizes_a = A_full.sum(axis=1).astype(float)
    sizes_b = B_full.sum(axis=1).astype(float)
    p_feat = float(A_full.shape[1])
    A = sizes_a[:, None]
    B = sizes_b[None, :]
    inter_exp = (A * B) / p_feat
    union_exp = A + B - inter_exp
    with np.errstate(divide="ignore", invalid="ignore"):
        J = inter_exp / union_exp
    vals = J.reshape(-1)
    vals = vals[np.isfinite(vals)]
    return float(np.mean(vals)) if vals.size > 0 else float("nan")


def _overlap_within(bin_full: Optional[np.ndarray]) -> float:
    if bin_full is None or bin_full.size == 0 or bin_full.shape[0] <= 1:
        return float("nan")
    B = bin_full.astype(np.int32)
    row_sums = B.sum(axis=1).astype(np.int64)
    inter = (B @ B.T).astype(np.int64)
    mins = np.minimum(row_sums[:, None], row_sums[None, :])
    with np.errstate(divide="ignore", invalid="ignore"):
        O = inter / mins
    iu = np.triu_indices_from(O, k=1)
    vals = O[iu]
    vals = vals[np.isfinite(vals)]
    return float(np.mean(vals)) if vals.size > 0 else float("nan")


def _overlap_cross(A_full: Optional[np.ndarray], B_full: Optional[np.ndarray]) -> float:
    if (
        A_full is None
        or B_full is None
        or A_full.size == 0
        or B_full.size == 0
    ):
        return float("nan")
    A = A_full.astype(np.int32)
    B = B_full.astype(np.int32)
    rows_a = A.sum(axis=1).astype(np.int64)
    rows_b = B.sum(axis=1).astype(np.int64)
    inter = (A @ B.T).astype(np.int64)
    mins = np.minimum(rows_a[:, None], rows_b[None, :])
    with np.errstate(divide="ignore", invalid="ignore"):
        O = inter / mins
    vals = O.reshape(-1)
    vals = vals[np.isfinite(vals)]
    return float(np.mean(vals)) if vals.size > 0 else float("nan")


def _clean(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr
    arr = np.asarray(arr, dtype=float)
    mask = np.isfinite(arr)
    return arr[mask]


def _bootstrap_mean_diff(
    ref: np.ndarray,
    alt: np.ndarray,
    rng: np.random.Generator,
    iters: int = 4000,
) -> Tuple[float, Tuple[float, float]]:
    ref = _clean(ref)
    alt = _clean(alt)
    if ref.size == 0 or alt.size == 0:
        return float("nan"), (float("nan"), float("nan"))
    ref_mean = float(np.mean(ref))
    alt_mean = float(np.mean(alt))
    if iters <= 0:
        return alt_mean - ref_mean, (float("nan"), float("nan"))
    ref_idx = rng.integers(0, ref.size, size=(iters, ref.size))
    alt_idx = rng.integers(0, alt.size, size=(iters, alt.size))
    ref_samples = ref[ref_idx].mean(axis=1)
    alt_samples = alt[alt_idx].mean(axis=1)
    diff_samples = alt_samples - ref_samples
    lo, hi = np.percentile(diff_samples, [2.5, 97.5])
    return alt_mean - ref_mean, (float(lo), float(hi))


def _compute_panel_b_stats(mode: ModeData) -> PanelBStats:
    j_l = _jacc_within(mode.lasso)
    j_r = _jacc_within(mode.rf)
    j_c = _jacc_cross(mode.lasso, mode.rf)
    exp_l = _expected_j_mean_within(mode.lasso)
    exp_r = _expected_j_mean_within(mode.rf)
    exp_c = _expected_j_mean_cross(mode.lasso, mode.rf)
    ovl_l = _overlap_within(mode.lasso)
    ovl_r = _overlap_within(mode.rf)
    ovl_c = _overlap_cross(mode.lasso, mode.rf)
    return PanelBStats(j_l, j_r, j_c, exp_l, exp_r, exp_c, ovl_l, ovl_r, ovl_c)


def _panel_b_categories(
    non_stats: PanelBStats,
    cluster_stats: PanelBStats,
) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    categories = [
        ("Within LASSO", non_stats.jacc_lasso, cluster_stats.jacc_lasso),
        ("Within RF", non_stats.jacc_rf, cluster_stats.jacc_rf),
        ("Cross-method", non_stats.jacc_cross, cluster_stats.jacc_cross),
    ]
    cleaned: List[Tuple[str, np.ndarray, np.ndarray]] = []
    for label, arr_non, arr_cluster in categories:
        cleaned.append((label, _clean(arr_non), _clean(arr_cluster)))
    return cleaned


def _prepare_mode(
    base_dir: Path,
    phenotype_safe: str,
    clustered: bool,
    min_prev: float,
    cluster_threshold: float,
    cluster_metric: str,
    cluster_root: Path,
) -> ModeData:
    cache_root = base_dir / phenotype_safe / ".cache"
    lasso_raw, rf_raw, genes = _load_selection_matrices(cache_root)
    lasso = np.asarray(lasso_raw, dtype=bool)
    rf = np.asarray(rf_raw, dtype=bool)
    mapping: Optional[np.ndarray] = None
    cluster_labels: Optional[List[str]] = None

    if clustered:
        mapping, num_clusters = _load_cluster_mapping(
            genes,
            min_prev=min_prev,
            cluster_threshold=cluster_threshold,
            cluster_metric=cluster_metric,
            cluster_root=cluster_root,
        )
        lasso = _project_bins_to_clusters(lasso, mapping, num_clusters)
        rf = _project_bins_to_clusters(rf, mapping, num_clusters)
        cluster_labels = [f"C{i}" for i in range(num_clusters)]
        labels = list(cluster_labels)
        return ModeData(
            lasso=lasso,
            rf=rf,
            labels=labels,
            total_lasso=lasso_raw.shape[0],
            total_rf=rf_raw.shape[0],
            level="cluster",
            cluster_mapping=None,
            cluster_labels=cluster_labels,
        )

    try:
        mapping, num_clusters = _load_cluster_mapping(
            genes,
            min_prev=min_prev,
            cluster_threshold=cluster_threshold,
            cluster_metric=cluster_metric,
            cluster_root=cluster_root,
        )
        cluster_labels = [f"C{i}" for i in range(num_clusters)]
    except FileNotFoundError:
        mapping = None
        cluster_labels = None
    except RuntimeError:
        mapping = None
        cluster_labels = None

    labels = [str(g) for g in genes]
    return ModeData(
        lasso=lasso,
        rf=rf,
        labels=labels,
        total_lasso=lasso_raw.shape[0],
        total_rf=rf_raw.shape[0],
        level="gene",
        cluster_mapping=mapping,
        cluster_labels=cluster_labels,
    )


def _plot_panel_b(
    non_stats: PanelBStats,
    cluster_stats: PanelBStats,
    output_path: Path,
    seed: int,
) -> None:
    categories = _panel_b_categories(non_stats, cluster_stats)
    colors = {"non": "#1f78b4", "cluster": "#e31a1c"}
    rng = np.random.default_rng(seed)
    means_non = []
    means_cluster = []
    diffs = []
    cis: List[Tuple[float, float]] = []

    fig = plt.figure(figsize=(7.2, 5.6))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3.4, 1.2], hspace=0.25)
    ax = fig.add_subplot(gs[0])

    pos = np.arange(1, len(categories) + 1)
    offset = 0.18
    widths = 0.26
    data_non = [arr if arr.size > 0 else np.array([np.nan]) for _, arr, _ in categories]
    data_cluster = [arr if arr.size > 0 else np.array([np.nan]) for _, _, arr in categories]

    vp_non = ax.violinplot(data_non, positions=pos - offset, widths=widths, showmeans=True, showextrema=False)
    vp_cluster = ax.violinplot(data_cluster, positions=pos + offset, widths=widths, showmeans=True, showextrema=False)
    for pc in vp_non["bodies"]:
        pc.set_facecolor(colors["non"])
        pc.set_alpha(0.45)
        pc.set_edgecolor(colors["non"])
    vp_non["cmeans"].set_color(colors["non"])
    for pc in vp_cluster["bodies"]:
        pc.set_facecolor(colors["cluster"])
        pc.set_alpha(0.45)
        pc.set_edgecolor(colors["cluster"])
    vp_cluster["cmeans"].set_color(colors["cluster"])

    for idx, (label, arr_non, arr_cluster) in enumerate(categories):
        cn = arr_non
        cc = arr_cluster
        means_non.append(float(np.mean(cn)) if cn.size > 0 else float("nan"))
        means_cluster.append(float(np.mean(cc)) if cc.size > 0 else float("nan"))
        diff, ci = _bootstrap_mean_diff(cn, cc, rng)
        diffs.append(diff)
        cis.append(ci)
        ax.scatter(
            pos[idx] - offset,
            means_non[-1],
            color=colors["non"],
            s=18,
            zorder=3,
        )
        ax.scatter(
            pos[idx] + offset,
            means_cluster[-1],
            color=colors["cluster"],
            s=18,
            zorder=3,
        )

    ax.set_xticks(pos)
    ax.set_xticklabels([c[0] for c in categories])
    ax.set_ylabel("Jaccard similarity")
    ax.set_ylim(0, 1)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(
        [plt.Line2D([0], [0], color=colors["non"], lw=3), plt.Line2D([0], [0], color=colors["cluster"], lw=3)],
        ["Non-clustered", "Clustered"],
        loc="upper left",
        frameon=False,
    )

    ax_diff = fig.add_subplot(gs[1], sharex=ax)
    ax_diff.axhline(0.0, color="#555555", linewidth=1)
    for idx, (pos_val, diff, ci) in enumerate(zip(pos, diffs, cis)):
        if math.isnan(diff):
            continue
        ax_diff.bar(
            pos_val,
            diff,
            color=colors["cluster"],
            alpha=0.65,
            width=0.35,
        )
        if all(math.isnan(x) for x in ci):
            continue
        ax_diff.vlines(pos_val, ci[0], ci[1], color="#333333", linewidth=1.1)
    ax_diff.set_xticks(pos)
    ax_diff.set_xticklabels([c[0] for c in categories], rotation=0)
    ax_diff.set_xlabel("Similarity category")
    ax_diff.set_ylabel("Δ mean J (cluster - non)")
    ax_diff.grid(True, axis="y", alpha=0.3)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_panel_b_grid(
    stats_by_phen: Dict[str, Tuple[PanelBStats, PanelBStats]],
    output_path: Path,
    seed: int,
    cols: int,
) -> None:
    phen_names = sorted(stats_by_phen.keys())
    if not phen_names:
        return
    cols = max(1, min(cols, len(phen_names)))
    rows = int(math.ceil(len(phen_names) / float(cols)))
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(cols * 3.2, rows * 3.0),
        sharey=True,
    )
    if isinstance(axes, np.ndarray):
        axes_arr = axes.reshape(rows, cols)
    else:
        axes_arr = np.array([[axes]])
    colors = {"non": "#1f78b4", "cluster": "#e31a1c"}

    for idx, phen in enumerate(phen_names):
        row = idx // cols
        col = idx % cols
        ax = axes_arr[row, col]
        non_stats, cluster_stats = stats_by_phen[phen]
        categories = _panel_b_categories(non_stats, cluster_stats)
        pos = np.arange(1, len(categories) + 1)
        offset = 0.16
        widths = 0.22
        data_non = [arr if arr.size > 0 else np.array([np.nan]) for _, arr, _ in categories]
        data_cluster = [arr if arr.size > 0 else np.array([np.nan]) for _, _, arr in categories]
        vp_non = ax.violinplot(
            data_non,
            positions=pos - offset,
            widths=widths,
            showmeans=True,
            showextrema=False,
        )
        vp_cluster = ax.violinplot(
            data_cluster,
            positions=pos + offset,
            widths=widths,
            showmeans=True,
            showextrema=False,
        )
        for pc in vp_non["bodies"]:
            pc.set_facecolor(colors["non"])
            pc.set_alpha(0.35)
            pc.set_edgecolor(colors["non"])
        vp_non["cmeans"].set_color(colors["non"])
        for pc in vp_cluster["bodies"]:
            pc.set_facecolor(colors["cluster"])
            pc.set_alpha(0.35)
            pc.set_edgecolor(colors["cluster"])
        vp_cluster["cmeans"].set_color(colors["cluster"])

        for pos_idx, (label, arr_non, arr_cluster) in enumerate(categories):
            mean_non = float(np.mean(arr_non)) if arr_non.size > 0 else float("nan")
            mean_cluster = float(np.mean(arr_cluster)) if arr_cluster.size > 0 else float("nan")
            ax.scatter(
                pos[pos_idx] - offset,
                mean_non,
                color=colors["non"],
                s=10,
                zorder=3,
            )
            ax.scatter(
                pos[pos_idx] + offset,
                mean_cluster,
                color=colors["cluster"],
                s=10,
                zorder=3,
            )
        ax.set_xticks(pos)
        ax.tick_params(axis="x", length=0)
        if row == rows - 1:
            ax.set_xticklabels([c[0] for c in categories], rotation=30, ha="right", fontsize=8)
        else:
            ax.set_xticklabels([])
            ax.tick_params(axis="x", labelbottom=False)
        ax.set_ylim(0, 1)
        ax.grid(True, axis="y", alpha=0.25)
        title = phen.replace("_", " ")
        ax.set_title(title, fontsize=11)
        if col == 0:
            ax.set_ylabel("Jaccard")
        else:
            ax.set_ylabel("")

    for idx in range(len(phen_names), rows * cols):
        row = idx // cols
        col = idx % cols
        axes_arr[row, col].axis("off")

    fig.legend(
        [plt.Line2D([0], [0], color=colors["non"], lw=3), plt.Line2D([0], [0], color=colors["cluster"], lw=3)],
        ["Non-clustered", "Clustered"],
        loc="upper center",
        ncol=2,
        frameon=False,
        fontsize=11,
        bbox_to_anchor=(0.5, 0.995),
    )
    fig.supxlabel("Similarity category", fontsize=11)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _frequency_for_mode(
    mode: ModeData,
    labels: Sequence[str],
    target: str,
) -> np.ndarray:
    total = float(mode.total_lasso + mode.total_rf)
    freq = np.zeros(len(labels), dtype=float)
    if total <= 0:
        return freq

    if target == "cluster":
        counts = np.zeros(len(labels), dtype=float)
        if mode.level == "cluster":
            label_to_idx = {str(label): idx for idx, label in enumerate(mode.labels)}
            combined = (
                mode.lasso.astype(np.int32).sum(axis=0)
                + mode.rf.astype(np.int32).sum(axis=0)
            ).astype(float)
            for i, label in enumerate(labels):
                idx = label_to_idx.get(str(label))
                if idx is None or idx >= combined.size:
                    continue
                counts[i] = combined[idx]
        elif mode.cluster_mapping is not None and mode.cluster_labels is not None:
            mapping = mode.cluster_mapping
            if mapping is None or mapping.shape[0] != mode.lasso.shape[1]:
                raise ValueError("Cluster mapping does not match feature dimension for gene-level mode")
            label_to_cid = {str(label): idx for idx, label in enumerate(mode.cluster_labels)}
            gene_counts = (
                mode.lasso.astype(np.int32).sum(axis=0)
                + mode.rf.astype(np.int32).sum(axis=0)
            ).astype(float)
            gene_freq = gene_counts / total
            for i, label in enumerate(labels):
                cid = label_to_cid.get(str(label))
                if cid is None:
                    continue
                mask = mapping == cid
                if not np.any(mask):
                    continue
                counts[i] = float(np.mean(gene_freq[mask]) * total)
        else:
            raise ValueError("Cluster information unavailable for mode")
    else:
        counts = np.zeros(len(labels), dtype=float)
        label_to_idx = {str(label): idx for idx, label in enumerate(mode.labels)}
        combined = (
            mode.lasso.astype(np.int32).sum(axis=0)
            + mode.rf.astype(np.int32).sum(axis=0)
        )
        for i, label in enumerate(labels):
            idx = label_to_idx.get(str(label))
            if idx is None or idx >= combined.size:
                continue
            counts[i] = float(combined[idx])

    freq = counts / total
    return freq


def _panel_c_vectors(
    non_mode: ModeData,
    cluster_mode: ModeData,
    top_k: int,
) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray]:
    use_clusters = (
        cluster_mode.level == "cluster"
        and cluster_mode.labels
        and non_mode.cluster_mapping is not None
        and non_mode.cluster_labels is not None
    )

    if use_clusters:
        base_labels = list(cluster_mode.labels)
        freq_cluster_vec = _frequency_for_mode(cluster_mode, base_labels, target="cluster")
        freq_non_vec = _frequency_for_mode(non_mode, base_labels, target="cluster")
        combined = np.maximum(freq_cluster_vec, freq_non_vec)
        if top_k > 0:
            order = np.argsort(-combined)[:top_k]
            base_labels = [base_labels[i] for i in order]
            freq_cluster_vec = freq_cluster_vec[order]
            freq_non_vec = freq_non_vec[order]
        return base_labels, freq_non_vec, freq_cluster_vec, freq_cluster_vec - freq_non_vec

    freq_non = {
        str(label): val
        for label, val in zip(non_mode.labels, _frequency_for_mode(non_mode, non_mode.labels, target="native"))
    }
    freq_cluster = {
        str(label): val
        for label, val in zip(cluster_mode.labels, _frequency_for_mode(cluster_mode, cluster_mode.labels, target="native"))
    }
    combined: Dict[str, float] = {}
    for label, val in freq_non.items():
        combined[label] = max(combined.get(label, 0.0), float(val))
    for label, val in freq_cluster.items():
        combined[label] = max(combined.get(label, 0.0), float(val))
    ordered = sorted(combined.items(), key=lambda kv: kv[1], reverse=True)
    if top_k > 0:
        ordered = ordered[: top_k]
    labels = [lbl for lbl, _ in ordered]
    if not labels:
        labels = list(freq_non.keys())[:top_k]
    freq_non_vec = np.array([freq_non.get(lbl, 0.0) for lbl in labels], dtype=float)
    freq_cluster_vec = np.array([freq_cluster.get(lbl, 0.0) for lbl in labels], dtype=float)
    diff_vec = freq_cluster_vec - freq_non_vec
    return labels, freq_non_vec, freq_cluster_vec, diff_vec


def _plot_panel_c(
    non_mode: ModeData,
    cluster_mode: ModeData,
    output_path: Path,
    top_k: int,
) -> None:
    labels_sorted, freq_non_vec, freq_cluster_vec, diff_vec = _panel_c_vectors(
        non_mode,
        cluster_mode,
        top_k,
    )

    figure_width = max(6.0, 0.18 * len(labels_sorted) + 2.0)
    fig = plt.figure(figsize=(figure_width, 4.6))
    gs = gridspec.GridSpec(3, 1, height_ratios=[0.8, 2.2, 0.6], hspace=0.18)

    ax_top = fig.add_subplot(gs[0])
    xs = np.arange(len(labels_sorted))
    ax_top.plot(xs, freq_non_vec, color="#1f78b4", label="Non-clustered", linewidth=1.5)
    ax_top.plot(xs, freq_cluster_vec, color="#e31a1c", label="Clustered", linewidth=1.5)
    ax_top.set_ylabel("Frequency")
    ax_top.set_ylim(0, max(0.01, float(np.max([freq_non_vec.max(), freq_cluster_vec.max(), 0.01]))) * 1.1)
    ax_top.legend(loc="upper right", frameon=False, fontsize=8)
    ax_top.set_xticks([])
    ax_top.grid(True, axis="y", alpha=0.3)

    ax_heat = fig.add_subplot(gs[1])
    heat = np.vstack([freq_non_vec, freq_cluster_vec])
    heat_vmax = float(np.max(heat))
    if heat_vmax <= 0:
        heat_vmax = 0.01
    im = ax_heat.imshow(heat, aspect="auto", cmap="Greys", vmin=0.0, vmax=heat_vmax)
    ax_heat.set_yticks([0, 1])
    ax_heat.set_yticklabels(["Non-clustered", "Clustered"], rotation=0)
    ax_heat.set_xticks([])
    ax_heat.set_ylabel("Panel C rows")
    cbar_heat = fig.colorbar(
        im,
        ax=ax_heat,
        orientation="horizontal",
        fraction=0.08,
        pad=0.15,
    )
    cbar_heat.set_label("Selection frequency")
    cbar_heat.formatter = FormatStrFormatter("%.2f")
    cbar_heat.locator = MaxNLocator(4)
    cbar_heat.update_ticks()

    ax_diff = fig.add_subplot(gs[2], sharex=ax_heat)
    span = float(np.max(np.abs(diff_vec)))
    if span <= 0:
        span = 0.01
    diff_im = ax_diff.imshow(diff_vec[np.newaxis, :], aspect="auto", cmap="coolwarm", vmin=-span, vmax=span)
    ax_diff.set_yticks([0])
    ax_diff.set_yticklabels(["Δ freq (cluster - non)"])
    ax_diff.set_xticks([])
    cbar_diff = fig.colorbar(
        diff_im,
        ax=ax_diff,
        orientation="horizontal",
        fraction=0.35,
        pad=0.5,
    )
    cbar_diff.set_label("Δ (cluster - non)")
    cbar_diff.formatter = FormatStrFormatter("%.3f")
    tick_vals = np.linspace(-span, span, 5)
    cbar_diff.set_ticks(tick_vals)
    cbar_diff.update_ticks()
    ax_diff.grid(False)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fig.tight_layout()
    fig.supxlabel("Feature rank", fontsize=11)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_panel_c_grid(
    modes_by_phen: Dict[str, Tuple[ModeData, ModeData]],
    output_path: Path,
    top_k: int,
    cols: int,
) -> None:
    phen_names = sorted(modes_by_phen.keys())
    if not phen_names:
        return
    prepared: List[Tuple[str, List[str], np.ndarray, np.ndarray, np.ndarray]] = []
    heat_vmax = 0.0
    diff_span = 0.0
    for phen in phen_names:
        non_mode, cluster_mode = modes_by_phen[phen]
        labels, freq_non_vec, freq_cluster_vec, diff_vec = _panel_c_vectors(
            non_mode,
            cluster_mode,
            top_k,
        )
        heat = np.vstack([freq_non_vec, freq_cluster_vec])
        if heat.size > 0:
            heat_vmax = max(heat_vmax, float(np.max(heat)))
        if diff_vec.size > 0:
            diff_span = max(diff_span, float(np.max(np.abs(diff_vec))))
        prepared.append((phen, labels, freq_non_vec, freq_cluster_vec, diff_vec))
    if heat_vmax <= 0:
        heat_vmax = 0.01
    if diff_span <= 0:
        diff_span = 0.01

    cols = max(1, min(cols, len(phen_names)))
    rows = int(math.ceil(len(phen_names) / float(cols)))
    fig = plt.figure(figsize=(cols * 3.4, rows * 3.2))
    outer = gridspec.GridSpec(rows, cols, hspace=0.35, wspace=0.2)
    heat_axes: List[plt.Axes] = []
    diff_axes: List[plt.Axes] = []
    first_heat_im = None
    first_diff_im = None

    for idx, (phen, labels, freq_non_vec, freq_cluster_vec, diff_vec) in enumerate(prepared):
        row = idx // cols
        col = idx % cols
        sub = outer[row, col].subgridspec(2, 1, height_ratios=[1.0, 0.32], hspace=0.08)
        ax_heat = fig.add_subplot(sub[0])
        ax_diff = fig.add_subplot(sub[1], sharex=ax_heat)
        heat = np.vstack([freq_non_vec, freq_cluster_vec]) if freq_non_vec.size > 0 else np.zeros((2, 1))
        heat_im = ax_heat.imshow(
            heat,
            aspect="auto",
            cmap="Greys",
            vmin=0.0,
            vmax=heat_vmax,
        )
        if first_heat_im is None:
            first_heat_im = heat_im
        heat_axes.append(ax_heat)
        ax_heat.set_yticks([0, 1])
        if col == 0:
            ax_heat.set_yticklabels(["Non", "Cluster"], fontsize=8)
        else:
            ax_heat.set_yticklabels([])
            ax_heat.tick_params(axis="y", labelleft=False)
        ax_heat.set_xticks([])
        ax_heat.set_title(phen.replace("_", " "), fontsize=11)

        diff_data = diff_vec[np.newaxis, :] if diff_vec.size > 0 else np.zeros((1, heat.shape[1]))
        diff_im = ax_diff.imshow(
            diff_data,
            aspect="auto",
            cmap="coolwarm",
            vmin=-diff_span,
            vmax=diff_span,
        )
        if first_diff_im is None:
            first_diff_im = diff_im
        diff_axes.append(ax_diff)
        ax_diff.set_yticks([0])
        if col == 0:
            ax_diff.set_yticklabels(["Δ"], fontsize=8)
        else:
            ax_diff.set_yticklabels([])
            ax_diff.tick_params(axis="y", labelleft=False)
        ax_diff.set_xticks([])
        ax_diff.axhline(0.5, color="#f5f5f5", linewidth=0.6)

    total_cells = rows * cols
    for idx in range(len(prepared), total_cells):
        row = idx // cols
        col = idx % cols
        sub = outer[row, col].subgridspec(2, 1, height_ratios=[1.0, 0.32])
        fig.add_subplot(sub[0]).axis("off")
        fig.add_subplot(sub[1]).axis("off")

    if first_heat_im is not None:
        cax_heat = fig.add_axes([0.09, 0.04, 0.35, 0.03])
        cbar_heat = fig.colorbar(first_heat_im, cax=cax_heat, orientation="horizontal")
        cbar_heat.set_label("Selection frequency")
        cbar_heat.formatter = FormatStrFormatter("%.2f")
        cbar_heat.locator = MaxNLocator(4)
        cbar_heat.update_ticks()
    if first_diff_im is not None:
        cax_diff = fig.add_axes([0.56, 0.04, 0.35, 0.03])
        cbar_diff = fig.colorbar(first_diff_im, cax=cax_diff, orientation="horizontal")
        cbar_diff.set_label("Δ (cluster - non)")
        cbar_diff.formatter = FormatStrFormatter("%.3f")
        diff_span = abs(first_diff_im.get_clim()[1])
        ticks = np.linspace(-diff_span, diff_span, 5)
        cbar_diff.set_ticks(ticks)
        cbar_diff.update_ticks()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fig.tight_layout(rect=(0.02, 0.08, 0.98, 0.98))
    fig.supxlabel("Feature rank", fontsize=11)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _discover_phenotypes(non_base: Path, cluster_base: Path) -> List[str]:
    phenotypes: List[str] = []
    if not non_base.exists():
        return phenotypes
    for child in sorted(non_base.iterdir()):
        if not child.is_dir():
            continue
        cache_dir = child / ".cache"
        cluster_cache = cluster_base / child.name / ".cache"
        if (
            cache_dir.exists()
            and cluster_cache.exists()
            and _phenotype_has_caches(child.name, non_base, cluster_base)
        ):
            phenotypes.append(child.name)
    return phenotypes


def _resolve_phenotype_dir(name: str, non_base: Path, cluster_base: Path) -> str:
    candidates = [name, name.replace(" ", "_"), name.replace("_", " ")]
    for cand in candidates:
        safe = cand.strip()
        if not safe:
            continue
        non_cache = non_base / safe / ".cache"
        cluster_cache = cluster_base / safe / ".cache"
        if (
            non_cache.exists()
            and cluster_cache.exists()
            and _phenotype_has_caches(safe, non_base, cluster_base)
        ):
            return safe
    raise FileNotFoundError(
        f"Could not locate phenotype '{name}' under {non_base} (and matching clustering directory)."
    )


def _phenotype_has_caches(phen_safe: str, non_base: Path, cluster_base: Path) -> bool:
    for base in (non_base, cluster_base):
        cache_dir = base / phen_safe / ".cache"
        if not cache_dir.exists():
            return False
        for fname in REQUIRED_CACHE_FILES:
            if not (cache_dir / fname).exists():
                return False
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Panel B/C between clustered and non-clustered sporulation runs")
    parser.add_argument("--phenotype", type=str, default="Spore formation")
    parser.add_argument(
        "--phenotypes",
        type=str,
        nargs="*",
        default=None,
        help="Optional list of phenotype directory names to facet; use 'all' to include every available phenotype",
    )
    parser.add_argument("--min_prev", type=float, default=0.02)
    parser.add_argument("--cluster_threshold", type=float, default=0.7)
    parser.add_argument("--cluster_metric", type=str, default="ochiai")
    parser.add_argument("--bootstrap_seed", type=int, default=17)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--grid_top_k", type=int, default=20)
    parser.add_argument("--grid_cols", type=int, default=5)
    parser.add_argument(
        "--skip_individual",
        action="store_true",
        help="When faceting multiple phenotypes, skip per-phenotype figures.",
    )
    parser.add_argument(
        "--noncluster_base",
        type=Path,
        default=Path("sporulation/results/h1_multiplicity"),
    )
    parser.add_argument(
        "--cluster_base",
        type=Path,
        default=Path("sporulation/results/clustering"),
    )
    parser.add_argument(
        "--cluster_root",
        type=Path,
        default=Path("/vol/projects/BIFO/genomenet/yichen"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("sporulation/results"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.phenotypes:
        if len(args.phenotypes) == 1 and args.phenotypes[0].lower() == "all":
            phen_list = _discover_phenotypes(args.noncluster_base, args.cluster_base)
        else:
            phen_list = []
            for name in args.phenotypes:
                phen_list.append(
                    _resolve_phenotype_dir(name, args.noncluster_base, args.cluster_base)
                )
    else:
        phen_list = [
            _resolve_phenotype_dir(
                args.phenotype,
                args.noncluster_base,
                args.cluster_base,
            )
        ]

    if not phen_list:
        raise RuntimeError("No phenotypes selected for comparison.")

    mode_pairs: Dict[str, Tuple[ModeData, ModeData]] = {}
    stats_pairs: Dict[str, Tuple[PanelBStats, PanelBStats]] = {}

    for phen_safe in phen_list:
        non_mode = _prepare_mode(
            base_dir=args.noncluster_base,
            phenotype_safe=phen_safe,
            clustered=False,
            min_prev=args.min_prev,
            cluster_threshold=args.cluster_threshold,
            cluster_metric=args.cluster_metric,
            cluster_root=args.cluster_root,
        )
        cluster_mode = _prepare_mode(
            base_dir=args.cluster_base,
            phenotype_safe=phen_safe,
            clustered=True,
            min_prev=args.min_prev,
            cluster_threshold=args.cluster_threshold,
            cluster_metric=args.cluster_metric,
            cluster_root=args.cluster_root,
        )
        non_stats = _compute_panel_b_stats(non_mode)
        cluster_stats = _compute_panel_b_stats(cluster_mode)
        mode_pairs[phen_safe] = (non_mode, cluster_mode)
        stats_pairs[phen_safe] = (non_stats, cluster_stats)

    multiple = len(phen_list) > 1
    if not multiple or (multiple and not args.skip_individual):
        for phen_safe in phen_list:
            non_stats, cluster_stats = stats_pairs[phen_safe]
            non_mode, cluster_mode = mode_pairs[phen_safe]
            panel_b_path = output_dir / f"{phen_safe}_panelB_compare.png"
            _plot_panel_b(non_stats, cluster_stats, panel_b_path, seed=args.bootstrap_seed)
            panel_c_path = output_dir / f"{phen_safe}_panelC_compare.png"
            _plot_panel_c(
                non_mode,
                cluster_mode,
                panel_c_path,
                top_k=int(args.top_k),
            )
            print(f"Saved Panel B comparison to {panel_b_path}")
            print(f"Saved Panel C comparison to {panel_c_path}")

    if multiple:
        panel_b_grid_path = output_dir / "panelB_compare_grid.png"
        _plot_panel_b_grid(
            stats_pairs,
            panel_b_grid_path,
            seed=args.bootstrap_seed,
            cols=int(args.grid_cols),
        )
        panel_c_grid_path = output_dir / "panelC_compare_grid.png"
        _plot_panel_c_grid(
            mode_pairs,
            panel_c_grid_path,
            top_k=int(args.grid_top_k),
            cols=int(args.grid_cols),
        )
        print(f"Saved Panel B grid comparison to {panel_b_grid_path}")
        print(f"Saved Panel C grid comparison to {panel_c_grid_path}")


if __name__ == "__main__":
    main()
