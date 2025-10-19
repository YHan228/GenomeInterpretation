#!/usr/bin/env python3
"""Simulate IG-like attribution curves and visualize SaAUC and SaSNR vs Expected.

This script does NOT compute Integrated Gradients. It generates synthetic
attribution arrays with different shapes relative to a fixed ground-truth mask
that covers exactly 20% of positions. For each scenario, it computes:

- SaAUC: rank-based AUROC of contributions inside vs outside the mask
- SaSNR: fraction of squared attribution mass inside the mask
- Expected SaSNR: mask fraction (fixed at 0.2)

Outputs two figures by default:
- curves.png: example attribution curves (smoothed) with the ground-truth mask shaded
- metrics.png: SaAUC per scenario, and SaSNR vs Expected (0.2)

Usage:
  python simulate_sauc_sasnr.py --length 10000 --out_dir ./sim_out --seed 0

"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


# --------------------------- Metrics (SaAUC, SaSNR) ---------------------------


def _rankdata_average(x: np.ndarray) -> np.ndarray:
    """Average ranks for ties, 1-based ranks (like scipy.stats.rankdata, method='average')."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1, dtype=np.float64)
    # handle ties
    vals = x[order]
    i = 0
    n = len(x)
    while i < n:
        j = i + 1
        while j < n and vals[j] == vals[i]:
            j += 1
        if j - i > 1:
            avg = (i + 1 + j) / 2.0
            ranks[order[i:j]] = avg
        i = j
    return ranks


def _interval_mask(intervals: List[Tuple[int, int]], length: int) -> np.ndarray:
    """Return boolean mask for 0-based inclusive intervals within a window.

    intervals: list of (start, end) inclusive, 0-based
    length: total length L
    """
    mask = np.zeros(length, dtype=bool)
    for s, e in intervals:
        s0 = max(0, int(s))
        e0 = min(length - 1, int(e))
        if s0 <= e0:
            mask[s0 : e0 + 1] = True
    return mask


def saliency_auc_from_mask(attr: np.ndarray, intervals: List[Tuple[int, int]]) -> float:
    """Compute SaAUC via rank-based AUROC (Mann-Whitney) using the evaluation convention.

    attr: 1D array of non-negative attribution magnitudes for L positions
    intervals: list of (start,end) 0-based inclusive GT intervals
    """
    L = int(attr.shape[0])
    mask = _interval_mask(intervals, L)
    inside = attr[mask]
    outside = attr[~mask]
    n_pos = inside.size
    n_neg = outside.size
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    all_vals = np.concatenate([inside, outside])
    ranks = _rankdata_average(all_vals)
    ranks_pos = ranks[:n_pos].sum()
    auc = (ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def compute_sasnr(attr: np.ndarray, intervals: List[Tuple[int, int]]) -> float:
    """Compute SaSNR = sum(attr^2 inside) / sum(attr^2 everywhere)."""
    L = int(attr.shape[0])
    mask = _interval_mask(intervals, L)
    attr_sq = np.square(attr.astype(np.float64, copy=False))
    sum_sq_total = float(attr_sq.sum())
    if sum_sq_total <= 0.0:
        return float("nan")
    if mask.any():
        sum_sq_inside = float(attr_sq[mask].sum())
    else:
        sum_sq_inside = 0.0
    return float(sum_sq_inside / (sum_sq_total + 1e-12))


# --------------------------- Mask construction (20%) ---------------------------


def build_mask_intervals(length: int, cover_fraction: float = 0.2, n_intervals: int = 5) -> List[Tuple[int, int]]:
    """Construct evenly spaced, equal-length intervals covering exactly cover_fraction of positions.

    Intervals are 0-based inclusive. If rounding leaves leftover positions, the last
    interval absorbs the remainder to ensure exact coverage when possible.
    """
    L = int(length)
    frac = float(cover_fraction)
    k = int(n_intervals)
    total = int(round(frac * L))
    if total <= 0 or k <= 0:
        return []
    base = total // k
    rem = total - base * k
    # place interval starts evenly across the genome
    starts = np.linspace(0, max(0, L - 1), num=k, endpoint=True, dtype=np.int64)
    spans = [base + (1 if i < rem else 0) for i in range(k)]
    intervals: List[Tuple[int, int]] = []
    for i in range(k):
        span = int(spans[i])
        if span <= 0:
            continue
        s = int(starts[i])
        e = int(min(L - 1, s + span - 1))
        intervals.append((s, e))
    return intervals


# --------------------------- Scenario generation ---------------------------


@dataclass
class Scenario:
    name: str
    generator: Callable[[int, np.random.Generator, np.ndarray], np.ndarray]


def _normalize_and_abs(arr: np.ndarray) -> np.ndarray:
    """Ensure non-negative contributions and stabilize scale for visualization."""
    # emulate absolute contribution as in evaluation code: |(corrected_grad * onehot).sum_channels|
    x = np.abs(arr.astype(np.float64))
    return x


def _intervals_from_mask(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Recover contiguous (start,end) inclusive intervals from a boolean mask."""
    L = int(mask.size)
    if L == 0:
        return []
    b = mask.astype(np.int8)
    # detect edges
    edges = np.diff(np.concatenate([[0], b, [0]])).astype(int)
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0] - 1
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


def _add_lump(x: np.ndarray, center: int, width: int, amp: float) -> None:
    """Add a flat-topped lump (boxcar) to x around center with given width and amplitude."""
    L = x.size
    half = max(1, int(width // 2))
    s = max(0, int(center) - half)
    e = min(L, int(center) + half)
    if e > s:
        x[s:e] += float(amp)


def gen_inside_peaky(L: int, rng: np.random.Generator, mask: np.ndarray) -> np.ndarray:
    base = rng.normal(loc=0.0, scale=0.25, size=L)
    x = base.copy()
    # create a pronounced peak within every GT interval
    for s, e in _intervals_from_mask(mask):
        c = rng.integers(s, e + 1)
        w = max(60, int(0.05 * (e - s + 1)))
        _add_lump(x, c, w, amp=2.5)
    return _normalize_and_abs(x)


def gen_outside_peaky(L: int, rng: np.random.Generator, mask: np.ndarray) -> np.ndarray:
    base = rng.normal(loc=0.0, scale=1.0, size=L)
    boost = rng.normal(loc=3.2, scale=1.0, size=L) * (~mask).astype(np.float64)
    return _normalize_and_abs(base + boost)


def gen_uniform_noise(L: int, rng: np.random.Generator, mask: np.ndarray) -> np.ndarray:
    return _normalize_and_abs(rng.normal(loc=0.0, scale=1.0, size=L))


def gen_sparse_inside_spikes(L: int, rng: np.random.Generator, mask: np.ndarray) -> np.ndarray:
    x = rng.normal(loc=0.0, scale=0.15, size=L)
    intervals = _intervals_from_mask(mask)
    if intervals:
        # only a minority of intervals get any spikes at all
        k = max(1, int(np.ceil(0.4 * len(intervals))))
        chosen = rng.choice(len(intervals), size=k, replace=False)
        for idx in chosen:
            s, e = intervals[int(idx)]
            # add 1-3 narrow lumps inside this interval
            m = rng.integers(1, 4)
            for _ in range(int(m)):
                c = rng.integers(s, e + 1)
                _add_lump(x, c, width=40, amp=1.8)
    return _normalize_and_abs(x)


def gen_slightly_outside(L: int, rng: np.random.Generator, mask: np.ndarray) -> np.ndarray:
    # both regions informative; outside only slightly stronger
    inside = rng.normal(loc=1.0, scale=0.25, size=L) * mask.astype(np.float64)
    outside = rng.normal(loc=1.1, scale=0.25, size=L) * (~mask).astype(np.float64)
    noise = rng.normal(loc=0.0, scale=0.1, size=L)
    return _normalize_and_abs(inside + outside + noise)


# ----- New partial-ground-truth variants -----


def gen_partial_inside(L: int, rng: np.random.Generator, mask: np.ndarray, keep_fraction: float = 0.5) -> np.ndarray:
    """Only a subset of GT intervals get strong inside peaks; others remain weak.

    keep_fraction: fraction of GT intervals to boost (0..1).
    """
    base = rng.normal(loc=0.0, scale=0.2, size=L)
    x = base.copy()
    intervals = _intervals_from_mask(mask)
    if not intervals:
        return _normalize_and_abs(x)
    k = max(1, int(np.floor(float(keep_fraction) * len(intervals))))
    chosen = set(rng.choice(len(intervals), size=k, replace=False).tolist())
    for i, (s, e) in enumerate(intervals):
        if i in chosen:
            # strong peak inside selected interval
            c = rng.integers(s, e + 1)
            _add_lump(x, c, width=max(60, int(0.05 * (e - s + 1))), amp=2.2)
        else:
            # weak background signal inside non-selected GT interval
            c = rng.integers(s, e + 1)
            _add_lump(x, c, width=40, amp=0.3)
    return _normalize_and_abs(x)


def gen_partial_inside_with_outside_peaks(L: int, rng: np.random.Generator, mask: np.ndarray) -> np.ndarray:
    """Only some GT regions are strong; plus additional peaks outside the mask."""
    x = gen_partial_inside(L, rng, mask, keep_fraction=0.5)
    intervals_out = _intervals_from_mask(~mask)
    if intervals_out:
        # add a few strong outside peaks in randomly chosen outside intervals
        m = max(1, int(np.ceil(0.3 * len(intervals_out))))
        chosen = rng.choice(len(intervals_out), size=m, replace=False)
        for idx in chosen:
            s, e = intervals_out[int(idx)]
            c = rng.integers(s, e + 1)
            _add_lump(x, c, width=70, amp=2.4)
    return _normalize_and_abs(x)


def gen_partial_inside_with_dense_outside(L: int, rng: np.random.Generator, mask: np.ndarray) -> np.ndarray:
    """Only some GT regions strong; outside has broad, non-sparse signal (not peaky)."""
    x = gen_partial_inside(L, rng, mask, keep_fraction=0.5)
    # global random background so unselected GT windows are not systematically lower
    bg = rng.normal(loc=0.25, scale=0.08, size=L)
    x = x + bg
    # add dense outside (blockwise positive bias applied only to outside)
    win = 200
    nblocks = int(np.ceil(L / win))
    dense_add = np.zeros(L, dtype=np.float64)
    means = rng.normal(loc=0.12, scale=0.04, size=nblocks)  # non-sparse broad outside uplift
    for i in range(nblocks):
        s = i * win
        e = min(L, (i + 1) * win)
        dense_add[s:e] = means[i]
    dense_add[mask] = 0.0
    x = x + dense_add
    return _normalize_and_abs(x)


def gen_inside_strictly_higher(L: int, rng: np.random.Generator, mask: np.ndarray) -> np.ndarray:
    """Construct attribution where every inside bp > every outside bp by a margin."""
    base = rng.uniform(0.0, 1.0, size=L)
    x = base.copy()
    if (~mask).any():
        mx = float(np.max(base[~mask]))
    else:
        mx = float(np.max(base))
    margin = 0.2
    inside_vals = mx + margin + rng.uniform(0.0, 0.1, size=int(mask.sum()))
    x[mask] = inside_vals
    return _normalize_and_abs(x)


def list_default_scenarios() -> List[Scenario]:
    return [
        Scenario("inside_peaky", gen_inside_peaky),
        Scenario("partial_inside", lambda L, rng, m: gen_partial_inside(L, rng, m, keep_fraction=0.5)),
        Scenario("partial_in+outside_peaks", gen_partial_inside_with_outside_peaks),
        Scenario("partial_in+dense_outside", gen_partial_inside_with_dense_outside),
        # Replace outside_peaky with a diagnostic: inside strictly higher than outside
        Scenario("inside_strictly_higher", gen_inside_strictly_higher),
        Scenario("sparse_inside", gen_sparse_inside_spikes),
        Scenario("slightly_outside", gen_slightly_outside),
        Scenario("uniform_noise", gen_uniform_noise),
    ]


# --------------------------- Visualization ---------------------------


def _plot_curves(ax_list: List[plt.Axes], attrs: Dict[str, np.ndarray], intervals: List[Tuple[int, int]], smooth_win: int) -> None:
    L = len(next(iter(attrs.values()))) if attrs else 0
    xs = np.arange(L)
    # Shade mask
    for (s, e) in intervals:
        for ax in ax_list:
            ax.axvspan(s, e, color="#fee08b", alpha=0.3, lw=0)
    # Plot curves
    for ax, (name, arr) in zip(ax_list, attrs.items()):
        # smooth by block-mean of size smooth_win
        if smooth_win > 1:
            w = int(max(1, smooth_win))
            nblocks = int(np.ceil(L / w))
            block_means = np.array([arr[i*w : min(L, (i+1)*w)].mean() for i in range(nblocks)])
            bx = np.arange(nblocks) * w + min(w // 2, w - 1)
            ax.plot(bx, block_means, color="#2c7fb8", lw=1.2, alpha=0.6)
        else:
            ax.plot(xs, arr, color="#2c7fb8", lw=0.9, alpha=0.6)
        ax.set_title(name)
        ax.set_xlim(0, L - 1)
        ax.set_ylabel("attribution")
        ax.grid(True, alpha=0.2)
    ax_list[-1].set_xlabel("position")


def _plot_metrics_bar(ax_auc: plt.Axes, ax_snr: plt.Axes, results: List[Tuple[str, float, float]], expected: float) -> None:
    names = [r[0] for r in results]
    aucs = [r[1] for r in results]
    snrs = [r[2] for r in results]

    # SaAUC bar
    ax_auc.bar(np.arange(len(names)), aucs, color="#4575b4")
    ax_auc.set_xticks(np.arange(len(names)))
    ax_auc.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax_auc.set_ylabel("SaAUC")
    ax_auc.set_ylim(0.0, 1.0)
    ax_auc.set_title("SaAUC (inside vs outside)")
    ax_auc.grid(True, axis="y", alpha=0.3)

    # SaSNR vs Expected
    ax_snr.bar(np.arange(len(names)), snrs, color="#d7301f", label="SaSNR")
    ax_snr.hlines(expected, -0.5, len(names) - 0.5, colors="#666666", linestyles="--", label=f"Expected={expected:.2f}")
    ax_snr.set_xticks(np.arange(len(names)))
    ax_snr.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax_snr.set_ylabel("SaSNR")
    ax_snr.set_ylim(0.0, 1.0)
    ax_snr.set_title("SaSNR vs Expected")
    ax_snr.legend()
    ax_snr.grid(True, axis="y", alpha=0.3)


# --------------------------- Main ---------------------------


def run(length: int, out_dir: str, seed: int, smooth_win: int, n_intervals: int) -> None:
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(int(seed))

    # Fixed ground-truth coverage at 20%
    mask_intervals = build_mask_intervals(length, cover_fraction=0.2, n_intervals=int(n_intervals))
    mask = _interval_mask(mask_intervals, length)
    expected = float(mask.mean())

    scenarios = list_default_scenarios()

    # Generate example curves per scenario
    attrs: Dict[str, np.ndarray] = {}
    for sc in scenarios:
        arr = sc.generator(length, rng, mask)
        # Rescale per-scenario for consistent plotting dynamic range
        if np.max(arr) > 0:
            arr = arr / float(np.percentile(arr, 99.5))
        attrs[sc.name] = arr

    # Compute metrics per scenario
    results: List[Tuple[str, float, float]] = []
    for name, arr in attrs.items():
        sauc = saliency_auc_from_mask(arr, mask_intervals)
        sasnr = compute_sasnr(arr, mask_intervals)
        results.append((name, sauc, sasnr))

    # Figure 1: curves
    n = len(attrs)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig1, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 2.6 * nrows), sharex=True)
    axes_list = list(axes.flat) if hasattr(axes, "flat") else [axes]
    # pad with empty axes if needed
    for i in range(len(axes_list) - n):
        axes_list[-(i + 1)].axis("off")
    _plot_curves(axes_list[:n], attrs, mask_intervals, smooth_win=int(smooth_win))
    fig1.tight_layout()
    fig1.savefig(os.path.join(out_dir, "curves.png"), dpi=130, bbox_inches="tight")
    plt.close(fig1)

    # Figure 2: metrics
    fig2, (ax_auc, ax_snr) = plt.subplots(nrows=2, ncols=1, figsize=(10.5, 6.5))
    _plot_metrics_bar(ax_auc, ax_snr, results, expected)
    # annotate numeric values
    for i, (_, auc, snr) in enumerate(results):
        ax_auc.text(i, auc + 0.02, f"{auc:.2f}", ha="center", va="bottom", fontsize=9)
        ax_snr.text(i, snr + 0.02, f"{snr:.2f}", ha="center", va="bottom", fontsize=9)
    fig2.tight_layout()
    fig2.savefig(os.path.join(out_dir, "metrics.png"), dpi=130, bbox_inches="tight")
    plt.close(fig2)

    # Console summary
    print("Mask coverage (expected SaSNR):", f"{expected:.3f}")
    for name, auc, snr in results:
        print(f"{name:>16s} | SaAUC={auc:.3f} | SaSNR={snr:.3f} | Δ={snr-expected:+.3f}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Simulate SaAUC and SaSNR under different IG-like curves (mask fraction fixed at 0.2)")
    ap.add_argument("--length", type=int, default=10000, help="Sequence length (number of positions)")
    ap.add_argument("--out_dir", type=str, default="sim_sauc_sasnr_out", help="Output directory for figures")
    ap.add_argument("--seed", type=int, default=0, help="Random seed")
    ap.add_argument("--smooth_win", type=int, default=100, help="Block size for mean smoothing in bp (>=1)")
    ap.add_argument("--gt_intervals", type=int, default=5, help="Number of GT intervals composing the 20% mask")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(length=int(args.length), out_dir=str(args.out_dir), seed=int(args.seed), smooth_win=int(args.smooth_win), n_intervals=int(args.gt_intervals))


