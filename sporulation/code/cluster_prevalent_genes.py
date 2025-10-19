#!/usr/bin/env python3
"""Cluster ≥2% prevalence genes by co-occurrence/non-coexistence and save reusable mapping.

Outputs:
- Disk-efficient NPZ saved to an external absolute path (default:
  /vol/projects/BIFO/genomenet/yichen) with arrays:
  - genes: array[str] of kept gene names (train-prevalence ≥ min_prev)
  - cluster_ids: array[int] same length as genes (1..K)
  - prevalence_train: array[float] per-gene train prevalence
  - leaf_order: array[int] dendrogram leaf order for consistent plotting
  - phenotype, mode, linkage, min_prev, distance_threshold, n_clusters: scalars/strings

- A clustered correlation heatmap (no gene name ticks) written to results dir.

Notes:
- Feature set is selected by training prevalence (same policy as training scripts).
- Clustering similarity can be based on |correlation| (default, captures co-occurrence
  and non-coexistence), positive-only correlation, or negative-only correlation.
  Distance used for linkage is 1 - similarity.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, fcluster, leaves_list
from scipy.spatial.distance import squareform

# Reuse dataset builders for consistency
from train_lasso import _read_table

def _maybe_import_cupy():
    try:
        import cupy as cp  # type: ignore
        try:
            ndev = int(cp.cuda.runtime.getDeviceCount())
            if ndev <= 0:
                return None
        except Exception:
            return None
        return cp
    except Exception:
        return None


def compute_training_prevalence(X_train: np.ndarray) -> np.ndarray:
    if X_train.size == 0:
        return np.array([], dtype=float)
    return np.asarray(X_train.mean(axis=0), dtype=float)


def build_splits_from_pa(pa_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Phenotype-agnostic split builder using only presence/absence and 'split'.

    Returns X_train, X_val, X_test as float32 arrays and the feature_names list.
    """
    if "split" not in pa_df.columns:
        raise ValueError("Presence/absence matrix must include a 'split' column")
    if pa_df.index.name != "sample_id":
        if "sample_id" in pa_df.columns:
            pa_df = pa_df.set_index("sample_id")
        else:
            pa_df.index.name = "sample_id"
    splits_col = pa_df["split"].astype("string").fillna("unspecified")
    X_df = pa_df.drop(columns=["split"])  # all genes/features
    feature_names = list(X_df.columns)

    train_mask = splits_col == "train"
    val_mask = splits_col == "val"
    test_mask = splits_col == "test"

    X_train = X_df.loc[train_mask].to_numpy(dtype=np.float32)
    X_val = X_df.loc[val_mask].to_numpy(dtype=np.float32)
    X_test = X_df.loc[test_mask].to_numpy(dtype=np.float32)
    return X_train, X_val, X_test, feature_names


def compute_corr_matrix(
    X_bin: np.ndarray,
    use_gpu: bool,
    corr_dtype: str,
) -> np.ndarray:
    """Compute feature-by-feature Pearson correlation matrix in a RAM-conscious way.

    - X_bin: binary/fractional matrix [n_samples, n_features] (float32/float16)
    - use_gpu: if True and CuPy available, compute on GPU then downcast
    - corr_dtype: 'float16' or 'float32' for final stored matrix
    Returns corr in the requested dtype, with diag set to 1 and NaNs replaced by 0.
    """
    n, p = X_bin.shape
    if p == 0:
        return np.zeros((0, 0), dtype=np.float16 if corr_dtype == "float16" else np.float32)

    target_dtype = np.float16 if corr_dtype == "float16" else np.float32
    eps = np.float32(1e-6)

    if use_gpu:
        cp = _maybe_import_cupy()
        if cp is not None:
            Xg = cp.asarray(X_bin, dtype=cp.float16)
            mu = cp.mean(Xg, axis=0, dtype=cp.float32)
            var = cp.mean((Xg - mu) ** 2, axis=0, dtype=cp.float32)
            std = cp.sqrt(var + eps)
            Z = (Xg - mu) / std
            # Accumulate in float32, downcast after
            G = (Z.T @ Z) / cp.float32(n)
            corr_g = cp.clip(G, -1.0, 1.0)
            corr = cp.asnumpy(corr_g).astype(target_dtype, copy=False)
            del Xg, mu, var, std, Z, G, corr_g
        else:
            use_gpu = False
    if not use_gpu:
        Xf = np.asarray(X_bin, dtype=np.float16)
        mu = Xf.mean(axis=0, dtype=np.float32)
        # variance via E[(X-mu)^2]
        var = ((Xf - mu) ** 2).mean(axis=0, dtype=np.float32)
        std = np.sqrt(var + eps, dtype=np.float32)
        Z = (Xf - mu) / std
        # accumulate in float32 for stability
        G = (np.matmul(Z.T.astype(np.float32), Z.astype(np.float32)) / np.float32(n)).astype(np.float32)
        corr = np.clip(G, -1.0, 1.0).astype(target_dtype, copy=False)
        del Xf, mu, var, std, Z, G

    # Clean
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    # Ensure symmetry and diag=1
    try:
        d = min(corr.shape)
        corr.flat[::d + 1] = 1.0  # fast diagonal set
    except Exception:
        np.fill_diagonal(corr, 1.0)
    return corr


def compute_ochiai_similarity(X_binary: np.ndarray) -> np.ndarray:
    """Compute Ochiai (binary cosine) similarity matrix in [0,1] for presence data.

    S[i,j] = cooccur(i,j) / sqrt(count(i) * count(j))
    """
    n, p = X_binary.shape
    if p == 0:
        return np.zeros((0, 0), dtype=np.float16)
    if n <= 1:
        return np.eye(p, dtype=np.float16)
    Xf = X_binary.astype(np.float16, copy=False)
    # counts and co-occurrence in float32 for stability
    counts = Xf.sum(axis=0, dtype=np.float32)
    # Avoid zero divisions
    counts = np.maximum(counts, np.float32(1e-6))
    co = (np.matmul(Xf.T.astype(np.float32), Xf.astype(np.float32))).astype(np.float32)
    denom = np.sqrt(np.outer(counts, counts)).astype(np.float32)
    with np.errstate(divide='ignore', invalid='ignore'):
        S = np.clip(co / denom, 0.0, 1.0)
    np.fill_diagonal(S, 1.0)
    return S.astype(np.float16, copy=False)


def build_similarity_matrix(
    X_binary: np.ndarray,
    metric: str,
    mode: str,
    use_gpu: bool,
    corr_dtype: str,
) -> np.ndarray:
    """Return similarity matrix in [0,1].

    metric:
      - 'ochiai': presence-only Ochiai similarity
      - 'phi': Pearson correlation on binary data → similarity via mode
    mode (for 'phi'):
      - 'abs': similarity = |corr|
      - 'pos': similarity = max(corr, 0)
      - 'neg': similarity = max(-corr, 0)
    For 'ochiai', mode 'abs' and 'pos' both use Ochiai; 'neg' uses symmetric exclusion Ochiai.
    """
    n_samples, n_features = X_binary.shape
    if n_features == 0:
        return np.zeros((0, 0), dtype=np.float16)
    if n_samples <= 1:
        return np.eye(n_features, dtype=np.float16)

    if metric == "ochiai":
        Xb = (X_binary > 0).astype(np.float32, copy=False)
        S_pos = compute_ochiai_similarity(Xb)
        if mode in ("abs", "pos"):
            return S_pos
        # 'neg': symmetric exclusion Ochiai based on presence vs absence
        n = Xb.shape[0]
        A = Xb.astype(np.float32, copy=False)
        B = (1.0 - A).astype(np.float32, copy=False)
        cntA = A.sum(axis=0, dtype=np.float32)
        cntB = B.sum(axis=0, dtype=np.float32)
        cntA = np.maximum(cntA, np.float32(1e-6))
        cntB = np.maximum(cntB, np.float32(1e-6))
        M1 = (np.matmul(A.T, B)).astype(np.float32)  # i present, j absent
        M2 = M1.T  # j present, i absent
        denom1 = np.sqrt(np.outer(cntA, cntB))
        denom2 = denom1.T
        with np.errstate(divide='ignore', invalid='ignore'):
            S1 = np.clip(M1 / denom1, 0.0, 1.0)
            S2 = np.clip(M2 / denom2, 0.0, 1.0)
            S_neg = np.sqrt(S1 * S2)
        np.fill_diagonal(S_neg, 1.0)
        return S_neg.astype(np.float16, copy=False)
    elif metric == "phi":
        corr = compute_corr_matrix(X_binary, use_gpu=bool(use_gpu), corr_dtype=corr_dtype).astype(np.float16, copy=False)
        if mode == "abs":
            S = np.abs(corr).astype(np.float16, copy=False)
        elif mode == "pos":
            S = np.clip(corr, 0.0, 1.0).astype(np.float16, copy=False)
        elif mode == "neg":
            S = np.clip(-corr, 0.0, 1.0).astype(np.float16, copy=False)
        else:
            raise ValueError(f"Invalid mode: {mode}")
        np.fill_diagonal(S, 1.0)
        return S
    else:
        raise ValueError(f"Unknown metric: {metric}")


def _row_start_in_condensed(n: int, i: int) -> int:
    # Start index in condensed vector for row i (i<j) when flattened row-wise
    return i * (n - 1) - (i * (i - 1)) // 2


def condensed_from_similarity_stream(
    S: np.ndarray,
    work_dir: Path,
    dtype: str = "float16",
    block_rows: int = 1024,
) -> np.memmap:
    """Stream-build condensed distance vector (1 - similarity) from S without full NxN distance.

    Returns a memmap of length n*(n-1)//2 with requested dtype (float16 or float32).
    Caller is responsible for unlinking the file when done if desired.
    """
    n = int(S.shape[0])
    m = (n * (n - 1)) // 2
    work_dir.mkdir(parents=True, exist_ok=True)
    path = work_dir / "condensed_dvec.dat"
    dt = np.float16 if dtype == "float16" else np.float32
    dvec = np.memmap(path, dtype=dt, mode="w+", shape=(m,))

    for i0 in range(0, n, block_rows):
        i1 = min(n, i0 + block_rows)
        for i in range(i0, i1):
            if i + 1 >= n:
                continue
            row = S[i, i + 1 :].astype(np.float32, copy=False)
            dist = (1.0 - row).astype(dt, copy=False)
            start = _row_start_in_condensed(n, i)
            end = start + (n - i - 1)
            dvec[start:end] = dist
    dvec.flush()
    return dvec


def cluster_from_similarity_stream(
    S: np.ndarray,
    linkage_method: str,
    distance_threshold: float,
    n_clusters: int,
    work_dir: Path,
) -> Tuple[np.ndarray, np.ndarray]:
    n = int(S.shape[0])
    if n == 0:
        return np.zeros((0,), dtype=int), np.zeros((0,), dtype=int)
    dvec = condensed_from_similarity_stream(S, work_dir=work_dir, dtype="float16")
    # SciPy may upcast internally; memmap keeps RAM usage lower for construction
    Z = linkage(dvec, method=linkage_method)
    if int(n_clusters) > 0:
        labels = fcluster(Z, t=int(n_clusters), criterion="maxclust").astype(int)
    else:
        labels = fcluster(Z, t=float(distance_threshold), criterion="distance").astype(int)
    order = leaves_list(Z).astype(int)
    try:
        # Best-effort cleanup of the temporary memmap file
        path = Path(dvec.filename) if hasattr(dvec, "filename") else None
        del dvec
        if path is not None:
            try:
                os.remove(path)
            except Exception:
                pass
    except Exception:
        pass
    return labels, order


def _downsample_matrix_average(A: np.ndarray, max_dim: int) -> np.ndarray:
    """Downsample a square matrix by block-averaging to at most max_dim x max_dim.

    If max_dim <= 0, returns the input matrix unchanged.
    """
    n = A.shape[0]
    if max_dim is None or max_dim <= 0 or n <= max_dim:
        return A
    # Compute block size; ensure integer blocks cover the matrix
    b = int(np.ceil(n / max_dim))
    m = int(np.ceil(n / b))
    # Pad to multiple of b
    pad = m * b - n
    if pad > 0:
        A = np.pad(A, ((0, pad), (0, pad)), mode="edge")
    # Reshape and average
    A = A.reshape(m, b, m, b).mean(axis=(1, 3))
    return A


def plot_clustered_heatmap(
    M: np.ndarray,
    leaf_order: np.ndarray,
    out_path: Path,
    title: str,
    metric: str,
    heatmap_max_dim: int = 2500,
) -> None:
    if M.size == 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No genes after prevalence filter", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return
    M_ord = M[np.ix_(leaf_order, leaf_order)]
    M_ord = _downsample_matrix_average(M_ord, max_dim=int(heatmap_max_dim)).astype(np.float16, copy=False)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    if metric == "phi":
        im = ax.imshow(M_ord, cmap="coolwarm", vmin=-1.0, vmax=1.0, interpolation="nearest", aspect="auto")
        cbar_label = "Pearson r"
    else:
        im = ax.imshow(M_ord, cmap="viridis", vmin=0.0, vmax=1.0, interpolation="nearest", aspect="auto")
        cbar_label = "Ochiai"
    # Remove ticks and labels to keep file compact and readable
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel(cbar_label, rotation=90)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _cluster_console_summary(
    M: np.ndarray,
    labels: np.ndarray,
    kept_genes: List[str],
    mode: str,
    metric: str,
    distance_threshold: float,
) -> None:
    n = int(len(kept_genes))
    if n == 0 or labels.size == 0:
        print("[Summary] No genes kept after prevalence filter.", flush=True)
        return
    # Normalize labels to 0..K-1
    labs, inv = np.unique(labels.astype(int), return_inverse=True)
    idx_by_cluster: List[np.ndarray] = [np.where(inv == k)[0] for k in range(labs.size)]
    sizes = np.array([len(ix) for ix in idx_by_cluster], dtype=int)
    K = int(labs.size)
    singletons = int(np.sum(sizes == 1))
    non_singleton = K - int(np.sum(sizes == 1))
    # Size stats
    size_min = int(np.min(sizes)) if sizes.size > 0 else 0
    size_med = float(np.median(sizes)) if sizes.size > 0 else 0.0
    size_mean = float(np.mean(sizes)) if sizes.size > 0 else 0.0
    size_max = int(np.max(sizes)) if sizes.size > 0 else 0

    # Within-cluster similarity quality (histogram-based to save RAM)
    n_bins = 100
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1, dtype=np.float32)
    hist = np.zeros(n_bins, dtype=np.int64)
    sum_sim = 0.0
    cnt_sim = 0
    cnt_sim_ge = 0
    sim_thresh = 1.0 - float(distance_threshold) if distance_threshold > 0 else None

    for ix in idx_by_cluster:
        s = int(ix.size)
        if s <= 1:
            continue
        block = M[np.ix_(ix, ix)].astype(np.float32, copy=False)
        if metric == "phi":
            if mode == "abs":
                S = np.abs(block, dtype=np.float32)
            elif mode == "pos":
                S = np.clip(block, 0.0, 1.0)
            else:
                S = np.clip(-block, 0.0, 1.0)
        else:
            S = np.clip(block, 0.0, 1.0)
        iu = np.triu_indices(s, k=1)
        vals = S[iu]
        # Update stats
        sum_sim += float(vals.sum(dtype=np.float64))
        cnt_sim += int(vals.size)
        h, _ = np.histogram(vals, bins=bin_edges)
        hist += h.astype(np.int64)
        if sim_thresh is not None:
            cnt_sim_ge += int(np.sum(vals >= float(sim_thresh)))

    def _percentile_from_hist(h: np.ndarray, edges: np.ndarray, q: float) -> float:
        if h.sum() == 0:
            return float("nan")
        c = np.cumsum(h)
        target = q * c[-1]
        idx = int(np.searchsorted(c, target, side="left"))
        idx = min(max(idx, 0), len(edges) - 2)
        return float(0.5 * (edges[idx] + edges[idx + 1]))

    mean_sim = float(sum_sim / cnt_sim) if cnt_sim > 0 else float("nan")
    p50 = _percentile_from_hist(hist, bin_edges, 0.5)
    p75 = _percentile_from_hist(hist, bin_edges, 0.75)
    p90 = _percentile_from_hist(hist, bin_edges, 0.90)
    frac_ge = (float(cnt_sim_ge) / float(cnt_sim)) if (cnt_sim > 0 and sim_thresh is not None) else float("nan")

    # Print summary
    print(
        "[Summary] Clusters:",
        f"genes={n}",
        f"K={K}",
        f"singletons={singletons} ({(singletons/max(1,n))*100:.1f}%)",
        f"size[min/med/mean/max]={size_min}/{size_med:.1f}/{size_mean:.2f}/{size_max}",
        flush=True,
    )
    print(
        "[Summary] Within-cluster similarity (", metric, ", mode=", mode, "):",
        f"mean={mean_sim:.3f}",
        f"p50≈{p50:.3f}",
        f"p75≈{p75:.3f}",
        f"p90≈{p90:.3f}",
        (f"frac≥{sim_thresh:.2f}≈{frac_ge*100:.1f}%" if sim_thresh is not None and np.isfinite(frac_ge) else ""),
        sep="",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Cluster ≥2% prevalence genes (phenotype-agnostic) and save cluster mapping for downstream multiplicity")
    parser.add_argument("--input_dir", type=Path, default=Path("/vol/projects/BIFO/genomenet/yichen/phenotype/data/rfdata"))
    parser.add_argument("--external_out_dir", type=Path, default=Path("/vol/projects/BIFO/genomenet/yichen"))
    parser.add_argument("--output_dir", type=Path, default=Path("sporulation/results/clustering"))
    parser.add_argument("--min_prev", type=float, default=0.02)
    parser.add_argument("--metric", type=str, choices=["ochiai", "phi"], default="ochiai", help="Similarity metric: ochiai (default) or phi (Pearson)")
    parser.add_argument("--mode", type=str, choices=["abs", "pos", "neg"], default="abs", help="Similarity mode: abs/pos/neg (phi only uses sign; ochiai uses abs/pos as co-occur, neg as exclusion)")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU (CuPy) if available for correlation computation")
    parser.add_argument("--corr_dtype", type=str, choices=["float16", "float32"], default="float16", help="dtype for stored correlation matrix and distance stream")
    parser.add_argument("--linkage", type=str, choices=["average", "complete", "single"], default="average")
    parser.add_argument("--distance_threshold", type=float, default=0.30, help="Threshold on distance (1-similarity) if n_clusters==0")
    parser.add_argument("--n_clusters", type=int, default=0, help="If >0, use maxclust to cut the tree")
    parser.add_argument("--heatmap_max_dim", type=int, default=2500, help="Downsample heatmap to at most this dimension for RAM/PNG size")
    args = parser.parse_args()

    input_dir: Path = args.input_dir
    external_out_dir: Path = args.external_out_dir
    out_dir: Path = args.output_dir
    phen_safe = "all_samples"

    # Load presence/absence and build phenotype-agnostic splits
    pa_path = input_dir / "rf_presence_absence.parquet"
    pa_df = _read_table(pa_path)
    X_train, X_val, X_test, feature_names = build_splits_from_pa(pa_df)

    # Filter by train prevalence (policy used elsewhere)
    prev_train = compute_training_prevalence(X_train)
    keep_mask = prev_train >= float(args.min_prev)
    kept_indices = np.where(keep_mask)[0]
    genes_all: List[str] = list(feature_names)
    kept_genes = [genes_all[i] for i in kept_indices]
    prev_kept = prev_train[keep_mask].astype(np.float32)

    # Build matrix for correlation on train+val to stabilize estimates
    X_trval = np.concatenate([X_train, X_val], axis=0) if X_val.size > 0 else X_train
    X_bin = (X_trval[:, keep_mask] > 0).astype(np.float32)

    # Similarity matrix (RAM-conscious), stored in float16
    S = build_similarity_matrix(
        X_bin,
        metric=str(args.metric),
        mode=str(args.mode),
        use_gpu=bool(args.use_gpu),
        corr_dtype=str(args.corr_dtype),
    )
    # Similarity streaming and clustering (build distance condensed vector without full NxN distance)
    work_dir = out_dir / phen_safe / ".cache"
    labels, order = cluster_from_similarity_stream(
        S,
        linkage_method=str(args.linkage),
        distance_threshold=float(args.distance_threshold),
        n_clusters=int(args.n_clusters),
        work_dir=work_dir,
    )

    # Save disk-efficient mapping to external absolute path
    external_out_dir.mkdir(parents=True, exist_ok=True)
    # Build informative filename
    mode_tag = str(args.mode)
    if int(args.n_clusters) > 0:
        tag = f"{phen_safe}_minprev{args.min_prev:.3f}_metric-{args.metric}_mode-{mode_tag}_link-{args.linkage}_k-{int(args.n_clusters)}"
    else:
        tag = f"{phen_safe}_minprev{args.min_prev:.3f}_metric-{args.metric}_mode-{mode_tag}_link-{args.linkage}_thr-{args.distance_threshold:.2f}"
    npz_path = external_out_dir / f"gene_clusters_{tag}.npz"

    try:
        np.savez_compressed(
            npz_path,
            genes=np.array(kept_genes, dtype=str),
            cluster_ids=np.asarray(labels, dtype=np.int32),
            prevalence_train=np.asarray(prev_kept, dtype=np.float32),
            leaf_order=np.asarray(order, dtype=np.int32),
            phenotype=np.array("all_samples"),
            metric=np.array(str(args.metric)),
            mode=np.array(str(args.mode)),
            linkage=np.array(str(args.linkage)),
            min_prev=np.array(float(args.min_prev)),
            distance_threshold=np.array(float(args.distance_threshold)),
            n_clusters=np.array(int(args.n_clusters)),
            n_genes=np.array(int(len(kept_genes))),
            n_train_samples=np.array(int(X_train.shape[0])),
            n_val_samples=np.array(int(X_val.shape[0])),
        )
        print("Saved cluster mapping:", str(npz_path), flush=True)
    except Exception as e:
        print(f"[Error] Failed to save NPZ mapping to {npz_path}: {e}", flush=True)

    # Plot clustered correlation heatmap (no axis ticks)
    heat_dir = out_dir / phen_safe
    heat_path = heat_dir / "clustered_corr_heatmap.png"
    title = f"Clustered similarity — all samples (n={len(kept_genes)})"
    plot_clustered_heatmap(S, order, heat_path, title, metric=str(args.metric), heatmap_max_dim=int(args.heatmap_max_dim))
    print("Saved heatmap:", str(heat_path), flush=True)

    # Console summaries of clusters and quality
    try:
        _cluster_console_summary(
            M=S.astype(np.float32, copy=False),
            labels=labels,
            kept_genes=kept_genes,
            mode=str(args.mode),
            metric=str(args.metric),
            distance_threshold=float(args.distance_threshold),
        )
    except Exception as e:
        print(f"[Summary] Skipped detailed cluster summary: {e}", flush=True)


if __name__ == "__main__":
    main()

