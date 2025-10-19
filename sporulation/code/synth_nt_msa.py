"""
NT-only MSA and conservation metrics on synthetic datasets cached as .npz.

Each dataset_cache .npz has arrays:
  - X: shape (N, 4, L), one-hot NT sequences (A,C,G,T)
  - y: shape (N,), labels (float/0-1), positives contain an embedded motif
  - masks: shape (N, L), boolean mask for motif positions (True length ≈ motif length)

We align only the motif regions across positive samples by directly extracting the
masked positions (same length by construction), so no external aligner is needed.

Outputs per dataset (.npz file):
  - n_pos_used, motif_len, nt_mean_col_conservation, nt_mean_pairwise_identity
  - parsed target_conservation from filename if available (e.g., cons_0.700)

Aggregated outputs:
  - CSV summary over all matched datasets
  - Plots: measured_vs_target_conservation and distribution histograms

Parallelization: per-dataset in a thread pool with progress logs.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    try:
        import seaborn as sns  # type: ignore
    except Exception:
        sns = None
except Exception as e:
    raise RuntimeError("matplotlib is required for plotting") from e


# --------------------------- Plotting ---------------------------

def _init_plotting() -> None:
    if 'MPLBACKEND' not in os.environ:
        os.environ['MPLBACKEND'] = 'Agg'
    if sns is not None:
        sns.set_theme(context="paper", style="whitegrid", font_scale=1.2)
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def _savefig(fig: "plt.Figure", out_base: str) -> None:
    fig.tight_layout()
    fig.savefig(out_base + ".png", bbox_inches="tight")
    try:
        fig.savefig(out_base + ".pdf", bbox_inches="tight")
    except Exception:
        pass
    plt.close(fig)


# --------------------------- Core metrics ---------------------------

ALPH = np.array(list("ACGT"), dtype="U1")


def one_hot_to_seq(one_hot_array: np.ndarray) -> str:
    """(4, L) one-hot -> string of length L using argmax across channels."""
    if one_hot_array.ndim != 2 or one_hot_array.shape[0] != 4:
        raise ValueError(f"Expected (4, L) array, got {one_hot_array.shape}")
    idx = np.argmax(one_hot_array, axis=0)
    return "".join(ALPH[idx])


def column_conservation_nt(aln_seqs: List[str]) -> float:
    """Mean per-column conservation: majority fraction among non-gaps (no gaps here)."""
    if not aln_seqs:
        return float("nan")
    L = len(aln_seqs[0])
    if L == 0:
        return float("nan")
    arr = np.array([list(s) for s in aln_seqs])
    cons_vals: List[float] = []
    for j in range(L):
        col = arr[:, j]
        # no gaps expected; still ignore any non-ACGT just in case
        mask = np.isin(col, ALPH)
        vals = col[mask]
        if vals.size == 0:
            continue
        uniq, counts = np.unique(vals, return_counts=True)
        cons_vals.append(counts.max() / float(vals.size))
    return float(np.mean(cons_vals)) if cons_vals else float("nan")


def mean_pairwise_identity_nt(aln_seqs: List[str]) -> float:
    if not aln_seqs or len(aln_seqs) < 2:
        return float("nan")
    n = len(aln_seqs)
    L = len(aln_seqs[0])
    if any(len(s) != L for s in aln_seqs):
        return float("nan")
    vals: List[float] = []
    for i in range(n):
        ai = aln_seqs[i]
        for j in range(i + 1, n):
            bj = aln_seqs[j]
            matches = sum(1 for a, b in zip(ai, bj) if a == b)
            vals.append(matches / float(L))
    return float(np.mean(vals)) if vals else float("nan")


# --------------------------- Dataset processing ---------------------------

FILENAME_RE = re.compile(r"gc_(?P<gc>\d+\.\d+)_cons_(?P<cons>\d+\.\d+)")


def parse_targets_from_filename(path: str) -> Tuple[Optional[float], Optional[float]]:
    m = FILENAME_RE.search(os.path.basename(path))
    if not m:
        return None, None
    try:
        gc = float(m.group("gc"))
        cons = float(m.group("cons"))
        return gc, cons
    except Exception:
        return None, None


def list_npz_files(root: str, pattern_substr: Optional[str]) -> List[str]:
    files = []
    for name in os.listdir(root):
        if not name.endswith(".npz"):
            continue
        if pattern_substr and (pattern_substr not in name):
            continue
        files.append(os.path.join(root, name))
    return sorted(files)


def extract_motif_seqs_from_npz(npz_path: str, max_pos: Optional[int] = None, seed: int = 42) -> Tuple[List[str], Dict[str, object]]:
    data = np.load(npz_path)
    if not {"X", "y", "masks"}.issubset(data.files):
        raise ValueError(f"npz missing required arrays: {npz_path}")
    X = data["X"]  # (N, 4, L)
    y = data["y"].astype(float)  # (N,)
    masks = data["masks"]  # (N, L) bool
    if X.ndim != 3 or X.shape[1] != 4:
        raise ValueError(f"X must be (N,4,L); got {X.shape} in {npz_path}")
    if masks.shape[0] != X.shape[0] or masks.shape[1] != X.shape[2]:
        raise ValueError(f"masks shape {masks.shape} incompatible with X {X.shape} in {npz_path}")
    # Positive samples only
    pos_idx = np.where(y > 0.5)[0]
    if pos_idx.size == 0:
        return [], {"n_pos": 0, "motif_len": 0}
    rng = np.random.default_rng(seed)
    if max_pos is not None and pos_idx.size > max_pos:
        pos_idx = rng.choice(pos_idx, size=int(max_pos), replace=False)
    # Extract motif sequences via mask indices
    motif_seqs: List[str] = []
    motif_len: Optional[int] = None
    for idx in pos_idx:
        one_hot = X[idx]  # (4,L)
        mask = masks[idx].astype(bool)
        # Determine positions of motif (True)
        motif_positions = np.where(mask)[0]
        if motif_positions.size == 0:
            continue
        if motif_len is None:
            motif_len = int(motif_positions.size)
        # Convert full one-hot to string once, then slice
        seq = one_hot_to_seq(one_hot)
        motif_seq = "".join(seq[p] for p in motif_positions)
        # Sanity: ensure consistent length
        if motif_len is not None and len(motif_seq) != motif_len:
            # skip inconsistent masks
            continue
        motif_seqs.append(motif_seq)
    return motif_seqs, {"n_pos": len(motif_seqs), "motif_len": (motif_len or 0)}


def process_dataset(npz_path: str, max_pos: Optional[int]) -> Dict[str, object]:
    gc_t, cons_t = parse_targets_from_filename(npz_path)
    try:
        motif_seqs, info = extract_motif_seqs_from_npz(npz_path, max_pos=max_pos)
    except Exception as e:
        return {
            "dataset": os.path.basename(npz_path),
            "status": f"error: {e}",
        }
    n_used = info.get("n_pos", 0)
    mlen = info.get("motif_len", 0)
    if n_used < 2 or mlen <= 0:
        return {
            "dataset": os.path.basename(npz_path),
            "status": f"insufficient_data(n_pos={n_used}, motif_len={mlen})",
            "target_gc": gc_t,
            "target_conservation": cons_t,
        }
    nt_col_cons = column_conservation_nt(motif_seqs)
    nt_pid = mean_pairwise_identity_nt(motif_seqs)
    return {
        "dataset": os.path.basename(npz_path),
        "n_pos_used": n_used,
        "motif_len": mlen,
        "nt_mean_col_conservation": nt_col_cons,
        "nt_mean_pairwise_identity": nt_pid,
        "target_gc": gc_t,
        "target_conservation": cons_t,
        "status": "ok",
    }


# --------------------------- Main ---------------------------

def run_pipeline(dataset_dir: str, outdir: str, pattern: Optional[str], max_pos: Optional[int], n_workers: int) -> None:
    os.makedirs(outdir, exist_ok=True)
    print(f"[synth_nt_msa] Scanning datasets in: {os.path.abspath(dataset_dir)}", flush=True)
    files = list_npz_files(dataset_dir, pattern_substr=pattern)
    if not files:
        raise FileNotFoundError(f"No .npz datasets matched under {dataset_dir} with pattern substring='{pattern}'")
    print(f"[synth_nt_msa] Found {len(files)} dataset files.", flush=True)

    results: List[Dict[str, object]] = []
    n_workers = max(1, int(n_workers))
    print(f"[synth_nt_msa] Launching processing with n_workers={n_workers}...", flush=True)
    if n_workers > 1 and len(files) > 1:
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            fut_map = {ex.submit(process_dataset, p, max_pos): p for p in files}
            for i, fut in enumerate(as_completed(fut_map), 1):
                res = fut.result()
                results.append(res)
                if (i % 5) == 0 or i == len(fut_map):
                    print(f"[synth_nt_msa] Progress: {i}/{len(fut_map)} datasets.", flush=True)
    else:
        for i, p in enumerate(files, 1):
            res = process_dataset(p, max_pos=max_pos)
            results.append(res)
            print(f"[synth_nt_msa] Progress: {i}/{len(files)} datasets.", flush=True)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(outdir, "synth_nt_msa_summary.csv"), index=False)
    print(f"[synth_nt_msa] Wrote summary CSV with {len(df)} rows.", flush=True)

    # Plots
    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        print("[synth_nt_msa] No successful datasets to plot.", flush=True)
        return
    _init_plotting()
    # Pairwise identity vs target conservation (if target present)
    if ok["target_conservation"].notna().any():
        sub = ok.dropna(subset=["target_conservation"]).copy()
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        ax.scatter(sub["target_conservation"], sub["nt_mean_pairwise_identity"], s=30, alpha=0.9)
        ax.plot([0, 1], [0, 1], ls=":", c="#888888")
        ax.set_xlabel("Target conservation (toy)")
        ax.set_ylabel("Measured NT pairwise identity")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title("Pairwise identity vs target motif conservation")
        _savefig(fig, os.path.join(outdir, "pairwise_identity_vs_target_conservation"))
    # Distributions
    fig2, ax2 = plt.subplots(figsize=(5.5, 4.0))
    if sns is not None:
        sns.histplot(ok["nt_mean_col_conservation"], bins=30, kde=False, ax=ax2)
    else:
        ax2.hist(ok["nt_mean_col_conservation"].to_numpy(), bins=30)
    ax2.set_xlabel("NT mean column conservation")
    ax2.set_ylabel("Count of datasets")
    ax2.set_title("Distribution of measured conservation across datasets")
    _savefig(fig2, os.path.join(outdir, "conservation_distribution"))


def main():
    ap = argparse.ArgumentParser(description="NT-only synthetic MSA/conservation from cached .npz datasets")
    ap.add_argument("--dataset_dir", default="/home/yhan/GenomeInterpretation/dataset_cache", help="Directory with cached synthetic .npz datasets")
    ap.add_argument("--outdir", default="/home/yhan/GenomeInterpretation/sporulation/analysis_out/synth_nt_msa", help="Output directory")
    ap.add_argument("--pattern", default=None, help="Optional substring to filter dataset filenames (e.g., 'cons_0.70')")
    ap.add_argument("--max_pos", type=int, default=None, help="Max number of positive sequences to use per dataset")
    ap.add_argument("--n_workers", type=int, default=8, help="Parallel workers across datasets")
    args = ap.parse_args()

    run_pipeline(
        dataset_dir=args.dataset_dir,
        outdir=args.outdir,
        pattern=args.pattern,
        max_pos=args.max_pos,
        n_workers=args.n_workers,
    )


if __name__ == "__main__":
    main()


