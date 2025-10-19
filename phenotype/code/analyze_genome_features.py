#!/usr/bin/env python3
"""Analyze genome-scale sequence statistics and simple phenotype logits."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
import statsmodels.api as sm  # type: ignore
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler, SplineTransformer

from phenotype_utils import DATA_ROOT, build_labels_map_and_classes, phenotype_to_slug

# ---------------------------------------------------------------------------
# Sequence helpers
# ---------------------------------------------------------------------------

_BASES = "ACGT"
_BASE_TO_INT = {b: i for i, b in enumerate(_BASES)}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in name).strip("_")


def load_fasta_sequence(path: Path) -> str:
    seq_parts: List[str] = []
    with open(path, "r") as handle:
        for line in handle:
            if not line:
                continue
            if line.startswith(">"):
                continue
            seq_parts.append(line.strip())
    return "".join(seq_parts).upper()


def compute_base_stats(seq: str) -> Tuple[int, Dict[str, int], float, float]:
    arr = np.frombuffer(seq.encode("ascii", errors="ignore"), dtype=np.uint8)
    counts: Dict[str, int] = {}
    known_total = 0
    for base, code in ("A", 65), ("C", 67), ("G", 71), ("T", 84):
        cnt = int((arr == code).sum())
        counts[base] = cnt
        known_total += cnt
    n_count = int((arr == 78).sum())  # 'N'
    length = int(arr.size)
    gc = (counts.get("G", 0) + counts.get("C", 0)) / max(1, known_total)
    n_fraction = n_count / max(1, length)
    return length, counts, gc, n_fraction


def count_kmers(seq: str, k: int) -> np.ndarray:
    mask = (1 << (2 * k)) - 1
    counts = np.zeros(4 ** k, dtype=np.int64)
    value = 0
    span = 0
    for ch in seq:
        base_idx = _BASE_TO_INT.get(ch)
        if base_idx is None:
            value = 0
            span = 0
            continue
        value = ((value << 2) | base_idx) & mask
        span += 1
        if span >= k:
            counts[value] += 1
    return counts


def kmer_to_string(idx: int, k: int) -> str:
    chars = ["A"] * k
    for pos in range(k - 1, -1, -1):
        chars[pos] = _BASES[idx & 3]
        idx >>= 2
    return "".join(chars)


# ---------------------------------------------------------------------------
# File selection and feature extraction
# ---------------------------------------------------------------------------


def select_files_per_class(
    dataset_dir: Path,
    labels_map: Dict[str, int],
    max_per_class: Optional[int],
    seed: int,
) -> Dict[int, List[Path]]:
    class_files: Dict[int, List[Path]] = {label: [] for label in set(labels_map.values())}
    for entry in sorted(dataset_dir.iterdir()):
        if not entry.is_file():
            continue
        if not entry.name.lower().endswith((".fasta", ".fa", ".fna")):
            continue
        label_idx = labels_map.get(entry.name.strip().lower())
        if label_idx is None:
            continue
        class_files.setdefault(label_idx, []).append(entry)
    rng = np.random.default_rng(seed)
    selected: Dict[int, List[Path]] = {}
    for idx, files in class_files.items():
        if not files:
            continue
        if max_per_class is not None and len(files) > max_per_class:
            chosen = rng.choice(len(files), size=max_per_class, replace=False)
            selected[idx] = [files[i] for i in sorted(chosen)]
        else:
            selected[idx] = files
    return selected


def aggregate_statistics(
    dataset_name: str,
    file_map: Dict[int, List[Path]],
    classes: Sequence[str],
    k_values: Sequence[int],
    verbose: bool = False,
) -> Tuple[pd.DataFrame, Dict[int, np.ndarray]]:
    records: List[Dict] = []
    per_k_freqs: Dict[int, List[np.ndarray]] = {k: [] for k in k_values}

    total_files = sum(len(v) for v in file_map.values())
    processed = 0
    report_step = max(1, total_files // 50)
    next_report = report_step

    for class_idx, files in file_map.items():
        if verbose:
            print(
                f"[genome-stats] Dataset '{dataset_name}': processing class '{classes[class_idx]}' with {len(files)} genomes",
                flush=True,
            )
        for path in files:
            seq = load_fasta_sequence(path)
            if not seq:
                continue
            length, counts, gc, n_frac = compute_base_stats(seq)
            record = {
                "filename": path.name,
                "dataset": dataset_name,
                "class_idx": int(class_idx),
                "class_name": classes[class_idx],
                "length": int(length),
                "gc_content": float(gc),
                "n_fraction": float(n_frac),
            }
            for k in k_values:
                counts_k = count_kmers(seq, k)
                total_k = int(counts_k.sum())
                record[f"k{k}_total"] = total_k
                freq = counts_k.astype(np.float32)
                if total_k > 0:
                    freq /= float(total_k)
                per_k_freqs[k].append(freq)
            records.append(record)
            processed += 1
            if verbose and processed >= next_report:
                print(
                    f"[genome-stats]   processed {processed}/{total_files} genomes in '{dataset_name}'",
                    flush=True,
                )
                next_report += report_step
        if verbose:
            print(
                f"[genome-stats] Dataset '{dataset_name}': completed class '{classes[class_idx]}'",
                flush=True,
            )
    if processed == 0:
        df = pd.DataFrame(columns=["filename", "dataset", "class_idx", "class_name", "length", "gc_content", "n_fraction"])
    else:
        df = pd.DataFrame(records)
    freq_arrays = {
        k: (np.vstack(per_k_freqs[k]) if per_k_freqs[k] else np.zeros((0, 4 ** k), dtype=np.float32))
        for k in k_values
    }
    return df, freq_arrays


def load_or_compute_dataset(
    slug: str,
    dataset_name: str,
    dataset_dir: Path,
    labels_map: Dict[str, int],
    classes: Sequence[str],
    k_values: Sequence[int],
    cache_dir: Path,
    max_per_class: Optional[int],
    seed: int,
    force_recompute: bool,
    verbose: bool,
) -> Tuple[pd.DataFrame, Dict[int, np.ndarray]]:
    dataset_cache = cache_dir / slug
    ensure_dir(dataset_cache)
    df_path = dataset_cache / f"{dataset_name}_records.parquet"
    freq_paths = {k: dataset_cache / f"{dataset_name}_k{k}_freqs.npz" for k in k_values}

    need_compute = force_recompute or not df_path.exists() or any(not p.exists() for p in freq_paths.values())

    if need_compute:
        file_map = select_files_per_class(dataset_dir, labels_map, max_per_class, seed)
        if not file_map:
            return pd.DataFrame(), {k: np.zeros((0, 4 ** k), dtype=np.float32) for k in k_values}
        df, freq_arrays = aggregate_statistics(dataset_name, file_map, classes, k_values, verbose=verbose)
        df.to_parquet(df_path, index=False)
        for k, arr in freq_arrays.items():
            np.savez_compressed(freq_paths[k], freqs=arr)
    else:
        df = pd.read_parquet(df_path)
        freq_arrays = {k: np.load(path)["freqs"] for k, path in freq_paths.items()}
    return df, freq_arrays


def compute_class_kmer_stats(
    df: pd.DataFrame,
    freq_arrays: Dict[int, np.ndarray],
    classes: Sequence[str],
    k_values: Sequence[int],
) -> Tuple[Dict[int, Dict[int, np.ndarray]], Dict[int, Dict[int, float]]]:
    counts: Dict[int, Dict[int, np.ndarray]] = {
        idx: {k: np.zeros(4 ** k, dtype=np.float64) for k in k_values}
        for idx in range(len(classes))
    }
    totals: Dict[int, Dict[int, float]] = {
        idx: {k: 0.0 for k in k_values} for idx in range(len(classes))
    }
    if df.empty:
        return counts, totals
    for i, row in df.iterrows():
        class_idx = int(row["class_idx"])
        for k in k_values:
            total_k = float(row.get(f"k{k}_total", 0.0))
            if total_k <= 0:
                continue
            counts[class_idx][k] += freq_arrays[k][i] * total_k
            totals[class_idx][k] += total_k
    return counts, totals


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_histogram(
    data_by_class: Dict[str, Sequence[float]],
    bins: Sequence[float],
    xlabel: str,
    title: str,
    out_path: Path,
    density: bool = False,
) -> None:
    plt.figure(figsize=(10, 6))
    for name, values in data_by_class.items():
        if not values:
            continue
        plt.hist(
            values,
            bins=bins,
            alpha=0.55,
            edgecolor="black",
            linewidth=0.5,
            label=f"{name} (n={len(values)})",
            density=density,
        )
    plt.xlabel(xlabel)
    plt.ylabel("Density" if density else "Count")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_scatter(
    x_by_class: Dict[str, Sequence[float]],
    y_by_class: Dict[str, Sequence[float]],
    xlabel: str,
    ylabel: str,
    title: str,
    out_path: Path,
    max_points_per_class: int = 500,
    seed: int = 0,
) -> None:
    plt.figure(figsize=(10, 6))
    rng = np.random.default_rng(seed)
    for name in x_by_class:
        xs = np.asarray(x_by_class[name], dtype=float)
        ys = np.asarray(y_by_class[name], dtype=float)
        if xs.size == 0:
            continue
        if xs.size > max_points_per_class:
            idx = rng.choice(xs.size, size=max_points_per_class, replace=False)
            xs = xs[idx]
            ys = ys[idx]
        plt.scatter(xs, ys, label=name, alpha=0.6, s=20)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_top_kmers(
    kmers: List[str],
    log2fc: List[float],
    freq_class: List[float],
    freq_rest: List[float],
    class_name: str,
    k: int,
    out_path: Path,
) -> None:
    if not kmers:
        return
    x = np.arange(len(kmers))
    colors = ["#d95f02" if v < 0 else "#1b9e77" for v in log2fc]
    plt.figure(figsize=(max(8, len(kmers) * 0.4), 6))
    plt.bar(x, log2fc, color=colors, edgecolor="black", linewidth=0.5)
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.xticks(x, kmers, rotation=75, ha="right", fontsize=9)
    plt.ylabel("log2(freq_class / freq_rest)")
    plt.title(f"Top enriched {k}-mers for {class_name}")
    ylim = max(abs(min(log2fc)), abs(max(log2fc))) or 1.0
    plt.ylim(-ylim * 1.1, ylim * 1.1)
    for x_pos, logfc, f_c, f_r in zip(x, log2fc, freq_class, freq_rest):
        if not np.isfinite(logfc):
            continue
        va = "bottom" if logfc >= 0 else "top"
        offset = 0.02 * ylim
        y = logfc + (offset if logfc >= 0 else -offset)
        plt.text(
            x_pos,
            y,
            f"{f_c:.2e}\nvs {f_r:.2e}",
            ha="center",
            va=va,
            fontsize=7,
            rotation=90,
        )
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_kmer_heatmap(
    freqs_by_class: Dict[str, np.ndarray],
    k: int,
    out_path: Path,
) -> None:
    if not freqs_by_class:
        return
    class_names = list(freqs_by_class.keys())
    kmers = [kmer_to_string(i, k) for i in range(4 ** k)]
    data = np.vstack([freqs_by_class[name] for name in class_names])
    fig, ax = plt.subplots(figsize=(min(24, 2 + data.shape[1] * 0.25), 4 + data.shape[0] * 0.6))
    im = ax.imshow(data, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(kmers)))
    ax.set_xticklabels(kmers, rotation=90, fontsize=7)
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_yticklabels(class_names)
    ax.set_xlabel(f"{k}-mer")
    ax.set_title(f"Normalized {k}-mer frequencies by class")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Frequency")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary and logistic modelling
# ---------------------------------------------------------------------------

def summarize_per_class(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
    }


def build_summary(
    train_df: pd.DataFrame,
    classes: Sequence[str],
    enrichment: Dict[int, Dict[int, Dict[str, np.ndarray]]],
    k_values: Sequence[int],
    out_dir: Path,
) -> Dict:
    summary: Dict = {
        "total_genomes": int(len(train_df)),
        "classes": list(classes),
        "per_class_counts": train_df.groupby("class_name").size().to_dict(),
        "plots": {},
    }

    print("[genome-stats]   plotting GC/length distributions", flush=True)

    gc_by_class: Dict[str, List[float]] = {}
    length_by_class: Dict[str, List[float]] = {}
    n_by_class: Dict[str, List[float]] = {}
    for class_name in classes:
        sub = train_df[train_df["class_name"] == class_name]
        gc_vals = sub["gc_content"].tolist()
        len_vals = sub["length"].tolist()
        n_vals = sub["n_fraction"].tolist()
        gc_by_class[class_name] = gc_vals
        length_by_class[class_name] = len_vals
        n_by_class[class_name] = n_vals
        summary.setdefault("gc_content_stats", {})[class_name] = summarize_per_class(gc_vals)
        summary.setdefault("length_stats", {})[class_name] = summarize_per_class(len_vals)
        summary.setdefault("n_content_stats", {})[class_name] = summarize_per_class(n_vals)

    gc_plot = out_dir / "gc_content_distribution.png"
    plot_histogram(gc_by_class, bins=np.linspace(0, 1, 60), xlabel="GC content", title="GC content distribution by class", out_path=gc_plot, density=True)
    summary["plots"]["gc_content_distribution"] = str(gc_plot)

    if train_df.empty:
        length_bins = np.logspace(4, 6, 10)
    else:
        length_min = max(1.0, float(train_df["length"].min()))
        length_max = max(1.0, float(train_df["length"].max()))
        log_start = math.log10(length_min)
        log_stop = math.log10(length_max)
        if math.isclose(log_start, log_stop):
            log_stop += 0.5
        length_bins = np.logspace(log_start, log_stop, 50)
    length_plot = out_dir / "genome_length_distribution.png"
    plot_histogram(length_by_class, bins=length_bins, xlabel="Genome length (bp)", title="Genome length distribution by class", out_path=length_plot)
    summary["plots"]["genome_length_distribution"] = str(length_plot)

    scatter_plot = out_dir / "gc_vs_length_scatter.png"
    plot_scatter(length_by_class, gc_by_class, xlabel="Genome length (bp)", ylabel="GC content", title="GC vs length by class", out_path=scatter_plot)
    summary["plots"]["gc_vs_length_scatter"] = str(scatter_plot)

    for k in k_values:
        print(f"[genome-stats]   processing k={k} enrichment", flush=True)
        freqs_for_heatmap: Dict[str, np.ndarray] = {}
        for idx, class_name in enumerate(classes):
            freq = enrichment.get(idx, {}).get(k, {}).get("freq_class")
            if freq is None:
                continue
            freqs_for_heatmap[class_name] = freq
        heatmap_path = out_dir / f"k{k}_frequency_heatmap.png"
        plot_kmer_heatmap(freqs_for_heatmap, k=k, out_path=heatmap_path)
        summary["plots"][f"k{k}_frequency_heatmap"] = str(heatmap_path)

        enrichment_summary: Dict[str, List[Dict[str, float]]] = {}
        for idx, class_name in enumerate(classes):
            data = enrichment.get(idx, {}).get(k)
            if not data:
                enrichment_summary[class_name] = []
                continue
            log2fc = data["log2fc"]
            freq_class = data["freq_class"]
            freq_rest = data["freq_rest"]
            order = np.argsort(-np.abs(log2fc))[:20]
            kmers = [kmer_to_string(int(i), k) for i in order]
            top_log2fc = [float(log2fc[i]) for i in order]
            top_fc = [float(freq_class[i]) for i in order]
            top_rest = [float(freq_rest[i]) for i in order]
            enrichment_summary[class_name] = [
                {
                    "kmer": kmers[i],
                    "log2fc": top_log2fc[i],
                    "freq_class": top_fc[i],
                    "freq_rest": top_rest[i],
                }
                for i in range(len(order))
            ]
            bar_path = out_dir / f"k{k}_top_kmers_{class_name}.png"
            plot_top_kmers(kmers, top_log2fc, top_fc, top_rest, class_name, k, bar_path)
            summary["plots"][f"k{k}_top_kmers_{class_name}"] = str(bar_path)
        summary.setdefault("top_kmers", {})[f"k{k}"] = enrichment_summary

    return summary


def select_top_kmer_indices(
    enrichment: Dict[int, Dict[int, Dict[str, np.ndarray]]],
    k: int,
    top_n: int,
) -> List[int]:
    selected: set[int] = set()
    if top_n <= 0:
        return []
    for data in enrichment.values():
        if k not in data:
            continue
        log2fc = data[k]["log2fc"]
        if log2fc.size == 0:
            continue
        n = min(top_n, log2fc.size)
        if n <= 0:
            continue
        top = np.argpartition(-np.abs(log2fc), n - 1)[:n]
        selected.update(int(i) for i in top)
    return sorted(selected)


def build_design_matrices(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_freqs: Dict[int, np.ndarray],
    test_freqs: Dict[int, np.ndarray],
    k: int,
    selected_indices: Sequence[int],
    spline_knots: int = 6,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], SplineTransformer, List[str]]:
    n_frac_train = train_df["n_fraction"].astype(float)
    gc_train = train_df["gc_content"].astype(float).to_numpy().reshape(-1, 1)

    spline = SplineTransformer(degree=3, n_knots=spline_knots, include_bias=False)
    gc_spline_train = spline.fit_transform(gc_train)
    gc_spline_names = [f"gc_spline_{i}" for i in range(gc_spline_train.shape[1])]

    base_train = pd.DataFrame(
        {
            "gc_content": gc_train.ravel(),
            "n_fraction": n_frac_train,
        },
        index=train_df.index,
    )
    if gc_spline_train.shape[1] > 0:
        for i, name in enumerate(gc_spline_names):
            base_train[name] = gc_spline_train[:, i]

    if selected_indices:
        kmer_matrix = train_freqs[k][:, selected_indices]
        for j, idx in enumerate(selected_indices):
            base_train[f"k{k}_{kmer_to_string(idx, k)}"] = kmer_matrix[:, j]

    # Test matrix uses same spline transformer
    if not test_df.empty:
        gc_test = test_df["gc_content"].astype(float).to_numpy().reshape(-1, 1)
        gc_spline_test = spline.transform(gc_test)
        base_test = pd.DataFrame(
            {
                "gc_content": gc_test.ravel(),
                "n_fraction": test_df["n_fraction"].astype(float),
            },
            index=test_df.index,
        )
        if gc_spline_test.shape[1] > 0:
            for i, name in enumerate(gc_spline_names):
                base_test[name] = gc_spline_test[:, i]
        if selected_indices:
            kmer_matrix_test = test_freqs[k][:, selected_indices]
            for j, idx in enumerate(selected_indices):
                base_test[f"k{k}_{kmer_to_string(idx, k)}"] = kmer_matrix_test[:, j]
    else:
        base_test = pd.DataFrame(columns=base_train.columns)

    feature_names = list(base_train.columns)
    return base_train, base_test, feature_names, spline, gc_spline_names


def plot_partial_effect(
    feature_name: str,
    feature_values: np.ndarray,
    scaler: StandardScaler,
    clf: LogisticRegression,
    base_point: np.ndarray,
    feature_index: int,
    class_idx: int,
    class_label: str,
    out_path: Path,
) -> None:
    lo = float(np.percentile(feature_values, 5))
    hi = float(np.percentile(feature_values, 95))
    if math.isclose(lo, hi):
        lo, hi = float(feature_values.min()), float(feature_values.max())
    if math.isclose(lo, hi):
        lo -= 1e-3
        hi += 1e-3
    grid = np.linspace(lo, hi, 100)
    samples = np.tile(base_point, (grid.size, 1))
    samples[:, feature_index] = grid
    samples_scaled = scaler.transform(samples)
    probs = clf.predict_proba(samples_scaled)[:, class_idx]
    plt.figure(figsize=(6, 4))
    plt.plot(grid, probs, color="tab:blue")
    plt.xlabel(feature_name)
    plt.ylabel(f"P({class_label} | features)")
    plt.title(f"Partial effect: {feature_name} → {class_label}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_partial_effect_gc(
    gc_values: np.ndarray,
    scaler: StandardScaler,
    clf: LogisticRegression,
    base_point: np.ndarray,
    feature_names: List[str],
    gc_spline_names: List[str],
    spline: SplineTransformer,
    class_idx: int,
    class_label: str,
    out_path: Path,
) -> None:
    gc_idx = feature_names.index("gc_content")
    lo = float(np.percentile(gc_values, 5))
    hi = float(np.percentile(gc_values, 95))
    if math.isclose(lo, hi):
        lo = float(gc_values.min())
        hi = float(gc_values.max())
    if math.isclose(lo, hi):
        lo -= 1e-3
        hi += 1e-3
    grid = np.linspace(lo, hi, 100)
    samples = np.tile(base_point, (grid.size, 1))
    for i, val in enumerate(grid):
        samples[i, gc_idx] = val
        spline_vals = spline.transform(np.array([[val]])).ravel()
        for j, name in enumerate(gc_spline_names):
            idx = feature_names.index(name)
            samples[i, idx] = spline_vals[j]
    probs = clf.predict_proba(scaler.transform(samples))[:, class_idx]
    plt.figure(figsize=(6, 4))
    plt.plot(grid, probs, color="tab:blue")
    plt.xlabel("gc_content")
    plt.ylabel(f"P({class_label} | features)")
    plt.title(f"Partial effect: gc_content → {class_label}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    roc_auc: float,
    out_path: Path,
    title: str,
) -> None:
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="tab:blue", lw=2, label=f"ROC AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], color="black", lw=1, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def run_logistic_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_freqs: Dict[int, np.ndarray],
    test_freqs: Dict[int, np.ndarray],
    classes: Sequence[str],
    enrichment: Dict[int, Dict[int, Dict[str, np.ndarray]]],
    k_values: Sequence[int],
    top_kmer_features: int,
    out_dir: Path,
) -> Dict[str, Dict[str, object]]:
    results: Dict[str, Dict[str, object]] = {}
    unique_classes = sorted(train_df["class_idx"].unique().tolist())
    n_classes = len(unique_classes)
    if n_classes <= 1:
        print("[genome-stats] Logistic analysis skipped (only one class present)", flush=True)
        return results
    class_mapping = {orig: idx for idx, orig in enumerate(unique_classes)}
    inv_mapping = {idx: orig for orig, idx in class_mapping.items()}
    model_class_names = [classes[inv_mapping[idx]] for idx in range(n_classes)]
    train_labels = train_df["class_idx"].map(class_mapping).to_numpy()
    test_labels = test_df["class_idx"].map(class_mapping).to_numpy() if not test_df.empty else None

    for k in k_values:
        selected_indices = select_top_kmer_indices(enrichment, k, top_kmer_features)
        feature_train_df, feature_test_df, feature_names, spline, gc_spline_names = build_design_matrices(
            train_df,
            test_df,
            train_freqs,
            test_freqs,
            k,
            selected_indices,
        )
        if feature_train_df.empty:
            continue

        scaler = StandardScaler()
        X_train = feature_train_df.to_numpy(dtype=float)
        X_train_scaled = scaler.fit_transform(X_train)
        clf = LogisticRegression(
            max_iter=2000,
            solver="lbfgs",
            class_weight="balanced",
            C=0.5,
        )
        clf.fit(X_train_scaled, train_labels)

        bal_acc = None
        test_pred_path = None
        auc_score = None
        roc_path = None
        if test_labels is not None and len(test_labels) == feature_test_df.shape[0] and feature_test_df.shape[0] > 0:
            X_test_scaled = scaler.transform(feature_test_df.to_numpy(dtype=float))
            probs_all = clf.predict_proba(X_test_scaled)
            y_pred = probs_all.argmax(axis=1)
            bal_acc = float(balanced_accuracy_score(test_labels, y_pred))
            true_prob = probs_all[np.arange(len(test_labels)), test_labels]
            pred_df = pd.DataFrame(
                {
                    "filename": test_df["filename"].tolist(),
                    "class_name": test_df["class_name"].tolist(),
                    "true_label": test_labels,
                    "pred_label": y_pred,
                    "pred_prob_true_class": true_prob,
                }
            )
            test_pred_path = out_dir / f"logit_test_predictions_k{k}.parquet"
            if n_classes == 2 and len(np.unique(test_labels)) > 1:
                fpr, tpr, _ = roc_curve(test_labels, probs_all[:, 1])
                auc_score = float(auc(fpr, tpr))
                roc_path = out_dir / f"roc_curve_k{k}.png"
                plot_roc_curve(fpr, tpr, auc_score, roc_path, f"ROC Curve (k={k})")
            # enrich prediction table with readable labels and probs per class
            pred_df["true_label_name"] = [model_class_names[idx] for idx in test_labels]
            pred_df["pred_label_name"] = [model_class_names[idx] for idx in y_pred]
            for class_idx, class_label in enumerate(model_class_names):
                pred_df[f"pred_prob_{safe_name(class_label)}"] = probs_all[:, class_idx]
            pred_df.to_parquet(test_pred_path, index=False)

        probs_train_all = clf.predict_proba(X_train_scaled)
        eps = 1e-9
        ll_model = float(
            np.sum(np.log(probs_train_all[np.arange(len(train_labels)), train_labels] + eps))
        )
        class_counts = np.bincount(train_labels, minlength=n_classes).astype(float)
        class_probs = class_counts / max(1.0, class_counts.sum())
        ll_null = float(
            np.sum(np.log(class_probs[train_labels] + eps))
        )
        pseudo_r2 = float(1.0 - (ll_model / ll_null)) if ll_null != 0 else float("nan")

        coef_matrix = clf.coef_
        intercept_vec = clf.intercept_
        if n_classes == 2 and coef_matrix.shape[0] == 1:
            coef_matrix = np.vstack([-coef_matrix[0], coef_matrix[0]])
            intercept_vec = np.array([-intercept_vec[0], intercept_vec[0]])

        summary_lines = [
            f"Logistic regression (scikit-learn lbfgs, C={clf.C}, penalty=L2, classes={n_classes})",
            f"Training samples: {len(train_labels)} | class distribution: {[int(x) for x in class_counts.tolist()]}",
            f"Pseudo R^2 (McFadden): {pseudo_r2:.4f}",
            f"Balanced accuracy (test): {bal_acc if bal_acc is not None else 'n/a'}",
        ]
        if auc_score is not None:
            summary_lines.append(f"ROC AUC (test): {auc_score:.4f}")
        summary_lines.append("Coefficients per class:")
        for class_idx, class_label in enumerate(model_class_names):
            summary_lines.append(f"  [{class_label}] intercept: {intercept_vec[class_idx]:.6f}")
            for name, coef_val in zip(feature_names, coef_matrix[class_idx]):
                summary_lines.append(f"    {name}: {coef_val:.6f}")
        summary_text = "\n".join(summary_lines)
        summary_path = out_dir / f"logit_summary_k{k}.txt"
        with open(summary_path, "w") as handle:
            handle.write(summary_text + "\n")

        base_point = feature_train_df.mean().to_numpy(dtype=float)
        plot_features = ["gc_content", "log_length", "n_fraction"]
        coef_abs = pd.Series(np.linalg.norm(coef_matrix, axis=0), index=feature_names)
        sorted_features = [
            name
            for name in coef_abs.sort_values(ascending=False).index
            if name not in plot_features
        ]
        plot_features.extend(sorted_features[: min(3, len(sorted_features))])

        partial_plots: Dict[str, Dict[str, str]] = {}
        for class_idx, class_label in enumerate(model_class_names):
            class_key = safe_name(class_label) or f"class_{class_idx}"
            class_plots: Dict[str, str] = {}
            for name in plot_features:
                if name not in feature_names:
                    continue
                out_path = out_dir / f"partial_effect_k{k}_{name}_{class_key}.png"
                values = feature_train_df[name].to_numpy(dtype=float)
                if name == "gc_content":
                    plot_partial_effect_gc(
                        values,
                        scaler,
                        clf,
                        base_point.copy(),
                        feature_names,
                        gc_spline_names,
                        spline,
                        class_idx,
                        class_label,
                        out_path,
                    )
                else:
                    idx = feature_names.index(name)
                    plot_partial_effect(
                        name,
                        values,
                        scaler,
                        clf,
                        base_point.copy(),
                        idx,
                        class_idx,
                        class_label,
                        out_path,
                    )
                class_plots[name] = str(out_path)
            partial_plots[class_label] = class_plots

        coefficients: Dict[str, Dict[str, float]] = {}
        for class_idx, class_label in enumerate(model_class_names):
            coef_map = {name: float(value) for name, value in zip(feature_names, coef_matrix[class_idx])}
            coef_map["intercept"] = float(intercept_vec[class_idx])
            coefficients[class_label] = coef_map

        results[f"k{k}"] = {
            "selected_kmers": [name for name in feature_names if name.startswith(f"k{k}_")],
            "balanced_accuracy": bal_acc,
            "roc_auc": auc_score,
            "pseudo_r2": pseudo_r2,
            "coefficients": coefficients,
            "class_labels": model_class_names,
            "summary_path": str(summary_path),
            "partial_effect_plots": partial_plots,
            "test_predictions": str(test_pred_path) if test_pred_path is not None else None,
            "roc_curve_path": str(roc_path) if roc_path is not None else None,
        }
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Statistical analysis of phenotype training genomes")
    ap.add_argument("--phenotype", type=str, default="Spore formation", help="Phenotype column to analyze")
    ap.add_argument("--metadata", type=str, default="sporulation/microbe.cards table S1.xlsx", help="Path to metadata Excel file")
    ap.add_argument("--data_root", type=str, default=str(DATA_ROOT), help="Root directory containing train/validation/test FASTA")
    ap.add_argument("--k_values", type=str, default="3,4,6,8", help="Comma-separated list of k-mer sizes to profile")
    ap.add_argument("--max_genomes_per_class", type=int, default=None, help="Optional cap on number of genomes per class to process")
    ap.add_argument("--seed", type=int, default=1337, help="Random seed for sampling when applying caps")
    ap.add_argument("--out_dir", type=str, default=None, help="Optional output directory (defaults to phenotype/plots/genome_stats/<slug>)")
    ap.add_argument("--cache_dir", type=str, default=None, help="Cache directory under DATA_ROOT (defaults to DATA_ROOT/cache/genome_stats)")
    ap.add_argument("--force_recompute", action="store_true", help="Force recomputation even if cached features exist")
    ap.add_argument("--top_kmer_features", type=int, default=40, help="Top enriched k-mers per class (union) to include in logistic models")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    phenotype = args.phenotype.strip()
    slug = phenotype_to_slug(phenotype)

    data_root = Path(args.data_root).expanduser().resolve()
    train_dir = data_root / "train"
    test_dir = data_root / "test"

    train_dirs = [train_dir]

    try:
        metadata_df = pd.read_excel(args.metadata)
    except Exception as exc:
        raise RuntimeError(f"Failed to read metadata Excel at {args.metadata}: {exc}")

    labels_map, classes = build_labels_map_and_classes(
        metadata_df,
        phenotype_col=phenotype,
        file_col="Fasta file",
        train_dirs=train_dirs,
    )
    if not classes:
        raise RuntimeError(f"No classes inferred for phenotype '{phenotype}'")

    if not train_dir.is_dir():
        raise FileNotFoundError(f"Training directory not found: {train_dir}")
    if not test_dir.is_dir():
        print(f"[genome-stats] Warning: test directory not found ({test_dir}); logistic evaluation will be skipped", flush=True)

    k_values = [int(k.strip()) for k in args.k_values.split(",") if k.strip()]
    if not k_values:
        raise ValueError("At least one k-mer size must be specified")
    print(f"[genome-stats] Profiling k-mer sizes: {k_values}", flush=True)

    cache_dir = Path(args.cache_dir) if args.cache_dir else Path(DATA_ROOT) / "cache" / "genome_stats"
    ensure_dir(cache_dir)

    train_df, train_freqs = load_or_compute_dataset(
        slug,
        "train",
        train_dir,
        labels_map,
        classes,
        k_values,
        cache_dir,
        args.max_genomes_per_class,
        args.seed,
        args.force_recompute,
        verbose=True,
    )
    if train_df.empty:
        raise RuntimeError("No training genomes were processed; check phenotype labels and data availability")
    total_selected = len(train_df)
    print(f"[genome-stats] Selected {total_selected} training genomes across {len(classes)} classes", flush=True)

    if test_dir.is_dir():
        test_df, test_freqs = load_or_compute_dataset(
            slug,
            "test",
            test_dir,
            labels_map,
            classes,
            k_values,
            cache_dir,
            args.max_genomes_per_class,
            args.seed,
            args.force_recompute,
            verbose=True,
        )
        print(f"[genome-stats] Loaded test genomes: {len(test_df)}", flush=True)
    else:
        test_df, test_freqs = pd.DataFrame(), {k: np.zeros((0, 4 ** k), dtype=np.float32) for k in k_values}

    counts, totals = compute_class_kmer_stats(train_df, train_freqs, classes, k_values)
    enrichment: Dict[int, Dict[int, Dict[str, np.ndarray]]] = {idx: {} for idx in range(len(classes))}
    for k in k_values:
        all_counts = sum((counts[idx][k] for idx in range(len(classes))), start=np.zeros(4 ** k, dtype=np.float64))
        all_total = sum(totals[idx][k] for idx in range(len(classes)))
        for idx in range(len(classes)):
            counts_c = counts[idx][k]
            total_c = totals[idx][k]
            counts_rest = all_counts - counts_c
            total_rest = all_total - total_c
            freq_c = (counts_c + 1.0) / max(1.0, total_c + 1.0 * counts_c.size)
            if total_rest <= 0:
                freq_rest = np.full_like(freq_c, freq_c.mean())
            else:
                freq_rest = (counts_rest + 1.0) / max(1.0, total_rest + 1.0 * counts_rest.size)
            log2fc = np.log2(np.divide(freq_c, freq_rest, out=np.zeros_like(freq_c), where=freq_rest > 0))
            enrichment[idx][k] = {
                "freq_class": freq_c,
                "freq_rest": freq_rest,
                "log2fc": log2fc,
            }

    out_dir = Path(args.out_dir) if args.out_dir else Path("phenotype/plots/genome_stats") / slug
    ensure_dir(out_dir)

    print(f"[genome-stats] Building plots and summary in {out_dir}", flush=True)
    summary = build_summary(train_df, classes, enrichment, k_values, out_dir)

    stats_path = out_dir / "genome_feature_stats.parquet"
    train_df.to_parquet(stats_path, index=False)
    summary["feature_table"] = str(stats_path)

    print("[genome-stats] Fitting logistic baselines", flush=True)
    logistic_results = run_logistic_models(
        train_df,
        test_df,
        train_freqs,
        test_freqs,
        classes,
        enrichment,
        k_values,
        args.top_kmer_features,
        out_dir,
    )
    summary["logistic_models"] = logistic_results

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Analysis complete for phenotype '{phenotype}'. Summary written to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
