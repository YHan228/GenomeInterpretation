#!/usr/bin/env python3
"""Genome snippet analysis with GC%, k-mer heatmaps, and L1 logistic models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

from phenotype_utils import DATA_ROOT, build_labels_map_and_classes, phenotype_to_slug

_BASES = "ACGT"
_BASE_TO_INT = {b: i for i, b in enumerate(_BASES)}
_FASTA_SUFFIXES = (".fasta", ".fa", ".fna", ".fasta.gz", ".fa.gz", ".fna.gz")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in name).strip("_")


def load_fasta_sequence(path: Path) -> str:
    seq_parts: List[str] = []
    with open(path, "r") as handle:
        for line in handle:
            if not line or line.startswith(">"):
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


def gather_dataset_files(directories: Iterable[Path], labels_map: Dict[str, int]) -> Dict[int, List[Path]]:
    files_by_class: Dict[int, List[Path]] = {}
    for directory in directories:
        directory = Path(directory)
        if not directory.is_dir():
            continue
        for entry in sorted(directory.iterdir()):
            if not entry.is_file():
                continue
            lower_name = entry.name.strip().lower()
            if not lower_name.endswith(_FASTA_SUFFIXES):
                continue
            class_idx = labels_map.get(lower_name)
            if class_idx is None:
                continue
            files_by_class.setdefault(class_idx, []).append(entry)
    return files_by_class


def sample_snippet_dataset(
    dataset_name: str,
    files_by_class: Dict[int, List[Path]],
    classes: Sequence[str],
    k_values: Sequence[int],
    snippet_length: int,
    max_snippets: int,
    seed: int,
    verbose: bool = False,
) -> Tuple[pd.DataFrame, Dict[int, np.ndarray]]:
    k_values = sorted(set(int(k) for k in k_values))
    freq_template = {k: np.zeros((0, 4 ** k), dtype=np.float32) for k in k_values}
    if max_snippets <= 0 or not files_by_class:
        return pd.DataFrame(), freq_template

    available: List[Tuple[int, Path]] = []
    for class_idx, paths in files_by_class.items():
        for path in paths:
            available.append((class_idx, path))
    if not available:
        return pd.DataFrame(), freq_template

    rng = np.random.default_rng(seed)
    records: List[Dict] = []
    per_k_freqs: Dict[int, List[np.ndarray]] = {k: [] for k in k_values}
    seq_cache: Dict[Path, str] = {}
    target = int(max_snippets)
    max_attempts = max(target * 20, len(available) * 10, 1000)
    attempts = 0
    report_step = max(1, target // 10) if target > 0 else 1

    while len(records) < target and attempts < max_attempts:
        attempts += 1
        class_idx, path = available[int(rng.integers(len(available)))]
        seq = seq_cache.get(path)
        if seq is None:
            seq = load_fasta_sequence(path)
            seq_cache[path] = seq
        if len(seq) < snippet_length:
            continue
        max_start = len(seq) - snippet_length
        if max_start < 0:
            continue
        start = int(rng.integers(0, max_start + 1))
        snippet_seq = seq[start : start + snippet_length]
        if not snippet_seq:
            continue
        length, _, gc, n_frac = compute_base_stats(snippet_seq)
        record = {
            "snippet_id": f"{dataset_name}_{len(records):06d}",
            "dataset": dataset_name,
            "filename": path.name,
            "source_path": str(path),
            "class_idx": int(class_idx),
            "class_name": classes[class_idx],
            "snippet_start": int(start),
            "snippet_length": int(snippet_length),
            "length": int(length),
            "gc_content": float(gc),
            "n_fraction": float(n_frac),
        }
        for k in k_values:
            counts_k = count_kmers(snippet_seq, k)
            total_k = int(counts_k.sum())
            record[f"k{k}_total"] = total_k
            freq = counts_k.astype(np.float32)
            if total_k > 0:
                freq /= float(total_k)
            per_k_freqs[k].append(freq)
        records.append(record)
        if verbose and len(records) % report_step == 0:
            print(
                f"[genome-stats]   sampled {len(records)}/{target} snippets for '{dataset_name}'",
                flush=True,
            )

    if not records:
        return pd.DataFrame(), freq_template
    if len(records) < target:
        print(
            f"[genome-stats] Warning: sampled {len(records)} of requested {target} snippets for '{dataset_name}'",
            flush=True,
        )

    df = pd.DataFrame(records).reset_index(drop=True)
    freq_arrays = {
        k: (np.vstack(per_k_freqs[k]) if per_k_freqs[k] else np.zeros((0, 4 ** k), dtype=np.float32))
        for k in k_values
    }
    return df, freq_arrays


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


def plot_gc_content(train_df: pd.DataFrame, classes: Sequence[str], out_path: Path) -> None:
    gc_by_class: Dict[str, List[float]] = {}
    for class_name in classes:
        sub = train_df[train_df["class_name"] == class_name]
        gc_by_class[class_name] = sub["gc_content"].astype(float).tolist()
    bins = np.linspace(0, 1, 60)
    plot_histogram(
        gc_by_class,
        bins=bins,
        xlabel="GC content",
        title="GC content distribution by class",
        out_path=out_path,
        density=True,
    )


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
            if total_k <= 0 or freq_arrays[k].shape[0] <= i:
                continue
            counts[class_idx][k] += freq_arrays[k][i] * total_k
            totals[class_idx][k] += total_k
    return counts, totals


def compute_enrichment_statistics(
    train_df: pd.DataFrame,
    train_freqs: Dict[int, np.ndarray],
    classes: Sequence[str],
    k_values: Sequence[int],
) -> Dict[int, Dict[int, Dict[str, np.ndarray]]]:
    counts, totals = compute_class_kmer_stats(train_df, train_freqs, classes, k_values)
    enrichment: Dict[int, Dict[int, Dict[str, np.ndarray]]] = {idx: {} for idx in range(len(classes))}
    for k in k_values:
        all_counts = np.zeros(4 ** k, dtype=np.float64)
        for idx in range(len(classes)):
            all_counts += counts[idx][k]
        all_total = sum(totals[idx][k] for idx in range(len(classes)))
        for idx in range(len(classes)):
            counts_c = counts[idx][k]
            total_c = totals[idx][k]
            counts_rest = all_counts - counts_c
            total_rest = all_total - total_c
            freq_c = (counts_c + 1.0) / max(1.0, total_c + counts_c.size)
            if total_rest <= 0:
                freq_rest = np.full_like(freq_c, freq_c.mean())
            else:
                freq_rest = (counts_rest + 1.0) / max(1.0, total_rest + counts_rest.size)
            log2fc = np.log2(np.divide(freq_c, freq_rest, out=np.zeros_like(freq_c), where=freq_rest > 0))
            enrichment[idx][k] = {
                "freq_class": freq_c,
                "freq_rest": freq_rest,
                "log2fc": log2fc,
            }
    return enrichment


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
        top = np.argpartition(-np.abs(log2fc), n - 1)[:n]
        selected.update(int(i) for i in top)
    return sorted(selected)


def build_feature_matrices(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_freqs: Dict[int, np.ndarray],
    test_freqs: Dict[int, np.ndarray],
    k: int,
    selected_indices: Sequence[int],
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    base_train = pd.DataFrame(
        {
            "gc_content": train_df["gc_content"].astype(float).to_numpy(),
            "n_fraction": train_df["n_fraction"].astype(float).to_numpy(),
        }
    )
    if selected_indices:
        kmer_matrix = train_freqs[k][:, selected_indices]
        for j, idx in enumerate(selected_indices):
            base_train[f"k{k}_{kmer_to_string(idx, k)}"] = kmer_matrix[:, j]

    if not test_df.empty:
        base_test = pd.DataFrame(
            {
                "gc_content": test_df["gc_content"].astype(float).to_numpy(),
                "n_fraction": test_df["n_fraction"].astype(float).to_numpy(),
            }
        )
        if selected_indices and test_freqs[k].shape[0] == len(test_df):
            kmer_matrix_test = test_freqs[k][:, selected_indices]
            for j, idx in enumerate(selected_indices):
                base_test[f"k{k}_{kmer_to_string(idx, k)}"] = kmer_matrix_test[:, j]
    else:
        base_test = pd.DataFrame(columns=base_train.columns)

    feature_names = list(base_train.columns)
    if not test_df.empty:
        for name in feature_names:
            if name not in base_test.columns:
                base_test[name] = 0.0
        base_test = base_test[feature_names]
    return base_train, base_test, feature_names


def run_cross_validated_logit(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_freqs: Dict[int, np.ndarray],
    test_freqs: Dict[int, np.ndarray],
    classes: Sequence[str],
    enrichment: Dict[int, Dict[int, Dict[str, np.ndarray]]],
    k_values: Sequence[int],
    top_kmer_features: int,
    cv_folds: int,
    logit_cs: Sequence[float],
    out_dir: Path,
) -> Dict[str, Dict[str, object]]:
    results: Dict[str, Dict[str, object]] = {}
    if train_df.empty:
        return results
    unique_classes = sorted(train_df["class_idx"].unique().tolist())
    if len(unique_classes) <= 1:
        print("[genome-stats] Logistic analysis skipped (only one class present)", flush=True)
        return results
    class_mapping = {orig: idx for idx, orig in enumerate(unique_classes)}
    inv_mapping = {idx: orig for orig, idx in class_mapping.items()}
    model_class_names = [classes[inv_mapping[idx]] for idx in range(len(unique_classes))]
    train_labels = train_df["class_idx"].map(class_mapping).to_numpy()
    test_labels = test_df["class_idx"].map(class_mapping).to_numpy() if not test_df.empty else None
    candidate_cs = sorted({float(c) for c in logit_cs if c > 0})
    if not candidate_cs:
        raise ValueError("At least one positive C value must be provided for logistic regression.")
    dataset_values = (
        test_df["dataset"].tolist() if "dataset" in test_df.columns else ["test"] * len(test_df)
    )

    for k in k_values:
        selected_indices = select_top_kmer_indices(enrichment, k, top_kmer_features)
        feature_train_df, feature_test_df, feature_names = build_feature_matrices(
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
        X_train = scaler.fit_transform(feature_train_df.to_numpy(dtype=float))
        clf = LogisticRegressionCV(
            Cs=candidate_cs,
            penalty="l1",
            solver="saga",
            cv=cv_folds,
            scoring="balanced_accuracy",
            class_weight="balanced",
            max_iter=5000,
            n_jobs=None,
            multi_class="multinomial",
            refit=True,
        )
        clf.fit(X_train, train_labels)

        bal_acc = None
        test_pred_path = None
        if test_labels is not None and not feature_test_df.empty and len(test_labels) == feature_test_df.shape[0]:
            X_test = scaler.transform(feature_test_df.to_numpy(dtype=float))
            probs = clf.predict_proba(X_test)
            y_pred = probs.argmax(axis=1)
            bal_acc = float(balanced_accuracy_score(test_labels, y_pred))
            pred_df = pd.DataFrame(
                {
                    "snippet_id": test_df["snippet_id"].tolist(),
                    "dataset": dataset_values,
                    "filename": test_df["filename"].tolist(),
                    "true_class_idx": test_df["class_idx"].tolist(),
                    "true_class_name": test_df["class_name"].tolist(),
                    "true_label_encoded": test_labels.tolist(),
                    "pred_label_encoded": y_pred.tolist(),
                    "pred_class_idx": [unique_classes[idx] for idx in y_pred],
                    "pred_class_name": [classes[unique_classes[idx]] for idx in y_pred],
                }
            )
            for class_idx, class_label in enumerate(model_class_names):
                pred_df[f"pred_prob_{safe_name(class_label)}"] = probs[:, class_idx]
            test_pred_path = out_dir / f"logit_test_predictions_k{k}.parquet"
            pred_df.to_parquet(test_pred_path, index=False)

        best_c = {model_class_names[idx]: float(c_val) for idx, c_val in enumerate(np.ravel(clf.C_))}
        coefficients: Dict[str, Dict[str, float]] = {}
        coef_matrix = clf.coef_
        intercepts = clf.intercept_
        if len(model_class_names) == 2 and coef_matrix.shape[0] == 1:
            coef_matrix = np.vstack([-coef_matrix[0], coef_matrix[0]])
            intercepts = np.array([-intercepts[0], intercepts[0]])
        for class_idx, class_label in enumerate(model_class_names):
            coef_map = {name: float(value) for name, value in zip(feature_names, coef_matrix[class_idx])}
            coef_map["intercept"] = float(intercepts[class_idx])
            coefficients[class_label] = coef_map

        cv_score_summary: Dict[str, float] = {}
        if hasattr(clf, "scores_") and clf.scores_:
            for c_idx, c_val in enumerate(candidate_cs):
                class_scores = []
                for cls_scores in clf.scores_.values():
                    if cls_scores.shape[0] > c_idx:
                        class_scores.append(float(np.mean(cls_scores[c_idx])))
                if class_scores:
                    cv_score_summary[str(c_val)] = float(np.mean(class_scores))

        summary_lines = [
            f"L1 multinomial logistic regression (k={k})",
            f"Train snippets: {len(train_labels)}",
            f"Classes: {', '.join(model_class_names)}",
            f"Candidate Cs: {', '.join(str(c) for c in candidate_cs)}",
            f"Best C per class: {', '.join(f'{cls}={c:.4g}' for cls, c in best_c.items())}",
            f"Test balanced accuracy: {bal_acc if bal_acc is not None else 'n/a'}",
        ]
        summary_path = out_dir / f"logit_summary_k{k}.txt"
        with open(summary_path, "w") as handle:
            handle.write("\n".join(summary_lines) + "\n")

        results[f"k{k}"] = {
            "selected_kmers": [name for name in feature_names if name.startswith(f"k{k}_")],
            "balanced_accuracy_test": bal_acc,
            "best_c_per_class": best_c,
            "cv_scores": cv_score_summary,
            "coefficients": coefficients,
            "class_labels": model_class_names,
            "summary_path": str(summary_path),
            "test_predictions": str(test_pred_path) if test_pred_path is not None else None,
        }
    return results


def build_summary(
    phenotype: str,
    slug: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    snippet_length: int,
    classes: Sequence[str],
    k_values: Sequence[int],
    gc_plot_path: Optional[Path],
    heatmap_paths: Dict[int, Path],
    logistic_results: Dict[str, Dict[str, object]],
    train_table_path: Path,
    test_table_path: Optional[Path],
    parameters: Dict[str, object],
) -> Dict[str, object]:
    summary: Dict[str, object] = {
        "phenotype": phenotype,
        "slug": slug,
        "snippet_length": int(snippet_length),
        "train_snippets": int(len(train_df)),
        "test_snippets": int(len(test_df)),
        "classes": list(classes),
        "per_class_snippets": train_df.groupby("class_name").size().to_dict(),
        "k_values": list(map(int, k_values)),
        "plots": {},
        "logistic_models": logistic_results,
        "train_table": str(train_table_path),
        "parameters": parameters,
    }
    if not test_df.empty and test_table_path is not None:
        summary["test_table"] = str(test_table_path)
    if gc_plot_path is not None:
        summary["plots"]["gc_content_distribution"] = str(gc_plot_path)
    summary["plots"]["kmer_heatmaps"] = {f"k{k}": str(path) for k, path in heatmap_paths.items()}
    return summary


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Sample 1 Mbp snippets for phenotype analysis.")
    ap.add_argument("--phenotype", type=str, default="Spore formation", help="Phenotype column to analyze.")
    ap.add_argument("--metadata", type=str, default="sporulation/microbe.cards table S1.xlsx", help="Path to metadata Excel file.")
    ap.add_argument("--data_root", type=str, default=str(DATA_ROOT), help="Root directory containing train/validation/test FASTA.")
    ap.add_argument("--k_values", type=str, default="3,4,6", help="Comma-separated list of k-mer sizes to profile.")
    ap.add_argument("--train_snippets", type=int, default=400, help="Number of 1 Mbp snippets to sample from train+validation.")
    ap.add_argument("--test_snippets", type=int, default=200, help="Number of 1 Mbp snippets to sample from test.")
    ap.add_argument("--snippet_length", type=int, default=1_000_000, help="Length of each sampled snippet (bp).")
    ap.add_argument("--seed", type=int, default=1337, help="Random seed for snippet sampling.")
    ap.add_argument("--top_kmer_features", type=int, default=64, help="Union of enriched k-mers to include as features.")
    ap.add_argument("--cv_folds", type=int, default=5, help="Number of folds for cross-validated logistic regression.")
    ap.add_argument("--logit_cs", type=str, default="0.1,0.5,1.0,2.0", help="Comma-separated grid of C values for LogisticRegressionCV.")
    ap.add_argument("--out_dir", type=str, default=None, help="Output directory (defaults to phenotype/plots/genome_stats/<slug>).")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    phenotype = args.phenotype.strip()
    slug = phenotype_to_slug(phenotype)
    data_root = Path(args.data_root).expanduser().resolve()

    train_dir = data_root / "train"
    val_dir = data_root / "validation"
    test_dir = data_root / "test"

    train_dirs = [d for d in (train_dir, val_dir) if d.is_dir()]
    if not train_dirs:
        raise FileNotFoundError(f"No training/validation directories found under {data_root}")

    try:
        metadata_df = pd.read_excel(args.metadata)
    except Exception as exc:
        raise RuntimeError(f"Failed to read metadata Excel at {args.metadata}: {exc}") from exc

    labels_map, classes = build_labels_map_and_classes(
        metadata_df,
        phenotype_col=phenotype,
        file_col="Fasta file",
        train_dirs=[str(d) for d in train_dirs],
    )
    if not classes:
        raise RuntimeError(f"No classes inferred for phenotype '{phenotype}'")

    k_values = [int(k.strip()) for k in args.k_values.split(",") if k.strip()]
    if not k_values:
        raise ValueError("At least one k-mer size must be specified.")
    print(f"[genome-stats] Profiling k-mer sizes: {k_values}", flush=True)

    train_files = gather_dataset_files(train_dirs, labels_map)
    train_df, train_freqs = sample_snippet_dataset(
        "train",
        train_files,
        classes,
        k_values,
        args.snippet_length,
        args.train_snippets,
        args.seed,
        verbose=True,
    )
    if train_df.empty:
        raise RuntimeError("No training snippets were sampled; check phenotype labels and data availability.")
    print(f"[genome-stats] Sampled {len(train_df)} training snippets.", flush=True)

    if test_dir.is_dir():
        test_files = gather_dataset_files([test_dir], labels_map)
        test_df, test_freqs = sample_snippet_dataset(
            "test",
            test_files,
            classes,
            k_values,
            args.snippet_length,
            args.test_snippets,
            args.seed + 17,
            verbose=True,
        )
        print(f"[genome-stats] Sampled {len(test_df)} test snippets.", flush=True)
    else:
        print(f"[genome-stats] Warning: test directory not found ({test_dir}); evaluation will be skipped.", flush=True)
        test_df = pd.DataFrame()
        test_freqs = {k: np.zeros((0, 4 ** k), dtype=np.float32) for k in k_values}

    enrichment = compute_enrichment_statistics(train_df, train_freqs, classes, k_values)

    out_dir = Path(args.out_dir) if args.out_dir else Path("phenotype/plots/genome_stats") / slug
    ensure_dir(out_dir)

    gc_plot_path = out_dir / "gc_content_distribution.png"
    plot_gc_content(train_df, classes, gc_plot_path)

    heatmap_paths: Dict[int, Path] = {}
    for k in k_values:
        freqs_for_heatmap: Dict[str, np.ndarray] = {}
        for idx, class_name in enumerate(classes):
            freq = enrichment.get(idx, {}).get(k, {}).get("freq_class")
            if freq is not None:
                freqs_for_heatmap[class_name] = freq
        if freqs_for_heatmap:
            path = out_dir / f"k{k}_frequency_heatmap.png"
            plot_kmer_heatmap(freqs_for_heatmap, k, path)
            heatmap_paths[k] = path

    logit_cs = [float(c.strip()) for c in args.logit_cs.split(",") if c.strip()]
    logits = run_cross_validated_logit(
        train_df,
        test_df,
        train_freqs,
        test_freqs,
        classes,
        enrichment,
        k_values,
        args.top_kmer_features,
        args.cv_folds,
        logit_cs,
        out_dir,
    )

    train_table_path = out_dir / "train_snippets.parquet"
    train_df.to_parquet(train_table_path, index=False)
    test_table_path: Optional[Path] = None
    if not test_df.empty:
        test_table_path = out_dir / "test_snippets.parquet"
        test_df.to_parquet(test_table_path, index=False)

    summary = build_summary(
        phenotype,
        slug,
        train_df,
        test_df,
        args.snippet_length,
        classes,
        k_values,
        gc_plot_path,
        heatmap_paths,
        logits,
        train_table_path,
        test_table_path,
        parameters={
            "train_dirs": [str(d) for d in train_dirs],
            "test_dir": str(test_dir) if test_dir.exists() else None,
            "train_snippets_requested": args.train_snippets,
            "test_snippets_requested": args.test_snippets,
            "cv_folds": args.cv_folds,
            "logit_cs": logit_cs,
            "top_kmer_features": args.top_kmer_features,
        },
    )

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Analysis complete for phenotype '{phenotype}'. Summary written to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
