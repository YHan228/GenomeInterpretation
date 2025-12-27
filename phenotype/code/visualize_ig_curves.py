#!/usr/bin/env python3
"""Visualize ground-truth masks and IG attribution curves for standard vs. robust models."""

import argparse
import json
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import colors as mcolors  # noqa: E402
import numpy as np
import pandas as pd
import torch

from torch.cuda.amp import autocast

from model import SporulationModel
from phenotype_utils import normalize_label_value

from evaluation import (
    DEFAULT_DEVICE,
    DEFAULT_EVAL_DIR,
    DEFAULT_IG_STEPS,
    DEFAULT_PGD_EPS,
    DEFAULT_PGD_STEP_SIZE,
    DEFAULT_PGD_STEPS,
    DEFAULT_TEST_DIR,
    _FASTA,
    _interval_mask,
    compute_ig_attributions,
    correct_gradients,
    find_adversarial_baseline_pgd_batch,
    one_hot_encode,
)


def _load_manifest_and_samples(
    eval_dir: str,
    test_dir_override: Optional[str],
    target_class_override: Optional[str],
) -> Tuple[pd.DataFrame, Dict, str, str, int, List[str], Dict[str, int]]:
    samples_path = os.path.join(eval_dir, "samples.parquet")
    manifest_path = os.path.join(eval_dir, "eval_manifest.json")
    if not os.path.exists(samples_path):
        raise FileNotFoundError(f"samples.parquet not found under {eval_dir}; run eval_data.py first")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"eval_manifest.json not found under {eval_dir}; run eval_data.py first")

    samples = pd.read_parquet(samples_path)
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    seq_len = int(manifest.get("seq_len", 1_000_000))
    manifest_test_dirs = manifest.get("test_dirs") or []
    derived_test_dir = manifest_test_dirs[0] if manifest_test_dirs else DEFAULT_TEST_DIR
    test_dir = os.path.abspath(test_dir_override or derived_test_dir)

    classes = manifest.get("classes")
    class_to_id_manifest = manifest.get("class_to_id")
    if isinstance(class_to_id_manifest, dict) and class_to_id_manifest:
        mapped = {normalize_label_value(k): int(v) for k, v in class_to_id_manifest.items()}
        max_idx = max(mapped.values())
        classes = [None] * (max_idx + 1)
        for name, idx in mapped.items():
            if idx >= len(classes):
                classes.extend([None] * (idx - len(classes) + 1))
            classes[idx] = name
        for i, v in enumerate(classes):
            if v is None:
                classes[i] = f"class_{i}"
        class_to_id = {name: idx for idx, name in enumerate(classes)}
    else:
        if classes is None:
            label_col = samples.get("label")
            if label_col is not None and len(label_col.dropna()) > 0:
                unique_labels = sorted({int(x) for x in label_col.dropna().unique().tolist()})
            else:
                unique_labels = [0, 1]
            classes = [str(u) for u in unique_labels]
        classes = [normalize_label_value(c) for c in classes]
        class_to_id = {c: idx for idx, c in enumerate(classes)}

    if "label" not in samples.columns:
        if "label_name" in samples.columns:
            samples["label"] = samples["label_name"].map(
                lambda name: class_to_id.get(normalize_label_value(name), 0)
            ).astype(int)
        else:
            raise ValueError("samples.parquet missing 'label' column; regenerate eval data")
    samples["label"] = samples["label"].astype(int)

    if "label_name" not in samples.columns:
        samples["label_name"] = samples["label"].map(lambda idx: classes[idx] if 0 <= idx < len(classes) else str(idx))

    target_class = target_class_override or manifest.get("target_class")
    if target_class is not None:
        target_class = normalize_label_value(target_class)
    else:
        if "true" in class_to_id:
            target_class = "true"
        elif classes:
            target_class = classes[0]
        else:
            raise RuntimeError("Unable to determine target class from manifest or samples")

    if target_class not in class_to_id:
        raise ValueError(f"Target class '{target_class}' not found in classes {classes}")
    target_class_id = int(class_to_id[target_class])

    if "target_mask" not in samples.columns:
        samples["target_mask"] = samples["label"] == target_class_id

    return samples, manifest, test_dir, target_class, target_class_id, classes, class_to_id


def _load_model(model_path: str, num_classes: int, device: torch.device) -> SporulationModel:
    params = None
    meta_path_guess = os.path.join(os.path.dirname(os.path.abspath(model_path)), "best_meta.json")
    if os.path.exists(meta_path_guess):
        try:
            with open(meta_path_guess, "r") as f:
                meta_obj = json.load(f)
            if isinstance(meta_obj, dict) and isinstance(meta_obj.get("params"), dict):
                params = meta_obj["params"]
        except Exception:
            params = None
    model = SporulationModel(num_classes=num_classes, params=params).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def _compute_single_ig(
    model: SporulationModel,
    xb: torch.Tensor,
    target_class_id: int,
    device: torch.device,
    ig_steps: int,
    pgd_eps: float,
    pgd_steps: int,
    pgd_step_size: Optional[float],
) -> Tuple[np.ndarray, Dict]:
    xb_batch = xb.unsqueeze(0).to(device)
    yb = torch.full((1,), target_class_id, dtype=torch.long, device=device)
    baselines, stats = find_adversarial_baseline_pgd_batch(
        model,
        xb_batch,
        yb,
        device,
        num_iter=int(pgd_steps),
        epsilon=float(pgd_eps),
        step_size=(None if pgd_step_size is None else float(pgd_step_size)),
    )
    comp_baselines = xb_batch.mean(dim=1, keepdim=True).expand_as(xb_batch)
    use_baselines = torch.where(
        (baselines.abs().sum(dim=(1, 2)) > 0).view(-1, 1, 1),
        baselines,
        comp_baselines,
    )
    with autocast():
        raw_attr = compute_ig_attributions(model, xb_batch, use_baselines, steps=int(ig_steps))
    corr_attr = correct_gradients(raw_attr)
    contrib = (corr_attr * xb_batch).sum(dim=1).abs()[0].detach().cpu().numpy()
    return contrib, stats[0]


def _array_split_indices(length: int, max_points: int) -> List[np.ndarray]:
    if length <= max_points:
        return [np.arange(length)]
    bins = np.array_split(np.arange(length), max_points)
    return [b for b in bins if b.size > 0]


def _aggregate_curves(
    arrays: Sequence[np.ndarray],
    chunk_size: int,
    max_points: int,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    if not arrays:
        return np.array([]), []
    length = arrays[0].shape[0]
    if any(arr.shape[0] != length for arr in arrays):
        raise ValueError("All arrays must share the same length for aggregation")
    chunk = max(1, int(chunk_size))
    primary_bins = [
        np.arange(start, min(start + chunk, length))
        for start in range(0, length, chunk)
    ]
    x = np.array([b.mean() for b in primary_bins])
    aggregated: List[np.ndarray] = []
    for arr in arrays:
        aggregated.append(np.array([arr[b].mean() for b in primary_bins]))

    limit = max(1, int(max_points))
    if x.size > limit:
        bins = _array_split_indices(x.size, limit)
        x = np.array([x[b].mean() for b in bins])
        aggregated = [np.array([arr[b].mean() for b in bins]) for arr in aggregated]

    return x, aggregated


def _normalize_for_visual(
    curve: np.ndarray,
    percentile: float,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, float]:
    """Scale curve by the chosen percentile of |values| to minimize visual bias."""
    if curve.size == 0:
        return curve.copy(), 1.0
    abs_curve = np.abs(curve)
    scale = float(np.percentile(abs_curve, percentile))
    if not np.isfinite(scale) or scale <= eps:
        fallback = float(abs_curve.max(initial=0.0))
        scale = fallback if fallback > eps else 1.0
    return curve / scale, scale


def _plot_sample(
    sample_id: int,
    sample_row: pd.Series,
    std_curve: np.ndarray,
    rob_curve: np.ndarray,
    mask: np.ndarray,
    out_dir: str,
    max_points: int,
    smooth_bp: int,
    percentile: float,
    std_scale: float,
    rob_scale: float,
) -> str:
    x, [std_ds, rob_ds, mask_ds] = _aggregate_curves(
        [std_curve, rob_curve, mask.astype(np.float32)],
        smooth_bp,
        max_points,
    )

    fig, ax_curve = plt.subplots(1, 1, figsize=(12, 4))

    ymin = float(np.min(np.concatenate([std_ds, rob_ds])))
    ymax = float(np.max(np.concatenate([std_ds, rob_ds])))
    if not np.isfinite(ymin):
        ymin = 0.0
    if not np.isfinite(ymax):
        ymax = 1.0
    if abs(ymax - ymin) < 1e-9:
        span = 1.0
    else:
        span = ymax - ymin
    pad = 0.05 * span
    lower = ymin - pad
    upper = ymax + pad

    if mask_ds.size > 0 and np.max(mask_ds) > 0:
        base_rgba = mcolors.to_rgba("tab:green")
        norm_mask = mask_ds / mask_ds.max()
        colors = [
            (base_rgba[0], base_rgba[1], base_rgba[2], float(0.35 * val))
            for val in norm_mask
        ]
        edges = np.empty(x.size + 1, dtype=np.float64)
        if x.size <= 1:
            half_width = smooth_bp / 2.0
            center = float(x[0]) if x.size == 1 else 0.0
            edges[0] = center - half_width
            edges[1] = center + half_width
        else:
            edges[1:-1] = 0.5 * (x[:-1] + x[1:])
            edges[0] = x[0] - smooth_bp / 2.0
            edges[-1] = x[-1] + smooth_bp / 2.0
        widths = np.diff(edges)
        mask_height = upper - lower
        ax_curve.bar(
            x,
            mask_height,
            width=widths,
            bottom=lower,
            align="center",
            color=colors,
            edgecolor="none",
            linewidth=0,
        )
    legend_std = f"Standard model (|IG| P{percentile:.1f}={std_scale:.3g})"
    legend_rob = f"Robust model (|IG| P{percentile:.1f}={rob_scale:.3g})"
    ax_curve.plot(
        x,
        std_ds,
        label=legend_std,
        color="tab:blue",
        linewidth=1.0,
        alpha=0.75,
    )
    ax_curve.plot(
        x,
        rob_ds,
        label=legend_rob,
        color="tab:orange",
        linewidth=1.0,
        alpha=0.75,
    )
    ax_curve.set_ylim(lower, upper)
    if x.size > 0:
        ax_curve.set_xlim(float(x[0]), float(x[-1]))
    ax_curve.set_title(
        f"Sample {sample_id} | {sample_row['fasta_filename']}:{sample_row['seqid']} "
        f"[{int(sample_row['start'])}-{int(sample_row['end'])}]"
    )
    ax_curve.set_ylabel("IG contribution (|grad * input|)")
    ax_curve.set_xlabel("Position (bp)")
    ax_curve.legend(loc="upper right")

    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    base_name = os.path.join(out_dir, f"sample_{sample_id}")
    # Save in multiple formats
    for ext in [".png", ".pdf", ".svg"]:
        fig.savefig(base_name + ext, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return base_name + ".png"


def _parse_sample_ids(values: Optional[Iterable[str]]) -> List[int]:
    if not values:
        return []
    ids: List[int] = []
    for val in values:
        parts = str(val).replace(":", ",").split(",")
        for part in parts:
            part = part.strip()
            if not part:
                continue
            ids.append(int(part))
    return ids


def main() -> None:
    ap = argparse.ArgumentParser(description="Visualize IG curves for standard vs robust models")
    ap.add_argument("--eval_dir", type=str, default=DEFAULT_EVAL_DIR, help="Directory with eval_manifest.json and samples.parquet")
    ap.add_argument("--test_dir", type=str, default=None, help="Override FASTA directory (defaults to manifest entry)")
    ap.add_argument("--standard_model", type=str, default="phenotype/model/spore_formation/best_model.pth", help="Path to standard model checkpoint")
    ap.add_argument("--robust_model", type=str, default=None, help="Path to robust model checkpoint (overrides --robust_study if set)")
    ap.add_argument(
        "--robust_study",
        type=str,
        default="50epochs_linear_spore_formation",
        help="Study folder under phenotype/robust_model to select robust checkpoint",
    )
    ap.add_argument("--target_class", type=str, default=None, help="Override target class name")
    ap.add_argument("--num_samples", type=int, default=4, help="Number of target-class samples to visualize (ignored if sample ids provided)")
    ap.add_argument("--sample_id", action="append", help="Specific sample_id(s) to visualize; can be repeated or comma-separated")
    ap.add_argument("--ig_steps", type=int, default=DEFAULT_IG_STEPS, help="Number of IG steps along the path")
    ap.add_argument("--pgd_eps", type=float, default=DEFAULT_PGD_EPS, help="PGD epsilon for baseline search")
    ap.add_argument("--pgd_steps", type=int, default=DEFAULT_PGD_STEPS, help="PGD steps for baseline search")
    ap.add_argument("--pgd_step_size", type=float, default=DEFAULT_PGD_STEP_SIZE, help="Optional PGD step size (defaults to epsilon/10)")
    ap.add_argument("--max_points", type=int, default=4000, help="Maximum points when plotting (downsample via mean if longer)")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for sample selection")
    ap.add_argument(
        "--normalization_percentile",
        type=float,
        default=99.0,
        help="Percentile of |IG| used to rescale curves for fair visual comparison",
    )
    ap.add_argument(
        "--smooth_bp",
        type=int,
        default=100,
        help="Window size (bp) for averaging curves before optional downsampling",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="phenotype/plots/ig_curves",
        help="Parent directory for plots; a study-specific subfolder will be used",
    )
    ap.add_argument("--device", type=str, default=DEFAULT_DEVICE, help="Device for model inference (cuda/cpu)")
    args = ap.parse_args()

    device = torch.device(args.device)

    samples, manifest, test_dir, target_class, target_class_id, classes, class_to_id = _load_manifest_and_samples(
        args.eval_dir,
        args.test_dir,
        args.target_class,
    )
    seq_len = int(manifest.get("seq_len", 1_000_000))

    std_model = _load_model(args.standard_model, num_classes=max(2, len(classes)), device=device)
    robust_model_path = (
        args.robust_model
        if args.robust_model
        else os.path.join("phenotype", "robust_model", args.robust_study, "best_model.pth")
    )
    rob_model = _load_model(robust_model_path, num_classes=max(2, len(classes)), device=device)

    # Use a study-specific subfolder under the chosen out_dir to avoid overwriting
    study_out_dir = os.path.join(args.out_dir, args.robust_study)

    requested_ids = _parse_sample_ids(args.sample_id)
    rng = np.random.default_rng(int(args.seed))

    if requested_ids:
        target_rows = samples[samples["sample_id"].isin(requested_ids)]
        missing = sorted(set(requested_ids) - set(target_rows["sample_id"].tolist()))
        if missing:
            raise ValueError(f"Requested sample_id(s) not found: {missing}")
    else:
        target_rows = samples[samples["target_mask"]].copy()
        if target_rows.empty:
            raise RuntimeError("No positive samples (target_mask) available for visualization")
        if len(target_rows) > args.num_samples:
            idx = rng.choice(target_rows.index.values, size=args.num_samples, replace=False)
            target_rows = target_rows.loc[idx]

    out_records: List[Dict] = []

    for _, row in target_rows.iterrows():
        sample_id = int(row["sample_id"])
        fasta_path = row["fasta_filename"]
        if not os.path.isabs(fasta_path):
            fasta_path = os.path.join(test_dir, fasta_path)
        seq = _FASTA.fetch_window(str(fasta_path), str(row["seqid"]), int(row["start"]), int(row["end"]))
        if len(seq) < seq_len:
            seq = seq.ljust(seq_len, "N")
        xb = one_hot_encode(seq).to(device)

        try:
            intervals = json.loads(row.get("intervals_json", "[]")) if pd.notna(row.get("intervals_json", None)) else []
        except Exception:
            intervals = []
        intervals = [(int(a), int(b)) for a, b in intervals]
        mask = _interval_mask(intervals, xb.shape[1]).astype(np.float32)

        std_curve, std_stats = _compute_single_ig(
            std_model,
            xb,
            target_class_id,
            device,
            args.ig_steps,
            args.pgd_eps,
            args.pgd_steps,
            args.pgd_step_size,
        )
        rob_curve, rob_stats = _compute_single_ig(
            rob_model,
            xb,
            target_class_id,
            device,
            args.ig_steps,
            args.pgd_eps,
            args.pgd_steps,
            args.pgd_step_size,
        )

        std_disp, std_scale = _normalize_for_visual(std_curve, args.normalization_percentile)
        rob_disp, rob_scale = _normalize_for_visual(rob_curve, args.normalization_percentile)

        plot_path = _plot_sample(
            sample_id,
            row,
            std_disp,
            rob_disp,
            mask,
            study_out_dir,
            max(1, int(args.max_points)),
            max(1, int(args.smooth_bp)),
            float(args.normalization_percentile),
            std_scale,
            rob_scale,
        )

        out_records.append(
            {
                "sample_id": sample_id,
                "fasta_filename": row["fasta_filename"],
                "seqid": row["seqid"],
                "start": int(row["start"]),
                "end": int(row["end"]),
                "plot_path": plot_path,
                "visual_scale_percentile": float(args.normalization_percentile),
                "standard_scale_value": float(std_scale),
                "robust_scale_value": float(rob_scale),
                "smooth_bp": int(args.smooth_bp),
                "pgd_success_std": bool(std_stats.get("success", False)),
                "pgd_success_rob": bool(rob_stats.get("success", False)),
                "pgd_iters_std": int(std_stats.get("found_at_iter", 0)),
                "pgd_iters_rob": int(rob_stats.get("found_at_iter", 0)),
                "pgd_initial_logit_std": float(std_stats.get("initial_logit", 0.0)),
                "pgd_initial_logit_rob": float(rob_stats.get("initial_logit", 0.0)),
                "pgd_final_logit_std": float(std_stats.get("final_logit", 0.0)),
                "pgd_final_logit_rob": float(rob_stats.get("final_logit", 0.0)),
            }
        )
        print(
            f"Generated plot for sample {sample_id} -> {plot_path} | "
            f"baseline success (std={std_stats.get('success')}, rob={rob_stats.get('success')}) | "
            f"scales (std={std_scale:.3g}, rob={rob_scale:.3g}) | "
            f"smooth={int(args.smooth_bp)}bp",
            flush=True,
        )

    if out_records:
        summary_path = os.path.join(study_out_dir, "ig_visualization_summary.json")
        with open(summary_path, "w") as f:
            json.dump(out_records, f, indent=2)
        print(f"Wrote summary metadata to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
