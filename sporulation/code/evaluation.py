import argparse
import json
import os
import gzip
from functools import lru_cache
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from model import SporulationModel

# --------------------------- Editable defaults ---------------------------

DEFAULT_EVAL_DIR = "/vol/projects/BIFO/genomenet/yichen/phenotype/data/eval"
DEFAULT_TEST_DIR = "/vol/projects/BIFO/genomenet/yichen/phenotype/data/test"
DEFAULT_MODEL_PATH = "sporulation/model/best_sporulation_model.pth"
DEFAULT_ACC_BS = 16
DEFAULT_IG_BS = 1
DEFAULT_PGD_EPS = 0.1
DEFAULT_PGD_STEPS = 20
DEFAULT_PGD_STEP_SIZE = None
DEFAULT_SAUC_OUTSIDE_SUBSAMPLE = 200000
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_OUT_DIR = None
DEFAULT_IG_STEPS = 32
DEFAULT_TILE_LEN = None
DEFAULT_TILE_STRIDE = None
DEFAULT_TILE_AGG = "mean"

# Optional FASTA index access
try:
    from pyfaidx import Fasta  # type: ignore
    _PYFAIDX_AVAILABLE = True
except Exception:
    _PYFAIDX_AVAILABLE = False


# --------------------------- One-hot utilities ---------------------------

_ALPH = np.array(list("ACGT"), dtype="U1")
_to_ix = {b: i for i, b in enumerate(_ALPH)}


class _FastaAccessor:
    """Random access FASTA with optional pyfaidx and LRU contig cache fallback."""
    def __init__(self):
        # Use loose typing to avoid NameError when pyfaidx is not installed
        self._handles: Dict[str, object] = {}

    def _get_handle(self, fasta_path: str) -> Optional[object]:
        if not _PYFAIDX_AVAILABLE:
            return None
        h = self._handles.get(fasta_path)
        if h is not None:
            return h
        try:
            h = Fasta(fasta_path, as_raw=True, sequence_always_upper=True)
            self._handles[fasta_path] = h
            return h
        except Exception:
            return None

    @lru_cache(maxsize=64)
    def _contig_sequence(self, fasta_path: str, seqid: str) -> str:
        # Fallback: cache entire contig in memory from a linear scan once
        current = None
        chunks: List[str] = []
        # Support gz if pyfaidx is unavailable
        opener = gzip.open if fasta_path.endswith('.gz') else open
        with opener(fasta_path, "rt") as f:
            for line in f:
                if line.startswith(">"):
                    header = line[1:].strip().split()[0]
                    current = header
                    continue
                if current == seqid:
                    s = line.strip().upper()
                    if s:
                        chunks.append(s)
        return "".join(chunks)

    def fetch_window(self, fasta_path: str, seqid: str, start_1b: int, end_1b: int) -> str:
        if start_1b > end_1b:
            return ""
        # Try pyfaidx for O(1) slicing
        h = self._get_handle(fasta_path)
        if h is not None and seqid in h:
            try:
                # pyfaidx is 1-based inclusive indexing
                return str(h[seqid][start_1b:end_1b])
            except Exception:
                pass
        # Fallback to cached contig sequence
        full = self._contig_sequence(fasta_path, seqid)
        s0 = max(0, int(start_1b) - 1)
        e0 = min(len(full), int(end_1b))
        return full[s0:e0]


_FASTA = _FastaAccessor()


def one_hot_encode(seq: str) -> torch.Tensor:
    # Vectorized over ASCII codes for speed
    b = np.frombuffer(seq.encode('ascii', errors='ignore'), dtype=np.uint8)
    L = b.size
    out = np.zeros((4, L), dtype=np.float32)
    if L:
        out[0, b == 65] = 1.0  # 'A'
        out[1, b == 67] = 1.0  # 'C'
        out[2, b == 71] = 1.0  # 'G'
        out[3, b == 84] = 1.0  # 'T'
    return torch.from_numpy(out)


# --------------------------- PGD baseline (batched) ---------------------------

def find_adversarial_baseline_pgd_batch(model: nn.Module,
                                        xb_batch: torch.Tensor,
                                        yb_batch: torch.Tensor,
                                        dev: torch.device,
                                        num_iter: int = 20,
                                        epsilon: float = 0.1,
                                        step_size: Optional[float] = None) -> Tuple[torch.Tensor, List[Dict]]:
    """Vectorized PGD baseline finder adapted from toy_single_arch.py.

    Returns (baselines, stats_list). For samples where attack not needed/unsuccessful,
    the baseline is a zero-tensor of same shape.
    """
    batch_size = xb_batch.shape[0]
    adv_xb_batch = xb_batch.clone().detach()

    with torch.no_grad(), autocast():
        initial_logits = model(adv_xb_batch)
        # model outputs logits for 2 classes; take class-1 score for positives
        # For baseline attack we consider binary: y=1 needs to be flipped
        initial_pred_classes = torch.argmax(initial_logits, dim=1).long()

    is_correct = (initial_pred_classes == yb_batch)
    is_positive = (yb_batch == 1)
    active_mask = is_correct & is_positive

    if not active_mask.any():
        stats_list = []
        for i in range(batch_size):
            stats_list.append({
                'success': False,
                'initial_logit': float(initial_logits[i, 1].item()),
                'final_logit': float(initial_logits[i, 1].item()),
                'found_at_iter': num_iter,
                'initial_prediction_correct': bool(is_correct[i].item()),
            })
        return torch.zeros_like(xb_batch, device=dev), stats_list

    loss_fn = nn.CrossEntropyLoss(reduction='none')
    step_sz = float(epsilon) / 10.0 if (step_size is None or step_size <= 0) else float(step_size)

    success_mask = torch.zeros(batch_size, dtype=torch.bool, device=dev)
    success_iter = torch.full((batch_size,), int(num_iter), dtype=torch.long, device=dev)
    final_baselines = torch.zeros_like(xb_batch, device=dev)

    for iter_idx in range(num_iter):
        if not active_mask.any():
            break
        active_xb = adv_xb_batch[active_mask].detach().requires_grad_(True)
        active_labels = yb_batch[active_mask].long()
        with autocast():
            active_logits = model(active_xb)
            losses = loss_fn(active_logits, active_labels)
            loss = losses.mean()
        model.zero_grad()
        loss.backward()
        with torch.no_grad():
            grad_sign = active_xb.grad.sign()
            active_xb_new = active_xb + step_sz * grad_sign
            active_indices = torch.where(active_mask)[0]
            for j, idx in enumerate(active_indices):
                delta = active_xb_new[j] - xb_batch[idx]
                delta = torch.clamp(delta, -epsilon, epsilon)
                adv_xb_batch[idx] = torch.clamp(xb_batch[idx] + delta, 0, 1)
            # check flips
            current_logits = model(adv_xb_batch[active_mask])
            current_pred_classes = torch.argmax(current_logits, dim=1).float()
            flip_occurred = (current_pred_classes != initial_pred_classes[active_mask])
            for j, idx in enumerate(active_indices):
                if flip_occurred[j] and not success_mask[idx]:
                    success_mask[idx] = True
                    success_iter[idx] = iter_idx + 1
                    final_baselines[idx] = adv_xb_batch[idx].clone()
                    active_mask[idx] = False

    with torch.no_grad():
        final_logits = model(adv_xb_batch)

    stats_list: List[Dict] = []
    for i in range(batch_size):
        stats_list.append({
            'success': bool(success_mask[i].item()),
            'initial_logit': float(initial_logits[i, 1].item()),
            'final_logit': float(final_logits[i, 1].item()),
            'found_at_iter': int(success_iter[i].item()),
            'initial_prediction_correct': bool(is_correct[i].item()),
        })
    return final_baselines, stats_list


def compute_ig_attributions(model: nn.Module,
                            xb_batch: torch.Tensor,
                            baseline_batch: torch.Tensor,
                            steps: int = 32) -> torch.Tensor:
    """Memory-efficient IG: iterate per step without retaining a large graph."""
    grads_accum = torch.zeros_like(xb_batch)
    delta = xb_batch - baseline_batch
    for alpha in torch.linspace(0, 1, steps=steps, device=xb_batch.device, dtype=xb_batch.dtype):
        x_s = (baseline_batch + alpha * delta).detach().requires_grad_(True)
        with autocast():
            logits = model(x_s)
            target = logits[:, 1]
        grad = torch.autograd.grad(target.sum(), x_s, retain_graph=False)[0]
        grads_accum += grad
        del x_s, grad, logits, target
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    avg_grads = grads_accum / float(steps)
    ig = delta * avg_grads
    return ig.detach()


def correct_gradients(raw_attr: torch.Tensor) -> torch.Tensor:
    # Channel-wise mean removal
    return raw_attr - raw_attr.mean(dim=1, keepdim=True)


def _rankdata_average(x: np.ndarray) -> np.ndarray:
    """Average ranks for ties, 1-based ranks like scipy.stats.rankdata(method='average')."""
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


def saliency_auc_from_mask(attr: np.ndarray, intervals: List[Tuple[int, int]], outside_subsample: Optional[int] = None, rng: Optional[np.random.Generator] = None) -> float:
    """Compute SaAUC via rank-based AUROC (Mann-Whitney), optional subsampling outside.

    intervals are 0-based inclusive within window. attr is length L vector of scores.
    """
    L = attr.shape[0]
    mask = np.zeros(L, dtype=bool)
    for s, e in intervals:
        s0 = max(0, int(s))
        e0 = min(L - 1, int(e))
        if s0 <= e0:
            mask[s0 : e0 + 1] = True
    inside = attr[mask]
    outside = attr[~mask]
    n_pos = inside.size
    n_neg = outside.size
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    if outside_subsample is not None and n_neg > outside_subsample:
        rng = rng or np.random.default_rng(0)
        idx = rng.choice(n_neg, size=outside_subsample, replace=False)
        outside = outside[idx]
        n_neg = outside.size
    # Rank all
    all_vals = np.concatenate([inside, outside])
    ranks = _rankdata_average(all_vals)
    ranks_pos = ranks[:n_pos].sum()
    auc = (ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def batch_iter_indices(n: int, batch_size: int):
    i = 0
    while i < n:
        j = min(n, i + batch_size)
        yield i, j
        i = j


def main():
    ap = argparse.ArgumentParser(description="Evaluate sporulation model: accuracy and SaAUC (PGD+IG)")
    ap.add_argument("--eval_dir", type=str, default=DEFAULT_EVAL_DIR, help="Directory produced by eval_data.py")
    ap.add_argument("--test_dir", type=str, default=DEFAULT_TEST_DIR, help="FASTA directory for reading windows")
    ap.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH, help="Path to model .pth file")
    ap.add_argument("--acc_batch_size", type=int, default=DEFAULT_ACC_BS, help="Batch size for accuracy passes")
    ap.add_argument("--ig_batch_size", type=int, default=DEFAULT_IG_BS, help="Batch size for IG/PGD (use 1-2 for 1e6bp)")
    ap.add_argument("--pgd_epsilon", type=float, default=DEFAULT_PGD_EPS, help="PGD epsilon for baseline")
    ap.add_argument("--pgd_steps", type=int, default=DEFAULT_PGD_STEPS, help="PGD steps for baseline")
    ap.add_argument("--pgd_step_size", type=float, default=DEFAULT_PGD_STEP_SIZE, help="Optional PGD step size (default epsilon/10)")
    ap.add_argument("--sauc_outside_subsample", type=int, default=DEFAULT_SAUC_OUTSIDE_SUBSAMPLE, help="Optional subsample size for outside positions in SaAUC to reduce memory/time; set 0 to disable")
    ap.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    ap.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR, help="Output directory for predictions/sauc/metrics; default eval_dir")
    ap.add_argument("--ig_steps", type=int, default=DEFAULT_IG_STEPS, help="Number of IG steps along the path")
    ap.add_argument("--tile_len", type=int, default=DEFAULT_TILE_LEN, help="Optional tile length for forward passes (accuracy only)")
    ap.add_argument("--tile_stride", type=int, default=DEFAULT_TILE_STRIDE, help="Optional tile stride for forward passes (accuracy only)")
    ap.add_argument("--tile_agg", type=str, default=DEFAULT_TILE_AGG, choices=["mean", "logsumexp"], help="Aggregation over tile logits")
    args = ap.parse_args()

    dev = torch.device(args.device)

    # Load prepared samples
    manifest_path = os.path.join(args.eval_dir, "eval_manifest.json")
    samples_path = os.path.join(args.eval_dir, "samples.parquet")
    if not os.path.exists(samples_path):
        raise FileNotFoundError("samples.parquet not found; run eval_data.py first")
    samples = pd.read_parquet(samples_path)
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    seq_len = int(manifest.get("seq_len", 1_000_000))

    # Load model
    model = SporulationModel().to(dev)
    state = torch.load(args.model_path, map_location=dev)
    model.load_state_dict(state)
    model.eval()

    # Helper: tiled forward for a single one-hot (4,L)
    def _forward_tiled(one_hot: torch.Tensor) -> torch.Tensor:
        if args.tile_len is None or args.tile_len <= 0:
            with torch.no_grad(), autocast():
                return model(one_hot.unsqueeze(0))[0].squeeze(0)
        L = one_hot.shape[1]
        T = int(args.tile_len)
        S = int(args.tile_stride or args.tile_len)
        if T <= 0 or S <= 0:
            with torch.no_grad(), autocast():
                return model(one_hot.unsqueeze(0))[0].squeeze(0)
        starts = list(range(0, max(1, L - T + 1), S))
        if not starts:
            starts = [0]
        tiles: List[torch.Tensor] = []
        for st in starts:
            ed = min(L, st + T)
            tile = one_hot[:, st:ed]
            if tile.shape[1] < T:
                pad = torch.zeros((4, T - tile.shape[1]), dtype=tile.dtype, device=tile.device)
                tile = torch.cat([tile, pad], dim=1)
            tiles.append(tile)
        xb_tiles = torch.stack(tiles, dim=0).to(dev)  # (n_tiles, 4, T)
        with torch.no_grad(), autocast():
            logits_tiles = model(xb_tiles)  # (n_tiles, 2)
        if args.tile_agg == "logsumexp":
            agg = torch.logsumexp(logits_tiles, dim=0)
        else:
            agg = logits_tiles.mean(dim=0)
        return agg

    # Accuracy evaluation (all 5000)
    preds = []
    labels = []
    prob_pos = []
    logit_pos = []
    if args.tile_len is None or args.tile_len <= 0:
        for bi, (s, e) in enumerate(batch_iter_indices(len(samples), args.acc_batch_size)):
            sub = samples.iloc[s:e]
            x_list: List[torch.Tensor] = []
            y_list: List[int] = []
            for _, r in sub.iterrows():
                fasta_path = os.path.join(args.test_dir, r["fasta_filename"]) if not os.path.isabs(r["fasta_filename"]) else r["fasta_filename"]
                seq = _FASTA.fetch_window(fasta_path, str(r["seqid"]), int(r["start"]), int(r["end"]))
                if len(seq) < seq_len:
                    seq = seq.ljust(seq_len, "N")
                x_list.append(one_hot_encode(seq))
                y_list.append(int(r["label"]))
            xb = torch.stack(x_list).to(dev)
            yb = torch.tensor(y_list, dtype=torch.long, device=dev)
            with torch.no_grad(), autocast():
                logits = model(xb)
                p = torch.argmax(logits, dim=1)
                prob = F.softmax(logits, dim=1)[:, 1]
            logit_pos.extend(logits[:, 1].detach().cpu().numpy().tolist())
            preds.extend(p.cpu().numpy().tolist())
            labels.extend(yb.cpu().numpy().tolist())
            prob_pos.extend(prob.detach().cpu().numpy().tolist())
    else:
        # Tiled path: iterate sample-wise
        for _, r in samples.iterrows():
            fasta_path = os.path.join(args.test_dir, r["fasta_filename"]) if not os.path.isabs(r["fasta_filename"]) else r["fasta_filename"]
            seq = _FASTA.fetch_window(fasta_path, str(r["seqid"]), int(r["start"]), int(r["end"]))
            if len(seq) < seq_len:
                seq = seq.ljust(seq_len, "N")
            x = one_hot_encode(seq).to(dev)
            logits = _forward_tiled(x)
            p = int(torch.argmax(logits).item())
            pr = float(F.softmax(logits.unsqueeze(0), dim=1)[0, 1].item())
            preds.append(p)
            labels.append(int(r["label"]))
            prob_pos.append(pr)
            logit_pos.append(float(logits[1].item()))

    preds = np.array(preds, dtype=int)
    labels = np.array(labels, dtype=int)
    acc = float((preds == labels).mean()) if labels.size else 0.0
    print(f"Accuracy over all samples: {acc:.4f}")
    # Persist predictions
    out_dir = args.out_dir or args.eval_dir
    os.makedirs(out_dir, exist_ok=True)
    pred_df = samples[["sample_id"]].copy()
    pred_df["label"] = labels
    pred_df["pred"] = preds
    pred_df["prob_pos"] = prob_pos
    pred_df["logit_pos"] = logit_pos
    pred_path = os.path.join(out_dir, "predictions.parquet")
    pred_df.to_parquet(pred_path, index=False)

    # Interpretation on positives only
    pos_df = samples[samples["label"] == 1].reset_index(drop=True)
    if pos_df.empty:
        print("No positive samples for SaAUC; exiting.")
        # Write metrics with only accuracy
        with open(os.path.join(out_dir, "metrics.json"), "w") as f:
            json.dump({"accuracy": acc, "mean_sauc": None, "n_pos": 0, "n_all": int(len(samples))}, f, indent=2)
        return

    all_sauc: List[float] = []
    sauc_rows: List[Dict] = []
    rng = np.random.default_rng(0)
    outside_sub = None if (args.sauc_outside_subsample is None or int(args.sauc_outside_subsample) <= 0) else int(args.sauc_outside_subsample)
    for s, e in batch_iter_indices(len(pos_df), args.ig_batch_size):
        sub = pos_df.iloc[s:e]
        # Parse intervals first and decide which to process
        parsed_ivals: List[List[Tuple[int, int]]] = []
        nonempty_idx: List[int] = []
        empty_idx: List[int] = []
        for i, r in enumerate(sub.itertuples(index=False)):
            try:
                ivals = json.loads(getattr(r, "intervals_json")) if pd.notna(getattr(r, "intervals_json")) else []
            except Exception:
                ivals = []
            ivals = [(int(a), int(b)) for a, b in (ivals or [])]
            parsed_ivals.append(ivals)
            if ivals:
                nonempty_idx.append(i)
            else:
                empty_idx.append(i)

        # If all empty, skip sequence/IG/PGD and emit NaNs
        if len(nonempty_idx) == 0:
            for i in empty_idx:
                all_sauc.append(float('nan'))
                sauc_rows.append({
                    "sample_id": int(sub.iloc[i]["sample_id"]),
                    "sauc": float('nan'),
                    "pgd_success": None,
                    "pgd_iters": None,
                    "pgd_initial_logit": None,
                    "pgd_final_logit": None,
                })
            continue

        # Build batch only for non-empty masks
        x_list: List[torch.Tensor] = []
        sel_ivals: List[List[Tuple[int, int]]] = []
        sel_row_indices: List[int] = []
        for i in nonempty_idx:
            r = sub.iloc[i]
            fasta_path = os.path.join(args.test_dir, r["fasta_filename"]) if not os.path.isabs(r["fasta_filename"]) else r["fasta_filename"]
            seq = _FASTA.fetch_window(fasta_path, str(r["seqid"]), int(r["start"]), int(r["end"]))
            if len(seq) < seq_len:
                seq = seq.ljust(seq_len, "N")
            x_list.append(one_hot_encode(seq))
            sel_ivals.append(parsed_ivals[i])
            sel_row_indices.append(i)

        xb = torch.stack(x_list).to(dev)
        yb = torch.ones(xb.shape[0], device=dev, dtype=torch.long)

        # PGD baseline per batch
        baselines, stats = find_adversarial_baseline_pgd_batch(
            model, xb, yb, dev, num_iter=int(args.pgd_steps), epsilon=float(args.pgd_epsilon), step_size=(None if args.pgd_step_size is None else float(args.pgd_step_size))
        )
        # fallback to compositional baseline if not successful
        comp_baselines = xb.mean(dim=1, keepdim=True).expand_as(xb)
        use_baselines = torch.where(
            (baselines.abs().sum(dim=(1, 2)) > 0).view(-1, 1, 1), baselines, comp_baselines
        )

        # Integrated gradients and correction
        raw_attr = compute_ig_attributions(model, xb, use_baselines, steps=int(args.ig_steps))
        corr_attr = correct_gradients(raw_attr)
        contrib = (corr_attr * xb).sum(dim=1).abs().detach().cpu().numpy()

        # Emit SaAUC for processed subset
        for j in range(contrib.shape[0]):
            i = sel_row_indices[j]
            sauc = saliency_auc_from_mask(contrib[j], sel_ivals[j], outside_subsample=outside_sub, rng=rng)
            all_sauc.append(sauc)
            sauc_rows.append({
                "sample_id": int(sub.iloc[i]["sample_id"]),
                "sauc": float(sauc),
                "pgd_success": bool(stats[j]['success']),
                "pgd_iters": int(stats[j]['found_at_iter']),
                "pgd_initial_logit": float(stats[j]['initial_logit']),
                "pgd_final_logit": float(stats[j]['final_logit']),
            })

        # Emit NaNs for skipped rows in this sub-batch
        for i in empty_idx:
            all_sauc.append(float('nan'))
            sauc_rows.append({
                "sample_id": int(sub.iloc[i]["sample_id"]),
                "sauc": float('nan'),
                "pgd_success": None,
                "pgd_iters": None,
                "pgd_initial_logit": None,
                "pgd_final_logit": None,
            })

    mean_sauc = float(np.nanmean(all_sauc)) if all_sauc else float('nan')
    print(f"Mean SaAUC over positive samples: {mean_sauc:.4f}")

    # Persist SaAUC results and metrics
    sauc_df = pd.DataFrame(sauc_rows)
    sauc_path = os.path.join(out_dir, "sauuc.parquet")
    sauc_df.to_parquet(sauc_path, index=False)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump({
            "accuracy": acc,
            "mean_sauc": mean_sauc,
            "n_pos": int(len(pos_df)),
            "n_pos_sauc_used": int(np.sum(~np.isnan(all_sauc))) if all_sauc else 0,
            "n_all": int(len(samples)),
            "pgd_epsilon": float(args.pgd_epsilon),
            "pgd_steps": int(args.pgd_steps),
            "pgd_step_size": None if args.pgd_step_size is None else float(args.pgd_step_size),
            "acc_batch_size": int(args.acc_batch_size),
            "ig_batch_size": int(args.ig_batch_size),
            "sauc_outside_subsample": None if outside_sub is None else int(outside_sub),
            "ig_steps": int(args.ig_steps),
            "tile_len": None if args.tile_len is None else int(args.tile_len),
            "tile_stride": None if args.tile_stride is None else int(args.tile_stride),
            "tile_agg": str(args.tile_agg),
        }, f, indent=2)


if __name__ == "__main__":
    main()

