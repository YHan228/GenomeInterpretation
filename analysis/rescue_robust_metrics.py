import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

# Ensure project root (parent of this file's directory) is on sys.path
_this_dir = os.path.dirname(os.path.abspath(__file__))
_proj_root = os.path.dirname(_this_dir)
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

# Reuse training/eval utilities from the project root
try:
    import toy_single_arch as tsa
except Exception as e:
    raise SystemExit(f"Failed to import toy_single_arch from project root '{_proj_root}': {e}")


def _extract_top1_arch(standard_summary_path: str) -> Dict[str, object]:
    with open(standard_summary_path, 'r') as f:
        s = json.load(f)
    top = None
    top_list = s.get('top10_by_saliency_auc') or []
    if top_list:
        top = top_list[0]
    if top is None:
        best = s.get('best_trial')
        if isinstance(best, dict):
            top = {'params': best}
    if not top:
        raise RuntimeError("Could not locate Top-1 standard params in summary.json")
    p = dict(top.get('params', {}))
    arch = {
        'k1': int(p.get('k1', 21)), 'k2': int(p.get('k2', 7)), 'k3': int(p.get('k3', 7)),
        'c1': int(p.get('c1', 64)), 'c2': int(p.get('c2', 128)), 'c3': int(p.get('c3', 192)),
        'pool_w': int(p.get('pool_w', 50)), 'act1': str(p.get('act1', 'exp')),
        'drop_conv1': float(p.get('drop_conv1', 0.1)), 'drop_conv2': float(p.get('drop_conv2', 0.1)),
        'drop_conv3': float(p.get('drop_conv3', 0.1)), 'drop_fc': float(p.get('drop_fc', 0.5)),
    }
    # Ensure odd kernels
    for k in ['k1', 'k2', 'k3']:
        if arch[k] % 2 == 0:
            arch[k] = arch[k] + 1
    return arch


def _parse_dataset_from_name_or_summary(study_dir: str, summary: Dict) -> Tuple[float, float]:
    ds = summary.get('dataset') or {}
    gc_val = ds.get('gc_pos'); cons_val = ds.get('conservation')
    if gc_val is not None and cons_val is not None:
        return float(gc_val), float(cons_val)
    name = summary.get('study_name') or os.path.basename(study_dir)
    try:
        parts = name.split("_gc")
        if len(parts) > 1:
            tail = parts[1]
            gc_str, rest = tail.split("_cons", 1)
            cons_str = rest.split("_mode_robust", 1)[0]
            return float(gc_str), float(cons_str)
    except Exception:
        pass
    raise RuntimeError(f"Could not determine (gc, cons) for study: {study_dir}")


def reeval_top1_for_study(study_dir: str, arch: Dict[str, object], epochs: int, seeds: int, device: str = None) -> Tuple[float, float]:
    """
    Retrain the robust Top-1 configuration for this per-dataset study and return (wIoU, SNR).
    """
    summary_path = os.path.join(study_dir, 'summary.json')
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"No summary.json in {study_dir}")
    with open(summary_path, 'r') as f:
        s = json.load(f)

    top_list = s.get('top10_by_saliency_auc') or []
    if not top_list:
        raise RuntimeError(f"No top trials listed for {study_dir}")
    top1 = top_list[0]
    params = dict(top1.get('params', {}))

    # Dataset
    gc_val, cons_val = _parse_dataset_from_name_or_summary(study_dir, s)
    main_dataset = tsa.load_or_generate_dataset(gc_pos=gc_val, conservation=cons_val)
    dev = torch.device(device) if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bce = nn.BCEWithLogitsLoss()

    wiou_all: List[float] = []
    snr_all: List[float] = []

    for k in range(max(1, int(seeds))):
        seed_value = 4242 + k
        tsa.set_seeds(seed_value, deterministic=False)
        train_bs = int(params.get('train_batch_size', tsa.TRAIN_BATCH_SIZE))
        train_dl, val_dl, _ = tsa._make_dataloaders_for_hpo(main_dataset, seed=seed_value, train_batch_size=train_bs, eval_batch_size=tsa.DEFAULT_EVAL_BATCH_SIZE)

        # Build model from provided arch
        model = tsa.TinyCNN(**arch).to(dev)
        if hasattr(torch, 'compile'):
            try:
                model = torch.compile(model)
            except Exception:
                pass
        opt = torch.optim.AdamW(model.parameters(), lr=float(params.get('lr', 5e-4)), weight_decay=float(params.get('weight_decay', 1e-5)))
        writer = tsa.make_writer(log_dir=os.path.join(study_dir, f"rescue_seed_{seed_value}"))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', factor=0.5, patience=8, verbose=False)
        scaler = tsa.GradScaler()

        # Regime-specific training
        regime = params.get('regime', 'hotflip')
        schedule_flag = str(params.get('schedule', 'off')) == 'on'
        if regime == 'random_smoothing' and 'epsilon' in params:
            tsa.train_random_smoothing(model, train_dl, val_dl, bce, opt, dev, scaler, writer, scheduler,
                                       target_epsilon=float(params['epsilon']), epochs=epochs, early_stopping_patience=10)
        elif regime == 'gaussian_smoothing' and ('sigma2' in params or 'epsilon' in params):
            sigma2 = float(params.get('sigma2', params.get('epsilon', 1e-4)))
            tsa.train_gaussian_smoothing(model, train_dl, val_dl, bce, opt, dev, scaler, writer, scheduler,
                                         sigma2=sigma2, epochs=epochs, early_stopping_patience=10)
        elif regime == 'direct_hotflip' and 'max_flip_fraction' in params:
            tsa.train_direct_hotflip(model, train_dl, val_dl, bce, opt, dev, scaler, writer, scheduler,
                                     max_flip_fraction=float(params['max_flip_fraction']), epochs=epochs, use_scheduling=schedule_flag, early_stopping_patience=25, gc_pos=gc_val)
        else:
            # default to iterative hotflip
            mff = float(params.get('max_flip_fraction', 0.05))
            tsa.train_hotflip(model, train_dl, val_dl, bce, opt, dev, scaler, writer, scheduler,
                              max_flip_fraction=mff, epochs=epochs, use_scheduling=schedule_flag, early_stopping_patience=25, gc_pos=gc_val)

        writer.flush(); writer.close()

        # Evaluate (returns wIoU and SNR among others)
        wio, _, _, snr, _ = tsa.evaluate_model(model, val_dl, dev, pgd_cache={})
        wiou_all.append(float(wio)); snr_all.append(float(snr))

    # Aggregate
    wiou_mean = float(np.mean(wiou_all)) if wiou_all else 0.0
    snr_mean = float(np.mean(snr_all)) if snr_all else 0.0

    # Update summary.json in-place (Top1 only)
    top1['wIoU'] = wiou_mean
    top1['saliency_snr'] = snr_mean
    with open(summary_path, 'w') as f:
        json.dump(s, f, indent=2)
    return wiou_mean, snr_mean


def main():
    parser = argparse.ArgumentParser(description="Rescue script: compute wIoU and Saliency SNR for robust per-dataset studies and persist in summary.json (Top1 only).")
    parser.add_argument('--robust_root', type=str, required=True, help='Root folder containing robust per-dataset studies.')
    parser.add_argument('--standard_summary', type=str, required=True, help='Path to standard summary.json providing Top-1 architecture.')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--seeds', type=int, default=3)
    args = parser.parse_args()

    arch = _extract_top1_arch(args.standard_summary)
    print(f"Loaded Top-1 standard architecture: {arch}")

    # Iterate robust studies
    studies = [os.path.join(args.robust_root, d) for d in os.listdir(args.robust_root) if d.endswith('_mode_robust')]
    studies = [d for d in studies if os.path.isdir(d)]
    print(f"Found {len(studies)} robust studies.")

    for i, study_dir in enumerate(sorted(studies)):
        try:
            wiou, snr = reeval_top1_for_study(study_dir, arch, epochs=args.epochs, seeds=args.seeds)
            print(f"[{i+1}/{len(studies)}] {os.path.basename(study_dir)} -> wIoU={wiou:.3f}, SNR={snr:.3f}")
        except Exception as e:
            print(f"[{i+1}/{len(studies)}] {os.path.basename(study_dir)} -> FAILED: {e}")


if __name__ == '__main__':
    main()


