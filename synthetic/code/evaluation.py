"""
Evaluation methods for synthetic sequence experiments.
Includes PGD attacks, integrated gradients, saliency metrics, and effect size analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import numpy as np
from captum.attr import IntegratedGradients
from typing import Dict, List, Tuple, Optional
from synthetic.code.models import LogisticRegression


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

ANALYSIS_CHUNK_LEN = 60  # Window size for wIoU computation in vanilla experiments


# --------------------------------------------------------------------------- #
# PGD Adversarial Attacks
# --------------------------------------------------------------------------- #

def find_adversarial_baseline_pgd(model, xb: torch.Tensor, yb: torch.Tensor, dev: torch.device,
                                  num_iter: int = 20, epsilon: float = 0.1, step_size: float = 0.01):
    """
    Find adversarial baseline using PGD for Integrated Gradients.
    Single sample version.
    
    Parameters:
    - model: Neural network model
    - xb: Input sample
    - yb: Target label
    - dev: Device
    - num_iter: Number of PGD iterations
    - epsilon: Maximum perturbation size
    - step_size: Step size for PGD
    
    Returns:
    - Adversarial baseline
    - Statistics dictionary
    """
    adv_xb = xb.clone().detach()
    stats = {
        'success': False,
        'initial_logit': 0.0,
        'final_logit': 0.0,
        'found_at_iter': num_iter,
        'initial_prediction_correct': False
    }

    with torch.no_grad(), autocast():
        initial_logits, _ = model(adv_xb)
        initial_pred_class = (initial_logits > 0).float()
        stats['initial_logit'] = initial_logits.item()
        stats['final_logit'] = initial_logits.item()

    is_correct = initial_pred_class.item() == yb.item()
    stats['initial_prediction_correct'] = is_correct

    # Only run PGD for correct positive predictions
    if not is_correct or yb.item() == 0:
        return torch.zeros_like(xb, device=dev), stats

    loss_fn = nn.BCEWithLogitsLoss()
    step_size = epsilon / 10.0

    for i in range(num_iter):
        adv_xb.requires_grad = True
        with autocast():
            logits, _ = model(adv_xb)
            loss = loss_fn(logits, yb.expand_as(logits))
        
        model.zero_grad()
        loss.backward()
        
        with torch.no_grad():
            grad = adv_xb.grad.data
            adv_xb = adv_xb + step_size * grad.sign()
            
            delta = adv_xb - xb
            delta = torch.clamp(delta, -epsilon, epsilon)
            adv_xb = torch.clamp(xb + delta, 0, 1)

            current_logits, _ = model(adv_xb)
            current_pred_class = (current_logits > 0).float()
            
            if current_pred_class.item() != initial_pred_class.item():
                stats['success'] = True
                stats['final_logit'] = current_logits.item()
                stats['found_at_iter'] = i + 1
                return adv_xb.detach(), stats
    
    with torch.no_grad():
        final_logits, _ = model(adv_xb)
        stats['final_logit'] = final_logits.item()

    return torch.zeros_like(xb, device=dev), stats


def find_adversarial_baseline_pgd_batch_optimized(model, xb_batch: torch.Tensor, yb_batch: torch.Tensor, dev: torch.device,
                                                  num_iter: int = 20, epsilon: float = 0.1):
    """
    GPU-optimized batched PGD that minimizes synchronization.
    
    Parameters:
    - model: Neural network model
    - xb_batch: Batch of input samples
    - yb_batch: Batch of target labels
    - dev: Device
    - num_iter: Number of PGD iterations
    - epsilon: Maximum perturbation size
    
    Returns:
    - Batch of adversarial baselines
    - List of statistics dictionaries
    """
    batch_size = xb_batch.shape[0]
    adv_xb_batch = xb_batch.clone().detach()
    
    # Initialize stats - we'll update them at the end to minimize synchronization
    with torch.no_grad(), autocast():
        initial_logits, _ = model(adv_xb_batch)
        initial_pred_classes = (initial_logits > 0).float()
    
    # Determine which samples to attack (positive samples with correct predictions)
    is_correct = (initial_pred_classes == yb_batch)
    is_positive = (yb_batch == 1)
    active_mask = is_correct & is_positive
    
    # Early return if no samples need attacking
    if not active_mask.any():
        # Create stats with minimal synchronization
        stats_list = []
        for i in range(batch_size):
            stats_list.append({
                'success': False,
                'initial_logit': initial_logits[i].item(),
                'final_logit': initial_logits[i].item(),
                'found_at_iter': num_iter,
                'initial_prediction_correct': is_correct[i].item()
            })
        return torch.zeros_like(xb_batch, device=dev), stats_list
    
    loss_fn = nn.BCEWithLogitsLoss(reduction='none')
    step_size = epsilon / 10.0
    
    # Track success without synchronization
    success_mask = torch.zeros(batch_size, dtype=torch.bool, device=dev)
    success_iter = torch.full((batch_size,), num_iter, dtype=torch.long, device=dev)
    final_baselines = torch.zeros_like(xb_batch, device=dev)
    
    for iter_idx in range(num_iter):
        if not active_mask.any():
            break
        
        # Compute gradients only for active samples
        active_xb = adv_xb_batch[active_mask].detach().requires_grad_(True)
        
        with autocast():
            active_logits, _ = model(active_xb)
            active_labels = yb_batch[active_mask]
            losses = loss_fn(active_logits, active_labels)
            loss = losses.mean()
        
        model.zero_grad()
        loss.backward()
        
        # Update adversarial examples
        with torch.no_grad():
            grad_sign = active_xb.grad.sign()
            active_xb_new = active_xb + step_size * grad_sign
            
            # Get indices of active samples
            active_indices = torch.where(active_mask)[0]
            
            # Vectorized projection back to epsilon ball
            for j, idx in enumerate(active_indices):
                delta = active_xb_new[j] - xb_batch[idx]
                delta = torch.clamp(delta, -epsilon, epsilon)
                adv_xb_batch[idx] = torch.clamp(xb_batch[idx] + delta, 0, 1)
            
            # Check for successful flips
            current_logits, _ = model(adv_xb_batch[active_mask])
            current_pred_classes = (current_logits > 0).float()
            
            # Find newly successful attacks
            flip_occurred = (current_pred_classes != initial_pred_classes[active_mask])
            
            # Update success tracking
            for j, idx in enumerate(active_indices):
                if flip_occurred[j] and not success_mask[idx]:
                    success_mask[idx] = True
                    success_iter[idx] = iter_idx + 1
                    final_baselines[idx] = adv_xb_batch[idx].clone()
                    active_mask[idx] = False
    
    # Compute final logits
    with torch.no_grad():
        final_logits, _ = model(adv_xb_batch)
    
    # Create stats with single synchronization at the end
    stats_list = []
    for i in range(batch_size):
        stats_list.append({
            'success': success_mask[i].item(),
            'initial_logit': initial_logits[i].item(),
            'final_logit': final_logits[i].item() if success_mask[i] else final_logits[i].item(),
            'found_at_iter': success_iter[i].item(),
            'initial_prediction_correct': is_correct[i].item()
        })
    
    return final_baselines, stats_list


# --------------------------------------------------------------------------- #
# Saliency Analysis with Integrated Gradients
# --------------------------------------------------------------------------- #

def evaluate_model_vanilla(model, test_dl, dev, pgd_cache=None):
    """
    Evaluate model for vanilla experiments WITH wIoU metric.
    Used for toy_slurm.py experiments.
    
    Parameters:
    - model: Neural network model
    - test_dl: Test data loader
    - dev: Device
    - pgd_cache: Optional cache for PGD results
    
    Returns:
    - mean_wiou: Mean weighted IoU
    - accuracy: Test accuracy
    - mean_saliency_auc: Mean saliency AUC
    - mean_saliency_snr: Mean saliency SNR
    - pgd_stats: Dictionary of PGD statistics
    """
    print(f"Evaluating model (vanilla)...")
    SAMPLE_N = 50
    PGD_BATCH_SIZE = 25
    IG_BATCH_SIZE = 10
    
    # Get sequence length from data
    sample_batch = next(iter(test_dl))
    SEQ_LEN = sample_batch[0].shape[2]

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb, _ in test_dl:
            xb, yb = xb.to(dev), yb.to(dev)
            with autocast():
                logits, _ = model(xb)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == yb).sum().item()
            total += len(yb)
    accuracy = correct / total if total else 0
    print(f"  Test accuracy: {accuracy:.3f}")

    # For LogisticRegression, only accuracy is needed
    original_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    if isinstance(original_model, LogisticRegression):
        print("  (Logistic Regression model: skipping saliency and PGD evaluation)")
        return 0.0, accuracy, 0.0, 0.0, {'pgd_success_rate': 0, 'pgd_mean_iters_to_flip': 0}

    def model_for_captum(x):
        with autocast():
            return model(x)[0].unsqueeze(-1)

    ig = IntegratedGradients(model_for_captum)
    test_ds = test_dl.dataset
    
    # Handle both Subset and regular datasets
    if hasattr(test_ds, 'indices'):  # It's a Subset
        positive_subset_indices = [
            i for i, original_idx in enumerate(test_ds.indices)
            if test_ds.dataset.y[original_idx] == 1
        ]
    else:  # Regular dataset
        positive_subset_indices = [
            i for i in range(len(test_ds))
            if test_ds.y[i] == 1
        ]

    rng = np.random.default_rng(0)
    sample_n_actual = min(SAMPLE_N, len(positive_subset_indices))
    if sample_n_actual == 0:
        print("Warning: No positive samples in test set for evaluation.")
        return 0.0, accuracy, 0.0, 0.0, {'pgd_success_rate': 0, 'pgd_mean_iters_to_flip': 0}
        
    idxs = rng.choice(positive_subset_indices, size=sample_n_actual, replace=False)

    # --- PGD Caching ---
    if pgd_cache is None:
        pgd_cache = {}
    
    with torch.no_grad():
        if hasattr(model, '_orig_mod'):
            fingerprint = model._orig_mod.conv1.weight.detach().cpu().numpy().tobytes()[:64]
        else:
            fingerprint = model.conv1.weight.detach().cpu().numpy().tobytes()[:64]

    # --- Batched PGD Processing ---
    all_pgd_baselines = []
    all_pgd_stats = []
    
    if fingerprint in pgd_cache:
        print(f"  Using cached PGD results...")
        cached_data = pgd_cache[fingerprint]
        all_pgd_baselines = cached_data['baselines']
        all_pgd_stats = cached_data['stats']
    else:
        print(f"  Computing PGD baselines in batches...")
        for batch_start in range(0, len(idxs), PGD_BATCH_SIZE):
            batch_end = min(batch_start + PGD_BATCH_SIZE, len(idxs))
            batch_idxs = idxs[batch_start:batch_end]
            
            xb_list = [test_ds[i][0] for i in batch_idxs]
            yb_list = [test_ds[i][1] for i in batch_idxs]
            
            xb_batch = torch.stack(xb_list).to(dev)
            yb_batch = torch.tensor(yb_list, device=dev, dtype=torch.float)
            
            pgd_baselines_batch, pgd_stats_batch = find_adversarial_baseline_pgd_batch_optimized(
                model, xb_batch, yb_batch, dev
            )
            
            all_pgd_baselines.extend([pgd_baselines_batch[i] for i in range(len(batch_idxs))])
            all_pgd_stats.extend(pgd_stats_batch)
        
        pgd_cache[fingerprint] = {
            'baselines': all_pgd_baselines,
            'stats': all_pgd_stats
        }

    # --- Batched IG Processing ---
    results = []
    print(f"  Computing Integrated Gradients in batches...")
    
    for batch_start in range(0, len(idxs), IG_BATCH_SIZE):
        batch_end = min(batch_start + IG_BATCH_SIZE, len(idxs))
        batch_range = range(batch_start, batch_end)
        
        xb_list, mask_list, baseline_list = [], [], []
        
        for i in batch_range:
            idx = idxs[i]
            xb, _, mask = test_ds[idx]
            xb_list.append(xb)
            mask_list.append(mask)
            
            if all_pgd_stats[i]['success']:
                baseline_list.append(all_pgd_baselines[i].squeeze(0).cpu())
            else:
                proportions = xb.mean(dim=1, keepdim=True)
                baseline_list.append(proportions.expand_as(xb))
        
        xb_batch = torch.stack(xb_list).to(dev)
        baseline_batch = torch.stack(baseline_list).to(dev)
        
        raw_attributions_batch = ig.attribute(xb_batch, baselines=baseline_batch, target=0)
        
        for j, i in enumerate(batch_range):
            raw_attr = raw_attributions_batch[j]
            xb = xb_batch[j]
            mask = mask_list[j]
            
            corrected_attr = raw_attr - raw_attr.mean(dim=0, keepdim=True)
            attributions = np.abs((corrected_attr * xb).sum(0).cpu().numpy())

            # Compute wIoU metric
            window_sums = np.convolve(attributions, np.ones(ANALYSIS_CHUNK_LEN), mode='valid')
            best_window_start = np.argmax(window_sums)
            pred_mask_cont = np.zeros(SEQ_LEN, dtype=bool)
            pred_mask_cont[best_window_start:best_window_start + ANALYSIS_CHUNK_LEN] = True
            
            inter_cont = (pred_mask_cont & mask).sum()
            union_cont = (pred_mask_cont | mask).sum()
            iou_cont = inter_cont / union_cont if union_cont else 0

            # Compute saliency metrics
            inside_scores = attributions[mask]
            outside_scores = attributions[~mask]
            
            saliency_auc = (inside_scores[:, None] > outside_scores[None, :]).mean() if len(inside_scores) > 0 and len(outside_scores) > 0 else 0.5
            
            sum_sq_inside = np.sum(inside_scores**2)
            sum_sq_total = np.sum(attributions**2)
            saliency_snr = sum_sq_inside / (sum_sq_total + 1e-9)

            results.append(dict(iou_cont=iou_cont, saliency_auc=saliency_auc, saliency_snr=saliency_snr))

    # --- Aggregate Stats ---
    attackable_samples = [r for r in all_pgd_stats if r['initial_prediction_correct']]
    pgd_success_count = sum(1 for r in attackable_samples if r['success'])
    pgd_success_rate = pgd_success_count / len(attackable_samples) if attackable_samples else 0
    
    iters_to_flip = [r['found_at_iter'] for r in attackable_samples if r['success']]
    pgd_mean_iters_to_flip = np.mean(iters_to_flip) if iters_to_flip else 0

    pgd_stats = {
        "pgd_success_rate": pgd_success_rate,
        "pgd_mean_iters_to_flip": pgd_mean_iters_to_flip,
    }
    
    print(f"  PGD baseline success rate: {pgd_success_rate:.3f}")

    mean_iou_cont = np.mean([r['iou_cont'] for r in results]) if results else 0.0
    mean_saliency_auc = np.mean([r['saliency_auc'] for r in results]) if results else 0.0
    mean_saliency_snr = np.mean([r['saliency_snr'] for r in results]) if results else 0.0
    
    print(f"  Mean wIoU: {mean_iou_cont:.3f}")
    print(f"  Mean Saliency AUC: {mean_saliency_auc:.3f}")
    print(f"  Mean Saliency SNR: {mean_saliency_snr:.3f}")

    return mean_iou_cont, accuracy, mean_saliency_auc, mean_saliency_snr, pgd_stats


def evaluate_model(model, test_dl, dev, pgd_cache=None):
    """
    Evaluate model for complex experiments WITHOUT wIoU metric.
    Used for merged_experiment.py experiments.
    Includes separate motif and promoter region evaluation.
    
    Parameters:
    - model: Neural network model
    - test_dl: Test data loader
    - dev: Device
    - pgd_cache: Optional cache for PGD results
    
    Returns:
    - accuracy: Test accuracy
    - mean_saliency_auc: Mean saliency AUC (full mask)
    - mean_saliency_snr: Mean saliency SNR (full mask)
    - mean_motif_saliency_auc: Mean saliency AUC (motif-only)
    - mean_motif_saliency_snr: Mean saliency SNR (motif-only)
    - pgd_stats: Dictionary of PGD statistics
    """
    print(f"Evaluating model (complex)...")
    SAMPLE_N = 50
    PGD_BATCH_SIZE = 100
    IG_BATCH_SIZE = 50
    PROMOTER_MAX_LEN = 42  # hex1 + max_spacer + hex2

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb, _ in test_dl:
            xb, yb = xb.to(dev), yb.to(dev)
            with autocast():
                logits, _ = model(xb)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == yb).sum().item()
            total += len(yb)
    accuracy = correct / total if total else 0
    print(f"  Test accuracy: {accuracy:.3f}")

    # For LogisticRegression, only accuracy is needed
    original_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    if isinstance(original_model, LogisticRegression):
        print("  (Logistic Regression model: skipping saliency and PGD evaluation)")
        return accuracy, 0.0, 0.0, 0.0, 0.0, {'pgd_success_rate': 0, 'pgd_mean_iters_to_flip': 0}

    def model_for_captum(x):
        with autocast():
            return model(x)[0].unsqueeze(-1)

    ig = IntegratedGradients(model_for_captum)
    test_ds = test_dl.dataset
    
    # Handle both Subset and regular datasets
    if hasattr(test_ds, 'indices'):  # It's a Subset
        positive_subset_indices = [
            i for i, original_idx in enumerate(test_ds.indices)
            if test_ds.dataset.y[original_idx] == 1
        ]
    else:  # Regular dataset
        positive_subset_indices = [
            i for i in range(len(test_ds))
            if test_ds.y[i] == 1
        ]

    rng = np.random.default_rng(0)
    sample_n_actual = min(SAMPLE_N, len(positive_subset_indices))
    if sample_n_actual == 0:
        print("Warning: No positive samples in test set for evaluation.")
        return accuracy, 0.0, 0.0, 0.0, 0.0, {'pgd_success_rate': 0, 'pgd_mean_iters_to_flip': 0}
        
    idxs = rng.choice(positive_subset_indices, size=sample_n_actual, replace=False)

    # --- PGD Caching ---
    if pgd_cache is None:
        pgd_cache = {}
    
    with torch.no_grad():
        if hasattr(model, '_orig_mod'):
            fingerprint = model._orig_mod.conv1.weight.detach().cpu().numpy().tobytes()[:64]
        else:
            fingerprint = model.conv1.weight.detach().cpu().numpy().tobytes()[:64]

    # --- Batched PGD Processing ---
    all_pgd_baselines = []
    all_pgd_stats = []
    
    if fingerprint in pgd_cache:
        print(f"  Using cached PGD results...")
        cached_data = pgd_cache[fingerprint]
        all_pgd_baselines = cached_data['baselines']
        all_pgd_stats = cached_data['stats']
    else:
        print(f"  Computing PGD baselines in batches of {PGD_BATCH_SIZE}...")
        for batch_start in range(0, len(idxs), PGD_BATCH_SIZE):
            batch_end = min(batch_start + PGD_BATCH_SIZE, len(idxs))
            batch_idxs = idxs[batch_start:batch_end]
            
            xb_list = [test_ds[i][0] for i in batch_idxs]
            yb_list = [test_ds[i][1] for i in batch_idxs]
            
            xb_batch = torch.stack(xb_list).to(dev)
            yb_batch = torch.tensor(yb_list, device=dev, dtype=torch.float)
            
            pgd_baselines_batch, pgd_stats_batch = find_adversarial_baseline_pgd_batch_optimized(
                model, xb_batch, yb_batch, dev
            )
            
            all_pgd_baselines.extend([pgd_baselines_batch[i] for i in range(len(batch_idxs))])
            all_pgd_stats.extend(pgd_stats_batch)
        
        pgd_cache[fingerprint] = {
            'baselines': all_pgd_baselines,
            'stats': all_pgd_stats
        }

    # --- Batched IG Processing ---
    results = []
    results_motif_only = []  # Separate results for motif-only evaluation
    print(f"  Computing Integrated Gradients in batches of {IG_BATCH_SIZE}...")
    
    for batch_start in range(0, len(idxs), IG_BATCH_SIZE):
        batch_end = min(batch_start + IG_BATCH_SIZE, len(idxs))
        batch_range = range(batch_start, batch_end)
        
        xb_list, mask_list, baseline_list = [], [], []
        
        for i in batch_range:
            idx = idxs[i]
            xb, _, mask = test_ds[idx]
            xb_list.append(xb)
            mask_list.append(mask)
            
            if all_pgd_stats[i]['success']:
                baseline_list.append(all_pgd_baselines[i].squeeze(0).cpu())
            else:
                proportions = xb.mean(dim=1, keepdim=True)
                baseline_list.append(proportions.expand_as(xb))
        
        xb_batch = torch.stack(xb_list).to(dev)
        baseline_batch = torch.stack(baseline_list).to(dev)
        
        raw_attributions_batch = ig.attribute(xb_batch, baselines=baseline_batch, target=0)
        
        for j, i in enumerate(batch_range):
            raw_attr = raw_attributions_batch[j]
            xb = xb_batch[j]
            mask = mask_list[j]
            
            corrected_attr = raw_attr - raw_attr.mean(dim=0, keepdim=True)
            attributions = np.abs((corrected_attr * xb).sum(0).cpu().numpy())

            # Full mask evaluation (includes promoter)
            inside_scores = attributions[mask]
            outside_scores = attributions[~mask]
            
            saliency_auc = (inside_scores[:, None] > outside_scores[None, :]).mean() if len(inside_scores) > 0 and len(outside_scores) > 0 else 0.5
            
            sum_sq_inside = np.sum(inside_scores**2)
            sum_sq_total = np.sum(attributions**2)
            saliency_snr = sum_sq_inside / (sum_sq_total + 1e-9)

            results.append(dict(saliency_auc=saliency_auc, saliency_snr=saliency_snr))
            
            # Motif-only evaluation: create mask that excludes promoter region
            # Promoter is placed before the first motif, so find where mask starts
            mask_indices = np.where(mask)[0]
            if len(mask_indices) > 0:
                first_mask_pos = mask_indices[0]
                # Check if there's a gap in the mask (indicating promoter then motifs)
                gaps = np.diff(mask_indices)
                if np.any(gaps > 1):
                    # Find the first big gap - promoter ends there
                    gap_positions = np.where(gaps > 1)[0]
                    promoter_end = mask_indices[gap_positions[0]] + 1
                    # Create motif-only mask
                    motif_only_mask = mask.copy()
                    motif_only_mask[:promoter_end] = False
                else:
                    # No gap found, assume no clear separation or promoter at very start
                    # Conservative: exclude first PROMOTER_MAX_LEN positions of mask
                    motif_only_mask = mask.copy()
                    if first_mask_pos < PROMOTER_MAX_LEN:
                        motif_only_mask[:first_mask_pos + PROMOTER_MAX_LEN] = False
                
                # Compute motif-only metrics
                motif_inside_scores = attributions[motif_only_mask]
                motif_outside_scores = attributions[~motif_only_mask]
                
                if len(motif_inside_scores) > 0 and len(motif_outside_scores) > 0:
                    motif_saliency_auc = (motif_inside_scores[:, None] > motif_outside_scores[None, :]).mean()
                else:
                    motif_saliency_auc = 0.5
                
                motif_sum_sq_inside = np.sum(motif_inside_scores**2) if len(motif_inside_scores) > 0 else 0
                motif_saliency_snr = motif_sum_sq_inside / (sum_sq_total + 1e-9)
            else:
                # No mask found, use default values
                motif_saliency_auc = 0.5
                motif_saliency_snr = 0.0
                
            results_motif_only.append(dict(saliency_auc=motif_saliency_auc, saliency_snr=motif_saliency_snr))

    # --- Aggregate Stats ---
    attackable_samples = [r for r in all_pgd_stats if r['initial_prediction_correct']]
    pgd_success_count = sum(1 for r in attackable_samples if r['success'])
    pgd_success_rate = pgd_success_count / len(attackable_samples) if attackable_samples else 0
    
    iters_to_flip = [r['found_at_iter'] for r in attackable_samples if r['success']]
    pgd_mean_iters_to_flip = np.mean(iters_to_flip) if iters_to_flip else 0

    pgd_stats = {
        "pgd_success_rate": pgd_success_rate,
        "pgd_mean_iters_to_flip": pgd_mean_iters_to_flip,
    }
    
    print(f"  PGD baseline success rate: {pgd_success_rate:.3f}")

    mean_saliency_auc = np.mean([r['saliency_auc'] for r in results]) if results else 0.0
    mean_saliency_snr = np.mean([r['saliency_snr'] for r in results]) if results else 0.0
    mean_motif_saliency_auc = np.mean([r['saliency_auc'] for r in results_motif_only]) if results_motif_only else 0.0
    mean_motif_saliency_snr = np.mean([r['saliency_snr'] for r in results_motif_only]) if results_motif_only else 0.0
    
    print(f"  Mean Saliency AUC (full mask): {mean_saliency_auc:.3f}")
    print(f"  Mean Saliency SNR (full mask): {mean_saliency_snr:.3f}")
    print(f"  Mean Saliency AUC (motif-only): {mean_motif_saliency_auc:.3f}")
    print(f"  Mean Saliency SNR (motif-only): {mean_motif_saliency_snr:.3f}")

    return accuracy, mean_saliency_auc, mean_saliency_snr, mean_motif_saliency_auc, mean_motif_saliency_snr, pgd_stats


# --------------------------------------------------------------------------- #
# Effect Size Analysis (from toy_slurm.py)
# --------------------------------------------------------------------------- #

def compute_effect_sizes_fast(model, val_loader, dev, n_samples=50):
    """
    Compute effect sizes by randomly corrupting sequences and measuring prediction changes.
    
    Parameters:
    - model: Neural network model
    - val_loader: Validation data loader
    - dev: Device
    - n_samples: Number of samples to use
    
    Returns:
    - Dictionary of effect size statistics
    """
    model.eval()
    
    # Collect samples
    all_x, all_y, all_m = [], [], []
    for xb, yb, mb in val_loader:
        all_x.append(xb)
        all_y.append(yb)
        all_m.append(mb)
        if sum(len(x) for x in all_x) >= n_samples:
            break
    
    # Concatenate and limit to n_samples
    X = torch.cat(all_x)[:n_samples].to(dev)
    y = torch.cat(all_y)[:n_samples].to(dev)
    masks = torch.cat(all_m)[:n_samples]
    
    n_actual = len(X)
    seq_len = X.shape[2]
    
    with torch.no_grad():
        # Get original predictions
        with autocast():
            orig_logits, _ = model(X)
        orig_probs = torch.sigmoid(orig_logits)
        
        # Random corruption
        n_corrupt = max(1, int(0.01 * seq_len))  # Corrupt 1% of positions
        corrupted_X = X.clone()
        
        for i in range(n_actual):
            # Choose random positions to corrupt
            positions = torch.randperm(seq_len)[:n_corrupt]
            
            for pos in positions:
                # Get current base
                current_base = corrupted_X[i, :, pos].argmax()
                # Choose a different base randomly
                other_bases = [b for b in range(4) if b != current_base]
                new_base = np.random.choice(other_bases)
                # Apply corruption
                corrupted_X[i, :, pos] = 0
                corrupted_X[i, new_base, pos] = 1
        
        # Get corrupted predictions
        with autocast():
            corrupt_logits, _ = model(corrupted_X)
        corrupt_probs = torch.sigmoid(corrupt_logits)
        
        # Compute effect sizes
        prob_changes = (orig_probs - corrupt_probs).abs()
        
        # Separate by positive/negative samples
        pos_mask = (y == 1)
        neg_mask = (y == 0)
        
        effect_stats = {
            'mean_effect_positive': prob_changes[pos_mask].mean().item() if pos_mask.any() else 0,
            'mean_effect_negative': prob_changes[neg_mask].mean().item() if neg_mask.any() else 0,
            'std_effect_positive': prob_changes[pos_mask].std().item() if pos_mask.any() else 0,
            'std_effect_negative': prob_changes[neg_mask].std().item() if neg_mask.any() else 0,
            'n_samples': n_actual,
            'n_corrupt_positions': n_corrupt,
        }
    
    return effect_stats


# --------------------------------------------------------------------------- #
# Sequence Property Analysis
# --------------------------------------------------------------------------- #

def compute_sequence_properties_gpu(xb: torch.Tensor, masks: torch.Tensor = None) -> dict:
    """
    Compute various sequence properties on GPU.
    
    Parameters:
    - xb: One-hot encoded sequences (batch_size, 4, seq_len)
    - masks: Optional masks indicating signal regions
    
    Returns:
    - Dictionary of sequence properties
    """
    batch_size, _, seq_len = xb.shape
    
    # Convert to base indices
    seq_indices = torch.argmax(xb, dim=1)  # (batch_size, seq_len)
    
    # GC content (C=1, G=2)
    is_gc = (seq_indices == 1) | (seq_indices == 2)
    gc_content = is_gc.float().mean(dim=1)
    
    # Entropy (base distribution)
    base_counts = torch.zeros(batch_size, 4, device=xb.device)
    for b in range(4):
        base_counts[:, b] = (seq_indices == b).sum(dim=1).float()
    base_probs = base_counts / seq_len
    # Add small epsilon to avoid log(0)
    base_probs = base_probs + 1e-10
    entropy = -(base_probs * torch.log2(base_probs)).sum(dim=1)
    
    # Compute properties separately for masked and unmasked regions if masks provided
    if masks is not None:
        masks_tensor = torch.tensor(masks, device=xb.device, dtype=torch.bool)
        
        # GC content in signal regions
        gc_signal = []
        gc_background = []
        
        for i in range(batch_size):
            if masks_tensor[i].any():
                signal_indices = seq_indices[i][masks_tensor[i]]
                bg_indices = seq_indices[i][~masks_tensor[i]]
                
                gc_sig = ((signal_indices == 1) | (signal_indices == 2)).float().mean()
                gc_bg = ((bg_indices == 1) | (bg_indices == 2)).float().mean()
                
                gc_signal.append(gc_sig.item())
                gc_background.append(gc_bg.item())
            else:
                gc_signal.append(gc_content[i].item())
                gc_background.append(gc_content[i].item())
        
        props = {
            'gc_content_mean': gc_content.mean().item(),
            'gc_content_std': gc_content.std().item(),
            'gc_signal_mean': np.mean(gc_signal),
            'gc_background_mean': np.mean(gc_background),
            'entropy_mean': entropy.mean().item(),
            'entropy_std': entropy.std().item(),
        }
    else:
        props = {
            'gc_content_mean': gc_content.mean().item(),
            'gc_content_std': gc_content.std().item(),
            'entropy_mean': entropy.mean().item(),
            'entropy_std': entropy.std().item(),
        }
    
    return props


def compute_adversarial_changes(xb_orig: torch.Tensor, xb_adv: torch.Tensor, 
                               masks: torch.Tensor = None, gc_pos: float = 0.5) -> dict:
    """
    Analyze changes between original and adversarial examples.
    
    Parameters:
    - xb_orig: Original sequences
    - xb_adv: Adversarial sequences
    - masks: Optional masks indicating signal regions
    - gc_pos: Expected GC content
    
    Returns:
    - Dictionary of change statistics
    """
    batch_size = xb_orig.shape[0]
    
    # Find changed positions
    changes = (xb_orig != xb_adv).any(dim=1)  # (batch_size, seq_len)
    n_changes_per_sample = changes.sum(dim=1).float()
    
    # Analyze what bases were changed
    orig_bases = torch.argmax(xb_orig, dim=1)
    adv_bases = torch.argmax(xb_adv, dim=1)
    
    # Transition matrix (from -> to)
    transition_counts = torch.zeros(4, 4, device=xb_orig.device)
    for i in range(4):
        for j in range(4):
            if i != j:
                transition_counts[i, j] = ((orig_bases == i) & (adv_bases == j) & changes).sum()
    
    # GC content changes
    def gc_content(bases):
        return ((bases == 1) | (bases == 2)).float().mean(dim=1)
    
    gc_orig = gc_content(orig_bases)
    gc_adv = gc_content(adv_bases)
    gc_change = gc_adv - gc_orig
    
    def gc_to_optimal_acc(gc_content, gc_expected=gc_pos):
        """GC -> Optimal accuracy score (higher is better)"""
        diff = torch.abs(gc_content - gc_expected)
        return 1.0 - diff
    
    gc_opt_acc_orig = gc_to_optimal_acc(gc_orig)
    gc_opt_acc_adv = gc_to_optimal_acc(gc_adv)
    gc_opt_acc_change = gc_opt_acc_adv - gc_opt_acc_orig
    
    # Changes in signal vs background regions
    if masks is not None:
        masks_tensor = torch.tensor(masks, device=xb_orig.device, dtype=torch.bool)
        signal_changes = []
        bg_changes = []
        
        for i in range(batch_size):
            if masks_tensor[i].any():
                signal_changes.append(changes[i][masks_tensor[i]].float().mean().item())
                bg_changes.append(changes[i][~masks_tensor[i]].float().mean().item())
            else:
                bg_changes.append(changes[i].float().mean().item())
                signal_changes.append(0.0)
        
        signal_change_rate = np.mean(signal_changes)
        bg_change_rate = np.mean(bg_changes)
    else:
        signal_change_rate = 0.0
        bg_change_rate = changes.float().mean().item()
    
    # Information content change (simplified)
    def compute_ic(freqs):
        """Compute information content from frequency matrix"""
        # Shape: (batch_size, 4, seq_len) -> (4, seq_len)
        avg_freqs = freqs.mean(dim=0)
        # Add small epsilon to avoid log(0)
        avg_freqs = avg_freqs + 1e-10
        # IC = 2 - entropy
        entropy = -(avg_freqs * torch.log2(avg_freqs)).sum(dim=0)
        ic = 2 - entropy
        return ic.mean()
    
    ic_orig = compute_ic(xb_orig)
    ic_adv = compute_ic(xb_adv)
    
    stats = {
        'n_changes_mean': n_changes_per_sample.mean().item(),
        'n_changes_std': n_changes_per_sample.std().item(),
        'change_rate': changes.float().mean().item(),
        'signal_change_rate': signal_change_rate,
        'bg_change_rate': bg_change_rate,
        'gc_change_mean': gc_change.mean().item(),
        'gc_change_std': gc_change.std().item(),
        'gc_opt_acc_change_mean': gc_opt_acc_change.mean().item(),
        'ic_change': (ic_adv - ic_orig).item(),
        'transition_matrix': transition_counts.cpu().numpy(),
    }
    
    return stats 