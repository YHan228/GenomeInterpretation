import torch
import torch.nn.functional as F
import torch.nn as nn
from captum.attr import IntegratedGradients


def compute_integrated_gradients(model, inputs, baselines, target_class_index=0, apply_correction=True):
    """
    Compute Integrated Gradients with optional gradient correction.

    Args:
        model: The neural network model
        inputs: Input tensor of shape (batch, channels, seq_len)
        baselines: Baseline tensor of same shape as inputs
        target_class_index: Target class for attribution
        apply_correction: If True, apply gradient correction by subtracting
                         the mean across the channel dimension at each position.
                         This ensures attributions sum to zero at each position,
                         which is appropriate for one-hot encoded DNA sequences.

    Returns:
        Attributions as numpy array
    """
    ig = IntegratedGradients(model)
    attributions = ig.attribute(inputs,
                                baselines=baselines,
                                target=target_class_index,
                                return_convergence_delta=False)

    if apply_correction:
        # Apply gradient correction: subtract mean across nucleotide dimension
        # For shape (batch, seq_len, 4), dim=2 is the nucleotide dimension
        # (The model expects input as (batch, seq_len, 4) and permutes internally)
        corrected_attr = attributions - attributions.mean(dim=2, keepdim=True)
        return corrected_attr.cpu().numpy()

    return attributions.cpu().numpy()


def generate_direct_hotflip_examples_optimized(model, xb, yb, loss_fn, flip_fraction: float):
    """GPU-optimized Direct HotFlip using fully vectorized operations."""
    seq_len = xb.shape[2]
    batch_size = xb.shape[0]
    k_flips = int(flip_fraction * seq_len)
    
    if k_flips == 0:
        return xb.clone()
        
    adv_xb = xb.clone()
    adv_xb.requires_grad = True
    model.zero_grad()
    
    outputs = model(adv_xb)
    # Handle both tuple and tensor outputs
    if isinstance(outputs, tuple):
        logits = outputs[0]
    else:
        logits = outputs
    # Squeeze if necessary
    if logits.dim() > 1 and logits.shape[1] == 1:
        logits = logits.squeeze(-1)
    loss = loss_fn(logits, yb.squeeze(-1))
    loss.backward()
    
    grad = adv_xb.grad.data
    
    # Vectorized saliency computation
    current_bases_onehot = (adv_xb > 0.5).float()
    grad_at_current = (grad * current_bases_onehot).sum(dim=1, keepdim=True)
    saliency = grad - grad_at_current
    saliency.masked_fill_(current_bases_onehot.bool(), -1e9)
    
    # Detach for updates
    adv_xb = adv_xb.detach()
    
    # Reshape for top-k: (batch, 4, seq_len) -> (batch, 4*seq_len)
    saliency_flat = saliency.reshape(batch_size, -1)
    
    # Get top-k flips for each sequence
    topk_values, topk_indices = torch.topk(saliency_flat, k_flips, dim=1)
    
    # Convert to base and position indices
    topk_bases = topk_indices // seq_len  # (batch, k_flips)
    topk_positions = topk_indices % seq_len  # (batch, k_flips)
    
    # Apply all flips using advanced indexing
    for flip_idx in range(k_flips):
        batch_indices = torch.arange(batch_size, device=xb.device)
        positions = topk_positions[:, flip_idx]
        new_bases = topk_bases[:, flip_idx]
        
        # Find and zero out current bases
        current_bases = adv_xb[batch_indices, :, positions].argmax(dim=1)
        adv_xb[batch_indices, current_bases, positions] = 0.0
        
        # Set new bases
        adv_xb[batch_indices, new_bases, positions] = 1.0
    
    return adv_xb

def find_adversarial_baseline_pgd_for_probs_batch_optimized(model, xb_batch: torch.Tensor, yb_batch: torch.Tensor, dev: torch.device,
                                                  num_iter: int = 20, epsilon: float = 0.1):
    """GPU-optimized batched PGD for probability-based models."""
    batch_size = xb_batch.shape[0]
    adv_xb_batch = xb_batch.clone().detach()
    
    # Initialize stats 
    with torch.no_grad():
        outputs = model(adv_xb_batch)
        # Handle both tuple and tensor outputs
        if isinstance(outputs, tuple):
            initial_probs = outputs[0]
        else:
            initial_probs = outputs
        initial_pred_classes = (initial_probs > 0.5).float()
    
    is_correct = (initial_pred_classes.squeeze(-1) == yb_batch.squeeze(-1))
    is_positive = (yb_batch.squeeze(-1) == 1)
    active_mask = is_correct & is_positive
    
    if not active_mask.any():
        stats_list = []
        for i in range(batch_size):
            stats_list.append({
                'success': False,
                'initial_prob': initial_probs[i].item(),
                'final_prob': initial_probs[i].item(),
                'found_at_iter': num_iter,
                'initial_prediction_correct': is_correct[i].item()
            })
        return torch.zeros_like(xb_batch, device=dev), stats_list
    
    loss_fn = nn.BCELoss(reduction='none')
    step_size = epsilon / 10.0
    
    success_mask = torch.zeros(batch_size, dtype=torch.bool, device=dev)
    success_iter = torch.full((batch_size,), num_iter, dtype=torch.long, device=dev)
    final_baselines = torch.zeros_like(xb_batch, device=dev)
    
    for iter_idx in range(num_iter):
        if not active_mask.any():
            break
        
        active_xb = adv_xb_batch[active_mask].detach().requires_grad_(True)
        
        outputs = model(active_xb)
        # Handle both tuple and tensor outputs
        if isinstance(outputs, tuple):
            active_probs = outputs[0]
        else:
            active_probs = outputs
        active_labels = yb_batch[active_mask]
        losses = loss_fn(active_probs, active_labels)
        loss = losses.mean()
        
        model.zero_grad()
        loss.backward()
        
        with torch.no_grad():
            grad_sign = active_xb.grad.sign()
            active_xb_new = active_xb + step_size * grad_sign
            
            active_indices = torch.where(active_mask)[0]
            
            for j, idx in enumerate(active_indices):
                delta = active_xb_new[j] - xb_batch[idx]
                delta = torch.clamp(delta, -epsilon, epsilon)
                adv_xb_batch[idx] = torch.clamp(xb_batch[idx] + delta, 0, 1)
            
            outputs = model(adv_xb_batch[active_mask])
            # Handle both tuple and tensor outputs
            if isinstance(outputs, tuple):
                current_probs = outputs[0]
            else:
                current_probs = outputs
            current_pred_classes = (current_probs > 0.5).float()
            
            flip_occurred = (current_pred_classes.squeeze(-1) != initial_pred_classes[active_mask].squeeze(-1))
            
            if flip_occurred.dim() == 0:
                if flip_occurred.item() and not success_mask[active_indices[0]]:
                    success_mask[active_indices[0]] = True
                    success_iter[active_indices[0]] = iter_idx + 1
                    final_baselines[active_indices[0]] = adv_xb_batch[active_indices[0]].clone()
                    active_mask[active_indices[0]] = False
            else:
                for j, idx in enumerate(active_indices):
                    if flip_occurred[j] and not success_mask[idx]:
                        success_mask[idx] = True
                        success_iter[idx] = iter_idx + 1
                        final_baselines[idx] = adv_xb_batch[idx].clone()
                        active_mask[idx] = False
    
    with torch.no_grad():
        outputs = model(adv_xb_batch)
        # Handle both tuple and tensor outputs
        if isinstance(outputs, tuple):
            final_probs = outputs[0]
        else:
            final_probs = outputs
    
    stats_list = []
    for i in range(batch_size):
        stats_list.append({
            'success': success_mask[i].item(),
            'initial_prob': initial_probs[i].item(),
            'final_prob': final_probs[i].item(),
            'found_at_iter': success_iter[i].item(),
            'initial_prediction_correct': is_correct[i].item()
        })
    
    return final_baselines, stats_list 