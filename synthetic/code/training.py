"""
Training methods for synthetic sequence experiments.
Includes standard training, HotFlip adversarial training, and randomized smoothing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from typing import Optional, Dict, Any, Tuple
from .utils import concentration_from_epsilon


# --------------------------------------------------------------------------- #
# Basic Training Utilities
# --------------------------------------------------------------------------- #

def validate_epoch(model, loader, loss_fn, dev):
    """Calculates the loss on a validation set."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        for xb, yb, _ in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            with autocast():
                logits, _ = model(xb)
                loss = loss_fn(logits, yb)
            total_loss += loss.item()
            num_batches += 1
    return total_loss / num_batches if num_batches > 0 else 0


# --------------------------------------------------------------------------- #
# Standard Training
# --------------------------------------------------------------------------- #

def train_standard(model, train_loader, val_loader, loss_fn, optimizer, dev, scaler, writer, scheduler, 
                   epochs: int = 10, early_stopping_patience: int = 15, early_stopping_min_delta: float = 1e-4,
                   warmup_epochs: int = 5, early_stop_start_epoch: int = 60) -> None:
    """
    Standard training loop with warmup and delayed early stopping.
    
    Parameters:
    - model: Neural network model to train
    - train_loader: Training data loader
    - val_loader: Validation data loader
    - loss_fn: Loss function
    - optimizer: Optimizer
    - dev: Device (cuda/cpu)
    - scaler: GradScaler for mixed precision training
    - writer: TensorBoard SummaryWriter
    - scheduler: Learning rate scheduler
    - epochs: Number of training epochs
    - early_stopping_patience: Patience for early stopping
    - early_stopping_min_delta: Minimum improvement for early stopping
    - warmup_epochs: Number of warmup epochs
    - early_stop_start_epoch: Epoch to start early stopping
    """
    print(f"Starting standard training with warmup ({warmup_epochs} epochs) and early stopping after epoch {early_stop_start_epoch}")
    best_val_loss = float('inf')
    early_stopping_counter = 0
    
    # Setup warmup scheduler
    base_lr = optimizer.param_groups[0]['lr']
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
    )

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0
        for xb, yb, _ in train_loader:
            xb, yb = xb.to(dev), yb.to(dev)
            optimizer.zero_grad()
            with autocast():
                logits, _ = model(xb)
                loss = loss_fn(logits, yb)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            num_batches += 1
        
        avg_train_loss = total_loss / num_batches if num_batches > 0 else 0
        avg_val_loss = validate_epoch(model, val_loader, loss_fn, dev)

        # Handle learning rate scheduling
        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            scheduler.step(avg_val_loss)
            
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/validation', avg_val_loss, epoch)
        writer.add_scalar('LR/train', current_lr, epoch)
        print(f"  Epoch {epoch + 1}/{epochs}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, LR: {current_lr:.6f}")
        
        # Only start early stopping after specified epoch
        if epoch >= early_stop_start_epoch:
            # Check for improvement
            if (best_val_loss - avg_val_loss) > early_stopping_min_delta:
                best_val_loss = avg_val_loss
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
            
            if early_stopping_counter >= early_stopping_patience:
                print(f"  -> Early stopping at epoch {epoch + 1}")
                break
        else:
            # Still track best loss even before early stopping starts
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss


# --------------------------------------------------------------------------- #
# HotFlip Adversarial Training
# --------------------------------------------------------------------------- #

def generate_hotflip_examples_optimized(model, xb, yb, loss_fn, flip_fraction: float, 
                                       neighborhood_size: int = 20, penalize_nearby: bool = False):
    """
    Optimized HotFlip with optional neighborhood penalty.
    Iteratively flips bases to maximize loss.
    
    Parameters:
    - model: Neural network model
    - xb: Input batch
    - yb: Target labels
    - loss_fn: Loss function
    - flip_fraction: Fraction of sequence to flip
    - neighborhood_size: Size of neighborhood for penalty
    - penalize_nearby: Whether to penalize flips near previous flips
    """
    seq_len = xb.shape[2]
    k_flips = int(flip_fraction * seq_len)
    adv_xb = xb.clone()
    
    flipped_positions = set()
    
    for flip_idx in range(k_flips):
        adv_xb.requires_grad = True
        model.zero_grad()
        
        with autocast():
            logits, _ = model(adv_xb)
            loss = loss_fn(logits, yb)
        
        loss.backward()
        grad = adv_xb.grad.data
        
        # Compute saliency scores
        current_bases_onehot = (adv_xb > 0.5).float()
        grad_at_current_bases = (grad * current_bases_onehot).sum(dim=1, keepdim=True)
        saliency_scores = grad - grad_at_current_bases
        saliency_scores.masked_fill_(current_bases_onehot.bool(), -1e9)
        
        # Apply neighborhood penalty if requested
        if penalize_nearby and flipped_positions:
            penalty_mask = torch.zeros(xb.shape[0], seq_len, device=xb.device)
            for pos in flipped_positions:
                start = max(0, pos - neighborhood_size)
                end = min(seq_len, pos + neighborhood_size + 1)
                penalty_mask[:, start:end] = 1.0
            
            penalty_strength = 0.5 * (flip_idx / k_flips)
            saliency_scores -= penalty_strength * penalty_mask.unsqueeze(1) * saliency_scores.abs().max()
        
        # Find best flip
        best_flip_scores_per_pos, _ = saliency_scores.max(dim=1)
        best_pos_to_flip = best_flip_scores_per_pos.argmax(dim=1)
        best_new_base_idx = saliency_scores[range(len(xb)), :, best_pos_to_flip].argmax(dim=1)
        
        # Apply flip
        old_base_idx = adv_xb[range(len(xb)), :, best_pos_to_flip].argmax(dim=1)
        adv_xb = adv_xb.detach()
        adv_xb[range(len(xb)), old_base_idx, best_pos_to_flip] = 0.0
        adv_xb[range(len(xb)), best_new_base_idx, best_pos_to_flip] = 1.0
        
        # Track flipped positions
        for pos in best_pos_to_flip.cpu().numpy():
            flipped_positions.add(int(pos))
    
    return adv_xb


def generate_direct_hotflip_examples_optimized(model, xb, yb, loss_fn, flip_fraction: float):
    """
    One-shot "Direct" HotFlip implementation optimized for batch processing.
    Computes gradient once, finds the top-k flips, and applies them simultaneously.
    This is significantly faster than an iterative approach.
    
    Parameters:
    - model: Neural network model
    - xb: Input batch
    - yb: Target labels
    - loss_fn: Loss function
    - flip_fraction: Fraction of sequence to flip
    """
    seq_len = xb.shape[2]
    k_flips = int(flip_fraction * seq_len)
    
    if k_flips == 0:
        return xb.clone()
        
    adv_xb = xb.clone().requires_grad_(True)
    batch_size = xb.shape[0]

    # 1. Single forward/backward pass to get gradients
    model.zero_grad()
    with autocast():
        logits, _ = model(adv_xb)
        loss = loss_fn(logits, yb)
    loss.backward()
    grad = adv_xb.grad.data

    # 2. Saliency Score Calculation (vectorized)
    current_bases_mask = (adv_xb > 0.5)
    # grad_at_current is the gradient value for the current base at each position, broadcast across the 4 bases
    grad_at_current = (grad * adv_xb).sum(dim=1, keepdim=True)
    # Saliency is the change in loss, i.e., grad_for_new_base - grad_for_current_base
    saliency_scores = grad - grad_at_current
    saliency_scores.masked_fill_(current_bases_mask, -float('inf')) # Prevent flipping to the same base

    # 3. Find top-k flips non-iteratively
    # Find the best new base and its score for each position
    best_flip_scores_per_pos, best_new_base_idx_per_pos = saliency_scores.max(dim=1) # (B, L)
    
    # Now find the top k positions to flip among all L positions
    _, top_k_positions = torch.topk(best_flip_scores_per_pos, k=k_flips, dim=1) # (B, k)
    
    # 4. Apply all k flips in a batched manner
    adv_xb_final = xb.clone() # Apply flips to the original input
    batch_indices = torch.arange(batch_size, device=xb.device)[:, None]

    # Gather the new bases for the top-k positions
    top_k_new_bases = torch.gather(best_new_base_idx_per_pos, dim=1, index=top_k_positions)
    
    # Zero out the one-hot encoding at all positions that will be flipped
    adv_xb_final[batch_indices, :, top_k_positions] = 0.0

    # Set the new bases to 1 at the flipped positions using scatter
    adv_xb_final.scatter_(1, top_k_new_bases.unsqueeze(1), torch.ones(batch_size, 1, k_flips, device=xb.device))
    
    return adv_xb_final.detach()


def train_hotflip(
    model_class: nn.Module,
    X_train: np.ndarray, y_train: np.ndarray, masks_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray, masks_val: np.ndarray,
    batch_size: int, epochs: int, dev: torch.device,
    k_flips: int, n_augment: int,
    max_flip_fraction: float, use_scheduling: bool,
    warmup_epochs: int = 5, early_stop_start: int = 60
) -> Tuple[nn.Module, Dict]:
    """Train a model with iterative HotFlip adversarial examples."""
    print("Training with iterative HotFlip...")
    
    # Import SEQ_LEN from data module to avoid circular import
    from .data import VANILLA_SEQ_LEN, COMPLEX_SEQ_LEN
    # Determine sequence length based on data
    sample_batch = next(iter(train_loader))
    SEQ_LEN = sample_batch[0].shape[2]
    
    previous_val_loss = float('inf')
    early_stopping_counter = 0
    
    # Setup warmup scheduler
    base_lr = optimizer.param_groups[0]['lr']
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
    )

    # Training loop
    for epoch in range(epochs):
        model.train()
        
        # Determine current flip fraction if scheduling
        if use_scheduling:
            current_flip_fraction = max_flip_fraction * min(1.0, epoch / (epochs * 0.75))
        else:
            current_flip_fraction = max_flip_fraction
            
        # Determine current k_flips if scheduling
        if use_scheduling:
            current_k_flips = int(k_flips * min(1.0, epoch / (epochs * 0.75)))
        else:
            current_k_flips = k_flips
        
        adv_xb = generate_hotflip_examples_optimized(model, xb, yb, loss_fn, current_flip_fraction)
        optimizer.zero_grad()
        with autocast():
            logits_adv, _ = model(adv_xb)
            loss_adv = loss_fn(logits_adv, yb)

        scaler.scale(loss_adv).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss_adv.item()
        num_batches += 1

        avg_train_loss = total_loss / num_batches if num_batches > 0 else 0
        avg_val_loss = validate_epoch(model, val_loader, loss_fn, dev)
        
        # Handle learning rate scheduling
        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            if use_scheduling:
                scheduler.best = previous_val_loss
            scheduler.step(avg_val_loss)
            
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Loss/train_adversarial', avg_train_loss, epoch)
        writer.add_scalar('Loss/validation_adversarial', avg_val_loss, epoch)
        writer.add_scalar('LR/train_adversarial', current_lr, epoch)
        writer.add_scalar('Epsilon/train_adversarial', current_flip_fraction, epoch)
        print(f"  Epoch {epoch + 1}/{epochs}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Epsilon: {current_flip_fraction:.4f}, LR: {current_lr:.6f}")

        # Only start early stopping after specified epoch
        if epoch >= early_stop_start:
            # Early stopping
            if (previous_val_loss - avg_val_loss) > early_stopping_min_delta:
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
            
            if early_stopping_counter >= early_stopping_patience:
                print(f"  -> Early stopping at epoch {epoch + 1}")
                break
        
        previous_val_loss = avg_val_loss


def train_direct_hotflip(
    model_class: nn.Module,
    X_train: np.ndarray, y_train: np.ndarray, masks_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray, masks_val: np.ndarray,
    batch_size: int, epochs: int, dev: torch.device,
    k_flips: int, n_augment: int,
    use_scheduling: bool,
    warmup_epochs: int = 5, early_stop_start: int = 60
) -> Tuple[nn.Module, Dict]:
    """Train a model with direct (one-shot) HotFlip adversarial examples."""
    print("Training with direct HotFlip...")
    
    # Import SEQ_LEN from data module to avoid circular import
    from .data import VANILLA_SEQ_LEN, COMPLEX_SEQ_LEN
    # Determine sequence length based on data
    sample_batch = next(iter(train_loader))
    SEQ_LEN = sample_batch[0].shape[2]
    
    previous_val_loss = float('inf')
    early_stopping_counter = 0
    
    # Setup warmup scheduler
    base_lr = optimizer.param_groups[0]['lr']
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
    )

    # Training loop
    for epoch in range(epochs):
        model.train()
        
        # Determine current k_flips if scheduling
        if use_scheduling:
            current_k_flips = int(k_flips * min(1.0, epoch / (epochs * 0.75)))
        else:
            current_k_flips = k_flips
            
        adv_xb = generate_direct_hotflip_examples_optimized(model, xb, yb, loss_fn, current_flip_fraction)
        optimizer.zero_grad()
        with autocast():
            logits_adv, _ = model(adv_xb)
            loss_adv = loss_fn(logits_adv, yb)

        scaler.scale(loss_adv).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss_adv.item()
        num_batches += 1

        avg_train_loss = total_loss / num_batches if num_batches > 0 else 0
        avg_val_loss = validate_epoch(model, val_loader, loss_fn, dev)
        
        # Handle learning rate scheduling
        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            if use_scheduling:
                scheduler.best = previous_val_loss
            scheduler.step(avg_val_loss)
            
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Loss/train_direct_adversarial', avg_train_loss, epoch)
        writer.add_scalar('Loss/validation_direct_adversarial', avg_val_loss, epoch)
        writer.add_scalar('LR/train_direct_adversarial', current_lr, epoch)
        writer.add_scalar('Epsilon/train_direct_adversarial', current_flip_fraction, epoch)
        print(f"  Epoch {epoch + 1}/{epochs}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Epsilon: {current_flip_fraction:.4f}, LR: {current_lr:.6f}")

        # Only start early stopping after specified epoch
        if epoch >= early_stop_start:
            # Early stopping
            if (previous_val_loss - avg_val_loss) > early_stopping_min_delta:
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
            
            if early_stopping_counter >= early_stopping_patience:
                print(f"  -> Early stopping at epoch {epoch + 1}")
                break
        
        previous_val_loss = avg_val_loss


# --------------------------------------------------------------------------- #
# Randomized Smoothing Training (from toy_slurm.py)
# --------------------------------------------------------------------------- #

def generate_smoothed_batch(xb: torch.Tensor, concentration_major: float, dev: torch.device) -> torch.Tensor:
    """
    Generate a smoothed batch using Dirichlet noise.
    
    Parameters:
    - xb: Input batch (one-hot encoded)
    - concentration_major: Concentration parameter for Dirichlet distribution
    - dev: Device
    
    Returns:
    - Smoothed batch
    """
    batch_size, n_symbols, seq_len = xb.shape
    
    # Get indices of current bases
    current_bases = torch.argmax(xb, dim=1)  # (batch_size, seq_len)
    
    # Generate Dirichlet samples
    xb_smooth = torch.zeros_like(xb)
    for i in range(batch_size):
        for j in range(seq_len):
            current_base = current_bases[i, j].item()
            
            # Create concentration vector
            concentration = torch.ones(n_symbols) * 1.0
            concentration[current_base] = concentration_major
            
            # Sample from Dirichlet
            dirichlet_sample = torch.from_numpy(
                np.random.dirichlet(concentration.numpy())
            ).float().to(dev)
            
            xb_smooth[i, :, j] = dirichlet_sample
    
    return xb_smooth


def train_random_smoothing(model, train_loader, val_loader, loss_fn, optimizer, dev, scaler, writer, scheduler,
                          target_epsilon: float, epochs: int = 10, early_stopping_patience: int = 10, 
                          early_stopping_min_delta: float = 1e-4) -> None:
    """
    Training with randomized smoothing using Dirichlet noise.
    
    Parameters:
    - model: Neural network model to train
    - train_loader: Training data loader
    - val_loader: Validation data loader
    - loss_fn: Loss function
    - optimizer: Optimizer
    - dev: Device (cuda/cpu)
    - scaler: GradScaler for mixed precision training
    - writer: TensorBoard SummaryWriter
    - scheduler: Learning rate scheduler
    - target_epsilon: Target epsilon for randomized smoothing
    - epochs: Number of training epochs
    - early_stopping_patience: Patience for early stopping
    - early_stopping_min_delta: Minimum improvement for early stopping
    """
    concentration_major = concentration_from_epsilon(target_epsilon)
    print(f"Starting randomized smoothing training with epsilon={target_epsilon:.4f}, concentration={concentration_major:.2f}")
    
    best_val_loss = float('inf')
    early_stopping_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for xb, yb, _ in train_loader:
            xb, yb = xb.to(dev), yb.to(dev)
            
            # Generate smoothed batch
            xb_smooth = generate_smoothed_batch(xb, concentration_major, dev)
            
            optimizer.zero_grad()
            with autocast():
                logits, _ = model(xb_smooth)
                loss = loss_fn(logits, yb)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            num_batches += 1
        
        avg_train_loss = total_loss / num_batches if num_batches > 0 else 0
        avg_val_loss = validate_epoch(model, val_loader, loss_fn, dev)
        
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        writer.add_scalar('Loss/train_smoothing', avg_train_loss, epoch)
        writer.add_scalar('Loss/validation_smoothing', avg_val_loss, epoch)
        writer.add_scalar('LR/train_smoothing', current_lr, epoch)
        
        print(f"  Epoch {epoch + 1}/{epochs}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, LR: {current_lr:.6f}")
        
        # Early stopping
        if (best_val_loss - avg_val_loss) > early_stopping_min_delta:
            best_val_loss = avg_val_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
        
        if early_stopping_counter >= early_stopping_patience:
            print(f"  -> Early stopping at epoch {epoch + 1}")
            break 