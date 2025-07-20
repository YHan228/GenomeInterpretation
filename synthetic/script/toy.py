"""
Synthetic 1-kbp phenotype dataset:
    positives: high-GC background + one 60-bp causal block
    negatives: low-GC background, no causal block
CNN training + Integrated Gradients attribution quality
Author: <your-name>, 2025-06-29
"""

import itertools
import math
import os
import random
import string
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from captum.attr import IntegratedGradients, LayerGradCam
from torch.utils.data import DataLoader, Dataset, random_split
from torch.cuda.amp import autocast, GradScaler


# --------------------------------------------------------------------------- //
# 1. Configuration & Utilities
# --------------------------------------------------------------------------- #

WITH_CONFOUNDER = True # Global switch for GC-content difference


SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
EPSILONS = [0.001, 0.0025, 0.005, 0.01, 0.05]

def set_seeds(seed_value: int = 42) -> None:
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


set_seeds()  # initial seed

ALPH = np.array(list("ACGT"), dtype="U1")
to_ix = {b: i for i, b in enumerate(ALPH)}


def sample_background(length: int, gc: float) -> np.ndarray:
    """iid sampling with given GC content, returns char array"""
    p = np.array([(1 - gc) / 2, gc / 2, gc / 2, (1 - gc) / 2])  # A,C,G,T
    return np.random.choice(ALPH, size=length, p=p)


def random_chunk(length: int) -> np.ndarray:
    """60-bp random chunk with balanced GC ≈ 50 %"""
    return sample_background(length, GC_POS)


def mutate(chunk: np.ndarray, conservation: float, gc_target: float) -> np.ndarray:
    """Return a new chunk with given conservation level, with mutations
    sampled to match the target GC content."""
    mutated_chunk = chunk.copy()
    n_to_mutate = int(len(chunk) * (1.0 - conservation))
    pos_to_mutate = np.random.choice(len(chunk), n_to_mutate, replace=False)
    
    # Distribution for sampling new bases
    p = np.array([(1 - gc_target) / 2, gc_target / 2, gc_target / 2, (1 - gc_target) / 2])
    
    for pos in pos_to_mutate:
        original_base = mutated_chunk[pos]
        
        # Create a temporary probability distribution for the other 3 bases
        temp_p = p.copy()
        temp_p[to_ix[original_base]] = 0
        
        # If the sum is zero (shouldn't happen with 4 bases), fall back to uniform
        if temp_p.sum() == 0:
            mutated_chunk[pos] = np.random.choice(np.setdiff1d(ALPH, [original_base]))
        else:
            temp_p /= temp_p.sum() # Normalize to make it a probability distribution
            mutated_chunk[pos] = np.random.choice(ALPH, p=temp_p)

    return mutated_chunk


def embed(seq: np.ndarray, chunk: np.ndarray) -> Tuple[np.ndarray, int]:
    """Insert chunk at random non-overlapping position; return new seq and start idx"""
    L, l = len(seq), len(chunk)
    start = np.random.randint(0, L - l + 1)
    seq[start:start + l] = chunk
    return seq, start


def one_hot(seq: np.ndarray) -> np.ndarray:
    """(1000,) char → (4,1000) float32 one-hot"""
    arr = np.zeros((4, len(seq)), dtype=np.float32)
    for i, b in enumerate(seq):
        arr[to_ix[b], i] = 1.0
    return arr


def one_hot_to_seq(one_hot_tensor: torch.Tensor) -> str:
    """ (4, L) float tensor -> (L,) string """
    indices = torch.argmax(one_hot_tensor, dim=0).cpu().numpy()
    return "".join(ALPH[indices])


# --------------------------------------------------------------------------- #
# 2. Dataset generation
# --------------------------------------------------------------------------- #

SEQ_LEN = 1000
CHUNK_LEN = 60
N_TOTAL = 10000
POS_N = N_TOTAL // 2
NEG_N = N_TOTAL - POS_N

# Define GC content based on the global flag
GC_POS = 0.60 if WITH_CONFOUNDER else 0.50
GC_NEG = 0.50

X, y, masks = [], [], []

# The master chunk for positive examples must also have high GC content.
master_chunk = sample_background(CHUNK_LEN, gc=GC_POS)

for _ in range(POS_N):
    bg = sample_background(SEQ_LEN, gc=GC_POS)
    conservation = random.uniform(0.6, 0.8)
    chunk = mutate(master_chunk, conservation, gc_target=GC_POS)
    seq, start = embed(bg, chunk)
    X.append(one_hot(seq))
    y.append(1)
    mask = np.zeros(SEQ_LEN, dtype=bool)
    mask[start:start + CHUNK_LEN] = True
    masks.append(mask)

for _ in range(NEG_N):
    bg = sample_background(SEQ_LEN, gc=GC_NEG)
    X.append(one_hot(bg))
    y.append(0)
    masks.append(np.zeros(SEQ_LEN, dtype=bool))

X = torch.tensor(np.stack(X))          # (N,4,1000)
y = torch.tensor(y, dtype=torch.float) # (N,)
masks = np.stack(masks)                # (N,1000) bool


class SeqDS(Dataset):
    def __init__(self, xs, ys, ms):
        self.x, self.y, self.m = xs, ys, ms

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.m[idx]


ds = SeqDS(X, y, masks)
train_ds, test_ds = random_split(
    ds,
    [int(0.8 * N_TOTAL), N_TOTAL - int(0.8 * N_TOTAL)],
    generator=torch.Generator().manual_seed(42)
)
train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=128)


# --------------------------------------------------------------------------- #
# 3. Model definition
# --------------------------------------------------------------------------- #

class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # User-specified k1, with sensible defaults for subsequent layers
        self.k1, self.k2, self.k3 = 30, 3, 3

        # Calculate padding to keep sequence length constant *before* pooling
        p1 = (self.k1 - 1) // 2
        # Padding for subsequent layers operating on pooled output
        p2 = (self.k2 - 1) // 2
        p3 = (self.k3 - 1) // 2

        # Conv Block 1
        self.conv1 = nn.Conv1d(4, 32, kernel_size=self.k1, padding=p1)
        self.bn1 = nn.BatchNorm1d(32)
        self.dropout1 = nn.Dropout(0.1)

        # Conv Block 2
        self.conv2 = nn.Conv1d(32, 64, kernel_size=self.k2, padding=p2)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.1)

        # Conv Block 3
        self.conv3 = nn.Conv1d(64, 128, kernel_size=self.k3, padding=p3)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(0.1)

        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc_dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        # Conv Block 1: Motif scanning
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.exp(x)
        x = self.dropout1(x)
        
        # Localist pooling: Drastically downsample to get motif presence features
        x = F.max_pool1d(x, 50)

        # Conv Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        # Conv Block 3
        conv3_out = self.conv3(x)
        x = self.bn3(conv3_out)
        x = F.relu(x)
        x = self.dropout3(x)

        # FC Layer
        x = self.pool(x).squeeze(-1)
        x = self.fc_dropout(x)
        logits = self.fc(x)
        return logits.squeeze(-1), conv3_out
    
    def receptive_field(self) -> int:
        """
        For the localist architecture, the receptive field is conceptually the
        size of the initial motif scanners (k1), as subsequent layers operate
        on a heavily downsampled representation of motif presence.
        """
        return self.k1


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TinyCNN().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
bce = nn.BCEWithLogitsLoss()


# --------------------------------------------------------------------------- #
# 4. Training functions
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


def train_standard(model, train_loader, val_loader, loss_fn, optimizer, dev, scaler, scheduler, epochs: int = 10, early_stopping_patience: int = 15, early_stopping_min_delta: float = 1e-4) -> None:
    print(f"Starting standard training with early stopping (patience={early_stopping_patience}, min_delta={early_stopping_min_delta})...")
    best_val_loss = float('inf')
    early_stopping_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0
        for xb, yb, _ in train_loader:
            xb, yb = xb.to(dev), yb.to(dev)
            optimizer.zero_grad()
            with autocast():
                logits, conv_out = model(xb)
                loss = loss_fn(logits, yb)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            num_batches += 1
        
        avg_train_loss = total_loss / num_batches if num_batches > 0 else 0
        if not np.isfinite(avg_train_loss):
            print(f"  WARNING: NaN or Inf average train loss at epoch {epoch + 1}. Stopping training for this model.")
            break
            
        avg_val_loss = validate_epoch(model, val_loader, loss_fn, dev)

        scheduler.step(avg_val_loss)
        print(f"  Epoch {epoch + 1}/{epochs}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, LR: {scheduler.optimizer.param_groups[0]['lr']:.2E}")
        
        if (best_val_loss - avg_val_loss) > early_stopping_min_delta:
            best_val_loss = avg_val_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
        
        if early_stopping_counter >= early_stopping_patience:
            print(f"  -> Early stopping at epoch {epoch + 1} due to no improvement for {early_stopping_patience} epochs.")
            break


def generate_hotflip_examples(model, xb, yb, loss_fn, flip_fraction: float, 
                              neighborhood_size: int = 20, penalize_nearby: bool = False):
    seq_len = xb.shape[2]
    k_flips = int(flip_fraction * seq_len)
    adv_xb = xb.clone()
    forbidden_regions = torch.zeros_like(adv_xb[:, 0, :], dtype=torch.bool, device=xb.device)

    for _ in range(k_flips):
        adv_xb.requires_grad = True
        model.zero_grad()
        with autocast():
            logits, _ = model(adv_xb)
            loss = loss_fn(logits, yb)
        loss.backward()
        grad = adv_xb.grad.data
        current_bases_onehot = (adv_xb > 0.5).float()
        grad_at_current_bases = (grad * current_bases_onehot).sum(dim=1, keepdim=True)
        saliency_scores = grad - grad_at_current_bases
        saliency_scores.masked_fill_(current_bases_onehot.bool(), -1e9)
        best_flip_scores_per_pos, _ = saliency_scores.max(dim=1)
        if penalize_nearby:
            best_flip_scores_per_pos.masked_fill_(forbidden_regions, -1e9)
        best_pos_to_flip = best_flip_scores_per_pos.argmax(dim=1)
        best_new_base_idx = saliency_scores[range(len(xb)), :, best_pos_to_flip].argmax(dim=1)
        old_base_idx = adv_xb[range(len(xb)), :, best_pos_to_flip].argmax(dim=1)
        adv_xb = adv_xb.detach()
        adv_xb[range(len(xb)), old_base_idx, best_pos_to_flip] = 0.0
        adv_xb[range(len(xb)), best_new_base_idx, best_pos_to_flip] = 1.0
        if penalize_nearby:
            pos = best_pos_to_flip
            start = torch.clamp(pos - neighborhood_size, 0)
            end = torch.clamp(pos + neighborhood_size + 1, max=seq_len)
            indices = torch.arange(seq_len, device=xb.device).unsqueeze(0)
            newly_forbidden = (indices >= start.unsqueeze(1)) & (indices < end.unsqueeze(1))
            forbidden_regions |= newly_forbidden
    return adv_xb


def train_hotflip(model, train_loader, val_loader, loss_fn, optimizer, dev, scaler, scheduler,
                  max_flip_fraction: float, epochs: int = 10, use_scheduling: bool = True, early_stopping_patience: int = 25, early_stopping_min_delta: float = 1e-4) -> None:
    
    scheduling_str = "ON" if use_scheduling else "OFF"
    print(f"Starting HotFlip training with max_flip_fraction = {max_flip_fraction:.4f}, Scheduling: {scheduling_str}...")
    
    previous_val_loss = float('inf')
    early_stopping_counter = 0

    for epoch in range(epochs):
        
        max_flips = int(max_flip_fraction * SEQ_LEN)
        
        if use_scheduling:
            # Smart scheduling: Linearly ramp up from 1 flip to max_flips.
            num_flips = 1 + int(np.floor((max_flips - 1) * (epoch / (epochs - 1)))) if epochs > 1 else max_flips
            current_flip_fraction = num_flips / SEQ_LEN
        else:
            current_flip_fraction = max_flip_fraction
        
        model.train()
        total_loss = 0.0
        num_batches = 0
        for xb, yb, _ in train_loader:
            xb, yb = xb.to(dev), yb.to(dev)
            adv_xb = generate_hotflip_examples(model, xb, yb, loss_fn, current_flip_fraction)
            optimizer.zero_grad()
            with autocast():
                logits_adv, conv_out_adv = model(adv_xb)
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
        
        if use_scheduling:
            scheduler.best = previous_val_loss
        
        scheduler.step(avg_val_loss)

        print(f"  Epoch {epoch + 1}/{epochs}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Epsilon: {current_flip_fraction:.4f}, LR: {scheduler.optimizer.param_groups[0]['lr']:.2E}")

        if not np.isfinite(avg_train_loss):
            print(f"  WARNING: NaN or Inf average train loss at epoch {epoch + 1}. Stopping training for this model.")
            break

        if (previous_val_loss - avg_val_loss) > early_stopping_min_delta:
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
        
        previous_val_loss = avg_val_loss

        if early_stopping_counter >= early_stopping_patience:
            print(f"  -> Early stopping at epoch {epoch + 1} due to no improvement for {early_stopping_patience} consecutive epochs.")
            break


# --------------------------------------------------------------------------- #
# 5. Attribution-based evaluation (Integrated Gradients & Grad-CAM)
# --------------------------------------------------------------------------- #

def find_adversarial_baseline_pgd(model, xb: torch.Tensor, yb: torch.Tensor, dev: torch.device,
                                  num_iter: int = 20, epsilon: float = 0.1, step_size: float = 0.01):
    """
    Finds a baseline for IG using PGD to find a nearby adversarial example
    that flips the model's prediction. Returns the baseline and a stats dict.
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
        stats['final_logit'] = initial_logits.item() # Default final logit

    is_correct = initial_pred_class.item() == yb.item()
    stats['initial_prediction_correct'] = is_correct

    # We only run PGD if the initial prediction for a positive example is correct.
    if not is_correct or yb.item() == 0:
        return torch.zeros_like(xb, device=dev), stats

    loss_fn = nn.BCEWithLogitsLoss()
    
    # Use a more stable step size for sign-based PGD
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
            # Use sign() for more stable PGD updates
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
    
    # If no flip was found, final logit is the last one computed
    with torch.no_grad():
        final_logits, _ = model(adv_xb)
        stats['final_logit'] = final_logits.item()

    return torch.zeros_like(xb, device=dev), stats


def evaluate_model(model, model_name: str, test_ds, dev, produce_plots: bool = True):
    print(f"Evaluating model: {model_name}")
    SAMPLE_N = 100
    ANALYSIS_CHUNK_LEN = 60  # assumed window size

    # -- accuracy ----------------------------------------------------------------
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
    print(f"Test accuracy: {accuracy:.3f}")

    # -- Integrated Gradients ----------------------------------------------------
    def model_for_captum(x):
        with autocast():
            return model(x)[0].unsqueeze(-1)

    ig = IntegratedGradients(model_for_captum)

    positive_subset_indices = [
        i for i, original_idx in enumerate(test_ds.indices)
        if test_ds.dataset.m[original_idx].sum() > 0
    ]

    rng = np.random.default_rng(0)
    sample_n_actual = min(SAMPLE_N, len(positive_subset_indices))
    if sample_n_actual < SAMPLE_N:
        print(f"Warning: only {sample_n_actual} positive samples found "
              f"(requested {SAMPLE_N}).")
    if sample_n_actual == 0:
        print("Warning: No positive samples in test set for evaluation.")
        return 0.0, accuracy, 0.0, 0.0
        
    idxs = rng.choice(positive_subset_indices,
                      size=sample_n_actual,
                      replace=False)

    results = []
    pgd_results = []
    for idx in idxs:
        xb, y_scalar, mask = test_ds[idx]
        xb = xb.unsqueeze(0).to(dev)
        yb = torch.tensor([y_scalar], device=dev, dtype=torch.float)

        # Two-stage baseline strategy:
        # 1. Try to find a decision-boundary point using PGD.
        pgd_baseline, pgd_stat = find_adversarial_baseline_pgd(model, xb, yb, dev)
        pgd_results.append(pgd_stat)

        # 2. If PGD succeeds, use the adversarial example as the baseline.
        #    If it fails, fall back to the sequence's own nucleotide composition.
        if pgd_stat['success']:
            baseline = pgd_baseline
        else:
            proportions = xb.mean(dim=2, keepdim=True)
            baseline = proportions.expand_as(xb)

        raw_attributions = ig.attribute(xb, baselines=baseline, target=0)
        
        # Apply the gradient correction from Majdandzic et al., Genome Biology 2023.
        # This subtracts the mean attribution across all nucleotides at each position.
        corrected_attributions = raw_attributions - raw_attributions.mean(dim=1, keepdim=True)
        
        # Calculate final contribution scores using the corrected attributions,
        # taking the absolute value after projection as is standard for IG.
        attributions = np.abs((corrected_attributions * xb).sum(1).squeeze(0).cpu().numpy())

        # 1. contiguous Overlap
        window_sums = np.convolve(attributions,
                                  np.ones(ANALYSIS_CHUNK_LEN),
                                  mode='valid')
        best_window_start = np.argmax(window_sums)
        pred_mask_cont = np.zeros(SEQ_LEN, dtype=bool)
        pred_mask_cont[
            best_window_start:best_window_start + ANALYSIS_CHUNK_LEN
        ] = True
        inter_cont = (pred_mask_cont & mask).sum()
        union_cont = (pred_mask_cont | mask).sum()
        iou_cont = inter_cont / union_cont if union_cont else 0

        # 2. Saliency AUC
        inside_scores = attributions[mask]
        outside_scores = attributions[~mask]
        # Efficiently calculate AUC: probability that a random inside score is > a random outside score
        saliency_auc = (inside_scores[:, None] > outside_scores[None, :]).mean()

        # 3. Saliency Signal-to-Noise Ratio (fraction of energy in motif)
        sum_sq_inside = np.sum(inside_scores**2)
        sum_sq_total = np.sum(attributions**2)
        saliency_snr = sum_sq_inside / (sum_sq_total + 1e-9)

        results.append(
            dict(iou_cont=iou_cont,
                 saliency_auc=saliency_auc,
                 saliency_snr=saliency_snr,
                 attributions=attributions,
                 mask=mask,
                 cont_start=best_window_start)
        )

    if not results:
        print("No positives in sample set – increase SAMPLE_N.")
        return 0.0, accuracy, 0.0, 0.0

    # -- statistics --------------------------------------------------------------
    results.sort(key=lambda r: r['iou_cont'])
    mean_iou_cont = np.mean([r['iou_cont'] for r in results])
    mean_saliency_auc = np.mean([r['saliency_auc'] for r in results])
    mean_saliency_snr = np.mean([r['saliency_snr'] for r in results])
    print(f"Mean Overlap : {mean_iou_cont:.3f} on {len(results)} positive samples")
    print(f"Mean Saliency AUC: {mean_saliency_auc:.3f}")
    print(f"Mean Saliency SNR: {mean_saliency_snr:.3f}")

    if not produce_plots:
        return mean_iou_cont, accuracy, mean_saliency_auc, mean_saliency_snr

    # -- plotting ----------------------------------------------------------------
    fig, axs = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle(
        f'IG scores vs. position ({model_name.title()}, sorted by Overlap)'
    )
    n_res = len(results)
    mid1, mid2 = n_res // 2 - 1, n_res // 2
    plot_data = [results[0], results[1],
                 results[mid1], results[mid2],
                 results[-2], results[-1]]
    titles = ['Worst 1', 'Worst 2',
              'Median 1', 'Median 2',
              'Best 2', 'Best 1']

    for ax, data, title in zip(axs.flat, plot_data, titles):
        ax.plot(data['attributions'],
                label='IG score',
                color='black',
                linewidth=0.7)
        ax.set_title(f"{title}\nOverlap={data['iou_cont']:.3f}, AUC={data['saliency_auc']:.3f}, SNR={data['saliency_snr']:.2f}")
        ax.set_xlabel("Position")
        ax.set_ylabel("IG score")
        ax.grid(True, ls='--', alpha=0.6)

        # highlight true & predicted block
        gt_start = np.where(data['mask'])[0][0]
        ax.axvspan(gt_start,
                   gt_start + CHUNK_LEN,
                   color='red',
                   alpha=0.2,
                   lw=0,
                   label=f'Ground truth ({CHUNK_LEN} bp)')
        pred_start = data['cont_start']
        ax.axvspan(pred_start,
                   pred_start + ANALYSIS_CHUNK_LEN,
                   color='blue',
                   alpha=0.2,
                   lw=0,
                   label=f'Predicted ({ANALYSIS_CHUNK_LEN} bp)')
        ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{model_name}_ig_scores_plot.png")
    print(f"Saved plot → {model_name}_ig_scores_plot.png")

    # -- distribution plots ------------------------------------------------------
    wious = [r['iou_cont'] for r in results]
    saliency_aucs = [r['saliency_auc'] for r in results]
    saliency_snrs = [r['saliency_snr'] for r in results]

    fig_dist, axs_dist = plt.subplots(1, 3, figsize=(18, 5))
    fig_dist.suptitle(f'Evaluation Metric Distributions ({model_name.title()})')

    axs_dist[0].hist(wious, bins=20, alpha=0.75)
    axs_dist[0].set_title('Overlap')
    axs_dist[0].set_xlabel('Score')
    axs_dist[0].set_ylabel('Frequency')
    axs_dist[0].grid(True, ls='--', alpha=0.6)

    axs_dist[1].hist(saliency_aucs, bins=20, alpha=0.75)
    axs_dist[1].set_title('Saliency AUC')
    axs_dist[1].set_xlabel('Score')
    axs_dist[1].grid(True, ls='--', alpha=0.6)

    axs_dist[2].hist(saliency_snrs, bins=20, alpha=0.75, range=(0, 1.0))
    axs_dist[2].set_title('Saliency SNR')
    axs_dist[2].set_xlabel('Score')
    axs_dist[2].grid(True, ls='--', alpha=0.6)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{model_name}_metric_distributions.png")
    print(f"Saved plot → {model_name}_metric_distributions.png")

    return mean_iou_cont, accuracy, mean_saliency_auc, mean_saliency_snr


def analyze_adversarial_examples(model, test_ds, dev, loss_fn, flip_fraction: float):
    print("\n--- Adversarial Example Analysis ---")
    k_flips = int(flip_fraction * SEQ_LEN)
    positive_indices = [i for i, (_, y, _) in enumerate(test_ds) if y == 1]
    
    if not positive_indices:
        print("No positive examples found for analysis.")
        return

    counts = {
        'chunk': {'at_to_gc': 0, 'gc_to_at': 0, 'at_to_at': 0, 'gc_to_gc': 0},
        'bg': {'at_to_gc': 0, 'gc_to_at': 0, 'at_to_at': 0, 'gc_to_gc': 0}
    }
    gc_bases = {'G', 'C'}
    all_flip_distances = []

    print(f"  Analyzing flips for {len(positive_indices)} positive samples...")
    for idx in positive_indices:
        xb, _, mask = test_ds[idx]
        xb = xb.unsqueeze(0).to(dev)
        yb = torch.tensor([1.0], device=dev)
        
        adv_xb = generate_hotflip_examples(model, xb, yb, loss_fn, flip_fraction)
        
        original_seq_str = one_hot_to_seq(xb.squeeze(0))
        adv_seq_str = one_hot_to_seq(adv_xb.squeeze(0))

        flipped_indices = [i for i, (c1, c2) in enumerate(zip(original_seq_str, adv_seq_str)) if c1 != c2]

        if len(flipped_indices) > 1:
            sorted_indices = np.sort(flipped_indices)
            distances = np.diff(sorted_indices)
            all_flip_distances.extend(distances)

        for flip_idx in flipped_indices:
            loc = 'chunk' if mask[flip_idx] else 'bg'
            old_base, new_base = original_seq_str[flip_idx], adv_seq_str[flip_idx]
            old_is_gc, new_is_gc = old_base in gc_bases, new_base in gc_bases

            if not old_is_gc and new_is_gc:
                counts[loc]['at_to_gc'] += 1
            elif old_is_gc and not new_is_gc:
                counts[loc]['gc_to_at'] += 1
            elif not old_is_gc and not new_is_gc:
                counts[loc]['at_to_at'] += 1
            elif old_is_gc and new_is_gc:
                counts[loc]['gc_to_gc'] += 1

    # --- New stacked bar plot for HotFlip analysis ---
    total_flips = sum(sum(d.values()) for d in counts.values())
    if total_flips == 0:
        print("No flips were made during adversarial generation, skipping plot.")
        return

    labels = ['In Causal Chunk', 'In Background']
    data = {
        'AT → GC': [counts['chunk']['at_to_gc'], counts['bg']['at_to_gc']],
        'GC → AT': [counts['chunk']['gc_to_at'], counts['bg']['gc_to_at']],
        'AT → AT': [counts['chunk']['at_to_at'], counts['bg']['at_to_at']],
        'GC → GC': [counts['chunk']['gc_to_gc'], counts['bg']['gc_to_gc']],
    }
    
    total_chunk_flips = sum(counts['chunk'].values())
    total_bg_flips = sum(counts['bg'].values())
    
    labels = [f'In Causal Chunk\n(N={total_chunk_flips})', 
              f'In Background\n(N={total_bg_flips})']

    percentages = {key: [0.0, 0.0] for key in data}
    if total_chunk_flips > 0:
        for key in data:
            percentages[key][0] = 100 * data[key][0] / total_chunk_flips
    if total_bg_flips > 0:
        for key in data:
            percentages[key][1] = 100 * data[key][1] / total_bg_flips
            
    fig, ax = plt.subplots(figsize=(10, 7))
    bottom = np.zeros(len(labels))
    colors = {'AT → GC': '#2ca02c', 'GC → AT': '#d62728', 'AT → AT': '#1f77b4', 'GC → GC': '#ff7f0e'}

    for flip_type, values in percentages.items():
        p = ax.bar(labels, values, width=0.5, bottom=bottom, label=flip_type, color=colors[flip_type])
        raw_counts = data[flip_type]
        for i, (p_val, r_count) in enumerate(zip(values, raw_counts)):
            if p_val > 4:  # Add text only if segment is large enough
                y_pos = bottom[i] + p_val / 2
                ax.text(i, y_pos, str(r_count), ha='center', va='center', color='white', fontsize=10, fontweight='bold')
        bottom += values

    ax.set_ylabel('Percentage of Flips within Location (%)')
    ax.set_title(f'Composition of HotFlip Attacks (k={k_flips}, Total Flips: {total_flips})')
    ax.legend(title='Flip Type', bbox_to_anchor=(1.04, 1), loc='upper left')
    ax.set_ylim(0, 105)
    ax.grid(True, linestyle='--', alpha=0.6, axis='y')

    plt.tight_layout(rect=[0, 0, 0.85, 0.95])
    plt.savefig("hotflip_attack_composition.png")
    print("\nSaved plot → hotflip_attack_composition.png")

    if all_flip_distances:
        fig_dist, ax_dist = plt.subplots(figsize=(10, 6))
        neighborhood_size = 20  # From generate_hotflip_examples default
        ax_dist.hist(all_flip_distances, bins=50, range=(0, 200), label=f'Distances (k={k_flips})')
        ax_dist.axvline(neighborhood_size, color='r', linestyle='--', 
                        label=f'Neighborhood Penalty ({neighborhood_size} bp)')
        ax_dist.set_title('Distribution of Distances Between Consecutive Flips')
        ax_dist.set_xlabel('Distance (bp)')
        ax_dist.set_ylabel('Frequency')
        ax_dist.legend()
        ax_dist.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig("hotflip_flip_distances.png")
        print("Saved plot → hotflip_flip_distances.png")


# --------------------------------------------------------------------------- #
# 6. Experiment helpers
# --------------------------------------------------------------------------- #

def run_single_experiment(seed: int, epsilons_to_test: List[float], main_train_ds, test_ds):
    """
    1. Generates data for the seed
    2. Trains a standard model and evaluates wIoU & acc
    3. Trains robust models for each epsilon, evaluates each
    """
    print(f"\n{'=' * 20}  SEED {seed}  {'=' * 20}")
    
    # Create a validation set from the main training data
    train_size = int(0.9 * len(main_train_ds))
    val_size = len(main_train_ds) - train_size
    train_ds, val_ds = random_split(main_train_ds, [train_size, val_size], generator=torch.Generator().manual_seed(seed))

    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=128, num_workers=4, pin_memory=True)
    
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bce = nn.BCEWithLogitsLoss()
    scaler = GradScaler()
    epochs = 100 # Increased epochs for better convergence with scheduler

    # --- Standard model ---
    set_seeds(seed)
    standard_model = TinyCNN().to(dev)
    opt_standard = torch.optim.AdamW(standard_model.parameters(), lr=3e-4, weight_decay=1e-6)
    std_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_standard, 'min', factor=0.5, patience=8, verbose=True)
    
    train_standard(standard_model, train_dl, val_dl, bce, opt_standard, dev, scaler, std_scheduler, epochs=epochs)
    std_wiou, std_acc, std_auc, std_snr = evaluate_model(standard_model,
                                       f"standard_seed{seed}",
                                       test_ds,
                                       dev,
                                       produce_plots=False)

    # --- Robust models ---
    robust_wious, robust_accs, robust_aucs, robust_snrs = [], [], [], []
    for eps in epsilons_to_test:
        set_seeds(seed)
        mdl = TinyCNN().to(dev)
        opt = torch.optim.AdamW(mdl.parameters(), lr=3e-4, weight_decay=1e-6)
        rob_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', factor=0.5, patience=8, verbose=True)

        if eps == 0:
            print("Skipping HotFlip for eps=0, copying standard model results.")
            robust_wious.append(std_wiou)
            robust_accs.append(std_acc)
            robust_aucs.append(std_auc)
            robust_snrs.append(std_snr)
            continue
            
        train_hotflip(mdl, train_dl, val_dl, bce, opt, dev, scaler, rob_scheduler, 
                      max_flip_fraction=eps, epochs=epochs, use_scheduling=True)
        
        k_flips_for_name = int(eps * SEQ_LEN)
        wio, acc, auc, snr = evaluate_model(mdl,
                                  f"robust_k{k_flips_for_name}_seed{seed}",
                                  test_ds,
                                  dev,
                                  produce_plots=False)
        robust_wious.append(wio)
        robust_accs.append(acc)
        robust_aucs.append(auc)
        robust_snrs.append(snr)

    return std_wiou, std_acc, std_auc, std_snr, robust_wious, robust_accs, robust_aucs, robust_snrs


# --------------------------------------------------------------------------- #
# 7. Main entry-point
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    
    # Generate the single, large dataset for all experiments
    print(f"--- Generating a single dataset of size {N_TOTAL} ---")
    set_seeds(42) # Use a fixed seed for dataset generation
    master_chunk = sample_background(CHUNK_LEN, gc=GC_POS)
    X, y, masks = [], [], []
    for _ in range(POS_N):
        bg = sample_background(SEQ_LEN, gc=GC_POS)
        conservation = random.uniform(0.6, 0.9)
        chunk = mutate(master_chunk, conservation, gc_target=GC_POS)
        seq, start = embed(bg, chunk)
        X.append(one_hot(seq)); y.append(1)
        m = np.zeros(SEQ_LEN, dtype=bool); m[start:start + CHUNK_LEN] = True; masks.append(m)
    for _ in range(NEG_N):
        bg = sample_background(SEQ_LEN, gc=GC_NEG)
        X.append(one_hot(bg)); y.append(0)
        masks.append(np.zeros(SEQ_LEN, dtype=bool))

    X = torch.tensor(np.stack(X)); y = torch.tensor(y, dtype=torch.float); masks = np.stack(masks)
    ds = SeqDS(X, y, masks)
    
    # This split is now done once for all experiments
    main_train_ds, main_test_ds = random_split(
        ds,
        [int(0.8 * N_TOTAL), N_TOTAL - int(0.8 * N_TOTAL)],
        generator=torch.Generator().manual_seed(42)
    )

    print("--- single baseline run for visualisation ---")
    set_seeds(0)
    
    # Create a validation set from the main training data just for this viz run
    viz_train_size = int(0.9 * len(main_train_ds))
    viz_val_size = len(main_train_ds) - viz_train_size
    viz_train_ds, viz_val_ds = random_split(main_train_ds, [viz_train_size, viz_val_size], generator=torch.Generator().manual_seed(0))

    viz_train_dl = DataLoader(viz_train_ds, batch_size=64, shuffle=True)
    viz_val_dl = DataLoader(viz_val_ds, batch_size=128)

    # --- Standard model for visualization ---
    viz_model = TinyCNN().to(device)
    viz_opt = torch.optim.AdamW(viz_model.parameters(), lr=3e-4, weight_decay=1e-6)
    viz_scaler = GradScaler()
    epochs_viz = 100
    viz_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(viz_opt, 'min', factor=0.5, patience=8, verbose=True)

    train_standard(viz_model, viz_train_dl, viz_val_dl, bce, viz_opt, device, viz_scaler, viz_scheduler, epochs=epochs_viz)
    evaluate_model(viz_model, "standard_baseline", main_test_ds, device, True)
    
    # --- Adversarial Analysis Section ---
    print("\n--- Training a single HotFlip model for analysis ---")
    set_seeds(0)
    hotflip_model_for_analysis = TinyCNN().to(device)
    hotflip_opt = torch.optim.AdamW(hotflip_model_for_analysis.parameters(), lr=3e-4, weight_decay=1e-6)
    hotflip_scaler = GradScaler()
    hotflip_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(hotflip_opt, 'min', factor=0.5, patience=8, verbose=True)

    analysis_flip_fraction = 0.01 # Using 1% flips (k=10)
    train_hotflip(hotflip_model_for_analysis, viz_train_dl, viz_val_dl, bce, hotflip_opt, device, hotflip_scaler, hotflip_scheduler, max_flip_fraction=analysis_flip_fraction, epochs=epochs_viz, use_scheduling=True)
    analyze_adversarial_examples(hotflip_model_for_analysis, main_test_ds, device, bce, flip_fraction=analysis_flip_fraction)
    evaluate_model(hotflip_model_for_analysis, "hotflip_baseline", main_test_ds, device, True)

    print("\n--- baseline plots generated. multi-seed experiments start ---\n")

    all_std_wious, all_std_accs, all_std_aucs, all_std_snrs = [], [], [], []
    all_rob_wious, all_rob_accs, all_rob_aucs, all_rob_snrs = [], [], [], []

    for sd in SEEDS:
        sw, sa, sa_auc, sa_snr, rw, ra, ra_auc, ra_snr = run_single_experiment(sd, EPSILONS, main_train_ds, main_test_ds)
        all_std_wious.append(sw)
        all_std_accs.append(sa)
        all_std_aucs.append(sa_auc)
        all_std_snrs.append(sa_snr)
        all_rob_wious.append(rw)
        all_rob_accs.append(ra)
        all_rob_aucs.append(ra_auc)
        all_rob_snrs.append(ra_snr)

    # aggregate - wIoU
    std_wiou_mean = np.mean(all_std_wious)
    std_wiou_std = np.std(all_std_wious)
    rob_wious_arr = np.array(all_rob_wious)
    rob_mean = rob_wious_arr.mean(axis=0)
    rob_std = rob_wious_arr.std(axis=0)

    # aggregate - accuracy
    std_acc_mean = np.mean(all_std_accs)
    std_acc_std = np.std(all_std_accs)
    rob_acc_arr = np.array(all_rob_accs)
    rob_acc_mean = rob_acc_arr.mean(axis=0)
    rob_acc_std = rob_acc_arr.std(axis=0)

    # aggregate - saliency AUC
    std_auc_mean = np.mean(all_std_aucs)
    std_auc_std = np.std(all_std_aucs)
    rob_auc_arr = np.array(all_rob_aucs)
    rob_auc_mean = rob_auc_arr.mean(axis=0)
    rob_auc_std = rob_auc_arr.std(axis=0)

    # aggregate - saliency SNR
    std_snr_mean = np.mean(all_std_snrs)
    std_snr_std = np.std(all_std_snrs)
    rob_snr_arr = np.array(all_rob_snrs)
    rob_snr_mean = rob_snr_arr.mean(axis=0)
    rob_snr_std = rob_snr_arr.std(axis=0)

    # plot Overlap vs eps
    plt.figure(figsize=(12, 8))
    plt.plot(EPSILONS, rob_mean, marker='o', label='Robust mean')
    plt.fill_between(EPSILONS, rob_mean - rob_std, rob_mean + rob_std, alpha=0.2)
    plt.axhline(std_wiou_mean, color='r', ls='--',
                label=f'Standard mean ({std_wiou_mean:.3f})')
    plt.fill_between(EPSILONS,
                     std_wiou_mean - std_wiou_std,
                     std_wiou_mean + std_wiou_std,
                     color='r', alpha=0.1)
    plt.xscale('log')
    plt.xlabel('Epsilon (Fraction of Sequence Flipped)')
    plt.ylabel('Mean Overlap')
    plt.title('Epsilon vs Overlap (10 seeds)')
    plt.grid(True, which='both', ls='--')
    plt.legend()
    plt.savefig("multi_seed_fgsm_vs_overlap.png")
    print("Saved plot → multi_seed_fgsm_vs_overlap.png")

    # plot accuracy vs eps
    plt.figure(figsize=(12, 8))
    plt.plot(EPSILONS, rob_acc_mean, marker='o', label='Robust mean acc')
    plt.fill_between(EPSILONS, rob_acc_mean - rob_acc_std,
                     rob_acc_mean + rob_acc_std, alpha=0.2)
    plt.axhline(std_acc_mean, color='r', ls='--',
                label=f'Standard mean ({std_acc_mean:.3f})')
    plt.fill_between(EPSILONS,
                     std_acc_mean - std_acc_std,
                     std_acc_mean + std_acc_std,
                     color='r', alpha=0.1)
    plt.xscale('log')
    plt.xlabel('Epsilon (Fraction of Sequence Flipped)')
    plt.ylabel('Accuracy')
    plt.title('Epsilon vs accuracy (10 seeds)')
    plt.grid(True, which='both', ls='--')
    plt.legend()
    plt.savefig("multi_seed_fgsm_vs_acc.png")
    print("Saved plot → multi_seed_fgsm_vs_acc.png")

    # plot saliency AUC vs eps
    plt.figure(figsize=(12, 8))
    plt.plot(EPSILONS, rob_auc_mean, marker='o', label='Robust mean')
    plt.fill_between(EPSILONS, rob_auc_mean - rob_auc_std, rob_auc_mean + rob_auc_std, alpha=0.2)
    plt.axhline(std_auc_mean, color='r', ls='--', label=f'Standard mean ({std_auc_mean:.3f})')
    plt.fill_between(EPSILONS, std_auc_mean - std_auc_std, std_auc_mean + std_auc_std, color='r', alpha=0.1)
    plt.xscale('log')
    plt.xlabel('Epsilon (Fraction of Sequence Flipped)')
    plt.ylabel('Mean Saliency AUC')
    plt.title('Epsilon vs Saliency AUC (10 seeds)')
    plt.grid(True, which='both', ls='--')
    plt.legend()
    plt.savefig("multi_seed_fgsm_vs_saliency_auc.png")
    print("Saved plot → multi_seed_fgsm_vs_saliency_auc.png")

    # plot saliency SNR vs eps
    plt.figure(figsize=(12, 8))
    plt.plot(EPSILONS, rob_snr_mean, marker='o', label='Robust mean')
    plt.fill_between(EPSILONS, rob_snr_mean - rob_snr_std, rob_snr_mean + rob_snr_std, alpha=0.2)
    plt.axhline(std_snr_mean, color='r', ls='--', label=f'Standard mean ({std_snr_mean:.3f})')
    plt.fill_between(EPSILONS, std_snr_mean - std_snr_std, std_snr_mean + std_snr_std, color='r', alpha=0.1)
    plt.xscale('log')
    plt.xlabel('Epsilon (Fraction of Sequence Flipped)')
    plt.ylabel('Mean Saliency SNR')
    plt.title('Epsilon vs Saliency SNR (10 seeds)')
    plt.grid(True, which='both', ls='--')
    plt.legend()
    plt.savefig("multi_seed_fgsm_vs_saliency_snr.png")
    print("Saved plot → multi_seed_fgsm_vs_saliency_snr.png")
