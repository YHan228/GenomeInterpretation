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
    return sample_background(length, 0.50)


def mutate(chunk: np.ndarray, conservation: float) -> np.ndarray:
    """Return a new chunk with given conservation level (≈ %identity)"""
    mutated_chunk = chunk.copy()
    n_to_mutate = int(len(chunk) * (1.0 - conservation))
    pos_to_mutate = np.random.choice(len(chunk), n_to_mutate, replace=False)
    for pos in pos_to_mutate:
        original_base = mutated_chunk[pos]
        mutated_chunk[pos] = np.random.choice(np.setdiff1d(ALPH, [original_base]))
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
N_TOTAL = 5000
POS_N = N_TOTAL // 2
NEG_N = N_TOTAL - POS_N

# Define GC content based on the global flag
GC_POS = 0.60 if WITH_CONFOUNDER else 0.50
GC_NEG = 0.50

X, y, masks = [], [], []

master_chunk = random_chunk(CHUNK_LEN)

for _ in range(POS_N):
    bg = sample_background(SEQ_LEN, gc=GC_POS)
    conservation = random.uniform(0.6, 0.8)
    chunk = mutate(master_chunk, conservation)
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
        self.conv1 = nn.Conv1d(4, 32, 13, padding=6)
        self.conv2 = nn.Conv1d(32, 64, 7, padding=3)
        self.conv3 = nn.Conv1d(64, 128, 7, padding=3)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc   = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x)); x = F.max_pool1d(x, 2)
        x = F.relu(self.conv2(x)); x = F.max_pool1d(x, 2)
        x = F.relu(self.conv3(x)); x = F.max_pool1d(x, 2)
        x = self.pool(x).squeeze(-1)
        logits = self.fc(x) # Return raw logits
        return logits.squeeze(-1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TinyCNN().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
bce = nn.BCEWithLogitsLoss()


# --------------------------------------------------------------------------- #
# 4. Training functions
# --------------------------------------------------------------------------- #

def train_standard(model, loader, loss_fn, optimizer, dev, epochs: int = 10) -> None:
    print("Starting standard training...")
    for epoch in range(epochs):
        model.train()
        for xb, yb, _ in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
        print(f"  Epoch {epoch + 1}/{epochs} completed.")


def generate_hotflip_examples(model, xb, yb, loss_fn, flip_fraction: float, 
                              neighborhood_size: int = 20, penalize_nearby: bool = False):
    """
    Generates adversarial examples using iterative, greedy HotFlip.
    Includes optional penalty for flipping bases in a local neighborhood.
    """
    seq_len = xb.shape[2]
    k_flips = int(flip_fraction * seq_len)

    if flip_fraction > 0.05:
        print(f"Warning: HotFlip is changing {k_flips}/{seq_len} ({flip_fraction:.1%}) "
              f"of bases, which is > 5%. This may be unrealistic.")

    adv_xb = xb.clone()
    forbidden_regions = torch.zeros_like(adv_xb[:, 0, :], dtype=torch.bool, device=xb.device)

    for _ in range(k_flips):
        adv_xb.requires_grad = True
        model.zero_grad()
        logits = model(adv_xb)
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
            for i in range(len(xb)):
                pos = best_pos_to_flip[i].item()
                start = max(0, pos - neighborhood_size)
                end = min(seq_len, pos + neighborhood_size + 1)
                forbidden_regions[i, start:end] = True

    return adv_xb


def train_hotflip(model, loader, loss_fn, optimizer, dev,
                  flip_fraction: float, epochs: int = 10) -> None:
    print(f"Starting HotFlip training with flip_fraction = {flip_fraction:.4f} ...")
    for epoch in range(epochs):
        model.train()
        for xb, yb, _ in loader:
            xb, yb = xb.to(dev), yb.to(dev)

            # Generate adversarial examples for the batch
            adv_xb = generate_hotflip_examples(model, xb, yb, loss_fn, flip_fraction)
            
            optimizer.zero_grad()
            logits_adv = model(adv_xb)
            loss_adv = loss_fn(logits_adv, yb)
            loss_adv.backward()
            optimizer.step()
        print(f"  Epoch {epoch + 1}/{epochs} completed.")


def train_standard_verbose(model, loader, loss_fn, optimizer, dev, epochs: int = 10) -> None:
    print("Starting verbose standard training...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_samples = 0
        num_batches = len(loader)
        for i, (xb, yb, _) in enumerate(loader):
            xb, yb = xb.to(dev), yb.to(dev)
            
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct_preds += (preds == yb).sum().item()
            total_samples += len(yb)
            
            if (i + 1) % 10 == 0:
                avg_loss = running_loss / 10
                avg_acc = correct_preds / total_samples
                print(f"  Epoch {epoch+1}, Batch {i+1}/{num_batches} | "
                      f"Avg Loss (last 10): {avg_loss:.4f} | "
                      f"Running Acc: {avg_acc:.4f}")
                running_loss = 0.0
        
        epoch_acc = correct_preds / total_samples
        print(f"  Epoch {epoch + 1}/{epochs} completed. Final Training Accuracy: {epoch_acc:.4f}")


def train_hotflip_verbose(model, loader, loss_fn, optimizer, dev,
                          flip_fraction: float, epochs: int = 10) -> None:
    print(f"Starting verbose HotFlip training with flip_fraction = {flip_fraction:.4f} ...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_samples = 0
        num_batches = len(loader)
        for i, (xb, yb, _) in enumerate(loader):
            xb, yb = xb.to(dev), yb.to(dev)

            adv_xb = generate_hotflip_examples(model, xb, yb, loss_fn, flip_fraction)
            
            optimizer.zero_grad()
            logits_adv = model(adv_xb)
            loss_adv = loss_fn(logits_adv, yb)
            loss_adv.backward()
            optimizer.step()

            running_loss += loss_adv.item()
            preds = (torch.sigmoid(logits_adv) > 0.5).float()
            correct_preds += (preds == yb).sum().item()
            total_samples += len(yb)

            if (i + 1) % 10 == 0:
                avg_loss = running_loss / 10
                avg_acc = correct_preds / total_samples
                print(f"  Epoch {epoch+1}, Batch {i+1}/{num_batches} | "
                      f"Avg Adv Loss (last 10): {avg_loss:.4f} | "
                      f"Running Adv Acc: {avg_acc:.4f}")
                running_loss = 0.0

        epoch_acc = correct_preds / total_samples
        print(f"  Epoch {epoch + 1}/{epochs} completed. Final Adversarial Training Accuracy: {epoch_acc:.4f}")


# --------------------------------------------------------------------------- #
# 5. Attribution-based evaluation (Integrated Gradients & Grad-CAM)
# --------------------------------------------------------------------------- #

def evaluate_model(model, model_name: str, test_ds, dev, produce_plots: bool = True):
    print(f"Evaluating model: {model_name}")
    SAMPLE_N = 300
    ANALYSIS_CHUNK_LEN = 60  # assumed window size

    # -- accuracy ----------------------------------------------------------------
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb, _ in test_dl:
            xb, yb = xb.to(dev), yb.to(dev)
            logits = model(xb)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == yb).sum().item()
            total += len(yb)
    accuracy = correct / total if total else 0
    print(f"Test accuracy: {accuracy:.3f}")

    # -- Integrated Gradients ----------------------------------------------------
    def model_for_captum(x):
        return model(x).unsqueeze(-1)

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
    idxs = rng.choice(positive_subset_indices,
                      size=sample_n_actual,
                      replace=False)

    results = []
    for idx in idxs:
        xb, _, mask = test_ds[idx]
        xb = xb.unsqueeze(0).to(dev)
        attributions = (
            ig.attribute(xb, target=0)
              .abs()
              .sum(1)
              .squeeze(0)
              .cpu()
              .numpy()
        )

        # 1. contiguous wIoU
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

        results.append(
            dict(iou_cont=iou_cont,
                 saliency_auc=saliency_auc,
                 attributions=attributions,
                 mask=mask,
                 cont_start=best_window_start)
        )

    if not results:
        print("No positives in sample set – increase SAMPLE_N.")
        return (0.0, 0.0, 0.0) if not produce_plots else (0.0, accuracy, 0.0)

    # -- statistics --------------------------------------------------------------
    results.sort(key=lambda r: r['iou_cont'])
    mean_iou_cont = np.mean([r['iou_cont'] for r in results])
    mean_saliency_auc = np.mean([r['saliency_auc'] for r in results])
    print(f"Mean wIoU : {mean_iou_cont:.3f} on {len(results)} positive samples")
    print(f"Mean Saliency AUC: {mean_saliency_auc:.3f}")

    if not produce_plots:
        return mean_iou_cont, accuracy, mean_saliency_auc

    # -- plotting ----------------------------------------------------------------
    fig, axs = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle(
        f'IG scores vs. position ({model_name.title()}, sorted by wIoU)'
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
        ax.set_title(f"{title}\nwIoU={data['iou_cont']:.3f}, AUC={data['saliency_auc']:.3f}")
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

    fig_dist, axs_dist = plt.subplots(1, 2, figsize=(12, 5))
    fig_dist.suptitle(f'Evaluation Metric Distributions ({model_name.title()})')

    axs_dist[0].hist(wious, bins=20, alpha=0.75)
    axs_dist[0].set_title('Windowed IoU (wIoU)')
    axs_dist[0].set_xlabel('Score')
    axs_dist[0].set_ylabel('Frequency')
    axs_dist[0].grid(True, ls='--', alpha=0.6)

    axs_dist[1].hist(saliency_aucs, bins=20, alpha=0.75)
    axs_dist[1].set_title('Saliency AUC')
    axs_dist[1].set_xlabel('Score')
    axs_dist[1].grid(True, ls='--', alpha=0.6)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{model_name}_metric_distributions.png")
    print(f"Saved plot → {model_name}_metric_distributions.png")

    return mean_iou_cont, accuracy, mean_saliency_auc


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

def run_single_experiment(seed: int, epsilons_to_test: List[float], train_ds, test_ds):
    """
    1 Generates data for the seed
    2 Trains a standard model and evaluates wIoU & acc
    3 Trains robust models for each epsilon, evaluates each
    """
    print(f"\n{'=' * 20}  SEED {seed}  {'=' * 20}")

    # Use the provided datasets
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bce = nn.BCEWithLogitsLoss()

    # standard model ------------------------------------------------------------
    set_seeds(seed)
    standard_model = TinyCNN().to(dev)
    opt_standard = torch.optim.Adam(standard_model.parameters(), lr=1e-3)
    train_standard(standard_model, train_dl, bce, opt_standard, dev)
    std_wiou, std_acc, std_auc = evaluate_model(standard_model,
                                       f"standard_seed{seed}",
                                       test_ds,
                                       dev,
                                       produce_plots=False)

    # robust models -------------------------------------------------------------
    robust_wious, robust_accs, robust_aucs = [], [], []
    for eps in epsilons_to_test:
        set_seeds(seed)
        mdl = TinyCNN().to(dev)
        opt = torch.optim.Adam(mdl.parameters(), lr=1e-3)
        
        # Use fraction of flips `eps` to calculate `k_flips`
        if eps == 0: # Handle eps=0 case if it occurs
            print("Skipping HotFlip for eps=0, copying standard model results.")
            robust_wious.append(std_wiou)
            robust_accs.append(std_acc)
            robust_aucs.append(std_auc)
            continue
            
        train_hotflip(mdl, train_dl, bce, opt, dev, flip_fraction=eps, epochs=10)
        
        k_flips_for_name = int(eps * SEQ_LEN)
        wio, acc, auc = evaluate_model(mdl,
                                  f"robust_k{k_flips_for_name}_seed{seed}",
                                  test_ds,
                                  dev,
                                  produce_plots=False)
        robust_wious.append(wio)
        robust_accs.append(acc)
        robust_aucs.append(auc)

    return std_wiou, std_acc, std_auc, robust_wious, robust_accs, robust_aucs


# --------------------------------------------------------------------------- #
# 7. Main entry-point
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    
    # Generate the single, large dataset for all experiments
    print(f"--- Generating a single dataset of size {N_TOTAL} ---")
    set_seeds(42) # Use a fixed seed for dataset generation
    master_chunk = random_chunk(CHUNK_LEN)
    X, y, masks = [], [], []
    for _ in range(POS_N):
        bg = sample_background(SEQ_LEN, gc=GC_POS)
        conservation = random.uniform(0.6, 0.9)
        chunk = mutate(master_chunk, conservation)
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
    viz_model = TinyCNN().to(device)
    viz_opt = torch.optim.Adam(viz_model.parameters(), lr=1e-3)
    viz_train_dl = DataLoader(main_train_ds, batch_size=64, shuffle=True)

    train_standard_verbose(viz_model, viz_train_dl, bce, viz_opt, device)
    evaluate_model(viz_model, "standard_baseline", main_test_ds, device, True)
    # evaluate_model_gradcam(viz_model, "standard_baseline", main_test_ds, device, True)
    
    # --- Adversarial Analysis Section ---
    print("\n--- Training a single HotFlip model for analysis ---")
    set_seeds(0)
    hotflip_model_for_analysis = TinyCNN().to(device)
    hotflip_opt = torch.optim.Adam(hotflip_model_for_analysis.parameters(), lr=1e-3)
    
    analysis_flip_fraction = 0.0025 # Using 1% flips (k=10) for better distance visualization
    train_hotflip_verbose(hotflip_model_for_analysis, viz_train_dl, bce, hotflip_opt, device, flip_fraction=analysis_flip_fraction)
    analyze_adversarial_examples(hotflip_model_for_analysis, main_test_ds, device, bce, flip_fraction=analysis_flip_fraction)
    evaluate_model(hotflip_model_for_analysis, "hotflip_baseline", main_test_ds, device, True)

    print("\n--- baseline plots generated. multi-seed experiments start ---\n")

    all_std_wious, all_std_accs, all_std_aucs = [], [], []
    all_rob_wious, all_rob_accs, all_rob_aucs = [], [], []

    for sd in SEEDS:
        sw, sa, sa_auc, rw, ra, ra_auc = run_single_experiment(sd, EPSILONS, main_train_ds, main_test_ds)
        all_std_wious.append(sw)
        all_std_accs.append(sa)
        all_std_aucs.append(sa_auc)
        all_rob_wious.append(rw)
        all_rob_accs.append(ra)
        all_rob_aucs.append(ra_auc)

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

    # plot wIoU vs eps
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
    plt.ylabel('Mean wIoU')
    plt.title('Epsilon vs interpretability (10 seeds)')
    plt.grid(True, which='both', ls='--')
    plt.legend()
    plt.savefig("multi_seed_fgsm_vs_wiou.png")
    print("Saved plot → multi_seed_fgsm_vs_wiou.png")

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
