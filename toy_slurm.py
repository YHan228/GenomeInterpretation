"""
Synthetic 1-kbp phenotype dataset (SLURM-compatible version)
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
import argparse
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

# --- Hyperparameter Search Space ---
GC_HPARAMS = [0.55, 0.60, 0.65]
CONS_HPARAMS = [0.6, 0.7, 0.8]
# ---

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


set_seeds(42)  # initial seed for consistency

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
N_TOTAL = 10000
POS_N = N_TOTAL // 2
NEG_N = N_TOTAL - POS_N

def generate_dataset(gc_pos: float, conservation: float):
    """Generates the main dataset based on global config."""
    print(f"Generating dataset with GC_POS={gc_pos:.2f} and conservation={conservation:.2f}...")
    GC_NEG = 0.50

    X, y, masks = [], [], []
    master_chunk = random_chunk(CHUNK_LEN)

    for _ in range(POS_N):
        bg = sample_background(SEQ_LEN, gc=gc_pos)
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

    X = torch.tensor(np.stack(X))
    y = torch.tensor(y, dtype=torch.float)
    masks = np.stack(masks)
    
    return SeqDS(X, y, masks)


# --------------------------------------------------------------------------- #
# 3. Model and Dataset Classes
# --------------------------------------------------------------------------- #

class SeqDS(Dataset):
    def __init__(self, xs, ys, ms):
        self.x, self.y, self.m = xs, ys, ms

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.m[idx]

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
        logits = self.fc(x)
        return logits.squeeze(-1)


# --------------------------------------------------------------------------- #
# 4. Training and Evaluation
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
    seq_len = xb.shape[2]
    k_flips = int(flip_fraction * seq_len)
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
            pos = best_pos_to_flip
            start = torch.clamp(pos - neighborhood_size, 0)
            end = torch.clamp(pos + neighborhood_size + 1, max=seq_len)
            indices = torch.arange(seq_len, device=xb.device).unsqueeze(0)
            newly_forbidden = (indices >= start.unsqueeze(1)) & (indices < end.unsqueeze(1))
            forbidden_regions |= newly_forbidden
    return adv_xb

def train_hotflip(model, loader, loss_fn, optimizer, dev,
                  flip_fraction: float, epochs: int = 10) -> None:
    print(f"Starting HotFlip training with flip_fraction = {flip_fraction:.4f} ...")
    for epoch in range(epochs):
        model.train()
        for xb, yb, _ in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            adv_xb = generate_hotflip_examples(model, xb, yb, loss_fn, flip_fraction)
            optimizer.zero_grad()
            logits_adv = model(adv_xb)
            loss_adv = loss_fn(logits_adv, yb)
            loss_adv.backward()
            optimizer.step()
        print(f"  Epoch {epoch + 1}/{epochs} completed.")

def evaluate_model(model, test_dl, dev):
    print(f"Evaluating model...")
    SAMPLE_N = 300
    ANALYSIS_CHUNK_LEN = 60

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
    print(f"  Test accuracy: {accuracy:.3f}")

    def model_for_captum(x):
        return model(x).unsqueeze(-1)

    ig = IntegratedGradients(model_for_captum)
    test_ds = test_dl.dataset
    positive_subset_indices = [
        i for i, original_idx in enumerate(test_ds.indices)
        if test_ds.dataset.y[original_idx] == 1
    ]

    rng = np.random.default_rng(0)
    sample_n_actual = min(SAMPLE_N, len(positive_subset_indices))
    if sample_n_actual == 0:
        print("Warning: No positive samples in test set for evaluation.")
        return 0.0, accuracy, 0.0
        
    idxs = rng.choice(positive_subset_indices, size=sample_n_actual, replace=False)

    results = []
    for idx in idxs:
        xb, _, mask = test_ds[idx]
        xb = xb.unsqueeze(0).to(dev)
        attributions = ig.attribute(xb, target=0).abs().sum(1).squeeze(0).cpu().numpy()
        window_sums = np.convolve(attributions, np.ones(ANALYSIS_CHUNK_LEN), mode='valid')
        best_window_start = np.argmax(window_sums)
        pred_mask_cont = np.zeros(SEQ_LEN, dtype=bool)
        pred_mask_cont[best_window_start:best_window_start + ANALYSIS_CHUNK_LEN] = True
        inter_cont = (pred_mask_cont & mask).sum()
        union_cont = (pred_mask_cont | mask).sum()
        iou_cont = inter_cont / union_cont if union_cont else 0
        inside_scores = attributions[mask]
        outside_scores = attributions[~mask]
        saliency_auc = (inside_scores[:, None] > outside_scores[None, :]).mean()
        results.append(dict(iou_cont=iou_cont, saliency_auc=saliency_auc))

    mean_iou_cont = np.mean([r['iou_cont'] for r in results])
    mean_saliency_auc = np.mean([r['saliency_auc'] for r in results])
    print(f"  Mean wIoU: {mean_iou_cont:.3f}")
    print(f"  Mean Saliency AUC: {mean_saliency_auc:.3f}")

    return mean_iou_cont, accuracy, mean_saliency_auc

# --------------------------------------------------------------------------- #
# 5. Experiment Runner
# --------------------------------------------------------------------------- #

def run_single_experiment(seed: int, epsilons_to_test: List[float], main_ds):
    print(f"\n{'=' * 20}  SEED {seed}  {'=' * 20}")
    
    train_ds, test_ds = random_split(
        main_ds,
        [int(0.8 * N_TOTAL), N_TOTAL - int(0.8 * N_TOTAL)],
        generator=torch.Generator().manual_seed(seed) # Use seed for split
    )
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=128)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bce = nn.BCEWithLogitsLoss()

    set_seeds(seed)
    standard_model = TinyCNN().to(dev)
    opt_standard = torch.optim.Adam(standard_model.parameters(), lr=1e-3)
    train_standard(standard_model, train_dl, bce, opt_standard, dev)
    std_wiou, std_acc, std_auc = evaluate_model(standard_model, test_dl, dev)

    robust_wious, robust_accs, robust_aucs = [], [], []
    for eps in epsilons_to_test:
        set_seeds(seed)
        mdl = TinyCNN().to(dev)
        opt = torch.optim.Adam(mdl.parameters(), lr=1e-3)
        if eps > 0:
            train_hotflip(mdl, train_dl, bce, opt, dev, flip_fraction=eps, epochs=10)
            wio, acc, auc = evaluate_model(mdl, test_dl, dev)
            robust_wious.append(wio); robust_accs.append(acc); robust_aucs.append(auc)
        else: # Handle eps=0 case
            robust_wious.append(std_wiou); robust_accs.append(std_acc); robust_aucs.append(std_auc)

    return std_wiou, std_acc, std_auc, robust_wious, robust_accs, robust_aucs

# --------------------------------------------------------------------------- #
# 6. Main entry-point
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multi-seed robustness experiment.")
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save results and plots.')
    args = parser.parse_args()
    
    # --- Outer loop for Hyperparameter Search ---
    for gc_hparam in GC_HPARAMS:
        for cons_hparam in CONS_HPARAMS:
            
            run_output_dir = os.path.join(args.output_dir, f"gc_{gc_hparam:.2f}_cons_{cons_hparam:.2f}")
            os.makedirs(run_output_dir, exist_ok=True)
            
            print(f"\n{'#'*60}")
            print(f"## Starting HP experiment: GC={gc_hparam:.2f}, Conservation={cons_hparam:.2f}")
            print(f"## Results will be saved to: {run_output_dir}")
            print(f"{'#'*60}\n")
            
            main_dataset = generate_dataset(gc_pos=gc_hparam, conservation=cons_hparam)
            
            all_std_wious, all_std_accs, all_std_aucs = [], [], []
            all_rob_wious, all_rob_accs, all_rob_aucs = [], [], []

            for sd in SEEDS:
                sw, sa, sa_auc, rw, ra, ra_auc = run_single_experiment(sd, EPSILONS, main_dataset)
                all_std_wious.append(sw); all_std_accs.append(sa); all_std_aucs.append(sa_auc)
                all_rob_wious.append(rw); all_rob_accs.append(ra); all_rob_aucs.append(ra_auc)

            # --- Save Raw Results ---
            np.savez(
                os.path.join(run_output_dir, 'multi_seed_results.npz'),
                epsilons=EPSILONS, seeds=SEEDS,
                gc_pos=gc_hparam, conservation=cons_hparam,
                std_wious=all_std_wious, std_accs=all_std_accs, std_aucs=all_std_aucs,
                rob_wious=all_rob_wious, rob_accs=all_rob_accs, rob_aucs=all_rob_aucs
            )
            print(f"\nSaved raw results to {os.path.join(run_output_dir, 'multi_seed_results.npz')}")

            # --- Aggregate and Plot ---
            rob_wious_arr = np.array(all_rob_wious)
            rob_acc_arr = np.array(all_rob_accs)
            rob_auc_arr = np.array(all_rob_aucs)
            std_wiou_mean = np.mean(all_std_wious); std_wiou_std = np.std(all_std_wious)
            std_acc_mean = np.mean(all_std_accs); std_acc_std = np.std(all_std_accs)
            std_auc_mean = np.mean(all_std_aucs); std_auc_std = np.std(all_std_aucs)

            title_suffix = f"(GC={gc_hparam:.2f}, Cons={cons_hparam:.2f}, 10 Seeds)"

            # Plot 1: Mean performance vs Epsilon
            fig, axs = plt.subplots(1, 3, figsize=(24, 7))
            fig.suptitle(f'Mean Performance vs. Adversarial Training Strength {title_suffix}')
            metrics = [
                ('wIoU', rob_wious_arr, std_wiou_mean, std_wiou_std),
                ('Accuracy', rob_acc_arr, std_acc_mean, std_acc_std),
                ('Saliency AUC', rob_auc_arr, std_auc_mean, std_auc_std)
            ]
            for i, (name, rob_arr, std_mean, std_std) in enumerate(metrics):
                rob_mean = rob_arr.mean(axis=0); rob_std = rob_arr.std(axis=0)
                axs[i].plot(EPSILONS, rob_mean, marker='o', label='Robust mean')
                axs[i].fill_between(EPSILONS, rob_mean - rob_std, rob_mean + rob_std, alpha=0.2)
                axs[i].axhline(std_mean, color='r', ls='--', label=f'Standard mean ({std_mean:.3f})')
                axs[i].fill_between(EPSILONS, std_mean - std_std, std_mean + std_std, color='r', alpha=0.1)
                axs[i].set_xlabel('Epsilon (Fraction of Sequence Flipped)'); axs[i].set_ylabel(f'Mean {name}')
                axs[i].set_title(f'{name} vs. Epsilon'); axs[i].set_xscale('log'); axs[i].grid(True, which='both', ls='--'); axs[i].legend()
            plt.tight_layout(rect=[0, 0.03, 1, 0.93])
            plt.savefig(os.path.join(run_output_dir, "mean_performance_vs_epsilon.png"))
            print(f"Saved plot to {os.path.join(run_output_dir, 'mean_performance_vs_epsilon.png')}")

            # Plot 2: Per-seed trajectories
            fig, axs = plt.subplots(1, 3, figsize=(24, 7))
            fig.suptitle(f'Per-Seed Performance vs. Adversarial Training Strength {title_suffix}')
            for i, (name, rob_arr, _, _) in enumerate(metrics):
                for seed_idx in range(len(SEEDS)):
                    axs[i].plot(EPSILONS, rob_arr[seed_idx, :], marker='o', linestyle='-', alpha=0.4)
                axs[i].set_xlabel('Epsilon'); axs[i].set_ylabel(name)
                axs[i].set_title(f'Per-Seed {name} vs. Epsilon'); axs[i].set_xscale('log'); axs[i].grid(True, which='both', ls='--')
            plt.tight_layout(rect=[0, 0.03, 1, 0.93])
            plt.savefig(os.path.join(run_output_dir, "per_seed_trajectories.png"))
            print(f"Saved plot to {os.path.join(run_output_dir, 'per_seed_trajectories.png')}")
            
            # Plot 3: Delta plots (Robust - Standard)
            fig, axs = plt.subplots(1, 3, figsize=(24, 7))
            fig.suptitle(f'Improvement vs. Adversarial Training Strength {title_suffix}')
            for i, (name, rob_arr, _, _) in enumerate(metrics):
                std_arr = np.array(all_std_wious if name == 'wIoU' else all_std_accs if name == 'Accuracy' else all_std_aucs)
                delta = rob_arr - std_arr[:, np.newaxis]
                delta_mean = delta.mean(axis=0); delta_std = delta.std(axis=0)
                axs[i].plot(EPSILONS, delta_mean, marker='o', label=f'Mean Δ{name}')
                axs[i].fill_between(EPSILONS, delta_mean - delta_std, delta_mean + delta_std, alpha=0.2)
                axs[i].axhline(0, color='r', ls='--'); axs[i].set_xlabel('Epsilon')
                axs[i].set_ylabel(f'Δ{name}'); axs[i].set_title(f'Improvement in {name} vs. Epsilon')
                axs[i].set_xscale('log'); axs[i].grid(True, which='both', ls='--'); axs[i].legend()
            plt.tight_layout(rect=[0, 0.03, 1, 0.93])
            plt.savefig(os.path.join(run_output_dir, "delta_performance_vs_epsilon.png"))
            print(f"Saved plot to {os.path.join(run_output_dir, 'delta_performance_vs_epsilon.png')}")
            
            plt.close('all') # Close all figures to free memory for the next loop 