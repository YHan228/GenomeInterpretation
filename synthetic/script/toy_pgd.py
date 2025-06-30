"""
Synthetic 1-kbp phenotype dataset:
    positives: high-GC background + one 60-bp causal block
    negatives: low-GC background, no causal block
CNN training + Integrated Gradients attribution quality
Author: <your-name>, 2025-06-29
"""

import random, string, math, os, itertools
from typing import List, Tuple

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from captum.attr import IntegratedGradients, LayerGradCam
import matplotlib.pyplot as plt
import torchattacks

WITH_CONFOUNDER = True # Global switch for GC-content difference

def set_seeds(seed_value=42):
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    # The two lines below are not strictly necessary but ensure full determinism
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# set initial seed
set_seeds()

# ---------- 1. sequence utilities ------------------------------------------------

ALPH = np.array(list("ACGT"), dtype="U1")
to_ix = {b: i for i, b in enumerate(ALPH)}

def sample_background(length: int, gc: float) -> np.ndarray:
    """iid sampling with given GC content, returns char array"""
    p = np.array([(1 - gc) / 2, gc / 2, gc / 2, (1 - gc) / 2])  # A,C,G,T
    return np.random.choice(ALPH, size=length, p=p)

def random_chunk(length: int) -> np.ndarray:
    """60-bp random chunk with balanced GC ≈ 50 %"""
    return sample_background(length, 0.525)

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
    """(1000,) char -> (4,1000) float32 one-hot"""
    arr = np.zeros((4, len(seq)), dtype=np.float32)
    for i, b in enumerate(seq):
        arr[to_ix[b], i] = 1.0
    return arr

# ---------- 2. dataset generation ------------------------------------------------

SEQ_LEN = 1000
CHUNK_LEN = 60
N_TOTAL = 10000
POS_N = N_TOTAL // 2
NEG_N = N_TOTAL - POS_N

# Define GC content based on the global flag
GC_POS = 0.525 if WITH_CONFOUNDER else 0.50
GC_NEG = 0.50

X, y, masks = [], [], []  # data, label, ground-truth mask (pos only)

master_chunk = random_chunk(CHUNK_LEN)

for _ in range(POS_N):
    bg = sample_background(SEQ_LEN, gc=GC_POS)
    conservation = random.uniform(0.7, 0.8)
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
    masks.append(np.zeros(SEQ_LEN, dtype=bool))  # empty mask

X = torch.tensor(np.stack(X))          # (N,4,1000)
y = torch.tensor(y, dtype=torch.float) # (N,)
masks = np.stack(masks)                # (N,1000) bool

class SeqDS(Dataset):
    def __init__(self, xs, ys, ms): self.x, self.y, self.m = xs, ys, ms
    def __len__(self): return len(self.x)
    def __getitem__(self, idx): return self.x[idx], self.y[idx], self.m[idx]

ds = SeqDS(X, y, masks)
train_ds, test_ds = random_split(ds, [int(0.8 * N_TOTAL), N_TOTAL - int(0.8 * N_TOTAL)],
                                 generator=torch.Generator().manual_seed(42))
train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
test_dl  = DataLoader(test_ds , batch_size=128)

# ---------- 3. model -------------------------------------------------------------

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

# ---------- 4. training functions ------------------------------------------------

def train_standard(model, train_dl, bce, opt, device, epochs=8):
    print("Starting standard training...")
    for epoch in range(epochs):
        model.train()
        for xb, yb, _ in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = bce(model(xb), yb)
            loss.backward()
            opt.step()
        print(f"  Epoch {epoch+1}/{epochs} completed.")

def train_robust(model, train_dl, bce, opt, device, eps, epochs=8):
    print(f"Starting robust (PGD) training with eps={eps}...")
    alpha = eps / 4
    steps = 10
    for epoch in range(epochs):
        model.train()
        for xb, yb, _ in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            
            # --- Manual PGD Attack Generation on Logits ---
            # Start with a random perturbation
            adv_xb = xb.clone().detach() + torch.empty_like(xb).uniform_(-eps, eps)
            adv_xb = torch.clamp(adv_xb, min=0, max=1).detach() # Ensure valid one-hot range

            for _ in range(steps):
                adv_xb.requires_grad = True
                logits = model(adv_xb)
                loss = bce(logits, yb)
                model.zero_grad()
                loss.backward()
                
                grad = adv_xb.grad.data
                adv_xb = (adv_xb + alpha * grad.sign()).detach()
                # Project back into epsilon-ball and valid range
                delta = torch.clamp(adv_xb - xb, min=-eps, max=eps)
                adv_xb = torch.clamp(xb + delta, min=0, max=1).detach()
            # --- End Attack ---

            opt.zero_grad()
            logits_adv = model(adv_xb)
            loss_adv = bce(logits_adv, yb)
            loss_adv.backward()
            opt.step()
        print(f"  Epoch {epoch+1}/{epochs} completed.")

# ---------- 5. evaluation function -----------------------------------------------

def evaluate_model(model, model_name: str, test_ds, device, produce_plots=True):
    print(f"Evaluating model: {model_name}")
    
    SAMPLE_N = 300
    ANALYSIS_CHUNK_LEN = 60 # Researcher's assumption of window size

    # Accuracy
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb, _ in test_dl:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == yb).sum().item()
            total   += len(yb)
    accuracy = correct / total if total > 0 else 0
    print(f"Test accuracy: {accuracy:.3f}")

    # IG Attribution
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
        print(f"Warning: Found only {sample_n_actual} positive samples, less than the requested {SAMPLE_N}.")
    idxs = rng.choice(positive_subset_indices, size=sample_n_actual, replace=False)

    results = []
    for idx in idxs:
        xb, _, mask = test_ds[idx]
        xb = xb.unsqueeze(0).to(device)
        attributions = ig.attribute(xb, target=0).abs().sum(1).squeeze(0).cpu().numpy()

        # 1. IoU (original method)
        topk_idx = np.argsort(attributions)[-ANALYSIS_CHUNK_LEN:]
        pred_mask_pos = np.zeros(SEQ_LEN, dtype=bool); pred_mask_pos[topk_idx] = True
        inter_pos = (pred_mask_pos & mask).sum()
        union_pos = (pred_mask_pos | mask).sum()
        iou_pos = (inter_pos / union_pos if union_pos else 0)

        # 2. wIoU
        window_sums = np.convolve(attributions, np.ones(ANALYSIS_CHUNK_LEN), 'valid')
        best_window_start = np.argmax(window_sums)
        pred_mask_cont = np.zeros(SEQ_LEN, dtype=bool)
        pred_mask_cont[best_window_start:best_window_start + ANALYSIS_CHUNK_LEN] = True
        inter_cont = (pred_mask_cont & mask).sum()
        union_cont = (pred_mask_cont | mask).sum()
        iou_cont = (inter_cont / union_cont if union_cont else 0)

        # 3. Saliency AUC
        inside_scores = attributions[mask]
        outside_scores = attributions[~mask]
        saliency_auc = (inside_scores[:, None] > outside_scores[None, :]).mean()

        results.append(
            dict(iou_pos=iou_pos,
                 iou_cont=iou_cont,
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
    mean_iou_pos = np.mean([r['iou_pos'] for r in results])
    mean_iou_cont = np.mean([r['iou_cont'] for r in results])
    mean_saliency_auc = np.mean([r['saliency_auc'] for r in results])
    print(f"Mean IoU  : {mean_iou_pos:.3f} on {len(results)} positive samples")
    print(f"Mean wIoU : {mean_iou_cont:.3f}")
    print(f"Mean Saliency AUC: {mean_saliency_auc:.3f}")

    if not produce_plots:
        return mean_iou_cont, accuracy, mean_saliency_auc

    # -- plotting ----------------------------------------------------------------
    fig, axs = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle(f'IG Scores vs. Sequence Position ({model_name.title()} Model, Sorted by wIoU)')
    
    n_res = len(results)
    mid1_idx, mid2_idx = n_res // 2 - 1, n_res // 2
    
    plot_data = [results[0], results[1], 
                 results[mid1_idx], results[mid2_idx], 
                 results[-2], results[-1]]
    titles = ['Worst IoU 1', 'Worst IoU 2', 
              'Middle IoU 1', 'Middle IoU 2',
              'Best IoU 2', 'Best IoU 1']

    for i, (ax, data, title) in enumerate(zip(axs.flat, plot_data, titles)):
        ax.plot(data['attributions'], label='IG Score', color='black', linewidth=0.7)
        title_str = (f"{title}\n"
                     f"wIoU: {data['iou_cont']:.3f}, "
                     f"IoU: {data['iou_pos']:.3f}, AUC={data['saliency_auc']:.3f}")
        ax.set_title(title_str)
        ax.set_xlabel("Sequence Position")
        ax.set_ylabel("IG Score")
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Highlight ground truth (red) and predicted contiguous block (blue)
        gt_start = np.where(data['mask'])[0][0]
        ax.axvspan(gt_start, gt_start + CHUNK_LEN, color='red', alpha=0.2, lw=0, label=f'Ground Truth ({CHUNK_LEN}bp)')
        
        pred_start = data['cont_start']
        ax.axvspan(pred_start, pred_start + ANALYSIS_CHUNK_LEN, color='blue', alpha=0.2, lw=0, label=f'Predicted Block ({ANALYSIS_CHUNK_LEN}bp)')
        ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{model_name}_ig_scores_plot.png")
    print(f"Saved plot to {model_name}_ig_scores_plot.png")

    # Plotting distributions
    ious_pos = [r['iou_pos'] for r in results]
    ious_cont = [r['iou_cont'] for r in results]

    fig_dist, axs_dist = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    fig_dist.suptitle(f'Distribution of IoU Scores for Positive Samples ({model_name.title()} Model)')

    axs_dist[0].hist(ious_pos, bins=20, alpha=0.75, color='royalblue')
    axs_dist[0].set_title('IoU')
    axs_dist[0].set_xlabel('IoU Score')
    axs_dist[0].set_ylabel('Frequency')
    axs_dist[0].grid(True, linestyle='--', alpha=0.6)

    axs_dist[1].hist(ious_cont, bins=20, alpha=0.75, color='firebrick')
    axs_dist[1].set_title('wIoU')
    axs_dist[1].set_xlabel('IoU Score')
    axs_dist[1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{model_name}_iou_distributions.png")
    print(f"Saved IoU distribution plot to {model_name}_iou_distributions.png")
    return mean_iou_cont, accuracy, mean_saliency_auc

# ---------- 6. Main Execution Logic ---------------------------------------------

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
    standard_wIou, standard_acc, standard_auc = evaluate_model(standard_model, f"standard_seed{seed}", test_ds, dev, produce_plots=False)

    # 3. Train and evaluate robust models for each epsilon
    robust_wious, robust_accs, robust_aucs = [], [], []
    for eps in epsilons_to_test:
        set_seeds(seed)
        mdl = TinyCNN().to(dev)
        opt = torch.optim.Adam(mdl.parameters(), lr=1e-3)
        train_robust(mdl, train_dl, bce, opt, dev, eps=eps, epochs=10)
        wio, acc, auc = evaluate_model(mdl,
                                  f"robust_eps{eps}_seed{seed}",
                                  test_ds,
                                  dev,
                                  produce_plots=False)
        robust_wious.append(wio)
        robust_accs.append(acc)
        robust_aucs.append(auc)
    
    return standard_wIou, standard_acc, standard_auc, robust_wious, robust_accs, robust_aucs

# 7. Main entry-point
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    
    # Generate the single, large dataset for all experiments
    print(f"--- Generating a single dataset of size {N_TOTAL} ---")
    set_seeds(42)
    master_chunk = random_chunk(CHUNK_LEN, GC_POS)
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

    print("\n--- PGD experiments start ---\n")

    SEEDS = [42, 123, 1024, 0, 99]
    epsilons = [0.001, 0.005, 0.01, 0.025, 0.05] # Shortened list for PGD
    all_std_wious, all_std_accs, all_std_aucs = [], [], []
    all_rob_wious, all_rob_accs, all_rob_aucs = [], [], []

    for sd in SEEDS:
        sw, sa, sa_auc, rw, ra, ra_auc = run_single_experiment(sd, epsilons, main_train_ds, main_test_ds)
        all_std_wious.append(sw)
        all_std_accs.append(sa)
        all_std_aucs.append(sa_auc)
        all_rob_wious.append(rw)
        all_rob_accs.append(ra)
        all_rob_aucs.append(ra_auc)

    # Aggregate wIoU results
    standard_wIou_mean = np.mean(all_std_wious)
    standard_wIou_std = np.std(all_std_wious)
    robust_wIous_by_eps = np.array(all_rob_wious)
    robust_mean_by_eps = np.mean(robust_wIous_by_eps, axis=0)
    robust_std_by_eps = np.std(robust_wIous_by_eps, axis=0)

    # Aggregate accuracy results
    standard_acc_mean = np.mean(all_std_accs)
    standard_acc_std = np.std(all_std_accs)
    robust_accs_by_eps = np.array(all_rob_accs)
    robust_mean_acc_by_eps = np.mean(robust_accs_by_eps, axis=0)
    robust_std_acc_by_eps = np.std(robust_accs_by_eps, axis=0)

    # Aggregate saliency AUC results
    standard_auc_mean = np.mean(all_std_aucs)
    standard_auc_std = np.std(all_std_aucs)
    robust_aucs_by_eps = np.array(all_rob_aucs)
    robust_mean_auc_by_eps = np.mean(robust_aucs_by_eps, axis=0)
    robust_std_auc_by_eps = np.std(robust_aucs_by_eps, axis=0)

    # Plot wIoU results
    plt.figure(figsize=(12, 8))
    # Plot robust model results with variance
    plt.plot(epsilons, robust_mean_by_eps, marker='o', linestyle='-', label='Mean Robust Model wIoU')
    plt.fill_between(epsilons, robust_mean_by_eps - robust_std_by_eps, 
                     robust_mean_by_eps + robust_std_by_eps, alpha=0.2, 
                     label='Robust Model Std. Dev.')

    # Plot standard model baseline with variance
    plt.axhline(y=standard_wIou_mean, color='r', linestyle='--', 
                label=f'Mean Standard Model wIoU ({standard_wIou_mean:.3f})')
    plt.fill_between(epsilons, standard_wIou_mean - standard_wIou_std, 
                     standard_wIou_mean + standard_wIou_std, color='r', alpha=0.1,
                     label='Standard Model Std. Dev.')

    plt.title('Impact of Epsilon on Model Interpretability (Averaged Over 5 Seeds)')
    plt.xlabel('Epsilon (Adversarial Perturbation Size)')
    plt.ylabel('Mean Windowed IoU (wIoU)')
    plt.xscale('log')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.savefig("multi_seed_pgd_vs_wIou.png")
    print(f"\nSaved final multi-seed experiment plot to multi_seed_pgd_vs_wIou.png")

    # Plot Accuracy results
    plt.figure(figsize=(12, 8))
    plt.plot(epsilons, robust_mean_acc_by_eps, marker='o', linestyle='-', label='Mean Robust Model Accuracy')
    plt.fill_between(epsilons, robust_mean_acc_by_eps - robust_std_acc_by_eps, 
                     robust_mean_acc_by_eps + robust_std_acc_by_eps, alpha=0.2, 
                     label='Robust Model Std. Dev.')
    plt.axhline(y=standard_acc_mean, color='r', linestyle='--', 
                label=f'Mean Standard Model Accuracy ({standard_acc_mean:.3f})')
    plt.fill_between(epsilons, standard_acc_mean - standard_acc_std, 
                     standard_acc_mean + standard_acc_std, color='r', alpha=0.1,
                     label='Standard Model Std. Dev.')
    plt.title('Impact of PGD Epsilon on Model Accuracy (Averaged Over 5 Seeds)')
    plt.xlabel('Epsilon (Adversarial Perturbation Size)')
    plt.ylabel('Test Accuracy')
    plt.xscale('log')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.savefig("multi_seed_pgd_vs_acc.png")
    print(f"\nSaved final accuracy experiment plot to multi_seed_pgd_vs_acc.png")

    # Plot saliency AUC results
    plt.figure(figsize=(12, 8))
    plt.plot(epsilons, robust_mean_auc_by_eps, marker='o', label='Robust mean')
    plt.fill_between(epsilons, robust_mean_auc_by_eps - robust_std_auc_by_eps, 
                     robust_mean_auc_by_eps + robust_std_auc_by_eps, alpha=0.2)
    plt.axhline(standard_auc_mean, color='r', ls='--', label=f'Standard mean ({standard_auc_mean:.3f})')
    plt.fill_between(epsilons, standard_auc_mean - standard_auc_std, 
                     standard_auc_mean + standard_auc_std, color='r', alpha=0.1)
    plt.xscale('log')
    plt.xlabel('Epsilon')
    plt.ylabel('Mean Saliency AUC')
    plt.title('Epsilon vs Saliency AUC (5 seeds)')
    plt.grid(True, which='both', ls='--')
    plt.legend()
    plt.savefig("multi_seed_pgd_vs_saliency_auc.png")
    print("Saved plot → multi_seed_pgd_vs_saliency_auc.png")


