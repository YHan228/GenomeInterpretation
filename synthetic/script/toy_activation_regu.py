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
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt
import torchattacks

# --------------------------------------------------------------------------- //
# 1. Configuration & Utilities
# --------------------------------------------------------------------------- #

WITH_CONFOUNDER = True # Global switch for GC-content difference

def set_seeds(seed_value: int = 42) -> None:
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
N_TOTAL = 5000
POS_N = N_TOTAL // 2
NEG_N = N_TOTAL - POS_N

# Define GC content based on the global flag
GC_POS = 0.535 if WITH_CONFOUNDER else 0.50
GC_NEG = 0.50

X, y, masks = [], [], []  # data, label, ground-truth mask (pos only)

master_chunk = random_chunk(CHUNK_LEN)

for _ in range(POS_N):
    bg = sample_background(SEQ_LEN, gc=GC_POS)
    conservation = random.uniform(0.6, 0.9)
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
        conv3_out = F.relu(self.conv3(x)); x = F.max_pool1d(conv3_out, 2)
        pooled_out = self.pool(x).squeeze(-1)
        logits = self.fc(pooled_out)
        return logits.squeeze(-1), conv3_out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TinyCNN().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
bce = nn.BCEWithLogitsLoss()

# ---------- 4. training functions ------------------------------------------------

def train_standard(model, loader, loss_fn, optimizer, dev, epochs: int = 10) -> None:
    print("Starting standard training...")
    for epoch in range(epochs):
        model.train()
        for xb, yb, _ in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            optimizer.zero_grad()
            logits, _ = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
        print(f"  Epoch {epoch + 1}/{epochs} completed.")

def train_activation_regularized(model, loader, loss_fn, optimizer, dev,
                 lambda_l1: float, epochs: int = 10) -> None:
    print(f"Starting Activation L1 regularized training (lambda_l1={lambda_l1:.2E})...")
    for epoch in range(epochs):
        model.train()
        for xb, yb, _ in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            logits, conv_out = model(xb)
            class_loss = loss_fn(logits, yb)
            
            activation_l1_penalty = conv_out.abs().mean()
            total_loss = class_loss + (lambda_l1 * activation_l1_penalty)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        print(f"  Epoch {epoch + 1}/{epochs} completed.")

# ---------- 5. evaluation function -----------------------------------------------

def evaluate_model(model, model_name: str, test_ds, dev, produce_plots: bool = True):
    print(f"Evaluating model: {model_name}")
    SAMPLE_N = 300
    ANALYSIS_CHUNK_LEN = 60  # assumed window size

    # Accuracy
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb, _ in test_dl:
            xb, yb = xb.to(dev), yb.to(dev)
            logits, _ = model(xb)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == yb).sum().item()
            total += len(yb)
    accuracy = correct / total if total else 0
    print(f"Test accuracy: {accuracy:.3f}")

    # IG Attribution
    # We need a wrapper that returns only the logits for Captum
    def model_for_captum(x):
        logits, _ = model(x)
        return logits.unsqueeze(-1)

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

        # 1. IoU (top-k mask)
        topk_idx = np.argsort(attributions)[-ANALYSIS_CHUNK_LEN:]
        pred_mask_pos = np.zeros(SEQ_LEN, dtype=bool)
        pred_mask_pos[topk_idx] = True
        inter_pos = (pred_mask_pos & mask).sum()
        union_pos = (pred_mask_pos | mask).sum()
        iou_pos = inter_pos / union_pos if union_pos else 0

        # 2. contiguous wIoU
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

        # 3. Saliency AUC
        inside_scores = attributions[mask]
        outside_scores = attributions[~mask]
        # Efficiently calculate AUC: probability that a random inside score is > a random outside score
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
        ax.set_title(f"{title}\nwIoU={data['iou_cont']:.3f}, "
                     f"IoU={data['iou_pos']:.3f}, AUC={data['saliency_auc']:.3f}")
        ax.set_xlabel("Position")
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

def run_single_experiment(seed: int, lambdas_to_test: List[float], train_ds, test_ds):
    """
    1 Generates data for the seed
    2 Trains a standard model and evaluates wIoU & acc
    3 Trains robust models for each lambda, evaluates each
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

    # regularized models -------------------------------------------------------------
    reg_wious, reg_accs, reg_aucs = [], [], []
    for l1 in lambdas_to_test:
        set_seeds(seed)
        mdl = TinyCNN().to(dev)
        opt = torch.optim.Adam(mdl.parameters(), lr=1e-3)
        train_activation_regularized(mdl, train_dl, bce, opt, dev, lambda_l1=l1, epochs=10)
        wio, acc, auc = evaluate_model(mdl,
                                  f"reg_lambda_{l1}_seed{seed}",
                                  test_ds,
                                  dev,
                                  produce_plots=False)
        reg_wious.append(wio)
        reg_accs.append(acc)
        reg_aucs.append(auc)

    return std_wiou, std_acc, std_auc, reg_wious, reg_accs, reg_aucs

# 7. Multi-Seed Experiment Aggregation
# --------------------------------------------------------------------------- #

if __name__ == '__main__':
    # Generate the single, large dataset for all experiments
    print(f"--- Generating a single dataset of size {N_TOTAL} ---")
    set_seeds(42) # Use a fixed seed for dataset generation
    
    # Define GC content based on the global flag
    GC_POS = 0.535 if WITH_CONFOUNDER else 0.50
    GC_NEG = 0.50
    
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

    # --- Initial visualization of the baseline problem ---
    print("\n--- Running a single standard model for visualization ---")
    set_seeds(42)
    viz_model = TinyCNN().to(device)
    viz_opt = torch.optim.Adam(viz_model.parameters(), lr=1e-3)
    viz_train_dl = DataLoader(main_train_ds, batch_size=64, shuffle=True)
    bce = nn.BCEWithLogitsLoss()

    train_standard(viz_model, viz_train_dl, bce, viz_opt, device)
    evaluate_model(viz_model, "standard_baseline", main_test_ds, device, True)

    print("\n--- Activation L1 Regularization experiments start ---\n")
    
    SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    lambdas = [1e-4, 1e-3, 1e-2, 0.1, 1.0] 
    all_std_wious, all_std_accs, all_std_aucs = [], [], []
    all_reg_wious, all_reg_accs, all_reg_aucs = [], [], []

    for sd in SEEDS:
        sw, sa, sa_auc, rw, ra, ra_auc = run_single_experiment(sd, lambdas, main_train_ds, main_test_ds)
        all_std_wious.append(sw)
        all_std_accs.append(sa)
        all_std_aucs.append(sa_auc)
        all_reg_wious.append(rw)
        all_reg_accs.append(ra)
        all_reg_aucs.append(ra_auc)

    # aggregate - wIoU
    standard_wIou_mean = np.mean(all_std_wious)
    standard_wIou_std = np.std(all_std_wious)
    reg_wIous_by_lambda = np.array(all_reg_wious)
    reg_mean_by_lambda = np.mean(reg_wIous_by_lambda, axis=0)
    reg_std_by_lambda = np.std(reg_wIous_by_lambda, axis=0)

    # aggregate - accuracy
    standard_acc_mean = np.mean(all_std_accs)
    standard_acc_std = np.std(all_std_accs)
    reg_accs_by_lambda = np.array(all_reg_accs)
    reg_mean_acc_by_lambda = np.mean(reg_accs_by_lambda, axis=0)
    reg_std_acc_by_lambda = np.std(reg_accs_by_lambda, axis=0)

    # aggregate - saliency AUC
    std_auc_mean = np.mean(all_std_aucs)
    std_auc_std = np.std(all_std_aucs)
    reg_auc_by_lambda = np.array(all_reg_aucs)
    reg_mean_auc_by_lambda = np.mean(reg_auc_by_lambda, axis=0)
    reg_std_auc_by_lambda = np.std(reg_auc_by_lambda, axis=0)

    # plot wIoU vs lambda
    plt.figure(figsize=(12, 8))
    # Plot regularized model results with variance
    plt.plot(lambdas, reg_mean_by_lambda, marker='o', linestyle='-', label='Mean Regularized Model wIoU')
    plt.fill_between(lambdas, reg_mean_by_lambda - reg_std_by_lambda, 
                     reg_mean_by_lambda + reg_std_by_lambda, alpha=0.2, 
                     label='Regularized Model Std. Dev.')

    # Plot standard model baseline with variance
    plt.axhline(y=standard_wIou_mean, color='r', linestyle='--', 
                label=f'Mean Standard Model wIoU ({standard_wIou_mean:.3f})')
    plt.fill_between(lambdas, standard_wIou_mean - standard_wIou_std, 
                     standard_wIou_mean + standard_wIou_std, color='r', alpha=0.1,
                     label='Standard Model Std. Dev.')

    plt.title('Impact of Activation L1 Regularization on Interpretability (Averaged Over 10 Seeds)')
    plt.xlabel('Lambda (L1 Regularization Strength)')
    plt.ylabel('Mean Windowed IoU (wIoU)')
    plt.xscale('log')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.savefig("multi_seed_activation_regu_vs_wIou.png")
    print(f"\nSaved final multi-seed experiment plot to multi_seed_activation_regu_vs_wIou.png")

    # plot accuracy vs lambda
    plt.figure(figsize=(12, 8))
    plt.plot(lambdas, reg_mean_acc_by_lambda, marker='o', linestyle='-', label='Mean Regularized Model Accuracy')
    plt.fill_between(lambdas, reg_mean_acc_by_lambda - reg_std_acc_by_lambda, 
                     reg_mean_acc_by_lambda + reg_std_acc_by_lambda, alpha=0.2, 
                     label='Regularized Model Std. Dev.')
    plt.axhline(y=standard_acc_mean, color='r', linestyle='--', 
                label=f'Mean Standard Model Accuracy ({standard_acc_mean:.3f})')
    plt.fill_between(lambdas, standard_acc_mean - standard_acc_std, 
                     standard_acc_mean + standard_acc_std, color='r', alpha=0.1,
                     label='Standard Model Std. Dev.')
    plt.title('Impact of Activation L1 Regularization on Model Accuracy (Averaged Over 10 Seeds)')
    plt.xlabel('Lambda (L1 Regularization Strength)')
    plt.ylabel('Test Accuracy')
    plt.xscale('log')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.savefig("multi_seed_activation_regu_vs_acc.png")
    print(f"\nSaved final accuracy experiment plot to multi_seed_activation_regu_vs_acc.png")

    # plot saliency AUC vs lambda
    plt.figure(figsize=(12, 8))
    plt.plot(lambdas, reg_mean_auc_by_lambda, marker='o', linestyle='-', label='Mean Regularized Model Saliency AUC')
    plt.fill_between(lambdas, reg_mean_auc_by_lambda - reg_std_auc_by_lambda, 
                     reg_mean_auc_by_lambda + reg_std_auc_by_lambda, alpha=0.2, 
                     label='Regularized Model Std. Dev.')
    plt.axhline(y=std_auc_mean, color='r', linestyle='--', 
                label=f'Mean Standard Model Saliency AUC ({std_auc_mean:.3f})')
    plt.fill_between(lambdas, std_auc_mean - std_auc_std, 
                     std_auc_mean + std_auc_std, color='r', alpha=0.1,
                     label='Standard Model Std. Dev.')
    plt.title('Impact of Activation L1 Regularization on Saliency AUC (Averaged Over 10 Seeds)')
    plt.xlabel('Lambda (L1 Regularization Strength)')
    plt.ylabel('Mean Saliency AUC')
    plt.xscale('log')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.savefig("multi_seed_activation_regu_vs_saliency_auc.png")
    print(f"\nSaved final Saliency AUC experiment plot to multi_seed_activation_regu_vs_saliency_auc.png")

    # --- Final Visualization of Regularization Effect ---
    print("\n--- Generating final analysis visualization ---")
    set_seeds(42)
    
    # 1. Train a standard and a regularized model
    std_model = TinyCNN().to(device)
    std_opt = torch.optim.Adam(std_model.parameters(), lr=1e-3)
    print("Training standard model for visualization...")
    train_standard(std_model, main_train_ds, bce, std_opt, device)

    reg_model = TinyCNN().to(device)
    reg_opt = torch.optim.Adam(reg_model.parameters(), lr=1e-3)
    best_lambda = lambdas[np.argmax(reg_mean_by_lambda)]
    print(f"Training regularized model for visualization (best lambda = {best_lambda:.4f})...")
    train_activation_regularized(reg_model, main_train_ds, bce, reg_opt, device, lambda_l1=best_lambda)

    # 2. Get a batch of positive samples
    positive_indices = [i for i, (_, y, _) in enumerate(main_test_ds.dataset) if y.item() == 1]
    sample_batch = torch.stack([main_test_ds.dataset[i][0] for i in positive_indices[:64]])
    sample_batch = sample_batch.to(device)

    # 3. Get activations from both models
    with torch.no_grad():
        _, std_activations = std_model(sample_batch)
        _, reg_activations = reg_model(sample_batch)

    # 4. Create the plot
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Effect of Activation L1 Regularization on Internal Feature Maps', fontsize=16)

    # Heatmaps of activations
    im1 = axs[0, 0].imshow(std_activations[0].cpu().numpy(), aspect='auto', cmap='viridis')
    axs[0, 0].set_title('Standard Model: Activations (1st sample)')
    axs[0, 0].set_ylabel('Filter/Channel')
    fig.colorbar(im1, ax=axs[0, 0])

    im2 = axs[0, 1].imshow(reg_activations[0].cpu().numpy(), aspect='auto', cmap='viridis')
    axs[0, 1].set_title('Regularized Model: Activations (1st sample)')
    fig.colorbar(im2, ax=axs[0, 1])

    # Histograms of activation values
    axs[1, 0].hist(std_activations.flatten().cpu().numpy(), bins=50, log=True)
    axs[1, 0].set_title('Standard Model: Activation Histogram')
    axs[1, 0].set_xlabel('Activation Value')
    axs[1, 0].set_ylabel('Frequency (log scale)')
    
    axs[1, 1].hist(reg_activations.flatten().cpu().numpy(), bins=50, log=True)
    axs[1, 1].set_title('Regularized Model: Activation Histogram')
    axs[1, 1].set_xlabel('Activation Value')

    for ax_row in axs:
        for ax in ax_row:
            ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("activation_regularization_effect.png")
    print("\nSaved final activation analysis plot to activation_regularization_effect.png")


