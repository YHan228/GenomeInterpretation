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
    """ 60-bp random chunk with balanced GC ≈50 % """
    return sample_background(length, 0.5)

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
N_TOTAL = 3000
POS_N = N_TOTAL // 2
NEG_N = N_TOTAL - POS_N

X, y, masks = [], [], []  # data, label, ground-truth mask (pos only)

master_chunk = random_chunk(CHUNK_LEN)

for _ in range(POS_N):
    bg = sample_background(SEQ_LEN, gc=0.60)
    conservation = random.uniform(0.50, 0.75)
    chunk = mutate(master_chunk, conservation)
    seq, start = embed(bg, chunk)
    X.append(one_hot(seq))
    y.append(1)
    mask = np.zeros(SEQ_LEN, dtype=bool)
    mask[start:start + CHUNK_LEN] = True
    masks.append(mask)

for _ in range(NEG_N):
    bg = sample_background(SEQ_LEN, gc=0.45)
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
        self.conv1 = nn.Conv1d(4, 32, 7, padding=3)
        self.conv2 = nn.Conv1d(32, 64, 7, padding=3)
        self.conv3 = nn.Conv1d(64,128, 7, padding=3)
        self.fc1   = nn.Linear(128 * (SEQ_LEN // 8), 128)
        self.out   = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x)); x = F.max_pool1d(x, 2)
        x = F.relu(self.conv2(x)); x = F.max_pool1d(x, 2)
        x = F.relu(self.conv3(x)); x = F.max_pool1d(x, 2)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return torch.sigmoid(self.out(x)).squeeze(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TinyCNN().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
bce = nn.BCELoss()

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

def train_regularized(model, train_dl, bce, opt, device, lambda_l1, epochs=8):
    print(f"Starting L1-regularized training with lambda_l1={lambda_l1}...")
    for epoch in range(epochs):
        model.train()
        for xb, yb, _ in train_dl:
            xb.requires_grad = True # Need to get gradients with respect to input
            
            output = model(xb)
            class_loss = bce(output, yb)
            
            # Get input gradients for the batch
            input_grads = torch.autograd.grad(outputs=class_loss, inputs=xb,
                                               grad_outputs=torch.ones_like(class_loss),
                                               create_graph=True)[0]
            
            # Calculate L1 penalty on the attribution map to encourage sparsity
            attrs = input_grads.abs().sum(1) # Attribution map
            l1_penalty = attrs.mean()
            
            # Combine losses and update
            total_loss = class_loss + lambda_l1 * l1_penalty
            opt.zero_grad()
            total_loss.backward()
            opt.step()
    print(f"  Epoch {epoch+1}/{epochs} completed.")

# ---------- 5. evaluation function -----------------------------------------------

def evaluate_model(model, model_name: str, test_ds, device, produce_plots=True):
    print(f"Evaluating model: {model_name}")
    
    SAMPLE_N = 300
    ANALYSIS_CHUNK_LEN = 100 # Researcher's assumption of window size

    # Accuracy
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb, _ in test_dl:
            xb, yb = xb.to(device), yb.to(device)
            preds = (model(xb) > 0.5).float()
            correct += (preds == yb).sum().item()
            total   += len(yb)
    print(f"Test accuracy: {correct/total:.3f}")

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

        results.append({'iou_pos': iou_pos, 'iou_cont': iou_cont, 
                        'attributions': attributions, 'mask': mask,
                        'cont_start': best_window_start})

    if results:
        results.sort(key=lambda x: x['iou_cont'])
        mean_iou_pos = np.mean([r['iou_pos'] for r in results])
        mean_iou_cont = np.mean([r['iou_cont'] for r in results])
        print(f"Mean IoU over {len(results)} positive samples: {mean_iou_pos:.3f}")
        print(f"Mean wIoU over {len(results)} positive samples: {mean_iou_cont:.3f}")
        print(f"Note: wIoU = {CHUNK_LEN/ANALYSIS_CHUNK_LEN:.2f} is the best maximum achievable when {ANALYSIS_CHUNK_LEN}bp window overlaps perfectly with {CHUNK_LEN}bp ground truth")

        if not produce_plots:
            return mean_iou_cont

        # Plotting (sorted by wIoU)
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
                         f"IoU: {data['iou_pos']:.3f}")
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
        return mean_iou_cont

    else:
        print("No positives in sample set – increase SAMPLE_N.")
        if not produce_plots:
            return 0.0

# ---------- 6. Main Execution Logic ---------------------------------------------

def run_single_experiment(seed, lambdas_to_test):
    """
    Runs the full experiment for a single seed.
    1. Generates data
    2. Trains a standard model and evaluates its wIoU.
    3. Trains a regularized model for each lambda and evaluates its wIoU.
    Returns the wIoU for the standard model and a list of wIoUs for the regularized models.
    """
    print(f"\n{'='*20} RUNNING FOR SEED: {seed} {'='*20}")
    
    # 1. Generate data for this specific seed
    set_seeds(seed)
    master_chunk = random_chunk(CHUNK_LEN)
    X, y, masks = [], [], []
    for _ in range(POS_N):
        bg = sample_background(SEQ_LEN, gc=0.60)
        conservation = random.uniform(0.50, 0.75)
        chunk = mutate(master_chunk, conservation)
        seq, start = embed(bg, chunk)
        X.append(one_hot(seq))
        y.append(1)
        m = np.zeros(SEQ_LEN, dtype=bool); m[start:start + CHUNK_LEN] = True
        masks.append(m)
    for _ in range(NEG_N):
        bg = sample_background(SEQ_LEN, gc=0.45)
        X.append(one_hot(bg))
        y.append(0)
        masks.append(np.zeros(SEQ_LEN, dtype=bool))
    
    X = torch.tensor(np.stack(X)); y = torch.tensor(y, dtype=torch.float); masks = np.stack(masks)
    ds = SeqDS(X, y, masks)
    train_ds, test_ds = random_split(ds, [int(0.8 * N_TOTAL), N_TOTAL - int(0.8 * N_TOTAL)],
                                     generator=torch.Generator().manual_seed(seed))
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bce = nn.BCELoss()

    # 2. Train and evaluate standard model
    set_seeds(seed)
    standard_model = TinyCNN().to(device)
    opt_standard = torch.optim.Adam(standard_model.parameters(), lr=1e-3)
    train_standard(standard_model, train_dl, bce, opt_standard, device)
    standard_wIou = evaluate_model(standard_model, f"standard_seed{seed}", test_ds, device, produce_plots=False)

    # 3. Train and evaluate regularized models for each lambda
    reg_wIous = []
    for l1 in lambdas_to_test:
        set_seeds(seed) # Ensure every model starts from the same weights for this seed
        model = TinyCNN().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        train_regularized(model, train_dl, bce, optimizer, device, lambda_l1=l1, epochs=8)
        mean_wIou = evaluate_model(model, f"reg_lambda_{l1}_seed{seed}", test_ds, device, produce_plots=False)
        reg_wIous.append(mean_wIou)
    
    return standard_wIou, reg_wIous


# ---------- 7. Multi-Seed Experiment Aggregation --------------------------------
SEEDS = [42, 123, 1024, 0, 99]
lambdas = [1e-5, 1e-4, 1e-3, 1e-2, 0.1] # L1 penalties are often smaller
all_standard_wIous = []
all_reg_wIous = [] # This will be a list of lists

for seed in SEEDS:
    standard_wIou, reg_wIous_for_seed = run_single_experiment(seed, lambdas)
    all_standard_wIous.append(standard_wIou)
    all_reg_wIous.append(reg_wIous_for_seed)

# Aggregate results
standard_wIou_mean = np.mean(all_standard_wIous)
standard_wIou_std = np.std(all_standard_wIous)
reg_wIous_by_lambda = np.array(all_reg_wIous)
reg_mean_by_lambda = np.mean(reg_wIous_by_lambda, axis=0)
reg_std_by_lambda = np.std(reg_wIous_by_lambda, axis=0)

# Plot the final aggregated results
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

plt.title('Impact of L1 Regularization on Model Interpretability (Averaged Over 5 Seeds)')
plt.xlabel('Lambda (L1 Regularization Strength)')
plt.ylabel('Mean Windowed IoU (wIoU)')
plt.xscale('log')
plt.grid(True, which="both", ls="--")
plt.legend()
plt.savefig("multi_seed_l1_regu_vs_wIou.png")
print(f"\nSaved final multi-seed experiment plot to multi_seed_l1_regu_vs_wIou.png")


