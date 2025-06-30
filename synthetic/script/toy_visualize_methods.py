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
        conv3_out = F.relu(self.conv3(x)); x = F.max_pool1d(conv3_out, 2)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        logits = self.out(x).squeeze(1)
        return logits, conv3_out # Return both logits and internal activations

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
            logits, _ = model(xb) # Ignore activation output for standard training
            loss = bce(logits, yb)
            loss.backward()
            opt.step()
        print(f"  Epoch {epoch+1}/{epochs} completed.")

def train_fgsm(model, train_dl, bce, opt, device, eps, epochs=8):
    print(f"Starting robust (FGSM) training with eps={eps}...")
    for epoch in range(epochs):
        model.train()
        for xb, yb, _ in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            
            # --- FGSM Attack Generation ---
            xb.requires_grad = True
            logits, _ = model(xb)
            loss = bce(logits, yb)
            model.zero_grad()
            loss.backward()
            
            grad = xb.grad.data
            adv_xb = (xb + eps * grad.sign()).detach()
            # --- End Attack ---

            # Train on adversarial examples
            opt.zero_grad()
            logits_adv, _ = model(adv_xb)
            loss_adv = bce(logits_adv, yb)
            loss_adv.backward()
            opt.step()
    print(f"  Epoch {epoch+1}/{epochs} completed.")

def train_pgd(model, train_dl, bce, opt, device, eps, epochs=8):
    print(f"Starting robust (PGD) training with eps={eps}...")
    alpha = eps / 4
    steps = 10
    for epoch in range(epochs):
        model.train()
        for xb, yb, _ in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            
            # --- PGD Attack Generation ---
            # Start with a random perturbation
            adv_xb = xb.clone().detach() + torch.empty_like(xb).uniform_(-eps, eps)
            adv_xb = torch.clamp(adv_xb, min=0, max=1).detach()

            for _ in range(steps):
                adv_xb.requires_grad = True
                logits, _ = model(adv_xb)
                loss = bce(logits, yb)
                model.zero_grad()
                loss.backward()
                
                grad = adv_xb.grad.data
                adv_xb = (adv_xb + alpha * grad.sign()).detach()
                delta = torch.clamp(adv_xb - xb, min=-eps, max=eps)
                adv_xb = torch.clamp(xb + delta, min=0, max=1).detach()
            # --- End Attack ---
            
            # Train on adversarial examples
            opt.zero_grad()
            logits_adv, _ = model(adv_xb)
            loss_adv = bce(logits_adv, yb)
            loss_adv.backward()
            opt.step()
        print(f"  Epoch {epoch+1}/{epochs} completed.")

def train_activation_regularized(model, train_dl, bce, opt, device, lambda_l1, epochs=8):
    print(f"Starting Activation L1 regularized training (lambda_l1={lambda_l1:.2E})...")
    for epoch in range(epochs):
        model.train()
        for xb, yb, _ in train_dl:
            logits, conv3_out = model(xb)
            class_loss = bce(logits, yb)
            
            # L1 penalty on the activations of the final conv layer
            activation_l1_penalty = conv3_out.abs().mean()
            
            total_loss = class_loss + (lambda_l1 * activation_l1_penalty)
            opt.zero_grad()
            total_loss.backward()
            opt.step()
    print(f"  Epoch {epoch+1}/{epochs} completed.")

# ---------- 5. Visualization Logic ---------------------------------------------

def get_snr_distribution(model, test_ds, device):
    """
    Calculates the Signal-to-Noise Ratio (SNR) of attributions for all positive samples.
    SNR is defined as mean(attribution_inside_chunk) / mean(attribution_outside_chunk).
    """
    snr_values = []
    
    def model_for_captum(x):
        logits, _ = model(x)
        return logits.unsqueeze(-1)
    ig = IntegratedGradients(model_for_captum)
    
    positive_subset_indices = [
        i for i, original_idx in enumerate(test_ds.indices)
        if test_ds.dataset.m[original_idx].sum() > 0
    ]
    
    print(f"  Generating attributions for {len(positive_subset_indices)} positive samples...")
    for idx in positive_subset_indices:
        xb, _, mask = test_ds[idx]
        xb = xb.unsqueeze(0).to(device)
        
        attributions = ig.attribute(xb, target=0).abs().sum(1).squeeze(0).cpu().numpy()
        
        signal = attributions[mask].mean()
        noise = attributions[~mask].mean()
        
        # Add a small epsilon to avoid division by zero
        snr = signal / (noise + 1e-8)
        snr_values.append(snr)
        
    return snr_values


# ---------- 6. Main Execution ---------------------------------------------------
if __name__ == '__main__':
    print("--- Generating Representative Attribution Visualizations ---")

    # 1. Use the globally generated dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bce = nn.BCEWithLogitsLoss()

    # 2. Train all four models
    models = {}

    print("\n--- Training Standard Model ---")
    set_seeds(42)
    standard_model = TinyCNN().to(device)
    opt = torch.optim.Adam(standard_model.parameters(), lr=1e-3)
    train_standard(standard_model, train_dl, bce, opt, device)
    models['Standard'] = standard_model

    print("\n--- Training FGSM Model ---")
    set_seeds(42)
    fgsm_model = TinyCNN().to(device)
    opt = torch.optim.Adam(fgsm_model.parameters(), lr=1e-3)
    train_fgsm(fgsm_model, train_dl, bce, opt, device, eps=0.01)
    models['FGSM (eps=0.01)'] = fgsm_model

    print("\n--- Training PGD Model ---")
    set_seeds(42)
    pgd_model = TinyCNN().to(device)
    opt = torch.optim.Adam(pgd_model.parameters(), lr=1e-3)
    train_pgd(pgd_model, train_dl, bce, opt, device, eps=0.01)
    models['PGD (eps=0.01)'] = pgd_model

    print("\n--- Training Activation L1 Model ---")
    set_seeds(42)
    act_reg_model = TinyCNN().to(device)
    opt = torch.optim.Adam(act_reg_model.parameters(), lr=1e-3)
    train_activation_regularized(act_reg_model, train_dl, bce, opt, device, lambda_l1=0.1)
    models['Activation L1 (λ=0.1)'] = act_reg_model

    # 3. Generate and Plot SNR distributions
    snr_results = {}
    for name, model in models.items():
        print(f"\nCalculating SNR for: {name}")
        snr_results[name] = get_snr_distribution(model, test_ds, device)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.boxplot(snr_results.values())
    ax.set_xticklabels(snr_results.keys(), rotation=15, ha="right")
    ax.set_title('Interpretability Signal-to-Noise Ratio by Training Method', fontsize=16)
    ax.set_ylabel('SNR (Mean Attribution In / Mean Attribution Out)')
    ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig("snr_comparison.png")
    print("\nSaved final SNR comparison plot to snr_comparison.png")


