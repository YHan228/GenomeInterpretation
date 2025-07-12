"""
Tests a synthetic DNA sequence data generator for binary classification.

Positive examples are 1 kbp sequences containing 3-4 conserved, non-overlapping
blocks (40-70 bp each). These blocks are derived from a shared pool of 100
"ancestral" motifs, which are mutated to a specified conservation level before
being embedded in a background sequence of a given GC content. An optional
σ70-like promoter may be added upstream of the first block.

Negative examples are 1 kbp sequences of random background DNA, with GC content
drawn from a normal distribution around 0.50, containing no specific motifs.

The script trains a simple CNN on the generated data over multiple random seeds
to validate the dataset's utility. For the first run, it generates example
Integrated Gradients (IG) plots and metric distribution plots (Saliency AUC,
Saliency SNR) for model interpretability analysis.
"""
import argparse
import os
import random
import sys
from typing import Tuple
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from captum.attr import IntegratedGradients
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

# ------------------------------------------------------------------ #
# 0.  Global settings
# ------------------------------------------------------------------ #
SEQ_LEN                = 1000
BLOCKS_RANGE           = (3, 4)
BLOCK_LEN_MEAN         = 55
BLOCK_LEN_SD           = 15
BLOCK_LEN_MIN, BLOCK_LEN_MAX = 40, 70
PROMOTER_HEX_1, PROMOTER_HEX_2 = "TTGACA", "TATAAT"
PROMOTER_SPACER        = 17
MIN_GAP_BETWEEN_BLOCKS = 30
N_TOTAL = 5000
DEFAULT_BATCH_SIZE = 512
DEFAULT_EPOCHS = 50
DEFAULT_MOTIF_REPERTOIRE = 30
DEFAULT_TARGET_SIGNAL_FRAC = 0.20

# --------------------------------------------------------------------------- #
# 1. Utilities
# --------------------------------------------------------------------------- #

def set_seeds(seed_value: int = 42) -> None:
    """Sets random seeds for reproducibility."""
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

ALPH = np.array(list("ACGT"), dtype="U1")
to_ix = {b: i for i, b in enumerate(ALPH)}

def sample_background(length: int, gc: float) -> np.ndarray:
    """iid sampling with given GC content"""
    p = np.array([(1 - gc) / 2, gc / 2, gc / 2, (1 - gc) / 2])
    return np.random.choice(ALPH, size=length, p=p)

def mutate(chunk: np.ndarray, conservation: float) -> np.ndarray:
    """Return a new chunk with given conservation level"""
    mutated_chunk = chunk.copy()
    n_to_mutate = int(len(chunk) * (1.0 - conservation))
    pos_to_mutate = np.random.choice(len(chunk), n_to_mutate, replace=False)
    for pos in pos_to_mutate:
        original_base = mutated_chunk[pos]
        mutated_chunk[pos] = np.random.choice(np.setdiff1d(ALPH, [original_base]))
    return mutated_chunk

def one_hot(seq: np.ndarray) -> np.ndarray:
    """(L,) char -> (4,L) float32 one-hot"""
    arr = np.zeros((4, len(seq)), dtype=np.float32)
    for i, b in enumerate(seq):
        arr[to_ix[b], i] = 1.0
    return arr

def _trunc_norm(mu, sd, lo, hi, size=None):
    """N(μ,σ²) truncated to [lo,hi]."""
    while True:
        v = np.random.normal(mu, sd, size)
        if np.all((v >= lo) & (v <= hi)):
            return v.astype(int)

def _nonoverlap_positions(seq_len, lens):
    """Random non-overlapping start indices with ≥ MIN_GAP_BETWEEN_BLOCKS."""
    tries = 0
    while tries < 1000:
        starts = sorted(np.random.randint(0, seq_len - sum(lens) - (len(lens) - 1) * MIN_GAP_BETWEEN_BLOCKS + 1, size=len(lens)))
        
        # Adjust starts to include gaps
        offsets = np.array([0] + [l + MIN_GAP_BETWEEN_BLOCKS for l in lens[:-1]])
        starts = np.array(starts) + np.cumsum(offsets)

        if starts[-1] + lens[-1] <= seq_len:
             # Final check to ensure no logical error led to overlap
            if all(starts[i] + lens[i] + MIN_GAP_BETWEEN_BLOCKS <= starts[i+1] for i in range(len(starts)-1)):
                return starts.tolist()
        tries += 1
    raise RuntimeError("Could not place blocks without overlap.")

def build_promoter(gc):
    """Return 6-6-hexamer promoter string with spacer, GC tuned to background."""
    spacer = sample_background(PROMOTER_SPACER, gc)
    return np.array(list(PROMOTER_HEX_1 + ''.join(spacer) + PROMOTER_HEX_2), dtype="U1")

# --------------------------------------------------------------------------- #
# 2. Dataset Generation
# --------------------------------------------------------------------------- #

class SeqDS(Dataset):
    def __init__(self, xs, ys, ms):
        self.x, self.y, self.m = xs, ys, ms

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.m[idx]

def generate_dataset(gc_pos: float, conservation: float, target_signal_frac: float,
                     motif_repertoire: int = DEFAULT_MOTIF_REPERTOIRE,
                     include_partial_negatives: bool = True) -> Tuple[SeqDS, dict]:
    """
    New synthetic genome generator.
    * Positive examples have blocks + promoter, with count based on target_signal_frac.
    * Optional decoy negatives have 1-2 blocks but no promoter.
    * Negative background GC now drawn from N(0.50, 0.04) to add variance.
    """
    # Scale sample size to ensure enough examples per motif
    avg_blocks = max((target_signal_frac * SEQ_LEN) / BLOCK_LEN_MEAN, 1.0)
    required_pos = int(np.ceil(30 * motif_repertoire / avg_blocks))
    pos_n_default = N_TOTAL // 2
    POS_N = max(required_pos, pos_n_default)
    NEG_N = POS_N

    decoy_neg_n = int(0.2 * NEG_N) if include_partial_negatives else 0
    std_neg_n = NEG_N - decoy_neg_n
    promoter_only_neg_n = int(0.5 * std_neg_n)
    background_only_neg_n = std_neg_n - promoter_only_neg_n

    X, y, masks = [], [], []
    
    # Create a library of ancestral prototype blocks
    ancestral_pool = []
    for _ in range(motif_repertoire):
        ancestor_gc = np.clip(np.random.normal(gc_pos, 0.02), 0.25, 0.75)
        ancestor_seq = sample_background(BLOCK_LEN_MAX, gc=ancestor_gc)
        ancestral_pool.append(ancestor_seq)

    # --- Positive examples (must have promoter + sufficient blocks) ---
    n_pos_generated = 0
    realised_fracs_pos = []
    while n_pos_generated < POS_N:
        current_gc_pos = np.clip(np.random.normal(gc_pos, 0.04), 0.25, 0.75)
        bg = sample_background(SEQ_LEN, gc=current_gc_pos)
        
        n_blocks_mean = (target_signal_frac * SEQ_LEN) / BLOCK_LEN_MEAN
        n_blocks_low = int(np.floor(n_blocks_mean))
        n_blocks_high = int(np.ceil(n_blocks_mean))
        if n_blocks_low == n_blocks_high:
            n_blocks_high += 1
        n_blocks = np.random.randint(max(3, n_blocks_low), max(4, n_blocks_high))

        blk_lens = _trunc_norm(BLOCK_LEN_MEAN, BLOCK_LEN_SD, BLOCK_LEN_MIN, BLOCK_LEN_MAX, n_blocks)
        try:
            blk_starts = _nonoverlap_positions(SEQ_LEN, blk_lens)
        except RuntimeError:
            continue # Try again if placement fails

        mask = np.zeros(SEQ_LEN, dtype=bool)
        for blen, start in zip(blk_lens, blk_starts):
            ancestor = random.choice(ancestral_pool)
            master = ancestor[:blen]
            chunk = mutate(master, conservation)
            bg[start:start+blen] = chunk
            mask[start:start+blen] = True
        
        # Mandatory promoter for positive examples
        first_start = min(blk_starts)
        promoter_full_len = len(PROMOTER_HEX_1) + PROMOTER_SPACER + len(PROMOTER_HEX_2)
        if first_start < (promoter_full_len + MIN_GAP_BETWEEN_BLOCKS):
            continue # Not enough space for promoter, retry
            
        prom_seq = build_promoter(current_gc_pos)
        prom_pos = first_start - len(prom_seq) - MIN_GAP_BETWEEN_BLOCKS
        if prom_pos < 0:
            continue

        bg[prom_pos : prom_pos + len(prom_seq)] = prom_seq
        mask[prom_pos : prom_pos + len(prom_seq)] = True
        
        X.append(one_hot(bg))
        y.append(1)
        masks.append(mask)
        realised_fracs_pos.append(mask.sum() / SEQ_LEN)
        n_pos_generated += 1

    # --- Decoy negatives (1-2 blocks, no promoter) ---
    for _ in range(decoy_neg_n):
        bg_gc = np.clip(np.random.normal(0.50, 0.04), 0.25, 0.75)
        bg = sample_background(SEQ_LEN, gc=bg_gc)
        
        n_blocks = np.random.randint(1, 3)
        blk_lens = _trunc_norm(BLOCK_LEN_MEAN, BLOCK_LEN_SD, BLOCK_LEN_MIN, BLOCK_LEN_MAX, n_blocks)
        try:
            blk_starts = _nonoverlap_positions(SEQ_LEN, blk_lens)
        except RuntimeError:
            continue

        mask = np.zeros(SEQ_LEN, dtype=bool)
        for blen, start in zip(blk_lens, blk_starts):
            ancestor = random.choice(ancestral_pool)
            master = ancestor[:blen]
            chunk = mutate(master, conservation)
            bg[start:start+blen] = chunk
            mask[start:start+blen] = True
            
        X.append(one_hot(bg))
        y.append(0)
        masks.append(mask)

    # --- Promoter-only negatives ---
    for _ in range(promoter_only_neg_n):
        current_gc = np.clip(np.random.normal(0.50, 0.04), 0.25, 0.75)
        bg = sample_background(SEQ_LEN, gc=current_gc)

        prom_seq = build_promoter(current_gc)
        promoter_full_len = len(prom_seq)
        
        if SEQ_LEN >= promoter_full_len:
            prom_pos = np.random.randint(0, SEQ_LEN - promoter_full_len + 1)
            bg[prom_pos : prom_pos + promoter_full_len] = prom_seq

        X.append(one_hot(bg))
        y.append(0)
        masks.append(np.zeros(SEQ_LEN, dtype=bool))

    # --- Standard negatives (background only) ---
    for _ in range(background_only_neg_n):
        bg_gc  = np.clip(np.random.normal(0.50, 0.04), 0.25, 0.75)
        bg     = sample_background(SEQ_LEN, gc=bg_gc)
        X.append(one_hot(bg))
        y.append(0)
        masks.append(np.zeros(SEQ_LEN, dtype=bool))

    X = torch.from_numpy(np.stack(X)).float()
    y = torch.from_numpy(np.array(y)).float()
    masks = np.stack(masks)

    avg_realised_frac = np.mean(realised_fracs_pos) if realised_fracs_pos else 0.0
    summary = {
        "n_sequences": len(X), "n_positive": n_pos_generated, "n_decoy_negative": decoy_neg_n,
        "n_promoter_only_negative": promoter_only_neg_n,
        "motif_repertoire": motif_repertoire, "seed": np.random.get_state()[1][0].item(),
        "target_signal_frac": target_signal_frac, "avg_realised_frac": avg_realised_frac,
    }
    return SeqDS(X, y, masks), summary

# --------------------------------------------------------------------------- #
# 3. Model Definition
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
        logits = self.fc(x)
        return logits.squeeze(-1), None

# --------------------------------------------------------------------------- #
# 4. Training & Evaluation
# --------------------------------------------------------------------------- #

def validate_epoch(model, loader, loss_fn, dev):
    """Calculates the loss on a validation set."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for xb, yb, _ in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            with autocast():
                logits, _ = model(xb)
                loss = loss_fn(logits, yb)
            total_loss += loss.item()
    return total_loss / len(loader) if len(loader) > 0 else 0

def train_standard(model, train_loader, val_loader, epochs, dev, early_stopping_patience: int = 10, early_stopping_min_delta: float = 1e-4):
    """Standard model training with early stopping and LR scheduling."""
    print(f"  Standard training for {epochs} epochs (patience={early_stopping_patience}, min_delta={early_stopping_min_delta})...")
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=4, verbose=False)
    scaler = GradScaler()
    
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
                logits, _ = model(xb)
                loss = loss_fn(logits, yb)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            num_batches += 1
        
        avg_train_loss = total_loss / num_batches if num_batches > 0 else 0
        avg_val_loss = validate_epoch(model, val_loader, loss_fn, dev)
        
        scheduler.step(avg_val_loss)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"    Epoch {epoch + 1}/{epochs}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        if (best_val_loss - avg_val_loss) > early_stopping_min_delta:
            best_val_loss = avg_val_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
        
        if early_stopping_counter >= early_stopping_patience:
            print(f"  -> Early stopping at epoch {epoch + 1}.")
            break

def evaluate_model(model, test_dl, dev, model_name: str = "model", produce_plots: bool = False):
    """Evaluates model accuracy and interpretability metrics."""
    print("  Evaluating model...")
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

    def model_for_captum(x):
        return model(x)[0].unsqueeze(-1)

    ig = IntegratedGradients(model_for_captum)
    test_ds = test_dl.dataset
    positive_indices = [i for i, (_, y, _) in enumerate(test_ds) if y == 1]
    
    if not positive_indices:
        return accuracy, 0.0, 0.0

    sample_n = min(50, len(positive_indices))
    idxs = random.sample(positive_indices, sample_n)

    results = []
    for idx in idxs:
        xb, yb_scalar, mask = test_ds[idx]
        xb = xb.unsqueeze(0).to(dev)
        yb = torch.tensor([yb_scalar], device=dev, dtype=torch.float)
        
        # Use sample-specific nucleotide average as baseline
        proportions = xb.mean(dim=2, keepdim=True)
        baseline = proportions.expand_as(xb)
        attributions = ig.attribute(xb, baselines=baseline, target=0).abs().sum(1).squeeze(0).cpu().numpy()
        
        inside_scores = attributions[mask]
        outside_scores = attributions[~mask]
        saliency_auc = (inside_scores[:, None] > outside_scores[None, :]).mean() if len(inside_scores) > 0 and len(outside_scores) > 0 else 0.5
        
        sum_sq_inside = np.sum(inside_scores**2)
        sum_sq_total = np.sum(attributions**2)
        saliency_snr = sum_sq_inside / (sum_sq_total + 1e-9)

        results.append({'auc': saliency_auc, 'snr': saliency_snr, 'attributions': attributions, 'mask': mask})

    mean_auc = np.mean([r['auc'] for r in results]) if results else 0.0
    mean_snr = np.mean([r['snr'] for r in results]) if results else 0.0
    
    print(f"    Accuracy: {accuracy:.3f}, SaliencyAUC: {mean_auc:.3f}, SaliencySNR: {mean_snr:.3f}")
    
    if produce_plots and results:
        # Sort for IG plots
        results.sort(key=lambda r: r['auc'])
        
        fig, axs = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle(f'IG scores vs. position ({model_name.title()}, sorted by Saliency AUC)')
        
        n_res = len(results)
        indices_to_plot = []
        if n_res > 0: indices_to_plot.extend([0, n_res-1])
        if n_res > 2: indices_to_plot.extend([1, n_res-2])
        if n_res > 4: indices_to_plot.extend([n_res//2 - 1, n_res//2])
        indices_to_plot = sorted(list(set(indices_to_plot)))

        plot_data = [results[i] for i in indices_to_plot]
        
        base_titles = ['Worst', 'Worst', 'Median', 'Median', 'Best', 'Best']
        # Adjust titles based on how many plots we actually have
        if len(plot_data) == 2:
            titles = ['Worst', 'Best']
        elif len(plot_data) == 4:
            titles = ['Worst', 'Second Worst', 'Second Best', 'Best']
        else: # 6 plots
            titles = ['Worst 1', 'Worst 2', 'Median 1', 'Median 2', 'Best 2', 'Best 1']


        for i, ax in enumerate(axs.flat):
            if i >= len(plot_data):
                ax.axis('off')
                continue
            
            data = plot_data[i]
            title = f"{titles[i]}\nSaliency AUC={data['auc']:.3f}, SNR={data['snr']:.2f}"

            ax.plot(data['attributions'], label='IG score', color='black', linewidth=0.7)
            ax.set_title(title)
            ax.set_xlabel("Position")
            ax.set_ylabel("IG score")
            ax.grid(True, ls='--', alpha=0.6)

            # Highlight ground truth blocks
            mask = data['mask']
            in_block = False
            first_block_plotted = False
            for pos, is_mask in enumerate(mask):
                if is_mask and not in_block:
                    start = pos
                    in_block = True
                elif not is_mask and in_block:
                    end = pos
                    in_block = False
                    label = 'Ground truth' if not first_block_plotted else ""
                    ax.axvspan(start, end, color='red', alpha=0.2, lw=0, label=label)
                    if not first_block_plotted: first_block_plotted = True
            if in_block: # edge case for block at end
                 label = 'Ground truth' if not first_block_plotted else ""
                 ax.axvspan(start, len(mask), color='red', alpha=0.2, lw=0, label=label)
                 if not first_block_plotted: first_block_plotted = True

            if first_block_plotted:
                ax.legend()
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plot_path = f'{model_name}_ig_plots.png'
        plt.savefig(plot_path)
        plt.close(fig)
        print(f"Saved example IG plots to {plot_path}")

        # --- Distribution plots ---
        saliency_aucs = [r['auc'] for r in results]
        saliency_snrs = [r['snr'] for r in results]

        fig_dist, axs_dist = plt.subplots(1, 2, figsize=(12, 5))
        fig_dist.suptitle(f'Evaluation Metric Distributions ({model_name.title()})')

        axs_dist[0].hist(saliency_aucs, bins=20, alpha=0.75)
        axs_dist[0].set_title('Saliency AUC')
        axs_dist[0].set_xlabel('Score')
        axs_dist[0].set_ylabel('Frequency')
        axs_dist[0].grid(True, ls='--', alpha=0.6)

        # Clip SNR for better visualization
        snr_percentile = np.percentile(saliency_snrs, 99) if saliency_snrs else 10
        plot_range_snr = (0, max(snr_percentile, 10))
        axs_dist[1].hist(saliency_snrs, bins=20, alpha=0.75, range=plot_range_snr)
        axs_dist[1].set_title('Saliency SNR')
        axs_dist[1].set_xlabel('Score')
        axs_dist[1].grid(True, ls='--', alpha=0.6)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        dist_plot_path = f'{model_name}_metric_distributions.png'
        plt.savefig(dist_plot_path)
        plt.close(fig_dist)
        print(f"Saved metric distribution plots to {dist_plot_path}")
        
    return accuracy, mean_auc, mean_snr

# --------------------------------------------------------------------------- #
# 5. Baseline & Experiment Runner
# --------------------------------------------------------------------------- #

class LogisticRegression(nn.Module):
    """A simple logistic regression model for baseline comparison."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)

def run_gc_logit_regression(train_ds, val_ds, test_ds, dev):
    """Trains and evaluates a logistic regression model on GC content alone."""
    print("\n--- Running GC-Content-Only Logistic Regression Baseline ---")
    
    def extract_gc(dataset):
        gcs, labels = [], []
        # Handle both Subset and regular Dataset objects
        source_dataset = dataset.dataset if isinstance(dataset, torch.utils.data.Subset) else dataset
        indices = dataset.indices if isinstance(dataset, torch.utils.data.Subset) else range(len(source_dataset))
        
        for i in indices:
            x, y, _ = source_dataset[i]
            gc_content = (x[1].sum() + x[2].sum()) / SEQ_LEN
            gcs.append(gc_content.item())
            labels.append(y.item())
        return torch.tensor(gcs).float().unsqueeze(1), torch.tensor(labels).float()

    X_train_gc, y_train = extract_gc(train_ds)
    X_val_gc, y_val = extract_gc(val_ds)
    X_test_gc, y_test = extract_gc(test_ds)

    train_gc_ds = TensorDataset(X_train_gc, y_train)
    val_gc_ds = TensorDataset(X_val_gc, y_val)
    test_gc_ds = TensorDataset(X_test_gc, y_test)

    train_gc_dl = DataLoader(train_gc_ds, batch_size=DEFAULT_BATCH_SIZE, shuffle=True)
    val_gc_dl = DataLoader(val_gc_ds, batch_size=DEFAULT_BATCH_SIZE * 2)
    test_gc_dl = DataLoader(test_gc_ds, batch_size=DEFAULT_BATCH_SIZE * 2)

    model = LogisticRegression().to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(20): # Usually converges fast
        model.train()
        for x_gc, y_gc in train_gc_dl:
            x_gc, y_gc = x_gc.to(dev), y_gc.to(dev)
            optimizer.zero_grad()
            logits = model(x_gc)
            loss = loss_fn(logits, y_gc)
            loss.backward()
            optimizer.step()

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x_gc, y_gc in test_gc_dl:
            x_gc, y_gc = x_gc.to(dev), y_gc.to(dev)
            logits = model(x_gc)
            preds = torch.sigmoid(logits)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_gc.cpu().numpy())
            
    auroc = roc_auc_score(all_labels, all_preds)
    print(f"  GC-Only LogReg Test AUROC: {auroc:.4f}")
    return auroc


if __name__ == "__main__":
    try:
        torch.multiprocessing.set_start_method('forkserver', force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(description="Test new data generator with multiple seeds.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Max training epochs per model.")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Training batch size.")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers.")
    parser.add_argument("--gc_pos", type=float, default=0.6, help="GC content for positive samples' background.")
    parser.add_argument("--conservation", type=float, default=0.85, help="Conservation level for causal motifs.")
    parser.add_argument("--target_signal_frac", type=float, default=DEFAULT_TARGET_SIGNAL_FRAC, help="Target signal fraction for positive examples.")
    parser.add_argument("--motif_repertoire", type=int, default=DEFAULT_MOTIF_REPERTOIRE, help="Number of ancestral motifs.")
    parser.add_argument("--include_partial_negatives", type=lambda x: str(x).lower() not in ['false','0','no'], default=True,
                        help="Include 20% partial-segment decoy negatives (default: True).")
    args = parser.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {dev}")

    seeds = [42, 123, 456, 789, 1011]
    all_results = []

    for i, seed in enumerate(seeds):
        print(f"\n--- Running experiment {i+1}/{len(seeds)} with seed {seed} ---")
        
        set_seeds(seed)
        main_ds, gen_summary = generate_dataset(
            gc_pos=args.gc_pos,
            conservation=args.conservation,
            target_signal_frac=args.target_signal_frac,
            motif_repertoire=args.motif_repertoire,
            include_partial_negatives=args.include_partial_negatives
        )
        print("[DGP]", json.dumps(gen_summary, separators=(',',':')))
        
        train_size = int(0.7 * len(main_ds))
        val_size = int(0.15 * len(main_ds))
        test_size = len(main_ds) - train_size - val_size
        train_ds, val_ds, test_ds = random_split(main_ds, [train_size, val_size, test_size])

        # Run GC-only baseline model
        gc_auroc = run_gc_logit_regression(train_ds, val_ds, test_ds, dev)

        persistent = args.num_workers > 0
        train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, persistent_workers=persistent)
        val_dl = DataLoader(val_ds, batch_size=args.batch_size * 2, num_workers=args.num_workers, pin_memory=True, persistent_workers=persistent)
        test_dl = DataLoader(test_ds, batch_size=args.batch_size * 2, num_workers=args.num_workers, pin_memory=True, persistent_workers=persistent)

        set_seeds(42) # Fixed seed for model initialization
        model = TinyCNN().to(dev)
        if hasattr(torch, "compile"):
            try:
                model = torch.compile(model)
            except Exception as e:
                print(f"  torch.compile() failed, proceeding without it: {e}")
        
        train_standard(model, train_dl, val_dl, args.epochs, dev)
        
        produce_plots = (i == 0) # Only plot for the first seed run
        model_name = f"run_seed_{seed}"
        acc, auc, snr = evaluate_model(model, test_dl, dev, model_name=model_name, produce_plots=produce_plots)
        
        all_results.append({
            'seed': seed,
            'accuracy': acc,
            'saliency_auc': auc,
            'saliency_snr': snr,
            'gc_auroc': gc_auroc,
            'realised_frac': gen_summary['avg_realised_frac']
        })

    print("\n--- All runs complete. Summary: ---")
    df_results = pd.DataFrame(all_results)
    print(df_results.to_string())

    print("\n--- Averages across all runs: ---")
    print(df_results.mean().to_string())

    print("\nAnalysis complete.") 