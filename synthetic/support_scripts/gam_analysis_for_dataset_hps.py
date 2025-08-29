import argparse
import os
import random
import sys
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from captum.attr import IntegratedGradients
from torch.utils.data import DataLoader, Dataset, random_split
from torch.cuda.amp import autocast, GradScaler
import itertools
import glob

# Check for pygam installation
try:
    from pygam import GAM, s, te
except ImportError:
    print("Error: pygam is not installed. Please install it using 'pip install pygam'")
    sys.exit(1)

# --------------------------------------------------------------------------- #
# 1. Configuration & Utilities
# --------------------------------------------------------------------------- #

# Directory to cache synthetic datasets
DATASET_CACHE_DIR = "dataset_cache_gam"
os.makedirs(DATASET_CACHE_DIR, exist_ok=True)

# Default training parameters
DEFAULT_BATCH_SIZE = 512
DEFAULT_EPOCHS = 50
SEQ_LEN = 1000
CHUNK_LEN = 60
N_TOTAL = 5000  # Smaller dataset per point for faster iteration

def set_seeds(seed_value: int = 42) -> None:
    """Sets random seeds for reproducibility."""
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

set_seeds(42)

ALPH = np.array(list("ACGT"), dtype="U1")
to_ix = {b: i for i, b in enumerate(ALPH)}

def sample_background(length: int, gc: float) -> np.ndarray:
    """iid sampling with given GC content"""
    p = np.array([(1 - gc) / 2, gc / 2, gc / 2, (1 - gc) / 2])
    return np.random.choice(ALPH, size=length, p=p)

def random_chunk(length: int) -> np.ndarray:
    """60-bp random chunk with balanced GC"""
    return sample_background(length, 0.50)

def mutate(chunk: np.ndarray, conservation: float) -> np.ndarray:
    """Return a new chunk with given conservation level"""
    mutated_chunk = chunk.copy()
    n_to_mutate = int(len(chunk) * (1.0 - conservation))
    pos_to_mutate = np.random.choice(len(chunk), n_to_mutate, replace=False)
    for pos in pos_to_mutate:
        original_base = mutated_chunk[pos]
        mutated_chunk[pos] = np.random.choice(np.setdiff1d(ALPH, [original_base]))
    return mutated_chunk

def embed(seq: np.ndarray, chunk: np.ndarray) -> Tuple[np.ndarray, int]:
    """Insert chunk at random position"""
    L, l = len(seq), len(chunk)
    start = np.random.randint(0, L - l + 1)
    seq[start:start + l] = chunk
    return seq, start

def one_hot(seq: np.ndarray) -> np.ndarray:
    """(L,) char -> (4,L) float32 one-hot"""
    arr = np.zeros((4, len(seq)), dtype=np.float32)
    for i, b in enumerate(seq):
        arr[to_ix[b], i] = 1.0
    return arr

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

def _dataset_cache_path(gc_pos: float, conservation: float) -> str:
    return os.path.join(DATASET_CACHE_DIR, f"gc_{gc_pos:.4f}_cons_{conservation:.4f}.npz")

def generate_dataset(gc_pos: float, conservation: float):
    """Generates the dataset for a given hyperparameter combination."""
    POS_N = N_TOTAL // 2
    NEG_N = N_TOTAL - POS_N
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

def load_or_generate_dataset(gc_pos: float, conservation: float):
    """Load dataset from cache if present, otherwise generate and cache it."""
    cache_path = _dataset_cache_path(gc_pos, conservation)
    if os.path.exists(cache_path):
        data = np.load(cache_path)
        X = torch.tensor(data["X"], dtype=torch.float32)
        y = torch.tensor(data["y"], dtype=torch.float)
        masks = data["masks"]
        return SeqDS(X, y, masks)

    ds = generate_dataset(gc_pos=gc_pos, conservation=conservation)
    np.savez(cache_path, X=ds.x.cpu().numpy(), y=ds.y.cpu().numpy(), masks=ds.m)
    return ds

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
        return logits.squeeze(-1), None # Return None for compatibility

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

        print(f"    Epoch {epoch + 1}/{epochs}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        if (best_val_loss - avg_val_loss) > early_stopping_min_delta:
            best_val_loss = avg_val_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
        
        if early_stopping_counter >= early_stopping_patience:
            print(f"  -> Early stopping at epoch {epoch + 1}.")
            break

def find_adversarial_baseline_pgd(model, xb, yb, dev, num_iter=20, epsilon=0.1):
    """Finds a baseline for IG using PGD."""
    adv_xb = xb.clone().detach()
    with torch.no_grad(), autocast():
        initial_logits, _ = model(adv_xb)
        initial_pred_class = (initial_logits > 0).float()

    if initial_pred_class.item() != yb.item() or yb.item() == 0:
        return torch.zeros_like(xb, device=dev) # Return zero baseline if initial prediction is wrong or for negative samples

    loss_fn = nn.BCEWithLogitsLoss()
    step_size = epsilon / 10.0

    for _ in range(num_iter):
        adv_xb.requires_grad = True
        with autocast():
            logits, _ = model(adv_xb)
            loss = loss_fn(logits, yb.expand_as(logits))
        model.zero_grad()
        loss.backward()
        
        with torch.no_grad():
            adv_xb = adv_xb + step_size * adv_xb.grad.data.sign()
            delta = torch.clamp(adv_xb - xb, -epsilon, epsilon)
            adv_xb = torch.clamp(xb + delta, 0, 1)
            current_logits, _ = model(adv_xb)
            if (current_logits > 0).float().item() != initial_pred_class.item():
                return adv_xb.detach()
    
    return torch.zeros_like(xb, device=dev) # Return zero baseline if no flip found

def evaluate_model(model, test_dl, dev):
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
        return accuracy, 0.0, 0.0, 0.0

    sample_n = min(50, len(positive_indices))
    idxs = random.sample(positive_indices, sample_n)

    results = []
    for idx in idxs:
        xb, yb_scalar, mask = test_ds[idx]
        xb = xb.unsqueeze(0).to(dev)
        yb = torch.tensor([yb_scalar], device=dev, dtype=torch.float)
        
        baseline = find_adversarial_baseline_pgd(model, xb, yb, dev)
        attributions = ig.attribute(xb, baselines=baseline, target=0).abs().sum(1).squeeze(0).cpu().numpy()
        
        window_sums = np.convolve(attributions, np.ones(CHUNK_LEN), mode='valid')
        best_window_start = np.argmax(window_sums)
        pred_mask = np.zeros(SEQ_LEN, dtype=bool)
        pred_mask[best_window_start:best_window_start + CHUNK_LEN] = True
        
        inter = (pred_mask & mask).sum()
        union = (pred_mask | mask).sum()
        iou = inter / union if union else 0

        inside_scores = attributions[mask]
        outside_scores = attributions[~mask]
        saliency_auc = (inside_scores[:, None] > outside_scores[None, :]).mean()
        
        sum_sq_inside = np.sum(inside_scores**2)
        sum_sq_total = np.sum(attributions**2)
        saliency_snr = sum_sq_inside / (sum_sq_total + 1e-9)

        results.append({'iou': iou, 'auc': saliency_auc, 'snr': saliency_snr})

    mean_iou = np.mean([r['iou'] for r in results])
    mean_auc = np.mean([r['auc'] for r in results])
    mean_snr = np.mean([r['snr'] for r in results])
    
    print(f"    Accuracy: {accuracy:.3f}, wIoU: {mean_iou:.3f}, SaliencyAUC: {mean_auc:.3f}, SaliencySNR: {mean_snr:.3f}")
    return accuracy, mean_iou, mean_auc, mean_snr

def run_single_experiment(gc_pos: float, conservation: float, args: argparse.Namespace):
    """
    Runs the full train-and-evaluate pipeline for a single
    (gc_pos, conservation) combination.
    """
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- Running: GC={gc_pos:.3f}, Cons={conservation:.3f} ---")
    
    # 1. Data
    # Use the combo as a seed for data splitting to ensure consistency if re-run
    seed = int((gc_pos * 1000) + (conservation * 1000))
    set_seeds(seed)
    main_ds = load_or_generate_dataset(gc_pos=gc_pos, conservation=conservation)
    train_size = int(0.7 * len(main_ds))
    val_size = int(0.15 * len(main_ds))
    test_size = len(main_ds) - train_size - val_size
    train_ds, val_ds, test_ds = random_split(main_ds, [train_size, val_size, test_size])

    # Use persistent_workers=True to avoid re-initializing workers between epochs,
    # which is a major source of slowdowns.
    persistent = args.num_workers > 0
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, persistent_workers=persistent)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size * 2, num_workers=args.num_workers, pin_memory=True, persistent_workers=persistent)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size * 2, num_workers=args.num_workers, pin_memory=True, persistent_workers=persistent)

    # 2. Model & Training
    set_seeds(42) # Use fixed seed for model initialization for fair comparison
    model = TinyCNN().to(dev)
    if hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"  torch.compile() failed, proceeding without it: {e}")
    train_standard(model, train_dl, val_dl, args.epochs, dev)

    # 3. Evaluation
    return evaluate_model(model, test_dl, dev)

# --------------------------------------------------------------------------- #
# 5. Experiment Runner
# --------------------------------------------------------------------------- #

def run_experiment_suite(args):
    """Main loop to train models across the hyperparameter space (for local runs)."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {dev}")

    # Generate a grid of hyperparameter combinations
    gc_space = np.linspace(0.50, 0.70, args.num_gc_steps)
    cons_space = np.linspace(0.50, 0.95, args.num_cons_steps)
    hp_combos = list(itertools.product(gc_space, cons_space))
    
    print(f"Exploring {len(hp_combos)} combinations of (GC, Conservation)...")

    results_data = []
    for i, (gc_pos, conservation) in enumerate(hp_combos):
        acc, wiou, auc, snr = run_single_experiment(gc_pos, conservation, args)
        results_data.append({
            'gc_pos': gc_pos,
            'conservation': conservation,
            'accuracy': acc,
            'wIoU': wiou,
            'saliency_auc': auc,
            'saliency_snr': snr
        })

    return pd.DataFrame(results_data)

def run_single_combo_for_slurm(args):
    """
    Runs a single experiment combination based on SLURM's array_idx and
    saves the result to a unique .npz file.
    """
    # Generate the full list of combos to find our specific one
    gc_space = np.linspace(0.50, 0.70, args.num_gc_steps)
    cons_space = np.linspace(0.50, 0.95, args.num_cons_steps)
    hp_combos = list(itertools.product(gc_space, cons_space))

    # Select the combo for this array job
    array_idx = args.array_idx
    if array_idx < 0 or array_idx >= len(hp_combos):
        print(f"Error: --array_idx {array_idx} is out of range for {len(hp_combos)} total jobs.")
        sys.exit(1)
    
    gc_pos, conservation = hp_combos[array_idx]
    
    acc, wiou, auc, snr = run_single_experiment(gc_pos, conservation, args)

    # Save the result to a unique file to avoid race conditions
    results_dir = os.path.join(args.output_dir, "npz_results")
    os.makedirs(results_dir, exist_ok=True)
    result_path = os.path.join(results_dir, f"result_{array_idx}.npz")
    
    np.savez(
        result_path,
        gc_pos=gc_pos,
        conservation=conservation,
        accuracy=acc,
        wIoU=wiou,
        saliency_auc=auc,
        saliency_snr=snr
    )
    print(f"\nSaved single result to {result_path}")

# --------------------------------------------------------------------------- #
# 6. GAM Analysis & Plotting
# --------------------------------------------------------------------------- #

def aggregate_results_and_analyze(args):
    """Finds all npz files, aggregates them, and runs GAM analysis."""
    print("--- Aggregating results and running GAM analysis ---")
    npz_dir = os.path.join(args.output_dir, "npz_results")
    if not os.path.isdir(npz_dir):
        print(f"Error: The results directory '{npz_dir}' could not be found.")
        sys.exit(1)

    npz_files = glob.glob(os.path.join(npz_dir, "result_*.npz"))
    if not npz_files:
        print(f"No .npz result files found in {npz_dir}; nothing to aggregate.")
        sys.exit(1)
    
    print(f"Found {len(npz_files)} result files, building master dataframe...")
    all_data = []
    for f_path in npz_files:
        try:
            data = np.load(f_path)
            all_data.append({
                'gc_pos': data['gc_pos'].item(),
                'conservation': data['conservation'].item(),
                'accuracy': data['accuracy'].item(),
                'wIoU': data['wIoU'].item(),
                'saliency_auc': data['saliency_auc'].item(),
                'saliency_snr': data['saliency_snr'].item()
            })
        except Exception as e:
            print(f"Could not process file {f_path}: {e}")

    df_results = pd.DataFrame(all_data)
    
    # Save the aggregated dataframe
    master_csv_path = os.path.join(args.output_dir, 'master_results.csv')
    df_results.to_csv(master_csv_path, index=False)
    print(f"\nSaved master results table to {master_csv_path}")

    # Run the GAM analysis and plotting
    analyze_and_plot_gams(df_results, args.output_dir)

def analyze_and_plot_gams(df_results, output_dir):
    """Fits GAMs with interaction terms and plots the learned splines."""
    print("\n--- Fitting Generalized Additive Models (GAMs) ---")
    os.makedirs(output_dir, exist_ok=True)
    
    X = df_results[['gc_pos', 'conservation']].values
    metrics = ['accuracy', 'wIoU', 'saliency_auc', 'saliency_snr']
    
    for metric in metrics:
        y = df_results[metric].values
        
        print(f"\n--- GAM for metric: {metric} ---")
        
        # Fit GAM: s(0) for gc_pos, s(1) for conservation, te(0, 1) for the interaction
        # Reduced n_splines from 20 to 10 to prevent overfitting with a sample size of 225.
        lam = 0.6
        n_splines = 10
        gam = GAM(s(0, n_splines=n_splines, lam=lam) + 
                  s(1, n_splines=n_splines, lam=lam) + 
                  te(0, 1, n_splines=(n_splines, n_splines), lam=lam)).fit(X, y)
        
        print(gam.summary())

        # Plotting
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'GAM Analysis for {metric}', fontsize=16)

        # Plot for gc_pos
        ax = axes[0, 0]
        XX_gc = gam.generate_X_grid(term=0)
        pdep_gc, confi_gc = gam.partial_dependence(term=0, X=XX_gc, width=0.95)
        ax.plot(XX_gc[:, 0], pdep_gc)
        ax.plot(XX_gc[:, 0], confi_gc, c='r', ls='--')
        ax.set_title('Partial Effect of GC Content')
        ax.set_xlabel('GC Content')
        ax.set_ylabel('s(GC Content)')
        
        # Plot for conservation
        ax = axes[0, 1]
        XX_cons = gam.generate_X_grid(term=1)
        pdep_cons, confi_cons = gam.partial_dependence(term=1, X=XX_cons, width=0.95)
        ax.plot(XX_cons[:, 1], pdep_cons)
        ax.plot(XX_cons[:, 1], confi_cons, c='r', ls='--')
        ax.set_title('Partial Effect of Conservation')
        ax.set_xlabel('Causal Motif Conservation')
        ax.set_ylabel('s(Conservation)')
        
        # Common grid for the 2D plots
        XX_grid = gam.generate_X_grid(term=2, n=50)
        gc_grid = XX_grid[:,0].reshape((50, 50))
        cons_grid = XX_grid[:,1].reshape((50, 50))

        # Plot for interaction term
        ax = axes[1, 0]
        pdep_inter = gam.partial_dependence(term=2, X=XX_grid)
        pdep_grid = pdep_inter.reshape((50, 50))
        im = ax.pcolormesh(gc_grid, cons_grid, pdep_grid, cmap='viridis', shading='auto')
        ax.set_title('Interaction Effect: te(GC, Conservation)')
        ax.set_xlabel('GC Content')
        ax.set_ylabel('Conservation')
        fig.colorbar(im, ax=ax, label='Interaction Effect')

        # Plot for full model prediction
        ax = axes[1, 1]
        full_pred = gam.predict(XX_grid)
        pred_grid = full_pred.reshape((50, 50))
        im = ax.pcolormesh(gc_grid, cons_grid, pred_grid, cmap='viridis', shading='auto')
        ax.set_title(f'Full Model Prediction for {metric}')
        ax.set_xlabel('GC Content')
        ax.set_ylabel('Conservation')
        fig.colorbar(im, ax=ax, label=f'Predicted {metric}')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plot_path = os.path.join(output_dir, f'gam_plot_{metric}.png')
        plt.savefig(plot_path, dpi=200)
        plt.close(fig)
        print(f"Saved GAM plot to {plot_path}")

# --------------------------------------------------------------------------- #
# 7. Main Entry-point
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    # Set start method for multiprocessing to be safer on cluster environments
    try:
        torch.multiprocessing.set_start_method('forkserver', force=True)
        print("Multiprocessing start method set to 'forkserver'.")
    except RuntimeError:
        print("Multiprocessing start method already set or cannot be changed.")

    parser = argparse.ArgumentParser(description="Analyze dataset hyperparameter impact with GAMs.")
    parser.add_argument("--output_dir", type=str, default="gam_analysis_results", help="Where to save results and plots.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Max training epochs per model.")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Training batch size.")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers.")
    parser.add_argument("--num_gc_steps", type=int, default=10, help="Number of steps for GC content.")
    parser.add_argument("--num_cons_steps", type=int, default=10, help="Number of steps for conservation.")
    parser.add_argument(
        "--array_idx",
        type=int,
        default=None,
        help="Index from SLURM_ARRAY_TASK_ID for a single experiment combo.",
    )
    parser.add_argument(
        "--aggregate_only",
        action="store_true",
        help="Skip training, only aggregate npz results and generate plots.",
    )
    
    args = parser.parse_args()

    if args.aggregate_only:
        aggregate_results_and_analyze(args)
    elif args.array_idx is not None:
        run_single_combo_for_slurm(args)
    else:
        # Fallback to a local run for testing
        print("--- Running full experiment suite locally ---")
        df_results = run_experiment_suite(args)
        
        # Save the raw results
        os.makedirs(args.output_dir, exist_ok=True)
        results_csv_path = os.path.join(args.output_dir, 'master_results.csv')
        df_results.to_csv(results_csv_path, index=False)
        print(f"\nSaved master results table to {results_csv_path}")

        # Run the GAM analysis and plotting
        analyze_and_plot_gams(df_results, args.output_dir)

    print("\nAnalysis complete.") 