import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import balanced_accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model import SporulationModel

# --- Configuration ---
SEQ_LEN = 1_000_000
BATCH_SIZE = 16
EPOCHS = 200
LEARNING_RATE = 1e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = 'sporulation/data'
CSV_PATH = 'sporulation/sporeinfo.csv'
LOG_DIR = 'slurm_results/sporulation_experiment/tensorboard'
WEIGHT_DECAY = 1e-6
SCHEDULER_PATIENCE = 2
SCHEDULER_FACTOR = 0.5
EARLY_STOPPING_PATIENCE = 7
EARLY_STOPPING_MIN_DELTA = 1e-4
WARMUP_EPOCHS = 5
EARLY_STOP_START_EPOCH = 5
EPOCH_BUDGET = 20000
VAL_EPOCH_BUDGET = 10000

# --- FASTA Processing Utilities ---

def parse_fasta(file_path):
    """A simple FASTA parser."""
    sequence = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('>'):
                continue
            sequence.append(line.strip())
    return "".join(sequence)

def one_hot_encode(seq):
    """One-hot encodes a DNA sequence."""
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    one_hot = np.zeros((4, len(seq)), dtype=np.float32)
    for i, base in enumerate(seq.upper()):
        if base in mapping:
            one_hot[mapping[base], i] = 1.0
    return torch.from_numpy(one_hot)

# --- PyTorch Dataset ---

class FastaDataset(Dataset):
    """
    Dataset that preloads raw sequences, and on-the-fly samples a random
    chunk of `seq_len` and one-hot encodes it.
    For validation, sampling is deterministic based on a fixed seed.
    """
    def __init__(self, data_dir, labels_df, seq_len, epoch_budget=None, is_validation=False):
        self.data_dir = data_dir
        self.labels_map = {row['file']: row['ability_TRUE'] for _, row in labels_df.iterrows()}
        self.seq_len = seq_len
        self.is_validation = is_validation
        self.sequences = []
        self._preload_sequences()

        if epoch_budget:
            self.epoch_len = epoch_budget
        else:
            # Estimate epoch length based on total sequence length
            total_len = sum(len(s['sequence']) for s in self.sequences)
            self.epoch_len = total_len // self.seq_len
        
        if self.is_validation:
            self._generate_deterministic_samples()

    def _preload_sequences(self):
        print(f"Preloading sequences from {self.data_dir}...")
        file_list = [f for f in os.listdir(self.data_dir) if f.endswith(('.fasta', '.fa', '.fna'))]
        total_files = len(file_list)

        for i, file_name in enumerate(file_list):
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{total_files} files...")

            file_path = os.path.join(self.data_dir, file_name)
            label = self.labels_map.get(file_name)
            if label is None:
                continue

            sequence = parse_fasta(file_path)
            
            if not sequence:
                continue

            if len(sequence) < self.seq_len:
                # Pad shorter sequences
                sequence = sequence.ljust(self.seq_len, 'N')
            
            self.sequences.append({'sequence': sequence, 'label': label})
        
        print(f"Preloading complete. Total genomes: {len(self.sequences)}")

    def _generate_deterministic_samples(self):
        print("Generating deterministic validation samples...")
        self.deterministic_samples = []
        rng = np.random.RandomState(42)  # Use a fixed seed for reproducibility
        for i in range(self.epoch_len):
            genome_idx = i % len(self.sequences)
            sequence_info = self.sequences[genome_idx]
            sequence_str = sequence_info['sequence']
            
            start = 0
            if len(sequence_str) > self.seq_len:
                start = rng.randint(0, len(sequence_str) - self.seq_len + 1)
            
            self.deterministic_samples.append({
                'genome_idx': genome_idx,
                'start': start,
            })

    def __len__(self):
        return self.epoch_len

    def __getitem__(self, idx):
        if self.is_validation:
            sample_info = self.deterministic_samples[idx]
            genome_idx = sample_info['genome_idx']
            start = sample_info['start']
            
            sample = self.sequences[genome_idx]
            sequence_str = sample['sequence'][start : start + self.seq_len]
            label = sample['label']
        else:
            # Determine which genome to use based on index
            genome_idx = idx % len(self.sequences)
            sample = self.sequences[genome_idx]
            sequence_str = sample['sequence']
            label = sample['label']
            
            # Dynamically sample a random chunk
            if len(sequence_str) > self.seq_len:
                start = np.random.randint(0, len(sequence_str) - self.seq_len + 1)
                sequence_str = sequence_str[start : start + self.seq_len]

        # One-hot encode on the fly
        one_hot_seq = one_hot_encode(sequence_str)
        return one_hot_seq, torch.tensor(label, dtype=torch.long)

# --- Validation ---

def validate_epoch(model, loader, loss_fn):
    """Calculates the loss and balanced accuracy on a validation set."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            with autocast():
                logits = model(xb)
                loss = loss_fn(logits, yb)
            total_loss += loss.item()
            num_batches += 1
            
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(yb.cpu().numpy())
            
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    bal_acc = balanced_accuracy_score(all_labels, all_preds) if len(all_labels) > 0 else 0
    return avg_loss, bal_acc

# --- LR Finder ---

def find_lr(model, train_loader, loss_fn, start_lr=1e-8, end_lr=1.0, num_iter=100):
    """Performs a learning rate range test."""
    print("Starting LR Finder...")
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=start_lr, weight_decay=WEIGHT_DECAY)
    
    # Custom scheduler to increase LR exponentially
    gamma = (end_lr / start_lr) ** (1 / (num_iter -1))
    scheduler = LambdaLR(optimizer, lr_lambda=lambda step: gamma ** step)

    lrs = []
    losses = []
    best_loss = float('inf')
    
    # Use a fresh iterator for the loader
    data_iter = iter(train_loader)

    for i in range(num_iter):
        try:
            xb, yb = next(data_iter)
        except StopIteration:
            # Reset iterator if we've gone through the whole dataset
            data_iter = iter(train_loader)
            xb, yb = next(data_iter)
            
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        
        optimizer.zero_grad()
        logits = model(xb)
        loss = loss_fn(logits, yb)
        
        # Stop if loss explodes
        if loss > 4 * best_loss and i > 10:
            print("Loss exploded. Stopping LR Finder.")
            break
        if loss.item() < best_loss:
            best_loss = loss.item()

        loss.backward()
        optimizer.step()
        
        lrs.append(optimizer.param_groups[0]['lr'])
        losses.append(loss.item())
        
        scheduler.step()
        
        if (i + 1) % 10 == 0:
            print(f"  LR Finder Batch {i+1}/{num_iter}: LR={lrs[-1]:.2e}, Loss={losses[-1]:.4f}")

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(lrs, losses)
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.title('Learning Rate Finder')
    plt.grid(True)
    
    # Suggest a learning rate
    # Find the point of steepest descent before the minimum loss
    try:
        min_grad_idx = np.gradient(np.array(losses)).argmin()
        suggested_lr = lrs[min_grad_idx]
        print(f"\nLR Finder finished. Suggested LR (steepest gradient): {suggested_lr:.2e}")
        plt.axvline(x=suggested_lr, color='r', linestyle='--', label=f'Steepest: {suggested_lr:.2e}')
    except (ValueError, IndexError):
        print("\nCould not automatically suggest a learning rate.")

    plt.legend()
    plt.savefig('slurm_results/lr_finder_plot.png')
    print("LR finder plot saved to slurm_results/lr_finder_plot.png")
    plt.show()


# --- Training ---

def train():
    """Main training function."""
    print(f"Using device: {DEVICE}")

    # Load labels
    labels_df = pd.read_csv(CSV_PATH)

    # Create datasets and dataloaders
    train_dataset = FastaDataset(os.path.join(BASE_DIR, 'train'), labels_df, SEQ_LEN, epoch_budget=EPOCH_BUDGET)
    val_dataset = FastaDataset(os.path.join(BASE_DIR, 'validation'), labels_df, SEQ_LEN, 
                               epoch_budget=VAL_EPOCH_BUDGET, is_validation=True)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # Model setup
    model = SporulationModel().to(DEVICE)

    # Calculate class weights for loss function
    train_labels = [sample['label'] for sample in train_dataset.sequences]
    class_counts = pd.Series(train_labels).value_counts().sort_index()
    class_weights = (len(train_labels) / (2 * class_counts)).values
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
    print(f"Calculated class weights: {class_weights_tensor.cpu().numpy()}")
    
    loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=SCHEDULER_PATIENCE, factor=SCHEDULER_FACTOR)
    
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=WARMUP_EPOCHS
    )
    
    
    scaler = GradScaler()
    writer = SummaryWriter(LOG_DIR)
    
    best_val_loss = float('inf')
    early_stopping_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        num_batches = 0
        for i, (xb, yb) in enumerate(train_loader):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            
            optimizer.zero_grad()
            with autocast():
                logits = model(xb)
                loss = loss_fn(logits, yb)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_train_loss = total_loss / num_batches
        avg_val_loss, val_bal_acc = validate_epoch(model, val_loader, loss_fn)
        
        if epoch < WARMUP_EPOCHS:
            warmup_scheduler.step()
        else:
            scheduler.step(avg_val_loss)

        # After warmup, ensure the scheduler's best metric is in sync
        if epoch == WARMUP_EPOCHS:
            scheduler.best = best_val_loss

        current_lr = optimizer.param_groups[0]['lr']
        
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/validation', avg_val_loss, epoch)
        writer.add_scalar('Balanced_Accuracy/validation', val_bal_acc, epoch)
        writer.add_scalar('LR/train', current_lr, epoch)
        
        print(f"Epoch {epoch + 1}/{EPOCHS}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Bal Acc: {val_bal_acc:.4f}, LR: {current_lr:.6f}")
        
        # Early stopping logic based on validation loss
        if epoch >= EARLY_STOP_START_EPOCH:
            if (best_val_loss - avg_val_loss) > EARLY_STOPPING_MIN_DELTA:
                best_val_loss = avg_val_loss
                early_stopping_counter = 0
                torch.save(model.state_dict(), 'slurm_results/sporulation_experiment/best_sporulation_model.pth')
                print(f"  -> New best model saved with val_loss: {best_val_loss:.4f}")
            else:
                early_stopping_counter += 1
            
            if early_stopping_counter >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping at epoch {epoch + 1}")
                break
        elif avg_val_loss < best_val_loss:
             # Still save the best model found during the initial phase
            print(f"  -> New best model saved with val_loss: {avg_val_loss:.4f} (during warmup phase)")
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'slurm_results/sporulation_experiment/best_sporulation_model.pth')

    writer.close()
    print("Training finished.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train a sporulation prediction model.')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'find_lr'],
                        help='Set to "find_lr" to run the LR finder, otherwise trains the model.')
    args = parser.parse_args()

    if args.mode == 'find_lr':
        # --- Setup for LR Finder ---
        print(f"Using device: {DEVICE}")
        labels_df = pd.read_csv(CSV_PATH)
        train_dataset = FastaDataset(os.path.join(BASE_DIR, 'train'), labels_df, SEQ_LEN)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
        
        model = SporulationModel().to(DEVICE)
        
        train_labels = [sample['label'] for sample in train_dataset.sequences]
        class_counts = pd.Series(train_labels).value_counts().sort_index()
        class_weights = (len(train_labels) / (2 * class_counts)).values
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
        loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor)
        
        # Make sure slurm_results directory exists
        os.makedirs('slurm_results', exist_ok=True)
        
        find_lr(model, train_loader, loss_fn)
    else:
        train()
