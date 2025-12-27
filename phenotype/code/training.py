import os
import sys
from pathlib import Path
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
from model import SporulationModel, load_best_hparams_from_summary

try:
    from phenotype_utils import build_labels_map_and_classes, DATA_ROOT
except ImportError:  # pragma: no cover
    from .phenotype_utils import build_labels_map_and_classes, DATA_ROOT  # type: ignore

# --- Configuration ---
SEQ_LEN = 1_000_000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = str(DATA_ROOT)
# Phenotype-agnostic: read labels from the metadata Excel (phenotype column).
METADATA_XLSX = 'sporulation/microbe.cards table S1.xlsx'
PHENOTYPE_COL_DEFAULT = 'Spore formation'
FILE_COL_DEFAULT = 'Fasta file'
LOG_DIR = 'slurm_results/phenotype_experiment/tensorboard'
SCHEDULER_PATIENCE = 2
SCHEDULER_FACTOR = 0.5
EARLY_STOPPING_PATIENCE = 7
EARLY_STOPPING_MIN_DELTA = 1e-4
EARLY_STOP_START_EPOCH = 3
EPOCH_BUDGET = 4096
VAL_EPOCH_BUDGET = 2048
VAL_STEPS = 256

# Load best HPs from Optuna summary for optimization settings
_SUMMARY_PATH = os.path.join('spore_optuna', 'sporo_full_std_v2_cont_exp_sporulation', 'summary.txt')
_HP = load_best_hparams_from_summary(_SUMMARY_PATH)
LEARNING_RATE = float(_HP.get('lr', 1.0e-3))
WEIGHT_DECAY = float(_HP.get('weight_decay', 1.0e-6))
BATCH_SIZE = int(_HP.get('batch_size', 20))
GRAD_CLIP = float(_HP.get('grad_clip', 6.0))
EPOCHS = 40

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

# --- Metadata/labels utilities (phenotype-agnostic) ---

def _norm_basename(val: object) -> str:
    s = str(val) if not pd.isna(val) else ""
    s = os.path.basename(s)
    return s.strip().lower()

def read_metadata_table(xlsx_path: str) -> pd.DataFrame:
    md = pd.read_excel(xlsx_path)
    if "Fasta file" not in md.columns:
        raise ValueError("Expected column 'Fasta file' in metadata Excel")
    md["Fasta file_norm"] = md["Fasta file"].map(_norm_basename)
    return md

def _count_labeled_fastas_in_dir(dir_path: str, labels_map: dict) -> int:
    if not os.path.isdir(dir_path):
        return 0
    exts = ('.fasta', '.fa', '.fna')
    count = 0
    try:
        for name in os.listdir(dir_path):
            if name.endswith(exts):
                if labels_map.get(str(name).strip().lower()) is not None:
                    count += 1
    except Exception:
        return 0
    return count

def ensure_data_quality(metadata_df: pd.DataFrame, base_dir: str, phenotype_col: str, file_col: str,
                        min_train: int = 50, min_val: int = 10, min_test: int = 10) -> None:
    train_dir = os.path.join(base_dir, 'train')
    val_dir = os.path.join(base_dir, 'validation')
    test_dir = os.path.join(base_dir, 'test')
    labels_map, _classes = build_labels_map_and_classes(
        metadata_df,
        phenotype_col=phenotype_col,
        file_col=file_col,
        train_dirs=[train_dir],
    )
    n_train = _count_labeled_fastas_in_dir(train_dir, labels_map)
    n_val = _count_labeled_fastas_in_dir(val_dir, labels_map)
    n_test = _count_labeled_fastas_in_dir(test_dir, labels_map)
    if (n_train < min_train) or (n_val < min_val) or (n_test < min_test):
        print(f"[data-quality] Insufficient labeled FASTA counts: train={n_train}, val={n_val}, test={n_test}. "
              f"Require train>={min_train}, val>={min_val}, test>={min_test}. Aborting.", flush=True)
        sys.exit(1)

# --- PyTorch Dataset ---

class FastaDataset(Dataset):
    """
    Dataset that preloads raw sequences, and on-the-fly samples a random
    chunk of `seq_len` and one-hot encodes it.
    For validation, sampling is deterministic based on a fixed seed.
    """
    def __init__(self, data_dir, labels_df, seq_len, epoch_budget=None, is_validation=False, phenotype_col: str = PHENOTYPE_COL_DEFAULT, file_col: str = FILE_COL_DEFAULT):
        self.data_dir = data_dir
        # Build a mapping from FASTA basename -> class id using metadata Excel
        try:
            self.labels_map, self.classes = build_labels_map_and_classes(
                labels_df,
                phenotype_col=phenotype_col,
                file_col=file_col,
                train_dirs=[Path(BASE_DIR) / "train"],
            )
        except Exception:
            # Fallback for legacy CSV with columns ['file','ability_TRUE']
            if 'file' in labels_df.columns and 'ability_TRUE' in labels_df.columns:
                self.labels_map = {str(row['file']).strip().lower(): int(row['ability_TRUE']) for _, row in labels_df.iterrows()}
                self.classes = sorted(list({0, 1}))
            else:
                raise
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
            label = self.labels_map.get(str(file_name).strip().lower())
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

def validate_epoch(model, loader, loss_fn, max_batches=None):
    """Calculates the loss and balanced accuracy on a validation set.

    If max_batches is provided, evaluation stops after that many batches.
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for bi, (xb, yb) in enumerate(loader):
            if max_batches is not None and bi >= max_batches:
                break
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

def train(args=None):
    """Main training function."""
    print(f"Using device: {DEVICE}")

    # Load labels
    metadata_xlsx = METADATA_XLSX if args is None or getattr(args, 'metadata_xlsx', None) is None else args.metadata_xlsx
    phenotype_col = PHENOTYPE_COL_DEFAULT if args is None or getattr(args, 'phenotype_col', None) is None else args.phenotype_col
    file_col = FILE_COL_DEFAULT if args is None or getattr(args, 'file_col', None) is None else args.file_col
    metadata_df = read_metadata_table(metadata_xlsx)
    ensure_data_quality(metadata_df, BASE_DIR, phenotype_col, file_col, min_train=50, min_val=10, min_test=10)

    # Create datasets and dataloaders
    train_dataset = FastaDataset(os.path.join(BASE_DIR, 'train'), metadata_df, SEQ_LEN, epoch_budget=EPOCH_BUDGET, phenotype_col=phenotype_col, file_col=file_col)
    val_dataset = FastaDataset(os.path.join(BASE_DIR, 'validation'), metadata_df, SEQ_LEN,
                               epoch_budget=VAL_EPOCH_BUDGET, is_validation=True, phenotype_col=phenotype_col, file_col=file_col)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=min(BATCH_SIZE, 32), shuffle=False, num_workers=2)

    # Determine number of classes from training dataset
    train_labels = [sample['label'] for sample in train_dataset.sequences]
    num_classes = int(pd.Series(train_labels).nunique()) if len(train_labels) > 0 else 2

    # Model setup
    summary_path = _SUMMARY_PATH if args is None or getattr(args, 'summary_path', None) is None else args.summary_path
    model = SporulationModel(summary_path=summary_path, num_classes=num_classes).to(DEVICE)

    # Calculate class weights for loss function (supports multi-class)
    class_counts = pd.Series(train_labels).value_counts().sort_index()
    num_classes = max(1, len(class_counts))
    class_weights = (len(train_labels) / (num_classes * class_counts)).values
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
    print(f"Calculated class weights: {class_weights_tensor.cpu().numpy()}")
    
    loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=SCHEDULER_PATIENCE, factor=SCHEDULER_FACTOR)
    
    
    scaler = GradScaler()
    writer = SummaryWriter(LOG_DIR)
    
    # Ensure model save directory exists (under phenotype, per phenotype slug)
    phenotype_slug = phenotype_col.strip().lower().replace(' ', '_')
    model_root = os.path.join('phenotype', 'model', phenotype_slug)
    os.makedirs(model_root, exist_ok=True)
    
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
            # Grad clip from best Optuna trial
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_train_loss = total_loss / num_batches
        avg_val_loss, val_bal_acc = validate_epoch(model, val_loader, loss_fn, max_batches=VAL_STEPS)
        
        scheduler.step(avg_val_loss)

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
                torch.save(model.state_dict(), os.path.join(model_root, 'best_model.pth'))
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
            torch.save(model.state_dict(), os.path.join(model_root, 'best_model.pth'))

    writer.close()
    print("Training finished.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train a phenotype-agnostic genome sequence classifier.')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'find_lr'],
                        help='Set to "find_lr" to run the LR finder, otherwise trains the model.')
    parser.add_argument('--metadata_xlsx', type=str, default=METADATA_XLSX, help='Path to metadata Excel with phenotype columns')
    parser.add_argument('--phenotype_col', type=str, default=PHENOTYPE_COL_DEFAULT, help='Phenotype column name to use as label')
    parser.add_argument('--file_col', type=str, default=FILE_COL_DEFAULT, help='Column containing FASTA filenames')
    parser.add_argument('--summary_path', type=str, default=_SUMMARY_PATH, help='Optuna summary.txt path for best HPs')
    args = parser.parse_args()

    if args.mode == 'find_lr':
        # --- Setup for LR Finder ---
        print(f"Using device: {DEVICE}")
        metadata_df = read_metadata_table(args.metadata_xlsx)
        train_dataset = FastaDataset(os.path.join(BASE_DIR, 'train'), metadata_df, SEQ_LEN, phenotype_col=args.phenotype_col, file_col=args.file_col)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
        
        model = SporulationModel(summary_path=args.summary_path).to(DEVICE)
        
        train_labels = [sample['label'] for sample in train_dataset.sequences]
        class_counts = pd.Series(train_labels).value_counts().sort_index()
        class_weights = (len(train_labels) / (2 * class_counts)).values
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
        loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor)
        
        # Make sure slurm_results directory exists
        os.makedirs('slurm_results', exist_ok=True)
        
        find_lr(model, train_loader, loss_fn)
    else:
        train(args)
