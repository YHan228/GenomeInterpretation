#!/usr/bin/env python3
"""
Generate sequence logo plots from attribution analysis of toy_slurm.py experiments.
This script retrains only the best-performing models and creates logo visualizations
of their integrated gradient attributions.

Two visualization modes:
1. SHOW_ALL_NUCLEOTIDES = True (Motif Discovery Mode):
   - Shows attributions for all 4 nucleotides at each position
   - Reveals which pattern the model has learned
   - Best for checking if model recovered the causal motif
   
2. SHOW_ALL_NUCLEOTIDES = False (Importance Mode):
   - Shows attributions only for nucleotides present in input
   - Reveals importance of actual sequence elements
   - Best for understanding specific predictions

Usage:
    python toy_logo_plots.py

Author: Assistant, 2025
"""

import os
import sys

# Set environment variable to suppress multiprocessing resource tracker warnings
os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning'

# ============================================================================
# CONFIGURATION - Modify these instead of using command line arguments
# ============================================================================
RESULTS_DIR = "slurm_results/vanilla_run_v5_10seeds"
OUTPUT_DIR = "slurm_results/vanilla_logos"
CHECKPOINT_DIR = "slurm_results/vanilla_logos/checkpoints"
EPOCHS = 100
N_EXAMPLES = 5
SHOW_NEGATIVE = True  # Show negative attribution values
USE_FIXED_HPARAMS = True  # Use gc=0.6, cons=0.7 for all models
FIXED_GC = 0.6
FIXED_CONS = 0.7
SHOW_ALL_NUCLEOTIDES = False  # True: Show what model prefers (motif discovery)
                             # False: Show importance of actual nucleotides
# ============================================================================
import glob
import warnings
import tempfile
import shutil
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt
import logomaker

# Suppress the specific OSError warnings from temporary directory cleanup
warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils.tensorboard")
warnings.filterwarnings("ignore", message=".*Device or resource busy.*")

# Also suppress the actual OSError exceptions during cleanup
import logging
logging.getLogger('torch.utils.tensorboard').setLevel(logging.ERROR)

# Custom error handler for shutil.rmtree
def handle_remove_readonly(func, path, exc):
    """Error handler for Windows readonly files and busy resources."""
    import stat
    import errno
    if exc[1].errno in (errno.EACCES, errno.EBUSY, errno.ENOTEMPTY):
        # Ignore permission errors and busy resources
        pass
    else:
        raise

# Import components from toy_slurm.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from toy_slurm import (
    TinyCNN, SeqDS, load_or_generate_dataset, set_seeds,
    train_standard, train_hotflip, train_direct_hotflip,
    find_adversarial_baseline_pgd_batch_optimized,
    sample_background, one_hot,
    DEFAULT_BATCH_SIZE, DEFAULT_EVAL_BATCH_SIZE,
    SEQ_LEN, CHUNK_LEN, ALPH
)


def find_best_hyperparameters(results_dir: str, target_gc: float = None, target_cons: float = None) -> Dict[str, Dict]:
    """
    Analyze NPZ results to find the best hyperparameters for each model type.
    If target_gc and target_cons are specified, find best epsilon for that specific configuration.
    Returns a dictionary mapping model types to their best settings.
    """
    if target_gc is not None and target_cons is not None:
        print(f"Finding best epsilon values for fixed GC={target_gc}, Conservation={target_cons}...")
    else:
        print("Finding best hyperparameters from existing results...")
    
    # Load all NPZ files
    npz_files = glob.glob(os.path.join(results_dir, 'npz_results', '**', 'multi_seed_results.npz'), recursive=True)
    if not npz_files:
        raise ValueError(f"No result files found in {results_dir}")
    
    # Collect all results
    all_results = []
    for f_path in npz_files:
        try:
            data = np.load(f_path, allow_pickle=True)
            
            # Extract experiment info from path
            path_parts = f_path.split(os.sep)
            if 'iterative_hotflip' in f_path:
                mode = 'iterative_hotflip'
            elif 'direct_hotflip' in f_path:
                mode = 'direct_hotflip'
            else:
                continue
            
            if 'scheduled' in f_path:
                schedule = True
            elif 'no_schedule' in f_path:
                schedule = False
            else:
                continue
            
            gc_pos = float(data['gc_pos'].item())
            conservation = float(data['conservation'].item())
            
            # Get epsilon values
            if mode == 'iterative_hotflip':
                epsilons = data['epsilons']
            else:  # direct_hotflip
                epsilons = data.get('direct_hotflip_epsilons', data.get('epsilons', []))
            
            # Get mean saliency AUC for each epsilon across seeds
            rob_aucs = data.get('rob_aucs', [])
            if len(rob_aucs) > 0:
                # Average across seeds
                mean_aucs_per_epsilon = np.mean(rob_aucs, axis=0)
                
                for i, epsilon in enumerate(epsilons):
                    if i < len(mean_aucs_per_epsilon):
                        all_results.append({
                            'mode': mode,
                            'schedule': schedule,
                            'gc_pos': gc_pos,
                            'conservation': conservation,
                            'epsilon': epsilon,
                            'mean_auc': mean_aucs_per_epsilon[i]
                        })
            
        except Exception as e:
            print(f"Error processing {f_path}: {e}")
            continue
    
    if not all_results:
        raise ValueError("No valid results found")
    
    df = pd.DataFrame(all_results)
    
    # Find best epsilon for each model type
    best_params = {}
    
    # For standard model, use the target values or defaults
    best_params['standard'] = {
        'gc_pos': target_gc if target_gc is not None else 0.6,
        'conservation': target_cons if target_cons is not None else 0.7,
        'epsilon': 0.0
    }
    
    # For each robust model type, find the best performing configuration
    for mode in ['iterative_hotflip', 'direct_hotflip']:
        for schedule in [True, False]:
            model_key = f"{mode}_{'scheduled' if schedule else 'no_schedule'}"
            
            # Filter data for this model type
            model_df = df[(df['mode'] == mode) & (df['schedule'] == schedule)]
            
            # If target gc/cons specified, filter to only those results
            if target_gc is not None and target_cons is not None:
                # Allow small tolerance for floating point comparison
                tolerance = 0.001
                model_df = model_df[
                    (abs(model_df['gc_pos'] - target_gc) < tolerance) & 
                    (abs(model_df['conservation'] - target_cons) < tolerance)
                ]
            
            if not model_df.empty:
                # Find configuration with highest mean AUC
                best_idx = model_df['mean_auc'].idxmax()
                best_row = model_df.loc[best_idx]
                
                best_params[model_key] = {
                    'gc_pos': target_gc if target_gc is not None else best_row['gc_pos'],
                    'conservation': target_cons if target_cons is not None else best_row['conservation'],
                    'epsilon': best_row['epsilon']
                }
                
                if target_gc is not None and target_cons is not None:
                    print(f"Best epsilon for {model_key} at GC={target_gc}, Cons={target_cons}: "
                          f"Epsilon={best_row['epsilon']:.3f}, AUC={best_row['mean_auc']:.3f}")
                else:
                    print(f"Best params for {model_key}: GC={best_row['gc_pos']:.3f}, "
                          f"Cons={best_row['conservation']:.2f}, Epsilon={best_row['epsilon']:.3f}, "
                          f"AUC={best_row['mean_auc']:.3f}")
            else:
                # No results found for this configuration
                if target_gc is not None and target_cons is not None:
                    print(f"WARNING: No results found for {model_key} with GC={target_gc}, Cons={target_cons}")
                    # Use default epsilon
                    best_params[model_key] = {
                        'gc_pos': target_gc,
                        'conservation': target_cons,
                        'epsilon': 0.01  # Default epsilon
                    }
    
    return best_params


def train_best_models(best_params: Dict[str, Dict], device: torch.device, epochs: int = 50, 
                     checkpoint_dir: Optional[str] = None) -> Dict[str, nn.Module]:
    """
    Train models with the best hyperparameters.
    Returns a dictionary of trained models.
    """
    models = {}
    
    # We'll use a fixed seed for reproducibility
    seed = 42
    
    # Create checkpoint directory if specified
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create a single temporary directory for all tensorboard logs
    temp_dir = tempfile.mkdtemp(prefix='toy_logo_tb_')
    
    try:
        for model_key, params in best_params.items():
            print(f"\nProcessing {model_key} with best parameters...")
            
            # Check for existing checkpoint
            checkpoint_path = None
            if checkpoint_dir:
                checkpoint_path = os.path.join(checkpoint_dir, f"{model_key}_gc{params['gc_pos']:.3f}_cons{params['conservation']:.2f}.pt")
                
            # Initialize model
            set_seeds(seed)
            model = TinyCNN().to(device)
            
            # Try to load checkpoint
            if checkpoint_path and os.path.exists(checkpoint_path):
                print(f"  Loading checkpoint from {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                models[model_key] = model
                continue  # Skip training
            
            print(f"  Training from scratch...")
            
            # Load dataset
            dataset = load_or_generate_dataset(params['gc_pos'], params['conservation'])
            
            # Create train/val split (using same proportions as toy_slurm.py)
            train_size = int(0.85 * len(dataset))
            val_size = len(dataset) - train_size
            train_ds, val_ds = torch.utils.data.random_split(
                dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(seed)
            )
            
            # Create dataloaders (using num_workers=0 to avoid multiprocessing issues)
            train_loader = DataLoader(train_ds, batch_size=DEFAULT_BATCH_SIZE, shuffle=True, num_workers=0)
            val_loader = DataLoader(val_ds, batch_size=DEFAULT_EVAL_BATCH_SIZE, num_workers=0)
            
            # Initialize training components
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-6)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=8, verbose=True)
            loss_fn = nn.BCEWithLogitsLoss()
            scaler = GradScaler()
            
            # Create writer in temp directory
            from torch.utils.tensorboard import SummaryWriter
            writer_dir = os.path.join(temp_dir, model_key)
            writer = SummaryWriter(log_dir=writer_dir)
            
            # Train model (matching exact settings from toy_slurm.py)
            try:
                if model_key == 'standard':
                    train_standard(model, train_loader, val_loader, loss_fn, optimizer, device, 
                                  scaler, writer, scheduler, epochs=epochs, early_stopping_patience=15)
                elif 'iterative_hotflip' in model_key:
                    use_scheduling = 'scheduled' in model_key
                    train_hotflip(model, train_loader, val_loader, loss_fn, optimizer, device,
                                 scaler, writer, scheduler, max_flip_fraction=params['epsilon'],
                                 epochs=epochs, use_scheduling=use_scheduling, early_stopping_patience=25, 
                                 gc_pos=params['gc_pos'])
                elif 'direct_hotflip' in model_key:
                    use_scheduling = 'scheduled' in model_key
                    train_direct_hotflip(model, train_loader, val_loader, loss_fn, optimizer, device,
                                        scaler, writer, scheduler, max_flip_fraction=params['epsilon'],
                                        epochs=epochs, use_scheduling=use_scheduling, early_stopping_patience=25,
                                        gc_pos=params['gc_pos'])
            finally:
                # Ensure writer is closed and flushed
                writer.flush()
                writer.close()
                # Give it a moment to finish writing
                import time
                time.sleep(0.1)
            
            models[model_key] = model
            
            # Save checkpoint if directory specified
            if checkpoint_path:
                print(f"  Saving checkpoint to {checkpoint_path}")
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'gc_pos': params['gc_pos'],
                    'conservation': params['conservation'],
                    'epsilon': params['epsilon'],
                    'model_key': model_key
                }, checkpoint_path)
            
    finally:
        # Clean up temp directory, ignoring errors
        try:
            # Use custom error handler for robust cleanup
            shutil.rmtree(temp_dir, onerror=handle_remove_readonly)
        except:
            pass  # Ignore any cleanup errors
        
    return models


def select_examples(dataset: SeqDS, n_examples: int = 5, seed: int = 0) -> List[int]:
    """
    Select n_examples positive examples from the dataset for visualization.
    """
    # Find positive examples
    positive_indices = [i for i in range(len(dataset)) if dataset.y[i] == 1]
    
    # Use random selection with fixed seed
    rng = np.random.default_rng(seed)
    selected = rng.choice(positive_indices, size=min(n_examples, len(positive_indices)), replace=False)
    
    return list(selected)


def compute_attributions(model: nn.Module, dataset: SeqDS, example_indices: List[int], 
                        device: torch.device, multiply_by_input: bool = True) -> List[np.ndarray]:
    """
    Compute integrated gradient attributions for selected examples.
    If multiply_by_input is False, returns raw attributions showing model preferences.
    """
    model.eval()
    
    def model_for_captum(x):
        with autocast():
            return model(x)[0].unsqueeze(-1)
    
    ig = IntegratedGradients(model_for_captum)
    attributions = []
    
    # Prepare for PGD baseline computation
    pgd_examples = []
    for idx in example_indices:
        x, y, mask = dataset[idx]
        pgd_examples.append((x.unsqueeze(0).to(device), torch.tensor([y], device=device, dtype=torch.float)))
    
    # Batch compute PGD baselines
    xb_batch = torch.cat([x for x, _ in pgd_examples])
    yb_batch = torch.cat([y for _, y in pgd_examples])
    
    pgd_baselines, pgd_stats = find_adversarial_baseline_pgd_batch_optimized(
        model, xb_batch, yb_batch, device
    )
    
    # Compute attributions
    for i, idx in enumerate(example_indices):
        x, y, mask = dataset[idx]
        x = x.to(device).unsqueeze(0)
        
        # Use PGD baseline if successful, else compositional
        if pgd_stats[i]['success']:
            baseline = pgd_baselines[i].unsqueeze(0)
        else:
            proportions = x.mean(dim=2, keepdim=True)
            baseline = proportions.expand_as(x)
        
        # Compute IG
        raw_attr = ig.attribute(x, baselines=baseline, target=0)
        
        # Apply gradient correction (across nucleotide dimension)
        corrected_attr = raw_attr - raw_attr.mean(dim=2, keepdim=True)
        
        if multiply_by_input:
            # Multiply by input to get nucleotide-level attributions
            # x has shape (1, 4, length), so we need to transpose
            # Shape: (1, 4, length) -> (4, length) -> (length, 4)
            attr_scores = (corrected_attr * x).squeeze(0).cpu().numpy().T
        else:
            # Return raw corrected attributions (shows model preferences)
            # Shape: (1, 4, length) -> (4, length) -> (length, 4)
            attr_scores = corrected_attr.squeeze(0).cpu().numpy().T
        
        attributions.append(attr_scores)
    
    return attributions


def create_logo_comparison_plot(models: Dict[str, nn.Module], dataset: SeqDS, example_idx: int,
                               device: torch.device, output_path: str, plot_idx: int, gc_pos: float,
                               use_abs_values: bool = True):
    """
    Create comparison plots showing attribution logos for all model types.
    Generates both full sequence and zoomed plots.
    """
    x, y, mask = dataset[example_idx]
    
    # Compute attributions for each model
    model_attributions = {}
    for model_key, model in models.items():
        attrs = compute_attributions(model, dataset, [example_idx], device, 
                                   multiply_by_input=not SHOW_ALL_NUCLEOTIDES)
        model_attributions[model_key] = attrs[0]
    
    # Regenerate the original master motif (unmutated)
    rng_state = np.random.get_state()
    np.random.seed(42)  # Same seed as in generate_dataset
    master_chunk = sample_background(CHUNK_LEN, gc_pos)
    np.random.set_state(rng_state)
    
    # Get motif location
    motif_positions = np.where(mask)[0]
    if len(motif_positions) > 0:
        motif_start = motif_positions[0]
        motif_end = motif_positions[-1] + 1
        zoom_start = max(0, motif_start - 5)
        zoom_end = min(SEQ_LEN, motif_end + 5)
    else:
        motif_start = motif_end = zoom_start = zoom_end = 0
    
    # Create both full and zoomed plots
    for plot_type in ['full', 'zoomed']:
        n_models = len(models)
        fig, axes = plt.subplots(n_models + 2, 1, figsize=(20, 3 * (n_models + 2)))
        
        if plot_type == 'zoomed':
            seq_range = range(zoom_start, zoom_end)
            plot_width = zoom_end - zoom_start
        else:
            seq_range = range(SEQ_LEN)
            plot_width = SEQ_LEN
    
        # Model name mapping for display
        display_names = {
            'standard': 'Standard Training',
            'iterative_hotflip_no_schedule': 'Iterative HotFlip (No Schedule)',
            'iterative_hotflip_scheduled': 'Iterative HotFlip (Scheduled)',
            'direct_hotflip_no_schedule': 'Direct HotFlip (No Schedule)',
            'direct_hotflip_scheduled': 'Direct HotFlip (Scheduled)'
        }
        
        # Plot attribution logos for each model
        model_order = ['standard', 'iterative_hotflip_no_schedule', 'iterative_hotflip_scheduled',
                       'direct_hotflip_no_schedule', 'direct_hotflip_scheduled']
        
        for i, model_key in enumerate(model_order):
            if model_key not in model_attributions:
                continue
                
            ax = axes[i]
            scores = model_attributions[model_key]
            
            # Create DataFrame for logomaker
            counts_df = pd.DataFrame(data=0.0, columns=list('ACGT'), index=list(range(plot_width)))
            
            # Fill in attribution scores for the relevant range
            for j, pos in enumerate(seq_range):
                for base_idx, base in enumerate('ACGT'):
                    # Use the attribution score for this position and nucleotide
                    value = scores[pos, base_idx]
                    if use_abs_values:
                        value = abs(value)
                    counts_df.iloc[j, base_idx] = value
            
            # Create logo
            logo = logomaker.Logo(counts_df, ax=ax)
            ax.set_xlim(0, plot_width)
            ax.set_ylabel(display_names.get(model_key, model_key), fontsize=12)
            
            # Highlight motif region
            if mask.any() and motif_start < zoom_end and motif_end > zoom_start:
                highlight_start = max(0, motif_start - zoom_start) if plot_type == 'zoomed' else motif_start
                highlight_end = min(plot_width, motif_end - zoom_start) if plot_type == 'zoomed' else motif_end
                ax.axvspan(highlight_start, highlight_end, alpha=0.2, color='yellow')
            
            # Remove x-axis except for last subplot
            if i < n_models + 1:
                ax.set_xticklabels([])
                ax.set_xlabel('')
        
        # Add original motif (unmutated)
        ax = axes[-2]
        orig_df = pd.DataFrame(data=0.0, columns=list('ACGT'), index=list(range(plot_width)))
        
        if mask.any():
            # One-hot encode the master chunk
            master_one_hot = one_hot(master_chunk)
            
            for j, pos in enumerate(seq_range):
                if motif_start <= pos < motif_end:
                    chunk_pos = pos - motif_start
                    for base_idx in range(4):
                        if master_one_hot[base_idx, chunk_pos] > 0:
                            orig_df.iloc[j, base_idx] = 2.0
        
        logo = logomaker.Logo(orig_df, ax=ax)
        ax.set_xlim(0, plot_width)
        ax.set_ylabel('Original Motif\n(Unmutated)', fontsize=12)
        ax.set_xticklabels([])
        ax.set_xlabel('')
        
        # Highlight motif region
        if mask.any() and motif_start < zoom_end and motif_end > zoom_start:
            highlight_start = max(0, motif_start - zoom_start) if plot_type == 'zoomed' else motif_start
            highlight_end = min(plot_width, motif_end - zoom_start) if plot_type == 'zoomed' else motif_end
            ax.axvspan(highlight_start, highlight_end, alpha=0.2, color='blue')
        
        # Add mutated ground truth at bottom
        ax = axes[-1]
        gt_df = pd.DataFrame(data=0.0, columns=list('ACGT'), index=list(range(plot_width)))
        
        if mask.any():
            for j, pos in enumerate(seq_range):
                if motif_start <= pos < motif_end:
                    for base_idx in range(4):
                        if x[base_idx, pos] > 0:
                            gt_df.iloc[j, base_idx] = 2.0
        
        logo = logomaker.Logo(gt_df, ax=ax)
        ax.set_xlim(0, plot_width)
        ax.set_ylabel('Mutated Motif\n(Ground Truth)', fontsize=12)
        
        if plot_type == 'zoomed':
            # Add position labels for zoomed plot
            ax.set_xlabel('Position')
            ax.set_xticks(range(0, plot_width, 10))
            ax.set_xticklabels([str(zoom_start + i) for i in range(0, plot_width, 10)])
        else:
            ax.set_xlabel('Position')
        
        # Highlight motif region
        if mask.any() and motif_start < zoom_end and motif_end > zoom_start:
            highlight_start = max(0, motif_start - zoom_start) if plot_type == 'zoomed' else motif_start
            highlight_end = min(plot_width, motif_end - zoom_start) if plot_type == 'zoomed' else motif_end
            ax.axvspan(highlight_start, highlight_end, alpha=0.2, color='green')
        
        title_suffix = ' (Zoomed ±5bp)' if plot_type == 'zoomed' else ''
        mode_suffix = ' - Motif Discovery Mode' if SHOW_ALL_NUCLEOTIDES else ' - Importance Mode'
        plt.suptitle(f'Attribution Comparison - Example {plot_idx + 1}{title_suffix}{mode_suffix}', fontsize=16)
        plt.tight_layout()
        
        # Save with different filename for each plot type
        if plot_type == 'zoomed':
            save_path = output_path.replace('.pdf', '_zoomed.pdf')
        else:
            save_path = output_path
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved {plot_type} logo comparison plot to {save_path}")


def main():
    # Additional warning suppression for the main process
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    print(f"Configuration:")
    print(f"  Results dir: {RESULTS_DIR}")
    print(f"  Output dir: {OUTPUT_DIR}")
    print(f"  Checkpoint dir: {CHECKPOINT_DIR}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Examples: {N_EXAMPLES}")
    print(f"  Show negative values: {SHOW_NEGATIVE}")
    print(f"  Show all nucleotides: {SHOW_ALL_NUCLEOTIDES}")
    print(f"  Fixed hyperparameters: GC={FIXED_GC}, Cons={FIXED_CONS}")
    print()
    
    # Create output and checkpoint directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if CHECKPOINT_DIR:
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Find best hyperparameters
    if USE_FIXED_HPARAMS:
        # Find best epsilon values conditional on fixed GC and conservation
        best_params = find_best_hyperparameters(RESULTS_DIR, target_gc=FIXED_GC, target_cons=FIXED_CONS)
    else:
        # Find globally best hyperparameters
        best_params = find_best_hyperparameters(RESULTS_DIR)
    
    # Train models with best parameters
    models = train_best_models(best_params, device, epochs=EPOCHS, checkpoint_dir=CHECKPOINT_DIR)
    
    # Use the standard model's dataset for selecting examples
    # (all models should see the same test examples for fair comparison)
    standard_params = best_params['standard']
    dataset = load_or_generate_dataset(standard_params['gc_pos'], standard_params['conservation'])
    
    # Select examples
    example_indices = select_examples(dataset, n_examples=N_EXAMPLES)
    print(f"\nSelected {len(example_indices)} examples for visualization")
    
    # Generate logo plots
    for plot_idx, example_idx in enumerate(example_indices):
        output_path = os.path.join(OUTPUT_DIR, f'logo_comparison_example_{plot_idx + 1}.pdf')
        create_logo_comparison_plot(models, dataset, example_idx, device, output_path, plot_idx, 
                                   gc_pos=standard_params['gc_pos'],
                                   use_abs_values=not SHOW_NEGATIVE)
    
    print(f"\nAll logo plots saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main() 