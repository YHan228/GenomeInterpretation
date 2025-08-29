"""
Experiment configuration and execution logic.
Handles hyperparameter sweeps, SLURM array job mapping, and result saving.
"""

import torch
import numpy as np
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import json
import pickle

from synthetic.code.utils import set_seeds, log_gpu_stats
from synthetic.code.data import VANILLA_SEQ_LEN, COMPLEX_SEQ_LEN
from synthetic.code.models import TinyCNN, TinyCNN_Vanilla, LogisticRegression, get_model
from synthetic.code.training import train_standard, train_hotflip, train_direct_hotflip, train_random_smoothing
from synthetic.code.evaluation import evaluate_model, evaluate_model_vanilla, compute_effect_sizes_fast


# --------------------------------------------------------------------------- #
# Hyperparameter Configurations
# --------------------------------------------------------------------------- #

# Vanilla experiment hyperparameters
VANILLA_GC_POS_HPARAMS = [0.2, 0.35, 0.5, 0.65]  # GC content for positive examples
VANILLA_CONS_HPARAMS = [1.0, 0.9, 0.7, 0.5, 0.3]  # Conservation levels

# Complex experiment hyperparameters
COMPLEX_GC_GAP_HPARAMS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25]  # GC content gap
COMPLEX_CONS_HPARAMS = [1.0, 0.9, 0.7, 0.5]  # Conservation levels

# Training configuration
SCHEDULE_MODES = [True, False]  # Adversarial training with and without scheduling
SEEDS = [0, 1, 2]  # Random seeds for multiple runs
EPSILONS = [0.05, 0.1, 0.2]  # Epsilon values for randomized smoothing

# Default training parameters
DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 100
PREFETCH_FACTOR = 2


# --------------------------------------------------------------------------- #
# Array Job Mapping
# --------------------------------------------------------------------------- #

def get_vanilla_experiment_info(array_idx: int) -> Tuple[float, float, bool]:
    """
    Map SLURM array index to vanilla experiment configuration, including scheduling.
    Total combinations: 2 schedules * 4 GC * 5 cons = 40
    
    Parameters:
    - array_idx: SLURM array task ID
    
    Returns:
    - gc_pos: GC content for positive examples
    - conservation: Conservation level
    - use_scheduling: Whether to use a curriculum for adversarial training
    """
    n_gc = len(VANILLA_GC_POS_HPARAMS)
    n_cons = len(VANILLA_CONS_HPARAMS)
    n_schedules = len(SCHEDULE_MODES)
    
    n_per_schedule = n_gc * n_cons
    total_experiments = n_schedules * n_per_schedule
    
    if array_idx >= total_experiments:
        raise ValueError(f"Array index {array_idx} out of range. "
                        f"Total experiments: {total_experiments}")
    
    schedule_idx = array_idx // n_per_schedule
    remainder = array_idx % n_per_schedule
    gc_idx = remainder // n_cons
    cons_idx = remainder % n_cons
    
    use_scheduling = SCHEDULE_MODES[schedule_idx]
    gc_pos = VANILLA_GC_POS_HPARAMS[gc_idx]
    conservation = VANILLA_CONS_HPARAMS[cons_idx]
    
    return gc_pos, conservation, use_scheduling


def get_complex_experiment_info(array_idx: int) -> Tuple[float, float, bool]:
    """
    Map SLURM array index to complex experiment configuration, including scheduling.
    Total combinations: 2 schedules * 6 GC-gap * 4 cons = 48
    
    Parameters:
    - array_idx: SLURM array task ID
    
    Returns:
    - gc_gap: GC content gap between motifs and background
    - conservation: Conservation level
    - use_scheduling: Whether to use a curriculum for adversarial training
    """
    n_gc = len(COMPLEX_GC_GAP_HPARAMS)
    n_cons = len(COMPLEX_CONS_HPARAMS)
    n_schedules = len(SCHEDULE_MODES)
    
    n_per_schedule = n_gc * n_cons
    total_experiments = n_schedules * n_per_schedule
    
    if array_idx >= total_experiments:
        raise ValueError(f"Array index {array_idx} out of range. "
                        f"Total experiments: {total_experiments}")
    
    schedule_idx = array_idx // n_per_schedule
    remainder = array_idx % n_per_schedule
    gc_idx = remainder // n_cons
    cons_idx = remainder % n_cons
    
    use_scheduling = SCHEDULE_MODES[schedule_idx]
    gc_gap = COMPLEX_GC_GAP_HPARAMS[gc_idx]
    conservation = COMPLEX_CONS_HPARAMS[cons_idx]
    
    return gc_gap, conservation, use_scheduling


# --------------------------------------------------------------------------- #
# Experiment Execution
# --------------------------------------------------------------------------- #

def run_single_experiment(
    seed: int,
    train_fn: callable,
    eval_fn: callable,
    output_dir: Path,
    experiment_name: str,
    experiment_type: str = 'vanilla',
    **kwargs
) -> Dict[str, Any]:
    """
    Run a single experiment with a given seed.
    
    Parameters:
    - seed: Random seed
    - train_fn: Function to train the model
    - eval_fn: Function to evaluate the model
    - output_dir: Directory to save results
    - experiment_name: Name of the experiment
    - experiment_type: 'vanilla' or 'complex'
    - **kwargs: Additional parameters
    
    Returns:
    - Dictionary of results
    """
    print(f"\n{'='*60}")
    print(f"Running {experiment_name} with seed {seed}")
    print(f"{'='*60}")
    
    # Set seed
    set_seeds(seed)
    
    # Train model
    model, train_history = train_fn()
    
    # Evaluate model
    if experiment_type == 'vanilla':
        # Vanilla evaluation returns: wIoU, accuracy, AUC, SNR, pgd_stats
        wiou, accuracy, auc, snr, pgd_stats = eval_fn(model)
        results = {
            'seed': seed,
            'experiment': experiment_name,
            'wiou': wiou,
            'accuracy': accuracy,
            'saliency_auc': auc,
            'saliency_snr': snr,
            'pgd_stats': pgd_stats,
            'train_history': train_history
        }
    else:
        # Complex evaluation returns: accuracy, AUC, SNR, motif_AUC, motif_SNR, pgd_stats
        accuracy, auc, snr, motif_auc, motif_snr, pgd_stats = eval_fn(model)
        results = {
            'seed': seed,
            'experiment': experiment_name,
            'accuracy': accuracy,
            'saliency_auc': auc,
            'saliency_snr': snr,
            'motif_saliency_auc': motif_auc,
            'motif_saliency_snr': motif_snr,
            'pgd_stats': pgd_stats,
            'train_history': train_history
        }
    
    # Save model
    model_path = output_dir / f"{experiment_name}_seed{seed}_model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # Save results
    results_path = output_dir / f"{experiment_name}_seed{seed}_results.pkl"
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Results saved to {results_path}")
    
    return results


def run_experiment_for_hyperparams(
    gc_pos: Optional[float] = None,
    gc_gap: Optional[float] = None,
    conservation: float = 1.0,
    output_dir: Path = None,
    experiment_configs: List[Dict] = None,
    experiment_type: str = 'vanilla',
    seeds: List[int] = SEEDS
) -> Dict[str, List[Dict]]:
    """
    Run experiments for a specific hyperparameter combination across multiple seeds.
    
    Parameters:
    - gc_pos: GC content for positive examples (vanilla)
    - gc_gap: GC content gap (complex)
    - conservation: Conservation level
    - output_dir: Directory to save results
    - experiment_configs: List of experiment configurations
    - experiment_type: 'vanilla' or 'complex'
    - seeds: List of random seeds
    
    Returns:
    - Dictionary mapping experiment names to lists of results
    """
    # Validate inputs
    if experiment_type == 'vanilla' and gc_pos is None:
        raise ValueError("gc_pos required for vanilla experiments")
    if experiment_type == 'complex' and gc_gap is None:
        raise ValueError("gc_gap required for complex experiments")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Log experiment info
    log_path = output_dir / 'experiment_log.txt'
    with open(log_path, 'w') as f:
        f.write(f"Experiment Type: {experiment_type}\n")
        f.write(f"Timestamp: {datetime.now()}\n")
        if experiment_type == 'vanilla':
            f.write(f"GC Positive: {gc_pos}\n")
        else:
            f.write(f"GC Gap: {gc_gap}\n")
        f.write(f"Conservation: {conservation}\n")
        f.write(f"Seeds: {seeds}\n")
        f.write(f"Experiments: {[cfg['name'] for cfg in experiment_configs]}\n")
    
    # Run experiments
    all_results = {}
    
    for config in experiment_configs:
        exp_name = config['name']
        train_fn = config['train_fn']
        eval_fn = config['eval_fn']
        
        print(f"\n{'#'*80}")
        print(f"## Running {exp_name} experiments")
        print(f"{'#'*80}")
        
        exp_results = []
        for seed in seeds:
            try:
                results = run_single_experiment(
                    seed=seed,
                    train_fn=train_fn,
                    eval_fn=eval_fn,
                    output_dir=output_dir,
                    experiment_name=exp_name,
                    experiment_type=experiment_type
                )
                exp_results.append(results)
            except Exception as e:
                print(f"ERROR in {exp_name} seed {seed}: {e}")
                import traceback
                traceback.print_exc()
        
        all_results[exp_name] = exp_results
    
    # Save aggregated results
    summary = {}
    for exp_name, results_list in all_results.items():
        if not results_list:
            continue
            
        # Calculate mean and std across seeds
        if experiment_type == 'vanilla':
            metrics = ['wiou', 'accuracy', 'saliency_auc', 'saliency_snr']
        else:
            metrics = ['accuracy', 'saliency_auc', 'saliency_snr', 
                      'motif_saliency_auc', 'motif_saliency_snr']
        
        summary[exp_name] = {}
        for metric in metrics:
            values = [r[metric] for r in results_list if metric in r]
            if values:
                summary[exp_name][f'{metric}_mean'] = np.mean(values)
                summary[exp_name][f'{metric}_std'] = np.std(values)
        
        # PGD stats
        pgd_success_rates = [r['pgd_stats']['pgd_success_rate'] 
                            for r in results_list if 'pgd_stats' in r]
        if pgd_success_rates:
            summary[exp_name]['pgd_success_rate_mean'] = np.mean(pgd_success_rates)
            summary[exp_name]['pgd_success_rate_std'] = np.std(pgd_success_rates)
    
    # Save summary
    summary_path = output_dir / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    for exp_name, exp_summary in summary.items():
        print(f"\n{exp_name}:")
        for metric, value in exp_summary.items():
            print(f"  {metric}: {value:.4f}")
    
    return all_results


# --------------------------------------------------------------------------- #
# Sanity Check Experiments
# --------------------------------------------------------------------------- #

def run_sanity_checks(output_dir: Path, experiment_type: str = 'vanilla'):
    """
    Run quick sanity check experiments with small datasets.
    
    Parameters:
    - output_dir: Directory to save results
    - experiment_type: 'vanilla' or 'complex'
    """
    print("Running sanity check experiments...")
    
    # Import data loading functions
    from synthetic.code.data import generate_vanilla_dataset, generate_complex_dataset
    
    # Set device
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Generate small dataset
    n_samples = 1000
    if experiment_type == 'vanilla':
        X, y, masks = generate_vanilla_dataset(
            gc_pos=0.5, conservation=1.0, n_total=n_samples
        )
        model_class = TinyCNN_Vanilla
    else:
        X, y, masks, sample_types = generate_complex_dataset(
            gc_gap=0.1, conservation=1.0, n_total=n_samples
        )
        model_class = TinyCNN
    
    # Split data
    n_train = int(0.7 * n_samples)
    n_val = int(0.15 * n_samples)
    
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_test, y_test = X[n_train+n_val:], y[n_train+n_val:]
    
    # Test different models
    print("\n1. Testing Logistic Regression...")
    logistic_model = LogisticRegression(k=6).to(dev)
    
    # Create simple training loop for logistic regression
    optimizer = torch.optim.Adam(logistic_model.parameters(), lr=0.001)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    # Convert to tensors
    X_train_t = torch.from_numpy(X_train).float()
    y_train_t = torch.from_numpy(y_train).float()
    
    # Train for a few epochs
    logistic_model.train()
    for epoch in range(10):
        optimizer.zero_grad()
        logits, _ = logistic_model(X_train_t.to(dev))
        loss = criterion(logits, y_train_t.to(dev))
        loss.backward()
        optimizer.step()
        if epoch % 5 == 0:
            print(f"  Epoch {epoch}, Loss: {loss.item():.4f}")
    
    # Test accuracy
    logistic_model.eval()
    with torch.no_grad():
        X_test_t = torch.from_numpy(X_test).float().to(dev)
        y_test_t = torch.from_numpy(y_test).float().to(dev)
        logits, _ = logistic_model(X_test_t)
        preds = (torch.sigmoid(logits) > 0.5).float()
        acc = (preds == y_test_t).float().mean().item()
        print(f"  Logistic Regression Test Accuracy: {acc:.3f}")
    
    print(f"\n2. Testing {model_class.__name__}...")
    cnn_model = model_class().to(dev)
    
    # Quick training test
    optimizer = torch.optim.Adam(cnn_model.parameters(), lr=0.001)
    cnn_model.train()
    
    # Single batch forward/backward
    batch_size = min(32, n_train)
    batch_X = X_train_t[:batch_size].to(dev)
    batch_y = y_train_t[:batch_size].to(dev)
    
    optimizer.zero_grad()
    logits, _ = cnn_model(batch_X)
    loss = criterion(logits, batch_y)
    loss.backward()
    optimizer.step()
    print(f"  CNN forward/backward pass successful. Loss: {loss.item():.4f}")
    
    # Log GPU stats
    log_gpu_stats()
    
    print("\nSanity checks completed successfully!")
    

# --------------------------------------------------------------------------- #
# Result Analysis Utilities
# --------------------------------------------------------------------------- #

def load_experiment_results(output_dir: Path) -> Dict[str, Any]:
    """
    Load all experiment results from a directory.
    
    Parameters:
    - output_dir: Directory containing experiment results
    
    Returns:
    - Dictionary of results organized by experiment name and seed
    """
    results = {}
    
    # Find all result files
    result_files = list(output_dir.glob("*_results.pkl"))
    
    for file_path in result_files:
        # Parse filename
        parts = file_path.stem.split('_')
        exp_name = '_'.join(parts[:-2])  # Everything except seed and 'results'
        seed = int(parts[-2].replace('seed', ''))
        
        # Load results
        with open(file_path, 'rb') as f:
            result = pickle.load(f)
        
        # Organize by experiment name
        if exp_name not in results:
            results[exp_name] = {}
        results[exp_name][seed] = result
    
    return results


def analyze_convergence(train_history: Dict[str, List]) -> Dict[str, float]:
    """
    Analyze training convergence from history.
    
    Parameters:
    - train_history: Training history dictionary
    
    Returns:
    - Dictionary of convergence metrics
    """
    val_losses = train_history.get('val_loss', [])
    
    if not val_losses:
        return {}
    
    # Find best epoch
    best_epoch = np.argmin(val_losses)
    best_val_loss = val_losses[best_epoch]
    
    # Check if training converged (no improvement in last 10 epochs)
    if len(val_losses) > 10:
        recent_best = min(val_losses[-10:])
        converged = recent_best >= best_val_loss * 0.99
    else:
        converged = False
    
    # Calculate convergence rate (epochs to reach 90% of final performance)
    if len(val_losses) > 1:
        final_loss = val_losses[-1]
        target_loss = val_losses[0] - 0.9 * (val_losses[0] - final_loss)
        conv_epoch = next((i for i, loss in enumerate(val_losses) 
                          if loss <= target_loss), len(val_losses))
        conv_rate = conv_epoch / len(val_losses)
    else:
        conv_rate = 1.0
    
    return {
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'final_val_loss': val_losses[-1],
        'converged': converged,
        'convergence_rate': conv_rate,
        'total_epochs': len(val_losses)
    } 