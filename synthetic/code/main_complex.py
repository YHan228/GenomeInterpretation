#!/usr/bin/env python3
"""
Main entry point for complex (merged_experiment.py) experiments.
Supports SLURM array jobs for hyperparameter search and different experiment modes.
"""

import argparse
import os
import sys
import torch
from pathlib import Path
import numpy as np

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from synthetic.code.data import load_or_generate_complex_dataset, stratified_split
from synthetic.code.experiments import (
    get_complex_experiment_info,
    run_experiment_for_hyperparams,
    COMPLEX_GC_GAP_HPARAMS,
    COMPLEX_CONS_HPARAMS,
    SEEDS,
    SCHEDULE_MODES
)
from synthetic.code.utils import set_seeds


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Run complex synthetic sequence experiments')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save results')
    parser.add_argument('--experiment_mode', type=str, required=True,
                       choices=['adv_vs_std', 'direct_hotflip', 'all'],
                       help='Type of experiment to run')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for training')
    parser.add_argument('--array_idx', type=int, default=None,
                       help='SLURM array task ID')
    parser.add_argument('--aggregate_only', action='store_true',
                       help='Only generate plots from existing results')
    parser.add_argument('--test_mapping', action='store_true',
                       help='Test the array index mapping')
    
    args = parser.parse_args()
    
    # Test mapping mode
    if args.test_mapping:
        print("Testing array index mapping for complex experiments...")
        total_combos = len(SCHEDULE_MODES) * len(COMPLEX_GC_GAP_HPARAMS) * len(COMPLEX_CONS_HPARAMS)
        print(f"Total combinations: {len(SCHEDULE_MODES)} schedules × {len(COMPLEX_GC_GAP_HPARAMS)} GC-gap × {len(COMPLEX_CONS_HPARAMS)} cons = {total_combos}")
        for i in range(total_combos):
            try:
                gc_gap, conservation, use_scheduling = get_complex_experiment_info(i)
                print(f"  Index {i:2d}: gc_gap={gc_gap:.3f}, conservation={conservation:.1f}, schedule={use_scheduling}")
            except ValueError:
                break
        sys.exit(0)
    
    # Aggregate-only mode
    if args.aggregate_only:
        print("Running visualization analysis...")
        from synthetic.code.visualization import run_all_analyses
        run_all_analyses(Path(args.output_dir), experiment_type='complex')
        sys.exit(0)
    
    # Set device
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {dev}")
    
    # Get experiment configuration
    if args.array_idx is not None:
        # Single array job
        gc_gap, conservation, use_scheduling = get_complex_experiment_info(args.array_idx)
        print(f"SLURM array index {args.array_idx}: gc_gap={gc_gap}, conservation={conservation}, schedule={use_scheduling}")
        run_single_hyperparameter_set(args, gc_gap, conservation, use_scheduling, dev)
    else:
        # Run all experiments
        print("Running all hyperparameter combinations...")
        for use_scheduling in SCHEDULE_MODES:
            for gc_gap in COMPLEX_GC_GAP_HPARAMS:
                for conservation in COMPLEX_CONS_HPARAMS:
                    print(f"\n{'='*80}")
                    print(f"Running: schedule={use_scheduling}, gc_gap={gc_gap}, conservation={conservation}")
                    print(f"{'='*80}\n")
                    run_single_hyperparameter_set(args, gc_gap, conservation, use_scheduling, dev)


def run_single_hyperparameter_set(args, gc_gap, conservation, use_scheduling, dev):
    """Run experiments for a single hyperparameter combination."""
    # Import here to avoid circular imports
    from synthetic.code.training import train_standard, train_hotflip, train_direct_hotflip
    from synthetic.code.evaluation import evaluate_model
    from synthetic.code.models import TinyCNN, LogisticRegression
    
    # Prepare output directory
    schedule_str = "scheduled" if use_scheduling else "no_schedule"
    output_subdir = Path(args.output_dir) / schedule_str / f"gc{gc_gap}_cons{conservation}"
    output_subdir.mkdir(parents=True, exist_ok=True)
    
    # Generate or load dataset
    print(f"Loading/generating dataset...")
    X, y, masks, sample_types = load_or_generate_complex_dataset(
        gc_gap=gc_gap,
        conservation=conservation
    )
    
    # Create splits (70/15/15)
    n_total = len(X)
    n_train = int(0.7 * n_total)
    n_val = int(0.15 * n_total)
    n_test = n_total - n_train - n_val
    
    # Define experiment configurations based on mode
    experiment_configs = []
    
    # Always include standard training as baseline
    experiment_configs.append({
        'name': 'standard',
        'train_fn': lambda: train_standard_wrapper(
            X, y, masks, sample_types, n_train, n_val, n_test,
            args.batch_size, args.epochs, dev, model_class=TinyCNN
        ),
        'eval_fn': lambda model: evaluate_wrapper(
            model, X, y, masks, sample_types, n_train, n_val, n_test,
            args.batch_size, dev
        )
    })
    
    # Add sanity check models
    if args.experiment_mode in ['adv_vs_std', 'direct_hotflip', 'all']:
        # Logistic regression baseline
        experiment_configs.append({
            'name': 'logistic_regression',
            'train_fn': lambda: train_standard_wrapper(
                X, y, masks, sample_types, n_train, n_val, n_test,
                args.batch_size, min(50, args.epochs), dev, model_class=LogisticRegression
            ),
            'eval_fn': lambda model: evaluate_wrapper(
                model, X, y, masks, sample_types, n_train, n_val, n_test,
                args.batch_size, dev
            )
        })
    
    if args.experiment_mode in ['adv_vs_std', 'all']:
        # Iterative HotFlip adversarial training
        for epsilon in [0.05, 0.1, 0.15]:
            experiment_configs.append({
                'name': f'hotflip_eps{epsilon}',
                'train_fn': lambda eps=epsilon: train_hotflip_wrapper(
                    X, y, masks, sample_types, n_train, n_val, n_test,
                    args.batch_size, args.epochs, dev,
                    max_flip_fraction=eps, use_scheduling=use_scheduling,
                    k_flips=int(eps * 5000), n_augment=3
                ),
                'eval_fn': lambda model: evaluate_wrapper(
                    model, X, y, masks, sample_types, n_train, n_val, n_test,
                    args.batch_size, dev
                )
            })
    
    if args.experiment_mode in ['direct_hotflip', 'all']:
        # Direct HotFlip adversarial training
        for k_flips in [5, 10, 15]:
            experiment_configs.append({
                'name': f'direct_hotflip_k{k_flips}',
                'train_fn': lambda k=k_flips: train_direct_hotflip_wrapper(
                    X, y, masks, sample_types, n_train, n_val, n_test,
                    args.batch_size, args.epochs, dev,
                    k_flips=k, use_scheduling=use_scheduling, n_augment=3
                ),
                'eval_fn': lambda model: evaluate_wrapper(
                    model, X, y, masks, sample_types, n_train, n_val, n_test,
                    args.batch_size, dev
                )
            })
    
    # Run experiments across seeds
    run_experiment_for_hyperparams(
        gc_gap=gc_gap,
        conservation=conservation,
        output_dir=output_subdir,
        experiment_configs=experiment_configs,
        experiment_type='complex',
        seeds=SEEDS
    )


def train_standard_wrapper(X, y, masks, sample_types, n_train, n_val, n_test,
                          batch_size, epochs, dev, model_class):
    """Wrapper for standard training."""
    from synthetic.code.training import train_standard
    
    (X_train, y_train, masks_train,
     X_val, y_val, masks_val,
     X_test, y_test, masks_test) = stratified_split(
        X, y, masks, n_train=n_train, n_val=n_val, n_test=n_test,
        sample_types=sample_types, random_state=42
    )
    
    return train_standard(
        model_class=model_class,
        X_train=X_train, y_train=y_train, masks_train=masks_train,
        X_val=X_val, y_val=y_val, masks_val=masks_val,
        batch_size=batch_size, epochs=epochs, dev=dev,
        warmup_epochs=5, early_stop_start=60
    )


def train_hotflip_wrapper(X, y, masks, sample_types, n_train, n_val, n_test,
                         batch_size, epochs, dev, max_flip_fraction,
                         use_scheduling, k_flips, n_augment):
    """Wrapper for iterative HotFlip training."""
    from synthetic.code.training import train_hotflip
    from synthetic.code.models import TinyCNN
    
    (X_train, y_train, masks_train,
     X_val, y_val, masks_val,
     X_test, y_test, masks_test) = stratified_split(
        X, y, masks, n_train=n_train, n_val=n_val, n_test=n_test,
        sample_types=sample_types, random_state=42
    )
    
    return train_hotflip(
        model_class=TinyCNN,
        X_train=X_train, y_train=y_train, masks_train=masks_train,
        X_val=X_val, y_val=y_val, masks_val=masks_val,
        batch_size=batch_size, epochs=epochs, dev=dev,
        k_flips=k_flips, n_augment=n_augment,
        use_scheduling=use_scheduling, max_flip_fraction=max_flip_fraction,
        warmup_epochs=5, early_stop_start=60
    )


def train_direct_hotflip_wrapper(X, y, masks, sample_types, n_train, n_val, n_test,
                                batch_size, epochs, dev, use_scheduling, k_flips, n_augment):
    """Wrapper for direct HotFlip training."""
    from synthetic.code.training import train_direct_hotflip
    from synthetic.code.models import TinyCNN
    
    (X_train, y_train, masks_train,
     X_val, y_val, masks_val,
     X_test, y_test, masks_test) = stratified_split(
        X, y, masks, n_train=n_train, n_val=n_val, n_test=n_test,
        sample_types=sample_types, random_state=42
    )
    
    return train_direct_hotflip(
        model_class=TinyCNN,
        X_train=X_train, y_train=y_train, masks_train=masks_train,
        X_val=X_val, y_val=y_val, masks_val=masks_val,
        batch_size=batch_size, epochs=epochs, dev=dev,
        k_flips=k_flips, n_augment=n_augment,
        use_scheduling=use_scheduling,
        warmup_epochs=5, early_stop_start=60
    )


def evaluate_wrapper(model, X, y, masks, sample_types, n_train, n_val, n_test,
                    batch_size, dev):
    """Wrapper for model evaluation."""
    from synthetic.code.evaluation import evaluate_model
    
    test_indices = list(range(n_train + n_val, len(X)))
    X_test, y_test, masks_test = X[test_indices], y[test_indices], masks[test_indices]
    
    test_dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.from_numpy(X_test).float(),
            torch.from_numpy(y_test).float(),
            torch.from_numpy(masks_test)
        ),
        batch_size=batch_size * 4, shuffle=False
    )
    
    return evaluate_model(model, test_dl, dev)


if __name__ == '__main__':
    main() 