"""
Synthetic sequence experiments package.

This package contains modularized code for running synthetic DNA sequence
experiments with various neural network architectures and training methods.
"""

__version__ = "1.0.0"
__author__ = "Your Name"

# Import main components for easier access
from synthetic.code.models import TinyCNN, TinyCNN_Vanilla, LogisticRegression, get_model
from synthetic.code.data import (
    generate_vanilla_dataset,
    generate_complex_dataset,
    load_or_generate_vanilla_dataset,
    load_or_generate_complex_dataset,
    stratified_split
)
from synthetic.code.training import (
    train_standard,
    train_hotflip,
    train_direct_hotflip,
    train_random_smoothing
)
from synthetic.code.evaluation import (
    evaluate_model,
    evaluate_model_vanilla,
    compute_effect_sizes_fast,
    compute_sequence_properties_gpu,
    compute_adversarial_changes
)
from synthetic.code.experiments import (
    get_vanilla_experiment_info,
    get_complex_experiment_info,
    run_experiment_for_hyperparams
)
from synthetic.code.utils import set_seeds, log_gpu_stats

__all__ = [
    # Models
    'TinyCNN',
    'TinyCNN_Vanilla',
    'LogisticRegression',
    'get_model',
    
    # Data
    'generate_vanilla_dataset',
    'generate_complex_dataset',
    'load_or_generate_vanilla_dataset',
    'load_or_generate_complex_dataset',
    'stratified_split',
    
    # Training
    'train_standard',
    'train_hotflip',
    'train_direct_hotflip',
    'train_random_smoothing',
    
    # Evaluation
    'evaluate_model',
    'evaluate_model_vanilla',
    'compute_effect_sizes_fast',
    'compute_sequence_properties_gpu',
    'compute_adversarial_changes',
    
    # Experiments
    'get_vanilla_experiment_info',
    'get_complex_experiment_info',
    'run_experiment_for_hyperparams',
    
    # Utils
    'set_seeds',
    'log_gpu_stats',
] 