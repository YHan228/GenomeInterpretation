# Synthetic Sequence Experiments - Modular Code Structure

This directory contains a modular implementation of synthetic DNA sequence classification experiments with adversarial training, restructured from the original `toy_slurm.py` and `merged_experiment.py` scripts.

## Directory Structure

```
synthetic/code/
├── __init__.py           # Package initialization
├── utils.py              # Common utilities (GPU, sequence manipulation, visualization)
├── data.py               # Data generation for both vanilla and complex experiments
├── models.py             # Neural network architectures
├── training.py           # Training methods (standard, HotFlip, randomized smoothing)
├── evaluation.py         # Evaluation metrics (PGD attacks, saliency analysis)
├── experiments.py        # Experiment runners and configuration
├── main_toy.py           # Main entry point for vanilla experiments
├── main_merged.py        # Main entry point for complex experiments
└── README.md             # This file
```

## Module Descriptions

### `utils.py`
Common utilities used across all modules:
- Random seed management
- GPU optimization (prefetching, memory logging)
- Sequence manipulation (one-hot encoding, sampling, mutation)
- Visualization utilities (motif logos)
- Randomized smoothing utilities

### `data.py`
Data generation for two types of experiments:

**Vanilla Dataset** (from toy_slurm.py):
- Simple single-motif design
- 1000bp sequences
- Single ancestral motif with mutations
- Configurable GC content and conservation

**Complex Dataset** (from merged_experiment.py):
- Multi-block design with 3 repertoires
- 5000bp sequences
- 3 repertoires with 10 genes each
- Positive examples: 3 motifs (one from each repertoire) + promoter
- Negative decoys: 1-2 motifs from incomplete repertoires
- Includes dataset caching for efficiency

### `models.py`
Neural network architectures:
- `TinyCNN`: Main architecture with localist pooling
- `SimpleCNN`: Baseline without localist pooling
- `LogisticRegression`: k-mer count baseline
- `TinyCNNv0`: Legacy architecture for backward compatibility

### `training.py`
Training methods:
- Standard training with warmup and early stopping
- HotFlip adversarial training (iterative)
- Direct HotFlip (one-shot)
- Randomized smoothing with Dirichlet noise

### `evaluation.py`
Evaluation methods:
- PGD adversarial attacks (single and batched)
- Integrated Gradients saliency analysis
- Separate motif vs promoter evaluation
- Effect size analysis
- Sequence property analysis

### `experiments.py`
Experiment management:
- Array job configuration mapping
- Multi-seed experiment runners
- Hyperparameter sweep management
- Results saving (NPZ format)

## Usage

### Running Vanilla Experiments

From the project root directory:

```bash
# Test array job mapping
python run_toy_experiments.py --test_mapping

# Run a specific array job
python run_toy_experiments.py \
    --output_dir results/vanilla \
    --experiment_mode adv_vs_std \
    --array_idx 0

# Run full sweep
python run_toy_experiments.py \
    --output_dir results/vanilla \
    --experiment_mode adv_vs_std
```

### Running Complex Experiments

```bash
# Test array job mapping
python run_merged_experiments.py --test_mapping

# Run a specific array job
python run_merged_experiments.py \
    --output_dir results/complex \
    --experiment_mode direct_hotflip \
    --array_idx 5

# Run full sweep
python run_merged_experiments.py \
    --output_dir results/complex \
    --experiment_mode adv_vs_std
```

## Key Improvements from Original Scripts

1. **Modular Structure**: Clean separation of concerns with focused modules
2. **Unified Data Pipeline**: Both experiment types use consistent data interfaces
3. **Enhanced GPU Performance**: GPU prefetching and optimized batched operations
4. **Flexible Configuration**: Easy to add new models, training methods, or metrics
5. **Better Error Handling**: Improved error messages and validation
6. **Consistent API**: Standardized function signatures across modules
7. **Type Hints**: Added type annotations for better code clarity

## Extending the Code

To add new functionality:

1. **New Model Architecture**: Add to `models.py` and register in `get_model()`
2. **New Training Method**: Add to `training.py` following existing patterns
3. **New Evaluation Metric**: Add to `evaluation.py`
4. **New Data Generation**: Extend `data.py` with new dataset classes

## Dependencies

See `environment.yml` in the project root for required packages.

Main dependencies:
- PyTorch >= 1.9
- NumPy
- Captum (for Integrated Gradients)
- TensorBoard
- logomaker (optional, for motif visualization)

## Notes

- Dataset caching automatically creates cache directories
- TensorBoard logs are saved to `{output_dir}/tensorboard/`
- Results are saved as NPZ files in `{output_dir}/npz_results/`
- Use `--deterministic` flag for reproducible results (slower)
- Use `--no_compile` flag if torch.compile causes issues 