# Modular Code Structure Summary

I have successfully restructured your two large experiment scripts (`toy_slurm.py` and `merged_experiment.py`) into a clean, modular structure under `synthetic/code/`. Here's what was accomplished:

## Created Files

### Core Modules
1. **`utils.py`** (271 lines)
   - Common utilities for both experiments
   - GPU optimization (prefetching, memory logging)
   - Sequence manipulation functions
   - Visualization utilities
   - Random seed management

2. **`data.py`** (529 lines)
   - Unified data generation module
   - Vanilla dataset (single motif, 1000bp sequences)
   - Complex dataset (multi-block, 3 repertoires, 5000bp sequences)
   - Dataset caching functionality
   - Stratified splitting utilities

3. **`models.py`** (217 lines)
   - All neural network architectures
   - TinyCNN (main model with localist pooling)
   - SimpleCNN (baseline without localist pooling)
   - LogisticRegression (k-mer count baseline)
   - Model factory function

4. **`training.py`** (454 lines)
   - Standard training with warmup
   - HotFlip adversarial training (iterative)
   - Direct HotFlip (one-shot)
   - Randomized smoothing
   - Early stopping and learning rate scheduling

5. **`evaluation.py`** (530 lines)
   - PGD adversarial attacks
   - Integrated Gradients saliency analysis
   - Separate motif vs promoter evaluation
   - Effect size analysis
   - Sequence property analysis

6. **`experiments.py`** (468 lines)
   - Experiment configuration management
   - Array job mapping for SLURM
   - Multi-seed experiment runners
   - Results saving (NPZ format)

### Entry Points
7. **`main_toy.py`** (165 lines)
   - Main entry for vanilla experiments
   - Supports HotFlip and randomized smoothing

8. **`main_merged.py`** (153 lines)
   - Main entry for complex experiments
   - Supports iterative and direct HotFlip

### Supporting Files
9. **`__init__.py`** - Package initialization with key exports
10. **`README.md`** - Comprehensive documentation
11. **`run_toy_experiments.py`** - Standalone runner for vanilla experiments
12. **`run_merged_experiments.py`** - Standalone runner for complex experiments
13. **`test_modular_structure.py`** - Test script to verify imports

## Key Improvements

1. **Separation of Concerns**: Each module has a single, clear responsibility
2. **Code Reuse**: Common functionality is shared between both experiment types
3. **Maintainability**: Much easier to modify or extend individual components
4. **Testing**: Each module can be tested independently
5. **Documentation**: Clear docstrings and type hints throughout
6. **Performance**: Preserved all GPU optimizations from original scripts

## Usage Examples

### Running Vanilla Experiments
```bash
# Test array job mapping
python run_toy_experiments.py --test_mapping

# Run specific configuration
python run_toy_experiments.py \
    --output_dir results/vanilla \
    --experiment_mode adv_vs_std \
    --array_idx 0
```

### Running Complex Experiments
```bash
# Run with direct HotFlip
python run_merged_experiments.py \
    --output_dir results/complex \
    --experiment_mode direct_hotflip \
    --epochs 100
```

## Migration Notes

- All functionality from the original scripts is preserved
- The modular structure makes it easy to:
  - Add new models to `models.py`
  - Implement new training methods in `training.py`
  - Add evaluation metrics to `evaluation.py`
  - Create new data generation schemes in `data.py`
- Dataset caching paths remain compatible with existing caches
- Output formats (NPZ, TensorBoard) are unchanged

## Dependencies

The code requires the same dependencies as the original scripts:
- PyTorch
- NumPy
- Captum (for Integrated Gradients)
- TensorBoard
- matplotlib, pandas, seaborn
- logomaker (optional)

Note: To run the code, ensure you have activated the appropriate conda environment with all dependencies installed. 