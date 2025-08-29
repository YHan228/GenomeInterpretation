# HPO Approaches Comparison

## Old Approach (Per-Dataset)
- **Studies**: 65 separate studies (one per dataset combination)
- **Objective**: Multi-objective (Val Accuracy + Saliency AUC) with NSGA-II
- **Model selection**: Best model for EACH dataset
- **Parallelization**: Natural - each worker handles different dataset
- **Workers needed**: 27-65 for good coverage
- **Trial cost**: Low (single dataset evaluation)
- **Storage**: Multiple study results

## New Approach (Unified)
- **Studies**: 1 unified study
- **Objective**: Single-objective (Saliency AUC only) with TPE
- **Model selection**: ONE best model for ALL datasets
- **Parallelization**: Exploration-based - workers try different hyperparameters
- **Workers needed**: 4-8 optimal for TPE
- **Trial cost**: High (5-10 dataset evaluations)
- **Storage**: Single study result

## Performance Comparison

| Aspect | Old (Per-Dataset) | New (Unified) |
|--------|------------------|---------------|
| **Total GPU hours** | ~1000 hours (20 workers × 50h) | ~300 hours (6 workers × 50h) |
| **Model quality** | Specialized per dataset | Robust across datasets |
| **Hyperparameter efficiency** | Random/Grid search | Bayesian optimization |
| **Best for** | Dataset-specific optimization | General model discovery |
| **Trials needed** | 400 per dataset | 600 total |

## When to Use Each

### Use Old Approach When:
- You need dataset-specific models
- You have massive parallelization available (20+ GPUs)
- You want to understand dataset-specific behaviors
- Multi-objective trade-offs are important

### Use New Approach When:
- You need ONE robust model for deployment
- You have limited GPUs (4-8)
- You want better hyperparameter efficiency
- Single metric (Saliency AUC) is sufficient

## Migration Command Examples

### Old approach (example)
```bash
# 27 workers, each handles one dataset combo
export STUDY_NAME="per_dataset_study"
export SAMPLER="nsga2"
sbatch --array=1-27 slurm_scripts/run_optuna.sh
```

### New approach (recommended)
```bash
# 6 workers explore hyperparameters cooperatively
export STUDY_NAME="unified_study"
export SAMPLER="tpe"
sbatch --array=1-6 slurm_scripts/run_optuna_singlearch.sh
```

## Results Interpretation

### Old Results Location
```
optuna_results/
├── study_gc0.500_cons0.60_mode_standard/
├── study_gc0.500_cons0.70_mode_standard/
├── ... (65 directories)
└── study_gc0.800_cons0.80_mode_standard/
```

### New Results Location
```
optuna_results/
└── study_unified_mode_standard/
    ├── trials.csv
    └── summary.json
```

## Key Advantages of New Unified Approach

1. **Better generalization**: Model tested across all data variations
2. **Efficient search**: TPE learns from all trials globally
3. **Simpler deployment**: One model to maintain
4. **Lower total compute**: Fewer redundant explorations
5. **Cleaner results**: Single study to analyze

## Transition Tips

1. Start with a small test:
   ```bash
   export N_TRIALS=10
   sbatch --array=1-2 slurm_scripts/run_optuna_singlearch.sh
   ```

2. Verify it works, then scale up:
   ```bash
   export N_TRIALS=100
   sbatch --array=1-6 slurm_scripts/run_optuna_singlearch.sh
   ```

3. Compare results with old approach on a few key metrics

4. Gradually transition to unified model for production
