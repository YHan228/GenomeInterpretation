# Unified Optuna HPO Guide

## Overview
The new unified HPO approach finds ONE best model that performs well across ALL dataset variations (65 combinations: 13 GC values × 5 conservation levels).

## Key Changes from Previous Approach
- **Single objective**: Maximize Saliency AUC only (no longer multi-objective)
- **Unified model**: One model for all datasets (not per-dataset models)
- **TPE default**: Bayesian optimization for better sample efficiency
- **Heavier trials**: Each trial evaluates on 5-10 datasets

## Parallelization Strategy

### Recommended: Moderate Parallelization (4-8 workers)
TPE (Bayesian optimization) learns from completed trials, so moderate parallelization gives the best balance:

```bash
# Optimal for TPE: 6 parallel workers
sbatch --array=1-6 slurm_scripts/run_optuna_singlearch.sh

# Set study name and trials
export STUDY_NAME="my_unified_study"
export N_TRIALS=100  # Each worker will run 100 trials
sbatch --array=1-6 slurm_scripts/run_optuna_singlearch.sh
# Total: 600 trials across 6 workers
```

### Single GPU (Sequential)
Best for TPE learning but slower:

```bash
# Single worker, more trials
export N_TRIALS=600
sbatch slurm_scripts/run_optuna_singlearch.sh
```

### High Parallelization (10+ workers)
Only recommended if you need results very quickly:

```bash
# 12 parallel workers - faster but less efficient TPE learning
sbatch --array=1-12 slurm_scripts/run_optuna_singlearch.sh
```

## Database Setup

### Option 1: MariaDB (Recommended for cluster)
```bash
# Start MariaDB first
sbatch slurm_scripts/run_mariadb.sh
# Wait for it to start, check the log for the database host

# Set storage URL (replace DBHOST with actual host from log)
export OPTUNA_STORAGE="mysql+pymysql://optuna_user:optuna_pass@DBHOST:3306/optuna"

# The password should be URL-encoded if it contains special characters
# Use Python to encode: python -c "import urllib.parse; print(urllib.parse.quote('your_pass'))"
```

### Option 2: SQLite (Simple but not for high parallelization)
```bash
export OPTUNA_STORAGE="sqlite:////scratch/$USER/optuna_unified.db"
# Only use with 1-2 workers due to SQLite limitations
```

### Option 3: PostgreSQL (If available)
```bash
export OPTUNA_STORAGE="postgresql+psycopg2://user:pass@host:5432/optuna"
```

## Configuration Options

Set these environment variables before submitting:

```bash
# Basic configuration
export STUDY_NAME="unified_study_v1"     # Study name
export N_TRIALS=100                      # Trials per worker
export HPO_MODE="standard"               # or "robust" for adversarial training
export SAMPLER="tpe"                     # TPE (Bayesian) is default
export MAX_EPOCHS=40                     # Max epochs per trial
export NUM_SEEDS_PER_TRIAL=1            # Seeds to average per trial

# Submit job
sbatch --array=1-6 slurm_scripts/run_optuna_singlearch.sh
```

## Monitoring Progress

### Check study progress
```bash
# View latest results
cat optuna_results/${STUDY_NAME}_unified_mode_standard/summary.json

# Monitor worker logs
tail -f slurm_output/optuna_unified_*.out
```

### Visualize with Optuna Dashboard (if installed)
```bash
optuna-dashboard "${OPTUNA_STORAGE}"
# Then open browser to http://localhost:8080
```

## Post-Processing

### Get best hyperparameters
```bash
python -c "
import json
with open('optuna_results/YOUR_STUDY_unified_mode_standard/summary.json') as f:
    data = json.load(f)
    print('Best trial:', data['best_trial'])
    print('Best value:', data['best_value'])
"
```

### Re-evaluate best model with more seeds
```bash
# Find best trial number from summary.json
python toy_single_arch.py \
  --reval-trial TRIAL_NUMBER \
  --study-name YOUR_STUDY \
  --reval-seeds 10
```

## Example Full Workflow

```bash
# 1. Start database (if using MariaDB)
sbatch slurm_scripts/run_mariadb.sh
sleep 60  # Wait for DB to start

# 2. Get database host from log
DBHOST=$(grep "Database Host:" slurm_output/mariadb_*.out | awk '{print $3}')

# 3. Set configuration
export OPTUNA_STORAGE="mysql+pymysql://optuna_user:optuna_pass@${DBHOST}:3306/optuna"
export STUDY_NAME="unified_exp_$(date +%Y%m%d)"
export N_TRIALS=100
export MAX_EPOCHS=40

# 4. Launch HPO with 6 workers
sbatch --array=1-6 slurm_scripts/run_optuna_singlearch.sh

# 5. Monitor progress
watch -n 60 "grep 'Trial.*Overall metrics' slurm_output/optuna_unified_*.out | tail -5"

# 6. Check results when done
cat optuna_results/${STUDY_NAME}_unified_mode_standard/summary.json
```

## Computational Considerations

- **Memory**: Each worker loads all 65 datasets, requiring ~50GB RAM
- **GPU**: Each trial trains on multiple datasets, so GPU utilization is high
- **Time**: Each trial takes longer (5-10x) than old per-dataset approach
- **Storage**: Results are smaller (one study vs. 65 studies)

## Troubleshooting

### Database connection errors
- Check OPTUNA_STORAGE is correctly set
- Verify database is running: `squeue -u $USER`
- Check password encoding for special characters

### Out of memory
- Reduce batch size in the code
- Request more memory: `--mem=80G`

### Slow convergence
- Increase N_TRIALS
- Reduce parallelization for better TPE learning
- Check if pruning is too aggressive
