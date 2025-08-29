#!/bin/bash
#SBATCH --job-name=optuna_unified
#SBATCH --output=slurm_output/optuna_unified_%A_%a.out
#SBATCH --error=slurm_output/optuna_unified_%A_%a.err
#SBATCH --time=6-00:00:00
#SBATCH --partition=gpu
#SBATCH --qos=verylong
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=50G  # Increased memory since we load all datasets

# Unified Optuna HPO script for finding ONE best model across ALL datasets.
# Each worker explores different hyperparameters but evaluates on multiple datasets.
# 
# RECOMMENDED USAGE:
#   For TPE (Bayesian optimization): Use 4-8 parallel workers
#   sbatch --array=1-6 slurm_scripts/run_optuna_singlearch.sh
#
# The unified approach means:
#   - No dataset-specific studies (one unified study)
#   - Each trial evaluates on multiple datasets
#   - Objective: maximize average Saliency AUC across datasets
#   - TPE learns from previous trials to suggest better hyperparameters

source /home/yhan/miniconda3/etc/profile.d/conda.sh
conda activate genome

mkdir -p slurm_output
mkdir -p dataset_cache  # Essential for caching generated datasets

# Set unique temporary directory per worker to avoid resource/busy conflicts
if [ -d "/scratch/$USER" ]; then
  export TMPDIR="/scratch/$USER/optuna_tmp_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
else
  export TMPDIR="/tmp/optuna_tmp_${USER}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
fi
mkdir -p "$TMPDIR"
export MPLCONFIGDIR="$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT

# Resolve OPTUNA_STORAGE automatically when possible
if [ -z "$OPTUNA_STORAGE" ]; then
  # 1) Try latest summary.json
  latest_summary=$(ls -1t optuna_results/*/summary.json 2>/dev/null | head -1)
  if [ -n "$latest_summary" ]; then
    OPTUNA_STORAGE=$(python3 - <<'PY'
import json,sys
p=sys.argv[1]
try:
    with open(p) as f:
        d=json.load(f)
        v=d.get('storage','')
        if v:
            print(v)
except Exception:
    pass
PY
"$latest_summary")
  fi
fi

if [ -z "$OPTUNA_STORAGE" ] && [ -f "$HOME/.optuna_storage_url" ]; then
  read -r OPTUNA_STORAGE < "$HOME/.optuna_storage_url"
fi

if [ -z "$OPTUNA_STORAGE" ]; then
  echo "ERROR: OPTUNA_STORAGE not set and could not be inferred." >&2
  echo "Set it or place the URL in $HOME/.optuna_storage_url" >&2
  echo "Example: mysql+pymysql://optuna_user:<URL-ENCODED-PASS>@<DBHOST>:3306/optuna" >&2
  exit 1
fi

# Configuration variables with sensible defaults for unified approach
STUDY_NAME=${STUDY_NAME:-unified_$(date +%Y%m%d_%H%M%S)}
N_TRIALS=${N_TRIALS:-200}            # Fewer trials per worker since each is more expensive
HPO_MODE=${HPO_MODE:-standard}       # standard | robust
SAMPLER=${SAMPLER:-tpe}              # tpe (default for single-objective) | nsga2
PRUNER=${PRUNER:-hyperband}          # hyperband | median | none
MAX_EPOCHS=${MAX_EPOCHS:-40}
NUM_SEEDS_PER_TRIAL=${NUM_SEEDS_PER_TRIAL:-1}

OUTPUT_DIR="slurm_results/hpo_unified_${SLURM_ARRAY_JOB_ID}"
mkdir -p "${OUTPUT_DIR}"

echo "=========================================="
echo "Unified Optuna HPO Worker ${SLURM_ARRAY_TASK_ID}"
echo "=========================================="
echo "Study: ${STUDY_NAME}"
echo "Mode: ${HPO_MODE}"
echo "Trials per worker: ${N_TRIALS}"
echo "Sampler: ${SAMPLER} (TPE=Bayesian optimization)"
echo "Pruner: ${PRUNER}"
echo "Max epochs: ${MAX_EPOCHS}"
echo "Seeds per trial: ${NUM_SEEDS_PER_TRIAL}"
echo "Storage: ${OPTUNA_STORAGE}"
echo "Output: ${OUTPUT_DIR}"
echo ""
echo "This worker will explore hyperparameters and evaluate"
echo "each configuration on multiple datasets (5-10) to find"
echo "the single best model across all dataset variations."
echo "=========================================="

# Persist storage URL for future runs
echo -n "$OPTUNA_STORAGE" > "$HOME/.optuna_storage_url"

# Run the unified HPO
# Note: No --array_idx needed since we're not splitting by dataset
python -u toy_single_arch.py \
  --tune \
  --study-name "${STUDY_NAME}" \
  --n-trials ${N_TRIALS} \
  --sampler ${SAMPLER} \
  --pruner ${PRUNER} \
  --mode ${HPO_MODE} \
  --max-epochs ${MAX_EPOCHS} \
  --num-seeds-per-trial ${NUM_SEEDS_PER_TRIAL} \
  --eval-mode all \
  --pessimism-alpha 0 \
  --output_dir "${OUTPUT_DIR}"

echo ""
echo "Optuna unified worker ${SLURM_ARRAY_TASK_ID} finished."
echo "Results saved to optuna_results/${STUDY_NAME}_unified_mode_${HPO_MODE}/"
