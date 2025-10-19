#!/bin/bash
#SBATCH --job-name=optuna_sporulation_std
#SBATCH --output=sporulation_output/optuna_sporo_std_%A_%a.out
#SBATCH --error=sporulation_output/optuna_sporo_std_%A_%a.err
#SBATCH --time=6-00:00:00
#SBATCH --partition=gpu
#SBATCH --qos=verylong
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=80G

# Standard (architecture/optimizer) Optuna tuning for sporulation.
# Appends trials to an existing study if the same study name and storage are used.

source /home/yhan/miniconda3/etc/profile.d/conda.sh
conda activate genome

mkdir -p slurm_output

# Per-worker temp dirs to avoid matplotlib/cache collisions
if [ -d "/scratch/$USER" ]; then
  export TMPDIR="/scratch/$USER/optuna_sporo_std_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
else
  export TMPDIR="/tmp/optuna_sporo_std_${USER}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
fi
mkdir -p "$TMPDIR"
export MPLCONFIGDIR="$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT

# Resolve OPTUNA_STORAGE if possible
if [ -z "$OPTUNA_STORAGE" ] && [ -f "$HOME/.optuna_storage_url" ]; then
  read -r OPTUNA_STORAGE < "$HOME/.optuna_storage_url"
fi

# Default to local sqlite if not provided
if [ -z "$OPTUNA_STORAGE" ]; then
  OPTUNA_STORAGE="sqlite:///$(pwd)/optuna.db"
fi

# Persist for convenience (so future runs reuse the same DB)
echo -n "$OPTUNA_STORAGE" > "$HOME/.optuna_storage_url"

# Controls (can be overridden via env)
STUDY_NAME=${STUDY_NAME:-sporo_full_std_v2_cont_exp}
N_TRIALS=${N_TRIALS:-25}
PRUNER=${PRUNER:-hyperband}       # hyperband | median | none
MAX_EPOCHS=${MAX_EPOCHS:-25}
SEQ_LEN=${SEQ_LEN:-1000000}
EPOCH_BUDGET=${EPOCH_BUDGET:-4096}
VAL_EPOCH_BUDGET=${VAL_EPOCH_BUDGET:-2048}
VAL_STEPS=${VAL_STEPS:-64}
NUM_WORKERS=${NUM_WORKERS:-2}

# Optional: limit training steps per epoch if provided
TRAIN_STEPS_ARGS=()
if [ -n "$TRAIN_STEPS_PER_EPOCH" ]; then
  TRAIN_STEPS_ARGS+=(--train-steps-per-epoch "$TRAIN_STEPS_PER_EPOCH")
fi

# Unique seed per array worker to diversify TPE suggestions and training RNG
SEED_BASE=${SEED_BASE:-42}
WORKER_OFFSET=${SLURM_ARRAY_TASK_ID:-0}
SEED=$(( SEED_BASE + WORKER_OFFSET ))

echo "=========================================="
echo "Sporulation Optuna Worker (standard) ${SLURM_ARRAY_TASK_ID}"
echo "=========================================="
echo "Study: ${STUDY_NAME} (will append if exists)"
echo "Trials per worker: ${N_TRIALS}"
echo "Pruner: ${PRUNER}"
echo "Max epochs: ${MAX_EPOCHS}"
echo "Seq len: ${SEQ_LEN}"
echo "Train epoch budget: ${EPOCH_BUDGET}"
echo "Val epoch budget: ${VAL_EPOCH_BUDGET}"
echo "Val steps: ${VAL_STEPS}"
echo "Num workers: ${NUM_WORKERS}"
echo "Storage: ${OPTUNA_STORAGE}"
echo "Seed: ${SEED}"
echo "=========================================="

python -u sporulation/optuna_sporulation.py \
  --tune \
  --study-name "${STUDY_NAME}" \
  --n-trials ${N_TRIALS} \
  --pruner ${PRUNER} \
  --max-epochs ${MAX_EPOCHS} \
  --seq-len ${SEQ_LEN} \
  --epoch-budget ${EPOCH_BUDGET} \
  --val-epoch-budget ${VAL_EPOCH_BUDGET} \
  --val-steps ${VAL_STEPS} \
  --num-workers ${NUM_WORKERS} \
  --storage "${OPTUNA_STORAGE}" \
  --seed ${SEED} \
  --verbose \
  "${TRAIN_STEPS_ARGS[@]}"

echo ""
echo "Sporulation Optuna standard worker ${SLURM_ARRAY_TASK_ID} finished."
echo "Check spore_optuna/${STUDY_NAME}_sporulation/ for trials export (best in DB)."


