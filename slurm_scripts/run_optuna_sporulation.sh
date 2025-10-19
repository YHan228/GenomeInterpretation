#!/bin/bash
#SBATCH --job-name=optuna_sporulation
#SBATCH --output=sporulation_output/optuna_sporo_%A_%a.out
#SBATCH --error=sporulation_output/optuna_sporo_%A_%a.err
#SBATCH --time=6-00:00:00
#SBATCH --partition=gpu
#SBATCH --qos=verylong
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=150G

# Borrowed structure from run_optuna_singlearch.sh, adapted for sporulation

source /home/yhan/miniconda3/etc/profile.d/conda.sh
conda activate genome

mkdir -p slurm_output

# Per-worker temp dirs to avoid matplotlib/cache collisions
if [ -d "/scratch/$USER" ]; then
  export TMPDIR="/scratch/$USER/optuna_sporo_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
else
  export TMPDIR="/tmp/optuna_sporo_${USER}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
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

# Persist for convenience
echo -n "$OPTUNA_STORAGE" > "$HOME/.optuna_storage_url"

# Controls (can be overridden via env)
STUDY_NAME=${STUDY_NAME:-sporo_robust_strict}
N_TRIALS=${N_TRIALS:-25}
PRUNER=${PRUNER:-hyperband}       # hyperband | median | none
MAX_EPOCHS=${MAX_EPOCHS:-25}
SEQ_LEN=${SEQ_LEN:-1000000}       # Keep 1e6 default for parity with training
EPOCH_BUDGET=${EPOCH_BUDGET:-4096}
VAL_EPOCH_BUDGET=${VAL_EPOCH_BUDGET:-2048}
VAL_STEPS=${VAL_STEPS:-64}
TRAIN_STEPS_PER_EPOCH=${TRAIN_STEPS_PER_EPOCH:-}
NUM_WORKERS=${NUM_WORKERS:-2}

# Unique seed per array worker to diversify TPE suggestions and training RNG
SEED_BASE=${SEED_BASE:-42}
WORKER_OFFSET=${SLURM_ARRAY_TASK_ID:-0}
SEED=$(( SEED_BASE + WORKER_OFFSET ))

# Robust epsilon-only search bounds and eval/test roots
EPS_MIN=${EPS_MIN:-1e-6}
EPS_MAX=${EPS_MAX:-1e-2}
DATA_ROOT=${DATA_ROOT:-/vol/projects/BIFO/genomenet/yichen/phenotype/data}
EVAL_DIR=${EVAL_DIR:-${DATA_ROOT}/eval}
TEST_DIR=${TEST_DIR:-${DATA_ROOT}/test}

# Robust-mode efficiency defaults
ROBUST_EPOCH_BUDGET=${ROBUST_EPOCH_BUDGET:-4096}
ROBUST_VAL_EPOCH_BUDGET=${ROBUST_VAL_EPOCH_BUDGET:-2048}
ROBUST_ACC_BS=${ROBUST_ACC_BS:-64}
ROBUST_EVAL_IG_STEPS=${ROBUST_EVAL_IG_STEPS:-32}
ROBUST_EVAL_PGD_STEPS=${ROBUST_EVAL_PGD_STEPS:-10}
ROBUST_EVAL_SAUC_SUB=${ROBUST_EVAL_SAUC_SUB:-50000}

echo "=========================================="
echo "Sporulation Optuna Worker ${SLURM_ARRAY_TASK_ID}"
echo "=========================================="
echo "Study: ${STUDY_NAME}"
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
echo "--- Robust defaults ---"
echo "robust_epoch_budget=${ROBUST_EPOCH_BUDGET} robust_val_epoch_budget=${ROBUST_VAL_EPOCH_BUDGET}"
echo "hotflip: full top-k (no cap/stride)"
echo "acc_bs=${ROBUST_ACC_BS} ig_steps=${ROBUST_EVAL_IG_STEPS} pgd_steps=${ROBUST_EVAL_PGD_STEPS} sauc_sub=${ROBUST_EVAL_SAUC_SUB}"
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
  --robust-epsilon-only \
  --eps-min ${EPS_MIN} \
  --eps-max ${EPS_MAX} \
  --eval-dir ${EVAL_DIR} \
  --test-dir ${TEST_DIR} \
  --robust-epoch-budget ${ROBUST_EPOCH_BUDGET} \
  --robust-val-epoch-budget ${ROBUST_VAL_EPOCH_BUDGET} \
  --robust-acc-batch-size ${ROBUST_ACC_BS} \
  --robust-eval-ig-steps ${ROBUST_EVAL_IG_STEPS} \
  --robust-eval-pgd-steps ${ROBUST_EVAL_PGD_STEPS} \
  --robust-eval-sauc-subsample ${ROBUST_EVAL_SAUC_SUB} \
  --seed ${SEED} \
  --verbose

echo ""
echo "Sporulation Optuna worker ${SLURM_ARRAY_TASK_ID} finished."
echo "Check spore_optuna/${STUDY_NAME}_sporulation/ for trials export (best in DB)."
