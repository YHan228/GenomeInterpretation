#!/bin/bash
#SBATCH --job-name=optuna_pheno_robust
#SBATCH --output=phenotype_output/optuna_pheno_robust_%A_%a.out
#SBATCH --error=phenotype_output/optuna_pheno_robust_%A_%a.err
#SBATCH --time=2-00:00:00
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=80G
#SBATCH --array=0-7%8 # adjust to cover (num_phenotypes * chains_per_pheno)

# Robust Optuna tuning for sporulation (epsilon search) using direct hotflip regularization.

source /home/yhan/miniconda3/etc/profile.d/conda.sh
conda activate genome

mkdir -p phenotype_output

if [ -d "/scratch/$USER" ]; then
  export TMPDIR="/scratch/$USER/optuna_pheno_robust_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
else
  export TMPDIR="/tmp/optuna_pheno_robust_${USER}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
fi
mkdir -p "$TMPDIR"
export MPLCONFIGDIR="$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT

if [ -z "$OPTUNA_STORAGE" ] && [ -f "$HOME/.optuna_storage_url" ]; then
  read -r OPTUNA_STORAGE < "$HOME/.optuna_storage_url"
fi

if [ -z "$OPTUNA_STORAGE" ]; then
  OPTUNA_STORAGE="sqlite:///$(pwd)/optuna.db"
fi

echo -n "$OPTUNA_STORAGE" > "$HOME/.optuna_storage_url"

STUDY_NAME_BASE=${STUDY_NAME_BASE:-spore_cluster_robust_v1}
N_TRIALS=${N_TRIALS:-10}
MAX_EPOCHS=${MAX_EPOCHS:-25}
EPOCH_BUDGET=${EPOCH_BUDGET:-2048}
VAL_EPOCH_BUDGET=${VAL_EPOCH_BUDGET:-1024}
VAL_STEPS=${VAL_STEPS:-64}
NUM_WORKERS=${NUM_WORKERS:-2}
CHAINS_PER_PHENO=${CHAINS_PER_PHENO:-8}

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
PHENO_IDX=$(( TASK_ID / CHAINS_PER_PHENO ))
CHAIN_IDX=$(( TASK_ID % CHAINS_PER_PHENO ))

SEED_BASE=${SEED_BASE:-321}
SEED=$(( SEED_BASE + PHENO_IDX * 100 + CHAIN_IDX ))

PHENOS=(
  "Spore formation"
)

PHENO=${PHENOS[$PHENO_IDX]}
PHENO_SLUG=$(echo "$PHENO" | tr '[:upper:]' '[:lower:]' | sed 's/ /_/g')
if [ -z "$PHENO" ] || [ "$PHENO_SLUG" != "spore_formation" ]; then
  echo "Invalid phenotype selection (expected Spore formation). Index ${PHENO_IDX}, value '${PHENO}'." >&2
  exit 1
fi
STUDY_NAME="${STUDY_NAME_BASE}_${PHENO_SLUG}"

EVAL_ROOT=${EVAL_ROOT:-phenotype/data/eval}
EVAL_DIR="${EVAL_ROOT}/${PHENO_SLUG}"

echo "=========================================="
echo "Robust Optuna Phenotype Worker ${SLURM_ARRAY_TASK_ID}"
echo "Phenotype: ${PHENO} (index ${PHENO_IDX})"
echo "Chain: ${CHAIN_IDX}"
echo "Study: ${STUDY_NAME}"
echo "Trials per worker: ${N_TRIALS}"
echo "Max epochs: ${MAX_EPOCHS}"
echo "Train epoch budget: ${EPOCH_BUDGET}"
echo "Val epoch budget: ${VAL_EPOCH_BUDGET}"
echo "Seed: ${SEED}"
echo "Storage: ${OPTUNA_STORAGE}"
echo "Eval dir: ${EVAL_DIR}" 
echo "=========================================="

python -u phenotype/code/optuna_phenotype.py \
  --tune \
  --study-name "${STUDY_NAME}" \
  --n-trials ${N_TRIALS} \
  --max-epochs ${MAX_EPOCHS} \
  --epoch-budget ${EPOCH_BUDGET} \
  --val-epoch-budget ${VAL_EPOCH_BUDGET} \
  --val-steps ${VAL_STEPS} \
  --num-workers ${NUM_WORKERS} \
  --storage "${OPTUNA_STORAGE}" \
  --seed ${SEED} \
  --verbose \
  --metadata-xlsx "sporulation/microbe.cards table S1.xlsx" \
  --phenotype-col "${PHENO}" \
  --file-col "Fasta file" \
  --outdir-root "phenotype/model" \
  --eval-dir "${EVAL_ROOT}" \
  --robust-epsilon-only \
  --eps-min 1e-6 \
  --eps-max 1e-2 \
  --robust-epoch-budget ${ROBUST_EPOCH_BUDGET:-$EPOCH_BUDGET} \
  --robust-val-epoch-budget ${ROBUST_VAL_EPOCH_BUDGET:-$VAL_EPOCH_BUDGET} \
  --robust-acc-batch-size ${ROBUST_ACC_BATCH_SIZE:-16} \
  --robust-eval-ig-steps ${ROBUST_IG_STEPS:-16} \
  --robust-eval-pgd-steps ${ROBUST_PGD_STEPS:-10} \
  --robust-eval-sauc-subsample ${ROBUST_SAUC_SUB:-50000}

echo ""
echo "Optuna phenotype robust worker ${SLURM_ARRAY_TASK_ID} (pheno ${PHENO_SLUG}, chain ${CHAIN_IDX}) finished."
echo "Check phenotype/robust_model/${STUDY_NAME}/ for best checkpoints and metrics."
