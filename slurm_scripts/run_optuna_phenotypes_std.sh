#!/bin/bash
#SBATCH --job-name=optuna_pheno_std
#SBATCH --output=phenotype_output/optuna_pheno_std_%A_%a.out
#SBATCH --error=phenotype_output/optuna_pheno_std_%A_%a.err
#SBATCH --time=6-00:00:00
#SBATCH --partition=gpu
#SBATCH --qos=verylong
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --array=0-7%8 # for all phenos each 4 chains: 0-47

# Phenotype-agnostic standard Optuna tuning across 12 phenotypes, 4 chains each.

source /home/yhan/miniconda3/etc/profile.d/conda.sh
conda activate genome

mkdir -p phenotype_output

# Per-worker temp dirs to avoid matplotlib/cache collisions
if [ -d "/scratch/$USER" ]; then
  export TMPDIR="/scratch/$USER/optuna_pheno_std_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
else
  export TMPDIR="/tmp/optuna_pheno_std_${USER}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
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

# Controls
STUDY_NAME_BASE=${STUDY_NAME_BASE:-pheno_full_std_v1}
N_TRIALS=${N_TRIALS:-10}
PRUNER=${PRUNER:-hyperband}
MAX_EPOCHS=${MAX_EPOCHS:-25}
SEQ_LEN=${SEQ_LEN:-1000000}
EPOCH_BUDGET=${EPOCH_BUDGET:-2048}
VAL_EPOCH_BUDGET=${VAL_EPOCH_BUDGET:-1024}
VAL_STEPS=${VAL_STEPS:-64}
NUM_WORKERS=${NUM_WORKERS:-2}
CHAINS_PER_PHENO=8

# Optional: limit training steps per epoch if provided
TRAIN_STEPS_ARGS=()
if [ -n "$TRAIN_STEPS_PER_EPOCH" ]; then
  TRAIN_STEPS_ARGS+=(--train-steps-per-epoch "$TRAIN_STEPS_PER_EPOCH")
fi

# Derive phenotype index and chain index from array task id
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
PHENO_IDX=$(( TASK_ID / CHAINS_PER_PHENO ))
CHAIN_IDX=$(( TASK_ID % CHAINS_PER_PHENO ))

# Unique seed per chain
SEED_BASE=${SEED_BASE:-246}
SEED=$(( SEED_BASE + PHENO_IDX * 100 + CHAIN_IDX ))

# PHENOS=(
#   "Motility"
#   "Gram staining"
#   "Aerophilicity"
#   "Extreme environment tolerance"
#   "Biofilm formation"
#   "Animal pathogenicity"
#   "Biosafety level"
#   "Health association"
#   "Host association"
#   "Plant pathogenicity"
#   "Spore formation"
#   "Hemolysis"
# )

PHENOS=(
  #"Motility"
  #"Gram staining"
  "Spore formation"
)


PHENO=${PHENOS[$PHENO_IDX]}
PHENO_SLUG=$(echo "$PHENO" | tr '[:upper:]' '[:lower:]' | sed 's/ /_/g')
# Separate study name per phenotype (same across chains to aggregate into one study per phenotype)
STUDY_NAME="${STUDY_NAME_BASE}_${PHENO_SLUG}"

echo "=========================================="
echo "Optuna Phenotype Worker ${SLURM_ARRAY_TASK_ID}"
echo "=========================================="
echo "Phenotype: ${PHENO} (index ${PHENO_IDX})"
echo "Chain: ${CHAIN_IDX}"
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
echo "=========================================="

python -u phenotype/code/optuna_phenotype.py \
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
  --metadata-xlsx "sporulation/microbe.cards table S1.xlsx" \
  --phenotype-col "${PHENO}" \
  --file-col "Fasta file" \
  --outdir-root "phenotype/model" \
  "${TRAIN_STEPS_ARGS[@]}"

echo ""
echo "Optuna phenotype standard worker ${SLURM_ARRAY_TASK_ID} (pheno ${PHENO_SLUG}, chain ${CHAIN_IDX}) finished."
echo "Check phenotype/model/${PHENO_SLUG}/ for trials export (best in DB)."


