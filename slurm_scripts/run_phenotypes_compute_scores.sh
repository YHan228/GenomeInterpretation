#!/bin/bash
#SBATCH --job-name=phen_eval_scores
#SBATCH --output=phenotype_output/eval_scores_%A_%a.out
#SBATCH --error=phenotype_output/eval_scores_%A_%a.err
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --array=0-0 # 0-12

set -euo pipefail

source /home/yhan/miniconda3/etc/profile.d/conda.sh
conda activate genome

mkdir -p phenotype_output

# Configure data root (adjust if needed)
DATA_ROOT="/vol/projects/BIFO/genomenet/yichen/phenotype/data"

PHENOS=(
  # "Motility"
  # "Gram staining"
  # "Aerophilicity"
  # "Extreme environment tolerance"
  # "Biofilm formation"
  # "Animal pathogenicity"
  # "Biosafety level"
  # "Health association"
  # "Host association"
  # "Plant pathogenicity"
  "Spore formation"
  # "Hemolysis"
  # "Cell shape"
)

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
PHENO=${PHENOS[$TASK_ID]:-}

if [ -z "${PHENO}" ]; then
  echo "Invalid phenotype index: ${TASK_ID}" >&2
  exit 1
fi

PHENO_SLUG=$(echo "$PHENO" | tr '[:upper:]' '[:lower:]' | sed 's/ /_/g')

EVAL_DIR="$PWD/phenotype/data/eval/${PHENO_SLUG}"
MODEL_ROOT="$PWD/phenotype/model/${PHENO_SLUG}"
MODEL_PATH="${MODEL_ROOT}/best_model.pth"

if [ ! -f "${EVAL_DIR}/samples.parquet" ]; then
  echo "[ERR] Eval data not found for ${PHENO} in ${EVAL_DIR}. Run run_phenotypes_prepare_eval_data.sh first." >&2
  exit 2
fi

if [ ! -f "${MODEL_PATH}" ]; then
  echo "[WARN] Model not found at ${MODEL_PATH}. Skipping ${PHENO}." >&2
  exit 3
fi

python -u phenotype/code/evaluation.py \
  --phenotype "${PHENO}" \
  --eval_dir "${EVAL_DIR}" \
  --model_path "${MODEL_PATH}" \
  --acc_batch_size 8 \
  --ig_batch_size 1 \
  --ig_steps 32 \
  --pgd_epsilon 0.1 \
  --pgd_steps 20 \
  --pgd_step_size 0.01 \
  --sauc_outside_subsample 200000 \
  --out_dir "${EVAL_DIR}"

echo "Finished scoring for ${PHENO}"


