#!/bin/bash
#SBATCH --job-name=phen_prep_eval
#SBATCH --output=phenotype_output/prepare_eval_%A_%a.out
#SBATCH --error=phenotype_output/prepare_eval_%A_%a.err
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --array=0-12

set -euo pipefail

source /home/yhan/miniconda3/etc/profile.d/conda.sh
conda activate genome

mkdir -p phenotype_output
mkdir -p phenotype/data/eval

# Configure data root (adjust if needed)
DATA_ROOT="/vol/projects/BIFO/genomenet/yichen/phenotype/data"

PHENOS=(
  "Motility"
  "Gram staining"
  "Aerophilicity"
  "Extreme environment tolerance"
  "Biofilm formation"
  "Animal pathogenicity"
  "Biosafety level"
  "Health association"
  "Host association"
  "Plant pathogenicity"
  "Spore formation"
  "Hemolysis"
  "Cell shape"
)

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
PHENO=${PHENOS[$TASK_ID]:-}

if [ -z "${PHENO}" ]; then
  echo "Invalid phenotype index: ${TASK_ID}" >&2
  exit 1
fi

PHENO_SLUG=$(echo "$PHENO" | tr '[:upper:]' '[:lower:]' | sed 's/ /_/g')

EVAL_ROOT="$PWD/phenotype/data/eval"
EVAL_DIR="${EVAL_ROOT}/${PHENO_SLUG}"

python -u phenotype/code/eval_data.py \
  --phenotype "${PHENO}" \
  --metadata_xlsx "$PWD/sporulation/microbe.cards table S1.xlsx" \
  --processed_gff_dir "${DATA_ROOT}/processed_gff" \
  --test_dirs "${DATA_ROOT}/test" \
  --all_dirs "${DATA_ROOT}/train" "${DATA_ROOT}/validation" "${DATA_ROOT}/test" \
  --out_dir "${EVAL_ROOT}"

echo "Prepared eval data for ${PHENO} in ${EVAL_DIR}"


