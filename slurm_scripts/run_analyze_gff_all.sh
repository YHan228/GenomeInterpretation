#!/bin/bash
#SBATCH --job-name=analyze_gff
#SBATCH --output=phenotype_output/analyze_%A_%a.out
#SBATCH --error=phenotype_output/analyze_%A_%a.err
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --array=0-12

set -euo pipefail

REPO_ROOT="/home/yhan/GenomeInterpretation"

source /home/yhan/miniconda3/etc/profile.d/conda.sh
conda activate genome

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
cd "${REPO_ROOT}"

mkdir -p phenotype_output

DATA_ROOT="/vol/projects/BIFO/genomenet/yichen/phenotype/data"
INPUT_DIR="${DATA_ROOT}/processed_gff"
OUT_DIR_BASE="/home/yhan/GenomeInterpretation/phenotype/outputs"
mkdir -p "${OUT_DIR_BASE}"

WINDOWS=50000
WINDOW_SIZE=1000000
SEED=42

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

PHENO_SLUG=$(python - <<PY
from phenotype.code.phenotype_utils import phenotype_to_slug
print(phenotype_to_slug("${PHENO}"))
PY
)

if [ -z "${PHENO_SLUG}" ]; then
  echo "Failed to derive slug for phenotype '${PHENO}'" >&2
  exit 1
fi

echo "[analyze_gff] Starting analysis for phenotype: ${PHENO}" >&2
echo "[analyze_gff] Phenotype slug: ${PHENO_SLUG}" >&2

python -u /home/yhan/GenomeInterpretation/phenotype/code/analyze_gff.py \
  --input "${INPUT_DIR}" \
  --outdir "${OUT_DIR_BASE}" \
  --phenotype "${PHENO}" \
  --phenotype_type auto \
  --windows ${WINDOWS} \
  --window_size ${WINDOW_SIZE} \
  --seed ${SEED}

echo "[analyze_gff] Completed: ${PHENO}" >&2

