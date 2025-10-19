#!/bin/bash
#SBATCH --job-name=refit_topk
#SBATCH --output=slurm_output/refit_topk_%j.out
#SBATCH --error=slurm_output/refit_topk_%j.err
#SBATCH --time=6-00:00:00
#SBATCH --partition=gpu
#SBATCH --qos=verylong
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=50G

#
# Refit top-K models (from summary.json) across ALL 65 datasets.
#
# Usage examples:
#   sbatch slurm_scripts/run_refit_from_summary.sh
#   SUMMARY_JSON=/home/yhan/GenomeInterpretation/optuna_results/unified_8921050_unified_mode_standard/summary.json \
#   TOP_K=3 EPOCHS=40 REVAL_SEEDS=5 \
#   sbatch slurm_scripts/run_refit_from_summary.sh
#
# Optional environment variables:
#   SUMMARY_JSON   : Path to summary.json (default: latest *_unified_mode_standard/summary.json)
#   TOP_K          : Number of top trials to refit (default: 3)
#   EPOCHS         : Training epochs per refit (default: 40)
#   REVAL_SEEDS    : Number of seeds to average per dataset (default: 5)
#   OUTPUT_DIR     : Directory to write results (default: next to summary.json, refit_top${TOP_K})
#   BATCH_SIZE     : Train batch size (default: 512)
#   NUM_WORKERS    : DataLoader workers (default: 2)
#   DETERMINISTIC  : 1 to enable deterministic kernels (default: 0)

set -euo pipefail

source /home/yhan/miniconda3/etc/profile.d/conda.sh
conda activate genome

cd /home/yhan/GenomeInterpretation

mkdir -p slurm_output
mkdir -p dataset_cache

# Set unique temporary directory per job
if [ -d "/scratch/$USER" ]; then
  export TMPDIR="/scratch/$USER/refit_tmp_${SLURM_JOB_ID}"
else
  export TMPDIR="/tmp/refit_tmp_${USER}_${SLURM_JOB_ID}"
fi
mkdir -p "$TMPDIR"
export MPLCONFIGDIR="$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT

# Defaults
TOP_K=${TOP_K:-3}
EPOCHS=${EPOCHS:-40}
REVAL_SEEDS=${REVAL_SEEDS:-5}
BATCH_SIZE=${BATCH_SIZE:-512}
NUM_WORKERS=${NUM_WORKERS:-2}
DETERMINISTIC=${DETERMINISTIC:-0}

# Resolve SUMMARY_JSON if not provided
if [ -z "${SUMMARY_JSON:-}" ]; then
  # Prefer unified standard studies
  SUMMARY_JSON=$(ls -1t optuna_results/*_unified_mode_standard/summary.json 2>/dev/null | head -1 || true)
  # Fallback to any summary.json
  if [ -z "$SUMMARY_JSON" ]; then
    SUMMARY_JSON=$(ls -1t optuna_results/*/summary.json 2>/dev/null | head -1 || true)
  fi
fi

if [ -z "${SUMMARY_JSON:-}" ] || [ ! -f "$SUMMARY_JSON" ]; then
  echo "ERROR: SUMMARY_JSON not set and could not be inferred." >&2
  echo "Set SUMMARY_JSON to a valid summary.json path." >&2
  exit 1
fi

# Default OUTPUT_DIR next to summary.json
if [ -z "${OUTPUT_DIR:-}" ]; then
  base_dir=$(dirname "$SUMMARY_JSON")
  OUTPUT_DIR="${base_dir}/refit_top${TOP_K}"
fi
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Top-K Refit Job"
echo "=========================================="
echo "Summary JSON : ${SUMMARY_JSON}"
echo "Top-K        : ${TOP_K}"
echo "Epochs       : ${EPOCHS}"
echo "Seeds        : ${REVAL_SEEDS}"
echo "Batch size   : ${BATCH_SIZE}"
echo "Num workers  : ${NUM_WORKERS}"
echo "Deterministic: ${DETERMINISTIC}"
echo "Output dir   : ${OUTPUT_DIR}"
echo "=========================================="

DET_FLAG=""
if [ "$DETERMINISTIC" = "1" ]; then
  DET_FLAG="--deterministic"
fi

python -u toy_single_arch.py \
  --refit-from-summary "$SUMMARY_JSON" \
  --top-k ${TOP_K} \
  --epochs ${EPOCHS} \
  --reval-seeds ${REVAL_SEEDS} \
  --batch_size ${BATCH_SIZE} \
  --num_workers ${NUM_WORKERS} \
  --refit-output-dir "$OUTPUT_DIR" \
  ${DET_FLAG}

echo ""
echo "Refit job finished. Results saved to: ${OUTPUT_DIR}"


