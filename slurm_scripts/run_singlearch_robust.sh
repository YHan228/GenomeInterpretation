#!/bin/bash
#SBATCH --job-name=robust_hpo
#SBATCH --output=slurm_output/robust_hpo_%A_%a.out
#SBATCH --error=slurm_output/robust_hpo_%A_%a.err
#SBATCH --time=2-00:00:00
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=20G

source /home/yhan/miniconda3/etc/profile.d/conda.sh
conda activate genome
cd /home/yhan/GenomeInterpretation

mkdir -p slurm_output dataset_cache

# Set unique temporary directory per job (helps matplotlib/font cache and tmp conflicts)
if [ -d "/scratch/$USER" ]; then
  export TMPDIR="/scratch/$USER/robust_tmp_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
else
  export TMPDIR="/tmp/robust_tmp_${USER}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
fi
mkdir -p "$TMPDIR"
export MPLCONFIGDIR="$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT

# Enumerate all 65 (gc, cons) combos in the same order as toy_single_arch.GC_HPARAMS x CONS_HPARAMS
GC_LIST=(0.50 0.53 0.55 0.575 0.60 0.625 0.65 0.675 0.70 0.725 0.75 0.775 0.80)
CONS_LIST=(0.60 0.65 0.70 0.75 0.80)

IDX=${SLURM_ARRAY_TASK_ID}
GC_IDX=$(( IDX / ${#CONS_LIST[@]} ))
CONS_IDX=$(( IDX % ${#CONS_LIST[@]} ))
GC=${GC_LIST[$GC_IDX]}
CONS=${CONS_LIST[$CONS_IDX]}

STUDY_NAME="robust_${SLURM_ARRAY_JOB_ID}"
SUMMARY_STD="/home/yhan/GenomeInterpretation/optuna_results/unified_8921050_unified_mode_standard/summary.json"

N_TRIALS=${N_TRIALS:-150}
MAX_EPOCHS=${MAX_EPOCHS:-40}
NUM_SEEDS_PER_TRIAL=${NUM_SEEDS_PER_TRIAL:-1}
PRUNER=${PRUNER:-hyperband}
SAMPLER=${SAMPLER:-tpe}

# Resolve OPTUNA_STORAGE automatically when possible (borrowed from singlearch script)
if [ -z "$OPTUNA_STORAGE" ]; then
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
  echo "Example: mysql+pymysql://optuna_user:<URL-ENCODED-PASS>@<DBHOST>:3306/optuna?charset=utf8mb4" >&2
  exit 1
fi

# Persist storage URL for future runs
echo -n "$OPTUNA_STORAGE" > "$HOME/.optuna_storage_url"

python -u toy_single_arch.py \
  --tune \
  --study-name "${STUDY_NAME}" \
  --mode robust \
  --robust-arch-from-summary "${SUMMARY_STD}" \
  --dataset-gc ${GC} \
  --dataset-cons ${CONS} \
  --n-trials ${N_TRIALS} \
  --sampler ${SAMPLER} \
  --pruner ${PRUNER} \
  --storage "${OPTUNA_STORAGE}" \
  --max-epochs ${MAX_EPOCHS} \
  --num-seeds-per-trial ${NUM_SEEDS_PER_TRIAL} \
  --eval-mode all