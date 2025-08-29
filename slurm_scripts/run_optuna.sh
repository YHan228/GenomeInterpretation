#!/bin/bash
#SBATCH --job-name=optuna_tuning
#SBATCH --output=slurm_output/optuna_%A_%a.out
#SBATCH --error=slurm_output/optuna_%A_%a.err
#SBATCH --time=6-00:00:00
#SBATCH --partition=gpu
#SBATCH --qos=verylong
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=40G

# Dedicated Optuna HPO worker script. Each array task runs 1 worker (n_jobs=1).
# Workers coordinate via OPTUNA_STORAGE and study name.

source /home/yhan/miniconda3/etc/profile.d/conda.sh
conda activate genome

mkdir -p slurm_output

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
  echo "Example: mysql+pymysql://optuna_user:<ENC_PASS>@<DBHOST>:3306/optuna" >&2
  exit 1
fi

STUDY_NAME=${STUDY_NAME:-toycnn_$(date +%Y%m%d_%H%M%S)}
N_TRIALS=${N_TRIALS:-1000}
HPO_MODE=${HPO_MODE:-standard}       # standard | robust
SAMPLER=${SAMPLER:-nsga2}            # nsga2 | tpe
PRUNER=${PRUNER:-hyperband}          # hyperband | median | none
MAX_EPOCHS=${MAX_EPOCHS:-40}
NUM_SEEDS_PER_TRIAL=${NUM_SEEDS_PER_TRIAL:-1}

# Map dataset combos: always 27 (9 gc * 3 cons); schedule handled per submission for robust modes
NUM_HPO_JOBS=27
ARRAY_IDX=$((SLURM_ARRAY_TASK_ID % NUM_HPO_JOBS))

OUTPUT_DIR="slurm_results/hpo_${SLURM_ARRAY_JOB_ID}"
mkdir -p "${OUTPUT_DIR}"

echo "Starting Optuna worker ${SLURM_ARRAY_TASK_ID}: study_prefix=${STUDY_NAME}, trials=${N_TRIALS}, mode=${HPO_MODE}"
echo "Sampler=${SAMPLER}, Pruner=${PRUNER}, MaxEpochs=${MAX_EPOCHS}, SeedsPerTrial=${NUM_SEEDS_PER_TRIAL}, OPTUNA_STORAGE=${OPTUNA_STORAGE}"
echo "Combo index=${ARRAY_IDX}, Output=${OUTPUT_DIR}"

# Persist storage URL for future runs
echo -n "$OPTUNA_STORAGE" > "$HOME/.optuna_storage_url"

python -u toy_slurm.py \
  --tune \
  --study-name "${STUDY_NAME}" \
  --n-trials ${N_TRIALS} \
  --sampler ${SAMPLER} \
  --pruner ${PRUNER} \
  --mode ${HPO_MODE} \
  --max-epochs ${MAX_EPOCHS} \
  --num-seeds-per-trial ${NUM_SEEDS_PER_TRIAL} \
  --array_idx ${ARRAY_IDX} \
  --output_dir "${OUTPUT_DIR}"

echo "Optuna worker finished ${SLURM_ARRAY_TASK_ID}."

