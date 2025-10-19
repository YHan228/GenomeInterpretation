#!/bin/bash
#SBATCH --job-name=spor_train_lasso
#SBATCH --output=sporulation_output/train_lasso_%j.out
#SBATCH --error=sporulation_output/train_lasso_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=235G
#SBATCH --partition=cpu

# --- Environment Setup ---
source /home/yhan/miniconda3/etc/profile.d/conda.sh
conda activate genome

mkdir -p sporulation_output

DATA_ROOT="/vol/projects/BIFO/genomenet/yichen/phenotype/data"

# Avoid nested over-parallelism and oversubscription
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
# Limit joblib/loky when n_jobs=-1 is used inside scikit-learn
export LOKY_MAX_CPU_COUNT=${SLURM_CPUS_PER_TASK:-8}

echo "Training Random Forest (CPU) with HPO..."

# full block:

python -u /home/yhan/GenomeInterpretation/sporulation/code/train_lasso.py \
  --input_dir "${DATA_ROOT}/rfdata" \
  --output_dir "${DATA_ROOT}/rfdata" \
  --n_iter 75 \
  --cv 5 \
  --hpo_verbose 3 \
  --min_prev 0.02

echo "train_rf completed."

