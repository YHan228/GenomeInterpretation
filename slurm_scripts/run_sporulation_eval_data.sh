#!/bin/bash
#SBATCH --job-name=spor_eval_data
#SBATCH --output=sporulation_output/eval_data_%j.out
#SBATCH --error=sporulation_output/eval_data_%j.err
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --partition=cpu

# --- Environment Setup ---
source /home/yhan/miniconda3/etc/profile.d/conda.sh
conda activate genome

mkdir -p sporulation_output

DATA_ROOT="/vol/projects/BIFO/genomenet/yichen/phenotype/data"

echo "Preparing evaluation data (windows + masks)..."

python -u sporulation/code/eval_data.py \
  --test_dir "${DATA_ROOT}/test" \
  --processed_gff_dir "${DATA_ROOT}/processed_gff" \
  --sporeinfo_csv "sporulation/sporeinfo.csv" \
  --seq_len 1000000 \
  --n_pos 2500 \
  --n_neg 2500 \
  --seed 42 \
  --out_dir "${DATA_ROOT}/eval"

echo "eval_data completed."

