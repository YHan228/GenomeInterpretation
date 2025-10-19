#!/bin/bash
#SBATCH --job-name=spor_evaluation
#SBATCH --output=sporulation_output/evaluation_%j.out
#SBATCH --error=sporulation_output/evaluation_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --mem=120G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1

# --- Environment Setup ---
source /home/yhan/miniconda3/etc/profile.d/conda.sh
conda activate genome

mkdir -p sporulation_output

DATA_ROOT="/vol/projects/BIFO/genomenet/yichen/phenotype/data"

echo "Running evaluation (accuracy + SaAUC)..."

python -u sporulation/code/evaluation.py \
  --eval_dir "${DATA_ROOT}/eval" \
  --test_dir "${DATA_ROOT}/test" \
  --model_path "sporulation/model/best_sporulation_model.pth" \
  --acc_batch_size 8 \
  --ig_batch_size 1 \
  --ig_steps 32 \
  --pgd_epsilon 0.1 \
  --pgd_steps 20 \
  --pgd_step_size 0.01 \
  --sauc_outside_subsample 200000 \
  --tile_len 0 \
  --out_dir "${DATA_ROOT}/eval"

echo "evaluation completed."

