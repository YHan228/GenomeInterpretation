#!/bin/bash
#SBATCH --job-name=koo_task3_step1_training
#SBATCH --output=koo_output/task3_step1_training_%j.out
#SBATCH --error=koo_output/task3_step1_training_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --mem=80G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1

# --- Environment Setup ---
# This section is system-dependent and may require modification.
source /home/yhan/miniconda3/etc/profile.d/conda.sh
conda activate genome

# --- Output Directory Setup ---
# Create a directory for Slurm logs if it doesn't exist.
mkdir -p koo_output

# --- Execution ---
echo "Starting model training for Task 3, Step 1..."

# Execute the Python script for sequential training.
# The script's paths are relative to the project root, so we run it from here.
# The -u flag ensures that the output is unbuffered and written in real-time.
python -u koo/code/task3_step1_train_model.py

echo "Training script finished." 