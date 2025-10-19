#!/bin/bash
#SBATCH --job-name=sporulation_training_v2
#SBATCH --output=sporulation_output/training_%j.out
#SBATCH --error=sporulation_output/training_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --mem=100G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1

# --- Environment Setup ---
source /home/yhan/miniconda3/etc/profile.d/conda.sh
conda activate genome

# --- Execution ---
echo "Starting sporulation model v2 training..."

# Execute the Python script for training.
# The -u flag ensures that the output is unbuffered and written in real-time.
python -u sporulation/code/training.py --mode train

echo "Training script finished."
