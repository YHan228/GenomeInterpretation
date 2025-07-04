#!/bin/bash
#SBATCH --job-name=genome_robustness
#SBATCH --output=slurm_output/genome_robustness_%A_%a.out
#SBATCH --error=slurm_output/genome_robustness_%A_%a.err
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1 # Requesting 1 GPU per task (H100/V100/T4)
#SBATCH --array=0-41%10

# --- Setup Environment ---
# This section is system-dependent.
# You may need to load modules or activate a conda/virtual environment.
# Example for conda:
source /home/yhan/miniconda3/etc/profile.d/conda.sh
conda activate genome

# Create a unique output directory for this specific job array run
# All tasks in the array will write to the same top-level directory
TIMESTAMP=$(date +"%Y%m%d-%H%M%S")
UNIQUE_DIR_NAME="run_${TIMESTAMP}_${SLURM_ARRAY_JOB_ID}"
OUTPUT_DIR="slurm_results/${UNIQUE_DIR_NAME}"

mkdir -p $OUTPUT_DIR
mkdir -p slurm_output

echo "Starting Python experiment script for array index ${SLURM_ARRAY_TASK_ID}..."
# Each array index selects one (schedule, gc, conservation) combo.
python -u toy_slurm.py --output_dir $OUTPUT_DIR --epochs 30 --array_idx ${SLURM_ARRAY_TASK_ID}
echo "Experiment finished for task ${SLURM_ARRAY_TASK_ID}. Results are in $OUTPUT_DIR" 