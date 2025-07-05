#!/bin/bash
#SBATCH --job-name=genome_robustness
#SBATCH --output=slurm_output/genome_robustness_%A_%a.out
#SBATCH --error=slurm_output/genome_robustness_%A_%a.err
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1 # Requesting 1 GPU per task (H100/V100/T4)

# --- CHOOSE YOUR EXPERIMENT MODE ---
# To run, uncomment ONE of the three modes below.

# --- Mode 1: All experiments (126 jobs) ---
# This is the currently active mode.
##SBATCH --array=0-125%10
#EXPERIMENT_MODE="all"

# --- Mode 2: Standard vs. Adversarial comparison only (42 jobs) ---
#SBATCH --array=0-41%10
EXPERIMENT_MODE="adv_vs_std"

# --- Mode 3: Regularization comparison, including lambda sweep (105 jobs) ---
# #SBATCH --array=0-104%10
# EXPERIMENT_MODE="regu_comp"

# --- Sanity check to ensure a mode is active ---
if [ -z ${EXPERIMENT_MODE+x} ]; then
    echo "ERROR: You must uncomment one of the EXPERIMENT_MODE blocks in the script to select a run mode."
    exit 1
fi

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
echo "Running in mode: ${EXPERIMENT_MODE}"

# Each array index selects one (schedule, gc, conservation) combo based on the selected mode.
python -u toy_slurm.py --output_dir $OUTPUT_DIR --epochs 30 --array_idx ${SLURM_ARRAY_TASK_ID} --experiment_mode ${EXPERIMENT_MODE}
echo "Experiment finished for task ${SLURM_ARRAY_TASK_ID}. Results are in $OUTPUT_DIR" 