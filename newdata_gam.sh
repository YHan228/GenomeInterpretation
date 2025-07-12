#!/bin/bash
#SBATCH --job-name=final_gam_exp
#SBATCH --output=slurm_output/final_gam_%A_%a.out
#SBATCH --error=slurm_output/final_gam_%A_%a.err
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-99%20

# --- Configuration ---
# The hyperparameter grid is now defined within the Python script.
# This script runs all 80 combinations (8 GC Gaps x 10 Conservations).
# The '%20' at the end limits the number of concurrently running tasks to 20.

# --- Environment Setup ---
# This section is system-dependent.
# Update it to activate your specific conda or virtual environment.
source /home/yhan/miniconda3/etc/profile.d/conda.sh
conda activate genome

# --- Output Directory ---
# Create a single, unique directory for all results from this job array.
OUTPUT_DIR="slurm_results/final_gam_run_${SLURM_ARRAY_JOB_ID}"
mkdir -p "$OUTPUT_DIR"
mkdir -p slurm_output # For SLURM's own log files

echo "--- Starting SLURM Task for Final GAM Experiment ---"
echo "Job Array ID: ${SLURM_ARRAY_JOB_ID}"
echo "Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Output Directory: ${OUTPUT_DIR}"
echo "----------------------------------------------------"

# --- Guard against incorrect execution ---
if [ -z "${SLURM_ARRAY_TASK_ID}" ]; then
    echo "Error: This script is designed to be run as a SLURM job array." >&2
    echo "The SLURM_ARRAY_TASK_ID environment variable is not set." >&2
    echo "Please submit this script using 'sbatch run_final_experiment.sh'" >&2
    exit 1
fi

# --- Execute Python Script for a single combination ---
# The python script is run with the --array_idx argument, which tells it
# exactly which hyperparameter combination to run for this task.
python -u final_gam_experiment.py \
    --output_dir "${OUTPUT_DIR}" \
    --epochs 50 \
    --array_idx ${SLURM_ARRAY_TASK_ID} \
    --num_workers 2

echo "--- Finished SLURM Task ${SLURM_ARRAY_TASK_ID} ---"

# --------------------------------------------------------------------
# AGGREGATION STEP: Run this command manually after the job array completes.
# --------------------------------------------------------------------
# This command will find all the individual .npz result files, combine them,
# run the GAM analysis, and generate the final plots.
#
# srun --pty --mem=16G --time=0-00:30:00 python final_gam_experiment.py \
#    --output_dir "slurm_results/final_gam_run_{YOUR_JOB_ARRAY_ID}" \
#    --aggregate_only
#
# Replace {YOUR_JOB_ARRAY_ID} with the actual ID from your SLURM job.
# -------------------------------------------------------------------- 