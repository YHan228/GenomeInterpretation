#!/bin/bash
#SBATCH --job-name=gam_analysis
#SBATCH --output=slurm_output/gam_analysis_%A_%a.out
#SBATCH --error=slurm_output/gam_analysis_%A_%a.err
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-224%20

# --- Configuration ---
# Set the granularity of the hyperparameter grid search.
NUM_GC_STEPS=15
NUM_CONS_STEPS=15

# Calculate the total number of jobs for the SLURM array.
# The array is 0-indexed, so we subtract 1.
# TOTAL_JOBS=$((NUM_GC_STEPS * NUM_CONS_STEPS - 1)) # This is incorrect for #SBATCH

# Set the job array. The range must be hardcoded because #SBATCH directives
# are parsed before shell variables are assigned. With 15x15 steps, this is 225 jobs (0-224).
# The '%20' at the end limits the number of concurrently running tasks.

# --- Environment Setup ---
# This section is system-dependent.
# Update it to activate your specific conda or virtual environment.
source /home/yhan/miniconda3/etc/profile.d/conda.sh
conda activate genome

# --- Output Directory ---
# Create a single, unique directory for all results from this job array.
# This makes finding and aggregating results straightforward.
OUTPUT_DIR="slurm_results/gam_run_${SLURM_ARRAY_JOB_ID}"
mkdir -p "$OUTPUT_DIR"
mkdir -p slurm_output # For SLURM's own log files

echo "--- Starting SLURM Task ---"
echo "Job Array ID: ${SLURM_ARRAY_JOB_ID}"
echo "Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Output Directory: ${OUTPUT_DIR}"
echo "---------------------------"

# --- Guard against incorrect execution ---
# Exit with a helpful message if not run as a SLURM array task.
if [ -z "${SLURM_ARRAY_TASK_ID}" ]; then
    echo "Error: This script is designed to be run as a SLURM job array." >&2
    echo "The SLURM_ARRAY_TASK_ID environment variable is not set." >&2
    echo "Please submit this script using 'sbatch gam_data.sh'" >&2
    exit 1
fi

# --- Execute Python Script for a single combination ---
# The python script is run with the --array_idx argument, which tells it
# exactly which (GC, Conservation) combination to run for this task.
python -u gam_analysis_for_dataset_hps.py \
    --output_dir "${OUTPUT_DIR}" \
    --epochs 50 \
    --num_gc_steps ${NUM_GC_STEPS} \
    --num_cons_steps ${NUM_CONS_STEPS} \
    --array_idx ${SLURM_ARRAY_TASK_ID} \
    --num_workers 2

echo "--- Finished SLURM Task ${SLURM_ARRAY_TASK_ID} ---"

# --------------------------------------------------------------------
# AGGREGATION STEP: Run this command manually after the job array completes.
# --------------------------------------------------------------------
# This command will find all the individual .npz result files, combine them,
# run the GAM analysis, and generate the final plots.
#
# srun --pty --mem=16G --time=0-00:30:00 python gam_analysis_for_dataset_hps.py \
#    --output_dir "slurm_results/gam_run_${SLURM_ARRAY_JOB_ID}" \
#    --aggregate_only
#
# -------------------------------------------------------------------- 