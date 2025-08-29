#!/bin/bash
#SBATCH --job-name=direct_hotflip
#SBATCH --output=slurm_output_complex/v1_%A_%a.out
#SBATCH --error=slurm_output_complex/v1_%A_%a.err
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --mem=15G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# --- Experiment Configuration ---
# This script runs experiments ONLY for the Direct HotFlip method.
# The hyperparameter space is defined in merged_experiment.py:
#    - 2 sched * 6 GC levels * 4 cons levels = 48 jobs
#
# Total jobs = 48.
# We use a SLURM job array to manage these as individual tasks.
#SBATCH --array=0-47

# --- Environment Setup ---
# This section is system-dependent and may require modification.
source /home/yhan/miniconda3/etc/profile.d/conda.sh
conda activate genome

# --- Output Directory ---
# The directory is named after the master job array ID.
OUTPUT_DIR="slurm_results/direct_hotflip_run_${SLURM_ARRAY_JOB_ID}"
mkdir -p "$OUTPUT_DIR"
mkdir -p slurm_output_complex

# --- Task Dispatch Logic ---
# This script is now hardcoded to run only the 'direct_hotflip' mode.
EXPERIMENT_MODE="direct_hotflip"
# The SLURM array task ID directly corresponds to the python script's array_idx
ARRAY_IDX=${SLURM_ARRAY_TASK_ID}

echo "Starting task ${SLURM_ARRAY_TASK_ID} for job ${SLURM_ARRAY_JOB_ID}"
echo "  - Mode: ${EXPERIMENT_MODE}"
echo "  - Relative Index for Python: ${ARRAY_IDX}"
echo "  - Output Directory: ${OUTPUT_DIR}"

# Execute the Python script with the determined parameters
python -u merged_experiment.py \
    --output_dir "${OUTPUT_DIR}" \
    --epochs 100 \
    --array_idx ${ARRAY_IDX} \
    --experiment_mode ${EXPERIMENT_MODE} \
    --deterministic

echo "Experiment finished for task ${SLURM_ARRAY_TASK_ID}." 