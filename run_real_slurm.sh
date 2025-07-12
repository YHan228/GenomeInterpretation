#!/bin/bash
#SBATCH --job-name=real_genome_robustness
#SBATCH --output=slurm_output/real_genome_robustness_%A_%a.out
#SBATCH --error=slurm_output/real_genome_robustness_%A_%a.err
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1 # Requesting 1 GPU per task (H100/V100/T4)

# --- Unified Experiment Configuration for clean_real.py ---
# This script runs the full suite of experiments based on the new 3x3 HP grid.
# 1. Adversarial vs. Standard (adv_vs_std):
#    - 2 scheduling modes * 3 GC-gap levels * 3 conservation levels = 18 jobs
# 2. Randomized Smoothing (random_smoothing):
#    - 3 GC-gap levels * 3 conservation levels * 5 epsilon levels = 45 jobs
#
# Total jobs = 18 + 45 = 63.
# We use a SLURM job array to manage these as individual tasks.
# The '%20' limits the number of concurrently running tasks to 20.
#SBATCH --array=0-62%20

# --- Environment Setup ---
# This section is system-dependent and may require modification.
# e.g., source /path/to/your/conda.sh; conda activate your_env
source /home/yhan/miniconda3/etc/profile.d/conda.sh
conda activate genome

# --- Output Directory ---
# All tasks from this job array will write to a single, unique directory,
# avoiding the need for a consolidation script for a single experimental sweep.
# The directory is named after the master job array ID.
OUTPUT_DIR="slurm_results/real_run_${SLURM_ARRAY_JOB_ID}"
mkdir -p "$OUTPUT_DIR"
mkdir -p slurm_output

# --- Task Dispatch Logic ---
# Determine which experiment to run based on the array task ID.
# The Python script's get_experiment_info() function handles the detailed mapping.
NUM_ADV_VS_STD_JOBS=18 # 2 sched * 3 gc_gap * 3 cons

if [ ${SLURM_ARRAY_TASK_ID} -lt ${NUM_ADV_VS_STD_JOBS} ]; then
    EXPERIMENT_MODE="adv_vs_std"
    # The python script expects the array_idx relative to its mode
    ARRAY_IDX=${SLURM_ARRAY_TASK_ID}
else
    EXPERIMENT_MODE="random_smoothing"
    # Subtract the number of adv_vs_std jobs to get the correct relative index
    ARRAY_IDX=$((SLURM_ARRAY_TASK_ID - NUM_ADV_VS_STD_JOBS))
fi

echo "Starting task ${SLURM_ARRAY_TASK_ID} for job ${SLURM_ARRAY_JOB_ID}"
echo "  - Mode: ${EXPERIMENT_MODE}"
echo "  - Relative Index for Python: ${ARRAY_IDX}"
echo "  - Output Directory: ${OUTPUT_DIR}"

# Execute the cleaned-up Python script with the determined parameters
python -u clean_real.py \
    --output_dir "${OUTPUT_DIR}" \
    --epochs 100 \
    --array_idx ${ARRAY_IDX} \
    --experiment_mode ${EXPERIMENT_MODE}

echo "Experiment finished for task ${SLURM_ARRAY_TASK_ID}." 