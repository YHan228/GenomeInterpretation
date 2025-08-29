#!/bin/bash
#SBATCH --job-name=genome_robustness
#SBATCH --output=slurm_output/genome_robustness_%A_%a.out
#SBATCH --error=slurm_output/genome_robustness_%A_%a.err
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --mem=10G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1 # Requesting 1 GPU per task (H100/V100/T4)

# --- Unified Experiment Configuration ---
# This script runs the full suite of experiments, including:
# 1. Iterative HotFlip (adv_vs_std):
#    - 2 sched * 9 GC levels * 3 cons levels = 54 jobs
# 2. Direct HotFlip (direct_hotflip):
#    - 2 sched * 9 GC levels * 3 cons levels = 54 jobs
#
# Total jobs = 54 + 54 = 108.
# We use a SLURM job array to manage these as individual tasks.
#SBATCH --array=0-107

# --- Model Architecture ---
# This script executes `toy_slurm.py`, which uses the new, modernized `TinyCNN`
# architecture by default. The original model is preserved as `TinyCNNv0`.

# --- Environment Setup ---
# This section is system-dependent and may require modification.
# e.g., source /path/to/your/conda.sh; conda activate your_env
source /home/yhan/miniconda3/etc/profile.d/conda.sh
conda activate genome

# --- Output Directory ---
# All tasks from this job array will write to a single, unique directory,
# avoiding the need for a consolidation script for a single experimental sweep.
# The directory is named after the master job array ID.
OUTPUT_DIR="slurm_results/run_${SLURM_ARRAY_JOB_ID}"
mkdir -p "$OUTPUT_DIR"
mkdir -p slurm_output

# --- Task Dispatch: Fixed experiment sweep only (no HPO here) ---
NUM_ITERATIVE_JOBS=54 # 2 sched * 9 gc * 3 cons

if [ ${SLURM_ARRAY_TASK_ID} -lt ${NUM_ITERATIVE_JOBS} ]; then
    EXPERIMENT_MODE="adv_vs_std"
    # The python script expects the array_idx relative to its mode
    ARRAY_IDX=${SLURM_ARRAY_TASK_ID}
else
    EXPERIMENT_MODE="direct_hotflip"
    # Subtract the number of iterative jobs to get the correct relative index
    ARRAY_IDX=$((SLURM_ARRAY_TASK_ID - NUM_ITERATIVE_JOBS))
fi

echo "Starting task ${SLURM_ARRAY_TASK_ID} for job ${SLURM_ARRAY_JOB_ID}"
echo "  - Mode: ${EXPERIMENT_MODE}"
echo "  - Relative Index for Python: ${ARRAY_IDX}"
echo "  - Output Directory: ${OUTPUT_DIR}"

# Execute the Python script with the determined parameters
python -u toy_slurm.py \
    --output_dir "${OUTPUT_DIR}" \
    --epochs 100 \
    --array_idx ${ARRAY_IDX} \
    --experiment_mode ${EXPERIMENT_MODE} \
    --deterministic

echo "Experiment finished for task ${SLURM_ARRAY_TASK_ID}." 