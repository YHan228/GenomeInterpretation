#!/bin/bash
#SBATCH --job-name=clean_genome_robustness
#SBATCH --output=slurm_output/clean_genome_robustness_%A_%a.out
#SBATCH --error=slurm_output/clean_genome_robustness_%A_%a.err
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# --- Experiment Configuration ---
# Total 63 jobs split between two experiment types:
# - Adversarial vs Standard: jobs 0-17 (18 total)
# - Randomized Smoothing: jobs 18-62 (45 total)
#SBATCH --array=0-62%20

# --- Environment Setup ---
source /home/yhan/miniconda3/etc/profile.d/conda.sh
conda activate genome

# --- Output Directory ---
OUTPUT_DIR="slurm_results/clean_run_${SLURM_ARRAY_JOB_ID}"
mkdir -p "$OUTPUT_DIR"
mkdir -p slurm_output

# --- Task Dispatch ---
NUM_ADV_VS_STD_JOBS=18

if [ ${SLURM_ARRAY_TASK_ID} -lt ${NUM_ADV_VS_STD_JOBS} ]; then
    EXPERIMENT_MODE="adv_vs_std"
    ARRAY_IDX=${SLURM_ARRAY_TASK_ID}
else
    EXPERIMENT_MODE="random_smoothing"
    ARRAY_IDX=$((SLURM_ARRAY_TASK_ID - NUM_ADV_VS_STD_JOBS))
fi

echo "Starting task ${SLURM_ARRAY_TASK_ID} (${EXPERIMENT_MODE} index ${ARRAY_IDX})"
echo "Output: ${OUTPUT_DIR}"

# Run the experiment
python -u clean_real.py \
    --output_dir "${OUTPUT_DIR}" \
    --epochs 100 \
    --array_idx ${ARRAY_IDX} \
    --experiment_mode ${EXPERIMENT_MODE}

echo "Task ${SLURM_ARRAY_TASK_ID} completed." 