#!/bin/bash
#SBATCH --job-name=genome_robust_ext
#SBATCH --output=slurm_output/genome_robust_ext_%A_%a.out
#SBATCH --error=slurm_output/genome_robust_ext_%A_%a.err
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --mem=10G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1 # Requesting 1 GPU per task (H100/V100/T4)

# --- Unified Experiment Configuration ---
# This script runs the extension experiments for GC=0.675 and GC=0.7.
# It adds jobs for:
# 1. Iterative HotFlip (adv_vs_std):
#    - 2 sched * 2 GC levels * 3 cons levels = 12 jobs
# 2. Direct HotFlip (direct_hotflip):
#    - 2 sched * 2 GC levels * 3 cons levels = 12 jobs
#
# Total new jobs = 12 + 12 = 24.
# We use a SLURM job array to manage these as individual tasks.
#SBATCH --array=0-23

# --- Environment Setup ---
# This section is system-dependent and may require modification.
# e.g., source /path/to/your/conda.sh; conda activate your_env
source /home/yhan/miniconda3/etc/profile.d/conda.sh
conda activate genome

# --- Output Directory ---
# This script writes to an EXISTING output directory from a previous run.
OUTPUT_DIR="slurm_results/run_8745488"
mkdir -p "$OUTPUT_DIR"
mkdir -p slurm_output

# --- Task Dispatch Logic ---
# Decode SLURM_ARRAY_TASK_ID into the hyperparameter combination
# to calculate the correct --array_idx for toy_slurm.py.
# The python script now has 9 GC levels. We are running jobs for the last two.
# Total jobs = 2 modes * 2 schedules * 2 new GC * 3 cons = 24
# SLURM_ARRAY_TASK_ID is 0-23

# Cons (3 levels: 0.6, 0.7, 0.8)
cons_idx=$(( SLURM_ARRAY_TASK_ID % 3 ))

# New GC (2 levels: 0.675, 0.7)
gc_level_idx=$(( (SLURM_ARRAY_TASK_ID / 3) % 2 ))

# Schedule (2 levels: True, False)
schedule_idx=$(( (SLURM_ARRAY_TASK_ID / 6) % 2 ))

# Mode (2 levels: adv_vs_std, direct_hotflip)
mode_idx=$(( SLURM_ARRAY_TASK_ID / 12 ))

# Determine EXPERIMENT_MODE from mode_idx
if [ ${mode_idx} -eq 0 ]; then
    EXPERIMENT_MODE="adv_vs_std"
else
    EXPERIMENT_MODE="direct_hotflip"
fi

# The full GC_HPARAMS list in python is now 9 elements long.
# The original script ran for the first 7 (indices 0-6).
# The new GC values (0.675, 0.7) correspond to indices 7 and 8.
gc_idx=$(( 7 + gc_level_idx ))

# Calculate the python script's expected array_idx based on the combo.
# Formula from toy_slurm.py's get_experiment_combos function:
# idx = schedule_idx * (n_gc * n_cons) + gc_idx * n_cons + cons_idx
NUM_GC_TOTAL=9
NUM_CONS=3
offset_sched=$(( schedule_idx * NUM_GC_TOTAL * NUM_CONS ))
offset_gc=$(( gc_idx * NUM_CONS ))
ARRAY_IDX=$(( offset_sched + offset_gc + cons_idx ))

echo "Starting extension task ${SLURM_ARRAY_TASK_ID} for job ${SLURM_JOB_ID}"
echo "  - Decoded mode_idx=${mode_idx}, schedule_idx=${schedule_idx}, gc_level_idx=${gc_level_idx}, cons_idx=${cons_idx}"
echo "  - Mode: ${EXPERIMENT_MODE}"
echo "  - Python Array Index: ${ARRAY_IDX}"
echo "  - Output Directory: ${OUTPUT_DIR}"

# Execute the Python script with the determined parameters
python -u toy_slurm.py \
    --output_dir "${OUTPUT_DIR}" \
    --epochs 100 \
    --array_idx ${ARRAY_IDX} \
    --experiment_mode ${EXPERIMENT_MODE} \
    --deterministic

echo "Experiment finished for task ${SLURM_ARRAY_TASK_ID}." 