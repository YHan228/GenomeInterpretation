#!/bin/bash
#SBATCH --job-name=genome_robustness
#SBATCH --output=slurm_output/genome_robustness_%j.out
#SBATCH --error=slurm_output/genome_robustness_%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1

# --- Setup Environment ---
# This section is system-dependent.
# You may need to load modules or activate a conda/virtual environment.
# Example for conda:
source /home/yhan/miniconda3/etc/profile.d/conda.sh
conda activate genome

# Create output directories if they don't exist
OUTPUT_DIR="slurm_results"
mkdir -p $OUTPUT_DIR
mkdir -p slurm_output

echo "Starting Python experiment script..."
# Execute the cluster-specific python script
python -u toy_slurm.py --output_dir $OUTPUT_DIR
echo "Experiment finished. Results are in $OUTPUT_DIR" 