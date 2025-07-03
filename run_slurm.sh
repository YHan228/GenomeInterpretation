#!/bin/bash
#SBATCH --job-name=genome_robustness
#SBATCH --output=slurm_output/genome_robustness_%j.out
#SBATCH --error=slurm_output/genome_robustness_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:<YOUR_GPU_TYPE_HERE>:1

# --- Setup Environment ---
# This section is system-dependent.
# You may need to load modules or activate a conda/virtual environment.
# Example for conda:
# source /path/to/conda/etc/profile.d/conda.sh
# conda activate your_env_name

# Create output directories if they don't exist
OUTPUT_DIR="slurm_results"
mkdir -p $OUTPUT_DIR
mkdir -p slurm_output

echo "Starting Python experiment script..."
# Execute the cluster-specific python script
python toy_slurm.py --output_dir $OUTPUT_DIR
echo "Experiment finished. Results are in $OUTPUT_DIR" 