#!/bin/bash
#SBATCH --job-name=plot_aggregate
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --output=slurm_output/plot_aggregate_%j.out

# This script runs the aggregation and plotting step of the analysis.
# It should be pointed to a consolidated results directory.

source /home/yhan/miniconda3/etc/profile.d/conda.sh
conda activate genome

echo "Starting aggregation and plotting..."

python -u toy_slurm.py \
       --output_dir slurm_results/job_8726350_consolidated \
       --aggregate_only

echo "Plotting complete. Results are in slurm_results/job_8726350_consolidated/plots/" 