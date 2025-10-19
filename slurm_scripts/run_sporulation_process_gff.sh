#!/bin/bash
#SBATCH --job-name=spor_process_gff
#SBATCH --output=sporulation_output/process_gff_%j.out
#SBATCH --error=sporulation_output/process_gff_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --partition=cpu

# --- Environment Setup ---
source /home/yhan/miniconda3/etc/profile.d/conda.sh
conda activate genome

mkdir -p sporulation_output

# Avoid nested over-parallelism and oversubscription
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
# Limit joblib/loky when n_jobs=-1 is used inside scikit-learn
export LOKY_MAX_CPU_COUNT=${SLURM_CPUS_PER_TASK:-8}

echo "Processing GFF files to build RF datasets..."

python -u /home/yhan/GenomeInterpretation/sporulation/code/process_gff_for_rf.py

echo "process_gff completed."


