#!/bin/bash
#SBATCH --job-name=spor_codon_msa
#SBATCH --output=sporulation_output/codon_msa_%j.out
#SBATCH --error=sporulation_output/codon_msa_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=cpu

# --- Environment Setup ---
source /home/yhan/miniconda3/etc/profile.d/conda.sh
conda activate genome

mkdir -p sporulation_output

DATA_ROOT="/vol/projects/BIFO/genomenet/yichen/phenotype/data"

echo "Running codon-aware MSA pipeline (MACSE) for sporulation genes..."

python sporulation/code/codon_msa.py \
  --genome_dirs ${DATA_ROOT}/train ${DATA_ROOT}/validation \
  --sporeinfo sporulation/sporeinfo.csv \
  --intervals ${DATA_ROOT}/eval/genome_intervals.parquet \
  --outdir sporulation/analysis_out/codon_msa \
  --min_frequency 10 \
  --macse macse \
  --n_workers ${SLURM_CPUS_PER_TASK:-16}

echo "codon_msa completed."

