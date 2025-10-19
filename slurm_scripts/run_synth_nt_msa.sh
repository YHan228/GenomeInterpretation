#!/bin/bash
#SBATCH --job-name=synth_nt_msa
#SBATCH --output=sporulation_output/synth_nt_msa_%j.out
#SBATCH --error=sporulation_output/synth_nt_msa_%j.err
#SBATCH --time=8:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --partition=cpu

source /home/yhan/miniconda3/etc/profile.d/conda.sh
conda activate genome

mkdir -p sporulation_output

echo "Running NT-only synthetic MSA/conservation over cached datasets..."

python -u /home/yhan/GenomeInterpretation/sporulation/code/synth_nt_msa.py \
  --dataset_dir "/home/yhan/GenomeInterpretation/dataset_cache" \
  --outdir "/home/yhan/GenomeInterpretation/sporulation/analysis_out/synth_nt_msa" \
  --pattern "" \
  --max_pos 2000 \
  --n_workers ${SLURM_CPUS_PER_TASK:-16}

echo "synth_nt_msa completed."


