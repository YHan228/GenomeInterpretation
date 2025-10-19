#!/bin/bash
#SBATCH --job-name=analyze_gff
#SBATCH --output=sporulation_output/analyze_gff_%j.out
#SBATCH --error=sporulation_output/analyze_gff_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --partition=cpu

# --- Environment Setup ---
source /home/yhan/miniconda3/etc/profile.d/conda.sh
conda activate genome

mkdir -p sporulation_output

DATA_ROOT="/vol/projects/BIFO/genomenet/yichen/phenotype/data"

# --- I/O Paths ---
INPUT_DIR="${DATA_ROOT}/processed_gff"
OUTDIR="sporulation/analysis_out"
mkdir -p "${OUTDIR}"

# Optional: retained gene flags produced by retained_spore_genes_analysis.py
# Uncomment and set to enable redefining spore_related from allowed genes (rule in {a,b,c})
RETAINED_FLAGS="sporulation/analysis_out/retained_gene_flags.csv"
RETAINED_RULE="c"

# Optional: restrict to subset of genomes by glob on gff_filename
# TEST_GLOB="*test*"

WINDOWS=50000
WINDOW_SIZE=1000000
SEED=42

echo "Running analyze_gff.py..."

python -u sporulation/code/analyze_gff.py \
  --input "${INPUT_DIR}" \
  --outdir "${OUTDIR}" \
  --windows ${WINDOWS} \
  --window_size ${WINDOW_SIZE} \
  --seed ${SEED} \
  ${TEST_GLOB:+--test_glob "${TEST_GLOB}"} \
  ${RETAINED_FLAGS:+--retained_flags "${RETAINED_FLAGS}"} \
  ${RETAINED_RULE:+--retained_rule "${RETAINED_RULE}"}

rc=$?
if [[ $rc -ne 0 ]]; then
  echo "analyze_gff.py exited with code $rc" >&2
  exit $rc
fi

echo "analyze_gff completed. Outputs in ${OUTDIR}"

