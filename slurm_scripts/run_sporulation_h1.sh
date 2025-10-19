#!/bin/bash
#SBATCH --job-name=pheno_mult_cluster
#SBATCH --output=sporulation_output/h1_%j.out
#SBATCH --error=sporulation_output/h1_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --mem=300G
#SBATCH --partition=cpu
#SBATCH --array=0-12

# --- Environment Setup ---
source /home/yhan/miniconda3/etc/profile.d/conda.sh
conda activate genome

mkdir -p sporulation_output

DATA_ROOT="/vol/projects/BIFO/genomenet/yichen/phenotype/data"

# Avoid nested over-parallelism and oversubscription
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
# Limit joblib/loky when n_jobs=-1 is used inside scikit-learn/joblib
export LOKY_MAX_CPU_COUNT=${SLURM_CPUS_PER_TASK:-50}

PHENOS=(
  "Motility"
  "Gram staining"
  "Aerophilicity"
  "Extreme environment tolerance"
  "Biofilm formation"
  "Animal pathogenicity"
  "Biosafety level"
  "Health association"
  "Host association"
  "Plant pathogenicity"
  "Spore formation"
  "Hemolysis"
  "Cell shape"
)

PHENOTYPE=${PHENOS[$SLURM_ARRAY_TASK_ID]}

echo "Running multiplicity H1 experiment for phenotype: ${PHENOTYPE} (CPSS, Boruta, Perm VI, CPI)..."

python -u /home/yhan/GenomeInterpretation/sporulation/code/multiplicity_h1.py \
  --input_dir "${DATA_ROOT}/rfdata" \
  --output_dir /home/yhan/GenomeInterpretation/sporulation/results/h1_multiplicity \
  --phenotype "${PHENOTYPE}" \
  --seed 42 \
  --min_prev 0.02 \
  --cpss_pairs 100 \
  --cpss_tau 0.7 \
  --cpss_q 100 \
  --cpss_cv 5 \
  --boruta_runs 100 \
  --rf_trees 600 \
  --rf_max_depth 30 \
  --perm_reps 20 \
  --perm_topk 2000 \
  --cpi_candidates 500 \
  --cpi_reps 20 \
  --leaf_max_depth 5 \
  --leaf_min_samples 30 \
  --topn 100 \
  --net_edge_tau 0.2 \
  --clustered \
  --cluster_threshold 0.7 \
  --cluster_metric ochiai

echo "multiplicity_h1 completed."

