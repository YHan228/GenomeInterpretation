#!/bin/bash
#SBATCH --job-name=phen_genome_stats
#SBATCH --output=phenotype_output/genome_stats_%j.out
#SBATCH --error=phenotype_output/genome_stats_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=240G
#SBATCH --partition=cpu

# Usage: sbatch slurm_scripts/run_phenotypes_analyze_genome_features.sh [phenotype] [k_values] [max_genomes_per_class] [extra args...]
#        sbatch --array=1-13 slurm_scripts/run_phenotypes_analyze_genome_features.sh "" "" "" --top_kmer_features 80
# Example: sbatch slurm_scripts/run_phenotypes_analyze_genome_features.sh "Spore formation" "3,4,6,8" 250 --top_kmer_features 80 --force_recompute

set -euo pipefail

PHENOTYPE="Spore formation"
K_VALUES="3,4,6,8"
MAX_GENOMES=

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
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
    idx=$((SLURM_ARRAY_TASK_ID - 1))
    if (( idx >= 0 && idx < ${#PHENOS[@]} )); then
        PHENOTYPE="${PHENOS[idx]}"
    else
        echo "SLURM_ARRAY_TASK_ID ${SLURM_ARRAY_TASK_ID} out of range" >&2
        exit 1
    fi
fi

if [[ $# -gt 0 ]]; then
    PHENOTYPE="$1"
    shift
fi
if [[ $# -gt 0 ]]; then
    K_VALUES="$1"
    shift
fi
if [[ $# -gt 0 ]]; then
    MAX_GENOMES="$1"
    shift
fi

EXTRA_ARGS=("$@")

source /home/yhan/miniconda3/etc/profile.d/conda.sh
conda activate genome

mkdir -p phenotype_output

# Keep threaded libs from oversubscribing the allocated cores
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export LOKY_MAX_CPU_COUNT=${SLURM_CPUS_PER_TASK:-64}

CMD=(python -u /home/yhan/GenomeInterpretation/phenotype/code/analyze_genome_features.py \
    --phenotype "${PHENOTYPE}" \
    --k_values "${K_VALUES}" \
    --seed 1337)

if [[ -n "${MAX_GENOMES}" ]]; then
    CMD+=(--max_genomes_per_class "${MAX_GENOMES}")
fi

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    CMD+=("${EXTRA_ARGS[@]}")
fi

echo "Running genome feature analysis with phenotype='${PHENOTYPE}', k_values='${K_VALUES}', max_genomes_per_class='${MAX_GENOMES}'"
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    echo "Additional args: ${EXTRA_ARGS[*]}"
fi
"${CMD[@]}"

echo "Genome feature analysis completed."
