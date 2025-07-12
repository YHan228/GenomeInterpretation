#!/bin/bash
#
# Ad-hoc script to consolidate SLURM results from multiple directories
# sharing a common job ID into a single, clean directory.
#

# --- Configuration ---
# The parent directory containing all the individual run folders.
SOURCE_PARENT_DIR="slurm_results"

# --- Pre-flight Checks ---
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <SLURM_JOB_ID>"
    echo "Example: $0 8726751"
    exit 1
fi

JOB_ID=$1
DEST_DIR="${SOURCE_PARENT_DIR}/job_${JOB_ID}_consolidated"


# --- Main Logic ---
echo "Starting consolidation for job ID: ${JOB_ID}"
echo "Source pattern: ${SOURCE_PARENT_DIR}/*_${JOB_ID}"
echo "Destination: ${DEST_DIR}"

if [ ! -d "$SOURCE_PARENT_DIR" ]; then
    echo "Error: Source directory '$SOURCE_PARENT_DIR' not found."
    exit 1
fi

# Create the destination directory
mkdir -p "$DEST_DIR"

# Find all directories for the given JOB_ID and loop through them
find_results=$(find "$SOURCE_PARENT_DIR" -maxdepth 1 -type d -name "*_${JOB_ID}")

if [ -z "$find_results" ]; then
    echo "Warning: No directories found matching pattern '*_${JOB_ID}' in ${SOURCE_PARENT_DIR}"
    exit 0
fi

for run_dir in $find_results; do
    if [ -d "$run_dir" ]; then
        echo "Processing: $run_dir"

        # Use rsync to robustly merge all relevant subdirectories
        for subdir in npz_results tensorboard; do
            if [ -d "${run_dir}/${subdir}" ]; then
                echo "  -> Merging ${subdir}..."
                rsync -av --exclude='*/*' --include='*.png' "${run_dir}/${subdir}/" "${DEST_DIR}/${subdir}/" 2>/dev/null
                rsync -av "${run_dir}/${subdir}/" "${DEST_DIR}/${subdir}/"
            fi
        done
    fi
done

echo ""
echo "---"
echo "Consolidation complete!"
echo "All results for job ${JOB_ID} have been merged into: ${DEST_DIR}"
echo ""
echo "You can now generate the final plots by running:"
echo "python toy_slurm.py --aggregate_only --output_dir ${DEST_DIR}"
echo "" 
echo "Example:"
echo "./consolidate_results.sh 8727062"
echo "python toy_slurm.py --aggregate_only --output_dir slurm_results/job_8727062_consolidated"

## usage in console
# ./consolidate_results.sh 8726751