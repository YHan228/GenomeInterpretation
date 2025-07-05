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

        # Use rsync to robustly merge the npz_results directory
        if [ -d "${run_dir}/npz_results" ]; then
            echo "  -> Merging npz_results..."
            rsync -av "${run_dir}/npz_results/" "${DEST_DIR}/npz_results/"
        fi
        
        # Use rsync to robustly merge the tensorboard directory
        if [ -d "${run_dir}/tensorboard" ]; then
            echo "  -> Merging tensorboard..."
            rsync -av "${run_dir}/tensorboard/" "${DEST_DIR}/tensorboard/"
        fi
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

## usage in console
# ./consolidate_results.sh 8726751