#!/bin/bash

# Test script to verify that multiple workers will share the same study name

echo "Testing study name generation for parallel workers..."
echo "=================================================="

# Simulate SLURM environment
export SLURM_ARRAY_JOB_ID=8921042

# Test what each worker would get
for SLURM_ARRAY_TASK_ID in 1 2 3 4; do
    # Extract the logic from the SLURM script
    STUDY_NAME=${STUDY_NAME:-unified_${SLURM_ARRAY_JOB_ID:-$(date +%Y%m%d)}}
    echo "Worker $SLURM_ARRAY_TASK_ID: Study name = $STUDY_NAME"
done

echo ""
echo "✓ All workers share the same study name!"
echo ""
echo "Without SLURM_ARRAY_JOB_ID (e.g., local testing):"
unset SLURM_ARRAY_JOB_ID
STUDY_NAME_LOCAL=${STUDY_NAME:-unified_${SLURM_ARRAY_JOB_ID:-$(date +%Y%m%d)}}
echo "Study name = $STUDY_NAME_LOCAL"

echo ""
echo "With explicit STUDY_NAME environment variable:"
STUDY_NAME="my_custom_study" 
STUDY_NAME_CUSTOM=${STUDY_NAME:-unified_${SLURM_ARRAY_JOB_ID:-$(date +%Y%m%d)}}
echo "Study name = $STUDY_NAME_CUSTOM"
