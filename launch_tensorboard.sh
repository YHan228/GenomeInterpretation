#!/bin/bash

# This script launches TensorBoard on the login node and provides instructions
# for setting up an SSH tunnel to view it on your local machine.

# --- Configuration ---
# The root directory where all your slurm_results are stored.
LOG_ROOT="slurm_results" 
PORT=6007 # The port to run TensorBoard on (on the remote server)

# --- Find the most recent consolidated TensorBoard log directory ---
# We prioritize directories that follow the new, clean structure.
LATEST_LOG_DIR=$(find "$LOG_ROOT" -mindepth 1 -maxdepth 3 -type d -name "tensorboard" -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)

if [ -z "$LATEST_LOG_DIR" ]; then
    echo "Warning: Could not find a consolidated 'tensorboard' directory."
    echo "Falling back to searching for older 'run_*' directories..."
    LATEST_LOG_DIR=$(find "$LOG_ROOT" -mindepth 1 -maxdepth 1 -type d -name 'run_*' -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)
fi

if [ -z "$LATEST_LOG_DIR" ]; then
    echo "Error: Could not find any experiment run directories to launch TensorBoard."
    exit 1
fi

echo "Found latest log directory: $LATEST_LOG_DIR"
echo "TensorBoard will scan this directory for all logs."

# --- Instructions for SSH Tunnel ---
LOGIN_NODE_HOSTNAME=$(hostname)
echo ""
echo "--- To view TensorBoard, open a NEW terminal on YOUR LOCAL machine ---"
echo "--- and run the following command to create an SSH tunnel: ---"
echo ""
echo "ssh -N -L localhost:$PORT:$LOGIN_NODE_HOSTNAME:$PORT $USER@$LOGIN_NODE_HOSTNAME"
echo ""
echo "Once the tunnel is running, open your local web browser and go to:"
echo "http://localhost:$PORT"
echo ""
echo "---------------------------------------------------------------------"
echo ""

# --- Launch TensorBoard ---
echo "Starting TensorBoard... (Press Ctrl+C to stop)"

# Activate conda environment
source /home/yhan/miniconda3/etc/profile.d/conda.sh
conda activate genome

# Launch TensorBoard
tensorboard --logdir "$LATEST_LOG_DIR" --port "$PORT" --bind_all 