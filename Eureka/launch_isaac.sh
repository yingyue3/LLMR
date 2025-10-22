#!/bin/bash

# Isaac Sim Launch Script for Headless Operation
# This script configures the environment for running Isaac Sim without display

echo "Setting up Isaac Sim for headless operation..."

# Load required modules
module load StdEnv/2023
module load python/3.10
module load gcc

# Set environment variables for headless operation
export DISPLAY=:99
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export OMNI_DISABLE_SYNTHETIC_RECORDER=1
export OMNI_DISABLE_GPU=1

# Isaac Sim path
ISAAC_SIM_PATH="/lustre04/scratch/yingyue/isaacv/bin/isaacsim"

# Check if Isaac Sim exists
if [ ! -f "$ISAAC_SIM_PATH" ]; then
    echo "Error: Isaac Sim not found at $ISAAC_SIM_PATH"
    exit 1
fi

echo "Launching Isaac Sim in headless mode..."
echo "Isaac Sim path: $ISAAC_SIM_PATH"

# Launch Isaac Sim with headless flags
$ISAAC_SIM_PATH \
    --headless \
    --disable-gpu \
    --disable-window \
    --no-window \
    --ext-folder /tmp/isaac_extensions \
    --enable-cameras \
    --enable-remote-commands \
    --enable-remote-commands-port 8211 \
    "$@"

echo "Isaac Sim launched successfully!"

