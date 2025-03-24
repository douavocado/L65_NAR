#!/bin/bash

# Script to train and evaluate models for all configs in interp/configs/model_param_gnn
# Set up error handling
set -e

# Add current directory to Python path
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Directory containing model configurations
CONFIG_DIR="interp/configs/layers_gnn"
# Device to use (cuda or cpu)
DEVICE="cpu"
# Dataset to use
DATASET="all"
# Algorithm to use
ALGO="bellman_ford"

echo "Starting training and evaluation for all configs in $CONFIG_DIR"
mkdir -p logs

# Loop through all YAML files in the config directory
for CONFIG_FILE in $CONFIG_DIR/*.yaml; do
    # Extract the filename without path and extension for logging
    CONFIG_BASENAME=$(basename $CONFIG_FILE .yaml)
    
    echo "========================================================"
    echo "Processing config: $CONFIG_BASENAME"
    echo "========================================================"
    echo "Current working directory: $(pwd)"
    
    # Extract model_name from the config file
    MODEL_NAME=$(grep "model_name:" $CONFIG_FILE | awk '{print $2}' | tr -d '"')
    if [ -z "$MODEL_NAME" ]; then
        echo "Error: Could not extract model_name from $CONFIG_FILE"
        continue
    fi
    
    # Log files
    TRAIN_LOG="logs/${CONFIG_BASENAME}_train.log"
    EVAL_LOG="logs/${CONFIG_BASENAME}_eval.log"
    
    echo "Training model with config $CONFIG_FILE (model_name: $MODEL_NAME)"
    # Train the model
    python interp/train.py \
        --config $CONFIG_FILE \
        --dataset $DATASET \
        --algo $ALGO \
        --device $DEVICE \
        --num_epochs 150 \
        2>&1 | tee $TRAIN_LOG
    
    # Get the model directory path based on model_name and dataset
    MODEL_DIR="interp_checkpoints/${ALGO}/${MODEL_NAME}_${DATASET}"
    
    echo "Evaluating model in $MODEL_DIR"
    # Evaluate the model
    python interp/eval.py \
        --model_dir $MODEL_DIR \
        --algorithm $ALGO \
        --device $DEVICE \
        2>&1 | tee $EVAL_LOG
    
    echo "Completed training and evaluation for $CONFIG_BASENAME"
    echo ""
done

echo "All training and evaluation complete!"
