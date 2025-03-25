#!/bin/bash

# Script to evaluate all models found in interp_checkpoints for all algorithms
# Set up error handling
set -e

# Add current directory to Python path
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Device to use (cuda or cpu)
DEVICE="cpu"

# Base directory containing model folders
CHECKPOINT_BASE="interp_checkpoints"

echo "Starting evaluation for all models in $CHECKPOINT_BASE"
mkdir -p logs

# Get list of algorithms from the subdirectories in interp_checkpoints
VALID_ALGOS=("bellman_ford" "bfs" "mst_prim" "dijkstra")
ALGORITHMS=$(ls -d "$CHECKPOINT_BASE"/*/ | xargs -n 1 basename | grep -E "$(IFS="|"; echo "${VALID_ALGOS[*]}")")

# Loop through all algorithms
for ALGO in $ALGORITHMS; do
    echo "========================================================"
    echo "Processing algorithm: $ALGO"
    echo "========================================================"
    
    # Find all model directories for this algorithm
    MODEL_DIRS=$(ls -d "$CHECKPOINT_BASE/$ALGO"/*/)
    
    # Loop through all model directories
    for MODEL_DIR in $MODEL_DIRS; do
        # Remove trailing slash
        MODEL_DIR=${MODEL_DIR%/}
        
        # Extract the model name (basename of directory)
        MODEL_NAME=$(basename "$MODEL_DIR")
        
        echo "------------------------------------------------------"
        echo "Evaluating model: $MODEL_NAME"
        echo "Model directory: $MODEL_DIR"
        echo "------------------------------------------------------"
        
        # Log file
        EVAL_LOG="logs/${ALGO}_${MODEL_NAME}_eval.log"
        
        # Evaluate the model
        echo "Running evaluation..."
        python interp/eval.py \
            --model_dir "$MODEL_DIR" \
            --algorithm "$ALGO" \
            --device "$DEVICE" \
            2>&1 | tee "$EVAL_LOG"
        
        echo "Completed evaluation for $MODEL_NAME"
        echo ""
    done
    
    echo "Completed evaluations for algorithm: $ALGO"
    echo ""
done

echo "All evaluations complete!" 