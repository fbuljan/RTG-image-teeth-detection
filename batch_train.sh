#!/bin/bash
# Batch training script for YOLO on Mac
# Runs training in chunks to manage memory by killing Python between batches
#
# How it works:
# - Each batch trains for BATCH_EPOCHS epochs starting from the current weights
# - After each batch, Python exits and memory is freed
# - Next batch loads last.pt and trains for another BATCH_EPOCHS
# - Tracks total epochs via a counter file

set -e

CONFIG_PATH=$1
TOTAL_EPOCHS=${2:-50}
BATCH_EPOCHS=${3:-5}

if [ -z "$CONFIG_PATH" ]; then
    echo "Usage: ./batch_train.sh <config_path> [total_epochs] [batch_epochs]"
    echo ""
    echo "Arguments:"
    echo "  config_path   - Path to YOLO config YAML file (required)"
    echo "  total_epochs  - Total epochs to train (default: 50)"
    echo "  batch_epochs  - Epochs per batch before restart (default: 5)"
    echo ""
    echo "Example:"
    echo "  ./batch_train.sh configs/yolo/seg-enhanced-1.yaml 50 5"
    exit 1
fi

# Extract project and name from config file
PROJECT=$(grep -E "^project:" "$CONFIG_PATH" | awk '{print $2}' | tr -d '"' | tr -d "'")
NAME=$(grep -E "^name:" "$CONFIG_PATH" | awk '{print $2}' | tr -d '"' | tr -d "'")

# Defaults if not found in config
PROJECT=${PROJECT:-"runs-segmentation"}
NAME=${NAME:-"experiment"}

WEIGHTS_DIR="$PROJECT/$NAME/weights"
EPOCH_TRACKER="$PROJECT/$NAME/batch_epoch_count.txt"

echo "=========================================="
echo "       Batch YOLO Training for Mac"
echo "=========================================="
echo "Config:         $CONFIG_PATH"
echo "Total epochs:   $TOTAL_EPOCHS"
echo "Batch epochs:   $BATCH_EPOCHS"
echo "Output:         $PROJECT/$NAME"
echo "Weights dir:    $WEIGHTS_DIR"
echo "=========================================="
echo ""

# Initialize or read epoch counter
if [ -f "$EPOCH_TRACKER" ]; then
    current_epoch=$(cat "$EPOCH_TRACKER")
    echo "Resuming from epoch $current_epoch (found $EPOCH_TRACKER)"
else
    current_epoch=0
fi

batch=1

while [ $current_epoch -lt $TOTAL_EPOCHS ]; do
    # Calculate epochs for this batch
    remaining=$((TOTAL_EPOCHS - current_epoch))
    if [ $remaining -lt $BATCH_EPOCHS ]; then
        epochs_this_batch=$remaining
    else
        epochs_this_batch=$BATCH_EPOCHS
    fi

    echo ""
    echo "=== Batch $batch: training $epochs_this_batch epochs (total: $current_epoch -> $((current_epoch + epochs_this_batch))) ==="
    echo ""

    if [ -f "$WEIGHTS_DIR/last.pt" ] && [ $current_epoch -gt 0 ]; then
        # Continue from checkpoint - load weights and train for more epochs
        echo "Loading weights from $WEIGHTS_DIR/last.pt"
        python train_yolo.py \
            --config "$CONFIG_PATH" \
            --epochs $epochs_this_batch \
            --resume "$WEIGHTS_DIR/last.pt"
    else
        # First run - start fresh
        echo "Starting fresh training"
        python train_yolo.py \
            --config "$CONFIG_PATH" \
            --epochs $epochs_this_batch
    fi

    exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo ""
        echo "Training failed with exit code $exit_code"
        exit $exit_code
    fi

    # Update epoch counter
    current_epoch=$((current_epoch + epochs_this_batch))

    # Save progress to file
    mkdir -p "$PROJECT/$NAME"
    echo $current_epoch > "$EPOCH_TRACKER"

    echo ""
    echo "Progress: $current_epoch / $TOTAL_EPOCHS epochs completed"

    if [ $current_epoch -ge $TOTAL_EPOCHS ]; then
        echo ""
        echo "=========================================="
        echo "       Training Complete!"
        echo "=========================================="
        echo "Total epochs trained: $current_epoch"
        echo "Final model: $WEIGHTS_DIR/best.pt"
        echo "=========================================="
        break
    fi

    echo ""
    echo "Freeing memory before next batch..."
    sleep 5

    batch=$((batch + 1))
done
