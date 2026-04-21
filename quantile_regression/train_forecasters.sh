#!/usr/bin/env bash
set -euo pipefail
QUANTILES="0.01 0.1 0.25 0.5 0.75 0.9 0.99"
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
TRAIN_SCRIPT_PATH="$SCRIPT_DIR/train_quantile_forecaster.py"

echo "Starting the training process for all expert models..."
echo "======================================================"

# Check if the training script exists
if [ ! -f "$TRAIN_SCRIPT_PATH" ]; then
    echo "Error: Training script not found at $TRAIN_SCRIPT_PATH"
    exit 1
fi

# Loop through each quantile in the list
for q in $QUANTILES
do
  echo ""
  echo "------------------------------------------------------"
  echo ">>> Training model for quantile: $q"
  echo "------------------------------------------------------"
  
  # Call the Python training script with the current quantile
  python "$TRAIN_SCRIPT_PATH" --quantile $q
  
  # Check if the python script ran successfully
  if [ $? -ne 0 ]; then
    echo "Error: Training failed for quantile $q. Aborting."
    exit 1
  fi
done

echo ""
echo "======================================================"
echo "All expert and oracle models have been trained successfully."
echo "======================================================"
