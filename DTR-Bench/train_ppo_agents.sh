#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR"

# Script to train PPO agent with varying lambda_nk values from 0 to 10

echo "Starting reward_weighting.py runs for lambda_nk = 0 to 10..."

for lambda in {0..10}
do
    echo "Running training with lambda_nk = $lambda"
    python reward_weighting.py --lambda_nk $lambda --seed 1
    echo "Completed training for lambda_nk = $lambda"
    echo "--------------------------------------------"
done

echo "All runs completed."
