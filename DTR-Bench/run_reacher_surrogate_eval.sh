#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR"

ITERATIONS=5
SCRIPT="surrogate_eval_reacher.py"
METHODS=("eucl_no_potential" "eucl_w_potential" "learned_no_potential" "learned_w_potential")

for method in "${METHODS[@]}"; do
  for i in $(seq 1 "$ITERATIONS"); do
    echo "Running iteration $i with run name: $method..."
    python3 "$SCRIPT" --name "$method" --iter "$i" --all_lambdas
  done
done

echo "All $ITERATIONS iterations completed for ${#METHODS[@]} methods."
