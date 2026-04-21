#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
REPO_ROOT=$( cd -- "${SCRIPT_DIR}/.." &> /dev/null && pwd )
cd "$REPO_ROOT"

seeds=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19)

for seed in "${seeds[@]}"; do
  python NLOT/train.py dataset='quantile_data' geometry='neural_net_metric_eig' num_train_iters=1001 include_inverse_potential=False wandb_project="quantile_learned" ctransform_solver.max_iter=10 D=3 C=12 categorical=False collect_save_dir="$REPO_ROOT/NLOT/surrogate_models/ett_quantile/learned_no_potential" seed=$seed
  python NLOT/train.py dataset='quantile_data' geometry='neural_net_metric_eig' num_train_iters=1001 include_inverse_potential=True bandwidth=1.0 conditional_bandwidth=1.0 lambda=0.01 wandb_project="quantile_learned_w_potential" ctransform_solver.max_iter=10 D=3 C=12 categorical=False collect_save_dir="$REPO_ROOT/NLOT/surrogate_models/ett_quantile/learned_w_potential" seed=$seed
  python NLOT/train.py dataset='quantile_data' geometry='sq_euclidean_manifold' num_train_iters=1001 include_inverse_potential=False wandb_project="quantile_eucl" ctransform_solver.max_iter=10 D=3 C=12 categorical=False collect_save_dir="$REPO_ROOT/NLOT/surrogate_models/ett_quantile/eucl_no_potential" seed=$seed
  python NLOT/train.py dataset='quantile_data' geometry='sq_euclidean_manifold' num_train_iters=1001 include_inverse_potential=True bandwidth=1.0 conditional_bandwidth=1.0 lambda=0.01 wandb_project="quantile_eucl_w_potential" ctransform_solver.max_iter=10 D=3 C=12 categorical=False collect_save_dir="$REPO_ROOT/NLOT/surrogate_models/ett_quantile/eucl_w_potential" seed=$seed
done
