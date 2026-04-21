#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
REPO_ROOT=$( cd -- "${SCRIPT_DIR}/.." &> /dev/null && pwd )
cd "$REPO_ROOT"

seeds=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19)

for seed in "${seeds[@]}"; do
  python NLOT/train.py dataset='conditional_semicircles' geometry='neural_net_metric_eig' plotting.disable=False include_inverse_potential=True bandwidth=0.05 lambda=0.05 wandb_project="semicircles_learned_w_potential" ctransform_solver.max_iter=10 D=2 C=1 categorical=True num_categories=4 collect_save_dir="$REPO_ROOT/NLOT/surrogate_models/semicircles/learned_w_potential" seed=$seed

  python NLOT/train.py dataset='conditional_semicircles' geometry='sq_euclidean_manifold' include_inverse_potential=False wandb_project="semicircles_eucl" ctransform_solver.max_iter=10 D=2 C=1 categorical=True num_categories=4 collect_save_dir="$REPO_ROOT/NLOT/surrogate_models/semicircles/eucl_no_potential" seed=$seed

  python NLOT/train.py dataset='conditional_semicircles' geometry='sq_euclidean_manifold' plotting.disable=False include_inverse_potential=True bandwidth=0.05 lambda=0.05 wandb_project="semicircles_eucl_w_potential" ctransform_solver.max_iter=10 D=2 C=1 categorical=True num_categories=4 collect_save_dir="$REPO_ROOT/NLOT/surrogate_models/semicircles/eucl_w_potential" seed=$seed

  python NLOT/train.py dataset='conditional_semicircles' geometry='neural_net_metric_eig' include_inverse_potential=False wandb_project="semicircles_learned" ctransform_solver.max_iter=10 D=2 C=1 categorical=True num_categories=4 collect_save_dir="$REPO_ROOT/NLOT/surrogate_models/semicircles/learned_no_potential" seed=$seed

  python NLOT/train.py dataset='conditional_semicircles' geometry='neural_net_metric' include_inverse_potential=True bandwidth=0.05 lambda=0.05 wandb_project="semicircles_nlot_metric" ctransform_solver.max_iter=10 D=2 C=1 categorical=True num_categories=4 collect_save_dir="$REPO_ROOT/NLOT/surrogate_models/semicircles/nlot_metric" seed=$seed
  
  python NLOT/train.py dataset='conditional_semicircles' geometry='neural_net_metric' include_inverse_potential=False wandb_project="semicircles_pooladian" ctransform_solver.max_iter=10 D=2 C=1 categorical=True num_categories=4 collect_save_dir="$REPO_ROOT/NLOT/surrogate_models/semicircles/pooladian" seed=$seed
done
