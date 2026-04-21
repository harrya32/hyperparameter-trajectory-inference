#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
REPO_ROOT=$( cd -- "${SCRIPT_DIR}/.." &> /dev/null && pwd )
cd "$REPO_ROOT"

seeds=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19)
for seed in "${seeds[@]}"; do
  python NLOT/train.py dataset='reward_weighting_data' geometry='neural_net_metric_eig' include_inverse_potential=True bandwidth=1.0 conditional_bandwidth=1.0 lambda=0.01 wandb_project="reward_weighting_learned_w_potential" ctransform_solver.max_iter=3 D=2 C=7 categorical=False target_potential_dim_hidden=[64,64,64,64] source_map_dim_hidden=[64,64,64,64] collect_save_dir="$REPO_ROOT/NLOT/surrogate_models/reward_weighting/learned_w_potential" seed=$seed
  python NLOT/train.py dataset='reward_weighting_data' geometry='neural_net_metric' include_inverse_potential=False wandb_project="reward_weighting_pooladian" ctransform_solver.max_iter=3 D=2 C=7 categorical=False target_potential_dim_hidden=[64,64,64,64] source_map_dim_hidden=[64,64,64,64] collect_save_dir="$REPO_ROOT/NLOT/surrogate_models/reward_weighting/pooladian" seed=$seed
  python NLOT/train.py dataset='reward_weighting_data' geometry='neural_net_metric' include_inverse_potential=True bandwidth=1.0 conditional_bandwidth=1.0 lambda=0.01 wandb_project="reward_weighting_nlot_metric" ctransform_solver.max_iter=3 D=2 C=7 categorical=False target_potential_dim_hidden=[64,64,64,64] source_map_dim_hidden=[64,64,64,64] collect_save_dir="$REPO_ROOT/NLOT/surrogate_models/reward_weighting/nlot_metric" seed=$seed
  python NLOT/train.py dataset='reward_weighting_data' geometry='neural_net_metric_eig' include_inverse_potential=False wandb_project="reward_weighting_learned" ctransform_solver.max_iter=3 D=2 C=7 categorical=False target_potential_dim_hidden=[64,64,64,64] source_map_dim_hidden=[64,64,64,64] collect_save_dir="$REPO_ROOT/NLOT/surrogate_models/reward_weighting/learned_no_potential" seed=$seed
  python NLOT/train.py dataset='reward_weighting_data' geometry='sq_euclidean_manifold' include_inverse_potential=False wandb_project="reward_weighting_eucl" ctransform_solver.max_iter=3 D=2 C=7 categorical=False target_potential_dim_hidden=[64,64,64,64] source_map_dim_hidden=[64,64,64,64] collect_save_dir="$REPO_ROOT/NLOT/surrogate_models/reward_weighting/eucl_no_potential" seed=$seed
  python NLOT/train.py dataset='reward_weighting_data' geometry='sq_euclidean_manifold' include_inverse_potential=True bandwidth=1.0 conditional_bandwidth=1.0 lambda=0.01 wandb_project="reward_weighting_eucl_w_potential" ctransform_solver.max_iter=3 D=2 C=7 categorical=False target_potential_dim_hidden=[64,64,64,64] source_map_dim_hidden=[64,64,64,64] collect_save_dir="$REPO_ROOT/NLOT/surrogate_models/reward_weighting/eucl_w_potential" seed=$seed
done
