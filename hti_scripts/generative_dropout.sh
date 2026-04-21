#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
REPO_ROOT=$( cd -- "${SCRIPT_DIR}/.." &> /dev/null && pwd )
cd "$REPO_ROOT"

seeds=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19)

for seed in "${seeds[@]}"; do
  python NLOT/train.py dataset='2moons_dropout' geometry='neural_net_metric_eig' include_inverse_potential=True bandwidth=0.2 conditional_bandwidth=1.0 lambda=0.01 wandb_project="2moons_learned_w_potential" ctransform_solver.max_iter=10 D=2 C=1 categorical=True num_categories=2 target_potential_dim_hidden=[64,64,64,64] source_map_dim_hidden=[64,64,64,64] collect_save_dir="$REPO_ROOT/NLOT/surrogate_models/generative_dropout/learned_w_potential" seed=$seed

  python NLOT/train.py dataset='2moons_dropout' geometry='sq_euclidean_manifold' include_inverse_potential=False wandb_project="2moons_eucl_no_potential" ctransform_solver.max_iter=10 D=2 C=1 categorical=True num_categories=2 target_potential_dim_hidden=[64,64,64,64] source_map_dim_hidden=[64,64,64,64] collect_save_dir="$REPO_ROOT/NLOT/surrogate_models/generative_dropout/eucl_no_potential" seed=$seed

  python NLOT/train.py dataset='2moons_dropout' geometry='sq_euclidean_manifold' include_inverse_potential=True bandwidth=0.2 conditional_bandwidth=1.0 lambda=0.01 wandb_project="2moons_eucl_w_potential" ctransform_solver.max_iter=10 D=2 C=1 categorical=True num_categories=2 target_potential_dim_hidden=[64,64,64,64] source_map_dim_hidden=[64,64,64,64] collect_save_dir="$REPO_ROOT/NLOT/surrogate_models/generative_dropout/eucl_w_potential" seed=$seed

  python NLOT/train.py dataset='2moons_dropout' geometry='neural_net_metric_eig' include_inverse_potential=False wandb_project="2moons_learned_no_potential" ctransform_solver.max_iter=10 D=2 C=1 categorical=True num_categories=2 target_potential_dim_hidden=[64,64,64,64] source_map_dim_hidden=[64,64,64,64] collect_save_dir="$REPO_ROOT/NLOT/surrogate_models/generative_dropout/learned_no_potential" seed=$seed

  python NLOT/train.py dataset='2moons_dropout' geometry='neural_net_metric' include_inverse_potential=False wandb_project="2moons_pooladian" ctransform_solver.max_iter=10 D=2 C=1 categorical=True num_categories=2 target_potential_dim_hidden=[64,64,64,64] source_map_dim_hidden=[64,64,64,64] collect_save_dir="$REPO_ROOT/NLOT/surrogate_models/generative_dropout/pooladian" seed=$seed

  python NLOT/train.py dataset='2moons_dropout' geometry='neural_net_metric' include_inverse_potential=True bandwidth=0.2 conditional_bandwidth=1.0 lambda=0.01 wandb_project="2moons_nlot_metric" ctransform_solver.max_iter=10 D=2 C=1 categorical=True num_categories=2 target_potential_dim_hidden=[64,64,64,64] source_map_dim_hidden=[64,64,64,64] collect_save_dir="$REPO_ROOT/NLOT/surrogate_models/generative_dropout/nlot_metric" seed=$seed
done
