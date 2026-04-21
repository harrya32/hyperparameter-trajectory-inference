# Hyperparameter Trajectory Inference (HTI)

Code and experiment assets for [**Hyperparameter Trajectory Inference with Conditional Lagrangian Optimal Transport**](https://openreview.net/pdf?id=P5B97gZwRb), by Harry Amad and Mihaela van der Schaar (ICLR 2026).

This repository contains:
- Conditional Lagrangian optimal transport-based HTI surrogate training code.
- Data generation + evaluation pipelines for experiments in the paper.
- Reproducibility scripts that train different CLOT HTI methods across seeds.

The CLOT implementation extends on the code from [Pooladian et al. (2024)](https://github.com/facebookresearch/lagrangian-ot).

## Repository Structure

```text
HTI/
├── NLOT/                        # Core HTI training code
│   ├── lagrangian_ot/           # CLOT surrogate implementation
│   ├── data/                    # HTI training datasets (.pt)
│   ├── eval_data/               # Optional marginal eval datasets
│   ├── train.py                 # Main HTI training entrypoint
│   └── train.yaml               # Default training config
├── hti_scripts/                 # Multi-seed training scripts used for paper experiments
├── DTR-Bench/                   # Cancer and Reacher RL policy training and surrogate evaluation scripts
├── quantile_regression/         # ETT quantile forecasting training + HTI data generation
├── generative_dropout/          # Diffusion model with dropout training + HTI data generation
└── README.md
```

## Environment Setup

Create and activate a Python environment:

```bash
conda create -n hti_env python=3.10 -y
conda activate hti_env
```

Install HTI core dependencies:

```bash
pip install -r NLOT/requirements.txt
```

`NLOT/requirements.txt` pins `jax[cuda12]`. If your machine does not use CUDA 12,
install a compatible CPU/GPU JAX build first, then install the remaining requirements.

Install experiment-specific dependencies as needed:

```bash
pip install -r DTR-Bench/requirements.txt
pip install -r quantile_regression/requirements.txt
pip install -r generative_dropout/requirements.txt
```

For Reacher experiments, install MuJoCo support:

```bash
pip install "gymnasium[mujoco]"
```

## Train HTI on Your Own Data

### 1. Prepare data tensor

HTI expects a `.pt` tensor with shape:

```text
[num_timepoints, num_samples_per_timepoint, D + C]
```

Interpretation:
- First `D` columns: ambient dimensions OT operates in.
- Last `C` columns: conditioning variables.
- If `categorical=True`, the first conditioning column (`x[:, D]`) is treated as integer class id in `[0, num_categories - 1]`.

### 2. Run training

Use `dataset_path` for custom data, e.g.

```bash
python NLOT/train.py \
  dataset='my_dataset' \
  dataset_path='/absolute/path/to/my_dataset.pt' \
  D=2 C=3 categorical=False \
  geometry='neural_net_metric_eig' \
  include_inverse_potential=True \
  wandb=False
```

Notes:
- For custom datasets, plotting bounds are auto-inferred from first 2 ambient dims.
- You can override bounds explicitly:

```bash
python NLOT/train.py dataset='my_dataset' dataset_path='/path/my_dataset.pt' plot_bounds=[-3,3,-2,2]
```

- To train on a subset of transitions, use `num_pairs=<k>`.

### 3. Outputs

Each run writes to Hydra output dir:

```text
exp/local/<YYYY.MM.DD>/<HHMM>.<geometry>/
```

Checkpoints:
- `latest.pkl` in run dir.
- Optional collection copy in `collect_save_dir` (used by reproducibility scripts).

## Reproducing Paper Experiments

All provided `hti_scripts/*.sh` run seeds `0..19` and save checkpoints directly to:

```text
NLOT/surrogate_models/<experiment>/<method>/
```

### 1) Semicircles

Data is already included in `NLOT/data/conditional_semicircles.pt`. Optional regeneration:

```bash
cd NLOT
python generate_synth_data.py
cd ..
```

Train all HTI variants:

```bash
./hti_scripts/semicircles.sh
```

Evaluation in terms of NLL/C.D. can be found in wandb logs.

### 2) Cancer Reward Weighting (linear)

Optional: retrain RL agents and regenerate HTI dataset:

```bash
cd DTR-Bench
./train_ppo_agents.sh
python reward_agents_data_gen.py
cd ..
```

This writes HTI training data to `NLOT/data/reward_weighting_data_0_10.pt`.

Train HTI surrogates:

```bash
./hti_scripts/reward_weighting.sh
```

Evaluate in DTR environment:

```bash
cd DTR-Bench
./run_surrogate_eval.sh
cd ..
```

Results are written under:

```text
DTR-Bench/surrogate_plots_reward_weighting/<method>/
```

### 3) Cancer Reward Weighting (hinge)

Optional: retrain RL agents and regenerate HTI dataset:

```bash
cd DTR-Bench
./train_ppo_reward_weighting_hinge.sh
python reward_weighting_hinge_data_gen.py
cd ..
```

Train HTI surrogates:

```bash
./hti_scripts/reward_weighting_hinge.sh
```

Evaluate:

```bash
cd DTR-Bench
./run_hinge_surrogate_eval.sh
cd ..
```

Results are written under:

```text
DTR-Bench/surrogate_plots_hinge/<method>/
```

### 4) Reacher Reward Weighting

Dataset is included as `NLOT/data/reacher_data.pt`.

Optional regeneration from saved Reacher policies:

```bash
cd DTR-Bench
python reacher_data_gen.py
cd ..
```

This writes HTI training data to `NLOT/data/reacher_data.pt`

Train surrogates:

```bash
./hti_scripts/reacher.sh
```

Evaluate:

```bash
cd DTR-Bench
./run_reacher_surrogate_eval.sh
cd ..
```

Results are written under:

```text
DTR-Bench/surrogate_plots_reacher/<method>/
```

### 5) Quantile Regression

Dataset is included as `NLOT/data/quantile_data_new.pt`.

Optionally train NN quantile forecasters and generate HTI training data:

```bash
cd quantile_regression
./train_forecasters.sh
./generate_hti_data.sh
cd ..
```

This writes HTI training data to `NLOT/data/quantile_data_new.pt`

Train surrogates:

```bash
./hti_scripts/ett_quantiles.sh
```

Evaluate:

```bash
cd DTR-Bench
./run_surrogate_ett_quantile_eval.sh
cd ..
```

Results are written under:

```text
DTR-Bench/surrogate_plots_ett_quantile/<method>/
```

### 6) Generative Dropout

Optional HTI training data regeneration:

```bash
cd generative_dropout
python generate_2moons_dropout_data.py
cd ..
```

This writes HTI training data to `NLOT/data/diffusion_2moons_dropout.pt`

Train surrogates:

```bash
./hti_scripts/generative_dropout.sh
```

Evaluation in terms of W.D. can be found in wandb logs.

## Citation

```bibtex
@inproceedings{
  amad2026hyperparameter,
  title={Hyperparameter Trajectory Inference with Conditional Lagrangian Optimal Transport},
  author={Harry Amad and Mihaela van der Schaar},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://openreview.net/pdf?id=P5B97gZwRb}
}
```
