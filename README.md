# F-PIN: Fleet-Aware Permutation-Invariant Network for Vehicle Routing

This repository contains the implementation of **F-PIN**, a fleet-aware permutation-invariant neural architecture for solving fleet-constrained vehicle routing problems (FC-CVRP).

The model learns a fleet-conditioned edge representation ("heatmap") which can be combined with different decoding strategies to construct feasible routes.

---

## Overview

F-PIN explicitly models the interaction between:
- customers
- depot
- fleet (vehicles)

using a permutation-invariant set-based architecture with attention mechanisms.

The model predicts edge scores conditioned on the available fleet size and can be paired with:
- vehicle assignment decoding
- giant-tour + split decoding
- optional local search post-processing

---

## Installation

### 1. Clone repository

```bash
git clone <your-repo-url>
cd fpin
```

### 2. Create conda environment

```bash
conda env create -f environment.yml
conda activate l2o_py310
``` 

## Running Evaluation (Sanity Check)

```bash
python run_fpin.py env=cvrp60_unf test_cfg.time_limit=8 test_cfg.dataset_size=10 data_file_path=data/test_data/cvrp/uniform/cvrp60/fc-cvrp_k6_seed213298_size130.pt checkpoint_load_path=fpin/logs/ckpts/uniform_60/best-ep-n60_k6_unf_attn.pt test_cfg.add_ls=True model_cfg.model_args.max_fleet_length=6 eval_opts_cfg.nr_vehicles_eval=6
```

## Running Evaluation (Reproducing Results)

```bash
python run_fpin.py \
  env=cvrp20_unf \
  test_cfg.time_limit=8 \
  model_cfg.model_args.max_fleet_length=3 \
  eval_opts_cfg.nr_vehicles_eval=3 \
  test_cfg.dataset_size=1000 \
  data_file_path=data/test_data/cvrp/uniform/cvrp20/fc-cvrp_k3_seed213298_size1000.pt \
  checkpoint_load_path=fpin/logs/ckpts/uniform_20/best-ep-n20_k3_unf_attn.pt \
  test_cfg.add_ls=True
``` 

## Citation

##