#!/bin/bash
#SBATCH --job-name=RETRAIN_FPIN_E1_N100_M10
#SBATCH --output=/home/thyssens/fpin/cluster_logs/RETRAIN_FPIN_E1_N100_M10_%j.log
#SBATCH --error=/home/thyssens/fpin/cluster_logs/RETRAIN_FPIN_E1_N100_M10_%j.err
#SBATCH --ntasks=1
#SBATCH --partition=NGPU,GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=thyssens@ismll.de
#SBATCH --export=PYTHONIOENCODING=UTF-8

set -eo pipefail
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate l2o_py310
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO=/home/thyssens/fpin
cd "$REPO"; mkdir -p cluster_logs

python -u run_fpin.py \
  meta=train_base env=cvrp100_unf model=fpin hydra.job.chdir=true \
  model_cfg.model_args.max_fleet_length=10 \
  model_cfg.model_args.fleet_in_dim=260 \
  model_cfg.model_args.use_attn=True \
  model_cfg.model_args.vehicle_cond_edge_head=True \
  train_cfg.batch_size=64 \
  train_cfg.n_epochs=100 \
  train_cfg.lr=0.0001 \
  train_cfg.checkpoint_epochs=10 \
  train_cfg.run_name=fpin_e1_n100_k10_unf_attn \
  fixed_train_set=/fs-home/data/thyssens/ra_data/train_data/cvrp/uniform/targets/fc_hgs_clean/n100_k10/
