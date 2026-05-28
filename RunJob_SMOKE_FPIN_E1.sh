#!/bin/bash
#SBATCH --job-name=SMOKE_FPIN_E1
#SBATCH --output=/home/thyssens/fpin/cluster_logs/SMOKE_FPIN_E1_%j.log
#SBATCH --error=/home/thyssens/fpin/cluster_logs/SMOKE_FPIN_E1_%j.err
#SBATCH --ntasks=1
#SBATCH --partition=NGPU,GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --export=PYTHONIOENCODING=UTF-8

set -eo pipefail
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate l2o_py310
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
REPO=/home/thyssens/fpin
cd "$REPO"; mkdir -p cluster_logs

python -u run_fpin.py \
  meta=train_base env=cvrp50_unf model=fpin hydra.job.chdir=true \
  model_cfg.model_args.max_fleet_length=7 model_cfg.model_args.fleet_in_dim=260 \
  model_cfg.model_args.use_attn=True model_cfg.model_args.vehicle_cond_edge_head=True \
  train_cfg.n_epochs=1 train_cfg.batch_size=32 train_cfg.epoch_size=256 \
  train_cfg.nr_train_samples=2000 train_cfg.no_tensorboard=true \
  train_cfg.run_name=fpin_e1_SMOKE \
  fixed_train_set=/fs-home/data/thyssens/ra_data/train_data/cvrp/uniform/targets/converted_npz/
