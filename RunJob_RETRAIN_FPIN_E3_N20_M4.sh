#!/bin/bash
#SBATCH --job-name=RETRAIN_FPIN_E3_N20_M4
#SBATCH --output=cluster_logs/RETRAIN_FPIN_E3_N20_M4_%j.log
#SBATCH --error=cluster_logs/RETRAIN_FPIN_E3_N20_M4_%j.err
#SBATCH --ntasks=1
#SBATCH --partition=NGPU,GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --export=PYTHONIOENCODING=UTF-8

set -eo pipefail
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate l2o_py310
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${REPO:-$HOME/fpin}"; mkdir -p cluster_logs

python -u run_fpin.py \
  meta=train_base env=cvrp20_unf model=fpin hydra.job.chdir=true \
  model_cfg.model_args.max_fleet_length=4 model_cfg.model_args.fleet_in_dim=260 \
  model_cfg.model_args.use_attn=True model_cfg.model_args.vehicle_cond_edge_head=True \
  model_cfg.model_args.sinkhorn_assignment=True model_cfg.model_args.sinkhorn_iters=3 \
  train_cfg.batch_size=64 train_cfg.n_epochs=100 train_cfg.lr=0.0001 train_cfg.checkpoint_epochs=10 \
  train_cfg.run_name=fpin_e3_n20_k4_unf_attn \
  fixed_train_set=/fs-home/data/thyssens/ra_data/train_data/cvrp/uniform/targets/fc_hgs_clean/n20_k4/
