#!/bin/bash
# Retrain F-PIN with the corrected/aligned decode + E1 (vehicle-conditioned edge head) + E2
# (learnable temperature). Template for VRP50 M=7 — to make other cells, change:
#   --job-name, env=cvrpNN_unf, model_cfg.model_args.max_fleet_length=M,
#   train_cfg.run_name, and (if per-size) fixed_train_set.
#SBATCH --job-name=FPIN_E1_N50_M7
#SBATCH --output=/home/thyssens/fpin/cluster_logs/FPIN_E1_N50_M7_%j.log
#SBATCH --error=/home/thyssens/fpin/cluster_logs/FPIN_E1_N50_M7_%j.err
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

# >>> adjust REPO to where you rsync'd the repo on the cluster <<<
REPO=/home/thyssens/fpin
cd "$REPO"
mkdir -p cluster_logs

python -u run_fpin.py \
  meta=train_base env=cvrp50_unf model=fpin hydra.job.chdir=true \
  model_cfg.model_args.max_fleet_length=7 \
  model_cfg.model_args.fleet_in_dim=260 \
  model_cfg.model_args.use_attn=True \
  model_cfg.model_args.vehicle_cond_edge_head=True \
  train_cfg.batch_size=64 \
  train_cfg.n_epochs=100 \
  train_cfg.lr=0.0001 \
  train_cfg.checkpoint_epochs=10 \
  train_cfg.run_name=fpin_e1_n50_k7_unf_attn \
  fixed_train_set=/fs-home/data/thyssens/ra_data/train_data/cvrp/uniform/targets/converted_npz/
