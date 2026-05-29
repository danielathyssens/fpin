#!/bin/bash
# Evaluate a trained F-PIN checkpoint with the as-trained transition decoder.
# Edit the 4 vars under ">>> EDIT PER CELL <<<"; the data path is built from N and M.
#SBATCH --job-name=EVAL_FPIN
#SBATCH --output=/home/thyssens/fpin/cluster_logs/EVAL_FPIN_%j.log
#SBATCH --error=/home/thyssens/fpin/cluster_logs/EVAL_FPIN_%j.err
#SBATCH --ntasks=1
#SBATCH --partition=NGPU,GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --export=ALL

set -eo pipefail
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate l2o_py310
set -u
export PYTHONIOENCODING=UTF-8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO=/home/thyssens/fpin
cd "$REPO"; mkdir -p cluster_logs

# >>> EDIT PER CELL <<<
N=${N:-20}              # graph size: 20 | 50 | 100
M=${M:-3}               # fleet size k
ADD_LS=${ADD_LS:-False} # False=raw ; True=+OR-Tools post-process
DECODE=${DECODE:-assign_transition}   # assign_transition | assign_plain | assign_faithful
LSBATCH=${LSBATCH:-8}   # OR-Tools post-process parallelism (set ~= cpus-per-task)
CKPT=${CKPT:?set CKPT=.../checkpoints/best.pt}

DATA=data/test_data/cvrp/uniform/cvrp${N}/fc-cvrp_k${M}_seed213298_size1000.pt

python -u run_fpin.py \
  env=cvrp${N}_unf model=fpin cuda=True \
  test_cfg.dataset_size=1000 test_cfg.time_limit=8 test_cfg.add_ls=${ADD_LS} \
  model_cfg.model_args.fleet_in_dim=260 model_cfg.model_args.max_fleet_length=${M} \
  model_cfg.model_args.use_attn=True model_cfg.model_args.vehicle_cond_edge_head=True \
  eval_opts_cfg.nr_vehicles_eval=${M} \
  eval_opts_cfg.decode_v_assign_type=${DECODE} \
  test_cfg.ls_policy_cfg.batch_size=${LSBATCH} \
  data_file_path=${DATA} \
  checkpoint_load_path=${CKPT}
