#!/bin/bash
# F-PIN-A retrain @ N=20, M=3.
# F-PIN-A = F-PIN attention encoder + PIM 2022's MTSPSoftassign head.
# Iteratively normalizes (depot per-vehicle softmax) + (customer joint over
# (m, j)) + transpose, enforcing BOTH out-flow and in-flow per-customer
# uniqueness across all vehicles. Structurally prevents the head from
# emitting heatmaps that decode to > M routes -- the lever for matching
# PIM's empirical 0% fleet violation at this cell (vs F-PIN published 27.1%).
#
# Toy v2 (this repo) shows ATTN+MTSPSoftassign decodes to 0.10% gap-vs-
# optimum at N=6 -- best of 6 configs.
#
# E3 (sinkhorn_assignment) and F-PIN-S (joint_customer_norm) are DISABLED.
# Falsifiable claim: Cost_v < 114.04 (PIM measured) and viol <= 5% on
# VRP20 M=3 -> architectural advance.
#SBATCH --job-name=GWDG_FPIN_A_N20_M3
#SBATCH --partition=grete:shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -G A100:1
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=thyssens@ismll.de
#SBATCH --output=/projects/extern/nhr/nhr_ni/nim00026/dir.project/daniela/cluster_logs/GWDG_FPIN_A_N20_M3_%j.log
#SBATCH --error=/projects/extern/nhr/nhr_ni/nim00026/dir.project/daniela/cluster_logs/GWDG_FPIN_A_N20_M3_%j.err

set -eo pipefail
PROJ=/projects/extern/nhr/nhr_ni/nim00026/dir.project/daniela
mkdir -p "$PROJ/cluster_logs"
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate l2o_py310
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FPIN_OUTPUT_ROOT="$PROJ/fpin_outputs"
mkdir -p "$FPIN_OUTPUT_ROOT"
REPO=${REPO:-${SLURM_SUBMIT_DIR:-$HOME/repos/fpin}}
cd "$REPO"

RUN_NAME="fpin_a_n20_k3_unf_attn"
RESUME_GLOB="$FPIN_OUTPUT_ROOT/cvrp_20_uniform/train/fpin/${RUN_NAME}_*/checkpoints/best.pt"
LATEST_CKPT=$(ls -t $RESUME_GLOB 2>/dev/null | head -1 || true)
RUN_DIR="$FPIN_OUTPUT_ROOT/cvrp_20_uniform/train/fpin/${RUN_NAME}_$(date +%Y-%m-%d_%H-%M-%S)"

COMMON_ARGS=( \
  env=cvrp20_unf model=fpin hydra.job.chdir=true \
  "hydra.run.dir=${RUN_DIR}" \
  model_cfg.model_args.max_fleet_length=3 model_cfg.model_args.fleet_in_dim=260 \
  model_cfg.model_args.use_attn=True model_cfg.model_args.vehicle_cond_edge_head=True \
  model_cfg.model_args.sinkhorn_assignment=False \
  model_cfg.model_args.joint_customer_norm=False \
  model_cfg.model_args.softassign_head=True \
  model_cfg.model_args.softassign_layers=3 \
  train_cfg.batch_size=64 train_cfg.n_epochs=200 train_cfg.lr=0.0001 train_cfg.checkpoint_epochs=50 \
  "train_cfg.run_name=${RUN_NAME}" \
  fixed_train_set=${PROJ}/data/train_data/cvrp/uniform/targets/fc_hgs_clean/n20_k3/ \
)

if [ -n "${LATEST_CKPT:-}" ] && [ -f "${LATEST_CKPT}" ]; then
  echo "[RESUME] $LATEST_CKPT"
  srun python run_fpin.py meta=resume \
    test_cfg.checkpoint_load_path="${LATEST_CKPT}" \
    "${COMMON_ARGS[@]}"
else
  echo "[FRESH] No prior ${RUN_NAME} ckpt; training from scratch."
  srun python run_fpin.py meta=train_base "${COMMON_ARGS[@]}"
fi
