#!/bin/bash
# F-PIN-S retrain @ N=20, M=3: replaces per-vehicle row softmax with
# per-customer JOINT (vehicle, next-node) softmax (depot row unchanged).
# Mirrors PIM 2022 softassign structural prior in ONE softmax step instead
# of iterative Sinkhorn -> single LogSumExp gradient, no doubly-stochastic
# fixed-point pathology (cf. Mena et al. 2018; Cuturi 2013 OT).
# E3 (sinkhorn_assignment) is DISABLED for this run.
#
# Falsifiable claim: this beats published F-PIN row (Cost_v 120.09 / Viol 27.1%
# on VRP20 M=3) and ideally <= PIM 2022 published 135.5.
#SBATCH --job-name=GWDG_FPIN_S_N20_M3
#SBATCH --partition=grete:shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -G A100:1
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=thyssens@ismll.de
#SBATCH --output=/projects/extern/nhr/nhr_ni/nim00026/dir.project/daniela/cluster_logs/GWDG_FPIN_S_N20_M3_%j.log
#SBATCH --error=/projects/extern/nhr/nhr_ni/nim00026/dir.project/daniela/cluster_logs/GWDG_FPIN_S_N20_M3_%j.err

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

RUN_NAME="fpin_s_n20_k3_unf_attn"
RESUME_GLOB="$FPIN_OUTPUT_ROOT/cvrp_20_uniform/train/fpin/${RUN_NAME}_*/checkpoints/best.pt"
LATEST_CKPT=$(ls -t $RESUME_GLOB 2>/dev/null | head -1 || true)
RUN_DIR="$FPIN_OUTPUT_ROOT/cvrp_20_uniform/train/fpin/${RUN_NAME}_$(date +%Y-%m-%d_%H-%M-%S)"

COMMON_ARGS=( \
  env=cvrp20_unf model=fpin hydra.job.chdir=true \
  "hydra.run.dir=${RUN_DIR}" \
  model_cfg.model_args.max_fleet_length=3 model_cfg.model_args.fleet_in_dim=260 \
  model_cfg.model_args.use_attn=True model_cfg.model_args.vehicle_cond_edge_head=True \
  model_cfg.model_args.sinkhorn_assignment=False \
  model_cfg.model_args.joint_customer_norm=True \
  train_cfg.batch_size=64 train_cfg.n_epochs=200 train_cfg.lr=0.0001 train_cfg.checkpoint_epochs=10 \
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
