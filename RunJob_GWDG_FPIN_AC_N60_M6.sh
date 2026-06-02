#!/bin/bash
#SBATCH --job-name=GWDG_FPIN_AC_N60_M6
#SBATCH --partition=grete:shared
#SBATCH --nodes=1 --ntasks-per-node=1 -G A100:1 --time=48:00:00 --requeue
#SBATCH --mail-type=FAIL --mail-user=thyssens@ismll.de
#SBATCH --output=/projects/extern/nhr/nhr_ni/nim00026/dir.project/daniela/cluster_logs/GWDG_FPIN_AC_N60_M6_%j.log
#SBATCH --error=/projects/extern/nhr/nhr_ni/nim00026/dir.project/daniela/cluster_logs/GWDG_FPIN_AC_N60_M6_%j.err
set -eo pipefail
mkdir -p "/projects/extern/nhr/nhr_ni/nim00026/dir.project/daniela/cluster_logs"
source "${HOME}/miniconda3/etc/profile.d/conda.sh"; conda activate l2o_py310
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FPIN_OUTPUT_ROOT="/projects/extern/nhr/nhr_ni/nim00026/dir.project/daniela/fpin_outputs"; mkdir -p "${FPIN_OUTPUT_ROOT}"
REPO=${REPO:-${SLURM_SUBMIT_DIR:-$HOME/repos/fpin}}; cd "${REPO}"
RUN_NAME="fpin_ac_n60_k6_unf_attn"
RESUME_GLOB="${FPIN_OUTPUT_ROOT}/cvrp_60_uniform/train/fpin/${RUN_NAME}_*/checkpoints/best.pt"
LATEST_CKPT=$(ls -t $RESUME_GLOB 2>/dev/null | head -1 || true)
RUN_DIR="${FPIN_OUTPUT_ROOT}/cvrp_60_uniform/train/fpin/${RUN_NAME}_$(date +%Y-%m-%d_%H-%M-%S)"
COMMON=( env=cvrp60_unf model=fpin hydra.job.chdir=true "hydra.run.dir=${RUN_DIR}" \
  model_cfg.model_args.max_fleet_length=6 model_cfg.model_args.fleet_in_dim=260 \
  model_cfg.model_args.use_attn=True model_cfg.model_args.vehicle_cond_edge_head=True \
  model_cfg.model_args.sinkhorn_assignment=False model_cfg.model_args.joint_customer_norm=False \
  model_cfg.model_args.softassign_head=True model_cfg.model_args.softassign_layers=3 \
  model_cfg.model_args.vcount_aux_head=True +train_cfg.vcount_aux_w=0.1 \
  train_cfg.batch_size=64 train_cfg.n_epochs=200 train_cfg.lr=0.0001 train_cfg.checkpoint_epochs=50 \
  "train_cfg.run_name=${RUN_NAME}" fixed_train_set=/projects/extern/nhr/nhr_ni/nim00026/dir.project/daniela/data/train_data/cvrp/uniform/targets/fc_hgs_clean/n60_k6/ )
if [ -n "${LATEST_CKPT:-}" ] && [ -f "${LATEST_CKPT}" ]; then
  srun python run_fpin.py meta=resume test_cfg.checkpoint_load_path="${LATEST_CKPT}" "${COMMON[@]}"
else
  srun python run_fpin.py meta=train_base "${COMMON[@]}"
fi
