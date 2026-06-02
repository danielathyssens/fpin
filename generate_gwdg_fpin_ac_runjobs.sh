#!/bin/bash
# F-PIN-AC = F-PIN-A (softassign head) + vcount auxiliary supervision.
# Encoder learns fleet-aware representation via MSE on predicted vs target #vehicles.
set -eo pipefail
PROJ=${PROJ:-/projects/extern/nhr/nhr_ni/nim00026/dir.project/daniela}
TARGET_ROOT=${TARGET_ROOT:-$PROJ/data/train_data/cvrp/uniform/targets/fc_hgs_clean}
target_dir_for() {
  case "${1}_${2}" in
    20_3) printf '%s/n20_k3' "$TARGET_ROOT" ;; 20_4) printf '%s/n20_k4' "$TARGET_ROOT" ;;
    50_6) printf '%s/n50_k6' "$TARGET_ROOT" ;; 50_7) printf '%s/n50_k7' "$TARGET_ROOT" ;;
    60_6) printf '%s/n60_k6' "$TARGET_ROOT" ;; 60_7) printf '%s/n60_k7' "$TARGET_ROOT" ;;
    60_9) printf '%s/data/train_data/cvrp/uniform/targets/converted_npz' "$PROJ" ;;
    100_9) printf '%s/n100_k9' "$TARGET_ROOT" ;; 100_10) printf '%s/n100_k10' "$TARGET_ROOT" ;;
    *) return 1 ;;
  esac
}
batch_for() { case "${1}_${2}" in 100_10) printf '32' ;; *) printf '64' ;; esac; }
gen() {
  local n=$1; local m=$2; local name="GWDG_FPIN_AC_N${n}_M${m}"
  local targets bs
  targets=$(target_dir_for "$n" "$m"); bs=$(batch_for "$n" "$m")
  cat > "RunJob_${name}.sh" <<RJ
#!/bin/bash
#SBATCH --job-name=${name}
#SBATCH --partition=grete:shared
#SBATCH --nodes=1 --ntasks-per-node=1 -G A100:1 --time=48:00:00 --requeue
#SBATCH --mail-type=FAIL --mail-user=thyssens@ismll.de
#SBATCH --output=${PROJ}/cluster_logs/${name}_%j.log
#SBATCH --error=${PROJ}/cluster_logs/${name}_%j.err
set -eo pipefail
mkdir -p "${PROJ}/cluster_logs"
source "\${HOME}/miniconda3/etc/profile.d/conda.sh"; conda activate l2o_py310
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FPIN_OUTPUT_ROOT="${PROJ}/fpin_outputs"; mkdir -p "\${FPIN_OUTPUT_ROOT}"
REPO=\${REPO:-\${SLURM_SUBMIT_DIR:-\$HOME/repos/fpin}}; cd "\${REPO}"
RUN_NAME="fpin_ac_n${n}_k${m}_unf_attn"
RESUME_GLOB="\${FPIN_OUTPUT_ROOT}/cvrp_${n}_uniform/train/fpin/\${RUN_NAME}_*/checkpoints/best.pt"
LATEST_CKPT=\$(ls -t \$RESUME_GLOB 2>/dev/null | head -1 || true)
RUN_DIR="\${FPIN_OUTPUT_ROOT}/cvrp_${n}_uniform/train/fpin/\${RUN_NAME}_\$(date +%Y-%m-%d_%H-%M-%S)"
COMMON=( env=cvrp${n}_unf model=fpin hydra.job.chdir=true "hydra.run.dir=\${RUN_DIR}" \\
  model_cfg.model_args.max_fleet_length=${m} model_cfg.model_args.fleet_in_dim=260 \\
  model_cfg.model_args.use_attn=True model_cfg.model_args.vehicle_cond_edge_head=True \\
  model_cfg.model_args.sinkhorn_assignment=False model_cfg.model_args.joint_customer_norm=False \\
  model_cfg.model_args.softassign_head=True model_cfg.model_args.softassign_layers=3 \\
  model_cfg.model_args.vcount_aux_head=True +train_cfg.vcount_aux_w=0.1 \\
  model_cfg.model_args.add_demand_weights=True \\
  loss_cfg.start_weight=0.2 loss_cfg.pen_w=0.1 loss_cfg.load_w=0.2 \\
  train_cfg.batch_size=${bs} train_cfg.n_epochs=200 train_cfg.lr=0.0001 train_cfg.checkpoint_epochs=50 \\
  "train_cfg.run_name=\${RUN_NAME}" fixed_train_set=${targets}/ )
if [ -n "\${LATEST_CKPT:-}" ] && [ -f "\${LATEST_CKPT}" ]; then
  srun python run_fpin.py meta=resume test_cfg.checkpoint_load_path="\${LATEST_CKPT}" "\${COMMON[@]}"
else
  srun python run_fpin.py meta=train_base "\${COMMON[@]}"
fi
RJ
  echo "wrote RunJob_${name}.sh"
}
gen 20 3; gen 20 4; gen 50 6; gen 50 7; gen 60 6; gen 60 7; gen 60 9; gen 100 9; gen 100 10
