#!/bin/bash
# Eval every downloaded GWDG ckpt locally, raw + post-process, log to results/.
# Identifies (N, M) by reading vehicle_embed.weight.shape[0] from the state_dict
# (M) and the parent path (cvrp_<N>_uniform → N).
#
# Usage:
#   bash eval_gwdg_ckpts.sh                  # raw + LS post-process for every ckpt
#   CKPTS_ROOT=... DRY_RUN=1 bash eval_gwdg_ckpts.sh   # print commands, do not run
set -u

CKPTS_ROOT=${CKPTS_ROOT:-ckpts_from_gwdg}
RESULTS_DIR=${RESULTS_DIR:-results/eval_gwdg_$(date +%Y%m%d_%H%M%S)}
DECODE=${DECODE:-assign_transition}     # match training-time decoder
DRY_RUN=${DRY_RUN:-0}
mkdir -p "$RESULTS_DIR"

# --- Step 1: probe each ckpt for (N, M) ---
mapfile -t PROBED < <(python - <<'PY'
import glob, os, sys, torch
for p in sorted(glob.glob("ckpts_from_gwdg/cvrp_*/train/fpin/*/checkpoints/best-epoch-100.pt")):
    try:
        sd = torch.load(p, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "model" in sd:
            sd = sd["model"]
        m = sd["vehicle_embed.weight"].shape[0]
        # N from path: .../cvrp_<N>_uniform/...
        parts = p.split(os.sep)
        n_str = next((x for x in parts if x.startswith("cvrp_") and x.endswith("_uniform")), "")
        n = int(n_str.split("_")[1]) if n_str else -1
        print(f"{n}\t{m}\t{p}")
    except Exception as e:
        print(f"!ERROR\t{p}\t{e}", file=sys.stderr)
PY
)

echo "=== Probed ckpts ==="
printf '%s\n' "${PROBED[@]}"
echo "=== Results dir: $RESULTS_DIR ==="

# --- Step 2: emit + run eval commands ---
for line in "${PROBED[@]}"; do
  IFS=$'\t' read -r N M CKPT <<< "$line"
  [ -z "$N" ] && continue
  [ "$N" = "!ERROR" ] && continue
  TAG="N${N}_M${M}"
  DATA="data/test_data/cvrp/uniform/cvrp${N}/fc-cvrp_k${M}_seed213298_size1000.pt"
  if [ ! -f "$DATA" ]; then
    echo "[skip $TAG] test data missing: $DATA" >&2
    continue
  fi
  for ADD_LS in False True; do
    SUFFIX=$([ "$ADD_LS" = "False" ] && echo raw || echo ortools)
    LOG="$RESULTS_DIR/${TAG}_${SUFFIX}_${DECODE}.log"
    CMD="python -u run_fpin.py \
      meta=run env=cvrp${N}_unf model=fpin cuda=True \
      run_type=val \
      test_cfg.dataset_size=1000 test_cfg.time_limit=8 test_cfg.add_ls=${ADD_LS} \
      model_cfg.model_args.fleet_in_dim=260 \
      model_cfg.model_args.max_fleet_length=${M} \
      model_cfg.model_args.use_attn=True \
      model_cfg.model_args.vehicle_cond_edge_head=True \
      model_cfg.model_args.sinkhorn_assignment=True \
      model_cfg.model_args.sinkhorn_iters=3 \
      eval_opts_cfg.nr_vehicles_eval=${M} \
      eval_opts_cfg.decode_v_assign_type=${DECODE} \
      test_cfg.ls_policy_cfg.batch_size=8 \
      data_file_path=${DATA} \
      checkpoint_load_path=${CKPT}"
    echo
    echo "### $TAG  ADD_LS=$ADD_LS  decode=$DECODE  -> $LOG"
    echo "$CMD"
    if [ "$DRY_RUN" = "0" ]; then
      eval $CMD 2>&1 | tee "$LOG"
    fi
  done
done

echo
echo "=== Summary (Cost_v + violations) ==="
grep -E "Cost_v|Mean Capacity Violation|Mean Cost|infeasibility|best_ema_val_loss" \
  $RESULTS_DIR/*.log 2>/dev/null
