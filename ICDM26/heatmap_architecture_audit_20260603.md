# Heatmap Architecture Audit — 2026-06-03

## Scope

This note focuses on the question:

> Why is the F-PIN heatmap weaker than old PIM, even after adding a softassign-style head?

The goal is not another decode-only comparison. The goal is to isolate whether the
heatmap itself is failing at:

1. customer-to-vehicle clustering
2. within-vehicle ordering
3. capacity-aware mass allocation


## Coverage Matrix

This audit now covers the requested points without duplicating overlapping ablations:

1. `clustering vs ordering`
   Covered by [experiments/heatmap_failure_audit.py](/home/thyssens/Research/L2O/github_projects/fpin/experiments/heatmap_failure_audit.py:1)
   via `membership_acc`, `succ_ce_next`, and `succ_ce_correct_cluster`.
2. `capacity-awareness from raw heatmap`
   Covered by `expected_load_mean`, `overload_mean`, `overload_max_mean`, `any_overload_rate`.
3. `no vehicle-ID ablation`
   Covered by `model_cfg.model_args.use_vehicle_id_embedding=False`.
4. `old scoring head ablation`
   Covered by `model_cfg.model_args.vehicle_cond_edge_head=False`.
5. `encoder factorization`
   Covered by:
   - `use_graph_encoder=False`  -> PermInv-only path
   - `use_perm_inv_encoder=False` -> GraphEncoder-only path
   - `use_attn=False` -> no-attention PermInv path
6. `softassign strength sweep`
   Covered by `softassign_layers in {10,30,90}` and fixed-temperature variant.
7. `no extra demand weighting`
   Covered by `add_demand_weights=False`.
8. `old-PIM provenance`
   Covered by the code-path audit plus the checkpoint-family audit below.


## Budget Rationale

The audit is split into three budgets because “one setting for everything” is not principled here.

### Smoke

- `N20` only or `N20+N60` if you want a very quick direction check
- `10` epochs
- `10k` training samples
- purpose: filter out obviously bad variants before spending meaningful GPU time

### Medium

- default recommended two-scale audit
- `N20_k3` + `N50_k7`
- `30` epochs
- `30k` training samples
- purpose: enough optimization to stabilize directional effects, while still short enough to run several variants

### Full

- same two scales
- `75` epochs
- `60k` training samples
- purpose: verify that early rank ordering survives a nontrivial training horizon, without paying the full `200`-epoch cost for every ablation

Why these values:

- `10` epochs is enough for fast screening because your local baseline already reaches a useful train/val signal at that horizon.
- `30` epochs is the first point where architectural effects usually stop looking like pure initialization noise.
- `75` epochs is long enough that “only learns later” variants have a real chance, but still far cheaper than replicating every ablation at the full main-run budget.

Why `N20_k3` and `N50_k7`:

- `N20_k3` is the cheapest setting and already has the local HGS target file.
- `N50_k7` is the larger scale that still has direct old-PIM comparison values in the archive tables.
- `N60_k7` is still useful as an extrapolation scale, but it is no longer the default audit scale because old PIM was not one of the paper-era comparison points there.


## Generator Script

New generator:

- [experiments/generate_ablation_audit_commands.py](/home/thyssens/Research/L2O/github_projects/fpin/experiments/generate_ablation_audit_commands.py:1)

It does three jobs:

1. prints separate local or GWDG commands
2. emits `rsync` commands for missing local HGS target folders
3. can submit one sbatch script per training ablation with `--profile gwdg --submit`

### Recommended local medium audit

```bash
cd /home/thyssens/Research/L2O/github_projects/fpin
python experiments/generate_ablation_audit_commands.py \
  --profile local \
  --budget medium \
  --phase all \
  --scales n20_k3 n50_k7 \
  --skip-existing \
  --write-dir /tmp/ablation_audit_cmds
```

### Recommended GWDG medium audit submission

```bash
cd /projects/extern/nhr/nhr_ni/nim00026/dir.project/daniela/fpin
python experiments/generate_ablation_audit_commands.py \
  --profile gwdg \
  --budget medium \
  --phase train \
  --scales n20_k3 n50_k7 \
  --write-dir /tmp/ablation_audit_submit \
  --submit
```

### Quick local smoke, only the highest-priority replacements

```bash
cd /home/thyssens/Research/L2O/github_projects/fpin
python experiments/generate_ablation_audit_commands.py \
  --profile local \
  --budget smoke \
  --phase train \
  --scales n20_k3 \
  --variants ab_base ab_vidoff ab_oldhead ab_nodemandw ab_soft30 \
  --skip-existing
```


## Hard Clarifications

### 1. The current integrated `run_PIMold.py` path is **not** the archived `run_PIM.py` baseline path.

- `run_PIM.py` is **not present** in this repo.
- `run_PIMold.py` uses [models/PIMold/pimold.py](/home/thyssens/Research/L2O/github_projects/fpin/models/PIMold/pimold.py:61).
- `build_old_model()` there instantiates `VRPModel_attn1`, not `VRPModel_attn_soft`.

So the current integrated `PIMold` runner is not the same execution path as the archived
March evals that produced the `114.15` `Cost_v` regime.

### 2. The `114.15` `Cost_v` result comes from archived `run_PIM.py` logs using the `assign` decoder family.

Relevant references:

- [eval_logs/pim_raw_summary.md](/home/thyssens/Research/L2O/github_projects/fpin/eval_logs/pim_raw_summary.md:64)
- [eval_logs/pim_grouped_runs.csv](/home/thyssens/Research/L2O/github_projects/fpin/eval_logs/pim_grouped_runs.csv:11)
- [eval_logs/run_commands_fpin.md](/home/thyssens/Research/L2O/github_projects/fpin/eval_logs/run_commands_fpin.md:9)

The archived checkpoint reference in those command sheets is:

`models/PIM/PIM/logs/ckpts/uniform_20/best-ep-n20_k3_unf_attn.pt`

That path does not exist in the current repo snapshot, so the `114.15` baseline is
currently documented by logs, not reproducible from the checked-in runner alone.

### 3. The checked-in ICLR22 code snapshot points to **softassign disabled** in the publication path.

Relevant references:

- [sup-vrp-copy/ICLR_2022/code_iclr22/train.py](/home/thyssens/Research/L2O/github_projects/fpin/sup-vrp-copy/ICLR_2022/code_iclr22/train.py:68)
- [sup-vrp-copy/ICLR_2022/code_iclr22/VRPModel_attn.py](/home/thyssens/Research/L2O/github_projects/fpin/sup-vrp-copy/ICLR_2022/code_iclr22/VRPModel_attn.py:157)

The publication-code snapshot has:

- `n_SOFTlayers = 0` by default in training
- the `MTSPSoftassign` import commented out
- the `self.softassign(...)` call commented out

So the strongest code-based reading is:

> the ICLR22 training/eval branch you shipped was using the non-softassign path.

### 4. The local `models/PIMold/logs/vrp20|vrp50/VRP_model.pth` checkpoints look like the ICLR22-era code family, but the state dict alone cannot prove whether softassign was used.

What is true:

- they match the ICLR22 code naming scheme (`temperature_dists`, `beta_d_q`)
- they do **not** match the newer FPIN family
- they can be loaded into both `VRPModel_attn1` and `VRPModel_attn_soft`, because softassign adds no trainable parameters

What is **not** true:

- we cannot prove from the state dict alone that a particular checkpoint was trained with softassign on or off

So the current best answer is:

- the publication **code path** points to `softassign off`
- the archived `uniform_20` / `uniform_60` publication checkpoints referenced by the old `run_PIM.py` logs are **not present locally**
- therefore the exact `114.15` checkpoint provenance is still partially blocked by a missing artifact, not by missing code inspection

### 5. `fpin/logs/ckpts/uniform_20.zip` is **not** old PIM.

It contains the newer FPIN-family graph/vehicle embedding stack and should not be used as an old-PIM baseline checkpoint.


## Tiny Heatmap Audit Results

New script:

- [experiments/heatmap_failure_audit.py](/home/thyssens/Research/L2O/github_projects/fpin/experiments/heatmap_failure_audit.py:1)

This script measures:

- `membership_acc`: customer -> vehicle assignment accuracy after Hungarian alignment
- `succ_ce_next`: within-route successor cross-entropy on customer rows
- `overload_max_mean`: max expected per-vehicle overload from raw assignment mass
- `any_overload_rate`: how often any vehicle is overloaded before decode

### N20_M3, 32-instance CPU audit slice

| model | membership_acc | succ_ce_next | overload_max_mean | any_overload_rate |
|---|---:|---:|---:|---:|
| `pim_soft` | `0.7344` | `5.1515` | `0.1295` | `0.9063` |
| `fpin_a` | `0.5703` | `4.3718` | `0.3702` | `1.0000` |
| `fpin_ab` | `0.8406` | `4.3468` | `0.0427` | `0.8125` |

Interpretation:

- `AB` is a real heatmap improvement over `A` on clustering and overload.
- `A` is the weak point here, not softassign-in-general.
- `pim_soft` still stays stronger than `A` on clustering/overload.
- On this slice, `AB` actually beats `pim_soft` on clustering and overload, so the raw-eval gap is **not** explained by the heatmap alone anymore.

### N20_M3, 8-instance CPU sanity slice

| model | membership_acc | succ_ce_next | overload_max_mean | any_overload_rate |
|---|---:|---:|---:|---:|
| `pim_soft` | `0.7313` | `5.6858` | `0.1210` | `0.8750` |
| `pim_attn1` | `0.5000` | `10.0277` | `0.2199` | `1.0000` |
| `fpin_a` | `0.5688` | `4.1514` | `0.4373` | `1.0000` |
| `fpin_ab` | `0.8625` | `4.1283` | `0.0399` | `0.8750` |
| `fpin_ac` | `0.7625` | `3.0579` | `0.2611` | `1.0000` |
| `fpin_e3` | `0.2125` | `6.0614` | `0.0271` | `0.6250` |

Interpretation:

- `pim_attn1` is clearly a different regime from `pim_soft`.
- `AC` helps ordering CE on this tiny slice but still looks worse than `AB` on overload.
- `E3` appears to push low overload but very weak route-membership structure.


## Why These Architectural Changes Can Hurt

### Explicit vehicle IDs can weaken a permutation-invariant target

F-PIN injects explicit vehicle IDs in [fpin/data_utils/preprocess1.py](/home/thyssens/Research/L2O/github_projects/fpin/fpin/data_utils/preprocess1.py:45) and embeds them in [fpin/VRPModel_attn_new.py](/home/thyssens/Research/L2O/github_projects/fpin/fpin/VRPModel_attn_new.py:34).

Why this can hurt:

- vehicles are exchangeable, but IDs are arbitrary
- route labels in the target do not carry semantic meaning like “vehicle 1 is special”
- the model can waste capacity memorizing route-slot conventions instead of learning clustering geometry
- across seeds / instances, equally good solutions can appear under different vehicle orderings, so arbitrary IDs can inject label noise upstream of a permutation-invariant loss

This is exactly the sort of change that can help optimization locally and still hurt generalization / heatmap structure globally.

### The old head is more route-biased than the vehicle-conditioned MLP head

Old PIM-style scoring is closer to a structured compatibility score:

- vehicle query
- pairwise edge representation
- final dot-product / bilinear-like interaction

The newer F-PIN `vehicle_cond_edge_head=True` path in [fpin/VRPModel_attn_new.py](/home/thyssens/Research/L2O/github_projects/fpin/fpin/VRPModel_attn_new.py:279) is more expressive, but that flexibility cuts both ways:

- it can fit arbitrary per-vehicle edge templates
- it is less constrained to behave like a route-compatibility score
- it is easier to overfit vehicle-specific edge quirks instead of learning clean cluster-then-order structure

So “more expressive” does not imply “better heatmap prior”.


## Local Commands — Existing Checkpoint Heatmap Audits

Run from local repo root:

```bash
conda activate l2o_py310
cd /home/thyssens/Research/L2O/github_projects/fpin
mkdir -p eval_logs/heatmap_audit_20260603
LD=eval_logs/heatmap_audit_20260603
DATA_NPZ=data/train_data/cvrp/uniform/targets/fc_hgs_clean/n20_k3/targets20_m3_seed448689_size150000_hgs_t3.npz
```

### A

```bash
PYTHONPATH=. python experiments/heatmap_failure_audit.py \
  --model-family fpin \
  --ckpt ckpts_from_gwdg/cvrp_20_uniform/train/fpin/fpin_a_n20_k3_unf_attn_2026-06-02_14-37-17/checkpoints/best.pt \
  --data-npz "$DATA_NPZ" \
  --graph-size 20 --fleet-size 3 --limit 256 --batch-size 32 --device cuda \
  --tag fpin_a_n20_k3 \
  --softassign-head --softassign-layers 3 \
  --vehicle-cond-edge-head --use-attn --add-demand-weights \
  --out-json "$LD/fpin_a_n20_k3.json"
```

### AB

```bash
PYTHONPATH=. python experiments/heatmap_failure_audit.py \
  --model-family fpin \
  --ckpt ckpts_from_gwdg/cvrp_20_uniform/train/fpin/fpin_ab_n20_k3_unf_attn_2026-06-02_19-26-31/checkpoints/best.pt \
  --data-npz "$DATA_NPZ" \
  --graph-size 20 --fleet-size 3 --limit 256 --batch-size 32 --device cuda \
  --tag fpin_ab_n20_k3 \
  --softassign-head --softassign-layers 3 --softassign-log-domain \
  --vehicle-cond-edge-head --use-attn --add-demand-weights \
  --out-json "$LD/fpin_ab_n20_k3.json"
```

### AC

```bash
PYTHONPATH=. python experiments/heatmap_failure_audit.py \
  --model-family fpin \
  --ckpt ckpts_from_gwdg/cvrp_20_uniform/train/fpin/fpin_ac_n20_k3_unf_attn_2026-06-02_16-49-37/checkpoints/best.pt \
  --data-npz "$DATA_NPZ" \
  --graph-size 20 --fleet-size 3 --limit 256 --batch-size 32 --device cuda \
  --tag fpin_ac_n20_k3 \
  --softassign-head --softassign-layers 3 \
  --vehicle-cond-edge-head --use-attn --add-demand-weights --vcount-aux-head \
  --out-json "$LD/fpin_ac_n20_k3.json"
```

### E3

```bash
PYTHONPATH=. python experiments/heatmap_failure_audit.py \
  --model-family fpin \
  --ckpt ckpts_from_gwdg/cvrp_20_uniform/train/fpin/fpin_e3_n20_k3_unf_attn_2026-06-01_17-04-36/checkpoints/best-epoch-200.pt \
  --data-npz "$DATA_NPZ" \
  --graph-size 20 --fleet-size 3 --limit 256 --batch-size 32 --device cuda \
  --tag fpin_e3_n20_k3 \
  --sinkhorn-assignment --sinkhorn-iters 3 \
  --vehicle-cond-edge-head --use-attn \
  --out-json "$LD/fpin_e3_n20_k3.json"
```

### Old PIM Softassign Reference

```bash
PYTHONPATH=. python experiments/heatmap_failure_audit.py \
  --model-family pim_soft \
  --ckpt models/PIMold/logs/vrp20/VRP_model.pth \
  --data-npz "$DATA_NPZ" \
  --graph-size 20 --fleet-size 3 --limit 256 --batch-size 32 --device cuda \
  --tag pim_soft_n20_k3 \
  --pim-softassign-layers 90 \
  --out-json "$LD/pim_soft_n20_k3.json"
```

### Integrated PIMold `attn1` Reference

```bash
PYTHONPATH=. python experiments/heatmap_failure_audit.py \
  --model-family pim_attn1 \
  --ckpt models/PIMold/logs/vrp20/VRP_model.pth \
  --data-npz "$DATA_NPZ" \
  --graph-size 20 --fleet-size 3 --limit 256 --batch-size 32 --device cuda \
  --tag pim_attn1_n20_k3 \
  --out-json "$LD/pim_attn1_n20_k3.json"
```


## Local Commands — 10-Epoch Architecture A/B Smokes

Use `N20_M3` first. It is the cheapest setting with a real target file already present locally.

Base setup:

```bash
conda activate l2o_py310
cd /home/thyssens/Research/L2O/github_projects/fpin
TRAIN_SET=data/train_data/cvrp/uniform/targets/fc_hgs_clean/n20_k3/
```

### 1. No Vehicle ID

```bash
python run_fpin.py meta=train_base env=cvrp20_unf model=fpin cuda=True \
  model_cfg.model_args.softassign_head=True model_cfg.model_args.softassign_layers=3 \
  model_cfg.model_args.softassign_log_domain=True \
  model_cfg.model_args.sinkhorn_assignment=False model_cfg.model_args.joint_customer_norm=False \
  model_cfg.model_args.add_demand_weights=True \
  model_cfg.model_args.use_vehicle_id_embedding=False \
  model_cfg.model_args.max_fleet_length=3 \
  train_cfg.batch_size=32 train_cfg.n_epochs=10 train_cfg.lr=0.0001 \
  train_cfg.nr_train_samples=10000 train_cfg.checkpoint_epochs=2 \
  loss_cfg.start_weight=0.2 loss_cfg.pen_w=0.1 \
  train_cfg.run_name=fpin_ab_vidoff_n20_k3_smoke \
  fixed_train_set="$TRAIN_SET" \
  2>&1 | tee /tmp/fpin_ab_vidoff_n20_k3_smoke.log
```

### 2. Old Scoring Head

```bash
python run_fpin.py meta=train_base env=cvrp20_unf model=fpin cuda=True \
  model_cfg.model_args.softassign_head=True model_cfg.model_args.softassign_layers=3 \
  model_cfg.model_args.softassign_log_domain=True \
  model_cfg.model_args.sinkhorn_assignment=False model_cfg.model_args.joint_customer_norm=False \
  model_cfg.model_args.add_demand_weights=True \
  model_cfg.model_args.vehicle_cond_edge_head=False \
  model_cfg.model_args.max_fleet_length=3 \
  train_cfg.batch_size=32 train_cfg.n_epochs=10 train_cfg.lr=0.0001 \
  train_cfg.nr_train_samples=10000 train_cfg.checkpoint_epochs=2 \
  loss_cfg.start_weight=0.2 loss_cfg.pen_w=0.1 \
  train_cfg.run_name=fpin_ab_oldhead_n20_k3_smoke \
  fixed_train_set="$TRAIN_SET" \
  2>&1 | tee /tmp/fpin_ab_oldhead_n20_k3_smoke.log
```

### 3. No Extra Demand Weighting

```bash
python run_fpin.py meta=train_base env=cvrp20_unf model=fpin cuda=True \
  model_cfg.model_args.softassign_head=True model_cfg.model_args.softassign_layers=3 \
  model_cfg.model_args.softassign_log_domain=True \
  model_cfg.model_args.sinkhorn_assignment=False model_cfg.model_args.joint_customer_norm=False \
  model_cfg.model_args.add_demand_weights=False \
  model_cfg.model_args.max_fleet_length=3 \
  train_cfg.batch_size=32 train_cfg.n_epochs=10 train_cfg.lr=0.0001 \
  train_cfg.nr_train_samples=10000 train_cfg.checkpoint_epochs=2 \
  loss_cfg.start_weight=0.2 loss_cfg.pen_w=0.1 \
  train_cfg.run_name=fpin_ab_nodemandw_n20_k3_smoke \
  fixed_train_set="$TRAIN_SET" \
  2>&1 | tee /tmp/fpin_ab_nodemandw_n20_k3_smoke.log
```

### 4. No Graph Encoder

```bash
python run_fpin.py meta=train_base env=cvrp20_unf model=fpin cuda=True \
  model_cfg.model_args.softassign_head=True model_cfg.model_args.softassign_layers=3 \
  model_cfg.model_args.softassign_log_domain=True \
  model_cfg.model_args.sinkhorn_assignment=False model_cfg.model_args.joint_customer_norm=False \
  model_cfg.model_args.add_demand_weights=True \
  model_cfg.model_args.use_graph_encoder=False \
  model_cfg.model_args.max_fleet_length=3 \
  train_cfg.batch_size=32 train_cfg.n_epochs=10 train_cfg.lr=0.0001 \
  train_cfg.nr_train_samples=10000 train_cfg.checkpoint_epochs=2 \
  loss_cfg.start_weight=0.2 loss_cfg.pen_w=0.1 \
  train_cfg.run_name=fpin_ab_nograph_n20_k3_smoke \
  fixed_train_set="$TRAIN_SET" \
  2>&1 | tee /tmp/fpin_ab_nograph_n20_k3_smoke.log
```

### 5. No PermInvNet

```bash
python run_fpin.py meta=train_base env=cvrp20_unf model=fpin cuda=True \
  model_cfg.model_args.softassign_head=True model_cfg.model_args.softassign_layers=3 \
  model_cfg.model_args.softassign_log_domain=True \
  model_cfg.model_args.sinkhorn_assignment=False model_cfg.model_args.joint_customer_norm=False \
  model_cfg.model_args.add_demand_weights=True \
  model_cfg.model_args.use_perm_inv_encoder=False \
  model_cfg.model_args.max_fleet_length=3 \
  train_cfg.batch_size=32 train_cfg.n_epochs=10 train_cfg.lr=0.0001 \
  train_cfg.nr_train_samples=10000 train_cfg.checkpoint_epochs=2 \
  loss_cfg.start_weight=0.2 loss_cfg.pen_w=0.1 \
  train_cfg.run_name=fpin_ab_noperminv_n20_k3_smoke \
  fixed_train_set="$TRAIN_SET" \
  2>&1 | tee /tmp/fpin_ab_noperminv_n20_k3_smoke.log
```

### 6. No Attention in PermInvNet

```bash
python run_fpin.py meta=train_base env=cvrp20_unf model=fpin cuda=True \
  model_cfg.model_args.softassign_head=True model_cfg.model_args.softassign_layers=3 \
  model_cfg.model_args.softassign_log_domain=True \
  model_cfg.model_args.sinkhorn_assignment=False model_cfg.model_args.joint_customer_norm=False \
  model_cfg.model_args.add_demand_weights=True \
  model_cfg.model_args.use_attn=False \
  model_cfg.model_args.max_fleet_length=3 \
  train_cfg.batch_size=32 train_cfg.n_epochs=10 train_cfg.lr=0.0001 \
  train_cfg.nr_train_samples=10000 train_cfg.checkpoint_epochs=2 \
  loss_cfg.start_weight=0.2 loss_cfg.pen_w=0.1 \
  train_cfg.run_name=fpin_ab_noattn_n20_k3_smoke \
  fixed_train_set="$TRAIN_SET" \
  2>&1 | tee /tmp/fpin_ab_noattn_n20_k3_smoke.log
```

### 7. Stronger Softassign: 10 Layers

```bash
python run_fpin.py meta=train_base env=cvrp20_unf model=fpin cuda=True \
  model_cfg.model_args.softassign_head=True model_cfg.model_args.softassign_layers=10 \
  model_cfg.model_args.softassign_log_domain=True \
  model_cfg.model_args.sinkhorn_assignment=False model_cfg.model_args.joint_customer_norm=False \
  model_cfg.model_args.add_demand_weights=True \
  model_cfg.model_args.max_fleet_length=3 \
  train_cfg.batch_size=32 train_cfg.n_epochs=10 train_cfg.lr=0.0001 \
  train_cfg.nr_train_samples=10000 train_cfg.checkpoint_epochs=2 \
  loss_cfg.start_weight=0.2 loss_cfg.pen_w=0.1 \
  train_cfg.run_name=fpin_ab_soft10_n20_k3_smoke \
  fixed_train_set="$TRAIN_SET" \
  2>&1 | tee /tmp/fpin_ab_soft10_n20_k3_smoke.log
```

### 8. Stronger Softassign: 30 Layers

```bash
python run_fpin.py meta=train_base env=cvrp20_unf model=fpin cuda=True \
  model_cfg.model_args.softassign_head=True model_cfg.model_args.softassign_layers=30 \
  model_cfg.model_args.softassign_log_domain=True \
  model_cfg.model_args.sinkhorn_assignment=False model_cfg.model_args.joint_customer_norm=False \
  model_cfg.model_args.add_demand_weights=True \
  model_cfg.model_args.max_fleet_length=3 \
  train_cfg.batch_size=32 train_cfg.n_epochs=10 train_cfg.lr=0.0001 \
  train_cfg.nr_train_samples=10000 train_cfg.checkpoint_epochs=2 \
  loss_cfg.start_weight=0.2 loss_cfg.pen_w=0.1 \
  train_cfg.run_name=fpin_ab_soft30_n20_k3_smoke \
  fixed_train_set="$TRAIN_SET" \
  2>&1 | tee /tmp/fpin_ab_soft30_n20_k3_smoke.log
```

### 9. Stronger Softassign: 90 Layers

```bash
python run_fpin.py meta=train_base env=cvrp20_unf model=fpin cuda=True \
  model_cfg.model_args.softassign_head=True model_cfg.model_args.softassign_layers=90 \
  model_cfg.model_args.softassign_log_domain=True \
  model_cfg.model_args.sinkhorn_assignment=False model_cfg.model_args.joint_customer_norm=False \
  model_cfg.model_args.add_demand_weights=True \
  model_cfg.model_args.max_fleet_length=3 \
  train_cfg.batch_size=32 train_cfg.n_epochs=10 train_cfg.lr=0.0001 \
  train_cfg.nr_train_samples=10000 train_cfg.checkpoint_epochs=2 \
  loss_cfg.start_weight=0.2 loss_cfg.pen_w=0.1 \
  train_cfg.run_name=fpin_ab_soft90_n20_k3_smoke \
  fixed_train_set="$TRAIN_SET" \
  2>&1 | tee /tmp/fpin_ab_soft90_n20_k3_smoke.log
```

### 10. Fixed Sharper Temperature

```bash
python run_fpin.py meta=train_base env=cvrp20_unf model=fpin cuda=True \
  model_cfg.model_args.softassign_head=True model_cfg.model_args.softassign_layers=3 \
  model_cfg.model_args.softassign_log_domain=True \
  model_cfg.model_args.sinkhorn_assignment=False model_cfg.model_args.joint_customer_norm=False \
  model_cfg.model_args.add_demand_weights=True \
  model_cfg.model_args.learnable_temperature=False \
  model_cfg.model_args.initial_log_temperature=-2.0 \
  model_cfg.model_args.max_fleet_length=3 \
  train_cfg.batch_size=32 train_cfg.n_epochs=10 train_cfg.lr=0.0001 \
  train_cfg.nr_train_samples=10000 train_cfg.checkpoint_epochs=2 \
  loss_cfg.start_weight=0.2 loss_cfg.pen_w=0.1 \
  train_cfg.run_name=fpin_ab_fixtemp_n20_k3_smoke \
  fixed_train_set="$TRAIN_SET" \
  2>&1 | tee /tmp/fpin_ab_fixtemp_n20_k3_smoke.log
```


## Verified Launches

These two new architecture-ablation training paths were actually launched locally on CPU and completed 1 epoch successfully:

- `use_vehicle_id_embedding=False`
- `vehicle_cond_edge_head=False`

So those two are not just speculative config knobs; they do run end-to-end in the current codebase.


## Recommended Order

If time is tight, run these first:

1. `fpin_ab_vidoff_n20_k3_smoke`
2. `fpin_ab_oldhead_n20_k3_smoke`
3. `fpin_ab_nodemandw_n20_k3_smoke`
4. `fpin_ab_soft30_n20_k3_smoke`
5. `fpin_ab_nograph_n20_k3_smoke`

Reason:

- first isolate symmetry-breaking
- then isolate scorer bias
- then remove the extra pairwise demand weighting
- then test whether the head is simply under-enforced
- only then test larger body-factorization changes
