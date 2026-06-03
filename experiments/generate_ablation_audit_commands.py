#!/usr/bin/env python3
"""Generate architecture-audit commands for local runs or GWDG submission.

Prints separate shell commands by default so the user can launch them manually
or in parallel. With ``--profile gwdg --submit`` it writes one sbatch script
per training ablation and submits them.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List


REMOTE_ROOT = "/projects/extern/nhr/nhr_ni/nim00026/dir.project/daniela"
LOCAL_REPO = "/home/thyssens/Research/L2O/github_projects/fpin"
GWDG_REPO = f"{REMOTE_ROOT}/fpin"
GWDG_OUTPUT_ROOT = f"{REMOTE_ROOT}/fpin_outputs"
GWDG_TRAIN_TARGET_ROOT = f"{REMOTE_ROOT}/data/train_data/cvrp/uniform/targets/fc_hgs_clean"
GWDG_LOG_ROOT = f"{REMOTE_ROOT}/cluster_logs"


@dataclass(frozen=True)
class Profile:
    name: str
    repo: str
    train_target_root: str
    ckpt_existing_root: str
    output_root: str
    eval_root: str


@dataclass(frozen=True)
class Scale:
    key: str
    graph_size: int
    fleet_size: int
    env: str
    train_folder: str
    test_relpath: str
    test_dataset_size: int
    target_glob: str


@dataclass(frozen=True)
class Budget:
    name: str
    n_epochs: int
    nr_train_samples: int
    checkpoint_epochs: int
    local_batch_n20: int
    local_batch_n60: int
    gwdg_batch: int
    sbatch_time: str


PROFILES: Dict[str, Profile] = {
    "local": Profile(
        name="local",
        repo=LOCAL_REPO,
        train_target_root=f"{LOCAL_REPO}/data/train_data/cvrp/uniform/targets/fc_hgs_clean",
        ckpt_existing_root=f"{LOCAL_REPO}/ckpts_from_gwdg",
        output_root=f"{LOCAL_REPO}/ablation_runs",
        eval_root=f"{LOCAL_REPO}/eval_logs",
    ),
    "gwdg": Profile(
        name="gwdg",
        repo=GWDG_REPO,
        train_target_root=GWDG_TRAIN_TARGET_ROOT,
        ckpt_existing_root=GWDG_OUTPUT_ROOT,
        output_root=GWDG_OUTPUT_ROOT,
        eval_root=f"{GWDG_REPO}/eval_logs",
    ),
}


SCALES: Dict[str, Scale] = {
    "n20_k3": Scale(
        key="n20_k3",
        graph_size=20,
        fleet_size=3,
        env="cvrp20_unf",
        train_folder="n20_k3",
        test_relpath="data/test_data/cvrp/uniform/cvrp20/fc-cvrp_k3_seed213298_size1000.pt",
        test_dataset_size=1000,
        target_glob="targets20*.npz",
    ),
    "n50_k7": Scale(
        key="n50_k7",
        graph_size=50,
        fleet_size=7,
        env="cvrp50_unf",
        train_folder="n50_k7",
        test_relpath="data/test_data/cvrp/uniform/cvrp50/fc-cvrp_k7_seed213298_size1000.pt",
        test_dataset_size=1000,
        target_glob="targets50*.npz",
    ),
    "n60_k7": Scale(
        key="n60_k7",
        graph_size=60,
        fleet_size=7,
        env="cvrp60_unf",
        train_folder="n60_k7",
        test_relpath="data/test_data/cvrp/uniform/cvrp60/fc-cvrp_k7_seed213298_size130.pt",
        test_dataset_size=130,
        target_glob="targets60*.npz",
    ),
}


BUDGETS: Dict[str, Budget] = {
    "smoke": Budget(
        name="smoke",
        n_epochs=10,
        nr_train_samples=10_000,
        checkpoint_epochs=2,
        local_batch_n20=32,
        local_batch_n60=16,
        gwdg_batch=64,
        sbatch_time="06:00:00",
    ),
    "medium": Budget(
        name="medium",
        n_epochs=30,
        nr_train_samples=30_000,
        checkpoint_epochs=5,
        local_batch_n20=32,
        local_batch_n60=16,
        gwdg_batch=64,
        sbatch_time="16:00:00",
    ),
    "full": Budget(
        name="full",
        n_epochs=75,
        nr_train_samples=60_000,
        checkpoint_epochs=10,
        local_batch_n20=32,
        local_batch_n60=16,
        gwdg_batch=64,
        sbatch_time="36:00:00",
    ),
}


BASE_TRAIN_FLAGS = {
    "model_cfg.model_args.softassign_head": True,
    "model_cfg.model_args.softassign_layers": 3,
    "model_cfg.model_args.softassign_log_domain": True,
    "model_cfg.model_args.sinkhorn_assignment": False,
    "model_cfg.model_args.joint_customer_norm": False,
    "model_cfg.model_args.add_demand_weights": True,
    "model_cfg.model_args.use_attn": True,
    "model_cfg.model_args.vehicle_cond_edge_head": True,
    "loss_cfg.start_weight": 0.2,
    "loss_cfg.pen_w": 0.1,
}


TRAIN_VARIANTS = [
    ("ab_base", "AB baseline control", {}),
    ("ab_vidoff", "No vehicle-ID embedding", {
        "model_cfg.model_args.use_vehicle_id_embedding": False,
    }),
    ("ab_oldhead", "Old scoring head", {
        "model_cfg.model_args.vehicle_cond_edge_head": False,
    }),
    ("ab_nodemandw", "No extra demand weighting", {
        "model_cfg.model_args.add_demand_weights": False,
    }),
    ("ab_nograph", "No graph encoder", {
        "model_cfg.model_args.use_graph_encoder": False,
    }),
    ("ab_noperminv", "No PermInvNet", {
        "model_cfg.model_args.use_perm_inv_encoder": False,
    }),
    ("ab_noattn", "No attention in PermInvNet", {
        "model_cfg.model_args.use_attn": False,
    }),
    ("ab_soft10", "Softassign 10 layers", {
        "model_cfg.model_args.softassign_layers": 10,
    }),
    ("ab_soft30", "Softassign 30 layers", {
        "model_cfg.model_args.softassign_layers": 30,
    }),
    ("ab_soft90", "Softassign 90 layers", {
        "model_cfg.model_args.softassign_layers": 90,
    }),
    ("ab_fixtemp_m2", "Fixed sharper temperature", {
        "model_cfg.model_args.learnable_temperature": False,
        "model_cfg.model_args.initial_log_temperature": -2.0,
    }),
]


EXISTING_HEATMAP_CKPTS = {
    "n20_k3": [
        ("fpin_a_n20_k3", "fpin", "cvrp_20_uniform/train/fpin/fpin_a_n20_k3_unf_attn_2026-06-02_14-37-17/checkpoints/best.pt", {
            "--softassign-head": None, "--softassign-layers": 3,
            "--vehicle-cond-edge-head": None, "--use-attn": None, "--add-demand-weights": None,
        }),
        ("fpin_ab_n20_k3", "fpin", "cvrp_20_uniform/train/fpin/fpin_ab_n20_k3_unf_attn_2026-06-02_19-26-31/checkpoints/best.pt", {
            "--softassign-head": None, "--softassign-layers": 3, "--softassign-log-domain": None,
            "--vehicle-cond-edge-head": None, "--use-attn": None, "--add-demand-weights": None,
        }),
        ("fpin_ac_n20_k3", "fpin", "cvrp_20_uniform/train/fpin/fpin_ac_n20_k3_unf_attn_2026-06-02_16-49-37/checkpoints/best.pt", {
            "--softassign-head": None, "--softassign-layers": 3, "--vcount-aux-head": None,
            "--vehicle-cond-edge-head": None, "--use-attn": None, "--add-demand-weights": None,
        }),
        ("fpin_e3_n20_k3", "fpin", "cvrp_20_uniform/train/fpin/fpin_e3_n20_k3_unf_attn_2026-06-01_17-04-36/checkpoints/best-epoch-200.pt", {
            "--sinkhorn-assignment": None, "--sinkhorn-iters": 3,
            "--vehicle-cond-edge-head": None, "--use-attn": None,
        }),
        ("pim_soft_n20_k3", "pim_soft", "models/PIMold/logs/vrp20/VRP_model.pth", {
            "--pim-softassign-layers": 90,
        }),
        ("pim_attn1_n20_k3", "pim_attn1", "models/PIMold/logs/vrp20/VRP_model.pth", {}),
    ],
    "n50_k7": [
        ("fpin_a_n50_k7", "fpin", "cvrp_50_uniform/train/fpin/fpin_a_n50_k7_unf_attn_2026-06-02_14-37-20/checkpoints/best.pt", {
            "--softassign-head": None, "--softassign-layers": 3,
            "--vehicle-cond-edge-head": None, "--use-attn": None, "--add-demand-weights": None,
        }),
        ("fpin_ab_n50_k7", "fpin", "cvrp_50_uniform/train/fpin/fpin_ab_n50_k7_unf_attn_2026-06-03_00-43-20/checkpoints/best.pt", {
            "--softassign-head": None, "--softassign-layers": 3, "--softassign-log-domain": None,
            "--vehicle-cond-edge-head": None, "--use-attn": None, "--add-demand-weights": None,
        }),
        ("fpin_ac_n50_k7", "fpin", "cvrp_50_uniform/train/fpin/fpin_ac_n50_k7_unf_attn_2026-06-02_16-49-39/checkpoints/best.pt", {
            "--softassign-head": None, "--softassign-layers": 3, "--vcount-aux-head": None,
            "--vehicle-cond-edge-head": None, "--use-attn": None, "--add-demand-weights": None,
        }),
        ("fpin_e3_n50_k7", "fpin", "cvrp_50_uniform/train/fpin/fpin_e3_n50_k7_unf_attn_2026-06-01_17-04-42/checkpoints/best-epoch-200.pt", {
            "--sinkhorn-assignment": None, "--sinkhorn-iters": 3,
            "--vehicle-cond-edge-head": None, "--use-attn": None,
        }),
        ("pim_soft_n50_k7", "pim_soft", "models/PIMold/logs/vrp50/VRP_model.pth", {
            "--pim-softassign-layers": 90,
        }),
        ("pim_attn1_n50_k7", "pim_attn1", "models/PIMold/logs/vrp50/VRP_model.pth", {}),
    ],
    "n60_k7": [
        ("fpin_a_n60_k7", "fpin", "cvrp_60_uniform/train/fpin/fpin_a_n60_k7_unf_attn_2026-06-02_14-37-19/checkpoints/best.pt", {
            "--softassign-head": None, "--softassign-layers": 3,
            "--vehicle-cond-edge-head": None, "--use-attn": None, "--add-demand-weights": None,
        }),
        ("fpin_ab_n60_k7", "fpin", "cvrp_60_uniform/train/fpin/fpin_ab_n60_k7_unf_attn_2026-06-02_21-59-52/checkpoints/best.pt", {
            "--softassign-head": None, "--softassign-layers": 3, "--softassign-log-domain": None,
            "--vehicle-cond-edge-head": None, "--use-attn": None, "--add-demand-weights": None,
        }),
        ("fpin_ac_n60_k7", "fpin", "cvrp_60_uniform/train/fpin/fpin_ac_n60_k7_unf_attn_2026-06-02_16-49-39/checkpoints/best.pt", {
            "--softassign-head": None, "--softassign-layers": 3, "--vcount-aux-head": None,
            "--vehicle-cond-edge-head": None, "--use-attn": None, "--add-demand-weights": None,
        }),
    ],
}


def q(value: str) -> str:
    return shlex.quote(value)


def bool_str(value: bool) -> str:
    return "True" if value else "False"


def scalar_str(value) -> str:
    if isinstance(value, bool):
        return bool_str(value)
    return str(value)


def local_batch_size(scale: Scale, budget: Budget) -> int:
    return budget.local_batch_n20 if scale.graph_size == 20 else budget.local_batch_n60


def data_var_name(scale: Scale) -> str:
    return f"DATA_{scale.key.upper()}"


def train_dir(profile: Profile, stamp: str, budget: Budget, scale: Scale, variant_key: str) -> str:
    return f"{profile.output_root}/ablation_audit_{stamp}/{budget.name}/{scale.key}/{variant_key}"


def train_set_dir(profile: Profile, scale: Scale) -> str:
    return f"{profile.train_target_root}/{scale.train_folder}"


def maybe_target_sync_commands(profile: Profile, scale: Scale) -> List[str]:
    if profile.name != "local":
        return []
    local_dir = Path(train_set_dir(profile, scale))
    if local_dir.exists():
        return []
    return [
        f"mkdir -p {q(str(local_dir))}",
        "rsync -av --progress \\",
        f"  u25702@glogin-gpu.hpc.gwdg.de:{REMOTE_ROOT}/data/train_data/cvrp/uniform/targets/fc_hgs_clean/{scale.train_folder}/ \\",
        f"  {q(str(local_dir))}/",
    ]


def data_var_preamble(profile: Profile, scale: Scale) -> str:
    return f'{data_var_name(scale)}=$(ls {q(train_set_dir(profile, scale))}/{scale.target_glob} 2>/dev/null | head -1)'


def existing_ckpt_path(profile: Profile, rel_or_abs: str) -> str:
    if rel_or_abs.startswith("models/"):
        return f"{profile.repo}/{rel_or_abs}"
    return f"{profile.ckpt_existing_root}/{rel_or_abs}"


def append_heatmap_flag(parts: List[str], cli_key: str, value) -> None:
    if isinstance(value, bool):
        parts.append(cli_key if value else "--no-" + cli_key[2:])
    else:
        parts.append(f"{cli_key} {scalar_str(value)}")


def wrap_skip_existing(target_path: str, command: str) -> str:
    return (
        f"if [ -f {q(target_path)} ]; then\n"
        f"  echo \"[skip-existing] {target_path}\"\n"
        f"else\n"
        f"  {command}\n"
        f"fi"
    )


def build_heatmap_command(profile: Profile, stamp: str, scale: Scale, tag: str, family: str, ckpt_rel: str, extra_flags: Dict[str, object]) -> tuple[str, str]:
    out_dir = f"{profile.eval_root}/heatmap_audit_{stamp}"
    out_json = f"{out_dir}/{tag}.json"
    parts = [
        "PYTHONPATH=. python experiments/heatmap_failure_audit.py",
        f"--model-family {family}",
        f"--ckpt {q(existing_ckpt_path(profile, ckpt_rel))}",
        f'--data-npz "${data_var_name(scale)}"',
        f"--graph-size {scale.graph_size}",
        f"--fleet-size {scale.fleet_size}",
        "--limit 256",
        "--batch-size 32",
        "--device cuda",
        f"--tag {tag}",
    ]
    for k, v in extra_flags.items():
        if v is None:
            parts.append(k)
        else:
            append_heatmap_flag(parts, k, v)
    parts.append(f"--out-json {q(out_json)}")
    return " \\\n  ".join(parts), out_json


def train_batch_size(profile: Profile, scale: Scale, budget: Budget) -> int:
    if profile.name == "gwdg":
        return budget.gwdg_batch
    return local_batch_size(scale, budget)


def build_train_overrides(profile: Profile, scale: Scale, budget: Budget, variant_overrides: Dict[str, object], run_name: str) -> Dict[str, object]:
    merged = dict(BASE_TRAIN_FLAGS)
    merged.update({
        "model_cfg.model_args.max_fleet_length": scale.fleet_size,
        "train_cfg.batch_size": train_batch_size(profile, scale, budget),
        "train_cfg.n_epochs": budget.n_epochs,
        "train_cfg.lr": 0.0001,
        "train_cfg.nr_train_samples": budget.nr_train_samples,
        "train_cfg.checkpoint_epochs": budget.checkpoint_epochs,
        "train_cfg.run_name": run_name,
    })
    merged.update(variant_overrides)
    return merged


def build_train_command(profile: Profile, stamp: str, scale: Scale, budget: Budget, variant_key: str, variant_overrides: Dict[str, object]) -> tuple[str, str]:
    run_name = f"{variant_key}_{scale.key}_{budget.name}"
    run_dir = train_dir(profile, stamp, budget, scale, variant_key)
    overrides = build_train_overrides(profile, scale, budget, variant_overrides, run_name)
    parts = [
        "PYTHONPATH=. python run_fpin.py meta=train_base",
        f"env={scale.env}",
        "model=fpin",
        "cuda=True",
        "hydra.job.chdir=true",
        f"hydra.run.dir={q(run_dir)}",
        f"fixed_train_set={q(train_set_dir(profile, scale) + '/')}",
    ]
    for key, value in overrides.items():
        parts.append(f"{key}={scalar_str(value)}")
    return " \\\n  ".join(parts), f"{run_dir}/checkpoints/best.pt"


def build_eval_command(profile: Profile, stamp: str, scale: Scale, budget: Budget, variant_key: str, variant_overrides: Dict[str, object]) -> str:
    run_dir = train_dir(profile, stamp, budget, scale, variant_key)
    ckpt = f"{run_dir}/checkpoints/best.pt"
    parts = [
        "python run_fpin.py",
        f"env={scale.env}",
        "model=fpin",
        "test_cfg.time_limit=8",
        "eval_opts_cfg.post_process=False",
        "model_cfg.model_args.fleet_in_dim=260",
        f"model_cfg.model_args.max_fleet_length={scale.fleet_size}",
        "eval_opts_cfg.decode_v_assign_type=assign_transition",
        f"eval_opts_cfg.nr_vehicles_eval={scale.fleet_size}",
        f"test_cfg.dataset_size={scale.test_dataset_size}",
        f"data_file_path={q(f'{profile.repo}/{scale.test_relpath}')}",
        f"checkpoint_load_path={q(ckpt)}",
        "test_cfg.add_ls=False",
        "number_runs=1",
    ]
    eval_flags = dict(BASE_TRAIN_FLAGS)
    eval_flags.update({
        "model_cfg.model_args.max_fleet_length": scale.fleet_size,
    })
    eval_flags.update(variant_overrides)
    for key, value in eval_flags.items():
        if key.startswith("loss_cfg.") or key.startswith("train_cfg."):
            continue
        parts.append(f"{key}={scalar_str(value)}")
    return " \\\n  ".join(parts) + " 2>&1 | grep -E \"Average vehicle cost|Fleet violation|Feasible\""


def build_posttrain_heatmap_command(profile: Profile, stamp: str, scale: Scale, budget: Budget, variant_key: str, variant_overrides: Dict[str, object]) -> tuple[str, str]:
    run_dir = train_dir(profile, stamp, budget, scale, variant_key)
    out_dir = f"{profile.eval_root}/heatmap_audit_{stamp}/{budget.name}/{scale.key}"
    out_json = f"{out_dir}/{variant_key}.json"
    parts = [
        "PYTHONPATH=. python experiments/heatmap_failure_audit.py",
        "--model-family fpin",
        f"--ckpt {q(f'{run_dir}/checkpoints/best.pt')}",
        f'--data-npz "${data_var_name(scale)}"',
        f"--graph-size {scale.graph_size}",
        f"--fleet-size {scale.fleet_size}",
        "--limit 256",
        "--batch-size 32",
        "--device cuda",
        f"--tag {variant_key}_{scale.key}_{budget.name}",
    ]
    heatmap_flags = dict(BASE_TRAIN_FLAGS)
    heatmap_flags.update(variant_overrides)
    for key, value in heatmap_flags.items():
        if not key.startswith("model_cfg.model_args."):
            continue
        arg_name = key.split("model_cfg.model_args.", 1)[1]
        if arg_name == "max_fleet_length":
            continue
        cli_key = "--" + arg_name.replace("_", "-")
        append_heatmap_flag(parts, cli_key, value)
    parts.append(f"--out-json {q(out_json)}")
    return " \\\n  ".join(parts), out_json


def provenance_commands(profile: Profile) -> List[str]:
    return [
        "# Publication-code provenance checks",
        "rg -n \"n_SOFTlayers|self\\.softassign|MTSPSoftassign\" \\",
        f"  {q(profile.repo + '/sup-vrp-copy/ICLR_2022/code_iclr22/train.py')} \\",
        f"  {q(profile.repo + '/sup-vrp-copy/ICLR_2022/code_iclr22/VRPModel_attn.py')} \\",
        f"  {q(profile.repo + '/models/PIMold/src/train1.py')} \\",
        f"  {q(profile.repo + '/models/PIMold/src/train_soft.py')} \\",
        f"  {q(profile.repo + '/models/PIMold/src/eval1.py')} \\",
        f"  {q(profile.repo + '/models/PIMold/src/eval_soft.py')}",
    ]


def render_section(title: str, lines: Iterable[str]) -> str:
    body = "\n".join(lines)
    return f"\n# {title}\n{body}\n"


def write_sbatch_scripts(profile: Profile, stamp: str, scale_keys: List[str], budget: Budget, variant_keys: List[str], output_dir: Path, do_submit: bool) -> List[str]:
    submitted: List[str] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    key_to_variant = {key: (label, overrides) for key, label, overrides in TRAIN_VARIANTS}
    for scale_key in scale_keys:
        scale = SCALES[scale_key]
        for variant_key in variant_keys:
            label, overrides = key_to_variant[variant_key]
            run_name = f"{variant_key}_{scale.key}_{budget.name}"
            run_dir = train_dir(profile, stamp, budget, scale, variant_key)
            log_stub = f"ABL_{variant_key.upper()}_{scale.key.upper()}_{budget.name.upper()}"
            script_path = output_dir / f"{run_name}.sbatch.sh"
            parts = [
                "#!/bin/bash",
                f"#SBATCH --job-name={log_stub[:60]}",
                "#SBATCH --partition=grete:shared",
                "#SBATCH --nodes=1",
                "#SBATCH --ntasks-per-node=1",
                "#SBATCH -G A100:1",
                f"#SBATCH --time={budget.sbatch_time}",
                "#SBATCH --requeue",
                "#SBATCH --mail-type=FAIL",
                f"#SBATCH --output={GWDG_LOG_ROOT}/{run_name}_%j.log",
                f"#SBATCH --error={GWDG_LOG_ROOT}/{run_name}_%j.err",
                "set -eo pipefail",
                f"mkdir -p {q(GWDG_LOG_ROOT)}",
                'source "${HOME}/miniconda3/etc/profile.d/conda.sh"',
                "conda activate l2o_py310",
                "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
                f"export FPIN_OUTPUT_ROOT={q(GWDG_OUTPUT_ROOT)}",
                'mkdir -p "${FPIN_OUTPUT_ROOT}"',
                f"cd {q(profile.repo)}",
            ]
            cmd, _ = build_train_command(profile, stamp, scale, budget, variant_key, overrides)
            cmd = cmd.replace("PYTHONPATH=. python", "srun env PYTHONPATH=. python", 1)
            parts.append(cmd)
            script_path.write_text("\n".join(parts) + "\n")
            submitted.append(str(script_path))
            if do_submit:
                subprocess.run(["sbatch", str(script_path)], check=True)
    return submitted


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=sorted(PROFILES), default="local")
    ap.add_argument("--budget", choices=sorted(BUDGETS), default="medium")
    ap.add_argument("--scales", nargs="+", choices=sorted(SCALES), default=["n20_k3", "n50_k7"])
    ap.add_argument("--phase", choices=["all", "heatmap", "train"], default="all")
    ap.add_argument("--variants", nargs="*", default=[key for key, _, _ in TRAIN_VARIANTS])
    ap.add_argument("--stamp", default="20260603")
    ap.add_argument("--write-dir", default="")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--skip-existing", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    profile = PROFILES[args.profile]
    budget = BUDGETS[args.budget]

    missing_variants = [v for v in args.variants if v not in {k for k, _, _ in TRAIN_VARIANTS}]
    if missing_variants:
        raise SystemExit(f"Unknown variants: {missing_variants}")

    lines: List[str] = []
    lines.append(f"conda activate l2o_py310")
    lines.append(f"cd {q(profile.repo)}")

    for scale_key in args.scales:
        lines.extend(maybe_target_sync_commands(profile, SCALES[scale_key]))

    if args.phase in ("all", "heatmap"):
        lines.append("")
        lines.extend(provenance_commands(profile))
        for scale_key in args.scales:
            scale = SCALES[scale_key]
            lines.append("")
            lines.append(f"# Data preamble for {scale.key}")
            lines.append(data_var_preamble(profile, scale))
            lines.append(f"mkdir -p {q(f'{profile.eval_root}/heatmap_audit_{args.stamp}')}")
            for tag, family, ckpt_rel, extra_flags in EXISTING_HEATMAP_CKPTS.get(scale.key, []):
                lines.append("")
                cmd, out_json = build_heatmap_command(profile, args.stamp, scale, tag, family, ckpt_rel, extra_flags)
                lines.append(wrap_skip_existing(out_json, cmd) if args.skip_existing else cmd)

    if args.phase in ("all", "train"):
        lines.append("")
        lines.append(f"mkdir -p {q(f'{profile.output_root}/ablation_audit_{args.stamp}/{budget.name}')}")
        lines.append(f"mkdir -p {q(f'{profile.eval_root}/heatmap_audit_{args.stamp}/{budget.name}')}")
        variant_map = {key: (label, overrides) for key, label, overrides in TRAIN_VARIANTS}
        for scale_key in args.scales:
            scale = SCALES[scale_key]
            lines.append("")
            lines.append(f"# Training ablations for {scale.key} ({budget.name})")
            lines.append(data_var_preamble(profile, scale))
            lines.append(f"mkdir -p {q(f'{profile.eval_root}/heatmap_audit_{args.stamp}/{budget.name}/{scale.key}')}")
            for variant_key in args.variants:
                label, overrides = variant_map[variant_key]
                lines.append("")
                lines.append(f"# {label}")
                train_cmd, train_ckpt = build_train_command(profile, args.stamp, scale, budget, variant_key, overrides)
                lines.append(wrap_skip_existing(train_ckpt, train_cmd) if args.skip_existing else train_cmd)
                lines.append("")
                lines.append(f"# Post-train raw eval for {variant_key}")
                lines.append(build_eval_command(profile, args.stamp, scale, budget, variant_key, overrides))
                lines.append("")
                lines.append(f"# Post-train heatmap audit for {variant_key}")
                heat_cmd, heat_json = build_posttrain_heatmap_command(profile, args.stamp, scale, budget, variant_key, overrides)
                lines.append(wrap_skip_existing(heat_json, heat_cmd) if args.skip_existing else heat_cmd)

    text = "\n".join(lines).strip() + "\n"
    print(text)

    if args.write_dir:
        out_dir = Path(args.write_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"ablation_audit_{args.profile}_{args.budget}_{args.stamp}.sh").write_text(text)

    if args.submit:
        if args.profile != "gwdg":
            raise SystemExit("--submit is only supported with --profile gwdg")
        if args.phase not in ("all", "train"):
            raise SystemExit("--submit only submits training jobs")
        submit_dir = Path(args.write_dir) if args.write_dir else Path(profile.repo) / "batch_run_outputs" / f"ablation_audit_submit_{args.stamp}_{budget.name}"
        scripts = write_sbatch_scripts(profile, args.stamp, args.scales, budget, args.variants, submit_dir, do_submit=True)
        print("\n# Submitted sbatch scripts")
        for path in scripts:
            print(path)


if __name__ == "__main__":
    main()
