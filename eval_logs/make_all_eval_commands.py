from pathlib import Path

# -----------------------------------------------------------------------------
# Global defaults
# -----------------------------------------------------------------------------
NUMBER_RUNS = 1
FLEET_IN_DIM = 260
USE_ATTN = True

TIME_LIMITS = [5, 8]

INCLUDE_FPIN = True
INCLUDE_BASELINES = True
INCLUDE_NEURAL_BASELINES = True
INCLUDE_SEARCH_BASELINES = True
INCLUDE_NEUROLKH_ONLY_FOR_N100 = True

# Optional pruning knobs
INCLUDE_ASSIGNMENT_DECODER = True
ASSIGNMENT_ONLY_FOR_N60 = False
INCLUDE_LS_FOR_ASSIGNMENT = True

# -----------------------------------------------------------------------------
# F-PIN experiments
# -----------------------------------------------------------------------------
FPIN_EXPTS = [
    dict(
        name="n20_k3_uniform",
        dist="uniform",
        n=20, eval_k=3, model_k=3, dataset_size=1000,
        ckpt="models/PIM/PIM/logs/ckpts/uniform_20/best-ep-n20_k3_unf_attn.pt",
        data="data/test_data/cvrp/uniform/cvrp20/fc-cvrp_k3_seed213298_size1000.pt",
    ),
    dict(
        name="n20_k4_uniform",
        dist="uniform",
        n=20, eval_k=4, model_k=4, dataset_size=1000,
        ckpt="models/PIM/PIM/logs/ckpts/uniform_20/best-ep-n20_k4_unf_attn.pt",
        data="data/test_data/cvrp/uniform/cvrp20/fc-cvrp_k4_seed213298_size1000.pt",
    ),
    dict(
        name="n50_k6_uniform",
        dist="uniform",
        n=50, eval_k=6, model_k=6, dataset_size=1000,
        ckpt="models/PIM/PIM/logs/ckpts/uniform_50/best-ep-n50_k6_unf_attn.pt",
        data="data/test_data/cvrp/uniform/cvrp50/fc-cvrp_k6_seed213298_size1000.pt",
    ),
    dict(
        name="n50_k7_uniform",
        dist="uniform",
        n=50, eval_k=7, model_k=7, dataset_size=1000,
        ckpt="models/PIM/PIM/logs/ckpts/uniform_50/best-ep-n50_k7_unf_attn.pt",
        data="data/test_data/cvrp/uniform/cvrp50/fc-cvrp_k7_seed213298_size1000.pt",
    ),
    dict(
        name="n60_k6_uniform",
        dist="uniform",
        n=60, eval_k=6, model_k=6, dataset_size=130,
        ckpt="models/PIM/PIM/logs/ckpts/uniform_60/best-ep-n60_k6_unf_attn.pt",
        data="data/test_data/cvrp/uniform/cvrp60/fc-cvrp_k6_seed213298_size130.pt",
    ),
    dict(
        name="n60_k7_uniform",
        dist="uniform",
        n=60, eval_k=7, model_k=7, dataset_size=130,
        ckpt="models/PIM/PIM/logs/ckpts/uniform_60/best-ep-n60_k7_unf_attn.pt",
        data="data/test_data/cvrp/uniform/cvrp60/fc-cvrp_k7_seed213298_size130.pt",
    ),
    dict(
        name="n60_k9_uniform",
        dist="uniform",
        n=60, eval_k=9, model_k=9, dataset_size=130,
        ckpt="models/PIM/PIM/logs/ckpts/uniform_60/best-ep-n60_k9_unf_attn.pt",
        data="data/test_data/cvrp/uniform/cvrp60/fc-cvrp_k9_seed213298_size130.pt",
    ),
    dict(
        name="n100_k9_uniform",
        dist="uniform",
        n=100, eval_k=9, model_k=9, dataset_size=1000,
        ckpt="models/PIM/PIM/logs/ckpts/uniform_100/best-ep-n100_k9_unf_attn.pt",
        data="data/test_data/cvrp/uniform/cvrp100/fc-cvrp_k9_seed213298_size1000.pt",
    ),
    dict(
        name="n100_k10_uniform",
        dist="uniform",
        n=100, eval_k=10, model_k=10, dataset_size=1000,
        ckpt="models/PIM/PIM/logs/ckpts/uniform_100/best-ep-50-n100_k10_unf_attn.pt",
        data="data/test_data/cvrp/uniform/cvrp100/fc-cvrp_k10_seed213298_size1000.pt",
    ),

    # -------------------------------------------------------------------------
    # Explosion distribution (new)
    # -------------------------------------------------------------------------
    dict(
        name="n60_k7_explosion",
        dist="explosion",
        n=60, eval_k=7, model_k=7, dataset_size=130,
        ckpt="models/PIM/PIM/logs/ckpts/explosion_60/best-ep-n60_k7_exp_attn.pt",
        data="data/test_data/cvrp/explosion/cvrp60/fc-cvrp_k7_seed213298_size130.pt",
    ),
    dict(
        name="n60_k9_explosion",
        dist="explosion",
        n=60, eval_k=9, model_k=9, dataset_size=130,
        ckpt="models/PIM/PIM/logs/ckpts/explosion_60/best-ep-n60_k9_exp_attn.pt",
        data="data/test_data/cvrp/explosion/cvrp60/fc-cvrp_k9_seed213298_size130.pt",
    ),
]

# -----------------------------------------------------------------------------
# Baseline experiments
# -----------------------------------------------------------------------------
BASELINE_EXPTS = [
    dict(dist="uniform", n=20,  k=3,  data="data/test_data/cvrp/uniform/cvrp20/fc-cvrp_k3_seed213298_size1000.pt", dataset_size=1000),
    dict(dist="uniform", n=20,  k=4,  data="data/test_data/cvrp/uniform/cvrp20/fc-cvrp_k4_seed213298_size1000.pt", dataset_size=1000),
    dict(dist="uniform", n=50,  k=6,  data="data/test_data/cvrp/uniform/cvrp50/fc-cvrp_k6_seed213298_size1000.pt", dataset_size=1000),
    dict(dist="uniform", n=50,  k=7,  data="data/test_data/cvrp/uniform/cvrp50/fc-cvrp_k7_seed213298_size1000.pt", dataset_size=1000),
    dict(dist="uniform", n=60,  k=6,  data="data/test_data/cvrp/uniform/cvrp60/fc-cvrp_k6_seed213298_size130.pt", dataset_size=130),
    dict(dist="uniform", n=60,  k=7,  data="data/test_data/cvrp/uniform/cvrp60/fc-cvrp_k7_seed213298_size130.pt", dataset_size=130),
    dict(dist="uniform", n=100, k=9,  data="data/test_data/cvrp/uniform/cvrp100/fc-cvrp_k9_seed213298_size1000.pt", dataset_size=1000),
    dict(dist="uniform", n=100, k=10, data="data/test_data/cvrp/uniform/cvrp100/fc-cvrp_k10_seed213298_size1000.pt", dataset_size=1000),

    # Explosion: HGS + LKH only
    dict(dist="explosion", n=60, k=7, data="data/test_data/cvrp/explosion/cvrp60/fc-cvrp_k7_seed213298_size130.pt", dataset_size=130),
    dict(dist="explosion", n=60, k=9, data="data/test_data/cvrp/explosion/cvrp60/fc-cvrp_k9_seed213298_size130.pt", dataset_size=130),
]

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def section(title: str) -> list[str]:
    return [f"## {title}", ""]

def should_include_neurolkh(n: int) -> bool:
    if not INCLUDE_NEUROLKH_ONLY_FOR_N100:
        return True
    return n == 100

def is_explosion_experiment(ex: dict) -> bool:
    return ex.get("dist", "") == "explosion"

# -----------------------------------------------------------------------------
# F-PIN command builders
# -----------------------------------------------------------------------------
def make_fpin_cmd(
    n: int,
    eval_k: int,
    model_k: int,
    ckpt: str,
    data: str,
    dataset_size: int,
    time_limit: int,
    add_ls: bool,
    decode_mode: str,
) -> str:
    """
    decode_mode:
      - 'split'       -> giant_tour_split=True,  decode_vehicle_assignment=False
      - 'assignment'  -> giant_tour_split=False, decode_vehicle_assignment=True
    """
    #env = f"cvrp{n}_unf"
    env = f"cvrp{n}_exp" if "explosion" in data else f"cvrp{n}_unf"

    if decode_mode == "split":
        giant_tour_split = "True"
        decode_vehicle_assignment = "False"
    elif decode_mode == "assignment":
        giant_tour_split = "False"
        decode_vehicle_assignment = "True"
    else:
        raise ValueError(f"Unknown decode_mode: {decode_mode}")

    parts = [
        "python run_PIM.py",
        f"env={env}",
        f"test_cfg.time_limit={time_limit}",
        "eval_opts_cfg.post_process=False",
        f"model_cfg.model_args.fleet_in_dim={FLEET_IN_DIM}",
        f"model_cfg.model_args.max_fleet_length={model_k}",
        f"model_cfg.model_args.use_attn={'True' if USE_ATTN else 'False'}",
        f"eval_opts_cfg.nr_vehicles_eval={eval_k}",
        f"eval_opts_cfg.giant_tour_split={giant_tour_split}",
        f"eval_opts_cfg.decode_vehicle_assignment={decode_vehicle_assignment}",
        f"test_cfg.dataset_size={dataset_size}",
        f"data_file_path={data}",
        f"checkpoint_load_path={ckpt}",
        f"test_cfg.add_ls={'True' if add_ls else 'False'}",
        f"number_runs={NUMBER_RUNS}",
    ]
    return " ".join(parts)

# -----------------------------------------------------------------------------
# Baseline command builders
# -----------------------------------------------------------------------------
def base_args(n: int, k: int, data: str, dataset_size: int, time_limit: int):
    env = f"cvrp{n}_exp" if "explosion" in data else f"cvrp{n}_unf"
    return [
        f"env={env}",
        f"test_cfg.time_limit={time_limit}",
        f"data_file_path={data}",
        f"test_cfg.dataset_size={dataset_size}",
        f"env_kwargs.sampling_args.k={k}",
    ]

def make_am_cmd(n: int, k: int, data: str, dataset_size: int, time_limit: int, decode_type: str) -> str:
    parts = ["python run_AM.py"] + base_args(n, k, data, dataset_size, time_limit) + [
        f"test_cfg.decode_type={decode_type}"
    ]
    if decode_type == "sample":
        parts.append("test_cfg.sample_size=1280")
    return " ".join(parts)

def make_bq_cmd(n: int, k: int, data: str, dataset_size: int, time_limit: int) -> str:
    parts = ["python run_BQ.py"] + base_args(n, k, data, dataset_size, time_limit)
    return " ".join(parts)

def make_pomo_cmd(n: int, k: int, data: str, dataset_size: int, time_limit: int, pomo_size: int) -> str:
    parts = ["python run_POMO.py"] + base_args(n, k, data, dataset_size, time_limit) + [
        f"tester_cfg.pomo_size={pomo_size}"
    ]
    return " ".join(parts)

def make_parco_cmd(n: int, k: int, data: str, dataset_size: int, time_limit: int) -> str:
    parts = ["python run_PARCO.py"] + base_args(n, k, data, dataset_size, time_limit)
    return " ".join(parts)

def make_neurolkh_cmd(n: int, k: int, data: str, dataset_size: int, time_limit: int, policy: str) -> str:
    parts = ["python run_NeuroLKH.py", f"policy={policy}"] + base_args(n, k, data, dataset_size, time_limit)
    return " ".join(parts)

def make_hgs_cmd(n: int, k: int, data: str, dataset_size: int, time_limit: int) -> str:
    env = f"cvrp{n}_exp" if "explosion" in data else f"cvrp{n}_unf"
    parts = [
        "python run_HGS.py",
        f"env={env}",
        f"test_cfg.time_limit={time_limit}",
        f"policy_cfg.fleet_size={k}",
        f"data_file_path={data}",
        f"test_cfg.dataset_size={dataset_size}",
    ]
    return " ".join(parts)

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    lines = []
    lines += section("Merged evaluation commands")

    # -------------------------------------------------------------------------
    # F-PIN
    # -------------------------------------------------------------------------
    if INCLUDE_FPIN:
        for ex in FPIN_EXPTS:
            lines += section(
                f"F-PIN | {ex['name']} "
                f"(dist={ex['dist']}, n={ex['n']}, eval_k={ex['eval_k']}, model_k={ex['model_k']}, ds={ex['dataset_size']})"
            )

            for time_limit in TIME_LIMITS:
                lines.append(f"### F-PIN | time_limit={time_limit}")
                lines.append("")

                # split decoder
                lines.append("# split decoder | add_ls=False")
                lines.append(make_fpin_cmd(
                    n=ex["n"],
                    eval_k=ex["eval_k"],
                    model_k=ex["model_k"],
                    ckpt=ex["ckpt"],
                    data=ex["data"],
                    dataset_size=ex["dataset_size"],
                    time_limit=time_limit,
                    add_ls=False,
                    decode_mode="split",
                ))
                lines.append("")

                lines.append("# split decoder | add_ls=True")
                lines.append(make_fpin_cmd(
                    n=ex["n"],
                    eval_k=ex["eval_k"],
                    model_k=ex["model_k"],
                    ckpt=ex["ckpt"],
                    data=ex["data"],
                    dataset_size=ex["dataset_size"],
                    time_limit=time_limit,
                    add_ls=True,
                    decode_mode="split",
                ))
                lines.append("")

                # assignment decoder
                include_assignment = INCLUDE_ASSIGNMENT_DECODER
                if ASSIGNMENT_ONLY_FOR_N60 and ex["n"] != 60:
                    include_assignment = False

                if include_assignment:
                    lines.append("# assignment decoder | add_ls=False")
                    lines.append(make_fpin_cmd(
                        n=ex["n"],
                        eval_k=ex["eval_k"],
                        model_k=ex["model_k"],
                        ckpt=ex["ckpt"],
                        data=ex["data"],
                        dataset_size=ex["dataset_size"],
                        time_limit=time_limit,
                        add_ls=False,
                        decode_mode="assignment",
                    ))
                    lines.append("")

                    if INCLUDE_LS_FOR_ASSIGNMENT:
                        lines.append("# assignment decoder | add_ls=True")
                        lines.append(make_fpin_cmd(
                            n=ex["n"],
                            eval_k=ex["eval_k"],
                            model_k=ex["model_k"],
                            ckpt=ex["ckpt"],
                            data=ex["data"],
                            dataset_size=ex["dataset_size"],
                            time_limit=time_limit,
                            add_ls=True,
                            decode_mode="assignment",
                        ))
                        lines.append("")

            lines.append("")

    # -------------------------------------------------------------------------
    # Baselines
    # -------------------------------------------------------------------------
    if INCLUDE_BASELINES:
        for ex in BASELINE_EXPTS:
            n, k, data, dataset_size = ex["n"], ex["k"], ex["data"], ex["dataset_size"]
            dist = ex["dist"]

            lines += section(f"Baselines | dist={dist} | n{n}_k{k} (ds={dataset_size})")

            for time_limit in TIME_LIMITS:
                lines.append(f"### Baselines | time_limit={time_limit}")
                lines.append("")

                if is_explosion_experiment(ex):
                    # Explosion: HGS + LKH only
                    lines.append("#### Search-based baselines (explosion: HGS + LKH only)")
                    lines.append(make_neurolkh_cmd(n, k, data, dataset_size, time_limit, "lkh"))
                    lines.append(make_hgs_cmd(n, k, data, dataset_size, time_limit))
                    lines.append("")
                    continue

                if INCLUDE_NEURAL_BASELINES:
                    lines.append("#### Neural baselines")
                    lines.append(make_am_cmd(n, k, data, dataset_size, time_limit, "greedy"))
                    lines.append(make_am_cmd(n, k, data, dataset_size, time_limit, "sample"))
                    lines.append(make_bq_cmd(n, k, data, dataset_size, time_limit))
                    lines.append(make_pomo_cmd(n, k, data, dataset_size, time_limit, 1))
                    lines.append(make_pomo_cmd(n, k, data, dataset_size, time_limit, 20))
                    lines.append(make_parco_cmd(n, k, data, dataset_size, time_limit))
                    lines.append("")

                if INCLUDE_SEARCH_BASELINES:
                    lines.append("#### Search-based baselines")
                    if should_include_neurolkh(n):
                        lines.append(make_neurolkh_cmd(n, k, data, dataset_size, time_limit, "neuro_lkh"))
                    else:
                        lines.append(f"# skipped NeuroLKH for n={n} (only stable for N=100 so far)")
                    lines.append(make_neurolkh_cmd(n, k, data, dataset_size, time_limit, "lkh"))
                    lines.append(make_hgs_cmd(n, k, data, dataset_size, time_limit))
                    lines.append("")

            lines.append("")

    out = Path("run_commands_all_eval.md")
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")

if __name__ == "__main__":
    main()