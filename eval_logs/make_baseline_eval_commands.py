from pathlib import Path

# -----------------------------------------------------------------------------
# Global defaults
# -----------------------------------------------------------------------------
DATASET_SIZE_SMALL = 1000
DATASET_SIZE_60 = 130

TIME_LIMIT_NEURAL = 8
TIME_LIMIT_SEARCH = 8

INCLUDE_NEURAL_BASELINES = True
INCLUDE_SEARCH_BASELINES = True
INCLUDE_NEUROLKH_ONLY_FOR_N100 = True

EXPTS = [
    dict(n=20,  k=3,  data="data/test_data/cvrp/uniform/cvrp20/fc-cvrp_k3_seed213298_size1000.pt", dataset_size=1000),
    dict(n=20,  k=4,  data="data/test_data/cvrp/uniform/cvrp20/fc-cvrp_k4_seed213298_size1000.pt", dataset_size=1000),
    dict(n=50,  k=6,  data="data/test_data/cvrp/uniform/cvrp50/fc-cvrp_k6_seed213298_size1000.pt", dataset_size=1000),
    dict(n=50,  k=7,  data="data/test_data/cvrp/uniform/cvrp50/fc-cvrp_k7_seed213298_size1000.pt", dataset_size=1000),
    dict(n=60,  k=6,  data="data/test_data/cvrp/uniform/cvrp60/fc-cvrp_k6_seed213298_size130.pt", dataset_size=130),
    dict(n=60,  k=7,  data="data/test_data/cvrp/uniform/cvrp60/fc-cvrp_k7_seed213298_size130.pt", dataset_size=130),
    dict(n=100, k=9,  data="data/test_data/cvrp/uniform/cvrp100/fc-cvrp_k9_seed213298_size1000.pt", dataset_size=1000),
    dict(n=100, k=10, data="data/test_data/cvrp/uniform/cvrp100/fc-cvrp_k10_seed213298_size1000.pt", dataset_size=1000),
]


def base_args(n: int, k: int, data: str, dataset_size: int, time_limit: int):
    env = f"cvrp{n}_unf"
    return [
        f"env={env}",
        f"test_cfg.time_limit={time_limit}",
        f"data_file_path={data}",
        f"test_cfg.dataset_size={dataset_size}",
        f"env_kwargs.sampling_args.k={k}",
    ]


def make_am_cmd(n: int, k: int, data: str, dataset_size: int, decode_type: str) -> str:
    parts = ["python run_AM.py"] + base_args(n, k, data, dataset_size, TIME_LIMIT_NEURAL) + [
        f"test_cfg.decode_type={decode_type}"
    ]
    if decode_type == "sample":
        parts.append("test_cfg.sample_size=1280")
    return " ".join(parts)


def make_bq_cmd(n: int, k: int, data: str, dataset_size: int) -> str:
    parts = ["python run_BQ.py"] + base_args(n, k, data, dataset_size, TIME_LIMIT_NEURAL)
    return " ".join(parts)


def make_pomo_cmd(n: int, k: int, data: str, dataset_size: int, pomo_size: int) -> str:
    parts = ["python run_POMO.py"] + base_args(n, k, data, dataset_size, TIME_LIMIT_NEURAL) + [
        f"tester_cfg.pomo_size={pomo_size}"
    ]
    return " ".join(parts)


def make_parco_cmd(n: int, k: int, data: str, dataset_size: int) -> str:
    parts = ["python run_PARCO.py"] + base_args(n, k, data, dataset_size, TIME_LIMIT_NEURAL)
    return " ".join(parts)


def make_neurolkh_cmd(n: int, k: int, data: str, dataset_size: int, policy: str) -> str:
    parts = ["python run_NeuroLKH.py", f"policy={policy}"] + base_args(n, k, data, dataset_size, TIME_LIMIT_SEARCH)
    return " ".join(parts)


def make_hgs_cmd(n: int, k: int, data: str, dataset_size: int) -> str:
    env = f"cvrp{n}_unf"
    parts = [
        "python run_HGS.py",
        f"env={env}",
        f"test_cfg.time_limit={TIME_LIMIT_SEARCH}",
        f"policy_cfg.fleet_size={k}",
        f"data_file_path={data}",
        f"test_cfg.dataset_size={dataset_size}",
    ]
    return " ".join(parts)


def should_include_neurolkh(n: int) -> bool:
    if not INCLUDE_NEUROLKH_ONLY_FOR_N100:
        return True
    return n == 100


def main():
    lines = []

    for ex in EXPTS:
        n, k, data, dataset_size = ex["n"], ex["k"], ex["data"], ex["dataset_size"]

        lines.append(f"## n{n}_k{k}_uniform")
        lines.append("")

        if INCLUDE_NEURAL_BASELINES:
            lines.append("### Neural baselines")
            lines.append(make_am_cmd(n, k, data, dataset_size, "greedy"))
            lines.append(make_am_cmd(n, k, data, dataset_size, "sample"))
            lines.append(make_bq_cmd(n, k, data, dataset_size))
            lines.append(make_pomo_cmd(n, k, data, dataset_size, 1))
            lines.append(make_pomo_cmd(n, k, data, dataset_size, 20))
            lines.append(make_parco_cmd(n, k, data, dataset_size))
            lines.append("")

        if INCLUDE_SEARCH_BASELINES:
            lines.append("### Search-based baselines")

            if should_include_neurolkh(n):
                lines.append(make_neurolkh_cmd(n, k, data, dataset_size, "neuro_lkh"))
            else:
                lines.append(f"# skipped NeuroLKH for n={n} (only stable for N=100 so far)")

            lines.append(make_neurolkh_cmd(n, k, data, dataset_size, "lkh"))
            lines.append(make_hgs_cmd(n, k, data, dataset_size))
            lines.append("")

    out = Path("run_commands_baselines.md")
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()