from pathlib import Path

# global defaults
NUMBER_RUNS = 1
FLEET_IN_DIM = 260
USE_ATTN = True

# time budgets
TIME_LIMIT_CONSTR = 8
TIME_LIMIT_LS = 8

# experiments
# eval_k = number of vehicles used for evaluation / decoding
# model_k = max_fleet_length used to instantiate the model (must match checkpoint)
EXPTS = [
    dict(
        name="n20_k3",
        n=20, eval_k=3, model_k=3, dataset_size=1000,
        ckpt="models/PIM/PIM/logs/ckpts/uniform_20/best-ep-n20_k3_unf_attn.pt",
        data="data/test_data/cvrp/uniform/cvrp20/fc-cvrp_k3_seed213298_size1000.pt",
    ),
    dict(
        name="n20_k4",
        n=20, eval_k=4, model_k=4, dataset_size=1000,
        ckpt="models/PIM/PIM/logs/ckpts/uniform_20/best-ep-n20_k4_unf_attn.pt",
        data="data/test_data/cvrp/uniform/cvrp20/fc-cvrp_k4_seed213298_size1000.pt",
    ),
    dict(
        name="n50_k6",
        n=50, eval_k=6, model_k=6, dataset_size=1000,
        ckpt="models/PIM/PIM/logs/ckpts/uniform_50/best-ep-n50_k6_unf_attn.pt",
        data="data/test_data/cvrp/uniform/cvrp50/fc-cvrp_k6_seed213298_size1000.pt",
    ),
    dict(
        name="n50_k7",
        n=50, eval_k=7, model_k=7, dataset_size=1000,
        ckpt="models/PIM/PIM/logs/ckpts/uniform_50/best-ep-n50_k7_unf_attn.pt",
        data="data/test_data/cvrp/uniform/cvrp50/fc-cvrp_k7_seed213298_size1000.pt",
    ),
    dict(
        name="n60_k6",
        n=60, eval_k=6, model_k=6, dataset_size=130,
        ckpt="models/PIM/PIM/logs/ckpts/uniform_60/best-ep-n60_k6_unf_attn.pt",
        data="data/test_data/cvrp/uniform/cvrp60/fc-cvrp_k6_seed213298_size130.pt",
    ),
    dict(
        name="n60_k7",
        n=60, eval_k=7, model_k=7, dataset_size=130,
        ckpt="models/PIM/PIM/logs/ckpts/uniform_60/best-ep-n60_k7_unf_attn.pt",
        data="data/test_data/cvrp/uniform/cvrp60/fc-cvrp_k7_seed213298_size130.pt",
    ),
    # cross-eval: use k9 checkpoint on k7 dataset, but instantiate model with k=9
    dict(
        name="n60_k7_eval_with_k9_ckpt",
        n=60, eval_k=7, model_k=9, dataset_size=130,
        ckpt="models/PIM/PIM/logs/ckpts/uniform_60/best-ep-n60_k9_unf_attn.pt",
        data="data/test_data/cvrp/uniform/cvrp60/fc-cvrp_k7_seed213298_size130.pt",
    ),
    dict(
        name="n100_k9",
        n=100, eval_k=9, model_k=9, dataset_size=1000,
        ckpt="models/PIM/PIM/logs/ckpts/uniform_100/best-ep-n100_k9_unf_attn.pt",
        data="data/test_data/cvrp/uniform/cvrp100/fc-cvrp_k9_seed213298_size1000.pt",
    ),
    dict(
        name="n100_k10",
        n=100, eval_k=10, model_k=10, dataset_size=1000,
        ckpt="models/PIM/PIM/logs/ckpts/uniform_100/best-ep-50-n100_k10_unf_attn.pt",
        data="data/test_data/cvrp/uniform/cvrp100/fc-cvrp_k10_seed213298_size1000.pt",
    ),
]


def make_cmd(n: int, eval_k: int, model_k: int, ckpt: str, data: str, dataset_size: int, add_ls: bool) -> str:
    env = f"cvrp{n}_unf"
    time_limit = TIME_LIMIT_LS if add_ls else TIME_LIMIT_CONSTR

    parts = [
        "python run_PIM.py",
        f"env={env}",
        f"test_cfg.time_limit={time_limit}",
        "eval_opts_cfg.post_process=False",
        f"model_cfg.model_args.fleet_in_dim={FLEET_IN_DIM}",
        f"model_cfg.model_args.max_fleet_length={model_k}",
        f"model_cfg.model_args.use_attn={'True' if USE_ATTN else 'False'}",
        f"eval_opts_cfg.nr_vehicles_eval={eval_k}",
        f"test_cfg.dataset_size={dataset_size}",
        f"data_file_path={data}",
        f"checkpoint_load_path={ckpt}",
        f"test_cfg.add_ls={'True' if add_ls else 'False'}",
        f"number_runs={NUMBER_RUNS}",
    ]
    return " ".join(parts)


def main():
    lines = []
    for ex in EXPTS:
        title = (
            f"## {ex['name']} "
            f"(n={ex['n']}, eval_k={ex['eval_k']}, model_k={ex['model_k']})\n\n### F-PIN\n"
        )
        lines.append(title)

        lines.append(make_cmd(
            n=ex["n"],
            eval_k=ex["eval_k"],
            model_k=ex["model_k"],
            ckpt=ex["ckpt"],
            data=ex["data"],
            dataset_size=ex["dataset_size"],
            add_ls=False,
        ))
        lines.append("")

        lines.append(make_cmd(
            n=ex["n"],
            eval_k=ex["eval_k"],
            model_k=ex["model_k"],
            ckpt=ex["ckpt"],
            data=ex["data"],
            dataset_size=ex["dataset_size"],
            add_ls=True,
        ))
        lines.append("\n")

    out = Path("run_commands_fpin.md")
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()