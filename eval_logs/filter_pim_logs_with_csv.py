#!/usr/bin/env python3
"""
Parse raw PIM logs, generate:
- markdown summary
- csv summary

Optional:
- rename raw logs to inferred standardized names

Main purpose:
- recover run metadata from old raw PIM logs
- identify GT+Sp vs assign decoding
- identify pre/post decoder cutoff era
- compare repeated runs of same env/checkpoint/mode
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, List, Tuple


# ----------------------------
# Regex patterns
# ----------------------------

TS_SLASH = r"\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}"
TS_DASH = r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d+)?"

RE_MAIN_PROBLEM = re.compile(
    rf"(?P<ts>{TS_SLASH}) __main__: problem:\s*(?P<problem>\w+)"
)

RE_GRAPH_SIZE = re.compile(r"^graph_size:\s*(\d+)\s*$", re.MULTILINE)
RE_COORDS_DIST = re.compile(r"^coords_dist:\s*([A-Za-z0-9_]+)\s*$", re.MULTILINE)
RE_WEIGHTS_DIST = re.compile(r"^weights_dist:\s*([A-Za-z0-9_]+)\s*$", re.MULTILINE)
RE_DATA_FILE = re.compile(r"^data_file_path:\s*(.+?)\s*$", re.MULTILINE)
RE_CKPT = re.compile(r"^\s*checkpoint_load_path:\s*(.+?)\s*$", re.MULTILINE)

RE_TIME_LIMIT = re.compile(r"^\s*time_limit:\s*(\d+)\s*$", re.MULTILINE)
RE_ADD_LS = re.compile(r"^\s*add_ls:\s*(true|false)\s*$", re.MULTILINE)
RE_GIANT_TOUR_SPLIT = re.compile(r"^\s*giant_tour_split:\s*(true|false)\s*$", re.MULTILINE)
RE_DECODE_VEH_ASSIGN = re.compile(r"^\s*decode_vehicle_assignment:\s*(true|false)\s*$", re.MULTILINE)
RE_DECODE_V_ASSIGN_TYPE = re.compile(r"^\s*decode_v_assign_type:\s*([A-Za-z0-9_]+)\s*$", re.MULTILINE)
RE_NR_VEH_EVAL = re.compile(r"^\s*nr_vehicles_eval:\s*(\d+)\s*$", re.MULTILINE)

RE_AVG_BLOCK = re.compile(
    rf"(?P<ts>{TS_SLASH}) models\.runner_utils:\s*"
    r"\nAverage cost\s*:\s*(?P<avg_cost>.+?)\s*,\s*"
    r"\nAverage vehicle cost\s*:\s*(?P<avg_vcost>.+?)\s*,\s*"
    r"\nAverage number vehicles\s*:\s*(?P<avg_numveh>.+?)\s*,\s*"
    r"\nFeasible \(%\)\s*:\s*(?P<feasible>.+?)\s*,\s*"
    r"\nFleet violation \(%\)\s*:\s*(?P<fleetviol>.+?)\s*"
    r"\nAverage excess vehicles ΔK\s*:\s*(?P<delta_k>.+?)\s*$",
    re.MULTILINE | re.DOTALL,
)

RE_EXCEPTION_LINE = re.compile(
    r"(?P<etype>[A-Za-z_][A-Za-z0-9_]*Error|Exception)\s*:\s*(?P<msg>.*)"
)

RE_TRACEBACK = re.compile(r"Traceback \(most recent call last\):")
RE_RAW_PREFIX = re.compile(r"^(\d+_python_run_)PIM(\.py_.*)$")


# ----------------------------
# Data container
# ----------------------------

@dataclass
class RunSummary:
    file_name: str
    file_path: str

    start_ts: Optional[str]
    end_stats_ts: Optional[str]

    numeric_prefix: Optional[str]

    problem: Optional[str]
    graph_size: Optional[int]
    coords_dist: Optional[str]
    weights_dist: Optional[str]
    k: Optional[int]
    time_limit: Optional[int]

    data_file_path: Optional[str]
    checkpoint_load_path: Optional[str]
    checkpoint_short: Optional[str]

    env_key: Optional[str]
    run_group_key: Optional[str]

    add_ls: Optional[bool]
    giant_tour_split: Optional[bool]
    decode_vehicle_assignment: Optional[bool]
    has_assign_option: bool
    decode_v_assign_type: Optional[str]

    mode_suffix: str
    decoder_family: Optional[str]   # GT+Sp or assign
    postprocess: Optional[bool]     # inferred from add_ls
    decoder_era: Optional[str]      # none / v1 / v2 / explicit_assign_plain / explicit_assign_v2 / unknown_assign

    success: bool

    avg_cost: Optional[str]
    avg_vcost: Optional[str]
    avg_numveh: Optional[str]
    feasible: Optional[str]
    fleet_viol: Optional[str]
    delta_k: Optional[str]

    error_type: Optional[str]
    error_msg: Optional[str]

    inferred_stub_name: str


# ----------------------------
# Helpers
# ----------------------------

def extract_last_match(pattern: re.Pattern, text: str) -> Optional[re.Match]:
    matches = list(pattern.finditer(text))
    return matches[-1] if matches else None


def str_to_bool(x: Optional[str]) -> Optional[bool]:
    if x is None:
        return None
    x = x.strip().lower()
    if x == "true":
        return True
    if x == "false":
        return False
    return None


def clean_dist_short(dist: Optional[str]) -> Optional[str]:
    if not dist:
        return None
    d = dist.lower()
    if d.startswith("unif"):
        return "unf"
    return d


def safe_float_str(val: Optional[str]) -> Optional[str]:
    return val.strip() if val is not None else None


def is_raw_pim_log(path: Path) -> bool:
    name = path.name
    if "PIM" not in name or not name.endswith(".log"):
        return False

    # skip manually renamed logs
    skip_markers = ["GT+Sp", "assign", "_k"]
    if any(marker in name for marker in skip_markers):
        return False

    return bool(RE_RAW_PREFIX.match(name))


def extract_numeric_prefix(name: str) -> Optional[str]:
    m = re.match(r"^(\d+)_", name)
    return m.group(1) if m else None


def infer_mode_suffix(
    add_ls: Optional[bool],
    giant_tour_split: Optional[bool],
    decode_vehicle_assignment: Optional[bool],
    has_assign_option: bool,
) -> str:
    if not has_assign_option:
        if add_ls is True:
            return "_GT+Sp+post"
        if add_ls is False:
            return "_GT+Sp"
        return "_UNKNOWN"

    if giant_tour_split is True and decode_vehicle_assignment is False:
        if add_ls is True:
            return "_GT+Sp+post"
        if add_ls is False:
            return "_GT+Sp"

    if giant_tour_split is False and decode_vehicle_assignment is True:
        if add_ls is True:
            return "_assign+post"
        if add_ls is False:
            return "_assign"

    return "_UNKNOWN"


def infer_decoder_family(mode_suffix: str) -> Optional[str]:
    if "GT+Sp" in mode_suffix:
        return "GT+Sp"
    if "assign" in mode_suffix:
        return "assign"
    return None


def infer_decoder_era(
    has_assign_option: bool,
    decoder_family: Optional[str],
    decode_v_assign_type: Optional[str],
    start_ts: Optional[str],
    cutoff_ts: str,
) -> Optional[str]:
    if decoder_family != "assign":
        return "none"

    if decode_v_assign_type == "assign_plain":
        return "explicit_assign_plain"
    if decode_v_assign_type == "assign_v2":
        return "explicit_assign_v2"
    if decode_v_assign_type == "gt_split":
        return "gt_split_flag_conflict"

    if not has_assign_option:
        return "none"

    if start_ts is None:
        return "unknown_assign"

    return "v1" if start_ts < cutoff_ts else "v2"


def build_env_key(problem: Optional[str], graph_size: Optional[int], k: Optional[int], coords_dist: Optional[str]) -> Optional[str]:
    if not problem or graph_size is None or k is None:
        return None
    dist_short = clean_dist_short(coords_dist) or "unknown"
    return f"{problem}{graph_size}_k{k}_{dist_short}"


def build_run_group_key(
    env_key: Optional[str],
    checkpoint_short: Optional[str],
    decoder_family: Optional[str],
    postprocess: Optional[bool],
) -> Optional[str]:
    if env_key is None:
        return None
    post_str = "post" if postprocess else "nopost"
    ckpt = checkpoint_short or "unknown_ckpt"
    dec = decoder_family or "unknown_decoder"
    return f"{env_key}__{dec}__{post_str}__{ckpt}"


def shorten_checkpoint(path_str: Optional[str]) -> Optional[str]:
    if not path_str:
        return None
    return Path(path_str.strip()).name


def build_stub_name(
    original_name: str,
    mode_suffix: str,
    problem: Optional[str],
    graph_size: Optional[int],
    k: Optional[int],
    coords_dist: Optional[str],
    time_limit: Optional[int],
) -> str:
    m = RE_RAW_PREFIX.match(original_name)
    if not m:
        return Path(original_name).stem

    prefix = m.group(1)
    problem = problem or "cvrp"

    parts = [f"{prefix}PIM{mode_suffix}.py_env_{problem}"]
    if graph_size is not None:
        parts.append(str(graph_size))
    if k is not None:
        parts.append(f"k{k}")
    dist_short = clean_dist_short(coords_dist)
    if dist_short:
        parts.append(dist_short)
    if time_limit is not None:
        parts.append(f"test_cfg.time_limit_{time_limit}")

    return "_".join(parts)


def extract_error_summary(text: str) -> Tuple[Optional[str], Optional[str]]:
    lines = text.splitlines()

    for line in reversed(lines):
        m = RE_EXCEPTION_LINE.search(line.strip())
        if m:
            return m.group("etype"), m.group("msg").strip()

    tb_indices = [i for i, line in enumerate(lines) if RE_TRACEBACK.search(line)]
    if tb_indices:
        idx = tb_indices[-1]
        for line in lines[idx + 1:]:
            m = RE_EXCEPTION_LINE.search(line.strip())
            if m:
                return m.group("etype"), m.group("msg").strip()

    return None, None


# ----------------------------
# Parsing
# ----------------------------

def parse_log(path: Path, cutoff_ts: str) -> RunSummary:
    text = path.read_text(encoding="utf-8", errors="replace")

    main_match = extract_last_match(RE_MAIN_PROBLEM, text)
    start_ts = main_match.group("ts") if main_match else None
    problem = main_match.group("problem") if main_match else None

    graph_size_m = extract_last_match(RE_GRAPH_SIZE, text)
    graph_size = int(graph_size_m.group(1)) if graph_size_m else None

    coords_dist_m = extract_last_match(RE_COORDS_DIST, text)
    coords_dist = coords_dist_m.group(1) if coords_dist_m else None

    weights_dist_m = extract_last_match(RE_WEIGHTS_DIST, text)
    weights_dist = weights_dist_m.group(1) if weights_dist_m else None

    k_m = extract_last_match(RE_NR_VEH_EVAL, text)
    k = int(k_m.group(1)) if k_m else None

    tl_m = extract_last_match(RE_TIME_LIMIT, text)
    time_limit = int(tl_m.group(1)) if tl_m else None

    data_m = extract_last_match(RE_DATA_FILE, text)
    data_file_path = data_m.group(1).strip() if data_m else None

    ckpt_m = extract_last_match(RE_CKPT, text)
    checkpoint_load_path = ckpt_m.group(1).strip() if ckpt_m else None
    checkpoint_short = shorten_checkpoint(checkpoint_load_path)

    add_ls_m = extract_last_match(RE_ADD_LS, text)
    add_ls = str_to_bool(add_ls_m.group(1)) if add_ls_m else None

    gts_m = extract_last_match(RE_GIANT_TOUR_SPLIT, text)
    giant_tour_split = str_to_bool(gts_m.group(1)) if gts_m else None

    dva_m = extract_last_match(RE_DECODE_VEH_ASSIGN, text)
    decode_vehicle_assignment = str_to_bool(dva_m.group(1)) if dva_m else None
    has_assign_option = dva_m is not None

    dva_type_m = extract_last_match(RE_DECODE_V_ASSIGN_TYPE, text)
    decode_v_assign_type = dva_type_m.group(1).strip() if dva_type_m else None

    mode_suffix = infer_mode_suffix(add_ls, giant_tour_split, decode_vehicle_assignment, has_assign_option)
    decoder_family = infer_decoder_family(mode_suffix)
    postprocess = add_ls if add_ls is not None else None
    decoder_era = infer_decoder_era(
        has_assign_option=has_assign_option,
        decoder_family=decoder_family,
        decode_v_assign_type=decode_v_assign_type,
        start_ts=start_ts,
        cutoff_ts=cutoff_ts,
    )

    env_key = build_env_key(problem, graph_size, k, coords_dist)
    run_group_key = build_run_group_key(env_key, checkpoint_short, decoder_family, postprocess)

    inferred_stub_name = build_stub_name(
        original_name=path.name,
        mode_suffix=mode_suffix,
        problem=problem,
        graph_size=graph_size,
        k=k,
        coords_dist=coords_dist,
        time_limit=time_limit,
    )

    avg_match = extract_last_match(RE_AVG_BLOCK, text)
    if avg_match:
        return RunSummary(
            file_name=path.name,
            file_path=str(path),
            start_ts=start_ts,
            end_stats_ts=avg_match.group("ts"),
            numeric_prefix=extract_numeric_prefix(path.name),
            problem=problem,
            graph_size=graph_size,
            coords_dist=coords_dist,
            weights_dist=weights_dist,
            k=k,
            time_limit=time_limit,
            data_file_path=data_file_path,
            checkpoint_load_path=checkpoint_load_path,
            checkpoint_short=checkpoint_short,
            env_key=env_key,
            run_group_key=run_group_key,
            add_ls=add_ls,
            giant_tour_split=giant_tour_split,
            decode_vehicle_assignment=decode_vehicle_assignment,
            has_assign_option=has_assign_option,
            decode_v_assign_type=decode_v_assign_type,
            mode_suffix=mode_suffix,
            decoder_family=decoder_family,
            postprocess=postprocess,
            decoder_era=decoder_era,
            success=True,
            avg_cost=safe_float_str(avg_match.group("avg_cost")),
            avg_vcost=safe_float_str(avg_match.group("avg_vcost")),
            avg_numveh=safe_float_str(avg_match.group("avg_numveh")),
            feasible=safe_float_str(avg_match.group("feasible")),
            fleet_viol=safe_float_str(avg_match.group("fleetviol")),
            delta_k=safe_float_str(avg_match.group("delta_k")),
            error_type=None,
            error_msg=None,
            inferred_stub_name=inferred_stub_name,
        )

    err_type, err_msg = extract_error_summary(text)

    return RunSummary(
        file_name=path.name,
        file_path=str(path),
        start_ts=start_ts,
        end_stats_ts=None,
        numeric_prefix=extract_numeric_prefix(path.name),
        problem=problem,
        graph_size=graph_size,
        coords_dist=coords_dist,
        weights_dist=weights_dist,
        k=k,
        time_limit=time_limit,
        data_file_path=data_file_path,
        checkpoint_load_path=checkpoint_load_path,
        checkpoint_short=checkpoint_short,
        env_key=env_key,
        run_group_key=run_group_key,
        add_ls=add_ls,
        giant_tour_split=giant_tour_split,
        decode_vehicle_assignment=decode_vehicle_assignment,
        has_assign_option=has_assign_option,
        decode_v_assign_type=decode_v_assign_type,
        mode_suffix=mode_suffix,
        decoder_family=decoder_family,
        postprocess=postprocess,
        decoder_era=decoder_era,
        success=False,
        avg_cost=None,
        avg_vcost=None,
        avg_numveh=None,
        feasible=None,
        fleet_viol=None,
        delta_k=None,
        error_type=err_type,
        error_msg=err_msg,
        inferred_stub_name=inferred_stub_name,
    )


# ----------------------------
# Markdown output
# ----------------------------

def markdown_entry(run: RunSummary) -> str:
    annotations = []

    if not run.has_assign_option:
        annotations.append("no assign decode option yet")
    elif run.decoder_family == "assign":
        if run.decoder_era and run.decoder_era != "none":
            annotations.append(f"decoder era: {run.decoder_era}")

    ann = f" [{' ; '.join(annotations)}]" if annotations else ""

    lines = [
        f"**{run.start_ts or 'UNKNOWN_START_TIME'}**{ann}",
        f"**{run.inferred_stub_name}**",
    ]

    if run.success:
        lines += [
            f"{run.end_stats_ts} models.runner_utils:   ",
            f"Average cost  : {run.avg_cost},   ",
            f"Average vehicle cost : {run.avg_vcost},   ",
            f"Average number vehicles : {run.avg_numveh},   ",
            f"Feasible (%) : {run.feasible},   ",
            f"Fleet violation (%) : {run.fleet_viol} Average excess vehicles ΔK : {run.delta_k}",
        ]
    else:
        lines.append(f"*{run.error_type or 'Error'}*: {run.error_msg or 'No compact error message found.'}")

    return "\n".join(lines)


def write_markdown(runs: List[RunSummary], output_path: Path) -> None:
    output_path.write_text("\n\n".join(markdown_entry(r) for r in runs) + "\n", encoding="utf-8")


# ----------------------------
# CSV output
# ----------------------------

def write_csv(runs: List[RunSummary], output_path: Path) -> None:
    fieldnames = list(asdict(runs[0]).keys()) if runs else []
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for run in runs:
            writer.writerow(asdict(run))


# ----------------------------
# Optional renaming
# ----------------------------

def rename_logs(runs: List[RunSummary], dry_run: bool = False) -> None:
    for run in runs:
        old_path = Path(run.file_path)
        new_path = old_path.with_name(run.inferred_stub_name + ".log")
        if old_path.name == new_path.name:
            continue

        print(f"{old_path.name}  ->  {new_path.name}")
        if new_path.exists():
            print("  !! target exists, skipping")
            continue
        if not dry_run:
            old_path.rename(new_path)


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log_dir", type=Path)
    ap.add_argument("--md-output", type=Path, default=Path("pim_raw_summary.md"))
    ap.add_argument("--csv-output", type=Path, default=Path("pim_raw_summary.csv"))
    ap.add_argument("--cutoff-ts", type=str, default="2026/03/13 01:18:00")
    ap.add_argument("--rename", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    log_files = sorted([p for p in args.log_dir.rglob("*.log") if is_raw_pim_log(p)])
    if not log_files:
        print("No matching raw PIM logs found.")
        return

    runs = []
    for p in log_files:
        try:
            runs.append(parse_log(p, cutoff_ts=args.cutoff_ts))
        except Exception as e:
            runs.append(
                RunSummary(
                    file_name=p.name,
                    file_path=str(p),
                    start_ts=None,
                    end_stats_ts=None,
                    numeric_prefix=extract_numeric_prefix(p.name),
                    problem=None,
                    graph_size=None,
                    coords_dist=None,
                    weights_dist=None,
                    k=None,
                    time_limit=None,
                    data_file_path=None,
                    checkpoint_load_path=None,
                    checkpoint_short=None,
                    env_key=None,
                    run_group_key=None,
                    add_ls=None,
                    giant_tour_split=None,
                    decode_vehicle_assignment=None,
                    has_assign_option=False,
                    decode_v_assign_type=None,
                    mode_suffix="_UNKNOWN",
                    decoder_family=None,
                    postprocess=None,
                    decoder_era=None,
                    success=False,
                    avg_cost=None,
                    avg_vcost=None,
                    avg_numveh=None,
                    feasible=None,
                    fleet_viol=None,
                    delta_k=None,
                    error_type=type(e).__name__,
                    error_msg=str(e),
                    inferred_stub_name=Path(p.name).stem,
                )
            )

    runs.sort(key=lambda r: (r.start_ts is None, r.start_ts or "", r.file_name))

    write_markdown(runs, args.md_output)
    write_csv(runs, args.csv_output)

    print(f"Wrote markdown: {args.md_output}")
    print(f"Wrote csv:      {args.csv_output}")

    if args.rename or args.dry_run:
        rename_logs(runs, dry_run=args.dry_run)
        if args.dry_run:
            print("Dry-run only, no files renamed.")


if __name__ == "__main__":
    main()