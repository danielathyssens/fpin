#!/usr/bin/env python3
"""
Filter and summarize raw PIM log files into a markdown report.

What it does
------------
- Recursively scans a directory for raw PIM log files.
- Ignores already-renamed logs (those already containing k*, assign, GT+Sp, etc.).
- Extracts run metadata and final stats or error summaries.
- Writes a markdown summary file.
- Optionally renames raw logs based on inferred decode mode.

Typical usage
-------------
python filter_pim_logs.py /path/to/logs --output pim_raw_summary.md
python filter_pim_logs.py /path/to/logs --output pim_raw_summary.md --rename
python filter_pim_logs.py /path/to/logs --output pim_raw_summary.md --rename --dry-run

Notes
-----
- By default, files are NOT renamed.
- The numeric prefix like '004_' is preserved.
- For renamed filenames, only the first raw 'PIM' token is replaced by e.g. 'PIM_assign+post'.
- If the script cannot infer enough information, it still records the file in markdown.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple


# ---------- patterns ----------

TS_SLASH = r"\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}"
TS_DASH = r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d+)?"

RE_MAIN_PROBLEM = re.compile(
    rf"(?P<ts>{TS_SLASH}) __main__: problem:\s*(?P<problem>\w+)"
)

RE_GRAPH_SIZE = re.compile(r"^graph_size:\s*(\d+)\s*$", re.MULTILINE)
RE_NR_VEH_EVAL = re.compile(r"^\s*nr_vehicles_eval:\s*(\d+)\s*$", re.MULTILINE)
RE_TIME_LIMIT = re.compile(r"^\s*time_limit:\s*(\d+)\s*$", re.MULTILINE)
RE_ADD_LS = re.compile(r"^\s*add_ls:\s*(true|false)\s*$", re.MULTILINE)
RE_GIANT_TOUR_SPLIT = re.compile(
    r"^\s*giant_tour_split:\s*(true|false)\s*$", re.MULTILINE
)
RE_DECODE_VEH_ASSIGN = re.compile(
    r"^\s*decode_vehicle_assignment:\s*(true|false)\s*$", re.MULTILINE
)
RE_COORDS_DIST = re.compile(r"^coords_dist:\s*([A-Za-z0-9_]+)\s*$", re.MULTILINE)

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

RE_FLEET_IN_NAME = re.compile(r"(?<!\d)k\d+(?!\d)")
RE_RAW_PREFIX = re.compile(r"^(\d+_python_run_)PIM(\.py_.*)$")


# ---------- data containers ----------

@dataclass
class RunSummary:
    path: Path
    start_ts: Optional[str]
    problem: Optional[str]
    graph_size: Optional[int]
    k: Optional[int]
    time_limit: Optional[int]
    coords_dist: Optional[str]
    add_ls: Optional[bool]
    giant_tour_split: Optional[bool]
    decode_vehicle_assignment: Optional[bool]
    has_assign_option: bool
    mode_suffix: str
    new_stub_name: str
    success: bool
    stats_ts: Optional[str]
    avg_cost: Optional[str]
    avg_vcost: Optional[str]
    avg_numveh: Optional[str]
    feasible: Optional[str]
    fleet_viol: Optional[str]
    delta_k: Optional[str]
    error_type: Optional[str]
    error_msg: Optional[str]
    renamed_path: Optional[Path] = None


# ---------- helpers ----------

def str_to_bool(value: Optional[str]) -> Optional[bool]:
    if value is None:
        return None
    v = value.strip().lower()
    if v == "true":
        return True
    if v == "false":
        return False
    return None


def extract_last_match(pattern: re.Pattern, text: str) -> Optional[re.Match]:
    matches = list(pattern.finditer(text))
    return matches[-1] if matches else None


def is_raw_pim_log(path: Path) -> bool:
    name = path.name

    if "PIM" not in name:
        return False
    if not name.endswith(".log"):
        return False

    # Skip logs the user already renamed manually.
    skip_markers = [
        "GT+Sp",
        "assign",
        "_k",
    ]
    if any(marker in name for marker in skip_markers):
        return False

    # Must still be of the raw type.
    return bool(RE_RAW_PREFIX.match(name))


def infer_mode_suffix(
    add_ls: Optional[bool],
    giant_tour_split: Optional[bool],
    decode_vehicle_assignment: Optional[bool],
    has_assign_option: bool,
) -> str:
    """
    Build suffix after PIM:
      _GT+Sp
      _GT+Sp+post
      _assign
      _assign+post

    Fallbacks:
    - If assign option does not exist yet, assume GT+Sp / GT+Sp+post.
    - If mode can't be inferred, return '_UNKNOWN'.
    """
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


def build_stub_name(
    original_name: str,
    mode_suffix: str,
    graph_size: Optional[int],
    k: Optional[int],
    coords_dist: Optional[str],
    time_limit: Optional[int],
) -> str:
    """
    Create markdown display name like:
    009_python_run_PIM_GT+Sp.py_env_cvrp20_k4_unf_test_cfg.time_limit_5

    It preserves the numeric prefix from the raw filename.
    """
    m = RE_RAW_PREFIX.match(original_name)
    if not m:
        return original_name

    prefix = m.group(1)  # e.g. 009_python_run_
    tail = m.group(2)    # e.g. .py_env_cvrp50_unf_test_cfg.time_limit_8_...

    # Short dist label
    dist_short = None
    if coords_dist:
        if coords_dist.lower().startswith("unif"):
            dist_short = "unf"
        else:
            dist_short = coords_dist

    # Prefer reconstructed short standardized name.
    # Problem in examples is always cvrp, but use generic fallback if missing.
    problem = "cvrp"

    parts = [f"{prefix}PIM{mode_suffix}.py_env_{problem}"]

    if graph_size is not None:
        parts.append(str(graph_size))
    if k is not None:
        parts.append(f"k{k}")
    if dist_short:
        parts.append(dist_short)
    if time_limit is not None:
        parts.append(f"test_cfg.time_limit_{time_limit}")

    return "_".join(parts)


def extract_error_summary(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Return compact final error type/message from end of log.
    Priority:
    1) last Error/Exception line
    2) traceback-following error line
    """
    lines = text.splitlines()

    # Search backwards for explicit "...Error: ..."
    for line in reversed(lines):
        m = RE_EXCEPTION_LINE.search(line.strip())
        if m:
            return m.group("etype"), m.group("msg").strip()

    # Fallback: find traceback and inspect following lines
    tb_indices = [i for i, line in enumerate(lines) if RE_TRACEBACK.search(line)]
    if tb_indices:
        idx = tb_indices[-1]
        for line in lines[idx + 1:]:
            m = RE_EXCEPTION_LINE.search(line.strip())
            if m:
                return m.group("etype"), m.group("msg").strip()

    return None, None


def parse_log(path: Path) -> RunSummary:
    text = path.read_text(encoding="utf-8", errors="replace")

    main_match = extract_last_match(RE_MAIN_PROBLEM, text)
    start_ts = main_match.group("ts") if main_match else None
    problem = main_match.group("problem") if main_match else None

    graph_size_match = extract_last_match(RE_GRAPH_SIZE, text)
    graph_size = int(graph_size_match.group(1)) if graph_size_match else None

    k_match = extract_last_match(RE_NR_VEH_EVAL, text)
    k = int(k_match.group(1)) if k_match else None

    time_limit_match = extract_last_match(RE_TIME_LIMIT, text)
    time_limit = int(time_limit_match.group(1)) if time_limit_match else None

    coords_dist_match = extract_last_match(RE_COORDS_DIST, text)
    coords_dist = coords_dist_match.group(1) if coords_dist_match else None

    add_ls_match = extract_last_match(RE_ADD_LS, text)
    add_ls = str_to_bool(add_ls_match.group(1)) if add_ls_match else None

    gts_match = extract_last_match(RE_GIANT_TOUR_SPLIT, text)
    giant_tour_split = str_to_bool(gts_match.group(1)) if gts_match else None

    dva_match = extract_last_match(RE_DECODE_VEH_ASSIGN, text)
    decode_vehicle_assignment = (
        str_to_bool(dva_match.group(1)) if dva_match else None
    )
    has_assign_option = dva_match is not None

    mode_suffix = infer_mode_suffix(
        add_ls=add_ls,
        giant_tour_split=giant_tour_split,
        decode_vehicle_assignment=decode_vehicle_assignment,
        has_assign_option=has_assign_option,
    )

    stub_name = build_stub_name(
        original_name=path.name,
        mode_suffix=mode_suffix,
        graph_size=graph_size,
        k=k,
        coords_dist=coords_dist,
        time_limit=time_limit,
    )

    avg_match = extract_last_match(RE_AVG_BLOCK, text)

    if avg_match:
        return RunSummary(
            path=path,
            start_ts=start_ts,
            problem=problem,
            graph_size=graph_size,
            k=k,
            time_limit=time_limit,
            coords_dist=coords_dist,
            add_ls=add_ls,
            giant_tour_split=giant_tour_split,
            decode_vehicle_assignment=decode_vehicle_assignment,
            has_assign_option=has_assign_option,
            mode_suffix=mode_suffix,
            new_stub_name=stub_name,
            success=True,
            stats_ts=avg_match.group("ts"),
            avg_cost=avg_match.group("avg_cost").strip(),
            avg_vcost=avg_match.group("avg_vcost").strip(),
            avg_numveh=avg_match.group("avg_numveh").strip(),
            feasible=avg_match.group("feasible").strip(),
            fleet_viol=avg_match.group("fleetviol").strip(),
            delta_k=avg_match.group("delta_k").strip(),
            error_type=None,
            error_msg=None,
        )

    error_type, error_msg = extract_error_summary(text)

    return RunSummary(
        path=path,
        start_ts=start_ts,
        problem=problem,
        graph_size=graph_size,
        k=k,
        time_limit=time_limit,
        coords_dist=coords_dist,
        add_ls=add_ls,
        giant_tour_split=giant_tour_split,
        decode_vehicle_assignment=decode_vehicle_assignment,
        has_assign_option=has_assign_option,
        mode_suffix=mode_suffix,
        new_stub_name=stub_name,
        success=False,
        stats_ts=None,
        avg_cost=None,
        avg_vcost=None,
        avg_numveh=None,
        feasible=None,
        fleet_viol=None,
        delta_k=None,
        error_type=error_type,
        error_msg=error_msg,
    )


def markdown_entry(run: RunSummary, cutoff_ts: Optional[str] = None) -> str:
    """
    Build markdown block.
    If cutoff_ts is provided, add decoder version label:
      [old assign decoder era]
      [new assign decoder era]
    only when assign decode option existed and mode is assign.
    """
    annotations = []

    if not run.has_assign_option:
        annotations.append("no assign decode option yet")
    else:
        # Optional decoder-era note, useful for your reproducibility split
        if cutoff_ts and run.start_ts and run.mode_suffix.startswith("_assign"):
            if run.start_ts < cutoff_ts:
                annotations.append("assign decoder era: pre-2026/03/13 01:18:00 (v1)")
            else:
                annotations.append("assign decoder era: post-2026/03/13 01:18:00 (v2)")

    ann = ""
    if annotations:
        ann = " [" + "; ".join(annotations) + "]"

    start_ts = run.start_ts or "UNKNOWN_START_TIME"

    lines = [f"**{start_ts}**{ann}", f"**{run.new_stub_name}**"]

    if run.success:
        lines.extend([
            f"{run.stats_ts} models.runner_utils:   ",
            f"Average cost  : {run.avg_cost},   ",
            f"Average vehicle cost : {run.avg_vcost},   ",
            f"Average number vehicles : {run.avg_numveh},   ",
            f"Feasible (%) : {run.feasible},   ",
            f"Fleet violation (%) : {run.fleet_viol} Average excess vehicles ΔK : {run.delta_k}",
        ])
    else:
        etype = run.error_type or "Error"
        emsg = run.error_msg or "No compact error message found."
        lines.append(f"*{etype}*: {emsg}")

    return "\n".join(lines)


def sort_key(run: RunSummary):
    return (run.start_ts is None, run.start_ts or "", run.path.name)


def rename_log(run: RunSummary, dry_run: bool = False) -> Optional[Path]:
    old_path = run.path

    if run.new_stub_name == old_path.name.replace(".log", ""):
        # Should not happen often, but harmless.
        return old_path

    new_name = f"{run.new_stub_name}.log"
    new_path = old_path.with_name(new_name)

    if new_path.exists() and new_path != old_path:
        raise FileExistsError(f"Target already exists: {new_path}")

    if not dry_run:
        old_path.rename(new_path)

    return new_path


def write_markdown(
    runs: List[RunSummary],
    output_path: Path,
    cutoff_ts: Optional[str],
) -> None:
    blocks = [markdown_entry(run, cutoff_ts=cutoff_ts) for run in runs]
    output_path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")


# ---------- main ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "log_dir",
        type=Path,
        help="Directory containing raw log files (searched recursively).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("pim_raw_summary.md"),
        help="Output markdown file.",
    )
    parser.add_argument(
        "--rename",
        action="store_true",
        help="Actually rename matching raw log files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be renamed without changing files.",
    )
    parser.add_argument(
        "--cutoff-ts",
        default="2026/03/13 01:18:00",
        help=(
            "Timestamp separating old/new assign decoder eras. "
            "Used only as markdown annotation for assign runs."
        ),
    )
    args = parser.parse_args()

    log_dir = args.log_dir
    if not log_dir.exists():
        raise FileNotFoundError(f"Log directory does not exist: {log_dir}")

    raw_logs = sorted(
        [p for p in log_dir.rglob("*.log") if is_raw_pim_log(p)],
        key=lambda p: p.name,
    )

    if not raw_logs:
        print("No matching raw PIM logs found.")
        return

    runs: List[RunSummary] = []
    for path in raw_logs:
        try:
            run = parse_log(path)
            runs.append(run)
        except Exception as exc:
            runs.append(
                RunSummary(
                    path=path,
                    start_ts=None,
                    problem=None,
                    graph_size=None,
                    k=None,
                    time_limit=None,
                    coords_dist=None,
                    add_ls=None,
                    giant_tour_split=None,
                    decode_vehicle_assignment=None,
                    has_assign_option=False,
                    mode_suffix="_UNKNOWN",
                    new_stub_name=path.stem,
                    success=False,
                    stats_ts=None,
                    avg_cost=None,
                    avg_vcost=None,
                    avg_numveh=None,
                    feasible=None,
                    fleet_viol=None,
                    delta_k=None,
                    error_type=type(exc).__name__,
                    error_msg=str(exc),
                )
            )

    runs.sort(key=sort_key)

    if args.rename or args.dry_run:
        print("\nRename plan:")
        for run in runs:
            old_path = run.path
            new_name = f"{run.new_stub_name}.log"
            new_path = old_path.with_name(new_name)
            print(f"- {old_path.name}")
            print(f"  -> {new_path.name}")
            if args.rename or args.dry_run:
                try:
                    renamed = rename_log(run, dry_run=args.dry_run)
                    run.renamed_path = renamed
                except Exception as exc:
                    print(f"  !! rename failed: {exc}")

    write_markdown(runs, args.output, cutoff_ts=args.cutoff_ts)

    print(f"\nProcessed {len(runs)} raw PIM log(s).")
    print(f"Markdown summary written to: {args.output}")
    if args.dry_run:
        print("Dry-run mode: no files were renamed.")
    elif args.rename:
        print("Renaming completed.")
    else:
        print("No renaming performed. Use --rename to rename files.")


if __name__ == "__main__":
    main()