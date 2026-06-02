#!/usr/bin/env python3
"""Parse PIM eval logs (the `pim_*.log` files in eval_logs/pim_eval_<date>/)
and APPEND rows to the master CSV used by csv_to_latex_tables.py.

Does not rewrite existing rows. Idempotent: re-running with already-appended
rows skips duplicates (matched by (dist, N, M, method, decode, +post)).

Usage:
    python eval_logs/append_pim_eval_to_master.py \
        --logs eval_logs/pim_eval_20260602 \
        --csv  eval_logs/experiment_master_results_all.csv \
        [--ckpt-label PIM-Thyssens2022]

Filename convention assumed (matches the eval block I sent):
    pim_n{N}_k{M}_{raw|ls}.log
"""
from __future__ import annotations
import argparse
import csv
import re
from pathlib import Path
from typing import Optional


# --- regex over the standard runner_utils summary block --------------------

RE_VCOST = re.compile(r"Average vehicle cost\s*:\s*([0-9.\-eE]+)")
RE_COST  = re.compile(r"Average cost\s*:\s*([0-9.\-eE]+)")
RE_NV    = re.compile(r"Average number vehicles\s*:\s*([0-9.\-eE]+)")
RE_VIOL  = re.compile(r"Fleet violation \(%\)\s*:\s*([0-9.\-eE]+)")
RE_DK    = re.compile(r"Average excess vehicles\s+Δ?K\s*:\s*([0-9.\-eE]+)")
RE_FEAS  = re.compile(r"Feasible \(%\)\s*:\s*([0-9.\-eE]+)")
RE_TIME  = re.compile(r"Average Run Time \(total\)\s*:\s*([0-9.\-eE]+)")
RE_FNAME = re.compile(r"pim_n(\d+)_k(\d+)_(raw|ls)\.log")


def parse_log(path: Path) -> Optional[dict]:
    m = RE_FNAME.match(path.name)
    if not m:
        return None
    N = int(m.group(1))
    M = int(m.group(2))
    add_ls = (m.group(3) == "ls")
    text = path.read_text(errors="ignore")
    # find LAST occurrence of each (the summary at end-of-run)
    def last(pat):
        hits = pat.findall(text)
        return hits[-1] if hits else None
    cost = last(RE_COST)
    vcost = last(RE_VCOST)
    nv = last(RE_NV)
    viol = last(RE_VIOL)
    dk = last(RE_DK)
    feas = last(RE_FEAS)
    runtime = last(RE_TIME)
    if vcost is None:
        return None
    return {
        "N": N, "M": M, "add_ls": add_ls,
        "cost": cost, "cost_v": vcost, "num_vehicles": nv,
        "viol_pct": viol, "delta_k": dk, "feas_pct": feas, "runtime": runtime,
    }


def to_master_row(rec: dict, ckpt_label: str, dist: str = "uniform") -> dict:
    """Map parsed log -> master CSV schema. Method label and decode label are
    fixed for the old-PIM (softassign + greedy_path) baseline so that
    csv_to_latex_tables.py picks them up consistently in every table."""
    return {
        "split": "test",
        "dist": dist,
        "N": rec["N"],
        "M": rec["M"],
        "method": "PIM",
        "decode": "softassign",
        "+post": "yes" if rec["add_ls"] else "no",
        "checkpoint": ckpt_label,
        "cost": rec["cost"] or "",
        "cost_v": rec["cost_v"] or "",
        "num_vehicles": rec["num_vehicles"] or "",
        "viol_pct": rec["viol_pct"] or "",
        "delta_k": rec["delta_k"] or "",
        "feas_pct": rec["feas_pct"] or "",
        "runtime": rec["runtime"] or "",
        "notes": "PIM Thyssens 2022 baseline (cross-fleet eval where eval_k != ckpt_k)",
        "status": "done",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True, type=Path,
                    help="Directory containing pim_*.log files.")
    ap.add_argument("--csv", required=True, type=Path,
                    help="Master CSV to append to.")
    ap.add_argument("--ckpt-label", default="PIM-Thyssens2022",
                    help="Label for the 'checkpoint' column.")
    ap.add_argument("--dist", default="uniform")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # Read existing rows (idempotency)
    existing = set()
    if args.csv.exists():
        with args.csv.open() as fh:
            reader = csv.DictReader(fh)
            header = reader.fieldnames
            for r in reader:
                existing.add((r.get("dist"), r.get("N"), r.get("M"),
                              r.get("method"), r.get("decode"), r.get("+post")))
    else:
        raise SystemExit(f"Master CSV not found: {args.csv}")

    # Parse logs
    new_rows = []
    for log in sorted(args.logs.glob("pim_*.log")):
        rec = parse_log(log)
        if rec is None:
            print(f"  [skip] could not parse: {log.name}")
            continue
        row = to_master_row(rec, args.ckpt_label, dist=args.dist)
        key = (row["dist"], str(row["N"]), str(row["M"]),
               row["method"], row["decode"], row["+post"])
        if key in existing:
            print(f"  [skip-dup] {log.name} -> {key} already in master CSV")
            continue
        new_rows.append(row)
        print(f"  [+] {log.name} -> N={row['N']} M={row['M']} +post={row['+post']}  "
              f"cost_v={row['cost_v']}  viol%={row['viol_pct']}  feas%={row['feas_pct']}")

    if not new_rows:
        print("Nothing to append.")
        return

    if args.dry_run:
        print(f"\n[dry-run] would append {len(new_rows)} rows to {args.csv}")
        return

    # Append, preserving the existing header order
    with args.csv.open("a", newline="") as fh:
        # Filter row keys to header; missing fields stay empty
        writer = csv.DictWriter(fh, fieldnames=header, extrasaction="ignore")
        for r in new_rows:
            writer.writerow(r)
    print(f"\nAppended {len(new_rows)} PIM rows to {args.csv}")


if __name__ == "__main__":
    main()
