"""Target-aligned heatmap audit for FPIN and old PIM variants.

This script answers the question "is the heatmap failing at clustering or at
ordering?" on route-labeled HQ/HGS `.npz` data. It does not depend on a
heuristic decoder for the primary metrics.

Metrics:
  - membership_acc: customer -> vehicle assignment accuracy after Hungarian
    alignment of predicted vehicles to target routes
  - succ_ce_all: successor cross-entropy on target edges after alignment
  - succ_ce_correct_cluster: successor cross-entropy restricted to customer
    rows whose predicted vehicle assignment is already correct
  - expected_load / overload statistics from raw customer-to-vehicle mass
  - row / customer-joint entropy and customer commitment
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

from fpin.VRPModel_attn_new import VRP_Net as FPINNet
from fpin.utils_all.loss_utils import hungarian_match_from_membership


EPS = 1e-12


@dataclass
class AuditStats:
    tag: str
    model_family: str
    ckpt: str
    data_npz: str
    graph_size: int
    fleet_size: int
    num_instances: int
    membership_acc: float
    succ_ce_all: float
    succ_ce_start: float
    succ_ce_next: float
    succ_ce_correct_cluster: float
    row_entropy: float
    customer_joint_entropy: float
    customer_commitment: float
    expected_load_mean: float
    overload_mean: float
    overload_max_mean: float
    any_overload_rate: float
    log_temperature: float | None


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-family", choices=["fpin", "pim_soft", "pim_attn1"], required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-npz", required=True)
    ap.add_argument("--graph-size", type=int, default=20)
    ap.add_argument("--fleet-size", type=int, required=True)
    ap.add_argument("--limit", type=int, default=256)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--coords-dist", default="uniform")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out-json", default="")

    # FPIN architecture / ablation knobs.
    ap.add_argument("--use-attn", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--vehicle-cond-edge-head", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--sinkhorn-assignment", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--sinkhorn-iters", type=int, default=3)
    ap.add_argument("--joint-customer-norm", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--softassign-head", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--softassign-layers", type=int, default=3)
    ap.add_argument("--softassign-log-domain", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--vcount-aux-head", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--add-demand-weights", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--use-vehicle-id-embedding", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--use-graph-encoder", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--use-perm-inv-encoder", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--learnable-temperature", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--initial-log-temperature", type=float, default=0.0)

    # Old PIM knobs.
    ap.add_argument("--pim-softassign-layers", type=int, default=90)
    return ap


def _ensure_pim_src_on_path() -> None:
    pim_src = os.path.abspath("models/PIMold/src")
    if pim_src not in sys.path:
        sys.path.insert(0, pim_src)


def _pim_specs(graph_size: int, soft_layers: int) -> Dict[str, int | float]:
    if graph_size == 20:
        return dict(layers=7, main_dim=256, hidden=1024, cities_dim=3, soft_layers=soft_layers)
    if graph_size == 50:
        return dict(layers=9, main_dim=256, hidden=1024, cities_dim=3, soft_layers=soft_layers)
    if graph_size == 100:
        return dict(layers=7, main_dim=128, hidden=512, cities_dim=4, soft_layers=50 if soft_layers == 90 else soft_layers)
    raise ValueError(f"Unsupported graph_size for old PIM family: {graph_size}")


def load_model(args: argparse.Namespace, device: torch.device):
    if args.model_family == "fpin":
        model = FPINNet(
            layers=9,
            depot_in_dim=4,
            cities_in_dim=3,
            fleet_in_dim=260,
            cities_length=args.graph_size + 1,
            max_fleet_length=args.fleet_size,
            main_dim=256,
            avg_pool=False,
            residual=True,
            norm=True,
            ff_hidden_dim=1024,
            dropout=0.0,
            self_pool=False,
            embedding_norm=True,
            weighting=True,
            with_loads=True,
            use_attn=args.use_attn,
            regret_batches=1,
            add_demand_weights=args.add_demand_weights,
            vehicle_cond_edge_head=args.vehicle_cond_edge_head,
            sinkhorn_assignment=args.sinkhorn_assignment,
            sinkhorn_iters=args.sinkhorn_iters,
            joint_customer_norm=args.joint_customer_norm,
            softassign_head=args.softassign_head,
            softassign_layers=args.softassign_layers,
            softassign_log_domain=args.softassign_log_domain,
            global_edge_softmax=getattr(args, "global_edge_softmax", False),
            vcount_aux_head=args.vcount_aux_head,
            use_vehicle_id_embedding=args.use_vehicle_id_embedding,
            use_graph_encoder=args.use_graph_encoder,
            use_perm_inv_encoder=args.use_perm_inv_encoder,
            learnable_temperature=args.learnable_temperature,
            initial_log_temperature=args.initial_log_temperature,
        ).to(device).eval()
    else:
        _ensure_pim_src_on_path()
        spec = _pim_specs(args.graph_size, args.pim_softassign_layers)
        if args.model_family == "pim_soft":
            from VRPModel_attn_soft import VRP_Net as OldPIMNet
            soft_layers = spec["soft_layers"]
        else:
            from VRPModel_attn1 import VRP_Net as OldPIMNet
            soft_layers = 0

        model = OldPIMNet(
            spec["layers"], 4, spec["cities_dim"], 4, spec["main_dim"],
            False, True, True, spec["hidden"], 0.0, False, True,
            soft_layers, True, False, True, True
        ).to(device).eval()

    ckpt = torch.load(args.ckpt, map_location=device)
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state, strict=True)
    return model


def build_fpin_inputs(
    depots: np.ndarray,
    locs: np.ndarray,
    demands: np.ndarray,
    capacities: np.ndarray,
    fleet_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, ...]:
    B, N = demands.shape
    n_nodes = N + 1
    coords = np.concatenate([depots[:, None, :], locs], axis=1)
    dists = np.linalg.norm(coords[:, :, None, :] - coords[:, None, :, :], axis=-1).astype(np.float32)
    depot_centrality = ((n_nodes - 1) / np.clip(dists[:, 0].sum(axis=1), 1e-9, None)).astype(np.float32)

    frac_ids = (np.arange(1, fleet_size + 1, dtype=np.float32) / float(fleet_size))[None, :]
    veh_ids = np.arange(fleet_size, dtype=np.float32)[None, :]
    total_dem = demands.sum(axis=1, keepdims=True).astype(np.float32)

    fleet = np.zeros((B, fleet_size, 5), dtype=np.float32)
    fleet[:, :, 0] = np.broadcast_to(veh_ids, (B, fleet_size))
    fleet[:, :, 1] = np.broadcast_to(frac_ids, (B, fleet_size))
    fleet[:, :, 2] = float(fleet_size)
    fleet[:, :, 3] = 1.0
    fleet[:, :, 4] = np.broadcast_to(total_dem, (B, fleet_size))

    depot = np.concatenate([depots, np.zeros((B, 1), dtype=np.float32), depot_centrality[:, None]], axis=1)[:, None, :]
    customers = np.concatenate([locs, demands[..., None]], axis=2)
    demand_array = np.concatenate([np.zeros((B, 1), dtype=np.float32), demands], axis=1)[:, None, :]

    return (
        torch.from_numpy(fleet).to(device),
        torch.from_numpy(depot).to(device),
        torch.from_numpy(customers).to(device),
        torch.from_numpy(demand_array).to(device),
        torch.from_numpy(dists).to(device),
    )


def build_old_pim_inputs(
    depots: np.ndarray,
    locs: np.ndarray,
    demands: np.ndarray,
    capacities: np.ndarray,
    fleet_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, ...]:
    B, N = demands.shape
    n_nodes = N + 1
    coords = np.concatenate([depots[:, None, :], locs], axis=1)
    dists = np.linalg.norm(coords[:, :, None, :] - coords[:, None, :, :], axis=-1).astype(np.float32)
    depot_centrality = ((n_nodes - 1) / np.clip(dists[:, 0].sum(axis=1), 1e-9, None)).astype(np.float32)

    frac_ids = (np.arange(1, fleet_size + 1, dtype=np.float32) / float(fleet_size))[None, :]
    total_dem = demands.sum(axis=1, keepdims=True).astype(np.float32)

    fleet = np.zeros((B, fleet_size, 4), dtype=np.float32)
    fleet[:, :, 0] = np.broadcast_to(frac_ids, (B, fleet_size))
    fleet[:, :, 1] = float(fleet_size)
    fleet[:, :, 2] = 1.0
    fleet[:, :, 3] = np.broadcast_to(total_dem, (B, fleet_size))

    depot = np.concatenate([depots, np.zeros((B, 1), dtype=np.float32), depot_centrality[:, None]], axis=1)[:, None, :]
    customers = np.concatenate([locs, demands[..., None]], axis=2)
    demand_array = np.concatenate([np.zeros((B, 1), dtype=np.float32), demands], axis=1)[:, None, :]

    return (
        torch.from_numpy(fleet).to(device),
        torch.from_numpy(depot).to(device),
        torch.from_numpy(customers).to(device),
        torch.from_numpy(demand_array).to(device),
        torch.from_numpy(dists).to(device),
    )


def build_target_dense(
    solutions: Sequence[Sequence[Sequence[int]]],
    fleet_size: int,
    n_nodes: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B = len(solutions)
    target = torch.zeros((B, fleet_size, n_nodes, n_nodes), dtype=torch.float32, device=device)
    membership = torch.zeros((B, fleet_size, n_nodes - 1), dtype=torch.float32, device=device)
    for b, sol in enumerate(solutions):
        for m in range(fleet_size):
            route = sol[m] if m < len(sol) else [0, 0]
            for a, c in zip(route[:-1], route[1:]):
                target[b, m, a, c] = 1.0
            for node in route[1:-1]:
                membership[b, m, node - 1] = 1.0
    return target, membership


def membership_scores_from_probs(probs: torch.Tensor) -> torch.Tensor:
    inc = probs.amax(dim=2)
    out = probs.amax(dim=3)
    return torch.maximum(inc, out)[..., 1:]


def forward_probs(
    model,
    args: argparse.Namespace,
    fleet: torch.Tensor,
    depot: torch.Tensor,
    customers: torch.Tensor,
    demands: torch.Tensor,
    dists: torch.Tensor,
) -> torch.Tensor:
    with torch.no_grad():
        if args.model_family == "fpin":
            _, probs = model(depot, customers, fleet, demands, dists, sample=False, training=False)
        else:
            probs, *_ = model(depot, customers, fleet, demands, dists)
    return probs


def audit_batch(
    probs: torch.Tensor,
    target: torch.Tensor,
    target_mem: torch.Tensor,
    demand_norm_customers: torch.Tensor,
) -> Dict[str, float]:
    B, M, n_nodes, _ = probs.shape
    pred_mem = membership_scores_from_probs(probs)
    perm = hungarian_match_from_membership(pred_mem, target_mem)

    tgt_idx = perm.to(probs.device).unsqueeze(-1).unsqueeze(-1).expand(B, M, n_nodes, n_nodes)
    target_aligned = torch.gather(target, 1, tgt_idx)
    mem_idx = perm.to(probs.device).unsqueeze(-1).expand(B, M, n_nodes - 1)
    target_mem_aligned = torch.gather(target_mem, 1, mem_idx)

    pred_assign = pred_mem.argmax(dim=1)
    target_assign = target_mem_aligned.argmax(dim=1)
    assign_mass = pred_mem / pred_mem.sum(dim=1, keepdim=True).clamp_min(EPS)

    logp = probs.clamp_min(EPS).log()
    row_entropy = -(probs * logp).sum(dim=-1)

    cust = probs[:, :, 1:, :]
    cust_flat = cust.permute(0, 2, 1, 3).reshape(B, n_nodes - 1, M * n_nodes)
    cust_flat = cust_flat / cust_flat.sum(dim=-1, keepdim=True).clamp_min(EPS)
    cust_log = cust_flat.clamp_min(EPS).log()
    cust_joint_entropy = -(cust_flat * cust_log).sum(dim=-1)

    expected_load = (assign_mass * demand_norm_customers.unsqueeze(1)).sum(dim=-1)
    overload = (expected_load - 1.0).clamp_min(0.0)

    correct_cluster = (pred_assign == target_assign).unsqueeze(1).unsqueeze(-1)
    target_next_correct = target_aligned[:, :, 1:, :] * correct_cluster

    return {
        "membership_correct": float((pred_assign == target_assign).sum().item()),
        "membership_total": float(pred_assign.numel()),
        "succ_nll_all": float((-(target_aligned * logp)).sum().item()),
        "succ_edges_all": float(target_aligned.sum().item()),
        "succ_nll_start": float((-(target_aligned[:, :, :1, :] * logp[:, :, :1, :])).sum().item()),
        "succ_edges_start": float(target_aligned[:, :, :1, :].sum().item()),
        "succ_nll_next": float((-(target_aligned[:, :, 1:, :] * logp[:, :, 1:, :])).sum().item()),
        "succ_edges_next": float(target_aligned[:, :, 1:, :].sum().item()),
        "succ_nll_next_correct": float((-(target_next_correct * logp[:, :, 1:, :])).sum().item()),
        "succ_edges_next_correct": float(target_next_correct.sum().item()),
        "row_entropy_sum": float(row_entropy.sum().item()),
        "row_entropy_count": float(row_entropy.numel()),
        "cust_entropy_sum": float(cust_joint_entropy.sum().item()),
        "cust_entropy_count": float(cust_joint_entropy.numel()),
        "commit_sum": float(assign_mass.max(dim=1).values.sum().item()),
        "commit_count": float(assign_mass.max(dim=1).values.numel()),
        "expected_load_sum": float(expected_load.sum().item()),
        "expected_load_count": float(expected_load.numel()),
        "overload_sum": float(overload.sum().item()),
        "overload_count": float(overload.numel()),
        "overload_max_sum": float(overload.max(dim=-1).values.sum().item()),
        "overload_batch_count": float(overload.size(0)),
        "any_overload_sum": float((overload.max(dim=-1).values > 1e-6).float().sum().item()),
    }


def merge_stats(dst: Dict[str, float], src: Dict[str, float]) -> None:
    for k, v in src.items():
        dst[k] = dst.get(k, 0.0) + float(v)


def finalize_stats(args: argparse.Namespace, totals: Dict[str, float], model) -> AuditStats:
    log_temperature = None
    if hasattr(model, "log_temperature"):
        log_temperature = float(model.log_temperature.detach().reshape(-1)[0].item())

    return AuditStats(
        tag=args.tag or Path(args.ckpt).stem,
        model_family=args.model_family,
        ckpt=args.ckpt,
        data_npz=args.data_npz,
        graph_size=args.graph_size,
        fleet_size=args.fleet_size,
        num_instances=int(totals["num_instances"]),
        membership_acc=totals["membership_correct"] / max(1.0, totals["membership_total"]),
        succ_ce_all=totals["succ_nll_all"] / max(1.0, totals["succ_edges_all"]),
        succ_ce_start=totals["succ_nll_start"] / max(1.0, totals["succ_edges_start"]),
        succ_ce_next=totals["succ_nll_next"] / max(1.0, totals["succ_edges_next"]),
        succ_ce_correct_cluster=totals["succ_nll_next_correct"] / max(1.0, totals["succ_edges_next_correct"]),
        row_entropy=totals["row_entropy_sum"] / max(1.0, totals["row_entropy_count"]),
        customer_joint_entropy=totals["cust_entropy_sum"] / max(1.0, totals["cust_entropy_count"]),
        customer_commitment=totals["commit_sum"] / max(1.0, totals["commit_count"]),
        expected_load_mean=totals["expected_load_sum"] / max(1.0, totals["expected_load_count"]),
        overload_mean=totals["overload_sum"] / max(1.0, totals["overload_count"]),
        overload_max_mean=totals["overload_max_sum"] / max(1.0, totals["overload_batch_count"]),
        any_overload_rate=totals["any_overload_sum"] / max(1.0, totals["overload_batch_count"]),
        log_temperature=log_temperature,
    )


def batch_iterator(args: argparse.Namespace, device: torch.device):
    data = np.load(args.data_npz, allow_pickle=True)
    end = min(len(data["depots"]), args.offset + args.limit)
    indices = list(range(args.offset, end))
    n_nodes = args.graph_size + 1

    for start in range(0, len(indices), args.batch_size):
        idxs = indices[start:start + args.batch_size]
        depots = data["depots"][idxs].astype(np.float32)
        locs = data["locs"][idxs].astype(np.float32)
        demands = data["demands"][idxs].astype(np.float32)
        capacities = data["capacities"][idxs].astype(np.int64)
        solutions = [data["solutions"][i] for i in idxs]
        target, target_mem = build_target_dense(solutions, args.fleet_size, n_nodes, device)

        if args.model_family == "fpin":
            batch = build_fpin_inputs(depots, locs, demands, capacities, args.fleet_size, device)
        else:
            batch = build_old_pim_inputs(depots, locs, demands, capacities, args.fleet_size, device)
        yield batch, target, target_mem, torch.from_numpy(demands).to(device)


def main() -> None:
    args = build_argparser().parse_args()
    device = torch.device(args.device)
    model = load_model(args, device)
    totals: Dict[str, float] = {"num_instances": 0.0}

    for (fleet, depot, customers, demands, dists), target, target_mem, demand_norm in batch_iterator(args, device):
        probs = forward_probs(model, args, fleet, depot, customers, demands, dists)
        stats = audit_batch(probs, target, target_mem, demand_norm)
        stats["num_instances"] = float(target.size(0))
        merge_stats(totals, stats)

    summary = finalize_stats(args, totals, model)
    print(json.dumps(asdict(summary), indent=2, sort_keys=True))
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(asdict(summary), indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
