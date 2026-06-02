"""Evaluation logic for the original Thyssens-2022 "old PIM" baseline.

Mirrors the role of ``fpin/fpin.py:eval_model`` but drives the *old* softassign
perm-invariant model whose source lives in ``models/PIMold/src/`` (copied verbatim from
the original ICLR-2022 / cluster code base). We reuse the old model ONLY for forward +
greedy decode to produce routes; the routes are wrapped as ``formats.RPSolution`` and
scored by the shared ``data.cvrp_dataset.CVRPDataset.feasibility_check`` (cost_v = travel
distance + #routes * c_v, c_v in {35,50,80} for Q in {30,40,50}). This guarantees the PIM
baseline is scored identically to F-PIN.

The old src/ uses bare top-level imports (``from VRPModel_attn1 import ...``,
``from utils_all.get_path import ...``), so we put ``models/PIMold/src`` on sys.path before
importing it. Those bare names don't collide with the repo's package-qualified
``fpin.utils_all`` / ``data`` imports.
"""
import os
import sys
import time
import random
import logging
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.spatial import distance_matrix
import torch

from formats import CVRPInstance, RPSolution

logger = logging.getLogger(__name__)

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
_VRP100_SRC = os.path.join(_SRC, "vrp100_cluster", "supvrp_0")

# Per-size architecture (from src/eval1.py model_specs) — the checkpoints differ in shape.
OLD_MODEL_SPECS = {
    20:  dict(layers=7, main_dim=256, n_hidden=1024),
    50:  dict(layers=9, main_dim=256, n_hidden=1024),
    # VRP100 ckpt at sup-vrp-copy/ICLR_2022/.../outs100/outs_2/VRP_model_100.pth
    # differs from VRP20/VRP50 ckpts in TWO dims:
    #   - FF hidden  = 512 (not 1024); embed conv weight [512,128,1]
    #   - cities_dim = 4 (not 3); first embed conv weight [128,4,1]
    100: dict(layers=7, main_dim=128, n_hidden=512, cities_dim=4),
}
# Shared model_params (src/eval1.py), in constructor argument order after main_dim.
_SHARED = dict(
    depot_dim=4, cities_dim=3, fleet_dim=4, dropout=0.0, softassign_layers=0,
    avg_pool=False, residual=True, norm=True, self_pool=False, embedding_norm=True,
    weighting=True, with_loads=False, attn_dotproduct=True, memory_efficient=True,
)
# Fixed vehicle cost c_v by problem size (Golden FSMVRP; matches the paper / VEHICLE_COSTS).
VEHICLE_COST_BY_N = {20: 35.0, 50: 50.0, 60: 50.0, 100: 80.0}


def _ensure_src_on_path(problem_size: int = None):
    if _SRC not in sys.path:
        sys.path.insert(0, _SRC)
    if problem_size == 100 and _VRP100_SRC not in sys.path:
        sys.path.insert(0, _VRP100_SRC)


def build_old_model(problem_size: int, device: torch.device) -> torch.nn.Module:
    """Instantiate the old softassign VRP_Net with the size-specific architecture."""
    _ensure_src_on_path(problem_size)
    spec = OLD_MODEL_SPECS[problem_size]
    if problem_size == 100:
        # memory-modified VRP100 variant (M=11 OOM fix)
        from VRPModel_attn import VRP_Net  # noqa: E402  (from vrp100_cluster/supvrp_0)
    else:
        from VRPModel_attn1 import VRP_Net  # noqa: E402
    cities_dim = spec.get("cities_dim", _SHARED["cities_dim"])
    model = VRP_Net(
        spec["layers"], _SHARED["depot_dim"], cities_dim, _SHARED["fleet_dim"],
        spec["main_dim"], _SHARED["avg_pool"], _SHARED["residual"], _SHARED["norm"],
        spec["n_hidden"], _SHARED["dropout"], _SHARED["self_pool"], _SHARED["embedding_norm"],
        _SHARED["softassign_layers"], _SHARED["weighting"], _SHARED["with_loads"],
        _SHARED["attn_dotproduct"], _SHARED["memory_efficient"],
    )
    return model.to(device)


def _pack_routes(custs, demands, cap: float = 1.0):
    """First-fit-decreasing pack of customers into capacity-feasible routes (depot implied)."""
    routes = []
    loads = []
    for c in sorted(custs, key=lambda x: -float(demands[x])):
        d = float(demands[c])
        placed = False
        for i, load in enumerate(loads):
            if load + d <= cap + 1e-6:
                routes[i].append(int(c))
                loads[i] += d
                placed = True
                break
        if not placed:
            routes.append([int(c)])
            loads.append(d)
    return routes


def _route_distance(routes, coords) -> float:
    """Total euclidean travel distance over depot-closed routes (for LS warm-start cost)."""
    total = 0.0
    for r in routes:
        seq = list(r)
        if not seq:
            continue
        if seq[0] != 0:
            seq = [0] + seq
        if seq[-1] != 0:
            seq = seq + [0]
        for a, b in zip(seq[:-1], seq[1:]):
            total += float(np.linalg.norm(coords[a] - coords[b]))
    return total


def _cvrp_instance_to_old_input(inst: CVRPInstance, m: int):
    """Build the old model's (fleet, depot, customer, demand, dists) tuple from a CVRPInstance.

    Replicates the feature layout of src/data_utils/preprocess1.py:transform_Xs:
      fleet   [m,4]   = [vehicle_idx/m, m, capacity/capacity(=1.0), total_norm_demand]
      depot   [1,4]   = [x, y, depot_demand(=0), depot_centrality]
      custom  [n,3]   = [x, y, norm_demand]
      demand  [1,n+1] = normalized demands (depot first); already /Q in node_features[:,-1]
      dists   [n+1,n+1] = euclidean distances (model re-normalizes internally)
    """
    coords = np.asarray(inst.coords, dtype=np.float64)            # [n+1, 2], row 0 = depot
    nf = np.asarray(inst.node_features, dtype=np.float64)
    demand_norm = nf[:, inst.constraint_idx[0]].astype(np.float64)  # [n+1], depot=0, normalized by Q
    n_nodes = coords.shape[0]

    dists = distance_matrix(coords, coords, p=2)                  # [n+1, n+1] euclidean
    total_dem = float(demand_norm.sum())

    veh_idx = np.arange(1, m + 1, dtype=np.float64) / float(m)
    fleet = np.stack(
        [veh_idx, np.full(m, float(m)), np.ones(m), np.full(m, total_dem)], axis=1
    )  # [m, 4]

    depot_centrality = float(n_nodes - 1) / float(dists[0].sum())
    depot = np.array([[coords[0, 0], coords[0, 1], demand_norm[0], depot_centrality]])  # [1, 4]

    custom = np.concatenate([coords[1:], demand_norm[1:, None]], axis=1)  # [n, 3]
    demand_array = demand_norm.reshape(1, -1)                              # [1, n+1]
    return fleet, depot, custom, demand_array, dists


def eval_model_pimold(
    model: torch.nn.Module,
    data_rp: List[CVRPInstance],
    normalised_data: bool,
    problem_size: int,
    problem: str,
    device: torch.device,
    opts: Any = None,
) -> Tuple[Dict[str, Any], List[RPSolution]]:
    """Run the old PIM model over ``data_rp``; return ({}, List[RPSolution]).

    Same return shape as ``fpin.fpin.eval_model`` so the inherited BaseConstructionRunner
    scores the resulting solutions exactly like F-PIN.
    """
    _ensure_src_on_path(problem_size)
    from utils_all.get_path import greedy_path, make_valid  # noqa: E402

    model.eval()
    m = int(getattr(opts, "nr_vehicles_eval", None) or data_rp[0].max_num_vehicles)
    v_cost = VEHICLE_COST_BY_N.get(problem_size)
    logger.info(f"[PIMold] eval N={problem_size} | M={m} | c_v={v_cost} | n_inst={len(data_rp)}")

    sols: List[Any] = []
    costs: List[Any] = []
    times: List[float] = []
    n_fail = 0

    def _to_t(a):
        return torch.as_tensor(a, dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        for inst in data_rp:
            t0 = time.time()
            fleet, depot, custom, dem, dists = _cvrp_instance_to_old_input(inst, m)
            depot_t, custom_t, fleet_t = _to_t(depot), _to_t(custom), _to_t(fleet)
            dem_t, dists_t = _to_t(dem), _to_t(dists)

            probs, _loads, _sample = model(depot_t, custom_t, fleet_t, dem_t, dists_t)

            # greedy decode (random vehicle permutation), with FROM-direction fallback
            perm = list(range(m))
            random.shuffle(perm)
            path_idxs, remain = greedy_path(probs[0], dem_t[0], perm)
            _gsol, groutes, _gloads, missing = make_valid(path_idxs, probs[0], remain, dem_t[0])
            if list(missing):
                path_idxs, remain = greedy_path(probs[0].transpose(1, 2), dem_t[0], perm)
                _gsol, groutes, _gloads, missing = make_valid(path_idxs, probs[0], remain, dem_t[0])

            coords = np.asarray(inst.coords, dtype=np.float64)
            base = [list(map(int, r)) for r in groutes if len(r) >= 3]
            if list(missing):
                # greedy left customers unassigned: guarantee a feasible solution by packing
                # the missing customers into extra capacity-feasible routes (mirrors eval1.py's
                # guarantee_solution). These extra routes may exceed M -> counted as fleet
                # violations by the evaluator, which is the honest accounting of a decode failure
                # (no silent dropping). Never emit None (the LS warm-start requires a route list).
                dem_norm = np.asarray(inst.node_features, dtype=np.float64)[:, inst.constraint_idx[0]]
                extra = _pack_routes([int(x) for x in missing], dem_norm, cap=1.0)
                sol = base + extra
                n_fail += 1
            else:
                sol = base
            cost = _route_distance(sol, coords)
            sols.append(sol)
            costs.append(cost)
            times.append(time.time() - t0)

    logger.info(f"[PIMold] greedy unsolved (pre-scoring): {n_fail}/{len(data_rp)}")
    solutions = [
        RPSolution(
            solution=sols[i],
            cost=costs[i],
            num_vehicles=(len(sols[i]) if sols[i] else None),
            run_time=times[i],
            problem=problem,
            instance=data_rp[i],
        )
        for i in range(len(sols))
    ]
    return {}, solutions
