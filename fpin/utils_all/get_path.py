from __future__ import annotations

import time

import numpy as np
import random
from fpin.utils_all.basic_funcs import zeros, ones
import sys
import itertools
from typing import List, Tuple, Optional
import torch



# fast_seeds_from_heatmap (new):
# Adds Gumbel noise and does a simple, fully GPU-side greedy with
# global single-visit,
# per-vehicle capacity checks,
# depot wrapping.
# Pros: Much faster, stays on CUDA, batched-friendly.
# Cons: No beam/repair; feasibility is enforced during construction, but quality can be lower than SBS (still good enough as seeds for HGS).
@torch.no_grad()
def fast_seeds_from_heatmap(
    edge_probs: torch.Tensor,     # [B,M,N,N], probs, diag=0, on CUDA
    demands: torch.Tensor,        # [B,N] (or [B,1,N] squeeze)
    capacity: float | torch.Tensor,  # scalar or [B,M]
    K: int = 8,
    tau: float = 1.0,
    sigma: float = 0.5,
    depot_idx: int = 0,
) -> List[List[List[List[int]]]]:
    """
    GPU-side K seeds per instance:
    - global single-visit
    - per-vehicle capacity respected
    - Gumbel perturb + softmax temperature for diversity
    Returns: seeds[b][k][m] = route list (depot-wrapped).
    """
    device = edge_probs.device
    print('device in sampling', device)
    if demands.dim() == 3: demands = demands.squeeze(-2)
    B, M, N, _ = edge_probs.shape
    # diag=0
    I = torch.eye(N, device=device, dtype=torch.bool).view(1,1,N,N)
    P = edge_probs.masked_fill(I, 0.0).clamp_min(1e-12)

    # capacities -> [B,M]
    if not torch.is_tensor(capacity):
        capacity = torch.as_tensor(capacity, device=device)
    cap = capacity
    if cap.dim() == 0: cap = cap.view(1,1).expand(B,M)
    elif cap.dim() == 1:
        cap = (cap.view(1,M).expand(B,M) if cap.numel()==M else cap.view(B,1).expand(B,M))

    assert cap.shape == (B,M)

    seeds_all: List[List[List[List[int]]]] = []
    for b in range(B):
        seeds_b: List[List[List[int]]] = []
        dem = demands[b]  # [N]
        for _ in range(K):
            # global served mask
            served = torch.zeros(N, dtype=torch.bool, device=device)
            served[depot_idx] = True
            # per-vehicle
            cap_left = cap[b].clone()
            routes_k: List[List[int]] = []
            # Gumbel perturb per seed
            U = torch.rand_like(P[b])
            gumb = -torch.log(-torch.log(U))
            logp = torch.log(P[b]) + sigma * gumb
            # decode each vehicle
            for m in range(M):
                route = [depot_idx]
                load = 0.0
                curr = depot_idx
                local = torch.zeros(N, dtype=torch.bool, device=device); local[depot_idx]=True

                # expand until no feasible customer
                while True:
                    logits = logp[m, curr].clone() / max(tau, 1e-6)  # [N]
                    # allowed: not served, not local, fits capacity; depot allowed only if none feasible
                    feas = (~served) & (~local) & (dem <= cap_left[m] + 1e-12)
                    feas[depot_idx] = False
                    allow_depot = not feas.any()
                    allowed = feas.clone()
                    allowed[depot_idx] |= allow_depot
                    # forbid self-loop on non-depot
                    if curr != depot_idx: allowed[curr] = False

                    if not allowed.any():
                        if route[-1] != depot_idx: route.append(depot_idx)
                        break

                    logits[~allowed] = float('-inf')
                    # sample next (stochastic greedy)
                    probs_row = torch.softmax(logits, dim=-1)
                    nxt = torch.multinomial(probs_row, 1).item()

                    if nxt == depot_idx:
                        if route[-1] != depot_idx: route.append(depot_idx)
                        break
                    # accept customer
                    d = float(dem[nxt].item())
                    if load + d > cap_left[m] + 1e-12:
                        # close if cannot fit
                        if route[-1] != depot_idx: route.append(depot_idx)
                        break
                    route.append(nxt)
                    load += d
                    local[nxt] = True
                    served[nxt] = True
                    curr = nxt

                # commit load
                cap_left[m] = max(0.0, cap_left[m].item() - load)
                routes_k.append(route if route[-1]==depot_idx else route+[depot_idx])

                # early break if all served
                if ((~served) & (dem > 0)).sum().item() == 0:
                    # pad remaining vehicles with [0,0]
                    for mm in range(m+1, M):
                        routes_k.append([depot_idx, depot_idx])
                    break

            seeds_b.append(routes_k)
        seeds_all.append(seeds_b)
    return seeds_all




def sbs_decode(edge_probs, demands=None, capacities=None, beam_size=5, max_steps=50):
    """
    Stochastic Beam Search decoding for VRP using predicted edge_probs as policy.

    edge_probs: Tensor [B, M, N, N]
        Edge probability heatmaps (softmaxed).
    Returns:
        best_paths: List[List[List[int]]] of shape [B][M][steps]
    """
    B, M, N, _ = edge_probs.shape
    device = edge_probs.device
    edge_probs = edge_probs.clone()

    # log-probs for sampling with Gumbel noise
    log_probs = edge_probs.log()

    best_paths = []

    for b in range(B):
        batch_paths = []
        for m in range(M):
            beam = [([0], 0.0)]  # start at depot node 0, path + score

            for step in range(max_steps):
                candidates = []
                for path, score in beam:
                    last_node = path[-1]

                    # Sample k next nodes (excluding visited)
                    mask = torch.ones(N, device=device)
                    mask[path] = 0  # mask visited nodes
                    logits = log_probs[b, m, last_node]  # (N,)
                    logits = logits + (mask + 1e-6).log()  # large neg for visited

                    # Add Gumbel noise to logits (sampling w/o replacement)
                    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits)))
                    topk = torch.topk(logits + gumbel_noise, beam_size)

                    for i in range(beam_size):
                        next_node = topk.indices[i].item()
                        next_score = score + topk.values[i].item()
                        new_path = path + [next_node]
                        candidates.append((new_path, next_score))

                # Keep top beam_size candidates
                beam = sorted(candidates, key=lambda x: -x[1])[:beam_size]

            # pick best finished path
            best_path = beam[0][0]
            print('best_path', best_path)
            batch_paths.append(best_path)

        best_paths.append(batch_paths)

    return best_paths


@torch.no_grad()
def gumbel_perturbed_sbs_seeds(
    edge_probs: torch.Tensor,           # [B, M, N, N]  (row-stochastic, diag=0)
    demands: torch.Tensor,              # [B, N] or [B, 1, N]
    capacities: torch.Tensor,           # [B, M] or broadcastable
    *,
    K: int = 8,                         # seeds per instance
    sigma: float = 0.5,                 # Gumbel noise scale (0.3–0.8 typical)
    beam_size: int = 5,                 # SBS beam (keep modest)
    depot_idx: int = 0,
    dists: Optional[torch.Tensor] = None,  # [B,N,N] optional for repair
    repair: bool = True,
) -> Tuple[List[List[List[List[int]]]], torch.Tensor]:
    """
    Returns:
      seeds: list shape [B][K][M][T_v], depot-wrapped, feasible (SBS + repair)
      loads: [B, K, M] loads per vehicle seed
    """
    device = edge_probs.device
    B, M, N, _ = edge_probs.shape

    # 1) Mask diagonal to exact zero (row-stochastic but no self-loops)
    I = torch.eye(N, device=device, dtype=torch.bool)
    edge_probs = edge_probs.masked_fill(I.view(1,1,N,N), 0.0).clamp_min(1e-12)

    # 2) Make stable log-probs
    logp = edge_probs.log()  # [B,M,N,N]

    # 3) Gumbel noise and softmax row-wise (preserve stochasticity per row)
    #    Shape to [B,K,M,N,N] so we create K perturbed copies per instance.
    U = torch.rand(B, K, M, N, N, device=device)
    gumbel = -torch.log(-torch.log(U.clamp_min(1e-12)))
    Z = logp.unsqueeze(1) + sigma * gumbel          # [B,K,M,N,N]
    Pk = torch.softmax(Z, dim=-1)                   # row-wise normalize

    # 4) Flatten K into batch axis and run SBS once (fast)
    Pk_flat = Pk.view(B*K, M, N, N)
    if demands.dim() == 3:
        demands = demands.squeeze(-2)
    demands_flat = demands.unsqueeze(1).expand(B, K, N).reshape(B*K, N)
    # capacities broadcast: accept [B,M], [M], or scalar
    if not torch.is_tensor(capacities):
        capacities = torch.as_tensor(capacities, device=device)
    if capacities.dim() == 0:              # scalar
        capacities_flat = capacities.view(1,1).expand(B*K, M)
    elif capacities.dim() == 1:
        if capacities.numel() == M:        # [M] shared
            capacities_flat = capacities.view(1,M).expand(B*K, M).contiguous()
        elif capacities.numel() == B:      # [B]
            capacities_flat = capacities.view(B,1).expand(B, M).unsqueeze(1).expand(B, K, M).reshape(B*K, M)
        else:
            raise ValueError("capacities 1D must match M or B")
    else:                                   # [B,M]
        capacities_flat = capacities.unsqueeze(1).expand(B, K, M).reshape(B*K, M)

    if dists is not None:
        dists_flat = dists.unsqueeze(1).expand(B, K, N, N).reshape(B*K, N, N).contiguous()
    else:
        dists_flat = None

    # Reuse your SBS to enforce single-visit + capacity + optional repair
    seeds_flat, loads_flat = sbs_decode_vrp(
        edge_probs=Pk_flat, demands=demands_flat, capacities=capacities_flat,
        beam_size=beam_size, max_steps=200, depot_idx=depot_idx,
        dists=dists_flat, repair=repair, overflow_eps=0.0
    )
    # seeds_flat: list length B*K, each is [M][T_v]; loads_flat: [B*K, M]

    # 5) Unflatten back to [B][K][M][T_v]
    seeds: List[List[List[List[int]]]] = []
    loads = loads_flat.view(B, K, M).contiguous()
    for b in range(B):
        inst = []
        for k in range(K):
            inst.append(seeds_flat[b*K + k])  # list of M routes
        seeds.append(inst)

    return seeds, loads



from typing import List, Tuple, Optional
import torch

@torch.no_grad()
def sbs_decode_vrp(
    edge_probs: torch.Tensor,           # [B, M, N, N]  (softmaxed upstream; we use log)
    demands: torch.Tensor,              # [B, N] or [B, 1, N]
    capacities: torch.Tensor,           # [B, M] or broadcastable
    beam_size: int = 5,
    max_steps: int = 200,
    depot_idx: int = 0,
    dists: Optional[torch.Tensor] = None,   # [B, N, N] (optional, used by repair for cheapest insertion)
    repair: bool = True,
    overflow_eps: float = 0.0,              # 0.0 => strictly no overflow in repair
) -> Tuple[List[List[List[int]]], torch.Tensor]:
    """
    SBS decoder for CVRP with hard single-visit and capacity constraints + optional repair.

    Returns:
        paths: List[List[List[int]]] with shape [B][M][T_v], each starts/ends at depot
        loads: torch.Tensor [B, M] sum of served customer demands per vehicle

    Notes:
      - Depot is *disallowed* while any feasible (unserved, fits-capacity) customer remains.
      - Global single-visit is enforced via `served` mask.
      - If some customers remain unserved after all vehicles, an optional repair assigns them
        to vehicles with remaining capacity (cheapest insertion if `dists` provided; else append
        before the final depot). Repair is *strict*: no overflow beyond `overflow_eps`.
    """
    assert edge_probs.dim() == 4, "edge_probs must be [B, M, N, N]"
    B, M, N, _ = edge_probs.shape
    device = edge_probs.device

    # Stable log
    probs = edge_probs.clamp_min(1e-12)
    log_probs = probs.log()

    # Normalize demands to [B, N]
    if demands.dim() == 3:
        demands = demands.squeeze(-2)
    demands = demands.to(device)
    assert demands.shape == (B, N), f"demands must be [B, N], got {demands.shape}"

    # Capacities to [B, M]
    if not torch.is_tensor(capacities):
        capacities = torch.as_tensor(capacities, device=device)
    capacities = capacities.to(device)
    if capacities.dim() == 0:
        capacities = capacities.view(1, 1).expand(B, M)
    elif capacities.dim() == 1:
        assert capacities.numel() in (M, B), "capacities 1D must match M or B"
        capacities = (capacities.view(1, M).expand(B, M)
                      if capacities.numel() == M else
                      capacities.view(B, 1).expand(B, M))
    assert capacities.shape == (B, M), f"capacities must be [B, M], got {capacities.shape}"

    if dists is not None:
        assert dists.shape == (B, N, N), f"dists must be [B, N, N], got {dists.shape}"
        dists = dists.to(device)

    all_paths: List[List[List[int]]] = []
    all_loads = torch.zeros(B, M, device=device)

    for b in range(B):
        dem_b = demands[b]                 # [N]
        served = torch.zeros(N, dtype=torch.bool, device=device)
        served[depot_idx] = True           # depot excluded

        batch_paths: List[List[int]] = []
        veh_loads = torch.zeros(M, device=device)
        cap_left = capacities[b].clone()   # [M]

        # --- decode each vehicle ---
        for m in range(M):
            # Beam entry: (path, score, cap_left_m, local_vis, terminated)
            local_vis0 = torch.zeros(N, dtype=torch.bool, device=device)
            local_vis0[depot_idx] = True
            beam = [([depot_idx], 0.0, cap_left[m].item(), local_vis0, False)]

            for _ in range(max_steps):
                # stop if all beams terminated
                if all(term for _, _, _, _, term in beam):
                    break

                cand = []
                for path, score, cap_m, local_vis, term in beam:
                    if term:
                        cand.append((path, score, cap_m, local_vis, True))
                        continue

                    last = path[-1]

                    # Feasible customers for this vehicle and state
                    feas_customers = (~served) & (~local_vis) & (dem_b <= cap_m)
                    feas_customers[depot_idx] = False

                    # Depot allowed only if no feasible customers remain
                    allow_depot = not feas_customers.any()

                    allowed = feas_customers.clone()
                    allowed[depot_idx] = allow_depot

                    if last != depot_idx:
                        allowed[last] = False  # avoid self-loop (non-depot)

                    if not allowed.any():
                        # force close
                        if last != depot_idx:
                            new_path = path + [depot_idx]
                            cand.append((new_path, score, cap_m, local_vis.clone(), True))
                        else:
                            cand.append((path, score, cap_m, local_vis.clone(), True))
                        continue

                    logits = log_probs[b, m, last].clone()  # [N]
                    logits[~allowed] = float("-inf")
                    if not torch.isfinite(logits).any():
                        # numerical fallback: close tour
                        if last != depot_idx:
                            new_path = path + [depot_idx]
                            cand.append((new_path, score, cap_m, local_vis.clone(), True))
                        else:
                            cand.append((path, score, cap_m, local_vis.clone(), True))
                        continue

                    # Gumbel top-k expansion
                    gumbel = -torch.log(-torch.log(torch.rand_like(logits)))
                    noisy = logits + gumbel
                    k = min(beam_size, int(allowed.sum().item()))
                    vals, idxs = torch.topk(noisy, k)

                    for v, nxt in zip(vals.tolist(), idxs.tolist()):
                        if nxt == depot_idx:
                            new_path = path + [nxt]
                            cand.append((new_path, score + v, cap_m, local_vis.clone(), True))
                        else:
                            lv = local_vis.clone()
                            lv[nxt] = True
                            new_path = path + [nxt]
                            new_cap = cap_m - dem_b[nxt].item()
                            cand.append((new_path, score + v, new_cap, lv, False))

                # prune
                cand.sort(key=lambda x: -x[1])
                beam = cand[:beam_size] if cand else beam

            # pick best and close at depot
            if beam:
                beam.sort(key=lambda x: -x[1])
                best_path, _, cap_m_final, _, _ = beam[0]
            else:
                best_path, cap_m_final = [depot_idx, depot_idx], cap_left[m].item()

            if best_path[-1] != depot_idx:
                best_path = best_path + [depot_idx]

            # commit to global served and load
            load_m = 0.0
            for node in best_path:
                if node != depot_idx and not served[node]:
                    served[node] = True
                    load_m += dem_b[node].item()

            veh_loads[m] = load_m
            cap_left[m] = max(0.0, cap_left[m].item() - load_m)
            batch_paths.append(best_path)

            # optional fast exit if everything is served
            if ( (~served) & (dem_b > 0) ).sum().item() == 0:
                # fill remaining vehicles with [0,0]
                for mm in range(m+1, M):
                    batch_paths.append([depot_idx, depot_idx])
                    veh_loads[mm] = 0.0
                break

        # --- REPAIR (strict; no overflow beyond overflow_eps) ---
        if repair:
            unserved = ((~served) & (dem_b > 0)).nonzero(as_tuple=True)[0].tolist()
            if unserved:
                for cust in unserved:
                    demand_c = dem_b[cust].item()
                    # candidate vehicles with enough capacity
                    feas_veh = [mm for mm in range(M) if cap_left[mm].item() + 1e-12 >= demand_c + overflow_eps]

                    if not feas_veh:
                        # instance infeasible under strict capacity; skip (or raise)
                        # You can choose to allow mild overflow by setting `overflow_eps>0`.
                        continue

                    # choose vehicle and insertion position
                    best_mm, best_pos, best_delta = None, None, float("inf")
                    for mm in feas_veh:
                        path = batch_paths[mm]
                        # find position before final depot; test all arcs (i->j)
                        # positions: insert at index k (between path[k-1] and path[k])
                        if dists is not None:
                            # cheapest insertion
                            for k in range(1, len(path)):  # between k-1 and k
                                i = path[k-1]; j = path[k]
                                delta = dists[b, i, cust].item() + dists[b, cust, j].item() - dists[b, i, j].item()
                                if delta < best_delta:
                                    best_delta = delta
                                    best_mm = mm
                                    best_pos = k
                        else:
                            # append before final depot (simple, distance-agnostic)
                            best_mm = mm
                            best_pos = len(path) - 1
                            best_delta = 0.0
                            break  # good enough without distances

                    # commit insertion
                    if best_mm is not None:
                        batch_paths[best_mm].insert(best_pos, cust)
                        veh_loads[best_mm] += demand_c
                        cap_left[best_mm] -= demand_c
                        served[cust] = True

        all_paths.append(batch_paths)
        all_loads[b] = veh_loads

        # Final strict sanity (optional)
        # assert ((~served) & (dem_b > 0)).sum().item() == 0, "Unserved customers remain!"
        # assert torch.all(all_loads[b] <= capacities[b] + 1e-9), "Capacity overflow detected!"

    return all_paths, all_loads


def sample_routes_from_heatmap(Pm, demands, capacity, depot=0, topk=8, tau=1.0):
    """
    Pm: [M,N,N] probs for one instance (already diag=0)
    returns: list of M routes (each depot-wrapped)
    """
    M, N, _ = Pm.shape
    routes = []
    for m in range(M):
        visited = torch.zeros(N, dtype=torch.bool, device=Pm.device)
        visited[depot] = True
        load = 0.0
        curr = depot
        route = [depot]
        while True:
            logits = torch.log(Pm[m, curr].clamp_min(1e-9))  # [N]
            # mask visited & infeasible by capacity (if you can check)
            mask = visited.clone()
            logits = logits.masked_fill(mask, float('-inf'))
            # restrict to top-k to avoid noise
            idx = torch.topk(logits, k=min(topk, N)).indices
            sub = logits[idx] / tau
            probs = torch.softmax(sub, dim=-1)
            nxt = idx[torch.multinomial(probs, 1).item()].item()
            if mask[nxt]:  # no valid next
                break
            # capacity check (if demands is available):
            d = float(demands[nxt].item()) if nxt != depot else 0.0
            if load + d > capacity:
                break
            route.append(nxt)
            visited[nxt] = True
            load += d
            curr = nxt
            if visited.all():
                break
        route.append(depot)
        routes.append(route)
    return routes



def sbs_decode_with_constraints_OLD2(
    edge_probs: torch.Tensor,         # [B, M, N, N]  (probabilities or logits passed through softmax upstream)
    demands: torch.Tensor,            # [B, N] or [B, 1, N]
    capacities: torch.Tensor,         # [B, M] or scalar per vehicle (use torch.ones if all 1.0)
    beam_size: int = 5,
    max_steps: int = 200,             # allow enough steps to visit many customers
    depot_idx: int = 0
) -> Tuple[List[List[List[int]]], torch.Tensor]:
    """
Stochastic Beam Search decoding for CVRP with capacity + single-visit constraints.

    Returns:
        paths: List[List[List[int]]] with shape [B][M][T_v] (each tour starts/ends at depot)
        loads: torch.Tensor with shape [B, M] (sum of demands served by each vehicle)
    """
    assert edge_probs.dim() == 4, "edge_probs must be [B, M, N, N]"
    B, M, N, _ = edge_probs.shape
    device = edge_probs.device

    # Stable logs
    probs = edge_probs.clamp_min(1e-12)
    log_probs = probs.log()

    # Normalize demands to [B, N]
    if demands.dim() == 3:
        demands = demands.squeeze(-2)
    assert demands.shape == (B, N), f"demands must be [B, N], got {demands.shape}"
    demands = demands.to(device)

    # Capacities to [B, M]
    if not torch.is_tensor(capacities):
        capacities = torch.as_tensor(capacities, device=device)
    capacities = capacities.to(device)
    if capacities.dim() == 0:
        capacities = capacities.view(1, 1).expand(B, M)
    elif capacities.dim() == 1:
        assert capacities.numel() in (M, B), "capacities 1D must match M or B"
        if capacities.numel() == M:
            capacities = capacities.view(1, M).expand(B, M)
        else:
            capacities = capacities.view(B, 1).expand(B, M)
    assert capacities.shape == (B, M), f"capacities must be [B, M], got {capacities.shape}"

    all_batch_paths: List[List[List[int]]] = []
    all_batch_loads = torch.zeros(B, M, device=device)

    for b in range(B):
        # Global single-visit set (True => already served by some vehicle)
        served = torch.zeros(N, dtype=torch.bool, device=device)
        served[depot_idx] = True  # depot is not a customer

        batch_paths: List[List[int]] = []
        veh_loads = torch.zeros(M, device=device)

        for m in range(M):
            cap_left = capacities[b, m].item()

            # Beam entries: (path, score, cap_left, local_vis, terminated)
            local_init = torch.zeros(N, dtype=torch.bool, device=device)
            local_init[depot_idx] = True
            beam: List[Tuple[List[int], float, float, torch.Tensor, bool]] = [
                ([depot_idx], 0.0, cap_left, local_init, False)
            ]

            for step in range(max_steps):
                # Stop if all beams already terminated
                if all(term for _, _, _, _, term in beam):
                    break

                candidates: List[Tuple[List[int], float, float, torch.Tensor, bool]] = []

                for path, score, cap, local_vis, terminated in beam:
                    if terminated:
                        # Keep terminated beams as-is
                        candidates.append((path, score, cap, local_vis, True))
                        continue

                    last_node = path[-1]
                    dem_b = demands[b]  # [N]

                    # Feasible customer set for this vehicle at this state:
                    feas_customers = (~served) & (~local_vis) & (dem_b <= cap)
                    feas_customers[depot_idx] = False  # treat depot separately

                    # Allow depot only if no feasible customers remain
                    allow_depot = not feas_customers.any()

                    # Build allowed mask
                    allowed = feas_customers.clone()
                    if allow_depot:
                        allowed[depot_idx] = True
                    else:
                        allowed[depot_idx] = False

                    # Avoid trivial self-loop (stay on same node if it's not depot)
                    if last_node != depot_idx:
                        allowed[last_node] = False

                    if not allowed.any():
                        # If nothing allowed, force close at depot if not already there
                        if last_node != depot_idx:
                            new_path = path + [depot_idx]
                            candidates.append((new_path, score, cap, local_vis.clone(), True))
                        else:
                            candidates.append((path, score, cap, local_vis.clone(), True))
                        continue

                    # Prepare logits with mask
                    logits = log_probs[b, m, last_node].clone()  # [N]
                    logits[~allowed] = float("-inf")

                    # If numerical issues create all -inf, close at depot
                    if not torch.isfinite(logits).any():
                        if last_node != depot_idx:
                            new_path = path + [depot_idx]
                            candidates.append((new_path, score, cap, local_vis.clone(), True))
                        else:
                            candidates.append((path, score, cap, local_vis.clone(), True))
                        continue

                    # Stochastic expansion via Gumbel Top-k
                    gumbel = -torch.log(-torch.log(torch.rand_like(logits)))
                    noisy = logits + gumbel
                    k = min(beam_size, int(allowed.sum().item()))
                    top_vals, top_idx = torch.topk(noisy, k)

                    for v, nxt in zip(top_vals.tolist(), top_idx.tolist()):
                        if nxt == depot_idx:
                            # Terminate this tour
                            new_path = path + [nxt]
                            candidates.append((new_path, score + v, cap, local_vis.clone(), True))
                        else:
                            # Extend to a new feasible customer
                            new_path = path + [nxt]
                            new_score = score + v
                            new_cap = cap - dem_b[nxt].item()

                            lv = local_vis.clone()
                            lv[nxt] = True

                            candidates.append((new_path, new_score, new_cap, lv, False))

                # Prune to beam_size best
                candidates.sort(key=lambda x: -x[1])
                beam = candidates[:beam_size] if candidates else beam

                # If any beam has just terminated at depot AND there remain feasible customers,
                # we still keep exploring other non-terminated beams in subsequent iterations.
                # The outer "all(term...)" check will eventually stop when all are terminated.

            # Select best beam and ensure depot closure
            if beam:
                beam.sort(key=lambda x: -x[1])
                best_path, _, _, _, _ = beam[0]
            else:
                best_path = [depot_idx, depot_idx]

            if best_path[-1] != depot_idx:
                best_path = best_path + [depot_idx]

            # Compute load for this vehicle and mark customers served
            load = 0.0
            for node in best_path:
                if node != depot_idx and not served[node]:
                    load += demands[b, node].item()
                    served[node] = True  # global single-visit

            veh_loads[m] = load
            batch_paths.append(best_path)

            # Early stop if all customers served
            if served.sum().item() == 1 + (demands[b] > 0).sum().item():  # depot + all customers with >0 demand
                # fill remaining vehicles with trivial [0,0] tours
                for mm in range(m + 1, M):
                    batch_paths.append([depot_idx, depot_idx])
                    veh_loads[mm] = 0.0
                break

        all_batch_paths.append(batch_paths)
        all_batch_loads[b] = veh_loads

        # Optional: final sanity—if some customers remain unserved, you can add a repair step here.
        # For strict feasibility in every instance, ensure model logits don’t push depot too early
        # and capacities are sufficient for the instance.

    return all_batch_paths, all_batch_loads

def sbs_decode_with_constraints_OLD(
    edge_probs: torch.Tensor,
    demands: torch.Tensor,
    capacities: torch.Tensor,
    beam_size: int = 5,
    max_steps: int = 50
) -> List[List[List[int]]]:
    """
    Stochastic Beam Search decoding for VRP with capacity and feasibility constraints.

    Args:
        edge_probs (Tensor): [B, M, N, N] - edge probability heatmaps.
        demands (Tensor): [B, N] - customer demands.
        capacities (Tensor): [B, M] - vehicle capacities.
        beam_size (int): number of beams to keep per vehicle.
        max_steps (int): max steps per vehicle.

    Returns:
        List[List[List[int]]]: [B][M][steps] - decoded paths for each vehicle in each batch.
        - prevents visiting any customer more than once (both globally across vehicles and locally within the current path),
        - stops expanding a path once it returns to the depot,
        - always closes a path at the depot,
        - respects capacity constraints,
        - is robust to demands being shaped [B, 1, N] or [B, N],

works whether capacities is a tensor or you pass a scalar 1.0 equivalent.
    """
    B, M, N, _ = edge_probs.shape
    print('B, M, N', B, M, N)
    device = edge_probs.device
    edge_probs = edge_probs.clone()
    log_probs = edge_probs.log()

    all_batch_paths = []

    for b in range(B):
        batch_paths = []
        visited = set([0])  # depot already visited
        remaining_demands = demands[b].squeeze(0).clone()
        # print('remaining_demands', remaining_demands.size())
        depot_idx = 0

        for m in range(M):
            beam: List[Tuple[List[int], float, float]] = [([depot_idx], 0.0, capacities[b, m].item())]  # (path, score, capacity_left)

            for step in range(max_steps):
                candidates = []

                for path, score, cap_left in beam:
                    last_node = path[-1]

                    logits = log_probs[b, m, last_node]  # [N]
                    mask = torch.ones(N, device=device)
                    for i in range(N):
                        if i in visited or (remaining_demands[i] > cap_left and i != depot_idx):
                            mask[i] = 0
                    if mask.sum() == 0:
                        continue  # no feasible extension

                    logits = logits + (mask + 1e-6).log()
                    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits)))
                    topk = torch.topk(logits + gumbel_noise, beam_size)

                    for i in range(beam_size):
                        next_node = topk.indices[i].item()
                        new_path = path + [next_node]
                        new_score = score + topk.values[i].item()

                        if next_node == depot_idx:
                            candidates.append((new_path, new_score, cap_left))
                        else:
                            demand = remaining_demands[next_node].item()
                            if demand <= cap_left:
                                candidates.append((new_path, new_score, cap_left - demand))

                if not candidates:
                    break

                beam = sorted(candidates, key=lambda x: -x[1])[:beam_size]

                # If depot was selected, stop early
                if any(p[-1] == depot_idx for p, _, _ in beam):
                    break

            # pick best path
            best_path = beam[0][0] if beam else [depot_idx, depot_idx]
            batch_paths.append(best_path)

            # update visited and demands
            for node in best_path:
                if node != depot_idx:
                    visited.add(node)
                    remaining_demands[node] = 0.0

        all_batch_paths.append(batch_paths)

    return all_batch_paths




# make CURR GREEDY PATH VALID (single INSTANCE)
# '''fixing routes; returns capacity conform VRP sols ACCORDING to probabilities'''
def make_valid(all_idxs, probs, remaining_capa, demand):
    failed = None
    m = probs.size(0)
    n = probs.size(1)
    probs_cut = probs.clone()
    demands_cut = demand
    all_visited_ = []
    for i in range(m):
        #print('vehicle i', i)
        #print('visisted by vehicle i', targ_as_lst(all_idxs[i].max(1)[1], n))
        #print('visisted by vehicle i', targ_as_lst(all_idxs[i].max(1)[1], n)[1:-1])
        all_visited_.extend(targ_as_lst(all_idxs[i].max(1)[1], n)[1:-1])
    batch_missing = torch.tensor(np.setdiff1d(list(np.arange(1, n)), sorted(all_visited_)),
                                 device=probs_cut.device).long()
    batch_missing_ = batch_missing[demands_cut[0][batch_missing].sort(dim=0, descending=True)[1]]
    batch_missing_ = list(batch_missing_)
    # print('batch_missing_',batch_missing_)

    for j in batch_missing_:
        #print('missing node j:',j)
        #print('remaining_capa',remaining_capa)
        # choose possible vehicles
        available_vs = torch.where(remaining_capa + 0.000001 >= demands_cut[0][j].repeat(m))[0]
        #print('available_vs',available_vs)
        if list(available_vs):
            # check which direction has the highest prob for node j:
            if probs_cut[available_vs, :, j].max(1)[0].max(0)[0] > probs_cut[available_vs, j, :].max(1)[0].max(0)[0]:
                # TO-j-direction has highest
                v_to = available_vs[probs_cut[available_vs, :, j].max(1)[0].max(0)[1]]
                #print('v_to',v_to)
                # nodes available in v_to's route
                nodes_available = torch.tensor(targ_as_lst(all_idxs[v_to].max(1)[1], n)[:-1], device=v_to.device)
                c_to_idx = \
                torch.where(probs_cut[v_to, nodes_available, j] == probs_cut[v_to, nodes_available, j].max(0)[0].max())[
                    0]
                if list(c_to_idx):
                    c_to_idx = c_to_idx[0].unsqueeze(0)
                c_to = nodes_available[c_to_idx]
                #print('c_to',c_to)
                c_from_c_to = all_idxs[v_to, c_to, :].max(1)[1]
                all_idxs[v_to, c_to, :] = 0.0
                all_idxs[v_to, c_to, j] = 1.0
                # change prev. output node from c_to to be output node of j instead
                all_idxs[v_to, j, c_from_c_to] = 1.0
                #print('updated path for v_to', targ_as_lst(all_idxs[v_to].max(1)[1], n)[1:-1])
                v_ = v_to
            else:
                # FROM-j-direction has highest
                v_from = available_vs[probs_cut[available_vs, j, :].max(1)[0].max(0)[1]]
                #print('v_from',v_from)
                nodes_available = torch.tensor(targ_as_lst(all_idxs[v_from].max(1)[1], n)[:-1], device=v_from.device)
                c_from_idx = torch.where(
                    probs_cut[v_from, j, nodes_available] == probs_cut[v_from, j, nodes_available].max(0)[0].max())
                if list(c_from_idx):
                    c_from_idx = c_from_idx[0].unsqueeze(0)
                c_from = nodes_available[c_from_idx[0]]
                #print('c_from',c_from)
                c_to_c_from = all_idxs[v_from, :, c_from].max(0)[1]
                all_idxs[v_from, :, c_from] = 0.0
                all_idxs[v_from, j, c_from] = 1.0
                # change prev. input node to c_from to be input to j instead
                all_idxs[v_from, c_to_c_from, j] = 1.0
                #print('updated path for v_from', targ_as_lst(all_idxs[v_from].max(1)[1], n)[1:-1])
                v_ = v_from
            remaining_capa[v_] = remaining_capa[v_] - demands_cut[0][j]
        else:
            failed = 'yes'

    demands_ = demands_cut.unsqueeze(2).expand(m, n, n)
    final_loads = (demands_ * all_idxs).sum(dim=2).sum(dim=1)

    # check if really all cities covered:
    final_routes = []
    all_visited = []
    for i in range(m):
        #print('final route for i in m:',targ_as_lst(all_idxs[i].max(1)[1], n))
        final_routes.append(targ_as_lst(all_idxs[i].max(1)[1], n))
        all_visited.extend(targ_as_lst(all_idxs[i].max(1)[1], n)[1:-1])
    missing_final = torch.tensor(np.setdiff1d(list(np.arange(1, n)), sorted(all_visited)),
                                 device=final_loads.device).long()
    #print('missing_final',missing_final)

    return all_idxs, final_routes, final_loads, missing_final


# for ONE INSTANCE GET CURR GREEDY PATH (not necessarilly valid)
def greedy_path(probs, demand, current_perm):
    # only one instance probs and demands
    m = probs.size(0)
    n = probs.size(1)

    probs_ = probs.clone()
    all_idxs = zeros(*(probs_.size(0), probs_.size(1), probs_.size(1)))
    remaining_capa = ones(*(probs_.size(0),))
    other = ones(*(probs_.size(1), probs_.size(1))) * -99.0
    for v in current_perm:
        #print('CURRENT v:', v)
        # get starting batches
        curr_idx = probs_[v, :, :].max(1)[1][0]
        #print('starting idx for v',curr_idx)
        # remaining capa after starts for this vehicle
        demands_curr_idx = demand.squeeze(0)[curr_idx]
        #print('demands_curr_idx',demands_curr_idx)
        remaining_capa[v] = remaining_capa[v] - demands_curr_idx
        #print('remaining_capa[v]',remaining_capa[v])
        all_idxs[v, 0, curr_idx] = 1.0
        # update probs that curr. idx cannot be chosen anymore
        probs_[:, :, curr_idx] = -99.0
        # update probs for new remaining capa --> starting idx already
        # took capacity
        condition = (remaining_capa[v].unsqueeze(0).unsqueeze(1).expand(probs_.size(1),
                                                                            probs_.size(1)) >= demand.expand(
                probs_.size(1), probs_.size(1)))
        probs_[v, :, :] = torch.where(condition, probs_[v, :, :], other)
        while (curr_idx != 0):
            # get next idxs
            next_idx = probs_[v, :, :].max(1)[1][curr_idx]
            #print('next_idx',next_idx)
            # update current_idxs Mask
            all_idxs[v, curr_idx, next_idx] = 1.0
            # fix all_idxs for catched terminated paths (0 (curr) --> 0 (next))
            all_idxs[v, 0, 0] = 0.0
            # update probs after chosen
            probs_[:, :, next_idx] = -99.0
            # fix that depot is visitable multiple times
            probs_[:, :, 0] = probs[:, :, 0]
            # update remaining capa
            demands_next_idx = demand[:, next_idx]
            remaining_capa[v] = remaining_capa[v] - demands_next_idx
            #print('remaining_capa[v] in loop:',remaining_capa[v])
            # update probs
            condition = (remaining_capa[v].unsqueeze(0).unsqueeze(1).expand(probs_.size(1),
                                                                            probs_.size(1)) >= demand.expand(
                probs_.size(1), probs_.size(1)))
            probs_[v, :, :] = torch.where(condition, probs_[v, :, :], other)
            # update curr idxs
            curr_idx = next_idx

    return all_idxs, remaining_capa


# for ALL BATCHES GET CURR LOAD ESTIMATE (func) # _full
def load_estimate_full(probs, demand, perm_m):
    probs_ = probs.clone()
    b, m, n, _ = probs_.size()
    all_idxs = zeros(*(probs.size(0), probs.size(1), probs.size(2), probs.size(2)))
    arr = torch.arange(all_idxs.size(0)).cuda() if torch.cuda.is_available() else torch.arange(all_idxs.size(0))
    for v in perm_m:
        # get starting batches
        curr_batch_idxs = probs_[:, v, :, :].max(2)[1][:, 0]
        all_idxs[arr, v, zeros(*(curr_batch_idxs.size(0),)).long(), curr_batch_idxs] = 1.0
        probs_[arr, :, :, curr_batch_idxs] = -99.0
        while (curr_batch_idxs != 0).any():
            # get next idxs
            next_batch_idx = probs_[:, v, :, :].max(2)[1].gather(1, curr_batch_idxs.unsqueeze(1)).squeeze()
            # catch termintated paths
            next_batch_idx = torch.where(curr_batch_idxs == 0, curr_batch_idxs, next_batch_idx)
            # update current_idxs Mask
            all_idxs[arr, v, curr_batch_idxs, next_batch_idx] = 1.0
            # fix all_idxs for catched terminated paths (0 (curr) --> 0 (next))
            all_idxs[arr, v, 0, 0] = 0.0
            # update probs after chosen
            probs_[arr, :, :, next_batch_idx] = -99.0
            # fix that depot is visitable multiple times
            probs_[arr, :, :, 0] = probs[arr, :, :, 0]
            # update curr idxs
            curr_batch_idxs = next_batch_idx

    loads = (demand.unsqueeze(2).expand(b, m, n, n) * all_idxs).sum(dim=3).sum(dim=2)

    return all_idxs, loads

# Key wins:
# no [B,M,N,N] adjacency allocation
# no all_idxs[...] = 1 writes
# the loop is still there, but it’s way cheaper and fully on GPU
def load_estimate(probs, demand, perm_m):
    B, M, N, _ = probs.shape
    device = probs.device
    demand = demand.to(device).squeeze(-1)  # ensure [B,N]

    probs_ = probs.clone()
    visited = torch.zeros((B, N), device=device, dtype=torch.bool)
    visited[:, 0] = False

    loads = torch.zeros((B, M), device=device)
    next_of = torch.full((B, M, N), 0, device=device, dtype=torch.long)
    arr = torch.arange(B, device=device)

    for v in perm_m:
        curr = torch.zeros((B,), device=device, dtype=torch.long)
        active = torch.ones((B,), device=device, dtype=torch.bool)  # reset per vehicle
        left_depot = torch.zeros((B,), device=device, dtype=torch.bool)

        for _ in range(N):
            masked = probs_[:, v, :, :].masked_fill(visited[:, None, :], -1e9)
            masked[:, :, 0] = probs[:, v, :, 0]  # depot allowed

            nxt = masked[arr, curr].argmax(dim=-1)

            # terminate only after the vehicle has left depot at least once and returns
            returning = left_depot & (nxt == 0)
            active = active & (~returning)
            nxt = torch.where(active, nxt, torch.zeros_like(nxt))

            left_depot = left_depot | (nxt != 0)

            next_of[arr, v, curr] = nxt
            take = (nxt != 0) & (~visited[arr, nxt])
            loads[arr, v] += demand[arr, nxt] * take.float()
            visited[arr, nxt] |= take
            curr = nxt

            if (~active).all():
                break

    return next_of, loads

import torch
import torch.nn.functional as F

def _pool_sym_scores_from_logits(edge_logits: torch.Tensor, pool: str = "mean") -> torch.Tensor:
    """
    edge_logits: [B,M,N,N]
    returns S_sym: [B,N,N] in [0,1], symmetric, diag ~0 (left as-is; caller can mask)
    """
    p_v = torch.sigmoid(edge_logits)  # [B,M,N,N]
    if pool == "any":
        S = 1.0 - torch.prod(1.0 - p_v, dim=1)  # [B,N,N]
    elif pool == "mean":
        S = p_v.mean(dim=1)                     # [B,N,N]
    else:
        raise ValueError(f"pool must be 'mean' or 'any', got {pool}")
    S = 0.5 * (S + S.transpose(-1, -2))         # sym
    return S

def _routes_to_next_of_and_loads(routes_bm, demand_b: torch.Tensor, N: int, device):
    """
    routes_bm: list of M routes, each [0, ..., 0]
    demand_b:  [N] (demand[0]=0)
    returns next_of [M,N], loads [M]
    """
    M = len(routes_bm)
    next_of = torch.zeros((M, N), dtype=torch.long, device=device)
    loads = torch.zeros((M,), dtype=torch.float32, device=device)

    for v, r in enumerate(routes_bm):
        # ensure depot-wrapped
        if len(r) < 2:
            r = [0, 0]
        if r[0] != 0: r = [0] + r
        if r[-1] != 0: r = r + [0]

        load = 0.0
        for a, b in zip(r[:-1], r[1:]):
            next_of[v, a] = b
            if b != 0:
                load += float(demand_b[b].item())
        loads[v] = load
    return next_of, loads

def decode_heatmap_candidates_insertion(
    edge_logits: torch.Tensor,      # [B,M,N,N] (preferred)
    dists: torch.Tensor,            # [B,N,N]
    demand: torch.Tensor,           # [B,N] (demand[0]=0)
    capacity: float = 1.0,
    pool: str = "mean",

    # candidates
    cand_k: int = 32,               # top-K neighbors per node by score
    depot_cand_k: int = 64,         # allow more from depot row for seeding

    # objective mixing
    lambda_score: float = 0.15,     # how strongly heatmap biases insertion vs distance (0.05..0.3 typical)
    use_only_candidates: bool = True,  # if True, only insert customers that appear in candidate sets

    # fleet handling
    perm_m=None,                    # optional vehicle order; default 0..M-1

    # safety
    eps: float = 1e-9,
):
    """
    Deterministic feasible construction:
      1) pool+sym scores S_sym [B,N,N]
      2) build candidate sets per node (topK under S_sym)
      3) seed each vehicle with one customer from depot row (unique + capacity-feasible)
      4) iteratively best-insert remaining customers (capacity-feasible)

    Returns:
      routes: list length B of list length M of routes (each route depot-wrapped)
      next_of: [B,M,N]
      loads:   [B,M]
      stats:   dict with cost/served/feasible flags
    """
    device = edge_logits.device
    start_decode = time.time()
    B, M, N, _ = edge_logits.shape
    assert dists.shape[:2] == (B, N) and dists.shape[2] == N
    assert demand.shape == (B, N)

    if perm_m is None:
        perm_m = list(range(M))

    # pooled symmetric scores in [0,1]
    S = _pool_sym_scores_from_logits(edge_logits, pool=pool)  # [B,N,N]

    # mask diagonal
    diag = torch.eye(N, device=device, dtype=torch.bool)
    S = S.masked_fill(diag[None, :, :], 0.0)

    # precompute candidate indices per node: [B,N,K]
    K = min(int(cand_k), N)
    cand_idx = torch.topk(S, k=K, dim=-1).indices  # [B,N,K]

    # richer candidates from depot row for seeding: [B,K0]
    K0 = min(int(depot_cand_k), N-1)
    depot_scores = S[:, 0, :]                      # [B,N]
    depot_scores[:, 0] = -1.0
    depot_top = torch.topk(depot_scores, k=K0, dim=-1).indices  # [B,K0]

    routes_all = []
    next_of_all = torch.zeros((B, M, N), dtype=torch.long, device=device)
    loads_all = torch.zeros((B, M), dtype=torch.float32, device=device)

    total_cost = torch.zeros((B,), dtype=torch.float32, device=device)
    served_cnt = torch.zeros((B,), dtype=torch.long, device=device)
    feasible = torch.ones((B,), dtype=torch.bool, device=device)

    # ---- per instance (B usually smallish; this is readable + debuggable) ----
    for b in range(B):
        dem = demand[b]          # [N]
        dist = dists[b]          # [N,N]
        S_b = S[b]               # [N,N]
        cand_b = cand_idx[b]     # [N,K]
        depot_top_b = depot_top[b]

        # state
        unserved = set(range(1, N))
        used = [0.0 for _ in range(M)]
        routes = [[0, 0] for _ in range(M)]  # depot-wrapped, will insert between

        # ---- 1) seed routes with distinct customers from depot row ----
        # assign best unique feasible customer per vehicle
        for v in perm_m:
            chosen = None
            for u in depot_top_b.tolist():
                if u in unserved and (used[v] + float(dem[u].item()) <= capacity + 1e-8):
                    chosen = u
                    break
            if chosen is None:
                # leave vehicle unused
                continue
            # route [0, u, 0]
            routes[v] = [0, chosen, 0]
            used[v] += float(dem[chosen].item())
            unserved.remove(chosen)

        # ---- helper: compute best insertion position in a route ----
        def best_insertion_for_customer(route, u):
            """
            route: [0, ..., 0]
            returns (best_delta, best_pos_index)
              insert u between route[pos] and route[pos+1]
            """
            best_val = None
            best_pos = None
            # iterate over edges (a,b) in current route
            for pos in range(len(route) - 1):
                a = route[pos]
                c = route[pos + 1]
                # distance delta
                dc = float(dist[a, u].item() + dist[u, c].item() - dist[a, c].item())
                # heatmap gain (sym, uses S)
                ds = float(S_b[a, u].item() + S_b[u, c].item() - S_b[a, c].item())
                # combined objective (lower is better)
                val = dc - lambda_score * ds
                if best_val is None or val < best_val:
                    best_val = val
                    best_pos = pos
            return best_val, best_pos

        # ---- 2) iterative best insertion ----
        # To keep it cheap, only consider candidate customers for each route:
        # candidates = union of cand(node) for nodes on route.
        while len(unserved) > 0:
            best_move = None  # (val, v, u, pos)

            for v in perm_m:
                # capacity left
                rem = capacity - used[v]
                if rem <= 1e-8:
                    continue

                route = routes[v]
                # build route-based candidate set
                cand_set = set()
                if use_only_candidates:
                    for node in route:
                        if node == 0:
                            # allow more from depot row
                            cand_set.update(depot_top_b.tolist())
                        else:
                            cand_set.update(cand_b[node].tolist())
                    # keep only unserved + feasible by capacity
                    cand_list = [u for u in cand_set if (u in unserved and float(dem[u].item()) <= rem + 1e-8)]
                else:
                    # consider all unserved feasible
                    cand_list = [u for u in unserved if float(dem[u].item()) <= rem + 1e-8]

                if not cand_list:
                    continue

                # evaluate best insertion among these candidates
                for u in cand_list:
                    val, pos = best_insertion_for_customer(route, u)
                    if pos is None:
                        continue
                    if best_move is None or val < best_move[0]:
                        best_move = (val, v, u, pos)

            if best_move is None:
                # no feasible insertion found (should be rare if fleet/capacity consistent)
                feasible[b] = False
                break

            _, v_star, u_star, pos_star = best_move
            # apply insertion
            routes[v_star].insert(pos_star + 1, u_star)
            used[v_star] += float(dem[u_star].item())
            unserved.remove(u_star)

        # ---- 3) compute cost + successor map ----
        cost_b = 0.0
        seen = set()
        for v in range(M):
            r = routes[v]
            # ensure depot-wrapped
            if r[0] != 0: r = [0] + r
            if r[-1] != 0: r = r + [0]
            routes[v] = r

            for a, c in zip(r[:-1], r[1:]):
                cost_b += float(dist[a, c].item())
                if c != 0:
                    if c in seen:
                        feasible[b] = False
                    seen.add(c)

            if used[v] > capacity + 1e-6:
                feasible[b] = False

        # served count (unique customers served)
        served_cnt[b] = len(seen)
        if len(seen) != (N - 1):
            feasible[b] = False

        total_cost[b] = cost_b

        # next_of + loads tensors
        next_of_b, loads_b = _routes_to_next_of_and_loads(routes, dem, N=N, device=device)
        next_of_all[b] = next_of_b
        loads_all[b] = loads_b

        routes_all.append(routes)

    stats = {
        "total_cost": total_cost,     # [B]
        "served_cnt": served_cnt,     # [B]
        "feasible": feasible,         # [B]
    }
    print('decode took: ', time.time() - start_decode)
    return routes_all, next_of_all, loads_all, stats


def load_estimate_capacity(probs, demand, perm_m, capacity=1.0, max_steps=None, instance_id=None):
    """
    probs:  [B,M,N,N] (scores or probs; higher is better)
    demand: [B,N] or [B,N,1] or [B,1,N]
    returns:
      next_of: [B,M,N] successor map (next_of[b,v,cur]=nxt)
      loads:   [B,M] accumulated demand per vehicle
    """
    B, M, N, _ = probs.shape
    device = probs.device
    neg_inf = -1e9
    max_steps = N if max_steps is None else int(max_steps)
    paths = [[[] for _ in range(M)] for _ in range(B)]  # python lists, cheap for B small

    # demand -> [B,N]
    demand = demand.to(device)
    if demand.dim() == 3 and demand.size(-1) == 1: demand = demand.squeeze(-1)
    elif demand.dim() == 3 and demand.size(1) == 1: demand = demand.squeeze(1)

    # capacity -> [B]
    if torch.is_tensor(capacity):
        cap = capacity.to(device).view(-1)
        if cap.numel() == 1: cap = cap.expand(B)
    else:
        cap = torch.full((B,), float(capacity), device=device)

    visited = torch.zeros((B, N), device=device, dtype=torch.bool)
    visited[:, 0] = False

    next_of = torch.zeros((B, M, N), device=device, dtype=torch.long)
    loads = torch.zeros((B, M), device=device)

    arr = torch.arange(B, device=device)

    for v in perm_m:
        cur = torch.zeros((B,), device=device, dtype=torch.long)
        used = torch.zeros((B,), device=device)
        active = torch.ones((B,), device=device, dtype=torch.bool)
        left = torch.zeros((B,), device=device, dtype=torch.bool)

        # start each vehicle route with depot
        for b in range(B):
            paths[b][v].append(0)

        for _ in range(max_steps):
            scores = probs[arr, v, cur]            # [B,N]
            # visited + capacity masks on destinations
            scores = scores.masked_fill(visited, neg_inf)
            rem = (cap - used).clamp(min=0.0)      # [B]
            scores = scores.masked_fill(demand > rem[:, None], neg_inf)
            scores[:, 0] = probs[arr, v, cur, 0]   # depot always allowed
            # print('scores[0]', scores[0])

            # forbid staying at depot BEFORE leaving (prevents [0,0] everywhere)
            at_start = (cur == 0) & (~left) & active

            # feasible non-depot exists?
            feasible_non_depot = (scores[:, 1:] > (neg_inf / 2)).any(dim=-1)

            # if at start and no feasible customer, stop this vehicle for those batches
            dead_start = at_start & (~feasible_non_depot)

            # don't append anything new; just deactivate
            active = active & (~dead_start)

            scores[at_start, 0] = neg_inf
            scores[dead_start, 0] = 0.0  # harmless; they are inactive now anyway

            all_bad = scores.max(dim=-1).values <= (neg_inf / 2)

            # after computing all_bad
            dead_at_start = all_bad & (cur == 0) & (~left) & active
            if dead_at_start.any():
                # stop these batch elements for this vehicle (vehicle unused)
                active = active & (~dead_at_start)
                # (optional) ensure we don't write successors for them

            # only allow depot as fallback AFTER the vehicle has left
            scores[all_bad & left, 0] = 0.0

            # if we're at start and everything is masked, keep depot forbidden
            scores[all_bad & (~left) & (cur == 0), 0] = neg_inf

            # print("b0 v", v, "cur", int(cur[0].item()),
            #       "top5 idx/val:",
            #       torch.topk(scores[0], 5).indices.tolist(),
            #       torch.topk(scores[0], 5).values.tolist(),
            #       "depot_score:", float(scores[0, 0]))
            nxt = scores.argmax(dim=-1)            # [B]
            # print('nxt[0]', nxt[0])

            # record route step for active batches
            # for b in arr[active].tolist():
            #     paths[b][v].append(int(nxt[b].item()))

            # record route step for active batches
            for b in arr[active].tolist():
                if not (cur[b].item() == 0 and nxt[b].item() == 0 and (not left[b].item())):
                    paths[b][v].append(int(nxt[b].item()))

            # terminate after leaving and returning
            left = left | (nxt != 0)  # update left FIRST

            # terminate after leaving and returning
            returning = left & (nxt == 0)
            active = active & (~returning)
            nxt = torch.where(active, nxt, torch.zeros_like(nxt))

            # record successor
            # next_of[arr, v, cur] = nxt
            next_of[arr[active], v, cur[active]] = nxt[active] # --> only iterate further if still active batch series
            # print('next_of[0]', next_of[0])
            # print('next_of[0][v]', next_of[0][v])


            take = (nxt != 0) & (~visited[arr, nxt]) & active
            used = used + demand[arr, nxt] * take.float()
            loads[arr, v] = used
            visited[arr, nxt] |= take

            cur = nxt
            if (~active).all():
                break
        # print("b0 v", v, "succ0:", int(next_of[0, v, 0].item()))

    return next_of, loads, paths


# FOR TESTING --> return to depot only when you must (no feasible customer)
def load_estimate_capacity_test(probs, demand, perm_m, capacity=1.0, max_steps=None, instance_id=None):
    """
    probs:  [B,M,N,N] (scores or probs; higher is better)
    demand: [B,N] or [B,N,1] or [B,1,N]
    returns:
      next_of: [B,M,N] successor map (next_of[b,v,cur]=nxt)
      loads:   [B,M] accumulated demand per vehicle
    """
    B, M, N, _ = probs.shape
    device = probs.device
    neg_inf = -1e9
    max_steps = N if max_steps is None else int(max_steps)
    paths = [[[] for _ in range(M)] for _ in range(B)]  # python lists, cheap for B small

    # demand -> [B,N]
    demand = demand.to(device)
    if demand.dim() == 3 and demand.size(-1) == 1:
        demand = demand.squeeze(-1)
    elif demand.dim() == 3 and demand.size(1) == 1:
        demand = demand.squeeze(1)

    # capacity -> [B]
    if torch.is_tensor(capacity):
        cap = capacity.to(device).view(-1)
        if cap.numel() == 1:
            cap = cap.expand(B)
    else:
        cap = torch.full((B,), float(capacity), device=device)

    visited = torch.zeros((B, N), device=device, dtype=torch.bool)
    visited[:, 0] = False

    next_of = torch.zeros((B, M, N), device=device, dtype=torch.long)
    loads = torch.zeros((B, M), device=device)

    arr = torch.arange(B, device=device)

    for v in perm_m:
        cur = torch.zeros((B,), device=device, dtype=torch.long)
        used = torch.zeros((B,), device=device)
        active = torch.ones((B,), device=device, dtype=torch.bool)
        left = torch.zeros((B,), device=device, dtype=torch.bool)

        # start each vehicle route with depot
        for b in range(B):
            paths[b][v].append(0)

        for _ in range(max_steps):
            scores = probs[arr, v, cur]  # [B,N]

            # visited + capacity masks on destinations
            scores = scores.masked_fill(visited, neg_inf)
            rem = (cap - used).clamp(min=0.0)  # [B]
            scores = scores.masked_fill(demand > rem[:, None], neg_inf)

            # depot allowed by score (but we still may forbid it at start later)
            scores[:, 0] = probs[arr, v, cur, 0]

            # -------------------------
            # PATCH 2 (place it HERE)
            # Prevent argmax on all-masked rows and handle "no feasible move" safely.
            # -------------------------
            # For currently-active batch elements, check if ANY move is feasible at all.
            # (Using neg_inf/2 as your feasibility threshold.)
            row_has_any = (scores > (neg_inf / 2)).any(dim=-1)  # [B] bool

            # Among active batch elements, which ones are completely stuck?
            stuck = active & (~row_has_any)

            if stuck.any():
                # If vehicle already left: force depot return and end route for those batch elems
                stuck_after_left = stuck & left
                # If vehicle hasn't left and we're at depot: deactivate vehicle for those batch elems
                stuck_at_start = stuck & (~left) & (cur == 0)

                # Note: For stuck_at_start we just deactivate; we do NOT choose nxt via argmax.
                active = active & (~stuck_at_start)

                # For stuck_after_left we will force nxt=0 and treat them as returning below.
                # (We keep them active for this step so we can record the depot return cleanly.)
            # -------------------------
            # END PATCH 2
            # -------------------------

            # forbid staying at depot BEFORE leaving (prevents [0,0] everywhere)
            at_start = (cur == 0) & (~left) & active

            # feasible non-depot exists? (for active only, after masks)
            feasible_non_depot = (scores[:, 1:] > (neg_inf / 2)).any(dim=-1)

            # if at start and no feasible customer, stop this vehicle for those batches
            dead_start = at_start & (~feasible_non_depot)

            # deactivate those batches for this vehicle
            active = active & (~dead_start)

            # apply depot-forbid at start (only for still-active)
            scores[at_start, 0] = neg_inf

            # You had this line; keep it harmless (they are inactive now anyway)
            scores[dead_start, 0] = 0.0

            all_bad = scores.max(dim=-1).values <= (neg_inf / 2)

            # (Optional) dead-at-start secondary check (kept, but now mostly redundant)
            dead_at_start = all_bad & (cur == 0) & (~left) & active
            if dead_at_start.any():
                active = active & (~dead_at_start)

            # only allow depot as fallback AFTER the vehicle has left
            scores[all_bad & left, 0] = 0.0

            # if we're at start and everything is masked, keep depot forbidden
            scores[all_bad & (~left) & (cur == 0), 0] = neg_inf

            # Compute nxt for everyone (even inactive rows won't matter; we zero them later)
            nxt = scores.argmax(dim=-1)  # [B]

            # If Patch 2 found "stuck after left", force nxt=0 for those batches
            # (recompute mask cheaply here)
            row_has_any = (scores > (neg_inf / 2)).any(dim=-1)
            stuck = active & (~row_has_any)
            stuck_after_left = stuck & left
            if stuck_after_left.any():
                nxt = torch.where(stuck_after_left, torch.zeros_like(nxt), nxt)

            # record route step for active batches (avoid appending 0->0 at start)
            for b in arr[active].tolist():
                if not (cur[b].item() == 0 and nxt[b].item() == 0 and (not left[b].item())):
                    paths[b][v].append(int(nxt[b].item()))

            # update left FIRST
            left = left | (nxt != 0)

            # terminate after leaving and returning
            returning = left & (nxt == 0)
            active = active & (~returning)
            nxt = torch.where(active, nxt, torch.zeros_like(nxt))

            # record successor only for still-active
            next_of[arr[active], v, cur[active]] = nxt[active]

            take = (nxt != 0) & (~visited[arr, nxt]) & active
            used = used + demand[arr, nxt] * take.float()
            loads[arr, v] = used
            visited[arr, nxt] |= take

            cur = nxt
            if (~active).all():
                break

    return next_of, loads, paths


from typing import List, Sequence, Tuple
import torch

def split_route_by_capacity(
    route: Sequence[int],
    demand_1d: torch.Tensor,   # [N] on CPU or GPU
    capacity: float = 1.0,
) -> List[List[int]]:
    """
    Takes a single route like [0, a, b, c, 0] (may contain repeats/oddities),
    and splits it into multiple depot-to-depot routes so that each segment
    respects capacity.

    Returns a list of routes (each starts/ends with 0). Keeps the order.
    """
    # sanitize to python ints
    r = [int(x) for x in route]
    # strip leading/trailing depot for easier processing
    while len(r) > 0 and r[0] == 0: r = r[1:]
    while len(r) > 0 and r[-1] == 0: r = r[:-1]

    segs: List[List[int]] = []
    cur = [0]
    used = 0.0

    for node in r:
        if node == 0:
            continue
        d = float(demand_1d[node].item())
        # if node alone exceeds cap, we still create a "violating single" (you can drop/flag instead)
        if used + d > capacity and len(cur) > 1:
            cur.append(0)
            segs.append(cur)
            cur = [0]
            used = 0.0

        cur.append(node)
        used += d

    if len(cur) > 1:
        cur.append(0)
        segs.append(cur)

    return segs


def enforce_capacity_split(
    routes: List[List[int]],
    demand_1d: torch.Tensor,   # [N]
    capacity: float = 1.0,
) -> List[List[int]]:
    """
    Applies split_route_by_capacity to a list of routes.
    """
    out: List[List[int]] = []
    for r in routes:
        out.extend(split_route_by_capacity(r, demand_1d, capacity))
    return out


def load_estimate_capacity_(probs, demand, perm_m, capacity=1.0, max_steps=None):
    B, M, N, _ = probs.shape
    device = probs.device

    # demand -> [B,N]
    demand = demand.to(device)
    if demand.dim() == 3 and demand.size(-1) == 1:      # [B,N,1]
        demand = demand.squeeze(-1)
    elif demand.dim() == 3 and demand.size(1) == 1:     # [B,1,N]
        demand = demand.squeeze(1)
    elif demand.dim() != 2:
        raise RuntimeError(f"Unexpected demand shape: {tuple(demand.shape)}")
    if demand.size(1) != N:
        raise RuntimeError(f"demand N={demand.size(1)} vs probs N={N}")

    # capacity -> [B]
    if not torch.is_tensor(capacity):
        capacity = torch.full((B,), float(capacity), device=device)
    else:
        capacity = capacity.to(device).view(-1)
        if capacity.numel() == 1:
            capacity = capacity.expand(B)

    probs_ = probs.clone()
    visited = torch.zeros((B, N), device=device, dtype=torch.bool)
    visited[:, 0] = False  # depot allowed

    loads = torch.zeros((B, M), device=device)
    next_of = torch.full((B, M, N), 0, device=device, dtype=torch.long)

    arr = torch.arange(B, device=device)
    max_steps = N if max_steps is None else int(max_steps)

    for v in perm_m:
        curr = torch.zeros((B,), device=device, dtype=torch.long)
        used = torch.zeros((B,), device=device)
        active = torch.ones((B,), device=device, dtype=torch.bool)
        left_depot = torch.zeros((B,), device=device, dtype=torch.bool)

        for _ in range(max_steps):
            masked = probs_[:, v, :, :]  # [B,N,N]

            # mask visited destinations (except depot)
            masked = masked.masked_fill(visited[:, None, :], -1e9)

            # capacity mask
            remaining = (capacity - used).clamp(min=0.0)
            infeasible = demand > remaining[:, None]      # [B,N]
            infeasible[:, 0] = False
            masked = masked.masked_fill(infeasible[:, None, :], -1e9)

            # make depot selectable but not dominant
            masked[:, :, 0] = 0.0

            scores = masked[arr, curr]                     # [B,N]
            # if everything is masked, force depot
            all_bad = scores.max(dim=-1).values < -1e8
            scores[all_bad, 0] = 0.0

            nxt = scores.argmax(dim=-1)                    # [B]

            # terminate after leaving depot and returning
            returning = left_depot & (nxt == 0)
            active = active & (~returning)
            nxt = torch.where(active, nxt, torch.zeros_like(nxt))

            left_depot = left_depot | (nxt != 0)

            # next_of[arr, v, curr] = nxt
            next_of[arr, v].scatter_(1, curr.unsqueeze(1), nxt.unsqueeze(1))

            take = (nxt != 0) & (~visited[arr, nxt]) & active
            used = used + demand[arr, nxt] * take.float()
            loads[arr, v] = used

            visited[arr, nxt] |= take
            curr = nxt

            if (~active).all():
                break
            # print('next_of', next_of)
    # served = visited[:, 1:].float().mean()
    # n_served = visited[:, 1:].sum(dim=1).float().mean()
    # print('served', served)
    # print('n_served', n_served)

    # print('served = visited[:, 1:].float().mean()', visited[:, 1:].float().mean())
    # print('visited[:, 1:].sum(dim=1).float().mean()', visited[:, 1:].sum(dim=1).float().mean())
    return next_of, loads

@torch.no_grad()
def depot_logit_mins(edge_logits):
    # returns [N-1] mins over B,M for each i=1..N-1
    x = edge_logits[:, :, 1:, 0]          # [B,M,N-1]
    mins = x.amin(dim=(0,1))              # [N-1]
    return mins


def get_tour(idcs, n, start=0):
    # idcs_trg: [N] successor map: next[node] = successor
    nxt_idx = [start]
    cur = start
    for _ in range(n):
        nxt = int(idcs[cur].item())
        if nxt == 0:
            nxt_idx.append(0)
            return nxt_idx
        nxt_idx.append(nxt)
        cur = nxt
    nxt_idx.append(0)
    return nxt_idx

def targ_as_lst(idcs_trg, n, print_indcs=False):
    '''idcs_trg is a tensor containing the indices of next visited nodes'''
    nxt_idx = [0]
    # print('idcs_trg', idcs_trg)
    nxt = idcs_trg[0].item()
    for _ in range(n):
        cur_idx = nxt
        if cur_idx != 0:
            nxt_idx.append(cur_idx)
            nxt = idcs_trg[cur_idx].item()
    nxt_idx.append(0)
    return nxt_idx

def get_tour_(idcs, n):
    route = [0]
    seen = set([0])
    cur = int(idcs[0])
    for _ in range(n):
        if cur == 0 or cur in seen:
            break
        route.append(cur)
        seen.add(cur)
        cur = int(idcs[cur])
    route.append(0)
    return route


def is_valid_tour(path, n):
    return set(path[1:-1]) == set(range(1, n)) and path[0] == 0 and path[-1] == 0