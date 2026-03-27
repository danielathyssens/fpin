import torch
import numpy as np
import tqdm
from multiprocessing import Pool



def greedy_split_from_order(
    order,
    dists,
    dem,
    cap=1.0,
    vehicle_cost=0.0,
    coords=None,
):
    """
    Parameters
    ----------
    order : List
        Customer ids in 1..N (NO depot)
    dists : Tensor [N+1, N+1]
        Distance matrix including depot at index 0
    dem : Tensor [N+1]
        Demands including depot demand=0 at index 0
    cap : float
        Vehicle capacity (normalized or not – must match dem)
    vehicle_cost : float
        Optional fixed cost per vehicle

    Returns
    -------
    obj : float
        Total cost (travel + vehicle_costs)
    routes : List[List[int]]
        Routes as lists including depot, e.g. [0, i, j, k, 0]
    """

    depot = 0
    routes = []
    cur_route = [depot]

    obj = 0.0
    load = 0.0
    prev = depot
    ### CHECK
    # coords_np = coords.detach().cpu().numpy()

    for node in order: # .tolist():
        d = float(dem[node].item())

        # impossible instance (single customer too large)
        if d > cap + 1e-9:
            return float("inf"), None

        # start new route if capacity exceeded
        if load + d > cap + 1e-9:
            # close current route
            obj += torch.norm(coords[prev] - coords[depot], p=2)
                # float(dists[prev, depot].item()))
            obj += vehicle_cost
            cur_route.append(depot)
            routes.append(cur_route)

            # reset
            cur_route = [depot]
            load = 0.0
            prev = depot

        # go to node
        obj += torch.norm(coords[prev] - coords[node], p=2)
        # float(dists[prev, node].item())
        load += d
        cur_route.append(node)
        prev = node

    # close last route
    obj += torch.norm(coords[prev] - coords[depot], p=2)
    # float(dists[prev, depot].item())
    obj += vehicle_cost
    cur_route.append(depot)
    routes.append(cur_route)

    return obj, routes, len(routes)

def ffd_bins(dem: torch.Tensor, cap: float = 1.0) -> int:
    """
    dem: [N] including depot at 0, or [N-1] customers only
    returns number of bins used by First-Fit Decreasing
    """
    """
    dem: [N] or [1,N] or [B,N] (we assume single instance here)
    """
    # squeeze batch / singleton dims
    if dem.dim() > 1:
        dem = dem.squeeze()
    assert dem.dim() == 1, dem.shape

    # drop depot if present
    if dem.numel() > 0 and dem[0].abs() < 1e-12:
        d = dem[1:].clone()
    else:
        d = dem.clone()

    d = d[d > 1e-12]
    if d.numel() == 0:
        return 0
    d, _ = torch.sort(d, descending=True)

    bins = []  # remaining capacities
    for x in d.tolist():
        placed = False
        for bi in range(len(bins)):
            if bins[bi] >= x - 1e-12:
                bins[bi] -= x
                placed = True
                break
        if not placed:
            bins.append(cap - x)
    return len(bins)

# -----------------------------
# helpers
# -----------------------------
def route_load(route, dem_b):
    return sum(float(dem_b[node].item()) for node in route if node != 0)

def route_travel(route, D):
    if route is None or len(route) < 2:
        return 0.0
    c = 0.0
    for u, v in zip(route[:-1], route[1:]):
        c += float(D[u, v].item())
    return c

def insertion_delta(route, customer, D):
    """
    Best insertion delta of 'customer' into route [0, ..., 0].
    Returns (delta, pos).
    """
    best_delta = float("inf")
    best_pos = None
    for pos in range(len(route) - 1):
        i, j = route[pos], route[pos + 1]
        delta = float(D[i, customer] + D[customer, j] - D[i, j])
        if delta < best_delta:
            best_delta = delta
            best_pos = pos + 1
    return best_delta, best_pos

def greedy_order_from_start(score, start_j, N):
    """
    Build giant tour greedily from a fixed start node.
    score: [N,N]
    returns list of customers, length N-1
    """
    unv = torch.ones(N, device=score.device, dtype=torch.bool)
    unv[0] = False
    order = [start_j]
    unv[start_j] = False
    cur = start_j

    for _ in range(N - 2):
        cand = score[cur].clone()
        cand[~unv] = -1e9
        nxt = torch.argmax(cand).item()
        if cand[nxt].item() <= -1e8:
            rem = torch.nonzero(unv, as_tuple=False)
            if rem.numel() == 0:
                break
            nxt = rem[0].item()
        order.append(nxt)
        unv[nxt] = False
        cur = nxt
    return order

def greedy_capacity_split(order, dem_b, cap):
    """
    Always returns a capacity-feasible split along the giant tour.
    May use > M routes.
    """
    routes = []
    cur = [0]
    load = 0.0
    for node in order:
        q = float(dem_b[node].item())
        if load + q <= cap + 1e-9:
            cur.append(node)
            load += q
        else:
            cur.append(0)
            routes.append(cur)
            cur = [0, node]
            load = q
    cur.append(0)
    routes.append(cur)
    return routes

def repair_routes_to_M(routes, dem_b, D, M, cap):
    """
    Cheap repair:
    only tries to eliminate one extra route by moving its customers
    into the existing routes with cheapest feasible insertion.

    Assumes len(routes) is close to M, ideally M+1.
    """
    routes = [r[:] for r in routes if len(r) > 2]

    if len(routes) <= M:
        return routes

    # only try once: eliminate the lightest route
    loads = [sum(float(dem_b[node].item()) for node in r if node != 0) for r in routes]
    ridx = int(np.argmin(loads))
    extra_route = routes[ridx]
    customers = extra_route[1:-1]

    target_routes = [r[:] for j, r in enumerate(routes) if j != ridx]

    for cust in customers:
        q = float(dem_b[cust].item())

        best_t = None
        best_pos = None
        best_delta = float("inf")

        for j, target in enumerate(target_routes):
            load_j = sum(float(dem_b[node].item()) for node in target if node != 0)
            if load_j + q > cap + 1e-9:
                continue

            for pos in range(1, len(target)):
                i_prev = target[pos - 1]
                i_next = target[pos]
                delta = float(D[i_prev, cust] + D[cust, i_next] - D[i_prev, i_next])
                if delta < best_delta:
                    best_delta = delta
                    best_t = j
                    best_pos = pos

        if best_t is None:
            return routes  # repair failed, return original
        target_routes[best_t].insert(best_pos, cust)

    return target_routes

def repair_routes_to_M_ineff(routes, dem_b, D, M, cap):
    """
    Try to reduce #routes to <= M by moving customers from one route into others
    with residual capacity using cheapest insertion.
    """
    routes = [r[:] for r in routes if len(r) > 2]

    improved = True
    while len(routes) > M and improved:
        improved = False

        # eliminate lightest route first
        loads = [route_load(r, dem_b) for r in routes]
        ridx = int(np.argmin(loads))
        r = routes[ridx]
        customers = r[1:-1]

        # try to move all customers of route ridx elsewhere
        moves = []
        target_routes = [rt[:] for j, rt in enumerate(routes) if j != ridx]

        for cust in customers:
            q = float(dem_b[cust].item())
            best_t = None
            best_pos = None
            best_delta = float("inf")

            for j, target in enumerate(target_routes):
                if route_load(target, dem_b) + q > cap + 1e-9:
                    continue
                delta, pos = insertion_delta(target, cust, D)
                if delta < best_delta:
                    best_delta = delta
                    best_t = j
                    best_pos = pos

            if best_t is None:
                moves = None
                break
            moves.append((cust, best_t, best_pos))

            # apply tentatively so subsequent customers see updated load/structure
            target_routes[best_t].insert(best_pos, cust)

        if moves is not None:
            routes = target_routes
            improved = True

    return routes


# -----------------------------
# helpers
# -----------------------------
def route_travel_(route, D):
    if route is None or len(route) < 2:
        return 0.0
    c = 0.0
    for u, v in zip(route[:-1], route[1:]):
        c += float(D[u, v].item())
    return c


def greedy_capacity_split_(order, dem_b, cap):
    """
    Always capacity-feasible, may use > M routes.
    Returns routes without depot padding issues.
    """
    routes = []
    cur = [0]
    load = 0.0
    for node in order:
        q = float(dem_b[node].item())
        if load + q <= cap + 1e-9:
            cur.append(node)
            load += q
        else:
            cur.append(0)
            routes.append(cur)
            cur = [0, node]
            load = q
    cur.append(0)
    routes.append(cur)
    return routes


def greedy_order(score, dem_b, N, topk_start_seed, demand_bonus):
    """
    Build one demand-aware giant tour over customers 1..N-1.
    """
    unv = torch.ones(N, device=score.device, dtype=torch.bool)
    unv[0] = False

    # smart but cheap start:
    # among top depot-score candidates, pick the highest-demand one
    depot_scores = score[0, 1:]  # only customers
    k0 = min(topk_start_seed, N - 1)
    top_idx = torch.topk(depot_scores, k=k0).indices + 1
    start_j = top_idx[torch.argmax(dem_b[top_idx])].item()

    order = [start_j]
    unv[start_j] = False
    cur = start_j

    for _ in range(N - 2):
        cand = score[cur].clone()
        cand[~unv] = -1e9

        # tiny demand-aware bias: helps split without changing complexity
        cand = cand + demand_bonus * dem_b
        cand[~unv] = -1e9

        nxt = torch.argmax(cand).item()
        if cand[nxt].item() <= -1e8:
            rem = torch.nonzero(unv, as_tuple=False)
            if rem.numel() == 0:
                break
            nxt = rem[0].item()

        order.append(nxt)
        unv[nxt] = False
        cur = nxt

    return order


def exact_dp_split(order, dem_b, D, M, cap, vehicle_cost=0.0, allow_empty_routes=True):
    """
    Exact split of giant tour into <= M contiguous routes.
    Returns (routes, travel, k_used) or (None, inf, None).
    """
    T = order
    nC = len(T)
    if nC == 0:
        routes = [[0, 0] for _ in range(M)] if allow_empty_routes else []
        return routes, 0.0, 0

    q = dem_b[torch.tensor(T, device=dem_b.device)]
    pref = torch.zeros(nC + 1, device=dem_b.device)
    pref[1:] = torch.cumsum(q, dim=0)

    Ti = torch.tensor([0] + T, device=dem_b.device, dtype=torch.long)

    seg_cost = torch.full((nC + 1, nC + 1), float("inf"), device=dem_b.device)
    seg_ok = torch.zeros((nC + 1, nC + 1), dtype=torch.bool, device=dem_b.device)

    path_pref = torch.zeros(nC + 1, device=dem_b.device)
    for t in range(2, nC + 1):
        path_pref[t] = path_pref[t - 1] + D[Ti[t - 1], Ti[t]]

    for i in range(1, nC + 1):
        for j in range(i, nC + 1):
            load_ij = pref[j] - pref[i - 1]
            if load_ij <= cap + 1e-9:
                internal = path_pref[j] - path_pref[i]
                c = D[0, Ti[i]] + internal + D[Ti[j], 0]
                seg_cost[i, j] = c
                seg_ok[i, j] = True

    Kmax = min(M, nC)
    dp = torch.full((Kmax + 1, nC + 1), float("inf"), device=dem_b.device)
    prev = torch.full((Kmax + 1, nC + 1), -1, device=dem_b.device, dtype=torch.long)
    dp[0, 0] = 0.0

    for k in range(1, Kmax + 1):
        for j in range(1, nC + 1):
            best = float("inf")
            best_i = -1
            for i in range(1, j + 1):
                if not seg_ok[i, j]:
                    continue
                cand = dp[k - 1, i - 1] + seg_cost[i, j]
                if cand < best:
                    best = cand
                    best_i = i
            dp[k, j] = best
            prev[k, j] = best_i

    best_k = None
    best_obj = float("inf")
    best_travel = float("inf")

    for k in range(1, Kmax + 1):
        travel = dp[k, nC].item()
        if travel == float("inf"):
            continue
        obj = travel + vehicle_cost * k
        if obj < best_obj:
            best_obj = obj
            best_travel = travel
            best_k = k

    if best_k is None:
        return None, float("inf"), None

    routes = []
    k = best_k
    j = nC
    while k > 0 and j > 0:
        i = int(prev[k, j].item())
        if i < 1:
            break
        seg_customers = T[i - 1:j]
        routes.append([0] + seg_customers + [0])
        j = i - 1
        k -= 1
    routes.reverse()

    return routes, best_travel, best_k


# Optional: recommended if using GPUs or heavy libraries
# set_start_method('spawn', force=True)

def method_wrapper(args):
    """Unpacks the bundled tuple into the actual function."""
    return decode_mamd_split(*args)

def run_parallel_decode(dataset, num_workers, config):
    # dataset should be a list where each element is (edge_logits, dists, demands)
    # We bundle the instance data with the static config parameters
    task_args = [
        (
            item[0], # edge_logits
            item[1], # dists
            item[2], # demands
            config['capacity'],
            config['vehicle_cost'],
            config['max_nr_v_eval'],
            config['topk_start_seed'],
            config['demand_bonus']
        ) for item in dataset
    ]

    with Pool(processes=num_workers) as pool:
        # chunksize=1 is often better if decode_mamd_split is computationally heavy
        # to ensure better load balancing across workers.
        results = list(tqdm.tqdm(
            pool.imap(method_wrapper, task_args, chunksize=1),
            total=len(task_args),
            desc="Decoding MAMD Splits"
        ))

    # results is a list of (routes, costs, stats) tuples.
    # We need to unzip them:
    routes_v, costs_v, _ = zip(*results)
    return list(routes_v), torch.cat(costs_v) if isinstance(costs_v[0], torch.Tensor) else costs_v, None


def decode_mamd_split(
        edge_logits,  # [B, M, N, N] - The multi-agent logits
        dists,  # [B, N, N]
        demands,  # [B, N]
        capacity=1.0,
        vehicle_cost=0.0,
        max_nr_v_eval=11,
        topk_start_seed=3,
        demand_bonus=0.05
):
    """
    MAMD Decoder:
    Each vehicle 'm' generates a giant tour based on its own logit matrix.
    We split all M tours and pick the best one.
    """
    device = edge_logits.device
    B, M_logits, N, _ = edge_logits.shape
    M_max = max_nr_v_eval
    print('M_max', M_max)

    all_best_routes = []
    all_best_costs = torch.zeros(B, device=device)

    # Ensure demands is [B, N]
    if demands.dim() == 3:
        demands = demands.squeeze(1) if demands.size(1) == 1 else demands.squeeze(-1)

    for b in range(B):
        instance_dists = dists[b]
        instance_demands = demands[b]

        best_instance_cost = float('inf')
        best_instance_routes = None

        # MAMD Core: Iterate through each agent (vehicle index)
        for m in range(M_logits):
            # Get logits for agent 'm'
            score_m = edge_logits[b, m]

            # 1. Generate Giant Tour for this specific agent
            # Using your existing greedy_order logic
            order_m = greedy_order(score_m, instance_demands, N, topk_start_seed, demand_bonus)

            # 2. Split the tour
            # We try the standard order and the reversed order (standard Split heuristic)
            current_routes = None
            current_travel = float('inf')

            for trial_order in [order_m, list(reversed(order_m))]:
                routes, travel, k_used = exact_dp_split(
                    trial_order,
                    instance_demands,
                    instance_dists,
                    M_max,
                    capacity,
                    vehicle_cost=vehicle_cost
                )

                if routes is not None:
                    # Calculate total objective: Dist + (K * vehicle_cost)
                    total_obj = travel + (k_used * vehicle_cost)
                    if total_obj < current_travel:
                        current_travel = total_obj
                        current_routes = routes

            # 3. Fallback to Greedy if Agent 'm' failed to find a feasible DP split
            if current_routes is None:
                routes_greedy = greedy_capacity_split(order_m, instance_demands, capacity)
                # (Optional: insert your repair_routes_to_M logic here)
                travel_greedy = sum(route_travel(r, instance_dists) for r in routes_greedy)
                current_travel = travel_greedy + (len(routes_greedy) * vehicle_cost)
                current_routes = routes_greedy

            # Update best across all Agents for this instance
            if current_travel < best_instance_cost:
                best_instance_cost = current_travel
                best_instance_routes = current_routes

        all_best_routes.append(best_instance_routes)
        all_best_costs[b] = best_instance_cost  # Or just the routing part if preferred

    return all_best_routes[0], all_best_costs, None

def decode_parallel_greedy(
        edge_logits,      # [B, M, N, N]
        dists,            # [B, N, N]
        demands,          # [B, N]
        capacity=1.0,
        max_nr_v_eval=11,
        topk_per_vehicle=2,
        demand_bonus=0.03,
        dist_penalty=0.00,
):
    """
    Parallel Greedy Assignment Decoder (fleet-aware).

    Vehicles expand routes in parallel using their own heatmap slice.
    Conflicts are resolved globally.
    Depot decisions CLOSE the current route instead of silently resetting
    the vehicle and continuing inside the same route.
    """

    device = edge_logits.device
    B, M_logits, N, _ = edge_logits.shape
    M = min(M_logits, max_nr_v_eval)

    all_routes = []
    all_costs = torch.zeros(B, device=device)

    if demands.dim() == 3:
        demands = demands.squeeze(1) if demands.size(1) == 1 else demands.squeeze(-1)

    for b in range(B):
        A = edge_logits[b, :M]   # [M, N, N]
        D = dists[b]
        dem = demands[b]

        unvisited = set(range(1, N))

        # per-vehicle state
        cur = [0 for _ in range(M)]
        cap = [float(capacity) for _ in range(M)]
        current_routes = [[] for _ in range(M)]

        # finalized customer-only routes
        finished_routes = []

        # simple guard against infinite loops
        max_steps = N * M * 4
        steps = 0

        while unvisited and steps < max_steps:
            steps += 1
            proposals = []

            for m in range(M):
                i = cur[m]

                feasible = [
                    j for j in unvisited
                    if float(dem[j]) <= cap[m] + 1e-9
                ]

                # no feasible node: if route nonempty, close it
                if not feasible:
                    if current_routes[m]:
                        finished_routes.append(current_routes[m])
                        current_routes[m] = []
                        cur[m] = 0
                        cap[m] = float(capacity)
                    continue

                scores = []
                for j in feasible:
                    score = (
                        float(A[m, i, j])
                        + demand_bonus * float(dem[j])
                        - dist_penalty * float(D[i, j])
                    )
                    scores.append((score, j))

                scores.sort(reverse=True)
                best_score, best_j = scores[0]

                # depot option
                depot_score = float(A[m, i, 0])

                # if depot wins, CLOSE current route
                if current_routes[m] and depot_score > best_score:
                    finished_routes.append(current_routes[m])
                    current_routes[m] = []
                    cur[m] = 0
                    cap[m] = float(capacity)
                    continue

                # otherwise propose top-k feasible customers
                for score, j in scores[:topk_per_vehicle]:
                    proposals.append((score, m, j))

            if not proposals:
                break

            proposals.sort(reverse=True)
            assigned = set()

            for score, m, j in proposals:
                if j not in unvisited or j in assigned:
                    continue
                if float(dem[j]) > cap[m] + 1e-9:
                    continue

                current_routes[m].append(j)
                cap[m] -= float(dem[j])
                cur[m] = j
                assigned.add(j)

            for j in assigned:
                unvisited.remove(j)

        # close all still-open routes
        for m in range(M):
            if current_routes[m]:
                finished_routes.append(current_routes[m])

        # if still unvisited -> infeasible
        if unvisited:
            all_routes.append([])
            all_costs[b] = float("inf")
            continue

        # final sanity check: capacity
        feasible = True
        for r in finished_routes:
            cum_d = sum(float(dem[j]) for j in r)
            if cum_d > float(capacity) + 1e-6:
                feasible = False
                break

        if not feasible:
            all_routes.append([])
            all_costs[b] = float("inf")
            continue

        total_cost = 0.0
        for r in finished_routes:
            prev = 0
            for j in r:
                total_cost += float(D[prev, j])
                prev = j
            total_cost += float(D[prev, 0])

        all_routes.append(finished_routes)
        all_costs[b] = total_cost

    return all_routes, all_costs, None


def decode_giant_tour_split_dp(
    edge_logits,      # [B,M,N,N]
    dists,            # [B,N,N] incl depot 0
    demands,          # [B,1,N] or [B,N]
    capacity=1.0,
    pool="mean",          # "mean" | "any"
    pool_mode="logits",   # "logits" | "prob_logit"
    undirected=True,
    forbid_self=True,
    allow_empty_routes=True,
    max_nr_v_eval=11,
    vehicle_cost=0.0,
    demand_bonus=0.05,    # tiny bias in giant-tour construction
    topk_start_seed=5,    # only used to choose ONE smart start
):
    """
    Fast robust decoder:
      1) pool vehicle-conditioned edge logits into one score matrix
      2) build ONE demand-aware giant tour
      3) exact DP split once
      4) if that fails, try reversed order once
      5) if that still fails, use greedy capacity split (may use > M routes)

    Returns:
      all_routes : list length B, each is list of routes [0, ..., 0]
      all_costs  : [B] routing distance only
      stats      : dict
    """
    device = edge_logits.device
    B, _, N, _ = edge_logits.shape
    M = max_nr_v_eval


    # -----------------------------
    # demands -> [B,N]
    # -----------------------------
    dem = demands
    if dem.dim() == 3 and dem.size(1) == 1:
        dem = dem.squeeze(1)
    elif dem.dim() == 3 and dem.size(-1) == 1:
        dem = dem.squeeze(-1)
    dem = dem.to(device)

    # -----------------------------
    # pool edge logits -> L [B,N,N]
    # -----------------------------
    if pool_mode == "logits":
        if pool == "mean":
            L = edge_logits.mean(dim=1)
        elif pool == "any":
            p_v = torch.sigmoid(edge_logits)
            p_pool = 1.0 - torch.prod(1.0 - p_v, dim=1)
            p_pool = p_pool.clamp(1e-6, 1 - 1e-6)
            L = torch.log(p_pool) - torch.log1p(-p_pool)
        else:
            raise ValueError(pool)
    elif pool_mode == "prob_logit":
        p_v = torch.sigmoid(edge_logits)
        if pool == "mean":
            p_pool = p_v.mean(dim=1)
        elif pool == "any":
            p_pool = 1.0 - torch.prod(1.0 - p_v, dim=1)
        else:
            raise ValueError(pool)
        p_pool = p_pool.clamp(1e-6, 1 - 1e-6)
        L = torch.log(p_pool) - torch.log1p(-p_pool)
    else:
        raise ValueError(pool_mode)

    if forbid_self:
        diag = torch.eye(N, device=device, dtype=torch.bool)[None]
        L = L.masked_fill(diag, -1e9)

    if undirected:
        L = 0.5 * (L + L.transpose(-1, -2))

    # -----------------------------
    # decode batch
    # -----------------------------
    all_routes = []
    all_costs = torch.zeros(B, device=device)
    used_routes_count = []
    fallback_used = []

    for b in range(B):
        score = L[b]
        D = dists[b]
        dem_b = dem[b]

        # 1) build one smart giant tour
        # def greedy_order(score, dem_b, N, topk_start_seed, demand_bonus):
        order = greedy_order(score, dem_b, N, topk_start_seed, demand_bonus)

        # 2) exact split once
        routes, travel, k_used = exact_dp_split(order, dem_b, D, M, capacity,
                                                vehicle_cost=vehicle_cost,
                                                allow_empty_routes=allow_empty_routes)

        # 3) reverse-order retry only if needed
        if routes is None:
            order_rev = list(reversed(order))
            routes, travel, k_used = exact_dp_split(order_rev, dem_b, D, M, capacity, vehicle_cost=vehicle_cost,
                                                    allow_empty_routes=allow_empty_routes)

        # 4) final fallback: greedy capacity split (may exceed M)
        if routes is None:
            routes = greedy_capacity_split(order, dem_b, capacity)

            # cheap, targeted repair only when overflow is tiny
            if len(routes) == M + 1:
                routes_repaired = repair_routes_to_M(routes, dem_b, D, M, capacity)
                if len(routes_repaired) <= M:
                    routes = routes_repaired

            travel = sum(route_travel(r, D) for r in routes)
            k_used = len(routes)
            fallback_used.append(True)
        else:
            fallback_used.append(False)

        # pad only if <= M
        if allow_empty_routes and len(routes) < M:
            routes = routes + [[0, 0] for _ in range(M - len(routes))]

        all_routes.append(routes)
        all_costs[b] = travel
        used_routes_count.append(k_used)

    # -----------------------------
    # stats
    # -----------------------------
    covered = []
    feasible = []
    for b in range(B):
        seen = set()
        ok = True
        dem_b = dem[b]

        for r in all_routes[b]:
            load = 0.0
            for node in r:
                if node != 0:
                    if node in seen:
                        ok = False
                    seen.add(node)
                    load += float(dem_b[node].item())
            if load > capacity + 1e-6:
                ok = False

        covered.append(len(seen))
        feasible.append(ok and (len(seen) == (N - 1)))

    stats = {
        "feasible": feasible,
        "covered": covered,
        "N_customers": N - 1,
        "used_routes": used_routes_count,
        "fallback_used": fallback_used,
    }

    return all_routes, all_costs, stats


def decode_vehicle_assignment_1(
        edge_logits,      # [B, M, N, N]
        dists,            # [B, N, N]
        demands,          # [B, N]
        capacity=1.0,
        max_nr_v_eval=11,
        depot_weight=1.0,
        dist_penalty=0.01,
):
    """
    aka decode_vehicle_assignment_plain (from 12030000)
    Capacity-constrained node-to-vehicle assignment decoder.

    1) Assign each customer to one vehicle based on vehicle heatmap affinity.
    2) Reconstruct route within each cluster greedily.

    Guarantees:
    - ≤ M vehicles
    - capacity feasibility
    """

    device = edge_logits.device
    B, M_logits, N, _ = edge_logits.shape
    M = min(M_logits, max_nr_v_eval)

    all_routes = []
    all_costs = torch.zeros(B, device=device)

    if demands.dim() == 3:
        demands = demands.squeeze(1) if demands.size(1) == 1 else demands.squeeze(-1)

    for b in range(B):

        A = edge_logits[b, :M]
        D = dists[b]
        dem = demands[b]

        # ---------------------------
        # STEP 1: vehicle affinity
        # ---------------------------

        # score vehicle preference for each node
        affinity = torch.zeros(M, N, device=device)

        for m in range(M):

            depot_scores = A[m, 0]           # depot -> node
            incoming = A[m].max(dim=0).values  # best predecessor

            affinity[m] = depot_weight * depot_scores + incoming

        # ignore depot
        affinity[:,0] = -1e9

        # ---------------------------
        # STEP 2: capacity assignment
        # ---------------------------

        vehicle_caps = [capacity] * M
        clusters = [[] for _ in range(M)]

        # sort customers by difficulty (large demand first)
        customers = list(range(1, N))
        customers.sort(key=lambda j: float(dem[j]), reverse=True)

        for j in customers:

            # vehicles sorted by affinity
            scores = [(float(affinity[m,j]), m) for m in range(M)]
            scores.sort(reverse=True)

            assigned = False

            for score, m in scores:

                if dem[j] <= vehicle_caps[m] + 1e-9:
                    clusters[m].append(j)
                    vehicle_caps[m] -= float(dem[j])
                    assigned = True
                    break

            if not assigned:
                # fallback: assign to vehicle with most remaining capacity
                # best_m = max(range(M), key=lambda m: vehicle_caps[m])
                best_m = min(
                    range(M),
                    key=lambda m: max(0.0, dem[j] - vehicle_caps[m])
                )
                clusters[best_m].append(j)
                vehicle_caps[best_m] -= float(dem[j])

        if len(all_routes) > b:
            continue

        # ---------------------------
        # STEP 3: route reconstruction
        # ---------------------------

        routes = []

        for m in range(M):

            nodes = clusters[m]

            if not nodes:
                continue

            unvisited = set(nodes)
            cur = 0
            route = []

            while unvisited:

                best_j = None
                best_score = -1e18

                for j in unvisited:

                    score = float(A[m, cur, j]) - dist_penalty * float(D[cur,j])

                    if score > best_score:
                        best_score = score
                        best_j = j

                route.append(best_j)
                unvisited.remove(best_j)
                cur = best_j

            routes.append(route)

        # ---------------------------
        # STEP 4: compute cost
        # ---------------------------

        total_cost = 0.0

        for r in routes:

            prev = 0
            for j in r:
                total_cost += float(D[prev,j])
                prev = j

            total_cost += float(D[prev,0])

        all_routes.append(routes)
        all_costs[b] = total_cost

    return all_routes, all_costs, None


def decode_vehicle_assignment_v2(
        edge_logits,      # [B, M, N, N]
        dists,            # [B, N, N]
        demands,          # [B, N]
        capacity=1.0,
        max_nr_v_eval=11,
        depot_weight=1.0,
        dist_penalty=0.01,
):
    """
    Capacity-repaired node-to-vehicle assignment decoder.

    1) Assign each customer to one vehicle based on vehicle heatmap affinity.
    2) Reconstruct an order within each cluster greedily.
    3) If a cluster exceeds capacity, split it into multiple depot tours.

    Properties:
    - all returned tours are capacity-feasible (assuming each single customer demand <= capacity)
    - may use more than M tours if the initial cluster assignment overloads some vehicles
    """

    device = edge_logits.device
    B, M_logits, N, _ = edge_logits.shape
    M = min(M_logits, max_nr_v_eval)

    all_routes = []
    all_costs = torch.zeros(B, device=device)

    if demands.dim() == 3:
        demands = demands.squeeze(1) if demands.size(1) == 1 else demands.squeeze(-1)

    for b in range(B):

        A = edge_logits[b, :M]
        D = dists[b]
        dem = demands[b]

        # ---------------------------
        # STEP 1: vehicle affinity
        # ---------------------------
        affinity = torch.zeros(M, N, device=device)

        for m in range(M):
            depot_scores = A[m, 0]                 # depot -> node
            incoming = A[m].max(dim=0).values      # best predecessor
            affinity[m] = depot_weight * depot_scores + incoming

        affinity[:, 0] = -1e9  # ignore depot

        # ---------------------------
        # STEP 2: capacity-aware assignment
        # ---------------------------
        vehicle_caps = [capacity] * M
        clusters = [[] for _ in range(M)]

        customers = list(range(1, N))
        customers.sort(key=lambda j: float(dem[j]), reverse=True)

        for j in customers:
            scores = [(float(affinity[m, j]), m) for m in range(M)]
            scores.sort(reverse=True)

            assigned = False

            # first try feasible vehicle
            for score, m in scores:
                if dem[j] <= vehicle_caps[m] + 1e-9:
                    clusters[m].append(j)
                    vehicle_caps[m] -= float(dem[j])
                    assigned = True
                    break

            if not assigned:
                # fallback: still assign by best affinity / smallest shortfall
                best_m = min(
                    range(M),
                    key=lambda m: max(0.0, float(dem[j]) - vehicle_caps[m])
                )
                clusters[best_m].append(j)
                vehicle_caps[best_m] -= float(dem[j])

        # ---------------------------
        # STEP 3: route reconstruction + capacity repair
        # ---------------------------
        routes = []

        for m in range(M):
            nodes = clusters[m]
            if not nodes:
                continue

            # 3a) build one greedy ordering for the cluster
            unvisited = set(nodes)
            cur = 0
            ordered_nodes = []

            while unvisited:
                best_j = None
                best_score = -1e18

                for j in unvisited:
                    score = float(A[m, cur, j]) - dist_penalty * float(D[cur, j])
                    if score > best_score:
                        best_score = score
                        best_j = j

                ordered_nodes.append(best_j)
                unvisited.remove(best_j)
                cur = best_j

            # 3b) split ordered sequence into capacity-feasible tours
            cur_route = []
            cur_load = 0.0

            for j in ordered_nodes:
                dj = float(dem[j])

                # sanity: single node infeasible even alone
                if dj > capacity + 1e-9:
                    # still place it as singleton to avoid crashing;
                    # this should not happen on valid FC-CVRP instances
                    if cur_route:
                        routes.append(cur_route)
                        cur_route = []
                        cur_load = 0.0
                    routes.append([j])
                    continue

                if cur_load + dj <= capacity + 1e-9:
                    cur_route.append(j)
                    cur_load += dj
                else:
                    if cur_route:
                        routes.append(cur_route)
                    cur_route = [j]
                    cur_load = dj

            if cur_route:
                routes.append(cur_route)

        # ---------------------------
        # STEP 4: compute cost
        # ---------------------------
        total_cost = 0.0

        for r in routes:
            prev = 0
            for j in r:
                total_cost += float(D[prev, j])
                prev = j
            total_cost += float(D[prev, 0])

        all_routes.append(routes)
        all_costs[b] = total_cost

    return all_routes, all_costs, None