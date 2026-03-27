import torch


import numpy as np


import torch


def rl_decode_wrapper(
    *,
    use_eliteK: bool,
    L, dists_rl, dem_rl, coords_rl,
    cap: float,
    vehicle_cost: float,
    tau: float,
    K: int,
    greedy_giant_tour_order_fn,
    sample_giant_tour_order_fast_fn,
    greedy_split_from_order_fn,
):
    """
    Returns:
      decode_rl_loss, obj_sample_or_best, obj_base, impr
    All objs/impr are tensors shaped (B,) and detached (no grad).
    """
    if not use_eliteK:
        decode_rl_loss, obj_sample, obj_base = rl_loss_improvement_over_greedy(
            L=L,
            dists_rl=dists_rl,
            dem_rl=dem_rl,
            coords_rl=coords_rl,
            cap=cap,
            vehicle_cost=vehicle_cost,
            tau=tau,
            greedy_giant_tour_order_fn=greedy_giant_tour_order_fn,
            sample_giant_tour_order_fast_fn=sample_giant_tour_order_fast_fn,
            greedy_split_from_order_fn=greedy_split_from_order_fn,
        )
        if obj_sample is None or obj_base is None:
            return decode_rl_loss, obj_sample, obj_base, None
        impr = (obj_base - obj_sample).detach()
        return decode_rl_loss, obj_sample.detach(), obj_base.detach(), impr

    # --- eliteK ---
    decode_rl_loss, obj_best = rl_loss_eliteK_best_of_K(
        L, dists_rl, dem_rl, coords_rl,
        cap=cap, vehicle_cost=vehicle_cost,
        tau=tau, K=K,
        sample_giant_tour_order_fast_fn=sample_giant_tour_order_fast_fn,
        greedy_split_from_order_fn=greedy_split_from_order_fn,
    )
    if obj_best is None:
        return decode_rl_loss, None, None, None

    # baseline: greedy giant tour + greedy split (same eval pipeline)
    with torch.no_grad():
        orders_g = greedy_giant_tour_order_fn(L)  # whatever shape your greedy returns
        obj_base, _k_used, _feas = split_orders_to_obj(
            orders_g, dists_rl, dem_rl, coords_rl,
            cap=cap, vehicle_cost=vehicle_cost,
            greedy_split_from_order_fn=greedy_split_from_order_fn,
        )
        obj_base = obj_base.detach()
        obj_best = obj_best.detach()
        impr = (obj_base - obj_best)

    return decode_rl_loss, obj_best, obj_base, impr




@torch.no_grad()
def routes_feasible(routes, dem, cap=1.0):
    if routes is None:
        return False
    N = int(dem.numel())
    seen = []
    for r in routes:
        if len(r) < 2 or r[0] != 0 or r[-1] != 0:
            return False
        load = 0.0
        for v in r[1:-1]:
            if v <= 0 or v >= N:
                return False
            seen.append(v)
            load += float(dem[v].item())
        if load > cap + 1e-9:
            return False
    return len(seen) == (N - 1) and len(set(seen)) == (N - 1)


def split_orders_to_obj(
    orders, dists_rl, dem_rl, coords_rl,
    cap=1.0, vehicle_cost=0.0,
    greedy_split_from_order_fn=None,
):
    """
    orders: list length B_rl, each is list length N-1 (customers only)
    dists_rl: (B_rl, N, N)
    dem_rl:   (B_rl, N)
    coords_rl:(B_rl, N, 2)

    Returns:
      obj:    (B_rl,) float tensor (inf if infeasible)
      k_used: (B_rl,) float tensor (num routes used)
      feas:   (B_rl,) float tensor {0,1}
    """
    device = dists_rl.device
    B_rl = len(orders)

    obj_list, k_list, feas_list = [], [], []
    for b in range(B_rl):
        obj_b, routes_b, _ = greedy_split_from_order_fn(
            orders[b],
            dists_rl[b],
            dem_rl[b],
            cap=cap,
            vehicle_cost=vehicle_cost,
            coords=coords_rl[b],
        )

        # ensure tensor scalar -> python float -> tensor
        if torch.is_tensor(obj_b):
            obj_b = float(obj_b.detach().item())
        obj_list.append(obj_b)

        if routes_b is None:
            k_list.append(float("nan"))
            feas_list.append(0.0)
        else:
            k_list.append(float(len(routes_b)))
            feas_list.append(1.0 if routes_feasible(routes_b, dem_rl[b], cap=cap) else 0.0)

    obj = torch.tensor(obj_list, device=device, dtype=torch.float32)
    k_used = torch.tensor(k_list, device=device, dtype=torch.float32)
    feas = torch.tensor(feas_list, device=device, dtype=torch.float32)

    return obj, k_used, feas

    # return (obj, k_used) if return_k_used else obj


def rl_loss_improvement_over_greedy(
    L,                 # (B,N,N) pooled logits
    dists_rl,          # (B,N,N)
    dem_rl,            # (B,N)
    coords_rl,         # (B,N,2)
    cap: float,
    vehicle_cost: float,
    tau: float,
    greedy_giant_tour_order_fn,
    sample_giant_tour_order_fast_fn,
    greedy_split_from_order_fn,
):
    """
    One sampled giant tour per instance + baseline = greedy tour per instance.
    Grad flows ONLY through logp from the sampler.
    Returns:
      decode_rl_loss: scalar tensor (requires grad)
      obj_sample: (B,) tensor (no grad)
      obj_base: (B,) tensor (no grad)
    """
    # stochastic (training) orders + logp
    orders_s, logp = sample_giant_tour_order_fast_fn(L, tau=tau)   # logp: (B,)

    # hard guard
    if (not torch.isfinite(logp).all()):
        return None, None, None

    with torch.no_grad():
        # obj from sampled orders
        obj_sample, _, _ = split_orders_to_obj(
            orders_s, dists_rl, dem_rl, coords_rl, cap, vehicle_cost, greedy_split_from_order_fn
        )
        # baseline: greedy orders
        orders_g = greedy_giant_tour_order_fn(L)
        obj_base, _, _ = split_orders_to_obj(
            orders_g, dists_rl, dem_rl, coords_rl, cap, vehicle_cost, greedy_split_from_order_fn
        )

        finite = torch.isfinite(obj_sample) & torch.isfinite(obj_base)

    if finite.sum().item() < 2:
        return None, obj_sample, obj_base

    # reward = relative improvement over greedy (scale-free)
    R = ((obj_base - obj_sample) / (obj_base + 1e-6))[finite]  # higher is better
    lp = logp[finite]

    adv = R - R.mean()
    adv = adv / (adv.std(unbiased=False) + 1e-6)

    decode_rl_loss = -(adv.detach() * lp).mean()
    return decode_rl_loss, obj_sample, obj_base


def pool_logits_to_L(edge_logits: torch.Tensor,
                     pool: str = "any",
                     undirected: bool = True,
                     forbid_self: bool = True,
                     eps: float = 1e-6) -> torch.Tensor:
    """
    edge_logits: [B,M,N,N] (raw logits)
    returns L:  [B,N,N] pooled logits (logit of pooled probs)
    """
    B, M, N, _ = edge_logits.shape
    p_v = torch.sigmoid(edge_logits)  # [B,M,N,N]

    if pool == "any":
        p_pool = 1.0 - torch.prod(1.0 - p_v, dim=1)  # noisy-OR
    elif pool == "mean":
        p_pool = p_v.mean(dim=1)
    else:
        raise ValueError(pool)

    if undirected:
        p_pool = 0.5 * (p_pool + p_pool.transpose(-1, -2))

    if forbid_self:
        diag = torch.eye(N, device=edge_logits.device, dtype=torch.bool)[None]
        p_pool = p_pool.masked_fill(diag, 0.0)

    p_pool = p_pool.clamp(eps, 1 - eps)
    L = torch.log(p_pool) - torch.log1p(-p_pool)  # logit
    return L


def sample_giant_tour_order_fast(L, tau=1.0):
    # L: [B,N,N]
    B, N, _ = L.shape
    device = L.device

    unv = torch.ones(B, N, dtype=torch.bool, device=device)
    unv[:, 0] = False
    cur = torch.zeros(B, dtype=torch.long, device=device)

    order = torch.empty(B, N-1, dtype=torch.long, device=device)
    logp = torch.zeros(B, device=device)

    for t in range(N - 1):
        logits = L[torch.arange(B, device=device), cur]  # [B,N]
        logits = logits / max(tau, 1e-6)

        # mask visited + depot
        logits = logits.masked_fill(~unv, -1e9)
        logits[:, 0] = -1e9

        # stability (prevents NaNs)
        logits = logits - logits.max(dim=-1, keepdim=True).values
        probs = torch.softmax(logits, dim=-1)

        # safety clamp
        probs = probs.clamp_min(1e-12)
        # row_sum = probs.sum(dim=-1, keepdim=True)  # [B,1]
        # bad = row_sum.squeeze(1) <= 0
        #
        # if bad.any():
        #     # pick any remaining unvisited (first one) for those batches
        #     # (this keeps it finite and avoids NaNs)
        #     idx_bad = torch.where(bad)[0]
        #     for bb in idx_bad.tolist():
        #         remaining = torch.nonzero(unv[bb], as_tuple=False).squeeze(1)
        #         # depot is already forbidden; remaining should be non-empty
        #         probs[bb].zero_()
        #         probs[bb, remaining[0]] = 1.0
        #
        # probs = probs / probs.sum(dim=-1, keepdim=True)
        # nxt = torch.multinomial(probs, 1).squeeze(1)
        probs = probs / probs.sum(dim=-1, keepdim=True)

        nxt = torch.multinomial(probs, 1).squeeze(1)  # [B]
        logp = logp + torch.log(probs.gather(1, nxt[:, None]).squeeze(1))

        order[:, t] = nxt
        unv[torch.arange(B, device=device), nxt] = False
        cur = nxt

    return [order[b].tolist() for b in range(B)], logp


@torch.no_grad()
def sample_giant_tour_order(L, tau=1.0, eps=1e-12):
    """
    L: [B,N,N] pooled symmetric logits/scores (higher better), depot is 0.
    Returns:
      orders: list length B, each list length N-1 (customers only)
      logp:   [B] log prob of sampled order under sampling policy
    """
    B, N, _ = L.shape
    device = L.device

    # Safety: if L contains nan/inf, fix early to avoid poison
    L = torch.nan_to_num(L, neginf=-1e9, posinf=1e9)

    orders = []
    logp = torch.zeros(B, device=device)

    for b in range(B):
        score = L[b]  # [N,N]
        unv = torch.ones(N, device=device, dtype=torch.bool)
        unv[0] = False  # depot never selected

        cur = 0
        order = []

        for t in range(N - 1):
            logits = score[cur].clone()  # [N]

            # mask visited + mask depot
            logits[~unv] = -1e9
            logits[0] = -1e9

            # If EVERYTHING is masked, fall back to "pick any remaining unvisited"
            if not torch.isfinite(logits).any() or (logits.max() <= -1e8):
                remaining = torch.nonzero(unv, as_tuple=False).flatten()
                if remaining.numel() == 0:
                    break  # should not happen, but safe
                nxt = remaining[0].item()
                # treat as uniform fallback (no logp update, or add log(1/|rem|) if you want)
                order.append(nxt)
                unv[nxt] = False
                cur = nxt
                continue

            # Stable softmax: subtract max before exp
            z = logits / max(float(tau), 1e-6)
            z = z - z.max()
            p = torch.softmax(z, dim=-1)

            # clamp + renormalize to avoid exact zeros / NaNs
            p = torch.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
            p = p.clamp_min(0.0)
            s = p.sum()

            if s <= eps:
                remaining = torch.nonzero(unv, as_tuple=False).flatten()
                nxt = remaining[0].item()
                order.append(nxt)
                unv[nxt] = False
                cur = nxt
                continue

            p = p / s

            nxt = torch.multinomial(p, 1).item()
            logp[b] += torch.log(p[nxt].clamp_min(eps))

            order.append(nxt)
            unv[nxt] = False
            cur = nxt

        orders.append(order)

    return orders, logp


def rl_loss_eliteK_best_of_K(
    L, dists_rl, dem_rl, coords_rl,
    cap: float,
    vehicle_cost: float,
    tau: float,
    K: int,
    sample_giant_tour_order_fast_fn,
    greedy_split_from_order_fn,
):
    """
    Returns:
      decode_rl_loss (scalar) or None
      obj_best: (B,) no grad
    """
    B = L.size(0)
    logp_K = []
    orders_K = []

    # K stochastic orders
    for _ in range(K):
        orders_k, logp_k = sample_giant_tour_order_fast_fn(L, tau=tau)  # logp_k: (B,)
        orders_K.append(orders_k)                                       # list of len B
        logp_K.append(logp_k)

    logp = torch.stack(logp_K, dim=1)  # (B,K)
    if not torch.isfinite(logp).all():
        return None, None

    with torch.no_grad():
        obj = torch.empty((B, K), device=L.device, dtype=torch.float32)
        for k in range(K):
            # out: obj, k_used, feas
            obj[:, k], _, _ = split_orders_to_obj(
                orders_K[k], dists_rl, dem_rl, coords_rl, cap, vehicle_cost, greedy_split_from_order_fn
            )

        finite = torch.isfinite(obj).all(dim=1)  # require all K finite per instance for simplicity
        if finite.sum().item() < 2:
            return None, obj.min(dim=1).values

        obj_f = obj[finite]       # (Bf,K)
        logp_f = logp[finite]     # (Bf,K)

        best_k = obj_f.argmin(dim=1)  # (Bf,)
        obj_best = obj_f[torch.arange(obj_f.size(0), device=L.device), best_k]   # (Bf,)
        lp_best  = logp_f[torch.arange(logp_f.size(0), device=L.device), best_k] # (Bf,)

        R = -obj_best
        adv = (R - R.mean()) / (R.std(unbiased=False) + 1e-6)

    decode_rl_loss = -(adv.detach() * lp_best).mean()
    return decode_rl_loss, obj.min(dim=1).values  # obj_best per original B (roughly)

def succ_labels_from_sparse(targets_sparse, device=None):
    """
    targets_sparse: sparse COO [B,M,N,N] with exactly one successor per row (except maybe some rows)
    returns targ_succ: Long [B,M,N] with successor index, or -1 if missing
    """
    B, M, N, _ = targets_sparse.shape
    if device is None:
        device = targets_sparse.device

    idx = targets_sparse.coalesce().indices()  # [4, nnz] = (b,m,i,j)
    b, m, i, j = idx[0], idx[1], idx[2], idx[3]

    targ_succ = torch.full((B, M, N), -1, device=device, dtype=torch.long)
    targ_succ[b, m, i] = j
    return targ_succ

# def succ_topk_acc(edge_logits, targ_succ, k=5):
#     # ignore depot row if you want: i>=1
#     B,M,N,_ = edge_logits.shape
#     idx = torch.topk(edge_logits, k=k, dim=-1).indices  # [B,M,N,k]
#     hit = (idx == targ_succ[..., None]).any(dim=-1).float()  # [B,M,N]
#     # optionally mask i where targ_succ is invalid (e.g. 0 or self)
#     return hit.mean().item()

def succ_topk_acc(logits, targ_succ, k=5, ignore_index=-1):
    """
    logits:    [B,M,N,N]  (or [B,N,N] if you do union)
    targ_succ: [B,M,N]    (or [B,N]) dense Long, -1 ignored
    """
    # top-k predicted successors
    idx = logits.topk(k, dim=-1).indices  # [...,N,k]
    hit = (idx == targ_succ[..., None]).any(dim=-1)  # [...,N]

    valid = (targ_succ != ignore_index)
    if valid.any():
        return (hit & valid).float().sum().item() / valid.float().sum().item()
    return float("nan")

def sample_giant_tour_order_old(L: torch.Tensor, tau: float = 1.0):
    """
    L: [B,N,N] pooled logits/scores (higher better), depot is 0.
    Returns:
      orders: list length B, each list length N-1 (customers only)
      logp:   [B] log prob of sampled order under sampling policy (requires grad)
    """
    B, N, _ = L.shape
    device = L.device
    orders = []
    logp = torch.zeros(B, device=device, dtype=L.dtype)

    for b in range(B):
        score = L[b]  # [N,N]
        unv = torch.ones(N, device=device, dtype=torch.bool)
        unv[0] = False

        cur = 0
        order = []
        for _ in range(N - 1):
            logits = score[cur].clone()
            logits[~unv] = -1e9
            logits[0] = -1e9  # never pick depot

            probs = torch.softmax(logits / tau, dim=-1)
            nxt = torch.multinomial(probs, 1).item()

            logp[b] = logp[b] + torch.log(probs[nxt].clamp_min(1e-12))
            order.append(nxt)
            unv[nxt] = False
            cur = nxt

        orders.append(order)

    return orders, logp

def routes_feasible(routes, dem, cap=1.0, n_nodes=None):
    # routes: list of [0,...,0]
    # dem: Tensor[N]
    seen = []
    for r in routes:
        if len(r) < 2 or r[0] != 0 or r[-1] != 0:
            return False
        load = 0.0
        for v in r[1:-1]:
            if v == 0:
                return False
            seen.append(v)
            load += float(dem[v].item())
        if load > cap + 1e-9:
            return False

    if n_nodes is None:
        n_nodes = int(dem.numel())
    customers = list(range(1, n_nodes))
    return sorted(seen) == customers

def greedy_needed_k(order, dem_row, cap=1.0):
    k = 1
    load = 0.0
    for node in order:            # nodes are 1..N customers (no depot)
        d = float(dem_row[node].item())
        if d > cap + 1e-9:
            return 10**9
        if load + d <= cap + 1e-9:
            load += d
        else:
            k += 1
            load = d
    return k

@torch.no_grad()
def decode_split_dp_from_order(orders,
                               dists: torch.Tensor,      # [B,N,N]
                               demands: torch.Tensor,    # [B,N]
                               M: int,
                               capacity: float = 1.0,
                               vehicle_cost: float = 0.0,
                               allow_empty_routes: bool = True):
    """
    Returns:
      routes: list length B, each is list of routes (padded to M if allow_empty_routes)
      travel_cost: [B] (pure travel best_travel)
      stats: dict
      k_used: [B] chosen number of non-empty routes
      obj: [B] = best_travel + vehicle_cost * k_used
    """
    device = dists.device
    B, N, _ = dists.shape

    dem = demands
    if dem.dim() == 3 and dem.size(1) == 1:
        dem = dem.squeeze(1)
    elif dem.dim() == 3 and dem.size(-1) == 1:
        dem = dem.squeeze(-1)
    dem = dem.to(device)

    all_routes = []
    travel_cost = torch.full((B,), float("inf"), device=device)
    obj_cost = torch.full((B,), float("inf"), device=device)
    k_used = torch.zeros((B,), dtype=torch.long, device=device)

    for b in range(B):
        T = orders[b]               # list length N-1 (customers)
        nC = len(T)
        D = dists[b]                # [N,N]

        # prefix demand
        Tt = torch.tensor(T, device=device, dtype=torch.long)
        q = dem[b, Tt]                              # [nC]
        pref = torch.zeros(nC + 1, device=device)
        pref[1:] = torch.cumsum(q, dim=0)

        # helper array with 1-indexed customers: Ti[1]=T1
        Ti = torch.cat([torch.tensor([0], device=device), Tt], dim=0)  # [nC+1]

        # path prefix for internal edges along order
        path_pref = torch.zeros(nC + 1, device=device)
        for t in range(2, nC + 1):
            path_pref[t] = path_pref[t - 1] + D[Ti[t - 1], Ti[t]]

        seg_cost = torch.full((nC + 1, nC + 1), float("inf"), device=device)
        seg_ok = torch.zeros((nC + 1, nC + 1), dtype=torch.bool, device=device)

        for i in range(1, nC + 1):
            for j in range(i, nC + 1):
                load_ij = pref[j] - pref[i - 1]
                if load_ij <= capacity + 1e-9:
                    internal = path_pref[j] - path_pref[i]
                    c = D[0, Ti[i]] + internal + D[Ti[j], 0]
                    seg_cost[i, j] = c
                    seg_ok[i, j] = True
        # print('M', M)
        Kmax = min(M, nC)
        dp = torch.full((Kmax + 1, nC + 1), float("inf"), device=device)
        prev = torch.full((Kmax + 1, nC + 1), -1, dtype=torch.long, device=device)

        # pick k by travel + vehicle_cost*k
        best_k = -1
        best_travel = float("inf")
        best_obj = float("inf")

        dp[0, 0] = 0.0
        for k in range(1, Kmax + 1):
            for j in range(1, nC + 1):
                best = float("inf")
                best_i = -1
                load = 0.0
                for i in range(j, 0, -1):
                    load += float(q[i - 1].item())  # q is demand along order, length nC
                    if load > capacity + 1e-9:
                        break
                    cand = dp[k - 1, i - 1] + seg_cost[i, j]
                    if cand < best:
                        best = cand
                        best_i = i
                # for i in range(1, j + 1):
                #     if not seg_ok[i, j]:
                #         continue
                #     cand = dp[k - 1, i - 1] + seg_cost[i, j]
                #     if cand < best:
                #         best = cand
                #         best_i = i
                dp[k, j] = best
                prev[k, j] = best_i

        # compute the best value and best k on the fly while filling DP, and skip the final scan.
        tr = dp[k, nC].item()
        if tr < best_travel:
            best_travel = tr
            best_k = k

        if best_k < 0:
            needed_k = greedy_needed_k(T, dem[b], cap=capacity)
            print("DP FAIL",
                  "sum_dem", dem[b, 1:].sum().item(),
                  "max_dem", dem[b, 1:].max().item(),
                  "needed_k", needed_k,
                  "M", M,
                  "cap", capacity,
                  "seg_ok_count", int(seg_ok.sum().item()))

        # for k in range(1, Kmax + 1):
        #     tr = dp[k, nC].item()
        #     if tr == float("inf"):
        #         continue
        #     ob = tr + float(vehicle_cost) * k
        #     if ob < best_obj:
        #         best_obj = ob
        #         best_travel = tr
        #         best_k = k

        # backtrack
        routes = []
        k = best_k
        j = nC
        while k > 0 and j > 0:
            i = int(prev[k, j].item())
            if i < 1:
                break
            seg_customers = T[i - 1: j]
            routes.append([0] + seg_customers + [0])
            j = i - 1
            k -= 1
        routes.reverse()

        if allow_empty_routes and len(routes) < M:
            routes = routes + [[0, 0] for _ in range(M - len(routes))]

        all_routes.append(routes)
        travel_cost[b] = best_travel
        k_used[b] = best_k
        obj_cost[b] = best_obj

    # simple stats
    covered = []
    feasible = []
    for b in range(B):
        seen = set()
        ok = True
        for r in all_routes[b]:
            load = 0.0
            for node in r:
                if node != 0:
                    if node in seen:
                        ok = False
                    seen.add(node)
                    load += float(dem[b, node].item())
            if load > capacity + 1e-6:
                ok = False
        covered.append(len(seen))
        feasible.append(ok and (len(seen) == (N - 1)))

    stats = {"feasible": feasible, "covered": covered, "N_customers": N - 1}
    # print('k_used', k_used)
    return all_routes, travel_cost, stats, k_used, obj_cost


def hungarian_match_from_membership(pred_mem, tgt_mem, thr=0.2):
    """
    pred_mem, tgt_mem: [B,M,K] where K=N-1
    returns:
      perm_tgt_for_pred: [B,M] with entries in {0..M-1}
        meaning: pred vehicle v should be compared to target vehicle perm[v]
    """
    from scipy.optimize import linear_sum_assignment

    B, M, K = pred_mem.shape
    pred_bin = (pred_mem > thr).to(torch.bool)
    tgt_bin  = (tgt_mem  > 0.5).to(torch.bool)

    perms = []
    for b in range(B):
        # cost[v,u] = 1 - IoU(pred v, tgt u)
        P = pred_bin[b].cpu().numpy()  # [M,K]
        T = tgt_bin[b].cpu().numpy()   # [M,K]

        inter = (P[:, None, :] & T[None, :, :]).sum(axis=-1).astype(np.float32)  # [M,M]
        uni   = (P[:, None, :] | T[None, :, :]).sum(axis=-1).astype(np.float32)  # [M,M]

        # IoU edge-case: both empty -> IoU=1
        iou = np.where(uni == 0.0, 1.0, inter / (uni + 1e-9))
        cost = 1.0 - iou

        row_ind, col_ind = linear_sum_assignment(cost)  # rows are pred v, cols are tgt u
        perm = np.zeros(M, dtype=np.int64)
        perm[row_ind] = col_ind
        perms.append(torch.from_numpy(perm))
    return torch.stack(perms, dim=0)  # [B,M]


def permute_targets_sparse_by_perm(targets_sparse, perm, M):
    """
    targets_sparse: sparse COO tensor with indices [4, nnz] (b, m, i, j)
    perm: [B, M] mapping pred_m -> tgt_m  (can be cpu or cuda)
    Returns: sparse COO tensor on SAME device as targets_sparse.
    """
    ts = targets_sparse.coalesce()
    idx = ts.indices()   # [4, nnz] on same device as ts
    val = ts.values()
    device = val.device

    # Work on CPU (safe for Hungarian / indexing logic)
    idx_cpu = idx.cpu()
    b    = idx_cpu[0]
    m_old= idx_cpu[1]
    i    = idx_cpu[2]
    j    = idx_cpu[3]

    perm_cpu = perm.cpu()  # [B, M]

    # inverse permutation: inv[b, tgt_m] = pred_m
    inv = torch.empty_like(perm_cpu)
    inv.scatter_(1, perm_cpu, torch.arange(M, device=perm_cpu.device).expand_as(perm_cpu))

    m_new = inv[b, m_old]  # CPU

    new_idx_cpu = torch.stack([b, m_new, i, j], dim=0)  # ALL CPU now ✅

    # Move indices back to original device to build sparse tensor there
    new_idx = new_idx_cpu.to(device)
    new_val = val  # already on device

    return torch.sparse_coo_tensor(new_idx, new_val, ts.size(), device=device).coalesce()


def permute_targets_sparse_by_perm_old(targets_sparse, perm_tgt_for_pred, M):
    """
    targets_sparse indices are (b,m,i,j) with m being target-vehicle id.
    perm_tgt_for_pred[b,v] = u tells which original target vehicle u corresponds to pred vehicle v.

    We need inverse mapping inv[b,u] = v so we can rewrite each nonzero (b,u,i,j) -> (b,v,i,j).
    """
    idx = targets_sparse.coalesce().indices()  # [4, nnz]
    vals = targets_sparse.coalesce().values()

    b = idx[0]
    m_old = idx[1]

    # build inverse mapping per batch: inv[b, u] = v
    B = perm_tgt_for_pred.size(0)
    inv = torch.empty((B, M), device=perm_tgt_for_pred.device, dtype=torch.long)
    ar = torch.arange(M, device=perm_tgt_for_pred.device)
    inv.scatter_(1, perm_tgt_for_pred, ar.unsqueeze(0).expand(B, M))

    m_new = inv[b, m_old]
    idx_new = torch.stack([idx[0], m_new, idx[2], idx[3]], dim=0)

    return torch.sparse_coo_tensor(idx_new, vals, size=targets_sparse.size(), device=targets_sparse.device).coalesce()

def pred_membership_from_logits(edge_logits, customers_only=True):
    # edge_logits: [B,M,N,N]
    p = torch.sigmoid(edge_logits)  # [B,M,N,N]
    # incoming/outgoing presence (cheap reductions)
    inc = p.amax(dim=2)  # [B,M,N]  (max over i -> anything goes into j)
    out = p.amax(dim=3)  # [B,M,N]  (max over j -> anything leaves i)
    mem = torch.maximum(inc, out)
    if customers_only:
        mem = mem[..., 1:]  # drop depot
    return mem  # [B,M,N-1]


def tgt_membership_from_sparse(targets_sparse, B, M, N, device):
    """
    targets_sparse: sparse COO with indices [4, nnz] = (b, m, i, j)
    returns mem: [B,M,N-1] boolean-ish {0,1}
    """
    idx = targets_sparse.coalesce().indices()
    b = idx[0]; m = idx[1]; i = idx[2]; j = idx[3]

    mem = torch.zeros((B, M, N), device=device, dtype=torch.float32)

    # mark endpoints of edges (i and j); customers only later
    mem[b, m, i] = 1.0
    mem[b, m, j] = 1.0

    # optionally ignore depot presence
    mem[:, :, 0] = 0.0
    return mem[:, :, 1:]  # [B,M,N-1]



# Rank: early, small, then ramp
def sched_lambda_rank(e):
    if e < 3:
        return 0.1
    # elif e < 10:
    #     return 0.25 * (e - 3) / 7
    # elif e < 25:
    #     return 0.25 + 0.1 * (e - 10) / 15
    else:
        return 0.4  # not 0.5 for now

def sched_lambda_match(e):
    if e < 5:
        return 0.0
    elif e < 25:
        return 0.3 * (e - 5) / (25 - 5)   # 0 -> 0.3
    else:
        return 0.3

# OLDD
# def sched_lambda_rank(e):
#     if e < 10:
#         return 0.0
#     elif e < 40:
#         return 0.5 * (e - 10) / (40 - 10) # 0 -> 0.5
#     else:
#         return 0.5



# def sched_lambda_match(epoch: int) -> float:
#     if epoch < 5:
#         return 0.0
#     elif epoch < 20:
#         return 0.15 * (epoch - 5) / (20 - 5)
#     elif epoch < 60:
#         return 0.15 + 0.15 * (epoch - 20) / (60 - 20)
#     else:
#         return 0.30
#
# def sched_lambda_rank(epoch: int) -> float:
#     if epoch < 20:
#         return 0.0
#     elif epoch < 60:
#         return 0.5 * (epoch - 20) / (60 - 20)
#     else:
#         return 0.5