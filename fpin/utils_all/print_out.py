
import torch

def targ_as_lst(idcs_trg,n, print_indcs=False):
    '''idcs_trg is a tensor containing the indices of next visited nodes'''
    if print_indcs:
        print('idcs_trg',idcs_trg)
    nxt_idx=[0]
    nxt=idcs_trg[0].item()
    for _ in range(n):
        cur_idx=nxt
        if cur_idx !=0:
            nxt_idx.append(cur_idx)
            nxt=idcs_trg[cur_idx].item()
    nxt_idx.append(0)
    # print('nxt_idx', nxt_idx)
    return nxt_idx

def successor_to_path_(next_vec, n):
    # next_vec: tensor [N] where next_vec[cur]=nxt
    cur = 0
    out = [0]
    for _ in range(n+1):
        nxt = int(next_vec[cur].item())
        out.append(nxt)
        if nxt == 0:
            break
        cur = nxt
    if out[-1] != 0:
        out.append(0)
    return out

def successor_to_path(next_vec, n):
    """
    next_vec: tensor [N] where next_vec[cur] = nxt
    Returns a path starting at depot 0 following successors until depot or cycle.
    """
    cur = 0
    out = [0]
    seen = {0}
    for _ in range(n + 1):
        nxt = int(next_vec[cur].item())
        out.append(nxt)
        if nxt == 0:
            break
        if nxt in seen:  # cycle guard
            out.append(0)
            break
        seen.add(nxt)
        cur = nxt
    if out[-1] != 0:
        out.append(0)
    return out

def print_func(VRP_probs, predicted_tours, VRP_loads, targets, target_loads, with_loads, fleet_size=4):
    # n = VRP_probs.size(2)
    B, M, n, _ = VRP_probs.size()

    if predicted_tours is None:
        print("\nGREEDY PATH")
        print("(no RL giant-tour+split routes available for this diag batch)")
        return

    # plain succesor prediction Tour
    # succ_predictions = VRP_probs.argmax(dim=-1)  # [B,M,N] --> plain successor map
    L_dir = pool_probs_to_L_dir(VRP_probs, pool="mean")  # [B,N,N]
    targ_succ_u = union_succ_from_targets_sparse(
        targets, B=B, M=M, N=n, device=VRP_probs.device
    )
    # A) row-wise successor sanity (recommended)
    succ_pred_u = L_dir.argmax(dim=-1)  # [B,N]
    print("ROW i : pred_succ | targ_succ")
    for i in range(0, min(25, n)):
        print(f"{i:3d}: {int(succ_pred_u[0, i])} | {int(targ_succ_u[0, i])}")
    mask = (targ_succ_u[0] >= 0) & (targ_succ_u[0] != 0) & (torch.arange(n, device=VRP_probs.device) != 0)
    acc1 = (succ_pred_u[0, mask] == targ_succ_u[0, mask]).float().mean().item()
    print("succ_acc1_non_depot_succ =", acc1)
    print(f"PRED UNION GREEDY PATH: {greedy_path_from_logits(L_dir[0])}")

    # Determine M
    if torch.is_tensor(predicted_tours):
        m = predicted_tours.size(1)
    else:
        m = len(predicted_tours[0])

    fleet_size = min(fleet_size, m)
    print(f"[diag] VRP_probs M={M} | predicted_tours m={m} | requested fleet_size={fleet_size}")

    if with_loads and VRP_loads is not None:
        print("\nLOADS")
        print("predic demand_loads[0]", VRP_loads[0].detach().cpu())
        print("target demand_loads[0]", target_loads[0].detach().cpu())

    print("\nGREEDY PATH")

    for i in range(fleet_size):
        if torch.is_tensor(predicted_tours):
            pred_path = tour_from_next_of(predicted_tours[0, i])
        else:
            pred_path = predicted_tours[0][i]


        targ_succ = targets[0, i].to_dense().max(1)[1]
        # print('row_sum = targets[0,m-1].to_dense().sum(-1)', targets[0,m-1].to_dense().sum(-1))
        targ_path = targ_as_lst(targ_succ, n)

        print(f"CURR PRED CONSECUTIVE PATH  v_{i}: {pred_path}")
        print(f"TARGET CONSECUTIVE PATH for v_{i}: {targ_path}")

def pool_probs_to_L_dir(edge_probs, pool="mean", eps=1e-6):
    """
    edge_probs: [B,M,N,N] in [0,1]
    returns L_dir: [B,N,N] directed pooled logits (no symmetry)
    """
    B, M, N, _ = edge_probs.shape
    device = edge_probs.device

    if pool == "any":
        p_pool = 1.0 - torch.prod(1.0 - edge_probs, dim=1)   # [B,N,N]
    elif pool == "mean":
        p_pool = edge_probs.mean(dim=1)
    else:
        raise ValueError(pool)

    # forbid self
    diag = torch.eye(N, device=device, dtype=torch.bool)[None]
    p_pool = p_pool.masked_fill(diag, 0.0)

    # logit
    p_c = p_pool.clamp(eps, 1 - eps)
    L_dir = torch.log(p_c) - torch.log1p(-p_c)
    return L_dir

# def print_func(VRP_probs, predicted_tours, VRP_loads, targets, target_loads, with_loads, fleet_size=4):
#     """
#     VRP_probs:     [B,M,N,N] (only used for N)
#     predicted_tours:
#         - either list: [B_show][M][T] tour tensors (old "seeds" style)
#         - or tensor:  [B,M,N] successor map next_of (new decode output)
#     VRP_loads:     [B,M] loads per vehicle from decoder (optional)
#     targets:       sparse COO [B,M,N,N] ground-truth adjacency
#     target_loads:  [B,M] target loads
#     """
#     n = VRP_probs.size(2)
#
#     # Determine M
#     if torch.is_tensor(predicted_tours):
#         # next_of tensor [B,M,N]
#         m = predicted_tours.size(1)
#     else:
#         # list-of-list
#         m = len(predicted_tours[0])
#
#     fleet_size = min(fleet_size, m)
#
#     if with_loads and VRP_loads is not None:
#         print("\nLOADS")
#         print("predic demand_loads[0]", VRP_loads[0].detach().cpu())
#         print("target demand_loads[0]", target_loads[0].detach().cpu())
#
#     print("\nGREEDY PATH")
#
#     # batch 0
#     for i in range(fleet_size):
#         # Pred path
#         if torch.is_tensor(predicted_tours):
#             # successor map case
#             pred_path = tour_from_next_of(predicted_tours[0, i])
#         else:
#             # tour-list case
#             pred_t = predicted_tours[0][i]
#             pred_path = pred_t.detach().cpu().tolist()
#
#         # Target path from sparse adjacency -> successor indices
#         targ_succ = targets[0, i].to_dense().max(1)[1]
#         targ_path = targ_as_lst(targ_succ, n)
#
#         print(f"CURR PRED CONSECUTIVE PATH  v_{i}: {pred_path}")
#         print(f"TARGET CONSECUTIVE PATH for v_{i}: {targ_path}")
#
#

def union_succ_from_targets_sparse(targets_sparse, B, M, N, device):
    """
    targets_sparse: sparse COO [B,M,N,N] with exactly one 1 per row for the vehicle that owns the row.
    returns targ_succ_u: [B,N] with -1 for rows without a successor.
    """
    idx = targets_sparse.coalesce().indices()  # [4, nnz] (b,m,i,j)
    b = idx[0]; i = idx[2]; j = idx[3]

    T_u = torch.zeros((B, N, N), device=device, dtype=torch.float32)
    T_u[b, i, j] = 1.0

    # remove self just in case
    diag = torch.eye(N, device=device, dtype=torch.bool)
    T_u = T_u.masked_fill(diag[None], 0.0)

    row_sum = T_u.sum(dim=-1)                 # [B,N]
    targ_succ_u = T_u.argmax(dim=-1)          # [B,N]
    targ_succ_u = torch.where(row_sum > 0.5, targ_succ_u, torch.full_like(targ_succ_u, -1))
    return targ_succ_u

def greedy_path_from_logits(L_dir_1: torch.Tensor, start=0):
    # L_dir_1: [N,N] logits
    N = L_dir_1.size(0)
    cur = start
    visited = torch.zeros(N, dtype=torch.bool, device=L_dir_1.device)
    visited[start] = True
    tour = [start]

    for _ in range(N + 1):
        scores = L_dir_1[cur].clone()
        scores[visited] = -1e9
        scores[start] = scores[start]  # depot allowed
        nxt = int(scores.argmax().item())
        tour.append(nxt)
        if nxt == start:
            break
        if visited[nxt]:
            tour.append(start)
            break
        visited[nxt] = True
        cur = nxt

    if tour[-1] != start:
        tour.append(start)
    return tour

def tour_from_next_of(next_of_1v: torch.Tensor, start: int = 0):
    """
    Reconstruct [0, ..., 0] from a successor map but:
      - stops on depot
      - stops on cycles/repeats
      - stops if next is invalid
    """
    N = next_of_1v.size(0)
    tour = [start]
    cur = start
    seen = set([start])

    for _ in range(N + 1):
        nxt = int(next_of_1v[cur].item())
        # invalid / out of range -> close
        if nxt < 0 or nxt >= N:
            tour.append(0)
            break
        tour.append(nxt)
        # close on depot
        if nxt == 0:
            break
        # cycle -> force close
        if nxt in seen:
            tour.append(0)
            break
        seen.add(nxt)
        cur = nxt

    # ensure ends at depot
    if tour[-1] != 0:
        tour.append(0)
    return tour


def tour_from_next_of_old(next_of_1v: torch.Tensor, start: int = 0, max_steps: int = None):
    """next_of_1v: [N] successor map for ONE vehicle -> [0, a, b, ..., 0]."""
    N = next_of_1v.size(0)
    max_steps = N if max_steps is None else int(max_steps)
    tour = [start]
    cur = start
    for _ in range(max_steps):
        nxt = int(next_of_1v[cur].item())
        tour.append(nxt)
        if nxt == 0:
            break
        cur = nxt
    return tour


def greedy_tours(predicted_tours, n, fleet_size=4):

    # Determine M
    if torch.is_tensor(predicted_tours):
        # next_of tensor [B,M,N]
        m = predicted_tours.size(1)
    else:
        # list-of-list
        m = len(predicted_tours[0])

    fleet_size = min(fleet_size, m)

    greedy_tours_ = []
    for i in range(fleet_size):
        # Pred path
        if torch.is_tensor(predicted_tours):
            # successor map case
            pred_path = tour_from_next_of(predicted_tours[0, i])
        else:
            # tour-list case
            pred_t = predicted_tours[0][i]
            pred_path = pred_t.detach().cpu().tolist()
        greedy_tours_.append(pred_path)

    return greedy_tours_

def print_func_old(VRP_probs,predicted_tours,VRP_loads,targets,target_loads,with_loads,fleet_size=4):

    # print('targets', targets)
    # print('targets.to_dense().size()', targets.to_dense().size())
    # print('targets[0].to_dense().max(1)[1].size() IN PRINT FUNC', targets[0].to_dense().max(1)[1].size())
    target_path_b0 = [targ_as_lst(targets[0][i].to_dense().max(1)[1],
                                  VRP_probs.size(2), print_indcs=False) for i in range(fleet_size)]
    # print('target_path_b0', target_path_b0)
    # print('sample_path_b0[0]', sample_path_b0[0])
    # if isinstance(predicted_tours,list):
    #     sample_path_b0 = predicted_tours[0]
    # if not isinstance(predicted_tours[0][0], int):
    #     if not isinstance(predicted_tours[0][0], torch.Tensor):
    #         predicted_tours = predicted_tours[0]  # if multiple seeds in sample_path_b0 take first sampled sol
    #     else:
    #         predicted_tours = [t.detach().cpu().tolist() for t in predicted_tours]
    # if isinstance(target_path_b0,torch.Tensor):
    #     is_tensor=True
    # else:
    #     is_tensor=False
    # # print('sample_path_b0[0]', sample_path_b0[0])
    # #     sample_path_b0 = sample_path_b0[0]
    # # if not isinstance(sample_path_b0[0][0], int):
    # #     sample_path_b0 = sample_path_b0[0]  # if multiple seeds in sample_path_b0 take first sampled sol

    n=VRP_probs.size(2)
    if with_loads:
        print('\nLOADS')
        print('predic demand_loads[0]',VRP_loads[0])
        print('target demand_loads[0]',target_loads[0])
    # print('targets[0]', targets[0])
    # print('targets[0].to_dense().max(1)[1]', targets[0].to_dense().max(1)[1])
    # print('targets[0][0].to_dense()', targets[0][0].to_dense())
    print('\nGREEDY PATH')
    # path_idx is [B,M,N] successor map
    next_of_b0 = predicted_tours[0]  # rename predicted_tours -> next_of when you call print_func
    for i in range(fleet_size):
        print(f'CURR PRED CONSECUTIVE PATH  v_{i}: {successor_to_path(next_of_b0[i], n)}')
        print(f'TARGET CONSECUTIVE PATH for v_{i}: {targ_as_lst(targets[0][i].to_dense().max(1)[1], n)}')

    # print('CURR PRED CONSECUTIVE PATH  v1',sample_path_b0[1])
    # print('TARGET CONSECUTIVE PATH for v1',targ_as_lst(targets[0][1].to_dense().max(1)[1],n))
    # print('CURR PRED CONSECUTIVE PATH  v2',sample_path_b0[2])
    # print('TARGET CONSECUTIVE PATH for v2',targ_as_lst(targets[0][2].to_dense().max(1)[1],n))
    # print('CURR PRED CONSECUTIVE PATH  v3',sample_path_b0[3])
    # print('TARGET CONSECUTIVE PATH for v3',targ_as_lst(targets[0][3].to_dense().max(1)[1],n))
        
    # if vrp50:
    #     print('CURR PRED CONSECUTIVE PATH  v4',sample_path_b0[4])
    #     print('TARGET CONSECUTIVE PATH for v4',targ_as_lst(targets[0][4].to_dense().max(1)[1],n))
    #     print('CURR PRED CONSECUTIVE PATH  v5',sample_path_b0[5])
    #     print('TARGET CONSECUTIVE PATH for v5',targ_as_lst(targets[0][5].to_dense().max(1)[1],n))
    #     print('CURR PRED CONSECUTIVE PATH  v6',sample_path_b0[6])
    #     print('TARGET CONSECUTIVE PATH for v6',targ_as_lst(targets[0][6].to_dense().max(1)[1],n))