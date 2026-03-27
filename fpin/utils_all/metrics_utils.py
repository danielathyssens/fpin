import torch

@torch.no_grad()
def _sym_no_diag(x: torch.Tensor) -> torch.Tensor:
    x = 0.5 * (x + x.transpose(1,2))
    n = x.size(-1)
    diag = torch.eye(n, device=x.device, dtype=torch.bool).expand_as(x)
    return x.masked_fill(diag, 0.0)

@torch.no_grad()
def _flatten_upper_tri(P: torch.Tensor, T: torch.Tensor):
    # P,T: [B,N,N] probabilities and binary targets (symmetrized, no diag)
    B, N, _ = P.shape
    iu, ju = torch.triu_indices(N, N, offset=1, device=P.device)
    # shape -> [B, E] where E = N*(N-1)/2
    p_flat = P[:, iu, ju]
    t_flat = T[:, iu, ju]
    return p_flat, t_flat, (iu, ju)

@torch.no_grad()
def pr_auc_average_precision(P: torch.Tensor, T: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """
    Torch-only Average Precision (AP) a.k.a. PR-AUC.
    Equivalent to mean precision at the ranks of the positive examples.
    """
    P = P.clamp(0, 1)
    p_flat, t_flat, _ = _flatten_upper_tri(P, T)
    ap_list = []
    for b in range(p_flat.shape[0]):
        scores = p_flat[b]                  # [E]
        labels = t_flat[b] > 0.5            # [E] bool
        pos = labels.sum()
        if pos == 0:
            ap_list.append(torch.tensor(0.0, device=P.device))
            continue
        # sort by score desc
        order = torch.argsort(scores, descending=True)
        labels_sorted = labels[order].to(torch.float32)  # 1 at positive ranks
        # precision at each rank i: TP_i / (i+1); TP_i = cumsum(labels_sorted)
        tp_cum = torch.cumsum(labels_sorted, dim=0)
        ranks = torch.arange(1, labels_sorted.numel()+1, device=P.device, dtype=torch.float32)
        precision_at_i = tp_cum / ranks
        # AP = mean precision at positive ranks
        ap = (precision_at_i * labels_sorted).sum() / (pos + eps)
        ap_list.append(ap)
    return torch.stack(ap_list).mean()

@torch.no_grad()
def topk_edge_precision(P: torch.Tensor, T: torch.Tensor, K: int = 100) -> torch.Tensor:
    """
    Precision@K over upper-tri edges.
    If K > #edges, we cap to available edges.
    """
    P = P.clamp(0, 1)
    p_flat, t_flat, _ = _flatten_upper_tri(P, T)
    prec_list = []
    for b in range(p_flat.shape[0]):
        scores = p_flat[b]
        labels = (t_flat[b] > 0.5).to(torch.float32)
        E = scores.numel()
        k = min(K, E)
        if k == 0:
            prec_list.append(torch.tensor(0.0, device=P.device))
            continue
        topk_idx = torch.topk(scores, k=k, largest=True).indices
        hits = labels[topk_idx].sum()
        prec_list.append(hits / k)
    return torch.stack(prec_list).mean()

@torch.no_grad()
def make_union_probs_from_probs(edge_probs: torch.Tensor, agg: str = "max") -> torch.Tensor:
    """
    edge_probs: [B, M, N, N] probabilities (your attn_weights reshaped)
    returns P_u: [B, N, N] symmetrized, with zero diagonal
    """
    assert edge_probs.dim() == 4
    B, M, N, _ = edge_probs.shape
    # 1) aggregate over vehicles
    if agg == "max":
        P = edge_probs.max(dim=1).values
    elif agg == "mean":
        P = edge_probs.mean(dim=1)
    else:
        raise ValueError("agg must be 'max' or 'mean'")

    # 2) symmetrize
    P = 0.5 * (P + P.transpose(1, 2))

    # 3) zero diagonal (no self-edges)
    I = torch.eye(N, device=P.device, dtype=torch.bool)
    P = P.masked_fill(I, 0.0)
    return P


@torch.no_grad()
def make_union_targets(T: torch.Tensor) -> torch.Tensor:
    T = T.to(dtype=torch.float32)
    T = 0.5 * (T + T.transpose(1, 2))
    I = torch.eye(T.size(-1), device=T.device, dtype=torch.bool)
    T = T.masked_fill(I, 0.0)
    return T