from __future__ import annotations
from typing import List, Tuple, Optional
import torch
import torch.nn.functional as F

import numpy as np
import torch

# We distill by comparing predicted heatmap edge_probs against union of undirected edges in the HGS solution (across vehicles).
# --> gives permutation invariance over vehicles and direction invariance automatically.


def routes_to_union_edges_old(
    routes: List[List[int]],     # per-vehicle routes, **with** depot 0 at ends
    N: int,                      # total nodes incl. depot
) -> torch.Tensor:
    """
    Returns a [N, N] float target with undirected edge union (0/1).
    """
    tgt = torch.zeros((N, N), dtype=torch.float32)
    for r in routes:
        for u, v in zip(r[:-1], r[1:]):
            i, j = (u, v) if u <= v else (v, u)
            tgt[i, j] = 1.0
            tgt[j, i] = 1.0
    # zero diagonal (no self loops)
    tgt.fill_diagonal_(0.0)
    return tgt


def routes_to_union_edges(routes: list[list[int]], N: int) -> torch.Tensor:
    """
    routes: [[0, a, b, ..., 0], ...] using depot 0.
    returns [N, N] float {0,1} union of undirected edges across all routes.
    """
    M = torch.zeros((N, N), dtype=torch.float32)
    # print('routes', routes)
    for r in routes:
        # ensure depot at ends
        if r[0] != 0: r = [0] + r
        if r[-1] != 0: r = r + [0]
        for i, j in zip(r[:-1], r[1:]):
            M[i, j] = 1.0
            M[j, i] = 1.0
    # torch.fill_diagonal_(M, 0.0)
    M.fill_diagonal_(0.0)
    return M

def make_union_targets_batch(all_routes: list[list[list[int]]], N: int, device: torch.device) -> torch.Tensor:
    """
    all_routes: length B; each item is a list of routes for that instance.
    returns [B, N, N] float {0,1}
    """
    mats = [routes_to_union_edges(rs, N) for rs in all_routes]
    return torch.stack(mats, dim=0).to(device)

# Distillation loss (per-edge BCE) against your heatmap aggregated over vehicles (max/mean):

# @torch.no_grad()
def _make_sym_no_diag(x: torch.Tensor) -> torch.Tensor:
    # x: [B, N, N]
    x = 0.5 * (x + x.transpose(1, 2))
    n = x.size(-1)
    diag = torch.eye(n, device=x.device, dtype=torch.bool).expand_as(x)
    x = x.masked_fill(diag, 0.0)
    return x


def union_bce(
    edge_tensor: torch.Tensor,      # [B, M, N, N]  probs in [0,1] OR logits (see input_type)
    union_targets: torch.Tensor,    # [B, N, N] in {0,1}
    *,
    input_type: str = "probs",      # "probs" | "logits"
    agg: str = "mean",              # "mean" | "max" (mean spreads gradient early)
    auto_pos_weight: bool = True,
    pos_weight: float | None = None,
    eps: float = 1e-6,
    reduction: str = "mean",
    return_heatmap: bool = False,
    opts = None
) -> torch.Tensor:
    """
    Distill loss for union heatmap used by HGS.

    Steps:
      1) Convert model outputs to probabilities per-vehicle (if logits given).
      2) Aggregate across vehicles ("mean" or "max") -> P [B,N,N].
      3) Symmetrize & zero diagonal.
      4) Convert aggregated probs P -> logits Z = logit(P).
      5) BCEWithLogits(Z, T) with optional pos_weight.
    """
    assert edge_tensor.dim() == 4, "edge_tensor must be [B, M, N, N]"
    B, M, N, _ = edge_tensor.shape

    # 1) per-vehicle probs
    if input_type == "logits":
        probs_m = edge_tensor.sigmoid()                  # [B,M,N,N]
    elif input_type == "probs":
        probs_m = edge_tensor
    else:
        raise ValueError("input_type must be 'probs' or 'logits'")

    # 2) aggregate vehicles
    if agg == "mean":
        P = probs_m.mean(dim=1)                          # [B,N,N]
    elif agg == "max":
        P = probs_m.max(dim=1).values                    # [B,N,N]
    else:
        raise ValueError("agg must be 'mean' or 'max'")

    # 3) symmetrize + zero diagonal
    P = 0.5 * (P + P.transpose(1, 2))
    I = torch.eye(N, device=P.device, dtype=torch.bool)
    P = P.masked_fill(I, 0.0)

    T = union_targets.to(P.device, dtype=P.dtype)
    T = 0.5 * (T + T.transpose(1, 2)).masked_fill(I, 0.0)

    # 4) probs -> logits for numerically stable BCE
    Pc = P.clamp(eps, 1 - eps)
    Z = Pc.log() - torch.log1p(-Pc)  # logit(P)
    # Pc = P.clamp(eps, 1 - eps)
    # Z  = Pc.log() - (1 - Pc).log1p(-Pc)

    # 5) pos_weight
    if pos_weight is None and auto_pos_weight:
        I = torch.eye(N, device=T.device, dtype=torch.bool)
        # valid = ~I
        valid = (~I).unsqueeze(0).expand(B, N, N)  # [B,N,N]
        # pos = T[valid].sum()
        pos = T.masked_select(valid).sum()
        neg = valid.sum() - pos
        pos_weight = (neg / (pos + eps)).detach() * getattr(opts, "pos_weight_scale", 1.0)
        # pos = T.sum()
        # neg = T.numel() - pos
        # pos_weight = (neg / (pos + eps)).detach() * getattr(opts, "pos_weight_scale", 1.0)
        pos_weight = pos_weight.clamp(max=getattr(opts, "pos_weight_cap", 50.0))

    loss = F.binary_cross_entropy_with_logits(
        Z, T,
        pos_weight=None if pos_weight is None else torch.as_tensor(pos_weight, device=Z.device, dtype=Z.dtype),
        reduction=reduction
    )
    return (loss, P) if return_heatmap else loss


def degree_prior_loss(
    P: torch.Tensor,              # [B,N,N], sym, diag=0
    depot_idx: int = 0,
    target_degree: float = 2.0,
    reduction: str = "mean",
) -> torch.Tensor:
    B, N, _ = P.shape
    deg = P.sum(dim=-1)  # [B,N]
    mask = torch.ones(B, N, device=P.device, dtype=torch.bool)
    mask[:, depot_idx] = False
    return F.smooth_l1_loss(
        deg[mask],
        torch.full_like(deg[mask], target_degree, dtype=P.dtype),
        reduction=reduction
    )

def row_entropy(P: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # average row-wise entropy over batch and rows
    Q = P.clamp(eps, 1 - eps)
    ent = -(Q * Q.log()).sum(dim=-1).mean()
    return ent


def union_bce_from_probs(
    edge_probs: torch.Tensor,   # [B,M,N,N]  <-- your vrp_probs (requires_grad=True)
    union_targets: torch.Tensor # [B,N,N] in {0,1}
) -> torch.Tensor:
    # aggregate over vehicles (keeps grad)
    P = edge_probs.max(dim=1).values   # or .mean(dim=1) if you prefer
    P = _make_sym_no_diag(P)

    T = _make_sym_no_diag(union_targets.to(P.device, dtype=P.dtype))

    eps = 1e-6
    loss_pos = -T * torch.log(P.clamp_min(eps))
    loss_neg = -(1 - T) * torch.log((1 - P).clamp_min(eps))
    return (loss_pos + loss_neg).mean()

def edge_union_bce_with_logits(
    edge_logits: torch.Tensor,     # [B, M, N, N] (per-vehicle logits)
    union_targets: torch.Tensor,   # [B, N, N] in {0,1}
    agg: str = "max",              # 'max' or 'mean' across vehicles (in probability space)
    auto_pos_weight: bool = True,
    pos_weight: float | None = None,
    eps: float = 1e-6,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Collapses per-vehicle logits to a single heatmap, compares to union targets with BCE-with-logits.
    Steps:
      1) Convert per-vehicle logits -> probs, aggregate across vehicles ('max' or 'mean').
      2) Symmetrize and zero the diagonal.
      3) Convert aggregated probs back to logits via logit() with clamping.
      4) BCE-with-logits vs. union targets; supports pos_weight (auto by default).
    """
    assert edge_logits.dim() == 4, "edge_logits must be [B, M, N, N]"
    B, M, N, _ = edge_logits.shape

    # 1) aggregate across vehicles in PROB space
    probs_m = edge_logits.sigmoid()                               # [B, M, N, N]
    if agg == "max":
        P = probs_m.max(dim=1).values                              # [B, N, N]
    elif agg == "mean":
        P = probs_m.mean(dim=1)                                    # [B, N, N]
    else:
        raise ValueError("agg must be 'max' or 'mean'")

    # 2) symmetrize + remove self-loops
    P = _make_sym_no_diag(P)                                       # [B, N, N]

    # 3) convert probs -> logits (numerically stable BCE)
    Pc = P.clamp_(eps, 1 - eps)
    Z = torch.log(Pc) - torch.log1p(-Pc)                           # logit(P)

    # 4) targets (symmetrized, no diagonal)
    T = union_targets.to(P.device, dtype=P.dtype)
    T = _make_sym_no_diag(T)

    # 5) pos_weight
    if pos_weight is None and auto_pos_weight:
        # compute per-batch scalar pos_weight (can also do per-sample)
        pos = T.sum()
        neg = T.numel() - pos
        pos_weight = (neg / (pos + eps)).detach()

    loss = F.binary_cross_entropy_with_logits(
        Z, T, pos_weight=None if pos_weight is None else torch.as_tensor(pos_weight, device=Z.device, dtype=Z.dtype),
        reduction=reduction
    )
    return loss


def edge_union_bce_loss_old(
    edge_probs: torch.Tensor,     # [B, M, N, N]
    union_targets: torch.Tensor,  # [B, N, N] in {0,1}
    agg: str = "max",
    eps: float = 1e-9,
    pos_weight: float | None = None,
):
    # pos_weight ≈ (#neg / #pos)
    B, M, N, _ = edge_probs.shape

    P = edge_probs.max(dim=1).values if agg == "max" else edge_probs.mean(dim=1)  # [B, N, N]
    P = 0.5 * (P + P.transpose(1, 2))
    I = torch.eye(N, device=P.device, dtype=torch.bool)
    P = P.masked_fill(I, 0.0)

    T = union_targets.to(P.device)
    T = 0.5 * (T + T.transpose(1, 2))
    T = T.masked_fill(I, 0.0)

    # BCE with optional positive reweight
    Pw = 1.0 if pos_weight is None else pos_weight
    loss_pos = - T * torch.log(P.clamp_min(eps)) * Pw
    loss_neg = - (1 - T) * torch.log((1 - P).clamp_min(eps))
    # print('loss_pos[0]', loss_pos[0])
    # print('loss_neg[0]', loss_neg[0])
    return (loss_pos + loss_neg).mean()

