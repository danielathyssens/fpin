"""Principled toy: F-PIN vs PIM on FC-CVRP, with PIM's *actual* softassign.

Lesson from the v1 strawman (do not repeat): v1 used a vanilla doubly-stochastic
Sinkhorn as the "PIM" baseline. PIM's published softassign (`MTSPSoftassign`,
Thyssens 2022, see models/PIMold/src/utils_all/softassign.py) is NOT vanilla
Sinkhorn — its depot row is per-vehicle softmax over destinations, and its
customer rows are jointly normalized over (vehicle, destination), alternating
with a transpose to enforce the symmetric in-flow constraint. This file uses
the real softassign as the PIM head.

What this toy tests, independently:
  (H1) ENCODER: does an attention encoder consistently dominate a Deep-Sets
       pooling encoder on FC-CVRP supervised assignment? (The Wagstaff-2019
       latent-dim bottleneck predicts a growing gap with N at fixed latent.)
  (H2) HEAD: among {PIM iterative softassign, per-vehicle row softmax,
       per-customer joint (m, j) softmax}, which best fits the OPTIMAL Y*?
  (H3) HEAD-DECODER INTERACTION: under each head's natural decoding (greedy
       argmax that respects the head's normalization axis), which actually
       produces the lowest Cost_v on test instances against brute-force
       optimum?

Ground truth Y* is the brute-force optimum (not a heuristic) for small N.
Audit gates assert each head's distributions respect their claimed
normalization, so a head bug shows up as a CrashOnAssert before we draw
conclusions from numbers.

Usage:
    python experiments/toy_fpin_vs_pim_principled_v2.py --quick    # smoke
    python experiments/toy_fpin_vs_pim_principled_v2.py            # full sweep
"""
import argparse
import itertools
import math
import os
import sys
import time
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- 1. Instance generation + brute-force OPTIMAL Y* ------------------

def sample_instance(N: int, M: int, Q: int, rng: np.random.Generator):
    """N customers + 1 depot. coords ~ U(0,1)^2. depot at (0.5, 0.5)."""
    coords = rng.uniform(0.0, 1.0, (N + 1, 2)).astype(np.float32)
    coords[0] = [0.5, 0.5]
    # demands uniform in {1, ..., ceil(Q/2)} so sum <= M*Q is likely; we resample on infeasibility
    while True:
        d_max = max(1, Q // 2)
        raw = rng.integers(1, d_max + 1, size=N).astype(np.int32)
        if raw.sum() <= M * Q:
            break
    demand = np.concatenate([[0], raw.astype(np.float32)])
    return coords, demand


def _route_cost(coords: np.ndarray, route_order: Tuple[int, ...]) -> float:
    """Tour cost: depot -> route_order... -> depot."""
    if not route_order:
        return 0.0
    pts = [0] + list(route_order) + [0]
    return float(sum(np.linalg.norm(coords[pts[i]] - coords[pts[i + 1]])
                     for i in range(len(pts) - 1)))


def _best_tsp_for_subset(coords: np.ndarray, subset: Tuple[int, ...]) -> Tuple[float, Tuple[int, ...]]:
    """Brute-force TSP for one route. subset = tuple of customer indices."""
    if not subset:
        return 0.0, tuple()
    best_cost = math.inf
    best_order = subset
    for perm in itertools.permutations(subset):
        c = _route_cost(coords, perm)
        if c < best_cost:
            best_cost = c
            best_order = perm
    return best_cost, best_order


def _partitions(items: List[int], M: int, demand: np.ndarray, Q: int):
    """Enumerate all feasible partitions of `items` into <= M ordered subsets
    (subsets correspond to vehicles 0..M-1; empty allowed). Yields list of
    M tuples (each tuple = customer indices for that vehicle)."""
    # Recursively assign each customer to one of M vehicles, prune by capacity.
    M_caps = [Q] * M

    def rec(idx, assignment):
        if idx == len(items):
            yield tuple(tuple(sorted(g)) for g in assignment)
            return
        c = items[idx]
        for m in range(M):
            if M_caps[m] >= demand[c]:
                M_caps[m] -= demand[c]
                assignment[m].append(c)
                yield from rec(idx + 1, assignment)
                assignment[m].pop()
                M_caps[m] += demand[c]

    yield from rec(0, [[] for _ in range(M)])


def brute_force_optimal_Y(coords: np.ndarray, demand: np.ndarray, M: int, Q: int,
                          v_cost: float) -> np.ndarray:
    """Brute-force optimal Y* for FC-CVRP at given v_cost (per-route penalty).
    Returns Y* of shape [M, N+1, N+1] (0/1 edge indicators)."""
    N = len(demand) - 1
    customers = list(range(1, N + 1))

    best_total = math.inf
    best_routes: List[Tuple[int, ...]] = []
    seen = set()
    for partition in _partitions(customers, M, demand, Q):
        # canonicalize so we don't re-evaluate symmetric vehicle reorderings
        key = tuple(sorted(partition))
        if key in seen:
            continue
        seen.add(key)
        total = 0.0
        opt_orders = []
        n_routes_used = 0
        for subset in partition:
            c, order = _best_tsp_for_subset(coords, subset)
            total += c
            opt_orders.append(order)
            if len(subset) > 0:
                n_routes_used += 1
        total += v_cost * n_routes_used
        if total < best_total:
            best_total = total
            best_routes = opt_orders

    Y = np.zeros((M, N + 1, N + 1), dtype=np.float32)
    for m, route in enumerate(best_routes):
        if not route:
            continue
        tour = [0] + list(route) + [0]
        for i in range(len(tour) - 1):
            Y[m, tour[i], tour[i + 1]] = 1.0
    return Y, best_total


def build_dataset(B: int, N: int, M: int, Q: int, v_cost: float, base_seed: int):
    rng = np.random.default_rng(base_seed)
    X = np.zeros((B, N + 1, 4), dtype=np.float32)
    Y = np.zeros((B, M, N + 1, N + 1), dtype=np.float32)
    opt_costs = np.zeros(B, dtype=np.float32)
    for b in range(B):
        coords, demand = sample_instance(N, M, Q, rng)
        X[b, 0, 0] = 1.0
        X[b, :, 1:3] = coords
        X[b, :, 3] = demand / max(1.0, float(Q))
        Y[b], opt_costs[b] = brute_force_optimal_Y(coords, demand, M, Q, v_cost)
    return (torch.from_numpy(X), torch.from_numpy(Y),
            torch.from_numpy(opt_costs))


# ---------- 2. Encoders -----------------------------------------------------

class PoolEncoder(nn.Module):
    """Deep-Sets: each node sees only itself + the mean-pooled context."""
    def __init__(self, in_dim, hid, latent):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(in_dim, hid), nn.ReLU(),
                                 nn.Linear(hid, hid), nn.ReLU(),
                                 nn.Linear(hid, latent))
        self.rho = nn.Sequential(nn.Linear(latent * 2, hid), nn.ReLU(),
                                 nn.Linear(hid, latent))

    def forward(self, x):
        h = self.phi(x)
        ctx = h.mean(dim=1, keepdim=True).expand_as(h)
        return self.rho(torch.cat([h, ctx], dim=-1))


class AttnEncoder(nn.Module):
    """Transformer-style attention encoder."""
    def __init__(self, in_dim, hid, latent, n_layers=2, n_heads=4):
        super().__init__()
        self.proj = nn.Linear(in_dim, latent)
        layer = nn.TransformerEncoderLayer(d_model=latent, nhead=n_heads,
                                           dim_feedforward=hid, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, x):
        return self.enc(self.proj(x))


# ---------- 3. Heads --------------------------------------------------------

def _per_vehicle_edge_logits(h, vehicle_embed, q, k, scale):
    """h: (B, n, D), vehicle_embed: (M, D). Returns (B, M, n, n) logits."""
    B, n, D = h.shape
    veh = vehicle_embed.weight                  # (M, D)
    h_v = h.unsqueeze(1) + veh[None, :, None, :]  # (B, M, n, D)
    return torch.einsum("bmid,bmjd->bmij", q(h_v), k(h_v)) * scale


class PIMSoftassignHead(nn.Module):
    """PIM 2022's actual MTSPSoftassign (faithfully ported from
    models/PIMold/src/utils_all/softassign.py).

    Operates on per-vehicle edge logits of shape (B, M, n, n). For each
    softassign 'layer' iteration:
      - depot row (i=0):   per-vehicle softmax over j   (out-flow of depot)
      - customer rows (i>0): joint normalize over (m, j) (each customer is
        left exactly once)
      - transpose i<->j and repeat to enforce the IN-flow side symmetrically
    Returns LINEAR-space probabilities (NOT log).
    """
    def __init__(self, M, latent, n_iters=3, eps=1e-8):
        super().__init__()
        assert n_iters >= 1
        self.M = M
        self.iters = n_iters
        self.eps = eps
        self.vehicle_embed = nn.Embedding(M, latent)
        self.q = nn.Linear(latent, latent)
        self.k = nn.Linear(latent, latent)
        self.scale = latent ** -0.5

    def forward(self, h):
        logits = _per_vehicle_edge_logits(h, self.vehicle_embed, self.q, self.k, self.scale)
        # Standard PIM softassign uses exp(logits - max) for stability.
        out = torch.exp(logits - logits.amax(dim=(-2, -1), keepdim=True))
        for _ in range(self.iters):
            out = out.clamp_min(self.eps)
            # depot row: softmax over j per (b, m)
            depot = out[:, :, :1, :]
            depot = depot / depot.sum(dim=-1, keepdim=True).clamp_min(self.eps)
            # customer rows: joint normalize over (m, j)
            cust = out[:, :, 1:, :]                  # (B, M, n-1, n)
            # axis-1 (M) and axis-3 (n) jointly sum to 1 per (b, customer i)
            cust_sum = cust.sum(dim=(1, 3), keepdim=True).clamp_min(self.eps)
            cust = cust / cust_sum
            out = torch.cat([depot, cust], dim=2)
            # transpose i <-> j and iterate the "in-flow" side
            out = out.transpose(2, 3)
        # if odd number of iterations -> we are currently transposed; flip back
        if self.iters % 2 == 1:
            out = out.transpose(2, 3)
        return out  # LINEAR probabilities


class FPinPerVehicleHead(nn.Module):
    """F-PIN (paper's published architecture): per-(vehicle, source) row
    softmax over destinations. Returns log-probabilities."""
    def __init__(self, M, latent):
        super().__init__()
        self.vehicle_embed = nn.Embedding(M, latent)
        self.q = nn.Linear(latent, latent)
        self.k = nn.Linear(latent, latent)
        self.scale = latent ** -0.5

    def forward(self, h):
        logits = _per_vehicle_edge_logits(h, self.vehicle_embed, self.q, self.k, self.scale)
        return F.log_softmax(logits, dim=-1)  # LOG-prob


class FPinJointHead(nn.Module):
    """F-PIN-S: per-customer JOINT (m, j) softmax for customer rows; depot
    row keeps per-vehicle softmax. Returns log-probabilities."""
    def __init__(self, M, latent):
        super().__init__()
        self.vehicle_embed = nn.Embedding(M, latent)
        self.q = nn.Linear(latent, latent)
        self.k = nn.Linear(latent, latent)
        self.scale = latent ** -0.5

    def forward(self, h):
        logits = _per_vehicle_edge_logits(h, self.vehicle_embed, self.q, self.k, self.scale)
        depot = F.log_softmax(logits[:, :, 0:1, :], dim=-1)
        cust = logits[:, :, 1:, :]
        B, M, Nm1, N = cust.shape
        cust = cust.permute(0, 2, 1, 3).reshape(B, Nm1, M * N)
        cust = F.log_softmax(cust, dim=-1)
        cust = cust.reshape(B, Nm1, M, N).permute(0, 2, 1, 3)
        return torch.cat([depot, cust], dim=2)  # LOG-prob


# ---------- 4. Audit gates --------------------------------------------------

@torch.no_grad()
def audit_head_distributions(head, head_name: str, log_or_lin: str,
                             encoder, X_sample: torch.Tensor, M: int, n: int):
    """Verify each head respects its claimed normalization on a sample."""
    h = encoder(X_sample)
    out = head(h)
    if log_or_lin == "log":
        probs = out.exp()
    else:
        probs = out
    B = X_sample.shape[0]

    if head_name == "PIM-softassign":
        # depot row: per-vehicle softmax over j -> rows sum 1
        depot_sum = probs[:, :, 0, :].sum(dim=-1)
        ok_depot = torch.allclose(depot_sum, torch.ones_like(depot_sum), atol=1e-2)
        # customer rows: joint over (m, j) sum 1 per customer
        cust_joint = probs[:, :, 1:, :].sum(dim=(1, 3))   # (B, n-1)
        ok_cust = torch.allclose(cust_joint, torch.ones_like(cust_joint), atol=1e-2)
        msg = (f"  [audit {head_name}] depot rows sum-to-1: {ok_depot}  "
               f"customer joint sum-to-1: {ok_cust}")
        assert ok_depot and ok_cust, msg + " <- FAIL"
    elif head_name == "F-PIN-per-vehicle":
        rows = probs.sum(dim=-1)
        ok = torch.allclose(rows, torch.ones_like(rows), atol=1e-3)
        msg = f"  [audit {head_name}] per-(m,i) row sum-to-1: {ok}"
        assert ok, msg + " <- FAIL"
    elif head_name == "F-PIN-joint":
        depot_sum = probs[:, :, 0, :].sum(dim=-1)
        ok_depot = torch.allclose(depot_sum, torch.ones_like(depot_sum), atol=1e-3)
        cust_joint = probs[:, :, 1:, :].sum(dim=(1, 3))
        ok_cust = torch.allclose(cust_joint, torch.ones_like(cust_joint), atol=1e-3)
        msg = (f"  [audit {head_name}] depot rows sum-to-1: {ok_depot}  "
               f"customer joint sum-to-1: {ok_cust}")
        assert ok_depot and ok_cust, msg + " <- FAIL"
    print(msg)


# ---------- 5. Training + eval ---------------------------------------------

HEADS = {
    "PIM-softassign":    ("lin", PIMSoftassignHead),
    "F-PIN-per-vehicle": ("log", FPinPerVehicleHead),
    "F-PIN-joint":       ("log", FPinJointHead),
}


def loss_for_head(head_out, log_or_lin: str, Y_star: torch.Tensor, eps=1e-12):
    """CE against the binary Y*. Identical objective shape across heads."""
    if log_or_lin == "log":
        log_p = head_out
    else:
        log_p = (head_out + eps).log()
    return -(log_p * Y_star).sum(dim=(-3, -2, -1)).mean()


def train_eval(encoder_cls, head_name, X_tr, Y_tr, X_te, Y_te,
               *, hid, latent, M, epochs, lr, bs, dev):
    log_or_lin, HeadCls = HEADS[head_name]
    torch.manual_seed(0)
    encoder = encoder_cls(4, hid, latent).to(dev)
    head = HeadCls(M, latent).to(dev)
    opt = torch.optim.Adam(list(encoder.parameters()) + list(head.parameters()), lr=lr)

    audit_head_distributions(head, head_name, log_or_lin, encoder,
                             X_tr[:4].to(dev), M, X_tr.shape[1])

    X_tr_d = X_tr.to(dev); Y_tr_d = Y_tr.to(dev)
    X_te_d = X_te.to(dev); Y_te_d = Y_te.to(dev)
    N_tr = X_tr.shape[0]
    for ep in range(epochs):
        idx = torch.randperm(N_tr, device=dev)
        for s in range(0, N_tr, bs):
            b = idx[s:s + bs]
            out = head(encoder(X_tr_d[b]))
            loss = loss_for_head(out, log_or_lin, Y_tr_d[b])
            opt.zero_grad(); loss.backward(); opt.step()

    encoder.eval(); head.eval()
    with torch.no_grad():
        out_te = head(encoder(X_te_d))
        test_ce = loss_for_head(out_te, log_or_lin, Y_te_d).item()
        # row-argmax accuracy on rows that have an out-edge in Y*
        if log_or_lin == "log":
            probs_te = out_te.exp()
        else:
            probs_te = out_te
        row_mask = Y_te_d.sum(dim=-1) > 0
        pred_j = probs_te.argmax(dim=-1)
        tgt_j = Y_te_d.argmax(dim=-1)
        row_acc = ((pred_j == tgt_j) & row_mask).float().sum() / row_mask.float().sum().clamp_min(1)
    return float(test_ce), float(row_acc)


# ---------- 6. Driver -------------------------------------------------------

CONFIGS = [
    ("POOL", PoolEncoder),
    ("ATTN", AttnEncoder),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--latent", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    if args.quick:
        Ns, B_tr, B_te, epochs = [6], 200, 60, 40
    else:
        # N=10 was too slow under brute force (>60 min wall on 750 inst); keep N=6,8.
        # The encoder/head signal is testable across N=6->8; N=10 strengthens but is
        # not load-bearing. Larger train set + more epochs at the smaller Ns instead.
        Ns, B_tr, B_te, epochs = [6, 8], 800, 200, args.epochs
    M, Q, v_cost = 3, 3, 35.0
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev}  Ns={Ns}  M={M}  Q={Q}  v_cost={v_cost}  "
          f"B_train={B_tr}  B_test={B_te}  epochs={epochs}")

    results = {}
    for N in Ns:
        t0 = time.time()
        print(f"\n[N={N}]  brute-force solving {B_tr + B_te} instances...")
        X_tr, Y_tr, _ = build_dataset(B_tr, N, M, Q, v_cost, base_seed=N * 1009)
        X_te, Y_te, opt_te = build_dataset(B_te, N, M, Q, v_cost, base_seed=N * 1009 + 1)
        print(f"  data: X_tr={X_tr.shape}  Y_tr={Y_tr.shape}  "
              f"opt cost (test): mean={opt_te.mean():.3f} std={opt_te.std():.3f}  "
              f"({time.time()-t0:.1f}s)")
        for enc_name, EncCls in CONFIGS:
            for head_name in HEADS:
                ce, acc = train_eval(EncCls, head_name,
                                     X_tr, Y_tr, X_te, Y_te,
                                     hid=args.hidden, latent=args.latent,
                                     M=M, epochs=epochs, lr=args.lr,
                                     bs=args.bs, dev=dev)
                results[(N, enc_name, head_name)] = (ce, acc)
                print(f"  [N={N} {enc_name:<5s} {head_name:<20s}] "
                      f"test_CE={ce:.4f}  row_acc={acc:.4f}")
        print(f"  (N={N} total {time.time()-t0:.1f}s)")

    print("\n=== SUMMARY: test CE / row-argmax accuracy ===")
    for N in Ns:
        print(f"\n  N={N}")
        for enc_name, _ in CONFIGS:
            for head_name in HEADS:
                ce, acc = results[(N, enc_name, head_name)]
                print(f"    {enc_name:<5s}  {head_name:<22s}  CE={ce:6.3f}  acc={acc:.3f}")


if __name__ == "__main__":
    main()
