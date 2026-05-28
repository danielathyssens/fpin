"""Toy validation: attention vs. pooling on capacity-constrained ASSIGNMENT, vs. tightness.

Isolates the paper's inductive-bias claim from routing/geometry confounds. The task is pure
capacity bin-packing CO-ASSIGNMENT: given N items with demands and bins of capacity Q=1,
predict the N x N co-assignment matrix C_ij = 1 iff items i, j share a bin under a reference
packing (first-fit-decreasing). C is permutation-invariant and encodes the PAIRWISE "which
items pack together" relation.

Why this is the right test (pooling-adverse by construction): which items share a bin
depends on demand COMPLEMENTARITY (do their demands sum under Q, given everyone else) — a
pairwise/global relation. Deep-Sets pooling only sees per-item demand + the demand mean, so
it cannot resolve specific pairings; Set-Transformer attention can compare demands pairwise.
(A geometry-driven clustering target, by contrast, is exactly what mean-pooling is good at,
which is why an earlier geometric version did NOT isolate the claim.)

Tightness rho = sum(demand)/(M*Q). Prediction: attention - pooling co-assignment F1 gap is
small when loose (slack -> packing trivial) and GROWS as rho -> 1 (bins near-full ->
complementarity binds). Optionally add geometry distractor coords with --geom_dims 2.

Usage (quick):  python experiments/toy_attention_vs_pooling.py --quick
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def ffd_pack(demand, cap=1.0):
    """First-fit-decreasing bin packing -> per-item bin label. Pairwise/global by nature."""
    order = np.argsort(demand)[::-1]
    bins = []  # remaining capacity per open bin
    label = np.empty(len(demand), dtype=int)
    for i in order:
        placed = False
        for b, rem in enumerate(bins):
            if rem + 1e-9 >= demand[i]:
                label[i] = b; bins[b] -= demand[i]; placed = True; break
        if not placed:
            bins.append(cap - demand[i]); label[i] = len(bins) - 1
    return label


def make_batch(B, N, M, rho, cap=1.0, geom_dims=0, seed=None):
    rng = np.random.default_rng(seed)
    # demands scaled so total = rho * M * cap (the binding resource)
    raw = rng.random((B, N)).astype(np.float32) + 0.15
    demand = raw / raw.sum(1, keepdims=True) * (rho * M * cap)
    demand = np.clip(demand, 0.0, cap)  # no single item exceeds a bin
    feats = [demand[:, :, None]]
    if geom_dims:
        feats.append(rng.random((B, N, geom_dims)).astype(np.float32))  # distractor coords
    X = np.concatenate(feats, axis=2)
    C = np.zeros((B, N, N), dtype=np.float32)
    for b in range(B):
        lab = ffd_pack(demand[b], cap)
        C[b] = (lab[:, None] == lab[None, :]).astype(np.float32)
    return torch.tensor(X), torch.tensor(C)


class PoolingEncoder(nn.Module):  # Deep Sets
    def __init__(self, in_dim, d=128, layers=3):
        super().__init__()
        self.inp = nn.Linear(in_dim, d)
        self.blocks = nn.ModuleList([nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, d))
                                     for _ in range(layers)])

    def forward(self, x):
        h = F.relu(self.inp(x))
        for blk in self.blocks:
            pooled = h.mean(1, keepdim=True).expand_as(h)
            h = h + blk(torch.cat([h, pooled], dim=-1))
        return h


class AttentionEncoder(nn.Module):  # mini Set Transformer
    def __init__(self, in_dim, d=128, layers=3, heads=4):
        super().__init__()
        self.inp = nn.Linear(in_dim, d)
        self.attn = nn.ModuleList([nn.MultiheadAttention(d, heads, batch_first=True) for _ in range(layers)])
        self.ln1 = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.ff = nn.ModuleList([nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d)) for _ in range(layers)])
        self.ln2 = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])

    def forward(self, x):
        h = F.relu(self.inp(x))
        for a, ln1, ff, ln2 in zip(self.attn, self.ln1, self.ff, self.ln2):
            o, _ = a(h, h, h); h = ln1(h + o); h = ln2(h + ff(h))
        return h


def coassign_logits(h):
    return torch.matmul(h, h.transpose(1, 2)) / (h.size(-1) ** 0.5)


def f1_offdiag(logits, target):
    pred = (torch.sigmoid(logits) > 0.5).float()
    N = pred.size(1); eye = torch.eye(N, device=pred.device).bool()[None]
    pred = pred.masked_fill(eye, 0); tgt = target.masked_fill(eye, 0)
    tp = (pred * tgt).sum(); fp = (pred * (1 - tgt)).sum(); fn = ((1 - pred) * tgt).sum()
    return float((2 * tp / (2 * tp + fp + fn + 1e-9)).item())


def train_eval(model, in_dim, N, M, rho, steps, bs, geom, device, seed):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for s in range(steps):
        x, C = make_batch(bs, N, M, rho, geom_dims=geom, seed=seed + s * 7)
        x, C = x.to(device), C.to(device)
        logits = coassign_logits(model(x))
        pw = ((1 - C).sum() / (C.sum() + 1e-9)).clamp(1, 50)
        loss = F.binary_cross_entropy_with_logits(logits, C, pos_weight=pw)
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        x, C = make_batch(512, N, M, rho, geom_dims=geom, seed=99999)
        x, C = x.to(device), C.to(device)
        return f1_offdiag(coassign_logits(model(x)), C)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=24)
    ap.add_argument("--M", type=int, default=6)
    ap.add_argument("--rhos", type=float, nargs="+", default=[0.6, 0.75, 0.85, 0.95])
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--geom_dims", type=int, default=0, help="distractor coord dims (0 = pure packing)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.steps, args.rhos = 120, [0.6, 0.95]
    in_dim = 1 + args.geom_dims
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} task=bin-packing N={args.N} M={args.M} in_dim={in_dim} steps={args.steps}")
    print(f"{'rho':>6} {'pool_F1':>9} {'attn_F1':>9} {'gap(attn-pool)':>16}")
    for rho in args.rhos:
        torch.manual_seed(args.seed)
        f_pool = train_eval(PoolingEncoder(in_dim).to(device), in_dim, args.N, args.M, rho, args.steps, args.bs, args.geom_dims, device, args.seed)
        torch.manual_seed(args.seed)
        f_attn = train_eval(AttentionEncoder(in_dim).to(device), in_dim, args.N, args.M, rho, args.steps, args.bs, args.geom_dims, device, args.seed + 1000)
        print(f"{rho:>6.2f} {f_pool:>9.3f} {f_attn:>9.3f} {f_attn - f_pool:>16.3f}")


if __name__ == "__main__":
    main()
