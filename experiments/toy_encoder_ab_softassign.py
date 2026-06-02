"""Controlled encoder A/B with matched MTSPSoftassign head.

Falsifiable claim under test:
    H1: F-PIN's attention encoder produces LOWER-entropy MTSPSoftassign assignments
        than PIM's pooling encoder, at matched parameter budget + training epochs.
        Lower entropy = sharper, more committed assignments = better routing.

Setup:
    - shared MTSPSoftassign head (3 layers), shared FC-CVRP brute-force optimal Y*,
      shared training loop, shared loss.
    - 2 encoders: POOL (PIM-style DeepSets) and ATTN (F-PIN Transformer encoder).
    - matched latent dim + #layers + epochs.

Reported per N:
    - decoded Cost_v gap vs brute-force optimum   (cost signal)
    - mean per-customer assignment entropy        (sharpness signal: lower = better)
    - mean per-vehicle row entropy                (transition crispness)

If ATTN produces both lower entropy AND lower Cost_v gap -> H1 holds; F-PIN
encoder is the principled lever. If POOL ties or beats ATTN under matched
softassign head, the encoder advantage doesn't transfer through the head.

Audit gates assert softassign output is row+joint-stochastic before reporting.
"""
import argparse, time, math
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

# Reuse the brute-force solver + base encoders + softassign head from v2
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from toy_fpin_vs_pim_principled_v2 import (
    build_dataset, PoolEncoder, AttnEncoder, PIMSoftassignHead,
    decode_routes_greedy, cost_of_routes,
)


def loss_for_softassign(out, Y_star, eps=1e-12):
    log_p = (out + eps).log()
    return -(log_p * Y_star).sum(dim=(-3, -2, -1)).mean()


@torch.no_grad()
def measure_entropy(out_probs, eps=1e-12):
    """out_probs: (B, M, n, n) post-softassign linear probs.
    Returns (per_customer_entropy, per_vehicle_row_entropy)."""
    # per-customer joint distribution over (m, j) — measure how committed each customer is
    B, M, n, _ = out_probs.shape
    cust = out_probs[:, :, 1:, :]                          # (B, M, n-1, n)
    cust_flat = cust.permute(0, 2, 1, 3).reshape(B, n - 1, -1)  # (B, n-1, M*n)
    cust_flat = cust_flat / cust_flat.sum(dim=-1, keepdim=True).clamp_min(eps)
    ent_cust = -(cust_flat * cust_flat.clamp_min(eps).log()).sum(dim=-1).mean().item()
    # per-vehicle row entropy (transition sharpness)
    rows = out_probs / out_probs.sum(dim=-1, keepdim=True).clamp_min(eps)
    ent_row = -(rows * rows.clamp_min(eps).log()).sum(dim=-1).mean().item()
    return ent_cust, ent_row


def run(EncCls, name, X_tr, Y_tr, X_te, Y_te, opt_te, M, Q, v_cost,
        hid, latent, epochs, lr, bs, dev):
    torch.manual_seed(0)
    encoder = EncCls(4, hid, latent).to(dev)
    head = PIMSoftassignHead(M, latent, n_iters=3).to(dev)
    opt = torch.optim.Adam(list(encoder.parameters()) + list(head.parameters()), lr=lr)
    X_tr_d, Y_tr_d = X_tr.to(dev), Y_tr.to(dev)
    X_te_d, Y_te_d = X_te.to(dev), Y_te.to(dev)
    for ep in range(epochs):
        idx = torch.randperm(X_tr_d.shape[0], device=dev)
        for s in range(0, X_tr_d.shape[0], bs):
            b = idx[s:s + bs]
            out = head(encoder(X_tr_d[b]))
            loss = loss_for_softassign(out, Y_tr_d[b])
            opt.zero_grad(); loss.backward(); opt.step()
    encoder.eval(); head.eval()
    with torch.no_grad():
        out_te = head(encoder(X_te_d))
        # audit: row + joint sum-to-1
        depot_sum = out_te[:, :, 0, :].sum(dim=-1)
        cust_joint = out_te[:, :, 1:, :].sum(dim=(1, 3))
        assert torch.allclose(depot_sum, torch.ones_like(depot_sum), atol=1e-2)
        assert torch.allclose(cust_joint, torch.ones_like(cust_joint), atol=1e-2)
        ent_cust, ent_row = measure_entropy(out_te)
        # decoded Cost_v
        demand_norm = X_te[:, :, 3]
        routes_all = decode_routes_greedy(out_te, demand_norm, M, Q)
        coords_all = X_te[:, :, 1:3].cpu().numpy()
        costs = np.array([cost_of_routes(r, coords_all[b], v_cost)[0]
                          for b, r in enumerate(routes_all)])
        opt = opt_te.cpu().numpy()
        gap = ((costs - opt) / opt * 100).mean()
    return {"name": name, "ent_cust": ent_cust, "ent_row": ent_row,
            "cost_v": costs.mean(), "gap_pct": gap}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--epochs", type=int, default=100)
    args = ap.parse_args()
    Ns = [6] if args.quick else [6, 8]
    M, Q, v_cost = 3, 3, 35.0
    hid, latent, lr, bs = 128, 64, 1e-3, 64
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    B_tr = 200 if args.quick else 800
    B_te = 60 if args.quick else 200
    epochs = 30 if args.quick else args.epochs
    print(f"device={dev}  Ns={Ns}  epochs={epochs}  B_tr={B_tr}")
    results = {}
    for N in Ns:
        t0 = time.time()
        print(f"\n[N={N}] building data + solving optima...")
        X_tr, Y_tr, _ = build_dataset(B_tr, N, M, Q, v_cost, N * 1009)
        X_te, Y_te, opt_te = build_dataset(B_te, N, M, Q, v_cost, N * 1009 + 1)
        for EncCls, name in [(PoolEncoder, "POOL+softassign(PIM-like)"),
                              (AttnEncoder, "ATTN+softassign(F-PIN-A-like)")]:
            r = run(EncCls, name, X_tr, Y_tr, X_te, Y_te, opt_te, M, Q, v_cost,
                    hid, latent, epochs, lr, bs, dev)
            results[(N, name)] = r
            print(f"  [N={N} {name:<35s}] "
                  f"H(cust)={r['ent_cust']:.3f}  H(row)={r['ent_row']:.3f}  "
                  f"Cost_v={r['cost_v']:.3f}  gap%={r['gap_pct']:+.2f}")
        print(f"  (N={N} total {time.time() - t0:.1f}s)")
    print("\n=== SUMMARY: lower entropy = sharper assignments. Cost_v gap = decoded quality. ===")
    for N in Ns:
        for _, name in [(None, "POOL+softassign(PIM-like)"),
                        (None, "ATTN+softassign(F-PIN-A-like)")]:
            r = results[(N, name)]
            print(f"  N={N}  {name:<35s}  H_cust={r['ent_cust']:.3f}  "
                  f"H_row={r['ent_row']:.3f}  Cost_v={r['cost_v']:.3f}  gap%={r['gap_pct']:+.2f}")


if __name__ == "__main__":
    main()
