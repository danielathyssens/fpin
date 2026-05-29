"""Inspect a trained F-PIN heatmap to test the 'weak assignment' diagnosis.

Reuses the exact eval feeding path (prep_data_fpin -> prep_test_data -> move_to -> model).
For each instance it computes, from the per-vehicle heatmap Y = softmax(logits, dim=-1):

  * vehicle differentiation : per customer, (top1 - top2) vehicle affinity. Small => the
                              M vehicles compete for the same customers (assignment ill-posed).
  * inter-vehicle similarity: mean pairwise cosine between flattened Y[m]. ~1 => vehicles
                              are near-identical (no symmetry breaking).
  * implied load balance    : assign each customer to argmax-affinity vehicle, then report
                              max vehicle load and #overloaded (load > capacity).
  * row diffuseness         : mean normalized entropy of the next-node softmax rows (1=uniform).

Usage:
  python experiments/inspect_heatmap.py --ckpt <best.pt> --N 20 --M 4 \
      --data data/test_data/cvrp/uniform/cvrp20/fc-cvrp_k4_seed213298_size1000.pt \
      --n_inst 64 [--vehicle_cond_edge_head]   (flag => E1 head; omit => legacy head)
"""
import argparse
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
from fpin.VRPModel_attn_new import VRP_Net
from fpin.utils import prep_data_fpin
from fpin.data_utils.preprocess1 import prep_test_data
from fpin.utils_all.basic_funcs import move_to


def model_args(N, M, e1):
    return dict(layers=9, depot_in_dim=4, cities_in_dim=3, fleet_in_dim=260, cities_length=N,
                max_fleet_length=M, main_dim=256, avg_pool=False, residual=True, norm=True,
                ff_hidden_dim=1024, dropout=0.0, self_pool=False, embedding_norm=True,
                weighting=True, with_loads=True, use_attn=True, regret_batches=1,
                vehicle_cond_edge_head=e1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--M", type=int, required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--n_inst", type=int, default=64)
    ap.add_argument("--vehicle_cond_edge_head", action="store_true")
    args = ap.parse_args()
    dev = torch.device("cpu")

    model = VRP_Net(**model_args(args.N, args.M, args.vehicle_cond_edge_head)).to(dev)
    sd = torch.load(args.ckpt, map_location="cpu")
    sd = sd["model"] if isinstance(sd, dict) and "model" in sd else sd
    miss, unexp = model.load_state_dict(sd, strict=False)
    print(f"loaded ckpt | missing={len(miss)} unexpected={len(unexp)}")
    model.eval()

    data_rp = torch.load(args.data, map_location="cpu", weights_only=False)[: args.n_inst]
    data_kool = prep_data_fpin(data_rp, v_max=args.M)
    test_instances, _ = prep_test_data(args.N, data_kool, type_="uniform", normed_data=True, nr_veh=args.M)

    gaps, sims, max_loads, n_over, ent = [], [], [], [], []
    with torch.no_grad():
        for ti in test_instances:
            fleet_b, depot_b, custom_b, dem_b, dists_b = move_to(ti, dev, in_train=False)
            logits, _ = model(depot_b, custom_b, fleet_b, dem_b, dists_b, sample=False, training=False)
            Y = F.softmax(logits, dim=-1)[0]            # [M, n, n]
            M, n, _ = Y.shape
            dem = dem_b.reshape(-1)[:n].cpu().numpy()    # normalized demands, depot=0

            aff = Y[:, 0, :] + Y.max(dim=1).values        # [M, n] depot-edge + best incoming
            affc = aff[:, 1:]                              # [M, N] customers only
            top = affc.sort(dim=0, descending=True).values
            gaps.append(float((top[0] - top[1]).mean()))   # vehicle differentiation

            flat = Y.reshape(M, -1)
            cos = F.cosine_similarity(flat.unsqueeze(1), flat.unsqueeze(0), dim=-1)
            sims.append(float((cos.sum() - M) / (M * (M - 1))))  # mean off-diagonal

            assign = affc.argmax(dim=0).cpu().numpy()      # customer -> vehicle
            loads = np.zeros(M)
            for j, mm in enumerate(assign):
                loads[mm] += dem[j + 1]
            max_loads.append(float(loads.max()))
            n_over.append(int((loads > 1.0 + 1e-6).sum()))

            rows = Y.reshape(-1, n)
            e = -(rows * (rows + 1e-12).log()).sum(-1) / np.log(n)
            ent.append(float(e.mean()))

    def s(x): return f"{np.mean(x):.3f} +/- {np.std(x):.3f}"
    print(f"\n=== heatmap diagnosis over {len(gaps)} instances (N={args.N}, M={args.M}) ===")
    print(f"vehicle differentiation (top1-top2 affinity, higher=better): {s(gaps)}")
    print(f"inter-vehicle cosine sim (1=identical, lower=better)       : {s(sims)}")
    print(f"max implied vehicle load (<=1 feasible)                    : {s(max_loads)}")
    print(f"# overloaded vehicles per instance (0=feasible)            : {s(n_over)}")
    print(f"row diffuseness (norm. entropy, 1=uniform, lower=peakier)  : {s(ent)}")


if __name__ == "__main__":
    main()
