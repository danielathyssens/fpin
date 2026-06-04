###The classes and functions defined in this file originate from the source code in Kaempfer & Wolff et al. (2018) and were amended for the VRP###
import torch
import torch.nn as nn
import torch.nn.functional as F
from fpin.nets.perm_inv_net import PermInvNet
from fpin.utils_all.net_funcs import PairwiseLinear
from fpin.nets.encoder import GraphEncoder

class VRP_Net(nn.Module):
    '''Complete Net incl. Softassign producing final routes'''

    def __init__(self, layers, depot_in_dim, cities_in_dim, fleet_in_dim,
                 cities_length, max_fleet_length, main_dim,
                 avg_pool, residual, norm, ff_hidden_dim, dropout,
                 self_pool, embedding_norm, weighting, with_loads,
                 use_attn, regret_batches, seed_mode="fast", nr_seeds=8, seed_sigma=0.5,
                 add_demand_weights=False, vehicle_cond_edge_head=True,
                 sinkhorn_assignment=False, sinkhorn_iters=3,
                 joint_customer_norm=False,
                 softassign_head=False, softassign_layers=3,
                 softassign_log_domain=False,
                 global_edge_softmax=False,
                 vcount_aux_head=False,
                 use_vehicle_id_embedding=True,
                 use_graph_encoder=True,
                 use_perm_inv_encoder=True,
                 learnable_temperature=True,
                 initial_log_temperature=0.0):
        super().__init__()

        ##### ENCODER #####

        # Simple Fleet ID encoder
        self.max_fleet_length = max_fleet_length
        self.use_vehicle_id_embedding = use_vehicle_id_embedding
        self.use_graph_encoder = use_graph_encoder
        self.use_perm_inv_encoder = use_perm_inv_encoder
        self.vehicle_embed = nn.Embedding(max_fleet_length, main_dim)  # e.g. embed_dim = main_dims[vehicle_idx]
        self.fleet_project = nn.Linear(260, main_dim)  # 256 target dim

        # Graph encoder can be ablated to isolate whether message passing
        # helps or hurts the downstream routing heatmap.
        if use_graph_encoder:
            self.graph_embedding = GraphEncoder(
                node_dim=cities_in_dim + 1,
                edge_dim=1,  # distance as scalar edge feature
                hidden_dim=main_dim,
                num_layers=3,  # num GNN layers
                norm=norm,
                dropout=dropout
            )
        else:
            self.graph_embedding = None
            self.raw_node_proj = nn.Linear(cities_in_dim + 1, main_dim)

        # vehicle context (attention)
        self.veh_to_node_attn = nn.MultiheadAttention(
            embed_dim=main_dim, num_heads=4, batch_first=True
        )
        self.veh_ctx_ln = nn.LayerNorm(main_dim)

        # mix vehicle + context (MLP)
        self.veh_fuse = nn.Sequential(
            nn.Linear(2 * main_dim, main_dim),
            nn.ReLU(),
            nn.Linear(main_dim, main_dim),
        )

        # OPTIONAL weighting FOR POOLING
        if weighting:
            self.distance_to_weights = DistancesCapaToWeights(self_pool,
                                                              main_dim,
                                                              add_demand_weights=add_demand_weights)
        else:
            self.distance_to_weights = lambda dists, demands: None

        # init perm inv net
        if use_perm_inv_encoder:
            self.perm_inv_net = PermInvNet(layers,
                                           [depot_in_dim,
                                            cities_in_dim,
                                            fleet_in_dim], [main_dim] * 3,
                                           [ff_hidden_dim] * 3, avg_pool=False,
                                           residual=True, norm=True,
                                           dropout=dropout, embeddings=False,
                                           self_pool=False, embedding_norm=True,
                                           use_attn=use_attn)
        else:
            self.perm_inv_net = None

        # FC net comp. of single hidden layer
        self.linear_out_1 = PairwiseLinear(main_dim * 3, main_dim)

        ######## DECODER #######
        # Setup DECODER attention layer
        self.edge_attn = nn.MultiheadAttention(embed_dim=main_dim, num_heads=4, batch_first=True)

        # linear projection (DECODER) -- legacy bilinear head (vehicle-agnostic edge_kv,
        # vehicle enters only via final dot product). Kept for backward-compat / ablation.
        self.edge_proj = nn.Sequential(
            nn.Linear((2 * main_dim) + 1, 2 * main_dim),  # match the conv setup
            nn.ReLU(),
            nn.Linear(2 * main_dim, main_dim)
        )

        # E1: vehicle-conditioned edge head -- MLP over [vehicle, from-node, to-node, dist],
        # i.e. the vehicle is mixed into the edge representation BEFORE scoring (PIM-style,
        # richer per-vehicle differentiation than the bilinear head). E2: learnable decode
        # temperature. Both only instantiated when the new head is active, so legacy
        # checkpoints (bilinear head) still load with strict=True.
        self.vehicle_cond_edge_head = vehicle_cond_edge_head
        if vehicle_cond_edge_head:
            self.edge_mlp = nn.Sequential(
                nn.Linear(3 * main_dim + 1, main_dim), nn.ReLU(),
                nn.Linear(main_dim, main_dim), nn.ReLU(),
            )
            self.edge_score = nn.Linear(main_dim, 1)
            self.log_temperature = nn.Parameter(
                torch.full((1,), float(initial_log_temperature)),
                requires_grad=learnable_temperature,
            )  # temp = exp(.)

        # E3: Sinkhorn assignment head. Computes a soft customer->vehicle assignment
        # (Sinkhorn-normalized so each customer's vehicle distribution is a simplex), and
        # biases the per-vehicle edge logits toward each vehicle's assigned customers. This
        # structurally enforces the amortized-clustering / bin-packing constraint (each
        # customer served by one vehicle) and is trained end-to-end via the existing
        # next-node loss (no extra labels). Off by default -> retrain-1 (E1) unaffected.
        self.sinkhorn_assignment = sinkhorn_assignment
        self.sinkhorn_iters = sinkhorn_iters
        if sinkhorn_assignment:
            self.assign_q = nn.Linear(main_dim, main_dim)
            self.assign_k = nn.Linear(main_dim, main_dim)
            self.assign_bias_weight = nn.Parameter(torch.ones(1))

        # F-PIN-S: per-customer JOINT (vehicle, next-node) softmax replaces the
        # per-vehicle row softmax for customer rows. Matches PIM's softassign
        # structural prior (each customer is left by exactly one (vehicle, j))
        # in ONE softmax step instead of iterative Sinkhorn -> well-conditioned
        # gradients (single LogSumExp), no doubly-stochastic-fixed-point pathology
        # (cf. Mena et al. 2018 on Gumbel-Sinkhorn gradients; Cuturi 2013 OT).
        # Depot row unchanged (per-vehicle softmax over j).
        self.joint_customer_norm = joint_customer_norm

        # F-PIN-A: full PIM 2022 MTSPSoftassign at the head. Ported faithfully
        # from models/PIMold/src/utils_all/softassign.py. Iteratively normalizes
        # depot row (per-vehicle softmax over j) + customer rows (joint over
        # (m, j)), alternating with transpose to enforce both out-flow AND
        # in-flow constraints (each customer LEFT and ENTERED exactly once
        # across all vehicles). Structurally prevents the model from emitting
        # heatmaps that decode to > M routes -- the lever for matching PIM's
        # 0% fleet violation. Toy v2 (this repo) shows ATTN+MTSPSoftassign
        # decodes to 0.10% gap-vs-optimum at N=6 (best of 6 configs tested).
        self.softassign_head = softassign_head
        self.global_edge_softmax = global_edge_softmax
        self.softassign_layers = max(1, softassign_layers)
        # F-PIN-AB: log-domain Sinkhorn for the softassign head. Avoids the
        # numerical pathology of the multiplicative form (exp + clamp + divide)
        # at sharp distributions -- exactly the regime an attention encoder
        # drives toward (toy: H_cust ATTN=1.89 < POOL=2.14). Log-domain uses
        # logsumexp throughout, gradient stable at all entropy levels.
        # Mathematically equivalent fixed point; semantically identical output.
        # Reference: Peyre & Cuturi 2019, Computational Optimal Transport,
        # Sec. 4.2 (stable Sinkhorn iterations).
        self.softassign_log_domain = softassign_log_domain

        # F-PIN-C: auxiliary "vehicle count" prediction head. A tiny MLP that
        # consumes the (mean) pooled global feature and predicts the optimal
        # number of vehicles used by Y*. Loss-side: MSE against target count.
        # Forces the encoder to be fleet-aware at the representation level,
        # complementing F-PIN-A's structural-at-output prior. Returned as the
        # 3rd output of forward() when enabled; backwards-compatible (None when
        # disabled, ignored by the existing loss path).
        self.vcount_aux_head = vcount_aux_head
        if vcount_aux_head:
            self.vcount_mlp = nn.Sequential(
                nn.Linear(main_dim, main_dim // 2),
                nn.ReLU(),
                nn.Linear(main_dim // 2, 1),
            )

        # LOAD
        self.with_loads = with_loads

        # HELPERS
        self.register_buffer("_eye_cache", torch.empty(0), persistent=False)
        self._eye_cache_n = None
        self._vcount_pred = None    # set by forward() when vcount_aux_head=True

    def get_aux_loss(self, target):
        """F-PIN-C: auxiliary MSE loss on predicted vs target #vehicles used.
        Target #vehicles = count of vehicles with non-empty route in Y*.
        Returns None when vcount_aux_head is disabled OR no forward has run."""
        if not self.vcount_aux_head or self._vcount_pred is None:
            return None
        # target [B, M, n, n]; active vehicles = m where target[b, m, :, :].sum() > 0
        if target.is_sparse:
            t_dense = target.to_dense()
        else:
            t_dense = target
        with torch.no_grad():
            target_count = (t_dense.sum(dim=(-2, -1)) > 0).sum(dim=-1).float()
        return torch.nn.functional.mse_loss(self._vcount_pred, target_count)


    def forward(self, depot, customers, fleet, demands, dists, sample=False, training=True):

        # RETURNS PROBABILITIES
        # fleet:          Float(batch x fleet_length x fleet_in_dim)
        # depot:          Float(batch x 1 x cities_in_dim)
        # customers:      Float(batch x nr_customers x cities_in_dim)
        # dists:          Float(batch x from_city x to_city) incl. depot
        # demands:        Float(batch x 1 x (cities_in_dim+1))
        # -----------------------------------------------------------------------------------------------------------
        # res:            Float(batch x fleet x from_city x to_city)
        # pred_loads:     Float(batch x fleet) or Float(batch x 2 x fleet)
        # pred_demands:   Float(batch x fleet x from_city x to_city) or Float(batch x 2 x fleet x from_city x to_city)
        # sample_path_b0: Long(List of Lists)

        b = customers.size(0)  # batchsize
        n = customers.size(1) + 1  # nr. of customer nodes + 1 for depot
        m = fleet.size(1)  # nr. of vehicles
        m_max = self.max_fleet_length

        # PRE-COMPUTED DISTANCE WEIGHT
        # Float(batch x cities_length x cities_length x main_dim):
        weights_dists = self.distance_to_weights(dists, demands)

        # ENCODING

        ## FLEET
        vehicle_ids = fleet[:, :, 0].long()  # shape (B, M)
        veh_features = fleet[:, :, 1:]  # drop id column
        if self.use_vehicle_id_embedding:
            fleet_emb = self.vehicle_embed(vehicle_ids)  # (B, M, D)
        else:
            fleet_emb = torch.zeros(
                b, m, self.vehicle_embed.embedding_dim, device=fleet.device, dtype=fleet.dtype
            )
        fleet_emb_ = torch.cat([veh_features, fleet_emb], dim=-1)
        fleet_embedding = self.fleet_project(fleet_emb_)  # [B, M, 256]

        ## GRAPH
        node_feats = torch.cat((depot, torch.cat((customers, dists[:, 1:, :1]), dim=2)), dim=1)
        edge_feats = weights_dists if weights_dists is not None else dists.unsqueeze(-1)
        if self.use_graph_encoder:
            graph_emb = self.graph_embedding(node_feats, edge_feats)  # [B, N, main_dim]
        else:
            graph_emb = self.raw_node_proj(node_feats)  # [B, N, main_dim]
        # Split back into sets
        depot_embedding = graph_emb[:, :1, :]  # [B, 1, D]
        customers_embedding = graph_emb[:, 1:, :]  # [B, num_customers, D]

        # F-PIN-C: auxiliary fleet-count prediction from pooled customer feature.
        # Stored as attribute; loss is computed via model.get_aux_loss(target).
        if self.vcount_aux_head:
            global_feat = customers_embedding.mean(dim=1)             # [B, D]
            self._vcount_pred = self.vcount_mlp(global_feat).squeeze(-1)  # [B]
        else:
            self._vcount_pred = None

        ### PermInvPoolNet ###
        if self.use_perm_inv_encoder:
            if weights_dists is None:
                sets_weights = None
            else:
                depots_weights_dists = [weights_dists[:, :1, :1, :], weights_dists[:, :1, 1:, :], None]
                others_weights_dists = [weights_dists[:, 1:, :1, :], weights_dists[:, 1:, 1:, :], None]
                fleet_weights_dists = None
                sets_weights = [depots_weights_dists, others_weights_dists, fleet_weights_dists]

            ########## RUN PERM.Inv.NET --> Sets! ##########
            depot, customers, fleet = self.perm_inv_net([depot_embedding,
                                                         customers_embedding,
                                                         fleet_embedding],
                                                        weights=sets_weights)
        else:
            depot, customers, fleet = depot_embedding, customers_embedding, fleet_embedding

        ######### OUTPUT CONSTRUCTION ########
        cities = torch.cat((depot, customers), dim=1)  # (b, n, d_model)
        main_dim = cities.size(2)

        # Vehicle queries: vehicles attend to nodes, then fuse (shared by both heads)
        veh_ctx, _ = self.veh_to_node_attn(
            query=fleet, key=cities, value=cities,
        )  # -> [B,M,D]
        veh_ctx = self.veh_ctx_ln(veh_ctx)
        fleet_cond = self.veh_fuse(torch.cat([fleet, veh_ctx], dim=-1))  # [B,M,D]

        if self.vehicle_cond_edge_head:
            # E1: vehicle-conditioned edge MLP over [vehicle, h_i, h_j, dist_ij], scored per
            # vehicle. The vehicle is mixed into the edge representation BEFORE scoring, so
            # each vehicle gets a genuinely distinct, richer transition map (vs. the legacy
            # bilinear head where the edge encoding is vehicle-agnostic). The per-vehicle
            # loop keeps peak memory at O(b*n*n*d) and frees it each step.
            h_i = cities.unsqueeze(2).expand(b, n, n, main_dim)  # from-node
            h_j = cities.unsqueeze(1).expand(b, n, n, main_dim)  # to-node
            dist_e = dists.unsqueeze(-1)  # [b, n, n, 1]
            scores = []
            for mm in range(m):
                veh_m = fleet_cond[:, mm].view(b, 1, 1, main_dim).expand(b, n, n, main_dim)
                edge_in = torch.cat([veh_m, h_i, h_j, dist_e], dim=-1)  # [b, n, n, 3d+1]
                e = self.edge_mlp(edge_in)                              # [b, n, n, d]
                scores.append(self.edge_score(e).squeeze(-1))          # [b, n, n]
            edge_logits = torch.stack(scores, dim=1)                    # [b, m, n, n]
            edge_logits = edge_logits / torch.exp(self.log_temperature)  # E2: learnable temp
        else:
            # Legacy bilinear head: vehicle-agnostic edge_kv, vehicle enters via final dot.
            base_nodes = torch.cat(
                (cities.unsqueeze(2).expand(b, n, n, main_dim), cities.unsqueeze(1).expand(b, n, n, main_dim)),
                dim=3).view(b, n * n, 2 * main_dim)
            rel_dist = dists.view(b, n * n, 1)
            edge_kv = self.edge_proj(torch.cat([base_nodes, rel_dist], dim=2))  # (b, N*N, d)
            fleet_q_scaled = fleet_cond / 0.5
            edge_logits = torch.matmul(
                fleet_q_scaled, edge_kv.transpose(1, 2)
            ) / (main_dim ** 0.5)
            edge_logits = edge_logits.view(b, m, n, n)

        # E3: Sinkhorn customer->vehicle assignment -> bias incoming edges so each vehicle
        # focuses on its assigned customers (enforces one-vehicle-per-customer structure).
        if self.sinkhorn_assignment:
            q = self.assign_q(fleet_cond)                      # [b, m, d]
            k = self.assign_k(cities[:, 1:, :])                # [b, n-1, d]  customers only
            S = torch.matmul(q, k.transpose(1, 2)) / (main_dim ** 0.5)  # [b, m, n-1]
            logP = S
            for _ in range(self.sinkhorn_iters):
                logP = logP - torch.logsumexp(logP, dim=1, keepdim=True)  # customer -> simplex over vehicles
                logP = logP - torch.logsumexp(logP, dim=2, keepdim=True)  # vehicle  -> over customers
            logP = logP - torch.logsumexp(logP, dim=1, keepdim=True)      # final: log P(vehicle | customer)
            bias = torch.zeros_like(edge_logits)
            bias[:, :, :, 1:] = (self.assign_bias_weight * logP).unsqueeze(2)  # broadcast over from-node
            edge_logits = edge_logits + bias

        # Mask self-loops except depot
        if self._eye_cache_n != n or self._eye_cache.numel() == 0 or self._eye_cache.device != depot.device:
            eye = torch.eye(n, device=depot.device, dtype=torch.bool)
            eye[0, 0] = False
            self._eye_cache = eye
            self._eye_cache_n = n
        edge_logits = edge_logits.masked_fill(self._eye_cache[None, None, :, :], -30.0)

        # Normalization step.
        # F-PIN-A (softassign_head): apply PIM 2022's MTSPSoftassign at the head
        # to structurally enforce both out-flow (each customer left once across
        # all vehicles) and in-flow (each customer entered once) constraints.
        # Returns pre-normalized log-probs; loss must consume them directly
        # (loss_cfg.softassign_head=True skips the redundant log_softmax).
        # F-PIN-G (global_edge_softmax): PIM-2022's output normalization -- ONE softmax
        # over all n*n edges per vehicle, so each vehicle's heatmap sums to 1 globally and
        # concentrates mass on the ~n edges of a clean tour (maximally committed, vs F-PIN's
        # diffuse row/softassign mass). Keeps F-PIN's attention encoder + vehicle conditioning,
        # so it is NOT a revert to OG PIM. Diagonal masked to forbid self-loops (depot 0,0 kept
        # as a harmless no-op slot). Returns log-probs -> set loss_cfg.softassign_head=True.
        if self.global_edge_softmax:
            b_, m_, n_, _ = edge_logits.shape
            eye = torch.eye(n_, device=edge_logits.device, dtype=torch.bool)
            eye[0, 0] = False
            # large finite negative (not -inf): keeps log_probs finite so the loss's
            # -sum(target*log_probs) never hits 0*-inf=NaN on the masked diagonal.
            masked = edge_logits.masked_fill(eye.view(1, 1, n_, n_), -1e9)
            log_probs = F.log_softmax(masked.reshape(b_, m_, n_ * n_), dim=-1).reshape(b_, m_, n_, n_)
            edge_probs_for_decode = log_probs.exp()
            return log_probs, edge_probs_for_decode

        if self.softassign_head:
            if self.softassign_log_domain:
                # F-PIN-AB: log-domain Sinkhorn. Same fixed point as the
                # multiplicative form, but uses logsumexp throughout instead
                # of exp/clamp/divide -- gradient stable at sharp output
                # distributions (which is exactly what ATTN encoders produce,
                # toy: H_cust 1.89 vs POOL 2.14). No clamp_min(eps) hacks.
                log_out = edge_logits
                for _ in range(self.softassign_layers):
                    # depot row (i=0): per-vehicle softmax over j
                    depot_log = log_out[:, :, :1, :]
                    depot_log = depot_log - torch.logsumexp(depot_log, dim=-1, keepdim=True)
                    # customer rows (i>0): joint over (m, j)
                    cust_log = log_out[:, :, 1:, :]
                    cust_log = cust_log - torch.logsumexp(cust_log, dim=(1, 3), keepdim=True)
                    log_out = torch.cat([depot_log, cust_log], dim=2)
                    log_out = log_out.transpose(2, 3)
                if self.softassign_layers % 2 == 1:
                    log_out = log_out.transpose(2, 3)
                log_probs = log_out
                edge_probs_for_decode = log_probs.exp()
                return log_probs, edge_probs_for_decode
            # Original multiplicative form (kept for A/B against AB).
            eps = 1e-8
            out = torch.exp(edge_logits - edge_logits.amax(dim=(-2, -1), keepdim=True))
            for _ in range(self.softassign_layers):
                out = out.clamp_min(eps)
                # depot row (i=0): per-vehicle softmax over j -> out-degree(depot,m)=1
                depot = out[:, :, :1, :]
                depot = depot / depot.sum(dim=-1, keepdim=True).clamp_min(eps)
                # customer rows (i>0): joint over (m, j) -> each customer left once
                cust = out[:, :, 1:, :]
                cust_sum = cust.sum(dim=(1, 3), keepdim=True).clamp_min(eps)
                cust = cust / cust_sum
                out = torch.cat([depot, cust], dim=2)
                # transpose to enforce the in-flow side symmetrically
                out = out.transpose(2, 3)
            # if odd #iters -> currently transposed; flip back to (i_from, j_to)
            if self.softassign_layers % 2 == 1:
                out = out.transpose(2, 3)
            log_probs = (out + eps).log()
            edge_probs_for_decode = out
            return log_probs, edge_probs_for_decode

        # Default (F-PIN-original): per-(vehicle, source) softmax over destination j.
        # joint_customer_norm = True (F-PIN-S): customer rows use a per-customer
        # joint softmax over (m, j). Depot row keeps per-vehicle softmax over j.
        if self.joint_customer_norm:
            depot_log = F.log_softmax(edge_logits[:, :, 0:1, :], dim=-1)        # [B,M,1,n]
            cust = edge_logits[:, :, 1:, :]                                     # [B,M,n-1,n]
            B_, M_, Nm1, N_ = cust.shape
            cust = cust.permute(0, 2, 1, 3).reshape(B_, Nm1, M_ * N_)           # [B,n-1,M*n]
            cust = F.log_softmax(cust, dim=-1)
            cust = cust.reshape(B_, Nm1, M_, N_).permute(0, 2, 1, 3)             # [B,M,n-1,n]
            log_probs = torch.cat([depot_log, cust], dim=2)
            edge_probs_for_decode = log_probs.exp()
            # Loss path consumes probs via .log(); pass back probs that, when
            # .log()'d, recover log_probs.
            return log_probs, edge_probs_for_decode
        else:
            edge_probs_for_decode = torch.softmax(edge_logits, dim=-1)
            return edge_logits, edge_probs_for_decode


class DistancesCapaToWeights(nn.Module):
    def __init__(self, self_pool, main_dim, add_demand_weights):
        super().__init__()

        self.temperature_dists = nn.Parameter(torch.ones(main_dim), requires_grad=True)
        self.gamma_dists = nn.Parameter(torch.ones(main_dim), requires_grad=True)
        self.beta_d_q = nn.Parameter(torch.zeros(main_dim), requires_grad=True)
        self.temperature_q = nn.Parameter(torch.ones(main_dim), requires_grad=True)
        self.gamma_q = nn.Parameter(torch.ones(main_dim), requires_grad=True)

        self.self_pool = self_pool
        self.add_demand_weights = add_demand_weights
        self.main_dim = main_dim

    def forward(self, dists, demands):
        # demand: Float(b x 1 x n)
        # dists:  Float(b x n x n)
        # fleet:  Float(b x m x 4)
        # -------------------------------------
        # res:    Float(b x n x n x main_dim)

        b, n, _ = dists.size()

        ############ DISTS ############
        # normalize distances
        dists_mean = dists.sum(dim=1, keepdim=True).sum(dim=2, keepdim=True) / (n * n)
        dists = dists / dists_mean
        if self.add_demand_weights:
            ############ DEMANDS ############
            # encode demand weights as sum of 2 customer demands
            # Float(b x 1 x n x n)
            demands = demands[..., None] + demands[..., None, :]
            demands = demands.squeeze(1)
        else:
            demands = None

        # apply temperature
        # below: Float(b x n x n x main_dim)
        dists = dists.unsqueeze(3).expand(b, n, n, self.main_dim)
        dists = dists * self.temperature_dists.unsqueeze(0).unsqueeze(1).unsqueeze(2)
        if self.add_demand_weights:
            # below: Float(b x n x n x main_dim)
            demands = demands.unsqueeze(3).expand(b, n, n, self.main_dim)
            demands = demands * self.temperature_q.unsqueeze(0).unsqueeze(1).unsqueeze(2)

        # calc. weights
        weights_dists = 1 / dists.exp()  # Float(b x n x n x main_dim)
        weights_q = demands if self.add_demand_weights else None

        # remove self pool weight
        if not self.self_pool:
            n_range = torch.arange(n).long().to(device=weights_dists.device)
            # Float(b x n x n x main_dim)
            idx_dist = n_range.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand(b, n, 1, self.main_dim)
            weights_dists.scatter_(dim=2, index=idx_dist, value=0)
            if self.add_demand_weights:
                # Float(b x n x n x main_dim)
                idx_q = n_range.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand(b, n, 1, self.main_dim)
                weights_q.scatter_(dim=2, index=idx_q, value=0)

        # COMBINE weight_q and weights_dist
        # Float(b x m x n x main_dim)
        if self.add_demand_weights:
            weights_ = weights_dists * self.gamma_dists.unsqueeze(0).unsqueeze(1).unsqueeze(2) + \
                       weights_q * self.gamma_q.unsqueeze(0).unsqueeze(1).unsqueeze(2) + \
                       self.beta_d_q.unsqueeze(0).unsqueeze(1).unsqueeze(2)
        else:
            weights_ = weights_dists * self.gamma_dists.unsqueeze(0).unsqueeze(1).unsqueeze(2) + \
                       self.beta_d_q.unsqueeze(0).unsqueeze(1).unsqueeze(2)

        return weights_
