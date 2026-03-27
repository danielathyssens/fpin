###The classes and functions defined in this file originate from the source code in Kaempfer & Wolff et al. (2018) and were amended for the VRP###
import torch
import torch.nn as nn
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
                 add_demand_weights=False):
        super().__init__()

        ##### ENCODER #####

        # Simple Fleet ID encoder
        self.max_fleet_length = max_fleet_length
        self.vehicle_embed = nn.Embedding(max_fleet_length, main_dim)  # e.g. embed_dim = main_dims[vehicle_idx]
        self.fleet_project = nn.Linear(260, main_dim)  # 256 target dim

        # GAT graph encoder
        self.graph_embedding = GraphEncoder(
            node_dim=cities_in_dim + 1,
            edge_dim=1,  # distance as scalar edge feature
            hidden_dim=main_dim,
            num_layers=3,  # num GNN layers
            norm=norm,
            dropout=dropout
        )

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
        self.perm_inv_net = PermInvNet(layers,
                                       [depot_in_dim,
                                        cities_in_dim,
                                        fleet_in_dim], [main_dim] * 3,
                                       [ff_hidden_dim] * 3, avg_pool=False,
                                       residual=True, norm=True,
                                       dropout=dropout, embeddings=False,
                                       self_pool=False, embedding_norm=True,
                                       use_attn=use_attn)

        # FC net comp. of single hidden layer
        self.linear_out_1 = PairwiseLinear(main_dim * 3, main_dim)

        ######## DECODER #######
        # Setup DECODER attention layer
        self.edge_attn = nn.MultiheadAttention(embed_dim=main_dim, num_heads=4, batch_first=True)

        # linear projection (DECODER)
        self.edge_proj = nn.Sequential(
            nn.Linear((2 * main_dim) + 1, 2 * main_dim),  # match the conv setup
            nn.ReLU(),
            nn.Linear(2 * main_dim, main_dim)
        )

        # LOAD
        self.with_loads = with_loads

        # HELPERS
        self.register_buffer("_eye_cache", torch.empty(0), persistent=False)
        self._eye_cache_n = None


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
        fleet_emb = self.vehicle_embed(vehicle_ids)  # (B, M, D)
        fleet_emb_ = torch.cat([veh_features, fleet_emb], dim=-1)
        fleet_embedding = self.fleet_project(fleet_emb_)  # [B, M, 256]

        ## GRAPH
        node_feats = torch.cat((depot, torch.cat((customers, dists[:, 1:, :1]), dim=2)), dim=1)
        edge_feats = weights_dists if weights_dists is not None else dists.unsqueeze(-1)
        graph_emb = self.graph_embedding(node_feats, edge_feats)  # [B, N, main_dim]
        # Split back into sets
        depot_embedding = graph_emb[:, :1, :]  # [B, 1, D]
        customers_embedding = graph_emb[:, 1:, :]  # [B, num_customers, D]

        ### PermInvPoolNet ###
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

        ######### OUTPUT CONSTRUCTION ########
        # fleet: [batch_size, M, d_model] → queries
        # base_nodes: [batch_size, N×N, 2×d_model] → keys/values
        cities = torch.cat((depot, customers), dim=1)  # (b, n+1, d_model)
        main_dim = cities.size(2)
        # Create base_nodes by combining all pairs of cities (including depot)
        base_nodes = torch.cat(
            (cities.unsqueeze(2).expand(b, n, n, main_dim), cities.unsqueeze(1).expand(b, n, n, main_dim)),
            dim=3)  # (b, n, n, 2*main_dim)
        base_nodes = base_nodes.view(b, n * n, 2 * main_dim)  # (b, N×N, 2*main_dim)

        # Add pairwise distances
        rel_dist = dists.view(b, n * n, 1)  # (b, N×N, 1)
        base_nodes_with_dist = torch.cat([base_nodes, rel_dist], dim=2)  # (b, N×N, 2*main_dim + 1)

        # Project into key/value space
        edge_kv = self.edge_proj(base_nodes_with_dist)  # (b, N×N, d_model)

        # Prepare fleet queries
        # Vehicles attend to nodes
        veh_ctx, _ = self.veh_to_node_attn(
            query=fleet,  # [B,M,D]
            key=cities,  # [B,N,D]
            value=cities,  # [B,N,D]
        )  # -> [B,M,D]

        veh_ctx = self.veh_ctx_ln(veh_ctx)
        fleet_cond = self.veh_fuse(torch.cat([fleet, veh_ctx], dim=-1))  # [B,M,D]

        temperature = 0.5
        fleet_q_scaled = fleet_cond / temperature  # (b, M, d_model)

        # --- Explicit edge logits on (i, j) pairs
        edge_logits = torch.matmul(
            fleet_q_scaled, edge_kv.transpose(1, 2)
        ) / (fleet_q_scaled.size(-1) ** 0.5)  # [B, M, N*N]

        edge_logits = edge_logits.view(b, m, n, n)

        # Mask self-loops except depot
        if self._eye_cache_n != n or self._eye_cache.numel() == 0 or self._eye_cache.device != depot.device:
            eye = torch.eye(n, device=depot.device, dtype=torch.bool)
            eye[0, 0] = False
            self._eye_cache = eye
            self._eye_cache_n = n
        edge_logits = edge_logits.masked_fill(self._eye_cache[None, None, :, :], -30.0)

        # Probabilities for decoding (optional)
        edge_probs_for_decode = torch.sigmoid(edge_logits)  # [B, M, N, N]

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
