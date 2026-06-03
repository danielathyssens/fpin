import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
from utils_all.perm_inv_net_orig import PermInvNet
from utils_all.get_path import load_estimate, targ_as_lst
from utils_all.net_funcs import PairwiseLinear, FeedForwardLayer, LayerNorm
from utils_all.basic_funcs import to_variable, zeros, chunk_at, normalize_dims
from utils_all.softassign import MTSPSoftassign


class DistancesCapaToWeights(nn.Module):
    def __init__(self, self_pool, main_dim):
        super().__init__()

        # ,dev
        self.temperature_dists = nn.Parameter(torch.ones(main_dim), requires_grad=True)
        self.gamma_dists = nn.Parameter(torch.ones(main_dim), requires_grad=True)
        # self.beta_dists = nn.Parameter(torch.zeros(main_dim), requires_grad=True)
        self.beta_d_q = nn.Parameter(torch.zeros(main_dim), requires_grad=True)
        self.temperature_q = nn.Parameter(torch.ones(main_dim), requires_grad=True)
        self.gamma_q = nn.Parameter(torch.ones(main_dim), requires_grad=True)
        # self.beta_q = nn.Parameter(torch.zeros(main_dim), requires_grad=True)

        self.self_pool = self_pool
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

        ############ DEMANDS ############
        # encode demand weights as sum of 2 customer demands
        # Float(b x 1 x n x n)
        demands = demands[..., None] + demands[..., None, :]
        demands = demands.squeeze(1)

        # apply temperature 
        # below: Float(b x n x n x main_dim)
        dists = dists.unsqueeze(3).expand(b, n, n, self.main_dim)
        dists = dists * self.temperature_dists.unsqueeze(0).unsqueeze(1).unsqueeze(2)
        # below: Float(b x n x n x main_dim)
        demands = demands.unsqueeze(3).expand(b, n, n, self.main_dim)
        demands = demands * self.temperature_q.unsqueeze(0).unsqueeze(1).unsqueeze(2)

        # calc. weights
        weights_dists = 1 / dists.exp()  # Float(b x n x n x main_dim)
        weights_q = demands
        # print('weights_q.device',weights_q.device)
        # weights_q = 1 / demands.exp()   # Float(b x n x n x main_dim)

        # remove self pool weight
        if not self.self_pool:
            n_range = torch.arange(n, device=dists.device).long()
            # Float(b x n x n x main_dim)
            idx_dist = n_range.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand(b, n, 1, self.main_dim)
            weights_dists.scatter_(dim=2, index=idx_dist, value=0)
            # Float(b x n x n x main_dim)
            idx_q = n_range.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand(b, n, 1, self.main_dim)
            weights_q.scatter_(dim=2, index=idx_q, value=0)

        # shift
        # Float(b x n x n x main_dim)
        # weights_dists = weights_dists *self.gamma_dists.unsqueeze(0).unsqueeze(1).unsqueeze(2) + \
        #    self.beta_dists.unsqueeze(0).unsqueeze(1).unsqueeze(2)

        # COMBINE weight_q and weights_dist
        # Float(b x m x n x main_dim)
        weights_d_q = weights_dists * self.gamma_dists.unsqueeze(0).unsqueeze(1).unsqueeze(2) + \
                      weights_q * self.gamma_q.unsqueeze(0).unsqueeze(1).unsqueeze(2) + \
                      self.beta_d_q.unsqueeze(0).unsqueeze(1).unsqueeze(2)

        # Float(b x m x n x main_dim)
        # weights_q = weights_q * self.gamma_q.unsqueeze(0).unsqueeze(1).unsqueeze(2) + \
        #    self.beta_q.unsqueeze(0).unsqueeze(1).unsqueeze(2)

        return weights_d_q


class VRP_Net(nn.Module):
    '''Complete Net incl. Softassign producing final routes'''

    def __init__(self, layers, depot_in_dim, cities_in_dim, fleet_in_dim, main_dim,
                 avg_pool, residual, norm, ff_hidden_dim, dropout,
                 self_pool, embedding_norm, softassign_layers, weighting,
                 with_loads, attn_dotproduct, memory_efficient):
        super().__init__()

        # weighting...
        if weighting:
            self.distance_to_weights = DistancesCapaToWeights(self_pool,
                                                              main_dim)
        else:
            self.distance_to_weights = lambda dists: None

        # init perm inv net
        self.perm_inv_net = PermInvNet(layers,
                                       [depot_in_dim,
                                        cities_in_dim,
                                        fleet_in_dim], [main_dim] * 3,
                                       [ff_hidden_dim] * 3, avg_pool=False,
                                       residual=True, norm=True,
                                       dropout=dropout, embeddings=True,
                                       self_pool=False, embedding_norm=True)

        # FC net comp. of single hidden layer
        self.linear_out_1 = PairwiseLinear(main_dim * 3, main_dim)
        # self.linear_out_2 = PairwiseLinear(main_dim,1)

        # SOFTASSIGN
        self.softassign = MTSPSoftassign(softassign_layers)

        # LOAD
        self.with_loads = with_loads

        # OUTPUT construction -->loop and stack==>True OR --> convolve all==>False
        self.memory_efficient = memory_efficient

        # "attn dot product" with Q=vehicles and K=Base Nodes
        self.attn_dotproduct = attn_dotproduct

    def forward(self, depot, customers, fleet, demands, dists):

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

        # WEIGHTS FOR WEIGHTED LOO POOLING
        # # None | Float(batch x cities_length x cities_length x main_dim):
        weights_dists = self.distance_to_weights(dists, demands)

        if weights_dists is None:
            sets_weights = None

        else:
            depots_weights_dists = [weights_dists[:, :1, :1, :], weights_dists[:, :1, 1:, :], None]
            others_weights_dists = [weights_dists[:, 1:, :1, :], weights_dists[:, 1:, 1:, :], None]
            fleet_weights_dists = None
            sets_weights = [depots_weights_dists, others_weights_dists, fleet_weights_dists]

        ########## RUN PERM.Inv.NET --> Sets! ##########
        depot, customers, fleet = self.perm_inv_net([depot, customers, fleet], weights=sets_weights)

        ######### COMBINE SETS TO TENSOR --> SCORES! ###########
        cities = torch.cat((depot, customers), dim=1)  # Float(batch x cities_length +1 x main_dim)
        main_dim = cities.size(2)

        if self.attn_dotproduct:
            # fleet:  Float(batch x m x main_dim)
            # base_nodes: Float(batch x n+1*n+1 x (main_dim*2))

            # from_cities = cities.unsqueeze(2).expand(b, n, n, main_dim)     # Float(batch x cities_length x main_dim)
            # to_cities = cities.unsqueeze(1).expand_as(from_cities)
            base_nodes = torch.cat(
                (cities.unsqueeze(2).expand(b, n, n, main_dim), cities.unsqueeze(1).expand(b, n, n, main_dim)),
                dim=3)  # Float(batch x cities_length x cities_length x (main_dim*2))
            base_nodes = base_nodes.view(base_nodes.size(0), n * n,
                                         main_dim * 2)  # Float(batch x (cities_length*cities_length) x (main_dim*2))
            res = []
            for i, vehicle in enumerate(chunk_at(fleet, dim=1, squeeze=False)):  # Float(batch x 1 x main_dim)
                vehicle_exp = vehicle.expand(b, n * n,
                                             vehicle.size(2))  # Float(batch x (cities_length*cities_length) x main_dim)
                nodes = torch.cat((vehicle_exp, base_nodes), dim=2)
                nodes = self.linear_out_1(nodes)  # Float(batch x (cities_length*cities_length) x main_dim)
                nodes = F.relu(nodes)  # Float(batch x (cities_length*cities_length) x main_dim)
                # nodes = self.linear_out_2(nodes)                                 # Float(batch x (cities_length*cities_length) x 1)

                ##### Replace linear_out_2 with PART-ATTN #####
                logits = torch.matmul(vehicle,
                                      nodes.transpose(1, 2))  # Float(batch x 1 x (cities_length*cities_length))
                logits = logits.squeeze(-2) / math.sqrt(nodes.size(1))  # Float(batch x (cities_length*cities_length))
                # scores = torch.softmax(logits / 1.0, dim=-1)  # Float(batch x (cities_length*cities_length))
                scores = logits.unsqueeze(-1)
                scores = scores.view(b, n, n)  # Float(batch x cities_length x cities_length)
                res.append(scores)

            # SCORES
            output = torch.stack(res, dim=1)  # Float(batch x fleet_length x cities_length x cities_length)

        else:

            if self.memory_efficient:
                from_cities = cities.unsqueeze(2).expand(b, n, n, main_dim)  # Float(batch x cities_length x main_dim)
                to_cities = cities.unsqueeze(1).expand_as(from_cities)
                base_nodes = torch.cat((from_cities, to_cities),
                                       dim=3)  # Float(batch x cities_length x cities_length x (main_dim*2))
                base_nodes = base_nodes.view(base_nodes.size(0), n * n,
                                             main_dim * 2)  # Float(batch x (cities_length*cities_length) x (main_dim*2))

                res = []
                for i, vehicle in enumerate(chunk_at(fleet, dim=1, squeeze=False)):  # Float(batch x 1 x main_dim)
                    vehicle = vehicle.expand(b, n * n,
                                             vehicle.size(2))  # Float(batch x (cities_length*cities_length) x main_dim)
                    nodes = torch.cat((vehicle, base_nodes),
                                      dim=2)  # Float(batch x (cities_length*cities_length) x (main_dim*3))
                    nodes = self.linear_out_1(nodes)
                    nodes = F.relu(nodes)
                    nodes = self.linear_out_2(nodes)  # Float(batch x (cities_length*cities_length) x 1)
                    nodes = nodes.view(b, n, n)  # Float(batch x cities_length x cities_length)
                    res.append(nodes)
                # SCORES
                output = torch.stack(res, dim=1)  # Float(batch x fleet_length x cities_length x cities_length)

            else:
                # below: Float(batch x groups_length x cities_length x cities_length x main_dim)
                at_groups = fleet.unsqueeze(2).unsqueeze(3).expand(b, m, n, n, main_dim)
                from_cities = cities.unsqueeze(1).unsqueeze(3).expand_as(at_groups)
                to_cities = cities.unsqueeze(1).unsqueeze(2).expand_as(at_groups)
                # Float(batch x groups_length x cities_length x cities_length x (3 * main_dim))
                nodes = torch.cat((at_groups, from_cities, to_cities), dim=4)
                # Float(batch x (groups_length*cities_length*cities_length) x (3 * main_dim))
                nodes = nodes.view(b, m * n * n, main_dim * 3)
                # Float(batch x (groups_length*cities_length*cities_length) x (main_dim))
                nodes = self.linear_out_1(nodes)
                nodes = F.relu(nodes)
                nodes = self.linear_out_2(nodes)
                output = nodes.view(b, m, n, n)  # Float(batch x groups_length x cities_length x cities_length)

        ##### SET SCORE TO min_inf FOR STAYING AT SAME NODE #####
        min_inf = abs(output.min()) / 10
        mask_diag = torch.eye(n, n, device=output.device).bool()
        mask_diag[0, 0] = 0
        output = output.masked_fill(mask_diag.unsqueeze(0).unsqueeze(1).expand(b, m, n, n), min_inf)

        ##### SOFTASSIGN --> Semi multi-stochastic tensor #####
        result = self.softassign(output, demands)  # Float(batch x groups x from_city x to_city)

        # GET CURR PATHS and LOADS based on the probs
        if self.with_loads:
            # path_idx:                        Float(batch x fleet x n x n)
            # loads (uni-directional):         Float(batch x fleet)
            # pred_demands (uni-directional):  Float(batch x fleet x n x n)

            # get random permutation of vehicles
            perm_m = list(range(m))
            random.shuffle(perm_m)
            # ONLY TO-DIRECTION PATHS and LOADS
            path_idx, pred_loads = load_estimate(result, demands, perm_m)
            sample_path_b0 = [targ_as_lst(path_idx[0][i].max(1)[1], n) for i in perm_m]

        else:
            pred_loads = None
            sample_path_b0 = None

        return result, pred_loads, sample_path_b0
