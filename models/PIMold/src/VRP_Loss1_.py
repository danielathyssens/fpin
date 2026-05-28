import torch
import torch.nn as nn

# from collections import defaultdict
import itertools

from utils_all.basic_funcs import to_variable, zeros
from utils_all.basic_funcs import sparse_dense_mul, sparse_dense_mul_loss


class VRPLoss(nn.Module):
    def __init__(self, start_weight=0.5, simple_loss=False,
                 no_perms=False, size_average=True, with_penalty=True,
                 with_direct_demand=True, with_load_loss=True):
        super(VRPLoss, self).__init__()
        assert 0 <= start_weight <= 1, 'start_weight must be [0,1]'

        self.start_weight = start_weight
        self.simple_loss = simple_loss
        self.no_perms = no_perms
        self.size_average = size_average
        self.with_penalty = with_penalty
        self.with_direct_demand = with_direct_demand
        self.with_load_loss = with_load_loss
        self._permutations = {}
        # self.mse = nn.MSELoss(reduction='none')

    def get_permutations(self, fleet):
        # res: Long(perms x fleet)
        if fleet in self._permutations:
            return self._permutations[fleet]

        permutations = torch.LongTensor(list(itertools.permutations(range(fleet))))
        # permutations = to_variable(permutations)
        self._permutations[fleet] = permutations
        return permutations

    def forward(self, probs, curr_sol, loads, demands_pred, target, targ_loads, targ_dems):
        # probs:           Float(batch x m x n x n)
        # loads:           Float(batch x m) or Float(batch x 2 x m)
        # demands_pred:    Float(batch x m x n x n) or Float(batch x 2 x m x n x n)
        # target:          Byte(batch x m x n x n)
        # targ_loads:      Float(batch x m)
        # targ_dems:       Float(batch x m x n x n)

        # print('STARTING OF WHOLE LOSS CALCULATION')

        # get dimensionalities
        b, m, n, _ = probs.size()

        # Transform probs to log_probs and targets to float
        log_probs = probs.log()

        # calc loss without permutations
        if self.no_perms:
            losses = -log_probs * target  # Float(batch x fleet x from_city x to_city)
            # calc SIMPLE loss without permutations
            if self.simple_loss:
                weighted_losses = losses.sum(dim=3).sum(dim=2).sum(dim=1) / (n + m - 1)  # Float(batch)
            # calc WEIGHTED loss without permutations
            else:
                start_losses = losses[:, :, 0, :]  # Float(batch x m x n)
                next_losses = losses[:, :, 1:, :]  # Float(batch x m x (n-1) x n)
                start_losses = start_losses.sum(dim=2).sum(dim=1) / m  # Float(batch)
                next_losses = next_losses.sum(dim=3).sum(dim=2).sum(dim=1) / (n - 1)  # Float(batch)
                weighted_losses = start_losses * self.start_weight + next_losses * (
                        1 - self.start_weight)  # Float(batch)


        # calc permutation invariant loss
        else:
            # PREPROCESS TARGET, LOG LOSS, PRED_DEMANDS, PRED_LOADS
            # initial target:        Float(batch x m x n x n)
            # initial targ_dems:     Float(batch x m x n x n)
            # initial logProbs:      Float(batch x m x n x n)
            # initial loads:         Float(batch x m)
            # -----------------------------------------------------------------
            # transf. target:        Float(batch x m x 2 x m x n x n)
            # transf. targ_dems:     Float(batch x m x 2 x m x n x n)
            # transf. logProbs:      Float(batch x m x 2 x m x n x n)
            # transf. loads (pen):   Float(batch x m x 2 x m)
            # transf. loads (load):  Float(batch x m x m)

            # flip to and from city for TARGETS
            # sparse:
            stacked_target = torch.stack((target, target.transpose(2, 3)), dim=1)
            # # Float(batch x from_groups x 2 x to_groups x from_city x to_city) (for dense):
            # all_target = stacked_target.unsqueeze(1).expand(b, m, 2, m, n, n)
            # sparse:
            all_target = torch.stack([stacked_target] * m, dim=1)

            # Float(batch x from_groups x 2 x to_groups x from_city x to_city):
            log_probs = log_probs.unsqueeze(2).unsqueeze(3).expand_as(all_target)

            if self.with_penalty:
                loads_p = loads.unsqueeze(2).unsqueeze(3).expand(b, m, 2, m)  # for penalty
            if self.with_load_loss:
                if self.with_penalty:
                    loads_ = loads.unsqueeze(2).expand(b, m, m)  # for load loss
                    # loads_p = loads.unsqueeze(2).unsqueeze(3).expand(b, m, 2, m)  # for penalty
                else:
                    loads_ = loads.unsqueeze(2).expand(b, m, m)  # for load loss
            if self.with_direct_demand:
                curr_sol = curr_sol.unsqueeze(2).unsqueeze(3).expand_as(all_target)  # for inv. CE
                # demands_pred = demands_pred.unsqueeze(2).unsqueeze(3).expand_as(all_target)  # for inv. CE

            # INVERSE CE LOSS
            if self.with_direct_demand:
                # Losses for all directions and all vehicle combis + inverse CE
                # Float(batch x 2 x fleet x from_city x to_city):
                # stacked_targ_dems = torch.stack((targ_dems, targ_dems.transpose(2, 3)), dim=1)
                # # Float(batch x from_groups x 2 x to_groups x from_city x to_city):
                # all_demand_targs = stacked_targ_dems.unsqueeze(1).expand(b, m, 2, m, n, n)
                # Float(batch x from_groups x 2 x to_groups x from_city x to_city):
                # losses = (-log_probs * all_target) + 5 * torch.abs(demands_pred - all_demand_targs)
                losses = (-log_probs * all_target.to_dense()) + 0.7*torch.abs(curr_sol - all_target.to_dense())
            else:
                # Losses for all directions and all vehicle combis
                # # Float(batch x from_groups x 2 x to_groups x from_city x to_city) (for dense):
                # losses = -log_probs * all_target
                # sparse:
                losses = sparse_dense_mul_loss(all_target, -log_probs)

            # SIMPLE permutation invariant loss (NO WEIGHTS ON STARTS)
            if self.simple_loss:
                print('self.simple_loss')
                # Float(batch x from_groups x 2 x to_groups) (for dense):
                # weighted_losses = losses.sum(dim=5).sum(dim=4) / (n + m - 1)
                # sparse:
                weighted_losses = torch.sparse.sum(torch.sparse.sum(losses, dim=5), dim=4) / (n + m - 1)

            # WEIGHTED permutation invariant loss (WEIGHTS ON STARTS)
            else:
                # dense:
                if losses.is_sparse:
                    # Float(batch x from_groups x 2 x to_groups x to_city):
                    starts_losses = losses.to_dense()[:, :, :, :, 0, :]
                    # Float(batch x from_groups x 2 x to_groups x (from_city - 1) x to_city):
                    nexts_losses = losses.to_dense()[:, :, :, :, 1:, :]
                else:
                    starts_losses = losses[:, :, :, :, 0, :]
                    nexts_losses = losses[:, :, :, :, 1:, :]
                # Float(batch x from_groups x 2 x to_groups):
                starts_losses = starts_losses.sum(dim=4) / m
                nexts_losses = nexts_losses.sum(dim=5).sum(dim=4) / (n - 1)

                # PENALTY (in sparse)
                if self.with_penalty:
                    # Float(batch x m x 2 x m):
                    penalty = torch.where(loads_p >= 1.00001, torch.square(loads_p), zeros(1))
                    # Float(batch x m x 2 x m):
                    weighted_losses = (starts_losses * self.start_weight + nexts_losses * (
                            1 - self.start_weight))
                    weighted_losses = weighted_losses + (0.3 * penalty)

                else:
                    # Float(batch x from_groups x 2 x to_groups):
                    weighted_losses = (starts_losses * self.start_weight) + (
                            nexts_losses * (1 - self.start_weight))
                    # weighted_losses = weighted_losses.to_dense()

            # CHOOSE BEST DIRECTION for each match
            # Float(batch x 4 x 4):
            weighted_losses = weighted_losses.min(dim=2)[0]

            # CHOOSE BEST PERMUTATION
            perms_all = self.get_permutations(m)  # Long(perms x groups)
            # if more than 10 vehicles (like for VRP100) run in slices of perms
            if m >= 10:
                curr_min_mean = 90000
                for perms in torch.tensor_split(perms_all, 115):
                    perms = to_variable(perms).unsqueeze(0).expand(b, perms.size(0),
                                                                   perms.size(1))  # Long(batch x perms x m)
                    # LOAD LOSS
                    if self.with_load_loss:
                        loads_perms = loads_.gather(dim=1, index=perms)  # Float(batch x perms x groups)
                        targ_loads_ = targ_loads.unsqueeze(1).expand_as(loads_perms)   # Float(batch x perms x m)
                        perm_load_losses = torch.abs((loads_perms - targ_loads_))      # Float(batch x perms x m)
                        weighted_losses_ = weighted_losses.gather(dim=1, index=perms)  # Float(batch x perms x m)
                        # NOW BOTH weighted_losses and perm_load_losses are [b,perms,m]
                        weighted_losses_l = weighted_losses_ + 0.5 * perm_load_losses  # Float(batch x perms x m)
                        weighted_losses_curr = weighted_losses_l.sum(dim=2).min(dim=1)[0]  # Float(batch)
                    else:
                        weighted_losses_g = weighted_losses.gather(dim=1, index=perms)  # Float(batch x perms x m)
                        weighted_losses_curr = weighted_losses_g.sum(dim=2).min(dim=1)[0]  # Float(batch)

                    # only store best loss:
                    if weighted_losses_curr.mean(dim=0) < curr_min_mean:
                        curr_min_mean = weighted_losses_curr.mean(dim=0)
                        weighted_losses_f = weighted_losses_curr
            else:
                perms = to_variable(perms_all.unsqueeze(0).expand(b, perms_all.size(0), perms_all.size(1)))

                # LOAD LOSS
                if self.with_load_loss:
                    # loads_ --> now [b,m,m]
                    loads_perms = loads_.gather(dim=1, index=perms)                 # Float(batch x perms x m)
                    targ_loads = targ_loads.unsqueeze(1).expand_as(loads_perms)     # Float(batch x perms x m)
                    perm_load_losses = torch.abs((loads_perms - targ_loads))        # Float(batch x perms x m)
                    weighted_losses = weighted_losses.gather(dim=1, index=perms)    # Float(batch x perms x m)
                    # NOW BOTH weighted_losses and perm_load_losses are [b,perms,m]
                    weighted_losses_l = weighted_losses + 0.5*perm_load_losses      # Float(batch x perms x m)
                    weighted_losses_f = weighted_losses_l.sum(dim=2).min(dim=1)[0]  # Float(batch)
                else:
                    weighted_losses = weighted_losses.gather(dim=1, index=perms)  # Float(batch x perms x m)
                    weighted_losses_f = weighted_losses.sum(dim=2).min(dim=1)[0]  # Float(batch)

        if self.size_average:
            weighted_losses_f = weighted_losses_f.mean(dim=0)  # Float(1)

        return weighted_losses_f
