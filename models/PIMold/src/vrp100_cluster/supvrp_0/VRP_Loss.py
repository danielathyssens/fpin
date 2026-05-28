import torch
import torch.nn as nn

# from collections import defaultdict
import itertools

from utils_all.basic_funcs import to_variable, zeros
from utils_all.basic_funcs import sparse_dense_mul, sparse_dense_mul_loss


class VRPLoss(nn.Module):
    def __init__(self, start_weight=0.5, simple_loss=False,
                 no_perms=False, size_average=True, with_penalty=True,
                 with_load_loss=True):
        super(VRPLoss, self).__init__()
        assert 0 <= start_weight <= 1, 'start_weight must be [0,1]'

        self.start_weight = start_weight
        self.simple_loss = simple_loss
        self.no_perms = no_perms
        self.size_average = size_average
        self.with_penalty = with_penalty
        self.with_load_loss = with_load_loss

        self._permutations = {}

    def get_permutations(self, fleet):
        # res: Long(perms x fleet)
        if fleet in self._permutations:
            return self._permutations[fleet]

        permutations = torch.LongTensor(list(itertools.permutations(range(fleet))))
        # permutations = to_variable(permutations)
        # print(permutations)

        self._permutations[fleet] = permutations
        return permutations

    def forward(self, probs, loads, target, targ_loads):
        # probs:           Float(batch x group x from_city x to_city)
        # loads:           Float(batch x group) or Float(batch x 2 x group)
        # target:          Byte(batch x group x from_city x to_city)
        # targ_loads:      Float(batch x group)

        # print('STARTING OF WHOLE LOSS CALCULATION')

        # get dimensionalities
        b, m, n, _ = probs.size()

        # Transform probs to log_probs and targets to float
        log_probs = probs.log()
        
        #print('targ_loads.size()',targ_loads.size())

        # calc loss without permutations
        if self.no_perms:
            print('self.no_perms')
            losses = -log_probs * target  # Float(batch x fleet x from_city x to_city)
            # calc SIMPLE loss without permutations
            if self.simple_loss:
                print('self.no_perms AND self.simple_loss')
                weighted_losses = losses.sum(dim=3).sum(dim=2).sum(dim=1) / (n + m - 1)  # Float(batch)
            # calc WEIGHTED loss without permutations
            else:
                print('self.no_perms AND  NOT self.simple_loss')
                start_losses = losses[:, :, 0, :]  # Float(batch x fleet x to_city)
                next_losses = losses[:, :, 1:, :]  # Float(batch x fleet x (from_city-1) x to_city)
                start_losses = start_losses.sum(dim=2).sum(dim=1) / m  # Float(batch)
                next_losses = next_losses.sum(dim=3).sum(dim=2).sum(dim=1) / (n - 1)  # Float(batch)
                weighted_losses = start_losses * self.start_weight + next_losses * (
                        1 - self.start_weight)  # Float(batch)


        # calc permutation invariant loss
        else:
            # PREPROCESS TARGET, LOG LOSS, PRED_DEMANDS, PRED_LOADS
            # initial target:        Float(batch x fleet x from_city x to_city)
            # initial logProbs:      Float(batch x fleet x from_city x to_city)
            # initial loads:         Float(batch x fleet)
            # -----------------------------------------------------------------
            # transf. target:        Float(batch x fleet x 2 x fleet x from_city x to_city)
            # transf. logProbs:      Float(batch x fleet x 2 x fleet x from_city x to_city)
            # transf. loads:         Float(batch x fleet x fleet)

            # flip to and from city for TARGETS
            # # Float(batch x 2 x fleet x from_city x to_city) (for dense):
            # sparse:
            stacked_target = torch.stack((target, target.transpose(2, 3)), dim=1)
            # # Float(batch x from_groups x 2 x to_groups x from_city x to_city) (for dense):
            # all_target = stacked_target.unsqueeze(1).expand(b, m, 2, m, n, n)
            # sparse:
            all_target = torch.stack([stacked_target] * m, dim=1)

            # Float(batch x from_groups x 2 x to_groups x from_city x to_city):
            log_probs = log_probs.unsqueeze(2).unsqueeze(3).expand_as(all_target)

            if self.with_penalty:
                loads_ = loads.unsqueeze(2).expand(b, m, m)
            if self.with_load_loss and not self.with_penalty:
                loads_ = loads.unsqueeze(2).expand(b, m, m)

            losses = sparse_dense_mul_loss(all_target, -log_probs)

            # SIMPLE permutation invariant loss (NO WEIGHTS ON STARTS)
            if self.simple_loss:
                #print('self.simple_loss')
                # Float(batch x from_groups x 2 x to_groups) (for dense):
                # weighted_losses = losses.sum(dim=5).sum(dim=4) / (n + m - 1)
                # sparse:
                weighted_losses = torch.sparse.sum(torch.sparse.sum(losses, dim=5), dim=4) / (n + m - 1)
                #weighted_losses.to_dense()

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
                # Float(batch x from_groups x 2 x to_groups):
                weighted_losses = (starts_losses * self.start_weight) + (
                        nexts_losses * (1 - self.start_weight))

            # CHOOSE BEST DIRECTION for each match
            # Float(batch x from_groups x to_groups):
            if weighted_losses.is_sparse:
                weighted_losses = weighted_losses.to_dense().min(dim=2)[0]
                #weighted_losses = weighted_losses.min(dim=2)[0]
            else:
                weighted_losses = weighted_losses.min(dim=2)[0]

            # PENALTY
            if self.with_penalty:
                # Float(batch x from_groups x to_groups):
                penalty = torch.where(loads_ >= 1.00001, torch.square(loads_), zeros(1))
                weighted_losses = weighted_losses + (0.1 * penalty)

            # CHOOSE BEST PERMUTATION
            perms_all = self.get_permutations(m)  # Long(perms x groups)
            # if more than 10 vehicles (like for VRP100) run in slices of perms
            torch.cuda.empty_cache()
            if m >= 10:
                curr_min_mean = 90000
                for perms in torch.tensor_split(perms_all, 115):
                    perms = to_variable(perms).unsqueeze(0).expand(b, perms.size(0),
                                                                   perms.size(1))  # Long(batch x perms x groups)

                    # LOAD LOSS
                    if self.with_load_loss:
                        loads_perms = loads_.gather(dim=1, index=perms) # Float(batch x perms x groups)
                        #print('targ_loads.size()',targ_loads.size())
                        targ_loads_ = targ_loads.unsqueeze(1).expand_as(loads_perms)  # Float(batch x perms x groups)
                        perm_load_losses = torch.abs((loads_perms - targ_loads_)) # Float(batch x perms x groups)
                        weighted_losses_ = weighted_losses.gather(dim=1, index=perms)  # Float(batch x perms x groups)
                        #NOW BOTH weighted_losses and perm_load_losses are [b,perms,m]
                        weighted_losses_l = weighted_losses_ + 0.3*perm_load_losses    # Float(batch x perms x groups)
                        weighted_losses_curr = weighted_losses_l.sum(dim=2).min(dim=1)[0]  # Float(batch)
                        # OLD:
                        # perms_l = perms.unsqueeze(1).expand(b, 2, perms.size(1),
                        #                                    perms.size(2))  # Long(batch x 2 x perms x groups)
                        # loads_perms = loads_.gather(dim=2, index=perms_l)  # Float(batch x 2 x perms x groups)
                        # targ_loads_ = targ_loads.unsqueeze(1).unsqueeze(2).expand_as(loads_perms)
                        # perm_load_losses = torch.abs((loads_perms - targ_loads_))
                        # print('weighted_losses.size()',weighted_losses.size())
                        # weighted_losses_ = weighted_losses.gather(dim=1, index=perms)  # Float(batch x perms x groups)
                        # weighted_losses_ = weighted_losses_.sum(dim=2).min(dim=1)[0]  # Float(batch)
                        # perm_load_losses = perm_load_losses.sum(3).min(2)[0].min(1)[0]  # Float(batch)
                        # weighted_losses=weighted_losses+(0.3*torch.square(perm_load_losses))
                        # weighted_losses_curr = weighted_losses_ + (0.3 * perm_load_losses)
                    else:
                        weighted_losses_g = weighted_losses.gather(dim=1, index=perms)  # Float(batch x perms x groups)
                        weighted_losses_curr = weighted_losses_g.sum(dim=2).min(dim=1)[0]  # Float(batch)
                        # perm_load_losses = None

                    # only store best loss:
                    if weighted_losses_curr.mean(dim=0) < curr_min_mean:
                        curr_min_mean = weighted_losses_curr.mean(dim=0)
                        weighted_losses_f = weighted_losses_curr
            else:
                perms = to_variable(perms_all.unsqueeze(0).expand(b, perms_all.size(0), perms_all.size(1)))

                # LOAD LOSS
                if self.with_load_loss:
                    #loads_ --> now [b,m,m] (before: [b,m,2,m])
                    #perms_l = perms.unsqueeze(1).expand(b, 2, perms.size(1),
                    #                                    perms.size(2))  # Long(batch x 2 x perms x groups)
                    #loads_perms = loads_.gather(dim=2, index=perms_l)  # Float(batch x 2 x perms x groups)
                    loads_perms = loads_.gather(dim=1, index=perms) # Float(batch x perms x groups)
                    targ_loads = targ_loads.unsqueeze(1).expand_as(loads_perms)  # Float(batch x perms x groups)
                    perm_load_losses = torch.abs((loads_perms - targ_loads)) # Float(batch x perms x groups)
                    weighted_losses = weighted_losses.gather(dim=1, index=perms)  # Float(batch x perms x groups)
                    # NOW BOTH weighted_losses and perm_load_losses are [b,perms,m]
                    weighted_losses_l = weighted_losses + 0.5*perm_load_losses    # Float(batch x perms x groups)
                    weighted_losses_f = weighted_losses_l.sum(dim=2).min(dim=1)[0]  # Float(batch)
                else:
                    weighted_losses = weighted_losses.gather(dim=1, index=perms)  # Float(batch x perms x groups)
                    weighted_losses_f = weighted_losses.sum(dim=2).min(dim=1)[0]  # Float(batch)
                    # perm_load_losses = None

        if self.size_average:
            weighted_losses_f = weighted_losses_f.mean(dim=0)  # Float(1)

        return weighted_losses_f
