import torch
import torch.nn as nn
import torch.nn.functional as F

# from collections import defaultdict
import itertools
from itertools import tee
import math
from fpin.utils_all.basic_funcs import to_variable, zeros, sparse_dense_mul_loss, sparse_stack, safe_gather, sparse_transpose_dim2_dim3
from fpin.utils_all.loss_utils import succ_labels_from_sparse, hungarian_match_from_membership, permute_targets_sparse_by_perm, pred_membership_from_logits, tgt_membership_from_sparse, pool_logits_to_L

class VRPLoss(nn.Module):
    def __init__(self, start_weight=0.5, pen_w = 0.3, load_w = 0.5, simple_loss=False,
                 no_perms=False, size_average=True, with_penalty=True, with_load_loss=True, verbose=False,
                 joint_customer_norm=False):
        super(VRPLoss, self).__init__()
        assert 0 <= start_weight <= 1, 'start_weight must be [0,1]'

        self.start_weight = start_weight
        self.pen_w = pen_w
        self.load_w = load_w
        self.simple_loss = simple_loss
        self.no_perms = no_perms
        self.size_average = size_average
        self.with_penalty = with_penalty
        self.with_load_loss = with_load_loss
        self.verbose = verbose
        # F-PIN-S: when True, the model returns log_probs pre-normalized via
        # joint (vehicle, next-node) softmax for customer rows + per-vehicle
        # softmax for depot row. Skip the redundant log_softmax(dim=-1) below.
        self.joint_customer_norm = joint_customer_norm

        self._permutations = {}

    def get_permutations(self, fleet):
        # res: Long(perms x fleet)
        if fleet in self._permutations:
            return self._permutations[fleet]
        permutations = torch.LongTensor(list(itertools.permutations(range(fleet))))
        self._permutations[fleet] = permutations
        return permutations

    def sample_permutations(self, fleet, k=1000000):
        """
        Few thoughts:
        Keep k as large as your GPU memory can handle.
        Profile time and memory; you might get away with fewer permutations.
        If you want more efficient gradients, consider approximations like soft permutations
        or relaxations (Sinkhorn, Gumbel-Sinkhorn layers), but that’s more involved.
        """
        # total_perms = math.factorial(fleet)
        # k = int(s * math.factorial(fleet))
        permutations = []
        # for _ in range(k):
        #     perm = torch.randperm(fleet)
        #     permutations.append(perm)
        perms = torch.stack([torch.randperm(fleet) for _ in range(k)])
        return perms  # torch.stack(permutations)  # shape: (k, fleet)

    def forward(self, logits, loads, target, targ_loads, multi_gpu=False):
        # logits:           Float(batch x group x from_city x to_city)
        # loads:           Float(batch x group) or Float(batch x 2 x group)
        # target:          Byte(batch x group x from_city x to_city)
        # targ_loads:      Float(batch x group)

        # get dimensionalities
        b, m, n, _ = logits.size()
        # print("b",b)
        # print("m",m)
        # print("n",n)
        # Transform probs to log_probs and targets to float
        # log_probs = probs.log()
        if self.joint_customer_norm:
            # F-PIN-S: 'logits' is actually pre-normalized log_probs from the model
            # (depot per-vehicle softmax + customer joint (m, j) softmax). Use as-is.
            log_probs = logits
        else:
            log_probs = torch.log_softmax(logits, dim=-1)
        # print('in loss func:')
        # print('log_probs.device', log_probs.device)
        # print('log_probs.is_sparse', log_probs.is_sparse)
        # print('loads.device', loads.device)
        # print('target.device', target.device)
        # print('target.is_sparse', target.is_sparse)
        # print('targ_loads.device', targ_loads.device)
        # print('targ_loads.is_sparse', targ_loads.is_sparse)
        # print("log_probs:", log_probs.shape, log_probs.device)
        # print("targets:", target.shape, target.device)
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

        # elif self.edge_level_bce:
            # calculate only whether edge
        # calc permutation invariant loss
        else:
            # print("calculate permutation invariant loss")
            # PREPROCESS TARGET, LOG LOSS, PRED_DEMANDS, PRED_LOADS
            # initial target:        Float(batch x fleet x from_city x to_city)
            # initial logProbs:      Float(batch x fleet x from_city x to_city)
            # initial loads:         Float(batch x fleet)
            # -----------------------------------------------------------------
            # transf. target:        Float(batch x fleet x 2 x fleet x from_city x to_city)
            # transf. logProbs:      Float(batch x fleet x 2 x fleet x from_city x to_city)
            # transf. loads:         Float(batch x fleet x fleet)

            # flip to and from city for TARGETS
            # Float(batch x 2 x fleet x from_city x to_city) (for dense):
            # sparse:
            if multi_gpu:
                print('is multi_gpu')
                target_T = sparse_transpose_dim2_dim3(target)
                stacked_target = sparse_stack([
                    target,
                    target_T
                ], dim=1)
                all_target = sparse_stack([stacked_target for _ in range(m)], dim=1)
            else:
                stacked_target = torch.stack((target, target.transpose(2, 3)), dim=1)
                # print('stacked_target.is_sparse in multi_gpu', stacked_target.is_sparse)
                # Float(batch x from_groups x 2 x to_groups x from_city x to_city) (for dense):
                # sparse:
                all_target = torch.stack([stacked_target] * m, dim=1)
                # print('all_target.is_sparse', all_target.is_sparse)
                # Float(batch x from_groups x 2 x to_groups x from_city x to_city):
            log_probs = log_probs.unsqueeze(2).unsqueeze(3).expand_as(all_target)
            log_probs = log_probs.contiguous()
            # print('log_probs.is_sparse', log_probs.is_sparse)

            if self.with_penalty:
                loads_ = loads.unsqueeze(2).expand(b, m, m)
            if self.with_load_loss and not self.with_penalty:
                loads_ = loads.unsqueeze(2).expand(b, m, m)
            
            # pre-calculate all losses across directions and vehicle combinations
            # print('all_target.size()', all_target.size())
            # print('log_probs.size()', log_probs.size())
            losses = sparse_dense_mul_loss(all_target, -log_probs)
            # print('losses.size()', losses.size())
            # print('losses.device', losses.device)
            # SIMPLE permutation invariant loss (NO WEIGHTS ON STARTS)
            if self.simple_loss:
                # sparse:
                weighted_losses = torch.sparse.sum(torch.sparse.sum(losses, dim=5), dim=4) / (n + m - 1)
                weighted_losses.to_dense()

            # WEIGHTED permutation invariant loss (WEIGHTS ON STARTS)
            else:
                # print("calculate weighted permutation inv. loss")
                if losses.is_sparse:
                    # print("losses.is_sparse",losses.is_sparse)
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
                # print("weighted_losses.size()",weighted_losses.size())
                # print("weighted_losses.device", weighted_losses.device)
                # print("weighted losses[0][0]",weighted_losses[0][0])
            # CHOOSE BEST DIRECTION for each match
            # Float(batch x from_groups x to_groups):
            weighted_losses = weighted_losses.min(dim=2)[0]
            # print("weighted_losses.size()",weighted_losses.size())
            # PENALTY
            if self.with_penalty:
                # Float(batch x from_groups x to_groups):
                penalty = torch.where(loads_ >= 1.00001, torch.square(loads_), zeros(1).to(loads_.device))
                weighted_losses = weighted_losses + (self.pen_w * penalty.to(device=weighted_losses.device))

            # CHOOSE BEST PERMUTATION
            perms_all = self.get_permutations(m)  # Long(perms x groups)
            # get all perms
            perms = to_variable(perms_all.unsqueeze(0).expand(b, perms_all.size(0), perms_all.size(1)))
            # LOAD LOSS
            if self.with_load_loss:
                # loads_ --> [b,m,m]
                loads_perms = loads_.gather(dim=1, index=perms) # Float(batch x perms x groups)
                targ_loads = targ_loads.unsqueeze(1).expand_as(loads_perms)  # Float(batch x perms x groups)
                perm_load_losses = torch.abs((loads_perms - targ_loads)) # Float(batch x perms x groups)
                weighted_losses = weighted_losses.gather(dim=1, index=perms)  # Float(batch x perms x groups)
                weighted_losses_l = weighted_losses + (self.load_w*perm_load_losses)   # Float(batch x perms x groups)
                weighted_losses_f = weighted_losses_l.sum(dim=2).min(dim=1)[0]  # Float(batch)
            else:
                weighted_losses = weighted_losses.gather(dim=1, index=perms)  # Float(batch x perms x groups)
                weighted_losses_f = weighted_losses.sum(dim=2).min(dim=1)[0]  # Float(batch)

        if self.size_average:
            weighted_losses_f = weighted_losses_f.mean(dim=0)  # Float(1)

        return weighted_losses_f




