import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# classes taken from https://github.com/juho-lee/set_transformer/blob/master/modules.py
# @InProceedings{lee2019set,
#     title={Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks},
#     author={Lee, Juho and Lee, Yoonho and Kim, Jungtaek and Kosiorek, Adam and Choi, Seungjin and Teh, Yee Whye},
#     booktitle={Proceedings of the 36th International Conference on Machine Learning},
#     pages={3744--3753},
#     year={2019}
# }
class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O

class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)

class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)

class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)


import torch
import torch.nn as nn

class CrossAttentionPool(nn.Module):
    """
    Cross-set pooling via MAB:
      ctx_{i<-j} = MAB_{i,j}(Q = X_i, K = X_j)  -> [B, L_i, D_j]

    Replacement for your current AttentionPool:
      forward(xs, i, self_pool=True) returns list of [B, L_i, D_j] for each source j.

    Output dims are D_j (per source set), matching non-attention weighted pool(...)
               --> matching PairwiseLinear(in_dim_i + sum(in_dims), out_dim_i).
    """
    def __init__(
        self,
        in_dims,                # list of D_j for each set j
        num_heads=4,
        use_isab=False,         # compress source sets with ISAB before cross-attn (optional)
        num_inds=16,
        ln=False,
        isab_for_sources=None,  # None or list[bool] length len(in_dims)
    ):
        super().__init__()
        self.in_dims = list(in_dims)
        self.num_sets = len(in_dims)

        if isab_for_sources is None:
            isab_for_sources = [use_isab] * self.num_sets
        assert len(isab_for_sources) == self.num_sets

        # Optional source compressors: X_j -> H_j with same dim D_j but fewer tokens
        self.source_compress = nn.ModuleList()
        for j, d_j in enumerate(self.in_dims):
            if isab_for_sources[j]:
                # ISAB(dim_in=d_j, dim_out=d_j) keeps dims consistent
                self.source_compress.append(ISAB(d_j, d_j, num_heads=num_heads, num_inds=num_inds, ln=ln))
            else:
                self.source_compress.append(nn.Identity())

        # Cross-attention modules MAB_{i,j}:
        # MAB(dim_Q = D_i, dim_K = D_j, dim_V = D_j) -> output [B, L_i, D_j]
        self.mab = nn.ModuleList([
            nn.ModuleList([
                MAB(dim_Q=self.in_dims[i], dim_K=self.in_dims[j], dim_V=self.in_dims[j],
                    num_heads=num_heads, ln=ln)
                for j in range(self.num_sets)
            ])
            for i in range(self.num_sets)
        ])

    def forward(self, xs, i, self_pool=True):
        """
        xs: list of tensors [B, L_j, D_j]
        i:  target set index
        returns: list ctx_j with shapes [B, L_i, D_j]
        """
        x_i = xs[i]
        pooled_contexts = []
        for j, x_j in enumerate(xs):
            if (not self_pool) and (i == j):
                # shape must be [B, L_i, D_j]; if i==j then D_j==D_i
                pooled_contexts.append(torch.zeros(x_i.size(0), x_i.size(1), self.in_dims[j],
                                                  device=x_i.device, dtype=x_i.dtype))
                continue

            # Optionally compress source tokens (keeps dim D_j)
            k_j = self.source_compress[j](x_j)  # [B, L'_j, D_j] or [B, L_j, D_j]

            # Cross-attend: Q = target tokens, K = source tokens
            ctx_ij = self.mab[i][j](x_i, k_j)   # [B, L_i, D_j]
            pooled_contexts.append(ctx_ij)

        return pooled_contexts
