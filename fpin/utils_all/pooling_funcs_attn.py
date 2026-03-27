
###The classes and functions defined in this file originate from the source code in Kaempfer & Wolff et al. (2018) and were amended for the VRP###
###Code is fused with code provided in Lee et al. (2019) (Set Transformer)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .net_funcs import PairwiseLinear  # models.PIM.utils_all.
from .basic_funcs import strip_main_diagonal  # models.PIM.utils_all
from .set_transformer_modules import ISAB, SAB, CrossAttentionPool


def pool(nodes, from_length, i, is_avg=False, self_pool=True, weights=None):
    #, weights_q=None
    '''nodes --> nodes to pool from
       from_length --> length of group: l_i (for which to pool for)
       to_dim --> nr. of cols in one batch (256)'''
    # nodes:        Float(batch x to_length x to_dim)
    # weights:      None | Float(batch x from_length x to_length x to_dim)
    ######################################################################
    # res:          Float(batch x from_length x to_dim)
    
    # WEIGHTED POOLING or WEIGHTED LOO POOLING
    if weights is not None:
        #and weights_q is not None:
        # Float(batch x from_length x to_length x to_dim)
        nodes = nodes.unsqueeze(1).expand(nodes.size(0), from_length, nodes.size(1), nodes.size(2))
        #print('nodes.size() after "if weights_d"',nodes.size())
        # Float(batch x from_length x to_length x to_dim)
        new_nodes = nodes * weights
        
        # below: Float(batch x from_length x to_dim)
        if is_avg:
            return new_nodes.mean(dim=2)
        
        #WEIGHTED LOO MAX POOLING
        #elif not self_pool and nodes.size(1)> 1:
        #    assert nodes.size(1) == from_length, 'LOO pooling is possible only when the nodes pool over themselves!'
        #    #print('WEIGHTED LOO pooling')
        #   # Float(batch x from_length x (to_length -1) x to_dim)
        #    new_nodes = strip_main_diagonal(new_nodes.permute(0,3,1,2)).permute(0, 2, 3, 1)            
        #    pooled_val,pooled_idx=new_nodes.max(dim=2)  # Float(batch x from_length x to_dim)
        #    return pooled_val
        
        #WEIGHTED MAX POOLING
        else:
            return new_nodes.max(dim=2)[0]
            
        
    # LOO POOLING !!
    elif not self_pool and nodes.size(1)> 1:
        assert nodes.size(1) == from_length, 'LOO pooling is possible only when the nodes pool over themselves!'
        # Float(batch x from_length x to_length x to_dim)
        new_nodes = nodes.unsqueeze(1).repeat(1, from_length, 1, 1)
        # Float(batch x from_length (to_length -1) x to_dim)
        new_nodes = strip_main_diagonal(new_nodes.permute(0,3,1,2)).permute(0, 2, 3, 1)
        if is_avg:
            return new_nodes.mean(dim=2) # Float(batch x from_length x to_dim)
        else:
            # LOO MAX POOLING
            return new_nodes.max(dim=2)[0]  # Float(batch x from_length x to_dim)
        
    # REGULAR POOLING
    else:
        if is_avg:
            res = nodes.mean(dim=1) # Float(batch x to_dim)
        else:
            # REGULAR MAX POOLING
            res = nodes.max(dim=1)[0] # Float(batch x to_dim)
            res=res.unsqueeze(1).expand(nodes.size(0), from_length, nodes.size(2))  # Float(batch x from_length x to_dim)
        return res




class PoolLayer(nn.Module):
    ''' through pooling size of groups stays d_model = main_dim'''
    def __init__(self, in_dims, out_dims, is_avg=False, self_pool=True, use_attention=False):
        super(PoolLayer, self).__init__()
        assert len(in_dims) == len(out_dims), 'the number of groups must be consistent across a pooling layer!'
        
        
        self.linears = nn.ModuleList([
            PairwiseLinear(in_dim + sum(in_dims), out_dim) # (in_dim_i + sum(in_dims) + 1) --> +1 b/c capa or q vec
            for in_dim, out_dim
            in zip(in_dims, out_dims)
        ])
        self.is_avg = is_avg
        self.self_pool = self_pool
        self.use_attention = use_attention
        if use_attention:
            # num_inds: num_inds - only fore ISAB: how many inducing points mediate interaction across the set
            #           Think of num_inds as a compressed representation of the input set
            #           For small to medium sets (like in CVRP), 5–16 is standard

            # self.attn_pool = AttentionPool(in_dims, out_dim=out_dims[0],  # same dim for now
            #                                num_heads=4,  # num_heads
            #                                use_isab=False,  # use_isab
            #                                num_inds=4)   # num_inds
            self.attn_pool = CrossAttentionPool(
                in_dims=in_dims,
                num_heads=4,
                use_isab=False,  # start False for CVRP-100
                num_inds=16,
                ln=True,
                # optionally compress only customers (index 1) if you ever need it:
                # isab_for_sources=[False, True, False],
            )
            # in case in_dims very across groups:
            # self.attn_modules = nn.ModuleList()
            # for d_in in in_dims:
            #     if use_isab:
            #         self.attn_modules.append(ISAB(d_in, out_dim, num_heads, num_inds, ln=ln))
            #     else:
            #         self.attn_modules.append(SAB(d_in, out_dim, num_heads, ln=ln))
        
    def forward(self, xs, weights):
        # demands, capa_vec,
        #,weights_q=None
        # xs:       embedded+normed set--> list[Float(batch x length_i x in_dim_i)]
        # weights_d:  None | list[None | list[None | Float(batch x length_i x length_i x in_dim_i)]]
        # demands:    Float(batch x 1 x length_2)  
        # capa_vec:    Float(batch x 1 x length_3)
        ##########################################################################################
        # res:      list[Float(batch x length_i x out_dim_i)]
        
        if weights is None:
            weights = [None] * len(xs)
            
        #if weights_q is None:
        #    weights_q = [None] * len(xs)
        
        # capa_demand_lst=[demands[:,:,0],demands[:,0,1:],capa_vec]
            
        res = []
        # loop over number of groups (3) and the corresp tensors and weights
        for i, (x, weight) in enumerate(zip(xs, weights)):
            if weight is None:
                weight = [None] * len(xs)
            
            # HERE IS POOOLING HAPPENING
            if self.use_attention:
                pooled_contexts = self.attn_pool(xs, i, self_pool=self.self_pool)
            else:
                pooled_contexts = [
                    pool(y, x.size(1), i, is_avg=self.is_avg,
                         self_pool=(self.self_pool or j != i),
                         weights=w, )
                    for j, (y, w) in enumerate(zip(xs, weight))
                ]
            # if not self.use_attention:
            #     pooled_contexts = [
            #         pool(y, x.size(1), i, is_avg=self.is_avg, self_pool=(self.self_pool or j != i), weights=w,)
            #         for j, (y, w) in enumerate(zip(xs, weight))
            #     ]
            # else:
            #     pooled_contexts = self.attn_pool(xs, i, self_pool=self.self_pool)
            #, weight_q #, weights_q=w_q
                        
            # COMBINE each elem x_i,r with its context c'_i,1,r ... c'_i,3,r
            # AND ADD CAPA VECTOR OR DEMAND VECTOR RESPECTIVELY
            # below: Float(batch x length_i x (in_dim_i + sum(in_dims)))
            # print('x.size()',x.size())
            # print('pooled_contexts[0].size()', pooled_contexts[0].size())
            # print('pooled_contexts[1].size()', pooled_contexts[1].size())
            x_concat = torch.cat((x, *pooled_contexts),dim=2)
            # print('x_concat.size()', x_concat.size())
            #print('cap_dem.size()',cap_dem.size())
            #print('cap_dem.unsqueeze(2).size()',cap_dem.unsqueeze(2).size())
            # below: Float(batch x length_i x (in_dim_i + sum(in_dims) + 1))
            # x_2_concat = torch.cat((x_concat,cap_dem.unsqueeze(2)),dim=2)
            new_x = self.linears[i](x_concat) # Float(batch x length_i x out_dim_i)
            # print('new_x.size()', new_x.size())
            res.append(new_x)
            
        return res


class AttentionPool(nn.Module):
    def __init__(self, in_dims, out_dim, num_heads=4, use_isab=False, num_inds=16, ln=False):
        super().__init__()
        self.use_isab = use_isab
        self.attn_modules = nn.ModuleList()
        for d in in_dims:
            if use_isab:
                #  use a single dim_out. If your sets have different input dims (e.g., depot: 8, city: 16),
                #  you might need to wrap them in linear projectors to unify dims before attention.
                self.attn_modules.append(ISAB(d, out_dim, num_heads, num_inds, ln=ln))
            else:
                self.attn_modules.append(SAB(d, out_dim, num_heads, ln=ln))

    def forward(self, xs, i, self_pool=True):
        # For pooling into xs[i]
        # pooled = []
        # for j, x_j in enumerate(xs):
        #     if not self_pool and i == j:
        #         pooled.append(torch.zeros_like(x_j))
        #     else:
        #         pooled.append(self.attn_modules[j](x_j))  # SAB(x_j)
        # return pooled
        # fix dim issue of different sets:
        # xs: list of [B, L_i, D_i]
        # Target: xs[i], shape [B, L_i, D_i]
        pooled_contexts = []
        x = xs[i]
        for j, y in enumerate(xs):
            if not self_pool and i == j:
                # Optionally skip self pooling by returning zeros or similar
                pooled_contexts.append(torch.zeros_like(x))
            else:
                ctx = self.attn_modules[j](y)  # [B, L_j, D]
                ctx = ctx.mean(dim=1, keepdim=True)  # [B, 1, D]
                ctx = ctx.repeat(1, x.size(1), 1)  # [B, L_i, D]
                pooled_contexts.append(ctx)
        return pooled_contexts  # list of [B, L_i, D]

#from torch.nn._functions.dropout import FeatureDropout
class FeatureDropOut1d(nn.Module):
    def __init__(self, p=0.5, inplace=False):
        super(FeatureDropOut1d, self).__init__()
        if p < 0 or p > 1:
            raise ValueError('dropout probability has to be between 0 and 1, '
                             'but got {}'.format(p))
        self.p = p
        self.inplace = inplace
        #self.drop=nn.Dropout(p=p)
        
    def forward(self, x):
        # x:    Float(batch x length x channels)
        # res:  Float(batch x length x channels)
        #print('self.training',self.training)
        x = x.transpose(1, 2)  # Float(batch x channels x length)
        # self.p, self.training
        x = nn.functional.dropout(x, self.p, self.training,self.inplace)  # Float(batch x channels x length)
        #print('x.shape',x.shape)
        #x = FeatureDropout.apply(x, self.p, self.training, self.inplace)  # Float(batch x channels x length)
        x = x.transpose(1, 2)  # Float(batch x length x channels)
        return x