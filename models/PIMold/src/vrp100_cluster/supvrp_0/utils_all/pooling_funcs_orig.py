import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils_all.net_funcs import PairwiseLinear
from utils_all.basic_funcs import strip_main_diagonal,zeros

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
            #print('WEIGHTED MAX POOLING')
            return new_nodes.max(dim=2)[0]
            
        
    # LOO POOLING !!
    elif not self_pool and nodes.size(1)> 1:
        #print('LOO Pooling')
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
        #print('REGULAR Pooling')
        if is_avg:
            res = nodes.mean(dim=1) # Float(batch x to_dim)
        else:
            # REGULAR MAX POOLING
            res = nodes.max(dim=1)[0] # Float(batch x to_dim)
            res=res.unsqueeze(1).expand(nodes.size(0), from_length, nodes.size(2))  # Float(batch x from_length x to_dim)
        return res




class PoolLayer(nn.Module):
    ''' through pooling size of groups stays d_model = main_dim'''
    def __init__(self, in_dims, out_dims, is_avg=False, self_pool=True):
        super(PoolLayer, self).__init__()
        assert len(in_dims) == len(out_dims), 'the number of groups must be consistent across a pooling layer!'
        
        
        self.linears = nn.ModuleList([
            PairwiseLinear(in_dim + sum(in_dims), out_dim) # (in_dim_i + sum(in_dims) + 1) --> +1 b/c capa or q vec
            for in_dim, out_dim
            in zip(in_dims, out_dims)
        ])
        self.is_avg=is_avg
        self.self_pool=self_pool
        
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
            # capa_demand_lst,
            #, cap_dem
            #,weight_q
            #,weights_q
            #print('Pooling for context vectors i=',i)
            #print('i is',i)
            #print('weights_q[0]',weights_q[0])
            if weight is None:
                weight = [None] * len(xs)
                
            #if weight_q is None:
            #    weight_q = [None] * len(xs)
                
            #print('weight[0]',weight[0])
            #print('weight[1].size()',weight[1].size())
            
            # HERE IS POOOLING HAPPENING
            pooled_contexts = [
                pool(y, x.size(1), i, is_avg=self.is_avg, self_pool=(self.self_pool or j != i), weights=w,)
                for j, (y, w) in enumerate(zip(xs, weight))
            ]
            #, weight_q #, weights_q=w_q
                        
            # COMBINE each elem x_i,r with its context c'_i,1,r ... c'_i,3,r
            # AND ADD CAPA VECTOR OR DEMAND VECTOR RESPECTIVELY
            # below: Float(batch x length_i x (in_dim_i + sum(in_dims)))
            x_concat = torch.cat((x, *pooled_contexts),dim=2)
            #print('cap_dem.size()',cap_dem.size())
            #print('cap_dem.unsqueeze(2).size()',cap_dem.unsqueeze(2).size())
            # below: Float(batch x length_i x (in_dim_i + sum(in_dims) + 1))
            # x_2_concat = torch.cat((x_concat,cap_dem.unsqueeze(2)),dim=2)
            #print('x_2_concat.size()',x_2_concat.size())
            new_x = self.linears[i](x_concat) # Float(batch x length_i x out_dim_i)
            res.append(new_x)
            
        return res



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