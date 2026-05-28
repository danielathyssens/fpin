from collections import defaultdict
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def distribute(nets,xs):
    if isinstance(nets, (list,tuple,nn.ModuleList)):
        return [nets[i](x) for i,x in enumerate(xs)]
    else:
        return [nets(x) for x in xs]

class Distributed(nn.Module):
    def __init__(self,net,*args):
        super(Distributed, self).__init__()
        
                               
        self.nets = nn.ModuleList([
            net(*arg) for arg in zip(*args)
        ])
        
    def forward(self, xs):
        return distribute(self.nets,xs)


class PairwiseLinear(nn.Module):
    
    def __init__(self, in_dim, out_dim, bias=True):
        super(PairwiseLinear, self).__init__()
        
        
        self.conv = nn.Conv1d(in_channels=in_dim, 
                              out_channels=out_dim,
                              kernel_size=1, bias=bias)
        
    def forward(self, x):
        '''Applies a 1D convolution over an input signal composed of 
            several input planes. i.e. learns an internal representation of 
            a "two"-dim. input (feature learning)'''
        # x:    Float(batch x length x in_dim)
        # res:  Float(batch x length x out_dim)
        #print('x shape: ',x.shape)
        #PrintLayer()
        
        
        return self.conv(x.transpose(1,2)).transpose(1,2)


class FeedForwardLayer(nn.Module):
    def __init__(self, in_dims, hidden_dims, out_dims):
        super(FeedForwardLayer, self).__init__()
        assert len(in_dims) == len(hidden_dims) == len(out_dims), \
            'the number of groups muts be consistent across a feed forward layer!'

        self.linear_1 = Distributed(PairwiseLinear, in_dims, hidden_dims)
        self.linear_2 = Distributed(PairwiseLinear, hidden_dims, out_dims)

    def forward(self, xs):
        # xs:   list[batch x length_i x in_dim_i]
        # res:  list[batch x length_i x out_dim_i]
        xs = self.linear_1(xs)  # list[Float(batch x length_i x hidden_dim_i]]
        xs = distribute(F.relu, xs)  # list[Float(batch x length_i x hidden_dim_i]]
        xs = self.linear_2(xs)  # list[Float(batch x length_i x out_dim_i]]
        return xs

class LayerNorm(nn.Module):
    '''Layer norm like in Ba et al. (2016), removes from each activation
       the mean and divides by standard dev. of all activations in layer'''
    def __init__(self, dim, eps=1e-8):
        super(LayerNorm, self).__init__()
        
        self.eps=eps
        self.gamma = nn.Parameter(torch.ones(dim), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(dim), requires_grad=True)
        
    def forward(self,x):
        # x:     Float(batch x length x dim)
        # res:   Float(batch x length x dim)
        
        mean = x.mean(dim=2, keepdim=True)  # batch x length x 1
        # std calc. like this due to gradient=NaN if std=0 for some sample
        std = ((x - mean).pow(2).sum(dim=2, keepdim=True).div(x.size(2) - 1) + self.eps).sqrt()  # batch x length x 1
        
        norm_x = (x - mean.expand_as(x)) / std.expand_as(x)
        
        return norm_x *self.gamma + self.beta