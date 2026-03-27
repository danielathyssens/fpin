
###classes and functions defined in this file originate from the source code in Kaempfer & Wolff et al. (2018)
# and were amended for the FC-CVRP###

import torch
import torch.nn as nn
import torch.nn.functional as F

# from models.PIM.utils_all.basic_funcs import Identity
from fpin.utils_all.net_funcs import distribute,Distributed,PairwiseLinear,FeedForwardLayer,LayerNorm, SimpleGATLayer
from fpin.utils_all.pooling_funcs_attn import PoolLayer,FeatureDropOut1d


class PermInvNet(nn.Module):
    '''generic network architecture for Perm. Inv. Net (fig.2)'''
    
    def __init__(self,layers,in_dims,main_dims,ff_hidden_dims,
                 avg_pool=False,residual=True, norm=True, dropout=0.0,
                 embeddings=True,self_pool=False,
                 embedding_norm=True, use_attn=False):
        super().__init__()
        
        
        # EMBEDDING
        if embeddings:
            self.embeddings = Distributed(PairwiseLinear, in_dims, main_dims)
        else:
            # assert in_dims == main_dims, 'in_dims must equal main_dims if no embeddings are used'
            self.embeddings = None

        if embedding_norm:
            self.embedding_norms = Distributed(LayerNorm, main_dims)

        # POOL LAYERS
        self.pool_layers = nn.ModuleList([
            PoolLayer(main_dims, main_dims, is_avg=avg_pool, self_pool=self_pool, use_attention=use_attn)
            for _ in range(layers)
        ])
        
        self.layers = layers
        
        
        # FEED FORWARD LAYER
        self.feed_forwards = nn.ModuleList([
            FeedForwardLayer(main_dims, ff_hidden_dims, main_dims) for _ in range(layers)
        ])
        
        
        # RESIDUAL BLOCK
        if residual:
            self.apply_layer= lambda olds, news: [old + new for old, new in zip(olds, news)]
        else:
            self.apply_layer= lambda olds, news: distribute(F.relu, news)
        
        
        # POOL NORMS "for _ in range(layers)"
        self.pool_norms = nn.ModuleList([
            Distributed(LayerNorm, main_dims)
            if norm
            else None
            for _ in range(layers)
        ])
        
        # FEED FORWARD NORMS
        self.feed_forward_norms = nn.ModuleList([
            Distributed(LayerNorm, main_dims)
            if norm
            else None
            for _ in range(layers)
        ])


        # DROPOUT
        #if dropout > 0:
            #print('use dropout')
        #    self.dropout = Distributed(FeatureDropOut1d, [dropout] *len(main_dims))
        #else:
        #    self.dropout = Identity()

        
        
    def forward(self, xs, weights=None):
        """EXECUTES PermInvNet
        # xs:             list[Float(batch x length_i x in_dim_i)]
        # weights:  None | list[None | list[None | Float(batch x length_i x length_i x main_dim_i)]]
        # demands:        Float(batch x 1 x length_2) 
        # capa_vec:       Float(batch x length_3)
        ############################################################################################
        # res:            list[Float(batch x length_i x main_dim_i)]
        """
        if self.embeddings is not None:
            xs = self.embeddings(xs)  # --> Embedded Graph and Fleet in seperate encoder else plain MLP embedding
        xs = self.embedding_norms(xs)  # layer norm for embedding
        # "Groups are fed into i=1,...,N consecutive perm.inv. pooling blocks"
        for i in range(self.layers):
            
            # Call LOO Pooling Layer
            new_xs = self.pool_layers[i](xs, weights)
            # new_xs = self.dropout(new_xs)
            xs = self.apply_layer(xs, new_xs)   # resid. block
            xs = self.pool_norms[i](xs)          # norm perm inv pool layer
              
            # Shared FC --> Feed forward
            new_xs = self.feed_forwards[i](xs)
            xs = self.apply_layer(xs, new_xs)  # resid. block
            xs = self.feed_forward_norms[i](xs) # norm the shared FC layer
        

        return xs