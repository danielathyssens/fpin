import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


from utils_all.basic_funcs import normalize_dims


class MTSPSoftassign(nn.Module):
    def __init__(self, layers, eps=1e-8):
        super(MTSPSoftassign, self).__init__()
        self.eps = eps
        self.layers = layers
        
        #self.capa_mask=Capa_Mask(fleet_size=4,nodes=21)

    def forward(self, output, demands):
        # output:       Float(batch x groups x from_city x to_city)
        # demands:      Float(batch x 1 x from_city)
        # res:          Float(batch x groups x from_city x to_city)

        if self.layers > 0:
            # calculate the maximum per each row in the first softmax 
            #(of both depot and others)
            
            # Float(batch x groups x 1 x 1)
            depots_max = output[:, :, :1, :].max(dim=3, keepdim=True)[0]  
            # Float(batch x groups x 1 x to_city)
            depots_max = depots_max.expand(output.size(0), output.size(1), 1, output.size(3))
            # Float(batch x 1 x (from_city - 1) x to_city)
            others_max = output[:, :, 1:, :].max(dim=1, keepdim=True)[0].max(dim=3, keepdim=True)[0]
            # Float(batch x groups x (from_city - 1) x to_city)
            others_max = others_max.expand(output.size(0), output.size(1), output.size(2) - 1, output.size(3))
            output_max = torch.cat((depots_max, others_max), dim=2)  # Float(batch x groups x from_city x to_city)

            # subtract the maximum from the output
            output = output - output_max

        # calculate exponents and save intermediate results
        output = output.exp()

        # normalize the three dimensions in each layer
        for i in range(self.layers):
            output = output.clamp(min=self.eps)

            # normalize the depot and others in the (normal / inverse) matrix

            # ---> handles constr. eq2a and eq2c
            # (every vehicle leaves depot & every customer is left once)
            
            # below: Float(batch x groups x 1 x to_city)
            starts_output = output[:, :, :1, :]
            starts_output = normalize_dims(starts_output, (3,))

            # below: Float(batch x groups x (from_city - 1) x to_city)
            nexts_output = output[:, :, 1:, :]
            nexts_output = normalize_dims(nexts_output, (1, 3))

            # below: Float(batch x groups x from_city x to_city)
            output = torch.cat((starts_output, nexts_output), dim=2)
            output = output.transpose(2, 3)

        # fix the transposition ---> then handles constr. eq2b and eq2d
        # (every vehicle returns to depot & every customer is visited once)
        if self.layers % 2 != 0:
            output = output.transpose(2, 3)
    
        return output
