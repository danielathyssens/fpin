import numpy as np
import torch

def load_ckp(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, checkpoint['epoch']


def resc(t,t_max):
    '''to rescale between (min_t-(max_t/2)) and (max_t-(max_t/2))'''
    #(t/10)-15 #for t_max=300
    return (t-(t_max/2))/10


def sigmoid(t,t_max):
    '''t is current epoch int - returns sigmoid(t_rescaled)'''
    # scale epochs down
    t_sc = resc(t,t_max)
    return 1/(1.2+np.exp(-t_sc))

def constant_1(t,t_max):
	return 1

def constant_50(t,t_max):
	return 0.5
	
def constant_25(t,t_max):
	return 0.25
	
def constant_30(t,t_max):
	return 0.3