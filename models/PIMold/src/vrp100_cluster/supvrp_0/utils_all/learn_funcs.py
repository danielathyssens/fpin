import numpy as np
import torch

def load_ckp(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
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
    #x=x_/50
    #weight=1/(1+np.exp(-x))
    #weight=x/np.sqrt(1+x**2)
    #np.round(weight,3)
    return 1/(1.2+np.exp(-t_sc))

def constant_1(t,t_max):
	return 1

def constant_50(t,t_max):
	return 0.5
	
def constant_25(t,t_max):
	return 0.25
	
def constant_30(t,t_max):
	return 0.3


def stairs(t,t_max):
    if 0 <= t < 25:
        return 0.0
    if 25 <= t < 100:
        return 0.25
    #if 50 <= t < 75:
    #    return 0.2
    if 100 <= t < 150:
        return 0.5
    if 175<= t < 200:
        return 0.75 


def threshold(t,t_max):
    if 0 <= t < 50:
        return 0.15
    if 50 <= t < 150:
        return 0.175
    if 150<= t < 300:
        return 0.20