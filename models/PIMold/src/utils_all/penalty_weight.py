
import numpy as np


def resc(t,t_max):
    '''to rescale between (min_t-(max_t/2)) and (max_t-(max_t/2))'''
    #(t/10)-15 #for t_max=300
    return (t-(t_max/2))/10

def weight_sched_sig(t,t_max):
    '''t is current epoch int - returns sigmoid(t_rescaled)'''
    # scale epochs down
    t_sc = resc(t,t_max)
    #x=x_/50
    #weight=1/(1+np.exp(-x))
    #weight=x/np.sqrt(1+x**2)
    #np.round(weight,3)
    return 1/(1.2+np.exp(-t_sc))



def weight_sched_stairs(t,t_max):
if 0 <= t < 25:
    return 0.0
if 25 <= t < 75:
    return 0.15
#if 50 <= t < 75:
#    return 0.2
if 75 <= t < 125:
    return 0.35
if 100<= t < 175:
    return 0.50
if 175<= t < 200:
    return 0.75 