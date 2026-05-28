
import torch
import numpy as np
import random
from utils_all.basic_funcs import zeros, ones
import sys
import itertools

# make CURR GREEDY PATH VALID (single INSTANCE)
#'''fixing routes; returns capacity conform and valid VRP sols ACCORDING to               probabilities'''
def make_valid(all_idxs,probs,remaining_capa,demand):
    failed=None
    m = probs.size(0)
    n = probs.size(1)
    probs_cut=probs.clone()
    demands_cut=demand
    all_visited_=[]
    for i in range(m):
        #print(targ_as_lst(all_idxs[i].max(1)[1]))
        all_visited_.extend(targ_as_lst(all_idxs[i].max(1)[1],n)[1:-1])
    batch_missing=torch.tensor(np.setdiff1d(list(np.arange(1,n)),sorted(all_visited_)),device=probs_cut.device).long()
    #batch_missing=torch.tensor(np.setdiff1d(list(np.arange(1,n)),sorted(all_visited_))).long().cuda()
    batch_missing_=batch_missing[demands_cut[0][batch_missing].sort(dim=0,descending=True)[1]]
    batch_missing_=list(batch_missing_)
    
    for j in batch_missing_:
        # choose possible vehicles
        available_vs=torch.where(remaining_capa+0.000001>=demands_cut[0][j].repeat(m))[0]
        if list(available_vs):
            # check which direction has the highest prob for node j:
            if probs_cut[available_vs,:,j].max(1)[0].max(0)[0]>probs_cut[available_vs,j,:].max(1)[0].max(0)[0]:
                #print('TO-direction')
                # TO-j-direction has highest
                v_to=available_vs[probs_cut[available_vs,:,j].max(1)[0].max(0)[1]]
                #print('v_to',v_to)
                # nodes available in v_to's route
                #nodes_available=torch.tensor(targ_as_lst(all_idxs[v_to].max(1)[1],n)[:-1]).cuda()
                nodes_available=torch.tensor(targ_as_lst(all_idxs[v_to].max(1)[1],n)[:-1],device=v_to.device)
                #print('nodes_available',nodes_available)
                c_to_idx=torch.where(probs_cut[v_to,nodes_available,j]==probs_cut[v_to,nodes_available,j].max(0)[0].max())[0]
                if list(c_to_idx):
                    #print('list(c_to_idx)',list(c_to_idx))
                    c_to_idx=c_to_idx[0].unsqueeze(0)
                #print('c_to_idx',c_to_idx)
                #print('c_to_idx[0]',c_to_idx[0])
                c_to=nodes_available[c_to_idx]
                #print('c_to',c_to)
                c_from_c_to=all_idxs[v_to,c_to,:].max(1)[1]
                #print('c_from_c_to',c_from_c_to)
                all_idxs[v_to,c_to,:]=0.0
                all_idxs[v_to,c_to,j]=1.0
                # change prev. output node from c_to to be output node of j instead
                all_idxs[v_to,j,c_from_c_to]=1.0
                v_=v_to
            else:
                # FROM-j-direction has highest
                v_from=available_vs[probs_cut[available_vs,j,:].max(1)[0].max(0)[1]]
                nodes_available=torch.tensor(targ_as_lst(all_idxs[v_from].max(1)[1],n)[:-1],device=v_from.device)
                #nodes_available=torch.tensor(targ_as_lst(all_idxs[v_from].max(1)[1],n)[:-1]).cuda()
                c_from_idx=torch.where(probs_cut[v_from,j,nodes_available]==probs_cut[v_from,j,nodes_available].max(0)[0].max())
                if list(c_from_idx):
                    c_from_idx=c_from_idx[0].unsqueeze(0)
                c_from=nodes_available[c_from_idx[0]]
                c_to_c_from=all_idxs[v_from,:,c_from].max(0)[1]
                all_idxs[v_from,:,c_from]=0.0
                all_idxs[v_from,j,c_from]=1.0
                # change prev. input node to c_from to be input to j instead
                all_idxs[v_from,c_to_c_from,j]=1.0
                v_=v_from
            remaining_capa[v_]=remaining_capa[v_]-demands_cut[0][j]
        else:
            failed='yes'
            
            
    demands_=demands_cut.unsqueeze(2).expand(m,n,n)
    final_loads=(demands_*all_idxs).sum(dim=2).sum(dim=1)
    
    # check if really all cities covered:
    final_routes=[]
    all_visited=[]
    for i in range(m):
        final_routes.append(targ_as_lst(all_idxs[i].max(1)[1],n))
        all_visited.extend(targ_as_lst(all_idxs[i].max(1)[1],n)[1:-1])
    missing_final=torch.tensor(np.setdiff1d(list(np.arange(1,n)),sorted(all_visited)),device=final_loads.device).long()
    
    return all_idxs, final_routes, final_loads, missing_final



# for ONE INSTANCE GET CURR GREEDY PATH (not necessarilly valid)
def greedy_path(probs,demand,current_perm):
    # only one instance probs and demands
    m = probs.size(0)
    n = probs.size(1)
    
    probs_=probs.clone()
    all_idxs=zeros(*(probs_.size(0),probs_.size(1),probs_.size(1)))
    remaining_capa=ones(*(probs_.size(0),))
    other=ones(*(probs_.size(1),probs_.size(1)))*-99.0
    for v in current_perm:
        # get starting batches
        curr_idx= probs_[v,:,:].max(1)[1][0]
        # remaining capa after starts for this vehicle
        demands_curr_idx=demand.squeeze(0)[curr_idx]
        #print(remaining_capa)
        remaining_capa[v]=remaining_capa[v]-demands_curr_idx
        all_idxs[v,0,curr_idx]=1.0
        probs_[:,:,curr_idx]=-99.0
        while (curr_idx != 0):
            #get next idxs
            next_idx=probs_[v,:,:].max(1)[1][curr_idx]
            
            # update current_idxs Mask
            all_idxs[v,curr_idx,next_idx]=1.0
            
            # fix all_idxs for catched terminated paths (0 (curr) --> 0 (next))
            all_idxs[v,0,0]=0.0
            
            # update probs after chosen
            probs_[:,:,next_idx]=-99.0
            # fix that depot is visitable multiple times
            probs_[:,:,0]=probs[:,:,0]
            
            # update remaining capa
            demands_next_idx=demand[:,next_idx]
            remaining_capa[v]=remaining_capa[v]-demands_next_idx
            
            #update probs after remaining capa change
            condition=(remaining_capa[v].unsqueeze(0).unsqueeze(1).expand(probs_.size(1),probs_.size(1))>=demand.expand(probs_.size(1),probs_.size(1)))
            probs_[v,:,:]=torch.where(condition,probs_[v,:,:],other)
            
            #update curr idxs
            curr_idx= next_idx
    
    return all_idxs, remaining_capa


            
# for ALL BATCHES GET CURR LOAD ESTIMATE (func)
def load_estimate(probs,demand,perm_m):
    probs_=probs.clone()
    b,m,n,_=probs_.size()
    all_idxs=zeros(*(probs.size(0),probs.size(1),probs.size(2),probs.size(2)))
    arr = torch.arange(all_idxs.size(0)).cuda()
    for v in perm_m:
        # get starting batches
        curr_batch_idxs= probs_[:,v,:,:].max(2)[1][:,0]
        all_idxs[arr,v,zeros(*(curr_batch_idxs.size(0),)).long(),curr_batch_idxs]=1.0
        probs_[arr,:,:,curr_batch_idxs]=-99.0
        while (curr_batch_idxs != 0).any():
            #get next idxs
            next_batch_idx=probs_[:,v,:,:].max(2)[1].gather(1,curr_batch_idxs.unsqueeze(1)).squeeze()

            # catch termintated paths
            next_batch_idx=torch.where(curr_batch_idxs==0,curr_batch_idxs,next_batch_idx)

            # update current_idxs Mask
            all_idxs[arr,v,curr_batch_idxs,next_batch_idx]=1.0

            # fix all_idxs for catched terminated paths (0 (curr) --> 0 (next))
            all_idxs[arr,v,0,0]=0.0

            # update probs after chosen
            #probs_2=probs_.clone()
            probs_[arr,:,:,next_batch_idx]=-99.0
            # fix that depot is visitable multiple times
            probs_[arr,:,:,0]=probs[arr,:,:,0]
            
            #update curr idxs
            curr_batch_idxs= next_batch_idx

    loads = (demand.unsqueeze(2).expand(b,m,n,n)*all_idxs).sum(dim=3).sum(dim=2)
    
    return all_idxs, loads


def targ_as_lst(idcs_trg, n):
    '''idcs_trg is a tensor containing the indices of next visited nodes'''
    nxt_idx=[0]
    nxt=idcs_trg[0].item()
    for _ in range(n):
        cur_idx=nxt
        if cur_idx !=0:
            nxt_idx.append(cur_idx)
            nxt=idcs_trg[cur_idx].item()
    nxt_idx.append(0)
    return nxt_idx