
import torch
from utils_all.basic_funcs import percentile,zeros

def targ_as_lst(idcs_trg,n):
    '''idcs_trg is a tensor containing the indices of next visited nodes'''
    nxt_idx=[0]
    #nxt_idx=[idcs_trg[0].item()]
    nxt=idcs_trg[0].item()
    for _ in range(n):
        cur_idx=nxt
        #print(cur_idx)
        if cur_idx !=0:
            nxt_idx.append(cur_idx)
            #(idcs_trg==cur_idx).nonzero().item()
            nxt=idcs_trg[cur_idx].item()
            #print(nxt_idx)
    nxt_idx.append(0)
    #print(nxt_idx)
    return nxt_idx

def print_func(VRP_probs,sample_path_b0,VRP_loads,targets,target_loads,with_loads,vrp50,v5):
    
    # Recompute thresh for nodes to be likely to be visited
    #thresh=percentile(VRP_probs.max(3)[0],75)
    n=VRP_probs.size(2)
    # targets[0]=\
    # targets[0].to_dense()
    print(n)
    if with_loads:
        #print('\nCAPA VIOLATION')
        #print('OLD TOTAL Batch PENALTY',torch.where(VRP_loads>=1.00001,torch.square(VRP_loads),zeros(1)).sum())
              #torch.abs(torch.log(1 + VRP_violation)).sum())
        #print('\nBatch Penalties[:5]\n',torch.abs(torch.log(1 + VRP_violation))[:5])
        print('\nLOADS')
        print('predic demand_loads[0]',VRP_loads[0])
        print('target demand_loads[0]',target_loads[0])
        #print('\nPlain VRP_violation[0]',VRP_violation[0])  
    
    print('\nGREEDY PATH')
    #print('threshold:',thresh)
    #print('nodes visited (most likely) v0',torch.where(VRP_probs[0][0].max(1)[0]>=thresh,VRP_probs[0][0].max(1)[1],zeros(1,type='long')).unique(sorted=False))
    print('CURR PRED CONSECUTIVE PATH  v0',sample_path_b0[0])
    print('TARGET CONSECUTIVE PATH for v0',targ_as_lst(targets[0][0].to_dense().max(1)[1],n))
    #print('nodes visited (most likely) v1', torch.where(VRP_probs[0][1].max(1)[0]>=thresh,VRP_probs[0][1].max(1)[1],zeros(1,type='long')).unique(sorted=False))
    print('CURR PRED CONSECUTIVE PATH  v1',sample_path_b0[1])
    print('TARGET CONSECUTIVE PATH for v1',targ_as_lst(targets[0][1].to_dense().max(1)[1],n))
    #print('nodes visited (most likely) v2',torch.where(VRP_probs[0][2].max(1)[0]>=thresh,VRP_probs[0][2].max(1)[1],zeros(1,type='long')).unique(sorted=False))
    print('CURR PRED CONSECUTIVE PATH  v2',sample_path_b0[2])
    print('TARGET CONSECUTIVE PATH for v2',targ_as_lst(targets[0][2].to_dense().max(1)[1],n))
    #print('nodes visited (most likely) v3',torch.where(VRP_probs[0][3].max(1)[0]>=thresh,VRP_probs[0][3].max(1)[1],zeros(1,type='long')).unique(sorted=False))
    print('CURR PRED CONSECUTIVE PATH  v3',sample_path_b0[3])
    print('TARGET CONSECUTIVE PATH for v3',targ_as_lst(targets[0][3].to_dense().max(1)[1],n))
    #print('nodes visited (most likely) v4',torch.where(VRP_probs[0][4].max(1)[0]>=thresh,VRP_probs[0][4].max(1)[1],zeros(1,type='long')).unique(sorted=False))
    if v5:
        print('CURR PRED CONSECUTIVE PATH  v4',sample_path_b0[4])
        print('TARGET CONSECUTIVE PATH for v4',targ_as_lst(targets[0][4].to_dense().max(1)[1],n))
        #print('nodes visited (most likely) v5',torch.where(VRP_probs[0][5].max(1)[0]>=thresh,VRP_probs[0][5].max(1)[1],zeros(1,type='long')).unique(sorted=False))
        
    if vrp50:
        print('CURR PRED CONSECUTIVE PATH  v4',sample_path_b0[4])
        print('TARGET CONSECUTIVE PATH for v4',targ_as_lst(targets[0][4].to_dense().max(1)[1],n))
        #print('nodes visited (most likely) v5',torch.where(VRP_probs[0][5].max(1)[0]>=thresh,VRP_probs[0][5].max(1)[1],zeros(1,type='long')).unique(sorted=False))
        print('CURR PRED CONSECUTIVE PATH  v5',sample_path_b0[5])
        print('TARGET CONSECUTIVE PATH for v5',targ_as_lst(targets[0][5].to_dense().max(1)[1],n))
        #print('nodes visited (most likely) v6',torch.where(VRP_probs[0][6].max(1)[0]>=thresh,VRP_probs[0][6].max(1)[1],zeros(1,type='long')).unique(sorted=False))
        print('CURR PRED CONSECUTIVE PATH  v6',sample_path_b0[6])
        print('TARGET CONSECUTIVE PATH for v6',targ_as_lst(targets[0][6].to_dense().max(1)[1],n))
        #print('target nodes visisted v6 ---',targ_as_lst(targets[0][6].max(1)[1]))