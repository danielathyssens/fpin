import torch
from basic_funcs import zeros,percentile

def targ_as_lst(idcs_trg):
    '''idcs_trg is a tensor containing the indices of next visited nodes'''
    nxt_idx=[0]
    #nxt_idx=[idcs_trg[0].item()]
    nxt=idcs_trg[0].item()
    for _ in range(21):
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


def print_func(VRP_probs,VRP_loads,targets,target_loads,VRP_violation,with_loads):
    
    # Recompute thresh for nodes to be likely to be visited
    thresh=0.1
    #percentile(VRP_probs.max(2)[0],75)
    #VRP_probs.max(2)[0].mean(2)
    #percentile(VRP_probs.max(2)[0],75)
    
    if with_loads:
        print('\nCAPA VIOLATION')
        print('TOTAL Batch VIOLATION',VRP_violation.sum())
        print('TOTAL Batch PENALTY',torch.abs(torch.log(1 + VRP_violation)).sum())
        print('\nBatch Penalties[:5]\n',torch.abs(torch.log(1 + VRP_violation))[:5])
        print('\nLOADS')
        print('predic demand_loads[0]',VRP_loads[0])
        print('target demand_loads[0]',target_loads[0])
        #print('\nPlain VRP_violation[0]',VRP_violation[0])  
    
    print('\nNODES VISITED')
    print('threshold:',thresh)
    print('nodes visited (most likely) v0',torch.cat(torch.where(torch.where(VRP_probs[0][0].max(1)[0].unsqueeze(-1) == VRP_probs[0][0].max(0)[0].expand(21,21),
                                                                             VRP_probs[0][0].max(1)[0].unsqueeze(-1),zeros(1))>thresh)).unique())
    print('target nodes visisted v0 ---',targ_as_lst(targets[0][0].max(1)[1]))
    #print('threshold vehicle 1:',thresh[0][1])
    print('nodes visited (most likely) v1',torch.cat(torch.where(torch.where(VRP_probs[0][1].max(1)[0].unsqueeze(-1) == VRP_probs[0][1].max(0)[0].expand(21,21),
                                                                             VRP_probs[0][1].max(1)[0].unsqueeze(-1),zeros(1))>thresh)).unique())
    print('target nodes visisted v1 ---',targ_as_lst(targets[0][1].max(1)[1]))
    #print('threshold vehicle 2:',thresh[0][2])
    print('nodes visited (most likely) v2',torch.cat(torch.where(torch.where(VRP_probs[0][2].max(1)[0].unsqueeze(-1) == VRP_probs[0][2].max(0)[0].expand(21,21),
                                                                             VRP_probs[0][2].max(1)[0].unsqueeze(-1),zeros(1))>thresh)).unique())
    print('target nodes visisted v2 ---',targ_as_lst(targets[0][2].max(1)[1]))
    #print('threshold vehicle 3:',thresh[0][3])
    print('nodes visited (most likely) v3',torch.cat(torch.where(torch.where(VRP_probs[0][3].max(1)[0].unsqueeze(-1) == VRP_probs[0][3].max(0)[0].expand(21,21),
                                                                             VRP_probs[0][3].max(1)[0].unsqueeze(-1),zeros(1))>thresh)).unique())
    print('target nodes visisted v3 ---',targ_as_lst(targets[0][3].max(1)[1]))