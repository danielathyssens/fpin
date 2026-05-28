import numpy as np
import pickle
import torch
from itertools import tee


# help-functions
def filter_out(Y_dat):
    indices_None = [i for i, x in enumerate(Y_dat) if x == None]
    # print('indices_None',indices_None)
    # remove those samples
    Y_dat = [v for i, v in enumerate(Y_dat) if i not in indices_None]

    return Y_dat


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def transform_targets(Y_dat, m=7, n=51, capa=40):
    if isinstance(Y_dat, list):
        Y_dat = Y_dat
    else:
        Y_dat = [Y_dat]

    y_indices=[]
    y_values=[]
    y_sizes=[]
    all_target_instances = []
    count = 0
    # loop over target instances
    for Y in Y_dat:

        # initiate empty array for target tensor
        target_arr = np.zeros((m, n, n))
        # loop over vehicles in Y (one target instance)
        for j in range(m):
            # get sequence of visited customers
            sequence_of_customers = Y[j][0][0]
            # declare 1 for path from customer_k to customer_l
            for k, l in pairwise(sequence_of_customers):
                target_arr[j, k, l] = 1

        target_arr_byte = torch.from_numpy(target_arr).byte()  # .to_sparse()
        target_arr_byte_sp = target_arr_byte.to_sparse()
        i=target_arr_byte_sp.indices()
        v=target_arr_byte_sp.values()
        size=target_arr_byte_sp.size()
        y_indices.append(i)
        y_values.append(v)
        y_sizes.append(size)
        all_target_instances.append(target_arr_byte)
        count += 1

    return y_indices,y_values,y_sizes, all_target_instances


def import_and_filter(nr_of_files, path, vrp_size):
    print('nr of files to import: ', nr_of_files)
    Y_dat_all = []
    for i in range(0, nr_of_files):
        # load and filter data targets
        infile_3 = open(path + 'Y_vrp' + vrp_size + '_lst_new' + str(i) + '.pkl', 'rb')
        y_dat = pickle.load(infile_3)
        infile_3.close()
        y_dat_filtered = filter_out(y_dat)
        Y_dat_all.extend(y_dat_filtered)
    return Y_dat_all


def import_trainval_data(vrp_solved, nr_files):
    # get handles
    vrp_size = vrp_solved[3:]
    if vrp_size == '100':
        m, n, capa = 11, 101, 50
    elif vrp_size == '50':
        m, n, capa = 7, 51, 40
    elif vrp_size == '20':
        m, n, capa = 4, 21, 30

    # if not vrp_size=='20':
    Y_cleaned = import_and_filter(nr_files, 'data/' + vrp_solved + '_new/OR_data_files/', vrp_size)

    # transform Ys
    y_idx, y_val, y_size, target_instances = transform_targets(Y_cleaned, m, n, capa)
    
    #for g in range(len(y_idx)):
    #    
    #    if y_idx[g].size()[1] != 111:
    #            print('y_idx[g].size()',y_idx[g].size())
    #            print('y_val[g].size()',y_val[g].size())
    
    print('saving target indices..')
    with open('/content/gdrive/My Drive/Supervised_VRP/targets_i.pt', "wb") as f:
        torch.save(y_idx, f)
    print('saving target values..')
    with open('/content/gdrive/My Drive/Supervised_VRP/targets_v.pt', "wb") as f:
        torch.save(y_val, f)
    print('saving target shapes..')
    with open('/content/gdrive/My Drive/Supervised_VRP/targets_s.pt', "wb") as f:
        torch.save(y_size, f)
    #with open('/content/gdrive/My Drive/Supervised_VRP/targets_dense.pt', "wb") as f:
    #    torch.save(target_instances, f)

    #return dat_Y, dat_YLoad
