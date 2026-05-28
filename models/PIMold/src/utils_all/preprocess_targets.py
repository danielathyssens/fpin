#import libs
import numpy as np
import pandas as pd
import pickle
import torch
from itertools import tee

#functions
def filter_out(Y_dat):

    indices_None = [i for i, x in enumerate(Y_dat) if x == None]
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

    all_target_instances = []
    all_target_loads = []
    # loop over target instances
    for Y in Y_dat:

        # initiate empty array for target tensor
        target_arr = np.zeros((m, n, n))
        # initiate empty array for target acc. demands
        target_accD = np.zeros((m))
        # loop over vehicles in Y (one target instance)
        for j in range(m):
            # get sequence of visited customers
            sequence_of_customers = Y[j][0][0]
            # declare 1 for path from customer_k to customer_l
            for k, l in pairwise(sequence_of_customers):
                target_arr[j, k, l] = 1

            # GET ACC. DEMANDS FOR AUX_LOSS
            acc_demand = Y[j][0][1]
            target_accD[j] = max(acc_demand) / capa
            #
        all_target_instances.append(target_arr)
        all_target_loads.append(target_accD)

    return all_target_instances, all_target_loads

#import data
# nr_of_files=22
#path='data/VRP50_new/OR_data_files/'

def import_and_filter(nr_of_files,path,vrp_size):
    Y_dat_all = []
    for i in range(0, nr_of_files):
        # load data
        infile_ = open(path + 'Y_vrp'+vrp_size+'_lst_new' + str(i) + '.pkl', 'rb')
        # instance i, vehi_id=[0,1,2,3,4]
        # Y_dat[i][vehi_id][0] = [[cust_ids],[load]]
        # Y_dat[i][vehi_id][1] = capa,
        # Y_dat[i][vehi_id][2]= cost or dist (m)
        y_dat = pickle.load(infile_)
        infile_.close()
        y_dat_filtered = filter_out(y_dat)
        Y_dat_all.extend(y_dat_filtered)
    return Y_dat_all

# Press the green button in the gutter to run the script.
#if __name__ == '__main__':
#    Y_data = import_and_filter()

#