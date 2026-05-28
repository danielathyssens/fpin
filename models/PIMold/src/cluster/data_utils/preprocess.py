import numpy as np
import pickle
from scipy.spatial import distance_matrix
import torch
from itertools import tee


# help-functions
def filter_out(X_dat, X_dat_kool, Y_dat):
    indices_None = [i for i, x in enumerate(Y_dat) if x == None]
    # print('indices_None',indices_None)
    # remove those samples
    X_dat = [v for i, v in enumerate(X_dat) if i not in indices_None]
    X_dat_kool = [v for i, v in enumerate(X_dat_kool) if i not in indices_None]
    Y_dat = [v for i, v in enumerate(Y_dat) if i not in indices_None]

    return X_dat, X_dat_kool, Y_dat


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def transform_Xs(X_dat, X_dat_kool, nr_of_data=114895):
    nr_of_data = len(X_dat)
    Dists_lst = []  # list of distance matrices
    X_groups20_new_2d = []  # list of groups (training instances)
    print('Nr. of datapoints to preprocess: ', len(X_dat))

    for i in range(nr_of_data):
        # VEHICLE FLEET
        n_ = np.arange(1, X_dat[i]['num_vehicles'] + 1)
        vehicle_tuples = list(zip([(x / n_[-1]) for x in n_], [n_[-1]] * n_[-1],
                                  X_dat[i]['vehicle_capacities'],
                                  [sum(np.array(X_dat[i]['demands'])) / X_dat[i]['vehicle_capacities'][0]] * n_[-1]))
        vehicle_arrays = [np.asarray(x) for x in vehicle_tuples]
        vehicle_gr = np.vstack(vehicle_arrays)
        vehicle_gr[:, -2] = vehicle_gr[:, -2] / X_dat[i]['vehicle_capacities'][0]

        # DISTANCE MATRIX
        Dists_lst.append(X_dat[i]['distance_matrix'] / 1000)

        # DEMANDS
        demand_array = np.array(X_dat[i]['demands']) / X_dat[i]['vehicle_capacities'][0]
        demand_array = demand_array.reshape(1, len(demand_array))

        # DEPOT
        Depot_coord = np.array(X_dat_kool[i][0])
        Depot_coord = Depot_coord.reshape(1, len(Depot_coord))
        Depot_centrality = np.array(len(X_dat[i]['distance_matrix']) - 1) / np.sum(
            (X_dat[i]['distance_matrix'] / 1000)[0]).reshape(-1, 1)
        Depot_gr = np.concatenate((Depot_coord, demand_array[:, 0].reshape(-1, 1), Depot_centrality), axis=1)

        # CUSTOMER NODES
        Customer_coord = np.array(X_dat_kool[i][1])
        To_depot_dist = X_dat[i]['distance_matrix'][0] / 1000
        To_depot_dist = To_depot_dist.reshape(1, len(To_depot_dist))
        Customer_gr = np.concatenate(
            (Customer_coord, demand_array[:, 1:].transpose(), To_depot_dist[:, 1:].transpose()), axis=1)
        All_dist_mat = X_dat[i]['distance_matrix'] / 1000
        # Combine to Instances
        X_groups20_new_2d.append((vehicle_gr, Depot_gr, Customer_gr, demand_array,
                                  All_dist_mat))

        if i == 1:
            print('Example Input Data Sample:\n', X_groups20_new_2d[i])

    return X_groups20_new_2d


def transform_targets(Y_dat, m=7, n=51, capa=40):
    if isinstance(Y_dat, list):
        Y_dat = Y_dat
    else:
        Y_dat = [Y_dat]

    all_target_instances = []
    all_target_loads = []
    count = 0
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

        target_arr_byte = torch.from_numpy(target_arr).byte()  # .to_sparse()
        target_accD = torch.from_numpy(target_accD)
        if count == 0:
            print('numpy targets dtype:', target_arr.dtype)
            print('torch targets dtype:', target_arr_byte.dtype)
        all_target_instances.append(target_arr_byte)
        all_target_loads.append(target_accD)
        count += 1

    return all_target_instances, all_target_loads


def import_and_filter(nr_of_files, path, vrp_size):
    print('nr of files to import: ', nr_of_files)
    X_dat_kool_all, X_dat_all, Y_dat_all = [], [], []
    for i in range(0, nr_of_files):
        if not (vrp_size=='50' and i==9):
            # load and filter data X (orig)
            infile_1 = open(path + 'X_vrp' + vrp_size + '_lst_new' + str(i) + '.pkl', 'rb')
            # instance i
            # X_dat_kool[i][0]= depot loc, X_dat_kool[i][1]= node locs,
            # X_dat_kool[i][2]= demand, X_dat_kool[i][3]= capa
            x_dat_kool = pickle.load(infile_1)
            infile_1.close()
            # load and filter data X (dict)
            infile_2 = open(path + 'X_vrp' + vrp_size + '_dct_new' + str(i) + '.pkl', 'rb')
            # instance i
            # X_dat[i]['distance_matrix']=arr, X_dat[i]['num_vehicles']=int,
            # X_dat[i]['depot']= 0, X_dat[i]['demands']=lst, 
            # X_dat[i]['vehicle_capacities']=lst
            x_dat = pickle.load(infile_2)
            infile_2.close()
            # load and filter data targets
            infile_3 = open(path + 'Y_vrp' + vrp_size + '_lst_new' + str(i) + '.pkl', 'rb')
            # instance i, vehi_id=[0,1,2,3,4]
            # Y_dat[i][vehi_id][0] = [[cust_ids],[load]]
            # Y_dat[i][vehi_id][1] = capa,
            # Y_dat[i][vehi_id][2]= cost or dist (m)
            y_dat = pickle.load(infile_3)
            infile_3.close()
            x_dat_filtered, x_dat_kool_filtered, y_dat_filtered = filter_out(x_dat, x_dat_kool, y_dat)
            X_dat_all.extend(x_dat_filtered)
            X_dat_kool_all.extend(x_dat_kool_filtered)
            Y_dat_all.extend(y_dat_filtered)
        else:
            print("don't import file with id 9 for vrp50 --> corrupted")
    return X_dat_kool_all, X_dat_all, Y_dat_all


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
    X_kool_cleaned, X_cleaned, Y_cleaned = import_and_filter(nr_files, 'data/' + vrp_solved + '_new/OR_data_files/',
                                                             vrp_size)
    # transform Xs
    dat_X = transform_Xs(X_cleaned, X_kool_cleaned)
    # transform Ys
    dat_Y, dat_YLoad = transform_targets(Y_cleaned, m, n, capa)
    # else:
    #    dat_X=torch.load('data/VRP20_new/data_X.pt')
    #    dat_Y=torch.load('data/VRP20_new/data_Y.pt')
    #    dat_YLoad=torch.load('data/VRP20_new/data_YLoad.pt')

    return dat_X, dat_Y, dat_YLoad


def import_test_data(vrp_solved, test_data_path, accept_not_solvable=True):
    # get handles
    vrp_size = vrp_solved[3:]
    if vrp_size == '100':
        m, n, capa = 11, 101, 50
    elif vrp_size == '50':
        m, n, capa = 7, 51, 40
    elif vrp_size == '20':
        m, n, capa = 4, 21, 30
        # import original Kool et al Test Data
        # infile_kool_extra = open(test_data_path_extra, 'rb')
        # kool_test_extra = pickle.load(infile_kool_extra)
        # infile_kool_extra.close()

    # import original Kool et al Test Data
    infile_kool = open(test_data_path, 'rb')
    kool_test = pickle.load(infile_kool)
    infile_kool.close()

    # if vrp_size == '20':
    #    kool_test.extend(kool_test_extra)

    # create list of transformed dct instances & filter unsolvable
    dct_test = []
    kool_test_filtered = []
    if not accept_not_solvable:
        too_much_demand_test = 0
        for instance in kool_test:
            demand_lst = [0] + instance[2]
            if sum(demand_lst) < (capa*m):
                x_ = [instance[0]] + instance[1]
                X = np.array(x_)
                x_test = {'demands': demand_lst, 'depot': 0, 'distance_matrix': distance_matrix(X, X, p=2) * 1000,
                          'num_vehicles': m, 'vehicle_capacities': [capa] * m}
                dct_test.append(x_test)
                kool_test_filtered.append(instance)
            else:
                too_much_demand_test += 1

        print('cannot solve {} instances'.format(too_much_demand_test))
        print('have {} original instances'.format(len(kool_test_filtered)))
        print('have {} dict instances'.format(len(dct_test)))

    else:
        for instance in kool_test:
            demand_lst = [0] + instance[2]
            x_ = [instance[0]] + instance[1]
            X = np.array(x_)
            x_test = {'demands': demand_lst, 'depot': 0, 'distance_matrix': distance_matrix(X, X, p=2) * 1000,
                      'num_vehicles': m, 'vehicle_capacities': [capa] * m}
            dct_test.append(x_test)
            kool_test_filtered.append(instance)

        print('have {} original instances'.format(len(kool_test_filtered)))
        print('have {} dict instances'.format(len(dct_test)))

    # transform Xs
    dat_x = transform_Xs(dct_test, kool_test)

    return dat_x
