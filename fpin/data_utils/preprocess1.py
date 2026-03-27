import os
import hydra
import numpy as np
# import pickle
import pickle as pickle
from scipy.spatial import distance_matrix
import torch
from itertools import tee
import tracemalloc
from typing import Dict, Union, List, NamedTuple, Tuple, Any, Optional

# from formats import CVRPInstance
printt = True

RP_SPECS = {20: [5, 21, 30],
            50: [10, 51, 40],
            60: [7, 61, 40],
            100: [13, 101, 50]
            }

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


def transform_Xs(X_dat, X_dat_kool, is_normalised=False, nr_of_data=114895):
    nr_of_data = len(X_dat)
    X_groups20_new_2d = []  # list of groups (training instances)

    for i in range(nr_of_data):
        n_ = np.arange(1, X_dat[i]['num_vehicles'] + 1)
        vehicle_ids = np.arange(len(n_))  # 0-based indexing for embedding  # vehicle_ids,
        vehicle_tuples = list(zip(vehicle_ids, [(x / n_[-1]) for x in n_], [n_[-1]] * n_[-1],
                                  X_dat[i]['vehicle_capacities'],
                                  [sum(np.array(X_dat[i]['demands']))
                                   / X_dat[i]['vehicle_capacities'][0]] * n_[-1]))
        vehicle_gr = np.vstack([np.asarray(x) for x in vehicle_tuples])
        vehicle_gr[:, -2] = vehicle_gr[:, -2] / X_dat[i]['vehicle_capacities'][0]

        # DEMANDS
        X_dat[i]['demands'] = np.concatenate(([0], X_dat[i]['demands'])) if X_dat[i]['demands'][0] != 0 else X_dat[i]['demands']
        if not is_normalised:
            X_dat[i]['demands'] = [int(i) for i in X_dat[i]['demands']]
            demand_array = np.array(X_dat[i]['demands']) / X_dat[i]['vehicle_capacities'][0]
            demands_unnormed = X_dat[i]['demands']
        else:
            demand_array = np.array(X_dat[i]['demands'])
            demands_unnormed = np.round(np.array(X_dat[i]['demands']) *
                                        (X_dat[i]['vehicle_capacities'][0])).astype(int) #  + 1

        demand_array = demand_array.reshape(1, len(demand_array))
        Depot_centrality = np.array(len(X_dat[i]['coords_plus_dep']) - 1) / np.sum(
            (distance_matrix(X_dat[i]['coords_plus_dep'], X_dat[i]['coords_plus_dep'], p=2))[0]).reshape(-1, 1)

        Depot_gr = np.concatenate((np.array(X_dat_kool[i][0]).reshape(1, len(np.array(X_dat_kool[i][0]))),
                                   np.zeros((1,1)), Depot_centrality), axis=1)
        Customer_gr = np.concatenate(
            (np.array(X_dat_kool[i][1]),
             np.array(X_dat[i]['demands'][1:]).reshape(1, len(X_dat[i]['demands'][1:])).transpose()), axis=1)
        X_groups20_new_2d.append((vehicle_gr, Depot_gr, Customer_gr, demand_array,
                                  distance_matrix(X_dat[i]['coords_plus_dep'],
                                                  X_dat[i]['coords_plus_dep'], p=2),
                                  X_dat[i]['vehicle_capacities'][0], demands_unnormed))

    return X_groups20_new_2d


def transform_Xs_uchoa(X_dat, X_dat_kool, nr_of_data=114895):
    nr_of_data = len(X_dat)
    Dists_lst = []  # list of distance matrices
    X_groups20_new_2d = []  # list of groups (training instances)
    print('Nr. of datapoints to preprocess: ', len(X_dat))

    for i in range(nr_of_data):
        # correct X_dat_kool
        if not all(j < 2.5 for j in [item for sublist in X_dat_kool[i][1] for item in sublist]):
            kool_train_tr = (
                list(np.array(X_dat_kool[i][0]) / 1000), [list(np.array(x) / 1000) for x in X_dat_kool[i][1]],
                X_dat_kool[i][2], X_dat_kool[i][3])
        else:
            kool_train_tr = X_dat_kool[i]
        if i == 1:
            print(kool_train_tr)

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
        if i == 1:
            print('demand_array', demand_array)

        # DEPOT
        Depot_coord = np.array(kool_train_tr[0])
        if i == 1:
            print('Depot_coord', Depot_coord)
        Depot_coord = Depot_coord.reshape(1, len(Depot_coord))
        Depot_centrality = np.array(len(X_dat[i]['distance_matrix']) - 1) / np.sum(
            (X_dat[i]['distance_matrix'] / 1000)[0]).reshape(-1, 1)
        Depot_gr = np.concatenate((Depot_coord, demand_array[:, 0].reshape(-1, 1), Depot_centrality), axis=1)
        if i == 1:
            print('Depot_gr', Depot_gr)

        # CUSTOMER NODES
        Customer_coord = np.array(kool_train_tr[1])
        if i == 1:
            print('Customer_coord', Customer_coord)
        # To_depot_dist = X_dat[i]['distance_matrix'][0] / 1000
        # To_depot_dist = To_depot_dist.reshape(1, len(To_depot_dist))
        # Customer_gr = np.concatenate(
        #    (Customer_coord, demand_array[:, 1:].transpose(), To_depot_dist[:, 1:].transpose()), axis=1)
        # All_dist_mat = X_dat[i]['distance_matrix'] / 1000
        Customer_gr = np.concatenate(
            (Customer_coord, demand_array[:, 1:].transpose()), axis=1)
        if i == 1:
            print('Customer_gr', Customer_gr)

        # Dist_mat
        All_dist_mat = X_dat[i]['distance_matrix'] / 1000

        # Combine to Instances
        X_groups20_new_2d.append((vehicle_gr, Depot_gr, Customer_gr, demand_array,
                                  All_dist_mat, X_dat[i]["vehicle_capacities"][0], X_dat[i]['demands']))

        if i == 1:
            print('Depot_coord', Depot_coord)
            print('X_dat[i]["vehicle_capacities"][0]', X_dat[i]['vehicle_capacities'][0])
            print('Example Input Data Sample:\n', X_groups20_new_2d[i])

    return X_groups20_new_2d


def transform_targets(Y_dat, m=7, n=51, capa=40, is_in_loop=False):
    all_target_indices = []
    all_target_values = []
    all_target_loads = []
    m_idxs = {}
    had = []
    if isinstance(Y_dat, list):
        Y_dat = Y_dat
    else:
        Y_dat = [Y_dat]

    # print(m, n)

    all_target_instances = []
    all_target_loads = []
    count = 0
    # loop over target instances
    for Y in Y_dat:
        capa = Y[0][1]
        m_ = len(Y.keys()) - 1  # number of tours
        # if count in [0, 2, 3, 30]:
        #     print('Y', Y)
        #     print('capa', capa)
        #     print('m', m)
        #     print('nr_tours', m_)
        #     print('n', n)
        # initiate empty array for target tensor
        if n < 100:
            print('n<100')
            target_arr = np.zeros((m, n, n))
            idx_lst = None
        else:
            idx_lst = []
            target_arr = None
        # initiate empty array for target acc. demands
        target_accD = np.zeros((m))
        # loop over vehicles in Y (one target instance)
        for j in range(m):
            # if nr. tours == m (nr. available vehicles)
            try:
                # get sequence of visited customers
                sequence_of_customers = Y[j][0][0]
                # if count in [0, 2, 3, 30]:
                # print('sequence_of_customers', sequence_of_customers)
                # declare 1 for path from customer_k to customer_l
                for k, l in pairwise(sequence_of_customers):
                    if n < 100:
                        target_arr[j, k, l] = 1
                    else:
                        idx_lst.append(torch.tensor([j, k, l]))

                # GET ACC. DEMANDS FOR AUX_LOSS
                acc_demand = Y[j][0][1]
                target_accD[j] = max(acc_demand) / capa
            # if nr. tours < m (nr. available vehicles)
            except KeyError:
                j = m - 1  # (index from 0)
                # dummy empty seq. of customers
                sequence_of_customers = [0, 0]
                for k, l in pairwise(sequence_of_customers):
                    if n < 100:
                        target_arr[j, k, l] = 1
                    else:
                        idx_lst.append(torch.tensor([j, k, l]))
                # GET ACC. DEMANDS FOR AUX_LOSS
                acc_demand = 0
                target_accD[j] = 0  # max(acc_demand) / capa

        if n < 100:
            target_arr_byte = torch.from_numpy(target_arr).byte()  # .to_sparse()
            all_target_instances.append(target_arr_byte)
            target_accD = torch.from_numpy(target_accD)
            all_target_loads.append(target_accD)
            # if count == 0:
            #     print('numpy targets dtype:', target_arr.dtype)
            #     print('torch targets dtype:', target_arr_byte.dtype)
        else:
            all_target_indices.append(torch.stack(idx_lst).transpose(0, 1))
            all_target_values.append(torch.ByteTensor([1]).expand(len(idx_lst)))
            all_target_loads.append(torch.from_numpy(target_accD))
            # if count == 0:
            #     print('all_target_indices[0]', all_target_indices[0])
            #     print('all_target_values[0]:', all_target_values[0])

            # get sorted IDX dict
            # # sort indices based on possible fleet sizes
            # if m_ <= 20:
            #     if m_ in had:
            #         m_idxs[m_].append(idx)
            #     else:
            #         m_idxs[m_] = [idx]
            #         had.append(m_)

        count += 1
    # if is_in_loop:
    #     return torch.stack(all_target_instances), torch.stack(all_target_loads)
    # else:
    if n < 100:
        return all_target_instances, all_target_loads
    else:
        return (all_target_indices, all_target_values), all_target_loads


def transform_targets_uchoa(Y_dat, idx_range, m=7, n=51, capa=40):
    if isinstance(Y_dat, list):
        Y_dat = Y_dat
    else:
        Y_dat = list(Y_dat)
    print('isinstance(Y_dat, list)', isinstance(Y_dat, list))
    all_target_indices = []
    all_target_values = []
    all_target_loads = []
    m_idxs = {}
    had = []
    count = 0
    # loop over target instances
    for idx, Y in zip(idx_range, Y_dat):
        capa = Y[0][1]
        if count == 0:
            print('idx', idx)
            print('Y', Y)
        m_ = len(Y.keys()) - 1
        if count in [1, 2, 3, 30]:
            print('idx', idx)
            print('capa', capa)
            print('m', m)
            print('m_', m_)
            print('n', n)
            # initiate empty array for target indices
        # target_arr = np.zeros((m_, n, n))
        idx_lst = []
        # initiate empty array for target acc. demands
        target_accD = np.zeros((m_))
        # loop over vehicles in Y (one target instance)
        for j in range(m_):
            # get sequence of visited customers
            sequence_of_customers = Y[j][0][0]
            # declare 1 for path from customer_k to customer_l
            for k, l in pairwise(sequence_of_customers):
                # target_arr[j, k, l] = 1
                idx_lst.append(torch.tensor([j, k, l]))
                # GET ACC. DEMANDS FOR AUX_LOSS
            acc_demand = Y[j][0][1]
            target_accD[j] = max(acc_demand) / capa
        all_target_indices.append(torch.stack(idx_lst).transpose(0, 1))
        all_target_values.append(torch.ByteTensor([1]).expand(len(idx_lst)))
        all_target_loads.append(torch.from_numpy(target_accD))
        if count == 0:
            print('len(idx_lst)', len(idx_lst))
            print('all_target_indices[0]', all_target_indices[0])
            print('all_target_values[0]', all_target_values[0])

            # get sorted IDX dict
        # sort indices based on possible fleet sizes
        if m_ <= 20:
            if m_ in had:
                m_idxs[m_].append(idx)
            else:
                m_idxs[m_] = [idx]
                had.append(m_)

        count += 1

    print('idx_range[-1]', idx_range[-1])
    print('m_idxs.keys()', m_idxs.keys())

    return all_target_indices, all_target_values, all_target_loads, m_idxs


def import_and_filter(files, path, vrp_size, uchoa=False):
    print('files to import: ', files)
    X_dat_kool_all, X_dat_all, Y_dat_all = [], [], []
    # if not uchoa:
    begin = 0
    end = files
    for i in range(begin, end):
        if not (vrp_size == '50' and i == 9):
            # load and filter data X (orig)
            infile_1 = open(path + 'X_vrp' + vrp_size + '_lst_new' + str(i) + '.pkl', 'rb')
            # instance i
            x_dat_kool = pickle.load(infile_1)
            infile_1.close()
            # load and filter data X (dict)
            infile_2 = open(path + 'X_vrp' + vrp_size + '_dct_new' + str(i) + '.pkl', 'rb')
            # instance i
            x_dat = pickle.load(infile_2)
            infile_2.close()
            # load and filter data targets
            infile_3 = open(path + 'Y_vrp' + vrp_size + '_lst_new' + str(i) + '.pkl', 'rb')
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
    if isinstance(vrp_solved, int):
        vrp_size = str(vrp_solved)
        vrp_solved = 'VRP' + str(vrp_solved)
    else:
        vrp_size = vrp_solved[3:]
    if vrp_size == '100':
        m, n, capa = 11, 101, 50
    elif vrp_size == '50':
        m, n, capa = 7, 51, 40
    elif vrp_size == '20':
        m, n, capa = 4, 21, 30

    # if not vrp_size=='20':
    path_to_lib_data = os.path.join(hydra.utils.get_original_cwd(), "models/PIM/PIM/data/")
    X_kool_cleaned, X_cleaned, Y_cleaned = import_and_filter(nr_files, path_to_lib_data + vrp_solved
                                                             + '_new/OR_data_files/',
                                                             vrp_size)
    # transform Xs
    dat_X = transform_Xs(X_cleaned, X_kool_cleaned)
    # transform Ys
    dat_Y, dat_YLoad = transform_targets(Y_cleaned, m, n, capa)

    return dat_X, dat_Y, dat_YLoad


def filter_feasible_m(data, capa, m):
    dct_data, filtered_data, too_much_demand = [], [], 0
    for id_, instance in enumerate(data):
        if sum([0] + instance[2]) < (capa * m):

            if len([0] + instance[2]) >= 100:
                dct_data.append({'demands': [0] + instance[2], 'depot': 0,
                                 'coords_plus_dep': np.vstack([instance[0], instance[1]]), # coords_dep
                                 'num_vehicles': instance[-1] if instance[-1] is not None else m,
                                 'vehicle_capacities': [capa] * m})
            else:
                X = np.vstack([instance[0], instance[1]])
                x_test = {'demands': [0] + instance[2], 'depot': 0,
                          'coords_plus_dep': np.vstack([instance[0], instance[1]]),  # coords_dep
                          'distance_matrix': distance_matrix(X, X, p=2),  #  * 1000
                          'num_vehicles': instance[-1] if instance[-1] is not None else m,
                          'vehicle_capacities': [capa] * m}
                dct_data.append(x_test)

        else:
            too_much_demand += 1
            data.pop(id_)
        # print('dct_data[0]', dct_data[0])
    return dct_data, data, too_much_demand, True if too_much_demand == 0 else False


def prep_train_data(rp_size, data, accept_not_solvable=False, type_: str = "uniform", normed_data: bool = False):
    _, n, _ = RP_SPECS[rp_size] # stale
    m = int(data[0][-1])
    capa = int(data[0][-2])

    # create list of transformed dct instances & filter unsolvable
    filtered_dct_data, filtered_data, too_much_demand, can_solve_all = filter_feasible_m(data, capa, m)
    if not accept_not_solvable:
        data = filtered_data
        dct_data = filtered_dct_data
    else:
        # print('constructing dct_data')
        dct_data = [{'demands': [0] + instance[2],
                     'depot': 0,
                     'distance_matrix': distance_matrix(np.array([instance[0]] + instance[1]),
                                                        np.array([instance[0]] + instance[1]), p=2) * 1000,
                     'num_vehicles': m,
                     'vehicle_capacities': [capa] * m} for instance in data]

        print('cannot solve {} instances'.format(too_much_demand))
        print('have {} solvable instances'.format(len(filtered_data)))
        if not can_solve_all:
            print('Have unsolvable instances or full capacitated instances in Test set')

    if type_ != "OG_uchoa":
        current, peak = tracemalloc.get_traced_memory()
        # transform Xs
        return transform_Xs(dct_data, data, is_normalised=normed_data), can_solve_all
    else:
        i = 0
        for instance in data:
            demand_lst = [0] + instance[2]
            cap = instance[3]
            if i == 0:
                print('sum(demand_lst)', sum(demand_lst))

            x_ = [instance[0]] + instance[1]
            X = np.array(x_) / 1000
            x_test = {'demands': demand_lst, 'depot': 0, 'distance_matrix': distance_matrix(X, X, p=2) * 1000,
                      'num_vehicles': m, 'vehicle_capacities': [cap] * m}
            i += 1

        # transform Xs
        dat_x = transform_Xs(dct_data, data, is_normalised=normed_data)

    return dat_x, can_solve_all


def prep_test_data(rp_size,
                   data,
                   accept_not_solvable=True,
                   type_: str = "uniform",
                   normed_data: bool = False,
                   nr_veh: int = None):
    # get handles
    m = int(data[0][-1])
    n = rp_size+1
    capa = int(data[0][-2])

    # create list of transformed dct instances & filter unsolvable
    dct_test = []
    kool_test_filtered = []
    can_solve_all = None
    too_much_demand_test = 0
    dat_x = None
    if not accept_not_solvable:
        dct_test, kool_test_filtered, too_much_demand_test, can_solve_all = filter_feasible_m(data, m, n)
        if not can_solve_all:
            print('Have unsolvable instances or full capacitated instances in Test set')
            print('cannot solve {} instances'.format(too_much_demand_test))

    for i, instance in enumerate(data):
        demand_lst = [0] + instance[2]
        cap = instance[3]
        if i == 0:
            # print('instance', instance)
            print('sum(demand_lst)', sum(demand_lst))
            print('cap', cap)
            print('cap*m', cap * m)
            print('capa', capa)
            # print('instance', instance)
            print('instance[3]', instance[3])

        if sum(demand_lst) >= (capa * m):
            too_much_demand_test += 1
        x_ = np.vstack([instance[0], instance[1]])  #  [instance[0]] + instance[1]
        X = x_
        x_test = {'demands': demand_lst, 'depot': 0, 'distance_matrix': distance_matrix(X, X, p=2) * 1000,
                  'num_vehicles': m, 'vehicle_capacities': [cap] * m, 'coords_plus_dep': x_}
        dct_test.append(x_test)
        kool_test_filtered.append(instance)

    # transform Xs
    dat_x = transform_Xs(dct_test, data, is_normalised=normed_data)
    if too_much_demand_test != 0:
        print('Have unsolvable instances or full capacitated instances in Test set')
        can_solve_all = False
    else:
        can_solve_all = True

    return dat_x, can_solve_all
