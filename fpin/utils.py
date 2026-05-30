import os
import glob
import tracemalloc
import torch
import numpy as np
import time

import random
import warnings
from typing import List, Union, Tuple, Optional, Sequence
from omegaconf import DictConfig
from formats import CVRPInstance, RPSolution
import logging

from fpin.utils_all.get_path import greedy_path, make_valid
from fpin.utils_all.postprocessing import create_data_model, main
from fpin.data_utils.preprocess1 import transform_targets, prep_train_data, pairwise
from data.cvrp_dataset import CVRPDataset
from torch.utils.data import DataLoader
from fpin.utils_all.train_utils import load_ckp, transform_targets_to_dct
from formats import RPSolution

logger = logging.getLogger(__name__)

EPS = np.finfo(np.float32).eps


### TRAIN ###
def get_dataset(opts,
                rp_data_class: CVRPDataset = None,
                env_cfg: DictConfig = None,
                fixed_train_targets: List[RPSolution] = None,
                train_dat_load: str = "load_in_get_dataset",
                v_max=None,
                nr_datapoints=None  # optionally reduce training data size manually (for check)
                ):

    if opts.train_data == "use_local":
        print('train_dat_load', train_dat_load)
        if train_dat_load == "load_in_get_dataset":
            logger.info('loading local training data...')
            print('os.path.isfile(opts.train_dataset)', os.path.isfile(opts.train_dataset))
            print('opts.train_dataset', opts.train_dataset)
            if os.path.isfile(opts.train_dataset):
                if not isinstance(opts.train_dataset, dict):
                    rp_solution_data = np.load(opts.train_dataset)
                else:
                    local_solutions = torch.load(opts.train_dataset)
                    rp_solution_data = local_solutions["solutions"]
            elif os.path.isdir(opts.train_dataset):
                graph_size = str(env_cfg.sampling_args.graph_size)
                rp_solution_data = []
                if graph_size == '100':
                    current, peak = tracemalloc.get_traced_memory()
                    print(f"Current memory usage is {current / 10 ** 6}MB; Peak was {peak / 10 ** 6}MB")
                    for filename in os.listdir(opts.train_dataset):
                        if "targets" + graph_size in filename:
                            logger.info(f"loading {filename}...")
                            if filename.endswith(".npz"):
                                rp_solution_data.extend(np.load(os.path.join(opts.train_dataset, filename))[
                                    :nr_datapoints])
                            elif filename.endswith(".pkl"):
                                rp_solution_data.extend(
                                    torch.load(os.path.join(opts.train_dataset, filename))["solutions"][
                                    :nr_datapoints])  # [:20000]
                            else:
                                logger.info(f"Unknown file type{filename.split('.')[-1]}...")
                                raise NotImplementedError
                            current, peak = tracemalloc.get_traced_memory()
                            print(f"Current memory usage is {current / 10 ** 6}MB; Peak was {peak / 10 ** 6}MB")

                else:
                    for filename in os.listdir(opts.local_target_path):
                        if "s" + graph_size in filename:
                            logger.info(f"loading {filename}...")
                            if filename.endswith(".npz"):
                                rp_solution_data.extend(np.load(os.path.join(opts.train_dataset, filename)))
                            elif filename.endswith(".pkl"):
                                rp_solution_data.extend(
                                    torch.load(os.path.join(opts.train_dataset, filename))["solutions"])
                            else:
                                logger.info(f"Unknown file type{filename.split('.')[-1]}...")
                                raise NotImplementedError
            else:
                rp_solution_data = []
                warnings.warn("Unknow target data directory... ")
        elif train_dat_load == "load_in_Dataset":
            rp_solution_data = None
            # Load all npz data paths
            all_data = []
            folder = opts.train_dataset
            fleet_size = int(opts.fleet_size)
            graph_size_str = str(opts.graph_size)
            # Accept both legacy ("targets50_seed...") and HQ
            # ("targets50_m7_seed...size150000_hgs_t3.npz") naming. The old
            # filter "s<N>" matched neither HQ form ("size150000" does not
            # contain "s50") nor the FC-CVRP per-cell dirs.
            name_hint = "targets" + graph_size_str

            # Iterate over all .npz files in the folder
            for path in glob.glob(os.path.join(folder, "*.npz")):
                filename = os.path.basename(path)
                print('name_hint', name_hint)
                print('filename', filename)
                if name_hint not in filename:
                    continue
                data = np.load(path, allow_pickle=True)
                # Required HQ keys; bail with a clear message otherwise.
                missing = [k for k in ("depots", "locs", "demands",
                                       "capacities", "vehicle_limits",
                                       "solutions") if k not in data.files]
                if missing:
                    warnings.warn(
                        f"Skipping {path}: missing HQ keys {missing}; "
                        f"available={list(data.files)}"
                    )
                    continue
                solutions = data["solutions"]
                n_items = len(data["depots"])
                size = n_items if nr_datapoints is None else min(int(nr_datapoints), n_items)
                lengths = np.fromiter((len(sol) for sol in solutions[:size]), dtype=np.int32)
                valid_indices = np.where(lengths <= fleet_size)[0]
                print('fleet_size', fleet_size)
                print('valid_indices.shape', valid_indices.shape)
                print('ret_k = len(valid_indices) / size', len(valid_indices) / max(1, size))
                all_data.extend((path, i) for i in valid_indices)
            print('len(all_data)', len(all_data))
            if len(all_data) == 0:
                raise ValueError(
                    f"No training instances were loaded from {folder}. "
                    f"Checked filter='{name_hint}', fleet_size={fleet_size}."
                )
            print('all_data[0]', all_data[0])
            if len(all_data) > 4:
                print('all_data[4]', all_data[4])

        else:
            rp_solution_data = fixed_train_targets

        # if want to do the whole preprocessing pipeline (for RPSolutions list a priori)
        if isinstance(rp_solution_data, list):
            logger.info("transforming targets before torch.Dataset call...")
            print('rp_solution_data[0]', rp_solution_data[0])
            data_Y, data_Y_load, v_max = transform_local_targets(rp_solution_data, env_cfg)

            # train_data = [rp_solution_data[i].instance for i in range(len(rp_solution_data))]
            if rp_solution_data[0].instance is not None:
                data_kool = prep_data_fpin([rp_solution_data[i].instance for i in range(len(rp_solution_data))], v_max)
            else:
                data_kool = prep_data_fpin([rp_solution_data[i].problem for i in range(len(rp_solution_data))], v_max)
            current, peak = tracemalloc.get_traced_memory()
            print(f"Current memory usage after prep_data_pim is {current / 10 ** 6}MB; Peak was {peak / 10 ** 6}MB")
            print('opts.normalize_data', opts.normalize_data)
            print('data_kool[0]', data_kool[0])
            data_X, all_solvable = prep_train_data(env_cfg.sampling_args.graph_size,
                                                   data_kool,
                                                   type_=env_cfg.generator_args.coords_sampling_dist,
                                                   normed_data=opts.normalize_data)
            current, peak = tracemalloc.get_traced_memory()
            print(f"Current memory usage AFTER prep_train_data is {current / 10 ** 6}MB; Peak was {peak / 10 ** 6}MB")
        else:
            # do prep from npz files in datclass function and before act. train loop
            data_X, data_Y, data_Y_i, data_Y_v, data_Y_load, v_max = None, None, None, None, None, None

    else:
        # elif opts.hgs_target_path is None:
        if rp_data_class is not None:
            data_class_ = rp_data_class
        else:
            data_class_ = CVRPDataset(generator_args=env_cfg.generator_args,
                                      sampling_args=env_cfg.sampling_args,
                                      is_train=True)
        train_data = data_class_.sample(sample_size=opts.epoch_size,
                                        graph_size=env_cfg.sampling_args.graph_size, )
        #                               sub_samples=env_cfg.sampling_args.sub_samples)
        # val_data = data_class_.sample(sample_size=opts.val_size,
        #                               graph_size=opts.graph_size,
        #                               distribution=opts.distribution)

        # get pim X_data
        data_kool = prep_data_fpin(train_data)
        data_X, all_solvable = prep_train_data(env_cfg.sampling_args.graph_size, data_kool,
                                               type_=env_cfg.generator_args.coords_sampling_dist,
                                               normed_data=opts.normalize_data)

    if data_X is not None:
        # shuffle training instances
        if isinstance(data_Y, list):
            c = list(zip(data_X, data_Y, data_Y_load))
            random.shuffle(c)
            data_X, data_Y, data_YLoad = zip(*c)
            data_Y_i, data_Y_v = None, None
        else:
            c = list(zip(data_X, data_Y[0], data_Y[1], data_Y_load))
            random.shuffle(c)
            data_X, data_Y_i, data_Y_v, data_YLoad = zip(*c)
        tr = len(data_X) - int(len(data_X) * 0.2)  # TODO: ATTENTION: HARD CODING
        # Train Test Partition
        partition = {
            'train': [i for i in range(tr)],
            'val': [i for i in range(tr, len(data_X))]
        }
        # Generators
        if data_Y_i is not None:
            print('INTO DATASETVRP100...')
            train_set = DatasetVrp100(partition['train'], data_X, data_Y_i, data_Y_v, data_YLoad,
                                      vehicles_available=v_max,
                                      train_w_HGS=opts.HGS_on_the_fly)
            val_set = DatasetVrp100(partition['val'], data_X, data_Y_i, data_Y_v, data_YLoad)
            print('AFTER DATASETVRP100...')
        else:
            train_set = DatasetVrp(partition['train'], data_X, data_Y, data_YLoad, vehicles_available=v_max,
                                   train_w_HGS=opts.HGS_on_the_fly)
            val_set = DatasetVrp(partition['val'], data_X, data_Y, data_YLoad)

    else:
        # set up the torch.Dataset class with lazy dataloading, on-the-fly preprocessing
        # and caching npz files to avoid repeated disk I/O
        # Shuffle and split
        random.shuffle(all_data)
        tr = len(all_data) - int(len(all_data) * opts.val_split)
        train_indices = all_data[:tr]
        val_indices = all_data[tr:]

        # Use appropriate dataset class depending on opts / problem size
        train_set = DatasetVrpFromNPZ(train_indices, opts)
        val_set = DatasetVrpFromNPZ(val_indices, opts)

    return train_set, val_set


def transform_local_targets(local_solutions, env_cfg):
    targets_all, targets_i_all, targets_v_all, target_loads_all = [], [], [], []
    first_sol_inst = local_solutions[0].instance if local_solutions[0].instance is not None \
        else local_solutions[0].problem
    max_vs = np.max([len(solution_dct.solution) for solution_dct in local_solutions
                                  if solution_dct.solution is not None])
    print('first_sol_inst', first_sol_inst)
    vehicle_upper_bound = first_sol_inst.max_num_vehicles
    print('max_vs', max_vs)
    print('vehicle_upper_bound', vehicle_upper_bound)
    print('len(local_solutions)', len(local_solutions))
    for i, solution in enumerate(local_solutions):
        # current, peak = tracemalloc.get_traced_memory()
        # print(f"Current memory usage during transform targets is {current / 10 ** 6}MB; Peak was {peak / 10 ** 6}MB")

        if solution.solution is not None:
            # if i in [0, 2, 3, 30]:
            #     print('solution', solution)
            #     print('solution.solution', solution.solution)
            # print('transforming to SupVRP-targets in batch loop ...')
            solution_tuple = solution[0] if isinstance(solution, list) else solution
            solution_instance = solution_tuple.instance if solution_tuple.instance is not None else solution_tuple.problem
            targets_, target_loads_ = transform_targets(
                transform_targets_to_dct(solution_tuple, env_cfg.generator_args.weights_sampling_dist),
                m=max_vs,
                n=solution_instance.graph_size,
                capa=solution_instance.original_capacity if
                env_cfg.generator_args.weights_sampling_dist in
                ["random_int", "uchoa"]
                else solution_instance.vehicle_capacity)
            # print('type(targets_)', type(targets_))
            # print('len(targets_)', len(targets_))
            if i in [0, 30]:
                print('targets_[0][0]', targets_[0][0])
                print('targets_[1][0]', targets_[1][0])
            if env_cfg.sampling_args.graph_size < 100:
                targets_all.extend(targets_)
                target_loads_all.extend(target_loads_)
            else:
                # print('targets_[0][0]',targets_[0][0])
                # print('targets_[1]', targets_[1])
                targets_i_all.extend(targets_[0])
                targets_v_all.extend(targets_[1])
                target_loads_all.extend(target_loads_)
            # if i in [0, 2, 3, 30]:
            #     print('targets_[0].size()', targets_[0].size())
            #     print('targets_[0]', targets_[0])
            #     print('target_loads_', target_loads_)
    if env_cfg.sampling_args.graph_size < 100:
        print('len(targets_all)', len(targets_all))
        return targets_all, target_loads_all, max_vs
    else:
        print('nr_train_insts:', len(targets_i_all))
        print('len(targets_i_all[0])', len(targets_i_all[0]))
        return (targets_i_all, targets_v_all), target_loads_all, max_vs


class DatasetVrpFromNPZ(torch.utils.data.Dataset):
    """VRP Dataset using:
    - Lazy dataloading
    - Global in-RAM caching via GlobalCache
    - On-the-fly preprocessing (prep_X, prep_Y)
    """

    def __init__(self, data_index_list, opts):
        self.data_index_list = data_index_list  # list of (file_path, inner_idx)
        self.opts = opts

    def __len__(self):
        return len(self.data_index_list)

    def __getitem__(self, idx):
        file_path, inner_idx = self.data_index_list[idx]
        file_path = os.path.abspath(file_path)
        # print(f"[GETITEM] idx={idx}, file={file_path}, inner={inner_idx}")

        t0 = time.time()
        data = GlobalCache.get(file_path)
        # print(f"time for loading in getitem {time.time() - t0:.6f}s")

        # Extract VRP instance
        depot = data["depots"][inner_idx]
        locs = data["locs"][inner_idx]
        demands = data["demands"][inner_idx]
        capacity = data["capacities"][inner_idx]
        solution = data["solutions"][inner_idx]
        vehicle_limit = data["vehicle_limits"][inner_idx]
        vehicle_limit = vehicle_limit if not self.opts.fleet_size else self.opts.fleet_size
        # len(solution) >= vehicle_limit else len(solution)
        # print('vehicle_limit', vehicle_limit)
        capa_util = data["utilized_capacities"][inner_idx]

        instance = [(depot, locs, demands, capacity, vehicle_limit)]

        # Process X
        t1 = time.time()
        data_X, _ = prep_train_data(
            self.opts.graph_size,
            instance,
            type_=self.opts.coords_sampling_dist,
            normed_data=self.opts.normalize_data
        )
        X = data_X[0]
        # print(f"prep_X took {time.time() - t1:.2f}s")
        # print('X[2].shape', X[2].shape)
        # print('X[4].shape', X[4].shape)
        # print('X[0].shape[0]', X[0].shape[0])
        # print('self.opts.fleet_size', self.opts.fleet_size)
        # Process Y
        t2 = time.time()
        if self.opts.graph_size == 100:
            Y_i, Y_v, Y_loads = self.convert_solution_to_sparse(
                locs.shape[0], solution, m=self.opts.fleet_size, capa_util=capa_util
            )
            # print(f"prep_Y took {time.time() - t2:.2f}s")
            return X, (Y_i, Y_v), Y_loads, idx
        else:

            Y_dense, Y_load = self.convert_solution_to_dense(locs.shape[0],
                                                             solution, m=self.opts.fleet_size,
                                                             capa_util=capa_util)
            # print(f"prep_Y took {time.time() - t2:.2f}s")
            return X, Y_dense, Y_load, idx

    def convert_solution_to_sparse(self, n_nodes, solution, m, capa_util):
        indices = []
        values = []
        accum_demand = np.zeros(m)

        for j in range(m):
            try:
                sequence = solution[j]
            except IndexError:
                sequence = [0, 0]

            for k, l in pairwise(sequence):
                indices.append(torch.tensor([j, k, l]))

            try:
                accum_demand[j] = max(capa_util[j])
            except IndexError:
                accum_demand[j] = 0

        indices_all = torch.stack(indices).transpose(0, 1) if indices else torch.empty((3, 0), dtype=torch.long)
        values = torch.ByteTensor([1]).expand(len(indices))
        loads = torch.from_numpy(accum_demand)

        return indices_all, values, loads

    def convert_solution_to_dense(self, n_nodes, solution, m, capa_util):
        n_nodes_w_depot = n_nodes + 1
        mat = torch.zeros((m, n_nodes_w_depot, n_nodes_w_depot), dtype=torch.float32)
        accum_demand = np.zeros(m)
        # print('n_nodes', n_nodes)
        # for route in solution:
        #     for i in range(len(route) - 1):
        #         mat[:, route[i], route[i + 1]] = 1.0
        for j in range(m):
            try:
                sequence = solution[j]
            except IndexError:
                sequence = [0, 0]
            for i in range(len(sequence) - 1):
                mat[j, sequence[i], sequence[i + 1]] = 1.0

            try:
                accum_demand[j] = max(capa_util[j])
            except IndexError:
                accum_demand[j] = 0
        loads = torch.from_numpy(accum_demand)
        return mat, loads


# class DatasetVrpFromNPZ(torch.utils.data.Dataset):
#     """Characterizes VRPdataset for PyTorch with
#         - lazy dataloading,
#         - npz file caching to avoid repeated disk I/O and
#         - on-the-fly datapreprocessing"""
#
#     # _global_cache = {}
#
#     def __init__(self, data_index_list, opts):
#         self.data_index_list = all_data # data_index_list
#         self.opts = opts
#         # self.cache = {}
#
#         for file_path, _ in data_index_list:
#             if file_path not in DatasetVrpFromNPZ._global_cache:
#                 print(f"Loading {file_path} into global RAM cache...")
#                 DatasetVrpFromNPZ._global_cache[file_path] = {
#                     k: v.copy() for k, v in np.load(file_path, allow_pickle=True).items()
#                 }
#             self.cache[file_path] = DatasetVrpFromNPZ._global_cache[file_path]
#
#     # def __init__(self, data_index_list, opts):
#     #     self.data_index_list = data_index_list
#     #     self.cache = {}  # cache npz files to avoid repeated disk I/O
#     #     # load_time_init = time.time()
#     #     #self.cache = {fp: np.load(fp, allow_pickle=True) for fp, _ in data_index_list}
#     #     # print(f"loading data into self.cache in {time.time() - load_time_init:.3f} seconds")
#     #     self.opts = opts
#
#
#
#     def __len__(self):
#         return len(self.data_index_list)
#
#     def __getitem__(self, idx):
#         file_path, inner_idx = self.data_index_list[idx]
#         print(f"[GETITEM] idx={idx}, file={file_path}, inner={inner_idx}")
#         time_load_st = time.time()
#         # Load file if not already in cache
#         file_path = os.path.abspath(file_path)
#         if file_path not in self.cache:
#             # self.cache[file_path] = np.load(file_path, allow_pickle=True, mmap_mode='r')
#             self.cache[os.path.abspath(file_path)] = dict(np.load(file_path, allow_pickle=True))
#         # print("type(self.cache[file_path]['demands'])", type(self.cache[file_path]['demands']))
#
#         data = self.cache[file_path]
#         depot = data["depots"][inner_idx]
#         locs = data["locs"][inner_idx]
#         demands = data["demands"][inner_idx]
#         capacity = data["capacities"][inner_idx]
#         vehicle_limit = data["vehicle_limits"][inner_idx]
#         solution = data["solutions"][inner_idx]
#         capa_util = data["utilized_capacities"][inner_idx]
#         print('time for loading in getitem', time.time() - time_load_st)
#         # === Pack instance as required for prep_train_data ===
#         instance_tuple = (depot, locs, demands, capacity, vehicle_limit)
#         instance_list = [instance_tuple]
#
#         # === On-the-fly processing of the X Data ===
#         start = time.time()
#         data_X, _ = prep_train_data(
#             self.opts.graph_size,
#             instance_list,
#             type_=self.opts.coords_sampling_dist,
#             normed_data=self.opts.normalize_data
#         )
#         # print('time for preping X in getitem', time.time() - time_prepX_st)
#         print(f"prep_X took {time.time() - start:.2f}s")
#
#         X = data_X[0]  # Only one instance was passed
#         # print('X[2]/Customer_gr.shape', X[2].shape)
#         # print('X[0]/Customer_gr.shape', X[0].shape)
#         # === On-the-fly processing of the target solution ===
#         start = time.time()
#         if self.opts.graph_size == 100:
#             Y_i, Y_v, Y_loads = self.convert_solution_to_sparse(locs.shape[0],
#                                                                 solution,
#                                                                 m=X[0].shape[0],
#                                                                 capa_util=capa_util)
#             # print('time for preping Y in getitem', time.time() - time_prepY_st)
#             print(f"prep_Y took {time.time() - start:.2f}s")
#             # print('Y_i.shape', Y_i.shape)
#             # print('Y_i[0][0]', Y_i[0][0])
#             # print('Y_v.shape', Y_v.shape)
#             # print('Y_loads.shape', Y_loads.shape)
#             # print('Y_v[0][0]', Y_i[0][0])
#             return X, (Y_i, Y_v), Y_loads, idx
#         else:
#             Y_dense = self.convert_solution_to_dense(locs.shape[0], solution)
#             return X, Y_dense, None, idx
#
#     def convert_solution_to_sparse(self, n_nodes, solution, m, capa_util):
#         indices = []
#         values = []
#         indices_all, loads = None, None
#         accum_demand = np.zeros(m)
#         for j in range(m):
#             try:
#                 # get sequence of visited customers
#                 sequence_of_customers = solution[j]
#                 for k, l in pairwise(sequence_of_customers):
#                     indices.append(torch.tensor([j, k, l]))
#                 accum_demand[j] = max(capa_util[j])
#             except IndexError:
#                 j = m - 1  # (index from 0)
#                 # dummy empty seq. of customers
#                 sequence_of_customers = [0, 0]
#                 for k, l in pairwise(sequence_of_customers):
#                     indices.append(torch.tensor([j, k, l]))
#                 accum_demand[j] = 0
#             indices_all = torch.stack(indices).transpose(0, 1)
#             values = torch.ByteTensor([1]).expand(len(indices))
#             loads = torch.from_numpy(accum_demand)
#         return indices_all, values, loads
#
#     def convert_solution_to_dense(self, n_nodes, solution, m):
#         mat = torch.zeros((m, n_nodes, n_nodes), dtype=torch.float32)
#         for route in solution:
#             for i in range(len(route) - 1):
#                 mat[route[i], route[i + 1]] = 1.0
#         return mat


class DatasetVrp(torch.utils.data.Dataset):
    """Characterizes a dataset for PyTorch"""

    def __init__(self, list_IDs, X_dat, Y_dat, YLoad_dat, vehicles_available=4, train_w_HGS=False):
        """Initialization"""
        self.list_IDs = list_IDs
        self.X_dat = X_dat
        self.Y_dat = Y_dat
        self.YLoad_dat = YLoad_dat
        self.train_w_HGS = train_w_HGS
        if train_w_HGS:
            self.vehicles_available = vehicles_available

    def __len__(self):
        """Denotes total nr of samples"""
        return len(self.list_IDs)

    def __getitem__(self, index):
        """Generates one sample of data"""

        # Select sample
        ID = self.list_IDs[index]
        X = self.X_dat[ID]
        if self.train_w_HGS:
            # targets generated on the fly
            return X, None, None, ID
        else:
            # use stored targets
            y = self.Y_dat[ID]
            y_load = self.YLoad_dat[ID]
            return X, y, y_load, ID


class DatasetVrp100(torch.utils.data.Dataset):
    """Characterizes a dataset for PyTorch"""

    def __init__(self, list_IDs, X_dat, Y_dat_i, Y_dat_v, YLoad_dat, vehicles_available=4, train_w_HGS=False):
        """Initialization"""
        self.list_IDs = list_IDs
        self.X_dat = X_dat
        self.Y_dat_i = Y_dat_i
        self.Y_dat_v = Y_dat_v
        self.YLoad_dat = YLoad_dat
        self.train_w_HGS = train_w_HGS
        if train_w_HGS:
            self.vehicles_available = vehicles_available

    def __len__(self):
        """Denotes total nr of samples"""
        return len(self.list_IDs)

    def __getitem__(self, index):
        """Generates one sample of data"""

        # Select sample
        ID = self.list_IDs[index]
        X = self.X_dat[ID]

        if self.train_w_HGS:
            # targets generated on the fly
            return X, None, None, ID
        else:
            # use stored targets
            y_i = self.Y_dat_i[index]
            y_v = self.Y_dat_v[index]
            y = (y_i, y_v)
            # y = self.Y_dat[ID]
            y_load = self.YLoad_dat[ID]
            return X, y, y_load, ID


### EVAL ###
def make_valid_and_adjust(opts, veh, perm_v, probs, dem, v_cost, orig_vals, cvrp_instance):
    missing_, greedy_sol, greedy_routes, r_ = [], None, None, None
    _, dists_orig, capa_t, original_demands_t = orig_vals
    greedy_sols, greedy_routes_all, n_routes, dist_solved = [], [], [], []
    traveled_dists_orig_all, traveled_dists_scld_all, total_cost_v_orig = [], [], []
    # print('veh', veh)
    if opts.random_perm:
        r_ = list(range(veh))
        # print('r_', r_)
        random.shuffle(r_)
        p_idx, rem_cap = greedy_path(probs, dem, r_)
        # GREEDY VALID PATH for TO-direction (based on probs)
        greedy_sol, greedy_routes, final_loads, missing_ = make_valid(p_idx, probs, rem_cap, dem)
    # check all permutations of vehicles for repairing solutions - take valid one or best one
    else:
        # print('perm_v', perm_v)
        for perm in perm_v:
            p_idx, rem_cap = greedy_path(probs, dem, perm)
            # GREEDY VAL PATH for TO-direction (based on probs)
            greedy_sol, greedy_routes, final_loads, missing_ = make_valid(p_idx, probs, rem_cap, dem)
            # print('IF NOT RANDOM_PERM: *list(missing_)', list(missing_))
            if not list(missing_):
                # print('BREAK in NOT RANDOM_PERM')
                break
    # print('greedy_sol before check missing', greedy_sol)
    # print('greedy_routes before check missing', greedy_routes)
    # if not valid
    if list(missing_):
        # if missing_==1 for TO-path ==> get FROM-path
        if opts.random_perm:
            perm = r_
        p_idx, rem_cap = greedy_path(probs.transpose(1, 2), dem, perm)
        # GREEDY VAL PATH for FROM-direction (based on probs)
        greedy_sol, greedy_routes, final_loads, missing_ = make_valid(p_idx, probs, rem_cap, dem)
        if list(missing_):
            print('No Valid Solution in TO- and FROM- path')
            if opts.guarantee_solution:
                print('Move to guarantee solution... (Nr. Vehicles not guaranteed)')
                print('greedy_routes', greedy_routes)
                solution_guarantee = guarantee_solution(greedy_routes, original_demands_t, capa_t, missing_)
                greedy_routes = solution_guarantee
                print('greedy_routes after guarantee solution', greedy_routes)
                greedy_sol = None
        else:
            pass
            # print('Valid Solution in FROM- path')
    else:
        pass
        # print('Valid Solution in TO- path')
    # IF SOLVED (either TO- or FROM- direction)  --> SAVE + POSTPROCESS
    # print('IF SOLVED *list(missing_)', list(missing_))
    # if not list(missing_):
    # greedy_sols.append(greedy_sol)
    greedy_routes = [rout for rout in greedy_routes if len(rout) >= 3]
    n_routes = len([rout for rout in greedy_routes if len(rout) >= 3])
    # print('greedy_routes', greedy_routes)
    # get travelled distance
    # print('dists_orig', dists_orig[:5, :5])
    dist_mat = dists_orig.unsqueeze(0).expand(probs.size(0), probs.size(1), probs.size(1))  # .cuda
    # print('dist_mat.size()', dist_mat.size())
    # print('greedy_sol.size()', greedy_sol.size())
    # dist_scaled = dist_mat * 1000
    # dist_solved.append(dist_orig)
    if greedy_sol is not None:
        traveled_dists = greedy_sol * dist_mat
        # traveled_dists_scaled = greedy_sol * dist_scaled
        costs_orig = torch.sum(traveled_dists).item()
        # print('costs_orig', costs_orig)
        costs_orig_from_r, nr_v, routes_new = get_travel_costs(greedy_routes, cvrp_instance)
        cost_v_orig = costs_orig_from_r + (len([rout for rout in greedy_routes if len(rout) >= 3]) * v_cost)
        # print("costs_orig_from_r", costs_orig_from_r)
        # print("nr_v", nr_v)
    else:
        # get costs for guaranteed sol differently
        costs_orig_from_r, nr_v, routes_new = get_travel_costs(greedy_routes, cvrp_instance)
        cost_v_orig = costs_orig_from_r + (len([rout for rout in greedy_routes if len(rout) >= 3]) * v_cost)
        # print('routes_new', routes_new)
        assert routes_new == greedy_routes
    # traveled_dists_orig_all.append(torch.sum(traveled_dists_orig).item())
    # traveled_dists_scld_all.append(torch.sum(traveled_dists_scaled).item())

    # print('costs_orig', costs_orig)
    # costs_scld = torch.sum(traveled_dists_scaled).item()
    # total_cost_v_orig.append(torch.sum(traveled_dists_orig).item() + (len([rout for rout in greedy_routes
    #                                                                        if len(rout) >= 3]) * v_cost))

    return greedy_sol, greedy_routes, n_routes, missing_, costs_orig_from_r, cost_v_orig


def postprocess(greedy_routes, orig_vals, m_set, v_fixed_cost):
    locs, _, capa_t, original_demands_t = orig_vals
    locs = (locs[:, :2] * 1000).astype(int)
    # print('locs', locs[:5])
    m = len(greedy_routes)
    # print('m', m)
    init_solution, Sols_improved, Dist_improved, Dist_improved_orig, greedy_routes_improved = [], [], [], [], []
    nr_of_routes_improved, total_cost_v, total_cost_v_improved = [], [], []
    for r in greedy_routes:
        init_solution.append(r[1:-1])
    # create data for OR tools
    # original_demands_t
    # (dists_orig * 1000).type(torch.int16)
    OR_data = create_data_model(locs, init_solution, original_demands_t, capa_t, m)
    # list(map(round, Kool_Test_X[i][3][0] * capa_t)),
    sol_imp = main(OR_data, m_set)
    # Sols_improved.append(sol_improved)
    cost_imp = sol_imp['total_dist']
    cost_imp_orig = sol_imp['total_dist'] / 1000
    # print('Dist_improved_orig', Dist_improved_orig)
    # wrong:
    # routes_improved_wrong = [sol_improved[i][0][0][1:] for i in range(m) if sol_improved[i][0][0][1:]]
    # correct:
    routes_imp = [sol_imp[i][0][0][1:] for i in range(m) if len(sol_imp[i][0][0]) > 2]
    # greedy_routes_improved.append(routes_improved)
    # print('GREEDY ROUTES IMPROVED', greedy_routes_improved)
    # nr_of_routes_improved.append(len(routes_improved))
    total_cost_v = sol_imp['total_dist'] / 1000 + (len([rout for rout in greedy_routes if len(rout) >= 3])
                                                   * v_fixed_cost)
    # total_cost_v_improved_wrong.append(
    #     sol_improved['total_dist'] / 1000 + (len(routes_improved_wrong) * v_fixed_cost))
    total_cost_v_imp = sol_imp['total_dist'] / 1000 + (len(routes_imp) * v_fixed_cost)

    return sol_imp, routes_imp, len(routes_imp), cost_imp, cost_imp_orig, total_cost_v, total_cost_v_imp


def guarantee_solution(greedy_routes, original_dems, capa, missing_: list):
    init_solution = []
    for r in greedy_routes:
        init_solution.append(r[1:-1])
    print('init sol without extra routes:', init_solution)
    # add missing customers as route
    if sum([original_dems[x.item()] for x in missing_]) <= capa:
        # print('add only one extra tour')
        # np.sum(test_instance[3][0, missing_.cpu()]) < 1.000001:
        init_solution.append([x.item() for x in missing_])
        extra_count = 1
    else:
        # print('add MULTIPLE extra tours')
        extra_route = []
        extra_routes = []
        cum_demand = 0
        for x in missing_:
            # print('x in missing:', x)
            # print('cum_demand',cum_demand)
            # print('(cum_demand + original_demands_t[x.item()])',(cum_demand + original_demands_t[x.item()]))
            if (cum_demand + original_dems[x.item()]) <= capa:
                # print('add to EXISTING extra route')
                # (cum_demand + test_instance[3][0, x]) < 1.000001:
                extra_route.extend([x.item()])
                cum_demand += original_dems[x.item()]
                # cum_demand += test_instance[3][0, x]
                # print('Extra route now (AFTER APPEND):',extra_route)
                # print('(cum_demand + original_demands_t[x.item()])',(cum_demand + original_demands_t[x.item()]))
            else:
                # print('CREATE NEW EXTRA ROUTE')
                extra_routes.append(extra_route)
                extra_route = [x.item()]
                # print('Extra route now (AFTER NEW):',extra_route)
                cum_demand = original_dems[x.item()]
                # print('cum_demand after else',cum_demand)

            print('EXTRA ROUTES (ALL)', extra_routes)

        extra_routes.append(extra_route)
        extra_count = len(extra_routes)
        init_solution.extend(extra_routes)
        print('init_solution', init_solution)
    # check if all tours in init_solution are valid:
    routes_w_depot = []
    for route in init_solution:
        sum_tour = 0
        for vertex in route:
            sum_tour += original_dems[vertex]
        print('sum_tour', sum_tour)
        if sum_tour > capa + EPS:
            print('TOUR SUPRASSES CAPACITY', capa)
            print(sum_tour)
            break
        routes_w_depot.append([0] + route + [0])
    print('guaranteed solution:', routes_w_depot)
    return routes_w_depot


def prep_data_fpin(dat: List[CVRPInstance], v_max: int=None, offset=0):
    """preprocesses data format for AttentionModel-MDAM (i.e. from List[NamedTuple] to List[torch.Tensor])"""
    # if isinstance(dat[0], TSPInstance):
    #     return [torch.FloatTensor(row.coords) for row in (dat[offset:offset + len(dat)])]
    # elif isinstance(dat[0], CVRPInstance):
    print('dat[0]', dat[0])
    print('dat[0].node_features[2,-1]', dat[0].node_features[2,-1])
    return [make_cvrp_instance(args, v_max) for args in dat[offset:offset + len(dat)]]


def make_cvrp_instance(args, v_max, distribution_args=None):
    # depot, loc, demand, capacity, *args = args
    depot = args.coords[args.depot_idx[0]].tolist()
    loc = args.coords[1:, :].tolist()
    demand = args.node_features[1:, args.constraint_idx[0]].tolist()
    # print('demand', demand)
    # print('args.vehicle_capacity', args.vehicle_capacity)
    capacity = args.vehicle_capacity if args.vehicle_capacity > 1.0 else args.original_capacity
    # print('capacity', capacity)
    grid_size = 1
    if distribution_args is not None:
        depot_types, customer_types, grid_size = distribution_args
    # return {
    #     'loc': torch.tensor(loc, dtype=torch.float) / grid_size,
    #     'demand': torch.tensor(demand, dtype=torch.float),  # / capacity -> demands already normalized
    #     'depot': torch.tensor(depot, dtype=torch.float) / grid_size
    # }
    return depot, loc, demand, capacity, v_max if v_max is not None else args.max_num_vehicles


def make_RPSolution(problem, sols, costs, times, instances, running_sols=None) -> List[RPSolution]:
    """Parse model solution back to RPSolution for consistent evaluation"""
    # transform solution torch.Tensor -> List[List]
    # sol_list = [_get_sep_tours(problem, instance.graph_size, sol_) for sol_, instance in zip(sols, instances)]

    return [
        RPSolution(
            solution=sols[i],
            running_sols=running_sols[i] if running_sols else None,
            cost=costs[i] if sols[i] is not None else None,
            method_internal_cost=costs[i] if sols[i] is not None else None,
            num_vehicles=len(sols[i]) if sols[i] is not None else None,
            run_time=times[i] if sols[i] is not None else 0,
            problem=problem,
            instance=instances[i],
        )
        for i in range(len(sols))
    ]


def get_preliminaries(vrp_size, is_train=False, uchoa=False):
    # handles
    if vrp_size == 20:
        vrp_to_solve = 'VRP20'
        if not is_train:
            Q = 30
            m = 4
            n = 20
            v_fixed_cost = 35

            model_specs = {'dropout': 0.0,
                           'n_hidden': 1024,
                           'layers': 9,
                           'mainDimension': 256}
            return vrp_to_solve, m, v_fixed_cost, model_specs
        else:
            nr_files = 10  # 29
            pen_w = 0.3
            load_w = 0.5
            starts_weight = 0.5
            return nr_files, pen_w, load_w, starts_weight

    elif vrp_size == 50:
        vrp_to_solve = 'VRP50'
        if not is_train:
            Q = 40
            m = 7
            n = 50
            v_fixed_cost = 50

            model_specs = {'dropout': 0.0,
                           'n_hidden': 1024,
                           'layers': 9,
                           'mainDimension': 256}
            return vrp_to_solve, m, v_fixed_cost, model_specs
        else:
            nr_files = 23
            pen_w = 0.1
            load_w = 0.3
            starts_weight = 0.1
            return nr_files, pen_w, load_w, starts_weight

    elif vrp_size == 60:
        vrp_to_solve = 'VRP60'
        if not is_train:
            Q = 40
            m = 7
            n = 60
            v_fixed_cost = 56

            model_specs = {'dropout': 0.0,
                           'n_hidden': 1024,
                           'layers': 9,
                           'mainDimension': 256}
            return vrp_to_solve, m, v_fixed_cost, model_specs
        else:
            nr_files = 23
            pen_w = 0.1
            load_w = 0.3
            starts_weight = 0.1
            return nr_files, pen_w, load_w, starts_weight

    elif vrp_size == 100:
        vrp_to_solve = 'VRP100'
        if not is_train:
            Q = 50
            m = 7
            n = 100
            v_fixed_cost = 80

            model_specs = {'dropout': 0.0,
                           'n_hidden': 1024,
                           'layers': 9,
                           'mainDimension': 256}
            return vrp_to_solve, m, v_fixed_cost, model_specs
        else:
            if not uchoa:
                nr_files = 16
                pen_w = 0.1
                load_w = 0.3
                starts_weight = 0.1
            else:
                nr_files = 1
                pen_w = 0.1
                load_w = 0.3
                starts_weight = 0.3
            return nr_files, pen_w, load_w, starts_weight
    else:
        warnings.warn(f"No valid vrp_size specified: {vrp_size}")
        # print('Error: No valid vrp_size specified')
        return None, None, None, None


def get_travel_costs(routes: List[List], instance: CVRPInstance):
    depot = instance.depot_idx[0]
    coords = instance.coords
    demands = instance.node_features[:, instance.constraint_idx[0]]
    routes_ = []
    # capacity = instance.original_capacity
    capacity = instance.vehicle_capacity
    k, cost = 0, 0  # .0
    # print('routes', routes)
    for r in routes:
        # print('r', r)
        if r:
            if r[0] != depot:
                r = [depot] + r
            if r[-1] != depot:
                r.append(depot)
            transit = 0
            source = r[0]
            cum_d = 0
            for target in r[1:]:
                transit += np.linalg.norm(coords[source] - coords[target], ord=2)
                if target != depot:
                    cum_d += demands[target]
                source = target
            # print('cum_d', cum_d)
            # print('capacity', capacity)
            if cum_d > capacity + 0.002: # EPS:
                warnings.warn(f"Solution for instance {instance.instance_id} not feasible.  "
                              f"Capacity constr. violated: {cum_d}>{capacity}. Setting cost and k to 'inf'.")
                print('cum_d', cum_d)
                cost = float("inf")
                k = float("inf")
                break
            cost += transit
            k += 1 if r != [0,0] else 0
            routes_.append(r)
    # print('k', k)

    return cost, k, routes_



Route = Union[Sequence[int], torch.Tensor]

def routes_cost_from_dist(
    D: torch.Tensor,
    routes: List[Route],
) -> torch.Tensor:
    """
    Compute total travel cost for routes that already include depot nodes.

    Args:
        D: (N, N) distance matrix
        routes: list of routes, e.g. [0, 3, 7, 2, 0]

    Returns:
        Scalar tensor: total travel cost
    """
    assert D.ndim == 2 and D.shape[0] == D.shape[1], "D must be square (N,N)"
    N = D.shape[0]
    device = D.device

    total = D.new_zeros(())
    for r in routes:
        if isinstance(r, torch.Tensor):
            nodes = r.to(device=device).long().flatten()
        else:
            nodes = torch.tensor(r, device=device, dtype=torch.long)

        if nodes.numel() < 2:
            continue

        # Optional safety check (cheap, remove later)
        if torch.any(nodes < 0) or torch.any(nodes >= N):
            bad = nodes[(nodes < 0) | (nodes >= N)]
            raise ValueError(f"Route contains out-of-range nodes: {bad.tolist()} (N={N})")

        total = total + D[nodes[:-1], nodes[1:]].sum()

    return total

def transform_loaded_HGS_resutls(res_dict_list: list, data: List[CVRPInstance],
                                 int_prec: int = 1000, dataset_name: str = "cvrp"):
    objs = np.asarray([np.asarray(r['running_costs']) for r in res_dict_list if r is not None], dtype=object)
    # print('objs', objs)
    objs = objs / int_prec
    final_objs = [r['final_obj'] / int_prec for r in res_dict_list if r is not None]
    # print('final_objs', final_objs)
    final_rts = [r['runtime'] for r in res_dict_list if r is not None]
    # runtimes = np.array([r['runtimes'] for r in results if r is not None])
    # running_costs = np.array([r['running_costs'] for r in results if r is not None])
    # + prep_rt + init_sol_rt
    solutions = [
        RPSolution(
            solution=r['solution'] if r is not None else None,
            run_time=r['runtime'] if r is not None else None,
            running_costs=list(np.array(r['running_costs']) / int_prec) if r is not None else None,
            running_times=[r['running_times'][t] for t in range(len(r['running_times']))] if r is not None else None,
            num_vehicles=len(r['solution']) if r is not None else None,
            problem=dataset_name.upper(),
            instance=d
        ) for r, d in zip(res_dict_list, data)
    ]
    # print('solutions', solutions)

    results_ = {
        "objs": objs,
        "final_objs": final_objs,
        "runtime": final_rts,
    }
    return results_, solutions


def read_HGS_instance(filepath: str, scale_factor: int = None, env_cfg: DictConfig = None, normalize: bool = True):
    """
        taken from l2o meta
        For loading and parsing CVRPLIB format instances from HGS run_log.
        """
    file = open(filepath, "r")
    lines = [ll.strip() for ll in file]
    i = 0
    cap = 1.0
    dimension, locations, demand, node_features, capacity, K = None, None, None, None, None, None
    overall_inst_type, X_inst_type = "unknown", None  # for Uchoa (XE) type data store depot, coord distrib. type
    inst_id, int_loc = None, True
    scale_factor = 1000 if scale_factor is None else scale_factor
    # SCALE_FACTORS)[overall_inst_type] if overall_inst_type in SCALE_FACTORS.keys() else 1
    # print('self.scale_factor', self.scale_factor)
    while i < len(lines):
        line = lines[i]
        if line.startswith("NAME"):
            name = line.split(':')[1].strip()
            inst_id = name
            if "k" in name.split("-")[-1]:
                K = name.split("-")[-1][1:]
        if line.startswith("DIMENSION"):
            dimension = int(line.split(':')[1])
            node_features = np.zeros((dimension, 3), dtype=np.single)
        elif line.startswith("CAPACITY"):
            capacity = int(line.split(':')[1])
        elif line.startswith('NODE_COORD_SECTION'):
            try:
                locations = np.loadtxt(lines[i + 1:i + 1 + dimension], dtype=int)
            except ValueError:
                locations = np.loadtxt(lines[i + 1:i + 1 + dimension])
                int_loc = False
            i = i + dimension
        elif line.startswith('DEMAND_SECTION'):
            demand = np.loadtxt(lines[i + 1:i + 1 + dimension], dtype=int)
            i = i + dimension

        i += 1

    original_locations = locations[:, 1:]
    # print('original_locations', original_locations)
    # print('demand[:, 1:].squeeze()', demand[:, 1:].squeeze())
    # normalize coords and demands
    locations = original_locations / int(scale_factor) if normalize else original_locations
    demand = demand[:, 1:].squeeze() / int(scale_factor) if normalize else demand[:, 1:].squeeze()
    capacity = int(capacity) / int(scale_factor) if normalize else int(capacity)
    # print('demand', demand)
    # print('locations', locations)

    assert locations.max() <= 1000
    assert demand.min() >= 0
    node_features[:, :2] = locations
    node_features[:, -1] = demand / 1.0
    # print('node_features[:5]', node_features[:5])
    # add additional indicators
    depot_1_hot = np.zeros(dimension, dtype=np.single)
    depot_1_hot[0] = 1
    customer_1_hot = np.ones(dimension, dtype=np.single)
    customer_1_hot[0] = 0

    return CVRPInstance(
        coords=locations,
        node_features=np.concatenate((
            depot_1_hot[:, None],
            customer_1_hot[:, None],
            node_features
        ), axis=-1),
        graph_size=dimension,
        constraint_idx=[-1],  # demand is at last position of node features
        vehicle_capacity=1.0,  # demands are normalized
        time_limit=5,
        original_capacity=capacity,
        original_locations=original_locations,
        coords_dist=env_cfg.generator_args.coords_sampling_dist,
        depot_type=env_cfg.generator_args.coords_sampling_dist,
        demands_dist=env_cfg.generator_args.weights_sampling_dist,
        instance_id=inst_id,  # int(inst_id)
        type=X_inst_type if X_inst_type is not None else overall_inst_type,
        # demands_dist=None,
        max_num_vehicles=env_cfg.sampling_args.k,
    )


# global_cache_manager.py
# Global cache for NPZ files
class GlobalCache:
    _cache = {}

    @classmethod
    def get(cls, file_path):
        if file_path not in cls._cache:
            print(f"[GlobalCache] Loading {file_path} into RAM...")
            cls._cache[file_path] = dict(np.load(file_path, allow_pickle=True))
        return cls._cache[file_path]

    @classmethod
    def clear(cls):
        cls._cache.clear()
