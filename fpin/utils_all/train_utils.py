import torch
import numpy as np
from typing import NamedTuple, Union, Any, List, Optional


# from ..formats import CVRPInstance

class CVRPInstance(NamedTuple):
    """Typed Format Routing Problem Instance."""
    coords: Union[np.ndarray, torch.Tensor]
    node_features: Union[np.ndarray, torch.Tensor]
    graph_size: int
    vehicle_capacity: Union[Union[np.ndarray, torch.Tensor], float] = -1
    original_capacity: Union[int, np.ndarray, torch.Tensor] = None  # original_cap for NLNS or generally for Uchoa (
    # as they are non-uniform - array)
    max_num_vehicles: int = None  # soft upper limit
    depot_idx: List = [0]
    constraint_idx: List = [-1]
    time_limit: float = None  # overall time limit for solving this instance (needed to calculate PI)
    BKS: float = None  # Best Known Solution (for this particular instance so far)
    instance_id: Optional[int] = None  # Test instances of a particular dataset have an ID - due to BKS registry
    coords_dist: Union[int, str] = None  # For uchoa data it is an integer value, else 'unif' or 'gauss', 'mixed'
    depot_type: Union[int, str] = None  # For uchoa data it is an integer value, else 'unif' or 'gauss', 'mixed'
    demands_dist: Union[int, str] = None  # For uchoa -> integer value, else 'unif' or "random_int", "random_k_variant"
    original_locations: Union[np.ndarray, torch.Tensor] = None  # for NLNS
    type: str = None  # general "distribution" or type of data instance ('uniform', 'uchoa', 'dimacs', ...)

    #     demand: Union[np.ndarray, torch.Tensor] = None # for NLNS

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        info = [format_repr(k, v) for k, v in self._asdict().items()]
        return '{}({})'.format(cls, ', '.join(info))

    def __getitem__(self, key: Union[str, int]):
        if isinstance(key, int):
            key = self._fields[key]
        return getattr(self, key)

    def get(self, key: Union[str, int], default_val: Any = None):
        """Dict like getter method with default value."""
        try:
            return self[key]
        except AttributeError:
            return default_val

    def update(self, **kwargs):
        return self._replace(**kwargs)


class RPSolution(NamedTuple):
    """Typed wrapper for routing problem solutions."""
    solution: List[List]
    cost: float = None
    pi_score: float = None
    wrap_score: float = None
    num_vehicles: int = None
    run_time: float = None
    problem: str = None
    instance: CVRPInstance = None
    last_cost: Optional[float] = None  # previous instance cost that was PI-evaluated (for iterative PI computatn)
    last_runtime: Optional[float] = None  # previous instance runtime (for iterative PI computatn)
    running_costs: Optional[List] = None  # for PI and WRAP eval
    running_times: Optional[List] = None  # for PI and WRAP eval
    running_sols: Optional[List[List[List]]] = None  # for PI and WRAP eval
    instance_id: Optional[int] = None  # only needed for some models where solution lists are not sorted
    method_internal_cost: Optional[float] = None

    def update(self, **kwargs):
        return self._replace(**kwargs)


def load_ckp(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, checkpoint['epoch']


def transform_targets_to_dct(solution_tuple: Union[List[RPSolution], RPSolution] = None, weights_dist: str = "random_int"):

    # correct for the fact that RP Solution has changed after labels have been generated with old RPSolution definition
    solution_instance = solution_tuple.instance if solution_tuple.instance is not None else solution_tuple.problem
    # print('solution_instance', solution_instance)
    orig_capa = solution_instance.original_capacity
    if (solution_tuple.solution[0][0] != 0 and solution_tuple.solution[0][-1] != 0):
        solution_list = [[0] + sol + [0] for sol in solution_tuple.solution]
    else:
        solution_list = solution_tuple.solution
    # print('solution_list ', solution_list)
    # solution_list = [sol.insert(-1, 0) for sol in solution_list]
    # print('solution_list 2 ', solution_list)
    number_of_keys = len(solution_list)
    # print('number_of_keys', number_of_keys)
    # nodefeatures is already in original scale
    # print('weights_dist', weights_dist)
    if weights_dist == "random_int":
        if solution_instance.node_features[:, -1].all() < 1.1:
            original_demands = (solution_instance.node_features[:, -1] * solution_instance.original_capacity
                                        ).astype(int)
        else:
            original_demands = np.round(solution_instance.node_features[:, -1]
                                        ).astype(int)
        capa = orig_capa
    else:
        original_demands = solution_instance.node_features[:, -1]
        capa = solution_instance.vehicle_capacity
    acc_demand_all = []
    for key in range(number_of_keys):
        # print('key:', key)
        acc_dem = 0
        accumulated_demand_list = []
        for j in solution_list[key]:
            acc_dem += original_demands[j]
            accumulated_demand_list.append(acc_dem)
        assert accumulated_demand_list[-1] <= capa, f"INFEASIBLE TARGET SOLUTION: {accumulated_demand_list[-1]}"
        acc_demand_all.append(accumulated_demand_list)
    solution_dicts = {k: [[v_1, v_2], capa]
                      for k, (v_1, v_2) in enumerate(zip(solution_list, acc_demand_all))}
    solution_dicts['total_dist'] = solution_tuple.cost if solution_tuple.cost is not None else \
        solution_tuple.running_costs[-1]
    # print('solution_dicts', solution_dicts)
    return solution_dicts


def transform_to_CVRPInstance_batch(instance_batch: List[torch.tensor], inst_id: Union[int, list] = None,
                                    coords_dist: str = "uniform", depot_type: str = "uniform"):
    if len(instance_batch) == 1:
        # vehicle_b, depot_b, customer_b, demands_b, dist_mat_b, vehicle_capacity_b, demands_orig_b = instance_batch[0]
        # vehicle_b, depot_b, customer_b, demands_b = [vehicle_b], [depot_b], [customer_b], [demands_b]
        # dist_mat_b, vehicle_capacity_b, demands_orig_b = [dist_mat_b], [vehicle_capacity_b], [demands_orig_b]
        # print('customer_b.size', customer_b.size)
        vehicle_b = torch.FloatTensor(instance_batch[0][0]).unsqueeze(0)
        depot_b = torch.FloatTensor(instance_batch[0][1]).unsqueeze(0)
        customer_b = torch.FloatTensor(instance_batch[0][2]).unsqueeze(0)
        demands_b = torch.FloatTensor(instance_batch[0][3]).unsqueeze(0)
        dist_mat_b = torch.FloatTensor(instance_batch[0][4]).unsqueeze(0)
        vehicle_capacity_b = instance_batch[0][5]
        demands_orig_b = instance_batch[0][6]
        # print('customer_b.size(0)', customer_b.size(0))
        coords, demands = [], []
        for i in range(customer_b.size(0)):
            # print('i', i)
            # print('depot_b[i]', depot_b[i])
            # print('customer_b[i]', customer_b[i])
            coords_i = np.vstack((depot_b[i].numpy()[:, :-2], customer_b[i].numpy()[:, :-1]))
            # print('coords_i', coords_i)
            demands_i = demands_b.numpy()
            # print('demands_b[i].numpy()', demands_b[i].numpy())
            # print('vehicle_capacity_b', vehicle_capacity_b)
            coords.append((coords_i * 1000).astype(int))
            demands.append((demands_b[i].numpy()[0] * vehicle_capacity_b).astype(int))
        coords = np.stack(coords)
        # print('coords.shape', coords.shape)
        # print('coords[0]]', coords[0])
        demands = np.stack(demands)
        # print('demands.shape', demands.shape)
        # print('demands[0]', demands[0])
        graph_size = coords.shape[1]
    else:
        # (vehicle_gr, Depot_gr, Customer_gr, demand_array,dist_mat,vehicle_capacity, X_dat[i]['demands'])
        vehicle_b, depot_b, customer_b, demands_b, dist_mat_b, vehicle_capacity_b, demands_orig_b = instance_batch
        # print('customer_b.size(0)', customer_b.size(0))

        # print('customer_b[0]', customer_b[0])
        # print('demands_b[0]', demands_b[0])
        # some manipulation
        coords, demands = [], []
        # print('customer_b.size(0)', customer_b.size(0))
        for i in range(customer_b.size(0)):
            # print('i', i)
            # print('depot_b[i]', depot_b[i])
            # print('customer_b[i]', customer_b[i])
            coords_i = np.vstack((depot_b[i].numpy()[:, :-2], customer_b[i].numpy()[:, :-1]))
            # print('coords_i', coords_i)
            # demands_i = demands_b.numpy()
            coords.append((coords_i * 1000).astype(int))
            demands.append((demands_b[i].numpy()[0] * vehicle_capacity_b[i].item()).astype(int))
        coords = np.stack(coords)
        # print('coords.shape', coords.shape)
        # print('coords[0]]', coords[0])
        demands = np.stack(demands)
        # print('demands.shape', demands.shape)
        # print('demands[0]', demands[0])
        graph_size = coords.shape[1]

    node_features = create_nodes(customer_b.size(0), graph_size - 1, n_depots=1,
                                 features=[coords, demands])

    return [
        CVRPInstance(
            coords=coords[i],
            node_features=node_features[i],
            graph_size=graph_size,
            constraint_idx=[-1],  # demand is at last position of node features
            vehicle_capacity=1.0,  # demands are normalized
            original_capacity=vehicle_capacity_b if len(instance_batch) == 1 else vehicle_capacity_b[i].item(),
            max_num_vehicles=vehicle_b[i].shape[0],
            # BKS=None,
            instance_id=inst_id[i] if isinstance(inst_id, list) else i,
            coords_dist=coords_dist,
            depot_type=depot_type,
            # original_locations=None,
            # type=None,
        )
        for i in range(customer_b.size(0))
    ]


# _single
def transform_to_CVRPInstance(instance_tuple: tuple, inst_id: int = 0,
                              coords_dist: str = "uniform", depot_type: str = "uniform"):
    # (vehicle_gr, Depot_gr, Customer_gr, demand_array,dist_mat,vehicle_capacity, X_dat[i]['demands'])
    vehicle_gr, depot_gr, customer_gr, demands, dist_mat, vehicle_capacity, demands_orig = instance_tuple

    # some manipulation
    coords = np.vstack([depot_gr[:, :-2], customer_gr[:, :-1]])
    coords = np.stack([coords])
    demands = np.stack([demands[0]])
    graph_size = coords.shape[1]

    node_features = create_nodes(1, graph_size - 1, n_depots=1,
                                 features=[coords, demands])

    return [CVRPInstance(
        coords=coords[0],
        node_features=node_features[0],
        graph_size=graph_size,
        constraint_idx=[-1],  # demand is at last position of node features
        vehicle_capacity=1.0,  # demands are normalized
        original_capacity=vehicle_capacity,
        max_num_vehicles=vehicle_gr.shape[0],
        # BKS=None,
        instance_id=inst_id,
        coords_dist=coords_dist,
        depot_type=depot_type,
        # original_locations=None,
        # type=None,
    )]


def load_hgs_sol(sol_filename: str):
    tours = []
    with open(sol_filename, "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):  # read out solution tours
            if line[:5] == 'Route':
                l = line.strip().split()
                tours.append([int(idx) for idx in l[2:]])
            else:
                cost = float(line.strip().split()[1])
    # print('tours', tours)
    return tours, cost


def create_nodes(size: int,
                 graph_size: int,
                 features: list,
                 n_depots: int = 1):
    """Create node id and type vectors and concatenate with other features."""
    return np.dstack((
        np.broadcast_to(np.concatenate((  # add id and node type (depot / customer)
            np.array([1] * n_depots +
                     [0] * graph_size)[:, None],  # depot/customer type 1-hot
            np.array([0] * n_depots +
                     [1] * graph_size)[:, None],  # depot/customer type 1-hot
        ), axis=-1), (size, graph_size + n_depots, 2)),
        *features,
    ))


def format_repr(k, v, space: str = ' '):
    if isinstance(v, int) or isinstance(v, float):
        return f"{space}{k}={v}"
    elif isinstance(v, np.ndarray):
        return f"{space}{k}=ndarray_{list(v.shape)}"
    elif isinstance(v, torch.Tensor):
        return f"{space}{k}=tensor_{list(v.shape)}"
    elif isinstance(v, list) and len(v) > 3:
        return f"{space}{k}=list_{[len(v)]}"
    else:
        return f"{space}{k}={v}"
