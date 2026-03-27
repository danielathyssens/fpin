import itertools
import random
import os
import time
import torch
import numpy as np
from models.PIM.PIM.utils_all.basic_funcs import to_variable
from models.PIM.PIM.utils_all import postprocessing, get_path
from models.PIM.PIM.data_utils import preprocess_PIM
from models.PIM.PIM.VRPModel import VRP_Net
import pprint as pp


def run_eval_pim(opts, test_inst, model):
    # evaluate permutation invariant VRP model
    st_pim = time.time()
    outs_pim = eval_pim(opts, test_inst, dat_x_pim, i, model, pim_prelim)
    time_pim += time.time() - st_pim
    duration_pim = time.time() - st_pim
    if len(outs_pim) > 1:
        return appnd(results_pim, outs_pim, duration_pim)
    else:
        count_fails += 1
        return count_fails


def eval_pim(opts, test_instance, test_x, i, vrp_model, prelims) -> list:
    pp.pprint(vars(opts))
    Q, m, n, perm_v, v_fixed_cost = prelims.Q, prelims.m, prelims.n, prelims.perm_v, prelims.v_fixed_cost
    with torch.no_grad():
        # Unpack test instance
        fleet, depot, custom, demand, dists = unpack(test_instance)
        # Compute route-probabilities (forward pass)
        vrp_probs, vrp_loads, sample_path = vrp_model(depot, custom, fleet, demand, dists)
        # Greedy path & fix:
        pred_sol, routes, loads, missing_ = fixed_greedy_path(opts, vrp_probs, demand, perm_v, m)

    # postprocess & get cost --> outs is either [1] or list of results
    # nr_routes, routes_all, total_cost, total_cost_v = [], [], [], []
    outs = get_costs(opts, routes, test_x[i], Q, m, n, v_fixed_cost, pred_sol, vrp_probs, test_instance,
                     missing_)

    return outs


def get_costs(opts, routes, curr_inst, Q, m, n, v_fixed_cost, pred_sol, vrp_probs, test_instance, missing):
    # solved instance
    if not list(missing):
        if opts.post_process:
            cost, tours, nr_tours, cost_v = postpr(routes, curr_inst, Q, m, n, v_fixed_cost)
            return [cost, tours, nr_tours, cost_v]
        else:
            tours = [rout for rout in routes if len(rout) >= 3]
            nr_tours = len([rout for rout in routes if len(rout) >= 3])
            # get travelled distance for this solution
            cost, fixed_costs = get_dist(pred_sol, curr_inst, routes, v_fixed_cost, vrp_probs)
            cost_v = (cost.item() + fixed_costs)
            return [cost.item(), tours, nr_tours, cost_v]

    # not solved - but guarantee solution is set
    elif opts.guarantee_solution:
        init_tours, extra_count = guaranteed_tour(routes, test_instance, missing)
        cost, tours, nr_tours, cost_v = postpr(init_tours, curr_inst, Q, (m + extra_count), n, v_fixed_cost)
        return [cost, tours, nr_tours, cost_v]

    # not solved and no guaranteed solution
    else:
        return [1]


# get preliminaries for evaluating PIM
def prelim(opts, data, device):
    # get size-specific handles
    vrp, Q, m, n, perm_v, v_cost, pim_specs, pim_params = get_handles(opts)
    prelims = {
        'cvrp': vrp,
        'Q': Q,
        'm': m,
        'n': n,
        'perm_v': perm_v,
        'v_cost': v_cost,
        'pim_specs': pim_specs,
        'pim_params': pim_params
    }
    # import Test Data
    preped_x, all_solvable = preprocess_PIM.import_test_data(vrp, data)
    # get & restore model to evaluate
    model = init_model(opts, VRP_Net, pim_specs, pim_params, device)

    return model, preped_x, prelims


# get handles
def get_handles(opts):
    if opts.vrp_size == 20:
        vrp_to_solve = 'VRP20'
        Q = 30
        m = 4
        n = 20
        v_fixed_cost = 35
        r = list(range(m))
        perm_v = list(itertools.permutations(r))

        model_specs = {'dropout': 0.0,
                       'n_hidden': 1024,
                       'layers': 9,
                       'mainDimension': 256}

    elif opts.vrp_size == 50:
        vrp_to_solve = 'VRP50'
        Q = 40
        m = 7
        n = 50
        v_fixed_cost = 50
        r = list(range(m))
        perm_v = list(itertools.permutations(r))

        model_specs = {'dropout': 0.0,
                       'n_hidden': 1024,
                       'layers': 9,
                       'mainDimension': 256}

    elif opts.vrp_size == 100:
        vrp_to_solve = 'VRP100'
        Q = 50
        m = 11
        n = 100
        v_fixed_cost = 80
        r = list(range(m))
        perm_v = list(itertools.permutations(r))

        model_specs = {'dropout': 0.0,
                       'n_hidden': 1024,
                       'layers': 7,
                       'mainDimension': 128}
    else:
        print('Error: No valid vrp_size specified')
        vrp_to_solve = None
        Q = None
        m = None
        n = None
        v_fixed_cost = None
        perm_v = None
        model_specs = None

    # model parameters (indep. of model-size)
    model_params = {'with_loads': False, 'avg_pool': False, 'residual': True, 'norm': True,
                    'self_pool': False, 'embedding_norm': True, 'weighting': True, 'fleet_dim': 4,
                    'cities_dim': 3, 'depot_dim': 4}

    return vrp_to_solve, Q, m, n, perm_v, v_fixed_cost, model_specs, model_params


def init_model(opts, net, model_specs, model_params, device):
    model = os.path.join(opts.model_dir, 'VRP_model.pth')
    model_ = net(model_specs['layers'], model_params['depot_dim'], model_params['cities_dim'],
                 model_params['fleet_dim'], model_specs['mainDimension'], model_params['avg_pool'],
                 model_params['residual'], model_params['norm'], model_specs['n_hidden'],
                 model_specs['dropout'], model_params['self_pool'], model_params['embedding_norm'],
                 model_params['weighting'], model_params['with_loads'])
    model_.load_state_dict(torch.load(model))
    model_.to(device)
    model_.eval()

    return model_


def unpack(instance):
    # Unpack X_groups
    fleet_t = torch.FloatTensor(instance[0]).unsqueeze(0)
    depot_t = torch.FloatTensor(instance[1]).unsqueeze(0)
    custom_t = torch.FloatTensor(instance[2]).unsqueeze(0)
    demand_t = torch.FloatTensor(instance[3]).unsqueeze(0)
    dists_t = torch.FloatTensor(instance[4]).unsqueeze(0)

    # Transfer to GPU
    depot_t, custom_t, fleet_t, dists_t, demand_t = to_variable(depot_t, custom_t, fleet_t, dists_t, demand_t)

    return fleet_t, depot_t, custom_t, demand_t, dists_t


def fixed_greedy_path(opts, probs, demand, perm_v, m):
    if opts.random_perm:
        random.shuffle(list(range(m)))
        pred_path, cap_r = get_path.greedy_path(probs[0], demand[0], random.shuffle(list(range(m))))
        pred_sol, routes, loads, missing_ = get_path.make_valid(pred_path, probs[0], cap_r, demand[0])
        if not list(missing_):
            return pred_sol, routes, loads, missing_
        else:
            # if missing_ for TO-path ==> try FROM-path
            pred_path, cap_r = get_path.greedy_path(probs[0].transpose(1, 2), demand[0], random.shuffle(list(range(m))))
            pred_sol, routes, loads, missing_ = get_path.make_valid(pred_path, probs[0], cap_r, demand[0])
            return pred_sol, routes, loads, missing_
    else:
        count = 0
        for perm in perm_v:
            pred_path, cap_r = get_path.greedy_path(probs[0], demand[0], perm)
            # GREEDY VAL PATH for TO-direction (based on probs)
            pred_sol, routes, loads, missing_ = get_path.make_valid(pred_path, probs[0], cap_r, demand[0])
            if not list(missing_):
                return pred_sol, routes, loads, missing_
            else:
                # if missing_ for TO-path ==> try FROM-path
                pred_path, cap_r = get_path.greedy_path(probs[0].transpose(1, 2), demand[0], perm)
                pred_sol, routes, loads, missing_ = get_path.make_valid(pred_path, probs[0], cap_r, demand[0])
                if not list(missing_):
                    return pred_sol, routes, loads, missing_
                elif list(missing_) and count == len(perm_v):
                    return pred_sol, routes, loads, missing_


def get_dist(greedy_sol, curr_inst, greedy_routes, v_fixed_cost, vrp_probs):
    dist_orig = torch.FloatTensor(curr_inst[4]).unsqueeze(0).expand(vrp_probs[0].size(0),
                                                                    vrp_probs[0].size(1),
                                                                    vrp_probs[0].size(1)).cuda()
    traveled_dists = greedy_sol * dist_orig
    v_fixed_costs = len([rout for rout in greedy_routes if len(rout) >= 3]) * v_fixed_cost

    return torch.sum(traveled_dists), v_fixed_costs


def postpr(greedy_routes, curr_inst, Q, m, n, v_fixed_cost):
    dist_orig = torch.FloatTensor(curr_inst[4]).unsqueeze(0).expand(m, n, n).cuda()
    dist_scaled = dist_orig * 1000
    # preprocess routes and distance matrix
    # init_solution = []
    # for r in greedy_routes:
    #    init_solution.append(r[1:-1])
    # create data for OR tools
    OR_data = postprocessing.create_data_model(curr_inst[4] * 1000, [r[1:-1] for r in greedy_routes],
                                               list(map(int, curr_inst[3][0] * Q)), Q, m)
    sol_improved = postprocessing.main(OR_data)
    cost_improved = sol_improved['total_dist'] / 1000
    routes_improved = [sol_improved[i][0][0][1:] for i in range(m) if sol_improved[i][0][0][1:]]
    v_cost = cost_improved + (len(routes_improved) * v_fixed_cost)
    return cost_improved, routes_improved, len(routes_improved), v_cost


def guaranteed_tour(greedy_routes, test_inst, missing):
    init_solution = [r[1:-1] for r in greedy_routes]
    # for r in greedy_routes:
    #    init_solution.append(r[1:-1])
    # add missing customers as route
    if np.sum(test_inst[3][0, missing.cpu()]) < 1.000001:
        init_solution.append([x.item() for x in missing])
        extra_count = 1
    else:
        extra_route = []
        extra_routes = []
        cum_demand = 0
        for x in missing:
            if (cum_demand + test_inst[3][0, x]) < 1.00001:
                extra_route.extend([x.item()])
                cum_demand += test_inst[3][0, x]
            else:
                extra_routes.append(extra_route)
                extra_route = [x.item()]

        extra_routes.append(extra_route)
        # extra_count = len(extra_routes)
        init_solution.extend(extra_routes)
    return init_solution


def appnd(listt, outs, duration):
    # costs, tours, nr_tours, cost_v, duration
    listt.append((outs[0], outs[1], outs[2], outs[3], duration))
    return listt
