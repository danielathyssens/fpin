import os
from parse_options import get_options

# python lib imports
import pprint as pp
from collections import defaultdict
import itertools
import numpy as np
import random
import torch
import time

# model lib imports
# from VRP_Loss1 import VRPLoss
from VRPModel_attn1 import VRP_Net
from utils_all.basic_funcs import USE_GPU, to_variable
from utils_all.get_path import greedy_path, make_valid
from data_utils.preprocess1 import import_test_data
from utils_all.postprocessing import create_data_model, main


def eval(opts):
    # Pretty print the run args
    pp.pprint(vars(opts))
    # handles
    v5 = False

    if opts.vrp_size == 20:
        vrp_to_solve = 'VRP20'
        # VRP50 = False
        Q = 30
        m = 4
        n = 20

        model_specs = {'dropout': 0.0,
                       'n_hidden': 1024,
                       'layers': 7,
                       'n_SOFTlayers': 0,
                       'mainDimension': 256}

    elif opts.vrp_size == 50:
        vrp_to_solve = 'VRP50'
        # VRP50 = True
        Q = 40
        m = 7
        n = 50

        model_specs = {'dropout': 0.0,
                       'n_hidden': 1024,
                       'layers': 9,
                       'n_SOFTlayers': 0,
                       'mainDimension': 256}

    elif opts.vrp_size == 100:
        vrp_to_solve = 'VRP100'
        Q = 50
        m = 11
        n = 100

        model_specs = {'dropout': 0.0,
                       'n_hidden': 1024,
                       'layers': 7,
                       'n_SOFTlayers': 0,
                       'mainDimension': 128}
    else:
        print('Error: No valid vrp_size specified')
        vrp_to_solve = None

    # parameters
    model_params = {'with_loads': False, 'attn_dotproduct': True, 'memory_efficient': True,
                    'avg_pool': False, 'residual': True, 'norm': True, 'self_pool': False, 'embedding_norm': True,
                    'weighting': True, 'fleet_dim': 4, 'cities_dim': 3, 'depot_dim': 4}

    # import Test Data
    Kool_Test_X = import_test_data(vrp_to_solve, opts.eval_data)
    nr_of_datapoints = len(Kool_Test_X)

    # start evaluation
    greedy = opts.greedy
    post_process = opts.post_process
    random_perm = True
    model = os.path.join(opts.model_dir, 'VRP_model.pth')
    r = list(range(m))
    perm_v = list(itertools.permutations(r))
    # GREEDY SOL EMPTY LISTS
    # greedy_sol_all,greedy_routes_all,final_loads_all,missing_all=[],[],[],[]
    # wrong_distMat=[]
    nr_of_routes, total_cost_v, total_cost_v_orig, total_cost_v_improved = [], [], [], []
    instances_solved, instances_failed = [], []
    Sols_improved, Dist_improved, nr_of_routes_improved, greedy_routes_improved = [], [], [], []

    greedy_sols, greedy_routes_all, final_loads_all, distance_of_solved, demand_of_solved = [], [], [], [], []
    traveled_dists_scld_all, traveled_dists_orig_all, Dist_improved_orig = [], [], []
    greedy_sol_failed, greedy_routes_failed, final_loads_failed, missing_all = [], [], [], []
    count_fails_greedy = 0
    # save sample_probs,sample_dists,sample_demands to work with BeamSearch
    sample_probs_solved, sample_probs_failed, remain_capa_failed, remain_capa_solved = [], [], [], []
    # BEAM SEARCH SETTINGS
    fail_ID = []
    count_fails = 0
    beam_sizes = 2000  # [20,200]
    solss, solss_v, sols_tens = [], [], []

    # restore model from save for testing
    VRP_model_restored = VRP_Net(model_specs['layers'], model_params['depot_dim'], model_params['cities_dim'],
                                 model_params['fleet_dim'], model_specs['mainDimension'], model_params['avg_pool'],
                                 model_params['residual'], model_params['norm'], model_specs['n_hidden'],
                                 model_specs['dropout'], model_params['self_pool'], model_params['embedding_norm'],
                                 model_specs['n_SOFTlayers'], model_params['weighting'], model_params['with_loads'],
                                 model_params['attn_dotproduct'], model_params['memory_efficient'])

    VRP_model_restored.load_state_dict(torch.load(model))
    pytorch_total_params = sum(p.numel() for p in VRP_model_restored.parameters())
    print('VRP_model_restored params', pytorch_total_params)
    pytorch_total_params_tr = sum(p.numel() for p in VRP_model_restored.parameters() if p.requires_grad)
    print('VRP_model_restored tr. params', pytorch_total_params_tr)
    VRP_model_restored.eval()
    VRP_model_restored.cuda()

    # run evaluation
    # GET PREDICTIONS ON TEST DATA
    i = 0
    tic = time.time()
    with torch.no_grad():
        for test_instance in Kool_Test_X:
            # Unpack X_groups
            fleet_batch_t = torch.FloatTensor(test_instance[0]).unsqueeze(0)
            depot_batch_t = torch.FloatTensor(test_instance[1]).unsqueeze(0)
            custom_batch_t = torch.FloatTensor(test_instance[2]).unsqueeze(0)
            demand_batch_t = torch.FloatTensor(test_instance[3]).unsqueeze(0)
            All_dists_batch_t = torch.FloatTensor(test_instance[4]).unsqueeze(0)

            # Transfer to GPU
            depot_batch_t, custom_batch_t, fleet_batch_t, All_dists_batch_t, demand_batch_t = to_variable(depot_batch_t,
                                                                                                          custom_batch_t,
                                                                                                          fleet_batch_t,
                                                                                                          All_dists_batch_t,
                                                                                                          demand_batch_t)

            # Forward pass: Compute predicted y (ROUTE-Probs) by passing x to the model
            VRP_probs_t, VRP_loads_t, sample_path_b0_t = VRP_model_restored(depot_batch_t,
                                                                            custom_batch_t,
                                                                            fleet_batch_t,
                                                                            demand_batch_t,
                                                                            All_dists_batch_t)
            if greedy:

                #### get predicted path_idx and remaining_capa for all test instances ####
                if random_perm:
                    r_ = list(range(m))
                    random.shuffle(r_)
                    path_idxs, remain_capa = greedy_path(VRP_probs_t[0], demand_batch_t[0], r_)
                    # GREEDY VAL PATH for TO-direction (based on probs)
                    greedy_sol, greedy_routes, final_loads, missing_ = make_valid(path_idxs, VRP_probs_t[0],
                                                                                  remain_capa, demand_batch_t[0])

                else:
                    count = 0
                    for perm in perm_v:
                        # print('curr perm:',perm)
                        path_idxs, remain_capa = greedy_path(VRP_probs_t[0], demand_batch_t[0], perm)
                        # GREEDY VAL PATH for TO-direction (based on probs)
                        greedy_sol, greedy_routes, final_loads, missing_ = make_valid(path_idxs, VRP_probs_t[0],
                                                                                      remain_capa, demand_batch_t[0])

                        # print('missing_',missing_)
                        if not list(missing_):
                            # if missing_ == 0
                            break

                if list(missing_):
                    # if missing_ == 1
                    # print('missing_',missing_)
                    # print('Try FROM-direction')
                    # if missing_==1 for TO-path ==> get FROM-path
                    if random_perm:
                        perm = r_
                    path_idxs, remain_capa = greedy_path(VRP_probs_t[0].transpose(1, 2), demand_batch_t[0], perm)

                    # GREEDY VAL PATH for FROM-direction (based on probs)
                    greedy_sol, greedy_routes, final_loads, missing_ = make_valid(path_idxs, VRP_probs_t[0],
                                                                                  remain_capa, demand_batch_t[0])

                # IF SOLVED (either TO- or FROM- direction)  --> SAVE + POSTPROCESS
                if not list(missing_):
                    # missing_ == 0:
                    remain_capa_solved.append(remain_capa)
                    instances_solved.append(test_instance)
                    sample_probs_solved.append(VRP_probs_t)
                    greedy_sols.append(greedy_sol)
                    greedy_routes_all.append([rout for rout in greedy_routes if len(rout) >= 3])
                    nr_of_routes.append(len([rout for rout in greedy_routes if len(rout) >= 3]))
                    final_loads_all.append(final_loads)
                    demand_of_solved.append(Kool_Test_X[i][3])

                    # get travelled distance
                    dist_orig = torch.FloatTensor(Kool_Test_X[i][4]).unsqueeze(0).expand(VRP_probs_t[0].size(0),
                                                                                         VRP_probs_t[0].size(1),
                                                                                         VRP_probs_t[0].size(1)).cuda()
                    dist_scaled = dist_orig * 1000
                    distance_of_solved.append(dist_orig)
                    traveled_dists_orig = greedy_sol * dist_orig
                    traveled_dists_scaled = greedy_sol * dist_scaled
                    traveled_dists_orig_all.append(torch.sum(traveled_dists_orig).item())
                    traveled_dists_scld_all.append(torch.sum(traveled_dists_scaled).item())
                    total_cost_v_orig.append(torch.sum(traveled_dists_orig).item() + len([rout for rout in greedy_routes
                                                                                          if len(rout) >= 3]))
                    # APPLY OR-TOOLS Search Heuristic
                    if post_process:
                        # preprocess routes and distance matrix
                        init_solution = []
                        for r in greedy_routes:
                            init_solution.append(r[1:-1])
                        # create data for OR tools
                        OR_data = create_data_model(Kool_Test_X[i][4] * 1000, init_solution,
                                                    list(map(int, Kool_Test_X[i][3][0] * Q)), Q, VRP_probs_t[0].size(0))
                        sol_improved = main(OR_data)
                        Sols_improved.append(sol_improved)
                        Dist_improved.append(sol_improved['total_dist'])
                        Dist_improved_orig.append(sol_improved['total_dist'] / 1000)
                        routes_improved = [sol_improved[i][0][0][1:] for i in range(m) if sol_improved[i][0][0][1:]]
                        # print('routes_improved', routes_improved)
                        greedy_routes_improved.append(routes_improved)
                        nr_of_routes_improved.append(len(routes_improved))
                        # print('greedy_routes',greedy_routes)
                        # print('sol_improved[total_dist]/1000',sol_improved['total_dist']/1000)
                        # print('len([rout for rout in greedy_routes if len(rout)>=3])',len([rout for rout in greedy_routes if len(rout)>=3]))
                        total_cost_v.append(
                            sol_improved['total_dist'] / 1000 + len([rout for rout in greedy_routes if len(rout) >= 3]))
                        total_cost_v_improved.append(
                            sol_improved['total_dist'] / 1000 + len(routes_improved))

                elif opts.guarantee_solution:
                    #print('add route to sol before post process')
                    init_solution = []
                    for r in greedy_routes:
                        init_solution.append(r[1:-1])
                    # add missing customers as route
                    #print('test_instance[3][0, missing_]', test_instance[3][0, missing_.cpu()])
                    #print('sum(demand_batch_t[missing_]', np.sum(test_instance[3][0, missing_.cpu()]))
                    if np.sum(test_instance[3][0, missing_.cpu()]) < 1.000001:
                        init_solution.append([x.item() for x in missing_])
                        extra_count = 1
                    else:
                        extra_route = []
                        extra_routes = []
                        cum_demand = 0
                        for x in missing_:
                            #print('x', x)
                            #print('test_instance[3][0,x]', test_instance[3][0, x])
                            if (cum_demand + test_instance[3][0, x]) < 1.00001:
                                extra_route.extend([x.item()])
                                cum_demand += test_instance[3][0, x]
                                #print('extra_route', extra_route)
                            else:
                                extra_routes.append(extra_route)
                                extra_route = [x.item()]

                        extra_routes.append(extra_route)
                        extra_count = len(extra_routes)
                        #print(len(extra_routes))
                        #print('extra_routes', extra_routes)
                        init_solution.extend(extra_routes)
                    #print('init_solution', init_solution)
                    # create data for OR tools
                    OR_data = create_data_model(Kool_Test_X[i][4] * 1000, init_solution,
                                                list(map(int, Kool_Test_X[i][3][0] * Q)), Q, (m + extra_count))
                    sol_improved = main(OR_data)
                    Sols_improved.append(sol_improved)
                    Dist_improved.append(sol_improved['total_dist'])
                    Dist_improved_orig.append(sol_improved['total_dist'] / 1000)
                    #print('greedy_routes', greedy_routes)
                    routes_improved = [sol_improved[i][0][0][1:] for i in range(m + 1) if sol_improved[i][0][0][1:]]
                    #print('routes_improved', routes_improved)
                    greedy_routes_improved.append(routes_improved)
                    nr_of_routes_improved.append(len(routes_improved))
                    # print('sol_improved[total_dist]/1000',sol_improved['total_dist']/1000)
                    # print('len([rout for rout in greedy_routes if len(rout)>=3])',len([rout for rout in greedy_routes if len(rout)>=3]))
                    total_cost_v.append(
                        sol_improved['total_dist'] / 1000 + len([rout for rout in greedy_routes if len(rout) >= 3]))
                    total_cost_v_improved.append(
                        sol_improved['total_dist'] / 1000 + len(routes_improved))

                else:
                    #print('final fail')
                    remain_capa_failed.append(remain_capa)
                    sample_probs_failed.append(VRP_probs_t)
                    instances_failed.append(test_instance)
                    greedy_sol_failed.append(greedy_sol)
                    greedy_routes_failed.append(greedy_routes)
                    final_loads_failed.append(final_loads)
                    missing_all.append(missing_)
                    count_fails_greedy += 1

            else:
                ########### DO BEAM SEARCH ON PROBS OUTPUT ##########
                sol, sol_tensor = run_beam(All_dists_batch_t, VRP_probs_t,
                                           demand_batch_t, beam_sizes, capacity=Q,
                                           raw_solution=False, no_beam_search=False)
                if sol_tensor is None:
                    count_fails += 1
                    fail_ID.append(i)

                else:
                    solss.append(sol.item())
                    sols_tens.append(sol_tensor)

            # INCREMENT FOR TEST_INSTANCE
            i += 1

    toc = time.time()
    hours, rem = divmod(toc - tic, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Ran testing in {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))

    # get outputs for greedy
    if greedy:
        greedy_solved = (nr_of_datapoints - count_fails_greedy) / nr_of_datapoints
        final_costs_orig_ = np.mean(traveled_dists_orig_all)
        final_costs_v = np.mean(total_cost_v)
        final_cost_v_orig = np.mean(total_cost_v_orig)
        percentage_solved_ = greedy_solved * 100
        # print(len(Dist_improved),len(greedy_sol_solved))
        print('count fails GREEDY:', count_fails_greedy)
        print('GREEDY percentage solved', percentage_solved_)
        print('FINAL OBJ ORIG (GREEDY):', final_costs_orig_)
        print('AVG Nr of routes ORIG', np.mean(nr_of_routes))
        print('total_cost_v ORIG routes+imporved dists', final_costs_v)
        print('total_cost_v ORIG', final_cost_v_orig)
        if post_process:
            final_costs = np.mean(Dist_improved_orig)
            print('FINAL OBJ POSTPROCESSED:', final_costs)
            print('AVG Nr of routes POSTPROCESSED', np.mean(nr_of_routes_improved))
            print('total_cost_v POSTPROCESSED', np.mean(total_cost_v_improved))


    else:
        print(solss)
        print('count fails:', count_fails)
        print(np.mean(solss))
        final_costs = np.mean(solss)
        final_costs_v = None
        percentage_solved_ = ((nr_of_datapoints - count_fails) / nr_of_datapoints) * 100

    return traveled_dists_orig_all, total_cost_v, Dist_improved_orig, nr_of_routes, nr_of_routes_improved


if __name__ == "__main__":
    final_costs_orig, final_v_costs_orig, final_costs_pp, r_len, r_len_imprvd = eval(get_options())
    save_outs = get_options().model_dir
    # Save nr_routes, costs
    with open(os.path.join(save_outs, 'eval_outs/routes_len.pt'), "wb") as f:
        torch.save(r_len, f)
    with open(os.path.join(save_outs, 'eval_outs/routes_len_pp.pt'), "wb") as f:
        torch.save(r_len_imprvd, f)
    with open(os.path.join(save_outs, 'eval_outs/final_costs.pt'), "wb") as f:
        torch.save(final_costs_orig, f)
    with open(os.path.join(save_outs, 'eval_outs/final_costs_pp.pt'), "wb") as f:
        torch.save(final_costs_pp, f)
