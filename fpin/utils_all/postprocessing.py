from __future__ import print_function
import torch
import numpy as np
import random
from .basic_funcs import zeros, ones
import sys
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp


def create_data_model(locs, init_routes, demand_lst, capa, num_vehicles):
    """Stores the data for the problem."""
    data = {}
    data['distance_matrix'] = calculate_distances(locations=locs, distance_metric=l2_distance)
    data['initial_routes'] = init_routes
    data['num_vehicles'] = num_vehicles
    data['depot'] = 0
    data['demands'] = demand_lst
    data['vehicle_capacities'] = [capa] * num_vehicles

    return data

def print_solution(data, manager, routing, solution):
    """Prints solution on console."""
    total_distance = 0
    total_load = 0
    route_plan = {}
    max_route_distance = 0
    #print(solution)
    #################
    #from me
    #info_v=[]
    #################
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
        route_distance = 0
        route_load = 0
        #################
        #from me
        info_v = []
        plan_v = []
        cust_ids = []
        load_lst = []
        #################
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route_load += data['demands'][node_index]
            plan_output += ' {} -> '.format(manager.IndexToNode(index))
            #################
            #from me
            cust_ids.append(node_index)
            load_lst.append(route_load)
            ##################
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id)
        plan_output += '{}\n'.format(manager.IndexToNode(index))
        plan_output += 'Distance of the route: {}m\n'.format(route_distance)
        #print('\n plan_output',plan_output)
        max_route_distance = max(route_distance, max_route_distance)

        #################
        capa = data['vehicle_capacities'][vehicle_id]
        ##################
        #print('Maximum of the route distances: {}m'.format(max_route_distance))   
        ##################
        #from me
        plan_v.append(cust_ids)
        plan_v.append(load_lst)
        #print('plan_v',plan_v)
        ##################

        total_distance += route_distance
        total_load += route_load

        ##################
        #from me
        info_v.append(plan_v)
        info_v.append(capa)
        info_v.append(total_distance)
        route_plan[vehicle_id] = info_v
        ##################

    #print('\n THIS IS THE ROUT PLAN FOR VEHICLE 0:')
    #print(route_plan[0])
    #print('\n')
    #print('Total distance of all routes: {}m'.format(total_distance))
    #print('Total load of all routes: {}'.format(total_load))
    #################
    #from me
    route_plan['total_dist'] = total_distance
    #################

    return route_plan


def main(data, m_to_use):
    """Solve the CVRP problem."""
    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix'])-1,
                                           data['num_vehicles'], data['depot'])

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    # Create and register a transit callback.
    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add Capacity constraint.
    def demand_callback(from_index):
        """Returns the demand of the node."""
        # Convert from routing variable Index to demands NodeIndex.
        from_node = manager.IndexToNode(from_index)
        return data['demands'][from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(
        demand_callback)
    assert len(data['vehicle_capacities']) == data['num_vehicles']
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        data['vehicle_capacities'],  # vehicle maximum capacities
        True,  # start cumul to zero
        'Capacity')

    # print(data['initial_routes'])
    initial_solution = routing.ReadAssignmentFromRoutes(data['initial_routes'],
                                                        True)
    print_solution(data, manager, routing, initial_solution)

    # SETTING "ADVANCED" SEARCH HEURISTIC
    # search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    # search_parameters.local_search_metaheuristic = (
    #    routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    # search_parameters.time_limit.seconds = 30
    # search_parameters.log_search = True

    # Set default search parameters.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.time_limit.seconds = 10

    # Solve the problem.
    solution = routing.SolveFromAssignmentWithParameters(
        initial_solution, search_parameters)

    # Print solution on console.
    if solution:
        plan = print_solution(data, manager, routing, solution)

    else:
        plan = print('No solution')

    return plan


def targ_as_lst(idcs_trg, n):
    '''idcs_trg is a tensor containing the indices of next visited nodes'''
    nxt_idx = [0]
    nxt = idcs_trg[0].item()
    for _ in range(n):
        cur_idx = nxt
        if cur_idx != 0:
            nxt_idx.append(cur_idx)
            nxt = idcs_trg[cur_idx].item()
    nxt_idx.append(0)
    return nxt_idx


def calculate_distances(locations, distance_metric=None, round_to_int=True):
    """Calculate distances between locations as matrix.
    If no distance_metric is specified, uses l2 euclidean distance"""
    metric = l2_distance if distance_metric is None else distance_metric

    num_locations = len(locations)
    matrix = {}

    for from_node in range(num_locations):
        matrix[from_node] = {}
        for to_node in range(num_locations):
            x1 = locations[from_node][0]
            y1 = locations[from_node][1]
            x2 = locations[to_node][0]
            y2 = locations[to_node][1]
            if round_to_int:
                matrix[from_node][to_node] = int(round(metric(x1, y1, x2, y2), 0))
            else:
                matrix[from_node][to_node] = metric(x1, y1, x2, y2)

    return matrix


def l2_distance(x1, y1, x2, y2):
    """Normal 2d euclidean distance."""
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
