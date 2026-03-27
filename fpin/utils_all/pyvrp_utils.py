from __future__ import annotations
from typing import List, Tuple, Optional

import numpy as np
import torch

from pyvrp import Model, Population, GeneticAlgorithm, PenaltyManager, RandomNumberGenerator
from pyvrp.stop import MaxRuntime, MaxIterations
from pyvrp.search import LocalSearch
from pyvrp.crossover import selective_route_exchange as srex  # default is fine; SREX also available
from pyvrp import Solution, ProblemData, GeneticAlgorithmParams
from pyvrp.diversity import broken_pairs_distance
from pyvrp import (
    Model, GeneticAlgorithm, GeneticAlgorithmParams,
    Population, CostEvaluator, RandomNumberGenerator,
    PenaltyManager, PenaltyParams
)
from pyvrp.search import (
    NODE_OPERATORS,
    ROUTE_OPERATORS,
    LocalSearch,
    compute_neighbours,
)

import matplotlib.pyplot as plt

from pyvrp.plotting import plot_result

# PyVRP takes integers internally; we’ll scale distances/costs.
# Routes given to Solution(data, routes) are client indices only (no depot, clients are 1..N if your depot is 0).
# Initial population seeding is supported directly in GeneticAlgorithm(initial_solutions=...)


def to_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().contiguous().numpy()


def build_biased_costs(
        dist_mat: torch.Tensor,  # [N, N], N = 1 + num_clients, float
        edge_probs: torch.Tensor,  # [M, N, N] or [N, N]; if per-vehicle, we’ll max/mean over M
        tau: float = 0.5,
        alpha: float = 1.0,
        agg: str = "max",  # how to aggregate per-vehicle probs -> single matrix
        scale: int = 10_000,  # int scaling for PyVRP
        verbose: bool = False
) -> np.ndarray:
    """
    Returns an integer N x N biased cost matrix for PyVRP.
    """
    D = to_np(dist_mat).astype(np.float64)
    D = 0.5 * (D + D.T);
    np.fill_diagonal(D, 0.0)

    # print("D min/max:", float(dist_mat.min()), float(dist_mat.max()))
    # print("P min/max:", float(edge_probs.min()), float(edge_probs.max()))

    if edge_probs.dim() == 3:
        P = edge_probs.detach().cpu().float().max(
            dim=0).values if agg == "max" else edge_probs.detach().cpu().float().mean(dim=0)
        P = P.numpy().astype(np.float64)
    else:
        P = to_np(edge_probs).astype(np.float64)

    # Symmetrise (safer; HGS expects metric-like costs)
    P = np.clip(P, 0.0, 1.0)
    P = 0.5 * (P + P.T);
    np.fill_diagonal(P, 0.0)

    C = alpha * D + tau * (1.0 - P)
    C_int = np.rint(C * scale).astype(np.int64)
    np.fill_diagonal(C_int, 0)
    if verbose:
        print("D min/max:", float(dist_mat.min()), float(dist_mat.max()))
        print("P min/max:", float(edge_probs.min()), float(edge_probs.max()))
        print("C_int min/max:", float(C_int.min()), float(C_int.max()))

    return C_int


def build_pyvrp_instance(
        coords: np.ndarray,  # [N, 2] float (optional if you only have dist matrix)
        demands: np.ndarray,  # [N] with demands[0] == 0 (depot)
        capacity: int,  # integer capacity (scale your normalised demand accordingly)
        num_vehicles: int,
        biased_costs: np.ndarray,  # [N, N] int costs from build_biased_costs
        verbose: bool = False
) -> Tuple[Model, ProblemData]:
    """
    Constructs a CVRP instance in PyVRP with a single depot (node 0),
    unit duration, and biased distance costs.
    """
    m = Model()

    # One vehicle type, repeated num_vehicles times.
    # print("cost_int min/max:", int(cost_int.min()), int(cost_int.max()))
    if verbose:
        print("sum demand:", float(demands.sum()),
              "cap:", capacity,
              "cap:", int(round(capacity)))

    # print('coords.shape', coords.shape)
    # print('demands.shape', demands.shape)
    coords = to_np(coords)
    # demands = to_np(demands)
    # print('coords[0, 0]', coords[0, 0])
    # print('demands[0', demands[0])
    m.add_vehicle_type(num_vehicles, capacity=capacity)
    # m.add_vehicle_type(capacity=int(capacity), num_available=int(num_vehicles))

    # Add depot and clients. Coords optional, but nice for plotting.
    # depot = (
    depot = m.add_depot(x=coords[0, 0], y=coords[0, 1])
    clients = []
    # int(demands[i]
    for i in range(1, coords.shape[0]):
        clients.append(
            m.add_client(

                x=coords[i, 0],
                y=coords[i, 1],
                delivery=int(demands[i]),  # integer (scale if needed)

            ))

    # Add edges with our biased integer costs.
    #locs = m.locations  # [depot] + clients, same order of creation

    # print('biased_costs[0]', biased_costs[0])
    locs = [depot] + clients  # length N
    N = len(locs)
    # print(N)
    for i in range(N):
        for j in range(N):
            m.add_edge(locs[i], locs[j], distance=int(biased_costs[i, j]))

    # print('m', m)

    data = m.data()
    # print('data', data)
    return m, data


def routes_to_solution(
        data: ProblemData,
        routes: List[List[int]],
        depot_idx: int = 0,
) -> Solution:
    """
    Convert your per-vehicle routes (each includes depot indices 0 at ends)
    to a PyVRP Solution(routes=[client lists]), where clients are 1..N-1.

    Input route format example:
      [0, 12, 9, 25, 0]  -> stored as [12, 9, 25] in Solution
    """
    clean = []
    for r in routes:
        r = list(r)
        if len(r) > 2 and r[0] == depot_idx:
            r = r[1:]
        if len(r) >= 1 and r[-1] == depot_idx:
            r = r[:-1]
        if r == [0]:
            continue
        # Map 0..N-1 -> (clients are 1..N-1)
        for v in r:
            assert v != depot_idx, "Client list must not contain depot"
        clean.append([int(v) for v in r])  # already 1..N-1 if you used 0 as depot
    return Solution(data, clean)


def run_hgs_with_seeds(
        data: ProblemData,
        seed_solutions: List[Solution],
        time_sec: float = 1.0,
        max_iters: Optional[int] = None,
        rng_seed: int = 42,
        plot: bool = False,
        verbose: bool = False
):
    """
    Minimal HGS run with your seeds injected into the initial population.
    """

    # from pyvrp import XorShift128
    from pyvrp import RandomNumberGenerator
    from pyvrp.search import LocalSearch, compute_neighbours

    rng = RandomNumberGenerator(seed=42)  # e.g., seed = 42
    neighbours = compute_neighbours(data)  # optionally: compute_neighbours(data, num_neighbours=50)

    params = GeneticAlgorithmParams(
        repair_probability=0.8,  # default is fine
        nb_iter_no_improvement=200  # default is fine
    )

    # Typical default choices (can tune later)
    pm = PenaltyManager(initial_penalties=([1e4], 1e4, 1e4), params=PenaltyParams()).init_from(data)
    pop = Population(diversity_op=broken_pairs_distance)

    # print('pop.num_feasible()', pop.num_feasible())
    # print('pop.num_infeasible()', pop.num_infeasible())
    cost_eval = CostEvaluator(load_penalties=[20],  # capacity penalty; large enough to discourage infeasibility
                              tw_penalty=20,  # time window penalty (harmless if no TWs)
                              dist_penalty=0.0  # base distance multiplier)
                              )
    for sol in seed_solutions:
        pop.add(sol, cost_evaluator=cost_eval)

    if verbose:
        print('pop.num_feasible()', pop.num_feasible())
        print('pop.num_infeasible()', pop.num_infeasible())
        print('cost_eval.penalised_cost(seed_solutions[0])', cost_eval.penalised_cost(seed_solutions[0]))
        print('seed_solutions', seed_solutions)
    rng = RandomNumberGenerator(rng_seed)
    ls = LocalSearch(data, rng, neighbours)
    #

    for node_op in NODE_OPERATORS:
        ls.add_node_operator(node_op(data))

    for route_op in ROUTE_OPERATORS:
        ls.add_route_operator(route_op(data))

    new_sol = ls.search(seed_solutions[0], cost_eval)
    if verbose:
        print('cost_eval.penalised_cost(new_sol)', cost_eval.penalised_cost(new_sol))
        print('new_sol.routes()', new_sol.routes())
        print('seed_solutions[0].is_feasible()', seed_solutions[0].is_feasible())

    # print('new_sol.is_feasible()', new_sol.is_feasible())
    ga = GeneticAlgorithm(
        data=data,
        penalty_manager=pm,
        rng=rng,
        population=pop,
        search_method=ls,
        crossover_op=srex,
        initial_solutions=seed_solutions,
        params=params,  # defaults are already good
    )

    stop = MaxRuntime(time_sec) if max_iters is None else MaxIterations(max_iters)
    res = ga.run(stop=stop)  # , display=False
    if verbose:
        print('res', res)
        print('res.best.is_feasible()', res.best.is_feasible())
    if plot:
        fig = plt.figure(figsize=(12, 8))

        plot_result(res, data, fig=fig)
        plt.tight_layout()
    return res.best
