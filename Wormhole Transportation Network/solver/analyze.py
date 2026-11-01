from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np

from wormhole_udp import wormhole_traversal_udp
from solver.solution_utils import chromosome_to_paths


ROOT = Path(__file__).resolve().parents[1]

DATABASE_PATH = ROOT / "data" / "database.npz"
EXAMPLE_PATH = ROOT / "data" / "example.npy"


def calculate_arrival_times(problem, paths):
    """
    Calculate arrival time for each ship.
    """

    arrival_times = []

    for ship_idx, path in enumerate(paths):

        total_mean = 0.0

        for src, tgt in zip(path[:-1], path[1:]):

            total_mean += float(
                problem.network[src][tgt]["m"]
            )

        arrival_time = (
            total_mean
            + float(problem.delays[ship_idx])
        )

        arrival_times.append(arrival_time)

    return np.array(arrival_times, dtype=float)


def calculate_ship_costs(problem, paths):
    """
    Calculate accumulated variance for every ship,
    including node revisit penalties.
    """

    visits = defaultdict(list)

    # -----------------------------------
    # Record the time each node is visited
    # -----------------------------------

    for ship_idx, path in enumerate(paths):

        current_time = float(
            problem.delays[ship_idx]
        )

        for jump, (src, tgt) in enumerate(
            zip(path[:-1], path[1:])
        ):

            visits[src].append(
                (
                    ship_idx,
                    jump,
                    current_time,
                )
            )

            current_time += float(
                problem.network[src][tgt]["m"]
            )

        # Also register arrival at destination
        visits[path[-1]].append(
            (
                ship_idx,
                len(path) - 1,
                current_time,
            )
        )

    # -----------------------------------
    # Calculate revisit penalties
    # -----------------------------------

    penalties = {}

    for node_visits in visits.values():

        ordered_visits = sorted(
            node_visits,
            key=lambda item: item[2],
        )

        for order, (
            ship_idx,
            jump,
            _
        ) in enumerate(ordered_visits):

            if order >= 10:
                penalty = 1e10
            else:
                penalty = 10.0 ** order

            penalties[
                (ship_idx, jump)
            ] = penalty

    # -----------------------------------
    # Calculate cost for each ship
    # -----------------------------------

    costs = np.zeros(
        problem.n_ships,
        dtype=float,
    )

    for ship_idx, path in enumerate(paths):

        for jump, (src, tgt) in enumerate(
            zip(path[:-1], path[1:])
        ):

            variance = float(
                problem.network[src][tgt]["v"]
            )

            penalty = penalties[
                (ship_idx, jump)
            ]

            costs[ship_idx] += (
                variance * penalty
            )

    # Same scaling used by the UDP
    return costs * 10000.0


def main():

    problem = wormhole_traversal_udp(
        DATABASE_PATH
    )

    chromosome = np.load(
        EXAMPLE_PATH
    )

    paths = chromosome_to_paths(
        chromosome,
        problem.n_ships,
        problem.jump_limit,
        problem.destination,
    )

    # Official UDP result
    official_result = problem.fitness(
        chromosome
    )

    # ---------------------
    # Validate origins
    # ---------------------

    origins_valid = all(
        paths[ship_idx][0]
        in problem.origins[ship_idx]

        for ship_idx
        in range(problem.n_ships)
    )

    # ---------------------
    # Validate paths
    # ---------------------

    paths_valid = all(
        nx.is_path(
            problem.network,
            path,
        )

        for path in paths
    )

    # ---------------------
    # Calculate statistics
    # ---------------------

    arrivals = calculate_arrival_times(
        problem,
        paths,
    )

    costs = calculate_ship_costs(
        problem,
        paths,
    )

    arrival_range = (
        arrivals.max()
        - arrivals.min()
    )

    # ---------------------
    # Print network info
    # ---------------------

    print("\nNETWORK")
    print("-------")

    print(
        f"Nodes: {problem.n_bh}"
    )

    print(
        f"Edges: {problem.n_wh}"
    )

    print(
        f"Destination: "
        f"{problem.destination}"
    )

    print(
        f"Arrival window: "
        f"{problem.window}"
    )

    # ---------------------
    # Validation
    # ---------------------

    print("\nVALIDATION")
    print("----------")

    print(
        f"Origins valid: "
        f"{origins_valid}"
    )

    print(
        f"Paths valid: "
        f"{paths_valid}"
    )

    print(
        f"Arrival range: "
        f"{arrival_range:.9f}"
    )

    print(
        f"Window remaining: "
        f"{problem.window - arrival_range:.9f}"
    )

    # ---------------------
    # Ships
    # ---------------------

    print("\nSHIP ANALYSIS")
    print("-------------")

    for ship_idx, (
        path,
        cost,
        arrival
    ) in enumerate(
        zip(
            paths,
            costs,
            arrivals,
        ),
        start=1,
    ):

        print(
            f"Ship {ship_idx:2d}"
            f" | nodes={len(path):3d}"
            f" | cost={cost:10.6f}"
            f" | arrival={arrival:.6f}"
        )

    # ---------------------
    # Bottleneck
    # ---------------------

    bottleneck_idx = int(
        np.argmax(costs)
    )

    print("\nBOTTLENECK")
    print("----------")

    print(
        f"Ship: "
        f"{bottleneck_idx + 1}"
    )

    print(
        f"Cost: "
        f"{costs[bottleneck_idx]:.6f}"
    )

    # ---------------------
    # Official result
    # ---------------------

    print("\nOFFICIAL UDP RESULT")
    print("-------------------")

    print(
        f"Fitness: "
        f"{official_result[0]:.6f}"
    )

    print(
        f"Origin violation: "
        f"{official_result[1]}"
    )

    print(
        f"Path violation: "
        f"{official_result[2]}"
    )

    print(
        f"Window violation: "
        f"{official_result[3]:.9f}"
    )


if __name__ == "__main__":
    main()