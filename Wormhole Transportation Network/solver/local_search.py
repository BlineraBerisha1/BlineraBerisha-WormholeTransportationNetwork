from pathlib import Path

import numpy as np

from wormhole_udp import wormhole_traversal_udp

from solver.solution_utils import (
    chromosome_to_paths,
    paths_to_chromosome,
)

from solver.evaluation import (
    evaluate_solution,
)

from solver.analyze import (
    calculate_arrival_times,
    calculate_ship_costs,
)


ROOT = Path(__file__).resolve().parents[1]

DATABASE_PATH = (
    ROOT
    / "data"
    / "database.npz"
)

EXAMPLE_PATH = (
    ROOT
    / "data"
    / "example.npy"
)

RESULTS_DIR = (
    ROOT
    / "results"
)


# Number of optimization rounds
MAX_ROUNDS = 5

# We first rank candidates using a cheap heuristic.
# Only the best candidates are evaluated using the
# full official UDP.
MAX_UDP_EVALUATIONS_PER_ROUND = 80

EPSILON = 1e-12


def find_replacement_candidates(
    problem,
    paths,
    ship_idx,
    arrival_times,
):
    """
    Find possible single-node replacements for one ship.

    Original:

        A -> B -> C

    Candidate:

        A -> X -> C

    X is valid only if both edges exist:

        A -> X
        X -> C
    """

    path = paths[ship_idx]

    candidates = []

    # Arrival times of all OTHER ships
    other_arrivals = np.delete(
        arrival_times,
        ship_idx,
    )

    # Do not modify:
    # position 0       -> origin
    # last position    -> destination

    for position in range(
        1,
        len(path) - 1,
    ):

        previous_node = path[position - 1]
        current_node = path[position]
        next_node = path[position + 1]

        # --------------------------------
        # Current two edges
        # --------------------------------

        old_mean = (
            float(
                problem.network
                [previous_node]
                [current_node]["m"]
            )
            +
            float(
                problem.network
                [current_node]
                [next_node]["m"]
            )
        )

        old_variance = (
            float(
                problem.network
                [previous_node]
                [current_node]["v"]
            )
            +
            float(
                problem.network
                [current_node]
                [next_node]["v"]
            )
        )

        # --------------------------------
        # Find X such that:
        #
        # previous -> X
        # X -> next
        # --------------------------------

        outgoing = set(
            problem.network.successors(
                previous_node
            )
        )

        incoming = set(
            problem.network.predecessors(
                next_node
            )
        )

        possible_nodes = (
            outgoing
            & incoming
        )

        # Current node is not a modification
        possible_nodes.discard(
            current_node
        )

        # Destination should stay only
        # at the end of the path
        possible_nodes.discard(
            problem.destination
        )

        # --------------------------------
        # Evaluate candidates cheaply
        # --------------------------------

        for candidate_node in possible_nodes:

            candidate_node = int(
                candidate_node
            )

            # Avoid creating loops/repeated nodes
            if candidate_node in path:
                continue

            new_mean = (
                float(
                    problem.network
                    [previous_node]
                    [candidate_node]["m"]
                )
                +
                float(
                    problem.network
                    [candidate_node]
                    [next_node]["m"]
                )
            )

            mean_delta = (
                new_mean
                - old_mean
            )

            new_arrival = (
                arrival_times[ship_idx]
                + mean_delta
            )

            # -----------------------------
            # Window pre-check
            # -----------------------------

            if len(other_arrivals) > 0:

                max_gap = np.max(
                    np.abs(
                        other_arrivals
                        - new_arrival
                    )
                )

                if (
                    max_gap
                    > problem.window + 1e-7
                ):
                    continue

            # -----------------------------
            # Variance heuristic
            # -----------------------------

            new_variance = (
                float(
                    problem.network
                    [previous_node]
                    [candidate_node]["v"]
                )
                +
                float(
                    problem.network
                    [candidate_node]
                    [next_node]["v"]
                )
            )

            variance_delta = (
                new_variance
                - old_variance
            )

            candidates.append(
                {
                    "position": position,
                    "old_node": current_node,
                    "new_node": candidate_node,
                    "variance_delta": (
                        variance_delta
                    ),
                    "mean_delta": mean_delta,
                }
            )

    # Candidates expected to reduce variance
    # are checked first.
    candidates.sort(
        key=lambda candidate:
        candidate["variance_delta"]
    )

    return candidates


def main():

    print()
    print("LOCAL SEARCH")
    print("============")

    problem = wormhole_traversal_udp(
        DATABASE_PATH
    )

    # --------------------------------
    # Load starting solution
    # --------------------------------

    best_chromosome = np.load(
        EXAMPLE_PATH
    ).astype(np.int32)

    best_result = evaluate_solution(
        problem,
        best_chromosome,
    )

    if not best_result["valid"]:
        raise ValueError(
            "Starting solution is not valid."
        )

    print(
        f"\nStarting fitness: "
        f"{best_result['fitness']:.6f}"
    )

    # --------------------------------
    # Optimization rounds
    # --------------------------------

    for round_number in range(
        1,
        MAX_ROUNDS + 1,
    ):

        print()
        print(
            f"ROUND {round_number}"
        )
        print(
            "-" * 30
        )

        # Convert current best chromosome
        # into paths.
        paths = chromosome_to_paths(
            best_chromosome,
            problem.n_ships,
            problem.jump_limit,
            problem.destination,
        )

        # --------------------------------
        # Find current bottleneck ship
        # --------------------------------

        ship_costs = (
            calculate_ship_costs(
                problem,
                paths,
            )
        )

        bottleneck_idx = int(
            np.argmax(ship_costs)
        )

        print(
            f"Bottleneck ship: "
            f"{bottleneck_idx + 1}"
        )

        print(
            f"Bottleneck cost: "
            f"{ship_costs[bottleneck_idx]:.6f}"
        )

        # --------------------------------
        # Arrival times
        # --------------------------------

        arrival_times = (
            calculate_arrival_times(
                problem,
                paths,
            )
        )

        # --------------------------------
        # Generate candidate replacements
        # --------------------------------

        candidates = (
            find_replacement_candidates(
                problem,
                paths,
                bottleneck_idx,
                arrival_times,
            )
        )

        print(
            f"Possible replacements: "
            f"{len(candidates)}"
        )

        if not candidates:

            print(
                "No replacement candidates found."
            )

            break

        # --------------------------------
        # Evaluate only best heuristic
        # candidates using official UDP
        # --------------------------------

        candidates_to_test = candidates[
            :MAX_UDP_EVALUATIONS_PER_ROUND
        ]

        round_best_chromosome = None
        round_best_result = None
        round_best_change = None

        tested = 0

        for candidate in candidates_to_test:

            tested += 1

            # Copy all paths
            candidate_paths = [
                path.copy()
                for path in paths
            ]

            position = (
                candidate["position"]
            )

            new_node = (
                candidate["new_node"]
            )

            # Replace B with X
            candidate_paths[
                bottleneck_idx
            ][position] = new_node

            # Convert back to chromosome
            candidate_chromosome = (
                paths_to_chromosome(
                    candidate_paths,
                    problem.n_ships,
                    problem.jump_limit,
                    problem.destination,
                )
            )

            # Official evaluation
            result = evaluate_solution(
                problem,
                candidate_chromosome,
            )

            # Invalid?
            if not result["valid"]:
                continue

            # Must improve current best
            if (
                result["fitness"]
                >=
                best_result["fitness"]
                - EPSILON
            ):
                continue

            # Best solution found
            # during THIS round
            if (
                round_best_result is None
                or
                result["fitness"]
                <
                round_best_result["fitness"]
            ):

                round_best_chromosome = (
                    candidate_chromosome.copy()
                )

                round_best_result = result

                round_best_change = (
                    candidate
                )

        print(
            f"UDP evaluations: {tested}"
        )

        # --------------------------------
        # No improvement
        # --------------------------------

        if round_best_result is None:

            print(
                "No valid improvement "
                "found this round."
            )

            break

        # --------------------------------
        # Accept best improvement
        # --------------------------------

        old_fitness = (
            best_result["fitness"]
        )

        best_chromosome = (
            round_best_chromosome
        )

        best_result = (
            round_best_result
        )

        improvement = (
            old_fitness
            - best_result["fitness"]
        )

        print()
        print(
            f"Changed node "
            f"{round_best_change['old_node']}"
            f" -> "
            f"{round_best_change['new_node']}"
        )

        print(
            f"Position: "
            f"{round_best_change['position']}"
        )

        print(
            f"Old fitness: "
            f"{old_fitness:.6f}"
        )

        print(
            f"New fitness: "
            f"{best_result['fitness']:.6f}"
        )

        print(
            f"Improvement: "
            f"{improvement:.6f}"
        )

        print(
            f"Window constraint: "
            f"{best_result['window_violation']:.9f}"
        )

    # --------------------------------
    # Save best solution
    # --------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        RESULTS_DIR
        / "best_local_search.npy"
    )

    np.save(
        output_path,
        best_chromosome,
    )

    print()
    print("=" * 40)

    print(
        f"FINAL FITNESS: "
        f"{best_result['fitness']:.6f}"
    )

    print(
        f"VALID: "
        f"{best_result['valid']}"
    )

    print(
        f"Origin violations: "
        f"{best_result['origin_violation']}"
    )

    print(
        f"Path violations: "
        f"{best_result['path_violation']}"
    )

    print(
        f"Window violation: "
        f"{best_result['window_violation']:.9f}"
    )

    print(
        f"Saved to: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()