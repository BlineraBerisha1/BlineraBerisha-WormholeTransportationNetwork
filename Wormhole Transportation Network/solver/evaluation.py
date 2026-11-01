import numpy as np


def evaluate_solution(problem, chromosome):
    """
    Evaluate a complete chromosome using the official UDP.

    Returns fitness, constraints and whether the solution
    is valid.
    """

    chromosome = np.asarray(
        chromosome,
        dtype=np.int32,
    )

    expected_length = (
        problem.n_ships
        * problem.jump_limit
    )

    if len(chromosome) != expected_length:
        raise ValueError(
            f"Invalid chromosome length. "
            f"Expected {expected_length}, "
            f"got {len(chromosome)}."
        )

    result = problem.fitness(chromosome)

    fitness = float(result[0])
    origin_violation = int(result[1])
    path_violation = int(result[2])
    window_violation = float(result[3])

    valid = (
        origin_violation == 0
        and path_violation == 0
        and window_violation <= 0
    )

    return {
        "fitness": fitness,
        "origin_violation": origin_violation,
        "path_violation": path_violation,
        "window_violation": window_violation,
        "valid": valid,
    }


def is_better(candidate, current):
    """
    Decide whether candidate solution is better.

    A valid solution is always preferred over an invalid one.
    If both are valid, lower fitness wins.
    """

    if candidate["valid"] and not current["valid"]:
        return True

    if not candidate["valid"]:
        return False

    if not current["valid"]:
        return True

    return (
        candidate["fitness"]
        < current["fitness"]
    )