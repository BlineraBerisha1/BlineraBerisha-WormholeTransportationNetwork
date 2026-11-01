def evaluate_solution(problem, chromosome):

    fitness, origin, path, window = problem.fitness(
        chromosome
    )

    valid = (
        origin == 0
        and path == 0
        and window <= 0
    )

    return {
        "fitness": float(fitness),
        "origin_violation": int(origin),
        "path_violation": int(path),
        "window_violation": float(window),
        "valid": valid,
    }