from pathlib import Path

import numpy as np

from wormhole_udp import wormhole_traversal_udp


ROOT = Path(__file__).resolve().parents[1]

DATABASE_PATH = ROOT / "data" / "database.npz"
EXAMPLE_PATH = ROOT / "data" / "example.npy"


def main():
    problem = wormhole_traversal_udp(DATABASE_PATH)

    solution = np.load(EXAMPLE_PATH)

    result = problem.fitness(solution)

    fitness = result[0]
    origin_violation = result[1]
    path_violation = result[2]
    window_violation = result[3]

    valid = (
        origin_violation == 0
        and path_violation == 0
        and window_violation <= 0
    )

    print("\nBASELINE SOLUTION")
    print("-----------------")
    print(f"Chromosome length: {len(solution)}")
    print(f"Fitness: {fitness:.6f}")
    print(f"Origin violations: {origin_violation}")
    print(f"Path violations: {path_violation}")
    print(f"Window constraint: {window_violation:.9f}")
    print(f"Valid solution: {valid}")


if __name__ == "__main__":
    main()