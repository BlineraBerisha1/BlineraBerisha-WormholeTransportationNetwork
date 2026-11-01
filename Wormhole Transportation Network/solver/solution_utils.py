import numpy as np


def chromosome_to_paths(
    chromosome: np.ndarray,
    n_ships: int,
    jump_limit: int,
    destination: int,
):
    """
    Converts the chromosome into one path for each ship.

    The chromosome does not contain the destination.
    The destination is appended once at the end of each path.
    """

    paths = []

    for ship_idx in range(n_ships):

        start = ship_idx * jump_limit
        end = start + jump_limit

        section = chromosome[start:end]

        # Stop path at the first 0
        zero_positions = np.flatnonzero(section == 0)

        if zero_positions.size > 0:
            section = section[:zero_positions[0]]

        path = [int(node) for node in section]

        # UDP automatically considers the final jump
        # towards the destination.
        path.append(int(destination))

        paths.append(path)

    return paths