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

def paths_to_chromosome(
    paths,
    n_ships,
    jump_limit,
    destination
):
    chromosome = np.zeros(
        n_ships * jump_limit,
        dtype=int
    )

    for ship_idx, path in enumerate(paths):

        # Destination must NOT be stored
        path_without_destination = path.copy()

        if (
            path_without_destination
            and path_without_destination[-1] == destination
        ):
            path_without_destination = path_without_destination[:-1]

        if len(path_without_destination) > jump_limit:
            raise ValueError(
                f"Ship {ship_idx + 1} exceeds jump limit."
            )

        start = ship_idx * jump_limit
        end = start + len(path_without_destination)

        chromosome[start:end] = path_without_destination

    return chromosome