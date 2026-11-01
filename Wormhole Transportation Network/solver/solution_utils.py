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
    n_ships: int,
    jump_limit: int,
    destination: int,
):
    """
    Convert the 12 paths back into the chromosome format.

    The destination is NOT stored in the chromosome because
    wormhole_udp.py appends it automatically.
    """

    if len(paths) != n_ships:
        raise ValueError(
            f"Expected {n_ships} paths, got {len(paths)}."
        )

    chromosome = np.zeros(
        n_ships * jump_limit,
        dtype=np.int32,
    )

    for ship_idx, path in enumerate(paths):

        stored_path = [int(node) for node in path]

        # Remove destination before storing chromosome
        if (
            stored_path
            and stored_path[-1] == destination
        ):
            stored_path = stored_path[:-1]

        if len(stored_path) > jump_limit:
            raise ValueError(
                f"Ship {ship_idx + 1} has too many nodes: "
                f"{len(stored_path)} > {jump_limit}"
            )

        start = ship_idx * jump_limit
        end = start + len(stored_path)

        chromosome[start:end] = stored_path

    return chromosome