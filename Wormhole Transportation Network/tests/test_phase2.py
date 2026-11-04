import networkx as nx
import numpy as np
from solver.pathfinding import neighbors
from solver.phase2 import encode, decode


def test_neighbors_preserve_edges_and_endpoints():
    graph = nx.DiGraph()
    graph.add_edges_from([(1, 2), (2, 4), (1, 3), (3, 4), (1, 4)])
    candidates = neighbors(graph, [1, 2, 4], np.random.default_rng(42))
    assert [1, 4] in candidates
    assert [1, 3, 4] in candidates
    assert all(p[0] == 1 and p[-1] == 4 for p in candidates)


def test_encoding_destination_implicit():
    class UDP:
        n_ships = 2
        jump_limit = 5
        destination = 105
        @staticmethod
        def _chromosome_to_paths(x):
            paths = []
            for row in x.reshape(2, 5):
                zero = np.flatnonzero(row == 0)
                nodes = row[:int(zero[0])] if len(zero) else row
                paths.append(np.r_[nodes, 105])
            return paths, 3
    udp = UDP()
    x = encode(udp, [[1, 2, 105], [4, 5, 105]])
    assert x.dtype == np.int32
    assert x.tolist() == [1, 2, 0, 0, 0, 4, 5, 0, 0, 0]
    assert decode(udp, x) == [[1, 2, 105], [4, 5, 105]]
