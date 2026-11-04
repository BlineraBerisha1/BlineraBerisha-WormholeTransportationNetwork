"""Small graph helpers using the official UDP's directed NetworkX graph.

"""


def edge_mean(graph, a, b):
    return float(graph[a][b]['m'])


def edge_variance(graph, a, b):
    return float(graph[a][b]['v'])


def path_mean(graph, nodes):
    return sum(edge_mean(graph, a, b) for a, b in zip(nodes, nodes[1:]))


def is_valid_path(graph, nodes):
    return len(nodes) >= 2 and all(graph.has_edge(a, b) for a, b in zip(nodes, nodes[1:]))


def arrivals(udp, paths):
    return [float(udp.delays[i]) + path_mean(udp.network, p) for i, p in enumerate(paths)]
