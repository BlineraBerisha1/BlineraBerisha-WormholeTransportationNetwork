"""Local path mutations: direct shortcuts and two-edge reroutes.

Each produced path keeps the origin and destination, and all its edges exist.
"""
from solver.network import is_valid_path


def neighbors(graph, path, rng, max_candidates=24):
    """Return up to max_candidates DIFFERENT valid local paths.

    Uses random sampling of triples to avoid scanning all 2M edges.
    """
    if len(path) < 3:
        return []
    result = []
    seen = set()
    positions = list(range(1, len(path) - 1))
    rng.shuffle(positions)
    for pos in positions:
        a, old, c = path[pos - 1], path[pos], path[pos + 1]
        if graph.has_edge(a, c):
            candidate = path[:pos] + path[pos + 1:]
            key = tuple(candidate)
            if key not in seen and is_valid_path(graph, candidate):
                seen.add(key)
                result.append(candidate)
                if len(result) >= max_candidates:
                    break
        # Intersect a's outgoing neighbors with c's incoming neighbors.
        # This normally visits ~200 neighbors, rather than all graph edges.
        succ = graph.succ[a]
        pred = graph.pred[c]
        common = succ.keys() if len(succ) <= len(pred) else pred.keys()
        alternatives = [int(node) for node in common
                        if node != old and node != a and node != c
                        and node not in path and node in succ and node in pred]
        if alternatives:
            order = rng.permutation(len(alternatives))
            for index in order[:max_candidates]:
                candidate = path.copy()
                candidate[pos] = alternatives[int(index)]
                key = tuple(candidate)
                if key not in seen:
                    seen.add(key)
                    result.append(candidate)
                    if len(result) >= max_candidates:
                        break
        if len(result) >= max_candidates:
            break
    return result
