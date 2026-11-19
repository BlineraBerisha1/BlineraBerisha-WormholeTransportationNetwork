"""Numba-accelerated Lagrangian path construction and graph lookahead."""
import numpy as np
from numba import njit

@njit(cache=True)
def _build_one(offsets, neighbors, means, variances, dest_edges, origins,
               delay, lo, hi, lagrange, noise, occupancy, rng_seed, jump_limit, potential, discount, destination,
               preferred_start=-1):
    """Generate one directed path with an arrival-time resource constraint.

    The returned row lists *source* nodes; the terminal destination is implicit.
    Feasibility is checked again with the unmodified UDP before recording.
    """
    np.random.seed(rng_seed)
    n_nodes = len(offsets) - 2
    visited = np.zeros(n_nodes + 1, np.uint8)
    row = np.zeros(jump_limit, np.int32)
    # Pick a random permitted origin. Trying different starts is part of ALNS.
    origin = int(origins[np.random.randint(len(origins))]) if preferred_start <= 0 else preferred_start
    node = origin
    t = np.float32(delay)
    base_variance = 0.
    crowd_cost = 0.
    needed = 0.
    if t < lo: needed = lo - t
    elif t > hi: needed = hi - t
    direction = 1. if needed >= 0 else -1.
    visited[node] = 1
    for k in range(jump_limit):
        # First, try to terminate at the destination by one or two edges.
        # The 2-edge exit avoids requiring a direct edge to the destination.
        best_end = 1e15
        last_n = -1
        e = dest_edges[node]
        if e >= 0:
            arrival = t + means[e]
            if lo <= arrival <= hi:
                best_end = float(variances[e]) + 0.0025 * occupancy[node]
                last_n = 0
        if k + 1 < jump_limit:
            for j in range(offsets[node], offsets[node + 1]):
                nxt = neighbors[j]
                if nxt == node or visited[nxt] != 0:
                    continue
                end_j = dest_edges[nxt]
                if end_j < 0:
                    continue
                arrival = t + means[j] + means[end_j]
                if arrival < lo or arrival > hi:
                    continue
                cost = (float(variances[j]) + float(variances[end_j])
                        + 0.0025 * (occupancy[nxt] + occupancy[node]))
                if cost < best_end:
                    best_end, last_n = cost, int(nxt)
        if last_n >= 0:
            row[k] = node
            base_variance += best_end
            if last_n != 0:
                row[k + 1] = last_n
                return row, k + 2, base_variance + crowd_cost, np.float32(t)
            return row, k + 1, base_variance + crowd_cost, np.float32(t)

        if k + 2 >= jump_limit:
            return row, 0, 1e10, t

        remain = (lo - t) if direction > 0 else (t - hi)
        lam = lagrange
        if remain > 0:
            req = remain / max(1, jump_limit - k - 2)
            if req > 0.0008: lam += (req - .0008) * 250.
            if req > 0.00155: lam += (req - .00155) * 1300.

        start = offsets[node]
        end = offsets[node + 1]
        min_score = 1e15
        next_node = -1
        chosen = -1
        for j in range(start, end):
            nxt = neighbors[j]
            if nxt == 0 or nxt == node or nxt == destination:
                continue
            if visited[nxt] != 0:
                continue
            # Favor low-variance wormholes; Lagrange price encourages required
            # travel-time sign. Penalize shared nodes to limit revisit ranks.
            score = (float(variances[j]) - lam * direction * float(means[j])
                     + discount * float(potential[nxt]) + 0.002 * occupancy[nxt])
            score += noise * (np.random.random() - 0.5)
            if score < min_score:
                min_score = score
                next_node = int(nxt)
                chosen = j
        if chosen < 0:
            return row, 0, 1e10, t
        row[k] = node
        visited[next_node] = 1
        base_variance += float(variances[chosen])
        crowd_cost += 0.002 * occupancy[node]
        t = np.float32(t + means[chosen])
        node = next_node
    return row, 0, 1e10, t

@njit(cache=True)
def _node_potentials(offsets,targets,means,variances,lam,direction,
                     discount=0.80,iterations=16):
    """Discounted Lagrangian lookahead on all 1,998,600 directed edges.

    Values approximate future low-variance accessibility. Repeated-visit
    avoidance remains enforced in the route constructor.
    """
    n=len(offsets)-2
    old=np.zeros(n+1,dtype=np.float32)
    for it in range(iterations):
        fresh=np.zeros(n+1,dtype=np.float32)
        for src in range(1,n+1):
            best=1.e10
            for j in range(offsets[src],offsets[src+1]):
                val=variances[j]-lam*direction*means[j]+discount*old[targets[j]]
                if val<best:best=val
            fresh[src]=best
        old=fresh
    return old
