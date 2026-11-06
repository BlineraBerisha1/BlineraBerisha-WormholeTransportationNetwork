"""Phase 4.3: bottleneck-aware adaptive neighborhood search for SpOC2.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from wormhole_udp import wormhole_traversal_udp


@dataclass(frozen=True)
class Move:
    ship: int
    pos: int
    nodes: tuple[int, ...]  # replaces len(nodes) consecutive internal waypoints
    dt: float
    dv: float  # UNPENALIZED variance proxy; not official fitness!

    @property
    def key(self):
        return (self.ship, self.pos, self.nodes)


def official(udp, chromosome):
    f, origin, path, window = map(float, udp.fitness(chromosome)[:4])
    return {
        "fitness": f,
        "origin_violations": origin,
        "path_violations": path,
        "window_violation": window,
        "feasible": origin == 0 and path == 0 and window <= 0,
    }


def edge_mv(graph, a, b):
    return float(graph[a][b]["m"]), float(graph[a][b]["v"])


def generate_moves(udp, paths, ship, length, rng, *, positions=48,
                   attempts=8, keep=32):
    """Replace 1 or 2 internal nodes, preserving graph edges and path length."""
    graph = udp.network
    path = paths[ship]
    maxpos = len(path) - length - 1
    if maxpos < 1:
        return []
    starts = np.arange(1, maxpos + 1)
    if len(starts) > positions:
        starts = rng.choice(starts, size=positions, replace=False)
    existing = set(path)
    moves = {}
    for raw_pos in starts:
        pos = int(raw_pos)
        old_chain = path[pos-1:pos+length+1]
        a, b = old_chain[0], old_chain[-1]
        old_m = old_v = 0.0
        for e1, e2 in zip(old_chain, old_chain[1:]):
            em, ev = edge_mv(graph, e1, e2)
            old_m += em
            old_v += ev
        successors = graph.succ[a]
        predecessors = graph.pred[b]
        if not successors or not predecessors:
            continue
        if length == 1:
            small = successors if len(successors) <= len(predecessors) else predecessors
            options = list(small)
            if len(options) > attempts:
                options = [options[int(i)] for i in rng.choice(len(options), size=attempts, replace=False)]
            for u in options:
                u = int(u)
                if u in existing or u == udp.destination or u not in successors or u not in predecessors:
                    continue
                m1, v1 = edge_mv(graph, a, u)
                m2, v2 = edge_mv(graph, u, b)
                move = Move(ship, pos, (u,), m1+m2-old_m, v1+v2-old_v)
                moves[move.key] = move
        else:
            options = list(successors)
            if len(options) > attempts:
                options = [options[int(i)] for i in rng.choice(len(options), size=attempts, replace=False)]
            for u in options:
                u = int(u)
                if u in existing or u == udp.destination:
                    continue
                mids = graph.succ[u]
                small = mids if len(mids) < len(predecessors) else predecessors
                choices = list(small)
                if len(choices) > 12:
                    choices = [choices[int(i)] for i in rng.choice(len(choices), size=12, replace=False)]
                for v in choices:
                    v = int(v)
                    if v == u or v in existing or v == udp.destination:
                        continue
                    if v not in mids or v not in predecessors:
                        continue
                    m1, v1 = edge_mv(graph, a, u)
                    m2, v2 = edge_mv(graph, u, v)
                    m3, v3 = edge_mv(graph, v, b)
                    move = Move(ship, pos, (u, v), m1+m2+m3-old_m,
                                v1+v2+v3-old_v)
                    moves[move.key] = move
    pool = sorted(moves.values(), key=lambda z: (z.dv, abs(z.dt)))
    # Blend greedy and diverse candidates for escaping local optima.
    greedy = pool[: max(1, keep // 2)]
    remainder = pool[len(greedy):]
    random_count = min(max(0, keep-len(greedy)), len(remainder))
    if random_count:
        greedy += [remainder[int(i)] for i in rng.choice(len(remainder), random_count, replace=False)]
    return greedy


def get_times(udp, paths):
    # Use the official float32 arrival calculation on the unchanged valid paths.
    _, ts = udp._compute_arrival_gaps(paths)
    return np.asarray(ts, dtype=np.float64)


def quick_window_ok(times, moves, window, tolerance=0.004):
    # This is an approximate screen only: the UDP uses float32 accumulation.
    # A small tolerance prevents rejecting borderline candidates too early.
    ts = times.copy()
    for mv in moves:
        ts[mv.ship] += mv.dt
    return float(np.max(ts)-np.min(ts)) <= window + tolerance


def apply_moves(base, moves, jump_limit):
    out = base.copy()
    for mv in moves:
        start = mv.ship * jump_limit + mv.pos
        out[start:start+len(mv.nodes)] = mv.nodes
    return out


def get_candidates(udp, paths, times, variances, rng, pool_size, counts):
    n = udp.n_ships
    rank = np.argsort(-np.asarray(variances))
    bottlenecks = [int(i) for i in rank[:min(4, n)]]
    by_ship = {
        i: {
            1: generate_moves(udp, paths, i, 1, rng, keep=pool_size),
            2: generate_moves(udp, paths, i, 2, rng, keep=pool_size),
        }
        for i in range(n)
    }
    candidate_groups = {"single_bridge": [], "paired_bridge": [], "paired_short": []}
    for ship in bottlenecks:
        for move in by_ship[ship][2]:
            if quick_window_ok(times, (move,), udp.window):
                candidate_groups["single_bridge"].append((move,))
            else:
                counts["window_prefilter"] += 1
        for other in range(n):
            if other == ship:
                continue
            # Long segment on a bottleneck ship + shorter compensating change.
            for a in by_ship[ship][2]:
                for b in by_ship[other][1]:
                    if quick_window_ok(times, (a,b), udp.window):
                        candidate_groups["paired_bridge"].append((a,b))
                    else:
                        counts["window_prefilter"] += 1
            # Also retain timing-aware short/short pairs from Phase 4.2.
            for a in by_ship[ship][1]:
                for b in by_ship[other][1]:
                    if quick_window_ok(times, (a,b), udp.window):
                        candidate_groups["paired_short"].append((a,b))
                    else:
                        counts["window_prefilter"] += 1
    # Prioritize proxy variance on bottlenecks; final ranking is NOT official.
    def score(group):
        return (sum((2.0 if m.ship in bottlenecks[:2] else 1.0)*m.dv for m in group),
                abs(sum(m.dt for m in group)))
    for name, groups in candidate_groups.items():
        counts[f"{name}_generated"] += len(groups)
        groups.sort(key=score)
        # Limit memory and spend some budget on diverse moves, not only lowest proxy.
        cap = 900
        if len(groups) > cap:
            best = groups[:cap//2]
            tail = groups[cap//2:]
            chosen = rng.choice(len(tail), size=cap-len(best), replace=False)
            candidate_groups[name] = best + [tail[int(i)] for i in chosen]
    return candidate_groups, bottlenecks


def optimize(udp, input_x, *, budget=350, seed=43, rounds=8, pool_size=20):
    if budget < 1:
        raise ValueError("--budget must be at least 1")
    rng = np.random.default_rng(seed)
    x = np.asarray(input_x, dtype=np.int32).reshape(-1).copy()
    if x.size != udp.n_ships*udp.jump_limit:
        raise ValueError(f"Expected {udp.n_ships*udp.jump_limit} chromosome values, got {x.size}")
    first = official(udp, x)
    if not first["feasible"]:
        raise ValueError(f"Input is not feasible: {first}")
    best_x = x.copy()
    best = first
    evals = 1
    counts = Counter()
    history = []
    weights = {"single_bridge": 1.0, "paired_bridge": 2.0, "paired_short": 1.0}
    seen = {best_x.tobytes()}
    operator_names = list(weights)

    for round_id in range(1, rounds+1):
        if evals >= budget:
            break
        paths, _ = udp._chromosome_to_paths(best_x)
        paths = [list(map(int, path)) for path in paths]
        times = get_times(udp, paths)
        variances = np.asarray(udp._accumulate_variances(paths), dtype=float)
        groups, bottlenecks = get_candidates(udp, paths, times, variances, rng,
                                             pool_size, counts)
        remaining = budget-evals
        this_round = min(remaining, max(15, int(np.ceil(remaining/(rounds-round_id+1)))))
        improved = False
        tested = 0
        # Adaptive weighted operator choice, without replacement within each pool.
        cursors = {name: 0 for name in operator_names}
        while tested < this_round:
            active = [name for name in operator_names if cursors[name] < len(groups[name])]
            if not active:
                break
            probs = np.array([weights[name] for name in active], dtype=float)
            probs /= probs.sum()
            op = str(rng.choice(active, p=probs))
            changes = groups[op][cursors[op]]
            cursors[op] += 1
            candidate = apply_moves(best_x, changes, udp.jump_limit)
            sig = candidate.tobytes()
            if sig in seen:
                counts["duplicate"] += 1
                continue
            seen.add(sig)
            result = official(udp, candidate)
            evals += 1
            tested += 1
            counts["official_candidates"] += 1
            if not result["feasible"]:
                counts["official_invalid"] += 1
                weights[op] = max(0.25, weights[op]*0.995)
            elif result["fitness"] < best["fitness"] - 1e-9:
                best_x, best = candidate, result
                counts["accepted"] += 1
                counts[f"accepted_{op}"] += 1
                weights[op] = min(8.0, weights[op]*1.5)
                improved = True
                print(f"Round {round_id}: {op} -> {best['fitness']:.9f}", flush=True)
                break
            else:
                counts["not_improved"] += 1
                weights[op] = max(0.25, weights[op]*0.998)
        history.append({"round": round_id, "fitness": best["fitness"],
                        "improved": improved, "official_evaluations": evals,
                        "bottleneck_ships_1based": [s+1 for s in bottlenecks],
                        "evaluations_this_round": tested})
        print(f"Completed round {round_id}: fitness={best['fitness']:.9f}, evaluations={evals}/{budget}", flush=True)
        if tested == 0:
            break
    return best_x, {"initial": first, "final": best, "official_evaluations": evals,
                    "rejections_and_counts": dict(counts),
                    "operator_weights": weights, "history": history}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default="results/best_phase4_2.npy")
    p.add_argument("--database", default="data/database.npz")
    p.add_argument("--output", default="results/best_phase4_3.npy")
    p.add_argument("--budget", type=int, default=350)
    p.add_argument("--rounds", type=int, default=8)
    p.add_argument("--pool-size", type=int, default=20)
    p.add_argument("--seed", type=int, default=43)
    a = p.parse_args()
    udp = wormhole_traversal_udp(database=a.database)
    x = np.load(a.input, allow_pickle=False)
    out, report = optimize(udp, x, budget=a.budget, rounds=a.rounds,
                           pool_size=a.pool_size, seed=a.seed)
    dest = Path(a.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.save(dest, out)
    dest.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved: {dest} and {dest.with_suffix('.json')}")


if __name__ == "__main__":
    main()
