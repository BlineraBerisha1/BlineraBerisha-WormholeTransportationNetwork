"""Phase 4.2: constraint-aware, paired single-waypoint path search.
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
    node: int
    dt: float
    dv: float  # only a cheap ranking proxy, NOT the official fitness delta


def arrivals(udp, paths):
    return np.asarray([
        sum(float(udp.network[a][b]["m"]) for a, b in zip(p, p[1:])) + float(udp.delays[i])
        for i, p in enumerate(paths)
    ], dtype=np.float64)


def evaluate(udp, x):
    values = udp.fitness(x)
    fit, origin, path, window = map(float, values[:4])
    return {
        "fitness": fit, "origin_violations": origin,
        "path_violations": path, "window_violation": window,
        "feasible": origin == 0.0 and path == 0.0 and window <= 0.0,
    }


def make_moves(udp, paths, per_ship, rng):
    graph = udp.network
    moves = [[] for _ in paths]
    for ship, path in enumerate(paths):
        existing = set(map(int, path))
        for pos in range(1, len(path) - 1):
            a, old, b = map(int, path[pos-1:pos+2])
            old_m = float(graph[a][old]["m"]) + float(graph[old][b]["m"])
            old_v = float(graph[a][old]["v"]) + float(graph[old][b]["v"])
            # Only consider nodes that can connect both endpoints.
            successors = graph.succ[a]
            predecessors = graph.pred[b]
            choices = successors.keys() if len(successors) < len(predecessors) else predecessors.keys()
            for node in choices:
                node = int(node)
                if node in existing or node == udp.destination:
                    continue
                if node not in successors or node not in predecessors:
                    continue
                new_m = float(graph[a][node]["m"]) + float(graph[node][b]["m"])
                new_v = float(graph[a][node]["v"]) + float(graph[node][b]["v"])
                moves[ship].append(Move(ship, pos, node, new_m-old_m, new_v-old_v))
        # Keep useful improvements AND timing-compensation options.
        pool = sorted(moves[ship], key=lambda m: (m.dv, abs(m.dt)))
        best = pool[:max(1, per_ship // 2)]
        leftover = pool[max(1, per_ship // 2):]
        if leftover:
            indices = rng.choice(len(leftover), size=min(per_ship-len(best), len(leftover)), replace=False)
            best.extend(leftover[int(i)] for i in indices)
        moves[ship] = best
    return moves


def optimize(udp, x, budget=1200, per_ship=24, seed=42, rounds=6):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=np.int32).copy()
    baseline = evaluate(udp, x)
    if not baseline["feasible"]:
        raise ValueError(f"Starting solution is not feasible: {baseline}")
    best_x = x.copy()
    best = baseline
    counts = Counter()
    official_evaluations = 1
    history = []

    for round_idx in range(rounds):
        if official_evaluations >= budget:
            break
        paths, _ = udp._chromosome_to_paths(best_x)
        paths = [list(map(int, p)) for p in paths]
        times = arrivals(udp, paths)
        moves = make_moves(udp, paths, per_ship, rng)
        costs = udp._accumulate_variances(paths)
        ship_order = sorted(range(udp.n_ships), key=lambda i: -costs[i])
        candidates = []

        # Try a simultaneous change in two different ships.
        for i in ship_order:
            for j in range(udp.n_ships):
                if i == j:
                    continue
                for mi in moves[i]:
                    for mj in moves[j]:
                        # Pre-filter arrival constraint without costly UDP evaluation.
                        new_times = times.copy()
                        new_times[i] += mi.dt
                        new_times[j] += mj.dt
                        if new_times.max() - new_times.min() > udp.window - 1e-5:
                            counts["window_prefilter"] += 1
                            continue
                        # Ranked by estimated variance improvement and time stability.
                        score = (mi.dv if i == ship_order[0] else 0.5*mi.dv) + 0.5*mj.dv
                        candidates.append((score, abs(mi.dt+mj.dt), mi, mj))

        candidates.sort(key=lambda t: (t[0], t[1]))
        counts["ranked_candidates"] += len(candidates)
        improved = False
        for _, _, mi, mj in candidates:
            if official_evaluations >= budget:
                break
            candidate = best_x.copy()
            candidate[mi.ship*udp.jump_limit + mi.pos] = mi.node
            candidate[mj.ship*udp.jump_limit + mj.pos] = mj.node
            result = evaluate(udp, candidate)
            official_evaluations += 1
            if result["origin_violations"] != 0:
                counts["origin_rejected"] += 1
            elif result["path_violations"] != 0:
                counts["path_rejected"] += 1
            elif result["window_violation"] > 0:
                counts["window_rejected"] += 1
            elif result["fitness"] >= best["fitness"] - 1e-9:
                counts["not_improved"] += 1
            else:
                counts["accepted"] += 1
                best_x, best = candidate, result
                improved = True
                print(f"Round {round_idx+1}: new best fitness = {best['fitness']:.9f}")
                break  # recompute moves around the newly improved solution
        history.append({"round": round_idx+1, "fitness": best["fitness"],
                        "improved": improved, "official_evaluations": official_evaluations})
        if not improved:
            break
    return best_x, {"initial": baseline, "final": best,
                    "official_evaluations": official_evaluations,
                    "rejections": dict(counts), "history": history}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/best_phase4.npy")
    parser.add_argument("--database", default="data/database.npz")
    parser.add_argument("--output", default="results/best_phase4_2.npy")
    parser.add_argument("--budget", type=int, default=1200)
    parser.add_argument("--per-ship", type=int, default=24)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    udp = wormhole_traversal_udp(database=args.database)
    initial = np.load(args.input)
    best_x, report = optimize(udp, initial, args.budget, args.per_ship, args.seed, args.rounds)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, best_x)
    report_file = Path(args.output).with_suffix(".json")
    report_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved {args.output} and {report_file}")


if __name__ == "__main__":
    main()
