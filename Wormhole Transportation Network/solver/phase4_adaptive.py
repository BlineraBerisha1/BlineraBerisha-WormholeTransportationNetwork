"""Phase 4.1: adaptive bottleneck-guided iterated local search for SpOC2.
"""
import argparse
import json
import math
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import numpy as np

from solver.phase2 import decode, encode, evaluate
from solver.phase3 import _arrival, _changes, _pair_candidates, optimize as local_optimize, ship_costs


def _bridge(graph, a, b, rng, hops, trials):
    """Random short alternative connection from node a to node b.

    Graph is the directed networkx graph supplied by the challenge.
    A hop count of 2/3/4 changes a small segment without global Dijkstra.
    """
    if hops == 2:
        # The predecessor set of b allows O(degree(a)) membership checks.
        predecessors = set(graph.predecessors(b)) if graph.is_directed() else set(graph.neighbors(b))
        possible = [int(x) for x in graph.successors(a) if x in predecessors and x not in (a, b)] if graph.is_directed() else [int(x) for x in graph.neighbors(a) if x in predecessors and x not in (a, b)]
        if possible:
            return [a, int(possible[int(rng.integers(len(possible)))]), b]
        return None
    next_nodes = list(graph.successors(a) if graph.is_directed() else graph.neighbors(a))
    if not next_nodes:
        return None
    for _ in range(trials):
        x = int(next_nodes[int(rng.integers(len(next_nodes)))])
        if x in (a, b):
            continue
        if hops == 3:
            next2 = list(graph.successors(x) if graph.is_directed() else graph.neighbors(x))
            valid = [int(y) for y in next2 if y not in (a, x, b) and graph.has_edge(y, b)]
            if valid:
                return [a, x, int(valid[int(rng.integers(len(valid)))]), b]
        else:
            second = list(graph.successors(x) if graph.is_directed() else graph.neighbors(x))
            if not second:
                continue
            y = int(second[int(rng.integers(len(second)))])
            if y in (a, x, b):
                continue
            third = list(graph.successors(y) if graph.is_directed() else graph.neighbors(y))
            valid = [int(z) for z in third if z not in (a, x, y, b) and graph.has_edge(z, b)]
            if valid:
                return [a, x, y, int(valid[int(rng.integers(len(valid)))]), b]
    return None


def adaptive_changes(udp, paths, ship, rng, count, strength):
    """Combine Phase 3 neighbors with randomized multi-hop segment replacements."""
    graph = udp.network
    original = list(map(int, paths[ship]))
    baseline = _changes(udp, paths, ship, rng, max(2, count // 2))
    candidates = list(baseline)
    seen = {tuple(item[0]) for item in candidates}
    attempts = max(20, count * 12 * strength)
    for _ in range(attempts):
        if len(candidates) - 1 >= count:
            break
        # Select a segment of 1 to 3 existing edges. Substitute 2-4 hops.
        if len(original) < 3:
            break
        left = int(rng.integers(0, len(original) - 1))
        right = min(len(original) - 1, left + int(rng.integers(1, min(4, len(original) - left))))
        hops = int(rng.integers(2, min(5, strength + 2)))
        bridge = _bridge(graph, original[left], original[right], rng, hops, trials=8)
        if bridge is None:
            continue
        path = original[:left] + bridge + original[right + 1:]
        key = tuple(path)
        if key in seen or len(path) - 1 > udp.jump_limit or len(path) < 2:
            continue
        if any(n == 0 for n in path[:-1]) or path[0] not in udp.origins[ship] or path[-1] != udp.destination:
            continue
        if any(not graph.has_edge(a, b) for a, b in zip(path, path[1:])):
            continue
        seen.add(key)
        arrival = _arrival(udp, ship, path)
        variance = sum(float(graph[a][b]['v']) for a, b in zip(path, path[1:]))
        candidates.append((path, arrival, variance))
    return candidates


class FitnessCache:
    """Bounded cache across outer rounds. Only official evaluations are cached."""
    def __init__(self, capacity=20000):
        self.capacity = capacity
        self.data = OrderedDict()
        self.evaluations = 0
        self.hits = 0

    def check(self, udp, chromosome):
        key = chromosome.tobytes()
        if key in self.data:
            self.hits += 1
            self.data.move_to_end(key)
            return self.data[key], False
        stats = evaluate(udp, chromosome)
        self.evaluations += 1
        self.data[key] = stats
        if len(self.data) > self.capacity:
            self.data.popitem(last=False)
        return stats, True


def perturb(udp, current, current_stats, rng, cache, *, attempts, candidates, strength, temperature, partners):
    paths = decode(udp, current)
    arrivals = np.asarray([_arrival(udp, k, p) for k, p in enumerate(paths)])
    costs = ship_costs(udp, paths)
    order = list(map(int, np.argsort(-costs)))
    # 70% focus on bottleneck ships, 30% diversify randomly.
    focus = order[:min(max(2, strength), len(order))]
    focus += [int(x) for x in rng.permutation(udp.n_ships) if int(x) not in focus]
    before = cache.evaluations
    tested = set()
    for i in focus:
        candidates_i = adaptive_changes(udp, paths, i, rng, candidates, strength)
        other = [j for j in order if j != i]
        preferred = [int(np.argmin(arrivals)), int(np.argmax(arrivals))]
        selected = list(dict.fromkeys([j for j in preferred + other if j != i]))
        # A few random partners improve coverage in stalled search.
        if strength >= 3:
            selected = list(dict.fromkeys(selected[:2] + list(map(int, rng.permutation([j for j in range(udp.n_ships) if j != i])))))
        for j in selected[:max(1, partners + strength - 1)]:
            candidates_j = adaptive_changes(udp, paths, j, rng, candidates, strength)
            pair_options = _pair_candidates(udp, paths, arrivals, i, j, candidates_i, candidates_j,
                                            max(attempts, candidates * candidates))
            if not pair_options:
                continue
            # Balance heuristic quality and novelty.
            shortlist = pair_options[:max(16, candidates * 4)]
            indices = rng.permutation(len(shortlist))
            for index in indices:
                if cache.evaluations - before >= attempts:
                    return current, current_stats, cache.evaluations - before, False
                pi, pj = shortlist[int(index)]
                trial_paths = list(paths)
                trial_paths[i], trial_paths[j] = pi, pj
                chromosome = encode(udp, trial_paths)
                key = chromosome.tobytes()
                if key == current.tobytes() or key in tested:
                    continue
                tested.add(key)
                stats, fresh = cache.check(udp, chromosome)
                if not fresh or not stats['feasible']:
                    continue
                delta = stats['fitness'] - current_stats['fitness']
                if delta <= 1e-9 or rng.random() < math.exp(-min(max(delta, 0) / max(temperature, 1e-9), 700)):
                    return chromosome, stats, cache.evaluations - before, True
    return current, current_stats, cache.evaluations - before, False


def optimize(udp, start, *, rounds=8, local_iterations=6, seed=42, candidates=12,
             evaluations=15, partners=3, kick_attempts=40, temperature=2.0,
             cooling=0.92, stall_threshold=3, output=None):
    if rounds < 0 or local_iterations < 0 or min(candidates, evaluations, partners, kick_attempts, stall_threshold) < 1:
        raise ValueError('Invalid non-positive search parameter')
    if temperature <= 0 or not (0 < cooling <= 1):
        raise ValueError('temperature > 0 and 0 < cooling <= 1 required')
    rng = np.random.default_rng(seed)
    best = np.asarray(start, dtype=np.int32).reshape(-1).copy()
    cache = FitnessCache()
    best_stats, _ = cache.check(udp, best)
    if not best_stats['feasible']:
        raise ValueError(f'Starting solution is infeasible: {best_stats}')
    current = best.copy()
    current_stats = dict(best_stats)
    history = []
    no_improvement = 0
    global_improvements = 0
    kicks_accepted = 0
    local_evaluations = 0
    for r in range(1, rounds + 1):
        strength = min(4, 2 + no_improvement // stall_threshold)
        if no_improvement >= 2 * stall_threshold:
            # Return to best rather than following a degraded solution forever.
            current, current_stats = best.copy(), dict(best_stats)
        temp = temperature * cooling ** (r - 1) * (1.0 + 0.5 * (strength - 1))
        current, current_stats, kick_checks, accepted = perturb(
            udp, current, current_stats, rng, cache, attempts=kick_attempts,
            candidates=candidates, strength=strength, temperature=temp, partners=partners)
        kicks_accepted += int(accepted)
        # Reuse the tested Phase 3 implementation for local refinement.
        refined, info = local_optimize(
            udp, current, iterations=local_iterations,
            seed=int(rng.integers(0, 2**31 - 1)),
            candidates_per_ship=candidates,
            evaluations_per_iteration=evaluations,
            partners=partners)
        local_evaluations += int(info['official_evaluations'])
        current, current_stats = refined, info['final']
        if not current_stats['feasible']:
            raise AssertionError('Local search generated an infeasible result')
        improved = current_stats['fitness'] < best_stats['fitness'] - 1e-8
        if improved:
            best, best_stats = current.copy(), dict(current_stats)
            global_improvements += 1
            no_improvement = 0
            if output is not None:
                np.save(output, best)
        else:
            no_improvement += 1
        row = {'round': r, 'strength': strength, 'temperature': temp,
               'perturb_accepted': accepted, 'kick_evaluations': kick_checks,
               'local_evaluations': info['official_evaluations'],
               'local_accepted_moves': info['accepted_moves'],
               'current_fitness': current_stats['fitness'],
               'best_fitness': best_stats['fitness'], 'new_global_best': improved}
        history.append(row)
        print(f'Round {r}/{rounds} | strength={strength} | current={current_stats["fitness"]:.6f} | '
              f'BEST={best_stats["fitness"]:.6f} | improved={improved}', flush=True)
    return best, {'initial': evaluate(udp, start), 'final': best_stats,
                  'best_improvements': global_improvements,
                  'perturbations_accepted': kicks_accepted,
                  'official_evaluations': cache.evaluations + local_evaluations,
                  'cache_hits': cache.hits, 'history': history}


def main():
    p = argparse.ArgumentParser(description='Adaptive feasible search for Wormhole SpOC2')
    p.add_argument('--db', default='data/database.npz')
    p.add_argument('--input', default='results/best_phase4_1206_backup.npy')
    p.add_argument('--output', default='results/best_phase4_adaptive.npy')
    p.add_argument('--rounds', type=int, default=8)
    p.add_argument('--local-iterations', type=int, default=6)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--candidates', type=int, default=12)
    p.add_argument('--evaluations', type=int, default=15)
    p.add_argument('--partners', type=int, default=3)
    p.add_argument('--kick-attempts', type=int, default=40)
    p.add_argument('--temperature', type=float, default=2.0)
    p.add_argument('--cooling', type=float, default=0.92)
    p.add_argument('--stall-threshold', type=int, default=3)
    args = p.parse_args()
    if Path(args.input).resolve() == Path(args.output).resolve():
        p.error('--output must differ from --input')
    if not Path(args.input).is_file():
        p.error(f'Input not found: {args.input}')
    from wormhole_udp import wormhole_traversal_udp
    udp = wormhole_traversal_udp(database=args.db)
    x = np.load(args.input, allow_pickle=False).astype(np.int32)
    if x.shape != (udp.n_ships * udp.jump_limit,):
        p.error('Invalid chromosome length/shape')
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    best, report = optimize(
        udp, x, rounds=args.rounds, local_iterations=args.local_iterations,
        seed=args.seed, candidates=args.candidates, evaluations=args.evaluations,
        partners=args.partners, kick_attempts=args.kick_attempts,
        temperature=args.temperature, cooling=args.cooling,
        stall_threshold=args.stall_threshold, output=output)
    verified = evaluate(udp, best)
    if not verified['feasible'] or verified != report['final']:
        raise AssertionError(f'Official verification mismatch: {verified} != {report["final"]}')
    np.save(output, best)
    report.update({'timestamp': datetime.now().isoformat(timespec='seconds'),
                   'input': args.input, 'output': str(output), 'seed': args.seed,
                   'rounds': args.rounds, 'local_iterations': args.local_iterations,
                   'candidates': args.candidates, 'evaluations': args.evaluations,
                   'partners': args.partners, 'kick_attempts': args.kick_attempts,
                   'temperature': args.temperature, 'cooling': args.cooling,
                   'stall_threshold': args.stall_threshold})
    stats_file = output.with_name(output.stem + '_statistics.json')
    stats_file.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('Final verified:', verified, flush=True)
    print(f'Saved: {output} and {stats_file}', flush=True)


if __name__ == '__main__':
    main()
