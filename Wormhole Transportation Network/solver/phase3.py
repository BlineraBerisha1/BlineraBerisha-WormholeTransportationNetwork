"""Phase 3: feasibility-preserving paired ship local search.

Run from the existing project root:
  python -m solver.phase3 --input results/best_phase2.npy --iterations 40

"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from solver.network import path_mean
from solver.pathfinding import neighbors
from solver.phase2 import decode, encode, evaluate


def ship_costs(udp, paths):
    """Official per-ship penalized variance, used to focus search on bottlenecks."""
    return np.asarray(udp._accumulate_variances(paths), dtype=float) * float(udp._fitness_scaling)


def candidate_changes(udp, paths, ship, rng, limit):
    """Generate different valid paths with a cached arrival and raw variance."""
    graph = udp.network
    generated = neighbors(graph, paths[ship], rng, max_candidates=limit)
    out = []
    for p in generated:
        if len(p) < 2 or len(p) - 1 > udp.jump_limit:
            continue
        if p[0] not in udp.origins[ship] or p[-1] != udp.destination:
            continue
        mean = path_mean(graph, p)
        raw_var = sum(float(graph[a][b]['v']) for a, b in zip(p, p[1:]))
        out.append((p, float(udp.delays[ship]) + mean, raw_var))
    return out


def arrival_spread(arrival):
    return float(np.max(arrival) - np.min(arrival))


def prioritize_combinations(udp, paths, arrivals, i, j, changes_i, changes_j, limit):
    """Yield promising single and paired changes with arrival spread within window.

    Ranking is a cheap heuristic, NOT the official objective.  Official
    fitness is evaluated later before any candidate is accepted.
    """
    window = float(udp.window)
    graph = udp.network
    other_nodes = {int(n) for k, path in enumerate(paths) if k not in (i, j) for n in path[:-1]}
    # No-change options allow single-ship moves too.
    original_i = (paths[i], float(arrivals[i]), sum(float(graph[a][b]['v']) for a, b in zip(paths[i], paths[i][1:])))
    original_j = (paths[j], float(arrivals[j]), sum(float(graph[a][b]['v']) for a, b in zip(paths[j], paths[j][1:])))
    all_i = [original_i] + changes_i
    all_j = [original_j] + changes_j
    proposals = []
    for a_idx, (pa, ta, va) in enumerate(all_i):
        for b_idx, (pb, tb, vb) in enumerate(all_j):
            if a_idx == 0 and b_idx == 0:
                continue
            updated = np.array(arrivals, dtype=float)
            updated[i], updated[j] = ta, tb
            spread = arrival_spread(updated)
            # Tolerance only for rounding, official evaluator remains decisive.
            if spread > window + 1e-6:
                continue
            shared = len(set(pa[:-1]) & other_nodes) + len(set(pb[:-1]) & other_nodes)
            shared += len(set(pa[:-1]) & set(pb[:-1]))
            # Prefer few shared nodes and lower raw variance, and secondarily tighter arrivals.
            proxy = (shared * 1e6 + va + vb, spread)
            proposals.append((proxy, a_idx, b_idx))
    proposals.sort(key=lambda t: t[0])
    for _, a_idx, b_idx in proposals[:limit]:
        yield all_i[a_idx][0], all_j[b_idx][0]


def optimize(udp, start, iterations=40, seed=42, candidates_per_ship=8, evaluations_per_iteration=8):
    if iterations < 0 or candidates_per_ship < 1 or evaluations_per_iteration < 1:
        raise ValueError('Iterations must be >= 0; candidate and evaluation limits must be >= 1.')
    rng = np.random.default_rng(seed)
    best = np.asarray(start, dtype=np.int32).copy()
    stats = evaluate(udp, best)
    if not stats['feasible']:
        raise ValueError('Starting chromosome is invalid according to official UDP fitness.')
    paths = decode(udp, best)
    arrivals = np.array([float(udp.delays[k]) + path_mean(udp.network, p) for k, p in enumerate(paths)])
    history = [{'iteration': 0, 'accepted': False, **stats}]
    evaluated = 0
    accepted_count = 0
    for step in range(1, iterations + 1):
        costs = ship_costs(udp, paths)
        # Alternate between the objective bottleneck and random exploration.
        i = int(np.argmax(costs)) if step % 3 else int(rng.integers(udp.n_ships))
        # Partner most helpful for balancing the arrival-time extremes.
        if step % 2:
            j = int(np.argmax(arrivals)) if arrivals[i] <= np.median(arrivals) else int(np.argmin(arrivals))
        else:
            j = int(rng.integers(udp.n_ships - 1))
            if j >= i:
                j += 1
        if i == j:
            j = (i + 1) % udp.n_ships
        options_i = candidate_changes(udp, paths, i, rng, candidates_per_ship)
        options_j = candidate_changes(udp, paths, j, rng, candidates_per_ship)
        accepted = False
        for new_i, new_j in prioritize_combinations(
            udp, paths, arrivals, i, j, options_i, options_j, evaluations_per_iteration
        ):
            trial_paths = list(paths)
            trial_paths[i] = new_i
            trial_paths[j] = new_j
            trial = encode(udp, trial_paths)
            s = evaluate(udp, trial)
            evaluated += 1
            if s['feasible'] and s['fitness'] < stats['fitness'] - 1e-8:
                best, stats, paths = trial, s, trial_paths
                arrivals[i] = float(udp.delays[i]) + path_mean(udp.network, new_i)
                arrivals[j] = float(udp.delays[j]) + path_mean(udp.network, new_j)
                accepted = True
                accepted_count += 1
                break
        history.append({'iteration': step, 'accepted': accepted, 'fitness': stats['fitness'],
                        'arrival_spread': arrival_spread(arrivals)})
        if accepted or step % 5 == 0:
            print(f'Iteration {step}/{iterations} | fitness={stats["fitness"]:.6f} | '
                  f'accepted={accepted} | official_evaluations={evaluated}', flush=True)
    summary = {'final': stats, 'accepted_moves': accepted_count,
               'official_evaluations': evaluated, 'history': history}
    return best, summary


def main():
    parser = argparse.ArgumentParser(description='Paired ship optimization for Wormhole.')
    parser.add_argument('--db', default='data/database.npz')
    parser.add_argument('--input', default='results/best_phase2.npy')
    parser.add_argument('--output', default='results')
    parser.add_argument('--iterations', type=int, default=40)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--candidates', type=int, default=8)
    parser.add_argument('--evaluations', type=int, default=8)
    args = parser.parse_args()
    source = Path(args.input)
    if not source.is_file():
        parser.error(f'Input not found: {source}. Use --input results/best_local_search.npy if needed.')
    from wormhole_udp import wormhole_traversal_udp
    udp = wormhole_traversal_udp(database=args.db)
    x = np.load(source, allow_pickle=False).astype(np.int32)
    if x.shape != (udp.n_ships * udp.jump_limit,):
        parser.error('Chromosome shape must be (n_ships * jump_limit,).')
    initial = evaluate(udp, x)
    print('Initial:', initial, flush=True)
    result, report = optimize(udp, x, iterations=args.iterations, seed=args.seed,
                              candidates_per_ship=args.candidates,
                              evaluations_per_iteration=args.evaluations)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / 'best_phase3.npy', result)
    report.update({'timestamp': datetime.now().isoformat(timespec='seconds'),
                   'input': str(source), 'seed': args.seed, 'iterations': args.iterations,
                   'candidates_per_ship': args.candidates,
                   'evaluations_per_iteration': args.evaluations, 'initial': initial})
    (out / 'phase3_statistics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    # One final official evaluation of the saved representation.
    verified = evaluate(udp, np.load(out / 'best_phase3.npy', allow_pickle=False))
    if verified != report['final']:
        raise AssertionError(f'Saved solution mismatch: {verified} != {report["final"]}')
    print('Final verified:', verified, flush=True)
    print(f'Saved {out / "best_phase3.npy"} and {out / "phase3_statistics.json"}', flush=True)


if __name__ == '__main__':
    main()
