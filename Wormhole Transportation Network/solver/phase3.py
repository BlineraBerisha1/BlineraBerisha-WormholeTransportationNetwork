"""Phase 3 v2: bottleneck-guided, paired-ship local search.

Put this file at solver/phase3.py. Run from the project root:
    python -m solver.phase3 --input results/best_phase3.npy --iterations 60

Only solver.phase3 is replaced. Phase 2, official UDP and existing results
are untouched. New best is written to results/best_phase3_improved.npy.
"""
import argparse
import json
from itertools import zip_longest
from datetime import datetime
from pathlib import Path

import numpy as np

from solver.network import path_mean
from solver.pathfinding import neighbors
from solver.phase2 import decode, encode, evaluate


def ship_costs(udp, paths):
    """Actual per-ship costs, including time-ordered revisit penalties."""
    return np.asarray(udp._accumulate_variances(paths), dtype=float) * float(udp._fitness_scaling)


def _raw_variance(graph, path):
    return sum(float(graph[a][b]['v']) for a, b in zip(path, path[1:]))


def _arrival(udp, ship, path):
    return float(udp.delays[ship]) + float(path_mean(udp.network, path))


def _changes(udp, paths, ship, rng, count):
    """Return valid, deduplicated paths with computed arrival/variance."""
    original = tuple(map(int, paths[ship]))
    output = [(list(original), _arrival(udp, ship, original), _raw_variance(udp.network, original))]
    seen = {original}
    # Ask for additional candidates because neighbors may return duplicates.
    for proposed in neighbors(udp.network, paths[ship], rng, max_candidates=count * 2):
        p = tuple(map(int, proposed))
        if p in seen or len(p) < 2 or len(p) - 1 > udp.jump_limit:
            continue
        if p[0] not in udp.origins[ship] or p[-1] != udp.destination:
            continue
        if not all(udp.network.has_edge(a, b) for a, b in zip(p, p[1:])):
            continue
        if 0 in p[:-1]:  # zero is the chromosome padding sentinel
            continue
        seen.add(p)
        output.append((list(p), _arrival(udp, ship, p), _raw_variance(udp.network, p)))
        if len(output) - 1 >= count:
            break
    return output


def _pair_candidates(udp, paths, arrival_times, i, j, opts_i, opts_j, limit):
    """Quickly screen candidate pairs; official fitness remains authoritative.

    Ranking combines bottleneck raw-variance changes, visit-count changes,
    and proximity to the window; it does NOT pretend to be official fitness.
    """
    # Keep counts for unchanged ships; avoid building massive node sets repeatedly.
    counts = {}
    for path in paths:
        for node in path[:-1]:
            counts[int(node)] = counts.get(int(node), 0) + 1
    old_raw_i, old_raw_j = opts_i[0][2], opts_j[0][2]
    old_nodes = list(map(int, paths[i][:-1])) + list(map(int, paths[j][:-1]))
    base = dict(counts)
    for node in old_nodes:
        base[node] -= 1
    other_arrivals = np.delete(arrival_times, [i, j])
    lo = float(other_arrivals.min()) if other_arrivals.size else float('inf')
    hi = float(other_arrivals.max()) if other_arrivals.size else -float('inf')
    proposals = []
    for ai, (pi, ti, vi) in enumerate(opts_i):
        for bj, (pj, tj, vj) in enumerate(opts_j):
            if ai == 0 and bj == 0:
                continue
            span = max(hi, ti, tj) - min(lo, ti, tj)
            # Do not prune borderline float32 roundoff; official evaluator decides.
            if span > float(udp.window) + 2e-5:
                continue
            new_counts = {}
            for node in pi[:-1]:
                node = int(node)
                new_counts[node] = new_counts.get(node, 0) + 1
            for node in pj[:-1]:
                node = int(node)
                new_counts[node] = new_counts.get(node, 0) + 1
            # Change in collision pairs relative to current paths; cheap estimate.
            collision_delta = 0
            for node in set(old_nodes) | set(new_counts):
                before = counts.get(node, 0)
                after = base.get(node, 0) + new_counts.get(node, 0)
                collision_delta += max(0, after - 1) - max(0, before - 1)
            raw_delta = (vi - old_raw_i) + (vj - old_raw_j)
            # Revisit multiplier is 10; collision reductions are valuable but
            # not an absolute veto on otherwise promising candidates.
            heuristic = raw_delta + 0.04 * collision_delta + 0.0001 * span
            proposals.append((heuristic, ai, bj))
    proposals.sort(key=lambda t: t[0])
    return [(opts_i[ai][0], opts_j[bj][0]) for _, ai, bj in proposals[:limit]]


def optimize(udp, start, iterations=60, seed=42, candidates_per_ship=12,
             evaluations_per_iteration=20, partners=3, checkpoint=None):
    """Monotonic improvements with official evaluator and no worsening moves."""
    if iterations < 0 or min(candidates_per_ship, evaluations_per_iteration, partners) < 1:
        raise ValueError('iterations must be >=0; candidates/evaluations/partners >=1')
    rng = np.random.default_rng(seed)
    best = np.asarray(start, dtype=np.int32).reshape(-1).copy()
    stats = evaluate(udp, best)
    if not stats['feasible']:
        raise ValueError(f'Starting solution is not feasible: {stats}')
    paths = decode(udp, best)
    at = np.asarray([_arrival(udp, k, p) for k, p in enumerate(paths)])
    cache = {best.tobytes()}
    history = [{'iteration': 0, 'accepted': False, **stats}]
    evaluated = 0
    accepted_count = 0
    for step in range(1, iterations + 1):
        costs = ship_costs(udp, paths)
        ranking = np.argsort(-costs)
        # Focus top bottleneck, but periodically broaden to top three.
        focus = [int(ranking[0])] if step % 4 else [int(x) for x in ranking[:3]]
        candidate_groups = []
        for i in focus:
            remaining = [int(x) for x in ranking if x != i]
            # Mix arrival extremes with high-cost ships; reproducible exploration.
            preferred = [int(np.argmin(at)), int(np.argmax(at))] + remaining[:2]
            rng.shuffle(remaining)
            preferred += remaining
            partners_chosen = []
            for j in preferred:
                if j != i and j not in partners_chosen:
                    partners_chosen.append(j)
                if len(partners_chosen) >= min(partners, udp.n_ships - 1):
                    break
            opts_i = _changes(udp, paths, i, rng, candidates_per_ship)
            for j in partners_chosen:
                opts_j = _changes(udp, paths, j, rng, candidates_per_ship)
                pair_proposals = _pair_candidates(udp, paths, at, i, j, opts_i, opts_j,
                                                   evaluations_per_iteration)
                candidate_groups.append([(i, j, pi, pj) for pi, pj in pair_proposals])
        # Round-robin keeps each pair ranked, but explores different pairs.
        rng.shuffle(candidate_groups)
        candidate_trials = (candidate for group in zip_longest(*candidate_groups)
                            for candidate in group if candidate is not None)
        accepted = False
        checks_this_step = 0
        for i, j, pi, pj in candidate_trials:
            if checks_this_step >= evaluations_per_iteration:
                break
            trial_paths = list(paths)
            trial_paths[i], trial_paths[j] = pi, pj
            trial = encode(udp, trial_paths)
            key = trial.tobytes()
            if key in cache:
                continue
            cache.add(key)
            s = evaluate(udp, trial)
            evaluated += 1
            checks_this_step += 1
            if s['feasible'] and s['fitness'] < stats['fitness'] - 1e-8:
                best, stats, paths = trial, s, trial_paths
                at[i], at[j] = _arrival(udp, i, pi), _arrival(udp, j, pj)
                accepted = True
                accepted_count += 1
                if checkpoint is not None:
                    np.save(checkpoint, best)
                break
        history.append({'iteration': step, 'accepted': accepted,
                        'fitness': stats['fitness'],
                        'arrival_spread': float(np.max(at) - np.min(at)),
                        'official_evaluations': evaluated})
        if accepted or step % 5 == 0:
            print(f'Iteration {step}/{iterations} | fitness={stats["fitness"]:.6f} | '
                  f'accepted={accepted} | evaluations={evaluated}', flush=True)
    return best, {'final': stats, 'accepted_moves': accepted_count,
                  'official_evaluations': evaluated, 'history': history}


def main():
    parser = argparse.ArgumentParser(description='Improved paired-ship Phase 3 search.')
    parser.add_argument('--db', default='data/database.npz')
    parser.add_argument('--input', default='results/best_phase3.npy')
    parser.add_argument('--output', default='results/best_phase3_improved.npy')
    parser.add_argument('--iterations', type=int, default=60)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--candidates', type=int, default=12)
    parser.add_argument('--evaluations', type=int, default=20)
    parser.add_argument('--partners', type=int, default=3)
    args = parser.parse_args()
    source = Path(args.input)
    if not source.is_file():
        parser.error(f'Input not found: {source}')
    from wormhole_udp import wormhole_traversal_udp
    udp = wormhole_traversal_udp(database=args.db)
    x = np.load(source, allow_pickle=False).astype(np.int32)
    if x.shape != (udp.n_ships * udp.jump_limit,):
        parser.error('Chromosome shape must be (n_ships * jump_limit,)')
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    initial = evaluate(udp, x)
    print('Initial:', initial, flush=True)
    result, report = optimize(udp, x, iterations=args.iterations, seed=args.seed,
                              candidates_per_ship=args.candidates,
                              evaluations_per_iteration=args.evaluations,
                              partners=args.partners, checkpoint=output)
    # Verify before writing final output (checkpoint already saved if improved).
    verified = evaluate(udp, result)
    if verified != report['final'] or not verified['feasible']:
        raise AssertionError(f'Final verification mismatch: {verified}')
    np.save(output, result)
    report.update({'timestamp': datetime.now().isoformat(timespec='seconds'),
                   'input': str(source), 'output': str(output),
                   'seed': args.seed, 'iterations': args.iterations,
                   'candidates_per_ship': args.candidates,
                   'evaluations_per_iteration': args.evaluations,
                   'partners': args.partners, 'initial': initial})
    report_path = output.with_name(output.stem + '_statistics.json')
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('Final verified:', verified, flush=True)
    print(f'Saved {output} and {report_path}', flush=True)


if __name__ == '__main__':
    main()
