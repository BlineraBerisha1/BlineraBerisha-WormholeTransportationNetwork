""" Phase 2 local search for the Wormhole Transportation Network problem.
This is a single-ship local search that improves the solution from phase 1.
python -m solver.phase2 --iterations 30 --seed 42

Additive upgrade: existing baseline.py, local_search.py, and evaluation.py are untouched.
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from solver.network import arrivals
from solver.pathfinding import neighbors


def decode(udp, x):
    paths, _ = udp._chromosome_to_paths(x)
    return [list(map(int, p)) for p in paths]


def encode(udp, paths):
    out = np.zeros((udp.n_ships, udp.jump_limit), dtype=np.int32)
    for i, p in enumerate(paths):
        if not p or p[-1] != udp.destination:
            raise ValueError(f'Ship {i+1}: invalid destination')
        internal = p[:-1]   # Destination is implied by the UDP!
        if not internal or len(internal) > udp.jump_limit:
            raise ValueError(f'Ship {i+1}: invalid path length')
        out[i, :len(internal)] = internal
    return out.ravel()


def evaluate(udp, x):
    score, origin, path, window = (float(v) for v in udp.fitness(x))
    return {'fitness': score, 'origin_violations': origin,
            'path_violations': path, 'window_violation': window,
            'feasible': origin == 0 and path == 0 and window <= 1e-7}


def optimize(udp, start, iterations, seed):
    rng = np.random.default_rng(seed)
    best = np.asarray(start, dtype=np.int32).copy()
    stats = evaluate(udp, best)
    if not stats['feasible']:
        raise ValueError('Starting chromosome is not feasible. Use a validated baseline.')
    history = [{'iteration': 0, **stats}]
    paths = decode(udp, best)
    for iteration in range(1, iterations + 1):
        ship = int(rng.integers(0, udp.n_ships))
        changes = neighbors(udp.network, paths[ship], rng, max_candidates=12)
        accepted = False
        for revised in changes:
            if len(revised) - 1 > udp.jump_limit:
                continue
            trial_paths = list(paths)
            trial_paths[ship] = revised
            # Fast rejection; official evaluator decides final validity.
            at = arrivals(udp, trial_paths)
            if max(at) - min(at) > float(udp.window) + 1e-5:
                continue
            trial = encode(udp, trial_paths)
            trial_stats = evaluate(udp, trial)
            if trial_stats['feasible'] and trial_stats['fitness'] < stats['fitness'] - 1e-9:
                best, stats, paths = trial, trial_stats, trial_paths
                accepted = True
                break
        history.append({'iteration': iteration, 'accepted': accepted, **stats})
        if iteration % 10 == 0 or accepted:
            print(f'Iteration {iteration}: fitness={stats["fitness"]:.6f}, accepted={accepted}', flush=True)
    return best, stats, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', default='data/database.npz')
    parser.add_argument('--input', default='results/best_local_search.npy')
    parser.add_argument('--iterations', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', default='results')
    args = parser.parse_args()
    if args.iterations < 0:
        parser.error('--iterations must be nonnegative')
    input_path = Path(args.input)
    if not input_path.is_file():
        parser.error(f'Missing {input_path}; use --input data/example.npy')
    from wormhole_udp import wormhole_traversal_udp
    udp = wormhole_traversal_udp(database=args.db)
    starting = np.load(input_path).astype(np.int32)
    if starting.size != udp.n_ships * udp.jump_limit:
        parser.error('Input chromosome has incorrect length')
    initial = evaluate(udp, starting)
    print('Initial:', initial, flush=True)
    final_x, final, history = optimize(udp, starting, args.iterations, args.seed)
    folder = Path(args.output)
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / 'best_phase2.npy', final_x)
    report = {'timestamp': datetime.now().isoformat(timespec='seconds'),
              'source': str(input_path), 'seed': args.seed,
              'iterations': args.iterations, 'initial': initial, 'final': final,
              'history': history}
    (folder / 'phase2_statistics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('Final:', final, flush=True)
    print('Saved: results/best_phase2.npy and results/phase2_statistics.json', flush=True)


if __name__ == '__main__':
    main()
