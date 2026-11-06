"""Phase 4: feasible iterated local search with annealed perturbations.

Add as solver/phase4.py alongside existing Phase 2/3. Run from project root:
 python -m solver.phase4 --input results/best_phase3_1212_backup.npy --rounds 12

Requires solver/phase3.py to be the improved version with _changes and optimize.
Official wormhole_udp.fitness() is the only decision authority for feasibility
and the best objective. Intermediate states may be worse but must be feasible.
"""
import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np

from solver.phase2 import decode, encode, evaluate
from solver.phase3 import _changes, _pair_candidates, _arrival, optimize as phase3_optimize, ship_costs


def perturb(udp, current, current_stats, rng, attempts, candidates, temperature, seen):
    """Try coordinated two-ship changes, accepting feasible worse moves sometimes.

    The timing window is tight, so a single-ship kick often cannot be feasible.
    """
    paths = decode(udp, current)
    costs = ship_costs(udp, paths)
    ranking = [int(s) for s in np.argsort(-costs)]
    arrivals = np.array([_arrival(udp, k, p) for k, p in enumerate(paths)])
    focus = ranking[:min(4, udp.n_ships)]
    focus += [int(x) for x in rng.permutation(udp.n_ships) if int(x) not in focus]
    checks = 0
    for i in focus:
        partners = [int(x) for x in rng.permutation(udp.n_ships) if int(x) != i]
        # Include the extremes, plus random partner choices.
        partners = list(dict.fromkeys([int(np.argmin(arrivals)),
                                       int(np.argmax(arrivals))] + partners))
        options_i = _changes(udp, paths, i, rng, candidates)
        for j in partners:
            if i == j:
                continue
            options_j = _changes(udp, paths, j, rng, candidates)
            pairs = _pair_candidates(udp, paths, arrivals, i, j,
                                     options_i, options_j, candidates * candidates)
            # Randomize the top promising choices instead of always taking first.
            pool = pairs[:max(15, candidates * 3)]
            for index in rng.permutation(len(pool)):
                if checks >= attempts:
                    return current, current_stats, checks, False
                pi, pj = pool[int(index)]
                trial_paths = list(paths)
                trial_paths[i], trial_paths[j] = pi, pj
                trial = encode(udp, trial_paths)
                key = trial.tobytes()
                if key in seen:
                    continue
                seen.add(key)
                stats = evaluate(udp, trial)
                checks += 1
                if not stats['feasible']:
                    continue
                delta = stats['fitness'] - current_stats['fitness']
                if delta <= 0 or rng.random() < math.exp(-min(delta / temperature, 700)):
                    return trial, stats, checks, True
    return current, current_stats, checks, False


def optimize(udp, start, rounds=12, local_iterations=8, seed=42,
             candidates=12, evaluations=15, partners=3,
             kick_attempts=35, temperature=2.0, cooling=0.90,
             output=None):
    if rounds < 0 or local_iterations < 0 or kick_attempts < 1:
        raise ValueError('rounds/local_iterations must be >=0; kick_attempts >=1')
    if min(candidates, evaluations, partners) < 1 or temperature <= 0 or not (0 < cooling <= 1):
        raise ValueError('candidates/evaluations/partners >=1; temperature >0; 0<cooling<=1')
    rng = np.random.default_rng(seed)
    best = np.asarray(start, dtype=np.int32).reshape(-1).copy()
    best_stats = evaluate(udp, best)
    if not best_stats['feasible']:
        raise ValueError(f'Input is not feasible: {best_stats}')
    current, current_stats = best.copy(), dict(best_stats)
    seen = {best.tobytes()}
    log = []
    total_evaluations = 0
    improvements = 0
    perturb_accepts = 0
    for round_no in range(1, rounds + 1):
        # A restart reduces the chance of walking indefinitely on one plateau.
        if round_no > 1 and round_no % 5 == 0:
            current, current_stats = best.copy(), dict(best_stats)
        temp = max(temperature * cooling ** (round_no - 1), 1e-9)
        current, current_stats, kick_checks, changed = perturb(
            udp, current, current_stats, rng, kick_attempts, candidates, temp, seen)
        total_evaluations += kick_checks
        perturb_accepts += int(changed)
        # Phase 3 performs monotonic greedy refinement from the perturbed point.
        refined, info = phase3_optimize(
            udp, current, iterations=local_iterations,
            seed=int(rng.integers(0, 2**31 - 1)),
            candidates_per_ship=candidates,
            evaluations_per_iteration=evaluations,
            partners=partners)
        total_evaluations += int(info['official_evaluations'])
        refined_stats = info['final']
        if not refined_stats['feasible']:
            raise AssertionError('Phase 3 unexpectedly returned infeasible solution')
        current, current_stats = refined, refined_stats
        improved = current_stats['fitness'] < best_stats['fitness'] - 1e-8
        if improved:
            best, best_stats = current.copy(), dict(current_stats)
            improvements += 1
            if output is not None:
                np.save(output, best)  # checkpoint new best immediately
        row = {'round': round_no, 'temperature': temp, 'perturb_accepted': changed,
               'kick_evaluations': kick_checks,
               'local_evaluations': info['official_evaluations'],
               'local_accepted_moves': info['accepted_moves'],
               'current_fitness': current_stats['fitness'],
               'best_fitness': best_stats['fitness'], 'new_global_best': improved}
        log.append(row)
        print(f'Round {round_no}/{rounds} | current={current_stats["fitness"]:.6f} | '
              f'BEST={best_stats["fitness"]:.6f} | kicked={changed} | '
              f'improved={improved} | evaluations={total_evaluations}', flush=True)
    return best, {'initial': evaluate(udp, start), 'final': best_stats,
                  'best_improvements': improvements, 'perturbations_accepted': perturb_accepts,
                  'official_evaluations': total_evaluations, 'history': log}


def main():
    p = argparse.ArgumentParser(description='Feasible iterated local search for SpOC2 wormholes')
    p.add_argument('--db', default='data/database.npz')
    p.add_argument('--input', default='results/best_phase3_1212_backup.npy')
    p.add_argument('--output', default='results/best_phase4.npy')
    p.add_argument('--rounds', type=int, default=12)
    p.add_argument('--local-iterations', type=int, default=8)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--candidates', type=int, default=12)
    p.add_argument('--evaluations', type=int, default=15)
    p.add_argument('--partners', type=int, default=3)
    p.add_argument('--kick-attempts', type=int, default=35)
    p.add_argument('--temperature', type=float, default=2.0)
    p.add_argument('--cooling', type=float, default=0.9)
    a = p.parse_args()
    if not Path(a.input).is_file():
        p.error(f'Input not found: {a.input}')
    if Path(a.input).resolve() == Path(a.output).resolve():
        p.error('Choose a different --output to preserve your input!')
    from wormhole_udp import wormhole_traversal_udp
    udp = wormhole_traversal_udp(database=a.db)
    x = np.load(a.input, allow_pickle=False).astype(np.int32)
    if x.shape != (udp.n_ships * udp.jump_limit,):
        p.error('Invalid chromosome shape')
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    print('Starting official evaluation:', evaluate(udp, x), flush=True)
    best, report = optimize(udp, x, rounds=a.rounds, local_iterations=a.local_iterations,
                            seed=a.seed, candidates=a.candidates, evaluations=a.evaluations,
                            partners=a.partners, kick_attempts=a.kick_attempts,
                            temperature=a.temperature, cooling=a.cooling, output=out)
    official = evaluate(udp, best)
    if not official['feasible'] or official != report['final']:
        raise AssertionError(f'Final official verification failed: {official}')
    np.save(out, best)
    report.update({'timestamp': datetime.now().isoformat(timespec='seconds'),
                   'input': a.input, 'output': str(out), 'seed': a.seed,
                   'rounds': a.rounds, 'local_iterations': a.local_iterations,
                   'candidates': a.candidates, 'evaluations': a.evaluations,
                   'partners': a.partners, 'kick_attempts': a.kick_attempts,
                   'temperature': a.temperature, 'cooling': a.cooling})
    report_path = out.with_name(out.stem + '_statistics.json')
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('Final verified:', official, flush=True)
    print(f'Saved: {out}, {report_path}', flush=True)


if __name__ == '__main__':
    main()
