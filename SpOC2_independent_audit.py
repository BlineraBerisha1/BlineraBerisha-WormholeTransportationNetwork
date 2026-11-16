"""Independent, read-only audit of an ESA SpOC2 saved solution.

Place this script next to verify.py in the project directory. Run:
    python SpOC2_independent_audit.py

It deliberately does not call solver.competitive, solver.optimize, or the
project's FastEvaluator. It checks raw graph edges and reconstructs global
visit-rank costs, then compares the result against the original ESA UDP.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DB = ROOT / 'data' / 'database.npz'
SOLUTION = ROOT / 'results' / 'best_solution.npy'
EVALUATOR = ROOT / 'wormhole_udp.py'
EXPECTED_SHA256 = {
    'wormhole_udp.py': 'd2c1aad6ed82dd8ca812376dd3a4a538e2d9306093a966004a278722169801ff',
    'database.npz': 'e256666a761b0f76234d43170196ab32901083382dbacdb4f8ea82b8e875f900',
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def require(value: bool, message: str):
    if not value:
        raise AssertionError(message)


def audit() -> None:
    for p in [DB, SOLUTION, EVALUATOR]:
        require(p.is_file(), f'Missing required file: {p}')

    print('1. Checking original challenge files...')
    for name, path in [('wormhole_udp.py', EVALUATOR), ('database.npz', DB)]:
        actual = sha256(path)
        require(actual == EXPECTED_SHA256[name],
                f'{name} differs from the original supplied project! SHA256: {actual}')
        print(f'   [PASS] {name}: original SHA256 matches')

    x = np.load(SOLUTION, allow_pickle=False)
    require(x.shape == (6000,) and np.issubdtype(x.dtype, np.integer),
            f'Expected a 6000-integer chromosome, got shape={x.shape}, dtype={x.dtype}')
    print('2. Checking chromosome, origins and every directed edge...')

    with np.load(DB, allow_pickle=False) as z:
        edges = np.asarray(z['edges'], dtype=np.int32)
        mv = np.asarray(z['meanvar'], dtype=np.float32)
        origins = np.asarray(z['origins'], dtype=np.int32)
        destination = int(z['destination'])
        delays = np.asarray(z['delays'], dtype=np.float32)
        window = float(z['window'])
        jump_limit = int(z['jump_limit'])
    require(len(delays) == 12 and jump_limit == 500, 'Unexpected challenge dimensions')
    require(len(edges) == len(mv), 'Edge/weight count mismatch')

    # Independent directed-edge lookup by sorted integer (src, dst) key.
    base = int(max(edges.max(), destination)) + 1
    keys = edges[:, 0].astype(np.int64) * base + edges[:, 1]
    order = np.argsort(keys)
    keys = keys[order]
    mv = mv[order]
    require(bool(np.all(keys[1:] > keys[:-1])), 'Database contains duplicate directed edges')

    def get_edges(sources, targets):
        looked_up = np.asarray(sources, dtype=np.int64)*base + np.asarray(targets, dtype=np.int64)
        idx = np.searchsorted(keys, looked_up)
        if np.any(idx >= len(keys)) or not np.array_equal(keys[idx], looked_up):
            raise AssertionError('At least one route contains an invalid directed edge')
        return mv[idx, 0], mv[idx, 1]

    rows = x.reshape((12, 500))
    paths = []
    means_by_ship = []
    vars_by_ship = []
    for ship, row in enumerate(rows):
        zero = np.flatnonzero(row == 0)
        length = int(zero[0]) if len(zero) else len(row)
        require(1 <= length <= jump_limit, f'Ship {ship+1}: invalid length {length}')
        require(np.all(row[:length] > 0), f'Ship {ship+1}: invalid node ID')
        require(np.all(row[length:] == 0), f'Ship {ship+1}: nonzero padding after first zero')
        path = row[:length]
        require(int(path[0]) in origins[ship].tolist(), f'Ship {ship+1}: invalid origin {path[0]}')
        target = np.r_[path[1:], destination]
        means, variances = get_edges(path, target)
        paths.append(path)
        means_by_ship.append(means)
        vars_by_ship.append(variances)
        print(f'   [PASS] Ship {ship+1:2}: origin {int(path[0]):5}, jumps {length:3}, '
              f'repeated nodes {length-len(np.unique(path)):2}')

    print('3. Independently reconstructing arrival times and visit-rank costs...')
    arrivals = np.empty(12, dtype=np.float32)
    visits = defaultdict(list)
    for ship, path in enumerate(paths):
        # Original evaluator computes arrival = cast(float sum of edge means) + delay.
        arrivals[ship] = np.float32(float(sum(float(a) for a in means_by_ship[ship])) + float(delays[ship]))
        # Official visit ordering uses iterative float32 departure times.
        t = np.float32(delays[ship])
        for jump, (node, m, v) in enumerate(zip(path, means_by_ship[ship], vars_by_ship[ship])):
            visits[int(node)].append((ship, jump, t, float(v)))
            t = np.float32(t + m)

    per_ship = np.zeros(12, dtype=np.float64)
    for node in visits:
        # Python's stable sort matches the official tie rule (ships/jumps in traversal order).
        ranked = sorted(visits[node], key=lambda item: item[2])
        for rank, (ship, jump, t, variance) in enumerate(ranked):
            penalty = 10.0**rank if rank < 10 else 1e10
            per_ship[ship] += variance * penalty
    per_ship *= 10000.0
    independent_fitness = float(per_ship.max())

    print('4. Comparing every measurement with the UNMODIFIED official evaluator...')
    from wormhole_udp import wormhole_traversal_udp
    official = wormhole_traversal_udp(database=str(DB))
    fit = official.fitness(x)
    official_paths, _ = official._chromosome_to_paths(x)
    official_per_ship = official._accumulate_variances(official_paths) * 10000.0
    _, official_arrivals = official._compute_arrival_gaps(official_paths)

    require(fit[1] == 0, f'Official origin constraint violation: {fit[1]}')
    require(fit[2] == 0, f'Official directed-path violation: {fit[2]}')
    require(fit[3] <= 0, f'Official arrival-window violation: {fit[3]}')
    require(np.isclose(independent_fitness, fit[0], atol=0.02, rtol=0),
            f'Independent fitness {independent_fitness} differs from official {fit[0]}')
    np.testing.assert_allclose(per_ship, official_per_ship, atol=0.02, rtol=0)
    np.testing.assert_allclose(arrivals, official_arrivals, atol=0.0001, rtol=0)
    assert float(official_arrivals.max() - official_arrivals.min()) <= window + 1e-6

    submission = ROOT / 'results' / 'submission.json'
    if submission.is_file():
        data = json.loads(submission.read_text(encoding='utf-8'))
        require(isinstance(data, list) and len(data) > 0, 'Malformed submission JSON wrapper')
        require(np.array_equal(np.asarray(data[0]['decisionVector'], dtype=np.int32), x),
                'Submission JSON contains a DIFFERENT chromosome! Run python verify.py')
        print('   [PASS] submission.json contains exactly the verified best chromosome')
    else:
        print('   [INFO] submission.json not present; run python verify.py to generate it')

    report_path = ROOT / 'results' / 'verification_report.json'
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding='utf-8'))
        require(np.isclose(float(report['fitness']), float(fit[0]), atol=1e-5, rtol=0),
                'Verification report is stale; run python verify.py')
        print('   [PASS] verification_report.json matches the current chromosome')

    print('\nFINAL AUDIT: PASS')
    print(f'Official fitness:           {float(fit[0]):.12f}')
    print(f'Independent fitness:        {independent_fitness:.12f}')
    print(f'Origin violations:         {fit[1]}')
    print(f'Path violations:           {fit[2]}')
    print(f'Window violation:          {float(fit[3]):.12f}')
    print(f'Arrival-time range:        {float(official_arrivals.min()):.7f} to {float(official_arrivals.max()):.7f}')
    print(f'Bottleneck ship:           {int(np.argmax(official_per_ship))+1}')
    print('All 12 origins, directed routes, jump limits, global revisit penalties,')
    print('arrival synchronization, and solution exports have passed.')


if __name__ == '__main__':
    try:
        audit()
    except Exception as e:
        print(f'\nFINAL AUDIT: FAIL -- {type(e).__name__}: {e}')
        raise SystemExit(1)
