"""Verify a solution using the original ESA SpOC2 UDP and export submit-ready files.

Usage: python verify.py --solution results/best_solution.npy
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
from solver.optimize import ROOT
from wormhole_udp import wormhole_traversal_udp
from solver.output.formatter import SolutionFormatter


def verify(solution: Path, database: Path, output: Path):
    array = np.load(solution, allow_pickle=False)
    if array.shape != (6000,) or not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f'Expected a 6000-integer chromosome, found {array.shape} {array.dtype}')
    udp = wormhole_traversal_udp(database=str(database))
    result = udp.fitness(array)
    feasible = int(result[1]) == 0 and int(result[2]) == 0 and float(result[3]) <= 0
    paths,_ = udp._chromosome_to_paths(array)
    variances = udp._accumulate_variances(paths)*udp._fitness_scaling
    _,arrival = udp._compute_arrival_gaps(paths)
    report = {
        'fitness':float(result[0]),
        'origin_violations':int(result[1]),
        'path_violations':int(result[2]),
        'window_violation':float(result[3]),
        'feasible':bool(feasible),
        'fitness_per_ship':{str(i+1):float(v) for i,v in enumerate(variances)},
        'arrival_times':{str(i+1):float(t) for i,t in enumerate(arrival)},
        'path_lengths':{str(i+1):int(len(p)-1) for i,p in enumerate(paths)},
        'bottleneck_ship':int(np.argmax(variances)+1),
    }
    if not feasible:
        raise ValueError(f'SOLUTION FAILED OFFICIAL VALIDATION: {report}')
    output.mkdir(parents=True,exist_ok=True)
    np.save(output/'best_solution.npy',array)
    (output/'decision_vector.json').write_text(json.dumps(array.astype(int).tolist())+'\n')
    SolutionFormatter().create_submission_json(
        array.astype(np.int32),str(output/'submission.json'),
        submission_name=f'Valid solution fitness {float(result[0]):.9f}',
        submission_description='Official SpOC2 evaluator: all constraints satisfied',
    )
    (output/'verification_report.json').write_text(json.dumps(report,indent=2)+'\n')
    (output/'fitness_summary.txt').write_text(
        'ESA SpOC2 Wormhole Transportation Network\n'
        f'OFFICIALLY VERIFIED FITNESS: {report["fitness"]:.12f}\n'
        f'Feasible: {report["feasible"]}\n'
        f'Origin violations: {report["origin_violations"]}\n'
        f'Path violations: {report["path_violations"]}\n'
        f'Window violation: {report["window_violation"]:.12f}\n'
        f'Bottleneck ship: {report["bottleneck_ship"]}\n'
    )
    print(json.dumps({k:report[k] for k in ['fitness','feasible','origin_violations','path_violations','window_violation','bottleneck_ship']},indent=2))
    print('Verified files written to',output.resolve())
    return report


if __name__ == '__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--solution',type=Path,default=ROOT/'results/best_solution.npy')
    ap.add_argument('--data',type=Path,default=ROOT/'data/database.npz')
    ap.add_argument('--output-dir',type=Path,default=ROOT/'results')
    options=ap.parse_args()
    verify(options.solution,options.data,options.output_dir)
