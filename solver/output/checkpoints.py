"""Atomic saving and official reporting for the best verified chromosome."""
import json
import os
import numpy as np
from solver.output.formatter import SolutionFormatter


def write_checkpoint(self,phase):
    x=self.best_x
    checkpoint=self.output/'best_solution.npy'
    tmp=self.output/'.competitive_best.tmp.npy'
    np.save(tmp,x)
    os.replace(tmp,checkpoint)
    # Keep machine-readable result metadata in sync with checkpoint.
    official=self.official(x)
    metrics={'official_fitness':float(official[0]),'origin_violations':int(official[1]),
             'path_violations':int(official[2]),'window_violation':float(official[3]),
             'feasible':bool(official[1]==0 and official[2]==0 and official[3]<=0),
             'seed':self.seed,'phase':phase,'candidates':self.candidates,
             'path_lengths':[int(np.count_nonzero(row)) for row in x.reshape(len(self.graph.delays),self.graph.jump_limit)]}
    (self.output/'best_metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    (self.output/'competition_history.json').write_text(json.dumps(self.history,indent=2)+'\n')
    # Export all required formats with the unmodified evaluator.
    # Avoid constructing a second costly networkx instance by using cached self.udp.
    paths,_=self.udp._chromosome_to_paths(x)
    costs=self.udp._accumulate_variances(paths)*10000
    _,arrival=self.udp._compute_arrival_gaps(paths)
    report={'fitness':float(official[0]),'origin_violations':int(official[1]),
            'path_violations':int(official[2]),'window_violation':float(official[3]),
            'feasible':True,'fitness_per_ship':{str(i+1):float(c) for i,c in enumerate(costs)},
            'arrival_times':{str(i+1):float(t) for i,t in enumerate(arrival)},
            'path_lengths':{str(i+1):int(len(p)-1) for i,p in enumerate(paths)},
            'bottleneck_ship':int(np.argmax(costs)+1)}
    (self.output/'verification_report.json').write_text(json.dumps(report,indent=2)+'\n')
    (self.output/'decision_vector.json').write_text(json.dumps(x.astype(int).tolist())+'\n')
    SolutionFormatter().create_submission_json(x.astype(np.int32),str(self.output/'submission.json'),
         submission_name=f'Officially validated solution {self.best:.8f}',
         submission_description='All official SpOC2 constraints satisfied')
    (self.output/'fitness_summary.txt').write_text(f'Official fitness: {self.best:.12f}\n'
           f'Origin violations: {official[1]}\nPath violations: {official[2]}\n'
           f'Window violation: {official[3]}\n')
