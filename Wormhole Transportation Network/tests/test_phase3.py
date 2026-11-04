import networkx as nx
import numpy as np
from solver.phase3 import arrival_spread, candidate_changes, prioritize_combinations, optimize
from solver.phase2 import encode, decode, evaluate


class TinyUDP:
    n_ships = 2
    jump_limit = 5
    destination = 9
    window = 1.0
    delays = np.zeros(2)
    origins = [{1}, {5}]
    _fitness_scaling = 10000

    def __init__(self):
        self.network = nx.DiGraph()
        for a, b, m, v in [(1,2,1,.04),(2,9,1,.04), (1,3,1,.01),(3,9,1,.01),
                            (5,6,1,.04),(6,9,1,.04),(5,7,1,.01),(7,9,1,.01)]:
            self.network.add_edge(a,b,m=m,v=v)

    def _chromosome_to_paths(self,x):
        paths = []
        for row in x.reshape(2,5):
            zeros=np.flatnonzero(row==0)
            p=row[:zeros[0]] if len(zeros) else row
            paths.append(np.r_[p,9])
        return paths, 3

    def _accumulate_variances(self,paths):
        return np.array([sum(self.network[a][b]['v'] for a,b in zip(p,p[1:])) for p in paths])

    def fitness(self,x):
        paths,_=self._chromosome_to_paths(x)
        eq0=sum(int(p[0] not in self.origins[i]) for i,p in enumerate(paths))
        eq1=sum(int(not all(self.network.has_edge(a,b) for a,b in zip(p,p[1:]))) for p in paths)
        times=[sum(self.network[a][b]['m'] for a,b in zip(p,p[1:])) for p in paths]
        return [float(max(self._accumulate_variances(paths)))*10000,eq0,eq1,max(times)-min(times)-self.window]


def test_candidates_and_pair_combinations():
    u=TinyUDP()
    p=[[1,2,9],[5,6,9]]
    rng=np.random.default_rng(5)
    a=candidate_changes(u,p,0,rng,8)
    b=candidate_changes(u,p,1,rng,8)
    assert any(x[0]==[1,3,9] for x in a)
    pairs=list(prioritize_combinations(u,p,np.array([2.,2.]),0,1,a,b,20))
    assert any(x==[1,3,9] and y==[5,7,9] for x,y in pairs)
    assert arrival_spread([2,2])==0


def test_improvement_and_validity():
    u=TinyUDP()
    x=encode(u,[[1,2,9],[5,6,9]])
    best,report=optimize(u,x,iterations=16,seed=7,candidates_per_ship=8,evaluations_per_iteration=15)
    assert evaluate(u,best)['feasible']
    assert report['final']['fitness'] <= evaluate(u,x)['fitness']
    assert all(p[-1]==9 for p in decode(u,best))
