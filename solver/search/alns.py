"""Active competitive solver coordinating construction, ALNS and official checks.

No genetic algorithm is used. Core search logic was moved from the original
solver/competitive.py without changing path search, scoring or acceptance.
"""
import json
import time
from pathlib import Path
import numpy as np
from solver.paths import ROOT
from solver.data.csr import CSRNetwork
from solver.data.sparse import SparseNetwork
from solver.model.chromosome import to_paths, to_chromosome
from solver.evaluation.fast import FastEvaluator
from solver.heuristics.construction import RouteConstruction
from solver.search.moves import NeighborhoodMoves
from solver.output.checkpoints import write_checkpoint


class CompetitiveSolver(NeighborhoodMoves, RouteConstruction):
    def __init__(self, data:Path, output:Path, seed:int=2026, log:bool=True):
        self.data=Path(data)
        self.output=Path(output)
        self.output.mkdir(parents=True,exist_ok=True)
        self.graph=CSRNetwork(self.data)
        self.seed=seed
        self.rng=np.random.default_rng(seed)
        self.log=log
        self.sparse=SparseNetwork(self.data)
        self.proxy=FastEvaluator(self.sparse)
        self.potential_cache={}
        self.discount=0.80
        self.end_edges_cache={}
        self.best_sum=np.inf
        self.udp=None
        self.best=np.inf
        self.best_x=None
        self.best_phase=None
        self.history=[]
        self.candidates=0
        self.valid=0
        self.start=time.monotonic()
    def official(self,x):
        if self.udp is None:
            from wormhole_udp import wormhole_traversal_udp
            self.udp=wormhole_traversal_udp(database=str(self.data))
        return self.udp.fitness(np.asarray(x,dtype=np.int32))
    def _read_history(self):
        path=self.output/'competition_history.json'
        if path.is_file():
            try:
                self.history=json.loads(path.read_text())
                if not isinstance(self.history,list):self.history=[]
            except (ValueError, OSError):self.history=[]
    def init(self, candidates=()):
        self._read_history()
        all_candidates=[self.output/'best_solution.npy',ROOT/'results/best_solution.npy',ROOT/'data/example.npy']
        all_candidates.extend(Path(p) for p in candidates)
        selected=None
        for f in dict.fromkeys(all_candidates):
            if not f.exists():continue
            try:
                x=np.load(f,allow_pickle=False).astype(np.int32)
                if x.shape!=(self.graph.jump_limit*len(self.graph.delays),):continue
                official=self.official(x)
                if int(official[1])==0 and int(official[2])==0 and official[3]<=0:
                    if float(official[0])<self.best:
                        self.best=float(official[0]);selected=x.copy();self.best_phase='baseline'
                        self.best_sum=float(np.sum(self.proxy.score(to_paths(x,self.sparse)).costs))
                        if self.log: print('Feasible input:',f,'fitness',self.best,flush=True)
            except (ValueError, OSError) as exc:
                if self.log: print('Skipping invalid candidate',f,repr(exc),flush=True)
        if selected is None:raise ValueError('No officially feasible starting chromosome supplied')
        self.best_x=selected
        self._save('baseline')
        self.baseline=self.best
        return self.best
    def _save(self,phase):
        return write_checkpoint(self,phase)
    def evaluate(self,chromosome,phase):
        self.candidates+=1
        # Cheap directed-path and window prefilter avoids official work on failures.
        paths=to_paths(chromosome,self.sparse)
        try:proxy=self.proxy.score(paths)
        except ValueError:return False
        if proxy.window>-.00005 or proxy.objective > self.best+0.05:
            return False
        official=self.official(chromosome)
        if official[1]!=0 or official[2]!=0 or official[3]>0:
            return False
        self.valid+=1
        f=float(official[0])
        candidate_sum=float(np.sum(proxy.costs))
        is_record=f<self.best-1e-8
        is_tie=(abs(f-self.best)<1e-7 and candidate_sum<self.best_sum-0.005)
        if is_record or is_tie:
            self.best=f
            self.best_sum=candidate_sum
            self.best_x=chromosome.copy()
            self.best_phase=phase
            self.history.append({'phase':phase,'fitness':f,'record':bool(is_record),'window_violation':float(official[3]),
                     'candidates':self.candidates,'elapsed_seconds':round(time.monotonic()-self.start,2)})
            self._save(phase)
            if self.log and is_record:print(f'NEW OFFICIAL BEST {f:.9f} phase={phase} candidates={self.candidates}',flush=True)
            return True
        return False
    def run(self,minutes=10,phase2_trials=40,seed_count=12,alns_trials=30):
        g=self.graph
        deadline=time.monotonic()+max(0,minutes)*60
        # Phase 2/3: independently construct variable-length routes for each
        # possible window, but use occupancy penalties across all 12 ships.
        phase_start=time.monotonic()
        windows=np.linspace(-.60,-.38,max(3,seed_count))
        for idx,lo in enumerate(windows):
            if time.monotonic()>=deadline:break
            x=self.new_joint(float(lo),trials=phase2_trials)
            if x is not None:self.evaluate(x,'joint_variable_length')
            if self.log: print(f'Window {idx+1}/{len(windows)} [{lo:.4f},{lo+1:.4f}]; best={self.best:.8f}',flush=True)
        if self.log: print(f'Construction phase: {time.monotonic()-phase_start:.1f}s, best={self.best:.8f}',flush=True)
        # Phase 4: rebuild bottleneck ships while holding existing arrival band,
        # and occasionally restart all 12 routes in a shifted window.
        step=0
        failure=0
        while time.monotonic()<deadline:
            step+=1
            score=self.proxy.score(to_paths(self.best_x,self.sparse))
            most_expensive=np.argsort(-score.costs)
            if self.rng.random()<.1:
                lo=float(self.rng.uniform(-.59,-.39))
                x=self.new_joint(lo,trials=max(6,phase2_trials//2))
                success=(x is not None and self.evaluate(x,'ALNS: shifted-window restart'))
            else:
                s=int(most_expensive[self.rng.integers(0,min(4,len(most_expensive)))])
                # Existing arrival time bounds all other ships. Entire route
                # can be replaced at any length and by any permitted origin.
                other=np.delete(score.arrival,s)
                lo=float(other.max()-g.window+.0015)
                hi=float(other.min()+g.window-.0015)
                if self.rng.random()<.72:
                    success=self.reroute_segment(s,trials=alns_trials)
                else:
                    success=self.rebuild(s,lo,hi,trials=alns_trials)
            if success:failure=0
            else:failure+=1
            if step%20==0 and self.log:
                print(f'ALNS {step} best={self.best:.8f} candidates={self.candidates} failed={failure}',flush=True)
        self._save('final_official_validation')
        f=self.official(self.best_x)
        assert f[1]==0 and f[2]==0 and f[3]<=0
        print('FINAL OFFICIAL',list(map(float,f)), 'candidates', self.candidates, flush=True)
        return f
