"""Earlier, optional stochastic two-hop rewiring search.

This implementation is retained because it is real, working algorithmic code.
The primary active optimizer is solver.search.alns.CompetitiveSolver.
"""
from pathlib import Path
import json
import math
import time
import numpy as np
from solver.data.sparse import SparseNetwork
from solver.model.chromosome import to_paths, to_chromosome
from solver.evaluation.fast import FastEvaluator

class Optimizer:
    def __init__(self, data_path:Path, out_dir:Path, seed:int=42, checkpoint:str='best_solution.npy'):
        self.net=SparseNetwork(data_path)
        self.eval=FastEvaluator(self.net)
        self.rng=np.random.default_rng(seed)
        self.out=out_dir
        self.out.mkdir(parents=True,exist_ok=True)
        self.checkpoint=self.out/checkpoint
        self.tried=0;self.accepted=0;self.improvements=0
        self.seed=seed;self.history=[]
        self._official=None
        self._dbpath=data_path

    def official(self,x:np.ndarray):
        if self._official is None:
            from wormhole_udp import wormhole_traversal_udp
            self._official=wormhole_traversal_udp(database=str(self._dbpath))
        return self._official.fitness(x)

    @staticmethod
    def feasible(f)->bool:
        return int(f[1])==0 and int(f[2])==0 and float(f[3])<=0

    def init(self,source:Path):
        x=np.load(source)
        fitness=self.official(x)
        if not self.feasible(fitness):raise ValueError(f'Initial solution not feasible: {fitness}')
        self.paths=to_paths(x,self.net)
        self.curr=self.eval.score(self.paths)
        self.best_x=x.copy()
        self.best_official=float(fitness[0]);self.best_official_window=float(fitness[3])
        self.best=self.best_official
        self._save_best('initial',fitness)
        print(f'Validated feasible start: {self.best_official:.9f}, window={float(fitness[3]):.9f}',flush=True)

    def _save_best(self,phase:str,official_fitness):
        temporary=self.checkpoint.with_suffix(".tmp.npy")
        np.save(temporary,self.best_x)
        temporary.replace(self.checkpoint)
        metrics={'official_fitness':float(official_fitness[0]),'origin_violations':int(official_fitness[1]),'path_violations':int(official_fitness[2]),'window_violation':float(official_fitness[3]),'feasible':self.feasible(official_fitness),'seed':self.seed,'phase':phase,'evaluations':self.tried,'accepted':self.accepted,'improvements':self.improvements,'path_lengths':[int(np.count_nonzero(q)) for q in self.best_x.reshape(self.net.n_ships,self.net.jump_limit)]}
        (self.out/'best_metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
        (self.out/'history.json').write_text(json.dumps(self.history,indent=2)+'\n')

    def try_mutation(self, temperature:float=0.0, two=False):
        net=self.net
        # Weighted search tends to improve high-cost ships, while occasionally
        # fixing other ships in anticipation of changing bottlenecks.
        rank=np.argsort(self.curr.costs)
        idx=int(self.rng.integers(0,net.n_ships))
        if self.rng.random()<0.7:
            idx=int(rank[-int(self.rng.integers(1,6))])
        path=self.paths[idx]
        if len(path)<2:return
        j=int(self.rng.integers(1,len(path))) # may rewire final waypoint
        predecessor=int(path[j-1]); successor=int(path[j+1]) if j+1<len(path) else net.destination
        bridges=net.bridges(predecessor,successor)
        if len(bridges)==0:return
        # choose candidate from near-optimal edge-variance, or randomly for diversification
        if len(bridges)>1 and self.rng.random()<0.7:
            candidates=self.rng.choice(bridges,size=min(len(bridges),8),replace=False)
            m1,v1=net.values(np.full(len(candidates),predecessor),candidates)
            m2,v2=net.values(candidates,np.full(len(candidates),successor))
            # Favor low variance at new edges, with some noise.
            weights=v1+v2+self.rng.random(len(candidates))*0.0002
            new=int(candidates[np.argmin(weights)])
        else:new=int(self.rng.choice(bridges))
        if new==int(path[j]):return
        old=int(path[j]);means_old,_=net.values(np.array([predecessor,old]),np.array([old,successor])); means_new,_=net.values(np.array([predecessor,new]),np.array([new,successor]))
        delta=float(np.sum(means_new,dtype=np.float64)-np.sum(means_old,dtype=np.float64))
        changed=path.copy();changed[j]=new
        self.consider(idx,changed,delta,temperature)

    def consider(self,idx:int,changed:np.ndarray,delta:float,temperature:float):
        net=self.net
        path=self.paths[idx]
        # Window prefilter based on current float32 arrivals.
        check=self.curr.arrival.copy();check[idx]=np.float32(check[idx]+delta)
        if float(check.max()-check.min())>net.window-0.000015:return
        self.tried+=1
        self.paths[idx]=changed
        try:next_score=self.eval.score(self.paths)
        except ValueError:
            self.paths[idx]=path;return
        if next_score.window> -0.000015:
            self.paths[idx]=path;return
        # Objective uses variance of bottleneck ship. A sum term keeps other
        # ships competitive and facilitates progress when bottleneck changes.
        current_energy=self.curr.objective+0.10*np.mean(self.curr.costs)
        next_energy=next_score.objective+0.10*np.mean(next_score.costs)
        energy_delta=next_energy-current_energy
        allow=energy_delta < -1e-8 or (temperature>0 and energy_delta<20 and self.rng.random()<math.exp(-max(0.0,energy_delta)/temperature))
        if not allow:
            self.paths[idx]=path;return
        # Check with official rules before recording an improved incumbent.
        self.curr=next_score
        self.accepted+=1
        if next_score.objective < self.best_official - 0.001:
            chromosome=to_chromosome(self.paths,net)
            official_fitness=self.official(chromosome)
            if self.feasible(official_fitness) and float(official_fitness[0]) < self.best_official-1e-8:
                self.best_official=float(official_fitness[0]);self.best_official_window=float(official_fitness[3])
                self.best_x=chromosome.copy();self.improvements+=1
                self.history.append({'step':self.tried,'fitness':self.best_official,'window':self.best_official_window,'accepted':self.accepted})
                self._save_best('stochastic_local_search',official_fitness)
                print(f'  NEW BEST {self.best_official:.9f}, evals={self.tried}, window={self.best_official_window:.8f}',flush=True)
            elif not self.feasible(official_fitness):
                # Revert a move if approximation incorrectly predicted feasibility.
                self.paths[idx]=path
                self.curr=self.eval.score(self.paths)
                self.accepted-=1

    def try_double(self, temperature:float=0.0):
        """Replace two adjacent waypoints with a valid three-edge bridge."""
        net=self.net
        rank=np.argsort(self.curr.costs)
        ship=int(rank[-int(self.rng.integers(1,7))]) if self.rng.random()<0.8 else int(self.rng.integers(net.n_ships))
        path=self.paths[ship]
        if len(path)<4:return
        j=int(self.rng.integers(1,len(path)-1))
        a=int(path[j-1]);b=int(path[j+2]) if j+2<len(path) else net.destination
        out=net.out[a]
        if not len(out):return
        u_options=self.rng.choice(out,size=min(len(out),10),replace=False)
        options=[]
        for u in u_options:
            v_options=net.bridges(int(u),b)
            if not len(v_options):continue
            candidates=self.rng.choice(v_options,size=min(len(v_options),6),replace=False)
            src=np.r_[np.full(len(candidates),a),np.full(len(candidates),int(u)),candidates]
            dst=np.r_[np.full(len(candidates),int(u)),candidates,np.full(len(candidates),b)]
            m,v=net.values(src,dst)
            n=len(candidates)
            vari=v[:n]+v[n:2*n]+v[2*n:]
            best=int(np.argmin(vari+self.rng.random(n)*0.0003))
            options.append((float(vari[best]),int(u),int(candidates[best]),float(m[best]+m[n+best]+m[2*n+best])))
        if not options:return
        if self.rng.random()<0.75:
            _,u,v,new_mean=min(options)
        else:
            _,u,v,new_mean=options[int(self.rng.integers(len(options)))]
        if int(path[j])==u and int(path[j+1])==v:return
        old_src=np.array([a,path[j],path[j+1]],dtype=np.int32)
        old_dst=np.array([path[j],path[j+1],b],dtype=np.int32)
        old_m,_=net.values(old_src,old_dst)
        delta=float(new_mean-np.sum(old_m,dtype=np.float64))
        changed=path.copy();changed[j]=u;changed[j+1]=v
        self.consider(ship,changed,delta,temperature)

    def run(self,minutes:float,max_iterations:int|None=None,progress_interval:int=3000):
        t0=time.monotonic()
        deadline=t0+minutes*60
        i=0
        while time.monotonic()<deadline and (max_iterations is None or i<max_iterations):
            i+=1
            # Let diversified search occasionally move uphill, cooling over time.
            frac=min(1,(time.monotonic()-t0)/(minutes*60))
            temp=0.75*(1-frac)**2
            if self.rng.random()<0.33:
                self.try_double(temp)
            else:
                self.try_mutation(temp)
            if i%progress_interval==0:
                print(f'Iteration {i}; best {self.best_official:.6f}; current {self.curr.objective:.6f}; accepted {self.accepted}',flush=True)
                # Re-center search periodically on best valid incumbent.
                if self.curr.objective>self.best_official+1.0:
                    self.paths=to_paths(self.best_x,self.net)
                    self.curr=self.eval.score(self.paths)
        # The artifact to submit is always the last officially validated incumbent.
        fitness=self.official(self.best_x)
        assert self.feasible(fitness),f'Final invalid: {fitness}'
        self._save_best('final_official_validation',fitness)
        print(f'FINAL official fitness={float(fitness[0]):.12f} constraints={list(map(float,fitness[1:]))}; iterations={i}',flush=True)
        return fitness
