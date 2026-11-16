"""Feasibility-first, reproducible hybrid stochastic local search for SpOC2.

Keeps the original challenge data, UDP evaluator and GA implementation intact.
The primary entry point here starts from a *verified feasible* solution, samples
valid two-edge rewires, preserves arrival synchronization, checkpoints each new
incumbent, and verifies every published solution using the official UDP.

Run: python -m solver.optimize --minutes 3 --seed 42
Resume: python -m solver.optimize --resume results/best_solution.npy --minutes 3
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Sequence

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


class SparseNetwork:
    """Compact, vectorized, directed network lookup (IDs are 1-based)."""
    def __init__(self, data_path: Path):
        with np.load(data_path) as z:
            edges = np.asarray(z['edges'], dtype=np.int32)
            mv = np.asarray(z['meanvar'], dtype=np.float32)
            self.origins = np.asarray(z['origins'], dtype=np.int32)
            self.destination = int(z['destination'])
            self.delays = np.asarray(z['delays'], dtype=np.float32)
            self.window = float(z['window'])
            self.jump_limit = int(z['jump_limit'])
        self.n_ships = len(self.delays)
        self.n_nodes = int(edges.max())
        self.base = self.n_nodes + 1
        keys = edges[:,0].astype(np.int64)*self.base + edges[:,1]
        order = np.argsort(keys)
        self.keys = keys[order]
        self.means = mv[order,0]
        self.variances = mv[order,1]
        # Outgoing neighbor lists sorted by node, just views into one array.
        src = self.keys//self.base
        counts = np.bincount(src, minlength=self.n_nodes+1)
        offsets = np.r_[0,counts.cumsum()]
        targets = (self.keys%self.base).astype(np.int32)
        self.out = [targets[offsets[i]:offsets[i+1]] for i in range(self.n_nodes+1)]
        # Incoming lists sorted by source, for quick two-hop rewiring.
        rev = np.lexsort((edges[:,0],edges[:,1]))
        rev_edges = edges[rev]
        inc_counts=np.bincount(rev_edges[:,1],minlength=self.n_nodes+1)
        inc_offsets=np.r_[0,inc_counts.cumsum()]
        from_nodes=rev_edges[:,0]
        self.inc=[from_nodes[inc_offsets[i]:inc_offsets[i+1]] for i in range(self.n_nodes+1)]
        self._bridge_cache:dict[tuple[int,int], np.ndarray]={}

    def values(self, source: np.ndarray, dest: np.ndarray):
        keys = np.asarray(source,dtype=np.int64)*self.base + np.asarray(dest,dtype=np.int64)
        idx = np.searchsorted(self.keys,keys)
        valid = idx < len(self.keys)
        if np.any(valid):
            valid &= self.keys[np.minimum(idx,len(self.keys)-1)]==keys
        if not np.all(valid):
            raise ValueError('Attempted a nonexistent directed edge')
        return self.means[idx],self.variances[idx]

    def bridges(self, a:int,b:int)->np.ndarray:
        key = (a,b)
        found=self._bridge_cache.get(key)
        if found is None:
            found=np.intersect1d(self.out[a],self.inc[b],assume_unique=True)
            # Avoid cache growing without bound in lengthy sessions.
            if len(self._bridge_cache)>90000:
                self._bridge_cache.clear()
            self._bridge_cache[key]=found
        return found


def to_paths(chromosome:np.ndarray, network:SparseNetwork)->list[np.ndarray]:
    arr=np.asarray(chromosome,dtype=np.int32).reshape(network.n_ships,network.jump_limit)
    paths=[]
    for row in arr:
        cut=np.flatnonzero(row==0)
        length=int(cut[0]) if len(cut) else len(row)
        paths.append(row[:length].copy())
    return paths


def to_chromosome(paths:Sequence[np.ndarray],network:SparseNetwork)->np.ndarray:
    x=np.zeros(network.n_ships*network.jump_limit,dtype=np.int32)
    if len(paths)!=network.n_ships:raise ValueError('Expected 12 ship paths')
    for ship,path in enumerate(paths):
        path=np.asarray(path,dtype=np.int32)
        if not 1<=len(path)<=network.jump_limit:raise ValueError('Invalid path length')
        x[ship*network.jump_limit:ship*network.jump_limit+len(path)]=path
    return x


@dataclass
class Score:
    objective:float
    costs:np.ndarray
    arrival:np.ndarray
    window:float


class FastEvaluator:
    """Vectorized proxy for ranking proposals; official UDP verifies outputs."""
    def __init__(self,network:SparseNetwork):self.net=network
    def score(self,paths:Sequence[np.ndarray])->Score:
        net=self.net
        source=[];dest=[];ship=[];jump=[]
        for s,p in enumerate(paths):
            p=np.asarray(p,dtype=np.int32)
            n=len(p)
            source.append(p)
            dest.append(np.r_[p[1:],net.destination].astype(np.int32))
            ship.append(np.full(n,s,dtype=np.int16))
            jump.append(np.arange(n,dtype=np.int16))
        source=np.concatenate(source)
        dest=np.concatenate(dest)
        ship=np.concatenate(ship)
        means,var=net.values(source,dest)
        # Accumulate per ship with float32 arithmetic, matching official data precision.
        arrival=np.zeros(net.n_ships,dtype=np.float32)
        departure=np.empty_like(means)
        for s in range(net.n_ships):
            idx=np.flatnonzero(ship==s)
            t=net.delays[s]
            for j in idx:
                departure[j]=t
                t=np.float32(t+means[j])
            # The UDP uses numpy float32 sum followed by float32 addition of delay.
            arrival[s]=np.float32(np.sum(means[idx],dtype=np.float32)+net.delays[s])
        # Stable tie-break for visits is ship/jump traversal order.
        order=np.lexsort((np.arange(len(source)), departure,source))
        sorted_nodes=source[order]
        boundary=np.r_[True,sorted_nodes[1:]!=sorted_nodes[:-1]]
        inds=np.arange(len(source))
        first=np.maximum.accumulate(np.where(boundary,inds,0))
        ranks=np.minimum(inds-first,10)
        weight=np.power(10.0,ranks)
        rank_weight=np.empty_like(weight)
        rank_weight[order]=weight
        costs=np.bincount(ship,weights=np.asarray(var,dtype=np.float64)*rank_weight,minlength=net.n_ships)*10000.0
        window=float(arrival.max()-arrival.min()-net.window)
        return Score(float(costs.max()),costs,arrival,window)


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


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',type=Path,default=ROOT/'data/database.npz')
    ap.add_argument('--resume',type=Path,default=None,help='Start from existing verified .npy')
    ap.add_argument('--minutes',type=float,default=3)
    ap.add_argument('--iterations',type=int,default=None)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--output-dir',type=Path,default=ROOT/'results')
    args=ap.parse_args()
    opt=Optimizer(args.data,args.output_dir,args.seed)
    if args.resume is not None: initial=args.resume
    elif opt.checkpoint.exists():initial=opt.checkpoint
    else:initial=ROOT/'data/example.npy'
    opt.init(initial)
    opt.run(args.minutes,args.iterations)

if __name__=='__main__':main()
