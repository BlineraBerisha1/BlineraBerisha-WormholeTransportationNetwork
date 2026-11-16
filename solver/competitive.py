"""Variable-length, synchronized route construction and adaptive large-neighborhood search.

The official, unmodified ``wormhole_udp.py`` remains the only source of truth.
This module uses a CSR graph and Numba accelerated stochastic Lagrangian
route builders. It verifies and atomically checkpoints every improvement.

Run ``python -m solver.competitive --minutes 10`` or use ``--minutes 0`` to
validate existing output. Resume is automatic and never overwrites a better
feasible incumbent.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
from numba import njit

from solver.optimize import ROOT, SparseNetwork, FastEvaluator, to_paths, to_chromosome


@njit(cache=True)
def _build_one(offsets, neighbors, means, variances, dest_edges, origins,
               delay, lo, hi, lagrange, noise, occupancy, rng_seed, jump_limit, potential, discount, destination,
               preferred_start=-1):
    """Generate one directed path with an arrival-time resource constraint.

    The returned row lists *source* nodes; the terminal destination is implicit.
    Feasibility is checked again with the unmodified UDP before recording.
    """
    np.random.seed(rng_seed)
    n_nodes = len(offsets) - 2
    visited = np.zeros(n_nodes + 1, np.uint8)
    row = np.zeros(jump_limit, np.int32)
    # Pick a random permitted origin. Trying different starts is part of ALNS.
    origin = int(origins[np.random.randint(len(origins))]) if preferred_start <= 0 else preferred_start
    node = origin
    t = np.float32(delay)
    base_variance = 0.
    crowd_cost = 0.
    needed = 0.
    if t < lo: needed = lo - t
    elif t > hi: needed = hi - t
    direction = 1. if needed >= 0 else -1.
    visited[node] = 1
    for k in range(jump_limit):
        # First, try to terminate at the destination by one or two edges.
        # The 2-edge exit avoids requiring a direct edge to the destination.
        best_end = 1e15
        last_n = -1
        e = dest_edges[node]
        if e >= 0:
            arrival = t + means[e]
            if lo <= arrival <= hi:
                best_end = float(variances[e]) + 0.0025 * occupancy[node]
                last_n = 0
        if k + 1 < jump_limit:
            for j in range(offsets[node], offsets[node + 1]):
                nxt = neighbors[j]
                if nxt == node or visited[nxt] != 0:
                    continue
                end_j = dest_edges[nxt]
                if end_j < 0:
                    continue
                arrival = t + means[j] + means[end_j]
                if arrival < lo or arrival > hi:
                    continue
                cost = (float(variances[j]) + float(variances[end_j])
                        + 0.0025 * (occupancy[nxt] + occupancy[node]))
                if cost < best_end:
                    best_end, last_n = cost, int(nxt)
        if last_n >= 0:
            row[k] = node
            base_variance += best_end
            if last_n != 0:
                row[k + 1] = last_n
                return row, k + 2, base_variance + crowd_cost, np.float32(t)
            return row, k + 1, base_variance + crowd_cost, np.float32(t)

        if k + 2 >= jump_limit:
            return row, 0, 1e10, t

        remain = (lo - t) if direction > 0 else (t - hi)
        lam = lagrange
        if remain > 0:
            req = remain / max(1, jump_limit - k - 2)
            if req > 0.0008: lam += (req - .0008) * 250.
            if req > 0.00155: lam += (req - .00155) * 1300.

        start = offsets[node]
        end = offsets[node + 1]
        min_score = 1e15
        next_node = -1
        chosen = -1
        for j in range(start, end):
            nxt = neighbors[j]
            if nxt == 0 or nxt == node or nxt == destination:
                continue
            if visited[nxt] != 0:
                continue
            # Favor low-variance wormholes; Lagrange price encourages required
            # travel-time sign. Penalize shared nodes to limit revisit ranks.
            score = (float(variances[j]) - lam * direction * float(means[j])
                     + discount * float(potential[nxt]) + 0.002 * occupancy[nxt])
            score += noise * (np.random.random() - 0.5)
            if score < min_score:
                min_score = score
                next_node = int(nxt)
                chosen = j
        if chosen < 0:
            return row, 0, 1e10, t
        row[k] = node
        visited[next_node] = 1
        base_variance += float(variances[chosen])
        crowd_cost += 0.002 * occupancy[node]
        t = np.float32(t + means[chosen])
        node = next_node
    return row, 0, 1e10, t


@njit(cache=True)
def _node_potentials(offsets,targets,means,variances,lam,direction,
                     discount=0.80,iterations=16):
    """Discounted Lagrangian lookahead on all 1,998,600 directed edges.

    Values approximate future low-variance accessibility. Repeated-visit
    avoidance remains enforced in the route constructor.
    """
    n=len(offsets)-2
    old=np.zeros(n+1,dtype=np.float32)
    for it in range(iterations):
        fresh=np.zeros(n+1,dtype=np.float32)
        for src in range(1,n+1):
            best=1.e10
            for j in range(offsets[src],offsets[src+1]):
                val=variances[j]-lam*direction*means[j]+discount*old[targets[j]]
                if val<best:best=val
            fresh[src]=best
        old=fresh
    return old


class CSRNetwork:
    def __init__(self, file: Path):
        with np.load(file) as data:
            edges = np.asarray(data['edges'], dtype=np.int32)
            weights = np.asarray(data['meanvar'], dtype=np.float32)
            self.origins = np.asarray(data['origins'], dtype=np.int32)
            self.destination = int(data['destination'])
            self.delays = np.asarray(data['delays'], dtype=np.float32)
            self.window = float(data['window'])
            self.jump_limit = int(data['jump_limit'])
        self.n = max(int(edges.max()), self.destination)
        order = np.lexsort((edges[:, 1], edges[:, 0]))
        edges = edges[order]
        weights = weights[order]
        self.offsets = np.r_[0, np.bincount(edges[:,0],minlength=self.n+1).cumsum()].astype(np.int64)
        self.targets = np.ascontiguousarray(edges[:,1],dtype=np.int32)
        self.means = np.ascontiguousarray(weights[:,0],dtype=np.float32)
        self.variances = np.ascontiguousarray(weights[:,1],dtype=np.float32)
        self.dest_edges = np.full(self.n+1,-1,dtype=np.int32)
        for node in range(1,self.n+1):
            left,right=self.offsets[node],self.offsets[node+1]
            j=np.searchsorted(self.targets[left:right],self.destination)
            if j < right-left and self.targets[left+j]==self.destination:
                self.dest_edges[node]=left+j


class CompetitiveSolver:
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
        from verify import verify
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
        from solver.output.formatter import SolutionFormatter
        SolutionFormatter().create_submission_json(x.astype(np.int32),str(self.output/'submission.json'),
             submission_name=f'Officially validated solution {self.best:.8f}',
             submission_description='All official SpOC2 constraints satisfied')
        (self.output/'fitness_summary.txt').write_text(f'Official fitness: {self.best:.12f}\n'
               f'Origin violations: {official[1]}\nPath violations: {official[2]}\n'
               f'Window violation: {official[3]}\n')

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

    def build(self,ship,lo,hi,occupancy,trials=25,origin=None,endpoint=None,max_steps=None):
        g=self.graph
        best_row=None;best_cost=np.inf
        endpoint=g.destination if endpoint is None else int(endpoint)
        limit=g.jump_limit if max_steps is None else int(max_steps)
        ending=self.dest_edges(endpoint)
        origins=g.origins[ship] if origin is None else np.array([origin],np.int32)
        # Sample distinct Lagrange prices and noise levels: high prices make
        # faster routes; low prices explore longer, lower-variance routes.
        for trial in range(trials):
            required=abs(float(g.delays[ship])-(lo if g.delays[ship]<lo else hi))
            if required>0.30:
                lam=float(self.rng.choice([.035,.043,.05,.057,.065,.073,.083,.095,.11]))
            elif required>0.08:
                lam=float(self.rng.choice([.027,.04,.05,.065,.08,.11,.16]))
            else:
                lam=float(self.rng.choice([.02,.05,.08,.14,.25]))
            direction=1 if float(g.delays[ship])<lo else -1
            # Discounted graph lookahead is cached by lagrange parameter and sign.
            key=(lam,direction,self.discount)
            if key not in self.potential_cache:
                self.potential_cache[key]=_node_potentials(g.offsets,g.targets,g.means,g.variances,
                                    lam,float(direction),self.discount,20)
            potential=self.potential_cache[key]
            noise=float(self.rng.choice([0.,.00001,.000025,.00005,.000085,.00013]))
            random_seed=int(self.rng.integers(1,2**31-1))
            row,n,cost,_=_build_one(g.offsets,g.targets,g.means,g.variances,g.dest_edges,
                      origins,g.delays[ship] if origin is None else np.float32(0),np.float32(lo),np.float32(hi),
                      lam,noise,occupancy,random_seed,limit,potential,self.discount,endpoint,
                      -1 if origin is None else int(origin))
            if n>0 and cost < best_cost:
                best_cost=cost;best_row=row.copy()
        return best_row,best_cost

    @staticmethod
    def occupancy(x,exclude=-1):
        arr=x.reshape(12,500)
        occur=np.zeros(10001,dtype=np.int32)
        for s in range(12):
            if s==exclude:continue
            for n in arr[s]:
                if n>0:occur[n]+=1
        return occur

    def dest_edges(self, endpoint:int):
        if endpoint==self.graph.destination:
            return self.graph.dest_edges
        if endpoint not in self.end_edges_cache:
            edges=np.full(self.graph.n+1,-1,dtype=np.int32)
            for u in self.sparse.inc[endpoint]:
                node=int(u)
                a,b=self.graph.offsets[node],self.graph.offsets[node+1]
                idx=np.searchsorted(self.graph.targets[a:b],endpoint)
                if idx<b-a and self.graph.targets[a+idx]==endpoint:
                    edges[node]=a+idx
            if len(self.end_edges_cache)>2500:self.end_edges_cache.clear()
            self.end_edges_cache[endpoint]=edges
        return self.end_edges_cache[endpoint]

    def reroute_segment(self,ship,seg_start=None,seg_length=None,trials=15):
        g=self.graph
        paths=to_paths(self.best_x,self.sparse)
        p=paths[ship]
        k=len(p)
        if k<30:return False
        seg_length=int(self.rng.integers(18,min(140,k-2))) if seg_length is None else min(int(seg_length),k-1)
        seg_start=int(self.rng.integers(0,k-seg_length)) if seg_start is None else int(seg_start)
        a=seg_start
        b=seg_start+seg_length
        endpoint=int(p[b])
        old_sources=p[a:b]
        old_targets=np.r_[p[a+1:b],endpoint].astype(np.int32)
        old_means,_=self.sparse.values(old_sources,old_targets)
        old_duration=float(np.sum(old_means,dtype=np.float32))
        score=self.proxy.score(paths)
        others=np.delete(score.arrival,ship)
        lower=float(others.max()-g.window+.002)
        upper=float(others.min()+g.window-.002)
        lo=float(old_duration+lower-score.arrival[ship])
        hi=float(old_duration+upper-score.arrival[ship])
        if lo>=hi:return False
        occ=self.occupancy(self.best_x,exclude=ship)
        for n in p[:a]:occ[n]+=1
        for n in p[b:]:occ[n]+=1
        max_seg=min(k-1-a, min(g.jump_limit-k+seg_length,seg_length+80))
        max_seg=min(g.jump_limit-k+seg_length,seg_length+80)
        if max_seg<3:return False
        row,_=self.build(ship,lo,hi,occ,trials=trials,origin=int(p[a]),
                         endpoint=endpoint,max_steps=max_seg)
        if row is None:return False
        fragment=row[row!=0]
        revised=np.r_[p[:a],fragment,p[b:]]
        if len(revised)>g.jump_limit:return False
        newpaths=list(paths)
        newpaths[ship]=revised
        return self.evaluate(to_chromosome(newpaths,self.sparse),'ALNS: variable-length segment rebuild')

    def new_joint(self,lo,trials=40):
        hi=lo+self.graph.window
        occupancy=np.zeros(self.graph.n+1,dtype=np.int32)
        rows=np.zeros((len(self.graph.delays),self.graph.jump_limit),dtype=np.int32)
        # Extreme ships first: they are harder to synchronize and dominate cost.
        order=np.argsort(-abs(self.graph.delays-(lo+hi)/2))
        for s in order:
            allocated=(trials*12 if s in [0,11] else trials*4 if s in [1,10] else trials)
            row,_=self.build(int(s),lo+.002,hi-.002,occupancy,trials=allocated)
            if row is None:return None
            rows[s]=row
            for n in row:
                if n>0: occupancy[n]+=1
        return rows.reshape(-1)

    def rebuild(self,ship,lo,hi,trials):
        occupancy=self.occupancy(self.best_x,exclude=ship)
        row,_=self.build(ship,lo,hi,occupancy,trials=trials)
        if row is None:return False
        x=self.best_x.copy()
        x.reshape(12,500)[ship]=row
        return self.evaluate(x,'ALNS: whole-route bottleneck rebuild')

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


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=ROOT/'data/database.npz')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'results')
    parser.add_argument('--resume',type=Path,default=None)
    parser.add_argument('--minutes',type=float,default=10.)
    parser.add_argument('--seed',type=int,default=20261115)
    parser.add_argument('--construction-trials',type=int,default=40)
    parser.add_argument('--windows',type=int,default=12)
    parser.add_argument('--alns-trials',type=int,default=30)
    parser.add_argument('--discount',type=float,default=.80,help='Discount factor for Lagrangian graph lookahead, 0..1')
    args=parser.parse_args()
    opt=CompetitiveSolver(args.data,args.output_dir,args.seed)
    opt.discount=args.discount
    opt.init([args.resume] if args.resume is not None else [])
    opt.run(args.minutes,args.construction_trials,args.windows,args.alns_trials)

if __name__=='__main__':main()
