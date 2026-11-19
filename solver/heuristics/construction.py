"""Candidate ship routes, arrival-window targeting, graph lookahead and occupancy.

These methods are part of the actual competitive optimization algorithm.
"""
import numpy as np
from solver.heuristics.route_builder import _build_one, _node_potentials


class RouteConstruction:
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
