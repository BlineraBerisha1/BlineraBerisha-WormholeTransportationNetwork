"""Fast candidate-ranking evaluator; the original ESA UDP is the final authority."""
from dataclasses import dataclass
from typing import Sequence
import numpy as np
from solver.data.sparse import SparseNetwork

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
