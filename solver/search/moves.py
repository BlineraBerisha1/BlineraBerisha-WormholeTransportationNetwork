"""ALNS neighborhood moves: segment rerouting and full-route rebuilding."""
import numpy as np
from solver.model.chromosome import to_paths, to_chromosome


class NeighborhoodMoves:
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
    def rebuild(self,ship,lo,hi,trials):
        occupancy=self.occupancy(self.best_x,exclude=ship)
        row,_=self.build(ship,lo,hi,occupancy,trials=trials)
        if row is None:return False
        x=self.best_x.copy()
        x.reshape(12,500)[ship]=row
        return self.evaluate(x,'ALNS: whole-route bottleneck rebuild')
