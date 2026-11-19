"""Compact directed-edge lookup and two-hop bridge queries."""
from pathlib import Path
import numpy as np

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
