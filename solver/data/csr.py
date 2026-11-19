"""Compressed-sparse-row (CSR) view of the authentic ESA graph.

Used for fast directed-edge access during route construction.
"""
from pathlib import Path
import numpy as np

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
