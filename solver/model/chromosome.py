"""Conversions between 12 variable-length ship paths and the padded chromosome."""
from typing import Sequence
import numpy as np
from solver.data.sparse import SparseNetwork

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
