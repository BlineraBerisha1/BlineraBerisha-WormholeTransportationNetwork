"""Lightweight tests for the included feasible solution and serialization."""
from pathlib import Path
import numpy as np
from solver.optimize import ROOT, SparseNetwork, FastEvaluator, to_paths, to_chromosome


def test_example_round_trip():
    net = SparseNetwork(ROOT/'data/database.npz')
    x=np.load(ROOT/'data/example.npy')
    assert x.dtype == np.int32 and x.shape == (6000,)
    assert np.array_equal(x,to_chromosome(to_paths(x,net),net))


def test_fast_evaluator_matches_example():
    net=SparseNetwork(ROOT/'data/database.npz')
    score=FastEvaluator(net).score(to_paths(np.load(ROOT/'data/example.npy'),net))
    assert abs(score.objective-1243.8905117232935)<0.001
    assert score.window<=0


def test_included_checkpoint_is_feasible():
    """Test with the official evaluator; dataset loading is the expensive part."""
    from wormhole_udp import wormhole_traversal_udp
    x=np.load(ROOT/'results/best_solution.npy')
    f=wormhole_traversal_udp(database=str(ROOT/'data/database.npz')).fitness(x)
    assert int(f[1])==int(f[2])==0
    assert float(f[3])<=0
