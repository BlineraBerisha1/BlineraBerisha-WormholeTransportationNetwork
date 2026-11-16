"""Integration and mathematical regression tests against the unchanged ESA UDP."""
from pathlib import Path

import numpy as np
import pytest

from solver.competitive import CSRNetwork, _node_potentials, _build_one
from solver.optimize import ROOT, FastEvaluator, SparseNetwork, to_chromosome, to_paths


@pytest.fixture(scope='module')
def data():
    return ROOT / 'data' / 'database.npz'


@pytest.fixture(scope='module')
def net(data):
    return SparseNetwork(data)


@pytest.fixture(scope='module')
def original_udp(data):
    from wormhole_udp import wormhole_traversal_udp
    return wormhole_traversal_udp(database=str(data))


def test_csr_validated_against_reference_database(data,net):
    csr=CSRNetwork(data)
    with np.load(data) as archive:
        edges=archive['edges']
        mv=archive['meanvar'].astype(np.float32)
    rng=np.random.default_rng(701)
    indices=rng.choice(len(edges),size=200,replace=False)
    sources=edges[indices,0]
    targets=edges[indices,1]
    mean,var=net.values(sources,targets)
    np.testing.assert_array_equal(mean,mv[indices,0])
    np.testing.assert_array_equal(var,mv[indices,1])
    for a,b in zip(sources[:25],targets[:25]):
        assert int(b) in csr.targets[csr.offsets[a]:csr.offsets[a+1]]
    assert len(csr.dest_edges)==10001
    assert csr.jump_limit==500


def test_undamaged_baseline_and_solved_checkpoint(original_udp,net):
    baseline=np.load(ROOT/'data/example.npy',allow_pickle=False)
    solved=np.load(ROOT/'results/best_solution.npy',allow_pickle=False)
    assert baseline.shape==solved.shape==(6000,)
    assert np.issubdtype(solved.dtype,np.integer)
    for x in [baseline,solved]:
        f=original_udp.fitness(x)
        assert f[1]==0 and f[2]==0 and f[3]<=0
    assert float(original_udp.fitness(solved)[0]) < 1175.543385
    paths=to_paths(solved,net)
    np.testing.assert_array_equal(solved,to_chromosome(paths,net))
    for ship,path in enumerate(paths):
        assert 1<=len(path)<=500
        assert path[0] in net.origins[ship]


def test_fast_score_matches_official_reference(original_udp,net):
    x=np.load(ROOT/'results/best_solution.npy',allow_pickle=False)
    paths=to_paths(x,net)
    estimate=FastEvaluator(net).score(paths)
    official=original_udp.fitness(x)
    assert abs(estimate.objective-float(official[0]))<0.02
    assert abs(estimate.window-float(official[3]))<0.0001
    with_dest=[np.append(p,net.destination) for p in paths]
    np.testing.assert_allclose(estimate.costs,
        original_udp._accumulate_variances(with_dest)*10000,rtol=1.e-6,atol=0.02)


def test_reconstruction_produces_directed_feasible_path(data,net):
    graph=CSRNetwork(data)
    occupied=np.zeros(graph.n+1,dtype=np.int32)
    potentials=_node_potentials(graph.offsets,graph.targets,graph.means,
         graph.variances,.06,1.,.80,5)
    # Ship 7 starts within [-0.49, 0.51]; any short terminal path in the
    # interval is valid under the official ship-specific delay.
    row, n, objective,_=_build_one(graph.offsets,graph.targets,graph.means,
         graph.variances,graph.dest_edges,graph.origins[6],graph.delays[6],
         np.float32(-.49),np.float32(.51),.06,.00001,occupied,
         12039,500,potentials,.8,graph.destination,-1)
    assert 1<=n<=500
    assert row[0] in net.origins[6]
    assert np.all(row[n:]==0)
    assert objective>0
    path=row[:n]
    sources=path
    targets=np.r_[path[1:],net.destination]
    m,v=net.values(sources,targets)
    arrival=float(np.float32(np.sum(m,dtype=np.float32)+net.delays[6]))
    assert -.49<=arrival<=.51
    assert np.all(v>0)


def test_cannot_score_invalid_edges(net):
    with pytest.raises(ValueError):
        net.values(np.array([0],np.int32),np.array([0],np.int32))
