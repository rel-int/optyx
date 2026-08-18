"""Tests for the proposal B experiment, examples/beyond_1wl.py."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(
    os.path.dirname(__file__), "..", "examples"))

import beyond_1wl as b  # noqa: E402


def as_networkx(graph):
    import networkx as nx
    return nx.Graph(
        [(u, v) for u, nbrs in enumerate(graph) for v in nbrs])


def test_pairs_are_1wl_equivalent_but_not_isomorphic():
    import networkx as nx
    for left, right in b.PAIRS.values():
        assert b.wl_equivalent(left, right)
        assert not nx.is_isomorphic(as_networkx(left), as_networkx(right))


def test_the_control_pair_is_distinguished_by_1wl():
    assert not b.wl_equivalent(*b.CONTROL)


def test_star_coupler_is_a_port_symmetric_unitary():
    unitary = b.star_coupler(3)
    assert np.allclose(unitary @ unitary.conj().T, np.eye(4))
    swap_ports = np.eye(4)[[1, 0, 2, 3]]
    assert np.allclose(swap_ports @ unitary @ swap_ports, unitary)


def test_the_step_conserves_probability():
    graph = b.PAIRS["2C3 vs C6"][0]
    leak, feedback = b.step_blocks(b.graph_cmap(graph))
    norms = (np.abs(leak) ** 2).sum(1) + (np.abs(feedback) ** 2).sum(1)
    assert np.allclose(norms, 1)
    curves = b.escape_curves(graph, 100)
    assert ((curves >= 0) & (curves.cumsum(1) <= 1 + 1e-8)).all()
    assert curves.sum(1).min() > 0.99


def test_escape_curves_are_relabel_invariant():
    graph = b.PAIRS["decalin vs bicyclopentyl"][0]
    relabel = [(len(graph) - 1 - v) for v in range(len(graph))]
    relabelled = tuple(
        tuple(sorted(relabel[u] for u in graph[relabel[v]]))
        for v in range(len(graph)))
    assert np.allclose(
        b.escape_curves(graph, 8), b.escape_curves(relabelled, 8))


def test_the_photonic_map_separates_beyond_1wl():
    left, right = b.PAIRS["2C3 vs C6"]
    curves = b.escape_curves(left, 8), b.escape_curves(right, 8)
    assert np.allclose(curves[0][:, :2], curves[1][:, :2])
    assert np.abs(curves[0] - curves[1])[:, 2:].max() > 1e-3
    assert b.separation(*b.CONTROL, 8) > 1e-3


def test_the_decohered_ablation_is_blind_but_not_broken():
    assert b.separation(*b.PAIRS["2C3 vs C6"], 12, decohered=True) < 1e-12
    assert b.separation(*b.CONTROL, 12, decohered=True) > 1e-4


def test_unrolling_matches_the_path_iteration():
    from optyx.channel import Diagram, Measure
    from optyx.photonic import Create
    graph = b.path(2)
    cmap = b.graph_cmap(graph)
    n_ticks, source = 1, len(cmap.paired)
    state = Diagram.id().tensor(*[
        Create(1 if wire == source else 0)
        for wire in range(len(cmap.memory))])
    unrolled = state >> cmap.unroll(n_ticks)
    probabilities = unrolled >> Measure(unrolled.cod)
    joint = np.asarray(probabilities.eval().tensor.array).real.reshape(
        [2] * len(unrolled.cod))
    n_prediction = len(cmap.prediction)
    contracted = []
    for tick in range(n_ticks + 1):
        axes = range(tick * n_prediction, (tick + 1) * n_prediction)
        marginal = joint.sum(axis=tuple(
            axis for axis in range(len(unrolled.cod))
            if axis not in axes))
        contracted.append(sum(
            marginal[index] for index in np.ndindex(*marginal.shape)
            if sum(index) == 1))
    iterated = b.escape_curves(graph, n_ticks + 1)[-1]
    assert np.allclose(contracted, iterated, atol=1e-8)


def test_classical_embeddings_are_bounded_by_1wl():
    pytest.importorskip("torch")
    pair = b.PAIRS["2C3 vs C6"]
    assert b.embedding_separation(
        b.vanilla_gnn_embedding, *pair, n_seeds=2) < 1e-12
    assert b.embedding_separation(
        b.vanilla_gnn_embedding, *b.CONTROL, n_seeds=2) > 1e-3


def test_mapnn_embeddings_are_bounded_by_1wl():
    pytest.importorskip("torch")
    pytest.importorskip("discopy.neural")
    pair = b.PAIRS["2C3 vs C6"]
    assert b.embedding_separation(
        b.mapnn_embedding, *pair, n_seeds=2) < 1e-12
    assert b.embedding_separation(
        b.mapnn_embedding, *b.CONTROL, n_seeds=2) > 1e-3
