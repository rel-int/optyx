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


def test_the_coupler_is_a_port_symmetric_unitary():
    unitary = b.coupler(3)
    assert np.allclose(unitary @ unitary.conj().T, np.eye(5))
    swap_ports = np.eye(5)[[0, 2, 1, 3, 4]]
    assert np.allclose(swap_ports @ unitary @ swap_ports, unitary)


def test_the_step_conserves_probability():
    graph = b.PAIRS["2C3 vs C6"][0]
    amplitudes = b.step_amplitudes(b.graph_cmap(graph))
    assert np.allclose((np.abs(amplitudes) ** 2).sum(axis=1), 1)
    matrix = b.transfer(amplitudes, len(graph), 0, 30)
    captured = (np.abs(matrix) ** 2).sum(axis=0)
    assert (captured <= 1 + 1e-9).all()
    assert (captured[:15] > 0.9).all()


def test_the_profile_is_relabel_invariant():
    graph = b.PAIRS["decalin vs bicyclopentyl"][0]
    relabel = [(len(graph) - 1 - v) for v in range(len(graph))]
    relabelled = tuple(
        tuple(sorted(relabel[u] for u in graph[relabel[v]]))
        for v in range(len(graph)))
    assert np.allclose(b.profile(graph, 6), b.profile(relabelled, 6))


def test_the_driven_map_separates_beyond_1wl():
    assert b.separation(*b.PAIRS["2C3 vs C6"], 8) > 1e-3
    assert b.separation(*b.CONTROL, 8) > 1e-3


def test_the_decohered_ablation_is_blind_but_not_broken():
    assert b.separation(*b.PAIRS["2C3 vs C6"], 8, decohered=True) < 1e-12
    assert b.separation(*b.CONTROL, 8, decohered=True) > 1e-4


def test_one_box_is_a_stateful_channel():
    box = b.vertex_box(2)
    channel = b.stateful_channel(box)
    assert channel.dom == box.ports
    assert channel.cod == box.ports @ box.prediction


def test_unrolling_matches_the_transfer_matrix():
    from optyx.channel import Diagram, Measure
    from optyx.photonic import Create
    graph, source, n_ticks = b.path(2), 0, 1
    cmap = b.graph_cmap(graph)
    drives = Diagram.id().tensor(*[
        Create(1 if wire == source else 0) for wire in range(len(graph))])
    memory = Diagram.id().tensor(*[
        Create(0) for _ in range(len(cmap.memory))])
    unrolled = drives @ drives @ memory >> cmap.unroll(n_ticks)
    probabilities = unrolled >> Measure(unrolled.cod)
    tensor = probabilities.eval().tensor
    joint = np.asarray(tensor.array).real.reshape(
        tuple(int(dim) for dim in tensor.cod.inside))
    counts = [np.arange(dim) for dim in joint.shape]
    matrix = b.transfer(
        b.step_amplitudes(cmap), len(graph), source, n_ticks + 1)
    mean = (np.abs(matrix) ** 2).sum(axis=1)
    green = matrix @ matrix.conj().T
    squared = np.abs(matrix) ** 2
    for wire in range(len(unrolled.cod)):
        axes = tuple(a for a in range(len(unrolled.cod)) if a != wire)
        assert np.isclose(joint.sum(axes) @ counts[wire], mean[wire])
    for wire, other in [(0, 5), (2, 3), (1, 6)]:
        axes = tuple(a for a in range(len(unrolled.cod))
                     if a not in (wire, other))
        marginal = joint.sum(axes)
        correlation = np.einsum(
            "ij,i,j->", marginal, counts[wire], counts[other])
        coincidence = (
            mean[wire] * mean[other]
            + np.abs(green[wire, other]) ** 2
            - 2 * (squared[wire] * squared[other]).sum())
        assert np.isclose(correlation, coincidence, atol=1e-8)


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
