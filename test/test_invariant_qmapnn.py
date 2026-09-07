"""Fast tests for the invariant QMapNN experiment of
``experiments/invariant_qmapnn``: the Schur lemma on the cell, the
assembled step matrix against ``CMap.step.to_path()``, the two-photon
certificate against optyx's contraction of the driven functor image on
a triangle, and the rotation-system, relabelling and probability
validations on a path — all at toy sizes."""

import os
import sys

import networkx as nx
import numpy as np
import pytest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "experiments", "invariant_qmapnn"))

from cells import (  # noqa: E402
    CELLS, dart_permutation, haar_unitary, invariant_cell, rotate)
from engine import assemble, bins, joint, statistics  # noqa: E402
from ladder import adjacency, relabelled, separation  # noqa: E402


def score(cell, left, right, n_ticks=3, certificate="two-photon"):
    one, two = (bins(*statistics(cell(graph), n_ticks, certificate)[:2])
                for graph in (left, right))
    return separation(one, two)


@pytest.mark.parametrize("degree", [1, 2, 3, 5])
def test_cell_is_unitary_and_commutes_with_dart_permutations(degree):
    rng = np.random.default_rng(degree)
    unitary = invariant_cell(degree, haar_unitary(3, rng), rng.uniform())
    assert np.abs(unitary.conj().T @ unitary - np.eye(degree + 2)).max() \
        < 1e-12
    for _ in range(50):
        permutation = dart_permutation(degree, rng.permutation(degree))
        assert np.abs(permutation.T @ unitary @ permutation
                      - unitary).max() <= 1e-12


def test_assembled_step_matches_the_functor_image():
    for graph in (adjacency(nx.path_graph(3)),
                  rotate(adjacency(nx.cycle_graph(4)), 3)):
        cmap = CELLS["canonical"](graph)
        step = np.asarray(cmap.step.to_path().array, dtype=complex)
        assert np.abs(assemble(cmap) - step).max() < 1e-12


def test_two_photon_certificate_matches_the_contraction():
    from optyx.channel import Diagram, Discard, Measure, qmode
    from optyx.core.backends import DiscopyBackend
    from optyx.photonic import Create
    n_ticks, slots = 2, ((0, 0), (1, 2))
    cmap = CELLS["canonical"](adjacency(nx.cycle_graph(3)))
    n = len(cmap.dom)
    photons = [[0] * n for _ in range(n_ticks)]
    for tick, vertex in slots:
        photons[tick][vertex] += 1
    drive = Diagram.id().tensor(
        *[Create(k) for tick in photons for k in tick])
    memory = Diagram.id().tensor(*[Create(0)] * len(cmap.memory))
    readout = Diagram.id().tensor(
        *[Discard(qmode)] * (n_ticks - 1) * n) @ Measure(qmode ** n)
    tensor = DiscopyBackend().eval(
        drive @ memory >> cmap.unroll(n_ticks - 1) >> readout).tensor
    dims = tuple(int(dim) for dim in tensor.cod.inside)
    contracted = np.asarray(tensor.array).real.reshape(dims)
    certificate = joint(cmap, n_ticks, *slots)
    assert abs(contracted.sum() - 1) < 1e-12
    assert np.abs(
        contracted - certificate[tuple(map(slice, dims))]).max() < 1e-14


def test_rotation_systems_and_relabelling_on_a_path():
    path = adjacency(nx.path_graph(4))
    for name in ("canonical", "ladder"):
        cell = CELLS[name]
        assert score(cell, path, relabelled(path)) <= 1e-14
        assert max(score(cell, path, rotate(path, seed))
                   for seed in (1, 2)) <= 1e-14
    assert max(score(CELLS["generic"], path, rotate(path, seed))
               for seed in (1, 2)) > 1e-6


def test_certificates_conserve_probability():
    cmap = CELLS["canonical"](adjacency(nx.path_graph(4)))
    for certificate in ("two-photon", "one-photon"):
        nodes, total, mass = statistics(cmap, 3, certificate)
        assert abs(mass - 1) < 1e-12
        assert np.allclose(nodes.sum(axis=1), 1, atol=1e-12)
