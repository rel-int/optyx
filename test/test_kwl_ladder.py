"""Smoke tests for the k-WL ladder benchmark of
``experiments/kwl_ladder``: the WL hierarchy on classic pairs, the five
functors on a triangle, and the exact photonic certificates against
optyx's own semantics — all at toy sizes."""

import os
import sys

import networkx as nx
import numpy as np
import pytest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "experiments", "kwl_ladder"))

from dataset import fwl2, fwl3, level, wl1  # noqa: E402
from models import MODELS, adjacency  # noqa: E402
import exact  # noqa: E402


def test_wl_hierarchy_on_classic_pairs():
    c6, p6 = nx.cycle_graph(6), nx.path_graph(6)
    two_triangles = nx.disjoint_union(nx.cycle_graph(3),
                                      nx.cycle_graph(3))
    assert level(c6, p6) == 1
    assert level(two_triangles, c6) == 2
    assert wl1(two_triangles) == wl1(c6)
    assert fwl2(two_triangles) != fwl2(c6)
    relabelled = nx.relabel_nodes(c6, {i: (i * 5) % 6 for i in range(6)})
    assert wl1(c6) == wl1(relabelled)
    assert fwl2(c6) == fwl2(relabelled)
    assert fwl3(c6) == fwl3(relabelled)


def test_functors_on_a_triangle():
    triangle = adjacency(nx.cycle_graph(3))
    for model in MODELS.values():
        cmap = model(triangle)
        assert len(cmap.prediction) == 3
        assert cmap.boundary or model.name.startswith("qubit")


def test_passive_assembly_matches_cmap_step():
    from exact import PassiveLayout, assemble
    from models import PassiveModel
    graph = adjacency(nx.path_graph(3))
    model = PassiveModel()
    cmap = model(graph)
    step = np.asarray(cmap.step.to_path().array, dtype=complex)[
        :len(cmap.dom) + len(cmap.memory), :]
    ours = assemble(graph, [model.tap] * 3, PassiveLayout(model))
    assert np.abs(ours - step).max() < 1e-12


def test_certificates_conserve_probability():
    graph = adjacency(nx.cycle_graph(3))
    for name in ("passive", "bell", "active"):
        machine = exact.machine_for(MODELS[name], graph)
        bins = exact.aggregate(
            machine, 3,
            engine="branch" if name == "active" else "spacetime")
        assert abs(sum(bins.values()) - 1) < 1e-9


def test_spacetime_engine_matches_branching_engine():
    graph = adjacency(nx.path_graph(3))
    machine = exact.machine_for(MODELS["passive"], graph)
    slow = exact.run_pair(machine, 0, 2, 2)
    fast = exact.passive_pair_bins(
        machine, exact.spacetime_transfer(machine, 2), 2, 0, 2)
    keys = set(slow) | set(fast)
    assert max(abs(slow.get(k, 0.0) - fast.get(k, 0.0))
               for k in keys) < 1e-10


@pytest.mark.parametrize("name", ["qubit"])
def test_qubit_model_is_a_channel(name):
    import contract
    graph = adjacency(nx.path_graph(2))
    distribution = contract.trajectory_distribution(
        MODELS[name], graph, 2, cell=0)
    assert abs(distribution.sum() - 1) < 1e-9
    assert distribution.min() > -1e-12
