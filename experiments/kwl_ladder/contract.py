"""
Tensor-network contraction of the qubit models.

The functor image of a qubit model is contracted through optyx's
standard pipeline — unroll, double, ``to_tensor`` — handed to quimb
with a greedy path (the doubled networks here are shallow and sparse,
so path search dominates fancier optimisers). The readout is one
contraction per cell: that cell's prediction bits over every tick kept
open, every other prediction discarded, which marginalises it — the
per-cell *trajectory distribution*.

The invariant summary of a graph is the multiset of trajectory
distributions over its cells, compared outcome by outcome through the
sorted values across cells: sorting is 1-Lipschitz, so the comparison
is stable and permutation-invariant. ``max_bond`` truncates the
contraction on graphs too entangled to contract exactly; the truncation
is reported, never silent, and the relabelling-invariance check
measures the error scale it introduces.
"""

import numpy as np

from optyx.channel import Diagram, Discard
from optyx.core.backends import QuimbBackend


def backend_for(optimizer, max_bond=None):
    """A quimb backend: ``greedy`` contracts the toy graphs exactly;
    ``compressed`` bounds every intermediate bond by ``max_bond``
    through a compressed hyperoptimizer — the approximation whose error
    scale the relabelling-invariance runs measure; ``hyper`` searches
    for an exact path with kahypar and slicing."""
    if optimizer == "greedy":
        params = {"optimize": "greedy"}
        if max_bond is not None:
            params["max_bond"] = max_bond
        return QuimbBackend(contraction_params=params)
    if optimizer == "compressed":
        from cotengra import ReusableHyperCompressedOptimizer
        hyper = ReusableHyperCompressedOptimizer(
            max_repeats=6, parallel=False)
        return QuimbBackend(
            hyperoptimiser=hyper,
            contraction_params={"max_bond": max_bond or 64})
    from cotengra import ReusableHyperOptimizer
    hyper = ReusableHyperOptimizer(
        max_repeats=12, parallel=False,
        slicing_opts={"target_size": 2 ** 26})
    params = {"max_bond": max_bond} if max_bond is not None else {}
    return QuimbBackend(hyperoptimiser=hyper,
                        contraction_params=params)


def trajectory_distribution(model, graph, n_ticks, cell,
                            optimizer="greedy", max_bond=None,
                            unrolled=None):
    """The distribution of one cell's prediction bits over every tick,
    the other cells' predictions discarded."""
    if unrolled is None:
        unrolled = model.unrolled(graph, n_ticks)
    n = len(graph)
    keep = Diagram.id().tensor(*[
        Diagram.id(ob) if i % n == cell else Discard(ob)
        for i, ob in enumerate(unrolled.cod)])
    backend = backend_for(optimizer, max_bond)
    result = backend.eval(unrolled >> keep)
    return np.asarray(result.tensor.array).real.reshape(-1)


def summary(model, graph, n_ticks, optimizer="greedy", max_bond=None):
    """The multiset of per-cell trajectory distributions, as a matrix
    with one sorted column per outcome."""
    unrolled = model.unrolled(graph, n_ticks)
    rows = np.stack([
        trajectory_distribution(
            model, graph, n_ticks, cell, optimizer, max_bond, unrolled)
        for cell in range(len(graph))])
    return np.sort(rows, axis=0)
