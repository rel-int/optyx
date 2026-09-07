"""
The permutation-invariant linear-optical cell, its contrasts, and the
functor from a graph with a rotation system to an
:class:`optyx.interaction.CMap`.

A cell of degree ``d`` is a passive unitary on ``d + 2`` modes. Its
inputs are the drive, the ``d`` in-darts and the memory; its outputs
are the output mode, the ``d`` out-darts and the memory. The symmetric
group ``S_d`` acts on the darts, simultaneously on the in- and
out-darts, and trivially on the rest. By Schur, the permutation
representation on the darts is the trivial representation plus the
standard one, each with multiplicity one, so the unitaries commuting
with the action are exactly

    ``U(W, psi) = B^dagger . [W (+) exp(i psi) I_{d-1}] . B``

for a fixed balanced multiport ``B`` sending the symmetric vector
``(1, ..., 1) / sqrt(d)`` to the first dart, ``W`` in ``U(3)`` acting
on the symmetric dart, the memory and the drive, and one phase ``psi``
on the ``d - 1`` non-symmetric darts: :func:`invariant_cell`.

Every cell of a graph shares the same ``(W, psi)`` whatever its degree,
so the network is a graph invariant by construction; its output does
not depend on the rotation system, i.e. on the order in which each
vertex lists its incident edges. Two contrast cells measure that claim:
:class:`LadderCell`, the interferometer of ``experiments/kwl_ladder``
without its tap, which is port-symmetric too but with degree-dependent
parameters, and :class:`GenericCell`, a Haar-random unitary per degree
which is not.
"""

import numpy as np

from optyx.channel import qmode
from optyx.interaction import Box
from optyx.photonic import Create, Gate

from ladder import Model, coupler


def haar_unitary(n, rng):
    """A Haar-random unitary of ``U(n)`` (Mezzadri, arXiv:math-ph/0609050)."""
    gaussian = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    q, r = np.linalg.qr(gaussian / 2 ** .5)
    return q * (np.diag(r) / np.abs(np.diag(r)))


def householder(degree):
    """The real reflection of ``degree`` modes sending the symmetric
    vector to the first mode, the identity when there is one mode."""
    symmetric = np.full(degree, degree ** -.5)
    normal = symmetric - np.eye(degree)[0]
    if normal @ normal < 1e-24:
        return np.eye(degree)
    return np.eye(degree) - 2 * np.outer(normal, normal) / (normal @ normal)


def invariant_cell(degree, W, psi):
    """The ``S_d``-invariant unitary on modes ``(darts, memory, drive)``."""
    modes = degree + 2
    balanced = np.eye(modes, dtype=complex)
    balanced[:degree, :degree] = householder(degree)
    middle = np.exp(1j * psi) * np.eye(modes, dtype=complex)
    trivial = [0, degree, degree + 1]
    middle[np.ix_(trivial, trivial)] = W
    return balanced.conj().T @ middle @ balanced


def dart_permutation(degree, permutation):
    """A permutation of the darts, the identity on memory and drive."""
    matrix = np.eye(degree + 2)
    matrix[:degree, :degree] = np.eye(degree)[list(permutation)]
    return matrix


def rotate(graph, seed):
    """The same graph under another rotation system: every vertex lists
    its neighbours in a shuffled order, so its darts change ports."""
    rng = np.random.default_rng(seed)
    return tuple(tuple(int(u) for u in rng.permutation(nbrs))
                 for nbrs in graph)


class Cell(Model):
    """A functor from graphs with a rotation system to maps of passive
    cells: the box of a vertex reads the drive, the in-darts and the
    memory and writes the output, the out-darts and the memory, and an
    edge pairs the ports of its two endpoints, so each endpoint's
    out-dart is the other's in-dart at the next tick. An isolated
    vertex gets ``dangling`` darts left open: read from the
    environment, i.e. the vacuum, and written back to it, i.e.
    discarded."""

    dangling = 0

    def unitary(self, degree):
        """The cell on modes ``(drive, darts, memory)``."""
        raise NotImplementedError

    def port(self, position):
        return 1 + position

    def drive(self, degree):
        return Create(1)

    def vertex(self, degree):
        darts = degree or self.dangling
        modes = darts + 2
        name = f"{self.name}{darts}"
        gate = Gate(self.unitary(darts), modes, modes, name)
        return Box(name, qmode, qmode ** darts, gate, memory=qmode)


class InvariantCell(Cell):
    """The invariant cell with ``(W, psi)`` shared by every degree. An
    isolated vertex has no dart to reflect, so it gets the degree-one
    cell with its dart open, the ``d -> 0`` limit of the ansatz."""

    dangling = 1

    def __init__(self, W, psi, name):
        self.W, self.psi, self.name = np.asarray(W, dtype=complex), psi, name

    def unitary(self, degree):
        order = [degree + 1] + list(range(degree)) + [degree]
        return invariant_cell(degree, self.W, self.psi)[np.ix_(order, order)]


class LadderCell(Cell):
    """The port-symmetric interferometer of ``experiments/kwl_ladder``
    without its tap: invariant, with degree-dependent parameters."""

    name = "ladder"

    def unitary(self, degree):
        return coupler(degree)


class GenericCell(Cell):
    """A Haar-random unitary per degree, shared by the vertices of that
    degree: a passive cell that reads the rotation system."""

    name = "generic"

    def __init__(self, seed=0):
        self.seed = seed

    def unitary(self, degree):
        return haar_unitary(
            degree + 2, np.random.default_rng([self.seed, degree]))


def canonical():
    """R1: ``psi = pi`` and one Haar-random ``W``, drawn once."""
    return InvariantCell(
        haar_unitary(3, np.random.default_rng(0)), np.pi, "canonical")


def grover():
    """The Grover point: ``psi = pi`` and a diagonal ``W``, so the darts
    carry the Grover coin ``2J/d - I`` and the drive never enters."""
    return InvariantCell(
        np.diag(np.exp(1j * np.array([.3, .7, 1.1]))), np.pi, "grover")


def seeded(seed):
    """R2: ``W`` Haar-random and ``psi`` uniform, from one seed."""
    rng = np.random.default_rng(seed)
    return InvariantCell(
        haar_unitary(3, rng), rng.uniform(0, 2 * np.pi), f"seed-{seed:02d}")


SEEDS = range(1, 21)
CELLS = {cell.name: cell for cell in (
    [canonical(), grover(), LadderCell(), GenericCell()]
    + [seeded(seed) for seed in SEEDS])}
