"""
The five QMapNN models, each a functor from a graph to an
:class:`optyx.interaction.CMap` whose cells are physical CPTP maps.

A graph is a tuple of sorted neighbour tuples. A model provides
``vertex(degree)`` — the interpretation of one cell — and ``drive``, the
state fed to the cell's boundary ports at every tick; ``model(graph)``
is the functor, keeping the combinatorial map and replacing every vertex
by its cell:

- :class:`PassiveModel` — a port-symmetric interferometer on one drive
  mode, the message modes and a coherent memory, one photon into the
  drive at every tick, a beam-splitter tap as prediction.
- :class:`BellModel` — the same interferometer on *two* drive rails
  carrying one half of a dual-rail Bell pair per tick; the other half is
  rotated by a fixed beam splitter inside the cell and exits with the
  reflections, so the record correlates the network with an entangled
  local reference.
- :class:`ActiveModel` — the cell of the companion notebook
  ``beyond_3wl``: messages and memory carry ``qmode @ bit``, a threshold
  detector on the tap broadcasts its click one hop and the latched bit
  re-programs the tap through a bit-controlled beam splitter.
- :class:`QubitModel` — a qubit cell: a fresh ancilla in ``|+>``, a
  commuting star of CZ gates between ancilla, memory and every message
  port, fixed single-qubit rotations, and the ancilla measured in the X
  basis as the prediction. Only the prediction is measured.
- :class:`QubitFeedforwardModel` — the qubit cell with classical
  messages beside the quantum ones: the ancilla click is broadcast one
  hop along the edge bits, OR-latched into a stored bit which drives a
  bit-controlled phase on the memory qubit.

Every cell is port-symmetric and every parameter is fixed and shared, so
each model is a graph invariant by construction.
"""

from optyx import classical, photonic
from optyx.channel import Diagram, Ty, bit, qmode, qubit
from optyx.interaction import Box, CMap
from optyx.photonic import Create, Gate, TBS
from optyx.qubits import H, Ket, Measure, Scalar, X, Z

import numpy as np


def adjacency(graph):
    """A networkx graph as a tuple of sorted neighbour tuples."""
    nodes = sorted(graph.nodes)
    index = {v: i for i, v in enumerate(nodes)}
    return tuple(
        tuple(sorted(index[u] for u in graph.neighbors(v)))
        for v in nodes)


def unitary(coupling):
    """The unitary ``exp(i H)`` of a hermitian coupling matrix."""
    coupling = coupling + np.triu(coupling, 1).conj().T
    energies, modes = np.linalg.eigh(coupling)
    return modes @ np.diag(np.exp(1j * energies)) @ modes.conj().T


def coupler(degree, theta=0.4, phi=0.7, chi=1.1, drive=0.8):
    """The port-symmetric interferometer of ``beyond_3wl``: modes are
    the drive, the message ports and the memory."""
    modes = degree + 2
    ports, memory = range(1, degree + 1), modes - 1
    coupling = np.zeros((modes, modes), dtype=complex)
    for port in ports:
        coupling[port, memory] = theta * np.exp(1j * phi) / degree ** .5
        coupling[0, port] = 0.25 * np.exp(1j * 0.5) / degree ** .5
        for other in ports:
            if port != other:
                coupling[port, other] = chi / degree
    coupling[0, memory] = drive * np.exp(1j * 0.3)
    coupling[memory, memory] = 0.9
    return unitary(coupling)


def bell_coupler(degree, theta=0.4, phi=0.7, chi=1.1):
    """The interferometer of the Bell cell: two drive rails with
    different couplings, the message ports and the memory."""
    modes = degree + 3
    ports, memory = range(2, degree + 2), modes - 1
    coupling = np.zeros((modes, modes), dtype=complex)
    for port in ports:
        coupling[port, memory] = theta * np.exp(1j * phi) / degree ** .5
        coupling[0, port] = 0.25 * np.exp(1j * 0.5) / degree ** .5
        coupling[1, port] = 0.35 * np.exp(1j * 1.3) / degree ** .5
        for other in ports:
            if port != other:
                coupling[port, other] = chi / degree
    coupling[0, memory] = 0.8 * np.exp(1j * 0.3)
    coupling[1, memory] = 0.5 * np.exp(1j * 2.1)
    coupling[0, 1] = 0.6 * np.exp(1j * 0.9)
    coupling[memory, memory] = 0.9
    return unitary(coupling)


class Model:
    """A functor from graphs to :class:`optyx.interaction.CMap`."""

    name = None
    atoms = 1

    def vertex(self, degree) -> Box:
        raise NotImplementedError

    def drive(self, degree) -> Diagram:
        """The state fed to one cell's boundary ports at every tick."""
        return Diagram.id(Ty())

    def port(self, position):
        """The first port atom of the ``position``-th neighbour."""
        raise NotImplementedError

    def __call__(self, graph) -> CMap:
        boxes = [self.vertex(len(nbrs)) for nbrs in graph]
        edges = [
            ((u, self.port(nbrs.index(v)) + a),
             (v, self.port(graph[v].index(u)) + a))
            for u, nbrs in enumerate(graph) for v in nbrs if u < v
            for a in range(self.atoms)]
        return CMap(boxes, edges)

    def unrolled(self, graph, n_ticks) -> Diagram:
        """The driven protocol over ``n_ticks``, memory in the vacuum,
        one drive state into every cell at every tick."""
        cmap = self(graph)
        tick = Diagram.id().tensor(
            *[self.drive(len(nbrs)) for nbrs in graph])
        memory = Diagram.id().tensor(*[
            Ket(0) if ob == list(qubit)[0] else
            classical.Bit(0) if ob == list(bit)[0] else Create(0)
            for ob in cmap.memory])
        return (Diagram.id().tensor(*[tick] * n_ticks) @ memory
                >> cmap.unroll(n_ticks - 1))


class PassiveModel(Model):
    """One photon into every cell at every tick, passive optics."""

    name = "passive"

    def __init__(self, tap=0.3):
        self.tap = tap

    def port(self, position):
        return 1 + position

    def drive(self, degree):
        return Create(1)

    def vertex(self, degree):
        modes = degree + 2
        mix = Gate(coupler(degree), modes, modes, f"mix{degree}")
        channel = mix >> Diagram.id(qmode ** (modes - 1)) @ (
            qmode @ Create(0) >> TBS(self.tap))
        return Box(f"cell{degree}", qmode, qmode ** degree, channel,
                   memory=qmode, prediction=qmode)


CNOT = Z(1, 2) @ qubit >> qubit @ X(2, 1) @ Scalar(2 ** .5)

BELL_STATE = (Ket(0) @ Ket(0) >> H() @ qubit >> CNOT
              >> photonic.DualRail(2))


class BellModel(Model):
    """A dual-rail Bell pair into every cell at every tick: rails one
    and two carry the first qubit into the interferometer, rails three
    and four keep the second qubit as a rotated local reference exiting
    with the reflections."""

    name = "bell"

    def __init__(self, tap=0.3, reference=0.25):
        self.tap, self.reference = tap, reference

    def port(self, position):
        return 4 + position

    def drive(self, degree):
        return BELL_STATE

    def vertex(self, degree):
        modes = degree + 3
        mix = Gate(bell_coupler(degree), modes, modes, f"bmix{degree}")
        sort = Diagram.permutation(
            [0, 1] + list(range(4, degree + 5)) + [2, 3],
            qmode ** (degree + 5))
        channel = (
            sort
            >> mix @ TBS(self.reference)
            >> Diagram.id(qmode ** (modes - 1))
            @ (qmode @ Create(0) >> TBS(self.tap)) @ qmode ** 2
            >> Diagram.permutation(
                [0, 1, degree + 4, degree + 5]
                + list(range(2, degree + 2)) + [degree + 2, degree + 3],
                qmode ** (degree + 6)))
        return Box(f"bell{degree}", qmode ** 4, qmode ** degree, channel,
                   memory=qmode, prediction=qmode)


class ActiveModel(Model):
    """The active cell of ``beyond_3wl``: classical bits beside the
    coherent messages, a threshold detector on the tap, the click
    broadcast one hop, the latched bit re-programming the tap. With
    ``flood`` the cell broadcasts its latch instead of its click, so
    every cell relays the bits it receives and a click spreads one
    further hop per tick."""

    atoms = 2

    def __init__(self, tap=0.3, kick=0.5, flood=False):
        self.tap, self.kick, self.flood = tap, kick, flood
        self.name = "active-flood" if flood else "active"

    def port(self, position):
        return 1 + 2 * position

    def drive(self, degree):
        return Create(1)

    def vertex(self, degree):
        tap, kick = self.tap, self.kick
        modes = degree + 2
        mix = Gate(coupler(degree), modes, modes, f"mix{degree}")
        source = qmode @ (qmode @ bit) ** degree @ (qmode @ bit)
        sort = Diagram.permutation(
            [0] + [1 + 2 * j for j in range(degree)] + [1 + 2 * degree]
            + [2 + 2 * j for j in range(degree)] + [2 + 2 * degree],
            source)
        channel = (
            sort
            >> Diagram.id(qmode ** modes) @ classical.Or(degree + 1)
            >> Diagram.id(qmode ** modes) @ classical.CopyBit(2)
            >> mix @ bit @ bit
            >> Diagram.id(qmode ** (modes - 1))
            @ (qmode @ Create(0) >> TBS(tap)) @ bit @ bit
            >> Diagram.id(qmode ** (modes - 1))
            @ Diagram.swap(qmode ** 2, bit) @ bit
            >> Diagram.id(qmode ** (modes - 1))
            @ classical.BitControlledGate(TBS(kick)) @ bit
            >> Diagram.id(qmode ** modes)
            @ photonic.PhotonThresholdMeasurement() @ bit
            >> (Diagram.id(qmode ** modes)
                @ classical.Or(2)
                >> Diagram.id(qmode ** modes)
                @ classical.CopyBit(degree + 2)
                if self.flood else
                Diagram.id(qmode ** modes)
                @ classical.CopyBit(degree + 2) @ bit
                >> Diagram.id(qmode ** modes @ bit ** (degree + 1))
                @ classical.Or(2)))
        n_wires = modes + degree + 2
        gather = ([0] + [i for j in range(degree)
                         for i in (1 + j, modes + j)]
                  + [modes - 1, n_wires - 1, modes + degree])
        channel = channel >> Diagram.permutation(gather, channel.cod)
        return Box(f"{self.name}{degree}", qmode,
                   (qmode @ bit) ** degree, channel,
                   memory=qmode @ bit, prediction=bit)


CZ = (Z(1, 2) @ qubit >> qubit @ H() @ qubit >> qubit @ Z(2, 1)) \
    @ Scalar(2 ** .5)


def on_pair(gate, first, second, total):
    """A two-qubit gate applied at positions ``first, second``."""
    order = [first, second] + [
        x for x in range(total) if x not in (first, second)]
    inverse = [order.index(x) for x in range(total)]
    return (Diagram.permutation(order, qubit ** total)
            >> gate @ qubit ** (total - 2)
            >> Diagram.permutation(inverse, qubit ** total))


def qubit_core(degree, alpha, beta):
    """A fresh ancilla in ``|+>``, the commuting CZ star between
    ancilla, memory and every port, and fixed rotations: wires are the
    ancilla, the ports and the memory."""
    total = degree + 2
    diagram = Ket('+') @ qubit ** (degree + 1)
    for port in range(1, degree + 1):
        diagram >>= on_pair(CZ, 0, port, total)
        diagram >>= on_pair(CZ, port, total - 1, total)
    diagram >>= on_pair(CZ, 0, total - 1, total)
    return diagram >> Diagram.id().tensor(
        *([H()] + [Z(1, 1, alpha)] * degree + [X(1, 1, beta)]))


class QubitModel(Model):
    """Quantum messages, unitary cells, only the prediction measured."""

    name = "qubit"

    def __init__(self, alpha=0.35, beta=0.55):
        self.alpha, self.beta = alpha, beta

    def port(self, position):
        return position

    def vertex(self, degree):
        total = degree + 2
        channel = (
            qubit_core(degree, self.alpha, self.beta)
            >> Diagram.permutation(
                list(range(1, total)) + [0], qubit ** total)
            >> Diagram.id(qubit ** (total - 1)) @ Measure(1))
        return Box(f"qcell{degree}", qubit ** degree, Ty(), channel,
                   memory=qubit, prediction=bit)


class QubitFeedforwardModel(Model):
    """The qubit cell with classical messages: the ancilla click is
    broadcast one hop, OR-latched, and the latched bit drives a phase
    on the memory qubit."""

    name = "qubit-ff"
    atoms = 2

    def __init__(self, alpha=0.35, beta=0.55, gamma=0.45):
        self.alpha, self.beta, self.gamma = alpha, beta, gamma

    def port(self, position):
        return 2 * position

    def vertex(self, degree):
        total = degree + 2
        source = (qubit @ bit) ** degree @ (qubit @ bit)
        sort = Diagram.permutation(
            [2 * j for j in range(degree + 1)]
            + [2 * j + 1 for j in range(degree + 1)], source)
        kick = classical.BitControlledGate(Z(1, 1, self.gamma))
        diagram = (
            sort
            >> qubit ** (degree + 1) @ classical.Or(degree + 1)
            >> qubit ** (degree + 1) @ classical.CopyBit(2)
            >> qubit ** degree
            @ Diagram.swap(qubit, bit) @ bit
            >> qubit ** degree @ kick @ bit
            >> qubit_core(degree, self.alpha, self.beta) @ bit
            >> Diagram.permutation(
                list(range(1, total)) + [0], qubit ** total) @ bit
            >> qubit ** (total - 1) @ Measure(1) @ bit
            >> qubit ** (total - 1)
            @ classical.CopyBit(degree + 2) @ bit
            >> qubit ** (total - 1) @ bit ** (degree + 1)
            @ classical.Or(2))
        gather = ([i for j in range(degree)
                   for i in (j, total - 1 + j)]
                  + [degree, 2 * total - 2, 2 * total - 3])
        channel = diagram >> Diagram.permutation(gather, diagram.cod)
        return Box(f"qfcell{degree}", (qubit @ bit) ** degree, Ty(),
                   channel, memory=qubit @ bit, prediction=bit)


MODELS = {model.name: model for model in (
    PassiveModel(), BellModel(), ActiveModel(),
    ActiveModel(flood=True), QubitModel(), QubitFeedforwardModel())}
