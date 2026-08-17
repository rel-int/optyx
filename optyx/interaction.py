"""
Overview
--------

Combinatorial maps of recurrent channels.

A :class:`CMap` is a combinatorial map: a list of :class:`Box` carrying
channels, together with edges pairing their ports. Its semantics is the
recurrent protocol :meth:`CMap.protocol`, a :class:`optyx.channel.Diagram`
with a feedback loop on the paired ports. It has finite stream semantics
through :meth:`CMap.unroll` and stationary semantics through
:meth:`CMap.fix`.

This is the Int (geometry of interaction) construction of Joyal, Street
and Verity applied to the feedback category of channels:
:meth:`optyx.channel.Diagram.feedback` plays the role of a delayed trace,
so the structure is compact closed up to time shifts.

Every port of a box is both read and written at each time step, so a box
:code:`x -> y` carries a channel from :code:`x @ y` to :code:`x @ y` and an
edge carries messages in both directions. A paired port writes into the
memory read by its partner at the next time step; an unpaired port reads
from the domain and writes to the codomain, so :code:`dom == cod` is the
tensor of the unpaired port types.

>>> from optyx.channel import qubit
>>> from optyx.qubits import Z, X, Scalar
>>> cnot = Z(1, 2) @ qubit >> qubit @ X(2, 1) @ Scalar(2 ** 0.5)
>>> box = Box("f", qubit, qubit, cnot)
>>> cmap = CMap([box, box], [((0, 1), (1, 0))])
>>> assert cmap.dom == cmap.cod == qubit ** 2
>>> assert cmap.memory == qubit ** 2
>>> assert cmap.unroll(1).dom == cmap.dom ** 2 @ cmap.memory

A box can also carry an internal memory :code:`m` and a prediction
:code:`o`: its channel then goes from :code:`x @ y @ m` to
:code:`x @ y @ m @ o`. The memory is a feedback loop from the box to
itself, never a port of the map; the prediction is written to the
environment at every step but never read, appended to the codomain of the
protocol. A box that copies its memory predicts it without consuming it:

>>> readout = Box("readout", Ty(), Ty(), Z(1, 2),
...     memory=qubit, prediction=qubit)
>>> cmap = CMap([readout], [])
>>> assert cmap.dom == cmap.cod == Ty()
>>> assert cmap.memory == cmap.prediction == qubit
>>> assert cmap.protocol.cod == qubit

Recurrent tensor networks
-------------------------

A :class:`CMap` turns into a tensor network in two orthogonal directions:
*space* — its boxes and edges — and *time* — the unrolled ticks. Each stage
of the construction is a diagram the user can inspect and draw. The
running example is the smallest purely quantum map with every feature:
three cells in a triangle, each with an internal memory and a prediction.
A cell entangles its memory with the message on its outgoing port by a
CNOT, then copies its memory to its prediction:

>>> cell = Box("cell", qubit, qubit, qubit @ (cnot >> qubit @ Z(1, 2)),
...     memory=qubit, prediction=qubit)
>>> triangle = CMap(3 * [cell],
...     [((0, 1), (1, 0)), ((1, 1), (2, 0)), ((2, 1), (0, 0))])
>>> assert triangle.dom == triangle.cod == Ty()
>>> assert triangle.memory == qubit ** 9
>>> assert triangle.prediction == qubit ** 3

**One tick is a channel diagram.** :attr:`CMap.step` composes three
diagrams: :attr:`CMap.read`, a permutation routing :code:`dom @ memory`
onto the inputs of the boxes, then :attr:`CMap.parallel`, the tensor
product of the local channels, then :attr:`CMap.write`, a permutation
routing their outputs to :code:`cod @ prediction @ memory` — each paired
port to the memory read by its partner at the next tick, each internal
memory back to its own box. The permutations carry no data: all the
computation sits in the boxes, as in a premonoidal normal form.

>>> assert triangle.step \\
...     == triangle.read >> triangle.parallel >> triangle.write
>>> triangle.step.draw(figsize=(10, 5),
...     path="docs/_static/interaction_step.png")

.. image:: /_static/interaction_step.png
    :align: center

**Feedback closes the time loop.** :attr:`CMap.protocol` applies
:meth:`optyx.channel.Diagram.feedback` to the step, closing
:attr:`CMap.memory` — the paired ports followed by the internal memories —
onto itself with a one-tick delay. The memory is a *delay*, which makes
the protocol the feedback of a monoidal stream (Di Lavore, de Felice and
Román, LICS 2022) rather than a trace: nothing is computed yet, the loop
is syntax.

>>> triangle.protocol.draw(figsize=(10, 5),
...     path="docs/_static/interaction_protocol.png")

.. image:: /_static/interaction_protocol.png
    :align: center

**Unrolling gives a finite diagram.** :meth:`CMap.unroll` composes
:code:`n_steps + 1` copies of the step, threading the memory from each
tick to the next; the result is an ordinary
:class:`optyx.channel.Diagram` whose width is the memory cut and whose
depth is the number of ticks.

>>> triangle.unroll(1).draw(figsize=(10, 8),
...     path="docs/_static/interaction_unroll.png")

.. image:: /_static/interaction_unroll.png
    :align: center

**Contraction closes the time loop the other way.** The standard optyx
pipeline turns any of these diagrams into numbers:
:meth:`optyx.channel.Diagram.double` maps a channel diagram to its Kraus
map beside the conjugate, :code:`to_tensor()` translates it to a
:class:`discopy.tensor.Diagram`, and DisCoPy contracts it — in one
``einsum`` under the NumPy, JAX or PyTorch array backend, or through the
Quimb backends of :mod:`optyx.core.backends` with optimised and
compressed contraction paths. :meth:`CMap.fix` runs this pipeline at an
approximately stationary time, instead of a finite unrolling: with every
memory prepared at :code:`Ket(0)` the triangle is stationary from the
start, and the contraction reads :code:`Ket(0)` off all three
predictions.

>>> import numpy as np
>>> from optyx.qubits import Ket
>>> from optyx.core.backends import DiscopyBackend
>>> fixed = triangle.fix(
...     initial_state=Diagram.id().tensor(*9 * [Ket(0)]),
...     max_steps=2, backend=DiscopyBackend())
>>> expected = Diagram.id().tensor(*3 * [Ket(0)])
>>> assert np.allclose(fixed.density_matrix,
...     expected.double().to_tensor().eval().array)

See :doc:`/notebooks/fixpoints` for the certificates behind
:meth:`optyx.channel.Diagram.fix`, which :meth:`CMap.fix` delegates to.

Types and diagrams
------------------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Box
    CMap
"""

from __future__ import annotations

from discopy.utils import AxiomError

from optyx.channel import Diagram, Ty

__all__ = ["Box", "CMap", "Ty"]


class Box:
    """
    A box of a :class:`CMap`, carrying a recurrent channel on its ports.

    Parameters:
        name : The name of the box.
        dom : The input ports.
        cod : The output ports.
        channel : A :class:`optyx.channel.Diagram` from
            :code:`dom @ cod @ memory` to
            :code:`dom @ cod @ memory @ prediction`, reading and writing
            every port, updating the memory and writing the prediction.
        memory : The internal memory, fed back to the box itself at the
            next time step, empty by default.
        prediction : The type written to the environment at every time
            step but never read, empty by default.

    >>> from optyx.channel import qubit
    >>> from optyx.qubits import H, Z
    >>> assert Box("h", Ty(), qubit, H()).ports == qubit
    >>> mem = Box("m", Ty(), Ty(), Z(1, 2), memory=qubit, prediction=qubit)
    >>> assert mem.ports == Ty()
    """

    def __init__(self, name: str, dom: Ty, cod: Ty, channel: Diagram,
                 memory: Ty = Ty(), prediction: Ty = Ty()):
        source = dom @ cod @ memory
        target = source @ prediction
        if (channel.dom, channel.cod) != (source, target):
            raise AxiomError(
                f"{channel} is not a channel from {source} to {target}")
        self.name, self.dom, self.cod = name, dom, cod
        self.memory, self.prediction = memory, prediction
        self.channel = channel

    @property
    def ports(self) -> Ty:
        """The pairable ports of the box, its domain then its codomain."""
        return self.dom @ self.cod

    def __repr__(self):
        extra = "" if not self.memory @ self.prediction else (
            f", memory={repr(self.memory)}"
            f", prediction={repr(self.prediction)}")
        return (f"Box({repr(self.name)}, {repr(self.dom)}, "
                f"{repr(self.cod)}{extra})")

    def __str__(self):
        return self.name

    def __eq__(self, other):
        return isinstance(other, Box) and (
            self.name, self.dom, self.cod, self.memory, self.prediction,
            self.channel) == (
            other.name, other.dom, other.cod, other.memory, other.prediction,
            other.channel)

    def __hash__(self):
        return hash(
            (self.name, self.dom, self.cod, self.memory, self.prediction))


class CMap:
    """
    A combinatorial map: boxes carrying channels and edges pairing ports.

    A port is a pair :code:`(box, port)` of indices, the second indexing
    :attr:`Box.ports`. An edge is a pair of distinct ports of equal type,
    and every port belongs to at most one edge: an edge between two ports
    of the same map is a cup or a cap of the compact closed structure. The
    memory and prediction of a box are not ports and cannot be paired.

    Parameters:
        boxes : The boxes of the map.
        edges : The pairs of ports glued by the map.

    >>> from optyx.channel import qubit
    >>> from optyx.qubits import Z, X, Scalar
    >>> cnot = Z(1, 2) @ qubit >> qubit @ X(2, 1) @ Scalar(2 ** 0.5)
    >>> box = Box("f", qubit, qubit, cnot)
    >>> cmap = CMap([box, box], [((0, 1), (1, 0))])
    >>> assert cmap.boundary == [(0, 0), (1, 1)]
    >>> assert cmap.memory == qubit ** 2

    Both ports of an edge must have the same type, and a port cannot be
    paired twice:

    >>> from optyx.channel import qmode
    >>> from optyx.photonic import BS
    >>> CMap([box, Box("g", qmode, qmode, BS)], [((0, 0), (1, 0))])
    Traceback (most recent call last):
    ...
    discopy.utils.AxiomError: (0, 0) and (1, 0) have types qubit and qmode.
    >>> CMap([box, box], [((0, 1), (1, 0)), ((0, 1), (1, 1))])
    Traceback (most recent call last):
    ...
    discopy.utils.AxiomError: (0, 1) is paired twice.
    """

    def __init__(self, boxes, edges):
        self.boxes, self.edges = list(boxes), [
            tuple(edge) for edge in edges]
        self.partner = {}
        for source, target in self.edges:
            for port in (source, target):
                if port not in self.ports:
                    raise AxiomError(f"{port} is not a port.")
                if port in self.partner:
                    raise AxiomError(f"{port} is paired twice.")
            if self.port_type(source) != self.port_type(target):
                raise AxiomError(
                    f"{source} and {target} have types "
                    f"{self.port_type(source)} and {self.port_type(target)}.")
            self.partner[source] = target
            self.partner[target] = source

    @property
    def ports(self) -> list:
        """The pairable ports of the map, in box order."""
        return [(i, j) for i, box in enumerate(self.boxes)
                for j in range(len(box.ports))]

    @property
    def memories(self) -> list:
        """The internal memory wires of the boxes, in box order."""
        return [(i, len(box.ports) + j)
                for i, box in enumerate(self.boxes)
                for j in range(len(box.memory))]

    @property
    def predictions(self) -> list:
        """The prediction wires of the boxes, in box order."""
        return [(i, len(box.ports @ box.memory) + j)
                for i, box in enumerate(self.boxes)
                for j in range(len(box.prediction))]

    def port_type(self, port) -> Ty:
        """The type of a port, memory or prediction wire."""
        box, index = port
        box = self.boxes[box]
        return (box.ports @ box.memory @ box.prediction)[index]

    @property
    def boundary(self) -> list:
        """The unpaired ports, in box order."""
        return [port for port in self.ports if port not in self.partner]

    @property
    def paired(self) -> list:
        """The paired ports, in box order."""
        return [port for port in self.ports if port in self.partner]

    @property
    def memory(self) -> Ty:
        """
        The type fed back: one wire per paired port, then the internal
        memory of each box.
        """
        return Ty().tensor(*map(
            self.port_type, self.paired + self.memories))

    @property
    def prediction(self) -> Ty:
        """The type written to the environment by the boxes at every step."""
        return Ty().tensor(*map(self.port_type, self.predictions))

    @property
    def dom(self) -> Ty:
        """The type read from the environment by the unpaired ports."""
        return Ty().tensor(*map(self.port_type, self.boundary))

    @property
    def cod(self) -> Ty:
        """
        The type written to the environment by the unpaired ports, equal to
        :attr:`dom` because every port is both read and written. The
        predictions are appended to it in the codomain of the protocol.
        """
        return self.dom

    @property
    def parallel(self) -> Diagram:
        """The tensor of the channels of every box."""
        return Diagram.id().tensor(*[box.channel for box in self.boxes])

    def permutation(self, sources, targets) -> Diagram:
        """The permutation of ports from :code:`sources` to :code:`targets`."""
        return Diagram.permutation(
            [sources.index(port) for port in targets],
            Ty().tensor(*map(self.port_type, sources)))

    @property
    def wires(self) -> list:
        """
        The wires in the order they are fed back: the boundary read from
        and written to the environment, then the paired ports, then the
        internal memories.
        """
        return self.boundary + self.paired + self.memories

    @property
    def inputs(self) -> list:
        """The wires read by the channels of the boxes, in box order."""
        return [(i, j) for i, box in enumerate(self.boxes)
                for j in range(len(box.ports @ box.memory))]

    @property
    def outputs(self) -> list:
        """The wires written by the channels of the boxes, in box order."""
        return [(i, j) for i, box in enumerate(self.boxes)
                for j in range(
                    len(box.ports @ box.memory @ box.prediction))]

    @property
    def read(self) -> Diagram:
        """
        The permutation from :code:`dom @ memory` to the channel inputs,
        feeding each port from the environment or from the memory it is
        fed back, and each internal memory from its own box.
        """
        return self.permutation(self.wires, self.inputs)

    @property
    def write(self) -> Diagram:
        """
        The permutation from the channel outputs to
        :code:`cod @ prediction @ memory`, sending each port to the
        environment or to the memory read by its partner, each prediction
        to the environment and each internal memory back to its box.
        """
        return self.permutation(
            self.outputs, self.boundary + self.predictions + [
                self.partner[port] for port in self.paired
            ] + self.memories)

    @property
    def step(self) -> Diagram:
        """One message-passing step before the memory is fed back."""
        return self.read >> self.parallel >> self.write

    @property
    def protocol(self) -> Diagram:
        """
        The recurrent protocol of the map: the channels of the boxes in
        parallel, permuted by the edges and fed back on the paired ports
        and internal memories, with the predictions after the boundary.

        >>> from optyx.channel import qubit
        >>> from optyx.qubits import Z, X, Scalar
        >>> cnot = Z(1, 2) @ qubit >> qubit @ X(2, 1) @ Scalar(2 ** 0.5)
        >>> cmap = CMap([Box("f", qubit, qubit, cnot)], [((0, 0), (0, 1))])
        >>> assert cmap.protocol.dom == cmap.dom
        >>> assert cmap.protocol.mem == cmap.memory
        """
        return self.step.feedback(
            dom=self.dom, cod=self.cod @ self.prediction, mem=self.memory)

    def unroll(self, n_steps: int = 1) -> Diagram:
        """
        The protocol unrolled over :code:`n_steps + 1` time steps — as in
        :meth:`optyx.channel.Diagram.unroll`, :code:`n_steps` counts
        unrollings — a channel diagram evaluated as a tensor network by
        the existing backends.

        >>> from optyx.channel import qubit
        >>> from optyx.qubits import Z, X, Scalar
        >>> cnot = Z(1, 2) @ qubit >> qubit @ X(2, 1) @ Scalar(2 ** 0.5)

        The memory starts open at the end of the domain and is discarded
        after the last step, the default boundaries of
        :meth:`optyx.channel.Diagram.feedback`; :meth:`fix` is where other
        boundaries are chosen.

        >>> cmap = CMap([Box("f", qubit, qubit, cnot)], [((0, 0), (0, 1))])
        >>> assert cmap.unroll(2).dom == cmap.memory
        >>> assert cmap.unroll(2).cod == Ty()
        """
        return self.protocol.unroll(n_steps)

    def fix(self, input_state=None, initial_state=None, **params):
        """
        Approximate the stationary output of the recurrent protocol, the
        boundary followed by the predictions.

        ``input_state`` prepares the boundary input afresh at every time
        step, inside the loop so that the stationary certificates of
        :meth:`optyx.channel.Diagram.fix` can apply. It is required when
        the map has unpaired ports. ``initial_state`` prepares the feedback
        memory, the paired ports followed by the internal memories, before
        the first time step. Remaining parameters are those of
        :meth:`optyx.channel.Diagram.fix`.

        A box that swaps its output port with its internal memory is a
        delay line: fed a photon at every step, its stationary output is
        that photon.

        >>> import numpy as np
        >>> from optyx.channel import qmode
        >>> from optyx.photonic import Create
        >>> wait = Box("wait", Ty(), qmode,
        ...     Diagram.swap(qmode, qmode), memory=qmode)
        >>> fixed = CMap([wait], []).fix(Create(1), Create(0), max_chi=None)
        >>> assert np.allclose(fixed.density_matrix, [[0, 0], [0, 1]])
        """
        if input_state is None:
            if self.dom:
                raise ValueError(
                    "input_state is required for a map with a boundary.")
            input_state = Diagram.id()
        if (input_state.dom, input_state.cod) != (Ty(), self.dom):
            raise AxiomError(
                f"input_state must be a state of type {self.dom}.")
        step = input_state @ Diagram.id(self.memory) >> self.step
        return step.feedback(
            dom=Ty(), cod=self.cod @ self.prediction, mem=self.memory,
            state=initial_state).fix(**params)

    def to_drawing(self):
        """The drawing of the protocol."""
        return self.protocol.to_drawing()

    def draw(self, **params):
        """Draw the protocol."""
        return self.protocol.draw(**params)

    def __matmul__(self, other: CMap) -> CMap:
        """
        The disjoint union of two maps, reindexing the ports of the second.

        >>> from optyx.channel import qubit
        >>> from optyx.qubits import Z, X, Scalar
        >>> cnot = Z(1, 2) @ qubit >> qubit @ X(2, 1) @ Scalar(2 ** 0.5)
        >>> cmap = CMap([Box("f", qubit, qubit, cnot)], [((0, 0), (0, 1))])
        >>> assert (cmap @ cmap).memory == cmap.memory ** 2
        """
        shift = len(self.boxes)
        return CMap(self.boxes + other.boxes, self.edges + [
            ((i + shift, j), (k + shift, ll))
            for (i, j), (k, ll) in other.edges])

    def __repr__(self):
        return f"CMap({repr(self.boxes)}, {repr(self.edges)})"

    def __eq__(self, other):
        return isinstance(other, CMap) and (
            self.boxes, self.edges) == (other.boxes, other.edges)
