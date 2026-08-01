"""
Combinatorial maps of recurrent channels.

A :class:`CMap` is a combinatorial map: a list of :class:`Box` carrying
channels, together with edges pairing their ports. Its semantics is the
recurrent protocol :meth:`CMap.protocol`, a :class:`optyx.channel.Diagram`
with a feedback loop on the paired ports. It has finite stream semantics
through :meth:`CMap.unroll` and stationary semantics through
:meth:`CMap.fix`.

This is the Int (geometry of interaction) construction applied to the
feedback category of channels: :meth:`optyx.channel.Diagram.feedback` plays
the role of a delayed trace, so the structure is compact closed up to time
shifts.

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
>>> assert cmap.unroll(2).dom == cmap.dom ** 2 @ cmap.memory
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
        channel : A :class:`optyx.channel.Diagram` from :code:`dom @ cod`
            to :code:`dom @ cod`, reading and writing every port.

    >>> from optyx.channel import qubit
    >>> from optyx.qubits import H
    >>> assert Box("h", Ty(), qubit, H()).ports == qubit
    """

    def __init__(self, name: str, dom: Ty, cod: Ty, channel: Diagram):
        ports = dom @ cod
        if (channel.dom, channel.cod) != (ports, ports):
            raise AxiomError(
                f"{channel} is not a channel from {ports} to {ports}")
        self.name, self.dom, self.cod = name, dom, cod
        self.channel = channel

    @property
    def ports(self) -> Ty:
        """The ports of the box, its domain followed by its codomain."""
        return self.dom @ self.cod

    def __repr__(self):
        return f"Box({repr(self.name)}, {repr(self.dom)}, {repr(self.cod)})"

    def __str__(self):
        return self.name

    def __eq__(self, other):
        return isinstance(other, Box) and (
            self.name, self.dom, self.cod, self.channel) == (
            other.name, other.dom, other.cod, other.channel)

    def __hash__(self):
        return hash((self.name, self.dom, self.cod))


class CMap:
    """
    A combinatorial map: boxes carrying channels and edges pairing ports.

    A port is a pair :code:`(box, port)` of indices, the second indexing
    :attr:`Box.ports`. An edge is a pair of distinct ports of equal type,
    and every port belongs to at most one edge.

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
        """The ports of the map, in box order."""
        return [(i, j) for i, box in enumerate(self.boxes)
                for j in range(len(box.ports))]

    def port_type(self, port) -> Ty:
        """The type of a port."""
        box, index = port
        return self.boxes[box].ports[index]

    @property
    def boundary(self) -> list:
        """The unpaired ports, in box order."""
        return [port for port in self.ports if port not in self.partner]

    @property
    def memory(self) -> Ty:
        """The type fed back, one wire per paired port."""
        return Ty().tensor(*[
            self.port_type(port) for port in self.ports
            if port in self.partner])

    @property
    def dom(self) -> Ty:
        """The type read from the environment by the unpaired ports."""
        return Ty().tensor(*map(self.port_type, self.boundary))

    @property
    def cod(self) -> Ty:
        """
        The type written to the environment by the unpaired ports, equal to
        :attr:`dom` because every port is both read and written.
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
        The ports in the order they are fed back, the boundary read from
        and written to the environment followed by the paired ports.
        """
        return self.boundary + [
            port for port in self.ports if port in self.partner]

    @property
    def read(self) -> Diagram:
        """
        The permutation from :code:`dom @ memory` to the ports, feeding
        each port from the environment or from the memory it is fed back.
        """
        return self.permutation(self.wires, self.ports)

    @property
    def write(self) -> Diagram:
        """
        The permutation from the ports to :code:`cod @ memory`, sending
        each port to the environment or to the memory read by its partner.
        """
        return self.permutation(
            self.ports, [self.partner.get(port, port)
                         for port in self.wires])

    @property
    def step(self) -> Diagram:
        """One message-passing step before the memory is fed back."""
        return self.read >> self.parallel >> self.write

    @property
    def protocol(self) -> Diagram:
        """
        The recurrent protocol of the map: the channels of the boxes in
        parallel, permuted by the edges and fed back on the paired ports.

        >>> from optyx.channel import qubit
        >>> from optyx.qubits import Z, X, Scalar
        >>> cnot = Z(1, 2) @ qubit >> qubit @ X(2, 1) @ Scalar(2 ** 0.5)
        >>> cmap = CMap([Box("f", qubit, qubit, cnot)], [((0, 0), (0, 1))])
        >>> assert cmap.protocol.dom == cmap.dom
        >>> assert cmap.protocol.mem == cmap.memory
        """
        return self.step.feedback(
            dom=self.dom, cod=self.cod, mem=self.memory)

    def unroll(self, n_steps: int = 1) -> Diagram:
        """
        The protocol unrolled over :code:`n_steps` time steps, a channel
        diagram evaluated as a tensor network by the existing backends.

        >>> from optyx.channel import qubit
        >>> from optyx.qubits import Z, X, Scalar
        >>> cnot = Z(1, 2) @ qubit >> qubit @ X(2, 1) @ Scalar(2 ** 0.5)
        >>> cmap = CMap([Box("f", qubit, qubit, cnot)], [((0, 0), (0, 1))])
        >>> assert cmap.unroll(3).dom == cmap.unroll(3).cod
        """
        return self.protocol.unroll(n_steps)

    def fix(self, input_state=None, initial_state=None, **params):
        """
        Approximate the stationary boundary output of the recurrent protocol.

        ``input_state`` prepares the boundary input afresh at every time step.
        It is required when the map has unpaired ports. ``initial_state``
        prepares the feedback memory for the power method; the eigen method
        computes a unique stationary memory independently of this choice.
        Remaining parameters are those of :meth:`optyx.channel.Diagram.fix`.

        >>> import numpy as np
        >>> from optyx.channel import qubit
        >>> from optyx.core.backends import DiscopyBackend
        >>> from optyx.qubits import Ket
        >>> wire = Box("wire", qubit, qubit ** 2, Diagram.id(qubit ** 3))
        >>> cmap = CMap([wire], [((0, 1), (0, 2))])
        >>> result = cmap.fix(
        ...     Ket(1), Ket(0) @ Ket(0), n_steps=2,
        ...     backend=DiscopyBackend())
        >>> assert np.allclose(result.density_matrix, [[0, 0], [0, 1]])
        """
        if input_state is None:
            if self.dom:
                raise ValueError(
                    "input_state is required for a map with a boundary.")
            input_state = Diagram.id()
        if (input_state.dom, input_state.cod) != (Ty(), self.dom):
            raise AxiomError(
                f"input_state must be a state of type {self.dom}.")
        protocol = self.step.feedback(
            dom=self.dom, cod=self.cod, mem=self.memory,
            initial_state=initial_state)
        return (input_state >> protocol).fix(**params)

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

    def glue(self, *edges) -> CMap:
        """
        The map with extra edges, gluing boundary ports together: this is
        composition in the compact closed structure, and a single edge
        between two boundary ports of the same map is a cup or a cap.

        >>> from optyx.channel import qubit
        >>> from optyx.qubits import Z, X, Scalar
        >>> cnot = Z(1, 2) @ qubit >> qubit @ X(2, 1) @ Scalar(2 ** 0.5)
        >>> box = Box("f", qubit, qubit, cnot)
        >>> cmap = CMap([box, box], []).glue(((0, 1), (1, 0)))
        >>> assert cmap.memory == qubit ** 2
        """
        return CMap(self.boxes, self.edges + list(edges))

    def __repr__(self):
        return f"CMap({repr(self.boxes)}, {repr(self.edges)})"

    def __eq__(self, other):
        return isinstance(other, CMap) and (
            self.boxes, self.edges) == (other.boxes, other.edges)
