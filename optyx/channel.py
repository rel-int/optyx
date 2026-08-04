"""

Overview
--------

Implements classical-quantum channels.

Quantum channels are completely positive maps acting on
the doubled space :code:`H @ H` for a Hilbert space :code:`H`.
These can be initialised from the Kraus decomposition,
given as an :code:`diagram.Diagram` with domain :code:`H` and
codomain :code:`H @ E` for an auxiliary space :code:`E`,
called the environment, which is not observed.

Channels can moreover have a classical interface,
in the form of input :code:`bit` or :code:`mode` types.
The Kraus map is then given by an :class:`diagram.Diagram`
with domain :code:`H @ C` and codomain :code:`H @ C @ E`,
where the classical type :code:`C` represents
the classical inputs or outputs of the computation.
In the doubled picture, encoding or measuring a classical type
is implemented through instances of :class:`diagram.Spider`.

This module allows to build an arbitrary syntactic :class:`Diagram`
from instances of :class:`Channel`.
The :code:`Diagram.double` method returns an :class:`diagram.Diagram`,
whose tensor evaluation gives all the relevant statistics of the circuit.
A channel doubles into its Kraus map beside the conjugate, with the
environment bent round and shared:

>>> from discopy.symmetric import Equation
>>> from optyx import photonic
>>> Equation(photonic.PhotonLoss(.5), photonic.PhotonLoss(.5).double(),
...          symbol="$\\mapsto$").draw(
...     figsize=(9, 3), path="docs/_static/doubling.png")

.. image:: /_static/doubling.png
    :align: center

Stateful channels
-----------------

A channel diagram is **stateful** when :meth:`Diagram.feedback` closes its
last outputs back into its last inputs, one time step later. The type carried
between two steps is the **memory**:

>>> loop = (photonic.Create(1) @ qmode >> photonic.MZI(.06, 0)).feedback(
...     mem=qmode, initial_state=photonic.Create(0))
>>> assert (loop.dom, loop.cod, loop.mem) == (Ty(), qmode, qmode)
>>> loop.draw(figsize=(4, 3), path="docs/_static/feedback.png")

.. image:: /_static/feedback.png
    :align: center

**Stream semantics.** :meth:`Diagram.stream` gives the stream and the ordered
:class:`diagram.FeedbackBoundary` of each loop; :meth:`Diagram.now` opens the
loops again into the one-step process from :code:`dom @ mem` to
:code:`cod @ mem`, and :meth:`Diagram.unroll` plugs the boundaries in at the
first and last of `n` steps. The boundary is the memory wire only — the
inputs and outputs of each tick stay open:

>>> stream, (boundary,) = loop.stream()
>>> assert boundary.mem == qmode and boundary.final_effect is None
>>> assert loop.now() == stream.now
>>> Equation(loop, loop.unroll(2), symbol="$\\mapsto$").draw(
...     figsize=(11, 4), path="docs/_static/unroll.png")

.. image:: /_static/unroll.png
    :align: center

The memory is a *delay*, which makes this the feedback of a monoidal stream
(Di Lavore, de Felice and Román, LICS 2022) rather than a trace (Katis,
Sabadini and Walters, 2002): the loop unrolls a step at a time instead of
being closed by a fixed point.

**Fixpoints.** That fixed point is the other semantics.
:meth:`Diagram.at_time` closes the loop for `n` steps and keeps only the last
output; :meth:`Diagram.fix` approximates the state it settles into, by
contracting an unrolling (:code:`method="power"`) or by diagonalising the
transfer matrix of one step (:code:`method="eigen"`):

>>> import numpy as np
>>> measured = loop >> photonic.NumberResolvingMeasurement(1)
>>> settled = measured.at_time(8).eval().prob_dist()
>>> fixed = measured.fix(method="eigen", chi=8).prob_dist()
>>> assert max(abs(settled[k] - v) for k, v in fixed.items()) < 1e-6
>>> Equation(loop.at_time(2), symbol="").draw(
...     figsize=(6, 3), path="docs/_static/at_time.png")

.. image:: /_static/at_time.png
    :align: center

:meth:`Diagram.stationary_state` is the solver underneath, defined for any
loop-free endomorphism, and :meth:`Diagram.unroll_depth` turns a loss per
round trip into a depth which is a guarantee rather than a search.

See :doc:`/notebooks/fixpoints` for what each of these returns, and for
using them to simulate feedback boson sampling.

Types
-----

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Ob
    Ty

Generators and diagrams
------------------------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Diagram
    Channel
    Measure
    Encode
    Discard
    Feedback

Examples
--------

A Channel is initialised by its Kraus map from `dom` to `cod @ env`.

>>> from optyx.core import zx, zw, diagram
>>> from optyx import photonic
>>> circ = (
...     photonic.Phase(0.25) @
...     photonic.BS @
...     photonic.Phase(0.56) >>
...     photonic.BS @ photonic.BS
... ).get_kraus()
>>> channel = Channel(name='circuit', kraus=circ,\\
...                   dom=qmode ** 4, cod=qmode ** 4, env=diagram.Ty())

We can check that this channel is causal:

>>> import numpy as np
>>> discards = Discard(qmode ** 4)
>>> rhs = (channel >> discards).double().to_tensor().eval().array
>>> lhs = (discards).double().to_tensor().eval().array
>>> assert np.allclose(lhs, rhs)

We can calculate the probability of an input-output pair:

>>> state = Channel('state', zw.Create(1, 0, 1, 0))
>>> effect = Channel('effect', zw.Select(1, 0, 1, 0))
>>> prob = (state >> channel >> effect).double(\\
...     ).to_tensor().eval().array
>>> amp = (zw.Create(1, 0, 1, 0) >> circ >> zw.Select(1, 0, 1, 0)\\
...     ).to_tensor().eval().array
>>> assert np.allclose(prob, np.absolute(amp) ** 2)

We can check that the probabilities of a normalised state sum to 1:

>>> bell_state = Channel('Bell', diagram.Scalar(1/np.sqrt(2)) @ zx.Z(0, 2))
>>> dual_rail = Channel('2R', diagram.dual_rail(2))
>>> measure = Discard(qmode ** 3) @ Measure(qmode)
>>> setup = bell_state >> dual_rail >> channel >> measure
>>> assert np.isclose(sum(setup.double().to_tensor().eval().array), 1)

We can construct a lossy optical channel and compute its probabilities:

>>> eff = 0.95
>>> kraus = zw.W(2) >> zw.Endo(np.sqrt(eff)) @ zw.Endo(np.sqrt(1 - eff))
>>> loss = Channel(str(eff), kraus, dom=qmode, cod=qmode, env=diagram.mode)
>>> uniform_loss = loss.tensor(*[loss for _ in range(3)])
>>> lossy_channel = channel >> uniform_loss
>>> lossy_prob = (state >> lossy_channel >> effect).double(\\
...     ).to_tensor().eval().array
>>> assert np.allclose(lossy_prob, prob * (eff ** 2))

**Diagrams from Bosonic Operators**

The :code:`from_bosonic_operator` method
supports creating :class:`path` diagrams:

>>> from optyx.core.zw import Split, Select, Id
>>> from optyx.core.diagram import Mode
>>> from optyx.photonic import Scalar
>>> d1 = Diagram.from_bosonic_operator(
...     n_modes=2,
...     operators=((0, False), (1, False), (0, True)),
...     scalar=2.1
... )

>>> annil = Channel(
...     "annil", Split(2) >> Select(1) @ Id(Mode(1))
... )
>>> create = annil.dagger()

>>> d2 = Scalar(2.1) @ annil @ qmode >> \\
... qmode @ annil >> create @ qmode

>>> assert d1 == d2

We can map ZX diagrams to :class:`path` diagrams using
dual-rail encoding. For example, we can create a GHZ state:

>>> from discopy.drawing import Equation
>>> from optyx.qubits import Z
>>> from optyx.photonic import DualRail
>>> ghz = Z(0, 3)
>>> ghz_path = ghz.to_dual_rail()
>>> Equation(ghz >> DualRail(3), ghz_path, \\
... symbol="$\\mapsto$").draw(figsize=(10, 10), \\
... path="docs/_static/ghz_dr.svg")

.. image:: /_static/ghz_dr.svg
    :align: center

"""

from __future__ import annotations

from importlib import import_module
import warnings
from numbers import Integral, Real

from cotengra import (
    HyperCompressedOptimizer,
    ReusableHyperCompressedOptimizer,
)
import numpy as np
from discopy import tensor
from discopy import monoidal, symmetric, frobenius, hypergraph
from discopy.cat import factory
from discopy.utils import AxiomError
from pytket.extensions.pyzx import pyzx_to_tk
from pyzx import extract_circuit
from optyx.core import diagram


class Ob(frobenius.Ob):
    """Basic object: bit, mode, qubit or qmode"""

    _classical = {
        "bit": "bit",
        "mode": "mode",
        "qubit": "bit",
        "qmode": "mode",
    }
    _quantum = {
        "bit": "qubit",
        "mode": "qmode",
        "qubit": "qubit",
        "qmode": "qmode",
    }

    @property
    def is_classical(self):
        """Classical objects are :code:`bit` and :code:`mode`."""
        return self.name not in ["qubit", "qmode"]

    @property
    def single(self):
        """Maps :code:`qubit` to :code:`diagram.bit`
        and :code:`qmode` to :code:`diagram.mode`."""
        return diagram.Ty(self._classical[self.name])

    @property
    def double(self):
        """Maps :code:`qubit` to :code:`diagram.bit @ diagram.bit`
        and :code:`qmode` to :code:`diagram.mode @ diagram.mode`."""
        if self.is_classical:
            return diagram.Ty(self.name)
        name = self._classical[self.name]
        return diagram.Ty(name, name)


MAX_UNROLL = 64
MAX_TRUNCATION = 64


@factory
class Ty(frobenius.Ty):
    """Classical and quantum types."""

    generator_factory = Ob

    def single(self):
        """Returns the diagram.Ty obtained by mapping
        :code:`qubit` to :code:`bit` and :code:`qmode` to :code:`mode`"""
        return diagram.Ty().tensor(*[ob.single for ob in self.inside])

    def double(self):
        """Returns the diagram.Ty obtained by mapping
        :code:`qubit` to :code:`bit @ bit`
        and :code:`qmode` to :code:`mode @ mode`"""
        return diagram.Ty().tensor(*[ob.double for ob in self.inside])

    @staticmethod
    # pylint: disable=invalid-name
    def from_optyx(ty):
        """
        Get quantum types from core/diagram.Ty.
        """
        assert isinstance(ty, diagram.Ty)
        # pylint: disable=protected-access
        return Ty(*[Ob._quantum[ob.name] for ob in ty.inside])

    def needs_inflation(self) -> bool:
        """
        Diagrams with at least one :code:`qmode` need inflation.
        """
        return any(ob.name == "qmode" for ob in self.inside)

    # pylint: disable=invalid-name
    def inflate(self, d) -> Ty:
        """
        Inflate the type.
        """
        return (mode**0).tensor(
                *(o**d if o.needs_inflation() else o for o in self)
        )

    def double_axes(self):
        """
        The positions in :meth:`double` of the classical wires and of the
        ket/bra pairs of the quantum ones. A classical object doubles to one
        wire, a quantum object to two.

        >>> assert (bit @ qubit).double_axes() == ([0], [(1, 2)])
        """
        classical, quantum, position = [], [], 0
        for ob in self.inside:
            if ob.is_classical:
                classical.append(position)
                position += 1
            else:
                quantum.append((position, position + 1))
                position += 2
        return classical, quantum

    def dagger_axes(self):
        """
        The permutation of :meth:`double` exchanging ket and bra on every
        quantum wire. Transposing a doubled array along it is the dagger,
        so a state is Hermitian when it is fixed by that transposition.

        >>> assert (bit @ qubit).dagger_axes() == [0, 2, 1]
        """
        axes = list(range(len(self.double())))
        for row, column in self.double_axes()[1]:
            axes[row], axes[column] = column, row
        return axes


bit = Ty("bit")
mode = Ty("mode")
qubit = Ty("qubit")
qmode = Ty("qmode")


@factory
class Diagram(frobenius.Diagram):
    """Classical-quantum circuits over qubits and optical modes"""

    ob = Ty
    grad = tensor.Diagram.grad
    unroll = diagram.Diagram.unroll
    boundary = diagram.Diagram.boundary
    one_step = diagram.Diagram.one_step

    def feedback(self, dom=None, cod=None, mem=None,
                 state=None, effect=None) -> Diagram:
        """
        Feed the last `mem` outputs of a channel diagram from `dom @ mem`
        to `cod @ mem` back into its last `mem` inputs,
        one time step later.

        Parameters:
            dom : The domain of the result, `arg.dom[:-len(mem)]` by default.
            cod : The codomain of the result, `arg.cod[:-len(mem)]`
                by default.
            mem : The memory type fed back, `arg.cod[-1:]` by default.
            state : The boundary plugged in the input memory before the
                first time step of :meth:`unroll`, a diagram with
                `cod == mem`. `None` leaves the wire open; there is no
                natural state to default to, though on `qmode` the
                zero-photon state `photonic.Create(0)` is the usual choice.
            effect : The boundary plugged in the output memory after the
                last time step, a diagram with `dom == mem`. `None` gives
                `Discard(mem)`, which traces the memory out.

        A channel diagram is the one place where a boundary has a natural
        default: discarding is the unique causal effect, so
        :meth:`Feedback.default_effect` is `Discard(mem)` here while
        :meth:`Feedback.default_state` stays the open wire.

        >>> from optyx.photonic import Create
        >>> wait = Diagram.swap(qmode, qmode).feedback(state=Create(0))
        >>> assert wait.dom == wait.cod == qmode
        >>> assert wait.mem == qmode
        >>> assert wait.state == Create(0)
        >>> assert wait.effect == Discard(qmode)

        The result composes with any other channel diagram and is
        unrolled by :meth:`unroll`.
        """
        return self.feedback_factory(
            self, dom=dom, cod=cod, mem=mem, state=state, effect=effect)

    def now(self) -> Diagram:
        """
        Open every feedback loop to obtain one time step of the stateful
        diagram, from ``dom @ memory`` to ``cod @ memory``.

        This reuses :meth:`stream`, so the memory order is exactly the
        order used by :meth:`unroll`. Boundary ``initial_state`` and
        ``final_effect`` values are metadata and do not change this channel.

        It is a method rather than a property for the same reason
        :meth:`stream` is: it rebuilds the stream on every call. The
        ``now`` of a :class:`discopy.stream.Stream` is a field of a value
        already built, and costs nothing to read.

        >>> from optyx.photonic import Create
        >>> wait = Diagram.swap(qmode, qmode).feedback(
        ...     initial_state=Create(0))
        >>> assert wait.now() == Diagram.swap(qmode, qmode)
        """
        stream, _ = self.stream()
        return stream.now

    def at_time(self, n_steps: int) -> Diagram:
        """
        The state of a stateful diagram after `n_steps` time steps: every
        output before the last one and the final memory are discarded.

        The diagram must be a state and every feedback loop needs an
        `initial_state`. A loop's `final_effect` belongs to finite unrolling
        and is deliberately ignored here: stationary-state simulation traces
        out the memory rather than conditioning on it.

        >>> from optyx.qubits import Ket
        >>> source = (Discard(qubit) @ Ket(0) @ Ket(0)).feedback(
        ...     mem=qubit, initial_state=Ket(1))
        >>> assert source.at_time(3).dom == Ty()
        >>> assert source.at_time(3).cod == qubit
        """
        if self.dom:
            raise ValueError(
                "at_time builds a state, so the diagram must have an empty "
                f"domain, got dom={self.dom}.")
        if not isinstance(n_steps, Integral) or isinstance(n_steps, bool) \
                or n_steps < 1:
            raise ValueError("n_steps must be a positive integer.")
        semantics = self.stream()
        if any(boundary.initial_state is None
               for boundary in semantics.boundaries):
            raise ValueError(
                "Every feedback loop needs an initial_state.")
        initial = self.id(Ty()).tensor(*(
            boundary.initial_state for boundary in semantics.boundaries))
        unrolled = semantics.stream.unroll(n_steps - 1).now
        past = Discard(self.cod ** (n_steps - 1)) \
            if n_steps > 1 else self.id(Ty())
        memory = semantics.stream.mem.now
        return initial >> unrolled \
            >> past @ self.id(self.cod) @ Discard(memory)

    def normalisation(self, state=None, dimensions=None):
        """
        The scalar ``state >> self >> Discard(cod)``: feed ``self`` with
        ``state`` and discard every output. For a state this is its trace,
        and it is the factor which rescales ``self`` to a causal process.

        ``state`` is the array of a state over ``dom``, for the case where
        it comes out of a linear solve rather than out of a diagram. It is
        only optional when ``self`` is already a state.

        >>> from optyx.qubits import Ket
        >>> assert np.isclose(Ket(0).normalisation(), 1)
        >>> assert np.isclose((Scalar(.5) @ Ket(0)).normalisation(), .25)

        A process needs a state, since without one there is no scalar:

        >>> try:
        ...     Diagram.id(qubit).normalisation()
        ... except ValueError as error:
        ...     assert str(error) == (
        ...         "normalisation is a scalar, so it needs a state over "
        ...         "dom=qubit.")
        >>> assert np.isclose(
        ...     Diagram.id(qubit).normalisation([[.5, 0], [0, .5]]), 1)
        """
        if state is None:
            if self.dom != Ty():
                raise ValueError(
                    "normalisation is a scalar, so it needs a state over "
                    f"dom={self.dom}.")
            return (self >> Discard(self.cod)).double().to_tensor(
                dimensions).eval().array
        discarded = (self >> Discard(self.cod)).double().to_tensor(
            list(np.shape(state)))
        return (tensor.Box(
            "State", tensor.Dim(1), discarded.dom, np.asarray(state))
            >> discarded).eval().array

    def is_causal(self, dimensions=None, tol: float = 1e-6):
        """
        A process is causal when discarding its outputs is the same as
        discarding its inputs, ``self >> Discard(cod) == Discard(dom)``.
        For a state this says the trace is one, i.e. that it is normalised.

        Rescaling a process is tensoring it with a scalar, and that is
        exactly what breaks causality:

        >>> from optyx.qubits import Ket
        >>> assert Ket(0).is_causal()
        >>> assert not (Scalar(.5) @ Ket(0)).is_causal()
        >>> assert np.isclose((Scalar(.5) @ Ket(0)).normalisation(), .25)
        """
        left = (self >> Discard(self.cod)).double().to_tensor(
            dimensions).eval().array
        right = Discard(self.dom).double().to_tensor(
            dimensions).eval().array
        return bool(np.linalg.norm(left - right) <= tol)

    def is_hermitian(self, state, tol: float = 1e-6):
        """
        A state over ``cod`` is Hermitian when transposing it along
        :meth:`Ty.dagger_axes` gives its conjugate back.
        """
        array = np.asarray(state)
        return bool(np.allclose(
            array, np.transpose(array.conjugate(), self.cod.dagger_axes()),
            atol=tol, rtol=0))

    def is_positive(self, state, tol: float = 1e-6):
        """
        A state over ``cod`` is positive when every classical block of it is
        a positive semidefinite matrix on the ket/bra pairs. Positivity is
        only defined for a Hermitian state, so the Hermitian part is what
        gets diagonalised; pair this with :meth:`is_hermitian`.

        This diagonalises one matrix per classical outcome, so it costs as
        much as the eigensolve which produced the state: call it to check a
        result, not on the path of an evaluation.
        """
        array = np.asarray(state)
        array = (array + np.transpose(
            array.conjugate(), self.cod.dagger_axes())) / 2
        classical, quantum = self.cod.double_axes()
        rows, columns = map(list, zip(*quantum)) if quantum else ([], [])
        blocks = np.transpose(array, classical + rows + columns).reshape(
            int(np.prod([array.shape[axis] for axis in classical], dtype=int)),
            int(np.prod([array.shape[axis] for axis in rows], dtype=int)),
            int(np.prod([array.shape[axis] for axis in rows], dtype=int)))
        return bool(
            min(np.linalg.eigvalsh(block).min() for block in blocks) >= -tol)

    def stationary_state(
            self, dimensions: list[int] = None, tol: float = 1e-6):
        """Return the unique stationary state of a loop-free endomorphism.

        The diagram denotes the transfer operator; its doubled tensor is
        projected back to ``dimensions`` before the eigenspace at eigenvalue
        one is found, then the eigenvector is divided by its
        :meth:`normalisation`.

        Being a density matrix does not follow from causality, since this
        method accepts any loop-free endomorphism rather than only a channel.
        It is not checked here: :meth:`is_hermitian` and :meth:`is_positive`
        cost a diagonalisation, so they are exposed for a caller to run on a
        result rather than spent on every evaluation.

        Parameters:
            dimensions : Dimensions of the doubled input wires, all two by
                default. Optical simulations typically use ``cutoff`` for
                every mode in ``self.dom.double()``.
            tol : Tolerance for causality of the truncated transfer map.

        >>> from optyx import classical
        >>> transition = classical.ClassicalBox(
        ...     "Transition",
        ...     diagram.Box("Transition", diagram.bit, diagram.bit,
        ...                 array=np.array([[.9, .1], [.4, .6]])),
        ...     bit, bit)
        >>> state = transition.stationary_state()
        >>> assert np.allclose(state, [.8, .2])

        Both the fixed-point equation and the trace are compositions: the
        state is stationary when post-composing it with the transfer operator
        gives it back, and its trace is what post-composing it with
        :class:`Discard` evaluates to. Drawn together, the two equations the
        method solves and then checks:

        >>> from discopy.drawing import Equation
        >>> fixed = classical.ClassicalBox(
        ...     r"$\\rho_\\star$",
        ...     diagram.Box(r"$\\rho_\\star$", diagram.Ty(), diagram.bit,
        ...                 array=state),
        ...     Ty(), bit)
        >>> Equation(fixed >> transition, fixed,
        ...          fixed >> Discard(bit), symbol="$=$").draw(
        ...     figsize=(8, 3), path="docs/_static/stationary_state.svg")
        >>> assert np.isclose(state.sum(), 1)

        .. image:: /_static/stationary_state.svg
            :align: center

        A non-unique stationary state is a design choice, not an arbitrary
        eigensolver choice:

        >>> try:
        ...     Diagram.id(bit).stationary_state()
        ... except ValueError as error:
        ...     assert str(error) == (
        ...         "The stationary state is not unique "
        ...         "(fixed-space dimension 2).")
        """
        if any(isinstance(box, Feedback) for box in self.boxes):
            raise ValueError(
                "stationary_state is defined for diagrams without feedback "
                "loops; call now first.")
        if self.dom != self.cod:
            raise ValueError(
                "stationary_state is defined for endomorphisms, got "
                f"{self.dom} -> {self.cod}.")
        if not isinstance(tol, Real) or isinstance(tol, bool) \
                or not np.isfinite(tol) or tol <= 0:
            raise ValueError("tol must be a positive finite real number.")

        if dimensions is None:
            dimensions = [2] * len(self.dom.double())
        if len(dimensions) != len(self.dom.double()) or any(
                not isinstance(value, Integral) or isinstance(value, bool)
                or value <= 0 for value in dimensions):
            raise ValueError(
                "dimensions must contain one positive integer per doubled "
                "input wire.")
        dimensions = [int(value) for value in dimensions]
        tensor_transfer = self.double().to_tensor(dimensions)
        operator = tensor_transfer >> tensor.Diagram.tensor(*(
            diagram.EmbeddingTensor(source.inside[0], target)
            for source, target in zip(tensor_transfer.cod, dimensions)))
        matrix = operator.eval().array.reshape(
            int(np.prod(dimensions, dtype=int)), -1)

        discard = Discard(self.dom).double().to_tensor(dimensions)
        trace_residual = np.linalg.norm(
            (operator >> discard).eval().array - discard.eval().array)
        if trace_residual > tol:
            raise ValueError(
                "The truncated transfer map is not causal "
                f"(residual {trace_residual}); increase cutoff.")

        eigenvalues, eigenvectors = np.linalg.eig(matrix.T)
        eigen_tolerance = 100 * len(matrix) * np.finfo(float).eps \
            * max(1, np.linalg.norm(matrix))
        fixed = np.flatnonzero(abs(eigenvalues - 1) <= eigen_tolerance)
        if not fixed.size:
            raise ValueError(
                "The truncated transfer map has no stationary state; "
                "increase cutoff or check that the channel is trace "
                "preserving.")
        if len(fixed) != 1:
            raise ValueError(
                "The stationary state is not unique "
                f"(fixed-space dimension {len(fixed)}).")
        state = eigenvectors[:, fixed[0]]
        residual = np.linalg.norm(state @ matrix - state)
        if residual > eigen_tolerance:
            raise ValueError(
                f"The stationary-state residual {residual} is too large.")
        state = state.reshape(dimensions)
        trace = self.id(self.cod).normalisation(state)
        if not np.isfinite(trace) or abs(trace) <= tol:
            raise ValueError(
                "The stationary state has zero or non-finite trace.")
        return np.real_if_close(state / trace)

    def unroll_depth(self, tol: float = 1e-6, loss: float = 0):
        """
        The number of time steps at which unrolling approximates the fixed
        point within ``tol``.

        A memory which loses a fraction ``loss`` of itself per round trip
        forgets at least that much whatever the rest of the loop does, which
        bounds the second eigenvalue modulus of the transfer channel by the
        transmissivity :math:`\\gamma = 1 - loss` uniformly over diagrams —
        see :doc:`/notebooks/fixpoints`, where the bound is proved and
        sharpened to an equality. So

        .. math:: n^\\star
            = \\lceil\\log(tol) / \\log(\\gamma)\\rceil

        is a depth rather than a guess, and it does not grow with the memory.
        Without loss no finite depth is guaranteed: the gap depends on the
        diagram and can be arbitrarily close to one, so :meth:`fix` searches
        instead.

        >>> loop = Diagram.swap(qmode, qmode).feedback(
        ...     initial_state=Discard(Ty()))
        >>> assert loop.unroll_depth(1e-6, loss=.5) == 20
        >>> assert loop.unroll_depth(1e-6, loss=.05) == 270
        """
        if not isinstance(loss, Real) or isinstance(loss, bool) \
                or not 0 < loss < 1:
            raise ValueError(
                "A depth is only guaranteed for a loss in (0, 1), got "
                f"loss={loss}.")
        return int(np.ceil(np.log(tol) / np.log(1 - loss)))

    def truncation_dimension(self, tol: float = 1e-6,
                             maximum: int = MAX_TRUNCATION):
        """
        The smallest memory dimension at which the transfer map of one time
        step is still causal within ``tol``.

        Truncating each memory wire to a finite dimension throws away the
        weight above it, so the truncated map discards more than it should
        and :meth:`stationary_state` refuses it. Doubling from two until
        :meth:`is_causal` holds gives the dimension `"eigen"` needs, and
        qubit memories stop at the first step.

        >>> from optyx.qubits import Ket
        >>> source = (Discard(qubit) @ Ket(0) @ Ket(0)).feedback(
        ...     mem=qubit, initial_state=Ket(1))
        >>> assert source.truncation_dimension() == 2
        """
        step = self.now()
        memory = step.cod[len(self.cod):]
        transfer = step >> Discard(self.cod) @ self.id(memory)
        dimension = 2
        while dimension <= maximum:
            dimensions = [
                dimension if ob.inside[0].name == "mode" else 2
                for ob in memory.double()]
            if transfer.is_causal(dimensions, tol):
                return dimension
            dimension *= 2
        raise ValueError(
            f"No memory dimension below {maximum} makes the transfer map "
            f"causal within tol={tol}.")

    def fix(self, n_steps: int = None, chi: int = None, *,
            method: str = "power", tol: float = 1e-6, loss: float = 0,
            backend=None):
        """
        Approximate a stationary state of a stateful diagram as a density
        matrix over its codomain.

        A diagram with a feedback loop has stream semantics through
        :meth:`stream` and :meth:`unroll`, and approximate fixed-point
        semantics through :meth:`fix`. Both use the same open one-step
        process; the fixed-point semantics evolves an approximate stationary
        memory once, then discards the next memory and returns only the visible
        output.

        The stationary state is the fixed point of the transfer channel
        which one time step induces on the memory of the feedback loops.
        It is the limit of :meth:`at_time` only when iteration converges;
        periodic channels can have a fixed state while their iterates cycle.

        Parameters:
            n_steps : The number of time steps to unroll, `"power"` only.
                Given by :meth:`unroll_depth` when `None`.
            chi : The dimension at which the approximation is truncated:
                the bond dimension of a compressed contraction for
                `"power"`, the dimension of each memory wire for `"eigen"`.
                `"power"` contracts exactly when it is `None`, `"eigen"`
                takes it from :meth:`truncation_dimension`.
            method : Either `"power"`, contracting the unrolling as a tensor
                network, or `"eigen"`, diagonalising the transfer matrix,
                see below.
            tol : The error to approximate the fixed point within.
            loss : The fraction of the memory lost per round trip. A
                positive loss makes the depth a guarantee rather than a
                search, see :meth:`unroll_depth`.
            backend : An optional
                :class:`optyx.core.backends.AbstractBackend` used by the
                `"power"` method. DisCoPy evaluates with ``tensor.Functor``;
                Quimb can contract exactly or use Cotengra compression
                bounded by `chi`.

        The `"power"` method unrolls the diagram and contracts the doubled
        tensor network, exactly or with bonds bounded by `chi`: this is the
        power iteration on the transfer channel. The `"eigen"` method builds
        the transfer matrix of one time step and diagonalises it, exact and
        cheaper whenever `chi ** len(memory)` is small, with no `n_steps` at
        all.

        There is one truncation dimension and one depth, so `chi` and
        `n_steps` are the values to use where they apply and the caps where
        they are searched for; there are no separate maxima.

        A loop which reprepares its memory at every time step forgets its
        initial state, so it is its own stationary state:

        >>> from optyx.qubits import Ket
        >>> source = (Discard(qubit) @ Ket(0) @ Ket(0)).feedback(
        ...     mem=qubit, initial_state=Ket(1))
        >>> fixed = source.fix(method="eigen")
        >>> assert np.allclose(
        ...     fixed.density_matrix, [[1, 0], [0, 0]])

        See :doc:`/notebooks/fixpoints` for the semantic diagram, agreement
        map and contraction planning.
        """
        if self.dom:
            raise ValueError(
                "The stationary state of a diagram with a domain is not "
                f"defined, got dom={self.dom}.")
        if not any(isinstance(box, Feedback) for box in self.boxes):
            raise ValueError(
                "The diagram has no feedback loop, so it is already its "
                "own stationary state.")
        if method not in ("power", "eigen"):
            raise ValueError(
                f"Unknown method {method}, use 'power' or 'eigen'.")
        for name, value in {"n_steps": n_steps, "chi": chi}.items():
            if value is None:
                continue
            if not isinstance(value, Integral) or isinstance(value, bool) \
                    or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if not isinstance(tol, Real) or isinstance(tol, bool) \
                or not np.isfinite(tol) or tol <= 0:
            raise ValueError("tol must be a positive finite real number.")
        if not isinstance(loss, Real) or isinstance(loss, bool) \
                or not 0 <= loss < 1:
            raise ValueError("loss must be a real number in [0, 1).")
        if method == "eigen":
            if backend is not None:
                raise ValueError("backend is only used by method='power'.")
            return self.eigen_fix(
                self.truncation_dimension(tol) if chi is None else chi, tol)
        return self.power_fix(n_steps, chi, tol, loss, backend)

    def power_fix(self, n_steps, chi, tol, loss, backend):
        """Approximate the stationary output by contracting an unrolling.

        Each finite-time channel is evaluated through the existing backend
        interface. DisCoPy uses :class:`tensor.Functor`; Quimb converts the
        same doubled channel to a tensor network and may ask Cotengra for
        exact or compressed contraction. The loop depends only on
        :meth:`optyx.core.backends.AbstractBackend.eval`.

        :meth:`fix` is the public validated dispatcher. ``chi`` bounds the
        bond dimensions when given and the contraction is exact when it is
        not. ``n_steps`` is taken from :meth:`unroll_depth` when the loop is
        lossy, and searched for by doubling otherwise, since without loss no
        depth is guaranteed.

        The two local functions below are closures rather than methods on
        purpose. ``contract`` memoises evaluations in a cache that lives for
        the duration of one call and must not be shared between calls with
        different backends, and both read ``backend``, ``chi`` and ``tol``.
        As methods they would take those as arguments and the cache would
        have to live on the diagram, which is immutable.
        """
        backends = import_module("optyx.core.backends")

        if backend is None:
            backend = backends.QuimbBackend(
                hyperoptimiser=None if chi is None
                else HyperCompressedOptimizer())
        elif not isinstance(backend, backends.AbstractBackend):
            raise ValueError(
                "backend must implement the AbstractBackend interface.")

        compressed = chi is not None and isinstance(
            getattr(backend, "hyperoptimiser", None),
            (ReusableHyperCompressedOptimizer, HyperCompressedOptimizer))
        result_cache = {}

        def contract(steps):
            if steps not in result_cache:
                extra = {"max_bond": chi} if compressed else {}
                result_cache[steps] = self.at_time(steps).eval(
                    backend, **extra)
            return result_cache[steps]

        def normalised(result):
            state = np.asarray(result.density_matrix)
            trace = self.id(self.cod).normalisation(state)
            trace_tolerance = 100 * state.size * np.finfo(float).eps
            if not np.isfinite(trace) or abs(trace) <= trace_tolerance:
                raise ValueError(
                    "Contraction returned zero or non-finite trace.")
            return state / trace

        depth = n_steps if n_steps is not None else (
            self.unroll_depth(tol, loss) if loss else None)
        if depth is not None:
            result = contract(depth)
        else:
            depth, result = 2, contract(2)
            while np.linalg.norm(
                    normalised(result)
                    - normalised(contract(depth - 1))) >= tol:
                if depth >= MAX_UNROLL:
                    warnings.warn(
                        f"fix did not converge to tol={tol} within "
                        f"{MAX_UNROLL} steps; give n_steps or a positive "
                        "loss.", UserWarning, stacklevel=3)
                    break
                depth = min(2 * depth, MAX_UNROLL)
                result = contract(depth)
        return backends.EvalResult(
            tensor.Box(
                "Result", tensor.Dim(1), result.tensor.cod,
                normalised(result)),
            output_types=self.cod, state_type=backends.StateType.DM)

    def eigen_fix(self, chi: int = 2, tol: float = 1e-6):
        """
        The stationary state obtained by diagonalising the transfer matrix
        which one time step induces on the memory.

        ``chi`` is the local dimension of each optical mode, spanning
        occupations ``0`` through ``chi - 1``. Qubit dimensions remain
        two. Fresh photons may enlarge the output memory axes, which are
        projected back to the same cutoff before diagonalisation; excessive
        lost trace raises. A non-unique stationary memory state raises rather
        than choosing an arbitrary eigenvector.
        """
        step = self.now()
        memory = step.cod[len(self.cod):]
        dimensions = [
            chi if ob.inside[0].name == "mode" else 2
            for ob in memory.double()]
        transfer = step >> Discard(self.cod) @ self.id(memory)
        readout = step >> self.id(self.cod) @ Discard(memory)
        state = transfer.stationary_state(dimensions, tol)
        tensor_diagram = readout.double().to_tensor(dimensions)
        memory_state = tensor.Box(
            "Stationary memory", tensor.Dim(1), tensor_diagram.dom, state)
        density_matrix = (memory_state >> tensor_diagram).eval().array
        backends = import_module("optyx.core.backends")
        return backends.EvalResult(
            tensor.Box(
                "Result", tensor.Dim(1), tensor_diagram.cod,
                np.real_if_close(density_matrix)),
            output_types=self.cod, state_type=backends.StateType.DM)

    def needs_inflation(self) -> bool:
        """
        If the domain or codomain need inflation,
        the diagram needs inflation.
        """
        return self.dom.needs_inflation() or self.cod.needs_inflation()

    # pylint: disable=invalid-name
    def inflate(self, d):
        r"""Translates from an indistinguishable setting
        to a distinguishable one. For a map on :math:`F(\mathbb{C})`,
        obtain a map on :math:`F(\mathbb{C})^{\widetilde{\otimes} d}`."""
        assert isinstance(d, int), "Dimension must be an integer"
        assert d > 0, "Dimension must be positive"

        return frobenius.Functor(
            lambda x: x.inflate(d),
            lambda f: f.inflate(d),
            dom=Diagram,
            cod=Diagram,
        )(self)

    def double(self):
        """Returns the diagram.Diagram obtained by
        doubling every quantum dimension
        and building the completely positive map."""
        return frobenius.Functor(
            lambda x: x.double(), lambda f: f.double(),
            dom=Diagram, cod=diagram.Diagram
        )(self)

    @property
    def is_pure(self):
        """
        Check if the diagram is pure, i.e. it does not
        contain any discards or measures acting on quantum types,
        and does not prepare quantum types from classical types.
        """
        are_layers_pure = []
        are_layers_classical = []
        for layer in self:
            generator = layer.inside[0][1]

            if isinstance(generator, Feedback) and not all(
                part.is_pure for part in (
                    generator.arg, generator.state, generator.effect)
            ):
                return False

            # if we have a discard/measure
            # acting on quantum types, it's not pure
            if (
                isinstance(generator, (Discard, Measure)) and
                any(not ty.is_classical for ty in generator.dom.inside)
            ):
                return False
            if hasattr(generator, 'env') and generator.env != diagram.Ty():
                return False

            # if we prepare quantum from classical types, it's not pure
            if (
                isinstance(generator, Encode) and
                any(ty.is_classical for ty in generator.cod.inside)
            ):
                return False

            # if we're mixing classical and quantum types, it's not pure
            are_layers_pure.append(
                any(ty.is_classical for ty in generator.cod.inside) or
                any(ty.is_classical for ty in generator.dom.inside) or
                isinstance(generator, Discard)
            )

            # assume all classical maps are pure
            are_layers_classical.append(
                all(ty.is_classical for ty in generator.cod.inside) and
                all(ty.is_classical for ty in generator.dom.inside)
            )

        return not any(are_layers_pure) or all(are_layers_classical)

    def get_kraus(self):
        """
        Obtain the Kraus map of a pure circuit.
        """
        assert self.is_pure, "Cannot get a Kraus map of non-pure circuit"
        kraus_maps = [diagram.Id(self.dom.single())]
        for layer in self:
            left = diagram.Ty().tensor(*[ty.single()
                                       for ty in layer.inside[0][0]])
            right = diagram.Ty().tensor(*[ty.single()
                                        for ty in layer.inside[0][2]])
            generator = layer.inside[0][1]

            if isinstance(generator, Swap):
                kraus_maps.append(
                    left @ diagram.Swap(generator.dom.single()[0],
                                        generator.cod.single()[1]) @ right
                )
            elif isinstance(generator, Feedback):
                kraus_maps.append(
                    left @ generator.get_kraus() @ right
                )
            else:
                kraus_maps.append(
                    left @ generator.kraus @ right
                )

        if len(kraus_maps) == 1:
            return kraus_maps[0]
        return kraus_maps[0].then(
            *kraus_maps[1:]
        )

    def to_path(self, dtype: type = complex):
        """Returns the :class:`Matrix` normal form
        of a :class:`Diagram`.
        In other words, it is the underlying matrix
        representation of a :class:`path` and :class:`photonic` diagrams."""
        # pylint: disable=import-outside-toplevel
        from optyx.core import path

        assert self.is_pure, "Diagram must be pure to convert to path."

        return frobenius.Functor(
            ob_map=len,
            ar_map=lambda f: f.get_kraus().to_path(dtype),
            cod=path.Matrix[dtype],
        )(self)

    def decomp(self):
        # pylint: disable=protected-access
        return frobenius.Functor(
            ob_map=lambda x: qubit**len(x),
            ar_map=lambda arr: arr._decomp(),
            cod=Diagram,
        )(self)

    def to_dual_rail(self):
        """Convert to dual-rail encoding."""

        assert self.is_pure, "Diagram must be pure to convert to dual rail."

        return frobenius.Functor(
            ob_map=lambda x: qmode**(2*len(x)),
            ar_map=lambda arr: arr._to_dual_rail(),
            cod=Diagram,
        )(self.decomp())

    def to_tket(self):  # pragma: no cover
        """
        Convert to tket circuit. The circuit must be a pure circuit.
        """

        assert self.is_pure, "Diagram must be pure to convert to tket."

        kraus_maps = []
        for layer in self:
            left = layer.inside[0][0]
            right = layer.inside[0][2]
            generator = layer.inside[0][1]

            kraus_maps.append(
                diagram.Bit(len(left)) @
                generator.kraus @
                diagram.Bit(len(right))
            )

        # pylint: disable=no-value-for-parameter
        return pyzx_to_tk(
            extract_circuit(
                diagram.Diagram.then(
                    *kraus_maps
                ).to_pyzx()
            ).to_basic_gates()
        )

    def to_pyzx(self):
        """Convert to PyZX circuit. The circuit must be a pure circuit."""
        assert self.is_pure, "Diagram must be pure for conversion."

        return self.get_kraus().to_pyzx()

    @classmethod
    def from_tket(cls, tket_circuit):
        """Convert from tket circuit."""
        # pylint: disable=import-outside-toplevel
        from optyx.qubits import Circuit
        return Circuit(tket_circuit)

    @classmethod
    def from_pyzx(cls, pyzx_circuit):
        """Convert from PyZX circuit."""
        # pylint: disable=import-outside-toplevel
        from optyx.qubits import Circuit
        return Circuit(pyzx_circuit)

    @classmethod
    def from_discopy(cls, discopy_circuit):
        """Convert from discopy circuit."""
        # pylint: disable=import-outside-toplevel
        from optyx.qubits import Circuit
        return Circuit(discopy_circuit)

    # @classmethod
    # def from_bosonic_operator(cls, n_modes, operators, scalar=1):
    #     return Channel(
    #         "Bosonic operator",
    #         diagram.Diagram.from_bosonic_operator(
    #             n_modes, operators, scalar=scalar
    #         )
    #     )

    @classmethod
    def from_bosonic_operator(cls, n_modes, operators, scalar=1):
        """Create a :class:`zw` diagram from a bosonic operator."""
        # pylint: disable=import-outside-toplevel
        from optyx.core import zw
        from optyx.photonic import Scalar

        # pylint: disable=invalid-name
        d = Diagram.id(qmode**n_modes)
        annil = Channel("annil", zw.Split(2) >> zw.Select(1) @ zw.Id(1))
        create = annil.dagger()
        for idx, dagger in operators:
            if not 0 <= idx < n_modes:
                raise ValueError(f"Index {idx} out of bounds.")
            box = create if dagger else annil
            d = d >> qmode**idx @ box @ qmode**(n_modes - idx - 1)

        if scalar != 1:
            # pylint: disable=invalid-name
            d = Scalar(scalar) @ d
        return d

    @classmethod
    def from_graphix(cls, measurement_pattern):
        """Convert from Graphix measurement pattern."""
        # pylint: disable=import-outside-toplevel
        from optyx.qubits import Circuit
        return Circuit(measurement_pattern)

    @classmethod
    def from_perceval(cls, p):
        """
        Convert pcvl.Circuit or pcvl.Processor
        into optyx diagrams.

        Cannot convert objects involving components
        acting on polarisation modes, time delays,
        and with symbols.
        """
        # pylint: disable=import-outside-toplevel
        from optyx import photonic
        from optyx.utils import perceval_conversion
        import perceval as pcvl

        if isinstance(p, pcvl.Circuit):
            p_ = pcvl.Processor("SLOS", p.m)
            p_.add(0, p)

            p_new = pcvl.Processor("SLOS", p.m)
            for c in p_.flatten():
                p_new.add(c[0][0], c[1])
            p = p_new

        n_modes = p.circuit_size
        circuit = photonic.Id(n_modes)
        heralds = p.heralds

        circuit = perceval_conversion.heralds_diagram(
            heralds, n_modes, circuit, "in"
        ) >> circuit

        for wires, component in p.components:
            left = circuit.cod[:min(wires)]
            right = circuit.cod[max(wires) + 1:]

            if isinstance(component, pcvl.Detector):
                box = perceval_conversion.detector(component, wires)
            elif isinstance(
                component,
                pcvl.components.feed_forward_configurator.FFCircuitProvider
            ):
                box, left, right = perceval_conversion.ff_circuit_provider(
                    component, wires, circuit
                )
            elif isinstance(
                component,
                pcvl.components.feed_forward_configurator.FFConfigurator
            ):
                box, left, right = perceval_conversion.ff_configurator(
                    component, wires, circuit
                )
            elif isinstance(component, pcvl.components.Barrier):
                continue
            elif hasattr(component, "U"):
                box = perceval_conversion.unitary(component, wires)
            else:
                raise ValueError(
                    f"Unsupported perceval component type: {type(component)}"
                )

            circuit >>= (left @ box @ right)

        circuit >>= perceval_conversion.heralds_diagram(
            heralds, n_modes, circuit, "out"
        )
        if p.post_select_fn is not None:
            circuit >>= perceval_conversion.postselection(circuit, p)
        return circuit

    # pylint: disable=invalid-name
    def __pow__(self, n):
        if n == 1:
            return self
        return self @ self ** (n - 1)

    def eval(self, backend=None, **kwargs):
        """
        Evaluate the diagram using the specified backend.
        If no backend is specified, it uses the QuimbBackend.
        """
        # pylint: disable=import-outside-toplevel
        from optyx.core.backends import QuimbBackend
        if backend is None:
            backend = QuimbBackend()

        return backend.eval(self, **kwargs)


class Channel(Diagram, frobenius.Box):
    """
    Channel initialised by its Kraus map.
    """

    def __init__(
            self,
            name,
            kraus,
            dom=None,
            cod=None,
            env=diagram.Ty()):
        assert isinstance(kraus, diagram.Diagram)
        if dom is None:
            dom = Ty.from_optyx(kraus.dom)
        if cod is None:
            cod = Ty.from_optyx(kraus.cod)
        assert kraus.dom == dom.single()
        assert kraus.cod == cod.single() @ env
        self.kraus = kraus
        self.env = env
        super().__init__(name, dom, cod)

    def double(self):
        """
        Returns the :class:`diagram.Diagram` representing
        the action of the channel as a CP map on the doubled space.
        """

        def get_spiders(dom):
            spiders = diagram.Id()
            # pylint: disable=invalid-name
            for ob in dom.inside:
                if ob.is_classical:
                    box = diagram.Spider(1, 2, ob.single)
                else:
                    box = diagram.Id(ob.double)
                spiders @= box
            return spiders

        # pylint: disable=invalid-name
        def get_perm(n):
            return sorted(sorted(list(range(n))), key=lambda i: i % 2)

        cod = self.cod.single()
        top_spiders = get_spiders(self.dom)
        top_perm = diagram.Diagram.permutation(
            get_perm(len(top_spiders.cod)), top_spiders.cod
        )
        swap_env = diagram.Id(cod @ self.env) @ diagram.Diagram.swap(
            cod, self.env
        )
        discard = (
            diagram.Id(cod)
            @ diagram.Diagram.spiders(2, 0, self.env)
            @ diagram.Id(cod)
        )
        new_cod = diagram.Ty().tensor(*[ty @ ty for ty in cod])
        bot_perm = diagram.Diagram.permutation(
            get_perm(2 * len(cod)), new_cod
        ).dagger()
        bot_spiders = get_spiders(self.cod).dagger()
        top = top_spiders >> top_perm
        bot = swap_env >> discard >> bot_perm >> bot_spiders
        return top >> self.kraus @ self.kraus.conjugate() >> bot

    def dagger(self):
        return Channel(
            name=self.name + ".dagger()",
            kraus=self.kraus.dagger(),
            dom=self.cod,
            cod=self.dom,
        )

    def _decomp(self):
        # pylint: disable=import-outside-toplevel
        raise NotImplementedError(
            "Decomposition is only implemented for ZX channels."
        )

    def _to_dual_rail(self):
        raise TypeError(
            "Only ZX channels can be converted to dual rail."
            )

    def lambdify(self, *symbols, **kwargs):
        # Non-symbolic gates can be returned directly
        return lambda *xs: self

    def subs(self, *args) -> Diagram:
        syms, exprs = zip(*args)
        return self.lambdify(*syms)(*exprs)

    def inflate(self, d):
        r"""Translates from an indistinguishable setting
        to a distinguishable one. For a map on :math:`F(\mathbb{C})`,
        obtain a map on :math:`F(\mathbb{C})^{\widetilde{\otimes} d}`."""

        return Channel(
            name=self.name + f"^{d}",
            kraus=self.kraus.inflate(d) if
            self.needs_inflation() else self.kraus,
            dom=self.dom.inflate(d),
            cod=self.cod.inflate(d),
        )


class Spider(frobenius.Spider, Channel):  # pragma: no cover
    """
    Spider as a channel.
    """
    def __init__(self, n_legs_in: int, n_legs_out: int, typ: Ty, data=None,
                 **params):
        super().__init__(
            n_legs_in, n_legs_out, typ, data=data, **params
        )
        self.kraus = diagram.Spider(
            n_legs_in, n_legs_out, typ.single()
        )
        self.env = diagram.Ty()


class Sum(symmetric.Sum, Diagram):
    """
    Formal sum of optyx channel diagrams
    """

    __ambiguous_inheritance__ = (symmetric.Sum,)

    def double(self):
        return diagram.Diagram.sum_factory([t.double() for t in self])

    def grad(self, var, **params):
        """Gradient with respect to :code:`var`."""
        if var not in self.free_symbols:
            return self.sum_factory((), self.dom, self.cod)
        return sum(term.grad(var, **params) for term in self.terms)

    def get_kraus(self):
        if len(self.terms) == 0:
            return diagram.Scalar(0)

        return diagram.Diagram.sum_factory(
            [term.get_kraus() for term in self.terms]
        )


class CQMap(Diagram, frobenius.Box):
    """
    Channel initialised by its Density matrix.
    """

    def __init__(self, name, density_matrix, dom, cod):
        assert isinstance(density_matrix, diagram.Diagram)
        assert density_matrix.dom == dom.double()
        assert density_matrix.cod == cod.double()

        self.density_matrix = density_matrix
        super().__init__(name, dom, cod)

    def double(self):
        return self.density_matrix

    def dagger(self):
        return CQMap(
            name=self.name + ".dagger()",
            density_matrix=self.density_matrix.dagger(),
            dom=self.cod,
            cod=self.dom,
        )

    def inflate(self, d):
        r"""
        Translates from an indistinguishable setting
        to a distinguishable one. For a map on
        :math:`F(\mathbb{C}^d)`,
        obtain a map on :math:`F(\mathbb{C})^{\widetilde{\otimes} d}`.
        """

        return CQMap(
            name=self.name + f"^{d}",
            density_matrix=self.density_matrix.inflate(d) if
            self.needs_inflation() else self.density_matrix,
            dom=self.dom.inflate(d),
            cod=self.cod.inflate(d)
        )

    # pylint: disable=invalid-name
    def __pow__(self, n):
        if n == 1:
            return self
        return self @ self ** (n - 1)


class Swap(frobenius.Swap, Channel):
    def dagger(self):
        return self


class Feedback(monoidal.Bubble, Diagram, frobenius.Box):
    """
    A feedback loop connecting the last `mem` outputs of a channel diagram
    back to its last `mem` inputs, one time step later.

    See :class:`optyx.core.diagram.Feedback`; doubling the loop gives the
    loop of the doubled diagram, so `double` and `unroll` commute.

    The one boundary with a natural default is the effect: discarding is the
    unique causal effect on a channel, so :meth:`default_effect` is
    :class:`Discard` while :meth:`default_state` stays the open wire.

    Example
    -------
    The feedback of `CNOT >> SWAP` unrolls to a ladder of CNOT gates,
    here with the plus state as initial state and the output memory
    discarded by default; doubling commutes with unrolling on the nose:

    >>> from optyx.qubits import Z, X, Scalar, Ket
    >>> cnot = Z(1, 2) @ qubit >> qubit @ X(2, 1) @ Scalar(2 ** 0.5)
    >>> ladder = (cnot >> Diagram.swap(qubit, qubit)).feedback(
    ...     state=Ket("+"))
    >>> assert ladder.effect == Discard(qubit)
    >>> assert ladder.unroll(1).double() == ladder.double().unroll(1)
    """
    __ambiguous_inheritance__ = (monoidal.Bubble, frobenius.Box)

    @classmethod
    def default_state(cls, mem) -> Diagram:
        """The identity on `mem`: a channel has no natural initial state."""
        return Diagram.id(mem)

    @classmethod
    def default_effect(cls, mem) -> Diagram:
        """:class:`Discard` on `mem`, the unique causal effect."""
        return Discard(mem)

    def __init__(self, arg, dom=None, cod=None, mem=None,
                 state=None, effect=None):
        mem = arg.cod[-1:] if mem is None else mem
        dom = arg.dom[:len(arg.dom) - len(mem)] if dom is None else dom
        cod = arg.cod[:len(arg.cod) - len(mem)] if cod is None else cod
        if (arg.dom, arg.cod) != (dom @ mem, cod @ mem):
            raise AxiomError(
                f"{arg} is not a diagram from {dom @ mem} to {cod @ mem}")
        self.mem = mem
        self.state = self.default_state(mem) if state is None else state
        self.effect = self.default_effect(mem) if effect is None else effect
        if self.state.cod != mem:
            raise AxiomError(
                f"{self.state} is not a state on the memory {mem}")
        if self.effect.dom != mem:
            raise AxiomError(
                f"{self.effect} is not an effect on the memory {mem}")
        monoidal.Bubble.__init__(
            self, arg, dom=dom, cod=cod, method="feedback_operator")
        frobenius.Box.__init__(self, str(self), dom, cod)

    __str__ = diagram.Feedback.__str__
    __repr__ = diagram.Feedback.__repr__
    dagger = diagram.Feedback.dagger
    to_drawing = diagram.Feedback.to_drawing

    def double(self):
        """The feedback loop of the doubled diagram."""
        return Diagram.double(self.arg).feedback(
            mem=self.mem.double(), state=Diagram.double(self.state),
            effect=Diagram.double(self.effect))

    def get_kraus(self):
        """The feedback loop of the Kraus map."""
        return Diagram.get_kraus(self.arg).feedback(
            mem=self.mem.single(), state=Diagram.get_kraus(self.state),
            effect=Diagram.get_kraus(self.effect))

    def inflate(self, d):
        return Diagram.inflate(self.arg, d).feedback(
            mem=self.mem.inflate(d), state=Diagram.inflate(self.state, d),
            effect=Diagram.inflate(self.effect, d))


class Measure(Channel):
    """Measuring a qubit or qmode corresponds to
    applying a 2 -> 1 spider in the doubled picture.

    >>> dom = qubit @ bit @ qmode @ mode
    >>> print(dom.single())
    bit @ bit @ mode @ mode
    >>> assert Measure(dom).double().cod == dom.single()
    """
    draw_as_measures = True

    def __init__(self, dom):
        cod = Ty(*[Ob._classical[ob.name] for ob in dom.inside])
        kraus = diagram.Id(dom.single())
        super().__init__(name="Measure", kraus=kraus, dom=dom, cod=cod)

    def inflate(self, d):
        r""" A specific choice of inflation for the Measure channel.
        The diagram discards the internal states and measures
        the number of photons in the modes. Only qmodes are inflated.
        The bit, qubit and mode are not inflated.
        """

        diagrams = [self._measure_wire(ob, d) for ob in self.dom]
        return diagram.Diagram.tensor(*diagrams)

    # pylint: disable=invalid-name
    def _measure_wire(self, ob, d):
        """Return the diagram that measures one `ob`."""
        # pylint: disable=import-outside-toplevel
        from optyx.core.zw import Add
        if ob.needs_inflation():
            return Measure(ob ** d) >> CQMap(
                "Gather photons", Add(d), mode ** d, mode
            )
        return Measure(ob)


class Encode(Channel):
    """Encoding a bit or mode corresponds to
    applying a 1 -> 2 spider in the doubled picture.

    >>> dom = qubit @ bit @ qmode @ mode
    >>> assert len(Encode(dom).double().cod) == 8
    """
    draw_as_measures = True

    def __init__(self,
                 dom,
                 internal_states: tuple[list[int]] = None):
        cod = Ty(*[Ob._quantum[ob.name] for ob in dom.inside])
        kraus = diagram.Id(dom.single())
        if internal_states is not None:
            if not isinstance(internal_states, tuple):
                internal_states = (internal_states,)
            assert len(internal_states) == sum(
                [1 if ob.name == "mode" else 0 for ob in dom.inside]
            ), "# of internal states must match the number of modes in dom"
            assert len(set(len(i) for i in internal_states)) == 1, \
                "All internal states must be of the same length"

        super().__init__(name="Encode", kraus=kraus, dom=dom, cod=cod)
        self.internal_states = internal_states

    def inflate(self, d):
        r"""
        The internal states are used to encode the modes only.
        Bit and qubit are not encoded, qmode is inflated and
        mode is encoded.
        The diagram is a dagger of the inflation of
        the Measure channel with the difference
        that instead of discarding becoming a maximally mixed state,
        we apply the encoding of the internal states.
        """

        if any(
            ob.name == "mode" for ob in self.dom.inside
        ):
            assert self.internal_states is not None, \
                "Internal states must be provided for encoding"
            assert all(
                len(internal_state) == d for
                internal_state in self.internal_states
            ), "All internal states must have length d"

        amps_iter = iter(self.internal_states or [])
        diagrams = [self._encode_wire(ob, d, amps_iter) for ob in self.dom]
        return diagram.Diagram.tensor(*diagrams)

    def _encode_wire(self, ob, d, amps_iter):
        """Return the diagram that encodes *one* object `ob`.

        `amps_iter` yields the internal‑state vectors for `mode` wires.
        """
        # pylint: disable=import-outside-toplevel
        from optyx.core.zw import Add, Endo

        if ob == mode:
            amps = next(amps_iter)
            amp_layer = diagram.Diagram.tensor(*[Endo(a) for a in amps])
            return (
                CQMap("Add†", Add(d).dagger(), mode, mode ** d)
                >> Encode(mode ** d)
                >> Channel("Amplitudes", amp_layer)
            )
        if ob == qmode:
            return Encode(qmode ** d)
        return Encode(ob)


class Discard(Channel):
    """Discarding a qubit or qmode corresponds to
    applying a 2 -> 0 spider in the doubled picture.

    >>> assert Discard(qmode).double() == diagram.Spider(2, 0, diagram.mode)
    """
    draw_as_discards = True

    def __init__(self, dom):
        env = dom.single()
        kraus = diagram.Id(dom.single())
        super().__init__("Discard", kraus, dom=dom, cod=Ty(), env=env)

    def inflate(self, d):
        """
        Distinguishable setting for the Discard channel.
        """
        return Discard(self.dom.inflate(d))


class Functor(frobenius.Functor):
    """
    A hypergraph functor is a compact functor that preserves spiders.

    Parameters:
        ob_map (Mapping[Ty, Ty]) :
            Map from atomic :class:`Ty` to :code:`cod.ob`.
        ar_map (Mapping[Box, Diagram]) : Map from :class:`Box` to :code:`cod`.
        cod : The codomain of the functor, a :class:`Diagram` subclass.
    """
    dom = cod = Diagram

    def __call__(self, other):
        if isinstance(other, Feedback):
            return self(other.arg).feedback(
                dom=self(other.dom), cod=self(other.cod), mem=self(other.mem),
                state=self(other.state), effect=self(other.effect))
        return super().__call__(other)


class Hypergraph(hypergraph.Hypergraph):  # pragma: no cover
    functor = Functor


Id = Diagram.id
Scalar = lambda s: Channel(  # noqa: E731
    name=f"Scalar({s})",
    kraus=diagram.Scalar(s),
    dom=Ty(),
    cod=Ty()
)


Diagram.spider_factory = Spider
Diagram.hypergraph_factory = Hypergraph
Diagram.braid_factory = Swap
Diagram.sum_factory = Sum
Diagram.feedback_factory = Feedback
Diagram.functor_factory = Functor
