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

Functions
---------

.. autosummary::
    :template: function.rst
    :nosignatures:
    :toctree:

    stationary_vector
    frobenius_distance
    doubled_dimensions
    density_trace
    normalise_density_matrix


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

import warnings
from numbers import Integral, Real

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


bit = Ty("bit")
mode = Ty("mode")
qubit = Ty("qubit")
qmode = Ty("qmode")


def frobenius_distance(state, other):
    """Return the Frobenius distance between two state arrays.

    This metric does not require pairing doubled wires into matrix rows and
    columns, so it also applies to states with classical outputs.

    >>> assert np.isclose(
    ...     frobenius_distance(np.eye(2) / 2, np.eye(2) / 2), 0)
    >>> assert np.isclose(
    ...     frobenius_distance(np.diag([1, 0]), np.diag([0, 1])),
    ...     np.sqrt(2))
    """
    return np.linalg.norm(np.asarray(state) - np.asarray(other))


def stationary_vector(superoperator):
    """Return the unique stationary row vector of a superoperator.

    Optyx tensor arrays act on states from the right, hence a stationary
    vector ``state`` satisfies ``state @ superoperator == state``. The null
    space is computed with the numerical-rank tolerance of NumPy's SVD,
    independently of the convergence tolerance used by :meth:`Diagram.fix`.

    >>> transition = np.array([[.9, .1], [.4, .6]])
    >>> state = stationary_vector(transition)
    >>> assert np.allclose(state / state.sum(), [.8, .2])

    A non-unique stationary state is a design choice, not an arbitrary
    eigensolver choice:

    >>> stationary_vector(np.eye(2))
    Traceback (most recent call last):
    ...
    ValueError: The stationary state is not unique (fixed-space dimension 2).
    """
    matrix = np.asarray(superoperator)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("The superoperator must be a square matrix.")
    fixed = matrix.T - np.eye(matrix.shape[0], dtype=matrix.dtype)
    _, singular_values, vectors = np.linalg.svd(fixed)
    scale = max(
        np.linalg.norm(matrix, ord=2),
        singular_values[0] if singular_values.size else 0)
    rank_tolerance = max(fixed.shape) * np.finfo(float).eps * scale
    rank = np.count_nonzero(singular_values > rank_tolerance)
    nullity = matrix.shape[0] - rank
    if nullity == 0:
        raise ValueError(
            "The truncated transfer map has no stationary state; "
            "increase cutoff or check that the channel is trace preserving.")
    if nullity != 1:
        raise ValueError(
            "The stationary state is not unique "
            f"(fixed-space dimension {nullity}).")
    state = vectors[-1].conjugate()
    residual = np.linalg.norm(state @ matrix - state)
    residual_tolerance = max(matrix.shape) * np.finfo(float).eps \
        * max(1, np.linalg.norm(matrix))
    if residual > residual_tolerance:
        raise ValueError(
            f"The stationary-state residual {residual} is too large.")
    return state


def doubled_dimensions(ty: Ty, cutoff: int) -> list[int]:
    """Return local dimensions for the doubled representation of ``ty``.

    Classical bits and modes have one tensor axis; quantum wires have one
    axis for the ket and one for the bra.
    """
    dimensions = []
    for ob in ty.inside:
        if ob.name == "bit":
            dimensions.append(2)
        elif ob.name == "qubit":
            dimensions.extend((2, 2))
        elif ob.name == "mode":
            dimensions.append(cutoff)
        elif ob.name == "qmode":
            dimensions.extend((cutoff, cutoff))
        else:
            raise ValueError(f"Unknown channel object {ob}.")
    return dimensions


def density_trace(state, cod: Ty):
    """Return the categorical trace of a doubled state over ``cod``.

    Unlike :func:`numpy.trace`, this also sums classical outcomes and handles
    any interleaving of classical and quantum output wires.
    """
    array = np.asarray(state)
    dimensions = list(array.shape)
    discard = Discard(cod).double().to_tensor(dimensions).eval().array
    return np.tensordot(array, discard, axes=array.ndim)


def normalise_density_matrix(state, cod: Ty, tol: float):
    """Normalise and validate a classical-quantum density matrix.

    Validation uses the categorical trace, the dagger induced by ``cod`` and
    positivity of each classical block.
    """
    array = np.asarray(state)
    trace = density_trace(array, cod)
    if not np.isfinite(trace) or abs(trace) <= tol:
        raise ValueError("The stationary state has zero or non-finite trace.")
    array = array / trace

    classical, rows, columns, position = [], [], [], 0
    for ob in cod.inside:
        if ob.is_classical:
            classical.append(position)
            position += 1
        else:
            rows.append(position)
            columns.append(position + 1)
            position += 2
    dagger_axes = list(range(array.ndim))
    for row, column in zip(rows, columns):
        dagger_axes[row], dagger_axes[column] = column, row
    dagger = np.transpose(array.conjugate(), dagger_axes)
    if not np.allclose(array, dagger, atol=tol, rtol=0):
        raise ValueError("The stationary state is not Hermitian.")
    array = (array + dagger) / 2

    classical_dimension = int(np.prod(
        [array.shape[axis] for axis in classical], dtype=int))
    quantum_dimension = int(np.prod(
        [array.shape[axis] for axis in rows], dtype=int))
    blocks = np.transpose(array, classical + rows + columns).reshape(
        classical_dimension, quantum_dimension, quantum_dimension)
    if min(np.linalg.eigvalsh(block).min() for block in blocks) < -tol:
        raise ValueError("The stationary state is not positive.")
    return np.real_if_close(array)


@factory
class Diagram(frobenius.Diagram):
    """Classical-quantum circuits over qubits and optical modes"""

    ob = Ty
    grad = tensor.Diagram.grad
    unroll = diagram.Diagram.unroll
    to_stream = diagram.Diagram.to_stream

    def feedback(self, dom=None, cod=None, mem=None,
                 initial_state=None, final_effect=None) -> Diagram:
        """
        Feed the last `mem` outputs of a channel diagram from `dom @ mem`
        to `cod @ mem` back into its last `mem` inputs,
        one time step later.

        Parameters:
            dom : The domain of the result, `arg.dom[:-len(mem)]` by default.
            cod : The codomain of the result, `arg.cod[:-len(mem)]`
                by default.
            mem : The memory type fed back, `arg.cod[-1:]` by default.
            initial_state : Optional state of type `mem` plugged in the input
                memory by :meth:`unroll`, default `None` for an open wire;
                a natural choice on `qmode` is the zero-photon state
                `photonic.Create(0)`.
            final_effect : Optional effect on `mem` plugged in the output
                memory at the last time step of :meth:`unroll`, default
                `None` for an open wire; a natural choice is `Discard(mem)`.

        >>> from optyx.photonic import Create
        >>> wait = Diagram.swap(qmode, qmode).feedback(
        ...     initial_state=Create(0), final_effect=Discard(qmode))
        >>> assert wait.dom == wait.cod == qmode
        >>> assert wait.mem == qmode

        The result composes with any other channel diagram and is
        unrolled over `n_steps` time steps by :meth:`unroll`.
        """
        return self.feedback_factory(
            self, dom=dom, cod=cod, mem=mem,
            initial_state=initial_state, final_effect=final_effect)

    def one_step(self) -> Diagram:
        """
        Open every feedback loop to obtain one time step of the stateful
        diagram, from ``dom @ memory`` to ``cod @ memory``.

        This reuses :meth:`to_stream`, so the memory order is exactly the
        order used by :meth:`unroll`. Boundary ``initial_state`` and
        ``final_effect`` values are metadata and do not change this channel.

        >>> from optyx.photonic import Create
        >>> wait = Diagram.swap(qmode, qmode).feedback(
        ...     initial_state=Create(0))
        >>> assert wait.one_step() == Diagram.swap(qmode, qmode)
        """
        return self.to_stream().stream.now

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
        semantics = self.to_stream()
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

    def fix(self, n_steps: int = None, chi: int = None,
            method: str = "power", tol: float = 1e-6,
            cutoff: int = 2, backend=None,
            max_steps: int = 64, max_chi: int = 64):
        """
        Approximate a stationary state of a stateful diagram as a density
        matrix over its codomain.

        A diagram with a feedback loop has stream semantics through
        :meth:`to_stream` and :meth:`unroll`, and approximate fixed-point
        semantics through :meth:`fix`. Both use the same open one-step
        process; the fixed-point semantics evolves an approximate stationary
        memory once, then discards the next memory and returns only the visible
        output.

        The stationary state is the fixed point of the transfer channel
        which one time step induces on the memory of the feedback loops.
        It is the limit of :meth:`at_time` only when iteration converges;
        periodic channels can have a fixed state while their iterates cycle.

        Parameters:
            n_steps : The number of time steps unrolled by the `"power"`
                method, doubled until convergence when `None`.
            chi : The bound on the bond dimensions for a compressed tensor
                backend, doubled until convergence when `None` and ignored by
                exact backends.
            method : Either `"power"`, contracting the unrolling as a tensor
                network, or `"eigen"`, diagonalising the transfer matrix,
                see below.
            tol : The distance below which two successive
                approximations are deemed converged.
            cutoff : The dimension of each memory wire, `"eigen"` only.
            max_steps : The number of time steps at which the doubling
                gives up and warns, `"power"` only.
            max_chi : The bond dimension at which the doubling gives up
                and warns, `"power"` only.
            backend : An optional
                :class:`optyx.core.backends.TensorBackend` used by the
                `"power"` method. Exact DisCoPy backends can select NumPy,
                JAX or PyTorch; Quimb backends can contract exactly or use
                Cotengra compression bounded by `chi`.

        The `"power"` method unrolls the diagram and contracts the doubled
        tensor network. Its default compressed backend bounds bonds by `chi`:
        this is the scalable power iteration on the transfer channel. Exact
        tensor backends are useful baselines for networks which fit in
        memory. The `"eigen"` method builds the transfer matrix of one time
        step and diagonalises it, exact and cheaper whenever
        `cutoff ** len(memory)` is small, with no `n_steps` at all.

        A loop which reprepares its memory at every time step forgets its
        initial state, so it is its own stationary state:

        >>> from optyx.qubits import Ket
        >>> source = (Discard(qubit) @ Ket(0) @ Ket(0)).feedback(
        ...     mem=qubit, initial_state=Ket(1))
        >>> fixed = source.fix(method="eigen")
        >>> assert np.allclose(
        ...     fixed.density_matrix, [[1, 0], [0, 0]])

        See :doc:`/examples/fixpoints` for the semantic diagram, contraction
        planning and a backend benchmark.
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
        for name, value in {
                "n_steps": n_steps, "chi": chi, "cutoff": cutoff,
                "max_steps": max_steps, "max_chi": max_chi}.items():
            if value is None and name in ("n_steps", "chi"):
                continue
            if not isinstance(value, Integral) or isinstance(value, bool) \
                    or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if not isinstance(tol, Real) or isinstance(tol, bool) \
                or not np.isfinite(tol) or tol <= 0:
            raise ValueError("tol must be a positive finite real number.")
        if method == "eigen":
            if backend is not None:
                raise ValueError("backend is only used by method='power'.")
            return self.eigen_fix(cutoff, tol)
        return self.power_fix(
            n_steps, chi, tol, backend, max_steps, max_chi)

    def power_fix(self, n_steps, chi, tol, backend, max_steps, max_chi):
        """Approximate the stationary output by compressed iteration.

        The finite-time channel is first doubled and compiled by
        :meth:`to_tensor` to a :class:`discopy.tensor.Diagram`. The fixed-point
        algorithm knows no contraction library: it passes that tensor diagram
        to :meth:`optyx.core.backends.TensorBackend.contract`. A backend may
        then use the exact tensor functor under NumPy, JAX or PyTorch, exact
        Quimb contraction, or Cotengra compression.

        :meth:`fix` is the public validated dispatcher. ``n_steps`` is refined
        independently. ``chi`` is also refined when the backend advertises
        compression; exact backends ignore it because they truncate no bonds.
        """
        # pylint: disable=import-outside-toplevel
        from optyx.core.backends import (
            EvalResult, QuimbBackend, StateType, TensorBackend)

        if backend is None:
            backend = QuimbBackend.compressed()
        elif not isinstance(backend, TensorBackend):
            raise ValueError(
                "backend must implement the TensorBackend interface.")

        tensor_cache, result_cache = {}, {}

        def compile_tensor(steps):
            if steps not in tensor_cache:
                tensor_cache[steps] = \
                    self.at_time(steps).double().to_tensor()
            return tensor_cache[steps]

        def contract(steps, bond):
            max_bond = bond if backend.supports_compression else None
            key = steps, max_bond
            if key not in result_cache:
                result_cache[key] = EvalResult(
                    backend.contract(
                        compile_tensor(steps), max_bond=max_bond),
                    output_types=self.cod, state_type=StateType.DM)
            return result_cache[key]

        def normalised(result):
            state = result.density_matrix
            trace = density_trace(state, self.cod)
            if not np.isfinite(trace) or abs(trace) <= tol:
                raise ValueError(
                    "Compressed contraction returned zero or non-finite "
                    "trace.")
            return state / trace

        def refine_steps(bond):
            if n_steps is not None:
                return n_steps, contract(n_steps, bond), True
            steps = min(2, max_steps)
            result = contract(steps, bond)
            if steps == 1:
                return steps, result, False
            while True:
                previous = contract(steps - 1, bond)
                if frobenius_distance(
                        normalised(result), normalised(previous)) < tol:
                    return steps, result, True
                if steps == max_steps:
                    return steps, result, False
                steps = min(2 * steps, max_steps)
                result = contract(steps, bond)

        bond = chi if chi is not None else min(4, max_chi)
        _, result, steps_converged = refine_steps(bond)
        bond_converged = chi is not None or not backend.supports_compression
        if chi is None and backend.supports_compression:
            while bond < max_chi:
                next_bond = min(2 * bond, max_chi)
                _, next_result, next_steps_converged = \
                    refine_steps(next_bond)
                if frobenius_distance(
                        normalised(next_result), normalised(result)) < tol:
                    bond_converged = True
                    bond, result = next_bond, next_result
                    steps_converged = next_steps_converged
                    break
                bond, result = next_bond, next_result
                steps_converged = next_steps_converged

        failures = []
        if n_steps is None and not steps_converged:
            failures.append(
                f"n_steps before max_steps={max_steps}")
        if chi is None and not bond_converged:
            failures.append(f"chi before max_chi={max_chi}")
        if failures:
            warnings.warn(
                f"fix did not converge to tol={tol} in "
                + " or ".join(failures), UserWarning, stacklevel=3)
        return result

    def eigen_fix(self, cutoff: int = 2, tol: float = 1e-6):
        """
        The stationary state obtained by diagonalising the transfer matrix
        which one time step induces on the memory.

        ``cutoff`` is the local dimension of each optical mode, spanning
        occupations ``0`` through ``cutoff - 1``. Qubit dimensions remain
        two. A non-unique stationary memory state raises rather than choosing
        an arbitrary eigenvector.
        """
        # pylint: disable=import-outside-toplevel
        from optyx.core.backends import EvalResult, StateType
        step = self.one_step()
        memory = step.cod[len(self.cod):]
        dimensions = doubled_dimensions(memory, cutoff)
        memory_dimension = int(np.prod(dimensions, dtype=int))
        transfer = (step >> Discard(self.cod) @ self.id(memory)).double()
        readout = (step >> self.id(self.cod) @ Discard(memory)).double()
        matrix = transfer.to_tensor(dimensions).eval().array.reshape(
            memory_dimension, memory_dimension)
        discard = Discard(memory).double().to_tensor(
            dimensions).eval().array.reshape(memory_dimension)
        trace_residual = np.linalg.norm(matrix @ discard - discard)
        if trace_residual > tol:
            raise ValueError(
                "The truncated transfer map is not trace preserving "
                f"(residual {trace_residual}); increase cutoff.")
        state = stationary_vector(matrix).reshape(dimensions)
        state = normalise_density_matrix(state, memory, tol)
        tensor_diagram = readout.to_tensor(dimensions)
        memory_state = tensor.Box(
            "Stationary memory", tensor.Dim(1), tensor_diagram.dom, state)
        density_matrix = (memory_state >> tensor_diagram).eval().array
        density_matrix = normalise_density_matrix(
            density_matrix, self.cod, tol)
        return EvalResult(
            tensor.Box(
                "Result", tensor.Dim(1), tensor_diagram.cod,
                density_matrix),
            output_types=self.cod, state_type=StateType.DM)

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
                    generator.arg,
                    generator.initial_state, generator.final_effect)
                if part is not None
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

    Example
    -------
    The feedback of `CNOT >> SWAP` unrolls to a ladder of CNOT gates,
    here with the plus state as initial state and the output memory
    discarded; doubling commutes with unrolling on the nose:

    >>> from optyx.qubits import Z, X, Scalar, Ket
    >>> cnot = Z(1, 2) @ qubit >> qubit @ X(2, 1) @ Scalar(2 ** 0.5)
    >>> ladder = (cnot >> Diagram.swap(qubit, qubit)).feedback(
    ...     initial_state=Ket("+"), final_effect=Discard(qubit))
    >>> assert ladder.unroll(2).double() == ladder.double().unroll(2)
    """
    __ambiguous_inheritance__ = (monoidal.Bubble, frobenius.Box)

    def __init__(self, arg, dom=None, cod=None, mem=None,
                 initial_state=None, final_effect=None):
        mem = arg.cod[-1:] if mem is None else mem
        dom = arg.dom[:len(arg.dom) - len(mem)] if dom is None else dom
        cod = arg.cod[:len(arg.cod) - len(mem)] if cod is None else cod
        if (arg.dom, arg.cod) != (dom @ mem, cod @ mem):
            raise AxiomError(
                f"{arg} is not a diagram from {dom @ mem} to {cod @ mem}")
        self.mem = mem
        self.initial_state, self.final_effect = initial_state, final_effect
        monoidal.Bubble.__init__(
            self, arg, dom=dom, cod=cod, method="feedback_operator")
        frobenius.Box.__init__(self, str(self), dom, cod)

    __str__ = diagram.Feedback.__str__
    __repr__ = diagram.Feedback.__repr__
    dagger = diagram.Feedback.dagger
    to_drawing = diagram.Feedback.to_drawing

    def double(self):
        """The feedback loop of the doubled diagram."""
        initial_state = None if self.initial_state is None \
            else Diagram.double(self.initial_state)
        final_effect = None if self.final_effect is None \
            else Diagram.double(self.final_effect)
        return Diagram.double(self.arg).feedback(
            mem=self.mem.double(),
            initial_state=initial_state, final_effect=final_effect)

    def get_kraus(self):
        """The feedback loop of the Kraus map."""
        initial_state = None if self.initial_state is None \
            else Diagram.get_kraus(self.initial_state)
        final_effect = None if self.final_effect is None \
            else Diagram.get_kraus(self.final_effect)
        return Diagram.get_kraus(self.arg).feedback(
            mem=self.mem.single(),
            initial_state=initial_state, final_effect=final_effect)

    def inflate(self, d):
        initial_state = None if self.initial_state is None \
            else Diagram.inflate(self.initial_state, d)
        final_effect = None if self.final_effect is None \
            else Diagram.inflate(self.final_effect, d)
        return Diagram.inflate(self.arg, d).feedback(
            mem=self.mem.inflate(d),
            initial_state=initial_state, final_effect=final_effect)


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
            plugs = {
                attr: self(getattr(other, attr))
                for attr in ("initial_state", "final_effect")
                if getattr(other, attr) is not None}
            return self(other.arg).feedback(
                dom=self(other.dom), cod=self(other.cod),
                mem=self(other.mem), **plugs)
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
