Fixed points of feedback diagrams
=================================

Optyx models a stateful process by feeding some output wires of a channel
back into its input one time step later.  If the visible input is empty,
:meth:`optyx.channel.Diagram.fix` computes a stationary density matrix on
the visible output.  This page explains the semantics shared by feedback,
finite unrolling and fixed-point evaluation.

From feedback to a one-step channel
-----------------------------------

Suppose ``step`` has type ``dom @ mem -> cod @ mem``.  Calling
:meth:`optyx.channel.Diagram.feedback` on it hides the memory wires and
stores optional boundary data:

``initial_state``
    A state on ``mem`` used at the first time step of a finite unrolling.

``final_effect``
    An effect on ``mem`` used at the last time step of a finite unrolling.

The underlying process has two useful one-step views::

                       +------------------+
    dom -------------->|                  |--------------> cod
                       |     one_step     |
    memory ----------->|                  |--------------> memory'
                       +------------------+
                              |                         |
                 transfer: discard cod     readout: discard memory'

The **transfer channel** evolves the loop memory and discards ``cod``.  Its
stationary memory state satisfies ``rho @ transfer == rho`` in Optyx's row
vector convention.  The **readout channel** applies the same step and
discards the next memory.  Consequently, :meth:`~optyx.channel.Diagram.fix`
returns ``rho >> readout`` on ``cod``; it does not return ``rho`` itself.

Stream semantics
----------------

:meth:`optyx.core.diagram.Diagram.to_stream` is the shared conversion used
by :meth:`~optyx.core.diagram.Diagram.unroll` and
:meth:`~optyx.channel.Diagram.one_step`.  It deliberately is a method, not a
cached ``.stream`` property: conversion is cheap, and callers receive a new
immutable :class:`optyx.core.diagram.Stream` value rather than
mutable shared state.  Its ``stream.now`` is the open one-step channel and
never includes ``initial_state`` or ``final_effect``.  Its boundary records
are ordered left to right, with an outer loop before loops nested inside it.
This makes memory order identical in every consumer without exposing a
second, independently maintained feedback interpretation.

The related operations differ as follows:

* :meth:`~optyx.core.diagram.Diagram.unroll` produces every visible time-bin
  output and applies both stored boundaries.
* :meth:`~optyx.channel.Diagram.one_step` opens every loop and applies no
  boundary.
* :meth:`~optyx.channel.Diagram.at_time` requires an empty ``dom`` and an
  ``initial_state`` for every loop, discards earlier visible outputs and the
  final memory, and deliberately ignores ``final_effect``.
* :meth:`~optyx.channel.Diagram.fix` also requires an empty ``dom``.  It
  computes a stationary output by either compressed iteration or a transfer
  eigensolve.

For example, this loop prepares zero on every step, so its stationary output
does not depend on the initial memory:

.. doctest::

    >>> import numpy as np
    >>> from optyx.channel import Diagram, Discard, qubit
    >>> from optyx.qubits import Ket
    >>> source = (Discard(qubit) @ Ket(0) @ Ket(0)).feedback(
    ...     mem=qubit, initial_state=Ket(1))
    >>> np.allclose(source.fix(method="eigen").density_matrix,
    ...             [[1, 0], [0, 0]])
    True

Choosing a method
-----------------

``method="power"`` is the default.  The algorithm has one compilation path,
independent of the numerical library::

    at_time(n_steps) -> double() -> to_tensor() -> tensor.Diagram
                                                   |
                                                   v
                                      TensorBackend.contract()

The first three operations construct the finite doubled network; they do not
contract it.  The resulting :class:`discopy.tensor.Diagram` is the stable
boundary between Optyx and execution.  A
:class:`optyx.core.backends.TensorBackend` decides how that network is
contracted:

* :class:`~optyx.core.backends.DiscopyBackend` uses
  :class:`discopy.tensor.Functor` exactly.  Its optional backend name selects
  NumPy, JAX, PyTorch or another array implementation registered by DisCoPy.
* :class:`~optyx.core.backends.QuimbBackend` with no optimiser contracts the
  same diagram exactly with Quimb.
* :meth:`~optyx.core.backends.QuimbBackend.compressed` uses Cotengra and
  Quimb's compressed contraction.  This is the default because it can bound
  intermediate tensor sizes.

For example, the exact NumPy and Quimb choices are interchangeable at the
fixed-point API::

    from optyx.core.backends import DiscopyBackend, QuimbBackend

    numpy_result = process.fix(
        n_steps=8, backend=DiscopyBackend("numpy"))
    quimb_result = process.fix(
        n_steps=8, backend=QuimbBackend())

``n_steps`` is the unrolling depth.  ``chi`` is the maximum bond dimension
only for a backend which advertises compression support.  Exact backends
accept and ignore it, so the fixed-point code does not need backend-specific
branches.  When either adaptive axis is ``None``, Optyx refines the applicable
axis until consecutive normalised density matrices are within the Frobenius
tolerance ``tol``.  ``max_steps`` and ``max_chi`` cap only their corresponding
adaptive axes.  If a cap is reached, the last result is returned with a
:class:`UserWarning` naming the axis which did not converge.  Explicit values
are used as written and are not limited by the adaptive caps.

``method="eigen"`` builds the doubled transfer matrix, verifies that it is
trace preserving to ``tol``, and finds the nullspace of ``transfer.T - I``.
It is preferable when the truncated memory matrix fits in RAM: it has no
unrolling or bond-dimension approximation and it can return a stationary
state even when iteration is periodic.  It raises if the fixed space is not
one-dimensional rather than selecting an arbitrary state.  The stationary
memory and returned output are normalised and checked for Hermiticity and
positivity.  ``backend`` is not used by this method.

Truncation and assumptions
--------------------------

The power method has two distinct errors.  Finite ``n_steps`` measures how
far the initial memory state is from its long-time limit.  A compressed Quimb
backend adds contraction error by truncating intermediate bonds to ``chi``;
repeat at larger ``chi`` to measure it.  Exact tensor-functor and exact Quimb
backends have no bond truncation, so only finite-depth, physical cutoff and
floating-point errors remain.  The
:doc:`notebooks/fixpoint_benchmark` notebook estimates exact contraction cost
before execution and compares these choices.

All integer size parameters and ``tol`` must be positive; ``tol`` must also
be finite.  Fixed-point evaluation currently accepts only diagrams with
``dom == Ty()`` and at least one feedback loop.  A non-empty domain would
need a policy for preparing a fresh visible input at every time step, which
is not part of this API.

For ``method="eigen"``, ``cutoff`` is the local dimension of every optical
memory wire: occupations ``0`` through ``cutoff - 1``.  Classical bits and
qubits remain two-dimensional.  This is a per-mode rectangular truncation,
not a joint cutoff on total photon number.  Increase it when the transfer
map fails the trace-preservation check; physical truncation error must be
assessed by repeating the calculation at larger cutoffs.

A unique stationary state is not the same as convergence from a particular
initial state.  Periodic channels can have a unique stationary mixture while
their finite-time states cycle.  Conversely, a channel with several fixed
states needs an additional selection rule, so the eigen method raises.  The
power method follows the supplied ``initial_state`` and warns when its
successive finite-time states fail to settle.

Further reading
---------------

The transfer/readout split corresponds to the partial-density-matrix and
Kraus-superoperator views in Yu. A. Biriukov and I. V. Dyakonov,
`Simulation of boson sampling with optical feedback
<https://doi.org/10.48550/arXiv.2602.05566>`_, arXiv:2602.05566 [quant-ph]
(2026).  That paper also studies spatiotemporal unfolding and correlation
tensors; Optyx's ``power`` and ``eigen`` names describe contraction
strategies for the common channel rather than names of those four methods.
