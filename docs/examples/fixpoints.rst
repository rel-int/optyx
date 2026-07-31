Fixed points of feedback diagrams
=================================

A diagram with a feedback loop has stream semantics through
:meth:`optyx.core.diagram.Diagram.to_stream` and
:meth:`~optyx.core.diagram.Diagram.unroll`, and approximate fixed-point
semantics through :meth:`optyx.channel.Diagram.fix`. Both interpretations use
the same open one-step channel from ``dom @ memory`` to ``cod @ memory``.

Boson sampling with optical feedback
------------------------------------

The setup of Biriukov and Dyakonov injects a fixed bosonic product state
:math:`\psi` into the external modes at every time step, applies an
:math:`M`-mode unitary :math:`U`, exposes :math:`M-L` output modes and feeds
the remaining :math:`L` modes back as memory. For one feedback mode, Optyx
writes this directly as ``(psi @ qmode >> U).feedback(...)``.

The example uses a seeded Haar-random three-mode unitary and three product
Fock states on the two fresh input modes.

.. doctest::

    >>> import numpy as np
    >>> from discopy import tensor
    >>> from discopy.symmetric import Equation
    >>> from optyx import photonic
    >>> from optyx.channel import (
    ...     Channel, Diagram, Discard, density_trace,
    ...     frobenius_distance, qmode)
    >>> from optyx.core import diagram
    >>> from optyx.core.backends import DiscopyBackend

    >>> def random_unitary(size, seed=7):
    ...     rng = np.random.default_rng(seed)
    ...     matrix = rng.normal(size=(size, size)) + 1j * rng.normal(
    ...         size=(size, size))
    ...     unitary, upper = np.linalg.qr(matrix)
    ...     diagonal = np.diag(upper)
    ...     return unitary * (diagonal / np.abs(diagonal))

    >>> U = photonic.Gate(random_unitary(3), 3, 3, "U")
    >>> np.allclose(U.array.conjugate().T @ U.array, np.eye(3))
    True
    >>> occupations = ((1, 0), (0, 1), (1, 1))
    >>> def feedback_sampler(occupation):
    ...     psi = photonic.Create(*occupation)
    ...     return (psi @ qmode >> U).feedback(
    ...         mem=qmode, initial_state=photonic.Create(0))

    >>> process = feedback_sampler((1, 0))
    >>> str(process.dom), str(process.cod), str(process.mem)
    ('Ty()', 'qmode @ qmode', 'qmode')
    >>> isinstance(process.at_time(2).double().to_tensor(), tensor.Diagram)
    True

Its open step implements both maps in the paper. Tracing out the detected
modes updates the loop state,
:math:`\rho \mapsto \operatorname{Tr}_{\mathrm{det}}[
U(|\psi\rangle\!\langle\psi|\otimes\rho)U^\dagger]`; tracing out the next
memory instead gives the visible output. ``to_stream`` repeats the first map.
``fix`` approximates its stationary state :math:`\rho_\star` and applies the
second map once, so the result has no memory wire.

The following one-step picture uses a single box as a drawing placeholder for
the stationary mixed memory. It is not an extra numerical implementation.

.. doctest::

    >>> psi = photonic.Create(1, 0)
    >>> stationary = Channel(
    ...     r"$\rho_\star$",
    ...     diagram.Box(
    ...         r"$\rho_\star$", diagram.Ty(), diagram.Mode(2)),
    ...     cod=qmode, env=diagram.Mode(1))
    >>> readout = (
    ...     psi @ stationary >> U
    ...     >> Diagram.id(qmode ** 2) @ Discard(qmode))
    >>> Equation(process, readout, symbol=r"$\mapsto$").draw(
    ...     path="docs/_static/fixpoint.png")

.. image:: /_static/fixpoint.png
    :align: center
    :width: 480px

Where the methods agree
-----------------------

The eigen method truncates the loop occupation and diagonalises its one-step
transfer matrix. The power method unrolls from vacuum and compiles
``at_time(n_steps).double().to_tensor()``. Finite-depth and cutoff results can
have different local shapes, so the comparison pads absent higher occupations
with zeros before taking a Frobenius distance.

.. doctest::

    >>> def normalised(result, cod):
    ...     state = np.asarray(result.density_matrix)
    ...     return state / density_trace(state, cod)

    >>> def padded_distance(left, right):
    ...     shape = tuple(max(a, b) for a, b in zip(
    ...         left.shape, right.shape))
    ...     def pad(array):
    ...         result = np.zeros(shape, dtype=complex)
    ...         result[tuple(slice(0, n) for n in array.shape)] = array
    ...         return result
    ...     return frobenius_distance(pad(left), pad(right))

    >>> depths = (2, 4, 6)
    >>> distances = {}
    >>> for occupation in occupations:
    ...     sampler = feedback_sampler(occupation)
    ...     reference = normalised(
    ...         sampler.fix(method="eigen", cutoff=9), sampler.cod)
    ...     distances[occupation] = tuple(
    ...         padded_distance(normalised(sampler.fix(
    ...             n_steps=steps, chi=16,
    ...             backend=DiscopyBackend()), sampler.cod), reference)
    ...         for steps in depths)
    >>> for occupation, values in distances.items():
    ...     print(occupation, *(f"{value:.2e}" for value in values))
    (1, 0) 7.52e-02 2.75e-04 1.35e-06
    (0, 1) 8.79e-03 4.23e-05 2.08e-07
    (1, 1) 3.37e-02 1.07e-04 5.26e-07

At tolerance ``1e-5``, the agreement map is therefore:

.. list-table:: Power agrees with the cutoff-nine eigen result
    :header-rows: 1

    * - :math:`\psi`
      - 2 steps
      - 4 steps
      - 6 steps
    * - :math:`|1,0\rangle`
      - no
      - no
      - yes
    * - :math:`|0,1\rangle`
      - no
      - no
      - yes
    * - :math:`|1,1\rangle`
      - no
      - no
      - yes

This is evidence for this seeded example, not a convergence bound. Repeat the
map after increasing ``cutoff``; the eigen method raises when truncation loses
more trace than ``tol``.

Plan before contracting
-----------------------

An exact Quimb path can estimate arithmetic and memory cost without executing
the contraction:

.. code-block:: python

    def contraction_plan(network, assumed_gflops=10):
        quimb_network = network.to_quimb()
        info = quimb_network.contraction_info(
            optimize="greedy",
            output_inds=sorted(quimb_network.outer_inds()))
        flops = float(info.opt_cost)
        largest = int(info.largest_intermediate)
        return {
            "tensors": len(quimb_network.tensor_map),
            "flops": flops,
            "largest_complex128_bytes": 16 * largest,
            "heuristic_seconds": flops / (assumed_gflops * 1e9),
        }

    sampler = feedback_sampler((1, 0))
    plans = {
        steps: contraction_plan(
            sampler.at_time(steps).double().to_tensor())
        for steps in depths
    }

The time estimate excludes Python overhead, memory traffic, array compilation
and device transfers. Avoid an exact run when its largest intermediate does
not fit in memory.

Choosing approximation controls
--------------------------------

There are three independent accuracy axes:

* Increase ``n_steps`` to test convergence from ``initial_state``.
* Increase ``chi`` to test bond compression; only compressed Quimb uses it.
* Increase ``cutoff`` to test optical truncation in the eigen method.

``method="power"`` can adapt ``n_steps`` and ``chi`` independently, subject to
``max_steps`` and ``max_chi``. ``method="eigen"`` checks trace preservation,
uniqueness, normalisation, Hermiticity and positivity in the truncated space.
Both methods require ``dom == Ty()`` and at least one feedback loop.

Further reading
---------------

The transfer/readout construction follows Yu. A. Biriukov and
I. V. Dyakonov, `Simulation of boson sampling with optical feedback
<https://doi.org/10.48550/arXiv.2602.05566>`_, arXiv:2602.05566 [quant-ph]
(2026).
