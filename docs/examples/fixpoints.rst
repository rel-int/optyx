Fixed points of feedback diagrams
=================================

A diagram with a feedback loop has stream semantics through
:meth:`optyx.core.diagram.Diagram.to_stream` and
:meth:`~optyx.core.diagram.Diagram.unroll`, and approximate fixed-point
semantics through :meth:`optyx.channel.Diagram.fix`. Both interpretations use
the same open one-step channel from ``dom @ memory`` to ``cod @ memory``.

Semantics
---------

``to_stream`` returns that channel together with the ordered boundary data for
each loop. ``unroll`` applies the initial states, repeats the channel and then
applies the final effects. ``fix`` instead evolves an approximate stationary
memory once and discards the next memory, leaving only the visible output.
The boundary order is left to right, with an outer loop before any nested loop.
``one_step`` applies no boundaries; ``at_time`` applies every initial state but
ignores ``final_effect`` because stationary evaluation traces out memory rather
than conditioning on it.

This example resets a qubit memory to zero at every step. On the left, the
memory wire is fed back. On the right, the stationary memory enters one step
and the next memory is discarded.

.. doctest::

    >>> import numpy as np
    >>> from discopy import tensor
    >>> from discopy.symmetric import Equation
    >>> from optyx.channel import Diagram, Discard, qubit
    >>> from optyx.qubits import Ket
    >>> process = (Discard(qubit) @ Ket(0) @ Ket(0)).feedback(
    ...     mem=qubit, initial_state=Ket(1))
    >>> readout = (Ket(0) >> process.one_step()
    ...            >> Diagram.id(qubit) @ Discard(qubit))
    >>> Equation(process, readout, symbol=r"$\mapsto$").draw(
    ...     path="docs/_static/fixpoint.png")

.. image:: /_static/fixpoint.png
    :align: center
    :width: 480px

The eigen method constructs the transfer and readout tensors directly. The
power method compiles a finite approximation to the common tensor language.

.. doctest::

    >>> fixed = process.fix(method="eigen")
    >>> network = process.at_time(4).double().to_tensor()
    >>> isinstance(network, tensor.Diagram)
    True
    >>> np.allclose(fixed.density_matrix, [[1, 0], [0, 0]])
    True

Choosing a method
-----------------

``method="power"`` is the default. It increases ``n_steps`` until consecutive
normalised outputs are within the Frobenius tolerance. A compressed backend can
also increase ``chi`` independently; ``max_steps`` and ``max_chi`` cap only
their respective adaptive axes and a warning identifies an unconverged axis.

``method="eigen"`` constructs the doubled transfer matrix for one step, checks
trace preservation after truncation and solves its stationary nullspace. It is
exact when the truncated memory fits in RAM and can find a stationary mixture
for a periodic process. It raises when the fixed space is not one-dimensional,
or when the resulting state is not normalised, Hermitian and positive.

Both methods require ``dom == Ty()`` and at least one feedback loop. For the
eigen method, ``cutoff`` is the local dimension of each optical memory mode;
qubits remain two-dimensional. This rectangular per-mode cutoff is separate
from the contraction bond dimension ``chi``.

Plan before contracting
-----------------------

Every power backend consumes the same ``tensor.Diagram``. An exact Quimb path
can estimate arithmetic and memory cost without executing the contraction:

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

    plans = {
        steps: contraction_plan(
            process.at_time(steps).double().to_tensor())
        for steps in (2, 4, 8)
    }

The time estimate is only a scale: Python overhead, memory traffic, array
compilation and device transfers are not included. Avoid an exact run when its
largest intermediate does not fit in memory.

Compare methods and backends
----------------------------

The eigen method is the small-memory reference; every power case changes only
the executor. JAX and PyTorch are optional DisCoPy array backends; exact
backends ignore ``chi``.

.. code-block:: python

    from importlib.util import find_spec
    from time import perf_counter

    from optyx.channel import frobenius_distance
    from optyx.core.backends import DiscopyBackend, QuimbBackend

    def normalised(result):
        state = np.asarray(result.density_matrix)
        return state / np.trace(state)

    reference = normalised(fixed)
    cases = [
        ("eigen", lambda: process.fix(method="eigen")),
        ("power / NumPy", lambda: process.fix(
            n_steps=4, chi=4, backend=DiscopyBackend("numpy"))),
        ("power / Quimb exact", lambda: process.fix(
            n_steps=4, chi=4, backend=QuimbBackend())),
        ("power / Quimb compressed", lambda: process.fix(
            n_steps=4, chi=4, backend=QuimbBackend.compressed())),
    ]
    if find_spec("jax"):
        cases.append(("power / JAX", lambda: process.fix(
            n_steps=4, chi=4, backend=DiscopyBackend("jax"))))
    if find_spec("torch"):
        cases.append(("power / PyTorch", lambda: process.fix(
            n_steps=4, chi=4, backend=DiscopyBackend("pytorch"))))

    def time_case(name, run):
        started = perf_counter()
        try:
            result = run()
        except Exception as error:
            return {"backend": name, "error": repr(error)}
        return {
            "backend": name,
            "seconds": perf_counter() - started,
            "distance_to_eigen": frobenius_distance(
                normalised(result), reference),
        }

    if __name__ == "__main__":
        benchmark = [time_case(name, run) for name, run in cases]

There are three independent accuracy controls:

* Increase ``n_steps`` to test convergence from ``initial_state``.
* Increase ``chi`` to test bond compression; only compressed Quimb uses it.
* Increase ``cutoff`` to test optical truncation in the eigen method.

Compression helps when exact intermediates approach available memory. On small
networks, optimisation and device startup can cost more than contraction.
Report both runtime and distance to an eigen or feasible exact reference.

Further reading
---------------

The transfer/readout split follows Yu. A. Biriukov and I. V. Dyakonov,
`Simulation of boson sampling with optical feedback
<https://doi.org/10.48550/arXiv.2602.05566>`_, arXiv:2602.05566 [quant-ph]
(2026). Their spatiotemporal unfolding, partial-density-matrix,
Kraus-superoperator and correlation-tensor views describe the same feedback
process; ``power`` and ``eigen`` name Optyx's contraction strategies.
