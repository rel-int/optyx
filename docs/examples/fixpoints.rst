Fixed points of feedback diagrams
=================================

A diagram with a feedback loop has stream semantics through
:meth:`optyx.core.diagram.Diagram.to_stream` and
:meth:`~optyx.core.diagram.Diagram.unroll`, and approximate fixed-point
semantics through :meth:`optyx.channel.Diagram.fix`. Both interpretations use
the same open one-step channel from ``dom @ memory`` to ``cod @ memory``.

Part 1: boson sampling with optical feedback
--------------------------------------------

Biriukov and Dyakonov inject a fixed bosonic product state :math:`\psi` into
the external modes at every time step, apply an :math:`M`-mode unitary
:math:`U`, expose :math:`M-L` output modes and feed the remaining :math:`L`
modes back as memory. For one feedback mode, Optyx writes the setup directly
as ``(psi @ qmode >> U).feedback(...)``.

The example uses a seeded Haar-random three-mode unitary and three product
Fock states on the two fresh input modes.

.. doctest::

    >>> import numpy as np
    >>> from discopy import tensor
    >>> from discopy.symmetric import Equation
    >>> from optyx import photonic
    >>> from optyx.channel import Channel, Diagram, Discard, qmode
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

Its open step contains the two maps used by the paper. Tracing out the detected
modes updates the loop state,
:math:`\rho \mapsto \operatorname{Tr}_{\mathrm{det}}[
U(|\psi\rangle\!\langle\psi|\otimes\rho)U^\dagger]`; tracing out the next
memory instead gives the visible output. ``method="power"`` repeats the first
map through a finite tensor-network unrolling and applies the readout at the
last step. ``method="eigen"`` constructs its cutoff transfer operator, finds
the stationary memory eigenstate and applies the same readout once. The
correlation-tensor reconstruction and permanent-based unfolding described in
the paper are separate algorithms and are not aliases for these two methods.

The following executed drawing uses one box as a placeholder for the computed
stationary mixed memory.

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

Approximating the stationary distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let :math:`D` be the truncated Hilbert-space dimension of the feedback memory,
so its vectorised density matrix has dimension :math:`D^2`. The useful
complexity comparison is between errors in the stationary distribution, not
between backend names.

.. list-table:: Power and eigen approximations
    :header-rows: 1

    * - Method
      - Numerical object
      - Leading dense cost
      - Approximation controls
    * - Power
      - An :math:`n`-step tensor-network ladder
      - Exact contraction is network-dependent; a quasi-1D contraction with
        local size :math:`p` and bond cap :math:`\chi` is typically
        :math:`O(n p \chi^3)` time and :math:`O(p \chi^2)` memory
      - Increase :math:`n` for finite-time error and :math:`\chi` for bond
        compression error
    * - Eigen
      - A dense :math:`D^2 \times D^2` transfer matrix
      - :math:`O(D^6)` eigensolve time and :math:`O(D^4)` matrix memory
      - Increase the Fock cutoff; trace preservation detects insufficient
        truncation

The power estimates are a quasi-1D guide rather than a promise for every
diagram; the chosen contraction path determines the actual exponent. The
eigen method avoids a time-depth limit but becomes prohibitive as the number
of memory modes grows.

Finite-depth and cutoff results can have different local shapes, so this
executed comparison pads absent higher occupations with zeros before taking a
Frobenius distance.

.. doctest::

    >>> def padded_distance(left, right):
    ...     shape = tuple(max(a, b) for a, b in zip(
    ...         left.shape, right.shape))
    ...     def pad(array):
    ...         result = np.zeros(shape, dtype=complex)
    ...         result[tuple(slice(0, n) for n in array.shape)] = array
    ...         return result
    ...     return np.linalg.norm(pad(left) - pad(right))

    >>> depths = (2, 4, 6)
    >>> distances = {}
    >>> for occupation in occupations:
    ...     sampler = feedback_sampler(occupation)
    ...     reference = sampler.fix(method="eigen", cutoff=9).density_matrix
    ...     distances[occupation] = tuple(
    ...         padded_distance(sampler.fix(
    ...             n_steps=steps, chi=16,
    ...             backend=DiscopyBackend()).density_matrix, reference)
    ...         for steps in depths)
    >>> for occupation, values in distances.items():
    ...     print(occupation, *(f"{value:.2e}" for value in values))
    (1, 0) 7.52e-02 2.75e-04 1.35e-06
    (0, 1) 8.79e-03 4.23e-05 2.08e-07
    (1, 1) 3.37e-02 1.07e-04 5.26e-07

.. list-table:: Executed Frobenius error against the cutoff-nine eigen result
    :header-rows: 1

    * - :math:`\psi`
      - 2 steps
      - 4 steps
      - 6 steps
    * - :math:`|1,0\rangle`
      - :math:`7.52\,10^{-2}`
      - :math:`2.75\,10^{-4}`
      - :math:`1.35\,10^{-6}`
    * - :math:`|0,1\rangle`
      - :math:`8.79\,10^{-3}`
      - :math:`4.23\,10^{-5}`
      - :math:`2.08\,10^{-7}`
    * - :math:`|1,1\rangle`
      - :math:`3.37\,10^{-2}`
      - :math:`1.07\,10^{-4}`
      - :math:`5.26\,10^{-7}`

This is evidence for this seeded example, not a convergence bound.

Part 2: learning fixed points
-----------------------------

Consider a tunable two-mode beam splitter. A photon enters its first input at
every step, while the second output is fed back into the second input. Optyx's
:class:`~optyx.photonic.MZI` exposes a mixing phase ``theta`` and an external
phase ``phi``. The output is one optical mode and the loop memory is the other.

.. doctest::

    >>> def learner(theta, phi):
    ...     splitter = photonic.MZI(theta, phi)
    ...     return (photonic.Create(1) @ qmode >> splitter).feedback(
    ...         mem=qmode, initial_state=photonic.Create(0))

    >>> example = learner(.04, .125)
    >>> len(example.one_step().dom), len(example.one_step().cod)
    (1, 2)
    >>> str(example.dom), str(example.cod), str(example.mem)
    ('Ty()', 'qmode', 'qmode')

The next executed sweep compares depths two and four with the cutoff-six eigen
state. ``active_dimension`` counts output occupations with probability above
``1e-4``; it measures effective support, while ``shape`` reports the actual
density-matrix allocation. Both matter: cutoff fixes the allocated size, but
parameter tuning can concentrate the stationary state in a smaller physical
subspace.

.. doctest::

    >>> theta_values = np.array([0, .02, .04, .06])
    >>> phi_values = np.array([0, .125, .25, .375])
    >>> errors = np.empty((len(phi_values), len(theta_values), 2))
    >>> active_dimensions = np.empty(
    ...     (len(phi_values), len(theta_values)), dtype=int)
    >>> shapes = {}
    >>> for row, phi in enumerate(phi_values):
    ...     for column, theta in enumerate(theta_values):
    ...         candidate = learner(theta, phi)
    ...         reference = candidate.fix(
    ...             method="eigen", cutoff=6).density_matrix
    ...         active_dimensions[row, column] = np.count_nonzero(
    ...             np.real(np.diag(reference)) > 1e-4)
    ...         shapes[theta, phi] = reference.shape
    ...         for depth_index, depth in enumerate((2, 4)):
    ...             approximate = candidate.fix(
    ...                 n_steps=depth, chi=16,
    ...                 backend=DiscopyBackend()).density_matrix
    ...             errors[row, column, depth_index] = padded_distance(
    ...                 approximate, reference)

    >>> np.max(np.ptp(errors, axis=0)) < 1e-12
    True
    >>> np.all(active_dimensions == active_dimensions[0])
    True
    >>> for column, theta in enumerate(theta_values):
    ...     print(
    ...         f"{theta:.2f}", shapes[theta, 0],
    ...         active_dimensions[0, column],
    ...         *(f"{value:.2e}" for value in errors[0, column]))
    0.00 (6, 6) 1 0.00e+00 0.00e+00
    0.02 (7, 7) 3 1.42e-02 2.16e-07
    0.04 (7, 7) 4 5.06e-02 1.15e-05
    0.06 (7, 7) 4 9.26e-02 9.51e-05

.. list-table:: Executed learning sweep at ``phi = 0``
    :header-rows: 1

    * - ``theta``
      - Density-matrix shape
      - Active dimension
      - Error after 2 steps
      - Error after 4 steps
    * - 0.00
      - :math:`6 \times 6`
      - 1
      - 0
      - 0
    * - 0.02
      - :math:`7 \times 7`
      - 3
      - :math:`1.42\,10^{-2}`
      - :math:`2.16\,10^{-7}`
    * - 0.04
      - :math:`7 \times 7`
      - 4
      - :math:`5.06\,10^{-2}`
      - :math:`1.15\,10^{-5}`
    * - 0.06
      - :math:`7 \times 7`
      - 4
      - :math:`9.26\,10^{-2}`
      - :math:`9.51\,10^{-5}`

The four tested values of ``phi`` agree to numerical precision: for a fresh
Fock input and one-mode occupation readout, this external phase changes
amplitudes but not the occupation dynamics. The mixing phase is the useful
control here. Moving it towards zero makes the beam splitter refresh the loop
more strongly; the fixed point is reached sooner and occupies fewer Fock
levels.

The sweep also generates a two-panel picture directly from these arrays.

.. doctest::

    >>> import matplotlib.pyplot as plt
    >>> figure, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    >>> error_image = axes[0].imshow(
    ...     np.log10(errors[:, :, 0] + 1e-18), origin="lower", aspect="auto")
    >>> _ = axes[0].set_title(r"$\log_{10}$ error after 2 steps")
    >>> support_image = axes[1].imshow(
    ...     active_dimensions, origin="lower", aspect="auto", vmin=1, vmax=4)
    >>> _ = axes[1].set_title("Active Fock dimension")
    >>> for axis in axes:
    ...     _ = axis.set_xticks(range(len(theta_values)), theta_values)
    ...     _ = axis.set_yticks(range(len(phi_values)), phi_values)
    ...     _ = axis.set_xlabel(r"mixing phase $\theta$")
    >>> _ = axes[0].set_ylabel(r"external phase $\phi$")
    >>> figure.colorbar(error_image, ax=axes[0], shrink=.8)
    <matplotlib.colorbar.Colorbar object at ...>
    >>> figure.colorbar(support_image, ax=axes[1], shrink=.8, ticks=range(1, 5))
    <matplotlib.colorbar.Colorbar object at ...>
    >>> figure.tight_layout()
    >>> figure.savefig(
    ...     "docs/_static/fixpoint-learning.png", dpi=160,
    ...     bbox_inches="tight")
    >>> plt.close(figure)

.. image:: /_static/fixpoint-learning.png
    :align: center
    :width: 720px

Increase ``n_steps`` to test convergence from ``initial_state``, increase
``chi`` to test bond compression, and increase ``cutoff`` to test optical
truncation in the eigen method. ``method="eigen"`` checks trace preservation,
uniqueness, normalisation, Hermiticity and positivity in the truncated space.

Further reading
---------------

The transfer/readout construction follows Yu. A. Biriukov and
I. V. Dyakonov, `Simulation of boson sampling with optical feedback
<https://doi.org/10.48550/arXiv.2602.05566>`_, arXiv:2602.05566 [quant-ph]
(2026).
