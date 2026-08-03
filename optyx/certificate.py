"""

Overview
--------

Certified window depth and Fock truncation for feedback loops.

A feedback loop forgets its initial memory at a rate fixed by the loop block
of its one-step unitary. This module reads that rate off an :math:`L \\times L`
matrix, so a depth and a cutoff are decided before any Fock space is built.
Everything here is linear algebra on the mode matrix: no doubled diagram, no
tensor network, no density matrix, no eigenvalue of a superoperator.

The statements are proved in the wiki, *Pre-requisites for equivalence between
Stationary Boson Sampling and Boson Sampling* (`01`), *The truncation bound*
(`03`) and *An algorithm for Stationary Boson Sampling* (`06`).

Classes
-------------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Blocks


Functions
-------------

    .. autosummary::
        :template: function.rst
        :nosignatures:
        :toctree:

        window_error
        certified_depth
        mean_occupation
        certified_cutoff

Examples of usage
------------------

A loop which empties in one round trip is certified at depth one, whatever the
accuracy asked for, because its loop block is nilpotent:

>>> import numpy as np
>>> unitary = np.eye(2, dtype=complex)  # 02, (2.1) at x = 1
>>> blocks = Blocks(unitary, loop=1, injected=(1,))
>>> assert blocks.loop_block.shape == (1, 1)
>>> assert np.allclose(blocks.loop_block, 0)
>>> assert certified_depth(blocks, tol=1e-12) == 1
"""

from __future__ import annotations

from math import ceil, log, sqrt
from typing import NamedTuple

import numpy as np

MAX_DEPTH = 4096
MAX_CUTOFF = 4096


class Blocks(NamedTuple):
    """
    The four blocks of the one-step unitary of a feedback loop, with the Fock
    state injected at every step.

    The unitary acts on :math:`L + M` modes, the first :math:`L` of which are
    the memory. Inputs are ordered memory then injection, outputs detection
    then memory, which is the order :meth:`optyx.channel.Diagram.now` builds
    and :meth:`optyx.channel.Diagram.unroll` reuses.

    Parameters:
        unitary : The one-step unitary, acting on column vectors.
        loop : The number :math:`L` of memory modes.
        injected : The occupation number injected into each external input.

    The block convention is a hypothesis and not a convention: reading the
    exiting block where the entering one belongs is invisible at :math:`L = 1`
    and wrong from :math:`L = 2`. :meth:`check` is what makes it safe.
    """

    unitary: np.ndarray
    loop: int
    injected: tuple[int, ...]

    @property
    def external(self) -> int:
        """The number :math:`M` of external modes."""
        return len(self.injected)

    @property
    def loop_block(self) -> np.ndarray:
        """:math:`U_{\\ell\\ell}`, memory in to memory out."""
        return np.asarray(self.unitary)[self.external:, :self.loop]

    @property
    def entering_block(self) -> np.ndarray:
        """:math:`U_{x\\ell}`, injection in to memory out."""
        return np.asarray(self.unitary)[self.external:, self.loop:]

    @property
    def height(self) -> int:
        """:math:`\\bar q`, the largest occupation of a single input."""
        return max(self.injected)

    def check(self, depth: int = 4, tol: float = 1e-10) -> None:
        """
        Assert the exact power identity of `01`, (1.13):

        .. math::
            U_{\\ell\\ell}^{k}\\big(U_{\\ell\\ell}^{k}\\big)^\\dagger
            \\;=\\; \\mathbb I_L - \\sum_{i<k} w_i w_i^\\dagger,
            \\qquad w_i = U_{\\ell\\ell}^{\\,i} U_{x\\ell}

        One line, and it catches a non-unitary step, a wrong power, and the
        entering block read where the exiting one belongs. It does **not**
        catch memory modes taken from the wrong end of the unitary: row
        orthonormality holds for any partition, so the identity is blind to
        it. That ordering is derived from the types by
        :meth:`optyx.channel.Diagram.blocks` rather than assumed, and pinned
        by a test comparing :func:`mean_occupation` against an occupation
        measured from :meth:`optyx.channel.Diagram.at_time`.
        """
        loop_block, entering = self.loop_block, self.entering_block
        power, leaked = np.eye(self.loop, dtype=complex), np.zeros(
            (self.loop, self.loop), dtype=complex)
        for _ in range(depth):
            emitted = power @ entering
            leaked = leaked + emitted @ emitted.conjugate().T
            power = loop_block @ power
            residual = np.linalg.norm(
                power @ power.conjugate().T + leaked - np.eye(self.loop))
            if residual > tol:
                raise ValueError(
                    f"The power identity fails at residual {residual}: the "
                    "step is not unitary, or the memory modes are not the "
                    "first block of it.")


def _constant(height: int) -> float:
    """:math:`K(\\bar q)` of `03`, Theorem A."""
    return (height + 1) * (sqrt(6 * height * (height + 1)) + height)


def window_error(blocks: Blocks, depth: int, loss: float = 0) -> float:
    """
    The certified trace distance to the stationary state after ``depth``
    round trips, :math:`\\Gamma(k) = 4K(\\bar q)\\Theta_k` of `03`, Theorem A
    and Corollary 3.18, with

    .. math:: \\Theta_k = \\sum_r \\arcsin^2\\big(\\gamma^{k/2}\\beta_r\\big)

    for :math:`\\beta_r` the singular values of :math:`U_{\\ell\\ell}^{\\,k}`
    and :math:`\\gamma = 1 - loss` the round-trip transmissivity.

    The bound is uniform over the initial memory, so it certifies the distance
    to :math:`\\rho_\\infty` and not merely the gap between two iterates.
    """
    power = np.linalg.matrix_power(blocks.loop_block, depth)
    values = np.linalg.svd((1 - loss) ** (depth / 2) * power,
                           compute_uv=False)
    theta = float(np.sum(np.arcsin(np.clip(values, 0, 1)) ** 2))
    return 4 * _constant(blocks.height) * theta


def certified_depth(blocks: Blocks, tol: float = 1e-6, loss: float = 0,
                    maximum: int = MAX_DEPTH) -> int:
    """
    The smallest depth at which :func:`window_error` is within ``tol``.

    :math:`\\Theta_k` is monotone decreasing, so accumulating
    :math:`B_{k+1} = U_{\\ell\\ell}B_k` and taking one SVD per step costs
    :math:`O(kL^3)` in total and stops at the first depth that clears the
    target. This is a certificate for the instance at hand, not an asymptotic
    estimate: two loops with the same spectral radius can need very different
    depths and the singular values see the difference.

    >>> import numpy as np
    >>> unitary = np.eye(2, dtype=complex)  # 02, (2.1) at x = 1
    >>> blocks = Blocks(unitary, loop=1, injected=(1,))
    >>> assert certified_depth(blocks, tol=1e-9) == 1
    """
    blocks.check()
    power = np.eye(blocks.loop, dtype=complex)
    constant = 4 * _constant(blocks.height)
    for depth in range(1, maximum + 1):
        power = blocks.loop_block @ power
        values = np.linalg.svd((1 - loss) ** (depth / 2) * power,
                               compute_uv=False)
        theta = float(np.sum(np.arcsin(np.clip(values, 0, 1)) ** 2))
        if constant * theta <= tol:
            return depth
    raise ValueError(
        f"No depth below {maximum} certifies tol={tol}: the loop block has "
        "spectral radius one, or the target is too small.")


def mean_occupation(blocks: Blocks) -> float:
    """
    The stationary mean photon number of the loop, `06`, (6.9):

    .. math::
        Y = \\mathbb I_L + U_{\\ell\\ell}^\\dagger Y U_{\\ell\\ell},
        \\qquad Z = U_{x\\ell}^\\dagger Y U_{x\\ell},
        \\qquad \\lambda = \\sum_a q_a Z_{aa}

    :math:`Y` is :math:`L \\times L` and :math:`Z` is :math:`M \\times M`, so
    solving directly for :math:`Z` is a dimension error whenever
    :math:`M \\ne L`. The Stein equation is solved once, in closed form.
    """
    loop_block = blocks.loop_block
    size = blocks.loop
    operator = np.eye(size ** 2) - np.kron(
        loop_block.conjugate().T, loop_block.T)
    solution = np.linalg.solve(operator, np.eye(size).reshape(-1))
    memory = solution.reshape(size, size)
    entering = blocks.entering_block
    occupation = entering.conjugate().T @ memory @ entering
    return float(np.real(
        sum(count * occupation[mode, mode]
            for mode, count in enumerate(blocks.injected))))


def _tail(cutoff: int, mean: float) -> float:
    """
    :math:`\\eta(N, \\lambda) = \\min_t 1/[(1-t)(1+t/\\lambda)^N]` of `06`,
    Proposition 6.10 (iv), at its minimiser :math:`t = (N-\\lambda)/(N+1)`.
    Evaluated in logarithms, since the two factors overflow separately.
    """
    if cutoff <= mean:
        return 1.0
    logarithm = log(cutoff + 1) - log(1 + mean) + cutoff * (
        log(mean) + log(cutoff + 1) - log(cutoff) - log(1 + mean))
    return float(np.exp(logarithm))


def certified_cutoff(blocks: Blocks, depth: int, tol: float = 1e-6,
                     maximum: int = MAX_CUTOFF) -> int:
    """
    The smallest total photon number at which truncating the loop keeps the
    mass deficit below ``tol`` over ``depth`` steps, by `06`,
    Proposition 6.10 (iv): :math:`1 - \\operatorname{tr}\\tilde\\rho_k \\le
    k\\,\\eta(N_{\\max}, \\lambda)`, so
    :math:`N_{\\max} = O(\\lambda\\log(k/\\varepsilon))`.

    This is an a priori estimate and it does not have to be trusted: by
    Proposition 6.10 (ii) the error actually made is **exactly** the mass
    deficit, which an evaluation reports for free. Use this to size the
    computation, and the measured deficit to certify its result.
    """
    mean = mean_occupation(blocks)
    cutoff = max(1, ceil(mean))
    while cutoff <= maximum:
        if depth * _tail(cutoff, mean) <= tol:
            return cutoff
        cutoff += 1
    raise ValueError(
        f"No cutoff below {maximum} keeps the mass deficit below {tol} over "
        f"{depth} steps, at mean occupation {mean}.")
