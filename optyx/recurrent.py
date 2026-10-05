"""
Overview
--------

Passive recurrent linear-optical networks and their output streams.

A recurrent network is an interferometer on :math:`L + x` modes whose first
:math:`L` outputs are fed back into its first :math:`L` inputs one tick
later, while a product Fock state :math:`|q\\rangle` is injected into the
other :math:`x` inputs at every tick and the other :math:`x` outputs are
detected by photon-number-resolving detectors. The first :math:`L` modes are
the *loop*, the other :math:`x` are *external*. A loss channel of
transmissivity :math:`\\gamma` acts on every loop mode once per round trip,
and the detectors have efficiency :math:`\\eta`.

With the loop started in the vacuum, the state of the loop conditioned on the
detected history is pure and has a definite photon number: the input is a
product Fock state, the evolution is passive and the photons lost on the way
are sampled rather than traced out. One tick is then:

1. append the injected photons and apply the interferometer to the
   :math:`N + |q|` photons in the :math:`L + x` modes, one rotation of its
   :func:`givens` decomposition at a time,
2. sample the pattern on the external outputs and collapse the loop on it,
3. sample how many photons each loop mode loses and collapse again,
4. thin the detected pattern binomially, which is exact for inefficient
   detectors since loss before a number measurement is classical.

:meth:`Recurrent.sample` iterates this. A tick holds one vector of the
:math:`(N + |q|)`-photon sector of the :math:`L + x` modes and applies
:math:`(L + x)(L + x - 1) / 2` rotations to it, so its memory and time are
governed by the loop photon number :math:`N`, whose stationary mean is
:meth:`Recurrent.occupation`. Started from the vacuum, the loop forgets its
initial state at the rate certified by :meth:`Recurrent.burn_in`, so the
window sampled after that many ticks is within the requested total variation
distance of the stationary stream.

The same trajectory step, with every outcome enumerated rather than sampled,
gives the exact joint distribution of a window in
:meth:`Recurrent.distribution`. Evolving a pure state through the part of
the circuit a measurement depends on, then measuring and collapsing, is the
progressive simulation of Novák et al. [NRM+25]_ for loop-based time-bin
interferometers; here the cut is the loop itself, whatever the
interferometer.

Classes
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Recurrent
    LoopState
    Measurement

Functions
---------

.. autosummary::
    :template: function.rst
    :nosignatures:
    :toctree:

    sector
    position
    pairs
    twomode
    givens
    interfere
    rotate

Example
-------

A delay line swaps the loop with the external mode, so it detects at every
tick the photon injected one tick earlier:

>>> delay = Recurrent([[0, 1], [1, 0]], loop=1, inputs=(1, ))
>>> delay.sample(ticks=3, seed=0)
[(0,), (1,), (1,)]
>>> delay.burn_in(1e-9)
1
>>> delay.occupation()
1.0

.. [NRM+25] S. Novák, D. D. Roberts, A. Makarovskiy, R. García-Patrón and
    W. R. Clements, Boundaries for quantum advantage with single photons and
    loop-based time-bin interferometers, Quantum 9, 1915 (2025).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations_with_replacement, product
from math import comb, factorial, prod

import numpy as np


@lru_cache(maxsize=None)
def sector(modes: int, photons: int) -> np.ndarray:
    """
    The occupations of `photons` photons in `modes` modes, one per row, in
    lexicographic order: a basis of the `photons`-photon sector of Fock space.

    >>> sector(2, 2).tolist()
    [[0, 2], [1, 1], [2, 0]]
    >>> len(sector(4, 3)) == comb(4 + 3 - 1, 3)
    True
    """
    rows = [np.bincount(np.asarray(choice, dtype=int), minlength=modes)
            for choice in combinations_with_replacement(range(modes), photons)]
    occupations = np.array(rows, dtype=int).reshape(-1, modes)
    return occupations[np.lexsort(occupations.T[::-1])]


@lru_cache(maxsize=None)
def keys(modes: int, photons: int) -> tuple:
    """
    The sorted integer keys of the occupations of :func:`sector` and the
    permutation sorting them, to look occupations up in the sector.
    """
    values = sector(modes, photons) @ (photons + 1) ** np.arange(modes)
    order = np.argsort(values)
    return values[order], order


def position(occupations: np.ndarray, photons: int) -> np.ndarray:
    """
    The positions of `occupations` in :func:`sector`.

    >>> position(np.array([[1, 1], [2, 0]]), 2).tolist()
    [1, 2]
    """
    modes = occupations.shape[1]
    values, order = keys(modes, photons)
    return order[np.searchsorted(
        values, occupations @ (photons + 1) ** np.arange(modes))]


@lru_cache(maxsize=None)
def pairs(modes: int, photons: int, mode: int) -> tuple:
    """
    The rows of :func:`sector` grouped by everything but the occupations of
    `mode` and `mode + 1`: for every total :math:`s` of these two modes, an
    array with one group per row, ordered by the occupation of `mode`.

    >>> pairs(2, 1, 0)
    ((1, array([[0, 1]])),)
    """
    occupations = sector(modes, photons)
    total = occupations[:, mode] + occupations[:, mode + 1]
    rest = occupations.copy()
    rest[:, [mode, mode + 1]] = 0
    order = np.lexsort((occupations[:, mode],
                       rest @ (photons + 1) ** np.arange(modes), total))
    return tuple(
        (int(s), order[total[order] == s].reshape(-1, s + 1))
        for s in np.unique(total))


def twomode(matrix: np.ndarray, photons: int) -> np.ndarray:
    """
    The action of a two-mode `matrix` on the `photons`-photon sector of two
    modes, in the basis :math:`|k, s - k\\rangle` ordered by :math:`k`: the
    column :math:`k` expands
    :math:`(u_{00} a^\\dagger + u_{10} b^\\dagger)^k
    (u_{01} a^\\dagger + u_{11} b^\\dagger)^{s - k}|0\\rangle
    / \\sqrt{k! (s - k)!}`.

    A balanced beam splitter bunches two photons:

    >>> splitter = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    >>> twomode(splitter, 2)[:, 1].round(3).tolist()
    [-0.707, 0.0, 0.707]
    """
    result = np.zeros((photons + 1, photons + 1), dtype=complex)
    norms = np.sqrt([factorial(m) * factorial(photons - m)
                     for m in range(photons + 1)])
    for k in range(photons + 1):
        first = [comb(k, a) * matrix[1, 0] ** (k - a) * matrix[0, 0] ** a
                 for a in range(k + 1)]
        second = [comb(photons - k, b) * matrix[1, 1] ** (photons - k - b)
                  * matrix[0, 1] ** b for b in range(photons - k + 1)]
        result[:, k] = np.convolve(first, second) * norms / norms[k]
    return np.real_if_close(result)


def givens(unitary: np.ndarray) -> tuple:
    """
    The phases :math:`d` and the rotations of adjacent modes
    :math:`(i_1, u_1), \\dots, (i_K, u_K)` with
    :math:`U = u_1 \\cdots u_K \\, \\mathrm{diag}(d)`, the decomposition of
    the Reck scheme.

    >>> unitary = np.array([[0, 1j], [1, 0]])
    >>> phases, rotations = givens(unitary)
    >>> rotation = np.eye(2, dtype=complex)
    >>> for mode, matrix in rotations:
    ...     rotation[mode:mode + 2, mode:mode + 2] = matrix
    >>> bool(np.allclose(rotation @ np.diag(phases), unitary))
    True
    """
    current, rotations = np.array(unitary, dtype=complex), []
    size = len(current)
    for column in range(size - 1):
        for row in range(size - 1, column, -1):
            top, bottom = current[row - 1, column], current[row, column]
            norm = np.hypot(abs(top), abs(bottom))
            if abs(bottom) <= 1e-15:
                continue
            rotation = np.array(
                [[np.conj(top), np.conj(bottom)], [-bottom, top]]) / norm
            current[row - 1:row + 1] = rotation @ current[row - 1:row + 1]
            rotations.append((row - 1, rotation.conj().T))
    return np.diag(current), rotations


def interfere(unitary: np.ndarray, vector: np.ndarray,
              photons: int) -> np.ndarray:
    """
    The action of the mode matrix `unitary` on a `vector` of the
    `photons`-photon sector, one rotation of :func:`givens` at a time.

    >>> splitter = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    >>> interfere(splitter, np.array([0, 1, 0]), 2).real.round(3).tolist()
    [-0.707, 0.0, 0.707]
    """
    phases, rotations = givens(unitary)
    return rotate(phases, rotations, vector, photons,
                  lambda step, total: twomode(rotations[step][1], total))


def rotate(phases, rotations, vector, photons, table) -> np.ndarray:
    """
    Apply the decomposition `phases, rotations` of :func:`givens` to a
    `vector` of the `photons`-photon sector, where `table(step, total)` is
    :func:`twomode` of rotation `step` on `total` photons.
    """
    vector = np.asarray(vector, dtype=complex) * np.prod(
        phases ** sector(len(phases), photons), axis=1)
    for step in reversed(range(len(rotations))):
        for total, groups in pairs(len(phases), photons, rotations[step][0]):
            vector[groups] = vector[groups] @ table(step, total).T
    return vector


@dataclass
class LoopState:
    """
    A pure state of the loop with a definite photon number: the amplitude
    `amplitudes[i]` of the occupation `occupations[i]`.

    >>> LoopState.vacuum(2)
    LoopState(occupations=[[0, 0]], amplitudes=[1.0])
    """
    occupations: np.ndarray
    amplitudes: np.ndarray

    def __post_init__(self):
        self.occupations = np.asarray(self.occupations, dtype=int)
        self.amplitudes = np.asarray(self.amplitudes)

    def __repr__(self):
        return (f"LoopState(occupations={self.occupations.tolist()}, "
                f"amplitudes={self.amplitudes.tolist()})")

    def __eq__(self, other):
        return isinstance(other, LoopState) \
            and np.array_equal(self.occupations, other.occupations) \
            and np.array_equal(self.amplitudes, other.amplitudes)

    @staticmethod
    def vacuum(modes: int) -> LoopState:
        """ The loop with no photon. """
        return LoopState([[0] * modes], [1.])

    @property
    def photons(self) -> int:
        """ The photon number of the loop. """
        return int(self.occupations[0].sum())

    def normalised(self) -> LoopState:
        """ The same state with unit norm. """
        return LoopState(
            self.occupations,
            self.amplitudes / np.linalg.norm(self.amplitudes))

    def fingerprint(self, decimals: int = 12) -> tuple:
        """
        A key equal for two states that differ by a global phase, up to
        `decimals` digits.

        >>> LoopState([[1]], [1j]).fingerprint() == LoopState(
        ...     [[1]], [-1]).fingerprint()
        True
        """
        amplitudes = self.amplitudes[np.abs(self.amplitudes) > 0]
        phase = np.conj(amplitudes[0]) / np.abs(amplitudes[0])
        rounded = np.round(self.amplitudes * phase + 0j, decimals) + 0
        return self.occupations.tobytes(), rounded.tobytes()


@dataclass
class Measurement:
    """
    The output of one tick grouped by the pattern on the external outputs:
    pattern `patterns[i]` has probability `weights[i]` and occurs in the
    rows `rows[i]` of the loop `occupations` and output amplitudes `vector`.
    """
    patterns: np.ndarray
    weights: np.ndarray
    rows: list
    occupations: np.ndarray
    vector: np.ndarray

    def pattern(self, index: int) -> tuple:
        """ The pattern of outcome `index`. """
        return tuple(int(n) for n in self.patterns[index])

    def collapse(self, index: int) -> LoopState:
        """ The loop state left by outcome `index`. """
        rows = self.rows[index]
        return LoopState(
            self.occupations[rows], self.vector[rows]).normalised()


class Recurrent:
    """
    A passive recurrent linear-optical network.

    Parameters:
        unitary : The :math:`(L + x) \\times (L + x)` mode matrix, mapping
            input modes (columns) to output modes (rows); the first `loop`
            modes are fed back.
        loop : The number :math:`L` of loop modes.
        inputs : The photon numbers injected into the :math:`x` external
            inputs at every tick.
        transmissivity : The intensity transmissivity :math:`\\gamma` of each
            loop mode per round trip.
        efficiency : The efficiency :math:`\\eta` of the detectors.

    >>> network = Recurrent([[0, 1], [1, 0]], loop=1, inputs=(1, ))
    >>> network
    Recurrent([[0, 1], [1, 0]], loop=1, inputs=(1,), transmissivity=1.0, \
efficiency=1.0)
    >>> eval(repr(network)) == network
    True
    """
    def __init__(self, unitary, loop: int, inputs: tuple,
                 transmissivity: float = 1., efficiency: float = 1.):
        self.unitary = np.asarray(unitary)
        self.loop, self.inputs = loop, tuple(int(q) for q in inputs)
        self.transmissivity = float(transmissivity)
        self.efficiency = float(efficiency)
        if self.unitary.shape != (loop + len(self.inputs), ) * 2:
            raise ValueError(
                f"Expected a {loop + len(self.inputs)}-mode matrix, "
                f"got shape {self.unitary.shape}.")
        if not np.allclose(
                self.unitary.conj().T @ self.unitary, np.eye(len(unitary))):
            raise ValueError("The mode matrix must be unitary.")
        self.phases, self.rotations = givens(self.unitary)
        self.tables = {}

    def table(self, step: int, total: int) -> np.ndarray:
        """
        :func:`twomode` of rotation `step` of :attr:`rotations` on `total`
        photons, computed once and kept in :attr:`tables`.
        """
        if (step, total) not in self.tables:
            self.tables[step, total] = twomode(self.rotations[step][1], total)
        return self.tables[step, total]

    def __repr__(self):
        return (f"Recurrent({self.unitary.tolist()}, loop={self.loop}, "
                f"inputs={self.inputs}, "
                f"transmissivity={self.transmissivity}, "
                f"efficiency={self.efficiency})")

    def __eq__(self, other):
        return isinstance(other, Recurrent) \
            and np.array_equal(self.unitary, other.unitary) \
            and (self.loop, self.inputs, self.transmissivity,
                 self.efficiency) == (other.loop, other.inputs,
                                      other.transmissivity, other.efficiency)

    @property
    def modes(self) -> int:
        """ The number :math:`L + x` of modes. """
        return len(self.unitary)

    @property
    def block(self) -> np.ndarray:
        """
        The loop block :math:`\\sqrt\\gamma U_{\\ell\\ell}`, the amplitude a
        loop mode keeps over one round trip including its loss.
        """
        return np.sqrt(self.transmissivity) \
            * self.unitary[:self.loop, :self.loop]

    def occupation(self) -> float:
        """
        The mean photon number of the loop in the stationary state,
        :math:`\\mathrm{tr}\\, C` for the one-particle correlation matrix
        :math:`C = \\gamma (U_{\\ell\\ell} C U_{\\ell\\ell}^\\dagger
        + U_{\\ell x} Q U_{\\ell x}^\\dagger)` with
        :math:`Q = \\mathrm{diag}(q)`.

        Without loss every loop mode holds :math:`\\bar q` photons when
        :math:`\\bar q` is injected into every external mode, whatever the
        interferometer:

        >>> from scipy.stats import unitary_group
        >>> unitary = unitary_group.rvs(5, random_state=1)
        >>> network = Recurrent(unitary, loop=3, inputs=(2, 2))
        >>> bool(np.isclose(network.occupation(), 6))
        True
        """
        block, size = self.block, self.loop
        source = self.unitary[:self.loop, self.loop:]
        drive = self.transmissivity * source @ np.diag(self.inputs) \
            @ source.conj().T
        stein = np.eye(size ** 2) - np.kron(block, block.conj())
        correlations = np.linalg.solve(stein, drive.reshape(-1))
        return float(np.real(np.trace(correlations.reshape(size, size))))

    def burn_in(self, tol: float, max_depth: int = 10 ** 4) -> int:
        """
        The number :math:`k` of ticks after which the stream started from the
        vacuum loop is within total variation `tol` of the stationary stream.

        It is the smallest :math:`k` with
        :math:`2 K(\\bar q) \\sum_r \\arcsin^2 \\beta_r \\leq` `tol`, for
        :math:`\\beta_r` the singular values of the :math:`k`-th power of
        :attr:`block` and
        :math:`K(\\bar q) = (\\bar q + 1)[\\sqrt{6 \\bar q (\\bar q + 1)}
        + \\bar q]`: half the trace distance bound between the loop states
        of the stationary boson sampling study, since any window is a
        channel applied to the loop.

        >>> lossy = Recurrent(
        ...     [[0.6, 0.8], [0.8, -0.6]], 1, (1, ), transmissivity=.5)
        >>> lossy.burn_in(1e-3) < Recurrent(
        ...     [[0.6, 0.8], [0.8, -0.6]], 1, (1, )).burn_in(1e-3)
        True
        """
        qbar = max(self.inputs, default=0)
        constant = (qbar + 1) * (np.sqrt(6 * qbar * (qbar + 1)) + qbar)
        power = np.eye(self.loop)
        for depth in range(max_depth + 1):
            values = np.clip(np.linalg.svd(power, compute_uv=False), 0, 1)
            if 2 * constant * np.sum(np.arcsin(values) ** 2) <= tol:
                return depth
            power = self.block @ power
        raise ValueError(
            f"The bound does not reach tol={tol} within {max_depth} ticks.")

    def evolve(self, state: LoopState) -> np.ndarray:
        """
        The output state of the interferometer on the loop `state` and the
        injected photons, a vector of the :math:`N + |q|`-photon sector of
        the :math:`L + x` modes.
        """
        photons = state.photons + sum(self.inputs)
        occupations = np.hstack([
            state.occupations,
            np.tile(self.inputs, (len(state.occupations), 1))])
        vector = np.zeros(len(sector(self.modes, photons)), dtype=complex)
        vector[position(occupations, photons)] = state.amplitudes
        return rotate(self.phases, self.rotations, vector, photons, self.table)

    def measure(self, state: LoopState) -> Measurement:
        """
        The output of the interferometer on the loop `state` and the
        injected photons, grouped by the pattern on the external outputs.
        """
        vector = self.evolve(state)
        occupations = sector(self.modes, state.photons + sum(self.inputs))
        patterns, inverse = np.unique(
            occupations[:, self.loop:], axis=0, return_inverse=True)
        inverse = inverse.reshape(-1)
        weights = np.bincount(inverse, np.abs(vector) ** 2, len(patterns))
        order = np.argsort(inverse, kind="stable")
        bounds = np.concatenate(
            [[0], np.cumsum(np.bincount(inverse, minlength=len(patterns)))])
        return Measurement(
            patterns, weights, [order[bounds[i]:bounds[i + 1]]
                                for i in range(len(patterns))],
            occupations[:, :self.loop], vector)

    def detections(self, state: LoopState):
        """
        Every pattern on the external outputs with its probability and the
        loop state it leaves, before loss on the loop.
        """
        measurement = self.measure(state)
        for index, weight in enumerate(measurement.weights):
            if weight > 1e-14:
                yield measurement.pattern(index), float(weight), \
                    measurement.collapse(index)

    def losses(self, state: LoopState, mode: int):
        """
        Every number of photons lost by loop `mode` over one round trip,
        with its probability and the loop state it leaves.
        """
        number = state.occupations[:, mode]
        for lost in range(int(number.max()) + 1):
            rows = number >= lost
            amplitudes = state.amplitudes[rows] * np.sqrt(
                [comb(int(n), lost) for n in number[rows]]
                * self.transmissivity ** (number[rows] - lost)
                * (1 - self.transmissivity) ** lost)
            weight = float(np.sum(np.abs(amplitudes) ** 2))
            if weight <= 1e-14:
                continue
            occupations = state.occupations[rows].copy()
            occupations[:, mode] -= lost
            yield weight, LoopState(occupations, amplitudes).normalised()

    def thinnings(self, pattern: tuple):
        """
        Every pattern the detectors record when `pattern` reaches them, with
        its probability.
        """
        for recorded in product(*(range(n + 1) for n in pattern)):
            weight = prod(
                comb(n, k) * self.efficiency ** k
                * (1 - self.efficiency) ** (n - k)
                for n, k in zip(pattern, recorded))
            if weight > 0:
                yield recorded, weight

    def tick(self, state: LoopState, rng: np.random.Generator):
        """
        One tick of a trajectory: the recorded pattern and the next loop
        state.
        """
        measurement = self.measure(state)
        index = rng.choice(
            len(measurement.weights), p=normalise(measurement.weights))
        pattern, state = measurement.pattern(index), \
            measurement.collapse(index)
        if self.transmissivity < 1:
            for mode in range(self.loop):
                outcomes = list(self.losses(state, mode))
                state = outcomes[rng.choice(
                    len(outcomes),
                    p=normalise([weight for weight, _ in outcomes]))][1]
        recorded = rng.binomial(pattern, self.efficiency)
        return tuple(int(n) for n in recorded), state

    def trajectory(self, ticks: int, seed=None):
        """
        The recorded patterns and the loop photon numbers of `ticks` ticks
        of one trajectory started from the vacuum loop.
        """
        rng, state = np.random.default_rng(seed), LoopState.vacuum(self.loop)
        patterns, photons = [], []
        for _ in range(ticks):
            photons.append(state.photons)
            pattern, state = self.tick(state, rng)
            patterns.append(pattern)
        return patterns, photons

    def sample(self, ticks: int, burn_in: int = 0, seed=None) -> list:
        """
        A sample of the patterns recorded over `ticks` ticks, after
        `burn_in` ticks started from the vacuum loop.
        """
        patterns, _ = self.trajectory(burn_in + ticks, seed)
        return patterns[burn_in:]

    def distribution(self, ticks: int, burn_in: int = 0) -> dict:
        """
        The exact probability of every sequence of patterns recorded over
        `ticks` ticks after `burn_in` ticks, by enumerating trajectories.

        >>> delay = Recurrent([[0, 1], [1, 0]], loop=1, inputs=(1, ))
        >>> delay.distribution(2)
        {((0,), (1,)): 1.0}
        """
        branches = [((), 1., LoopState.vacuum(self.loop))]
        for time in range(burn_in + ticks):
            branches = merge(
                (history + (recorded, ) * (time >= burn_in),
                 weight * detected * lost * thinned, after)
                for history, weight, state in branches
                for pattern, detected, loop in self.detections(state)
                for lost, after in self.lost(loop)
                for recorded, thinned in self.thinnings(pattern))
        result = {}
        for history, weight, _ in branches:
            result[history] = result.get(history, 0) + weight
        return result

    def lost(self, state: LoopState):
        """
        Every loop state left by the losses of one round trip, with its
        probability.
        """
        branches = [(1., state)]
        if self.transmissivity < 1:
            for mode in range(self.loop):
                branches = [
                    (weight * lost, after) for weight, loop in branches
                    for lost, after in self.losses(loop, mode)]
        return branches


def merge(branches) -> list:
    """
    Add up the weights of the branches with the same history and the same
    loop state up to a global phase, since a mixture of copies of one pure
    state is that state.
    """
    merged = {}
    for history, weight, state in branches:
        key = (history, ) + state.fingerprint()
        _, total, _ = merged.get(key, (history, 0., state))
        merged[key] = (history, total + weight, state)
    return [merged[key] for key in sorted(merged)]


def normalise(weights) -> np.ndarray:
    """ The probability vector proportional to `weights`. """
    weights = np.asarray(weights, dtype=float)
    return weights / weights.sum()
