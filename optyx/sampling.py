"""
Overview
--------

Sampling the outputs of closed diagrams with feedback loops, tick after
tick: :meth:`optyx.channel.Diagram.sample`.

Any such diagram is sampled by its :class:`Unravelling`, a quantum
trajectory: one tick runs :meth:`one_step` box by box on a :class:`CQState`,
a pure state of its quantum wires and definite values of its classical
ones. A box contracts its Kraus map into the state, linear optics through
the :func:`givens` decomposition of its mode matrix and anything else one
box of its Kraus map at a time; its environment and its classical outputs
are measured in the number basis, sampled and collapsed. Measurements,
classical boxes and classical control inside the loops, qubits and
photons with internal states are all sampled this way, at a cost
exponential in the number of quantum wires alive at once.

A passive diagram is sampled through its :class:`Interferometer`, a recurrent
linear-optical network: an interferometer on :math:`L + x` modes whose first
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

1. hold the :math:`N + |q|` photons of the loop and the injected photons in
   the modes they occupy,
2. measure the external outputs one at a time along a :class:`Sweep`: bring
   the output to a held mode by a chain of rotations, sample its photon
   number and collapse,
3. map the held modes left at the end onto the loop and sample how many
   photons each loop mode loses, collapsing again,
4. thin the detected pattern binomially, which is exact for inefficient
   detectors since loss before a number measurement is classical.

:meth:`Interferometer.sample` iterates this. With :math:`k` occupied external
inputs, a tick holds one vector of the :math:`(N + |q|)`-photon sector of at
most :math:`L + k + 1` modes, the vacuum inputs never enter and every
measured output leaves, and applies at most :math:`x (L + k)` rotations to
it, plus :math:`L (L - 1) / 2` to land on the loop. So its memory and time
are governed by the loop photon number :math:`N`, whose stationary mean is
:meth:`Interferometer.occupation`. Started from the vacuum, the loop forgets
its initial state at the rate certified by :meth:`Interferometer.burn_in`, so
the window sampled after that many ticks is within the requested total
variation distance of the stationary stream.

Partially distinguishable photons are sampled by colouring them: each
injected photon is in the internal state shared by all photons with
probability :math:`p`, the :attr:`Interferometer.indistinguishability`, and
in one of its own otherwise [RMC+18]_. Photons of different colours never
interfere, so each colour runs its own trajectory and the detectors record
the sum of their patterns. A photon of its own is a single-photon walk,
forgotten once it has left the loop.

A passive diagram, linear optics with losses and discards, is turned into
its :class:`Interferometer` by :meth:`Interferometer.from_diagram`: its
discards and losses are environment outputs, measured and forgotten.

The same trajectory step, with every outcome enumerated rather than sampled,
gives the exact joint distribution of a window in
:meth:`Interferometer.distribution`. Evolving a pure state through the part of
the circuit a measurement depends on, then measuring and collapsing, is the
progressive simulation of Novák et al. [NRM+25]_ for loop-based time-bin
interferometers; here the cut is the loop itself, whatever the
interferometer, and within a tick each external output is measured as soon
as the held modes reach it.

The same tick written with JAX is :meth:`Interferometer.kernel`, a
:class:`Kernel` with a cap on the loop photon number: it compiles, maps over
many trajectories, and differentiates in the mode matrix, the loss, the
efficiency and the indistinguishability. Every outcome it samples is
recorded, so that :meth:`Kernel.log_prob` replays a trajectory for the
score-function estimator of a gradient, and :meth:`Kernel.distribution`
replays every one of a small network for the exact gradient. JAX is an
optional dependency: ``pip install optyx[jax]``.

Classes
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Interferometer
    FockState
    Sweep
    Unravelling
    CQState
    Kernel

Functions
---------

.. autosummary::
    :template: function.rst
    :nosignatures:
    :toctree:

    complete
    observe
    levels
    sector
    position
    pairs
    twomode
    eliminate
    givens
    interfere
    rotate
    pad
    marginal
    collapse
    binomials
    rank
    expansion
    powers
    twomodes
    turn
    move
    givens_rotation
    safe_log
    choose
    choices

Example
-------

A delay line swaps the loop with the external mode, so it detects at every
tick the photon injected one tick earlier:

>>> delay = Interferometer([[0, 1], [1, 0]], loop=1, inputs=(1, ))
>>> delay.sample(ticks=3, seed=0)
[(0,), (1,), (1,)]
>>> delay.burn_in(1e-9)
1
>>> delay.occupation()
1.0

The kernel samples many trajectories at once, here two of three ticks:

>>> delay.kernel(cap=1).sample(ticks=3, shots=2).tolist()
[[[0], [1], [1]], [[0], [1], [1]]]

.. [RMC+18] J. J. Renema, A. Menssen, W. R. Clements, G. Triginer,
    W. S. Kolthammer and I. A. Walmsley, Efficient classical algorithm for
    boson sampling with partially distinguishable photons, PRL 120, 220502
    (2018).

.. [NRM+25] S. Novák, D. D. Roberts, A. Makarovskiy, R. García-Patrón and
    W. R. Clements, Boundaries for quantum advantage with single photons and
    loop-based time-bin interferometers, Quantum 9, 1915 (2025).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from itertools import product
from math import comb, factorial, prod

import numpy as np

try:
    import jax
    from jax import numpy as jnp
except ImportError:  # pragma: no cover
    jax = jnp = None


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
    if modes == 0:
        return np.zeros((int(photons == 0), 0), dtype=np.int8)
    return np.vstack([
        np.hstack([np.full((len(rest), 1), first, dtype=np.int8), rest])
        for first in range(photons + 1)
        for rest in [sector(modes - 1, photons - first)]])


def encode(occupations: np.ndarray, photons: int) -> np.ndarray:
    """
    One integer per row of `occupations`, the digits of base `photons + 1`.

    >>> encode(np.array([[1, 2], [2, 0]]), 2).tolist()
    [7, 2]
    """
    values = np.zeros(len(occupations), dtype=np.int64)
    for mode in reversed(range(occupations.shape[1])):
        values = values * (photons + 1) + occupations[:, mode]
    return values


@lru_cache(maxsize=None)
def lookup(modes: int, photons: int) -> tuple:
    """
    The sorted integer keys of the occupations of :func:`sector` and the
    permutation sorting them, to look occupations up in the sector.
    """
    values = encode(sector(modes, photons), photons)
    order = np.argsort(values)
    return values[order], order


def position(occupations: np.ndarray, photons: int) -> np.ndarray:
    """
    The positions of `occupations` in :func:`sector`.

    >>> position(np.array([[1, 1], [2, 0]]), 2).tolist()
    [1, 2]
    """
    values, order = lookup(occupations.shape[1], photons)
    return order[np.searchsorted(values, encode(occupations, photons))]


@lru_cache(maxsize=None)
def pairs(modes: int, photons: int, mode: int) -> tuple:
    """
    The rows of :func:`sector` grouped by everything but the occupations of
    `mode` and `mode + 1`: for every total :math:`s` of these two modes, an
    array with one group per row, ordered by the occupation of `mode`.

    >>> [(total, groups.tolist()) for total, groups in pairs(2, 1, 0)]
    [(1, [[0, 1]])]
    """
    occupations = sector(modes, photons)
    total = occupations[:, mode] + occupations[:, mode + 1]
    rest = occupations.copy()
    rest[:, [mode, mode + 1]] = 0
    order = np.lexsort((occupations[:, mode],
                       encode(rest, photons), total)).astype(np.int32)
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


def eliminate(matrix: np.ndarray, column: int) -> list:
    """
    Zero the entries of `column` below the diagonal of `matrix` in place, by
    rotations of adjacent rows from the bottom up: the rotations
    :math:`(i, r)` acting as :math:`r` on rows :math:`i, i + 1`, in the
    order they are applied.

    >>> matrix = np.array([[0], [0.6], [0.8]], dtype=complex)
    >>> [mode for mode, _ in eliminate(matrix, 0)]
    [1, 0]
    >>> np.abs(matrix).round(3).tolist()
    [[1.0], [0.0], [0.0]]
    """
    rotations = []
    for row in range(len(matrix) - 1, column, -1):
        top, bottom = matrix[row - 1, column], matrix[row, column]
        if abs(bottom) <= 1e-15:
            continue
        rotation = np.array([[np.conj(top), np.conj(bottom)], [-bottom, top]]
                            ) / np.hypot(abs(top), abs(bottom))
        matrix[row - 1:row + 1] = rotation @ matrix[row - 1:row + 1]
        rotations.append((row - 1, rotation))
    return rotations


def givens(unitary: np.ndarray) -> tuple:
    """
    The phases :math:`d` and the rotations of adjacent modes
    :math:`(i_1, u_1), \\dots, (i_K, u_K)` with
    :math:`U = u_1 \\cdots u_K \\, \\mathrm{diag}(d)`, the decomposition of
    the Reck scheme, one :func:`eliminate` per column.

    >>> unitary = np.array([[0, 1j], [1, 0]])
    >>> phases, rotations = givens(unitary)
    >>> rotation = np.eye(2, dtype=complex)
    >>> for mode, matrix in rotations:
    ...     rotation[mode:mode + 2, mode:mode + 2] = matrix
    >>> bool(np.allclose(rotation @ np.diag(phases), unitary))
    True
    """
    current = np.array(unitary, dtype=complex)
    rotations = [(mode, rotation.conj().T)
                 for column in range(len(current) - 1)
                 for mode, rotation in eliminate(current, column)]
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
class FockState:
    """
    A pure state of modes with a definite photon number, such as the loop of
    an :class:`Interferometer`: the amplitude `amplitudes[i]` of the
    occupation `occupations[i]`.

    >>> FockState.vacuum(2)
    FockState(occupations=[[0, 0]], amplitudes=[1.0])
    """
    occupations: np.ndarray
    amplitudes: np.ndarray

    def __post_init__(self):
        self.occupations = np.asarray(self.occupations, dtype=int)
        self.amplitudes = np.asarray(self.amplitudes)

    def __repr__(self):
        return (f"FockState(occupations={self.occupations.tolist()}, "
                f"amplitudes={self.amplitudes.tolist()})")

    def __eq__(self, other):
        return isinstance(other, FockState) \
            and np.array_equal(self.occupations, other.occupations) \
            and np.array_equal(self.amplitudes, other.amplitudes)

    @staticmethod
    def vacuum(modes: int) -> FockState:
        """ The loop with no photon. """
        return FockState([[0] * modes], [1.])

    @property
    def photons(self) -> int:
        """ The photon number of the loop. """
        return int(self.occupations[0].sum())

    def normalised(self) -> FockState:
        """ The same state with unit norm. """
        return FockState(
            self.occupations,
            self.amplitudes / np.linalg.norm(self.amplitudes))

    def fingerprint(self, decimals: int = 12) -> tuple:
        """
        A key equal for two states that differ by a global phase, up to
        `decimals` digits.

        >>> FockState([[1]], [1j]).fingerprint() == FockState(
        ...     [[1]], [-1]).fingerprint()
        True
        """
        amplitudes = self.amplitudes[np.abs(self.amplitudes) > 0]
        phase = np.conj(amplitudes[0]) / np.abs(amplitudes[0])
        rounded = np.round(self.amplitudes * phase + 0j, decimals) + 0
        return self.occupations.tobytes(), rounded.tobytes()


@dataclass
class Sweep:
    """
    The progressive measurement of the external outputs of a recurrent
    network whose photons are in the input modes `columns`, the loop and the
    occupied external inputs.

    The state of one tick lives in the modes spanned by the images of these
    inputs, the *held* modes, :math:`k + L` of them for :math:`k` occupied
    inputs. Measuring the external output :math:`e_j` needs only the span of
    :math:`e_j` and the held modes: when :math:`e_j` is not in the span, a
    vacuum mode along its orthogonal part joins the held modes, then the
    rotations `chains[j]` (applied in order) bring :math:`e_j` to the first
    held mode, which is measured and dropped. The held modes left after the
    last external output are in the span of the loop, `final` maps them onto
    it. The sweep never holds more than :math:`L + k + 1` modes, and the
    schedule does not depend on the outcomes.

    Parameters:
        loop : The number :math:`L` of loop modes.
        columns : The occupied input modes, the loop first.
        grows : Whether a vacuum mode joins before each external output.
        chains : The rotations bringing each external output to the first
            held mode.
        final : The phases and rotations of :func:`givens` mapping the held
            modes left at the end, padded with vacuum, onto the loop.

    >>> sweep = Sweep.from_unitary([[0, 1], [1, 0]], loop=1, columns=(0, 1))
    >>> sweep.grows, sweep.chains
    ((False,), ([],))
    """
    loop: int
    columns: tuple
    grows: tuple
    chains: tuple
    final: tuple
    tables: dict = field(default_factory=dict, repr=False)

    @classmethod
    def from_unitary(cls, unitary, loop: int, columns: tuple,
                     tol: float = 1e-9) -> Sweep:
        """
        The sweep of the mode matrix `unitary` with photons in `columns`.
        """
        unitary = np.asarray(unitary, dtype=complex)
        basis, grows, chains = unitary[:, list(columns)], [], []
        for target in np.eye(len(unitary))[loop:]:
            residual = target - basis @ (basis.conj().T @ target)
            grows.append(bool(np.linalg.norm(residual) > tol))
            if grows[-1]:
                basis = np.hstack([
                    basis, residual[:, None] / np.linalg.norm(residual)])
            overlaps = (basis.conj().T @ target)[:, None]
            chains.append(eliminate(overlaps, 0))
            change = np.eye(basis.shape[1], dtype=complex)
            for mode, rotation in chains[-1]:
                change[mode:mode + 2] = rotation @ change[mode:mode + 2]
            basis = (basis @ change.conj().T)[:, 1:]
        final = givens(complete(basis[:loop]))
        return cls(loop, tuple(columns), tuple(grows), tuple(chains), final)

    def widths(self) -> list:
        """
        The number of held modes before each external output and at the end.

        >>> Sweep.from_unitary(np.eye(3), loop=1, columns=(0, 2)).widths()
        [2, 2, 1]
        """
        result = [len(self.columns)]
        for grows in self.grows:
            result.append(result[-1] + grows - 1)
        return result

    def probed(self) -> list:
        """
        The number of held modes when each external output is measured.

        >>> Sweep.from_unitary(np.eye(3), loop=1, columns=(0, 2)).probed()
        [3, 2]
        """
        return [width + grows
                for width, grows in zip(self.widths(), self.grows)]

    def table(self, key: tuple, rotation: np.ndarray,
              total: int) -> np.ndarray:
        """
        :func:`twomode` of `rotation` on `total` photons, computed once and
        kept in :attr:`tables` under `key`.
        """
        if key + (total, ) not in self.tables:
            self.tables[key + (total, )] = twomode(rotation, total)
        return self.tables[key + (total, )]

    def start(self, state: FockState, inputs: tuple) -> np.ndarray:
        """
        The loop `state` and the photons injected into the occupied inputs,
        a vector of the :math:`N + |q|`-photon sector of the held modes.
        """
        photons = state.photons + sum(inputs)
        injected = [inputs[column - self.loop]
                    for column in self.columns[self.loop:]]
        occupations = np.hstack([
            state.occupations,
            np.tile(injected, (len(state.occupations), 1)).astype(int)])
        vector = np.zeros(
            len(sector(len(self.columns), photons)), dtype=complex)
        vector[position(occupations, photons)] = state.amplitudes
        return vector

    def probe(self, index: int, vector: np.ndarray,
              photons: int) -> np.ndarray:
        """
        The held state before external output `index`, with that output
        brought to the first held mode, ready to be measured.
        """
        modes = self.widths()[index]
        if self.grows[index]:
            vector, modes = pad(vector, modes, photons), modes + 1
        chain = self.chains[index][::-1]
        return rotate(
            np.ones(modes), chain, vector, photons,
            lambda step, total: self.table(
                (index, step), chain[step][1], total))

    def finish(self, vector: np.ndarray, photons: int) -> FockState:
        """ The held state after the last external output, on the loop. """
        modes = self.widths()[-1]
        vector = pad(vector, modes, photons, self.loop - modes)
        phases, rotations = self.final
        return FockState(sector(self.loop, photons), rotate(
            phases, rotations, vector, photons,
            lambda step, total: self.table(
                ("final", step), rotations[step][1], total)))

    def sample(self, state: FockState, inputs: tuple,
               rng: np.random.Generator) -> tuple:
        """
        The pattern on the external outputs, sampled one output at a time,
        and the loop state it leaves.
        """
        vector, photons = self.start(state, inputs), \
            state.photons + sum(inputs)
        pattern = np.zeros(len(self.grows), dtype=int)
        for index, modes in enumerate(self.probed()):
            probed = self.probe(index, vector, photons)
            probabilities = marginal(probed, modes, photons)
            pattern[index] = rng.choice(
                len(probabilities), p=normalise(probabilities))
            vector = collapse(probed, modes, photons, pattern[index])
            photons -= pattern[index]
        return pattern, self.finish(vector, photons)

    def outcomes(self, state: FockState, inputs: tuple):
        """
        Every pattern on the external outputs with its probability and the
        loop state it leaves, enumerated one output at a time.
        """
        branches = [((), 1., self.start(state, inputs),
                     state.photons + sum(inputs))]
        for index, modes in enumerate(self.probed()):
            branches = [
                (pattern + (number, ), weight * probability,
                 collapse(probed, modes, photons, number), photons - number)
                for pattern, weight, vector, photons in branches
                for probed in [self.probe(index, vector, photons)]
                for number, probability in enumerate(
                    marginal(probed, modes, photons))
                if probability > 1e-14]
        for pattern, weight, vector, photons in branches:
            yield pattern, weight, self.finish(vector, photons)


def pad(vector: np.ndarray, modes: int, photons: int,
        extra: int = 1) -> np.ndarray:
    """
    A `vector` of the `photons`-photon sector of `modes` modes, with `extra`
    modes in the vacuum appended.

    >>> pad(np.array([1, 2]), 2, 1).tolist()
    [0, 1, 2]
    """
    occupations = sector(modes + extra, photons)
    result = np.zeros(len(occupations), dtype=np.asarray(vector).dtype)
    result[occupations[:, modes:].sum(axis=1) == 0] = vector
    return result


def marginal(vector: np.ndarray, modes: int, photons: int) -> np.ndarray:
    """
    The probability of each photon number of the first mode in a `vector`
    of the `photons`-photon sector of `modes` modes.

    >>> marginal(np.array([0.6, 0, 0.8]), 2, 2).round(2).tolist()
    [0.36, 0.0, 0.64]
    """
    return np.bincount(sector(modes, photons)[:, 0],
                       np.abs(vector) ** 2, photons + 1)


def collapse(vector: np.ndarray, modes: int, photons: int,
             number: int) -> np.ndarray:
    """
    The normalised state of the other modes when the first mode of a
    `vector` of the `photons`-photon sector of `modes` modes holds `number`
    photons, a vector of the `photons - number`-photon sector.

    >>> collapse(np.array([0.6, 0, 0.8]), 2, 2, 0).tolist()
    [1.0]
    """
    rows = vector[sector(modes, photons)[:, 0] == number]
    return rows / np.linalg.norm(rows)


class Interferometer:
    """
    A passive recurrent linear-optical network: an interferometer whose
    first `loop` outputs are fed back into its first `loop` inputs one tick
    later.

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
        visible : The external outputs that are recorded, all of them by
            default; the others are measured and forgotten, as the
            environment of a loss or a discard.
        indistinguishability : The probability :math:`p` that an injected
            photon is in the internal state shared by all photons rather
            than in one of its own, the model of Renema et al. [RMC+18]_.

    >>> network = Interferometer([[0, 1], [1, 0]], loop=1, inputs=(1, ))
    >>> network
    Interferometer([[0, 1], [1, 0]], loop=1, inputs=(1,), transmissivity=1.0, \
efficiency=1.0, visible=(0,), indistinguishability=1.0)
    >>> eval(repr(network)) == network
    True
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(self, unitary, loop: int, inputs: tuple,
                 transmissivity: float = 1., efficiency: float = 1.,
                 visible: tuple = None, indistinguishability: float = 1.):
        self.unitary = np.asarray(unitary)
        self.loop, self.inputs = loop, tuple(int(q) for q in inputs)
        self.transmissivity = float(transmissivity)
        self.efficiency = float(efficiency)
        self.visible = tuple(range(len(self.inputs))) if visible is None \
            else tuple(int(i) for i in visible)
        self.indistinguishability = float(indistinguishability)
        if self.unitary.shape != (loop + len(self.inputs), ) * 2:
            raise ValueError(
                f"Expected a {loop + len(self.inputs)}-mode matrix, "
                f"got shape {self.unitary.shape}.")
        if not np.allclose(
                self.unitary.conj().T @ self.unitary, np.eye(len(unitary))):
            raise ValueError("The mode matrix must be unitary.")
        self.sweeps = {}

    @classmethod
    def from_diagram(cls, diagram, indistinguishability: float = 1.):
        """
        The recurrent network of a closed :class:`optyx.channel.Diagram` with
        feedback loops: its :meth:`one_step` is dilated to a passive matrix
        from the memory and the created photons to the outputs, the memory
        and the environment of its discards and losses, then completed to a
        unitary with vacuum inputs. The outputs of the diagram are recorded,
        its environment is not.

        >>> from optyx.channel import Diagram, qmode
        >>> from optyx.photonic import Create
        >>> delay = (Create(1) @ qmode >> Diagram.swap(qmode, qmode)
        ...     ).feedback(state=Create(0))
        >>> Interferometer.from_diagram(delay).distribution(2)
        {((0,), (1,)): 1.0}
        """
        # pylint: disable=import-outside-toplevel
        from optyx.channel import Swap
        initial, _ = diagram.boundary()
        step = diagram.one_step()
        loop = len(step.dom)
        if len(diagram.dom):
            raise ValueError("Only closed diagrams can be sampled.")
        if levels(initial) or levels(step):
            raise NotImplementedError(
                "Photons with internal states are not passive.")
        if any(not hasattr(box, "kraus") and not isinstance(box, Swap)
               for box in initial.boxes + step.boxes):
            raise NotImplementedError(
                "Classical boxes are not passive linear optics.")
        if any(ob.name != "qmode" for ob in (step.dom @ step.cod).inside):
            raise NotImplementedError(
                "Only optical modes can be sampled; leave the outputs as "
                "qmode, they are measured by number resolving detectors.")
        vacuum = initial.dilate().to_path()
        if vacuum.dom or any(vacuum.creations) or vacuum.udom != loop:
            raise NotImplementedError(
                "The loops must start in the vacuum, e.g. Create(0).")
        matrix = step.dilate().to_path()
        if matrix.selections or not np.isclose(
                matrix.normalisation * matrix.scalar, 1):
            raise NotImplementedError(
                "Only passive linear optics can be sampled: no selections.")
        outputs = len(diagram.cod)
        rows = np.asarray(matrix.array).T
        rows = np.vstack([rows[outputs:outputs + loop], rows[:outputs],
                          rows[outputs + loop:]])
        unitary = complete(rows)
        inputs = tuple(matrix.creations) + (0, ) * (
            len(unitary) - loop - len(matrix.creations))
        return cls(unitary, loop, inputs, visible=tuple(range(outputs)),
                   indistinguishability=indistinguishability)

    def sweep(self, inputs: tuple = None) -> Sweep:
        """
        The :class:`Sweep` of the loop and the occupied external inputs,
        computed once and kept in :attr:`sweeps`.
        """
        inputs = self.inputs if inputs is None else inputs
        columns = tuple(range(self.loop)) + tuple(
            self.loop + mode for mode, number in enumerate(inputs) if number)
        if columns not in self.sweeps:
            self.sweeps[columns] = Sweep.from_unitary(
                self.unitary, self.loop, columns)
        return self.sweeps[columns]

    def __repr__(self):
        return (f"Interferometer({self.unitary.tolist()}, loop={self.loop}, "
                f"inputs={self.inputs}, "
                f"transmissivity={self.transmissivity}, "
                f"efficiency={self.efficiency}, visible={self.visible}, "
                f"indistinguishability={self.indistinguishability})")

    def __eq__(self, other):
        return isinstance(other, Interferometer) \
            and np.array_equal(self.unitary, other.unitary) \
            and (self.loop, self.inputs, self.transmissivity,
                 self.efficiency, self.visible, self.indistinguishability) \
            == (other.loop, other.inputs, other.transmissivity,
                other.efficiency, other.visible, other.indistinguishability)

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
        >>> network = Interferometer(unitary, loop=3, inputs=(2, 2))
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
        :math:`2 K(\\bar q) \\sum_r \\arcsin^2 \\beta_r
        + (1 - p) \\bar q \\sum_r \\beta_r^2 \\leq` `tol`, for
        :math:`\\beta_r` the singular values of the :math:`k`-th power of
        :attr:`block` and
        :math:`K(\\bar q) = (\\bar q + 1)[\\sqrt{6 \\bar q (\\bar q + 1)}
        + \\bar q]`. The first term is half the trace distance bound between
        the loop states of the stationary boson sampling study, since any
        window is a channel applied to the loop; the second bounds the mean
        number of photons in a colour of their own still in the loop.

        >>> lossy = Interferometer(
        ...     [[0.6, 0.8], [0.8, -0.6]], 1, (1, ), transmissivity=.5)
        >>> lossy.burn_in(1e-3) < Interferometer(
        ...     [[0.6, 0.8], [0.8, -0.6]], 1, (1, )).burn_in(1e-3)
        True
        """
        qbar = max(self.inputs, default=0)
        constant = (qbar + 1) * (np.sqrt(6 * qbar * (qbar + 1)) + qbar)
        private = (1 - self.indistinguishability) * qbar
        power = np.eye(self.loop)
        for depth in range(max_depth + 1):
            values = np.clip(np.linalg.svd(power, compute_uv=False), 0, 1)
            if 2 * constant * np.sum(np.arcsin(values) ** 2) \
                    + private * np.sum(values ** 2) <= tol:
                return depth
            power = self.block @ power
        raise ValueError(
            f"The bound does not reach tol={tol} within {max_depth} ticks.")

    def detections(self, state: FockState, inputs: tuple = None):
        """
        Every pattern on the external outputs with its probability and the
        loop state it leaves, before loss on the loop.
        """
        inputs = self.inputs if inputs is None else inputs
        yield from self.sweep(inputs).outcomes(state, inputs)

    def losses(self, state: FockState, mode: int):
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
            yield weight, FockState(occupations, amplitudes).normalised()

    def thinnings(self, pattern: tuple):
        """
        Every pattern the detectors record when `pattern` reaches the visible
        outputs, with its probability.
        """
        pattern = [pattern[i] for i in self.visible]
        for recorded in product(*(range(n + 1) for n in pattern)):
            weight = prod(
                comb(n, k) * self.efficiency ** k
                * (1 - self.efficiency) ** (n - k)
                for n, k in zip(pattern, recorded))
            if weight > 0:
                yield recorded, weight

    def step(self, state: FockState, inputs: tuple,
             rng: np.random.Generator):
        """
        One tick of one colour: the pattern on the external outputs, sampled
        along the :meth:`sweep`, and the next loop state, after the loss on
        the loop.
        """
        pattern, state = self.sweep(inputs).sample(state, inputs, rng)
        if self.transmissivity < 1:
            for mode in range(self.loop):
                outcomes = list(self.losses(state, mode))
                state = outcomes[rng.choice(
                    len(outcomes),
                    p=normalise([weight for weight, _ in outcomes]))][1]
        return pattern, state

    def colour(self, rng: np.random.Generator):
        """
        The injection of one tick split into colours: the photons in the
        shared internal state, then one single photon per photon in an
        internal state of its own.
        """
        if self.indistinguishability == 1:
            return self.inputs, []
        shared = rng.binomial(self.inputs, self.indistinguishability)
        alone = [mode for mode, (total, kept) in enumerate(
            zip(self.inputs, shared)) for _ in range(total - kept)]
        return tuple(int(n) for n in shared), [
            tuple(int(mode == other) for mode in range(len(self.inputs)))
            for other in alone]

    def tick(self, colours: list, rng: np.random.Generator):
        """
        One tick of a trajectory with one loop state per colour, the shared
        colour first: the recorded pattern and the next colours. A colour
        of its own is forgotten once its photon has left the loop.
        """
        shared, alone = self.colour(rng)
        empty = (0, ) * len(self.inputs)
        work = [(colours[0], shared)] \
            + [(state, empty) for state in colours[1:]] \
            + [(FockState.vacuum(self.loop), inputs) for inputs in alone]
        total, after = np.zeros(len(self.inputs), dtype=int), []
        for state, inputs in work:
            pattern, state = self.step(state, inputs, rng)
            total, after = total + pattern, after + [state]
        after = after[:1] + [state for state in after[1:] if state.photons]
        recorded = rng.binomial(total[list(self.visible)], self.efficiency)
        return tuple(int(n) for n in recorded), after

    def trajectory(self, ticks: int, seed=None):
        """
        The recorded patterns and the loop photon numbers of `ticks` ticks
        of one trajectory started from the vacuum loop.
        """
        rng = np.random.default_rng(seed)
        colours, patterns, photons = [FockState.vacuum(self.loop)], [], []
        for _ in range(ticks):
            photons.append(sum(state.photons for state in colours))
            pattern, colours = self.tick(colours, rng)
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
        `ticks` ticks after `burn_in` ticks, by enumerating trajectories of
        indistinguishable photons.

        >>> delay = Interferometer([[0, 1], [1, 0]], loop=1, inputs=(1, ))
        >>> delay.distribution(2)
        {((0,), (1,)): 1.0}
        """
        if self.indistinguishability != 1:
            raise NotImplementedError(
                "The exact distribution enumerates indistinguishable photons "
                "only; sample partially distinguishable ones.")
        branches = [((), 1., FockState.vacuum(self.loop))]
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

    def kernel(self, cap: int = None, private: int = None) -> Kernel:
        """
        The :class:`Kernel` of this network, with the loop photon cap `cap`
        and the cap `private` on photons of their own in the loop. Both
        default to far in the tail of the stationary loop photon number of
        :meth:`occupation`, :math:`\\bar n + 6 \\sqrt{\\bar n} + 2`, and a
        trajectory past them is an error.
        """
        mean = self.occupation()
        cap = int(np.ceil(mean + 6 * np.sqrt(mean) + 2)) if cap is None \
            else cap
        mean *= 1 - self.indistinguishability
        private = int(np.ceil(mean + 6 * np.sqrt(mean) + 2)) \
            if private is None else private
        return Kernel(self, cap, private)

    def lost(self, state: FockState):
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


QUANTUM = ("qmode", "qubit")


@dataclass
class CQState:
    """
    A state of the wires of a diagram in a quantum trajectory: a pure state
    of the quantum wires and a definite value on each classical one.
    `values[i]` is the value of wire `i`, or `None` when it is quantum, in
    which case it is an axis of `vector`, in the order of the wires.
    `vector` is normalised, and `weight` is the factor its norm squared was
    divided by since the start of the last :meth:`Unravelling.run`.

    >>> CQState.empty()
    CQState(vector=array(1.), values=[], weight=1.0)
    """
    vector: np.ndarray
    values: list
    weight: float = 1.

    @staticmethod
    def empty() -> CQState:
        """ The state of no wires. """
        return CQState(np.ones(()), [])

    def axes(self, start: int, stop: int) -> list:
        """ The axes of `vector` of the quantum wires in `start:stop`. """
        return [self.values[:i].count(None) for i in range(start, stop)
                if self.values[i] is None]

    def measure(self, start: int, stop: int, rng: np.random.Generator):
        """
        Measure the wires in `start:stop` in the number basis: their values,
        sampled for the quantum ones, and the state of the other wires.
        """
        axes = self.axes(start, stop)
        outcome, vector = observe(self.vector, axes, rng)
        sampled = iter(outcome)
        return tuple(next(sampled) if value is None else value
                     for value in self.values[start:stop]), CQState(
            vector, self.values[:start] + self.values[stop:], self.weight)


def observe(vector: np.ndarray, axes: list, rng: np.random.Generator):
    """
    Sample the number basis on the `axes` of a pure state `vector`: the
    outcome and the normalised state of the other axes.

    >>> observe(np.array([[0, 0.6], [0.8, 0]]), [0], np.random.default_rng(0))
    ((1,), array([1., 0.]))
    """
    if not axes:
        return (), vector
    others = [axis for axis in range(vector.ndim) if axis not in axes]
    probabilities = np.sum(np.abs(vector) ** 2, axis=tuple(others)).ravel()
    flat = rng.choice(len(probabilities), p=normalise(probabilities))
    outcome = np.unravel_index(flat, [vector.shape[axis] for axis in axes])
    index = [slice(None)] * vector.ndim
    for axis, value in zip(axes, outcome):
        index[axis] = value
    collapsed = vector[tuple(index)]
    return tuple(int(n) for n in outcome), collapsed / np.sqrt(
        probabilities[flat])


class Unravelling:
    """
    The quantum trajectories of a closed :class:`optyx.channel.Diagram`
    with feedback loops, whatever it holds: optical modes, qubits,
    measurements, classical boxes and classically controlled boxes.

    A tick runs :meth:`one_step` layer by layer on a :class:`CQState`. A box
    contracts its Kraus map into the state, its classical inputs selecting
    the slice of their values, and its environment and classical outputs
    are measured in the number basis, sampled and collapsed: the Kraus
    operators :math:`(\\langle e| \\otimes 1) K` unravel the channel
    :math:`\\rho \\mapsto \\mathrm{tr}_E(K \\rho K^\\dagger)`. A classical box
    samples its outputs from its inputs. The outputs of the diagram are
    then measured and recorded, those of the memory carried to the next
    tick. The loops start in the state of :meth:`boundary`, run on the
    same way.

    Photons with internal states are sampled on the diagram inflated to
    as many :func:`levels`, every optical mode becoming as many modes whose
    counts the detectors add up.

    Each optical mode holds at most `cap` photons, a qubit is a two-level
    wire, so that every tick is a fixed sequence of contractions on fixed
    shapes; a trajectory past the cap raises, and so does a box that is not
    trace preserving, i.e. a postselection. The cost of a tick is
    exponential in the number of quantum wires alive at once: passive
    diagrams are better sampled by their :class:`Interferometer`.

    >>> from optyx.channel import Diagram, qmode
    >>> from optyx.photonic import Create
    >>> delay = (Create(1) @ qmode >> Diagram.swap(qmode, qmode)
    ...     ).feedback(state=Create(0))
    >>> Unravelling(delay).sample(ticks=3, seed=0)
    [(0,), (1,), (1,)]
    """
    def __init__(self, diagram, cap: int = 4):
        if diagram.dom:
            raise ValueError("Only closed diagrams can be sampled.")
        if cap < 1:
            raise ValueError("The cap must be at least one photon.")
        self.diagram, self.cap = diagram, int(cap)
        self.initial, _ = diagram.boundary()
        self.step = diagram.one_step()
        self.levels = max(levels(self.initial), levels(self.step))
        if self.levels:
            self.initial = self.initial.inflate(self.levels)
            self.step = self.step.inflate(self.levels)
        self.tensors = {}

    def __repr__(self):
        return f"Unravelling({self.diagram!r}, cap={self.cap})"

    def dimension(self, ob, value=None) -> int:
        """
        The dimension of a wire of type `ob`: `cap + 1` for an optical
        mode, two for a qubit or a bit, and enough for its `value` for a
        classical mode, at least two since a dimension one is the unit.
        """
        if ob.name == "qmode":
            return self.cap + 1
        if ob.name == "mode":
            return max(value + 1, 2)
        return 2

    def tensor(self, box, dims: tuple) -> np.ndarray:
        """
        The array of a box of a Kraus map, or of the density matrix of a
        classical box, on inputs of dimensions `dims`, with an axis per
        input then per output, computed once.
        """
        key = id(box), dims
        if key not in self.tensors:
            if hasattr(box, "density_matrix"):
                array = np.asarray(box.density_matrix.to_tensor(
                    input_dims=list(dims)).eval().array)
                if array.ndim != len(box.dom) + len(box.cod):
                    raise NotImplementedError(
                        f"{box} has an output of dimension one.")
            else:
                outputs = box.determine_output_dimensions(list(dims))
                array = np.asarray(box.truncation(
                    list(dims), outputs).eval().array).reshape(
                        dims + tuple(outputs))
            self.tensors[key] = box, array
        return self.tensors[key][1]

    def unitary(self, box):
        """
        The mode matrix of `box` when it is linear optics on optical modes,
        a unitary with no environment, created or selected photon, and
        `None` otherwise; computed once.
        """
        key = id(box), "unitary"
        if key not in self.tensors:
            unitary = None
            if not box.env and all(ob.name == "qmode" for ob in (
                    box.dom @ box.cod).inside):
                try:
                    matrix = box.kraus.to_path()
                    array = np.asarray(matrix.array, dtype=complex).T
                    if not matrix.creations and not matrix.selections \
                            and np.isclose(
                                matrix.normalisation * matrix.scalar, 1) \
                            and array.shape == (len(box.dom), ) * 2 \
                            and np.allclose(array.conj().T @ array,
                                            np.eye(len(array))):
                        unitary = givens(array)
                except NotImplementedError:
                    pass
            self.tensors[key] = box, unitary
        return self.tensors[key][1]

    def interfere(self, vector: np.ndarray, axes: list,
                  decomposition: tuple) -> np.ndarray:
        """
        Apply the mode matrix with :func:`givens` `decomposition` to the
        optical modes on `axes` of `vector`: a phase per mode, then each
        rotation as the :func:`twomode` blocks of its pair of modes, cut to
        `cap + 1` levels, raising when the cut loses amplitude.
        """
        phases, rotations = decomposition
        numbers = np.arange(self.cap + 1)
        for axis, phase in zip(axes, phases):
            shape = [1] * vector.ndim
            shape[axis] = self.cap + 1
            vector = vector * (phase ** numbers).reshape(shape)
        for step in reversed(range(len(rotations))):
            mode, matrix = rotations[step]
            first, second = axes[mode], axes[mode + 1]
            vector = self.cut(np.moveaxis(np.tensordot(
                vector, self.pair(id(decomposition), step, matrix),
                axes=([first, second], [2, 3])), [-2, -1], [first, second]),
                [first, second])
        return vector

    def cut(self, vector: np.ndarray, axes) -> np.ndarray:
        """
        Cut the `axes` of `vector` to `cap + 1` levels, raising when the cut
        loses amplitude.
        """
        for axis in axes:
            if vector.shape[axis] > self.cap + 1:
                if np.sum(np.abs(np.take(vector, range(
                        self.cap + 1, vector.shape[axis]), axis)) ** 2) \
                        > 1e-12:
                    raise ValueError(
                        f"A mode holds more than cap={self.cap} photons; "
                        "sample with a larger cap.")
                vector = np.take(vector, range(self.cap + 1), axis)
        return vector

    def pair(self, key, step: int, matrix: np.ndarray) -> np.ndarray:
        """
        The action of a two-mode `matrix` on two modes of at most `cap`
        photons each, an array with axes the two outputs, of up to
        `2 cap` photons, then the two inputs; computed once.
        """
        if (key, step) not in self.tensors:
            size = self.cap + 1
            array = np.zeros((2 * size - 1, ) * 2 + (size, size),
                             dtype=complex)
            for total in range(2 * size - 1):
                table = twomode(matrix, total)
                for k in range(max(0, total - size + 1),
                               min(total, size - 1) + 1):
                    array[np.arange(total + 1), total - np.arange(
                        total + 1), k, total - k] = table[:, k]
            self.tensors[key, step] = matrix, array
        return self.tensors[key, step][1]

    def evolve(self, vector: np.ndarray, start: int, box) -> np.ndarray:
        """
        Apply `box`, a box of a Kraus map, to the axes of `vector` from
        `start`: contract its array, put its outputs in their place and cut
        each to `cap + 1` levels, raising when the cut loses amplitude.
        """
        # pylint: disable=import-outside-toplevel
        from optyx.core.diagram import Swap
        width = len(box.dom)
        if isinstance(box, Swap):
            return np.swapaxes(vector, start, start + 1)
        array = self.tensor(box, vector.shape[start:start + width])
        outputs = array.ndim - width
        vector = np.tensordot(vector, array, axes=(
            list(range(start, start + width)), list(range(width))))
        return self.cut(np.moveaxis(
            vector, list(range(vector.ndim - outputs, vector.ndim)),
            list(range(start, start + outputs))),
            range(start, start + outputs))

    def apply(self, state: CQState, offset: int, box,
              rng: np.random.Generator) -> CQState:
        """ The state after `box`, on the wires from `offset`. """
        # pylint: disable=import-outside-toplevel
        from optyx.channel import Swap
        stop = offset + len(box.dom)
        values = state.values[offset:stop]
        if isinstance(box, Swap):
            vector = np.swapaxes(state.vector, *state.axes(offset, stop)) \
                if values.count(None) == 2 else state.vector
            return CQState(vector, state.values[:offset] + values[::-1]
                           + state.values[stop:], state.weight)
        if not hasattr(box, "kraus"):
            dims = tuple(self.dimension(ob, value)
                         for ob, value in zip(box.dom.inside, values))
            return self.classical(state, offset, box, self.tensor(
                box, dims)[tuple(values)], rng)
        unitary = self.unitary(box)
        if unitary is not None:
            return CQState(self.interfere(
                state.vector, state.axes(offset, stop), unitary),
                state.values, state.weight)
        return self.contract(state, offset, box, rng)

    def contract(self, state: CQState, offset: int, box,
                 rng: np.random.Generator) -> CQState:
        """
        The state after a `box` with a Kraus map, on the wires from
        `offset`: its inputs are moved to the last axes, its Kraus map is
        applied one box at a time by :meth:`evolve`, its classical outputs
        and its environment are measured and its quantum outputs put in
        their place.
        """
        vector, kept = self.embed(state, offset, box)
        for inner, inner_offset in zip(box.kraus.boxes, box.kraus.offsets):
            vector = self.evolve(vector, kept + inner_offset, inner)
        norm = np.sum(np.abs(vector) ** 2)
        if norm == 0:
            raise NotImplementedError(
                f"{box} annihilates the state: postselection cannot be "
                "sampled.")
        outputs = list(box.cod.inside)
        outcome, vector = observe(vector / np.sqrt(norm), [
            kept + i for i, ob in enumerate(outputs)
            if ob.name not in QUANTUM] + list(range(
                kept + len(outputs), vector.ndim)), rng)
        for axis, ob in enumerate(
                [ob for ob in outputs if ob.name in QUANTUM], kept):
            widths = [(0, 0)] * vector.ndim
            widths[axis] = (0, self.dimension(ob) - vector.shape[axis])
            vector = np.pad(vector, widths)
        before = state.values[:offset].count(None)
        sampled = iter(outcome)
        return CQState(np.moveaxis(
            vector, list(range(kept, vector.ndim)),
            list(range(before, before + vector.ndim - kept))),
            state.values[:offset] + [
                None if ob.name in QUANTUM else next(sampled)
                for ob in outputs] + state.values[offset + len(box.dom):],
            state.weight * norm)

    def embed(self, state: CQState, offset: int, box) -> tuple:
        """
        The vector of `state` with the inputs of `box` on its last axes,
        those of classical values as basis states, and the number of axes
        before them.
        """
        stop = offset + len(box.dom)
        vector, legs = state.vector, []
        axes = iter(state.axes(offset, stop))
        for ob, value in zip(box.dom.inside, state.values[offset:stop]):
            if value is None:
                legs.append(next(axes))
                continue
            vector = np.multiply.outer(vector, np.eye(
                self.dimension(ob, value))[value])
            legs.append(vector.ndim - 1)
        kept = vector.ndim - len(legs)
        return np.moveaxis(
            vector, legs, list(range(kept, vector.ndim))), kept

    def classical(self, state: CQState, offset: int, box,
                  probabilities: np.ndarray, rng: np.random.Generator):
        """
        The state after a classical `box` whose transition `probabilities`
        from the values of its inputs are given: its outputs sampled.
        """
        if any(ob.name in QUANTUM for ob in (box.dom @ box.cod).inside):
            raise NotImplementedError(
                f"{box} has no Kraus map and quantum wires: it cannot be "
                "sampled.")
        probabilities = np.real(probabilities)
        if np.sum(probabilities) <= 0:
            raise NotImplementedError(
                f"{box} annihilates the state: postselection cannot be "
                "sampled.")
        flat = rng.choice(probabilities.size,
                          p=normalise(probabilities.ravel()))
        outcome = np.unravel_index(flat, probabilities.shape)
        return CQState(state.vector, state.values[:offset] + [
            int(n) for n in outcome] + state.values[
                offset + len(box.dom):],
            state.weight * np.sum(probabilities))

    def run(self, diagram, state: CQState,
            rng: np.random.Generator) -> CQState:
        """
        The state after the layers of `diagram`, one box at a time, checked
        to be trace preserving as a whole: a box may scale the state, as a
        spider does until the scalar that normalises it, but a run that
        does not end with weight one is a postselection.
        """
        for layer in diagram:
            left, box, _ = layer.inside[0]
            state = self.apply(state, len(left), box, rng)
        if not np.isclose(state.weight, 1, atol=1e-6):
            raise NotImplementedError(
                f"The diagram is not trace preserving (weight "
                f"{state.weight:.3g}): postselection cannot be sampled.")
        return CQState(state.vector, state.values)

    def trajectory(self, ticks: int, seed=None, observable=None) -> list:
        """
        The outputs measured over `ticks` ticks of one trajectory started
        from the state of the loops, or the values of an `observable` of
        its outputs measured by :meth:`observe`.
        """
        rng = np.random.default_rng(seed)
        widths = [self.levels if self.levels and ob.name == "qmode" else 1
                  for ob in self.diagram.cod.inside]
        state, patterns = self.run(
            self.initial, CQState.empty(), rng), []
        for _ in range(ticks):
            state = self.run(self.step, state, rng)
            if observable is not None:
                value, state = self.observe(state, observable, rng)
                patterns.append(value)
                continue
            pattern, state = state.measure(0, sum(widths), rng)
            bounds = np.cumsum([0] + widths)
            patterns.append(tuple(sum(pattern[bounds[i]:bounds[i + 1]])
                                  for i in range(len(widths))))
        return patterns

    def spectrum(self, observable) -> tuple:
        """
        The eigenvalues and an orthonormal basis of eigenvectors, as
        columns, of the operator of an `observable` on the modes of at most
        `cap` photons, by the Schur decomposition of its normal matrix;
        computed once.
        """
        # pylint: disable=import-outside-toplevel
        from scipy.linalg import schur
        key = id(observable), "spectrum"
        if key not in self.tensors:
            dims = tuple(self.dimension(ob) for ob in observable.dom.inside)
            array = np.asarray(observable.operator.get_kraus().to_tensor(
                input_dims=list(dims)).eval().array)
            if array.ndim != 2 * len(dims):
                raise NotImplementedError(
                    f"{observable} has an output of dimension one.")
            array = array[(Ellipsis, ) + tuple(
                slice(size) for size in dims)]
            array = np.pad(array, [(0, 0)] * len(dims) + [
                (0, size - width) for size, width
                in zip(dims, array.shape[len(dims):])])
            matrix = array.reshape(prod(dims), prod(dims)).T
            if not np.allclose(matrix @ matrix.conj().T,
                               matrix.conj().T @ matrix):
                raise ValueError(
                    f"{observable} is not normal on modes of at most "
                    f"cap={self.cap} photons.")
            triangular, vectors = schur(matrix, output="complex")
            self.tensors[key] = observable, (np.diag(triangular), vectors)
        return self.tensors[key][1]

    def observe(self, state: CQState, observable,
                rng: np.random.Generator) -> tuple:
        """
        Measure an `observable` on the outputs of `state`, its first wires:
        an eigenvalue of its operator sampled with the probability of its
        eigenvector, and the state of the other wires left. The outputs are
        dropped, so measuring degenerate eigenspaces one eigenvector at a
        time leaves the other wires in the same mixture.
        """
        width = len(observable.dom)
        if self.levels or any(
                value is not None for value in state.values[:width]):
            raise NotImplementedError(
                "An operator is measured on quantum outputs without "
                "internal states.")
        values, vectors = self.spectrum(observable)
        amplitudes = vectors.conj().T @ state.vector.reshape(
            len(vectors), -1)
        probabilities = np.sum(np.abs(amplitudes) ** 2, axis=1)
        index = rng.choice(len(values), p=normalise(probabilities))
        value = values[index].real if np.isclose(values[index].imag, 0) \
            else values[index]
        return value, CQState(
            (amplitudes[index] / np.sqrt(probabilities[index])).reshape(
                state.vector.shape[width:]), state.values[width:])

    def sample(self, ticks: int, burn_in: int = 0, seed=None) -> list:
        """
        A sample of the outputs measured over `ticks` ticks, after
        `burn_in` ticks.
        """
        return self.trajectory(burn_in + ticks, seed)[burn_in:]


def levels(diagram) -> int:
    """
    The number of internal levels of the photons of a channel `diagram`,
    the length of the internal states of its boxes, zero when they have
    none.

    >>> from optyx.photonic import Create
    >>> levels(Create(1, internal_states=([1, 0], ))), levels(Create(1))
    (2, 0)
    """
    return max([len(states[0]) for box in diagram.boxes
                for inner in getattr(getattr(box, "kraus", None), "boxes", [])
                for states in [getattr(inner, "internal_states", None)]
                if states is not None] + [0])


def complete(isometry: np.ndarray) -> np.ndarray:
    """
    A unitary whose first columns are those of `isometry`, the others
    spanning the orthogonal complement of its range.

    >>> complete(np.array([[0.6], [0.8]])).round(2).tolist()
    [[0.6, -0.8], [0.8, 0.6]]
    """
    isometry = np.asarray(isometry)
    if not np.allclose(
            isometry.conj().T @ isometry, np.eye(isometry.shape[1])):
        raise ValueError("Only passive linear optics can be sampled.")
    _, _, conjugate = np.linalg.svd(isometry.conj().T)
    return np.hstack([isometry, conjugate[isometry.shape[1]:].conj().T])


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


@lru_cache(maxsize=None)
def binomials(modes: int, photons: int) -> np.ndarray:
    """
    The size :math:`\\binom{m + r - 1}{r}` of the :math:`r`-photon sector of
    :math:`m` modes, for every :math:`m \\leq` `modes` and
    :math:`r \\leq` `photons`, clipped to 32-bit integers.

    >>> binomials(3, 2).tolist()
    [[1, 0, 0], [1, 1, 1], [1, 2, 3], [1, 3, 6]]
    """
    table = np.array([[comb(m + r - 1, r) if m else int(r == 0)
                       for r in range(photons + 1)]
                      for m in range(modes + 1)])
    return np.minimum(table, np.iinfo(np.int32).max).astype(np.int32)


def rank(occupations, photons, table):
    """
    The positions in :func:`sector` of the rows of `occupations`, each with
    `photons` photons, from the :func:`binomials` `table`: the combinatorial
    number system, where mode :math:`i` with :math:`r_i` photons left to
    place contributes :math:`D(W - i, r_i) - D(W - i, r_i - n_i)` for
    :math:`D(m, r)` the size of the :math:`r`-photon sector of :math:`m`
    modes. Rows with another number of photons have no meaning.

    >>> rank(sector(3, 2), 2, binomials(3, 2)).tolist()
    [0, 1, 2, 3, 4, 5]
    """
    occupations, table = jnp.asarray(occupations, jnp.int32), \
        jnp.asarray(table)
    remaining = jnp.asarray(photons, dtype=jnp.int32)[..., None] \
        - jnp.cumsum(occupations, axis=-1) + occupations
    modes = jnp.arange(occupations.shape[-1], 0, -1)
    return jnp.sum(
        table[modes, jnp.clip(remaining, 0, table.shape[1] - 1)]
        - table[modes, jnp.clip(remaining - occupations, 0,
                                table.shape[1] - 1)], axis=-1)


@lru_cache(maxsize=None)
def expansion(photons: int) -> tuple:
    """
    The terms of :func:`twomode` for every number of photons up to
    `photons`: the flat index of the entry :math:`(s, m, k)` each adds to,
    its coefficient and the powers of :math:`u_{00}, u_{10}, u_{01},
    u_{11}` it multiplies.
    """
    terms = [
        ((total * (photons + 1) + m) * (photons + 1) + k,
         comb(k, a) * comb(total - k, m - a) * np.sqrt(
             factorial(m) * factorial(total - m)
             / factorial(k) / factorial(total - k)),
         (a, k - a, m - a, total - k - m + a))
        for total in range(photons + 1)
        for k in range(total + 1)
        for m in range(total + 1)
        for a in range(max(0, m - total + k), min(k, m) + 1)]
    index, coefficients, exponents = zip(*terms)
    return np.array(index), np.array(coefficients), np.array(exponents)


def powers(values, photons: int):
    """
    The powers :math:`0, \\dots,` `photons` of each of `values`, one row
    each, as products so that they are differentiable at zero.
    """
    values = jnp.asarray(values)
    return jnp.cumprod(jnp.concatenate([
        jnp.ones(values.shape + (1, ), values.dtype),
        jnp.repeat(values[..., None], photons, axis=-1)], axis=-1), axis=-1)


def twomodes(matrix, photons: int):
    """
    :func:`twomode` of `matrix` on every number :math:`s \\leq` `photons` of
    photons at once, a tensor with entries :math:`(s, m, k)`, written with
    JAX so that it is differentiable in `matrix`.

    >>> splitter = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    >>> bool(np.allclose(twomodes(splitter, 2)[2], twomode(splitter, 2)))
    True
    """
    index, coefficients, exponents = expansion(photons)
    matrix = jnp.asarray(matrix)
    table = powers(jnp.stack([matrix[0, 0], matrix[1, 0], matrix[0, 1],
                              matrix[1, 1]]), photons)
    terms = coefficients * jnp.prod(
        table[jnp.arange(4), exponents], axis=-1)
    return jnp.zeros((photons + 1) ** 3, terms.dtype).at[index].add(
        terms).reshape((photons + 1, ) * 3)


def turn(vector, occupations, photons: int, mode, tensor, table):
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals
    """
    The rotation with :func:`twomodes` `tensor` of the modes `mode` and
    `mode + 1` of a `vector` of the `photons`-photon sector with
    `occupations`: the amplitude of each occupation gathers those of the
    occupations differing from it on these two modes. Their :func:`rank`
    differs only in the terms of these two modes, which depend on the
    photons :math:`k` put in the first one only through
    :math:`g(r - k)`, for :math:`r` the photons left before it and
    :math:`g` the difference of two rows of the :func:`binomials` `table`.
    """
    size, modes = occupations.shape
    first = jnp.take(occupations, mode, axis=1)
    total = first + jnp.take(occupations, mode + 1, axis=1)
    remaining = photons - jnp.take(
        np.cumsum(occupations, axis=1) - occupations, mode, axis=1)
    numbers = jnp.arange(photons + 1)
    table = jnp.asarray(table)
    difference = table[modes - mode - 1, :photons + 1] \
        - table[modes - mode, :photons + 1]
    targets = jnp.arange(size)[:, None] - difference[remaining - first][
        :, None] + difference[jnp.clip(
            remaining[:, None] - numbers[None, :], 0, photons)]
    targets = jnp.where(numbers[None, :] <= total[:, None], targets, size)
    padded = jnp.concatenate([vector, jnp.zeros(1, vector.dtype)])
    rows = tensor.reshape(-1, photons + 1)[total * (photons + 1) + first]
    return jnp.sum(rows * padded[targets], axis=1)


def move(vector, targets, size: int):
    """
    The vector of length `size` with the entries of `vector` added at
    `targets`, those at `size` being dropped.
    """
    return jnp.zeros(size + 1, vector.dtype).at[targets].add(vector)[:size]


def givens_rotation(top, bottom):
    """
    The rotation :math:`r` of two adjacent modes with
    :math:`r (t, b)^T = (\\sqrt{|t|^2 + |b|^2}, 0)^T`, as in
    :func:`eliminate` but without branching, the identity when both are
    zero, and that norm.
    """
    square = jnp.real(top * jnp.conj(top) + bottom * jnp.conj(bottom))
    safe = square > 1e-30
    top, bottom = jnp.where(safe, top, 1), jnp.where(safe, bottom, 0)
    norm = jnp.sqrt(jnp.where(safe, square, 1))
    matrix = jnp.array([[jnp.conj(top), jnp.conj(bottom)],
                        [-bottom, top]]) / norm
    return jnp.where(safe, matrix, jnp.eye(2, dtype=matrix.dtype)), \
        jnp.where(safe, norm, 0)


def safe_log(probability):
    """ The logarithm of `probability`, zero where it vanishes. """
    return jnp.log(jnp.where(probability > 0, probability, 1))


class Kernel:
    """
    One tick of an :class:`Interferometer` as a Markov kernel on the states
    of its loop, written with JAX: fixed shapes, so that it compiles, runs
    over many trajectories at once and differentiates.

    It runs the :class:`Sweep` of :meth:`Interferometer.step` with every
    photon number up to the cap :math:`M = n^\\star + |q|` compiled once:
    each external output switches on the photon number :math:`n` of the
    held state and works in the :math:`n`-photon sector of the held modes,
    a vector held in a buffer of the size of the :math:`M`-photon sector.
    The occupations that a rotation, a collapse or a loss pairs are found by
    :func:`rank`, so the only tables are the occupations of each sector.
    The loop is held between ticks in the sectors of :math:`n^\\star` photons
    in :math:`L + 1` modes, the last one counting the photons missing to
    :math:`n^\\star`, so that every photon number has its place.

    Photons in an internal state of their own are a padded batch of at most
    `private` single-photon walks. A trajectory whose loop holds more than
    `cap` photons, or more than `private` photons of their own, is flagged
    as an overflow.

    Every outcome a trajectory samples is recorded, so that
    :meth:`log_prob` replays it: the score-function estimator of the
    gradient of an expectation :math:`E_\\theta[f]` is the mean of
    :math:`f \\nabla_\\theta \\log p_\\theta`. The parameters are those of
    :meth:`parameters`; they change the values of the loss, the efficiency
    and the indistinguishability but not whether each is simulated, which
    is decided by the :class:`Interferometer`.

    Parameters:
        interferometer : The interferometer.
        cap : The photon cap :math:`n^\\star` of the loop.
        private : The cap on photons of their own in the loop.

    >>> delay = Interferometer([[0, 1], [1, 0]], loop=1, inputs=(1, ))
    >>> delay.kernel(cap=2).sample(ticks=3, shots=2).tolist()
    [[[0], [1], [1]], [[0], [1], [1]]]
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-locals,too-many-public-methods
    def __init__(self, interferometer: Interferometer, cap: int,
                 private: int = 0):
        if jax is None:  # pragma: no cover
            raise ImportError("The kernel needs JAX: pip install jax.")
        self.interferometer, self.cap = interferometer, int(cap)
        self.private = int(private) \
            if interferometer.indistinguishability < 1 else 0
        self.sweep = interferometer.sweep()
        self.photons = self.cap + sum(interferometer.inputs)
        self.width = max(self.sweep.probed() + [len(self.sweep.columns)])
        self.table = binomials(
            max(self.sweep.probed() + [interferometer.loop]) + 2,
            self.photons)
        self.compiled = {}

    def __repr__(self):
        return (f"Kernel({self.interferometer!r}, cap={self.cap}, "
                f"private={self.private})")

    def __eq__(self, other):
        return isinstance(other, Kernel) \
            and self.interferometer == other.interferometer \
            and (self.cap, self.private) == (other.cap, other.private)

    def size(self, modes: int, photons: int = None) -> int:
        """
        The length of a buffer of the sectors of `modes` modes with at most
        `photons` photons, :math:`M` by default.
        """
        photons = self.photons if photons is None else photons
        return comb(modes + photons - 1, photons) if modes else 1

    @staticmethod
    def occupations(modes: int, photons: int) -> np.ndarray:
        """ The occupations of :func:`sector` as 32-bit integers. """
        return sector(modes, photons).astype(np.int32)

    def parameters(self) -> dict:
        """
        The values the kernel differentiates in: the mode matrix, the
        transmissivity, the efficiency and the indistinguishability.
        """
        network = self.interferometer
        return {"unitary": jnp.asarray(network.unitary, dtype=complex),
                "transmissivity": jnp.asarray(network.transmissivity),
                "efficiency": jnp.asarray(network.efficiency),
                "indistinguishability":
                    jnp.asarray(network.indistinguishability)}

    def schedule(self, unitary) -> tuple:
        """
        The rotations of the :class:`Sweep` computed from the mode matrix
        with :func:`givens_rotation`, so that they are differentiable in it:
        for each external output the chain bringing it to the first held mode,
        padded with identities to the same length and stacked, then the
        phases and rotations mapping the held modes left onto the loop, as
        in :func:`givens`, all in the order they are applied.
        """
        unitary, loop = jnp.asarray(unitary), self.interferometer.loop
        basis, chains = unitary[:, list(self.sweep.columns)], []
        for index, grows in enumerate(self.sweep.grows):
            target = jnp.eye(len(unitary), dtype=unitary.dtype)[loop + index]
            if grows:
                residual = target - basis @ (basis.conj().T @ target)
                basis = jnp.hstack([basis, residual[:, None] / jnp.sqrt(
                    jnp.real(jnp.vdot(residual, residual)))])
            column, chain = basis.conj().T @ target, []
            for row in range(basis.shape[1] - 1, 0, -1):
                matrix, norm = givens_rotation(column[row - 1], column[row])
                column = column.at[row - 1].set(norm).at[row].set(0)
                basis = basis.at[:, row - 1:row + 1].set(
                    basis[:, row - 1:row + 1] @ matrix.conj().T)
                chain.append((row - 1, matrix))
            chains += chain + [(0, jnp.eye(2, dtype=unitary.dtype))] * (
                self.width - basis.shape[1])
            basis = basis[:, 1:]
        current, final = basis[:loop], []
        for column in range(min(current.shape[1], loop - 1)):
            for row in range(loop - 1, column, -1):
                matrix, _ = givens_rotation(
                    current[row - 1, column], current[row, column])
                current = current.at[row - 1:row + 1].set(
                    matrix @ current[row - 1:row + 1])
                final.append((row - 1, matrix.conj().T))
        phases = jnp.concatenate([
            jnp.diagonal(current),
            jnp.ones(loop - current.shape[1], unitary.dtype)])
        modes, tensors = self.tensors(chains)
        steps = (len(self.sweep.grows), self.width - 1)
        return (modes.reshape(steps), tensors.reshape(
            steps + tensors.shape[1:])), phases, self.tensors(final[::-1])

    def tensors(self, chain: list) -> tuple:
        """
        The modes of the rotations `chain`, pairs of a mode and a matrix,
        and their :func:`twomodes` tensors up to :math:`M` photons.
        """
        indices = np.array([mode for mode, _ in chain], dtype=np.int32)
        if not chain:
            return indices, jnp.zeros((0, ) + (self.photons + 1, ) * 3)
        return indices, jax.vmap(lambda matrix: twomodes(
            matrix, self.photons))(jnp.stack([matrix for _, matrix in chain]))

    def rotate(self, vector, modes: int, photons: int, chain: tuple):
        """
        Apply the rotations `chain` of :meth:`schedule`, their modes and
        :func:`twomodes` tensors, to a `vector` of the `photons`-photon
        sector of `modes` modes, as one scan.
        """
        indices, tensors = chain
        if indices.size == 0:
            return vector
        occupations = self.occupations(modes, photons)

        def body(vector, step):
            mode, tensor = step
            return turn(vector, occupations, photons, mode, tensor,
                        self.table), None
        return jax.lax.scan(body, vector, (
            jnp.asarray(indices),
            tensors[:, :photons + 1, :photons + 1, :photons + 1]))[0]

    def start(self, loop, photons, injected):
        """
        The held state of the loop buffer with `photons` photons and the
        `injected` photons in the occupied inputs, in the buffer of the held
        modes, the modes past them in the vacuum.
        """
        occupations = self.occupations(
            self.interferometer.loop + 1, self.cap)
        held = jnp.hstack([
            occupations[:, :-1],
            jnp.broadcast_to(injected, (len(occupations), len(injected))),
            jnp.zeros((len(occupations), self.width - len(self.sweep.columns)),
                      jnp.int32)])
        targets = jnp.where(
            occupations[:, -1] == self.cap - photons,
            rank(held, photons + jnp.sum(injected), self.table),
            self.size(self.width))
        return move(loop, targets, self.size(self.width))

    def measure(self, held, photons, chain: tuple, key, forced):
        """
        Measure the next external output of the held state, `chain` bringing
        it to the first held mode: switch on the photon number of the held
        state, rotate, sample the photon number of the first mode (or take
        the `forced` one) and collapse, shifting the other modes down by one.
        Returns the next held state, its photon number, the outcome, its log
        probability and whether it was possible.
        """
        def branch(photons):
            def run(held):
                vector = self.rotate(
                    held[:comb(self.width + photons - 1, photons)],
                    self.width, photons, chain)
                occupations = self.occupations(self.width, photons)
                probabilities = jax.ops.segment_sum(
                    jnp.abs(vector) ** 2, occupations[:, 0],
                    self.photons + 1)
                number = choose(key, probabilities, forced)
                probability = probabilities[number]
                targets = jnp.where(
                    occupations[:, 0] == number,
                    rank(jnp.hstack([
                        occupations[:, 1:],
                        jnp.zeros((len(occupations), 1), jnp.int32)]),
                        photons - number, self.table),
                    len(occupations))
                return held.at[:len(occupations)].set(move(
                    vector / jnp.sqrt(jnp.where(
                        probability > 0, probability, 1)), targets,
                    len(occupations))), number, \
                    safe_log(probability), probability > 0
            return run
        held, number, logp, possible = jax.lax.switch(
            photons, [branch(n) for n in range(self.photons + 1)], held)
        return held, (photons - number).astype(jnp.int32), number, logp, \
            possible

    def land(self, phases, final, held, photons, transmissivity, subkeys,
             forced):
        """
        Map the held modes left after the last output onto the loop, sample
        the photons each loop mode loses (or take the `forced` numbers) and
        store the loop in its buffer. Returns the buffer, the loop photon
        number, the losses, their log probability, whether they were
        possible and whether the loop holds more than the cap.
        """
        loop, modes = self.interferometer.loop, self.sweep.widths()[-1]
        lossy = self.interferometer.transmissivity < 1

        def branch(photons):
            def run(held):
                occupations = self.occupations(self.width, photons)
                vector = move(
                    held[:len(occupations)], jnp.where(
                        jnp.sum(occupations[:, modes:], axis=1) == 0,
                        rank(jnp.hstack([occupations[:, :modes], jnp.zeros(
                            (len(occupations), loop + 1 - modes),
                            jnp.int32)]), photons, self.table),
                        comb(loop + photons, photons)),
                    comb(loop + photons, photons))
                occupations = self.occupations(loop + 1, photons)
                vector = vector * jnp.prod(powers(phases, photons)[
                    jnp.arange(loop), occupations[:, :-1]], axis=1)
                vector = self.rotate(vector, loop + 1, photons, final)
                lost = jnp.zeros(loop, jnp.int32)
                logp, possible = 0., True
                for mode in range(loop * lossy):
                    vector, number, step_logp, step_possible = self.lose(
                        vector, occupations, mode, photons, transmissivity,
                        subkeys[mode],
                        None if forced is None else forced[mode])
                    lost = lost.at[mode].set(number)
                    logp, possible = logp + step_logp, \
                        possible & step_possible
                left = photons - jnp.sum(lost)
                targets = jnp.where(
                    (occupations[:, -1] == jnp.sum(lost)) & (left <= self.cap),
                    rank(jnp.hstack([occupations[:, :-1], jnp.broadcast_to(
                        self.cap - left, (len(occupations), 1))]),
                        self.cap, self.table),
                    self.size(loop + 1, self.cap))
                buffer = move(vector, targets, self.size(loop + 1, self.cap))
                return buffer, lost, logp, possible
            return run
        buffer, lost, logp, possible = jax.lax.switch(
            photons, [branch(n) for n in range(self.photons + 1)], held)
        left = photons - jnp.sum(lost)
        overflow = left > self.cap
        vacuum = jnp.zeros_like(buffer).at[0].set(1)
        return jnp.where(overflow, vacuum, buffer), \
            jnp.where(overflow, 0, left).astype(jnp.int32), lost, logp, \
            possible, overflow

    def lose(self, vector, occupations, mode: int, photons: int,
             transmissivity, key, forced):
        """
        Sample the number of photons that loop `mode` loses (or take the
        `forced` one) and move them to the last mode, which counts them: the
        Kraus operator losing :math:`l` of :math:`n` photons has amplitude
        :math:`\\sqrt{\\binom{n}{l} \\gamma^{n - l} (1 - \\gamma)^l}`.
        """
        number = occupations[:, mode]
        lost = jnp.arange(photons + 1)
        kept = jnp.maximum(number[:, None] - lost[None, :], 0)
        amplitudes = jnp.where(
            lost[None, :] <= number[:, None],
            jnp.sqrt(jnp.asarray(choices(photons))[number])
            * powers(jnp.sqrt(transmissivity), photons)[kept]
            * powers(jnp.sqrt(1 - transmissivity), photons)[lost][None, :],
            0)
        probabilities = jnp.zeros(self.photons + 1).at[:photons + 1].set(
            jnp.abs(vector) ** 2 @ jnp.abs(amplitudes) ** 2)
        chosen = choose(key, probabilities, forced)
        probability = probabilities[chosen]
        targets = jnp.where(
            number >= chosen,
            rank(jnp.asarray(occupations).at[:, mode].add(-chosen)
                 .at[:, -1].add(chosen),
                 photons, self.table), len(vector))
        return move(vector * jnp.take(amplitudes, jnp.minimum(
            chosen, photons), axis=1)
                    / jnp.sqrt(jnp.where(probability > 0, probability, 1)),
                    targets, len(vector)), \
            chosen, safe_log(probability), probability > 0

    def walk(self, unitary, transmissivity, walkers, active, private,
             subkeys, forced):
        """
        One tick of the photons in an internal state of their own, each a
        single-photon walk: those in the loop (`walkers`, the loop amplitudes,
        where `active`) and those injected now (where `private`, one per
        injected photon). Each is detected at an external output or stays in
        the loop, where it survives the loss or not. Returns the detections,
        the walks left, compacted to the first `private` slots, the outcomes,
        their log probability, whether they were possible and whether the
        walks left overflow their slots.
        """
        loop = self.interferometer.loop
        external = len(self.interferometer.inputs)
        injected = np.zeros((len(self.injected()), loop + external))
        injected[np.arange(len(injected)), loop + self.injected()] = 1
        inputs = jnp.vstack([jnp.hstack([
            walkers, jnp.zeros((len(walkers), external), walkers.dtype)]),
            jnp.asarray(injected, walkers.dtype)])
        alive = jnp.concatenate([active, private])
        outputs = inputs @ unitary.T
        stay = jnp.sum(jnp.abs(outputs[:, :loop]) ** 2, axis=1)
        probabilities = jnp.hstack([
            jnp.abs(outputs[:, loop:]) ** 2, stay[:, None]])
        steps = choose(subkeys[0], probabilities,
                       None if forced is None else forced["walk"])
        stays = steps == external
        logp = jnp.where(alive, safe_log(jnp.take_along_axis(
            probabilities, steps[:, None], axis=1)[:, 0]), 0)
        possible = jnp.all(~alive | (jnp.take_along_axis(
            probabilities, steps[:, None], axis=1)[:, 0] > 0))
        survive = jnp.ones(len(alive), bool)
        if self.interferometer.transmissivity < 1:
            survive = jax.random.bernoulli(
                subkeys[1], transmissivity, (len(alive), )) \
                if forced is None else forced["survive"]
            logp = logp + jnp.where(alive & stays, jnp.where(
                survive, jnp.log(transmissivity),
                safe_log(1 - transmissivity)), 0)
        detections = jnp.sum(jax.nn.one_hot(steps, external + 1)[:, :external]
                             * alive[:, None], axis=0).astype(jnp.int32)
        left = alive & stays & survive
        walkers = outputs[:, :loop] / jnp.sqrt(jnp.where(
            stay > 0, stay, 1))[:, None]
        order = jnp.argsort(~left, stable=True)[:self.private]
        return detections, walkers[order], left[order], \
            {"walk": steps, "survive": survive}, jnp.sum(logp), possible, \
            jnp.sum(left) > self.private

    def injected(self) -> np.ndarray:
        """ The external input of each photon injected at a tick. """
        return np.array([mode for mode, number in enumerate(
            self.interferometer.inputs) for _ in range(number)], dtype=int)

    def vacuum(self, dtype=complex) -> tuple:
        """
        The state of a trajectory started from the vacuum: the loop buffer,
        its photon number, the private walks and which are active.
        """
        return (jnp.zeros(self.size(self.interferometer.loop + 1, self.cap),
                          dtype).at[0].set(1),
                jnp.asarray(0, jnp.int32),
                jnp.zeros((self.private, self.interferometer.loop), dtype),
                jnp.zeros(self.private, bool))

    def colour(self, probability, key, forced):
        """
        Which injected photons are in an internal state of their own, each
        with probability :math:`1 - p`, or the `forced` ones; with their log
        probability.
        """
        injected = len(self.injected())
        if self.interferometer.indistinguishability == 1:
            return jnp.zeros(injected, bool), 0.
        private = jax.random.bernoulli(key, 1 - probability, (injected, )) \
            if forced is None else forced
        return private, jnp.sum(jnp.where(
            private, safe_log(1 - probability), safe_log(probability)))

    def shared(self, params: dict, schedule: tuple, loop, photons, private,
               subkeys, forced):
        """
        One tick of the photons in the shared internal state: inject those
        not `private`, measure every external output along the
        :class:`Sweep` and land on the loop. Returns the loop buffer and its
        photon number, the pattern and the losses, their log probability,
        whether they were possible and whether the loop overflows.
        """
        chains, phases, final = schedule
        loop_modes = self.interferometer.loop
        columns = np.array(self.sweep.columns[loop_modes:]) - loop_modes
        injected = jnp.asarray(self.interferometer.inputs)[columns] \
            - jax.ops.segment_sum(private.astype(jnp.int32), jnp.asarray(
                np.searchsorted(columns, self.injected())), len(columns))
        external = len(self.sweep.grows)

        def step(carry, step):
            chain, key, number = step
            held, held_photons, number, logp, possible = self.measure(
                *carry, chain, key, None if forced is None else number)
            return (held, held_photons), (number, logp, possible)
        (held, held_photons), (pattern, logp, possible) = jax.lax.scan(
            step, (self.start(loop, photons, injected),
                   (photons + jnp.sum(injected)).astype(jnp.int32)),
            (chains, jax.random.split(subkeys[0], external) if forced is None
             else jnp.zeros((external, 2), jnp.uint32),
             jnp.zeros(external, jnp.int32) if forced is None
             else forced["shared"]))
        loop, photons, lost, land_logp, land_possible, overflow = self.land(
            phases, final, held, held_photons, params["transmissivity"],
            jax.random.split(subkeys[1], loop_modes) if forced is None
            else [None] * loop_modes,
            None if forced is None else forced["lost"])
        return loop, photons, pattern, lost, jnp.sum(logp) + land_logp, \
            jnp.all(possible) & land_possible, overflow

    def thin(self, efficiency, detected, key, forced):
        """
        The numbers recorded by detectors of efficiency :math:`\\eta` on the
        `detected` photons, binomially, or the `forced` ones; with their log
        probability and whether they were possible.
        """
        if self.interferometer.efficiency == 1:
            return detected, 0., True
        recorded = jax.random.binomial(key, detected, efficiency).astype(
            jnp.int32) if forced is None else forced
        gammaln = jax.scipy.special.gammaln
        possible = recorded <= detected
        return recorded, jnp.sum(jnp.where(
            possible, gammaln(detected + 1.) - gammaln(recorded + 1.)
            - gammaln(detected - recorded + 1.)
            + recorded * safe_log(efficiency)
            + (detected - recorded) * safe_log(1 - efficiency), 0)), \
            jnp.all(possible)

    def tick(self, params: dict, schedule: tuple, state: tuple, key,
             forced: dict = None):
        """
        One tick of a trajectory from `state`, sampling with `key`, or
        replaying the `forced` outcomes of an earlier tick when given. Returns
        the next state and a dictionary of the recorded pattern, the
        outcomes, their log probability, whether they were possible and
        whether the loop overflows.
        """
        loop, photons, walkers, active = state
        subkeys = jax.random.split(key, 6) if forced is None else [None] * 6
        forced = {} if forced is None else forced
        private, logp = self.colour(
            params["indistinguishability"], subkeys[0], forced.get("private"))
        loop, photons, pattern, lost, step_logp, possible, overflow = \
            self.shared(params, schedule, loop, photons, private,
                        subkeys[1:3], forced or None)
        outcomes = {"private": private, "shared": pattern, "lost": lost,
                    "walk": jnp.zeros(len(private) + self.private, jnp.int32),
                    "survive": jnp.zeros(len(private) + self.private, bool)}
        logp = logp + step_logp
        if self.interferometer.indistinguishability < 1:
            detections, walkers, active, walked, step_logp, step_possible, \
                crowded = self.walk(
                    params["unitary"], params["transmissivity"], walkers,
                    active, private, subkeys[3:5], forced or None)
            outcomes.update(walked)
            pattern = pattern + detections
            logp, possible = logp + step_logp, possible & step_possible
            overflow = overflow | crowded
        recorded, step_logp, step_possible = self.thin(
            params["efficiency"],
            pattern[np.array(self.interferometer.visible, dtype=int)],
            subkeys[5], forced.get("recorded"))
        outcomes["recorded"] = recorded
        return (loop, photons, walkers, active), {
            "recorded": recorded, "outcomes": outcomes,
            "logp": logp + step_logp, "possible": possible & step_possible,
            "overflow": overflow}

    def run(self, key, ticks: int, params: dict = None) -> dict:
        """
        One trajectory of `ticks` ticks from the vacuum loop, sampled with
        `key`: the outputs of :meth:`tick`, stacked over ticks.
        """
        params = self.parameters() if params is None else params
        schedule = self.schedule(params["unitary"])

        def body(state, key):
            return self.tick(params, schedule, state, key)
        return jax.lax.scan(body, self.vacuum(params["unitary"].dtype),
                            jax.random.split(key, ticks))[1]

    def trajectories(self, ticks: int, shots: int = 1, seed: int = 0,
                     params: dict = None, batch_size: int = None) -> dict:
        """
        `shots` independent trajectories of `ticks` ticks from the vacuum
        loop: :meth:`run`, compiled once and mapped over the shots,
        `batch_size` at a time in parallel.
        """
        params = self.parameters() if params is None else params
        if ticks not in self.compiled:
            self.compiled[ticks] = jax.jit(
                lambda keys, params: jax.lax.map(
                    lambda key: self.run(key, ticks, params), keys,
                    batch_size=batch_size))
        return self.compiled[ticks](
            jax.random.split(jax.random.PRNGKey(seed), shots), params)

    def sample(self, ticks: int, burn_in: int = 0, shots: int = 1,
               seed: int = 0, params: dict = None) -> np.ndarray:
        """
        The patterns recorded over `ticks` ticks after `burn_in` ticks, for
        `shots` trajectories: an array of shape
        `(shots, ticks, len(visible))`. Raises when a trajectory overflows
        the caps.
        """
        result = self.trajectories(burn_in + ticks, shots, seed, params)
        if np.any(result["overflow"]):
            raise ValueError(
                f"A trajectory overflowed the caps cap={self.cap}, "
                f"private={self.private}; build a kernel with larger ones.")
        return np.asarray(result["recorded"])[:, burn_in:]

    def log_prob(self, outcomes: dict, params: dict = None):
        """
        The log probability of the trajectories whose `outcomes` were
        recorded by :meth:`trajectories`, replayed from the vacuum loop, and
        whether each was possible: differentiable in `params`.
        """
        params = self.parameters() if params is None else params
        schedule = self.schedule(params["unitary"])

        def replay(outcomes):
            def body(state, forced):
                state, result = self.tick(params, schedule, state, None,
                                          forced)
                return state, (result["logp"], result["possible"])
            _, (logp, possible) = jax.lax.scan(
                body, self.vacuum(params["unitary"].dtype), outcomes)
            return jnp.sum(logp), jnp.all(possible)
        return jax.lax.map(replay, outcomes)

    def distribution(self, ticks: int, burn_in: int = 0,
                     params: dict = None) -> dict:
        """
        The exact probability of every sequence of patterns recorded over
        `ticks` ticks after `burn_in` ticks, by replaying every sequence of
        outcomes with :meth:`log_prob`: differentiable in `params`, for small
        networks of indistinguishable photons. Impossible sequences are kept,
        with probability zero.

        >>> delay = Interferometer([[0, 1], [1, 0]], loop=1, inputs=(1, ))
        >>> {history: round(float(probability), 6) for history, probability
        ...  in delay.kernel(cap=2).distribution(2).items() if probability}
        {((0,), (1,)): 1.0}
        """
        network = self.interferometer
        if network.indistinguishability < 1:
            raise NotImplementedError(
                "The exact distribution enumerates indistinguishable photons "
                "only; sample partially distinguishable ones.")
        if self.cap < (burn_in + ticks) * sum(network.inputs):
            raise ValueError(
                "The exact distribution needs a cap of every photon injected, "
                f"{(burn_in + ticks) * sum(network.inputs)}.")
        per_tick = [
            (shared, lost, recorded)
            for shared in product(range(self.photons + 1),
                                  repeat=len(network.inputs))
            for lost in product(range(self.photons + 1), repeat=network.loop
                                * (network.transmissivity < 1))
            if sum(shared) + sum(lost) <= self.photons
            for recorded in product(*(
                range(shared[i] + 1) if network.efficiency < 1
                else [shared[i]] for i in network.visible))]
        sequences = list(product(per_tick, repeat=burn_in + ticks))
        outcomes = {
            "private": jnp.zeros((len(sequences), burn_in + ticks,
                                  len(self.injected())), bool),
            "shared": jnp.array([[shared for shared, _, _ in sequence]
                                 for sequence in sequences], jnp.int32),
            "lost": jnp.array([[lost + (0, ) * (
                network.loop - len(lost)) for _, lost, _ in sequence]
                for sequence in sequences], jnp.int32),
            "walk": jnp.zeros((len(sequences), burn_in + ticks,
                               len(self.injected())), jnp.int32),
            "survive": jnp.zeros((len(sequences), burn_in + ticks,
                                  len(self.injected())), bool),
            "recorded": jnp.array([[recorded for _, _, recorded in sequence]
                                   for sequence in sequences], jnp.int32)}
        logp, possible = self.log_prob(outcomes, params)
        histories = [tuple(recorded for _, _, recorded in sequence[burn_in:])
                     for sequence in sequences]
        distinct = sorted(set(histories))
        totals = jax.ops.segment_sum(
            jnp.where(possible, jnp.exp(logp), 0),
            jnp.array([distinct.index(history) for history in histories]),
            len(distinct))
        return dict(zip(distinct, totals))


def choose(key, probabilities, forced=None):
    """
    An index sampled with `key` with the weights `probabilities` along the
    last axis, or the `forced` one.
    """
    if forced is not None:
        return jnp.asarray(forced, jnp.int32)
    return jax.random.categorical(
        key, jnp.log(jnp.where(probabilities > 0, probabilities, 0))
    ).astype(jnp.int32)


@lru_cache(maxsize=None)
def choices(photons: int) -> np.ndarray:
    """
    The binomial coefficients :math:`\\binom{n}{k}` for
    :math:`k, n \\leq` `photons`.

    >>> choices(2).tolist()
    [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 2.0, 1.0]]
    """
    return np.array([[comb(n, k) for k in range(photons + 1)]
                     for n in range(photons + 1)], dtype=float)
