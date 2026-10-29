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

Light
-----

A :class:`Source` feeds the inputs of a diagram at every tick. Light is
sorted by two independent properties: the state of a tick, Gaussian or
not and classical (a mixture of coherent states) or not, and whether a
phase reference is kept from tick to tick. Light with no phase reference
injects photon numbers; light with one injects superpositions of them,
whose fields from successive ticks interfere.

.. list-table::
    :header-rows: 1

    * - source
      - a tick
      - Gaussian
      - classical
      - phase
      - sampled by
    * - :class:`Fock`
      - number state
      - no
      - no
      - none
      - :class:`Interferometer`, :class:`Unravelling`, any :math:`p`
    * - :class:`Coherent`
      - Poisson
      - no
      - yes
      - random
      - :class:`Interferometer`, :class:`Unravelling`, :class:`Fields`
    * - :class:`Thermal`
      - geometric
      - yes
      - yes
      - random
      - :class:`Interferometer`, :class:`Unravelling`, :class:`Fields`,
        :class:`Gaussian`
    * - :class:`PhaseLocked`
      - coherent state
      - yes
      - yes
      - locked
      - :class:`Unravelling` (kets), :class:`Fields`, :class:`Gaussian`
    * - :class:`Squeezed`
      - squeezed vacuum
      - yes
      - no
      - locked
      - :class:`Unravelling` (kets), :class:`Gaussian`
    * - :class:`Displaced`
      - its source, displaced
      - as its source
      - as its source
      - locked
      - :class:`Fields`, :class:`Gaussian`

Every source has exact one-particle means through passive diagrams.

- The exact means are the one-particle means of a passive diagram,
  :meth:`Interferometer.advance`, :meth:`Interferometer.correlations` and
  :meth:`Interferometer.field`. They see only the means of the source and
  the field of a :class:`PhaseLocked` laser, not its photon statistics nor
  the indistinguishability :math:`p`.
- The :class:`Interferometer` samples photon numbers through passive
  diagrams, the cheapest in the loop's photons, with a certified burn-in.
- The :class:`Unravelling` samples any diagram, feed-forward included, at a
  cost exponential in its wires; superpositions enter as kets cut at its
  cap, which raises when the cut loses norm. Squeezed light with number
  resolving detectors is Gaussian boson sampling: exact but exponential.
- The :class:`Gaussian` samples Gaussian light through passive diagrams,
  its outputs read by homodyne detection: means and covariances of the
  quadratures, a cost polynomial in the modes.
- The :class:`Fields` sample classical light through linear optics,
  detectors and feed-forward, by drawing the amplitudes of coherent states
  from the P function of the source: no cap, a cost linear in the wires.

Classes
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Interferometer
    Source
    Fock
    Coherent
    Thermal
    PhaseLocked
    Squeezed
    Displaced
    Shifted
    FockState
    Sweep
    Unravelling
    Fields
    Gaussian
    CQState

Functions
---------

.. autosummary::
    :template: function.rst
    :nosignatures:
    :toctree:

    colouring
    ket
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
from math import comb, lgamma, prod

import numpy as np
from discopy import symmetric


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
def keys(modes: int, photons: int) -> tuple:
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
    values, order = keys(occupations.shape[1], photons)
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
    logs = np.array([lgamma(m + 1) + lgamma(photons - m + 1)
                     for m in range(photons + 1)])
    for k in range(photons + 1):
        first = [comb(k, a) * matrix[1, 0] ** (k - a) * matrix[0, 0] ** a
                 for a in range(k + 1)]
        second = [comb(photons - k, b) * matrix[1, 1] ** (photons - k - b)
                  * matrix[0, 1] ** b for b in range(photons - k + 1)]
        result[:, k] = np.convolve(first, second) * np.exp(
            (logs - logs[k]) / 2)
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


class Source:
    """
    The photons injected into the external inputs of an
    :class:`Interferometer` at a tick: a product state diagonal in the Fock
    basis, drawn independently for each input and tick, with mean `means`.
    Having no phase reference, it is sampled exactly by drawing the photon
    numbers of each tick.

    >>> Coherent((1., 0.)).draw(np.random.default_rng(0))
    (1, 0)
    """
    def __init__(self, means):
        self.means = tuple(float(mean) for mean in means)

    def __len__(self):
        return len(self.means)

    def __repr__(self):
        return f"{type(self).__name__}({self.means})"

    def __eq__(self, other):
        return type(self) is type(other) and self.means == other.means

    def draw(self, rng: np.random.Generator) -> tuple:
        """ The photon numbers injected at one tick. """
        raise NotImplementedError

    def inject(self, rng: np.random.Generator, levels: int) -> tuple:
        """
        The state injected into each input at one tick: a photon number,
        or the ket of a superposition of photon numbers, cut to `levels`,
        for a source with a phase reference.
        """
        return self.draw(rng)

    @property
    def superposed(self) -> bool:
        """
        Whether a tick injects superpositions of photon numbers, whose
        phase is kept from tick to tick, rather than photon numbers.
        """
        return False

    def amplitudes(self, rng: np.random.Generator) -> np.ndarray:
        """
        The amplitudes of a coherent state drawn from the Glauber P
        function of the source, for classical light: a mixture of coherent
        states. Light that is not raises.
        """
        raise NotImplementedError(
            f"{self} is not a mixture of coherent states.")

    @property
    def field(self) -> np.ndarray:
        """
        The mean field :math:`\\langle a \\rangle` of each input, kept
        from tick to tick: zero for light with no phase reference.
        """
        return np.zeros(len(self), dtype=complex)

    def quadratures(self) -> tuple:
        """
        The means and the covariance of the quadratures
        :math:`x = (a + a^\\dagger) / 2` and
        :math:`p = (a - a^\\dagger) / 2i` of the inputs, all the :math:`x`
        then all the :math:`p`, for Gaussian light: the vacuum has
        covariance :math:`1 / 4`. Light that is not Gaussian raises.

        >>> [x.tolist() for x in Fock((0, )).quadratures()]
        [[0.0, 0.0], [[0.25, 0.0], [0.0, 0.25]]]
        """
        raise NotImplementedError(f"{self} is not Gaussian light.")

    def pad(self, modes: int) -> Source:
        """ The same source with `modes` more inputs in the vacuum. """
        return type(self)(self.means + (0, ) * modes)

    def shift(self, delta) -> Source:
        """
        A source of means `means + delta`: the same kind of light for
        :class:`Coherent` and :class:`Thermal`, a :class:`Shifted` one
        otherwise.

        >>> Coherent((1., 2.)).shift((.5, -.5))
        Coherent((1.5, 1.5))
        >>> Fock((1, )).shift((.25, )).means
        (1.25,)
        """
        return Shifted(self, delta)

    def tensor(self, other: Source) -> Source:
        """
        The source of both, `self` on the first inputs: a :class:`Product`
        of sources of different kinds, adjacent ones of the same kind
        merged.

        >>> Fock((1, )) @ Fock((2, ))
        Fock((1, 2))
        >>> Coherent((1., )) @ Fock((1, )) @ Fock((0, ))
        Coherent((1.0,)) @ Fock((1, 0))
        """
        factors = []
        for factor in self.factors + other.factors:
            if factors and type(factors[-1]) is type(factor):
                factors[-1] = factors[-1].join(factor)
            elif len(factor):
                factors.append(factor)
        return Product(*factors) if len(factors) > 1 \
            else factors[0] if factors else Fock(())

    def join(self, other: Source) -> Source:
        """ The source of both, of the same kind, `self` first. """
        return type(self)(self.means + other.means)

    @property
    def factors(self) -> tuple:
        """ The sources side by side, of different kinds. """
        return (self, )

    __matmul__ = tensor


class Fock(Source):
    """
    Exactly `photons[a]` photons into input :math:`a` at every tick.

    >>> Fock((1, 0)).draw(None), Fock((1, 0)).means
    ((1, 0), (1.0, 0.0))
    """
    def __init__(self, photons):
        self.photons = tuple(int(number) for number in photons)
        super().__init__(self.photons)

    def __repr__(self):
        return f"Fock({self.photons})"

    def draw(self, rng: np.random.Generator) -> tuple:
        return self.photons

    def amplitudes(self, rng: np.random.Generator) -> np.ndarray:
        if any(self.photons):
            return super().amplitudes(rng)
        return np.zeros(len(self), dtype=complex)

    def quadratures(self) -> tuple:
        if any(self.photons):
            return super().quadratures()
        return np.zeros(2 * len(self)), np.eye(2 * len(self)) / 4


class Coherent(Source):
    """
    A coherent state of mean photon number `means[a]` into input :math:`a`,
    with a uniformly random phase at every tick: a Poisson number of
    photons.
    """
    def draw(self, rng: np.random.Generator) -> tuple:
        return tuple(int(number) for number in rng.poisson(self.means))

    def amplitudes(self, rng: np.random.Generator) -> np.ndarray:
        return np.sqrt(self.means) * np.exp(
            2j * np.pi * rng.random(len(self)))

    def shift(self, delta) -> Source:
        means = np.add(self.means, delta)
        if np.any(means < 0):
            raise ValueError(f"A negative mean photon number: {means}.")
        return type(self)(means)


class Thermal(Source):
    """
    A thermal state of mean photon number `means[a]` into input :math:`a`
    at every tick: :math:`P(n) = \\bar n^n / (1 + \\bar n)^{n + 1}`.
    """
    def draw(self, rng: np.random.Generator) -> tuple:
        return tuple(int(number) - 1 for number in rng.geometric(
            1 / (1 + np.array(self.means))))

    def amplitudes(self, rng: np.random.Generator) -> np.ndarray:
        return np.sqrt(np.array(self.means) / 2) * (
            rng.normal(size=len(self)) + 1j * rng.normal(size=len(self)))

    def quadratures(self) -> tuple:
        return np.zeros(2 * len(self)), np.diag(
            np.tile(2 * np.array(self.means) + 1, 2) / 4)

    shift = Coherent.shift


class PhaseLocked(Source):
    """
    A coherent state of amplitude `amplitudes[a]` into input :math:`a`,
    the same at every tick: a laser with a phase reference, whose fields
    from successive ticks interfere. Its mean is
    :math:`|\\alpha|^2`.

    >>> laser = PhaseLocked((1j, 0))
    >>> laser.means, laser.field.tolist()
    ((1.0, 0.0), [1j, 0j])
    >>> np.round(laser.inject(None, 16)[0][:3], 3).tolist()
    [(0.607+0j), 0.607j, (-0.429+0j)]

    Cutting the ket to fewer levels than it needs raises:

    >>> laser.inject(None, 3)
    Traceback (most recent call last):
    ...
    ValueError: The source loses 0.08 of its norm above 2 photons: \
sample with a larger cap.
    """
    def __init__(self, amplitudes):
        self.coherent = tuple(complex(alpha) for alpha in amplitudes)
        super().__init__(np.abs(self.coherent) ** 2)

    def __repr__(self):
        return f"PhaseLocked({self.coherent})"

    def __eq__(self, other):
        return isinstance(other, PhaseLocked) \
            and self.coherent == other.coherent

    def draw(self, rng: np.random.Generator) -> tuple:
        raise NotImplementedError(
            f"{self} injects superpositions of photon numbers with a phase "
            "kept from tick to tick: sample it with the Unravelling or the "
            "Fields.")

    def inject(self, rng: np.random.Generator, levels: int) -> tuple:
        numbers = np.arange(levels)
        norms = np.exp(-np.array([lgamma(n + 1) for n in numbers]) / 2)
        return tuple(ket(np.exp(-abs(alpha) ** 2 / 2) * alpha ** numbers
                         * norms) for alpha in self.coherent)

    @property
    def superposed(self) -> bool:
        return True

    def amplitudes(self, rng: np.random.Generator) -> np.ndarray:
        return np.array(self.coherent)

    @property
    def field(self) -> np.ndarray:
        return np.array(self.coherent)

    def quadratures(self) -> tuple:
        return np.r_[np.real(self.coherent), np.imag(self.coherent)], \
            np.eye(2 * len(self)) / 4

    def pad(self, modes: int) -> Source:
        return PhaseLocked(self.coherent + (0, ) * modes)

    def join(self, other: Source) -> Source:
        return PhaseLocked(self.coherent + other.coherent)

    def shift(self, delta) -> Source:
        means = np.add(self.means, delta)
        if np.any(means < 0):
            raise ValueError(f"A negative mean photon number: {means}.")
        return PhaseLocked([
            alpha * np.sqrt(mean / abs(alpha) ** 2) if alpha else np.sqrt(mean)
            for alpha, mean in zip(self.coherent, means)])


class Squeezed(Source):
    """
    A squeezed vacuum of squeezing `squeezing[a]` into input :math:`a`,
    the same at every tick: Gaussian light that is not classical, of mean
    :math:`\\sinh^2 r`, its photons in pairs.

    >>> round(Squeezed((np.arcsinh(1.), )).means[0], 6)
    1.0
    >>> np.round(Squeezed((.3, )).inject(None, 24)[0][:5].real, 3).tolist()
    [0.978, 0.0, -0.201, 0.0, 0.051]
    """
    def __init__(self, squeezing):
        self.squeezing = tuple(float(r) for r in squeezing)
        super().__init__(np.sinh(self.squeezing) ** 2)

    def __repr__(self):
        return f"Squeezed({self.squeezing})"

    def __eq__(self, other):
        return isinstance(other, Squeezed) \
            and self.squeezing == other.squeezing

    draw = PhaseLocked.draw
    superposed = PhaseLocked.superposed

    def inject(self, rng: np.random.Generator, levels: int) -> tuple:
        pairs = np.arange(levels) // 2
        weights = np.exp(np.array([
            lgamma(2 * k + 1) / 2 - lgamma(k + 1) for k in pairs])) \
            / 2 ** pairs
        return tuple(ket(np.where(
            np.arange(levels) % 2, 0, (-np.tanh(r)) ** pairs * weights
            / np.sqrt(np.cosh(r)))) for r in self.squeezing)

    def quadratures(self) -> tuple:
        squeezing = np.array(self.squeezing)
        return np.zeros(2 * len(self)), np.diag(
            np.r_[np.exp(-2 * squeezing), np.exp(2 * squeezing)] / 4)

    def pad(self, modes: int) -> Source:
        return Squeezed(self.squeezing + (0, ) * modes)

    def join(self, other: Source) -> Source:
        return Squeezed(self.squeezing + other.squeezing)


def ket(amplitudes: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """
    The ket of a superposition of photon numbers, raising when cutting it
    to its levels loses more than `tol` of its norm.

    >>> ket(np.array([.6, .8])).real.tolist()
    [0.6, 0.8]
    """
    weight = float(np.sum(np.abs(amplitudes) ** 2))
    if weight < 1 - tol:
        raise ValueError(
            f"The source loses {1 - weight:.2g} of its norm above "
            f"{len(amplitudes) - 1} photons: sample with a larger cap.")
    return np.asarray(amplitudes, dtype=complex)


class Shifted(Source):
    """
    The photons of a `source` with the means moved by `delta`: a Poisson
    number of photons more on an input whose mean goes up, each photon
    kept with probability :math:`1 + \\delta / \\bar n` on one whose mean
    goes down.

    >>> shifted = Shifted(Fock((2, 1)), (.5, -.5))
    >>> shifted.means, shifted.draw(np.random.default_rng(0))
    ((2.5, 0.5), (2, 0))
    >>> shifted.pad(1) @ shifted
    Shifted(Fock((2, 1, 0, 2, 1)), (0.5, -0.5, 0.0, 0.5, -0.5))
    """
    def __init__(self, source: Source, delta):
        self.source, self.delta = source, tuple(float(d) for d in delta)
        means = np.add(source.means, self.delta)
        if len(self.delta) != len(source) or np.any(means < 0):
            raise ValueError(
                f"Cannot shift the means {source.means} by {self.delta}.")
        super().__init__(means)

    def __repr__(self):
        return f"Shifted({self.source!r}, {self.delta})"

    def pad(self, modes: int) -> Source:
        return Shifted(self.source.pad(modes), self.delta + (0., ) * modes)

    def join(self, other: Source) -> Source:
        return Shifted(self.source @ other.source, self.delta + other.delta)

    def __eq__(self, other):
        return isinstance(other, Shifted) \
            and (self.source, self.delta) == (other.source, other.delta)

    def draw(self, rng: np.random.Generator) -> tuple:
        drawn = np.array(self.source.draw(rng))
        up, down = np.maximum(self.delta, 0), np.minimum(self.delta, 0)
        kept = 1 + np.divide(down, self.source.means, out=np.zeros(len(
            self)), where=np.array(self.source.means) > 0)
        return tuple(int(n) for n in rng.binomial(drawn, kept)
                     + rng.poisson(up))


class Product(Source):
    """
    Independent sources side by side, the first on the first inputs.

    >>> source = Coherent((1., )) @ Fock((1, ))
    >>> source.draw(np.random.default_rng(0)), source.means
    ((1, 1), (1.0, 1.0))
    """
    def __init__(self, *factors: Source):
        self.parts = factors
        super().__init__(sum((factor.means for factor in factors), ()))

    @property
    def factors(self) -> tuple:
        return self.parts

    def __repr__(self):
        return " @ ".join(map(repr, self.factors))

    def __eq__(self, other):
        return isinstance(other, Product) and self.factors == other.factors

    def draw(self, rng: np.random.Generator) -> tuple:
        return sum((factor.draw(rng) for factor in self.factors), ())

    def inject(self, rng: np.random.Generator, levels: int) -> tuple:
        return sum((factor.inject(rng, levels) for factor in self.factors),
                   ())

    @property
    def superposed(self) -> bool:
        return any(factor.superposed for factor in self.factors)

    def amplitudes(self, rng: np.random.Generator) -> np.ndarray:
        return np.concatenate([
            factor.amplitudes(rng) for factor in self.factors])

    @property
    def field(self) -> np.ndarray:
        return np.concatenate([factor.field for factor in self.factors])

    def quadratures(self) -> tuple:
        offsets = np.cumsum([0] + [len(factor) for factor in self.factors])
        order = np.concatenate([
            np.r_[start:stop, len(self) + start:len(self) + stop]
            for start, stop in zip(offsets, offsets[1:])])
        means, covariance = np.zeros(2 * len(self)), np.zeros(
            (2 * len(self), 2 * len(self)))
        for start, stop, factor in zip(offsets, offsets[1:], self.factors):
            block = order[2 * start:2 * stop]
            means[block], covariance[np.ix_(block, block)] = \
                factor.quadratures()
        return means, covariance

    def pad(self, modes: int) -> Source:
        return self @ Fock((0, ) * modes)


class Displaced(Source):
    """
    The light of `source` displaced by `displacement`, as a strong laser
    mixed in on a beam splitter of transmissivity close to one does: its
    field moves by the displacement and its fluctuations stay. This is the
    modulator of a phase-locked electronic loop, which writes its signal on
    light whose noise the source sets: the vacuum for a laser, thermal or
    squeezed noise otherwise. It is sampled by the :class:`Fields` when the
    source is classical, and by the :class:`Gaussian` when it is Gaussian.

    >>> light = Displaced(Thermal((1., )), (2j, ))
    >>> light.means, light.field.tolist()
    ((5.0,), [2j])
    >>> [x.tolist() for x in light.quadratures()]
    [[0.0, 2.0], [[0.75, 0.0], [0.0, 0.75]]]
    """
    def __init__(self, source: Source, displacement):
        self.source = source
        self.displacement = tuple(complex(d) for d in displacement)
        if len(self.displacement) != len(source):
            raise ValueError(
                f"Expected a displacement of {len(source)} modes, got "
                f"{self.displacement}.")
        super().__init__(np.array(source.means) + np.abs(self.field) ** 2
                         - np.abs(source.field) ** 2)

    def __repr__(self):
        return f"Displaced({self.source!r}, {self.displacement})"

    def __eq__(self, other):
        return isinstance(other, Displaced) and (
            self.source, self.displacement) == (
                other.source, other.displacement)

    def draw(self, rng: np.random.Generator) -> tuple:
        raise NotImplementedError(
            f"{self} has a phase reference: sample it with the Fields or "
            "the Gaussian.")

    superposed = PhaseLocked.superposed

    def amplitudes(self, rng: np.random.Generator) -> np.ndarray:
        return self.source.amplitudes(rng) + np.array(self.displacement)

    @property
    def field(self) -> np.ndarray:
        return self.source.field + np.array(self.displacement)

    def quadratures(self) -> tuple:
        means, covariance = self.source.quadratures()
        return means + np.r_[np.real(self.displacement),
                             np.imag(self.displacement)], covariance

    def pad(self, modes: int) -> Source:
        return Displaced(self.source.pad(modes),
                         self.displacement + (0, ) * modes)

    def join(self, other: Source) -> Source:
        return Displaced(self.source @ other.source,
                         self.displacement + other.displacement)


def feeding(diagram, source: Source = None) -> Source:
    """
    The `source` feeding the inputs of `diagram` at every tick, checked
    to have one mode per input, none for a closed diagram.
    """
    if source is None and not diagram.dom:
        return Fock(())
    if source is None or len(source) != len(diagram.dom):
        raise ValueError(
            f"Expected a source of {len(diagram.dom)} modes to feed the "
            f"inputs {diagram.dom}, got {source}.")
    return source


def colouring(injection: tuple, indistinguishability: float,
              rng: np.random.Generator, optical: tuple = None) -> tuple:
    """
    Split the photons of an `injection` into colours: each photon is in
    the internal state shared by all photons with probability
    `indistinguishability`, and in one of its own otherwise [RMC+18]_.
    Returns the injection of the shared colour, then one injection per
    photon of its own, a single photon. Only the inputs flagged `optical`
    are split, all of them by default; the others, classical values, go
    unchanged to every colour.

    >>> colouring((2, 1), 1., None)
    ((2, 1), [])
    >>> colouring((1, 3), 0., np.random.default_rng(0), (True, False))
    ((0, 3), [(1, 3)])
    """
    optical = (True, ) * len(injection) if optical is None else optical
    if indistinguishability == 1:
        return tuple(injection), []
    drawn = rng.binomial(injection, indistinguishability)
    shared = [int(kept) if light else n
              for n, kept, light in zip(injection, drawn, optical)]
    empty = [0 if light else n for n, light in zip(injection, optical)]
    return tuple(shared), [
        tuple(empty[:mode] + [1] + empty[mode + 1:])
        for mode, (total, kept) in enumerate(zip(injection, shared))
        if optical[mode] for _ in range(total - kept)]


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
            inputs at every tick, a :class:`Fock` source.
        transmissivity : The intensity transmissivity :math:`\\gamma` of each
            loop mode per round trip.
        efficiency : The efficiency :math:`\\eta` of the detectors.
        visible : The external outputs that are recorded, all of them by
            default; the others are measured and forgotten, as the
            environment of a loss or a discard.
        indistinguishability : The probability :math:`p` that an injected
            photon is in the internal state shared by all photons rather
            than in one of its own, the model of Renema et al. [RMC+18]_,
            whatever the source.
        source : The :class:`Source` drawing the photons injected at each
            tick, in place of `inputs`: :class:`Fock`, :class:`Coherent` or
            :class:`Thermal`.

    >>> network = Interferometer([[0, 1], [1, 0]], loop=1, inputs=(1, ))
    >>> network
    Interferometer([[0, 1], [1, 0]], loop=1, inputs=(1,), transmissivity=1.0, \
efficiency=1.0, visible=(0,), indistinguishability=1.0)
    >>> eval(repr(network)) == network
    True

    A delay line fed with thermal light detects thermal light:

    >>> thermal = Interferometer(
    ...     [[0, 1], [1, 0]], loop=1, source=Thermal((2., )))
    >>> thermal
    Interferometer([[0, 1], [1, 0]], loop=1, source=Thermal((2.0,)), \
transmissivity=1.0, efficiency=1.0, visible=(0,), indistinguishability=1.0)
    >>> thermal.occupation()
    2.0
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(self, unitary, loop: int, inputs: tuple = None,
                 transmissivity: float = 1., efficiency: float = 1.,
                 visible: tuple = None, indistinguishability: float = 1.,
                 source: Source = None):
        if (inputs is None) == (source is None):
            raise ValueError("Give the inputs or a source.")
        self.unitary = np.asarray(unitary)
        self.loop = loop
        self.source = Fock(inputs) if source is None else source
        self.transmissivity = float(transmissivity)
        self.efficiency = float(efficiency)
        self.visible = tuple(range(len(self.source))) if visible is None \
            else tuple(int(i) for i in visible)
        self.indistinguishability = float(indistinguishability)
        if self.unitary.shape != (loop + len(self.source), ) * 2:
            raise ValueError(
                f"Expected a {loop + len(self.source)}-mode matrix, "
                f"got shape {self.unitary.shape}.")
        if not np.allclose(
                self.unitary.conj().T @ self.unitary, np.eye(len(unitary))):
            raise ValueError("The mode matrix must be unitary.")
        self.sweeps = {}

    @property
    def inputs(self) -> tuple:
        """ The photons injected at every tick by a :class:`Fock` source. """
        if not isinstance(self.source, Fock):
            raise NotImplementedError(
                f"{self.source} injects a random number of photons.")
        return self.source.photons

    @classmethod
    def from_diagram(cls, diagram, indistinguishability: float = 1.,
                     source: Source = None):
        """
        The recurrent network of a :class:`optyx.channel.Diagram` with
        feedback loops: its :meth:`one_step` is dilated to a passive matrix
        from the memory, the inputs and the created photons to the outputs,
        the memory and the environment of its discards and losses, then
        completed to a unitary with vacuum inputs. The inputs of the diagram
        are fed by `source` at every tick, its outputs are recorded and its
        environment is not.

        >>> from optyx.channel import Diagram, qmode
        >>> from optyx.photonic import Create
        >>> delay = (Create(1) @ qmode >> Diagram.swap(qmode, qmode)
        ...     ).feedback(state=Create(0))
        >>> Interferometer.from_diagram(delay).distribution(2)
        {((0,), (1,)): 1.0}
        """
        initial, _ = diagram.boundary()
        return cls.from_step(initial, diagram.one_step(), len(diagram.cod),
                             feeding(diagram, source), indistinguishability)

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    @classmethod
    def from_step(cls, initial, step, outputs: int, source: Source,
                  indistinguishability: float = 1.):
        """
        The recurrent network of the `step` of a diagram with feedback,
        from its inputs and memory to its first `outputs` outputs and its
        memory, the memory starting in the state `initial` and the inputs
        fed by `source`: :meth:`from_diagram` without the diagram.
        """
        inputs = len(source)
        loop = len(step.dom) - inputs
        if levels(initial) or levels(step):
            raise NotImplementedError(
                "Photons with internal states are not passive.")
        if any(not hasattr(box, "kraus")
               and not isinstance(box, symmetric.Swap)
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
        rows = np.asarray(matrix.array).T
        rows = np.vstack([rows[outputs:outputs + loop], rows[:outputs],
                          rows[outputs + loop:]])
        rows = np.hstack([rows[:, inputs:inputs + loop], rows[:, :inputs],
                          rows[:, inputs + loop:]])
        unitary = complete(rows)
        source = source @ Fock(matrix.creations)
        return cls(unitary, loop, visible=tuple(range(outputs)),
                   indistinguishability=indistinguishability,
                   source=source.pad(len(unitary) - loop - len(source)))

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
        injection = f"inputs={self.inputs}" \
            if isinstance(self.source, Fock) else f"source={self.source}"
        return (f"Interferometer({self.unitary.tolist()}, loop={self.loop}, "
                f"{injection}, "
                f"transmissivity={self.transmissivity}, "
                f"efficiency={self.efficiency}, visible={self.visible}, "
                f"indistinguishability={self.indistinguishability})")

    def __eq__(self, other):
        return isinstance(other, Interferometer) \
            and np.array_equal(self.unitary, other.unitary) \
            and (self.loop, self.source, self.transmissivity,
                 self.efficiency, self.visible, self.indistinguishability) \
            == (other.loop, other.source, other.transmissivity,
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
        The mean photon number of the loop in the stationary state, the
        trace of its :meth:`correlations`.

        Without loss every loop mode holds :math:`\\bar q` photons when
        :math:`\\bar q` is injected into every external mode, whatever the
        interferometer:

        >>> from scipy.stats import unitary_group
        >>> unitary = unitary_group.rvs(5, random_state=1)
        >>> network = Interferometer(unitary, loop=3, inputs=(2, 2))
        >>> bool(np.isclose(network.occupation(), 6))
        True
        """
        return float(np.real(np.trace(self.correlations())))

    def advance(self, correlations: np.ndarray, dephased: tuple = (),
                field: np.ndarray = None):
        """
        One round trip of the one-particle correlation matrix
        :math:`C_{ij} = \\langle a_i^\\dagger a_j \\rangle` of the loop
        at the start of a round trip, whose mean field
        :math:`x = \\langle a \\rangle` is `field`, zero by default. The
        loop modes in `dephased` lose their phase: their field and their
        coherence with the other modes. The fluctuations
        :math:`F = C - x x^\\dagger` and the means of the source with no
        phase reference add up as intensities, the fields of the loop and
        of a :class:`PhaseLocked` source as amplitudes. Returns the next
        :math:`C` and the mean counts of the visible outputs. These are
        exact whatever the photon statistics and the indistinguishability:
        one-particle observables do not see them.

        >>> delay = Interferometer([[0, 1], [1, 0]], loop=1, inputs=(1, ))
        >>> loop, counts = delay.advance(np.zeros((1, 1)))
        >>> loop.real.tolist(), counts.tolist()
        ([[1.0]], [0.0])
        """
        field = np.zeros(self.loop, dtype=complex) if field is None \
            else np.array(field, dtype=complex)
        fluctuations = correlations - np.outer(field.conj(), field)
        mask = np.ones((self.loop, self.loop), dtype=bool)
        mask[list(dephased), :] = mask[:, list(dephased)] = False
        diagonal = np.eye(self.loop, dtype=bool)
        fluctuations = np.where(mask | diagonal, fluctuations, 0) + np.diag(
            np.where(~mask.all(1), np.abs(field) ** 2, 0))
        field = np.where(mask.all(1), field, 0)
        source = self.source.field
        inputs = np.zeros((self.modes, self.modes), dtype=complex)
        inputs[:self.loop, :self.loop] = fluctuations
        inputs[self.loop:, self.loop:] = np.diag(
            np.array(self.source.means) - np.abs(source) ** 2)
        inputs += np.outer(np.r_[field, source].conj(), np.r_[field, source])
        outputs = self.unitary.conj() @ inputs @ self.unitary.T
        counts = self.efficiency * np.real(np.diag(outputs)[self.loop:])
        return self.transmissivity * outputs[:self.loop, :self.loop], \
            counts[list(self.visible)]

    def carry(self, field: np.ndarray, dephased: tuple = ()) -> tuple:
        """
        One round trip of the mean field of the loop: the field of the
        loop after it, and the field reaching the visible detectors, from
        the loop's `field`, the modes in `dephased` losing theirs, and the
        field of the source.

        >>> laser = Interferometer(
        ...     [[0, 1], [1, 0]], 1, source=PhaseLocked((1j, )))
        >>> [x.tolist() for x in laser.carry(np.array([2]))]
        [[1j], [(2+0j)]]
        """
        kept = np.ones(self.loop)
        kept[list(dephased)] = 0
        outputs = self.unitary @ np.r_[np.asarray(field) * kept,
                                       self.source.field]
        return np.sqrt(self.transmissivity) * outputs[:self.loop], \
            np.sqrt(self.efficiency) * outputs[self.loop:][list(self.visible)]

    def transfer(self) -> np.ndarray:
        """
        The stationary mode matrix from the external inputs to the visible
        outputs: a field :math:`\\alpha` injected at every tick reaches
        the detectors as :math:`T \\alpha` once the loop has settled,
        summing over every number of round trips,
        :math:`T = \\sqrt\\eta (U_{x x} + U_{x \\ell} (1 - \\sqrt\\gamma
        U_{\\ell \\ell})^{-1} \\sqrt\\gamma U_{\\ell x})`.

        >>> loop = Interferometer([[.6, .8], [.8, -.6]], 1, inputs=(0, ))
        >>> loop.transfer().real.round(6).tolist()
        [[1.0]]
        """
        loop, rows = self.loop, self.loop + np.array(self.visible, dtype=int)
        settled = np.linalg.solve(
            np.eye(loop) - self.block, np.sqrt(self.transmissivity)
            * self.unitary[:loop, loop:])
        return np.sqrt(self.efficiency) * (
            self.unitary[rows, loop:] + self.unitary[rows, :loop] @ settled)

    def field(self, dephased: tuple = ()) -> np.ndarray:
        """
        The mean field of the loop in the stationary state, the fixed point
        of :math:`x = \\sqrt\\gamma (U_{\\ell\\ell} x + U_{\\ell x} \\alpha)`
        for the field :math:`\\alpha` of the source, the dephased modes
        losing theirs: zero without a :class:`PhaseLocked` source.

        >>> lossy = Interferometer([[0.6, 0.8], [0.8, -0.6]], 1,
        ...     source=PhaseLocked((1, )), transmissivity=.5)
        >>> lossy.field().round(4).tolist()
        [(0.9825+0j)]
        """
        kept = np.ones(self.loop)
        kept[list(dephased)] = 0
        return np.linalg.solve(
            np.eye(self.loop) - self.block * kept,
            np.sqrt(self.transmissivity) * self.unitary[
                :self.loop, self.loop:] @ self.source.field)

    def correlations(self, dephased: tuple = ()) -> np.ndarray:
        """
        The one-particle correlation matrix of the loop in the stationary
        state, the fixed point of :meth:`advance` with the stationary
        :meth:`field`, by solving its linear equation: the Stein equation
        :math:`C = \\gamma (U_{\\ell\\ell} C U_{\\ell\\ell}^\\dagger
        + U_{\\ell x} Q U_{\\ell x}^\\dagger)` without dephasing or field.

        >>> lossy = Interferometer(
        ...     [[0.6, 0.8], [0.8, -0.6]], 1, (1, ), transmissivity=.5)
        >>> lossy.correlations().real.round(4).tolist()
        [[0.3902]]

        A laser with a phase reference fills the loop with its field rather
        than with its intensity, here more than twice as much:

        >>> locked = Interferometer([[0.6, 0.8], [0.8, -0.6]], 1,
        ...     source=PhaseLocked((1, )), transmissivity=.5)
        >>> locked.correlations().real.round(4).tolist()
        [[0.9654]]
        """
        size, field = self.loop ** 2, self.field(dephased)
        drive, _ = self.advance(
            np.zeros((self.loop, self.loop)), dephased, field)
        columns = [self.advance(basis.reshape(self.loop, self.loop),
                                dephased, field)[0].reshape(-1)
                   - drive.reshape(-1) for basis in np.eye(size)]
        return np.linalg.solve(np.eye(size) - np.array(columns).T,
                               drive.reshape(-1)).reshape(self.loop, self.loop)

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
        qbar = max(self.source.means, default=0)
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

    def padded(self, source: Source = None) -> Source:
        """
        The light injected at a tick: `source` on the first external
        inputs and the vacuum on the others, or the source of the network.
        """
        if source is None:
            return self.source
        if len(source) > len(self.source):
            raise ValueError(
                f"{source} has more modes than the {len(self.source)} "
                "external inputs.")
        if any(self.source.means[len(source):]):
            raise NotImplementedError(
                f"{self.source} feeds the inputs past those of {source}: "
                "they are not the vacuum.")
        return source.pad(len(self.source) - len(source))

    def fed(self, source: Source) -> Interferometer:
        """
        The same network fed by `source` on its first external inputs,
        :meth:`padded` with the vacuum.

        >>> delay = Interferometer([[0, 1], [1, 0]], loop=1, inputs=(1, ))
        >>> delay.fed(Coherent((2., ))).occupation()
        2.0
        """
        return Interferometer(
            self.unitary, self.loop, source=self.padded(source),
            transmissivity=self.transmissivity, efficiency=self.efficiency,
            visible=self.visible,
            indistinguishability=self.indistinguishability)

    def colour(self, rng: np.random.Generator, source: Source = None):
        """
        The injection of one tick, drawn from the source, or from `source`
        when given, split into colours: the photons in the shared internal
        state, then one single photon per photon in an internal state of
        its own.
        """
        return colouring(self.padded(source).draw(rng),
                         self.indistinguishability, rng)

    def tick(self, colours: list, rng: np.random.Generator,
             source: Source = None):
        """
        One tick of a trajectory with one loop state per colour, the shared
        colour first: the recorded pattern and the next colours. A colour
        of its own is forgotten once its photon has left the loop. The
        photons are drawn from `source` when given, as a modulator would
        set it for this tick.
        """
        shared, alone = self.colour(rng, source)
        empty = (0, ) * len(self.source)
        work = [(colours[0], shared)] \
            + [(state, empty) for state in colours[1:]] \
            + [(FockState.vacuum(self.loop), inputs) for inputs in alone]
        total, after = np.zeros(len(self.source), dtype=int), []
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
        if self.indistinguishability != 1 \
                or not isinstance(self.source, Fock):
            raise NotImplementedError(
                "The exact distribution enumerates a deterministic injection "
                "of indistinguishable photons only; sample the others.")
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

    def means(self) -> list:
        """
        The mean value of each wire: its value for a classical wire, its
        mean photon number, or its probability of one for a qubit, for a
        quantum one.

        >>> CQState(np.array([[0, 1], [0, 0]]), [None, 2, None]).means()
        [0.0, 2, 1.0]
        """
        axes = iter(range(self.vector.ndim))
        probabilities = np.abs(self.vector) ** 2
        return [value if value is not None else float(np.sum(
            probabilities, axis=tuple(
                other for other in range(self.vector.ndim) if other != axis))
            @ np.arange(self.vector.shape[axis]))
            for value in self.values
            for axis in [next(axes) if value is None else None]]

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

    Partially distinguishable photons are sampled by colouring the photons
    of the source, as the :class:`Interferometer` does, with probability
    `indistinguishability` of being in the shared internal state: the
    colours run in lockstep box by box, each on its own pure state, and
    share the values of the classical wires. A detector records the
    outcomes of all colours pooled by the `pool` of its box, the sum of
    the counts of a number resolving measurement, so that feed-forward
    acts on what the detector sees. This is exact because a detected
    photon is absorbed with its internal state, which the environment
    keeps. Boxes of the diagram creating photons, and qubits, are refused:
    the coloured photons come from the source.

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
    def __init__(self, diagram, cap: int = 4, source: Source = None,
                 indistinguishability: float = 1.):
        self.source = feeding(diagram, source)
        self.indistinguishability = float(indistinguishability)
        if any(ob.name not in ("qmode", "mode") for ob in diagram.dom.inside):
            raise NotImplementedError(
                "A source feeds optical or classical modes only.")
        if cap < 1:
            raise ValueError("The cap must be at least one photon.")
        self.diagram, self.cap = diagram, int(cap)
        self.initial, _ = diagram.boundary()
        self.step = diagram.one_step()
        self.levels = max(levels(self.initial), levels(self.step))
        if self.levels and self.diagram.dom:
            raise NotImplementedError(
                "A source feeds photons without internal states only.")
        if self.levels:
            self.initial = self.initial.inflate(self.levels)
            self.step = self.step.inflate(self.levels)
        if self.indistinguishability != 1 and (self.levels or any(
                ob.name == "qubit" for box in self.step.boxes
                for ob in (box.dom @ box.cod).inside)):
            raise NotImplementedError(
                "Colouring partially distinguishable photons needs optical "
                "modes without internal states and no qubits.")
        self.tensors = {}

    def __repr__(self):
        source = f", source={self.source}" if self.diagram.dom else ""
        colours = f", indistinguishability={self.indistinguishability}" \
            if self.indistinguishability != 1 else ""
        return f"Unravelling({self.diagram!r}, cap={self.cap}{source}" \
            f"{colours})"

    def feed(self, state: CQState, drawn: tuple) -> CQState:
        """
        The `state` of the memory with the inputs of one tick in front, as
        `drawn` from the source: a value on each classical mode, and on
        each optical mode a photon number or the ket of a superposition of
        photon numbers.
        """
        vector, values = state.vector, []
        for ob, number in reversed(list(zip(self.diagram.dom.inside, drawn))):
            if ob.name == "mode":
                values = [number] + values
                continue
            if np.ndim(number) == 0 and number > self.cap:
                raise ValueError(
                    f"The source injected {number} photons into a mode of "
                    f"at most cap={self.cap}.")
            vector = np.multiply.outer(np.eye(self.cap + 1)[number] if np.ndim(
                number) == 0 else number, vector)
            values = [None] + values
        return CQState(vector, values + state.values)

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
        width = len(box.dom)
        if isinstance(box, symmetric.Swap):
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
        stop = offset + len(box.dom)
        values = state.values[offset:stop]
        if isinstance(box, symmetric.Swap):
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

    def run(self, diagram, colours: list,
            rng: np.random.Generator) -> list:
        """
        The colours after the layers of `diagram`, one box at a time,
        each checked to be trace preserving as a whole: a box may scale a
        state, as a spider does until the scalar that normalises it, but a
        run that does not end with weight one is a postselection.
        """
        for layer in diagram:
            left, box, _ = layer.inside[0]
            colours = self.layer(colours, len(left), box, rng)
        for state in colours:
            if not np.isclose(state.weight, 1, atol=1e-6):
                raise NotImplementedError(
                    f"The diagram is not trace preserving (weight "
                    f"{state.weight:.3g}): postselection cannot be sampled.")
        return [CQState(state.vector, state.values) for state in colours]

    def layer(self, colours: list, offset: int, box,
              rng: np.random.Generator) -> list:
        """
        The colours after `box`, on the wires from `offset`: a classical
        box samples once and its values go to every colour, a box with no
        quantum input creates its photons in the shared colour only, and
        any other box acts on each colour, the classical outputs of a
        detector pooled across the colours by the `pool` of the box.
        """
        shared, others = colours[0], colours[1:]
        first = self.apply(shared, offset, box, rng)
        creates = hasattr(box, "kraus") and all(
            value is not None
            for value in shared.values[offset:offset + len(box.dom)])
        if creates and self.indistinguishability != 1:
            index = [slice(None)] * first.vector.ndim
            for axis in first.axes(offset, offset + len(box.cod)):
                index[axis] = 0
            if np.sum(np.abs(first.vector[tuple(index)]) ** 2) < 1 - 1e-12:
                raise NotImplementedError(
                    f"{box} creates photons, which cannot be coloured: feed "
                    "partially distinguishable photons through a source.")
        if not others:
            return [first]
        if not hasattr(box, "kraus"):
            return [first] + [CQState(state.vector, first.values, state.weight)
                              for state in others]
        if creates:
            return [first] + [self.vacant(state, offset, box, first.values)
                              for state in others]
        after = [first] + [self.apply(state, offset, box, rng)
                           for state in others]
        outputs = [i for i, value in enumerate(
            first.values[offset:offset + len(box.cod)], offset)
            if value is not None]
        if not outputs:
            return after
        pool = getattr(box, "pool", None)
        if pool is None or any(
                ob.name not in QUANTUM for ob in box.dom.inside):
            raise NotImplementedError(
                f"{box} has classical outputs but no pool to add up the "
                "outcomes of photons of different colours.")
        values = list(first.values)
        for i in outputs:
            values[i] = int(pool(state.values[i] for state in after))
        return [CQState(state.vector, values, state.weight)
                for state in after]

    def vacant(self, state: CQState, offset: int, box,
               values: list) -> CQState:
        """
        A colour after a `box` with no quantum input, such as the vacuum
        of a loop: its quantum outputs in the vacuum, its wires taking the
        `values` of the shared colour.
        """
        if any(ob.name not in QUANTUM for ob in box.cod.inside):
            raise NotImplementedError(
                f"{box} creates classical values from classical inputs "
                "only, sample it as a classical box.")
        vector = state.vector
        before = state.values[:offset].count(None)
        for axis, _ in enumerate(box.cod, before):
            vector = np.moveaxis(np.multiply.outer(
                vector, np.eye(self.cap + 1)[0]), -1, axis)
        return CQState(vector, values, state.weight)

    def vacuum(self, state: CQState) -> CQState:
        """
        The memory of a new colour: the vacuum on the quantum wires of
        `state` and its classical values.
        """
        vector = np.zeros(state.vector.shape, dtype=complex)
        vector[(0, ) * vector.ndim] = 1
        return CQState(vector, list(state.values))

    @staticmethod
    def occupations(colours: list) -> np.ndarray:
        """
        The mean photon numbers of the quantum wires of the memory, added
        up over its `colours`.

        >>> memory = CQState(np.array([0, 1]), [None, 3])
        >>> Unravelling.occupations([memory, memory])
        array([2.])
        """
        return np.sum([[mean for mean, value in zip(
            state.means(), state.values) if value is None]
            for state in colours], axis=0)

    def tick(self, colours: list, rng: np.random.Generator,
             source: Source = None):
        """
        One tick from the `colours` of the memory, the shared colour first:
        the inputs drawn from the source, or from `source` when given, and
        coloured, the step, and the outputs measured, the counts of the
        colours added up. Returns the outputs and the colours of the
        memory, a colour of its own forgotten once its photons have left.
        """
        source = self.source if source is None else feeding(
            self.diagram, source)
        optical = tuple(ob.name == "qmode" for ob in self.diagram.dom.inside)
        if source.superposed and self.indistinguishability != 1:
            raise NotImplementedError(
                f"{source} injects superpositions: its photons cannot "
                "be coloured.")
        shared, alone = colouring(source.inject(rng, self.cap + 1),
                                  self.indistinguishability, rng, optical)
        empty = tuple(0 if light else n for n, light in zip(shared, optical))
        colours = self.run(self.step, [self.feed(colours[0], shared)] + [
            self.feed(state, empty) for state in colours[1:]] + [
            self.feed(self.vacuum(colours[0]), single) for single in alone],
            rng)
        widths = [self.levels if self.levels and ob.name == "qmode" else 1
                  for ob in self.diagram.cod.inside]
        bounds = np.cumsum([0] + widths)
        measured = [state.measure(0, bounds[-1], rng) for state in colours]
        pooled = [sum(pattern[bounds[i]:bounds[i + 1]])
                  for i in range(len(widths))
                  for pattern, _ in measured[:1]]
        for pattern, _ in measured[1:]:
            pooled = [n + sum(pattern[bounds[i]:bounds[i + 1]]) * (
                ob.name == "qmode") for i, (n, ob) in enumerate(
                    zip(pooled, self.diagram.cod.inside))]
        memory = [state for _, state in measured]
        memory = memory[:1] + [state for state in memory[1:]
                               if abs(state.vector.flat[0]) ** 2 < 1 - 1e-12]
        return tuple(int(n) for n in pooled), memory

    def start(self, rng: np.random.Generator) -> list:
        """ The colours of the memory in the state of the loops. """
        return self.run(self.initial, [CQState.empty()], rng)

    def trajectory(self, ticks: int, seed=None) -> list:
        """
        The outputs measured over `ticks` ticks of one trajectory started
        from the state of the loops.
        """
        rng = np.random.default_rng(seed)
        colours, patterns = self.start(rng), []
        for _ in range(ticks):
            pattern, colours = self.tick(colours, rng)
            patterns.append(pattern)
        return patterns

    def sample(self, ticks: int, burn_in: int = 0, seed=None) -> list:
        """
        A sample of the outputs measured over `ticks` ticks, after
        `burn_in` ticks.
        """
        return self.trajectory(burn_in + ticks, seed)[burn_in:]


class Fields(Unravelling):
    """
    The trajectories of classical light, mixtures of coherent states,
    through a :class:`optyx.channel.Diagram` with feedback loops: linear
    optics, losses, detectors and classical feed-forward.

    A product of coherent states stays one under linear optics and loss,
    and detecting some of its modes leaves the others coherent, so the
    state of the optical wires is a vector of amplitudes, one per wire,
    and the classical wires hold values. A tick draws the amplitudes of
    the source from its Glauber P function: fixed for a
    :class:`PhaseLocked` laser, with a random phase for :class:`Coherent`
    light, Gaussian for :class:`Thermal` light. A box of linear optics
    multiplies the amplitudes by its mode matrix, read off its single
    photon amplitudes and checked on two photons, so a non-linear box
    raises. A number resolving detector draws a Poisson count of mean
    :math:`|\\alpha|^2`, a threshold detector clicks with probability
    :math:`1 - e^{-|\\alpha|^2}`, and a box with a `coherent` method, such
    as :class:`optyx.photonic.Dephasing`, says itself what it does to
    amplitudes. Classical boxes and classically controlled boxes are those
    of the :class:`Unravelling`. Nothing is cut: a tick costs the number of
    wires, not of photons.

    A beam splitter fed back on itself by a laser with a phase
    reference: the mean count of the samples is the exact mean of the
    :class:`Interferometer`, its field included.

    >>> from optyx.channel import qmode
    >>> from optyx.photonic import BS, Create
    >>> loop = BS.feedback(mem=qmode, state=Create(0))
    >>> laser = PhaseLocked((1, ))
    >>> samples = Fields(loop, source=laser).sample(4000, burn_in=20, seed=0)
    >>> network = Interferometer.from_diagram(loop, source=laser)
    >>> exact = network.advance(network.correlations(), (), network.field())[1]
    >>> bool(abs(np.mean(samples) - exact[0]) < .05)
    True
    """
    def __init__(self, diagram, source: Source = None):
        super().__init__(diagram, cap=2, source=source)
        if any(ob.name != "qmode" for ob in diagram.dom.inside) \
                or self.levels:
            raise NotImplementedError(
                "Fields are fed on optical modes, by photons without "
                "internal states.")

    def __repr__(self):
        return f"Fields({self.diagram!r}, source={self.source})"

    def start(self, rng: np.random.Generator) -> list:
        return self.run(self.initial, [CQState(
            np.zeros(0, dtype=complex), [])], rng)

    def matrix(self, box, values: tuple) -> np.ndarray:
        """
        The mode matrix of `box` on coherent amplitudes, its classical
        inputs taking `values`: its amplitude from one photon in each
        quantum input to one photon in each quantum output, with nothing
        in its environment. Checked to send two photons in an input to the
        squares of these, which linear optics does; raises otherwise.
        """
        key = id(box), values
        if key in self.tensors:
            return self.tensors[key][1]
        quantum = [i for i, value in enumerate(values) if value is None]
        outputs = [ob for ob in box.cod.inside if ob.name in QUANTUM]
        if len(outputs) != len(box.cod):
            raise NotImplementedError(
                f"{box} has classical outputs and is not a detector.")
        vacuum = self.transfer(box, values, 0, -1)
        if not np.isclose(abs(vacuum[(0, ) * vacuum.ndim]), 1):
            raise NotImplementedError(
                f"{box} creates photons: only light that is a mixture of "
                "coherent states is a field.")
        matrix, squares = (np.array([[self.transfer(box, values, photons, i)[
            tuple(photons * (j == k) for j in range(len(outputs) + len(
                box.env)))] for k in range(len(outputs))]
            for i, _ in enumerate(quantum)], dtype=complex).T.reshape(
                len(outputs), len(quantum)) for photons in (1, 2))
        if not np.allclose(squares, matrix ** 2):
            raise NotImplementedError(
                f"{box} is not linear optics: coherent states do not stay "
                "coherent through it.")
        self.tensors[key] = box, matrix
        return matrix

    def transfer(self, box, values: tuple, photons: int,
                 mode: int) -> np.ndarray:
        """
        The output of `box`, its classical inputs taking `values`, from
        `photons` photons in its quantum input `mode` and none in the
        others: an axis per quantum output, then per mode of its
        environment.
        """
        quantum = values.count(None)
        vector = np.zeros((3, ) * quantum, dtype=complex)
        vector[tuple(photons * (j == mode) for j in range(quantum))] = 1
        vector, kept = self.embed(CQState(vector, list(values)), 0, box)
        for inner, offset in zip(box.kraus.boxes, box.kraus.offsets):
            vector = self.evolve(vector, kept + offset, inner)
        return vector

    def apply(self, state: CQState, offset: int, box,
              rng: np.random.Generator) -> CQState:
        stop = offset + len(box.dom)
        values = state.values[offset:stop]
        if isinstance(box, symmetric.Swap) and values.count(None) == 2:
            first = state.values[:offset].count(None)
            vector = state.vector.copy()
            vector[[first, first + 1]] = vector[[first + 1, first]]
            return CQState(vector, state.values, state.weight)
        if isinstance(box, symmetric.Swap) or not hasattr(box, "kraus"):
            return super().apply(state, offset, box, rng)
        before = state.vector[:state.values[:offset].count(None)]
        after = state.vector[state.values[:stop].count(None):]
        inputs = state.vector[len(before):len(state.vector) - len(after)]
        if hasattr(box, "coherent"):
            return CQState(np.concatenate([before, box.coherent(
                inputs, rng), after]), state.values, state.weight)
        pool = getattr(box, "pool", None)
        if pool is not None:
            if any(ob.name not in QUANTUM for ob in box.dom.inside):
                raise NotImplementedError(
                    f"{box} measures classical wires.")
            means = np.abs(inputs) ** 2
            outcome = rng.poisson(means) if pool is sum \
                else rng.random(len(means)) < 1 - np.exp(-means)
            return CQState(np.concatenate([before, after]), state.values[
                :offset] + [int(n) for n in outcome] + state.values[stop:],
                state.weight)
        matrix = self.matrix(box, tuple(values))
        return CQState(np.concatenate([before, matrix @ inputs, after]),
                       state.values[:offset] + [None] * len(box.cod)
                       + state.values[stop:], state.weight)

    @staticmethod
    def occupations(colours: list) -> np.ndarray:
        """
        The mean photon numbers of the optical wires of the memory, the
        squares of their amplitudes.
        """
        return np.sum([np.abs(state.vector) ** 2 for state in colours],
                      axis=0)

    def transit(self, colours: list, rng: np.random.Generator,
                source: Source = None) -> CQState:
        """
        One tick from the `colours` of the memory, its outputs left
        unmeasured: their amplitudes or values first, then the memory. The
        amplitudes are drawn from `source` when given.
        """
        source = self.source if source is None else feeding(
            self.diagram, source)
        state = colours[0]
        state = CQState(np.concatenate([source.amplitudes(rng),
                                        state.vector]),
                        [None] * len(self.diagram.dom) + state.values)
        state, = self.run(self.step, [state], rng)
        return state

    def tick(self, colours: list, rng: np.random.Generator,
             source: Source = None):
        return self.detect(self.transit(colours, rng, source), rng)

    def detect(self, state: CQState, rng: np.random.Generator):
        """
        The outputs of a :meth:`transit` measured, a Poisson count of mean
        :math:`|\\alpha|^2` for each optical one, and the memory left.
        """
        outputs = len(self.diagram.cod)
        quantum = state.values[:outputs].count(None)
        counts = iter(rng.poisson(np.abs(state.vector[:quantum]) ** 2))
        pattern = tuple(int(next(counts)) if value is None else value
                        for value in state.values[:outputs])
        return pattern, [CQState(state.vector[quantum:],
                                 state.values[outputs:])]


class Gaussian:
    """
    The trajectories of Gaussian light through a passive recurrent
    :class:`Interferometer`, its visible outputs read by homodyne detection
    of their :math:`x` quadrature.

    Gaussian light stays Gaussian through linear optics and loss, and so
    does the loop conditioned on a homodyne outcome. The state of the loop
    is then the means :math:`\\mu` and the covariance :math:`V` of its
    quadratures :math:`x = (a + a^\\dagger) / 2` and
    :math:`p = (a - a^\\dagger) / 2i`, all the :math:`x` then all the
    :math:`p`, the vacuum having :math:`V = 1 / 4`. A tick:

    1. stacks the loop with the quadratures of the source,
       :meth:`Source.quadratures`, the environment in the vacuum;
    2. applies the real form of the mode matrix,
       :math:`S = \\begin{pmatrix} \\mathrm{Re}\\, U & -\\mathrm{Im}\\,
       U \\\\ \\mathrm{Im}\\, U & \\mathrm{Re}\\, U \\end{pmatrix}`,
       then mixes in the vacuum of the loss of the loop and of the
       inefficiency of the detectors;
    3. draws the :math:`x` quadratures of the visible outputs from their
       Gaussian marginal and conditions the loop on them: its means move by
       :math:`V_{\\ell o} V_{o o}^{-1} (x - \\mu_o)` and its covariance
       becomes the Schur complement
       :math:`V_{\\ell \\ell} - V_{\\ell o} V_{o o}^{-1} V_{o \\ell}`,
       whatever the outcome.

    Without a random generator the outcomes are their means: the
    deterministic mean field of the network.

    A beam splitter fed back on itself by a laser: once the loop has
    settled, homodyne detection reads the real part of the stationary field,
    with the spread of the vacuum, :math:`1 / 2`.

    >>> loop = Interferometer([[.6, .8], [.8, -.6]], 1,
    ...                       source=PhaseLocked((1, )))
    >>> samples = Gaussian(loop).sample(4000, burn_in=20, seed=0)
    >>> bool(abs(np.mean(samples) - loop.transfer()[0, 0].real) < .03)
    True
    >>> bool(abs(np.std(samples) - .5) < .03)
    True

    A delay line fed with squeezed light reads it below the vacuum noise:

    >>> delay = Interferometer([[0, 1], [1, 0]], 1, source=Squeezed((.5, )))
    >>> samples = Gaussian(delay).sample(4000, burn_in=2, seed=0)
    >>> bool(abs(np.var(samples) - np.exp(-1) / 4) < .01)
    True
    """
    def __init__(self, network: Interferometer):
        self.network = network
        unitary = network.unitary
        self.symplectic = np.block([[unitary.real, -unitary.imag],
                                    [unitary.imag, unitary.real]])
        modes, loop = network.modes, network.loop
        self.memory = np.r_[:loop, modes:modes + loop]
        self.inputs = np.r_[loop:modes, modes + loop:2 * modes]
        self.outputs = loop + np.array(network.visible, dtype=int)
        self.gains = np.ones(2 * modes)
        self.gains[self.memory] = np.sqrt(network.transmissivity)
        self.gains[self.outputs] = np.sqrt(network.efficiency)

    def __repr__(self):
        return f"Gaussian({self.network!r})"

    def __eq__(self, other):
        return isinstance(other, Gaussian) and self.network == other.network

    def start(self) -> tuple:
        """ The vacuum loop: its means and its covariance. """
        return np.zeros(2 * self.network.loop), np.eye(
            2 * self.network.loop) / 4

    def tick(self, memory: tuple, rng: np.random.Generator = None,
             source: Source = None) -> tuple:
        """
        One tick from the `memory`, the means and the covariance of the
        loop, fed by the source of the network or `source`: the homodyne
        outcomes of the visible outputs, their means without `rng`, and the
        memory they leave.
        """
        means, covariance = memory
        inputs, spread = self.network.padded(source).quadratures()
        size = len(self.gains)
        state, joint = np.zeros(size), np.zeros((size, size))
        state[self.memory], state[self.inputs] = means, inputs
        joint[np.ix_(self.memory, self.memory)] = covariance
        joint[np.ix_(self.inputs, self.inputs)] = spread
        state = self.gains * (self.symplectic @ state)
        joint = np.outer(self.gains, self.gains) * (
            self.symplectic @ joint @ self.symplectic.T) + np.diag(
                1 - self.gains ** 2) / 4
        kept, read = self.memory, self.outputs
        outcome = state[read] if rng is None else rng.multivariate_normal(
            state[read], joint[np.ix_(read, read)])
        regression = np.linalg.solve(joint[np.ix_(read, read)],
                                     joint[np.ix_(read, kept)]).T
        return outcome, (
            state[kept] + regression @ (outcome - state[read]),
            joint[np.ix_(kept, kept)] - regression @ joint[np.ix_(read, kept)])

    def trajectory(self, ticks: int, seed=None) -> list:
        """
        The homodyne outcomes of `ticks` ticks of one trajectory started
        from the vacuum loop.
        """
        rng = np.random.default_rng(seed)
        memory, outcomes = self.start(), []
        for _ in range(ticks):
            outcome, memory = self.tick(memory, rng)
            outcomes.append(tuple(float(x) for x in outcome))
        return outcomes

    def sample(self, ticks: int, burn_in: int = 0, seed=None) -> list:
        """
        A sample of the homodyne outcomes over `ticks` ticks, after
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
