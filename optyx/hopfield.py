"""
Overview
--------

A photonic chip in an electronic loop of :math:`\\tanh` units: the analog
optical computer of Microsoft [KGP+25]_, generalised to chips with photonic
loops of their own, to any light and to homodyne or photon counting.

:math:`N` units hold electronic states :math:`z`. At every electronic tick:

1. modulators write the activations :math:`r = \\tanh z` on the light of a
   source;
2. the chip, a passive diagram with or without feedback loops, runs
   `ratio` optical ticks with the modulators held;
3. detectors read the chip, :math:`m` being the mean readout over these
   ticks;
4. the electronics integrate :math:`z \\leftarrow (1 - a) z + a (g m + b)`,
   with leak :math:`a`, gain :math:`g` and bias :math:`b`.

The units are written on the light in one of two ways:

- **fields**: unit :math:`j` displaces the light of input :math:`j` by
  :math:`\\alpha r_j`, :class:`optyx.sampling.Displaced`, and homodyne
  detection reads :math:`m_i = x_i / \\alpha`. The light sets the noise:
  the vacuum for a laser, thermal or squeezed noise otherwise, sampled by
  the :class:`optyx.sampling.Gaussian`.
- **intensities**, on two rails: unit :math:`j` attenuates the light of
  inputs :math:`2j` and :math:`2j + 1` to the fractions :math:`[r_j]_+`
  and :math:`[r_j]_-` of their means :math:`q`, and photon counting reads
  :math:`m_i = (n_{2i} - n_{2i + 1}) / q_{2i}`. The light is any light
  without a phase reference: :class:`optyx.sampling.Coherent` and
  :class:`optyx.sampling.Thermal`, sampled by the
  :class:`optyx.sampling.Fields`, and single photons,
  :class:`optyx.sampling.Fock`, partially distinguishable or not, thinned
  by the modulators and sampled by the
  :class:`optyx.sampling.Interferometer`.

Once the chip has settled with the modulators held, the mean readout is
linear in the activations, :math:`\\langle m \\rangle = W r`, with the
couplings :math:`W` of :meth:`Hopfield.transfer`: the real part of the
stationary mode matrix :math:`T` of :meth:`optyx.sampling.Interferometer.\
transfer` for fields, the difference of the rails of the stationary
intensities for intensities. These are one-particle means, so the same
whatever the statistics of the light. A photonic loop changes :math:`W`
only, since :math:`T` sums over every number of round trips, and keeps it
symmetric for a reciprocal chip, :math:`U = U^\\top`. The fixed points of
the mean dynamics, :math:`z = g W r + b`, are then the critical points of
the Hopfield energy

.. math::

    E(r) = \\sum_i \\left(r_i\\, \\mathrm{artanh}\\, r_i
    + \\tfrac12 \\ln (1 - r_i^2)\\right) - \\tfrac12 g\\, r^\\top W r
    - b^\\top r,

and the dynamics settle into a unique one when :math:`g \\|W\\| < 1`.

The cost is :math:`C = \\tfrac12 (r_o - y)^2` on the activation of the
`output` unit, :math:`J(\\theta)` its stationary mean. Equilibrium
propagation [SB17]_ nudges the electronics by the drive
:math:`-\\beta (r_o - y)` on the output unit, the stationary points of
:math:`E + \\beta C`, and estimates

.. math::

    \\partial_\\theta J \\approx -\\frac{g}{4 \\beta} \\left(
    \\langle r^\\top \\partial_\\theta W r \\rangle_{+\\beta}
    - \\langle r^\\top \\partial_\\theta W r \\rangle_{-\\beta} \\right),

which is exact as :math:`\\beta \\to 0` for deterministic dynamics with a
symmetric :math:`W`. On hardware, :math:`r^\\top \\partial_\\theta W r` is
read by holding the modulators at :math:`r` and moving :math:`\\theta`.
With noisy light the stationary state is a distribution, which the
estimate averages over, and :meth:`Hopfield.gradient` measures the bias.

Classes
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Hopfield

Example
-------

Microsoft's architecture, with no photonic loop: two units read through a
symmetric beam splitter by homodyne detection of a bright laser. The
estimate is exact for the deterministic dynamics:

>>> from optyx.photonic import MZI
>>> from optyx.sampling import PhaseLocked
>>> chip = Hopfield(lambda theta: MZI(theta[0], 0), PhaseLocked((0, 0)),
...                 amplitude=10, gain=.8, bias=(.3, -.2), target=.5)
>>> assert np.allclose(chip.propagate((.3, ), beta=1e-4, exact=True),
...                    chip.gradient((.3, ), exact=True), atol=1e-6)

.. [KGP+25] K. P. Kalinin et al., Analog optical computer for AI inference
    and combinatorial optimization, Nature 645 (2025).

.. [SB17] B. Scellier and Y. Bengio, Equilibrium propagation: bridging the
    gap between energy-based models and backpropagation, Frontiers in
    Computational Neuroscience 11, 24 (2017).
"""

from __future__ import annotations

from itertools import islice

import numpy as np

from optyx.sampling import (
    Coherent, Displaced, FockState, Fields, Gaussian, Interferometer, Source)


class Hopfield:
    """
    A family of chips in an electronic loop of :math:`\\tanh` units, its
    stationary cost, the gradient of the cost and its equilibrium
    propagation estimate.

    Parameters:
        family : A function from the parameters :math:`\\theta` to a
            passive :class:`optyx.channel.Diagram`, with or without
            feedback loops, as many outputs as inputs: one per unit for
            fields, two for intensities.
        light : The :class:`optyx.sampling.Source` the modulators act on.
        amplitude : The displacement :math:`\\alpha` of a unit of
            activation one, for fields read by homodyne detection; without
            it, intensities on two rails read by photon counting.
        gain : The gain :math:`g` of the electronics.
        leak : The fraction :math:`a` of the state the electronics renew
            at a tick.
        bias : The bias :math:`b` of the units, zero by default.
        ratio : The number of optical ticks per electronic tick, the
            readout being their mean.
        output : The unit whose activation the cost reads.
        target : The target :math:`y` of the output.
        indistinguishability : The probability that a photon of the light
            is in the internal state shared by all.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    # pylint: disable=too-many-instance-attributes
    def __init__(self, family, light: Source, amplitude: float = None,
                 gain: float = 1., leak: float = .5, bias=None,
                 ratio: int = 1, output: int = -1, target: float = 0.,
                 indistinguishability: float = 1.):
        self.family, self.light, self.amplitude = family, light, amplitude
        self.gain, self.leak, self.ratio = float(gain), float(leak), ratio
        self.output, self.target = output, float(target)
        self.indistinguishability = float(indistinguishability)
        if amplitude is None and (len(light) % 2 or light.superposed):
            raise ValueError(
                f"Intensities on two rails need an even number of modes of "
                f"light with no phase reference, got {light}.")
        if amplitude is not None:
            light.quadratures()
        units = len(light) if amplitude is not None else len(light) // 2
        self.bias = np.zeros(units) if bias is None \
            else np.asarray(bias, dtype=float)

    @property
    def units(self) -> int:
        """ The number :math:`N` of units. """
        return len(self.bias)

    def network(self, theta, light: Source = None) -> Interferometer:
        """
        The :class:`optyx.sampling.Interferometer` of the chip at
        `theta`, fed by `light`, the light of the loop by default.
        """
        return Interferometer.from_diagram(
            self.family(theta), self.indistinguishability,
            self.light if light is None else light)

    def modulate(self, activations) -> Source:
        """
        The light the modulators set for the `activations`: displaced by
        :math:`\\alpha r` for fields, each rail attenuated to its part of
        :math:`r` for intensities.
        """
        activations = np.asarray(activations, dtype=float)
        if self.amplitude is not None:
            return Displaced(self.light, self.amplitude * activations)
        rails = np.ravel(np.column_stack([
            np.maximum(activations, 0), np.maximum(-activations, 0)]))
        return self.light.shift(np.multiply(self.light.means, rails - 1))

    def read(self, outcome) -> np.ndarray:
        """
        The readout of the units from the `outcome` of the detectors:
        the homodyne outcomes over :math:`\\alpha`, or the difference of
        the counts of the rails over their mean.
        """
        outcome = np.asarray(outcome, dtype=float)
        if self.amplitude is not None:
            return outcome / self.amplitude
        return (outcome[0::2] - outcome[1::2]) / np.array(
            self.light.means[0::2])

    def transfer(self, theta) -> np.ndarray:
        """
        The couplings :math:`W` of the chip at `theta`, the mean readout of
        settled light being :math:`W r`. For intensities, the readout is
        :math:`W_+ [r]_+ + W_- [r]_-`, so :math:`W = (W_+ - W_-) / 2`, and
        the readout is :math:`W r` when the chip is symmetric under the
        exchange of the rails, :math:`W_+ = -W_-`.
        """
        network = self.network(theta)
        if self.amplitude is not None:
            return np.real(network.transfer()[:, :self.units])
        means = np.array(self.light.means)
        columns = []
        for unit in np.eye(len(network.source))[:len(self.light)]:
            probe = Interferometer(
                network.unitary, network.loop, source=Coherent(unit),
                transmissivity=network.transmissivity,
                efficiency=network.efficiency, visible=network.visible)
            columns.append(probe.advance(probe.correlations())[1])
        rails = (np.array(columns).T[0::2] - np.array(columns).T[1::2]) \
            * means / means[0::2, None]
        return (rails[:, 0::2] - rails[:, 1::2]) / 2

    def optics(self, theta, exact: bool = False) -> tuple:
        """
        The optics of the chip at `theta`: a function starting its memory
        from a random generator and a function running an optical tick
        from the memory, a random generator and the light of the tick,
        returning the outcome of the detectors and the memory. With
        `exact`, the means: the mean field, or the one-particle
        correlations of the loop and the mean counts.
        """
        network = self.network(theta)
        if self.amplitude is not None:
            gaussian = Gaussian(network)
            return lambda rng: gaussian.start(), \
                lambda memory, rng, light: gaussian.tick(
                    memory, None if exact else rng, light)
        if exact:
            def tick(memory, rng, light):
                probe = Interferometer(
                    network.unitary, network.loop,
                    source=Coherent(network.padded(light).means),
                    transmissivity=network.transmissivity,
                    efficiency=network.efficiency, visible=network.visible)
                after, counts = probe.advance(memory)
                return counts, after
            return lambda rng: np.zeros((network.loop, network.loop)), tick
        try:
            self.light.amplitudes(np.random.default_rng(0))
        except NotImplementedError:
            return lambda rng: [FockState.vacuum(network.loop)], \
                network.tick
        fields = Fields(self.family(theta), self.light)
        return fields.start, fields.tick

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def run(self, theta, beta: float = 0., rng=None, start=None,
            exact: bool = False):
        """
        The electronic states after each electronic tick, from `start`,
        zero by default, with the drive of the nudge :math:`\\beta` and
        the random generator `rng`; the mean dynamics with `exact`.
        """
        begin, step = self.optics(theta, exact)
        memory = begin(rng)
        state = np.zeros(self.units) if start is None \
            else np.array(start, dtype=float)
        while True:
            activations = np.tanh(state)
            light = self.modulate(activations)
            readouts = []
            for _ in range(self.ratio):
                outcome, memory = step(memory, rng, light)
                readouts.append(self.read(outcome))
            drive = self.gain * np.mean(readouts, axis=0) + self.bias
            drive[self.output] -= beta * (
                activations[self.output] - self.target)
            state = (1 - self.leak) * state + self.leak * drive
            yield state

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def trajectory(self, theta, beta: float = 0., ticks: int = 100,
                   seed=None, start=None, exact: bool = False) -> np.ndarray:
        """
        The electronic states of `ticks` electronic ticks of :meth:`run`,
        one row each, from the random seed `seed`.
        """
        rng = None if exact else np.random.default_rng(seed)
        return np.array(list(islice(
            self.run(theta, beta, rng, start, exact), ticks)))

    def settle(self, theta, beta: float = 0., tol: float = 1e-12,
               ticks: int = 10 ** 5) -> np.ndarray:
        """
        The fixed point of the mean dynamics from zero, raising when they
        have not settled within `tol` after `ticks` electronic ticks.
        """
        previous = np.zeros(self.units)
        for state in islice(self.run(theta, beta, exact=True), ticks):
            if np.max(np.abs(state - previous)) < tol:
                return state
            previous = state
        raise ValueError(
            f"The mean dynamics have not settled within {tol} after {ticks} "
            "ticks.")

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def moments(self, theta, beta: float = 0., exact: bool = False,
                ticks: int = 1000, burn_in: int = 100, seed=None,
                shots: int = 1) -> tuple:
        """
        The stationary means of the activations and of their products,
        :math:`\\langle r \\rangle` and :math:`\\langle r r^\\top \\rangle`:
        at the fixed point with `exact`, otherwise over `ticks` electronic
        ticks after `burn_in` of `shots` trajectories from the child seeds
        of `seed`.
        """
        if exact:
            activations = np.tanh(self.settle(theta, beta))
            return activations, np.outer(activations, activations)
        activations = np.tanh(np.concatenate([
            self.trajectory(theta, beta, burn_in + ticks, child)[burn_in:]
            for child in np.random.SeedSequence(seed).spawn(shots)]))
        return np.mean(activations, axis=0), \
            activations.T @ activations / len(activations)

    def value(self, theta, exact: bool = False, **sampling) -> float:
        """
        The stationary cost
        :math:`J(\\theta) = \\langle \\tfrac12 (r_o - y)^2 \\rangle`.
        """
        means, products = self.moments(theta, exact=exact, **sampling)
        output = self.output % self.units
        return float(products[output, output] / 2
                     - self.target * means[output] + self.target ** 2 / 2)

    def gradient(self, theta, step: float = 1e-6, exact: bool = False,
                 **sampling) -> np.ndarray:
        """
        The gradient of :meth:`value` by central differences of `step`,
        sampled with the same seed on both sides.
        """
        theta = np.asarray(theta, dtype=float)
        return np.array([
            (self.value(theta + step * unit, exact, **sampling)
             - self.value(theta - step * unit, exact, **sampling)) / (2 * step)
            for unit in np.eye(len(theta))])

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def propagate(self, theta, beta: float = .1, step: float = 1e-6,
                  exact: bool = False, **sampling) -> np.ndarray:
        """
        The equilibrium propagation estimate of the gradient: the products
        of the activations under the nudges :math:`\\pm \\beta`, contracted
        with the derivative of the couplings, by central differences of
        `step`.
        """
        theta = np.asarray(theta, dtype=float)
        products = [self.moments(theta, sign * beta, exact, **sampling)[1]
                    for sign in (1, -1)]
        return np.array([
            -self.gain / (4 * beta) * np.sum(
                (self.transfer(theta + step * unit)
                 - self.transfer(theta - step * unit)) / (2 * step)
                * (products[0] - products[1]))
            for unit in np.eye(len(theta))])
