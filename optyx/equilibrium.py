"""
Overview
--------

Equilibrium propagation on photonic diagrams with feedback loops: the
gradient of a stationary expectation from a nudged run of the same device,
without differentiating through the loop.

A diagram with feedback whose inputs are fed by a
:class:`optyx.sampling.Source` at every tick settles into a stationary
regime. Its one-particle observables are the mean photon numbers
:math:`g` of its memory modes, at the cut where a tick starts, and the mean
counts :math:`o` of its outputs. A family of such diagrams, parametrised by
:math:`\\theta`, defines the cost

.. math::

    J(\\theta) = c \\cdot o(\\theta),

and a differentiable cost :math:`C(o)` is linearised at the free means,
:math:`c = \\nabla C(o)`.

Write one tick as the map :math:`(g, q) \\mapsto (g', o)` of the means of
the memory and of the source, which is linear for passive diagrams and
phase-randomised sources: :math:`g' = T g + S q` and :math:`o = R g + D q`
on mean photon numbers when the memory is dephased. The gradient of the
stationary cost is then

.. math::

    \\partial_a J = c \\cdot \\partial_a o + \\lambda \\cdot \\partial_a g',
    \\qquad \\lambda = (1 - T^\\top)^{-1} R^\\top c,

where :math:`\\partial_a` differentiates one tick only, at the stationary
state, and :math:`\\lambda` is the adjoint of the loop. Equilibrium
propagation [SB17]_ replaces the adjoint by a nudged run of the device:
:meth:`Equilibrium.propagate` injects the error back into the ports where
it is read, moving the means of the source to :math:`q \\pm \\beta c`, and
measures the response of the memory,

.. math::

    \\hat\\lambda = \\frac{g_{+\\beta} - g_{-\\beta}}{2 \\beta}
    = (1 - T)^{-1} S c,

which it contracts with the derivative of one tick, the vector-field form
of equilibrium propagation [SMBB18]_. The estimate is exact when
:math:`\\hat\\lambda = \\lambda`, i.e. when the response of the means is
reciprocal, :math:`R = S^\\top` and :math:`T = T^\\top`: for a reciprocal
mode matrix, :math:`U = U^\\top`, whose memory is dephased at the cut by
:class:`optyx.photonic.Dephasing`, the one-tick map of the means is
:math:`|U_{ij}|^2`, symmetric. Coherence between the memory modes,
non-reciprocal optics and feed-forward, whose classical memory the means do
not see, make it an approximation, measured against
:meth:`Equilibrium.gradient`.

Phase-locked light, a :class:`optyx.sampling.PhaseLocked` laser whose
memory keeps its phase, is trained on its fields rather than its
intensities: :meth:`Equilibrium.propagate_fields` nudges the field of the
source by the conjugate error :math:`\\pm \\beta\\, c\\, \\bar y`, reads the
response of the field of the memory, and is exact for a symmetric mode
matrix. Intensities miss the interference of the fields from successive
ticks, and :meth:`Equilibrium.propagate` is then biased.

For passive diagrams, linear optics with losses whose step may start by
dephasing memory modes, the means are exact one-particle quantities,
:meth:`optyx.sampling.Interferometer.correlations`, independent of the
statistics of the source and of the indistinguishability of its photons.
With feed-forward they are sampled by the
:class:`optyx.sampling.Unravelling`, with the same randomness for the
runs that are compared.

Classes
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Equilibrium

Example
-------

A loop of two memory modes and one port, its step a palindrome of
beam splitters between losses, so that its mode matrix is symmetric, and
its memory dephased at the cut:

>>> from optyx.channel import qmode
>>> from optyx.photonic import Create, MZI, Phase, PhotonLoss, Dephasing
>>> from optyx.sampling import Coherent
>>> def loop(theta):
...     half = MZI(theta[0], 0) @ qmode >> qmode @ MZI(theta[1], 0) \\
...         >> qmode @ qmode @ Phase(theta[2])
...     loss = qmode @ PhotonLoss(.9) @ PhotonLoss(.9)
...     step = qmode @ Dephasing() @ Dephasing() >> loss >> half \\
...         >> half[::-1] >> loss
...     return step.feedback(mem=qmode @ qmode, state=Create(0, 0))
>>> ep = Equilibrium(loop, Coherent((1., )), cost=(1., ))
>>> theta = (.2, .3, .1)
>>> assert np.allclose(ep.propagate(theta, exact=True),
...                    ep.gradient(theta, exact=True), atol=1e-6)

.. [SB17] B. Scellier and Y. Bengio, Equilibrium propagation: bridging the
    gap between energy-based models and backpropagation, Frontiers in
    Computational Neuroscience 11, 24 (2017).

.. [SMBB18] B. Scellier, A. Goyal, J. Binas, T. Mesnard and Y. Bengio,
    Generalization of equilibrium propagation to vector field dynamics,
    arXiv:1808.04873 (2018).
"""

from __future__ import annotations

import numpy as np

from optyx.photonic import Dephasing
from optyx.sampling import (
    Fields, Interferometer, PhaseLocked, Source, Unravelling)


class Equilibrium:
    """
    The stationary regime of a family of diagrams with feedback, and the
    gradient of a linear cost of its mean counts, by central differences
    and by equilibrium propagation.

    Parameters:
        family : A function from the parameters :math:`\\theta` to a
            :class:`optyx.channel.Diagram` with feedback, the same wiring
            for every :math:`\\theta`, with as many inputs as outputs:
            output :math:`k` is read where input :math:`k` is injected.
        source : The :class:`optyx.sampling.Source` feeding the inputs.
        cost : The weights :math:`c` of the mean counts of the outputs.
        indistinguishability : The probability that a photon of the
            source is in the internal state shared by all.
        cap : The most photons a mode holds in the
            :class:`optyx.sampling.Unravelling`.
        fields : Whether to sample classical light by its
            :class:`optyx.sampling.Fields` rather than its
            :class:`optyx.sampling.Unravelling`: no cap, a cost linear in
            the wires, phase-locked lasers included.
    """
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(self, family, source: Source, cost,
                 indistinguishability: float = 1., cap: int = 4,
                 fields: bool = False):
        self.family, self.source = family, source
        self.cost = np.asarray(cost, dtype=float)
        self.indistinguishability = float(indistinguishability)
        self.cap, self.fields = cap, fields
        if len(self.cost) != len(source):
            raise ValueError(
                f"Expected a cost on the {len(source)} ports of the source, "
                f"got {self.cost}.")

    def nudged(self, beta: float) -> Source:
        """ The source with its means moved by `beta` times the cost. """
        return self.source.shift(beta * self.cost) if beta else self.source

    def steered(self, beta: float, error) -> Source:
        """
        The :class:`optyx.sampling.PhaseLocked` source with its field moved
        by `beta` times the complex `error`: the nudge of
        :meth:`propagate_fields`.
        """
        if not isinstance(self.source, PhaseLocked):
            raise NotImplementedError(
                f"The fields of {self.source} are not phase-locked.")
        return PhaseLocked(np.add(self.source.coherent, beta * np.asarray(
            error)))

    def interferometer(self, theta, beta: float = 0.,
                       source: Source = None) -> tuple:
        """
        The :class:`optyx.sampling.Interferometer` of a passive diagram of
        the family, fed by `source`, the :meth:`nudged` source by default,
        and its loop modes dephased at the start of its step.
        """
        diagram = self.family(theta)
        initial, _ = diagram.boundary()
        step, inputs = diagram.one_step(), len(diagram.dom)
        boxes = [layer.inside[0][:2] for layer in step.inside]
        depth = next((i for i, (_, box) in enumerate(boxes)
                      if not isinstance(box, Dephasing)), len(boxes))
        dephased = [len(left) - inputs for left, _ in boxes[:depth]
                    if len(left) >= inputs]
        network = Interferometer.from_step(
            initial, step[depth:], len(diagram.cod),
            self.nudged(beta) if source is None else source,
            self.indistinguishability)
        return network, tuple(dephased)

    def unravelling(self, theta, beta: float = 0.,
                    source: Source = None) -> Unravelling:
        """
        The sampler of a diagram of the family, fed by `source`, the
        :meth:`nudged` source by default: its
        :class:`optyx.sampling.Unravelling`, or its
        :class:`optyx.sampling.Fields` with `fields`.
        """
        source = self.nudged(beta) if source is None else source
        if self.fields:
            return Fields(self.family(theta), source)
        return Unravelling(self.family(theta), self.cap, source,
                           self.indistinguishability)

    def exact(self, theta, beta: float = 0.) -> tuple:
        """
        The stationary one-particle correlation matrix of the memory of a
        passive diagram, its mean field and the mean counts of its outputs.
        """
        network, dephased = self.interferometer(theta, beta)
        correlations, field = network.correlations(dephased), \
            network.field(dephased)
        return correlations, field, \
            network.advance(correlations, dephased, field)[1]

    @staticmethod
    def stationary(sampler: Unravelling, others: list, ticks: int,
                   burn_in: int, rng: np.random.Generator):
        """
        The ticks of one trajectory of `sampler` after `burn_in`: for each,
        the mean photon numbers of the memory at the cut and the counts of
        the tick, then the same one tick later under each of the `others`,
        the same randomness for all of them.
        """
        colours = sampler.start(rng)
        for _ in range(burn_in):
            _, colours = sampler.tick(colours, rng)
        for _ in range(ticks):
            branch = int(rng.integers(2 ** 63))
            branched = [other.tick(colours, np.random.default_rng(branch))
                        for other in others]
            memory = sampler.occupations(colours)
            counts, colours = sampler.tick(colours, rng)
            yield np.concatenate([memory, counts]), [
                np.concatenate([other.occupations(after), counts])
                for other, (counts, after) in zip(others, branched)]

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def means(self, theta, beta: float = 0., ticks: int = 1000,
              burn_in: int = 100, seed=None, shots: int = 1,
              shifts=()) -> tuple:
        """
        The mean photon numbers of the memory at the cut and the mean
        counts of the outputs, over `ticks` ticks after `burn_in` of each
        of `shots` trajectories, sampled from the child seeds of `seed`.
        With `shifts`, a list of parameters, also the means one tick after
        each recorded state of the memory, under each of them, with the
        same randomness for all.
        """
        sampler = self.unravelling(theta, beta)
        others = [self.unravelling(shifted) for shifted in shifts]
        records = [record for child in np.random.SeedSequence(seed).spawn(
            shots) for record in self.stationary(
                sampler, others, ticks, burn_in, np.random.default_rng(child))]
        mean = np.mean([free for free, _ in records], axis=0)
        size = len(mean) - len(self.cost)
        return (mean[:size], mean[size:]), list(np.mean(
            [branched for _, branched in records], axis=0))

    def value(self, theta, exact: bool = False, **sampling) -> float:
        """ The stationary cost :math:`J(\\theta) = c \\cdot o(\\theta)`. """
        counts = self.exact(theta)[2] if exact \
            else self.means(theta, **sampling)[0][1]
        return float(self.cost @ counts)

    def gradient(self, theta, step: float = 1e-6, exact: bool = False,
                 **sampling) -> np.ndarray:
        """
        The gradient of :meth:`value` by central differences of `step`,
        the reference that :meth:`propagate` approximates; sampled with the
        same seed on both sides.
        """
        theta = np.asarray(theta, dtype=float)
        return np.array([
            (self.value(theta + step * unit, exact, **sampling)
             - self.value(theta - step * unit, exact, **sampling)) / (2 * step)
            for unit in np.eye(len(theta))])

    def propagate(self, theta, beta: float = .1, step: float = 1e-6,
                  exact: bool = False, **sampling) -> np.ndarray:
        """
        The equilibrium propagation estimate of the gradient: the response
        :math:`\\hat\\lambda` of the memory to the nudge of the source by
        :math:`\\pm \\beta c`, contracted with the derivative of one tick
        from the free stationary state, by central differences of `step`.
        """
        theta = np.asarray(theta, dtype=float)
        shifts = [theta + sign * step * unit
                  for unit in np.eye(len(theta)) for sign in (1, -1)]
        if exact:
            correlations, field, _ = self.exact(theta)
            nudges = [np.real(np.diag(self.exact(theta, sign * beta)[0]))
                      for sign in (1, -1)]
            ticks = []
            for shifted in shifts:
                network, dephased = self.interferometer(shifted)
                after, counts = network.advance(
                    correlations, dephased, field)
                ticks.append(np.concatenate([np.real(np.diag(after)),
                                             counts]))
        else:
            nudges = [self.means(theta, sign * beta, **sampling)[0][0]
                      for sign in (1, -1)]
            _, ticks = self.means(theta, shifts=shifts, **sampling)
        adjoint = (nudges[0] - nudges[1]) / (2 * beta)
        weights = np.concatenate([adjoint, self.cost])
        return np.array([weights @ (ticks[2 * a] - ticks[2 * a + 1])
                         for a in range(len(theta))]) / (2 * step)

    @staticmethod
    def flowing(sampler: Fields, others: list, ticks: int, burn_in: int,
                rng: np.random.Generator):
        """
        The ticks of one trajectory of the :class:`optyx.sampling.Fields`
        `sampler` after `burn_in`: for each, the field of the memory at the
        cut and the field of the outputs before they are detected, then the
        fields of the memory and the outputs one tick later under each of
        the `others`, the same randomness for all of them.
        """
        colours = sampler.start(rng)
        for _ in range(burn_in):
            _, colours = sampler.tick(colours, rng)
        size = len(sampler.diagram.cod)
        for _ in range(ticks):
            branch = int(rng.integers(2 ** 63))
            branched = [other.transit(colours, np.random.default_rng(branch))
                        for other in others]
            memory = colours[0].vector
            state = sampler.transit(colours, rng)
            yield np.r_[memory, state.vector[:size]], [
                np.r_[after.vector[size:], after.vector[:size]]
                for after in branched]
            _, colours = sampler.detect(state, rng)

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def flows(self, theta, source: Source = None, ticks: int = 1000,
              burn_in: int = 100, seed=None, shots: int = 1,
              shifts=()) -> tuple:
        """
        The mean fields of the memory at the cut and of the outputs before
        detection, sampled by the :class:`optyx.sampling.Fields` fed by
        `source`, and with `shifts` the same one tick later under each.
        """
        sampler = Fields(self.family(theta), source or self.source)
        others = [Fields(self.family(shifted), source or self.source)
                  for shifted in shifts]
        records = [record for child in np.random.SeedSequence(seed).spawn(
            shots) for record in self.flowing(
                sampler, others, ticks, burn_in, np.random.default_rng(child))]
        mean = np.mean([free for free, _ in records], axis=0)
        size = len(mean) - len(self.cost)
        return (mean[:size], mean[size:]), list(np.mean(
            [branched for _, branched in records], axis=0))

    def propagate_fields(self, theta, beta: float = .1, step: float = 1e-6,
                         exact: bool = False, **sampling) -> np.ndarray:
        """
        Equilibrium propagation on the fields of phase-locked light, whose
        memory keeps its phase. The cost is :math:`J = c \\cdot |y|^2` for
        the mean field :math:`y` of the outputs, and its gradient is
        :math:`2\\, \\mathrm{Re}\\,(\\lambda \\cdot \\partial_a x' + g \\cdot
        \\partial_a y)` for :math:`g = c\\, \\bar y` and the adjoint
        :math:`\\lambda` of the loop on fields. The nudge moves the field
        of the source by :math:`\\pm \\beta g`, injecting the conjugate
        error field where it is read, and the response
        :math:`\\hat\\lambda` of the field of the memory is the adjoint
        when the mode matrix is symmetric. The derivatives are those of
        one tick from the free stationary fields.

        >>> from optyx.channel import Diagram, qmode
        >>> from optyx.photonic import Create, MZI, PhotonLoss
        >>> def loop(theta):
        ...     half = [MZI(theta[0], 0), MZI(theta[1], 0)]
        ...     loss = qmode @ PhotonLoss(.8)
        ...     step = Diagram.then(loss, *half, *half[::-1], loss)
        ...     return step.feedback(mem=qmode, state=Create(0))
        >>> ep = Equilibrium(loop, PhaseLocked((1, )), cost=(1., ))
        >>> assert np.allclose(ep.propagate_fields((.2, .3), exact=True),
        ...                    ep.gradient((.2, .3), exact=True), atol=1e-5)
        """
        theta = np.asarray(theta, dtype=float)
        shifts = [theta + sign * step * unit
                  for unit in np.eye(len(theta)) for sign in (1, -1)]
        if exact:
            network, dephased = self.interferometer(theta)
            field = network.field(dephased)
            error = self.cost * np.conj(network.carry(field, dephased)[1])
            nudges = [self.interferometer(theta, source=self.steered(
                sign * beta, error))[0].field(dephased) for sign in (1, -1)]
            ticks = [np.r_[tuple(network.carry(field, dephased))] for
                     network, dephased in map(self.interferometer, shifts)]
        else:
            (_, output), _ = self.flows(theta, **sampling)
            error = self.cost * np.conj(output)
            nudges = [self.flows(theta, self.steered(sign * beta, error),
                                 **sampling)[0][0] for sign in (1, -1)]
            _, ticks = self.flows(theta, shifts=shifts, **sampling)
        weights = np.r_[(nudges[0] - nudges[1]) / (2 * beta), error]
        return np.array([2 * np.real(weights @ (
            ticks[2 * a] - ticks[2 * a + 1])) for a in range(len(theta))]) \
            / (2 * step)
