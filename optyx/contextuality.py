"""

Overview
--------

Measurement scenarios, empirical models and the contextual fraction.

A measurement :class:`Scenario` is a triple :math:`(X, \\mathcal{M}, O)`
of a set of measurements :math:`X`, a cover :math:`\\mathcal{M}` of
:math:`X` by **contexts**, the sets of measurements that can be performed
together, and a set of outcomes :math:`O_x` for each measurement
:math:`x \\in X`, after Abramsky and Brandenburger,
`The sheaf-theoretic structure of non-locality and contextuality
<https://doi.org/10.1088/1367-2630/13/11/113036>`_ (2011).

An :class:`EmpiricalModel` is a probability distribution :math:`e_C` on
the joint outcomes :math:`\\prod_{x \\in C} O_x` of each context
:math:`C \\in \\mathcal{M}`. It is **no-signalling** when the distributions
of two contexts have the same marginal on their intersection, and
**noncontextual** when they are all marginals of one distribution on the
**global assignments** :math:`\\prod_{x \\in X} O_x`.

The **noncontextual fraction** :math:`\\mathrm{NCF}(e)` is the largest
weight :math:`\\lambda` of a noncontextual model in a convex decomposition
:math:`e = \\lambda e^{NC} + (1 - \\lambda) e'` and the **contextual
fraction** is :math:`\\mathrm{CF}(e) = 1 - \\mathrm{NCF}(e)`, after
Abramsky, Barbosa and Mansfield, `Contextual fraction as a measure of
contextuality <https://doi.org/10.1103/PhysRevLett.119.050504>`_ (2017).
They are the value of the linear program

.. math::

    \\max \\; \\mathbf{1} \\cdot b
    \\quad \\text{subject to} \\quad
    \\mathbf{M} b \\leq v_e, \\; b \\geq 0

where :math:`b` is a subprobability distribution on global assignments,
:math:`v_e` the vector of the probabilities :math:`e_C(s)` and
:math:`\\mathbf{M}` the **incidence matrix** with :math:`\\mathbf{M}[(C, s),
g] = 1` when the global assignment :math:`g` restricts to :math:`s` on
:math:`C`. The number of global assignments is exponential in the number
of measurements, which bounds the scenarios this can decide.

The dual program :math:`\\min \\; v_e \\cdot y` subject to
:math:`\\mathbf{M}^T y \\geq \\mathbf{1}, y \\geq 0` reads off a
:class:`BellInequality` :math:`\\langle a, e \\rangle \\leq R` with
:math:`a = \\mathbf{1} - y` and :math:`R = |\\mathcal{M}| - 1`, satisfied by
every noncontextual model and violated by :math:`e` by exactly
:math:`\\mathrm{CF}(e)`.

Classes
-------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    Scenario
    EmpiricalModel
    BellInequality

Functions
---------

.. autosummary::
    :template: function.rst
    :nosignatures:
    :toctree:

    maximise_contextual_fraction
    local_measurement
    dual_rail_measurement

Message passing
---------------

.. autosummary::
    :template: class.rst
    :nosignatures:
    :toctree:

    MessagePassingScenario

Examples
--------

The Popescu-Rohrlich box is strongly contextual, i.e. no global assignment
is consistent with its support:

>>> chsh = Scenario(
...     ("a0", "a1", "b0", "b1"),
...     (("a0", "b0"), ("a0", "b1"), ("a1", "b0"), ("a1", "b1")))
>>> pr_box = EmpiricalModel(chsh, {
...     ("a0", "b0"): {(0, 0): .5, (1, 1): .5},
...     ("a0", "b1"): {(0, 0): .5, (1, 1): .5},
...     ("a1", "b0"): {(0, 0): .5, (1, 1): .5},
...     ("a1", "b1"): {(0, 1): .5, (1, 0): .5}})
>>> assert pr_box.is_no_signalling() and pr_box.is_strongly_contextual()
>>> pr_box.contextual_fraction()
1.0

Mixing it with its deterministic part lowers the contextual fraction, the
Bell inequality read off the dual being violated by exactly that much:

>>> mixed = .5 * pr_box + .5 * chsh.deterministic((0, 0, 0, 0))
>>> round(mixed.contextual_fraction(), 6)
0.5
>>> inequality = mixed.bell_inequality()
>>> round(inequality.violation(mixed), 6)
0.5
>>> assert inequality.violation(chsh.deterministic((1, 0, 1, 0))) <= 0

The empirical model of an optyx diagram with a classical input for the
setting and a classical output for the outcome of each party, here a
Bell test on the singlet at the Tsirelson bound:

>>> from optyx import bit, qubit, Diagram
>>> from optyx.qubits import Z, H, Scalar, Measure
>>> from optyx.classical import BitControlledGate
>>> alice = BitControlledGate(Z(1, 1, .25)) >> H() >> Measure(1)
>>> bob = Diagram.swap(qubit, bit) >> bit @ Z(1, 1, -.125)\\
...     >> BitControlledGate(Z(1, 1, .25)) >> H() >> Measure(1)
>>> bell_test = bit @ Scalar(.5 ** .5) @ Z(0, 2) @ bit >> alice @ bob
>>> tsirelson = EmpiricalModel.from_diagram(bell_test)
>>> tsirelson.scenario.measurements
((0, 0), (0, 1), (1, 0), (1, 1))
>>> tsirelson.scenario.contexts[0]
((0, 0), (1, 0))
>>> assert tsirelson.is_no_signalling()
>>> round(tsirelson.contextual_fraction(), 6)  # sqrt(2) - 1
0.414214

The contextual fraction is differentiable through the dual program, so
gradient ascent finds the measurement angles maximising it: here the
settings of each party are two free phases in the X-Y plane, and from
a contextual starting point the ascent reaches the Tsirelson bound, with
each party's settings a quarter turn apart.

>>> def bell_test(params):
...     a0, a1, b0, b1 = params
...     alice = bit @ Z(1, 1, a0) >> BitControlledGate(Z(1, 1, a1 - a0))\\
...         >> H() >> Measure(1)
...     bob = Diagram.swap(qubit, bit) >> bit @ Z(1, 1, b0)\\
...         >> BitControlledGate(Z(1, 1, b1 - b0)) >> H() >> Measure(1)
...     return bit @ Scalar(.5 ** .5) @ Z(0, 2) @ bit >> alice @ bob
>>> start = EmpiricalModel.from_diagram(bell_test([.3, .05, .17, .42]))
>>> round(start.contextual_fraction(), 3)
0.17
>>> params, best = maximise_contextual_fraction(
...     bell_test, [.3, .05, .17, .42])
>>> round(best.contextual_fraction(), 3)
0.414
>>> assert np.isclose(abs(params[1] - params[0]), .25)
>>> assert np.isclose(abs(params[3] - params[2]), .25)

A combinatorial map of recurrent channels,
:class:`optyx.interaction.CMap`, has a :class:`MessagePassingScenario`:
its nodes are the parties, each measuring the system it predicts at the
last tick with a local measurement chosen by a bit, and the messages
passed along the edges at the earlier ticks are what correlates them.
The measurements maximising the contextual fraction are learned by
gradient ascent from the most contextual of a few random starts, since
the fraction has no gradient inside the noncontextual polytope. Here a
source node sends half of a fresh Bell pair down its edge at every tick
and predicts the half it kept a tick later, when the sink predicts the
half it received, so the two predictions are a Bell pair from the second
tick on and the learned measurements approach the Tsirelson bound:

>>> from optyx import Discard
>>> from optyx.channel import Ty
>>> from optyx.qubits import Ket
>>> from optyx.interaction import Box, CMap
>>> source = Box("source", Ty(), qubit,
...     Discard(qubit) @ qubit >> Scalar(.5 ** .5) @ Z(0, 2) @ qubit,
...     memory=qubit, prediction=qubit)
>>> sink = Box("sink", qubit, Ty(), Ket(0) @ qubit, prediction=qubit)
>>> cmap = CMap([source, sink], [((0, 0), (1, 0))])
>>> scenario = MessagePassingScenario(cmap, Ket(0) ** 3, n_steps=1)
>>> tsirelson = scenario.model([0, 0, .25, .25, .25, .125, 0, .25])
>>> round(tsirelson.contextual_fraction(), 6)
0.414214
>>> angles, learned = scenario.learn(restarts=8, steps=5)
>>> assert learned.contextual_fraction() > .3

The scenario does not depend on the physical system. The same map on
dual-rail photons, where a node predicts a pair of modes and measures it
with a phase, a beam splitter and two photon counters, has pairs of
photon numbers as outcomes and the same contextual fraction, at the
angles putting both settings on a balanced beam splitter a quarter turn
of phase apart:

>>> from optyx import qmode
>>> from optyx.photonic import Create
>>> pair = qmode ** 2
>>> bell = (Scalar(.5 ** .5) @ Z(0, 2)).to_dual_rail()
>>> source = Box("source", Ty(), pair,
...     Discard(pair) @ pair >> bell @ pair, memory=pair, prediction=pair)
>>> sink = Box("sink", pair, Ty(), Create(0, 0) @ pair, prediction=pair)
>>> photonic = MessagePassingScenario(
...     CMap([source, sink], [((0, 0), (1, 0)), ((0, 1), (1, 1))]),
...     Create(0, 0, 0, 0, 0, 0), n_steps=1)
>>> counts = photonic.model([0, 0, .25, 0, .125, 0, .25, 0])
>>> counts.scenario.outcomes[(0, 0)]
((0, 1), (1, 0))
>>> round(counts.contextual_fraction(), 6)
0.414214
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from itertools import product

import numpy as np
from scipy.optimize import linprog

from optyx import channel
from optyx.channel import bit, qubit, qmode, Discard, Ty
from optyx.classical import Bit, Digit, BitControlledGate
from optyx.core.backends import StateType
from optyx.photonic import BBS, Phase, NumberResolvingMeasurement
from optyx.qubits import Measure, X, Z

Measurement = Hashable
Context = tuple[Measurement, ...]
Section = tuple[Hashable, ...]
Distribution = Mapping[Section, float]


class Scenario:
    """
    A measurement scenario :math:`(X, \\mathcal{M}, O)`.

    Parameters:
        measurements : The measurements :math:`X`, in a fixed order.
        contexts : The cover :math:`\\mathcal{M}`, a sequence of tuples of
            measurements whose union is :math:`X`.
        outcomes : The outcomes of each measurement, as one tuple shared by
            all of them or a mapping from measurements to tuples.

    Example
    -------
    >>> chsh = Scenario(("a0", "a1", "b0", "b1"), (("a0", "b0"),
    ...     ("a0", "b1"), ("a1", "b0"), ("a1", "b1")))
    >>> assert eval(repr(chsh)) == chsh
    >>> chsh.global_assignments()[:3]
    ((0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 1, 0))
    >>> chsh.restrict((0, 1, 1, 0), ("a1", "b1"))
    (1, 0)
    >>> chsh.incidence_matrix().shape
    (16, 16)
    """
    def __init__(self, measurements: Iterable[Measurement],
                 contexts: Iterable[Iterable[Measurement]],
                 outcomes: Sequence | Mapping[Measurement, Sequence] = (0, 1)
                 ) -> None:
        self.measurements = tuple(measurements)
        self.contexts = tuple(tuple(context) for context in contexts)
        self.outcomes = {
            x: tuple(outcomes[x] if isinstance(outcomes, Mapping)
                     else outcomes) for x in self.measurements}
        self.index = {x: i for i, x in enumerate(self.measurements)}
        covered = {x for context in self.contexts for x in context}
        if covered != set(self.measurements):
            raise ValueError(
                f"The contexts {self.contexts} do not cover the measurements "
                f"{self.measurements}.")

    def __repr__(self) -> str:
        outcomes = set(self.outcomes.values())
        outcomes = next(iter(outcomes)) if len(outcomes) == 1\
            else self.outcomes
        return f"Scenario({self.measurements}, {self.contexts}, "\
               f"outcomes={outcomes})"

    def __eq__(self, other) -> bool:
        return isinstance(other, Scenario) and (
            self.measurements, self.contexts, self.outcomes
        ) == (other.measurements, other.contexts, other.outcomes)

    def __hash__(self) -> int:
        return hash(repr(self))

    def sections(self, context: Iterable[Measurement]) -> tuple[Section, ...]:
        """ The joint outcomes :math:`\\prod_{x \\in C} O_x` of a context. """
        return tuple(product(*(self.outcomes[x] for x in context)))

    def global_assignments(self) -> tuple[Section, ...]:
        """ The joint outcomes of all the measurements. """
        return self.sections(self.measurements)

    def restrict(self, assignment: Section,
                 context: Iterable[Measurement]) -> Section:
        """ The restriction of a global assignment to a context. """
        return tuple(assignment[self.index[x]] for x in context)

    def rows(self) -> tuple[tuple[Context, Section], ...]:
        """
        The pairs of a context and one of its sections, in the order
        indexing the rows of the incidence matrix and the vector of a model.
        """
        return tuple((context, section)
                     for context in self.contexts
                     for section in self.sections(context))

    def incidence_matrix(self) -> np.ndarray:
        """
        The matrix :math:`\\mathbf{M}` with a row for each pair of a context
        :math:`C` and a section :math:`s`, a column for each global
        assignment :math:`g` and :math:`\\mathbf{M}[(C, s), g] = 1`
        when :math:`g` restricts to :math:`s` on :math:`C`.
        """
        assignments = self.global_assignments()
        return np.array([[
            float(self.restrict(assignment, context) == section)
            for assignment in assignments]
            for context, section in self.rows()])

    def deterministic(self, assignment: Section) -> EmpiricalModel:
        """ The noncontextual model of one global assignment. """
        return EmpiricalModel(self, {
            context: {self.restrict(assignment, context): 1.}
            for context in self.contexts})


class EmpiricalModel:
    """
    A probability distribution on the sections of each context of a
    scenario.

    Parameters:
        scenario : The measurement scenario.
        distributions : A mapping from each context to a mapping from its
            sections to their probabilities, the missing sections having
            probability zero.

    Example
    -------
    >>> scenario = Scenario("xy", (("x", "y"), ))
    >>> model = EmpiricalModel(scenario, {("x", "y"): {(0, 1): 1.}})
    >>> assert eval(repr(model)) == model == scenario.deterministic((0, 1))
    >>> model.marginal(("x", "y"), ("y", ))
    {(0,): 0.0, (1,): 1.0}
    """
    def __init__(self, scenario: Scenario,
                 distributions: Mapping[Context, Distribution]) -> None:
        self.scenario = scenario
        self.distributions = {
            context: {
                section: float(distributions.get(context, {}).get(section, 0.))
                for section in scenario.sections(context)}
            for context in scenario.contexts}
        for context, distribution in distributions.items():
            if context not in self.distributions:
                raise ValueError(f"{context} is not a context of {scenario}.")
            if set(distribution) - set(self.distributions[context]):
                raise ValueError(
                    f"{distribution} has keys which are not sections of "
                    f"the context {context}.")

    def __repr__(self) -> str:
        return f"EmpiricalModel({self.scenario!r}, {self.distributions})"

    def __eq__(self, other) -> bool:
        return isinstance(other, EmpiricalModel)\
            and self.scenario == other.scenario\
            and np.allclose(self.vector(), other.vector())

    def __hash__(self) -> int:
        return hash(self.scenario)

    def __add__(self, other: EmpiricalModel) -> EmpiricalModel:
        return self.map(lambda context, section:
                        self.distributions[context][section]
                        + other.distributions[context][section])

    def __rmul__(self, scalar: float) -> EmpiricalModel:
        return self.map(lambda context, section:
                        scalar * self.distributions[context][section])

    def map(self, function: Callable[[Context, Section], float]
            ) -> EmpiricalModel:
        """ The model with a function of each context and section as data. """
        return EmpiricalModel(self.scenario, {
            context: {section: function(context, section)
                      for section in distribution}
            for context, distribution in self.distributions.items()})

    def is_normalised(self, tol: float = 1e-9) -> bool:
        """ Whether each distribution is non-negative and sums to one. """
        return all(
            min(distribution.values()) >= -tol
            and abs(sum(distribution.values()) - 1.) <= tol
            for distribution in self.distributions.values())

    def vector(self) -> np.ndarray:
        """ The probabilities :math:`v_e` in the order of the rows. """
        return np.array([self.distributions[context][section]
                         for context, section in self.scenario.rows()])

    def marginal(self, context: Context,
                 measurements: Iterable[Measurement]) -> dict[Section, float]:
        """ The marginal of a context on some of its measurements. """
        context, measurements = tuple(context), tuple(measurements)
        index = [context.index(x) for x in measurements]
        marginal = {section: 0. for section in
                    self.scenario.sections(measurements)}
        for section, probability in self.distributions[context].items():
            marginal[tuple(section[i] for i in index)] += probability
        return marginal

    def is_no_signalling(self, tol: float = 1e-9) -> bool:
        """
        Whether the distributions of two contexts agree on the marginal
        of their intersection.
        """
        contexts = self.scenario.contexts
        for i, left in enumerate(contexts):
            for right in contexts[i + 1:]:
                common = tuple(x for x in left if x in right)
                if not np.allclose(
                        list(self.marginal(left, common).values()),
                        list(self.marginal(right, common).values()),
                        atol=tol):
                    return False
        return True

    def linear_program(self):
        """
        The solution of :math:`\\max \\; \\mathbf{1} \\cdot b` subject to
        :math:`\\mathbf{M} b \\leq v_e` and :math:`b \\geq 0`, with its
        value the noncontextual fraction, its argument :code:`x` the
        subprobability on global assignments and the marginals of the
        inequality constraints the solution :math:`-y` of the dual.
        """
        matrix = self.scenario.incidence_matrix()
        result = linprog(
            -np.ones(matrix.shape[1]), A_ub=matrix, b_ub=self.vector(),
            bounds=(0, None), method="highs")
        if not result.success:
            raise ValueError(
                f"The linear program failed: {result.message}")
        return result

    def noncontextual_fraction(self) -> float:
        """ The largest weight of a noncontextual model in the model. """
        return float(min(1., max(0., -self.linear_program().fun)))

    def contextual_fraction(self) -> float:
        """ One minus the noncontextual fraction. """
        return 1. - self.noncontextual_fraction()

    def is_contextual(self, tol: float = 1e-9) -> bool:
        """ Whether the model is not the marginal of a global distribution. """
        return self.contextual_fraction() > tol

    def is_strongly_contextual(self, tol: float = 1e-9) -> bool:
        """ Whether the contextual fraction is one. """
        return self.contextual_fraction() > 1 - tol

    def contextual_fraction_gradient(self) -> np.ndarray:
        """
        The gradient :math:`-y` of the contextual fraction with respect to
        the probabilities :math:`v_e`, in the order of the rows, for the
        solution :math:`y` of the dual program: the value of a linear
        program is piecewise linear in its right-hand side and its gradient
        is the dual solution, which the solver returns as the marginals of
        the inequality constraints. It is a subgradient where the dual is
        not unique, e.g. on the boundary of the noncontextual polytope, and
        constant on each context inside it, where the fraction is zero.
        """
        return np.array(self.linear_program().ineqlin.marginals)

    def bell_inequality(self) -> BellInequality:
        """
        The inequality :math:`\\langle \\mathbf{1} - y, e \\rangle \\leq
        |\\mathcal{M}| - 1` for the solution :math:`y` of the dual program,
        satisfied by every noncontextual model and violated by this one by
        its contextual fraction.
        """
        dual = -self.linear_program().ineqlin.marginals
        coefficients = {context: {} for context in self.scenario.contexts}
        for (context, section), y in zip(self.scenario.rows(), dual):
            coefficients[context][section] = 1. - float(y)
        return BellInequality(
            self.scenario, coefficients, len(self.scenario.contexts) - 1.)

    @staticmethod
    def conditionals(diagram: channel.Diagram, settings: Sequence[int],
                     backend=None) -> dict[tuple, dict[Section, float]]:
        """
        The distribution of the classical outputs of a diagram for each
        choice of its classical inputs, the choices ranging below
        :code:`settings`. The diagram is contracted once with its inputs
        open when the array has one entry per choice, which is the case
        for bits, and once per choice otherwise, e.g. for modes, which the
        open contraction truncates. Probabilities below :code:`1e-12` are
        numerical noise and left out.
        """
        choices = list(product(*(range(n) for n in settings)))
        result = diagram.eval(backend)
        array = np.array(result.tensor.array)
        if array.shape[:len(settings)] == tuple(settings):
            if result.state_type is StateType.AMP:
                array = abs(array) ** 2
            table = {choice: dict(np.ndenumerate(array[choice]))
                     for choice in choices}
        else:
            table = {choice: (channel.Diagram.tensor(*(
                Bit(x) if ob.name == "bit" else Digit(x)
                for x, ob in zip(choice, diagram.dom.inside)))
                >> diagram).eval(backend).prob_dist() for choice in choices}
        return {choice: {
            tuple(int(i) for i in section): float(np.real(probability))
            / float(np.real(sum(table[choice].values())))
            for section, probability in table[choice].items()
            if abs(probability) > 1e-12} for choice in choices}

    @classmethod
    def from_diagram(cls, diagram: channel.Diagram,
                     settings: Sequence[int] = None,
                     outcomes: Sequence[Sequence] = None,
                     backend=None, outputs: Sequence[int] = None
                     ) -> EmpiricalModel:
        """
        The empirical model of a diagram with one classical input choosing
        the measurement of each party and classical outputs giving their
        outcomes.

        Parameters:
            diagram : A channel diagram with classical domain and codomain.
            settings : The number of measurements of each party, two for
                each by default.
            outcomes : The outcomes of each party, by default those it
                gives with a nonzero probability for some settings.
            backend : The backend evaluating the diagram.
            outputs : The number of output wires of each party, one for
                each by default, the outcome of a party being the value of
                its wire or the tuple of the values of its wires.

        The measurements are the pairs :code:`(party, setting)`, the
        contexts the choices of one setting for every party and the
        sections the tuples of outcomes in the order of the parties.
        """
        dom, cod = diagram.dom.inside, diagram.cod.inside
        outputs = [1] * len(dom) if outputs is None else list(outputs)
        if sum(outputs) != len(cod) or not all(
                ob.is_classical for ob in dom + cod):
            raise ValueError(
                f"The domain {diagram.dom} and codomain {diagram.cod} must "
                f"be classical with {outputs} outputs for each input.")
        settings = tuple(2 for _ in dom) if settings is None else settings
        splits = list(zip(np.cumsum([0] + outputs), np.cumsum(outputs)))
        distributions = {tuple(enumerate(choice)): {
            tuple(section[i] if j == i + 1 else section[i:j]
                  for i, j in splits): probability
            for section, probability in distribution.items()}
            for choice, distribution
            in cls.conditionals(diagram, settings, backend).items()}
        if outcomes is None:
            outcomes = [sorted({section[i] for distribution in
                                distributions.values()
                                for section in distribution})
                        for i in range(len(dom))]
        scenario = Scenario(
            measurements=[(party, x) for party, n in enumerate(settings)
                          for x in range(n)],
            contexts=list(distributions),
            outcomes={(party, x): tuple(outcomes[party])
                      for party, n in enumerate(settings) for x in range(n)})
        return cls(scenario, distributions)


def maximise_contextual_fraction(
        setup: Callable[[np.ndarray], channel.Diagram],
        params: Sequence[float], steps: int = 10, learning_rate: float = .05,
        difference: float = 1e-3, **kwargs
        ) -> tuple[np.ndarray, EmpiricalModel]:
    """
    Gradient ascent on the contextual fraction of the empirical model of a
    parametrised setup, from a starting point.

    Parameters:
        setup : A function from an array of parameters to a diagram that
            :meth:`EmpiricalModel.from_diagram` accepts.
        params : The starting parameters.
        steps : The number of gradients computed.
        learning_rate : The initial step size, doubled after a step that
            improves the fraction and halved, up to a fixed number of
            times, after one that does not.
        difference : The step of the forward differences approximating the
            derivatives of the probabilities with respect to the parameters.
        kwargs : Passed to :meth:`EmpiricalModel.from_diagram`, with the
            outcomes of the starting point fixed throughout.

    The gradient of the fraction with respect to the probabilities is
    :meth:`EmpiricalModel.contextual_fraction_gradient`, exact from the
    dual program, and the chain rule contracts it with the derivatives of
    the probabilities with respect to the parameters, which are the only
    approximation. The fraction is piecewise linear in the probabilities,
    so a fixed step overshoots its kinks, which the backtracking is for;
    it is zero on the noncontextual polytope, so the ascent only moves
    from a contextual starting point.
    """
    params = np.array(params, dtype=float)
    model = EmpiricalModel.from_diagram(setup(params), **kwargs)
    kwargs["outcomes"] = [
        model.scenario.outcomes[(party, 0)]
        for party in range(len(model.scenario.contexts[0]))]
    for _ in range(steps):
        gradient = model.contextual_fraction_gradient()
        jacobian = np.array([(EmpiricalModel.from_diagram(
            setup(params + difference * direction), **kwargs).vector()
            - model.vector()) / difference
            for direction in np.eye(len(params))])
        direction = jacobian @ gradient
        for _ in range(8):
            candidate = params + learning_rate * direction
            improved = EmpiricalModel.from_diagram(setup(candidate), **kwargs)
            if improved.contextual_fraction() > model.contextual_fraction():
                params, model = candidate, improved
                learning_rate = 2 * learning_rate
                break
            learning_rate = learning_rate / 2
    return params, model


def local_measurement(angles: Sequence[float]) -> channel.Diagram:
    """
    A qubit measurement in one of two bases chosen by a bit, from
    :code:`bit @ qubit` to :code:`bit`, parametrised by four angles in
    turns: the first basis is the computational one rotated by
    :code:`Z(a0)` then :code:`X(b0)`, the second rotated further by
    :code:`Z(a1)` then :code:`X(b1)`. Each basis ranges over the Bloch
    sphere, the first with :code:`b0` as polar and :code:`a0` as
    azimuthal angle.

    >>> assert local_measurement([0, 0, 0, .25]).dom == bit @ qubit
    """
    a0, b0, a1, b1 = angles
    return bit @ (Z(1, 1, a0) >> X(1, 1, b0))\
        >> BitControlledGate(Z(1, 1, a1) >> X(1, 1, b1)) >> Measure(1)


def dual_rail_measurement(angles: Sequence[float]) -> channel.Diagram:
    """
    The photonic counterpart of :func:`local_measurement` on a dual-rail
    qubit, from :code:`bit @ qmode @ qmode` to :code:`mode @ mode`: a
    phase on the first mode then a biased beam splitter, the second
    setting adding a further phase and beam splitter, before counting the
    photons in both modes, so that the outcomes are pairs of photon
    numbers rather than bits.

    >>> from optyx import mode
    >>> assert dual_rail_measurement([0, 0, 0, .25]).cod == mode @ mode
    """
    a0, b0, a1, b1 = angles
    return bit @ (Phase(a0) @ qmode >> BBS(b0))\
        >> BitControlledGate(Phase(a1) @ qmode >> BBS(b1))\
        >> NumberResolvingMeasurement(2)


class MessagePassingScenario:
    """
    The measurement scenario of a combinatorial map of recurrent
    channels, :class:`optyx.interaction.CMap`, whatever the physical
    systems on its wires.

    The parties are the nodes with a prediction, the system a node writes
    to the environment at every tick. Each node has a family of local
    measurements of its prediction, chosen by one classical wire, and a
    context is a choice of one measurement for every node: the
    predictions of a tick are written at once, so any choice can be
    measured together. The empirical model of a choice of measurements is
    the distribution of their outcomes at the last of :code:`n_steps + 1`
    ticks, every other output being discarded. The nodes only interact
    through the messages passed along the edges at the earlier ticks, so
    the model is contextual only through them.

    Parameters:
        cmap : The map.
        initial_state : A state of :code:`cmap.memory` before the first
            tick.
        n_steps : The number of unrollings.
        input_state : A state of :code:`cmap.dom` prepared at every tick,
            required when the map has a boundary.
        measurements : For each node with a prediction, a function from
            an array of angles to a channel from :code:`S @ prediction`
            to a classical type, :code:`S` the single classical wire
            choosing the measurement. By default,
            :func:`local_measurement` for a qubit and
            :func:`dual_rail_measurement` for a pair of modes.
        sizes : The number of angles of each measurement, four for the
            defaults.
        settings : The number of measurements of each node, two for each
            by default, which is what a bit chooses between.

    >>> from optyx.qubits import Ket
    >>> from optyx.interaction import Box, CMap
    >>> cell = Box("cell", qubit, qubit, Z(2, 3), prediction=qubit)
    >>> ring = CMap([cell, cell], [((0, 1), (1, 0)), ((1, 1), (0, 0))])
    >>> scenario = MessagePassingScenario(ring, Ket(0) ** 4)
    >>> assert scenario.diagram(np.zeros(8)).dom == bit ** 2
    >>> model = scenario.model(np.zeros(8))
    >>> assert model.scenario.contexts[0] == ((0, 0), (1, 0))
    """
    def __init__(self, cmap, initial_state: channel.Diagram,
                 n_steps: int = 0, input_state: channel.Diagram = None, *,
                 measurements: Sequence[Callable] = None,
                 sizes: Sequence[int] = None,
                 settings: Sequence[int] = None) -> None:
        self.cmap, self.n_steps = cmap, n_steps
        self.nodes = [box for box in cmap.boxes if box.prediction]
        if measurements is None:
            measurements = [self.default_measurement(box.prediction)
                            for box in self.nodes]
            sizes = [4] * len(self.nodes)
        if input_state is None:
            if cmap.dom:
                raise ValueError(
                    "input_state is required for a map with a boundary.")
            input_state = channel.Diagram.id()
        self.measurements, self.sizes = list(measurements), list(sizes)
        self.settings = [2] * len(self.nodes)\
            if settings is None else list(settings)
        tick = cmap.cod @ cmap.prediction
        self.state = channel.Diagram.id().tensor(
            *[input_state] * (n_steps + 1)) @ initial_state\
            >> cmap.unroll(n_steps)\
            >> Discard(tick ** n_steps @ cmap.cod)\
            @ channel.Diagram.id(cmap.prediction)

    @staticmethod
    def default_measurement(system: Ty) -> Callable:
        """ The default family of measurements of a system. """
        if system == qubit:
            return local_measurement
        if system == qmode ** 2:
            return dual_rail_measurement
        raise ValueError(f"There is no default measurement of {system}.")

    def measured(self, angles: Sequence[float]) -> list[channel.Diagram]:
        """ The measurement of each node, for a flat array of angles. """
        angles = np.split(np.asarray(angles, dtype=float),
                          np.cumsum(self.sizes)[:-1])
        return [measurement(angle)
                for measurement, angle in zip(self.measurements, angles)]

    def diagram(self, angles: Sequence[float]) -> channel.Diagram:
        """
        The diagram from the settings of every node to their outcomes,
        for a flat array of angles.
        """
        measured = self.measured(angles)
        settings = Ty().tensor(*[m.dom[:1] for m in measured])
        offsets = np.cumsum([len(settings)] + [
            len(box.prediction) for box in self.nodes])
        permutation = [wire for i in range(len(self.nodes)) for wire
                       in [i] + list(range(offsets[i], offsets[i + 1]))]
        return settings @ self.state >> channel.Diagram.permutation(
            permutation, settings @ self.cmap.prediction)\
            >> channel.Diagram.id().tensor(*measured)

    def model(self, angles: Sequence[float], backend=None) -> EmpiricalModel:
        """ The empirical model of a choice of angles. """
        return EmpiricalModel.from_diagram(
            self.diagram(angles), settings=self.settings, backend=backend,
            outputs=[len(m.cod) for m in self.measured(angles)])

    def learn(self, restarts: int = 16, seed: int = 0, backend=None,
              **kwargs) -> tuple[np.ndarray, EmpiricalModel]:
        """
        The angles maximising the contextual fraction: the most contextual
        of :code:`restarts` uniformly random sets of angles is the starting
        point of :func:`maximise_contextual_fraction`, which takes the
        remaining keyword arguments. The fraction is zero on the
        noncontextual polytope and the ascent does not move there, which
        is what the restarts are for; when none is contextual, the first
        is returned as it is.
        """
        starts = np.random.default_rng(seed).uniform(
            0, 1, (restarts, sum(self.sizes)))
        fractions = [self.model(start, backend).contextual_fraction()
                     for start in starts]
        start = starts[int(np.argmax(fractions))]
        return maximise_contextual_fraction(
            self.diagram, start, settings=self.settings, backend=backend,
            outputs=[len(m.cod) for m in self.measured(start)], **kwargs)


class BellInequality:
    """
    An inequality :math:`\\langle a, e \\rangle \\leq R` on the empirical
    models of a scenario, with a coefficient for each section of each
    context.

    Example
    -------
    >>> scenario = Scenario("xy", (("x", ), ("y", )))
    >>> chsh = BellInequality(scenario, {
    ...     ("x", ): {(0, ): 1., (1, ): 0.},
    ...     ("y", ): {(0, ): 1., (1, ): 0.}}, bound=1.)
    >>> assert eval(repr(chsh)) == chsh
    >>> chsh.violation(scenario.deterministic((0, 0)))
    1.0
    """
    def __init__(self, scenario: Scenario,
                 coefficients: Mapping[Context, Mapping[Section, float]],
                 bound: float) -> None:
        self.scenario, self.bound = scenario, float(bound)
        self.coefficients = {
            context: {
                section: float(coefficients.get(context, {}).get(section, 0.))
                for section in scenario.sections(context)}
            for context in scenario.contexts}

    def __repr__(self) -> str:
        return f"BellInequality({self.scenario!r}, {self.coefficients}, "\
               f"bound={self.bound})"

    def __eq__(self, other) -> bool:
        return isinstance(other, BellInequality)\
            and (self.scenario, self.bound) == (other.scenario, other.bound)\
            and np.allclose(self.vector(), other.vector())

    def __hash__(self) -> int:
        return hash((self.scenario, self.bound))

    def vector(self) -> np.ndarray:
        """ The coefficients :math:`a` in the order of the rows. """
        return np.array([self.coefficients[context][section]
                         for context, section in self.scenario.rows()])

    def value(self, model: EmpiricalModel) -> float:
        """ The left-hand side :math:`\\langle a, e \\rangle` on a model. """
        return float(self.vector() @ model.vector())

    def violation(self, model: EmpiricalModel) -> float:
        """ The amount :math:`\\langle a, e \\rangle - R` by which a model
        exceeds the bound, negative when it does not. """
        return self.value(model) - self.bound
