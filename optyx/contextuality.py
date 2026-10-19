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
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from itertools import product

import numpy as np
from scipy.optimize import linprog

from optyx import channel
from optyx.classical import Bit, Digit

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

    @classmethod
    def from_diagram(cls, diagram: channel.Diagram,
                     settings: Sequence[int] = None,
                     outcomes: Sequence[Sequence] = None,
                     backend=None) -> EmpiricalModel:
        """
        The empirical model of a diagram with one classical input and one
        classical output for each party, the input choosing the party's
        measurement and the output giving its outcome.

        Parameters:
            diagram : A channel diagram with classical domain and codomain
                of the same length.
            settings : The number of measurements of each party, two for
                each by default.
            outcomes : The outcomes of each party, by default those it
                gives with a nonzero probability for some settings.
            backend : The backend evaluating the diagram.

        The measurements are the pairs :code:`(party, setting)`, the
        contexts the choices of one setting for every party and the
        sections the tuples of outcomes in the order of the parties.
        """
        dom, cod = diagram.dom.inside, diagram.cod.inside
        if len(dom) != len(cod) or not all(
                ob.is_classical for ob in dom + cod):
            raise ValueError(
                "The domain and codomain must be classical types of the "
                f"same length, got {diagram.dom} and {diagram.cod}.")
        settings = tuple(2 for _ in dom) if settings is None else settings
        distributions = {}
        for choice in product(*(range(n) for n in settings)):
            state = channel.Diagram.tensor(*(
                Bit(x) if ob.name == "bit" else Digit(x)
                for x, ob in zip(choice, dom)))
            distributions[tuple(enumerate(choice))] = {
                tuple(int(i) for i in section): float(probability)
                for section, probability
                in (state >> diagram).eval(backend).prob_dist().items()}
        if outcomes is None:
            outcomes = [sorted({section[i] for distribution in
                                distributions.values()
                                for section, p in distribution.items()
                                if p > 0}) for i in range(len(cod))]
        scenario = Scenario(
            measurements=[(party, x) for party, n in enumerate(settings)
                          for x in range(n)],
            contexts=list(distributions),
            outcomes={(party, x): tuple(outcomes[party])
                      for party, n in enumerate(settings) for x in range(n)})
        return cls(scenario, distributions)


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
