import numpy as np
import pytest

from optyx import bit, mode, qubit, Diagram, Encode
from optyx.classical import BitControlledGate
from optyx.contextuality import BellInequality, EmpiricalModel, Scenario
from optyx.photonic import NumberResolvingMeasurement
from optyx.qubits import H, Measure, Scalar, Z


chsh = Scenario(("a0", "a1", "b0", "b1"), (
    ("a0", "b0"), ("a0", "b1"), ("a1", "b0"), ("a1", "b1")))
triangle = Scenario("xyz", (("x", "y"), ("y", "z"), ("z", "x")))

pr_box = EmpiricalModel(chsh, {
    context: {(0, 0): .5, (1, 1): .5} if context != ("a1", "b1")
    else {(0, 1): .5, (1, 0): .5} for context in chsh.contexts})
specker = EmpiricalModel(triangle, {
    context: {(0, 1): .5, (1, 0): .5} for context in triangle.contexts})
signalling = EmpiricalModel(chsh, {
    ("a0", "b0"): {(0, 0): 1.}, ("a0", "b1"): {(1, 1): 1.},
    ("a1", "b0"): {(0, 0): 1.}, ("a1", "b1"): {(0, 0): 1.}})


def test_Scenario():
    assert eval(repr(chsh)) == chsh != triangle
    assert eval(repr(Scenario("xy", (("x", ), ("y", )),
                              {"x": (0, 1), "y": "abc"}))).outcomes == {
        "x": (0, 1), "y": ("a", "b", "c")}
    assert len(chsh.global_assignments()) == len(chsh.rows()) == 16
    assert chsh.incidence_matrix().sum(axis=0).tolist() == 16 * [4.]
    with pytest.raises(ValueError):
        Scenario("xyz", (("x", "y"), ))


def test_EmpiricalModel():
    assert eval(repr(pr_box)) == pr_box != specker
    assert pr_box.is_normalised() and not (2. * pr_box).is_normalised()
    assert (.5 * pr_box + .5 * pr_box) == pr_box
    assert pr_box.marginal(("a0", "b0"), ("b0", "a0")) == {
        (0, 0): .5, (0, 1): 0., (1, 0): 0., (1, 1): .5}
    assert pr_box.is_no_signalling() and not signalling.is_no_signalling()
    with pytest.raises(ValueError):
        EmpiricalModel(chsh, {("a0", "a1"): {(0, 0): 1.}})
    with pytest.raises(ValueError):
        EmpiricalModel(chsh, {("a0", "b0"): {(0, 0, 0): 1.}})


def test_contextual_fraction():
    for assignment in chsh.global_assignments():
        assert not chsh.deterministic(assignment).is_contextual()
    assert pr_box.is_strongly_contextual()
    assert specker.is_strongly_contextual()
    noise = 1 / 16 * sum(
        (chsh.deterministic(g) for g in chsh.global_assignments()[1:]),
        chsh.deterministic(chsh.global_assignments()[0]))
    for weight in (0., .3, .5, .7, 1.):
        isotropic = weight * pr_box + (1 - weight) * noise
        assert np.isclose(
            isotropic.contextual_fraction(), max(0., 2 * weight - 1))
        assert isotropic.is_contextual() == (weight > .5)


def test_BellInequality():
    inequality = BellInequality(chsh, {
        ("a0", "b0"): {(0, 0): 1.}}, bound=.5)
    assert eval(repr(inequality)) == inequality
    assert inequality.violation(pr_box) == 0.
    noise = 1 / 16 * sum(
        (chsh.deterministic(g) for g in chsh.global_assignments()[1:]),
        chsh.deterministic(chsh.global_assignments()[0]))
    for model in (pr_box, .6 * pr_box + .4 * noise, specker):
        inequality = model.bell_inequality()
        assert np.isclose(
            inequality.violation(model), model.contextual_fraction())
        for assignment in model.scenario.global_assignments():
            deterministic = model.scenario.deterministic(assignment)
            assert inequality.violation(deterministic) <= 1e-9


def test_from_diagram():
    alice = BitControlledGate(Z(1, 1, .25)) >> H() >> Measure(1)
    bob = Diagram.swap(qubit, bit) >> bit @ Z(1, 1, -.125)\
        >> BitControlledGate(Z(1, 1, .25)) >> H() >> Measure(1)
    bell_test = bit @ Scalar(.5 ** .5) @ Z(0, 2) @ bit >> alice @ bob
    tsirelson = EmpiricalModel.from_diagram(bell_test)
    assert tsirelson.is_no_signalling()
    assert np.isclose(tsirelson.contextual_fraction(), 2 ** .5 - 1)
    assert tsirelson == EmpiricalModel.from_diagram(
        bell_test, outcomes=((0, 1), (0, 1)))
    detector = Encode(mode) >> NumberResolvingMeasurement(1)
    counter = EmpiricalModel.from_diagram(detector @ detector, settings=(3, 1))
    assert counter.scenario.outcomes == {
        (0, 0): (0, 1, 2), (0, 1): (0, 1, 2), (0, 2): (0, 1, 2), (1, 0): (0, )}
    assert counter.distributions[((0, 2), (1, 0))] == {
        (0, 0): 0., (1, 0): 0., (2, 0): 1.}
    assert not counter.is_contextual()
    with pytest.raises(ValueError):
        EmpiricalModel.from_diagram(Measure(1))
    with pytest.raises(ValueError):
        EmpiricalModel.from_diagram(detector @ Encode(mode))
