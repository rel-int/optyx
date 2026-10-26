import numpy as np
import pytest

from optyx import bit, mode, qmode, qubit, Diagram, Discard, Encode
from optyx.channel import Ty
from optyx.classical import BitControlledGate
from optyx.contextuality import (
    BellInequality, EmpiricalModel, MessagePassingScenario, Scenario,
    local_measurement, maximise_contextual_fraction)
from optyx.interaction import Box, CMap
from optyx.photonic import Create, NumberResolvingMeasurement
from optyx.qubits import H, Ket, Measure, Scalar, Z


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


def test_maximise_contextual_fraction():
    def bell_test(params):
        a0, a1, b0, b1 = params
        alice = bit @ Z(1, 1, a0) >> BitControlledGate(Z(1, 1, a1 - a0))\
            >> H() >> Measure(1)
        bob = Diagram.swap(qubit, bit) >> bit @ Z(1, 1, b0)\
            >> BitControlledGate(Z(1, 1, b1 - b0)) >> H() >> Measure(1)
        return bit @ Scalar(.5 ** .5) @ Z(0, 2) @ bit >> alice @ bob
    start = EmpiricalModel.from_diagram(bell_test([0., .35, -.125, .125]))
    gradient = start.contextual_fraction_gradient()
    assert np.isclose(
        gradient @ start.vector(), start.contextual_fraction() - 1)
    assert gradient @ start.scenario.deterministic((0, 0, 0, 0)).vector()\
        <= -1 + 1e-9
    params, best = maximise_contextual_fraction(
        bell_test, [0., .35, -.125, .125], steps=3)
    assert best.contextual_fraction() > start.contextual_fraction()
    assert np.isclose(params[1] - params[0], .25, atol=.05)
    assert (best.vector() == EmpiricalModel.from_diagram(
        bell_test(params)).vector()).all()


def test_local_measurement():
    plus = Ket(0) >> H()
    measured = EmpiricalModel.conditionals(
        bit @ plus >> local_measurement([0, 0, .25, .25]), (2, ))
    assert np.allclose(list(measured[(0, )].values()), [.5, .5])
    assert len(measured[(1, )]) == 1


def bell_pair_map():
    source = Box("source", Ty(), qubit,
                 Discard(qubit) @ qubit >> Scalar(.5 ** .5) @ Z(0, 2) @ qubit,
                 memory=qubit, prediction=qubit)
    sink = Box("sink", qubit, Ty(), Ket(0) @ qubit, prediction=qubit)
    return CMap([source, sink], [((0, 0), (1, 0))])


def test_MessagePassingScenario():
    scenario = MessagePassingScenario(bell_pair_map(), Ket(0) ** 3, n_steps=1)
    angles = [0, 0, .25, .25, .25, .125, 0, .25]
    assert scenario.diagram(angles).dom == scenario.diagram(angles).cod\
        == bit ** 2
    assert np.isclose(
        scenario.model(angles).contextual_fraction(), 2 ** .5 - 1)
    first_tick = MessagePassingScenario(bell_pair_map(), Ket(0) ** 3)
    assert not first_tick.model(angles).is_contextual()
    with pytest.raises(ValueError):
        MessagePassingScenario(CMap([Box(
            "f", qubit, Ty(), Z(1, 3), prediction=qubit ** 2)], []), Ket(0))
    with pytest.raises(ValueError):
        MessagePassingScenario(
            CMap([Box("f", qubit, Ty(), Diagram.id(qubit))], []),
            Diagram.id())
    with pytest.raises(ValueError):
        MessagePassingScenario.default_measurement(mode)


def test_learn():
    scenario = MessagePassingScenario(bell_pair_map(), Ket(0) ** 3, n_steps=1)
    angles, model = scenario.learn(restarts=2, steps=1)
    assert angles.shape == (8, )
    assert model == scenario.model(angles)


def test_dual_rail_scenario():
    pair = qmode ** 2
    bell = (Scalar(.5 ** .5) @ Z(0, 2)).to_dual_rail()
    source = Box("source", Ty(), pair, Discard(pair) @ pair >> bell @ pair,
                 memory=pair, prediction=pair)
    sink = Box("sink", pair, Ty(), Create(0, 0) @ pair, prediction=pair)
    photonic = MessagePassingScenario(
        CMap([source, sink], [((0, 0), (1, 0)), ((0, 1), (1, 1))]),
        Create(0, 0, 0, 0, 0, 0), n_steps=1)
    counts = photonic.model([0, 0, .25, 0, .125, 0, .25, 0])
    assert counts.scenario.outcomes[(1, 1)] == ((0, 1), (1, 0))
    assert counts.is_no_signalling()
    assert np.isclose(counts.contextual_fraction(), 2 ** .5 - 1)
