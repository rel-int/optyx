import numpy as np
import pytest

from optyx.channel import qmode, mode, bit, Diagram
from optyx.photonic import (
    Create, MZI, Phase, PhotonLoss, Dephasing, NumberResolvingMeasurement)
from optyx import classical
from optyx.sampling import Fock, Coherent, Thermal
from optyx.equilibrium import Equilibrium


def palindrome(dephased=True, reciprocal=True, feedforward=False):
    """
    A family of loops of two memory modes and one port. Its step is a
    palindrome of layers between losses, so that its mode matrix is
    symmetric, unless it is not `reciprocal`; its memory is `dephased` at
    the cut; with `feedforward`, the parity of the count of a tick sets a
    phase in the middle of the next.
    """
    def loop(theta):
        layers = [MZI(theta[0], 0) @ qmode, qmode @ MZI(theta[1], 0),
                  qmode @ qmode @ Phase(theta[2])]
        back = layers[::-1] if reciprocal else [
            MZI(.1, .3) @ qmode, qmode @ MZI(.4, .7)]
        loss = qmode @ PhotonLoss(.8) @ PhotonLoss(.8)
        start = [qmode @ Dephasing() @ Dephasing()] if dephased else []
        if not feedforward:
            return Diagram.then(*start, loss, *layers, *back, loss).feedback(
                mem=qmode @ qmode, state=Create(0, 0))
        middle = qmode @ qmode @ Diagram.swap(qmode, bit) \
            >> qmode @ qmode @ classical.BitControlledGate(Phase(.25))
        parity = classical.ClassicalFunction(lambda n: [n[0] % 2], mode, bit)
        readout = NumberResolvingMeasurement(1) @ qmode @ qmode \
            >> classical.CopyN(2) @ qmode @ qmode \
            >> mode @ parity @ qmode @ qmode \
            >> mode @ Diagram.permutation([1, 2, 0], bit @ qmode @ qmode)
        return Diagram.then(
            *[box @ bit for box in start + [loss] + layers], middle, *back,
            loss, readout).feedback(
                mem=qmode @ qmode @ bit,
                state=Create(0, 0) @ classical.Bit(0))
    return loop


THETA = (.2, .3, .1)


@pytest.mark.parametrize("source, p", [
    (Fock((1, )), 1.), (Fock((2, )), .3), (Coherent((.6, )), 1.),
    (Thermal((.6, )), .5)])
def test_exact_for_a_dephased_reciprocal_loop(source, p):
    ep = Equilibrium(palindrome(), source, (1., ), indistinguishability=p)
    assert np.allclose(ep.propagate(THETA, exact=True),
                       ep.gradient(THETA, exact=True), atol=1e-6)


def test_one_body_means_ignore_the_statistics():
    values = {Equilibrium(palindrome(), source, (1., ), indistinguishability=p
                          ).value(THETA, exact=True)
              for source, p in [(Fock((1, )), 1.), (Fock((1, )), 0.),
                                (Coherent((1., )), 1.), (Thermal((1., )), 1.)]}
    assert np.allclose(list(values), min(values))


@pytest.mark.parametrize("dephased, reciprocal", [
    (False, True), (False, False), (True, False)])
def test_biased_otherwise(dephased, reciprocal):
    ep = Equilibrium(palindrome(dephased, reciprocal), Fock((1, )), (1., ))
    assert not np.allclose(ep.propagate(THETA, exact=True),
                           ep.gradient(THETA, exact=True), atol=1e-3)


def test_interferometer_peels_the_dephasing():
    network, dephased = Equilibrium(
        palindrome(), Fock((1, )), (1., )).interferometer(THETA, .5)
    assert dephased == (0, 1) and network.source.means[0] == 1.5
    _, dephased = Equilibrium(
        palindrome(False), Fock((1, )), (1., )).interferometer(THETA)
    assert dephased == ()


def test_sampled_estimates():
    ep = Equilibrium(palindrome(), Coherent((.6, )), (1., ), cap=6)
    sampling = dict(ticks=20, burn_in=10, seed=0, shots=2)
    (memory, counts), _ = ep.means(THETA, **sampling)
    assert memory.shape == (2, ) and counts.shape == (1, )
    assert ep.gradient(THETA, step=.05, **sampling).shape == (3, )
    assert ep.propagate(THETA, beta=.3, step=.05, **sampling).shape == (3, )
    exact = ep.exact(THETA)[1]
    assert abs(ep.value(THETA, ticks=400, burn_in=20, seed=1) - exact) < .1


def test_feedforward_is_sampled():
    ep = Equilibrium(palindrome(feedforward=True), Fock((1, )), (1., ),
                     indistinguishability=.5, cap=6)
    estimate = ep.propagate(THETA, beta=.3, step=.05, ticks=10, burn_in=5,
                            seed=0)
    assert np.all(np.isfinite(estimate))
    with pytest.raises(NotImplementedError):
        ep.exact(THETA)


def test_cost_on_the_ports():
    with pytest.raises(ValueError):
        Equilibrium(palindrome(), Fock((1, )), (1., 1.))
    ep = Equilibrium(palindrome(), Fock((1, )), (1., ))
    assert ep.nudged(0) == Fock((1, )) and ep.nudged(.5).means == (1.5, )
