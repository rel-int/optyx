import numpy as np
import pytest

from optyx.channel import Diagram, qmode
from optyx.photonic import BS, MZI, Phase, PhotonLoss, Create
from optyx.sampling import Coherent, Fock, PhaseLocked, Squeezed, Thermal
from optyx.hopfield import Hopfield

THETA = np.array([.2, .35, .15, .1, .3])


def looped(reciprocal=True):
    """
    Two units read through a chip with two loop modes, a palindrome of
    layers between losses unless it is not `reciprocal`.
    """
    def chip(theta):
        layers = [MZI(theta[0], 0) @ MZI(theta[1], 0),
                  qmode @ MZI(theta[2], 0) @ qmode,
                  Phase(theta[3]) @ qmode @ qmode @ Phase(theta[4])]
        back = layers[::-1] if reciprocal else [
            MZI(.1, .3) @ MZI(.2, .6), qmode @ MZI(.4, .7) @ qmode]
        loss = qmode @ qmode @ PhotonLoss(.7) @ PhotonLoss(.7)
        return Diagram.then(loss, *layers, *back, loss).feedback(
            mem=qmode @ qmode, state=Create(0, 0))
    return chip


def railed(theta):
    """
    Two units on two rails each, the chip symmetric under the exchange of
    the rails: beam splitters on the rails, a palindrome on their sums and
    on their differences, beam splitters again.
    """
    order = Diagram.permutation([0, 2, 1, 3], qmode ** 4)
    layers = [MZI(theta[0], 0) @ MZI(theta[1], 0),
              Phase(theta[2]) @ qmode @ Phase(theta[3]) @ qmode]
    return Diagram.then(BS @ BS, order, *layers, *layers[::-1], order, BS @ BS)


def electronics(**kwargs):
    return dict(gain=.8, bias=(.4, -.3), target=.6) | kwargs


@pytest.mark.parametrize("ratio", [1, 3])
def test_exact_with_a_photonic_loop(ratio):
    chip = Hopfield(looped(), PhaseLocked((0, 0)), amplitude=10,
                    **electronics(ratio=ratio))
    couplings = chip.transfer(THETA)
    assert np.allclose(couplings, couplings.T)
    assert np.allclose(chip.propagate(THETA, beta=1e-4, exact=True),
                       chip.gradient(THETA, exact=True), atol=1e-5)


def test_the_photonic_loop_changes_the_couplings_only():
    fixed = [Hopfield(looped(), PhaseLocked((0, 0)), amplitude=10,
                      **electronics(ratio=ratio)).settle(THETA)
             for ratio in (1, 4)]
    assert np.allclose(*fixed)
    chip = Hopfield(looped(), PhaseLocked((0, 0)), amplitude=10,
                    **electronics())
    activations = np.tanh(fixed[0])
    assert np.allclose(fixed[0], .8 * chip.transfer(THETA) @ activations
                       + chip.bias)


def test_biased_without_reciprocity():
    chip = Hopfield(looped(False), PhaseLocked((0, 0)), amplitude=10,
                    **electronics())
    estimate = chip.propagate(THETA, beta=1e-4, exact=True)
    gradient = chip.gradient(THETA, exact=True)
    assert estimate @ gradient / np.linalg.norm(estimate) \
        / np.linalg.norm(gradient) < .95


def test_exact_on_two_rails():
    chip = Hopfield(railed, Coherent((100., ) * 4), **electronics())
    theta = THETA[:4]
    assert np.allclose(chip.propagate(theta, beta=1e-4, exact=True),
                       chip.gradient(theta, exact=True), atol=1e-5)


@pytest.mark.parametrize("light, amplitude, ratio", [
    (PhaseLocked((0, 0)), 10, 1),
    (Squeezed((.5, .5)), 10, 1),
    (Thermal((.5, .5)), 10, 1),
])
def test_noisy_fields_sample_the_mean_dynamics(light, amplitude, ratio):
    chip = Hopfield(looped(), light, amplitude=amplitude,
                    **electronics(ratio=ratio))
    means, _ = chip.moments(THETA, ticks=400, burn_in=20, seed=0)
    exact, _ = chip.moments(THETA, exact=True)
    assert np.allclose(means, exact, atol=.02)


@pytest.mark.parametrize("light, ratio", [
    (Coherent((400., ) * 4), 1),
    (Thermal((400., ) * 4), 1),
    (Fock((1, ) * 4), 40),
])
def test_noisy_intensities_sample_the_mean_dynamics(light, ratio):
    chip = Hopfield(railed, light, **electronics(ratio=ratio))
    means, _ = chip.moments(THETA[:4], ticks=100, burn_in=10, seed=0)
    exact, _ = chip.moments(THETA[:4], exact=True)
    assert np.allclose(means, exact, atol=.05)


def test_the_readout_of_held_activations():
    chip = Hopfield(railed, Coherent((400., ) * 4), **electronics())
    activations = np.array([.5, -.8])
    begin, step = chip.optics(THETA[:4])
    rng = np.random.default_rng(0)
    memory, readouts = begin(rng), []
    for _ in range(2000):
        outcome, memory = step(memory, rng, chip.modulate(activations))
        readouts.append(chip.read(outcome))
    assert np.allclose(np.mean(readouts, axis=0),
                       chip.transfer(THETA[:4]) @ activations, atol=.01)


def test_settle_raises_when_the_dynamics_oscillate():
    chip = Hopfield(lambda theta: Phase(theta[0]), PhaseLocked((0, )),
                    amplitude=1, gain=-3, leak=1, bias=(.5, ))
    with pytest.raises(ValueError, match="not settled"):
        chip.settle((0, ), ticks=100)


def test_refusals():
    with pytest.raises(ValueError, match="two rails"):
        Hopfield(railed, Coherent((1., ) * 3))
    with pytest.raises(ValueError, match="two rails"):
        Hopfield(railed, PhaseLocked((1., ) * 4))
    with pytest.raises(NotImplementedError, match="not Gaussian"):
        Hopfield(railed, Fock((1, ) * 4), amplitude=1)
