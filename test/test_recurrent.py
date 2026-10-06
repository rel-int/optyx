from itertools import permutations
from math import factorial, prod

import numpy as np
import pytest
from scipy.stats import unitary_group

from optyx.recurrent import (
    Recurrent, LoopState, sector, position, givens, interfere)


def permanent(matrix):
    size = len(matrix)
    return sum(prod(matrix[i, sigma[i]] for i in range(size))
               for sigma in permutations(range(size)))


def unrolled(network, ticks):
    """
    The unitary of the interferometer unrolled over `ticks` ticks, on the
    loop, then the external modes of each tick, then the environment of
    the loop at each tick.
    """
    loop, external = network.loop, len(network.inputs)
    size = loop + ticks * (external + loop)
    total = np.eye(size, dtype=complex)
    gamma = network.transmissivity
    for time in range(ticks):
        fresh = loop + time * external
        environment = loop + ticks * external + time * loop
        step = np.eye(size, dtype=complex)
        modes = list(range(loop)) + list(range(fresh, fresh + external))
        step[np.ix_(modes, modes)] = network.unitary
        loss = np.eye(size, dtype=complex)
        for mode in range(loop):
            pair = [mode, environment + mode]
            loss[np.ix_(pair, pair)] = [
                [np.sqrt(gamma), -np.sqrt(1 - gamma)],
                [np.sqrt(1 - gamma), np.sqrt(gamma)]]
        total = loss @ step @ total
    return total


def brute_force(network, ticks):
    """
    The joint distribution of the detected patterns from the permanents of
    the unrolled interferometer, with perfect detectors.
    """
    loop, external = network.loop, len(network.inputs)
    matrix = unrolled(network, ticks)
    size = len(matrix)
    inputs = [mode for time in range(ticks) for k, q in enumerate(
        network.inputs) for mode in [loop + time * external + k] * q]
    photons = len(inputs)
    result = {}
    for output in sector(size, photons):
        rows = [mode for mode, n in enumerate(output) for _ in range(n)]
        amplitude = permanent(matrix[np.ix_(rows, inputs)]) / np.sqrt(
            prod(factorial(n) for n in output)
            * prod(factorial(q) for q in network.inputs) ** ticks)
        history = tuple(
            tuple(int(n) for n in output[
                loop + time * external: loop + (time + 1) * external])
            for time in range(ticks))
        result[history] = result.get(history, 0) + abs(amplitude) ** 2
    return result


def assert_close(left, right):
    for key in set(left) | set(right):
        assert np.isclose(left.get(key, 0), right.get(key, 0), atol=1e-10), key


@pytest.mark.parametrize("loop, inputs, transmissivity, ticks", [
    (1, (1, ), 1., 3),
    (1, (2, ), .7, 2),
    (2, (1, ), 1., 3),
    (2, (1, 0), .6, 2),
    (1, (1, 1), .8, 2),
])
def test_distribution_matches_permanents(loop, inputs, transmissivity, ticks):
    unitary = unitary_group.rvs(loop + len(inputs), random_state=loop + ticks)
    network = Recurrent(unitary, loop, inputs, transmissivity)
    assert_close(network.distribution(ticks), brute_force(network, ticks))


def test_distribution_is_normalised_with_inefficient_detectors():
    unitary = unitary_group.rvs(3, random_state=4)
    network = Recurrent(unitary, 2, (1, ), .9, efficiency=.5)
    distribution = network.distribution(2, burn_in=1)
    assert np.isclose(sum(distribution.values()), 1)
    perfect = Recurrent(unitary, 2, (1, ), .9).distribution(2, burn_in=1)
    assert distribution[((0, ), (0, ))] > perfect.get(((0, ), (0, )), 0)


def test_sampling_frequencies():
    unitary = unitary_group.rvs(3, random_state=7)
    network = Recurrent(unitary, 2, (1, ), .8)
    exact = network.distribution(1, burn_in=3)
    samples = [tuple(network.sample(1, burn_in=3, seed=seed))
               for seed in range(2000)]
    for history, weight in exact.items():
        assert abs(samples.count(history) / len(samples) - weight) < .04


def test_trajectory_photon_numbers():
    unitary = unitary_group.rvs(4, random_state=2)
    network = Recurrent(unitary, 3, (1, ))
    patterns, photons = network.trajectory(20, seed=1)
    injected = np.arange(20)
    detected = np.cumsum([0] + [sum(p) for p in patterns[:-1]])
    assert photons == list(injected - detected)


def test_occupation_matches_mean_loop_photons():
    unitary = unitary_group.rvs(3, random_state=5)
    network = Recurrent(unitary, 2, (1, ), .7)
    depth = network.burn_in(1e-6)
    means = []
    for seed in range(600):
        _, photons = network.trajectory(depth + 1, seed=seed)
        means.append(photons[-1])
    assert abs(np.mean(means) - network.occupation()) < .1


def test_lossless_occupation_is_universal():
    for seed in range(3):
        unitary = unitary_group.rvs(6, random_state=seed)
        assert np.isclose(Recurrent(unitary, 4, (1, 1)).occupation(), 4)


def test_burn_in_bounds_the_distance_to_stationarity():
    network = Recurrent([[.6, .8], [.8, -.6]], 1, (1, ), .5)
    depth = network.burn_in(.1)
    assert depth == 4
    late = network.distribution(1, burn_in=depth + 4)
    early = network.distribution(1, burn_in=depth)
    distance = sum(abs(late.get(k, 0) - early.get(k, 0))
                   for k in set(late) | set(early)) / 2
    assert distance <= .1


def test_interfere_matches_permanents():
    unitary = unitary_group.rvs(4, random_state=9)
    occupations = sector(4, 3)
    for column, inputs in enumerate(occupations):
        vector = np.zeros(len(occupations))
        vector[column] = 1
        output = interfere(unitary, vector, 3)
        columns = [m for m, n in enumerate(inputs) for _ in range(n)]
        for row, outputs in enumerate(occupations):
            rows = [m for m, n in enumerate(outputs) for _ in range(n)]
            expected = permanent(unitary[np.ix_(rows, columns)]) / np.sqrt(
                prod(factorial(n) for n in outputs)
                * prod(factorial(n) for n in inputs))
            assert np.isclose(output[row], expected)


def test_givens_and_index():
    unitary = unitary_group.rvs(5, random_state=8)
    phases, rotations = givens(unitary)
    product = np.eye(5, dtype=complex)
    for mode, matrix in rotations:
        step = np.eye(5, dtype=complex)
        step[mode:mode + 2, mode:mode + 2] = matrix
        product = product @ step
    assert np.allclose(product @ np.diag(phases), unitary)
    assert sector(3, 0).tolist() == [[0, 0, 0]]
    assert position(sector(3, 4), 4).tolist() == list(range(len(sector(3, 4))))


def test_loop_state():
    state = LoopState([[1, 0], [0, 1]], [3, 4])
    assert state.photons == 1
    assert np.allclose(state.normalised().amplitudes, [.6, .8])
    assert eval(repr(state)) == state


def test_errors():
    with pytest.raises(ValueError):
        Recurrent(np.eye(3), loop=1, inputs=(1, ))
    with pytest.raises(ValueError):
        Recurrent([[1, 1], [0, 1]], loop=1, inputs=(1, ))
    with pytest.raises(ValueError):
        Recurrent(np.eye(2), loop=1, inputs=(1, )).burn_in(1e-3, max_depth=5)


def lossy_diagram():
    from optyx.channel import qmode, Discard
    from optyx.photonic import Create, MZI, PhotonLoss
    step = (Create(1) @ Create(1) @ qmode
            >> MZI(0.3, 0.2) @ qmode >> qmode @ MZI(0.15, 0.7)
            >> qmode @ Discard(qmode) @ PhotonLoss(0.7))
    return step.feedback(mem=qmode, state=Create(0))


@pytest.mark.parametrize("unrollings", [1, 2])
def test_from_diagram_matches_the_unrolled_diagram(unrollings):
    from optyx.photonic import NumberResolvingMeasurement
    diagram = lossy_diagram()
    network = Recurrent.from_diagram(diagram)
    assert (network.loop, network.inputs, network.visible) \
        == (1, (1, 1, 0), (0, ))
    exact = (diagram.unroll(unrollings) >> NumberResolvingMeasurement(
        unrollings + 1)).eval().prob_dist()
    assert_close(
        network.distribution(unrollings + 1),
        {tuple((n, ) for n in key): value for key, value in exact.items()})


def test_partially_distinguishable_photons():
    from collections import Counter
    from optyx.channel import qmode, Discard
    from optyx.photonic import Create, MZI, NumberResolvingMeasurement
    p, mzi = .5, MZI(.2, .3)

    def internal(photon):
        state = np.zeros(3)
        state[0], state[photon] = np.sqrt(p), np.sqrt(1 - p)
        return state

    unrolled = (
        Create(1, internal_states=(internal(1), )) @ Create(0) >> mzi
        >> qmode @ Create(1, internal_states=(internal(2), )) @ qmode
        >> qmode @ mzi >> qmode @ qmode @ Discard(qmode)
        >> NumberResolvingMeasurement(2))
    exact = unrolled.inflate(3).eval().prob_dist()
    diagram = (Create(1) @ qmode >> mzi).feedback(
        mem=qmode, state=Create(0))
    network = Recurrent.from_diagram(diagram, indistinguishability=p)
    counts = Counter(tuple(n for (n, ) in network.sample(2, seed=seed))
                     for seed in range(4000))
    for key, value in exact.items():
        assert abs(counts[key] / 4000 - value) < .03


def test_distinguishable_photons_walk_alone():
    unitary = unitary_group.rvs(3, random_state=6)
    network = Recurrent(unitary, 2, (1, ), indistinguishability=0.)
    patterns, photons = network.trajectory(30, seed=2)
    assert photons == list(
        np.arange(30) - np.cumsum([0] + [sum(p) for p in patterns[:-1]]))
    assert network.burn_in(1e-2) >= Recurrent(unitary, 2, (1, )).burn_in(
        1e-2)
    with pytest.raises(NotImplementedError):
        network.distribution(1)


def test_sample_and_its_errors():
    from optyx.channel import Diagram, qmode
    from optyx.photonic import Create, NumberResolvingMeasurement
    assert len(lossy_diagram().sample(ticks=3, tol=1e-2, seed=0)) == 3
    with pytest.raises(ValueError):
        Recurrent.from_diagram(Diagram.swap(qmode, qmode).feedback(
            state=Create(0)))
    with pytest.raises(NotImplementedError):
        Recurrent.from_diagram((
            Create(1) @ qmode >> NumberResolvingMeasurement(1) @ qmode
        ).feedback(mem=qmode, state=Create(0)))
    with pytest.raises(NotImplementedError):
        Recurrent.from_diagram((Create(1) @ qmode >> Diagram.swap(
            qmode, qmode)).feedback(state=Create(1)))


def test_complete_refuses_a_non_isometry():
    from optyx.recurrent import complete
    with pytest.raises(ValueError):
        complete(np.array([[1.], [1.]]))
