from collections import Counter
from itertools import permutations
from math import comb, factorial, prod

import numpy as np
import pytest
from scipy.stats import unitary_group

from optyx.sampling import (
    Interferometer, FockState, Sweep, Unravelling, Fock, Coherent, Thermal,
    PhaseLocked, Squeezed, Fields, Displaced, Gaussian,
    Source, levels, sector, position, givens, interfere, twomode)


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
    (1, (1, 0, 0), .9, 2),
    (2, (0, 1, 1), 1., 2),
])
def test_distribution_matches_permanents(loop, inputs, transmissivity, ticks):
    unitary = unitary_group.rvs(loop + len(inputs), random_state=loop + ticks)
    network = Interferometer(unitary, loop, inputs, transmissivity)
    assert_close(network.distribution(ticks), brute_force(network, ticks))


def test_distribution_is_normalised_with_inefficient_detectors():
    unitary = unitary_group.rvs(3, random_state=4)
    network = Interferometer(unitary, 2, (1, ), .9, efficiency=.5)
    distribution = network.distribution(2, burn_in=1)
    assert np.isclose(sum(distribution.values()), 1)
    perfect = Interferometer(unitary, 2, (1, ), .9).distribution(2, burn_in=1)
    assert distribution[((0, ), (0, ))] > perfect.get(((0, ), (0, )), 0)


def test_sampling_frequencies():
    unitary = unitary_group.rvs(3, random_state=7)
    network = Interferometer(unitary, 2, (1, ), .8)
    exact = network.distribution(1, burn_in=3)
    samples = [tuple(network.sample(1, burn_in=3, seed=seed))
               for seed in range(2000)]
    for history, weight in exact.items():
        assert abs(samples.count(history) / len(samples) - weight) < .04


def test_trajectory_photon_numbers():
    unitary = unitary_group.rvs(4, random_state=2)
    network = Interferometer(unitary, 3, (1, ))
    patterns, photons = network.trajectory(20, seed=1)
    injected = np.arange(20)
    detected = np.cumsum([0] + [sum(p) for p in patterns[:-1]])
    assert photons == list(injected - detected)


def test_occupation_matches_mean_loop_photons():
    unitary = unitary_group.rvs(3, random_state=5)
    network = Interferometer(unitary, 2, (1, ), .7)
    depth = network.burn_in(1e-6)
    means = []
    for seed in range(600):
        _, photons = network.trajectory(depth + 1, seed=seed)
        means.append(photons[-1])
    assert abs(np.mean(means) - network.occupation()) < .1


def test_lossless_occupation_is_universal():
    for seed in range(3):
        unitary = unitary_group.rvs(6, random_state=seed)
        assert np.isclose(Interferometer(unitary, 4, (1, 1)).occupation(), 4)


def test_burn_in_bounds_the_distance_to_stationarity():
    network = Interferometer([[.6, .8], [.8, -.6]], 1, (1, ), .5)
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


@pytest.mark.parametrize("loop, inputs", [
    (2, (1, 0, 2)), (3, (0, 0, 1, 0)), (1, (2, 1)), (2, (0, 0))])
def test_detections_match_the_dense_interferometer(loop, inputs):
    modes = loop + len(inputs)
    unitary = unitary_group.rvs(modes, random_state=modes)
    network = Interferometer(unitary, loop, inputs)
    rng = np.random.default_rng(3)
    occupations = sector(loop, 2)
    state = FockState(occupations, rng.normal(size=len(occupations))
                      + 1j * rng.normal(size=len(occupations))).normalised()
    photons = 2 + sum(inputs)
    vector = np.zeros(len(sector(modes, photons)), dtype=complex)
    vector[position(np.hstack([occupations, np.tile(
        inputs, (len(occupations), 1))]), photons)] = state.amplitudes
    output, rows = interfere(unitary, vector, photons), sector(modes, photons)
    detected = list(network.detections(state))
    assert np.isclose(sum(weight for _, weight, _ in detected), 1)
    for pattern, weight, after in detected:
        mask = (rows[:, loop:] == pattern).all(axis=1)
        assert np.isclose(weight, np.sum(np.abs(output[mask]) ** 2))
        expected = FockState(rows[mask][:, :loop], output[mask]).normalised()
        assert np.allclose(np.abs(np.vdot(
            expected.amplitudes, after.amplitudes[position(
                expected.occupations, after.photons)])), 1)


def test_sweep_holds_the_loop_and_the_occupied_inputs():
    unitary = unitary_group.rvs(12, random_state=0)
    network = Interferometer(unitary, 3, (1, 0, 2, 0, 0, 0, 1, 0, 0))
    sweep = network.sweep()
    assert sweep is network.sweep() and sweep.columns == (0, 1, 2, 3, 5, 9)
    widths = sweep.widths()
    assert max(width + grows for width, grows in zip(widths, sweep.grows)) \
        == 3 + 3 + 1
    assert widths[-1] == 3
    assert Sweep.from_unitary(np.eye(2), 1, (0, )).widths() == [1, 1]


def test_equality_ignores_the_sign_of_zero():
    unitary = np.array([[1, -0j], [0j, 1]])
    assert eval(repr(Interferometer(unitary, 1, (1, )))) \
        == Interferometer(unitary, 1, (1, ))


def test_loop_state():
    state = FockState([[1, 0], [0, 1]], [3, 4])
    assert state.photons == 1
    assert np.allclose(state.normalised().amplitudes, [.6, .8])
    assert eval(repr(state)) == state


def test_errors():
    with pytest.raises(ValueError):
        Interferometer(np.eye(3), loop=1, inputs=(1, ))
    with pytest.raises(ValueError):
        Interferometer([[1, 1], [0, 1]], loop=1, inputs=(1, ))
    with pytest.raises(ValueError):
        Interferometer(np.eye(2), loop=1, inputs=(1, )).burn_in(
            1e-3, max_depth=5)


def test_twomode_is_unitary_beyond_twenty_photons():
    matrix = unitary_group.rvs(2, random_state=0)
    block = twomode(matrix, 25)
    assert np.allclose(block.conj().T @ block, np.eye(26))


def columns(unitary, loop, transmissivity, depth=100):
    """
    The probabilities q_c that a photon injected into the one external input
    is detected in the one external output a delay c later.
    """
    blocks = unitary[:loop, :loop], unitary[loop:, :loop]
    amplitudes, vector = [unitary[loop, loop]], unitary[:loop, loop]
    for _ in range(depth):
        vector = np.sqrt(transmissivity) * vector
        amplitudes.append(blocks[1][0] @ vector)
        vector = blocks[0] @ vector
    return np.abs(amplitudes) ** 2


def coefficient(series, order):
    """ The coefficient of x^order in the product of the power series. """
    result = np.eye(order + 1)[0]
    for coefficients in series:
        result = np.convolve(result, coefficients)[:order + 1]
    return result[order]


def bunching(source, overlap, q, order):
    """ The stationary factorial moments of bunching-rate.md. """
    if source is Thermal:
        return factorial(order) * q.sum() ** order
    if source is Coherent:
        return factorial(order) ** 2 * coefficient(
            [[x ** k / factorial(k) ** 2 for k in range(order + 1)]
             for x in q], order)
    gain = sum(comb(order, j) * overlap ** j * (1 - overlap) ** (order - j)
               * factorial(j) for j in range(order + 1))
    return factorial(order) * gain * coefficient([[1, x] for x in q], order)


@pytest.mark.parametrize("source, overlap", [
    (Fock, 1.), (Fock, .5), (Fock, 0.), (Coherent, 1.), (Thermal, 1.)])
def test_stationary_factorial_moments(source, overlap):
    unitary, transmissivity = unitary_group.rvs(2, random_state=1), .6
    network = Interferometer(
        unitary, loop=1, transmissivity=transmissivity,
        indistinguishability=overlap,
        source=Fock((1, )) if source is Fock else source((1., )))
    burn_in, runs = network.burn_in(1e-3), 100
    counts = np.array([[pattern[0] for pattern in network.sample(
        50, burn_in, seed)] for seed in range(runs)])
    q = columns(unitary, 1, transmissivity)
    for order in (1, 2):
        falling = prod(counts - k for k in range(order)).mean(axis=1)
        error = falling.std() / np.sqrt(runs)
        assert abs(falling.mean() - bunching(source, overlap, q, order)) \
            < 4 * error


def test_sources():
    thermal = Interferometer(np.eye(2)[::-1], loop=1, source=Thermal((1., )))
    assert eval(repr(thermal)) == thermal
    assert thermal.burn_in(1e-3) > 0 and thermal.occupation() == 1.
    assert Fock((1, )).pad(2) == Fock((1, 0, 0))
    assert Coherent((1., )).pad(1) == Coherent((1., 0.))
    assert Coherent((1., )) != Thermal((1., ))
    with pytest.raises(ValueError):
        Interferometer(np.eye(2), loop=1)
    with pytest.raises(ValueError):
        Interferometer(np.eye(2), loop=1, inputs=(1, ), source=Fock((1, )))
    with pytest.raises(NotImplementedError):
        thermal.inputs
    with pytest.raises(NotImplementedError):
        thermal.distribution(1)
    with pytest.raises(NotImplementedError):
        Source((1., )).draw(None)


def test_sources_on_diagrams():
    from optyx.channel import Diagram, Encode, qmode, mode
    from optyx.photonic import (
        Create, NumberResolvingMeasurement, PhotonLoss, BS)
    fibre = Diagram.swap(qmode, qmode).feedback(state=Create(0))
    network, _ = fibre.sampler(source=Coherent((2., )))
    assert network.source == Coherent((2., ))
    with pytest.raises(ValueError):
        fibre.sample()
    with pytest.raises(ValueError):
        fibre.sample(source=Coherent((1., 1.)))
    with pytest.raises(ValueError):
        (Create(1) @ qmode >> Diagram.swap(qmode, qmode)).feedback(
            state=Create(0)).sample(source=Fock((1, )))
    lossy = (Create(1) @ qmode @ qmode >> BS @ qmode >> qmode @ BS
             >> qmode @ qmode @ PhotonLoss(.5)).feedback(state=Create(0))
    network, _ = lossy.sampler(source=Thermal((1., )), burn_in=0)
    assert network.source == Thermal((1., )) @ Fock((1, 0))
    assert eval(repr(network.source)) == network.source
    unravelling = Unravelling(lossy, cap=6, source=Fock((1, )))
    assert "source=Fock((1,))" in repr(unravelling)
    exact = Interferometer.from_diagram(lossy, source=Fock((1, ))) \
        .distribution(1, burn_in=3)
    counts = Counter(tuple(unravelling.sample(1, 3, seed))
                     for seed in range(1000))
    assert sum(abs(counts[key] / 1000 - exact.get(key, 0))
               for key in set(counts) | set(exact)) / 2 < .05
    with pytest.raises(ValueError):
        Unravelling(fibre, cap=1, source=Fock((2, ))).sample(1)
    recycle = (Create(1) @ qmode >> NumberResolvingMeasurement(1) @ qmode
               ).feedback(mem=qmode, state=Create(0))
    assert len(recycle.sample(burn_in=0)) == 1
    encoded = (Encode(mode) @ qmode >> Diagram.swap(qmode, qmode)
               ).feedback(mem=qmode, state=Create(0))
    assert encoded.sample(ticks=2, burn_in=0, seed=0,
                          source=Fock((1, ))) == [(0, ), (1, )]


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
    network = Interferometer.from_diagram(diagram)
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
    network = Interferometer.from_diagram(diagram, indistinguishability=p)
    counts = Counter(tuple(n for (n, ) in network.sample(2, seed=seed))
                     for seed in range(4000))
    for key, value in exact.items():
        assert abs(counts[key] / 4000 - value) < .03


def test_distinguishable_photons_walk_alone():
    unitary = unitary_group.rvs(3, random_state=6)
    network = Interferometer(unitary, 2, (1, ), indistinguishability=0.)
    patterns, photons = network.trajectory(30, seed=2)
    assert photons == list(
        np.arange(30) - np.cumsum([0] + [sum(p) for p in patterns[:-1]]))
    assert network.burn_in(1e-2) >= Interferometer(unitary, 2, (1, )).burn_in(
        1e-2)
    with pytest.raises(NotImplementedError):
        network.distribution(1)


def test_sample_and_its_errors():
    from optyx.channel import Diagram, qmode
    from optyx.photonic import Create, NumberResolvingMeasurement
    assert len(lossy_diagram().sample(ticks=3, tol=1e-2, seed=0)) == 3
    with pytest.raises(ValueError):
        Interferometer.from_diagram(Diagram.swap(qmode, qmode).feedback(
            state=Create(0)))
    with pytest.raises(NotImplementedError):
        Interferometer.from_diagram((
            Create(1) @ qmode >> NumberResolvingMeasurement(1) @ qmode
        ).feedback(mem=qmode, state=Create(0)))
    with pytest.raises(NotImplementedError):
        Interferometer.from_diagram((Create(1) @ qmode >> Diagram.swap(
            qmode, qmode)).feedback(state=Create(1)))


def test_complete_refuses_a_non_isometry():
    from optyx.sampling import complete
    with pytest.raises(ValueError):
        complete(np.array([[1.], [1.]]))


def exact(diagram, ticks):
    """
    The exact distribution of the outputs of `ticks` ticks of a closed
    diagram, every quantum output measured, from its unrolling.
    """
    from optyx.channel import Diagram, Measure
    measured = diagram.unroll(ticks - 1)
    for i, ob in enumerate(measured.cod.inside):
        if ob.name in ("qmode", "qubit"):
            measured = measured >> Diagram.id(measured.cod[:i]) @ Measure(
                measured.cod[i:i + 1]) @ Diagram.id(measured.cod[i + 1:])
    if levels(measured):
        measured = measured.inflate(levels(measured))
    width = len(diagram.cod)
    return {tuple(tuple(key[i * width:(i + 1) * width])
                  for i in range(ticks)): value
            for key, value in measured.eval().prob_dist().items()
            if value > 1e-12}


def feedforward():
    from optyx.channel import qmode, mode, bit
    from optyx.photonic import Create, MZI, Phase, PhotonLoss
    from optyx.photonic import NumberResolvingMeasurement
    from optyx import classical
    parity = classical.ClassicalFunction(lambda x: [x[0] % 2], mode, bit)
    step = (Create(1) @ qmode @ qmode >> MZI(.3, .2) @ qmode
            >> NumberResolvingMeasurement(1) @ qmode @ qmode
            >> parity @ qmode @ qmode >> classical.CopyBit(2) @ qmode @ qmode
            >> bit @ classical.BitControlledGate(Phase(.3)) @ qmode
            >> bit @ MZI(.1, .4) >> bit @ qmode @ PhotonLoss(.8))
    return step.feedback(mem=qmode @ qmode, state=Create(0) @ Create(1))


def qubit_loop():
    from optyx.channel import qubit, bit, Diagram
    from optyx import classical, qubits
    cnot = qubits.Z(1, 2) @ qubit \
        >> qubit @ qubits.X(2, 1) @ qubits.Scalar(2 ** .5)
    step = (qubit @ qubits.Ket(0) >> qubits.X(1, 1, .15) @ qubit >> cnot
            >> qubit @ qubits.Measure(1) >> Diagram.swap(qubit, bit)
            >> classical.CopyBit(2) @ qubit >> bit @ classical.CtrlZ
            >> bit @ qubits.H())
    return step.feedback(mem=qubit, state=qubits.Ket(0))


def adder():
    from optyx.channel import qmode
    from optyx.photonic import Create, BS, NumberResolvingMeasurement
    from optyx import classical
    step = (Create(1) @ Create(1) @ qmode >> qmode @ BS >> BS @ qmode
            >> NumberResolvingMeasurement(2) @ qmode
            >> classical.Add(2) @ qmode)
    return step.feedback(mem=qmode, state=Create(0))


def internal_states():
    from optyx.channel import qmode
    from optyx.photonic import Create, MZI, PhotonLoss
    step = Create(1, internal_states=([.3 ** .5, .7 ** .5], )) @ qmode \
        >> MZI(.2, .3) >> qmode @ PhotonLoss(.6)
    return step.feedback(
        mem=qmode, state=Create(1, internal_states=([1, 0], )))


def kerr():
    from optyx.core import zw
    from optyx.channel import Channel, qmode
    from optyx.photonic import Create, BS
    nonlinear = Channel(
        "Kerr", zw.ZBox(1, 1, lambda n: 1j ** (n * n)), qmode, qmode)
    step = Create(1) @ qmode >> BS >> qmode @ nonlinear >> BS
    return step.feedback(mem=qmode, state=Create(1))


@pytest.mark.parametrize("diagram, ticks", [
    (feedforward, 2), (qubit_loop, 3), (adder, 2), (internal_states, 2),
    (kerr, 2), (lossy_diagram, 2)])
def test_unravelling_matches_the_unrolled_diagram(diagram, ticks):
    from collections import Counter
    diagram = diagram()
    unravelling = Unravelling(diagram)
    counts = Counter(tuple(unravelling.sample(ticks, seed=seed))
                     for seed in range(1000))
    for history, probability in exact(diagram, ticks).items():
        assert abs(counts[history] / 1000 - probability) < .05, history


def test_unravelling_refuses_what_it_cannot_sample():
    from optyx.channel import qmode, Diagram
    from optyx.photonic import Create, BS, Select
    from optyx import classical
    selected = (Create(1) @ qmode >> BS >> Select(1) @ qmode).feedback(
        mem=qmode, state=Create(0))
    with pytest.raises(NotImplementedError):
        Unravelling(selected).sample(2, seed=0)
    postselected = (qmode @ classical.Bit(1) >> qmode @ classical
                    .PostselectBit(0)).feedback(mem=qmode, state=Create(0))
    with pytest.raises(NotImplementedError):
        Unravelling(postselected).sample(1, seed=0)
    with pytest.raises(ValueError):
        Unravelling(lossy_diagram(), cap=1).sample(3, seed=0)
    with pytest.raises(ValueError):
        Unravelling((Create(2) @ qmode >> BS).feedback(
            mem=qmode, state=Create(0)), cap=1).sample(1, seed=0)
    assert repr(Unravelling(adder())).startswith("Unravelling(")
    with pytest.raises(ValueError):
        Unravelling(lossy_diagram(), cap=0)
    with pytest.raises(ValueError):
        Unravelling(Diagram.id(qmode))


def test_sample_dispatches_on_passivity():
    assert len(lossy_diagram().sample(ticks=2, seed=0)) == 2
    assert lossy_diagram().sample(ticks=2, seed=0) \
        == lossy_diagram().sample(ticks=2, seed=0, burn_in=Interferometer
                                  .from_diagram(lossy_diagram()).burn_in(
                                      1e-3))
    assert len(adder().sample(ticks=2, burn_in=1, seed=0)) == 2
    with pytest.raises(NotImplementedError):
        adder().sample(ticks=2)
    with pytest.raises(NotImplementedError):
        adder().sample(ticks=2, burn_in=1, indistinguishability=.5)
    with pytest.raises(NotImplementedError):
        Interferometer.from_diagram(internal_states())
    from optyx.channel import qmode, mode, bit
    from optyx.photonic import Create, BS, Phase, NumberResolvingMeasurement
    from optyx import classical
    parity = classical.ClassicalFunction(lambda n: [n[0] % 2], mode, bit)
    controlled = Create(1, 1) >> BS \
        >> NumberResolvingMeasurement(1) @ qmode >> parity @ qmode \
        >> classical.BitControlledGate(Phase(.5))
    with pytest.raises(NotImplementedError):
        Interferometer.from_diagram(controlled)
    assert len(controlled.sample(burn_in=0, shots=3, seed=0)) == 3


@pytest.mark.parametrize("name", ["delay", "mzi", "two inputs"])
def test_stationary_distribution_matches_fix(name):
    from optyx.channel import Diagram, qmode
    from optyx.photonic import (
        Create, MZI, BS, PhotonLoss, NumberResolvingMeasurement)
    step = {
        "delay": Create(1) @ qmode >> Diagram.swap(qmode, qmode)
        >> qmode @ PhotonLoss(.5),
        "mzi": Create(1) @ qmode >> MZI(.06, 0) >> qmode @ PhotonLoss(.7),
        "two inputs": Create(1, 1) @ qmode >> qmode @ BS
        >> MZI(.3, .2) @ qmode >> qmode @ qmode @ PhotonLoss(.5)}[name]
    loop = step.feedback(mem=qmode, state=Create(0))
    network = Interferometer.from_diagram(loop)
    burn_in = network.burn_in(1e-3)
    assert burn_in + 1 == loop.unroll_certificate(1e-3)
    exact = {history[0]: probability for history, probability
             in network.distribution(1, burn_in).items()}
    fixed = (loop >> NumberResolvingMeasurement(len(loop.cod))).fix(
        tol=1e-3, max_chi=None).prob_dist()
    for outcome in set(exact) | set(fixed):
        assert np.isclose(exact.get(outcome, 0), fixed.get(outcome, 0))


def switch_loop(copies=1):
    """
    A loop whose detector feeds the parity of its count forward to the
    setting of the next tick's switch, every optical wire in `copies`
    copies, the switch acting alike on each and the detector adding them
    up: with one photon per copy, the copies are photons in orthogonal
    internal states.
    """
    from optyx.channel import qmode, mode, bit, Diagram
    from optyx.photonic import MZI, PhotonLoss, NumberResolvingMeasurement
    from optyx import classical
    order = [k // 2 + copies * (k % 2) for k in range(2 * copies)]
    pairs = Diagram.permutation(order, qmode ** (2 * copies))
    switch = classical.BitControlledGate(*(
        pairs >> Diagram.tensor(*[gate] * copies) >> pairs.dagger()
        for gate in (MZI(.3, .2), MZI(.1, .4))))
    parity = classical.ClassicalFunction(lambda n: [n[0] % 2], mode, bit)
    return Diagram.permutation(
        [2 * copies] + list(range(2 * copies)), qmode ** (2 * copies) @ bit
    ) >> switch >> NumberResolvingMeasurement(copies) @ qmode ** copies \
        >> classical.Add(copies) @ qmode ** copies \
        >> classical.CopyN(2) @ qmode ** copies >> mode @ Diagram.permutation(
            list(range(1, copies + 1)) + [0], mode @ qmode ** copies) \
        >> mode @ Diagram.tensor(*[PhotonLoss(.8)] * copies) @ parity


def test_colouring_matches_distinguishable_copies():
    from optyx.channel import qmode, mode, bit, Diagram, Discard
    from optyx.photonic import Create
    from optyx import classical
    ticks = 2
    step = switch_loop(ticks)
    photon = [Create(*[int(i == time) for i in range(ticks)])
              for time in range(ticks)]
    state = photon[0] @ Create(*[0] * ticks) @ classical.Bit(0) >> step
    for time in range(1, ticks):
        state = state >> Diagram.id(mode ** time) @ (
            photon[time] @ qmode ** ticks @ bit >> step)
    exact = (state >> Diagram.id(mode ** ticks) @ Discard(
        qmode ** ticks @ bit)).eval().prob_dist()
    loop = switch_loop().feedback(
        mem=qmode @ bit, state=Create(0) @ classical.Bit(0))
    for p, close in ((0., True), (1., False)):
        unravelling = Unravelling(loop, source=Fock((1, )),
                                  indistinguishability=p)
        counts = Counter(tuple(n for (n, ) in unravelling.sample(
            ticks, seed=seed)) for seed in range(2000))
        assert close == all(abs(counts[history] / 2000 - probability) < .04
                            for history, probability in exact.items())


def test_colouring_agrees_with_the_interferometer():
    from optyx.channel import qmode
    from optyx.photonic import Create, MZI, PhotonLoss
    loop = (MZI(.3, .1) >> qmode @ PhotonLoss(.7)).feedback(
        mem=qmode, state=Create(0))
    network = Interferometer.from_diagram(
        loop, indistinguishability=.5, source=Fock((2, )))
    unravelling = Unravelling(
        loop, source=Fock((2, )), indistinguishability=.5, cap=8)
    assert "indistinguishability=0.5" in repr(unravelling)
    counts = [Counter(sampler.sample(1, burn_in=8, seed=seed)[0]
                      for seed in range(2000))
              for sampler in (network, unravelling)]
    for outcome in set(counts[0]) | set(counts[1]):
        assert abs(counts[0][outcome] - counts[1][outcome]) / 2000 < .04


def test_colouring_refusals():
    from optyx.channel import qmode, mode, Diagram, Measure
    from optyx.photonic import Create, BS
    from optyx import classical
    with pytest.raises(NotImplementedError):
        adder().sample(ticks=2, burn_in=1, indistinguishability=.5)
    with pytest.raises(NotImplementedError):
        Unravelling(qubit_loop(), indistinguishability=.5)
    unpooled = (classical.Digit(0) @ BS >> Diagram.swap(mode, qmode) @ qmode
                >> Measure(qmode @ mode) @ qmode).feedback(
                    mem=qmode, state=Create(0))
    with pytest.raises(NotImplementedError):
        Unravelling(unpooled, source=Fock((1, )),
                    indistinguishability=0.).sample(1, seed=0)


def test_shifted_sources():
    from optyx.sampling import Shifted
    assert Thermal((1., )).shift((.5, )) == Thermal((1.5, ))
    assert Fock((1, )).shift((-.5, )) == Shifted(Fock((1, )), (-.5, ))
    assert repr(Fock((1, )).shift((.5, ))) == "Shifted(Fock((1,)), (0.5,))"
    with pytest.raises(ValueError):
        Coherent((1., )).shift((-2., ))
    with pytest.raises(ValueError):
        Fock((1, )).shift((-2., ))
    rng = np.random.default_rng(0)
    draws = [Fock((2, 1)).shift((.5, -.5)).draw(rng) for _ in range(4000)]
    assert np.allclose(np.mean(draws, axis=0), (2.5, .5), atol=.05)


def test_dephased_correlations_match_the_unravelling():
    from optyx.channel import qmode
    from optyx.photonic import Create, MZI, PhotonLoss, Dephasing
    from optyx.sampling import CQState
    step = qmode @ Dephasing() @ qmode >> MZI(.2, .3) @ qmode \
        >> qmode @ MZI(.35, .1) >> qmode @ PhotonLoss(.6) @ PhotonLoss(.8)
    loop = step.feedback(mem=qmode @ qmode, state=Create(0, 0))
    network = Interferometer.from_step(
        Create(0, 0), step[1:], 1, Coherent((.3, )))
    exact = np.real(np.diag(network.correlations(dephased=(0, ))))
    assert not np.allclose(exact, np.diag(network.correlations()))
    unravelling = Unravelling(loop, cap=8, source=Coherent((.3, )))
    rng = np.random.default_rng(0)
    colours, means = [CQState.empty()], []
    colours = unravelling.run(unravelling.initial, colours, rng)
    for time in range(2500):
        _, colours = unravelling.tick(colours, rng)
        if time > 20:
            means.append(unravelling.occupations(colours))
    assert np.allclose(np.mean(means, axis=0), exact, atol=.03)


def histories(sampler, ticks, shots=2000, burn_in=0):
    return Counter(tuple(n for (n, *_) in sampler.sample(
        ticks, burn_in=burn_in, seed=seed)) for seed in range(shots))


def close(first, second, shots=2000, tol=.04):
    return all(abs(first[key] - second[key]) / shots < tol
               for key in set(first) | set(second))


@pytest.mark.parametrize("source", [Coherent((.6, )), Thermal((.25, ))])
def test_fields_match_the_unravelling_with_feed_forward(source):
    from optyx.channel import qmode, bit
    from optyx.photonic import Create
    from optyx import classical
    loop = switch_loop().feedback(
        mem=qmode @ bit, state=Create(0) @ classical.Bit(0))
    assert close(histories(Fields(loop, source), 3),
                 histories(Unravelling(loop, cap=10, source=source), 3))


def test_phase_locked_light_through_kets_and_fields():
    from optyx.channel import qmode
    from optyx.photonic import Create, MZI, PhotonLoss
    loop = (MZI(.3, .1) >> qmode @ PhotonLoss(.7)).feedback(
        mem=qmode, state=Create(0))
    laser = PhaseLocked((.3j, ))
    assert close(histories(Fields(loop, laser), 3),
                 histories(Unravelling(loop, cap=8, source=laser), 3))
    assert loop.sample(2, burn_in=1, seed=0, source=laser, cap=8) \
        == Unravelling(loop, cap=8, source=laser).sample(2, 1, seed=0)
    assert loop.sample(2, burn_in=1, seed=0, source=laser, fields=True) \
        == Fields(loop, laser).sample(2, 1, seed=0)
    with pytest.raises(NotImplementedError):
        loop.sample(2, source=laser)
    with pytest.raises(NotImplementedError):
        Unravelling(loop, source=laser, indistinguishability=.5).sample(1)
    with pytest.raises(NotImplementedError):
        Interferometer.from_diagram(loop, source=laser).sample(1)


def test_exact_means_with_a_field():
    from optyx.channel import qmode
    from optyx.photonic import Create, MZI, PhotonLoss
    step = MZI(.2, .3) @ qmode >> qmode @ MZI(.35, .1) \
        >> qmode @ PhotonLoss(.6) @ PhotonLoss(.8)
    loop = step.feedback(mem=qmode @ qmode, state=Create(0, 0))
    for source in (PhaseLocked((.8, )), Coherent((.64, ))):
        network = Interferometer.from_diagram(loop, source=source)
        field = network.field()
        exact = network.advance(network.correlations(), (), field)[1][0]
        sampled = np.mean(Fields(loop, source).sample(6000, 30, seed=1))
        assert abs(sampled - exact) < .03
    locked, random = (Interferometer.from_diagram(loop, source=source)
                      for source in (PhaseLocked((.8, )), Coherent((.64, ))))
    assert not np.isclose(np.trace(locked.correlations()),
                          np.trace(random.correlations()))
    assert np.allclose(locked.field(dephased=(0, 1)), locked.unitary[
        :2, 2:] @ locked.source.field)


def test_squeezed_light():
    from optyx.channel import qmode
    from optyx.photonic import Create, BS, PhotonLoss
    squeezed = Squeezed((.2, ))
    loop = (BS >> qmode @ PhotonLoss(.5)).feedback(
        mem=qmode, state=Create(0))
    network = Interferometer.from_diagram(loop, source=squeezed)
    exact = network.advance(network.correlations())[1][0]
    samples = Unravelling(loop, cap=14, source=squeezed).sample(
        4000, 10, seed=2)
    assert abs(np.mean(samples) - exact) < .02
    with pytest.raises(NotImplementedError):
        squeezed.amplitudes(np.random.default_rng(0))
    assert squeezed.pad(1) == Squeezed((.2, 0.)) \
        and squeezed @ squeezed == Squeezed((.2, .2))
    assert PhaseLocked((1, )).shift((3., )) == PhaseLocked((2, )) \
        and PhaseLocked((0, )).shift((4., )) == PhaseLocked((2, ))
    assert repr(PhaseLocked((1j, ))) == "PhaseLocked((1j,))"
    with pytest.raises(ValueError):
        PhaseLocked((1, )).shift((-2., ))
    assert (PhaseLocked((1, )) @ Fock((0, ))).superposed


def test_fields_refuse_what_is_not_a_field():
    from optyx.channel import qmode, Channel
    from optyx.core import zw
    from optyx.photonic import Create, BS
    loop = BS.feedback(mem=qmode, state=Create(0))
    with pytest.raises(NotImplementedError):
        Fields(loop, Fock((1, ))).sample(1, seed=0)
    assert Fields(loop, Fock((0, ))).sample(1, seed=0) == [(0, )]
    kerr = Channel("Kerr", zw.ZBox(1, 1, lambda n: 1j ** (n * n)),
                   qmode, qmode)
    with pytest.raises(NotImplementedError):
        Fields((BS >> qmode @ kerr).feedback(mem=qmode, state=Create(0)),
               Coherent((1., ))).sample(1, seed=0)
    with pytest.raises(NotImplementedError):
        Fields((Create(1) @ qmode >> BS).feedback(
            mem=qmode, state=Create(0))).sample(1, seed=0)
    assert repr(Fields(loop, Coherent((1., )))).startswith("Fields(")


def test_squeezed_quadratures_match_the_ket():
    levels, squeezing = 40, .4
    vector, = Squeezed((squeezing, )).inject(None, levels)
    lower = np.diag(np.sqrt(np.arange(1, levels)), 1)
    quadrature = (lower + lower.T) / 2
    spread = np.real(vector.conj() @ quadrature @ quadrature @ vector)
    assert np.isclose(spread, Squeezed((squeezing, )).quadratures()[1][0, 0])


def test_quadratures_of_a_product():
    means, covariance = (PhaseLocked((1 + 2j, )) @ Thermal((1., ))
                         @ Squeezed((0., ))).quadratures()
    assert means.tolist() == [1, 0, 0, 2, 0, 0]
    assert np.allclose(np.diag(covariance), [.25, .75, .25] * 2)
    with pytest.raises(NotImplementedError, match="not Gaussian"):
        Coherent((1., )).quadratures()


def test_gaussian_matches_the_exact_means():
    network = Interferometer(
        unitary_group.rvs(3, random_state=2), loop=2,
        source=Displaced(Thermal((.5, )), (1.5, )), transmissivity=.8,
        efficiency=.9)
    exact = np.real(network.transfer() @ network.source.field)
    gaussian = Gaussian(network)
    memory = gaussian.start()
    for _ in range(200):
        mean, memory = gaussian.tick(memory)
    assert np.allclose(mean, exact)
    samples = gaussian.sample(4000, burn_in=20, seed=0)
    assert np.allclose(np.mean(samples, axis=0), exact, atol=.04)
    with pytest.raises(ValueError, match="more modes"):
        network.padded(Fock((0, 0)))
    assert Gaussian(network) == eval(repr(Gaussian(network)))
    from optyx.channel import qmode
    from optyx.photonic import BS, Create
    lit = (Create(1) @ qmode >> BS).feedback(mem=qmode, state=Create(0))
    with pytest.raises(NotImplementedError, match="not the vacuum"):
        Interferometer.from_diagram(lit, source=Fock(())).padded(Fock(()))


def test_displaced_classical_light_through_the_fields():
    from optyx.channel import qmode
    from optyx.photonic import BS, Create
    loop = BS.feedback(mem=qmode, state=Create(0))
    light = Displaced(Thermal((.5, )), (1j, ))
    network = Interferometer.from_diagram(loop, source=light)
    exact = network.advance(network.correlations(), (), network.field())[1]
    samples = Fields(loop, light).sample(4000, burn_in=20, seed=0)
    assert abs(np.mean(samples) - exact[0]) < .06
    with pytest.raises(NotImplementedError, match="phase reference"):
        light.draw(None)
    assert light.pad(1) == Displaced(Thermal((.5, 0.)), (1j, 0))
    assert light @ light == Displaced(Thermal((.5, .5)), (1j, 1j))


def test_fields_through_swaps():
    from optyx.channel import Diagram, qmode
    from optyx.photonic import BS, Create
    loop = (BS @ qmode >> qmode @ Diagram.swap(qmode, qmode)).feedback(
        mem=qmode @ qmode, state=Create(0, 0))
    light = PhaseLocked((1, ))
    network = Interferometer.from_diagram(loop, source=light)
    exact = network.advance(network.correlations(), (), network.field())[1]
    samples = Fields(loop, light).sample(4000, burn_in=20, seed=0)
    assert abs(np.mean(samples) - exact[0]) < .05


def test_homodyne_needs_a_burn_in():
    from optyx.channel import qmode
    from optyx.photonic import BS, Create
    loop = BS.feedback(mem=qmode, state=Create(0))
    with pytest.raises(NotImplementedError, match="burn_in"):
        loop.sample(2, source=PhaseLocked((1, )), homodyne=True)


def test_shine_matches_the_fields():
    from optyx.channel import qmode
    from optyx.photonic import BS, Create, PhotonLoss
    loop = (BS >> qmode @ PhotonLoss(.7)).feedback(mem=qmode, state=Create(0))
    light = Thermal((2., ))
    network = Interferometer.from_diagram(loop, source=light)
    rng, field, counts = np.random.default_rng(0), np.zeros(1), []
    for _ in range(4000):
        count, field = network.shine(field, rng)
        counts.append(count[0])
    fields = Fields(loop, light).sample(4000, burn_in=20, seed=1)
    exact = network.advance(network.correlations())[1][0]
    assert abs(np.mean(counts[20:]) - exact) < .1
    assert abs(np.var(counts[20:]) - np.var(fields)) < .15 * np.var(fields)
