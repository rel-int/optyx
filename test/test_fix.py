import warnings

import numpy as np
import pytest
from cotengra import HyperCompressedOptimizer

from optyx import channel, classical, photonic, qubits
from optyx.channel import (
    Diagram, Discard, Scalar, Ty, bit, qmode, qubit,
)
from optyx.core.backends import DiscopyBackend, QuimbBackend


def source(photons=1, effect=None):
    """Reprepares the zero state at every time step, forgetting the memory."""
    return (Discard(qubit) @ qubits.Ket(0) @ qubits.Ket(0)).feedback(
        mem=qubit, state=qubits.Ket(photons), effect=effect)


def rotation(phase, photons=0):
    """Rotates the memory then copies it, one copy out and one back in."""
    return (qubits.X(1, 1, phase) >> qubits.Z(1, 2)).feedback(
        mem=qubit, state=qubits.Ket(photons))


def delay():
    """Outputs the previous memory and stores a fresh photon."""
    return (photonic.Create(1) @ qmode >> Diagram.swap(qmode, qmode)).feedback(
        mem=qmode, state=photonic.Create(0))


def sampler(reflectivity=.06):
    """A feedback boson sampler: one fresh photon per tick into an MZI."""
    return (photonic.Create(1) @ qmode >> photonic.MZI(reflectivity, 0)
            ).feedback(mem=qmode, state=photonic.Create(0))


def identity():
    """Copy the memory out without changing the memory."""
    return classical.CopyBit(2).feedback(
        mem=bit, state=classical.Bit(0))


def flip():
    """Flip the memory, copy it out, and feed one copy back."""
    return (classical.Not() >> classical.CopyBit(2)).feedback(
        mem=bit, state=classical.Bit(0))


class RecordingBackend(QuimbBackend):
    """A deterministic compressed backend recording per-call parameters."""

    def __init__(self):
        super().__init__(hyperoptimiser=HyperCompressedOptimizer())
        self.calls = []
        self.result = source().at_time(2).eval(DiscopyBackend())

    def eval(self, diagram, **extra):
        assert isinstance(diagram, Diagram)
        self.calls.append(
            (diagram.cod, {**self.contraction_params, **extra}))
        return self.result


def test_one_step():
    assert source().one_step() == (
        Discard(qubit) @ qubits.Ket(0) @ qubits.Ket(0))
    assert delay().one_step() == (
        photonic.Create(1) @ qmode >> Diagram.swap(qmode, qmode))


def test_at_time_types():
    for n_steps in range(4):
        assert source().at_time(n_steps).dom == Ty()
        assert source().at_time(n_steps).cod == qubit


def test_at_time_needs_a_state():
    loop = (Discard(qubit) @ qubits.Ket(0) @ qubits.Ket(0)).feedback(
        mem=qubit)
    with pytest.raises(ValueError, match="needs a state"):
        loop.at_time(2)


def test_unroll_takes_only_a_number_of_steps():
    """The boundaries belong to `feedback`, so `unroll` has no other
    parameter; `one_step` and `at_time` use `unroll_with_boundaries`."""
    with pytest.raises(TypeError):
        source().unroll(1, state=None)
    assert source().unroll(1) == source().unroll_with_boundaries(1)


@pytest.mark.parametrize("solver", ["fix", "eigen_fix"])
def test_fix_guards(solver):
    with pytest.raises(ValueError, match="domain"):
        getattr(Diagram.id(qubit) @ source(), solver)()
    with pytest.raises(ValueError, match="no feedback loop"):
        getattr(qubits.Ket(0), solver)()


@pytest.mark.parametrize("solver", ["fix", "eigen_fix"])
@pytest.mark.parametrize("kwargs", [
    {"chi": -1}, {"chi": True}, {"chi": 1.5},
    {"tol": 0}, {"tol": np.inf}, {"tol": np.nan},
    {"loss": 1}, {"loss": -.1},
])
def test_fix_validates_parameters(solver, kwargs):
    with pytest.raises(ValueError):
        getattr(source(), solver)(**kwargs)


def test_source_forgets_its_initial_state():
    expected = np.array([[1, 0], [0, 0]])
    assert np.allclose(
        source(photons=1).eigen_fix().density_matrix, expected, atol=1e-6)
    assert np.allclose(
        source(photons=1).fix(tol=1e-4, chi=8).density_matrix,
        expected, atol=1e-6)


def test_the_effect_does_not_change_the_stationary_state():
    conditioned, plain = source(effect=qubits.Bra(1)), source()
    assert conditioned.at_time(2) == plain.at_time(2)
    for solve in (lambda loop: loop.eigen_fix(),
                  lambda loop: loop.fix(tol=1e-4, chi=4)):
        assert np.allclose(
            solve(conditioned).density_matrix, solve(plain).density_matrix)


def test_eigen_against_at_time():
    """The eigensolve agrees with the state after enough time steps."""
    stationary = rotation(0.25).eigen_fix().density_matrix
    late = rotation(0.25).at_time(12).eval().density_matrix
    assert np.linalg.norm(stationary - late) < 1e-2


def test_power_converges_to_eigen():
    """Doubling the depth reaches the state the eigensolve computes."""
    diagram_ = rotation(0.25)
    assert np.linalg.norm(
        diagram_.fix(tol=1e-4).density_matrix
        - diagram_.eigen_fix().density_matrix) < 1e-2


def test_fix_is_a_state():
    """The stationary state is a normalised density matrix."""
    density_matrix = rotation(0.25).eigen_fix().density_matrix
    assert np.isclose(np.trace(density_matrix), 1)
    assert np.allclose(density_matrix, density_matrix.conjugate().T)
    assert min(np.linalg.eigvalsh(density_matrix)) > -1e-6


def test_power_normalises_the_contracted_state():
    backend = RecordingBackend()
    backend.result.tensor.array *= 1e-4
    result = source().fix(tol=1e-4, chi=4, backend=backend)
    assert np.allclose(result.density_matrix, [[1, 0], [0, 0]])


def test_power_does_not_alias_period_two(monkeypatch):
    """A period-two loop never converges, so the depth cap warns."""
    monkeypatch.setattr(channel, "MAX_UNROLL", 8)
    with pytest.warns(UserWarning, match="did not converge"):
        result = flip().fix(chi=4, tol=1e-6)
    expected = flip().at_time(7).eval().density_matrix
    assert np.allclose(result.density_matrix, expected)


def test_eigen_periodic_and_non_unique():
    assert np.allclose(flip().eigen_fix().density_matrix, [.5, .5])
    with pytest.raises(ValueError, match="not unique"):
        identity().eigen_fix()


def test_chi_bounds_the_bond_and_warns_past_the_budget():
    """`chi` bounds the bonds whenever it is given — lossless while the
    bonds fit — and warns only when the photon budget outgrows it, which is
    when the state itself no longer fits."""
    backend = RecordingBackend()
    source().fix(tol=1e-4, chi=8, backend=backend)
    assert all(params["max_bond"] == 8 for _, params in backend.calls)

    backend = RecordingBackend()
    source().fix(tol=1e-4, chi=None, backend=backend)
    assert all("max_bond" not in params for _, params in backend.calls)

    backend = RecordingBackend()
    with pytest.warns(UserWarning, match="needs dimension 2 but chi is 1"):
        source().fix(tol=1e-4, chi=1, backend=backend)
    assert [params["max_bond"] for _, params in backend.calls] == [1, 1]


def test_fix_truncates_a_growing_photon_budget():
    """A loop which accumulates photons outgrows a small `chi`: at the
    depth a loss of a half buys, this one needs eight dimensions, so
    `chi=6` warns that the state is truncated while `chi=8` holds the
    budget and stays silent."""
    assert [sampler().at_time(n).truncation_dimensions()[0]
            for n in range(4)] == [2, 3, 4, 5]
    with pytest.warns(UserWarning, match="needs dimension 8 but chi is 6"):
        truncated = sampler().fix(chi=6, loss=.5, tol=1e-2)
    assert np.isclose(np.trace(truncated.density_matrix), 1)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sampler().fix(chi=8, loss=.5, tol=1e-2)


def test_the_search_stops_where_the_budget_outgrows_chi():
    """On a lossless loop that accumulates photons the search cannot double
    forever — the boxes of the network grow with the budget, not just its
    bonds. The search stops at the deepest depth `chi` holds exactly and
    returns that state as the truncated approximation."""
    with pytest.warns(UserWarning, match="outgrows chi past depth 4"):
        truncated = sampler().fix(chi=5, tol=1e-6)
    expected = sampler().at_time(3).eval().density_matrix
    trace = np.trace(expected)
    assert np.allclose(
        truncated.density_matrix, expected / trace, atol=1e-6)


def test_loss_gives_a_depth_instead_of_a_search():
    """A lossy loop knows its depth without contracting anything."""
    assert source().unroll_depth(1e-6, loss=.5) == 20
    with pytest.raises(ValueError, match="loss in"):
        source().unroll_depth(1e-6, loss=0)
    backend = RecordingBackend()
    source().fix(chi=4, loss=.9, tol=1e-2, backend=backend)
    assert len(backend.calls) == 1


def test_backend_uses_existing_interface():
    with pytest.raises(ValueError, match="AbstractBackend"):
        source().fix(chi=4, backend=object())


def test_power_supports_numpy_tensor_functor():
    result = source().fix(tol=1e-4, chi=4, backend=DiscopyBackend())
    assert np.allclose(result.density_matrix, [[1, 0], [0, 0]])


def test_power_supports_exact_quimb():
    result = source().fix(tol=1e-4, chi=4, backend=QuimbBackend())
    assert np.allclose(result.density_matrix, [[1, 0], [0, 0]])


def test_photonic_delay_line():
    """A delay line fed one photon per step stabilises on a single photon."""
    density_matrix = delay().eigen_fix(chi=2).density_matrix
    assert np.allclose(density_matrix, [[0, 0], [0, 1]], atol=1e-6)


def test_eigen_loss_channel_on_every_optical_memory_wire():
    """Half the memory lost per round trip halves the delay line's photon:
    it stores a fresh photon each tick, which survives with probability a
    half, so the stationary memory is an even mixture of zero and one."""
    assert np.allclose(
        delay().eigen_fix(chi=3, loss=.5).density_matrix,
        np.diag([.5, .5, 0]), atol=1e-6)


def test_eigen_loss_leaves_classical_memory_alone():
    """There is no photon to lose in a `bit`, so the loss is the identity
    there and the periodic loop keeps its even stationary state."""
    assert np.allclose(flip().eigen_fix(loss=.5).density_matrix, [.5, .5])


def test_eigen_finds_its_own_cutoff():
    """Regression for the bug reported on PR #15: the cutoff search checked
    causality of the raw, unprojected transfer map, which is causal at every
    dimension whenever the step creates photons, so it always returned two
    and `eigen_fix` failed with no explicit `chi`. The search now measures
    the projected operator, and starts from the photon budget rather than
    from two."""
    loop = sampler()
    step = loop.one_step()
    transfer = step >> Discard(loop.cod) @ Diagram.id(
        step.cod[len(loop.cod):])
    assert transfer.truncation_dimensions() == [3, 3]
    density_matrix = loop.eigen_fix().density_matrix
    assert density_matrix.shape[0] > 3
    assert np.isclose(np.trace(density_matrix), 1)


def test_eigen_boson_sampler_truncates_memory_output():
    """Fresh photons can increase the untruncated output dimension."""
    reflectivity = .01
    unitary = np.array([
        [np.sqrt(reflectivity), np.sqrt(1 - reflectivity)],
        [np.sqrt(1 - reflectivity), -np.sqrt(reflectivity)],
    ])
    loop = (
        photonic.Create(1) @ qmode
        >> photonic.Gate(unitary, 2, 2, "U")
    ).feedback(mem=qmode, state=photonic.Create(0))
    density_matrix = loop.eigen_fix(chi=5).density_matrix
    assert density_matrix.shape == (6, 6)
    assert np.isclose(np.trace(density_matrix), 1)


def test_normalisation_is_the_trace_of_a_state():
    assert np.isclose(qubits.Ket(0).normalisation(), 1)
    assert np.isclose((Scalar(.5) @ qubits.Ket(0)).normalisation(), .25)
    with pytest.raises(ValueError, match="provide a state over it"):
        Diagram.id(qubit).normalisation()
    assert np.isclose(
        (qubits.Ket(0) >> Diagram.id(qubit)).normalisation(), 1)


def test_truncation_dimensions_are_the_photon_budget():
    """A photon only reaches the wires downstream of where it enters, so the
    budget differs per wire even when every wire is beam-split."""
    chain = photonic.Create(1) @ photonic.Create(0) >> photonic.BS
    for _ in range(3):
        tail = photonic.Id(1) @ photonic.Create(1) >> photonic.BS
        chain = chain >> photonic.Id(len(chain.cod) - 1) @ tail
    assert chain.truncation_dimensions() == [2, 2, 3, 3, 4, 4, 5, 5, 5, 5]
    assert qubits.Z(1, 2).truncation_dimensions() == [2, 2, 2, 2]
