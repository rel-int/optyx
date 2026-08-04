import numpy as np
import pytest
from cotengra import HyperCompressedOptimizer

from optyx import channel, classical, photonic, qubits
from optyx.channel import (
    CQMap, Diagram, Discard, Scalar, Ty, bit, qmode, qubit,
)
from optyx.core import diagram
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


def test_fix_guards():
    with pytest.raises(ValueError, match="domain"):
        (Diagram.id(qubit) @ source()).fix()
    with pytest.raises(ValueError, match="no feedback loop"):
        qubits.Ket(0).fix()
    with pytest.raises(ValueError, match="Unknown method"):
        source().fix(method="magic")
    with pytest.raises(ValueError, match="only used"):
        source().fix(method="eigen", backend=RecordingBackend())


@pytest.mark.parametrize("kwargs", [
    {"n_steps": 0}, {"n_steps": 1.5}, {"chi": -1}, {"chi": True},
    {"tol": 0}, {"tol": np.inf}, {"tol": np.nan},
    {"loss": 1}, {"loss": -.1},
])
def test_fix_validates_parameters(kwargs):
    with pytest.raises(ValueError):
        source().fix(**kwargs)


def test_source_forgets_its_initial_state():
    expected = np.array([[1, 0], [0, 0]])
    eigen = source(photons=1).fix(method="eigen")
    power = source(photons=1).fix(n_steps=4, chi=8)
    assert np.allclose(eigen.density_matrix, expected, atol=1e-6)
    assert np.allclose(power.density_matrix, expected, atol=1e-6)


def test_the_effect_does_not_change_the_stationary_state():
    conditioned = source(effect=qubits.Bra(1))
    plain = source()
    assert conditioned.at_time(2) == plain.at_time(2)
    for method, kwargs in (("eigen", {}),
                           ("power", {"n_steps": 2, "chi": 4})):
        lhs = conditioned.fix(method=method, **kwargs).density_matrix
        rhs = plain.fix(method=method, **kwargs).density_matrix
        assert np.allclose(lhs, rhs)


def test_eigen_against_at_time():
    """The eigensolve agrees with the state after enough time steps."""
    diagram = rotation(0.25)
    stationary = diagram.fix(method="eigen").density_matrix
    late = diagram.at_time(12).eval().density_matrix
    assert np.linalg.norm(stationary - late) < 1e-2


def test_power_converges_to_eigen():
    diagram = rotation(0.25)
    stationary = diagram.fix(method="eigen").density_matrix
    power = diagram.fix(n_steps=10, chi=16).density_matrix
    assert np.linalg.norm(stationary - power) < 1e-2


def test_fix_is_a_state():
    """The stationary state is a normalised density matrix."""
    density_matrix = rotation(0.25).fix(method="eigen").density_matrix
    assert np.isclose(np.trace(density_matrix), 1)
    assert np.allclose(density_matrix, density_matrix.conjugate().T)
    assert min(np.linalg.eigvalsh(density_matrix)) > -1e-6


def test_adaptive_defaults():
    """Doubling the depth and the bond dimension reaches the same state."""
    diagram = rotation(0.25)
    assert np.linalg.norm(
        diagram.fix(tol=1e-4).density_matrix
        - diagram.fix(method="eigen").density_matrix) < 1e-2


def test_power_normalises_the_contracted_state():
    backend = RecordingBackend()
    backend.result.tensor.array *= 1e-4
    result = source().fix(
        n_steps=2, chi=4, tol=1e-4, backend=backend)
    assert np.allclose(result.density_matrix, [[1, 0], [0, 0]])
    assert len(backend.calls) == 1


def test_power_does_not_alias_period_two(monkeypatch):
    """A period-two loop never converges, so the depth cap warns."""
    monkeypatch.setattr(channel, "MAX_UNROLL", 8)
    with pytest.warns(UserWarning, match="did not converge"):
        result = flip().fix(chi=4, tol=1e-6)
    expected = flip().at_time(7).eval().density_matrix
    assert np.allclose(result.density_matrix, expected)


def test_eigen_periodic_and_non_unique():
    assert np.allclose(
        flip().fix(method="eigen").density_matrix, [.5, .5])
    with pytest.raises(ValueError, match="not unique"):
        identity().fix(method="eigen")


def test_chi_bounds_the_bond_and_none_contracts_exactly():
    """chi is the truncation dimension: absent means no truncation."""
    backend = RecordingBackend()
    source().fix(n_steps=2, chi=8, backend=backend)
    assert [params["max_bond"] for cod, params in backend.calls if cod] \
        == [8]

    backend = RecordingBackend()
    source().fix(n_steps=2, chi=None, backend=backend)
    assert all(
        "max_bond" not in params for cod, params in backend.calls)


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
        source().fix(n_steps=2, chi=4, backend=object())


def test_power_supports_numpy_tensor_functor():
    result = source().fix(
        n_steps=2, chi=4, backend=DiscopyBackend())
    assert np.allclose(result.density_matrix, [[1, 0], [0, 0]])


def test_power_supports_exact_quimb():
    result = source().fix(
        n_steps=2, chi=4, backend=QuimbBackend())
    assert np.allclose(result.density_matrix, [[1, 0], [0, 0]])


def test_photonic_delay_line():
    """A delay line fed one photon per step stabilises on a single photon."""
    density_matrix = delay().fix(method="eigen", chi=2).density_matrix
    assert np.allclose(density_matrix, [[0, 0], [0, 1]], atol=1e-6)


def test_eigen_boson_sampler_truncates_memory_output():
    """Fresh photons can increase the untruncated output dimension."""
    reflectivity = .01
    unitary = np.array([
        [np.sqrt(reflectivity), np.sqrt(1 - reflectivity)],
        [np.sqrt(1 - reflectivity), -np.sqrt(reflectivity)],
    ])
    sampler = (
        photonic.Create(1) @ qmode
        >> photonic.Gate(unitary, 2, 2, "U")
    ).feedback(mem=qmode, state=photonic.Create(0))
    density_matrix = sampler.fix(
        method="eigen", chi=5).density_matrix
    assert density_matrix.shape == (6, 6)
    assert np.isclose(np.trace(density_matrix), 1)


def test_normalisation_is_the_trace_of_a_state():
    assert np.isclose(qubits.Ket(0).normalisation(), 1)
    assert np.isclose((Scalar(.5) @ qubits.Ket(0)).normalisation(), .25)
    with pytest.raises(ValueError, match="needs a state"):
        Diagram.id(qubit).normalisation()
    assert np.isclose(
        Diagram.id(qubit).normalisation([[.5, 0], [0, .5]]), 1)


def test_truncation_dimensions_are_the_photon_budget():
    """A photon only reaches the wires downstream of where it enters, so the
    budget differs per wire even when every wire is beam-split."""
    chain = photonic.Create(1) @ photonic.Create(0) >> photonic.BS
    for _ in range(3):
        tail = photonic.Id(1) @ photonic.Create(1) >> photonic.BS
        chain = chain >> photonic.Id(len(chain.cod) - 1) @ tail
    assert chain.truncation_dimensions() == [2, 2, 3, 3, 4, 4, 5, 5, 5, 5]
    assert qubits.Z(1, 2).truncation_dimensions() == [2, 2, 2, 2]
