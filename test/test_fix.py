import numpy as np
import pytest
from cotengra import HyperCompressedOptimizer

from optyx import classical, photonic, qubits
from optyx.channel import (
    CQMap, Diagram, Discard, Ty, bit, qmode, qubit,
)
from optyx.core import diagram
from optyx.core.backends import DiscopyBackend, QuimbBackend


def source(initial_state=1, final_effect=None):
    """Reprepares the zero state at every time step, forgetting the memory."""
    return (Discard(qubit) @ qubits.Ket(0) @ qubits.Ket(0)).feedback(
        mem=qubit, initial_state=qubits.Ket(initial_state),
        final_effect=final_effect)


def rotation(phase, initial_state=0):
    """Rotates the memory then copies it, one copy out and one back in."""
    return (qubits.X(1, 1, phase) >> qubits.Z(1, 2)).feedback(
        mem=qubit, initial_state=qubits.Ket(initial_state))


def delay():
    """Outputs the previous memory and stores a fresh photon."""
    return (photonic.Create(1) @ qmode >> Diagram.swap(qmode, qmode)).feedback(
        mem=qmode, initial_state=photonic.Create(0))


def identity():
    """Copy the memory out without changing the memory."""
    return classical.CopyBit(2).feedback(
        mem=bit, initial_state=classical.Bit(0))


def flip():
    """Flip the memory, copy it out, and feed one copy back."""
    return (classical.Not() >> classical.CopyBit(2)).feedback(
        mem=bit, initial_state=classical.Bit(0))


class RecordingBackend(QuimbBackend):
    """A deterministic compressed backend recording per-call parameters."""

    def __init__(self):
        super().__init__(hyperoptimiser=HyperCompressedOptimizer())
        self.calls = []
        self.result = qubits.Ket(0).eval(DiscopyBackend())
        self.trace = (
            qubits.Ket(0) >> Discard(qubit)).eval(DiscopyBackend())

    def eval(self, diagram, **extra):
        assert isinstance(diagram, Diagram)
        self.calls.append((diagram.cod, self.contraction_params))
        return self.result if diagram.cod else self.trace


def test_one_step():
    assert source().one_step() == (
        Discard(qubit) @ qubits.Ket(0) @ qubits.Ket(0))
    assert delay().one_step() == (
        photonic.Create(1) @ qmode >> Diagram.swap(qmode, qmode))


def test_at_time_types():
    for n_steps in range(1, 4):
        assert source().at_time(n_steps).dom == Ty()
        assert source().at_time(n_steps).cod == qubit


def test_at_time_needs_initial_state():
    loop = (Discard(qubit) @ qubits.Ket(0) @ qubits.Ket(0)).feedback(
        mem=qubit)
    with pytest.raises(ValueError, match="initial_state"):
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
    {"cutoff": 0}, {"max_steps": 0}, {"max_chi": 0},
    {"tol": 0}, {"tol": np.inf}, {"tol": np.nan},
])
def test_fix_validates_parameters(kwargs):
    with pytest.raises(ValueError):
        source().fix(**kwargs)


def test_source_forgets_its_initial_state():
    expected = np.array([[1, 0], [0, 0]])
    eigen = source(initial_state=1).fix(method="eigen")
    power = source(initial_state=1).fix(n_steps=4, chi=8)
    assert np.allclose(eigen.density_matrix, expected, atol=1e-6)
    assert np.allclose(power.density_matrix, expected, atol=1e-6)


def test_final_effect_does_not_change_stationary_state():
    conditioned = source(final_effect=qubits.Bra(1))
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


def test_power_does_not_alias_period_two():
    with pytest.warns(UserWarning, match="n_steps"):
        result = flip().fix(chi=4, tol=1e-6, max_steps=8)
    expected = flip().at_time(8).eval().density_matrix
    assert np.allclose(result.density_matrix, expected)


def test_eigen_periodic_and_non_unique():
    assert np.allclose(
        flip().fix(method="eigen").density_matrix, [.5, .5])
    with pytest.raises(ValueError, match="not unique"):
        identity().fix(method="eigen")


def test_adaptive_parameters_are_independent():
    backend = RecordingBackend()
    source().fix(
        n_steps=2, chi=None, backend=backend,
        max_steps=1, max_chi=8)
    assert [params["max_bond"] for cod, params in backend.calls if cod] \
        == [4, 8]

    backend = RecordingBackend()
    source().fix(
        n_steps=None, chi=100, backend=backend,
        max_steps=4, max_chi=64)
    assert [params["max_bond"] for cod, params in backend.calls if cod] \
        == [100, 100]


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
    density_matrix = delay().fix(method="eigen", cutoff=2).density_matrix
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
    ).feedback(mem=qmode, initial_state=photonic.Create(0))
    density_matrix = sampler.fix(
        method="eigen", cutoff=5).density_matrix
    assert density_matrix.shape == (6, 6)
    assert np.isclose(np.trace(density_matrix), 1)


def test_stationary_state_convention_and_rank():
    transition = CQMap(
        "Transition",
        diagram.Box(
            "Transition", diagram.bit, diagram.bit,
            array=np.array([[.9, .1], [.4, .6]])),
        bit, bit)
    state = transition.stationary_state()
    assert np.allclose(state, [.8, .2])
    almost_identity = CQMap(
        "Almost identity",
        diagram.Box(
            "Almost identity", diagram.bit, diagram.bit,
            array=np.array([[1 - 1e-9, 1e-9],
                            [1e-9, 1 - 1e-9]])),
        bit, bit)
    assert np.allclose(
        almost_identity.stationary_state(), [.5, .5])
    with pytest.raises(ValueError, match="not unique"):
        Diagram.id(bit).stationary_state()


@pytest.mark.parametrize(("state", "message"), [
    (np.array([[1, 1], [0, 0]]), "not Hermitian"),
    (np.diag([2, -1]), "not positive"),
])
def test_stationary_state_validates_density_matrix(state, message):
    trace = np.array([1, 0, 0, 1])
    replacement = CQMap(
        "Replacement",
        diagram.Box(
            "Replacement", diagram.bit ** 2, diagram.bit ** 2,
            array=np.outer(trace, state.reshape(-1))),
        qubit, qubit)
    with pytest.raises(ValueError, match=message):
        replacement.stationary_state()


def test_stationary_state_guards():
    with pytest.raises(ValueError, match="without feedback"):
        identity().stationary_state()
    with pytest.raises(ValueError, match="endomorphisms"):
        qubits.Ket(0).stationary_state()
    with pytest.raises(ValueError, match="dimensions"):
        qubits.Id(1).stationary_state([2])
    with pytest.raises(ValueError, match="tol"):
        qubits.Id(1).stationary_state(tol=0)


def test_ty_double_exposes_cutoff_dimensions():
    doubled = (qubit @ qmode).double()
    assert [3 if ob.inside[0].name == "mode" else 2 for ob in doubled] \
        == [2, 2, 3, 3]
