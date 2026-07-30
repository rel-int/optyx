import numpy as np
import pytest

from optyx import photonic, qubits
from optyx.channel import Diagram, Discard, Ty, qmode, qubit
from optyx.utils.misc import distance, fixed_point


def source(initial_state=1):
    """Reprepares the zero state at every time step, forgetting the memory."""
    return (Discard(qubit) @ qubits.Ket(0) @ qubits.Ket(0)).feedback(
        mem=qubit, initial_state=qubits.Ket(initial_state))


def rotation(phase, initial_state=0):
    """Rotates the memory then copies it, one copy out and one back in."""
    return (qubits.X(1, 1, phase) >> qubits.Z(1, 2)).feedback(
        mem=qubit, initial_state=qubits.Ket(initial_state))


def delay():
    """Outputs the previous memory and stores a fresh photon."""
    return (photonic.Create(1) @ qmode >> Diagram.swap(qmode, qmode)).feedback(
        mem=qmode, initial_state=photonic.Create(0))


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


def test_source_forgets_its_initial_state():
    expected = np.array([[1, 0], [0, 0]])
    for method in ("eigen", "power"):
        result = source(initial_state=1).fix(method=method, n_steps=4, chi=8)
        assert np.allclose(result.density_matrix, expected, atol=1e-6)


def test_eigen_against_at_time():
    """The eigensolve agrees with the state after enough time steps."""
    diagram = rotation(0.25)
    stationary = diagram.fix(method="eigen").density_matrix
    late = diagram.at_time(12).eval().density_matrix
    assert distance(stationary, late) < 1e-2


def test_power_converges_to_eigen():
    diagram = rotation(0.25)
    stationary = diagram.fix(method="eigen").density_matrix
    power = diagram.fix(n_steps=10, chi=16).density_matrix
    assert distance(stationary, power) < 1e-2


def test_fix_is_a_state():
    """The stationary state is a normalised density matrix."""
    density_matrix = rotation(0.25).fix(method="eigen").density_matrix
    assert np.isclose(np.trace(density_matrix), 1)
    assert np.allclose(density_matrix, density_matrix.conjugate().T)
    assert min(np.linalg.eigvalsh(density_matrix)) > -1e-6


def test_adaptive_defaults():
    """Doubling the depth and the bond dimension reaches the same state."""
    diagram = rotation(0.25)
    assert distance(
        diagram.fix(tol=1e-4).density_matrix,
        diagram.fix(method="eigen").density_matrix) < 1e-2


def test_warns_when_it_does_not_converge():
    """An unreachable tolerance stops at the cap with a warning."""
    with pytest.warns(UserWarning, match="did not converge"):
        rotation(0.25).fix(chi=4, tol=0, max_steps=8)


def test_photonic_delay_line():
    """A delay line fed one photon per step stabilises on a single photon."""
    density_matrix = delay().fix(method="eigen", cutoff=2).density_matrix
    assert np.allclose(density_matrix, [[0, 0], [0, 1]], atol=1e-6)


def test_fixed_point_helper():
    assert np.isclose(np.linalg.norm(fixed_point(np.eye(4))), 1)


def test_distance_helper():
    assert np.isclose(distance(np.eye(2), np.eye(2)), 0)
    assert np.isclose(distance(np.zeros(4), np.ones(4)), 2)
