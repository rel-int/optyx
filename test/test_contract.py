"""Tests for backend-neutral tensor contraction."""

from contextlib import nullcontext

from cotengra import ReusableHyperCompressedOptimizer
import numpy as np
import pytest

from optyx import qubits
from optyx.core import contract
from optyx.core.contract import contract_tensor


@pytest.mark.parametrize("backend", [None, "numpy", "quimb"])
def test_exact_backends(backend):
    network = qubits.Ket(0).double().to_tensor()
    result = contract_tensor(network, backend=backend)
    assert np.allclose(result.array, [[1, 0], [0, 0]])


@pytest.mark.parametrize(("backend", "module"), [
    ("jax", "jax"),
    ("pytorch", "torch"),
])
def test_array_backends(backend, module):
    pytest.importorskip(module)
    network = qubits.Ket(0).double().to_tensor()
    result = contract_tensor(network, backend=backend)
    assert np.allclose(np.asarray(result.array), [[1, 0], [0, 0]])


def test_array_backend_materialises_spiders(monkeypatch):
    network = qubits.Ket(0).double().to_tensor()

    def backend_context(_backend=None):
        return nullcontext(np)

    monkeypatch.setattr(contract.tensor, "backend", backend_context)
    result = contract_tensor(network, backend="accelerator")
    assert np.allclose(result.array, [[1, 0], [0, 0]])


def test_compressed_quimb():
    network = (qubits.Ket(0) >> qubits.H()).double().to_tensor()
    exact = contract_tensor(network, backend="numpy")
    optimizer = ReusableHyperCompressedOptimizer(
        methods=["greedy"], max_repeats=1, progbar=False)
    compressed = contract_tensor(
        network, backend="quimb", optimize=optimizer, max_bond=2)
    assert np.allclose(compressed.array, exact.array)


def test_sum():
    zero = qubits.Ket(0).double().to_tensor()
    one = qubits.Ket(1).double().to_tensor()
    result = contract_tensor(zero + one, backend="quimb")
    assert np.allclose(result.array, np.eye(2))


def test_quimb_only_parameters():
    network = qubits.Ket(0).double().to_tensor()
    with pytest.raises(ValueError, match="backend='quimb'"):
        contract_tensor(network, backend="numpy", max_bond=2)


def test_reserved_quimb_parameters():
    network = qubits.Ket(0).double().to_tensor()
    with pytest.raises(ValueError, match="cannot override output_inds"):
        contract_tensor(
            network, backend="quimb", output_inds=())
