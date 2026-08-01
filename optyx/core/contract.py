"""Backend-neutral contraction of DisCoPy tensor diagrams.

Functions
---------

.. autosummary::
    :nosignatures:
    :toctree:

    contract_tensor

``contract_tensor`` evaluates the output of ``Diagram.to_tensor`` without
coupling the diagram to an execution library. NumPy is the default exact
executor; JAX and PyTorch use the same exact semantics. Quimb accepts any
compatible Cotengra optimizer and optionally bounds intermediate bonds.
"""

from typing import Any

from cotengra import (
    HyperCompressedOptimizer,
    ReusableHyperCompressedOptimizer,
)
from discopy import tensor
import numpy as np
from quimb.tensor import Tensor


def _materialize_arrays(
        diagram: tensor.Diagram,
        backend: str,
        dtype: type = complex) -> tensor.Diagram:
    """Represent every box by a concrete array on one backend."""
    boxes = []
    with tensor.backend(backend) as array_module:
        for box in diagram.boxes:
            if isinstance(box, tensor.Spider):
                array = tensor.Tensor.spider_factory(
                    len(box.dom), len(box.cod), box.typ, box.phase).array
            else:
                array = box.array
            box = tensor.Box(box.name, box.dom, box.cod,
                             array_module.array(array, dtype=dtype))
            boxes.append(box)
    return tensor.Diagram.decode(
        diagram.dom, boxes=boxes, offsets=diagram.offsets, cod=diagram.cod)


def _contract_quimb(
        diagram: tensor.Diagram,
        optimize=None,
        max_bond: int = None,
        backend: str = None,
        tensor_backend: str = None,
        **params: Any) -> tensor.Tensor:
    """Contract with Quimb on one array backend."""
    reserved = {"output_inds"} & params.keys()
    if reserved:
        raise ValueError(
            "Contraction parameters cannot override "
            + ", ".join(sorted(reserved)) + ".")
    with tensor.backend(tensor_backend):
        network = diagram.to_quimb()
    if tensor_backend is not None:
        with tensor.backend(tensor_backend) as array_module:
            for quimb_tensor in network:
                data = quimb_tensor.data
                if isinstance(data, np.ndarray) and not data.flags.writeable:
                    data = data.copy()
                quimb_tensor.modify(data=array_module.array(
                    data, dtype=complex))
    else:
        for quimb_tensor in network:
            if quimb_tensor.data.dtype.kind in {"i", "u", "b"}:
                quimb_tensor.modify(data=quimb_tensor.data.astype(
                    np.complex128, copy=False))
    contract_params = {
        "output_inds": sorted(network.outer_inds()), **params}
    if backend is not None:
        contract_params["backend"] = backend
    if optimize is not None:
        contract_params["optimize"] = optimize
    compressed = max_bond is not None or isinstance(
        optimize,
        (ReusableHyperCompressedOptimizer, HyperCompressedOptimizer))
    if max_bond is not None:
        contract_params["max_bond"] = max_bond
    contract = network.contract_compressed if compressed else network.contract
    result = contract(**contract_params)
    array = result.data if isinstance(result, Tensor) else result
    with tensor.backend(tensor_backend):
        return tensor.Tensor(array, diagram.dom, diagram.cod)


def contract_tensor(
        diagram: tensor.Diagram,
        backend: str = None,
        optimize=None,
        max_bond: int = None,
        **params: Any) -> tensor.Tensor:
    """Contract a backend-neutral :class:`discopy.tensor.Diagram`.

    ``backend`` selects NumPy, JAX, PyTorch or Quimb arrays. JAX and PyTorch
    contractions preserve automatic differentiation. ``optimize`` may be any
    Quimb or Cotengra path optimizer; ``max_bond`` enables compressed
    contraction.

    Args:
        diagram: The tensor diagram to contract.
        backend: ``None``, ``"numpy"``, ``"jax"``, ``"pytorch"`` or
            ``"quimb"``.
        optimize: An optional Quimb or Cotengra path optimizer for Quimb,
            JAX or PyTorch.
        max_bond: An optional maximum intermediate bond dimension.
        params: Parameters forwarded to the selected contraction routine.

    Returns:
        The contracted tensor.

    >>> from optyx.qubits import Ket
    >>> network = Ket(0).double().to_tensor()
    >>> numpy_result = contract_tensor(network, backend="numpy")
    >>> quimb_result = contract_tensor(network, backend="quimb")
    >>> np.allclose(numpy_result.array, quimb_result.array)
    True

    Use ``backend="jax"`` or ``backend="pytorch"`` for differentiable
    accelerator arrays. Pass a Cotengra optimizer as ``optimize``.
    """
    if hasattr(diagram, "terms"):
        terms = [contract_tensor(
            term, backend, optimize, max_bond, **params)
                 for term in diagram.terms]
        array = sum((term.array for term in terms[1:]), terms[0].array)
        result_backend = None if backend == "quimb" else backend
        with tensor.backend(result_backend):
            return tensor.Tensor(array, diagram.dom, diagram.cod)
    if backend == "quimb":
        return _contract_quimb(diagram, optimize, max_bond, **params)
    if backend in ("jax", "pytorch"):
        dtype = params.pop("dtype", complex)
        diagram = _materialize_arrays(diagram, backend, dtype)
        array_backend = "torch" if backend == "pytorch" else backend
        return _contract_quimb(
            diagram, optimize, max_bond,
            backend=array_backend, tensor_backend=backend, **params)
    if optimize is not None or max_bond is not None:
        raise ValueError(
            "optimize and max_bond require backend='quimb', 'jax' "
            "or 'pytorch'.")
    dtype = params.pop("dtype", None)
    if backend not in (None, "numpy"):
        raise ValueError(f"Unknown tensor backend {backend!r}.")
    with tensor.backend(backend):
        return diagram.eval(**params) if dtype is None else diagram.eval(
            dtype=dtype, **params)
