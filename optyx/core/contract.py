"""Backend-neutral contraction of DisCoPy tensor diagrams.

Functions
---------

.. autosummary::
    :nosignatures:
    :toctree:

    contract_tensor

``contract_tensor`` evaluates the output of ``Diagram.to_tensor`` or its
combinatorial map without coupling the network to an execution library.
NumPy is the default exact executor; JAX and PyTorch preserve accelerator
arrays and gradients. Quimb accepts any compatible Cotengra optimizer and
optionally bounds intermediate bonds.
"""

from typing import Any

from cotengra import (
    HyperCompressedOptimizer,
    ReusableHyperCompressedOptimizer,
)
from discopy import tensor
import numpy as np
from quimb.tensor import Tensor, TensorNetwork


def _to_quimb(diagram):
    """Translate a tensor diagram or combinatorial map to Quimb."""
    if not isinstance(diagram, tensor.CMap):
        return diagram.to_quimb()
    wires, fresh = {}, iter(range(len(diagram.ports) * 2))
    for source, target in enumerate(diagram.edges):
        if source <= target:
            wire = next(fresh)
            wires[source] = wires[target] = f"i{wire}"
    tensors, output = [], []
    for port in range(len(diagram.dom)):
        external = f"o{next(fresh)}"
        dimension = int(np.prod(diagram.ports[port].obj.inside))
        tensors.append(Tensor(
            np.eye(dimension), inds=(external, wires[port])))
        output.append(external)
    start = len(diagram.dom)
    for box in diagram.boxes:
        arity, coarity = len(box.dom), len(box.cod)
        ports = list(range(start, start + arity)) + list(reversed(range(
            start + arity, start + arity + coarity)))
        tensors.append(Tensor(box.eval().array, inds=tuple(
            wires[port] for port in ports)))
        start += arity + coarity
    for port in range(
            len(diagram.ports) - len(diagram.cod), len(diagram.ports)):
        external = f"o{next(fresh)}"
        dimension = int(np.prod(diagram.ports[port].obj.inside))
        tensors.append(Tensor(
            np.eye(dimension), inds=(wires[port], external)))
        output.append(external)
    for loop in diagram.loops:
        wire = f"i{next(fresh)}"
        dimension = int(np.prod(loop.inside))
        tensors.append(Tensor(np.eye(dimension), inds=(wire, wire)))
    return TensorNetwork(tensors), output


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
    if isinstance(diagram, tensor.CMap):
        return tensor.CMap(
            diagram.dom, diagram.cod, tuple(boxes), diagram.edges,
            offsets=diagram.offsets, loops=diagram.loops)
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
        network = _to_quimb(diagram)
    if isinstance(network, tuple):
        network, output_inds = network
    else:
        output_inds = sorted(network.outer_inds())
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
    contract_params = {"output_inds": output_inds, **params}
    if optimize is not None:
        contract_params["optimize"] = optimize
    compressed = max_bond is not None or isinstance(
        optimize,
        (ReusableHyperCompressedOptimizer, HyperCompressedOptimizer))
    if max_bond is not None:
        contract_params["max_bond"] = max_bond
    contract = network.contract_compressed if compressed else network.contract
    result = contract(**contract_params)
    if isinstance(result, TensorNetwork):
        result = result.contract(
            output_inds=sorted(result.outer_inds()))
    array = result.data if isinstance(result, Tensor) else result
    with tensor.backend(tensor_backend):
        return tensor.Tensor(array, diagram.dom, diagram.cod)


def contract_tensor(
        diagram: tensor.Diagram,
        backend: str = None,
        optimize=None,
        max_bond: int = None,
        **params: Any) -> tensor.Tensor:
    """Contract a backend-neutral tensor diagram or combinatorial map.

    ``backend`` selects NumPy, JAX, PyTorch or Quimb arrays. JAX and PyTorch
    contractions preserve automatic differentiation. ``optimize`` may be any
    Quimb or Cotengra path optimizer; ``max_bond`` enables compressed
    contraction.

    Args:
        diagram: The :class:`discopy.tensor.Diagram` or
            :class:`discopy.tensor.CMap` to contract.
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
    if isinstance(diagram, tensor.CMap) and backend in (None, "numpy"):
        dtype = params.pop("dtype", complex)
        diagram = _materialize_arrays(diagram, "numpy", dtype)
        return _contract_quimb(
            diagram, optimize, max_bond, **params)
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
