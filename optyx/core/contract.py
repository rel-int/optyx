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


def _materialize_spiders(
        diagram: tensor.Diagram,
        dtype: type = complex) -> tensor.Diagram:
    """Replace structural spiders by boxes with concrete arrays."""
    boxes = []
    for box in diagram.boxes:
        if isinstance(box, tensor.Spider):
            spider = tensor.Tensor.spider_factory(
                len(box.dom), len(box.cod), box.typ, box.phase)
            box = tensor.Box(
                box.name, box.dom, box.cod,
                np.asarray(spider.array, dtype=dtype))
        boxes.append(box)
    return tensor.Diagram.decode(
        diagram.dom, boxes=boxes, offsets=diagram.offsets, cod=diagram.cod)


def _contract_quimb(
        diagram: tensor.Diagram,
        optimize=None,
        max_bond: int = None,
        **params: Any) -> tensor.Tensor:
    """Contract with Quimb and an optional Cotengra optimizer."""
    reserved = {"output_inds"} & params.keys()
    if reserved:
        raise ValueError(
            "Contraction parameters cannot override "
            + ", ".join(sorted(reserved)) + ".")
    network = diagram.to_quimb()
    for quimb_tensor in network:
        if quimb_tensor.data.dtype.kind in {"i", "u", "b"}:
            quimb_tensor.modify(data=quimb_tensor.data.astype(
                np.complex128, copy=False))
    contract_params = {
        "output_inds": sorted(network.outer_inds()), **params}
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
    return tensor.Box("Result", diagram.dom, diagram.cod, array)


def contract_tensor(
        diagram: tensor.Diagram,
        backend: str = None,
        optimize=None,
        max_bond: int = None,
        **params: Any) -> tensor.Tensor:
    """Contract a backend-neutral :class:`discopy.tensor.Diagram`.

    ``backend`` selects exact DisCoPy evaluation with NumPy, JAX or PyTorch.
    The value ``"quimb"`` selects Quimb, where ``optimize`` may be any
    Cotengra optimizer and ``max_bond`` enables compressed contraction.

    Args:
        diagram: The tensor diagram to contract.
        backend: ``None``, ``"numpy"``, ``"jax"``, ``"pytorch"`` or
            ``"quimb"``.
        optimize: An optional Quimb or Cotengra path optimizer.
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

    Use ``backend="jax"`` or ``backend="pytorch"`` for accelerator arrays.
    Pass a Cotengra optimizer as ``optimize`` with ``backend="quimb"``.
    """
    if hasattr(diagram, "terms"):
        terms = [contract_tensor(
            term, backend, optimize, max_bond, **params)
                 for term in diagram.terms]
        array = sum((term.array for term in terms[1:]), terms[0].array)
        return tensor.Box("Result", diagram.dom, diagram.cod, array)
    if backend == "quimb":
        return _contract_quimb(diagram, optimize, max_bond, **params)
    if optimize is not None or max_bond is not None:
        raise ValueError(
            "optimize and max_bond require backend='quimb'.")
    dtype = params.pop("dtype", None)
    if backend not in (None, "numpy"):
        dtype = dtype or complex
        diagram = _materialize_spiders(diagram, dtype)
    with tensor.backend(backend):
        return diagram.eval(**params) if dtype is None \
            else diagram.eval(dtype=dtype, **params)
