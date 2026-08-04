import numpy as np
import pytest

from optyx.core.diagram import Box, Ty, bit, mode


def test_array_state_infers_dims_from_array_shape():
    """optyx#28: a state (empty dom) with a mode cod wire used to be
    hardcoded to Dim(2) regardless of the array it carries."""
    state = Box("rho", Ty(), mode, array=np.array([1., 0., 0.]))
    tensor_box = state.to_tensor([])
    assert tensor_box.cod == tensor_box.cod.__class__(3)
    assert np.allclose(tensor_box.eval().array, [1., 0., 0.])


def test_array_endomorphism_respects_input_dims():
    """optyx#28 symptom B: dom/cod stayed Dim(2) even when the caller
    asked for a mode wire at a higher cutoff."""
    box = Box("f", mode, mode, array=np.eye(3))
    tensor_box = box.to_tensor([3])
    assert tensor_box.dom == tensor_box.dom.__class__(3)
    assert tensor_box.cod == tensor_box.cod.__class__(3)
    assert np.allclose(tensor_box.eval().array, np.eye(3))


def test_array_bit_wires_stay_at_two():
    state = Box("psi", Ty(), bit, array=np.array([1., 0.]))
    tensor_box = state.to_tensor([])
    assert tensor_box.cod == tensor_box.cod.__class__(2)


def test_array_size_mismatch_raises():
    box = Box("f", mode, mode, array=np.eye(3))
    with pytest.raises(ValueError):
        box.to_tensor([2])
