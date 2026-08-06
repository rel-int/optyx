import warnings

import numpy as np
import pytest

from discopy.utils import AxiomError

from optyx.channel import Diagram, Ty, qubit, qmode
from optyx.core.backends import DiscopyBackend
from optyx.interaction import Box, CMap
from optyx.photonic import BS, Create
from optyx.qubits import Bra, Ket, Scalar, X, Z


def cnot():
    return Z(1, 2) @ qubit >> qubit @ X(2, 1) @ Scalar(2 ** 0.5)


def box(name="f"):
    return Box(name, qubit, qubit, cnot())


def readout(name="m"):
    return Box(name, Ty(), Ty(), Z(1, 2), memory=qubit, prediction=qubit)


def test_box_axioms():
    with pytest.raises(AxiomError):
        Box("f", qubit, qubit, Diagram.id(qubit))
    with pytest.raises(AxiomError):
        Box("f", qmode, qmode, cnot())
    with pytest.raises(AxiomError):
        Box("f", qubit, qubit, cnot(), memory=qubit)
    with pytest.raises(AxiomError):
        Box("f", Ty(), Ty(), Z(1, 2), memory=qubit, prediction=qmode)


def test_box_ports():
    assert box().ports == qubit ** 2
    assert Box("g", Ty(), qubit ** 2, Diagram.id(qubit ** 2)).ports \
        == qubit ** 2
    assert readout().ports == Ty()


def test_box_equality():
    assert box() == box() and box() != box("g")
    assert readout() == readout() and readout() != box("m")
    assert Box("m", Ty(), Ty(), Diagram.id(qubit), memory=qubit) \
        != Box("m", Ty(), Ty(), Ket(0), prediction=qubit)


def test_edge_axioms():
    with pytest.raises(AxiomError):
        CMap([box()], [((0, 0), (0, 5))])
    with pytest.raises(AxiomError):
        CMap([box(), box()], [((0, 1), (1, 0)), ((0, 1), (1, 1))])
    with pytest.raises(AxiomError):
        CMap([box(), Box("g", qmode, qmode, BS)], [((0, 0), (1, 0))])


def test_memory_is_not_a_port():
    with pytest.raises(AxiomError, match="not a port"):
        CMap([readout(), readout()], [((0, 0), (1, 0))])


def test_boundary_and_memory():
    cmap = CMap([box(), box()], [((0, 1), (1, 0))])
    assert cmap.memory == qubit ** 2
    assert cmap.boundary == [(0, 0), (1, 1)]
    assert cmap.dom == cmap.cod == qubit ** 2


def test_no_edges_is_all_boundary():
    cmap = CMap([box()], [])
    assert cmap.memory == Ty()
    assert cmap.dom == qubit ** 2
    assert cmap.protocol.mem == Ty()
    assert cmap.protocol.one_step() == cmap.parallel


def test_protocol_types():
    cmap = CMap([box()], [((0, 0), (0, 1))])
    assert cmap.protocol.dom == cmap.dom == Ty()
    assert cmap.protocol.mem == cmap.memory == qubit ** 2


def test_memory_and_prediction_types():
    cmap = CMap([box(), readout()], [((0, 0), (0, 1))])
    assert cmap.dom == cmap.cod == Ty()
    assert cmap.memory == qubit ** 3
    assert cmap.prediction == qubit
    assert cmap.memories == [(1, 0)]
    assert cmap.predictions == [(1, 1)]
    assert cmap.protocol.dom == Ty()
    assert cmap.protocol.cod == qubit
    assert cmap.protocol.mem == qubit ** 3


def test_unroll_types():
    cmap = CMap([box(), box()], [((0, 1), (1, 0))])
    for n_steps in (0, 1, 2):
        unrolled = cmap.unroll(n_steps)
        assert unrolled.dom == cmap.dom ** (n_steps + 1) @ cmap.memory
        assert unrolled.cod == cmap.cod ** (n_steps + 1)


def test_unroll_prediction_types():
    cmap = CMap([readout()], [])
    for n_steps in (0, 1, 2):
        unrolled = cmap.unroll(n_steps)
        assert unrolled.dom == cmap.memory
        assert unrolled.cod == cmap.prediction ** (n_steps + 1)


def test_unroll_matches_protocol():
    cmap = CMap([box(), box()], [((0, 1), (1, 0))])
    assert cmap.unroll(1) == cmap.protocol.unroll(1)


def test_readout_predicts_its_memory():
    unrolled = Ket(1) >> CMap([readout()], []).unroll(1)
    array = unrolled.double().to_tensor().eval().array
    expected = (Ket(1) @ Ket(1)).double().to_tensor().eval().array
    assert np.allclose(
        np.asarray(array).flatten(), np.asarray(expected).flatten())


def without_certificate(call):
    """`fix` falls back to `power_fix` silently when no certificate
    applies, so the call warns about anything but the certificate."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = call()
    assert not [
        warning for warning in caught
        if "certificate does not apply" in str(warning.message)]
    return result


def fixed_wire():
    wire = Box("wire", qubit, qubit ** 2, Diagram.id(qubit ** 3))
    return CMap([wire], [((0, 1), (0, 2))])


def test_fix_requires_input_state():
    with pytest.raises(ValueError, match="input_state"):
        fixed_wire().fix(
            initial_state=Ket(0) @ Ket(0), max_steps=2,
            backend=DiscopyBackend())
    with pytest.raises(AxiomError, match="state of type"):
        fixed_wire().fix(
            Ket(0) @ Ket(0), Ket(0) @ Ket(0), max_steps=2,
            backend=DiscopyBackend())


def test_fix_reuses_protocol_fixed_point():
    result = without_certificate(lambda: fixed_wire().fix(
        Ket(1), Ket(0) @ Ket(0), max_steps=2,
        backend=DiscopyBackend()))
    assert np.allclose(result.density_matrix, [[0, 0], [0, 1]])


def test_fix_certifies_an_optical_delay():
    wait = Box("wait", Ty(), qmode,
               Diagram.swap(qmode, qmode), memory=qmode)
    fixed = CMap([wait], []).fix(Create(1), Create(0), max_chi=None)
    assert np.allclose(fixed.density_matrix, [[0, 0], [0, 1]])


def test_fix_with_internal_memory():
    result = without_certificate(lambda: CMap([readout()], []).fix(
        initial_state=Ket(1), max_steps=2, backend=DiscopyBackend()))
    assert np.allclose(result.density_matrix, [[0, 0], [0, 1]])


def test_tensor_is_disjoint_union():
    cmap = CMap([box()], [((0, 0), (0, 1))])
    both = cmap @ cmap
    assert both.boxes == cmap.boxes * 2
    assert both.memory == cmap.memory ** 2
    assert both.edges == [((0, 0), (0, 1)), ((1, 0), (1, 1))]


def test_glue_adds_edges():
    cmap = CMap([box(), box()], []).glue(((0, 1), (1, 0)))
    assert cmap == CMap([box(), box()], [((0, 1), (1, 0))])
    assert cmap.memory == qubit ** 2


def test_cup_on_one_box():
    cmap = CMap([box()], []).glue(((0, 0), (0, 1)))
    assert cmap.dom == Ty()
    assert cmap.memory == qubit ** 2


def test_read_and_write_are_permutations():
    cmap = CMap([box(), readout()], [((0, 1), (1, 0))][:0])
    channels = Ty().tensor(*map(cmap.port_type, cmap.inputs))
    assert cmap.read.dom == cmap.dom @ cmap.memory
    assert cmap.read.cod == channels
    assert cmap.write.dom \
        == Ty().tensor(*map(cmap.port_type, cmap.outputs))
    assert cmap.write.cod == cmap.cod @ cmap.prediction @ cmap.memory


def delay():
    swap = Diagram.swap(qubit, qubit)
    return CMap([Box("wait", qubit, qubit, swap)], [((0, 0), (0, 1))])


def test_closed_fix_needs_no_input_state():
    result = without_certificate(lambda: delay().fix(
        initial_state=Ket(0) @ Ket(0), max_steps=2,
        backend=DiscopyBackend()))
    assert np.allclose(result.density_matrix, 1)


def test_protocol_against_hand_built():
    swap = Diagram.swap(qubit, qubit)
    assert delay().protocol == (
        Diagram.id(qubit ** 2) >> swap >> swap
    ).feedback(dom=Ty(), cod=Ty(), mem=qubit ** 2)


@pytest.mark.parametrize("n_steps", (1, 2))
def test_memory_is_a_delay_line(n_steps):
    wait = Box("wait", Ty(), qubit,
               Diagram.swap(qubit, qubit), memory=qubit)
    inputs = [Ket(1)] + [Ket(0)] * n_steps + [Ket(0)]
    unrolled = Diagram.id().tensor(*inputs) \
        >> CMap([wait], []).unroll(n_steps)
    array = unrolled.double().to_tensor().eval().array
    outputs = [Ket(0)] + [Ket(1)] + [Ket(0)] * (n_steps - 1)
    expected = Diagram.id().tensor(*outputs).double().to_tensor()
    assert np.allclose(
        np.asarray(array).flatten(),
        np.asarray(expected.eval().array).flatten())


def test_drawing():
    cmap = CMap([box()], [((0, 0), (0, 1))])
    assert cmap.to_drawing() == cmap.protocol.to_drawing()


def test_gradient_through_the_memory_wire():
    torch = pytest.importorskip("torch")
    from discopy import tensor
    from optyx.channel import Channel
    from optyx.core.contract import contract_tensor
    from optyx.core.diagram import Box as CoreBox, bit as core_bit

    theta = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    array = torch.stack((
        torch.stack((torch.cos(theta), -torch.sin(theta))),
        torch.stack((torch.sin(theta), torch.cos(theta))),
    )).to(torch.float64)
    rotation = Channel(
        "R", CoreBox("R", core_bit, core_bit, array=array), qubit, qubit)
    memory = Box("m", Ty(), Ty(), rotation >> Z(1, 2),
                 memory=qubit, prediction=qubit)
    step = CMap([memory], []).step
    network = Ket(0) >> step >> Diagram.id(qubit) @ step \
        >> Bra(1) @ Bra(1) @ Bra(1)
    with tensor.backend("pytorch"):
        diagram = network.get_kraus().to_tensor()
        result = contract_tensor(
            diagram.to_map(), backend="pytorch", dtype=float)
    probability = result.array ** 2
    probability.backward()
    assert torch.allclose(probability, torch.sin(2 * theta) ** 2 / 4)
    assert torch.allclose(theta.grad, torch.sin(4 * theta) / 2)
