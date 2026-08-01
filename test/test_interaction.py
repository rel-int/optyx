import numpy as np
import pytest

from discopy.utils import AxiomError

from optyx.channel import Diagram, Ty, qubit, qmode
from optyx.core.backends import DiscopyBackend
from optyx.interaction import Box, CMap
from optyx.photonic import BS
from optyx.qubits import Ket, Scalar, X, Z


def cnot():
    return Z(1, 2) @ qubit >> qubit @ X(2, 1) @ Scalar(2 ** 0.5)


def box(name="f"):
    return Box(name, qubit, qubit, cnot())


def test_box_axioms():
    with pytest.raises(AxiomError):
        Box("f", qubit, qubit, Diagram.id(qubit))
    with pytest.raises(AxiomError):
        Box("f", qmode, qmode, cnot())


def test_box_ports():
    assert box().ports == qubit ** 2
    assert Box("g", Ty(), qubit ** 2, Diagram.id(qubit ** 2)).ports \
        == qubit ** 2


def test_edge_axioms():
    with pytest.raises(AxiomError):
        CMap([box()], [((0, 0), (0, 5))])
    with pytest.raises(AxiomError):
        CMap([box(), box()], [((0, 1), (1, 0)), ((0, 1), (1, 1))])
    with pytest.raises(AxiomError):
        CMap([box(), Box("g", qmode, qmode, BS)], [((0, 0), (1, 0))])


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
    assert cmap.unroll(1) == cmap.parallel


def test_protocol_types():
    cmap = CMap([box()], [((0, 0), (0, 1))])
    assert cmap.protocol.dom == cmap.dom == Ty()
    assert cmap.protocol.mem == cmap.memory == qubit ** 2


def test_unroll_types():
    cmap = CMap([box(), box()], [((0, 1), (1, 0))])
    for n_steps in (1, 2, 3):
        unrolled = cmap.unroll(n_steps)
        assert unrolled.dom == cmap.dom ** n_steps @ cmap.memory
        assert unrolled.cod == cmap.cod ** n_steps @ cmap.memory


def test_unroll_matches_protocol():
    cmap = CMap([box(), box()], [((0, 1), (1, 0))])
    assert cmap.unroll(1) == cmap.protocol.unroll(1)


def fixed_wire():
    wire = Box("wire", qubit, qubit ** 2, Diagram.id(qubit ** 3))
    return CMap([wire], [((0, 1), (0, 2))])


def test_fix_requires_input_state():
    with pytest.raises(ValueError, match="input_state"):
        fixed_wire().fix(
            initial_state=Ket(0) @ Ket(0), n_steps=2,
            backend=DiscopyBackend())
    with pytest.raises(AxiomError, match="state of type"):
        fixed_wire().fix(
            Ket(0) @ Ket(0), Ket(0) @ Ket(0), n_steps=2,
            backend=DiscopyBackend())


def test_fix_reuses_protocol_fixed_point():
    result = fixed_wire().fix(
        Ket(1), Ket(0) @ Ket(0), n_steps=2,
        backend=DiscopyBackend())
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
    cmap = CMap([box(), box()], [((0, 1), (1, 0))])
    ports = Ty().tensor(*map(cmap.port_type, cmap.ports))
    assert cmap.read.dom == cmap.dom @ cmap.memory
    assert cmap.read.cod == ports
    assert cmap.write.dom == ports
    assert cmap.write.cod == cmap.cod @ cmap.memory


def delay():
    swap = Diagram.swap(qubit, qubit)
    return CMap([Box("wait", qubit, qubit, swap)], [((0, 0), (0, 1))])


def test_closed_fix_needs_no_input_state():
    result = delay().fix(
        initial_state=Ket(0) @ Ket(0), n_steps=2,
        backend=DiscopyBackend())
    assert np.allclose(result.density_matrix, 1)


def test_protocol_against_hand_built():
    swap = Diagram.swap(qubit, qubit)
    assert delay().protocol == (
        Diagram.id(qubit ** 2) >> swap >> swap
    ).feedback(dom=Ty(), cod=Ty(), mem=qubit ** 2)


@pytest.mark.parametrize("n_steps", (1, 2, 3))
def test_self_edge_is_a_delay_line(n_steps):
    array = delay().unroll(n_steps).double().to_tensor().eval().array
    dim = int(np.prod(array.shape) ** 0.5)
    assert np.allclose(array.reshape(dim, dim), np.eye(dim))


def test_drawing():
    cmap = CMap([box()], [((0, 0), (0, 1))])
    assert cmap.to_drawing() == cmap.protocol.to_drawing()
