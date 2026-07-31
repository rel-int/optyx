import numpy as np
import pytest

from discopy.utils import AxiomError

from optyx import photonic
from optyx import classical, qubits
from optyx.channel import (
    Diagram, Discard, Feedback, Functor, bit, qmode, qubit
)
from optyx.core import diagram as core, path, zw


def delay(initial_state=None, final_effect=None):
    return Diagram.swap(qmode, qmode).feedback(
        initial_state=initial_state, final_effect=final_effect)


def test_feedback_types():
    wait = delay()
    assert wait.dom == wait.cod == qmode and wait.mem == qmode
    fb = (photonic.BS @ photonic.BS).feedback(mem=qmode ** 2)
    assert fb.dom == fb.cod == qmode ** 2 and fb.mem == qmode ** 2


def test_feedback_axioms():
    with pytest.raises(AxiomError):
        photonic.BS.feedback(dom=qmode ** 2, cod=qmode, mem=qmode)
    with pytest.raises(AxiomError):
        photonic.BS.feedback(dom=qmode, cod=qmode ** 2, mem=qmode)
    with pytest.raises(AxiomError):
        delay(initial_state=photonic.Create(0, 0)).unroll(1)
    with pytest.raises(AxiomError):
        delay(final_effect=Discard(qmode ** 2)).unroll(1)


def test_single_step_unrolling():
    wait, box = delay(), photonic.BS
    assert box.unroll(1) == box
    assert wait.unroll(1) == Diagram.swap(qmode, qmode)


def test_unroll_is_a_delay_line():
    wait = delay(
        initial_state=photonic.Create(1), final_effect=Discard(qmode))
    unrolled = wait.unroll(2)
    assert unrolled.dom == unrolled.cod == qmode ** 2
    probability = (
        photonic.Create(0, 0) >> unrolled >> photonic.Select(1, 0)
    ).double().to_tensor().eval().array
    assert np.isclose(probability, 1)


def test_unroll_open_wires():
    unrolled = delay().unroll(2)
    assert unrolled.dom == unrolled.cod == qmode ** 3
    with pytest.raises(ValueError):
        delay().unroll(0)


def test_diagram_has_no_stream_method():
    assert not hasattr(delay(), "stream")
    assert not hasattr(core.Diagram.swap(core.mode, core.mode), "stream")


def test_evaluation_raises_on_feedback():
    with pytest.raises(ValueError):
        delay().eval()
    with pytest.raises(ValueError):
        delay().double().to_tensor()
    with pytest.raises(ValueError):
        core.Diagram.swap(core.mode, core.mode).feedback().to_tensor()


def test_double_unroll_commute():
    wait = delay(
        initial_state=photonic.Create(0), final_effect=Discard(qmode))
    assert wait.unroll(2).double() == wait.double().unroll(2)
    lhs = wait.unroll(2).double().to_tensor().eval().array
    rhs = wait.double().unroll(2).to_tensor().eval().array
    assert np.allclose(lhs, rhs)


def composite_loops():
    fb = photonic.BS.feedback()
    inner = photonic.BS.feedback()
    return {
        "single": fb,
        "composite": photonic.Phase(0.3) >> fb >> photonic.Phase(0.2),
        "sequence": fb >> photonic.BS.feedback(),
        "tensor": fb @ photonic.Phase(0.1),
        "nested": (
            inner @ qmode >> Diagram.swap(qmode, qmode)).feedback(),
    }


@pytest.mark.parametrize("name", composite_loops().keys())
def test_to_path_raises_on_feedback(name):
    with pytest.raises(ValueError):
        composite_loops()[name].to_path()


def test_unrolled_to_path():
    unrolled = delay().unroll(2)
    amplitude = (
        photonic.Create(1, 0, 0) >> unrolled >> photonic.Select(0, 1, 0)
    ).to_path().eval().array
    assert np.isclose(amplitude, 1)
    with pytest.raises(ValueError):
        core.Diagram.swap(core.mode, core.mode).feedback().to_path()


def test_core_feedback_axioms():
    swap = core.Diagram.swap(core.mode, core.mode)
    with pytest.raises(AxiomError):
        swap.feedback(dom=core.mode ** 2, cod=core.mode, mem=core.mode)
    with pytest.raises(AxiomError):
        swap.feedback(dom=core.mode, cod=core.mode ** 2, mem=core.mode)
    with pytest.raises(AxiomError):
        swap.feedback(initial_state=zw.Create(0, 0)).unroll(1)
    with pytest.raises(AxiomError):
        swap.feedback(final_effect=zw.Select(0, 0)).unroll(1)


def test_memory_order():
    loops = composite_loops()
    sequence, nested = loops["sequence"], loops["nested"]
    assert sequence.unroll(1).dom == sequence.dom @ qmode ** 2
    assert nested.unroll(1).dom == nested.dom @ qmode ** 2


def test_stream_semantics_contract_and_order():
    left = photonic.BS.feedback(
        initial_state=photonic.Create(0),
        final_effect=photonic.Select(0))
    right = photonic.BS.feedback(
        initial_state=photonic.Create(1),
        final_effect=photonic.Select(1))
    for diagram in (left >> right, left @ right):
        semantics = diagram.to_stream()
        assert isinstance(semantics, core.StreamSemantics)
        assert semantics.stream.mem.now == qmode ** 2
        assert semantics.stream.now.dom == diagram.dom @ qmode ** 2
        assert semantics.stream.now.cod == diagram.cod @ qmode ** 2
        assert tuple(boundary.initial_state
                     for boundary in semantics.boundaries) == (
                         photonic.Create(0), photonic.Create(1))
        assert tuple(boundary.final_effect
                     for boundary in semantics.boundaries) == (
                         photonic.Select(0), photonic.Select(1))
        assert not any(isinstance(box, Feedback)
                       for box in semantics.stream.now.boxes)

    inner = photonic.BS.feedback(initial_state=photonic.Create(0))
    nested = (inner @ qmode >> Diagram.swap(qmode, qmode)).feedback(
        initial_state=photonic.Create(1))
    assert tuple(boundary.initial_state
                 for boundary in nested.to_stream().boundaries) == (
                     photonic.Create(1), photonic.Create(0))


def test_stream_semantics_is_uncached_and_boundary_free():
    wait = delay(
        initial_state=photonic.Create(0),
        final_effect=photonic.Select(0))
    first, second = wait.to_stream(), wait.to_stream()
    assert first == second and first is not second
    assert first.stream is not second.stream
    assert wait.one_step() == delay().one_step()
    assert wait.unroll(1) != wait.one_step()
    stateless = photonic.BS.to_stream()
    assert stateless.stream.now == photonic.BS
    assert stateless.boundaries == ()


def test_matrix_has_no_feedback():
    assert not hasattr(path.Matrix, "feedback")
    assert not hasattr(path, "Feedback")


def test_functor_maps_feedback():
    functor = Functor(ob_map=lambda x: x, ar_map=lambda f: f, cod=Diagram)
    wait = delay(initial_state=photonic.Create(0))
    image = functor(photonic.Phase(0.3) >> wait)
    image_loops = [b for b in image.boxes if isinstance(b, Feedback)]
    assert image_loops[0].initial_state == photonic.Create(0)
    core_functor = core.Functor(
        ob_map=lambda x: x, ar_map=lambda f: f, cod=core.Diagram)
    core_wait = core.Diagram.swap(core.mode, core.mode).feedback(
        initial_state=zw.Create(1), final_effect=zw.Select(0))
    core_image = core_functor(core_wait)
    assert core_image.initial_state == zw.Create(1)
    assert core_image.final_effect == zw.Select(0)


def test_get_kraus_and_purity():
    fb = photonic.BS.feedback()
    kraus = fb.get_kraus()
    assert isinstance(kraus, core.Feedback)
    assert kraus.mem == core.mode
    composite = photonic.Phase(0.3) >> fb
    assert any(isinstance(box, core.Feedback)
               for box in composite.get_kraus().boxes)
    assert fb.is_pure
    assert not photonic.BS.feedback(final_effect=Discard(qmode)).is_pure


def test_conjugate_and_inflate():
    core_wait = core.Diagram.swap(core.mode, core.mode).feedback(
        initial_state=zw.Create(1), final_effect=zw.Select(0))
    conjugated = core_wait.conjugate()
    assert conjugated.initial_state == zw.Create(1).conjugate()
    assert conjugated.final_effect == zw.Select(0).conjugate()
    inflated = core.Diagram.swap(core.mode, core.mode).feedback(
        initial_state=zw.Create(0)).inflate(2)
    assert inflated.mem == core.mode ** 2
    wait = delay(initial_state=photonic.Create(0))
    assert wait.inflate(2).mem == qmode ** 2


def test_str_repr_equality():
    fb = photonic.BS.feedback()
    assert str(fb) == "(BBS(0)).feedback()"
    assert "Feedback" in repr(fb)
    assert fb == photonic.BS.feedback()
    assert fb != photonic.BS.feedback(final_effect=Discard(qmode))
    with_state = delay(initial_state=photonic.Create(0))
    assert "initial_state" in str(with_state)
    assert "initial_state" in repr(with_state)


def test_dagger_raises():
    with pytest.raises(NotImplementedError):
        delay().dagger()
    with pytest.raises(NotImplementedError):
        core.Diagram.swap(core.mode, core.mode).feedback().dagger()


def test_to_drawing():
    assert delay().to_drawing() is not None
    core_wait = core.Diagram.swap(core.mode, core.mode).feedback()
    assert core_wait.to_drawing() is not None


def test_qubit_delay_line():
    wait = Diagram.swap(qubit, qubit).feedback(
        initial_state=qubits.Ket(0), final_effect=Discard(qubit))
    unrolled = wait.unroll(2)
    assert unrolled.dom == unrolled.cod == qubit ** 2
    probability = (
        qubits.Ket(1) @ qubits.Ket(1) >> unrolled
        >> qubits.Bra(0) @ qubits.Bra(1)
    ).double().to_tensor().eval().array
    assert np.isclose(probability, 1)
    assert wait.unroll(2).double() == wait.double().unroll(2)


def test_bit_delay_line():
    wait = Diagram.swap(bit, bit).feedback(
        initial_state=classical.Bit(1), final_effect=Discard(bit))
    unrolled = wait.unroll(2)
    assert unrolled.dom == unrolled.cod == bit ** 2
    probability = (
        classical.Bit(0, 0) >> unrolled >> classical.PostselectBit(1, 0)
    ).double().to_tensor().eval().array
    assert np.isclose(probability, 1)
