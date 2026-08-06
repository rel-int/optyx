import numpy as np
import pytest

from discopy.utils import AxiomError

from optyx import photonic
from optyx import classical, qubits
from optyx.channel import (
    Diagram, Discard, Feedback, Functor, bit, qmode, qubit
)
from optyx.core import diagram as core, path, zw


def delay(state=None, effect=None):
    return Diagram.swap(qmode, qmode).feedback(state=state, effect=effect)


def open_delay(state=None):
    return delay(state=state, effect=Diagram.id(qmode))


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


def test_boundary_axioms_are_checked_at_construction():
    """A mistyped boundary raises where it is given, not later in unroll."""
    with pytest.raises(AxiomError):
        delay(state=photonic.Create(0, 0))
    with pytest.raises(AxiomError):
        delay(effect=Discard(qmode ** 2))


def test_boundary_defaults():
    """`None` is an input spelling: the boundaries are always diagrams.
    A channel discards its memory, the pure layer leaves it open."""
    wait = delay()
    assert wait.state == Diagram.id(qmode)
    assert wait.effect == Discard(qmode)
    core_wait = core.Diagram.swap(core.mode, core.mode).feedback()
    assert core_wait.state == core_wait.effect == core.Diagram.id(core.mode)


def test_one_step_opens_the_loop():
    step = Diagram.swap(qmode, qmode)
    assert delay().one_step() == step
    assert delay(state=photonic.Create(0)).one_step() == step
    assert photonic.BS.one_step() == photonic.BS
    core_step = core.Diagram.swap(core.mode, core.mode)
    assert core_step.feedback(state=zw.Create(1)).one_step() == core_step


def test_unroll_counts_unrollings_not_time_steps():
    """As in `discopy.stream`, `unroll(0)` is one time step."""
    wait = open_delay()
    for n_steps in range(3):
        assert wait.unroll(n_steps).cod == qmode ** (n_steps + 2)
    assert photonic.BS.unroll(0) == photonic.BS
    with pytest.raises(ValueError):
        wait.unroll(-1)


def test_unroll_is_a_delay_line():
    wait = delay(state=photonic.Create(1))
    unrolled = wait.unroll(1)
    assert unrolled.dom == unrolled.cod == qmode ** 2
    probability = (
        photonic.Create(0, 0) >> unrolled >> photonic.Select(1, 0)
    ).double().to_tensor().eval().array
    assert np.isclose(probability, 1)


def test_boundaries_are_chosen_at_feedback():
    """`unroll` takes only the number of steps: the boundaries belong to
    `feedback`, and other boundaries mean another `feedback` call on the
    loop's argument. An identity boundary is an open wire."""
    wait = delay(state=photonic.Create(1))
    with pytest.raises(TypeError):
        wait.unroll(1, state=None)
    assert wait.unroll(1).dom == qmode ** 2
    opened = wait.arg.feedback(
        state=Diagram.id(qmode), effect=Diagram.id(qmode))
    assert opened.unroll(1).dom == opened.unroll(1).cod == qmode ** 3
    replaced = wait.arg.feedback(state=photonic.Create(0))
    assert replaced.unroll(1).dom == qmode ** 2
    assert replaced.unroll(1) != wait.unroll(1)


def test_evaluation_raises_on_feedback():
    with pytest.raises(ValueError):
        delay().eval()
    with pytest.raises(ValueError):
        delay().double().to_tensor()
    with pytest.raises(ValueError):
        core.Diagram.swap(core.mode, core.mode).feedback().to_tensor()


def test_double_unroll_commute():
    wait = delay(state=photonic.Create(0))
    assert wait.unroll(1).double() == wait.double().unroll(1)
    lhs = wait.unroll(1).double().to_tensor().eval().array
    rhs = wait.double().unroll(1).to_tensor().eval().array
    assert np.allclose(lhs, rhs)


def composite_loops():
    """Loops with an open memory: the default `Discard` effect would make
    them impure, and `to_path` would refuse them for that reason rather
    than for the feedback the guard is about."""
    def loop(arg, **boundary):
        return arg.feedback(effect=Diagram.id(qmode), **boundary)
    fb, inner = loop(photonic.BS), loop(photonic.BS)
    return {
        "single": fb,
        "composite": photonic.Phase(0.3) >> fb >> photonic.Phase(0.2),
        "sequence": fb >> loop(photonic.BS),
        "tensor": fb @ photonic.Phase(0.1),
        "nested": loop(inner @ qmode >> Diagram.swap(qmode, qmode)),
    }


@pytest.mark.parametrize("name", composite_loops().keys())
def test_to_path_raises_on_feedback(name):
    with pytest.raises(ValueError):
        composite_loops()[name].to_path()


def test_to_path_refuses_a_discarded_memory():
    """The default effect is a discard, so the loop is not pure either."""
    with pytest.raises(AssertionError):
        photonic.BS.feedback().to_path()


def test_unrolled_to_path():
    unrolled = open_delay().unroll(1)
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
        swap.feedback(state=zw.Create(0, 0))
    with pytest.raises(AxiomError):
        swap.feedback(effect=zw.Select(0, 0))


def test_memory_order():
    loops = composite_loops()
    sequence, nested = loops["sequence"], loops["nested"]
    assert sequence.unroll(0).dom == sequence.dom @ qmode ** 2
    assert nested.unroll(0).dom == nested.dom @ qmode ** 2


def test_matrix_has_no_feedback():
    assert not hasattr(path.Matrix, "feedback")
    assert not hasattr(path, "Feedback")


def test_functor_maps_feedback():
    functor = Functor(ob_map=lambda x: x, ar_map=lambda f: f, cod=Diagram)
    wait = delay(state=photonic.Create(0))
    image = functor(photonic.Phase(0.3) >> wait)
    image_loops = [b for b in image.boxes if isinstance(b, Feedback)]
    assert image_loops[0].state == photonic.Create(0)
    assert image_loops[0].effect == Discard(qmode)
    core_functor = core.Functor(
        ob_map=lambda x: x, ar_map=lambda f: f, cod=core.Diagram)
    core_wait = core.Diagram.swap(core.mode, core.mode).feedback(
        state=zw.Create(1), effect=zw.Select(0))
    core_image = core_functor(core_wait)
    assert core_image.state == zw.Create(1)
    assert core_image.effect == zw.Select(0)


def test_get_kraus_and_purity():
    """Discarding the memory is not pure, so the default effect makes a
    channel loop impure until the memory is left open."""
    fb = photonic.BS.feedback(effect=Diagram.id(qmode))
    kraus = fb.get_kraus()
    assert isinstance(kraus, core.Feedback)
    assert kraus.mem == core.mode
    composite = photonic.Phase(0.3) >> fb
    assert any(isinstance(box, core.Feedback)
               for box in composite.get_kraus().boxes)
    assert fb.is_pure
    assert not photonic.BS.feedback().is_pure


def test_conjugate_and_inflate():
    core_wait = core.Diagram.swap(core.mode, core.mode).feedback(
        state=zw.Create(1), effect=zw.Select(0))
    conjugated = core_wait.conjugate()
    assert conjugated.state == zw.Create(1).conjugate()
    assert conjugated.effect == zw.Select(0).conjugate()
    inflated = core.Diagram.swap(core.mode, core.mode).feedback(
        state=zw.Create(0)).inflate(2)
    assert inflated.mem == core.mode ** 2
    wait = delay(state=photonic.Create(0))
    assert wait.inflate(2).mem == qmode ** 2


def test_str_repr_equality():
    """Only a boundary that differs from the default is printed."""
    fb = photonic.BS.feedback()
    assert str(fb) == "(BBS(0)).feedback()"
    assert "Feedback" in repr(fb)
    assert fb == photonic.BS.feedback()
    assert fb != photonic.BS.feedback(effect=Diagram.id(qmode))
    with_state = delay(state=photonic.Create(0))
    assert "state" in str(with_state)
    assert "state" in repr(with_state)


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
    wait = Diagram.swap(qubit, qubit).feedback(state=qubits.Ket(0))
    unrolled = wait.unroll(1)
    assert unrolled.dom == unrolled.cod == qubit ** 2
    probability = (
        qubits.Ket(1) @ qubits.Ket(1) >> unrolled
        >> qubits.Bra(0) @ qubits.Bra(1)
    ).double().to_tensor().eval().array
    assert np.isclose(probability, 1)
    assert wait.unroll(1).double() == wait.double().unroll(1)


def test_bit_delay_line():
    wait = Diagram.swap(bit, bit).feedback(state=classical.Bit(1))
    unrolled = wait.unroll(1)
    assert unrolled.dom == unrolled.cod == bit ** 2
    probability = (
        classical.Bit(0, 0) >> unrolled >> classical.PostselectBit(1, 0)
    ).double().to_tensor().eval().array
    assert np.isclose(probability, 1)
