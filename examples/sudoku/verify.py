"""Verify the trainer's tensor semantics against optyx.

Two checks tie ``contraction.py`` to the ``optyx.interaction`` protocol:

* The sudoku ``Structure`` is the combinatorial data of a genuine
  ``interaction.CMap``: same ports, pairings, memories and predictions.

* On a two-box map with memory and prediction, the trainer's born-family
  scores match the composition of ``CMap.step`` -- the read, parallel
  and write permutation sandwich of ``optyx.interaction`` -- threaded
  over the memory for the same number of ticks, with plus states on the
  paired ports, zero on the internal memory and uniform effects at the
  end, evaluated through ``double().to_tensor()`` and
  ``optyx.core.contract.contract_tensor``: the squared single-copy
  amplitude of the trainer equals the doubled channel probability.
  (``CMap.unroll`` itself differs only in its final boundary, which
  discards the memory instead of postselecting it uniformly.)

A third check grounds the stochastic family: a classical channel on bit
wires doubles to the elementwise square of its kraus array, so the
non-negative digit tensors of the trainer are optyx classical channels
with kraus the elementwise square root. (Checked on a square kraus:
unequal arities hit the ``Channel.double`` bug reported as issue #51.)
"""

import numpy as np

import experiment as ex
import contraction as co


def array_box(name, dom, cod, array):
    """A core box whose ``conjugate`` conjugates the array in place.

    Works around ``optyx.core.diagram.Box.conjugate`` swapping ``dom``
    and ``cod`` on plain array boxes -- a dagger, not a conjugate --
    which breaks ``Channel.double`` (reported upstream).
    """
    from optyx.core.diagram import Box as CoreBox

    class ArrayBox(CoreBox):
        def conjugate(self):
            return array_box(
                self.name, self.dom, self.cod, np.conjugate(self.array))

    return ArrayBox(name, dom, cod, array=array)


def check_structure():
    from discopy.utils import AxiomError  # noqa: F401
    from optyx.channel import Ty, qubit
    from optyx.interaction import Box, CMap

    structure = ex.sudoku_structure()
    dummy_cell = np.zeros((2 ** 4, 2 ** 6))
    dummy_cell[0, 0] = 1
    from optyx.core.diagram import bit as core_bit
    from optyx.channel import Channel
    cell_channel = Channel(
        "cell", array_box("cell", core_bit ** 4, core_bit ** 6,
                          dummy_cell),
        qubit ** 4, qubit ** 6)
    constraint_channel = Channel(
        "constraint", array_box(
            "constraint", core_bit ** 4, core_bit ** 4, np.eye(2 ** 4)),
        qubit ** 4, qubit ** 4)
    boxes = [
        Box(f"cell_{i}", qubit ** 2, qubit, cell_channel,
            memory=qubit, prediction=qubit ** 2)
        for i in range(ex.N_CELLS)]
    boxes += [
        Box(f"constraint_{i}", qubit ** 2, qubit ** 2, constraint_channel)
        for i in range(ex.N_CONSTRAINTS)]
    cmap = CMap(boxes, list(structure.edges))
    assert len(cmap.boundary) == 0
    assert len(cmap.paired) == 2 * len(structure.edges)
    assert len(cmap.memories) == ex.N_CELLS
    assert cmap.memory == qubit ** (2 * len(structure.edges) + ex.N_CELLS)
    assert cmap.prediction == qubit ** (2 * ex.N_CELLS)
    assert cmap.protocol.cod == cmap.prediction
    print("structure check passed: the sudoku Structure is the",
          "combinatorial data of an interaction.CMap")


def check_born_toy(ticks=2, seed=11):
    import jax
    import jax.numpy as jnp
    from optyx.channel import Channel, Ty, qubit
    from optyx.core.contract import contract_tensor
    from optyx.core.diagram import bit as core_bit
    from optyx.interaction import Box, CMap
    from optyx.qubits import Ket

    jax.config.update("jax_enable_x64", True)
    random = np.random.default_rng(seed)

    def unitary(width):
        matrix = random.normal(size=(2 ** width, 2 ** width))
        q, _ = np.linalg.qr(matrix)
        return q

    cell_isometry = unitary(4)[:2 ** 2]      # port + memory -> + 2 preds
    constraint_unitary = unitary(1)
    cell_channel = Channel(
        "A", array_box("A", core_bit ** 2, core_bit ** 4, cell_isometry),
        qubit ** 2, qubit ** 4)
    constraint_channel = Channel(
        "B", array_box("B", core_bit, core_bit, constraint_unitary),
        qubit, qubit)
    cell_box = Box("A", qubit, Ty(), cell_channel,
                   memory=qubit, prediction=qubit ** 2)
    constraint_box = Box("B", qubit, Ty(), constraint_channel)
    cmap = CMap([cell_box, constraint_box], [((0, 0), (1, 0))])
    from optyx.channel import Diagram
    plus = Ket(0) >> Channel(
        "H", array_box("H", core_bit, core_bit,
                       np.array([[1, 1], [1, -1]]) / 2 ** .5),
        qubit, qubit)
    state = plus @ plus @ Ket(0)             # paired ports, then memory

    def threaded(pred_effects):
        """States >> ticks of CMap.step >> effects, memory postselected
        uniformly at the end as in the trainer."""
        diagram = state
        for effect in pred_effects:
            diagram = diagram >> cmap.step
            diagram = diagram >> (effect @ Diagram.id(cmap.memory))
        for _ in range(3):
            diagram = diagram >> (
                Diagram.id(diagram.cod[:-1]) @ plus.dagger())
        return diagram

    structure = ex.Structure(
        (ex.BoxSpec("cell", 1, 1, 1), ex.BoxSpec("constraint", 1, 0, 0)),
        (((0, 0), (1, 0)),))
    scores = co.make_scores(structure, ticks, "born")
    tensors = {
        "cell": jnp.asarray(
            cell_isometry.reshape((2, 2) + (2, 2, 4))),
        "constraint": jnp.asarray(constraint_unitary),
    }
    for digits in ([0, 0], [1, 3], [2, 1]):
        effects = np.ones((1, ticks, 4))
        for tick, digit in enumerate(digits[:-1]):
            onehot = np.zeros(4)
            onehot[digit] = 1
            effects[0, tick] = onehot
        raw = np.asarray(scores(tensors, jnp.asarray(effects)))[0]
        mine = raw / raw.sum()

        reference = []
        for final_digit in range(4):
            selected = [*digits[:-1], final_digit]
            pred_effects = [
                (Ket(digit // 2) @ Ket(digit % 2)).dagger()
                for digit in selected]
            diagram = threaded(pred_effects)
            value = contract_tensor(
                diagram.double().to_tensor().to_map(),
                backend="numpy").array
            reference.append(float(np.real(np.asarray(value).ravel()[0])))
        reference = np.asarray(reference)
        reference = reference / reference.sum()
        assert np.allclose(mine, reference, atol=1e-9), (mine, reference)
    print("born toy check passed: trainer scores match ticks of",
          "CMap.step >> double >> contract_tensor")


def check_classical_doubling(seed=5):
    from optyx.channel import Channel, bit
    from optyx.core.diagram import bit as core_bit

    random = np.random.default_rng(seed)
    kraus = random.normal(size=(4, 4))
    channel = Channel(
        "f", array_box("f", core_bit ** 2, core_bit ** 2, kraus),
        bit ** 2, bit ** 2)
    doubled = channel.double().to_tensor().eval().array
    assert np.allclose(
        np.asarray(doubled).reshape(4, 4), np.abs(kraus) ** 2)
    print("classical doubling check passed: bit-wire channels double to",
          "the elementwise square of their kraus array")


if __name__ == "__main__":
    check_structure()
    check_classical_doubling()
    check_born_toy()
