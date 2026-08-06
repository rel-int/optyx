# TODO

> Check out the PR https://github.com/rel-int/optyx/pull/16. We need to push
> this model to try to solve the sudoku task. One important concept for the
> interaction.CMap is that every cell: X -> Y should have additionally a memory
> type M and a prediction type O, so that the process inside it has type
> X @ Y @ M -> X @ Y @ M @ O. So we can run the same experiment we were running
> but such that every cell has 1 qubit internal memory. Make a plan for
> improvements and tests to solve the sudoku task in the notebook. Write the
> plan in TODO.md.

Stacked on #16. Mathematically, a box `X -> Y` of a `CMap` now carries three
kinds of wires: the message ports `X @ Y`, read and written at every step and
pairable by edges; a private memory `M`, a feedback loop from the box to
itself that never appears as a port of the map; and a prediction `O`, written
to the environment at every step but never read. The process inside the box
is a channel `X @ Y @ M -> X @ Y @ M @ O`, so the protocol of a map has
`dom` the unpaired ports, `cod == dom @ predictions` and `mem` the paired
ports followed by the internal memories: the `cod == dom` invariant of #16 is
dropped and the compact closed structure only glues along message ports.

## `interaction.Box` with memory and prediction

- [WIP] @claude-xz2c4u-2026-08-06 12:05 Extend `Box(name, dom, cod, channel, memory=Ty(), prediction=Ty())`,
      type-checking `channel` from `dom @ cod @ memory` to
      `dom @ cod @ memory @ prediction`; the defaults recover the boxes of
      #16 so every existing doctest and test stays valid.
- [WIP] @claude-xz2c4u-2026-08-06 12:05 Route each internal memory as a self-loop in `CMap`: it joins
      `CMap.memory` after the paired ports, in box order, and is fed back to
      the same box by `read` and `write`; document the wire-order convention.
- [WIP] @claude-xz2c4u-2026-08-06 12:05 Make predictions write-only boundary: `CMap.cod == dom @ predictions`
      in box order, with `read`, `write`, `step`, `protocol`, `unroll` and
      the drawings updated; state in the docstring that `glue` and the cups
      and caps act on message ports only.
- [WIP] @claude-xz2c4u-2026-08-06 12:05 Update `CMap.fix`: `input_state` stays of type `dom`,
      `initial_state` now prepares the paired ports and the internal
      memories; the stationary output includes the predictions.
- [WIP] @claude-xz2c4u-2026-08-06 12:05 Update `__matmul__`, `glue`, `__repr__`, `__eq__`, `__hash__` and the
      module docstring for the two new attributes.

## Sudoku with one qubit of cell memory

- [ ] Rebuild the notebook map: a cell is
      `Box("cell", qubit ** 3, qubit ** 3, channel, memory=qubit,
      prediction=qubit ** 2)` — three messages read, three written, one
      internal memory qubit, two prediction qubits; constraints stay
      `Box(qubit ** 4, qubit ** 4, channel)` with no memory or prediction.
- [ ] Cell channels become shared trainable isometries
      `qubit ** 7 -> qubit ** 9`: a parameterised nine-qubit real unitary
      applied to the input tensored with two fresh ancillas, keeping the
      rotation-layer parameterisation and the parameter count comparable
      to #16; constraint channels keep their eight-qubit unitary.
- [ ] Check the topology: 96 edges and 192 paired memory wires as in #16,
      plus 16 internal memory wires (208 total), empty `dom` and a `cod` of
      32 prediction qubits per step.
- [ ] Move clue injection entirely to the write side: at every step the
      prediction output of a clue cell is postselected on its digit, free
      cells meet uniform effects at intermediate steps and the candidate
      digit at the last step; pick and document the initial memory state.
- [ ] Update `unrolled_tensor_map`: internal memory ports connect a box to
      itself at the next step, prediction ports get one effect per step and
      no read; keep the direct `tensor.CMap` construction without a
      materialised permutation.

## Solving the task

- [ ] Replace two-candidate ranking by a per-cell readout: score the four
      digits of every hidden cell from the last-step prediction amplitudes,
      and report per-cell argmax accuracy and the full-grid solve rate on
      the held-out puzzles, alongside the ranking metric of #16.
- [ ] Batch the contraction over puzzles and candidates as in the
      neural-sudoku experiment of discopy#416: a batch index threaded
      through the boundary states and effects, one Cotengra path reused
      across the batch.
- [ ] Extend the training budget: more epochs with the held-out probability
      as stopping criterion, and a small sweep over `n_steps` in {2, 3, 4}
      and bond dimension in {4, 8, 16}, recording the loss curves.
- [ ] Ablate the memory: run the same experiment with `memory=Ty()` at a
      matched parameter count and report whether the qubit of internal
      memory improves the held-out metrics.
- [ ] Stretch: probe `CMap.fix` on one cell with its memory qubit, towards
      inferring `n_steps` from the stationary semantics during training.

## Tests

- [ ] `test/test_interaction.py`: `Box` rejects a channel whose type is not
      `dom @ cod @ memory -> dom @ cod @ memory @ prediction`.
- [ ] Protocol types on a one-box map with memory and prediction:
      `cod == dom @ prediction`, `protocol.mem` is the paired ports followed
      by the memory, and `unroll` agrees with a hand-built diagram.
- [ ] Backwards compatibility: default `memory` and `prediction` reproduce
      the protocol of #16 on its existing examples.
- [ ] `fix` on a box with one memory qubit and a known stationary
      prediction.
- [ ] Non-zero PyTorch gradient through the memory wire: a parameter acting
      only on the memory qubit of a small unrolled map, contracted with
      `contract_tensor`.
- [ ] Notebook assertions: topology counts, non-zero initial gradient,
      held-out metrics improving over their initial values, fresh-kernel
      execution.

## Docs and checks

- [ ] Module docstring, doctests and `docs/api.rst` updated for the new
      `Box` signature; a drawing of a cell with memory and prediction wires.
- [ ] `pflake8 optyx`, `pylint optyx/interaction.py --fail-under=9` and
      `coverage run -m pytest` green with coverage at least 95%.
