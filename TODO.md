# TODO

> Check out the PR https://github.com/rel-int/optyx/pull/16. We need to push
> this model to try to solve the sudoku task. One important concept for the
> interaction.CMap is that every cell: X -> Y should have additionally a memory
> type M and a prediction type O, so that the process inside it has type
> X @ Y @ M -> X @ Y @ M @ O. So we can run the same experiment we were running
> but such that every cell has 1 qubit internal memory. Make a plan for
> improvements and tests to solve the sudoku task in the notebook. Write the
> plan in TODO.md.

> Why would cod == dom @ predictions? I don't think this is needed. Go ahead
> and implement the changes to interaction.py. Then give a few more ideas for
> scaling while keeping the contractions doable on MacMini in the TODO.md

> Also rebase on the fixpoint PR

Stacked on #16 and merged with the fixpoint PR #15. Mathematically, a box
`X -> Y` of a `CMap` now carries three kinds of wires: the message ports
`X @ Y`, read and written at every step and pairable by edges; a private
memory `M`, a feedback loop from the box to itself that never appears as a
port of the map; and a prediction `O`, written to the environment at every
step but never read. The process inside the box is a channel
`X @ Y @ M -> X @ Y @ M @ O`. The map keeps `dom == cod`, the unpaired
ports: the predictions are appended to the codomain of the `protocol`
diagram only, its `mem` is the paired ports followed by the internal
memories, and the compact closed structure glues along message ports only.

## `interaction.Box` with memory and prediction

- [x] Extend `Box(name, dom, cod, channel, memory=Ty(), prediction=Ty())`,
      type-checking `channel` from `dom @ cod @ memory` to
      `dom @ cod @ memory @ prediction`; the defaults recover the boxes of
      #16 so every existing doctest and test stays valid.
- [x] Route each internal memory as a self-loop in `CMap`: it joins
      `CMap.memory` after the paired ports, in box order, and is fed back to
      the same box by `read` and `write`; document the wire-order convention.
- [x] Make predictions write-only: `CMap.cod` stays equal to `dom` and
      `CMap.prediction` is appended to the codomain of `protocol` in box
      order, with `read`, `write`, `step`, `unroll` and the drawings
      updated; state in the docstring that `glue` and the cups and caps
      act on message ports only.
- [x] Update `CMap.fix` to the #15 interface: `input_state` stays of type
      `dom` and is composed inside the loop so the stationary certificates
      apply, `initial_state` prepares the paired ports and the internal
      memories through `feedback(state=...)`, the remaining parameters are
      `tol`, `loss`, `chi`, `max_steps` and `backend`, and the stationary
      output includes the predictions.
- [x] Update `__matmul__`, `glue`, `__repr__`, `__eq__`, `__hash__` and the
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
      reading the number of message-passing steps off the certified depth
      of the stationary semantics during training.

## Scaling on a Mac Mini

- [ ] Replace the dense eight- and nine-qubit unitaries by shared brickwork
      circuits of two-qubit rotations: every tensor in the network becomes
      rank four, the compressed contraction cost grows with the gate count
      rather than `2 ** width`, and the parameter count stays comparable.
- [ ] Contract the unrolling as a boundary MPS in the time direction: treat
      each step as an MPO on the memory wires and sweep with truncation at
      `chi`, so peak memory is set by `chi` and the 208-wire cut, never by
      the full network; compare against Cotengra's compressed paths.
- [ ] Exploit locality of the loss: each cell's prediction after three
      steps only sees its light cone (its three constraints and their
      cells), so per-cell scores contract sub-networks a fraction of the
      660-box map; batch the cells that share a light-cone shape.
- [ ] Slice the Cotengra path over a few high-degree indices
      (`slicing_opts`) to cap peak memory at a constant factor of the
      largest sliced tensor, trading a small constant in time.
- [ ] Curriculum on sub-grids: pre-train the shared cell and constraint
      channels on a single row or a 2x2 block `CMap` (a handful of boxes),
      then fine-tune on the full grid; the parameters transfer because
      every box shares them.
- [ ] Keep everything real: the rotation ansatz stays orthogonal, halving
      memory and contraction constants against complex tensors, and
      `float32` forward passes with `float64` loss accumulation halve them
      again if the singular-value gradients stay stable.

## Tests

- [x] `test/test_interaction.py`: `Box` rejects a channel whose type is not
      `dom @ cod @ memory -> dom @ cod @ memory @ prediction`.
- [x] Protocol types on maps with memory and prediction:
      `protocol.cod == cod @ prediction`, `protocol.mem` is the paired
      ports followed by the internal memories, an internal memory with a
      swapped-out port is a delay line, and a copied memory is predicted
      without being consumed.
- [x] Backwards compatibility: default `memory` and `prediction` reproduce
      the protocol of #16 on its existing examples, updated to the #15
      boundary conventions (open initial memory, discarded final memory,
      `n_steps` counts unrollings).
- [x] `fix` on a box with one memory qubit and a known stationary
      prediction, and on an optical delay certified without warnings.
- [x] PyTorch gradient through the memory wire: a rotation on the memory
      qubit of a two-step readout network, contracted with
      `contract_tensor`, matches the analytic Born score and gradient.
- [ ] Notebook assertions: topology counts, non-zero initial gradient,
      held-out metrics improving over their initial values, fresh-kernel
      execution.

## Docs and checks

- [ ] Module docstring, doctests and `docs/api.rst` updated for the new
      `Box` signature; a drawing of a cell with memory and prediction wires.
- [ ] `pflake8 optyx`, `pylint optyx/interaction.py --fail-under=9` and
      `coverage run -m pytest` green with coverage at least 95%.
