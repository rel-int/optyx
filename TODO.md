# TODO

> I have a proposal for a new module in optyx https://github.com/rel-int/optyx/issues/13. Make a plan for implementation.

Solves #13. Blocked on #12: the module is built on `channel.Diagram.feedback`
and `unroll`, so the implementation stacks on `claude/optyx-issue-6-g9z1pv`
(rebase onto `main` once #12 lands).

Mathematically, `optyx.interaction` is the Int (geometry of interaction)
construction applied to the feedback category of optyx channels: feedback
plays the role of a delayed trace, so the result is compact closed up to
time shifts. A `CMap` is a combinatorial map whose boxes carry channels and
whose edges pair ports; its semantics is a recurrent protocol, evaluated by
unrolling to a tensor network.

## `optyx.interaction` module

- [WIP] @session_014tZqfLpxjnFUrMsKKFZUj2-2026-07-31 12:03 `interaction.Ty`: tuples of dimensions, wrapping `channel.Ty`
      (`bit`, `qubit`, `qmode`) so boxes plug into the existing channel layer
- [WIP] @session_014tZqfLpxjnFUrMsKKFZUj2-2026-07-31 12:03 `interaction.Box(name, dom, cod, channel)` where `channel` is an
      `optyx.channel.Diagram` with domain `dom @ cod` and codomain `dom @ cod`,
      as in the issue: every port is both read and written at each time step
      (a plain channel `dom -> cod` embeds by initialising / discarding)
- [WIP] @session_014tZqfLpxjnFUrMsKKFZUj2-2026-07-31 12:03 `CMap`: a set of boxes and a set of edges pairing output ports to input
      ports, type-checked at construction; unpaired ports are the boundary,
      giving the domain and codomain of the map
- [WIP] @session_014tZqfLpxjnFUrMsKKFZUj2-2026-07-31 12:03 `CMap.protocol`: the `channel.Diagram`
      `(parallel >> perm).feedback(dom, cod, mem)` where `parallel` is the
      tensor of all box channels, `perm` the permutation given by the edges
      (boundary ports first, memory last) and `mem` the paired ports
- [WIP] @session_014tZqfLpxjnFUrMsKKFZUj2-2026-07-31 12:03 `CMap.unroll(n_steps)` = `CMap.protocol.unroll(n_steps)`, a
      `channel.Diagram` evaluated as a tensor network by the existing backends
- [WIP] @session_014tZqfLpxjnFUrMsKKFZUj2-2026-07-31 12:03 compact closed structure: identity and composition glue maps along
      boundary ports, cups and caps are single edges; document and test which
      snake equations hold on the nose and which only up to a time delay
- [WIP] @session_014tZqfLpxjnFUrMsKKFZUj2-2026-07-31 12:03 drawing: reuse `to_drawing` on the protocol so a `CMap` and its
      unrollings can be rendered

## `CMap.fix`

- [WIP] @session_014tZqfLpxjnFUrMsKKFZUj2-2026-07-31 12:03 `CMap.fix(backend=None, tol=..., max_steps=..., chi=...)` approximates
      the stationary output distribution: unroll for doubling `n_steps`,
      contract with `QuimbBackend` and `HyperCompressedOptimizer` at bond
      dimension `chi`, stop when successive output distributions are within
      `tol` (total variation); ramp `chi` until the answer is stable in `chi`
- [WIP] @session_014tZqfLpxjnFUrMsKKFZUj2-2026-07-31 12:03 optional exact bound when `mem` is small: mixing time from the second
      largest eigenvalue modulus of the transfer channel of `protocol`
- [WIP] @session_014tZqfLpxjnFUrMsKKFZUj2-2026-07-31 12:03 return the `EvalResult` together with convergence diagnostics
      (`n_steps`, `chi`, distances per iteration)

## Differentiability and training

- [WIP] @session_014tZqfLpxjnFUrMsKKFZUj2-2026-07-31 12:03 end-to-end gradients: parametrised box channels, unrolled diagram
      contracted with torch-backed quimb tensors; test that gradients flow
      through `contract` and `contract_compressed`
- [WIP] @session_014tZqfLpxjnFUrMsKKFZUj2-2026-07-31 12:03 batching over classical inputs and targets as in the neural-sudoku
      experiment of discopy#416: a batch dimension threaded through
      evaluation, plus a loss helper against a target distribution

## Notebook

- [WIP] @session_014tZqfLpxjnFUrMsKKFZUj2-2026-07-31 12:03 `examples/sudoku.ipynb`: 4x4 sudoku as a `CMap` with 16 cell
      boxes and 12 block boxes (4 rows, 4 columns, 4 squares); each cell has
      3 block neighbours plus 2 prediction qubits (3 ports in, 5 out), each
      block has its 4 cells as neighbours (4 in, 4 out)
- [WIP] @session_014tZqfLpxjnFUrMsKKFZUj2-2026-07-31 12:03 message passing for `n_steps`, loss as distance from the correct
      prediction on the 2 extra qubits per cell, backpropagation through the
      tensor network; compare against the classical CMap-GNN of discopy#416
- [WIP] @session_014tZqfLpxjnFUrMsKKFZUj2-2026-07-31 12:03 stretch: make `n_steps` dynamic, inferred during training via
      `CMap.fix`

## Tests and docs

- [WIP] @session_014tZqfLpxjnFUrMsKKFZUj2-2026-07-31 12:03 `test/test_interaction.py`: type checking, protocol construction,
      unrolling against hand-built diagrams, snake equations, `fix`
      convergence on a small channel with known stationary state, gradients
- [WIP] @session_014tZqfLpxjnFUrMsKKFZUj2-2026-07-31 12:03 doctests, `docs/api.rst` entry, drawings for the docs
- [WIP] @session_014tZqfLpxjnFUrMsKKFZUj2-2026-07-31 12:03 `pflake8 optyx` and `coverage run -m pytest` green
