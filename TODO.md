# TODO

> So we need the tensor contraction and auto diff

> You ran an experiment with MapRNN but that was not the point! We need to simulate learning optyx channels to solve the sudoku, the network is quantum, not a classical NN

> The sudoku notebook should run a learning experiment on a sudoku dataser, similar to the MapRNN demonstration.

> Check out [rel-int/optyx#16](https://github.com/rel-int/optyx/pull/16), let's stack on the fixpoint implementation, we should be able to move on with TODO.md and write the notebook

> I have a proposal for a new module in optyx https://github.com/rel-int/optyx/issues/13. Make a plan for implementation.

Solves #13. The module is built on `channel.Diagram.feedback` and `unroll` from
#12, then stacks on the `channel.Diagram.fix` implementation in #15.

Mathematically, `optyx.interaction` is the Int (geometry of interaction)
construction applied to the feedback category of optyx channels: feedback
plays the role of a delayed trace, so the result is compact closed up to
time shifts. A `CMap` is a combinatorial map whose boxes carry channels and
whose edges pair ports; its semantics is a recurrent protocol, evaluated by
unrolling to a tensor network.

## Fixed-point stack and notebook round

- [x] Stack this PR on #15, merge its fixed-point
      implementation into the branch and keep the interaction diff isolated.
- [x] Define `CMap.fix` by exposing the stationary
      semantics of the recurrent `protocol`, without duplicating its
      convergence or backend logic.
- [x] Write `examples/sudoku.ipynb` as an executable
      construction of the 16 cell and 12 constraint boxes, finite message
      passing and the closed-map fixed-point boundary.
- [x] Add focused tests and run lint, coverage and
      notebook validation.

## Sudoku learning experiment

- [x] Replace the topology-only placeholder with a seeded MapRNN-style
      learning experiment whose message routes come from the Optyx `CMap`.
- [x] Split the complete 4x4 sudoku corpus before masking, train shared cell
      and constraint modules, and report held-out loss, cell accuracy and
      valid-grid accuracy.

## Quantum learning correction

- [WIP] @Codex-2026-08-01-05:30 Remove the classical MapRNN interpreter and
        parameterise actual Optyx channel tensors on the sudoku interaction
        graph.
- [WIP] @Codex-2026-08-01-05:30 Contract the unrolled quantum network through
        the backend-neutral tensor routine, backpropagate the sudoku loss and
        verify non-zero gradients and learning.

## `optyx.interaction` module

- [x] `interaction.Ty`: tuples of dimensions, wrapping `channel.Ty`
      (`bit`, `qubit`, `qmode`) so boxes plug into the existing channel layer
- [x] `interaction.Box(name, dom, cod, channel)` where `channel` is an
      `optyx.channel.Diagram` with domain `dom @ cod` and codomain `dom @ cod`,
      as in the issue: every port is both read and written at each time step
      (a plain channel `dom -> cod` embeds by initialising / discarding)
- [x] `CMap`: a set of boxes and a set of edges pairing output ports to input
      ports, type-checked at construction; unpaired ports are the boundary,
      giving the domain and codomain of the map
- [x] `CMap.protocol`: the `channel.Diagram`
      `(parallel >> perm).feedback(dom, cod, mem)` where `parallel` is the
      tensor of all box channels, `perm` the permutation given by the edges
      (boundary ports first, memory last) and `mem` the paired ports
- [x] `CMap.unroll(n_steps)` = `CMap.protocol.unroll(n_steps)`, a
      `channel.Diagram` evaluated as a tensor network by the existing backends
- [ ] compact closed structure: identity and composition glue maps along
      boundary ports, cups and caps are single edges; document and test which
      snake equations hold on the nose and which only up to a time delay
- [x] drawing: reuse `to_drawing` on the protocol so a `CMap` and its
      unrollings can be rendered

## `CMap.fix`

- [x] `CMap.fix(input_state=None, initial_state=None, **params)` prepares a
      fixed boundary input at every step, initializes the recurrent memory and
      delegates the stationary output to `channel.Diagram.fix`, including its
      power/eigen methods, backends and independent depth/bond refinement
- [ ] optional exact bound when `mem` is small: mixing time from the second
      largest eigenvalue modulus of the transfer channel of `protocol`
- [x] return the `EvalResult` from `channel.Diagram.fix`
- [ ] expose convergence diagnostics (`n_steps`, `chi`, distances per
      iteration) from the shared fixed-point implementation

## Differentiability and training

- [ ] end-to-end gradients: parametrised box channels, unrolled diagram
      contracted with torch-backed quimb tensors; test that gradients flow
      through `contract` and `contract_compressed`
- [ ] batching over classical inputs and targets as in the neural-sudoku
      experiment of discopy#416: a batch dimension threaded through
      evaluation, plus a loss helper against a target distribution

## Notebook

- [x] `examples/sudoku.ipynb`: 4x4 sudoku as a `CMap` with 16 cell
      boxes and 12 block boxes (4 rows, 4 columns, 4 squares); each cell has
      3 block neighbours plus 2 prediction qubits (3 ports in, 5 out), each
      block has its 4 cells as neighbours (4 in, 4 out)
- [x] demonstrate finite message passing for `n_steps` and define the loss as
      distance from the correct prediction on the 2 extra qubits per cell
- [x] derive port-addressed routes from `CMap.partner`, then train and evaluate
      shared cell and constraint updates with a notebook-local PyTorch MapRNN
- [ ] replace the topology's identity channels with trainable ansatzes,
      backpropagate through the Optyx tensor contraction and compare its
      convergence against the notebook-local MapRNN
- [ ] stretch: make `n_steps` dynamic, inferred during training via
      `CMap.fix`

## Tests and docs

- [x] `test/test_interaction.py`: type checking, protocol construction,
      unrolling against hand-built diagrams
- [x] `test/test_interaction.py`: `fix` convergence on a small channel with a
      known stationary state
- [ ] `test/test_interaction.py`: snake equations and gradients
- [x] doctests and `docs/api.rst` entry
- [ ] drawings for the docs
- [x] `pflake8 optyx` and `coverage run -m pytest` green
