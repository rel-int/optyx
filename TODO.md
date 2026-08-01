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

- [x] Replace the topology-only placeholder with a seeded quantum-channel
      learning experiment whose recurrent tensor routes come from the Optyx
      `CMap`.
- [x] Enumerate the complete 4x4 sudoku corpus, split selected solutions into
      training and held-out sets, and report conditional completion metrics.

## Quantum learning correction

- [x] Remove the classical MapRNN interpreter and parameterise actual Optyx
      channel tensors on the sudoku interaction graph.
- [x] Contract the unrolled quantum network through the backend-neutral tensor
      routine, backpropagate the sudoku loss and verify non-zero gradients and
      held-out learning.

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

- [x] end-to-end gradients: parametrised box channels, unrolled diagram
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
- [x] demonstrate three-step quantum message passing and define a conditional
      Born loss on the 2 prediction qubits per cell
- [x] compile port-addressed recurrence from `CMap.partner` directly to a
      combinatorial tensor map, avoiding a materialised 224-wire permutation
- [x] replace identity channels with shared trainable unitary ansatzes and
      backpropagate through compressed Optyx tensor contraction
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

> Two problems: 1) I see you added a lot of pylint disables, inline comments are not allowed in optyx as in discopy! remove them and check the pylint, 2) The tensor backend is a large addition, write an issue and open a separate PR directly on main that proposes a new routine for evaluating optyx tensor contractions. The logic needs to be simple and flexible accross quimb, cotengra, jax and pytorch

## Automatic differentiation

- [x] Preserve accelerator arrays and gradients while materialising structural
      tensors and contracting with a Cotengra path.
- [x] Test non-zero PyTorch gradients through an Optyx channel tensor rather
      than testing constant output alone.

## Tensor contraction routine

- [x] Implement issue #20 as one deterministic routine
      over ``discopy.tensor.Diagram`` for exact NumPy, JAX, PyTorch and Quimb
      contraction, with arbitrary Cotengra optimizers and optional bond limits.
- [x] Reuse the routine from the existing Quimb and DisCoPy evaluators, add
      concise documentation and tests, and run lint, pylint and coverage.
- [x] Open a draft pull request directly against ``main`` and leave every
      checklist item complete.
- [x] Cover structural spider materialisation when the optional JAX and
      PyTorch test dependencies are unavailable in CI.

> The notebook should be structured in two parts:
> 1) Boson sampling with feedback
> Explain the setup of Biriukov and Dyakonov and the two simulation methods implemented.
> 2) Learning fixpoints
> Start from a beam splitter 2 -> 2, we feed a photon at every time step to the first
> input, we feed the second output into the second input. We compute the fixpoint of an
> example (does it converge? how big is the density matrix?). Now, the beam splitter has
> two phase parameters (transmittivity, reflexivity) that we can adjust. We tune the
> parameters to find fixpoints that converge faster, to lower dimensional density matrices.
>
> Replace this section by a comparison of the complexity of the power method vs the eigen
> method in terms of the approximation of the stationary distribution.
>
> These hanging methods should be avoided, aither use inline, or define methods of the
> existing classes.
>
> This method could be defined over any optyx diagram without feedback loops, finding the
> eigenvectors of an operator.
>
> Isn't this in optyx Ty already?
>
> The trace can be obtained diagrammatically as the Discard, why do we evaluate it
> separately from the rest of the tensor contraction machinery? We should hand off the
> contraction to the backend and formulate everything diagrammatically
>
> This seems like part of a method about the stationary state and needs to be simplified

## Upstream fixed-point review

- [ ] `docs/examples/fixpoints.rst` part 1, boson sampling with feedback: state
      the Biriukov–Dyakonov setup and say which of its pieces `power` and
      `eigen` implement.
- [ ] Part 2, learning fixpoints: a beam splitter `2 -> 2` with one photon into
      the first input at every step and the second output fed back into the
      second input; report convergence and the density-matrix size.
- [ ] Part 2, tuning: adjust the two beam-splitter phases towards fixed points
      that converge faster and have lower-dimensional density matrices.
- [ ] Replace the backend-comparison section with a complexity comparison of
      `power` against `eigen` as approximations of the stationary distribution.
- [ ] Drop `doubled_dimensions`: `Ty` already carries its doubled dimensions,
      so read the cutoff convention off `Ty.double()`.
- [ ] Build the trace as a `Discard` diagram and hand it to the backend instead
      of calling `to_tensor` followed by a hand-written `np.tensordot`.
- [ ] Define `stationary_vector` over any Optyx diagram without feedback loops,
      as eigenvectors of the denoted operator rather than a superoperator array.
- [ ] Fold module-level fixed-point helpers into methods of existing classes or
      inline them at their single call site.
- [ ] Simplify normalisation as part of the stationary-state method.

The tuning work depends on differentiable contraction from #21. Landing #21
first keeps it to a few lines; without it, tuning falls back to a parameter
sweep.
