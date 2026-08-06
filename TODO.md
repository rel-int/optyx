# TODO

> Then it makes sense that the MapGNN performs much better. Let's try to
> increase the number of parameters

> Try again, allocating more compute to each box, and following the unchecked
> suggestions in the TODO.md to make it efficient. Let's run a few experiments
> in sequence on the GPU and test its limits for contraction

## GPU contraction-limit retry

Increase capacity without widening the recurrent graph.  A conditional
rotation on target qubit `q` partitions the computational basis into
`2 ** (w - 1)` pairs that differ only at `q`, and gives every pair an
independent real rotation angle.  Cycling the target through the qubits yields
an orthogonal, near-identity box ansatz whose whole layer is applied by one
vectorised row-pair update.  With product rotations before, between and after
these conditional layers, a depth-`d` cell has `265d + 9` parameters and a
constraint has `136d + 8`, or `401d + 17` shared parameters in total.  Depths
8, 16 and 32 therefore have 3,225, 6,433 and 12,849 parameters; the largest is
within 1.1% of the 12,980-parameter MapGNN in discopy#416.  Parameters remain
shared over boxes and time.  Depth only changes vectorised construction of the
dense local arrays; tensor-map indices are unchanged.  Ticks and compressed
bond dimension are benchmarked separately, using configuration records with
`depth`, `ticks`, `chi`, elapsed seconds, scalar-contraction count and live MPS
memory.

The GPU probes are capped at 48 scalar contractions: three four-way capacity
probes at depths 8, 16 and 32, followed by one four-way loss and backward pass
for ticks in `{2, 3, 4}` and `chi` in `{4, 8, 16}`, in increasing order.  The
ladder stops before the next configuration if a probe takes more than 30
seconds, holds more than 6 GiB of live MPS tensors, or raises an unsupported or
out-of-memory error.  Capacity pilots use at most 384 further contractions:
three depths times 64 training, 32 candidate-ranking and 32 targeted-readout
contractions.  Only the winner may use another 448 contractions: 192 for
training, 128 for all held-out candidate rankings and 128 for exact readout on
four grids.  Total experiment cost is at most 880 scalar contractions.  A
configuration is eligible for training only if its measured time projects the
whole training study below 15 minutes.  MPS is asserted, CPU fallback remains
fatal, paths are reused per `(ticks, chi)`, and the existing real `float32`
representation is retained.

- [x] Implement the vectorised conditional-rotation ansatz and configurable
      compression rank, cache one deterministic Cotengra path per
      `(ticks, chi)`, and add timed MPS-memory diagnostics.
- [x] Run the increasing GPU contraction ladder over two to four ticks and
      `chi` in `{4, 8, 16}`, respecting the time and memory stop conditions.
- [x] Replace the over-budget full-map per-cell loss by the induced local
      `CMap` containing the target, its seven peers and its three constraints;
      close missing message ports with `|+>` states/effects and rerun the
      three-tick ladder.
- [WIP] @codex-019fd73e-2026-08-06 19:48 At the largest affordable local three-tick configuration, compare
      conditional depths 8, 16 and 32 under the fixed pilot budget, then scale
      only the held-out winner.
- [ ] Execute the notebook in a fresh GPU-only kernel, record the limit and
      learning results, and update the conclusions and unchecked suggestions.

The full-map ladder reached all 12,849 parameters without pressure from box
construction: at two ticks, `chi=4`, `8` and `16` took 1.76, 8.54 and 25.11
seconds for a four-way forward/backward and held 0.67, 1.71 and 4.72 GiB of
live MPS tensors.  Three ticks at `chi=4` then took 117.33 seconds and held
10.37 GiB live (11.33 GiB including the MPS driver), crossing both stop limits;
higher three-tick bonds and all four-tick probes were skipped.  The next run
therefore uses the local-loss suggestion rather than training an over-budget
full map.

The local map has 8 cells, 3 constraints, 24 paired edges, 24 open boundary
wires and 56 recurrent wires.  At three ticks it reduces the tensor network
from 596 boxes / 1,856 ports to 337 boxes / 832 ports.  The 12,849-parameter
model completed `chi=4`, `8` and `16` four-way gradients in 3.27, 2.36 and
2.97 seconds using at most 0.58 GiB live MPS memory.  Four ticks completed
through `chi=8` in 4.71 seconds and 2.28 GiB, while `chi=16` hit the MPS
high-water mark after 82.18 seconds.  Thus three ticks at `chi=16` is the
largest configuration eligible for the 15-minute learning budget; four ticks
at `chi=16` is the measured local contraction limit.

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

> Check out the TODO.md in [rel-int/optyx#16](https://github.com/rel-int/optyx/pull/16).
> Let's solve these sudokus with recurrent channels! Use at_time with small
> unroll steps instead of fix in the experiments, as we are not sure it will
> converge. Decide between Jax and PyTorch to differentiate the tensor networks.
> Run the experiments in the notebook.

> That's bad! I think the reason for this is that the dataset is too small.
> Check out the sudoku notebook here
> [discopy/discopy#416](https://github.com/discopy/discopy/pull/416). Do you think
> we can reach the same dataset sizes? A second thing we can try is changing the
> ansatz for each box, make 2 other proposals to check. Let's experiment with
> increasing the dataset size and playing with the ansatz. Keep budgeting before
> running too big experiments, the tensor contractions will get expensive. Try
> your best, let's solve these sudokus!

> make sure you run your experiments on the GPU, there's another agent running
> some experiments on the CPU

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

The experiment below uses the finite semantics of `at_time(1)`, i.e. two
recurrent ticks, instead of assuming convergence of `fix`. Its direct
`tensor.CMap` representation keeps the same recurrent index routing without
materialising the 240-wire permutations. PyTorch was chosen for autodiff:
compressed gradients pass on this machine, whereas the installed experimental
JAX Metal backend fails its basic complex-array contraction (reported as #43).

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

- [x] Rebuild the notebook map: a cell is
      `Box("cell", qubit ** 3, qubit ** 3, channel, memory=qubit,
      prediction=qubit ** 2)` — three messages read, three written, one
      internal memory qubit, two prediction qubits; constraints stay
      `Box(qubit ** 4, qubit ** 4, channel)` with no memory or prediction.
- [x] Cell channels become shared trainable isometries
      `qubit ** 7 -> qubit ** 9`: a parameterised nine-qubit real unitary
      applied to the input tensored with two fresh ancillas, keeping the
      rotation-layer parameterisation and the parameter count comparable
      to #16; constraint channels keep their eight-qubit unitary.
- [x] Check the topology: 96 edges and 192 paired memory wires as in #16,
      plus 16 internal memory wires (208 total), empty `dom` and a `cod` of
      32 prediction qubits per step.
- [x] Move clue injection entirely to the write side: at every step the
      prediction output of a clue cell is postselected on its digit, free
      cells meet uniform effects at intermediate steps and the candidate
      digit at the last step; pick and document the initial memory state.
- [x] Update `at_time_tensor_map`: internal memory ports connect a box to
      itself at the next step, prediction ports get one effect per step and
      no read; keep the direct `tensor.CMap` construction without a
      materialised permutation.

## Solving the task

- [x] Replace two-candidate ranking by a per-cell readout: score the four
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
- [x] Replace the `CMap.fix` probe with `at_time(1)` on one recurrent wire;
      one unrolling gives two explicit ticks without assuming that the learned
      channel has converged.

## Executed finite-time experiment

- [x] PyTorch/Cotengra, `unroll_steps=1`, `max_bond=4`, 24 four-way updates:
      initial gradient norm 6.909; held-out correct-candidate probability
      0.4984 -> 0.5020; per-cell accuracy 21.9% -> 34.4%; candidate accuracy
      stayed at 50%; full-grid solve rate stayed at 0/4. A 96-update sweep
      reduced per-cell accuracy to 25%, so longer training is not assumed to
      help without a better schedule or batching.

## Dataset scale and box ansatz search

The notebook in discopy#416 stores 192 training puzzles and 64 test puzzles,
but its generator produces only 24 distinct completed grids and 22 of the test
solution patterns also occur in training. Of its 256 eight-clue puzzles, 69
admit two or three valid completions. We can match its record count while using
more information: sample 256 distinct solutions from the complete 288-grid
corpus, split them 192/64 before masking, and accept only clue masks with one
completion in that corpus.

All three channel families remain real orthogonal maps with two single-qubit
rotation layers. The baseline places 16 controlled rotations between the first
and last four qubits. The first alternative uses both directions of a cyclic
nearest-neighbour ring, so every message, memory and prediction qubit mixes at
34--36 parameters per channel. The second uses one controlled rotation for
every unordered qubit pair, raising the cell/constraint counts to 54/44 while
leaving their `2 ** 7 x 2 ** 9` and `2 ** 8 x 2 ** 8` tensors unchanged.
Consequently the recurrent contraction graph and its `chi=4` peak tensors do
not grow; only construction and differentiation of the local unitary does.

The model-selection pilot is capped at 480 scalar contractions per ansatz: 12
optimizer updates times four examples times four digits, two-candidate scores
before and after training on eight validation puzzles, and a final four-way
decode of their 64 hidden cells. At the measured baseline rate of about 0.20 s
per contraction, three ansatzes should take about five minutes. Gradients for
the four-example mini-batch are accumulated one example at a time to keep peak
memory near the original four-contraction update. Only the winner is allowed
the remaining 144 train cases and a broader held-out evaluation, budgeted at
about 1,200 further contractions.

The rerun uses PyTorch MPS in `float32` with
`PYTORCH_ENABLE_MPS_FALLBACK=0`, and the notebook asserts the device. Apple's
backend does not implement `torch.linalg.svd`; allowing its default fallback
would silently contend for the CPU. Compressed bonds therefore use a
deterministic rank-four range projection followed by MPS-native QR. This keeps
the same `chi=4` topology and makes an unsupported GPU operation fail instead
of migrating it to the other agent's CPU.

- [x] Build the disjoint 192/64 uniquely-solvable dataset and compare the
      bipartite, nearest-neighbour-ring and all-pairs box ansatzes under the
      fixed pilot budget; scale only the validation winner and record timings,
      contraction counts, losses, per-cell accuracy and full-grid solve rate.
- [x] If none of the three ansatzes beats random per-cell accuracy after the
      pilot, stop before the scale-up and diagnose the four-way energy and
      gradient distributions instead of spending the winner budget.

The final GPU-only run used exactly 2,656 contractions. The bipartite, ring and
all-pairs pilots reached 18.8%, 29.7% and 20.3% validation cell accuracy, so
the ring ansatz beat the 25% random baseline and received the scale-up. After
one 192-example pass, it assigned the correct completion 0.581 mean
probability over all 64 held-out puzzles and ranked it first 59.4% of the time.
Exact four-way decoding reached 26.6% hidden-cell accuracy on 16 held-out
puzzles and solved 0/16 full grids. Median training target probability was
0.040, median target margin was -3.02 and unclipped gradient norms ranged from
88.8 to 227,061, so the channel learns a weak constraint signal but the local
readout remains unstable even with clipping.

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
- [x] Notebook assertions: topology counts, non-zero initial gradient,
      held-out metrics improving over their initial values, fresh-kernel
      execution.

## Docs and checks

- [ ] Module docstring, doctests and `docs/api.rst` updated for the new
      `Box` signature; a drawing of a cell with memory and prediction wires.
- [ ] `pflake8 optyx`, `pylint optyx/interaction.py --fail-under=9` and
      `coverage run -m pytest` green with coverage at least 95%.
