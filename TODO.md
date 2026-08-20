# TODO

> I have a proposal for a new module in optyx https://github.com/rel-int/optyx/issues/13.
> Make a plan for implementation.

> Check out [rel-int/optyx#16](https://github.com/rel-int/optyx/pull/16), let's stack on the
> fixpoint implementation, we should be able to move on with TODO.md and write the notebook

> So we need the tensor contraction and auto diff

> You ran an experiment with MapRNN but that was not the point! We need to simulate learning
> optyx channels to solve the sudoku, the network is quantum, not a classical NN

> The sudoku notebook should run a learning experiment on a sudoku dataser, similar to the
> MapRNN demonstration.

> Check out the PR https://github.com/rel-int/optyx/pull/16. We need to push this model to try
> to solve the sudoku task. One important concept for the interaction.CMap is that every cell:
> X -> Y should have additionally a memory type M and a prediction type O, so that the process
> inside it has type X @ Y @ M -> X @ Y @ M @ O. So we can run the same experiment we were
> running but such that every cell has 1 qubit internal memory. Make a plan for improvements
> and tests to solve the sudoku task in the notebook. Write the plan in TODO.md.

> Why would cod == dom @ predictions? I don't think this is needed. Go ahead and implement the
> changes to interaction.py. Then give a few more ideas for scaling while keeping the
> contractions doable on MacMini in the TODO.md

> Also rebase on the fixpoint PR

> Check out the TODO.md in [rel-int/optyx#16](https://github.com/rel-int/optyx/pull/16). Let's
> solve these sudokus with recurrent channels! Use at_time with small unroll steps instead of
> fix in the experiments, as we are not sure it will converge. Decide between Jax and PyTorch
> to differentiate the tensor networks. Run the experiments in the notebook.

> That's bad! I think the reason for this is that the dataset is too small. Check out the
> sudoku notebook here [discopy/discopy#416](https://github.com/discopy/discopy/pull/416). Do
> you think we can reach the same dataset sizes? A second thing we can try is changing the
> ansatz for each box, make 2 other proposals to check. Let's experiment with increasing the
> dataset size and playing with the ansatz. Keep budgeting before running too big experiments,
> the tensor contractions will get expensive. Try your best, let's solve these sudokus!

> Then it makes sense that the MapGNN performs much better. Let's try to increase the number
> of parameters

> Try again, allocating more compute to each box, and following the unchecked suggestions in
> the TODO.md to make it efficient. Let's run a few experiments in sequence on the GPU and test
> its limits for contraction

> make sure you run your experiments on the GPU, there's another agent running some experiments
> on the CPU

> Check out the interaction PR https://github.com/rel-int/optyx/pull/16. CI is failing, we
> should avoid introducing heavy tests, let's remove the sudoku notebook entirely, I think it's
> beyond the reach of the current of the tensor contractions we can do. Make a plan with
> proposals for smaller, generated datasets that demonstrates the reasoning capabilities of
> photonic networks. Make sure that the method is linked to the literature and establishes the
> reasoning claim. Update the PR with a detailed TODO.md and a couple of proposals in the PR
> description.

> Additionally, write a documentation for the interaction module explaining how recurrent
> tensor networks are constructed from optyx Diagrams.

> from CMaps I mean

## What this PR ships

`optyx.interaction` implements #13: a `Box` is a typed local recurrent channel
`X @ Y @ M -> X @ Y @ M @ O` — message ports `X @ Y` read and written at every
tick, a private memory `M` fed back to the box itself, a prediction `O` written
to the environment and never read. A `CMap` is a list of boxes plus a pairing of
their ports; its semantics is `protocol`, an `optyx.channel.Diagram` with
feedback on the paired ports, with finite semantics `unroll` and stationary
semantics `fix`. This is the Int-construction of Joyal–Street–Verity applied to
the feedback category of channels: `@` is the disjoint union and the edges are
the compact closed structure. The recurrent tensor network is contracted with
DisCoPy's own methods (`eval`, `to_quimb`); the differentiable and compressed
contraction stays proposed in #21.

## Why the Sudoku notebook is gone

Kept here as the measured reason, not as work to resume.

4x4 Sudoku needs 16 cell boxes and 12 constraint boxes with 8- and 9-qubit
local channels, giving a three-tick network of ~660 tensor boxes and ~1,920
ports. Measured on the full map: three ticks at `chi=4` took 117 s per four-way
forward/backward and held 10.4 GiB of live MPS tensors; four ticks did not fit
at any bond. Restricting the loss to one cell's light cone brought this down to
337 boxes and 3 s per gradient, and three ansatzes were compared at
12,849 shared parameters. The best (nearest-neighbour ring) reached 26.6%
hidden-cell accuracy on held-out grids — the random baseline is 25% — and
solved 0 of 16 grids. Unclipped gradient norms spanned 88.8 to 227,061.

Two things are wrong and neither is fixed by more compute. The local channels
are dense `2 ** 8 x 2 ** 8` and `2 ** 7 x 2 ** 9` maps, so the tensor network
is expensive before any propagation happens; and 4x4 Sudoku needs constraints
to travel four hops in three ticks, which the trained map never learned, so
there is no signal separating "learned to propagate" from "learned the digit
marginals". A benchmark that cannot fail informatively is not evidence.

Nothing about `optyx.interaction` depends on the notebook: the module, its
doctests and `test/test_interaction.py` stand on their own.

## What counts as evidence of reasoning

Every experiment below is held to the same protocol, so that a positive result
means propagation and a negative result is informative. The last three items
are what the Sudoku notebook was missing.

- **Exact verifier.** Accuracy is measured against a checkable certificate
  (a parity, a distance, an isomorphism invariant), never partial credit.
- **Held-out by solution.** The split is disjoint in the *solution*, not just
  the instance, so a memorised answer table cannot transfer.
- **Out-of-distribution size.** Train on the small sizes, test on strictly
  larger ones. Box parameters are shared across boxes and ticks, so the same
  `CMap` weights apply verbatim at any size — this is the property that makes
  the test meaningful and that a fixed-width circuit ansatz does not have.
- **Test-time recurrence.** Train at `T_train` ticks, evaluate `unroll(T)` for
  `T > T_train` and plot accuracy against `T`. A propagating map improves with
  extra ticks and shows a knee at `T ~ diameter`; a pattern matcher is flat.
  This is the NeuroSAT and easy-to-hard signature and it is the single
  cheapest discriminating measurement we have.
- **Ablations that must fail.** `T = 1`; edges rewired at random; `memory=Ty()`;
  labels shuffled. Any ablation that still succeeds voids the claim.
- **Matched classical baseline.** The same `CMap` with `bit` wires instead of
  `qubit`/`qmode` wires, at matched parameter count. optyx types classical and
  quantum wires in one category, so this is a one-line change of encoding and
  isolates exactly what the quantum channel buys.
- **Pre-registered budget.** Contraction count and wall-clock cap stated before
  the run; negative results reported rather than re-tuned away.

### Literature

- D. Selsam et al., *Learning a SAT Solver from Single-Bit Supervision*,
  ICLR 2019, [arXiv:1802.03685](https://arxiv.org/abs/1802.03685) — message
  passing on a constraint graph trained on one bit of supervision, solving
  larger instances at test time "by simply running for more iterations".
- R. B. Palm, U. Paquet, O. Winther, *Recurrent Relational Networks*,
  NeurIPS 2018, [arXiv:1711.08028](https://arxiv.org/abs/1711.08028) — the
  recurrent message-passing formulation of Sudoku that discopy#416 reproduces.
- A. Schwarzschild et al., *Can You Learn an Algorithm? Generalizing from Easy
  to Hard Problems with Recurrent Networks*, NeurIPS 2021,
  [arXiv:2106.04537](https://arxiv.org/abs/2106.04537) — prefix sums and mazes;
  the accuracy-versus-recurrence protocol used above.
- P. Veličković, C. Blundell, *Neural Algorithmic Reasoning*, Patterns 2021,
  [arXiv:2105.02761](https://arxiv.org/abs/2105.02761), and the CLRS benchmark,
  ICML 2022, [arXiv:2205.15659](https://arxiv.org/abs/2205.15659) — step-wise
  hint supervision and size generalisation as the standard of evidence.
- K. Xu et al., *How Powerful are Graph Neural Networks?*, ICLR 2019,
  [arXiv:1810.00826](https://arxiv.org/abs/1810.00826); C. Morris et al., AAAI
  2019, [arXiv:1810.02244](https://arxiv.org/abs/1810.02244) — message passing
  is bounded by 1-WL.
- K. Brádler, S. Friedland, J. Izaac, N. Killoran, D. Su, *Graph isomorphism
  and Gaussian boson sampling*, Special Matrices 9:166–196, 2021,
  [arXiv:1810.10644](https://arxiv.org/abs/1810.10644); M. Schuld et al.,
  *Measuring the similarity of graphs with a Gaussian boson sampler*,
  Phys. Rev. A 101, 032314 (2020),
  [arXiv:1905.12646](https://arxiv.org/abs/1905.12646) — photon-counting
  statistics of a graph-encoded interferometer are graph invariants beyond
  1-WL.
- P. L. McMahon et al., *A fully programmable 100-spin coherent Ising machine
  with all-to-all connections*, Science 354:614–617 (2016),
  [doi:10.1126/science.aah5178](https://doi.org/10.1126/science.aah5178) — a
  photonic network that solves constraint problems by recurrent measurement
  feedback, i.e. the physical reading of `CMap.protocol`.
- E. Knill, R. Laflamme, G. J. Milburn, *A scheme for efficient quantum
  computation with linear optics*, Nature 409:46–52 (2001) — why the
  nonlinearity that makes these maps expressive is measurement and feedback,
  which is what a `CMap` tick is.

## Land the module

- [x] Remove `examples/sudoku.ipynb`.
- [x] Time every test and doctest that `optyx.interaction` and
      `optyx.core.contract` add, and cap the whole addition at 10 s of the
      `test` job: measured 2026-08-11, the module doctests take 0.57 s and
      every test this PR adds is below 0.7 s; the slow tail
      (`test_fix_truncates_a_growing_photon_budget`, 12.9 s) belongs to the
      fixpoint PR #15, not to this one.
- [x] Confirm the module needs no dependency that `pip install .[test]` does
      not already pull: `quimb` comes transitively through the hard
      dependency `graphix` and `cotengra` through `quimb`. Worth declaring
      both explicitly in `pyproject.toml` since `optyx.channel` and
      `optyx.core.contract` import them at module level — file as an issue.
- [x] Get one green run of `lint`, `test` and `docs` on this branch.
      Verified 2026-08-19: GitHub check-runs on the head commit are all
      `success` (lint, test, docs), and reproduced locally under
      Python 3.12 — `pflake8 optyx` clean; `pylint optyx --fail-under=9`
      (the CI invocation) rates 9.57/10; `coverage run -m pytest` passes
      1286 tests, `coverage report --fail-under=95` passes at 95% total;
      the docs job's `jupyter nbconvert --execute
      docs/notebooks/fixpoints.ipynb` and `sphinx-build docs
      docs/_build/html` both complete without error.
- [x] Merge the target branch in (never rebase, per RULES.md) and rerun.
      `git merge-base --is-ancestor
      origin/claude/optyx-new-module-plan-p4k3ip HEAD` is already true —
      the base is fully contained in this branch, nothing to merge.

## Proposal A — XOR chains: propagation over a distance

The smallest task whose answer provably needs `n` rounds of message passing.
`n` variables in a chain, constraints `b_i` with `x_i XOR x_{i+1} = b_i`, one
endpoint clamped, predict the other endpoint: the answer is the parity of all
`b_i`, so no local rule and no bounded number of ticks can produce it. This is
the prefix-sum task of Schwarzschild et al. in constraint form, and the
propagation core of NeuroSAT.

`CMap`: one `Box(f"v{i}", qubit, qubit, channel, memory=qubit ** 2,
prediction=qubit)` per variable, `edges = [((i, 1), (i + 1, 0))]`, so the
boundary is the two chain endpoints — the clamp is the `input_state`. The two
memory qubits hold the running estimate and the constraint bit `b_i`, prepared
by `initial_state`. Local channel is `qubit ** 4 -> qubit ** 5`, a
512-entry Kraus tensor; contracted at Kraus level with postselection, a 16-box
24-tick network is ~400 tensors of bond 2. No compression, no GPU.

- [ ] Generator in `test/fixtures` (or a small module the notebook imports):
      `xor_chain(n, rng)` returning constraints, clamp and certified answer;
      at `n = 4` enumerate the whole 16-instance space so the split is exact.
- [ ] Shared box ansatz: real orthogonal, two single-qubit rotation layers and
      a nearest-neighbour ring of controlled rotations — the ansatz that won
      the Sudoku pilot — at ~40 parameters per box, shared over boxes and ticks.
- [ ] Train at `n = 4`, `T_train = 4`, PyTorch autodiff through
      `contract_tensor`, budget 2,000 scalar contractions and 5 minutes CPU.
- [ ] Evaluate at `n = 8, 12, 16, 24` with `T = n`, and produce the
      accuracy-versus-`T` curve at fixed `n = 16` for `T = 1 .. 32`.
- [ ] Run the four ablations and the matched `bit`-wire classical baseline.
- [ ] Photonic encoding: repeat with single-rail `qmode` messages truncated at
      one photon, boxes built from `photonic.BS` and phase shifters with a
      heralded ancilla, so the trained object is a linear-optical network with
      measurement feedback rather than an abstract unitary. Report both
      encodings; state the local dimension and contraction count for each.
- [ ] Probe `CMap.fix` on the trained map: a chain that has propagated has a
      stationary state, so the fixed point should agree with `unroll(T)` for
      large `T`. This is the one place the stationary semantics of #15 is
      tested against something with a known answer.

## Proposal B — Photonic interference beyond 1-WL

A separation rather than a benchmark: exhibit graphs that no classical
message-passing scheme can tell apart and that a photonic `CMap` does. The
minimal generated pair is `C_6 + C_6` against `C_12` — both 2-regular, so 1-WL
assigns identical colours forever — followed by decalin against bicyclopentyl
(10 vertices) and, if those are too easy, the 4x4 rook's graph against the
Shrikhande graph (16 vertices, indistinguishable even by 3-WL).

`CMap`: one box per vertex with one `qmode` port per incident edge, a single
photon injected, boxes built from beam splitters and phase shifters, readout
from the photon-number statistics on the prediction wires after `T = 2 .. 4`
ticks. Two ports per box on the 2-regular pair: a 12-box, 4-tick network with
local dimension 3 truncated at one photon.

The claim has a proof obligation and an experiment. Classical message passing
on these pairs is provably constant (Xu et al., Morris et al.); the photonic
readout is a sum over closed walks with interference phases, and at the level
of Fock statistics is a permanent/hafnian of the adjacency matrix, which
Brádler et al. show is a complete set of graph invariants. Running the same
`CMap` with `bit` wires is the controlled ablation and must sit at chance.

- [x] Generator for the three 1-WL-equivalent pairs plus a 1-WL check, so the
      indistinguishability is asserted rather than asserted-in-prose:
      `examples/beyond_1wl.py`, asserted in `test/test_beyond_1wl.py`.
- [x] Encode a graph as a `CMap` and read out photon-number statistics;
      report the separation margin against shot noise for a stated sample
      count, not just the exact amplitude: margins 1.6e-2 (2C3/C6),
      2.2e-3 (2C6/C12), 4.3e-3 (decalin/bicyclopentyl), i.e. 1e5 to 5e6
      shots per vertex for five sigma; rook/Shrikhande are cospectral, so
      one photon measurably does not separate them (Gamble et al., PRA
      81, 052313).
- [x] Classical ablation at chance and a control that must separate: the
      decohered run of the same map (photon measured every tick) is exact
      zero on every 1-WL pair while separating `C6 vs P6`; the vanilla
      GNN and the MapNN land on the same bound. The `bit`-wire CMap form
      of the ablation stays open below.
- [x] Say plainly in the notebook what this does and does not show: it is a
      separation on graph invariants, related to walk and matching counts, not
      evidence of constraint propagation. Proposal A carries that claim.

## Proposal C — Mazes, if A and B land

The natural scale-up, and the one place the module meets an existing benchmark
without inventing a dataset: grid mazes from Schwarzschild et al., generated
locally. One box per cell with four `qmode` ports plus memory and prediction;
train on 3x3 with `T_train = 4`, test on 5x5 and 7x7 with `T = 12`, reading the
same accuracy-versus-`T` curve. Held for after A, because it costs an order of
magnitude more contraction and answers the same question.

- [ ] Only start this once A has a positive accuracy-versus-`T` knee.

## Contraction work these need

Carried over from the Sudoku plan because the new experiments still want them,
minus everything that only existed to make a 660-box network fit.

- [ ] Batch the contraction over instances: one batch index through the
      boundary states and effects, one Cotengra path reused across the batch.
- [ ] Contract the unrolling as a boundary MPS in the time direction, each
      tick an MPO on the memory wires, so peak memory is set by `chi` and the
      memory cut rather than by the whole network; compare with Cotengra's
      compressed paths.
- [ ] Keep everything real: orthogonal ansatz, `float32` forward with `float64`
      loss accumulation.

## Review of 2026-08-17

> This is not a good example (only a delay!). Give an example with three cells
> connected in a triangle. The example should be minimal such that it is purely
> quantum and every cell has an internal memory and a prediction output.

> Why is this file needed? Can't we just use the tensor contraction methods in
> DisCoPy?

> For the same simple protocol above, compute the fixpoint through contraction.
> This conract_tensor method should not be needed, as per my comment above.

> No point to this method. We can simply initialise the map directly with the
> right edges.

> Draw this somewhere in the doctests

- [x] Replace the delay example in the module docstring by a triangle of
      three cells, purely quantum, each with an internal memory and a
      prediction output: a cell is a single Z spider on its two message
      ports, its memory and its prediction. Started from uniform
      superpositions the cells synchronise and the stationary prediction
      is the GHZ mixture.
- [x] Remove `optyx.core.contract`: `QuimbBackend.eval` reverted to the
      DisCoPy `to_quimb` route, `test/test_contract.py` dropped, doctests
      use `eval` and `to_quimb`. The differentiable and compressed
      contraction — and the PyTorch gradient test that needed it — return
      to the scope of #21: DisCoPy's own einsum path currently breaks on
      diagrams with spiders or swaps under the PyTorch backend, so #21
      remains the place to add it.
- [x] Compute the fixpoint of the triangle protocol through contraction in
      the doctests, without `contract_tensor`.
- [x] Remove `CMap.glue`: maps are initialised directly with the right
      edges.
- [x] Draw `CMap.step` in the doctests, with `wire_labels=False` to keep
      the permutations legible; drawing `CMap.unroll` directly hits the
      multi-wire `Discard` drawing bug #35 (minimal case
      `Diagram.swap(qubit, qubit) >> Discard(qubit ** 2)`), so the docs
      draw two open ticks composed by hand instead.

## Docs and checks

- [x] `pflake8 optyx`, `pylint optyx/interaction.py optyx/core/contract.py
      --fail-under=9`, `coverage run -m pytest` with coverage at least 95%.
      `optyx/core/contract.py` no longer exists — removed by this PR's own
      "Remove `optyx.core.contract`" point above — so `pylint` is run on
      `optyx/interaction.py` alone (9.80/10) and, matching the CI `lint`
      job's actual invocation, on the whole package (9.57/10); both clear
      `--fail-under=9`. `pflake8 optyx` is clean. `coverage run -m pytest`
      passes 1286 tests; `coverage report -m` is 95% total (module-level
      lowest is `optyx/core/backends.py` and `optyx/qubits.py` at 92-93%),
      matching CI's `coverage report --fail-under=95`.
- [x] Module documentation explaining how recurrent tensor networks are
      constructed from a `CMap`: one tick as `read >> parallel >> write`,
      `protocol` as delayed feedback, `unroll` as a finite channel diagram,
      `double().to_tensor().to_map()` down to `contract_tensor`, with
      doctested drawings of the protocol and its unrolling.
- [x] A drawing of a box with memory and prediction wires in the module
      docstring, and `docs/api.rst` entry (already added) rendering.
      `docs/api.rst:11` already lists `optyx.interaction`, confirmed
      rendering in the local `sphinx-build` above. Judgement call on the
      drawing: `optyx/interaction.py`'s module docstring already draws
      `triangle.step`/`.protocol`/`two_ticks` (its `cell` box has both
      `memory` and `prediction`), and the `Box` docstring has a runnable
      `memory=qubit, prediction=qubit` example; on top of that, the
      "One box is a stateful channel" section of
      `docs/notebooks/beyond_1wl.ipynb` (added for the 2026-08-18 "Driven
      protocol" point above) draws exactly this — a single box's local
      channel with the memory fed back, `Create(1)` and its prediction
      tap. No further diagram needed.
- [ ] File as issues anything left unchecked when this PR is signed off.
      Done for this PR's own scope: the two items superseded by the
      2026-08-19 single-notebook revision are filed as rel-int/optyx#59
      and rel-int/optyx#60 and checked off above. Left open: `Proposal A`,
      `Proposal C` and `Contraction work these need` stay unchecked and
      unfiled on purpose — they are a different, larger piece of scope
      (training runs, maze scale-up) explicitly gated on an unresolved
      process question, whether an unattended overnight session may spend
      compute on multi-hour training (giodefelice/desire#17, open). Not
      this bounded pass's call to file or close; revisit once #17 answers.

## Proposal B as three models (2026-08-18)

> Implement proposal B in https://github.com/rel-int/optyx/pull/16 as a
> comparison between three models: 1. A vanilla GNN, 2. a MapNN from
> discopy.neural, and 3. an optyx.interaction.CMap where nodes are
> interferometers with coherent memory and coherent messages

- [x] Vanilla GNN: a textbook isotropic MPNN with permutation-invariant
      readout, run on the 1-WL-equivalent pairs; its outputs on the two
      graphs of a pair must be equal to float precision, over random
      parameter draws — the 1-WL bound observed, not assumed.
- [x] MapNN from `discopy.neural` (discopy#585): the same graphs
      interpreted as port-addressed interaction maps, same invariant
      readout, measured under the same separation test.
- [x] Photonic `CMap`: one `interaction.Box` per vertex, one `qmode`
      port per incident edge, a `qmode` coherent memory and a prediction
      tap; boxes are interferometers (beam splitters and phase shifters),
      one photon injected through the initial memory, escape-time and
      per-mode photon statistics as the readout.
- [x] Notebook `examples/beyond_1wl.ipynb` running all three models on
      the same pairs with the separation margin against shot noise, and
      stating what the comparison does and does not show.

Found while running the three models against discopy main (which
`discopy.neural` needs): two incompatibilities with the pinned discopy,
fixed here so that optyx runs on both — `unpack_layer` in
`optyx.utils.misc` replaces the pre-discopy#438 alternating-layer
indexing, and `optyx.core.path.Matrix` gets `@factory` so that functors
into it map swaps to path matrices.

- [x] `bit`-wire `CMap` ablation: the decohered walk of the notebook,
      expressed as the same `CMap` with classical wires. Superseded by the
      2026-08-19 single-notebook scope (no torch/discopy.neural, notebook
      only); filed as rel-int/optyx#59.
- [x] Multi-photon rook-vs-Shrikhande: needs interactions or measurement
      feedback (KLM) beyond one photon, and multi-photon contraction.
      Superseded by the 2026-08-19 instruction excluding rook vs
      Shrikhande from the notebook as too expensive; filed as
      rel-int/optyx#60.

## Driven protocol (2026-08-18)

> For the photonic CMap, the initial state in the memory should be Create(0),
> one photon should be input at every time step. Let's see if we can separate
> Rook and Shrikhande too. To illustrate the model, draw the interpretation of
> a single box in the CMap as a stateful channel (instead of the step of the
> protocol).

- [x] Drive the map through the
      boundary: vacuum initial memory, one photon injected at the source
      vertex at every tick, the mechanism of `CMap.fix`'s `input_state`.
- [x] Second-order photon
      statistics: with several photons in flight, two-photon interference
      gives permanental invariants beyond the single-particle spectrum;
      measure whether they separate rook 4x4 from Shrikhande. Measured:
      the driven margins grow to 3.5e-2 / 9.9e-3 / 1.5e-2 on the
      non-cospectral pairs (2e4 to 2.5e5 shots per source for five
      sigma), but rook vs Shrikhande stays at zero in the means, the
      pairwise coincidences and the three-photon coincidences -- and
      the two transfer matrices are equal up to an output relabelling
      and a phase per injection tick, which every permanent modulus is
      invariant under, so the statistics agree at every order however
      many photons are injected. Gamble et al. (PRA 81, 052313) prove
      the one- and two-walker cases for Hamiltonian walks; the
      all-orders statement here is checked directly for this encoding.
      The open item below sharpens to number-resolved feed-forward
      inside the loop.
- [x] Draw one box as a stateful
      channel — its local channel with the memory fed back — instead of the
      step of the protocol, in the notebook.

## One self-contained docs notebook (2026-08-19)

> Remove the hosting PR you've created in DisCoPy, you can just use
> https://github.com/discopy/discopy/pull/399 as base. Also, edit the optyx PR
> so that the contribution is a single self-contained notebook in the docs,
> showing the separation for all the graphs you used except for Rook vs
> Shriklande (that's too expensive for a documentation notebook). The notebook
> should simply state at the beginning that GNNs (and MapNNs) provably cannot
> distinguish between these graphs (giving a reference for GNNs). This way we
> don't even need the pytorch and discopy neural import in the PR. The notebook
> should be self-contained, defining the photonic CMap implementation and the
> separation results.

- [x] One self-contained
      notebook in `docs/notebooks/`, defining the photonic `CMap`
      implementation inline and showing the separation for every pair
      except rook vs Shrikhande, stating up front that GNNs (and MapNNs)
      provably cannot distinguish these graphs, with a reference:
      `docs/notebooks/beyond_1wl.ipynb`, committed executed against the
      pinned discopy and added to the docs toctree. Margins at eight
      ticks: 3.5e-2 (2C3 vs C6), 6.3e-3 (2C6 vs C12), 1.0e-2 (decalin
      vs bicyclopentyl), 4.2e-2 (control); decohered blind at 1e-16 on
      every pair; the fast pipeline validated against the contraction
      of `CMap.unroll` inside the notebook.
- [x] Remove
      `examples/beyond_1wl.py`, `examples/beyond_1wl.ipynb`,
      `test/test_beyond_1wl.py`, the torch path and the discopy-main
      compatibility changes, so the PR diff is the notebook alone. The
      compatibility changes (`unpack_layer`, `@factory` on
      `path.Matrix`) are reverted but preserved in history at d2e743c
      should optyx move its discopy pin; the two determinism fixes stay
      (seeded `chip_mzi`, exact contraction when the bonds fit
      `max_chi`), since both repaired tests that failed intermittently
      on this branch's CI.
- [x] Close the discopy
      hosting PR discopy#593: closed, discopy#585 can use discopy#399
      as base.

## Active photonics: classical feed-forward (2026-08-20)

> Add classical feedforward to the stateful channel and see if you can separate
> between Rook and Shriklande. Open a separate pull request with the full
> experiment, saying how which k-WL can be separated with passive vs active
> photonics

- [x] Classical feed-forward
      in the cell: threshold detection on the tap, the outcome stored in
      a classical `bit` in the memory and *broadcast one hop along the
      edges* (the ports become `qmode @ bit`), the stored bit controlling
      a second beam splitter — drawn as a stateful channel with
      `Create(1)` in the loop — and an exact two-photon simulation of
      the post-selected sector of the homogeneous drive. Measured, with
      the separation of rook 4x4 from Shrikhande: kick off, exactly
      blind (7e-18); detection re-programming only its own cell, still
      exactly blind (7e-18) — strong regularity absorbs one marked
      vertex; detection broadcast one hop, separation 1.7e-3 at twelve
      ticks (~9e6 post-selected pairs at five sigma), the classical
      layer writing the 2C3-vs-C6 neighbourhood into the optics. A
      conditional *phase* never separates however applied to one cell:
      conditional phases sit inside the gauge freedom. Addresses
      rel-int/optyx#60.
- [x] A separate pull
      request with the full experiment as a self-contained docs notebook,
      stating which k-WL classes passive and active photonics separate,
      with references: `docs/notebooks/beyond_3wl.ipynb`, committed
      executed, validated against `CMap.step.to_path()` and the
      contraction of `CMap.unroll` at machine precision.

## Literature positioning and paper proposal (2026-08-20)

> Can you look in the literature if anyone already showed that quantum models
> can separate graphs beyond 1-WL?

> Can you look more closely at the Kashefi paper? Except for the photonics, is
> there a difference between their message passing GNNs and our model? What
> does their result say about j-WL separation? At first glance it seems that
> they show that these QGNNs can distinguish up to 4-WL and not further, but
> there's some weird j-1 happening

> Ok that's interesting. Fold this in the notebook explanation and add a
> markdown file with a proposal for a new paper, introducing our quantum Map
> neural networks, their photonic instantiations and their advantage compared
> to previous models, with a proposal for experiments to show separations and
> the potential of our new model

- [x] Fold the literature
      into `docs/notebooks/beyond_3wl.ipynb`: prior quantum-beyond-1WL
      work (Emms et al. walk spectra, the cellular-algebra limitation
      theorems, EQGC, the Raj et al. message-passing QGNN) and the
      precise reading of arXiv:2606.26873 — set-based j-WL is tuple
      (j-1)-WL, proved for j <= 4, so their guarantee tops out at
      tuple 3-WL and cannot cover same-parameter strongly regular
      pairs; purely unitary circuits with no measurement in the loop.
- [x] A markdown proposal
      for a new paper introducing quantum Map neural networks, their
      photonic instantiation, their advantages over previous models,
      and the experiments to run:
      `docs/quantum_map_neural_networks.md`, with six proposed
      experiments (broadcast-radius ladder on CFI/BREC, the
      generalized-quadrangle pairs where walk invariants failed,
      trained cells with the bit-wire ablation, deeper photon
      sectors, the theory obligations, hardware feasibility).
