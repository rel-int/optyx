# TODO

> Check out https://github.com/rel-int/optyx/pull/16. The aim of this task is
> to scale up the experiments on modal GPUs to try to reach a good accuracy at
> the sudoku task. Open a new PR "sudoku experiment" , the aim is to learn to
> solve 4x4 sudokus with optyx.interaction. There is a reference notebook in a
> previous commit of the PR above. The point of comparison is
> https://github.com/discopy/discopy/pull/416. We want to solve the task using
> fewer parameters than the CMap GNN. The architecture is the same: one box per
> cell, per column, per row and per box. You can use the quimb/cotengra setup
> for GPU computation. Test different ansatze and unroll depths, progressively
> increasing the number of parameter. Report your results in the PR
> description.

- [x] Set up the Modal GPU environment: an image carrying this branch of
      optyx with quimb, cotengra and a differentiable GPU backend, and a
      smoke test contracting an `interaction.CMap` unrolling on the GPU.
- [x] Rebuild the leakage-free dataset of the removed reference notebook:
      288 completed grids, 256 distinct sampled solutions split 192/64
      before masking, eight-clue puzzles with a unique completion.
- [x] Build the sudoku `interaction.CMap` — one box per cell, per row, per
      column and per square — and its unrolled tensor networks at several
      tick counts; verify the batched GPU contraction against
      `optyx.core.contract.contract_tensor` on small cases.
- [x] Sanity-check expressivity before training: hand-coded constraint and
      cell channels (all-different indicators and digit memories) must
      decode held-out puzzles exactly, so the model class contains a solver.
- [x] Compare ansatze under a fixed pilot budget, progressively increasing
      the parameter count and the unroll depth: the Born/orthogonal-circuit
      family of the reference notebook against classical stochastic channels
      with factored non-negative cores, at parameter counts below the
      12,980 of the CMap GNN in discopy#416.
- [x] Scale the winning configuration on the GPU and report held-out
      per-cell accuracy and full-grid solve rate against the CMap GNN
      numbers (0.993 cell accuracy, 0.936 valid grids, 12,980 parameters).
- [x] Commit the experiment scripts and a results report, open the draft
      PR "sudoku experiment" stacked on #16 and report the results in its
      description.

> That's good but I want the experiment to be genuinely quantum. The
> classical stochastic channel should be learnable with a quantum model
> too.

> Do fill in an issue for the bug you found

- [x] File the `Channel.double()` bug on plain array kraus boxes:
      `Box.conjugate()` swaps dom and cod, and classical doubling
      mis-assembles when dom and cod arity differ
      ([#51](https://github.com/rel-int/optyx/issues/51)).
- [x] Make the leading ansatz genuinely quantum: every box parameterised
      by orthogonal-circuit angles -- cells as ten-qubit circuits with a
      coherent two-qubit memory, measured digit messages and predictions,
      constraints as nine-qubit measure-and-prepare channels with a
      measured verdict register -- so that the classical stochastic
      solver is contained in the quantum family as permutation circuits
      (Stinespring) and the classical family becomes its
      decohered-ablation baseline at matched parameter counts.
- [x] Certify quantum expressivity the same way as classical: the
      hand-written permutation circuits must decode held-out puzzles
      exactly through the doubled contraction with coherent memory.
