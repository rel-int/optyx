# Learning to solve 4x4 sudoku with `optyx.interaction`

The experiment of the removed reference notebook of
[#16](https://github.com/rel-int/optyx/pull/16), scaled up on Modal
GPUs. The point of comparison is the CMap GNN of
[discopy#416](https://github.com/discopy/discopy/pull/416): 12,980
parameters, 0.993 held-out cell accuracy, 0.936 valid grids. The goal
is to reach a good accuracy with **fewer parameters**, keeping the same
architecture: one box per cell, per row, per column and per square.

## Result

Two models solve the task **perfectly** -- 1.000 held-out cell accuracy,
64/64 grids solved -- with a fraction of the CMap GNN's parameters:

| model | parameters | cell accuracy | grids solved |
|---|---|---|---|
| CMap GNN (discopy#416) | 12,980 | 0.993 | 0.936 |
| **quantum channels** (this PR) | **4,492** | **1.000** | **1.000** |
| quantum channels, random init | 6,347 | 0.951 | 0.844 |
| classical stochastic ablation | 1,592 | 1.000 | 1.000 |

The quantum model is genuinely quantum: each cell is a ten-qubit
orthogonal circuit whose two-qubit private memory stays *coherent*
between ticks while its digit messages and prediction are measured;
each constraint is a nine-qubit measure-and-prepare channel whose
measured verdict register drives what it writes back. The classical
stochastic family is its decohered ablation, and by Stinespring it is
contained in the quantum family.

## Architecture

The `interaction.CMap` of #16: 16 cell boxes and 12 constraint boxes,
one bidirectional dimension-four digit wire pairing each cell with each
of its three constraints (48 edges), every port read and written at
every tick. Cells carry a private memory and write a prediction to the
environment at every tick; clue predictions are postselected on their
digit, free predictions are marginalised, and the readout leaves the
target cell's final prediction leg open -- its four digit scores in one
contraction. The loss is the cross-entropy of the normalised scores of
every hidden cell.

`verify.py` ties the trainer to optyx: the structure is the
combinatorial data of a genuine `interaction.CMap`, the born-family
scores match a brute-force simulation of the protocol that
`CMap.step` fixes, and classical bit-wire channels double to the
elementwise square of their kraus array, so the stochastic tensors are
optyx channels with kraus their elementwise square root.

## Exact contraction, and its limits

The unrolled protocol is contracted **exactly** -- no compressed bond --
with Cotengra paths on JAX `float64` arrays, batched over puzzles on the
GPU. Two design choices make this possible where the reference notebook
needed `chi=4` compression and 117 s per gradient:

1. **One digit wire per incidence** instead of two qubit wires per
   direction: the double-wire map contracts at width 2^52 at two ticks;
   the single-wire map at 2^16.
2. **Port-factored constraint writes**: what a constraint writes back is
   a product over its ports given one of `feedback` many read verdicts.
   The write side of a constraint would otherwise correlate its four
   neighbours and blow the treewidth; with the factorisation the
   feedback travels through a rank-`feedback` hyperedge. At `feedback=1`
   constraints cannot respond to what they read at all -- and, matching
   that, `feedback=1` never learns while `feedback=2` suffices.

Contraction widths of one readout expression (hyper-greedy paths,
`feedback` 2, classical dims; the born column has single-qubit wires):

| ticks | classical/quantum | born |
|---|---|---|
| 2 | 2^16 - 2^20 | 2^16 |
| 3 | 2^50 (kahypar 2^50) | 2^40 |
| 4 | 2^71 | -- |

Full-map exact contraction is therefore feasible at two ticks and
infeasible from three, even with kahypar: the depth axis beyond two
ticks needs the local light-cone maps of `local_structure` (the target,
its seven peers and its three constraints, outside ports unpaired), the
same approximation the reference notebook used. Two ticks turn out to
be sufficient.

## Solver certificates

Before any training, hand-written parameters certify that each family
contains an exact solver -- the expressivity gap behind the null result
of the reference notebook (26.6% against a 25% baseline at 12,849
parameters):

* `solver_cores` (classical): cell cores of bond four carry the memory
  digit and broadcast it; constraint read-chains of bond six track the
  claimed digits as a subset and postselect all-different; writes are
  uniform. Decodes 16/16 held-out puzzles at two ticks.
* `solver_quantum` (quantum): the same solver as permutation circuits
  -- the cell XORs its memory onto its messages and prediction, the
  constraint computes the all-different flag reversibly into a fresh
  ancilla read out as the verdict. Decodes 16/16.
* `solver_angles` (quantum, **inside the trainable ansatz**): the cell
  permutation is an XOR ladder of conditional-rotation layers (angle
  pi/2 where the memory control bit is one), and the constraint flag is
  a *single* conditional layer -- one independent angle per claim
  configuration is exactly a generalised multi-controlled flip. The
  solver lives at cell depth 8-10 and constraint depth 1 of the very
  parameterisation being trained. Decodes 16/16.

## Training

Dataset as in the reference notebook: all 288 completed grids, 256
distinct solutions sampled, split 192/64 *before* masking, eight-clue
puzzles with a unique completion, cells 0 and 1 always hidden. Adam on
the cross-entropy of exact per-cell marginals, batch 32.

**From random initialisation** (600 steps): the quantum family learns
best -- 45.3% at depths (8,8) and 6,347 parameters -- the classical
`square` family reaches 37.9%, and the fully coherent `born` family
stays at chance, matching the reference notebook's null result:
decoherence on the message wires is what makes the landscape tractable.
One-tick ablations sit at chance for every family (recurrence is
necessary), as does `feedback=1` (constraints must respond to their
reads).

**With longer training from random initialisation** (4,000 steps,
lr 0.01, depths (8,8), 6,347 parameters): the quantum family reaches
**0.951 cell accuracy and solves 84.4% of held-out grids** with no
initialisation prior at all -- already close to the CMap GNN's 0.993 /
0.936 at half its parameter count.

**With structured initialisation** ("cells": the cell starts at its
digit-broadcast plumbing -- a specific angle setting of the same
ansatz -- while the constraint logic starts random and is learned from
data): both families solve the task within a few hundred steps.

| family | config | init | seed | params | cell acc | solved |
|---|---|---|---|---|---|---|
| quantum | depths (8,1), lr 0.01 | cells | 7 | 4,492 | 1.000 | 1.000 |
| quantum | depths (10,1), lr 0.01 | cells | 7,8,9 | 5,536 | 1.000 | 1.000 |
| quantum | depths (10,4), lr 0.01 and 0.003 | cells | 7 | 6,331 | 1.000 | 1.000 |
| quantum | depths (10,4), lr 0.003 | solver+noise | 7 | 6,331 | 1.000 | 1.000 |
| square | bond 5, lr 0.01 | cells | 7 | 1,592 | 1.000 | 1.000 |
| square | bond 6, lr 0.01 | cells | 7,9 (of 7,8,9) | 2,240 | 1.000 | 1.000 |
| square | bond 4, lr 0.01 | cells | 7 | 1,056 | 0.678 | 0.031 |
| exp | bond 6, lr 0.01 | cells | 7 | 2,240 | 0.494 | 0 |

The quantum family solved in every structured run (four configurations,
five seeds); the classical family is seed-sensitive at 800 steps (seed 8
of bond 6 plateaued at 0.744) and needs bond at least five. The `exp`
(log-space) parameterisation of the same classical tensors plateaus
below 50%: the square link is what trains.

## Files

* `experiment.py` -- task, dataset, `interaction` structure, ansatze,
  solver certificates.
* `contraction.py` -- exact batched contraction of the unrolled
  protocol, box tensors per family, training loop.
* `verify.py` -- the optyx semantics checks.
* `test_solver.py` -- the solver certificates
  (`python test_solver.py <ticks> <square|quantum|angles>`).
* `modal_sweep.py` -- the Modal app
  (`modal run modal_sweep.py --configs configs.json --out out.json`).
* `results_pilot.json`, `results_pilot2.json`, `results_final.json` --
  raw metric histories of the reported runs.

Found and reported along the way:
[#51](https://github.com/rel-int/optyx/issues/51), `Channel.double()`
broken for plain array kraus boxes.
