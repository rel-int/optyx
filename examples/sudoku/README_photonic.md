# Solving sudoku with photonic channels

The sequel to the qubit experiment of this branch's base: the same
sudoku `interaction.CMap` -- one box per cell, per row, per column and
per square, the same 192/64 leakage-free dataset, two recurrent ticks,
exact contraction -- with every channel realised by **linear optics**.
Each cell is a small recurrent optical circuit: one or two fresh
photons injected at every tick, a mesh of Givens rotations (the
Clements interferometer layout), photon counting, and a recirculating
memory. Three models benchmark what carries the recurrence:

| loop | model | init | params | cell accuracy | grids solved |
|---|---|---|---|---|---|
| classical only | measured messages and memory, feedforward meshes | structured | **648** | **1.000** | **1.000** |
| classical only | same, learned from scratch | random | 984 | 0.793 | 0.062 |
| both | coherent memory, measured messages with feedforward | random | 872 | 0.637 | 0 |
| quantum only | coherent messages and memory, counters + learned post-processing | random | 109 | 0.541 | 0 |

The ordering is monotone in how much measurement sits in the loop:
the fully measured model solves, the mixed model learns to 0.64, the
fully coherent model to 0.54 -- each well above the 0.25 chance line,
none but the measured one solving grids.

For scale: the CMap GNN of discopy#416 uses 12,980 parameters for
0.993 / 0.936, and the qubit families of the base PR solve at 4,492
(quantum) and 1,592 (classical).

## Only a classical loop: a certified linear-optical solver

Cells route two injected photons through an eight-mode mesh whose
angles are feedforward-controlled by the measured incoming digits and
the memory digit; all modes are counted and the lowest occupied mode is
the digit the cell broadcasts. Constraints walk a single photon through
a mesh with one feedforward layer per claim; the verdict is whether it
reaches the accept mode.

Because the mesh layout is a sorting network, firing comparators at a
right angle realises any controlled mode permutation -- so the exact
solver is a closed-form angle setting (`photonic.solver_photonic`): the
cell routes its photons to the modes indexed by its memory digit, and
the constraint's photon walks the **subset automaton of the
all-different constraint**, which fits on exactly eight modes (live
states plus parked rejects). The certificate decodes 16/16 held-out
puzzles before any training. Trained from the broadcast plumbing with
random constraint logic, the model reaches 1.000 cell accuracy and
64/64 grids within 100 steps at 648 parameters -- one photon walking an
interferometer per constraint is enough to learn all-different.

## Only a quantum loop: light is what recurs

No feedforward anywhere: messages and memory are coherent optical
modes (doubled wires in the contraction), cells hold a two-mode
coherent memory and receive two fresh photons per tick, constraints
passively interfere their four port modes with one injected photon on
a fifth, traced-out mode. The only measurement is photon counting on
the two prediction modes, followed by a trainable local classical
post-processing (16 of the 109 parameters). Amplitudes are
collision-free boson amplitudes computed with Ryser permanents, exact
in the dual-rail-truncated sector.

The full map does not contract exactly -- width 2^40 at two ticks even
decomposed to beamsplitter level, measured before spending any GPU --
so this family trains and evaluates on the local light-cone maps
(width 2^18 at two ticks, 2^24 at three). Without the traced ancilla
and the learned post-processing the model sits at chance (two runs,
27-54 parameters); with them it learns to **0.541** cell accuracy at
two ticks and 109 parameters, 0.359 at three ticks -- twice chance
through nothing but two-photon interference, but far from solving.
Decoherence in the loop is what separates the families: the measured
model solves, the coherent model learns weakly, exactly the pattern of
the born-versus-measured qubit families in the base PR.

## Scaling the quantum loop: parameters are not the bottleneck

Can the coherent model be scaled past 0.541 without blowing up the
contraction? The ``photonic-scaled`` family (`photonic_scaled.py`)
grows parameters only in directions that leave the network width
untouched: internal vacuum ancillas inside each box (cells 7 to 11 or
12 modes, constraints 5 to 8 -- photons can scatter into them and be
traced out, a structured non-unitary channel), complex phased-Givens
meshes (two parameters per comparator), more mesh sweeps, and fixed
pure-loss channels of transmittivity 0.95 folded into every coherent
write leg. With no ancillas, real angles and no loss the family
reduces exactly to the pure one (verified to 1e-16 by
`check_scaled.py`). Four runs at two ticks on the local light cones:

| config | params | cell accuracy (final / best) |
|---|---|---|
| pure baseline, 7/5 modes | 109 | 0.506 / 0.541 |
| 11/8 modes, 4 complex sweeps, lossless | 680 | 0.463 / 0.479 |
| same, fixed loss 0.95 | 680 | 0.459 / 0.516 |
| 12/8 modes, 8 complex sweeps, lossless | 1,520 | 0.482 / 0.502 |
| same, 8,000 steps at half the rate | 1,520 | 0.396 / 0.406 |

The plateau does not move: an order of magnitude more parameters,
non-unitary ancilla channels and mild decoherence on the links all
land within noise of the 109-parameter model, and training longer
lands lower. What limits the purely coherent model is not its
parameter count but its structure -- a passive linear-optical loop
with counting only at the predictions has no nonlinearity inside the
recurrence, exactly what the measured families get from their
counters. Fock cutoff two on the message wires was the one width-side
scaling planned; its doubled cell tensor (nine-valued message legs)
is a 4.4 GB dense array before contraction, out of reach of the
A100, so the bigger-mesh run above took its budget slot.

## Contraction budget

All dimensions were fixed by the base PR's width measurements before
any GPU run: dimension-four measured wires and two ticks for the
classical and mixed loops (the established 2^16-2^24 regime), local
light cones for the coherent loop. The scaled family grows boxes, not
wires, so it contracts at exactly the pure family's width. GPU spend:
two runs per model plus a four-run scaling ladder, on Modal A100s,
everything else certified or smoke-tested on CPU first.
