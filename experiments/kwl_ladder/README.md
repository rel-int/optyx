# Climbing the k-WL ladder with QMapNNs

How far up the Weisfeiler-Leman hierarchy do quantum Map neural
networks reach? This experiment builds a dataset of graph pairs of
increasing k-WL indistinguishability — Cai-Fürer-Immerman pairs and the
BREC benchmark (arXiv:2304.07702) — and benchmarks five models on it.
Every model is a functor turning the graph, given as a combinatorial
map, into an `optyx.interaction.CMap` whose cells are physical CPTP
maps: the same wiring, five interpretations.

## The dataset

Rungs are *measured*, not taken from category labels, with our own
colour refinement (1-FWL), 2-FWL and 3-FWL implementations
(`dataset.py`; recall k-FWL == tuple (k+1)-WL):

- **control** — two pairs 1-WL already distinguishes;
- **1fwl-blind** — five pairs colour refinement (tuple 1/2-WL) cannot
  split but 2-FWL (tuple 3-WL) can (BREC basic/regular/extension, and 2C3 vs C6);
- **2fwl-blind** — five pairs 2-FWL (tuple 3-WL) cannot split but
  3-FWL (tuple 4-WL) can: rook
  4×4 vs Shrikhande, two SRG(26,10,3,4) pairs, two distance-regular
  pairs;
- **3fwl-blind** — five CFI pairs (n = 80 to 106) that our 3-FWL
  (tuple 4-WL) certifies as indistinguishable; they are non-isomorphic by the
  Cai-Fürer-Immerman construction, which also guarantees some higher
  k-FWL distinguishes them.

Two measurements about BREC fell out of the classification: every
*4-vertex-condition* pair (n = 63) and every *distance-regular* pair is
distinguished by 3-FWL — so under tuple 4-WL the benchmark's hardest
non-CFI categories all sit on the same rung as the strongly regular
pairs — and no CFI pair below 70 vertices survives 3-FWL either: the
genuinely 3-FWL-blind pairs start at n = 80.

## The five models

All cells are port-symmetric with fixed shared parameters, so every
model is a graph invariant by construction (`models.py`).

1. **passive** — a port-symmetric interferometer on one drive mode, the
   message modes and a coherent memory; one photon into every cell at
   every tick; a beam-splitter tap, measured, as prediction.
2. **bell** — the same optics on two drive rails carrying half of a
   dual-rail Bell pair per tick; the other half is rotated locally and
   measured with the reflections, so the record correlates the network
   with an entangled reference.
3. **active** — the cell of `docs/notebooks/beyond_3wl`: messages and
   memory carry `qmode @ bit`, a threshold detector watches the tap,
   the click is broadcast one hop along the edge bits and the latched
   OR re-programs the tap through a bit-controlled beam splitter.
4. **qubit** — a qubit cell: fresh `|+>` ancilla, a commuting CZ star
   over ancilla, memory and ports, fixed rotations, ancilla measured in
   the X basis as the prediction; quantum messages, unitary otherwise.
5. **qubit-ff** — the qubit cell with classical messages beside the
   quantum ones: the click is broadcast one hop, OR-latched, and
   drives a phase on the memory qubit.

## The engines

The photonic models are evaluated *exactly* on every rung through the
few-photon certificates of `exact.py`: the record statistics of the
attenuated, two-herald post-selected homogeneous drive are the uniform
average over injection pairs of the exact two-photon conditioned
dynamics. The step matrix is assembled cell by cell and checked against
`CMap.step.to_path()`; the passive and Bell ensembles are read off
permanents of one spacetime transfer matrix, the active ensemble
branches over classical records; the two engines agree bin for bin, and
the Bell certificate is validated entry by entry against optyx's own
contraction of the driven functor image (`test/test_kwl_ladder.py`
carries the fast half of these checks). Beyond 40 vertices the
ensemble averages over the edges instead of all pairs — a smaller but
equally invariant ensemble, recorded in the results.

The qubit models are contracted as doubled tensor networks through
quimb (`contract.py`); the readout is the multiset over cells of
per-cell prediction-bit trajectory distributions, compared through
sorted values. Exact contraction hits the expected wall — the memory
cut is two wires per edge — which confines them to the small rungs;
that wall is the classical-simulability boundary and is reported, not
hidden.

## Results

Separations at T = 8 (photonic) and T = 4 (qubit), largest bin-wise
difference between the two ensembles; `0.0e+00`-scale entries (1e-16
and below) are exact zeros of the model, not small numbers:

| pair | rung | passive | bell | active | active-flood | qubit | qubit-ff |
|---|---|---|---|---|---|---|---|
| control-c6-vs-p6 | control | 2.8e-02 | 2.0e-02 | 2.8e-02 |  | 3.2e-15 | — |
| control-k15-vs-p6 | control | 1.0e-02 | 6.2e-03 | 1.0e-02 |  |  |  |
| regular-24 | 1fwl-blind | 3.0e-03 | 1.5e-03 | 3.0e-03 |  |  |  |
| regular-47 | 1fwl-blind | 3.3e-03 | 1.3e-03 | 2.7e-03 |  |  |  |
| extension-09 | 1fwl-blind | 3.3e-03 | 1.8e-03 | 3.3e-03 |  |  |  |
| extension-43 | 1fwl-blind | 4.1e-03 | 2.4e-03 | 4.1e-03 |  |  |  |
| classic-2c3-vs-c6 | 1fwl-blind | 9.3e-03 | 4.5e-03 | 9.3e-03 |  |  |  |
| str-00 | 2fwl-blind | 6.9e-18 | 3.5e-18 | 1.2e-04 | 1.5e-05 |  |  |
| str-08 | 2fwl-blind | 1.7e-18 | 1.7e-18 | 3.3e-06 |  |  |  |
| str-09 | 2fwl-blind | 5.2e-18 | 3.5e-18 | 1.5e-05 |  |  |  |
| cfi-73 | 2fwl-blind | 8.7e-19 | 8.7e-19 | 8.7e-19 | 1.7e-18 |  |  |
| dr-18 | 2fwl-blind | 3.5e-18 | 3.5e-18 | 8.7e-19 | 1.4e-17 |  |  |
| cfi-83 | 3fwl-blind | 1.4e-17 | 8.7e-19 |  |  |  |  |
| cfi-88 | 3fwl-blind | 3.5e-18 | 3.5e-18 |  |  |  |  |
| cfi-95 | 3fwl-blind | 1.7e-18 | 8.7e-19 |  |  |  |  |
| cfi-98 | 3fwl-blind | 8.7e-19 |  |  |  |  |  |
| cfi-97 | 3fwl-blind | 8.7e-19 |  |  |  |  |  |

A blank cell was *not run*, by the pruning rule: a model already
exactly zero on every pair of a lower rung does not climb, so it is not
run higher — passive and Bell stop after the 2-FWL rung, the active
family stops after measuring zero on the locally-isomorphic pairs of
that same rung, and the qubit models stop at the controls. A dash is a
cell recorded as computationally trimmed. The invariance table repeats
the cells for a graph against a relabelling of itself: all exact zeros.

| pair | rung | passive | bell | active | active-flood | qubit | qubit-ff |
|---|---|---|---|---|---|---|---|
| regular-24 | 1fwl-blind | 6.9e-18 | 3.5e-18 | 6.9e-18 | 6.9e-18 |  |  |
| str-00 | 2fwl-blind |  |  | 1.4e-17 |  |  |  |

## Reading the table

1. **Two-photon interference climbs past 1-WL.** Every model with a
   photonic two-photon record separates every 1-FWL-blind pair at
   margins near 1e-3, an order below the controls.
2. **Entangled drives buy nothing.** The Bell model tracks the passive
   one at roughly half the margin on the rungs both separate, and is
   exactly zero from the 2-FWL rung on: detecting the local reference
   collapses the network photon onto fixed drive-rail superpositions,
   which is passive linear optics again — still inside the gauge and
   cellular-algebra ceiling of `beyond_3wl`.
3. **Classical feed-forward separates exactly the strongly regular
   pairs.** The one-hop active model splits all three SRG pairs of the
   2-FWL-blind rung (1.2e-4, 3.3e-6, 1.5e-5 at eight ticks) — and is
   *exactly* zero on the CFI pair and the distance-regular pair of the
   same rung. The relayed flood, whose kicked region grows one hop per
   tick without bound, changes nothing: it separates the rook pair
   (1.5e-5) and stays exactly zero on both locally-isomorphic pairs.
   The measured ceiling of measurement-plus-feed-forward photonics at
   the two-photon level is therefore not a k-WL rung at all: it is the
   class of pairs whose *grown click neighbourhoods* differ, which the
   Cai-Fürer-Immerman construction — locally isomorphic by design —
   defeats at any broadcast radius. Climbing past it needs a different
   resource: more photons in the post-selected sector, or non-local
   classical control.
4. **The commuting qubit cell is blind, and its controlled variant is
   the simulation wall.** The unitary qubit model measures 3.2e-15 on
   a pair that colour refinement itself separates: with the CZ star
   applied before any non-diagonal rotation, the measured ancilla's
   per-cell trajectory distribution turns out identical for every cell
   of every graph tried — cells of degree one and two give bitwise the
   same distribution. The qubit-ff model's doubled network, laden with
   classical copy spiders, defeated every exact contraction path on a
   six-vertex graph and the compressed optimizer of cotengra 0.8.2
   errors on every trial, so its cells are recorded as trimmed — the
   contraction wall is the classical-simulability boundary showing up
   on schedule.
5. **Everything is validated.** Probability mass is 1 to 1e-14 in
   every cell; the invariance rows are exact zeros; the assembled step
   matrices equal `CMap.step.to_path()`; the active engine matches
   optyx's contraction of the driven functor image entry by entry —
   which is how two discrepancies between the `beyond_3wl` notebook's
   simulator and its own drawn cell were found and fixed here (kicks
   on reflections, and a replaced rather than composed kick splitter;
   reported on #61). The corrected rook/Shrikhande margin is 1.2e-4
   against the notebook's 7.8e-4 — the separation survives, smaller.

## Reproducing

```shell
python experiments/kwl_ladder/dataset.py     # rebuild data/dataset.json
python experiments/kwl_ladder/run.py         # photonic models
python experiments/kwl_ladder/run.py --models qubit,qubit-ff
python experiments/kwl_ladder/report.py      # render the tables
```

The BREC data must be cloned to `graphpku/brec` (see `dataset.py`);
everything else is committed. `modal_app.py` shards the active-model
ensembles over Modal workers — the fan-out is validated against the
local engine on a triangle — but the sandbox this experiment ran in
blocks gRPC, so the committed results were all computed locally.
