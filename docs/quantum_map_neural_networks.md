# Quantum Map Neural Networks

*A paper proposal. The measured results quoted below are reproduced by
two self-contained documentation notebooks,
[`beyond_1wl.ipynb`](notebooks/beyond_1wl.ipynb) and
[`beyond_3wl.ipynb`](notebooks/beyond_3wl.ipynb); everything else is
proposed work.*

## Abstract

We propose *quantum Map neural networks* (QMapNNs): recurrent networks
of typed quantum channels wired by a combinatorial map, the quantum
extension of the Map neural networks executed by the geometry of
interaction in `discopy.neural`. A cell is a channel
`X @ Y @ M -> X @ Y @ M @ O` — message ports read and written at every
tick, a private memory fed back with a one-tick delay, a prediction
written to the environment — and a `CMap` wires one cell per vertex of
a graph, with quantum *and* classical wires typed side by side in one
category. The photonic instantiation makes every cell a small
interferometer with a coherent memory, driven by one photon per tick,
with number-resolved detectors whose outcomes are fed forward along
classical edge wires that re-program the optics mid-run. Measured on
the smallest hard instances, the model separates non-isomorphic graphs
at three strictly increasing levels — beyond 1-WL passively, and
beyond the 3-WL blindness on strongly regular graphs once the
classical feed-forward layer is on — with exact-zero ablations
isolating each resource. We position the model against message-passing
GNNs, quantum walk invariants, Gaussian-boson-sampling kernels and
subset-register quantum GNNs, and lay out the experiments that would
establish it as a hierarchy-climbing, hardware-native graph model.

## 1. Background: three ceilings

**Classical message passing is bounded by 1-WL.** A GNN with anonymous
inputs and permutation-invariant aggregation cannot distinguish graphs
that colour refinement cannot (Xu et al., ICLR 2019; Morris et al.,
AAAI 2019). The bound covers the MapNNs of `discopy.neural`, whose
geometry-of-interaction execution is isotropic message passing — and
we observed it as an exact equality of outputs on 1-WL-equivalent
pairs.

**Quantum walk invariants are bounded by the cellular algebra.**
Walk-based spectra pass 1-WL on many strongly regular pairs (Emms,
Hancock, Severini & Wilson, quant-ph/0701033), but every invariant of
this family — including the interacting two-walker statistics of
Gamble et al. (PRA 81, 052313) in their non-interacting regime —
factors through the graph's cellular (coherent) algebra (Smith,
arXiv:1103.0262; ENDM 2011): a Weisfeiler-Leman-shaped ceiling. Our
measured passive gauge-blindness, and the exact blindness of a
single-marked-cell feed-forward, are instances of this ceiling
observed at machine precision.

**Subset-register quantum GNNs are bounded by their initialisation.**
The message-passing QGNN of Raj, Coyle, Monbroussou, Ferreira-Martins,
Farias & Kashefi (arXiv:2606.26873) prepares a node register at
particle number `j`, so its basis states are the `j`-subsets of
vertices, and proves that one unitary layer performs one round of
*set-based* `j`-WL, for `2 <= j <= 4`. Set-based `j`-WL is
expressivity-equivalent to standard `(j-1)`-WL on tuples, and the
proof stops at `j = 4` because its closed-walk initialisation — traces
of the induced adjacency's second, third and fourth powers — stops
fixing induced subgraphs up to isomorphism at five vertices. The
proven ceiling is therefore tuple 3-WL: one level below the strongly
regular pairs. The circuit is purely unitary, trained variationally in
a polynomially-sized subspace — which avoids barren plateaus at the
declared cost of classical simulability with polynomial overhead.

## 2. The model

A **QMapNN** is an `optyx.interaction.CMap` over a graph: one cell per
vertex, one pair of ports per edge, plus per-cell memory and
prediction. Three design choices carry the claims.

1. **Typed hybrid wires.** Ports, memory and predictions are objects
   of one symmetric monoidal category of channels, where `qmode`,
   `qubit`, `bit` and `mode` coexist: a port can be `qmode @ bit` — a
   coherent message *and* a classical message per edge. Classical
   feed-forward is not post-processing; it is a wire.
2. **Recurrence with memory.** The semantics is a feedback category
   (monoidal streams): the same cell runs at every tick, memory
   carries state, `unroll(T)` gives finite semantics and `fix` a
   stationary one. Parameters are shared across vertices and ticks, so
   the same weights apply verbatim at any graph size and any depth —
   the property that makes size-generalisation claims meaningful.
3. **Measurement in the loop.** Cells may measure and condition later
   optics on the outcome. This is the KLM resource — the one that
   makes linear optics universal — and, as measured below, the one
   that crosses the cellular-algebra ceiling.

**Photonic instantiation.** A cell of degree `d` is an interferometer
on `d + 2` modes — drive, messages, memory — port-symmetric so the
readout is a graph invariant, with a beam-splitter tap into a fresh
vacuum mode as prediction. The drive inputs one photon per tick at
every cell. The *active* cell adds: a threshold detector on the tap; a
stored classical bit, OR-ed with the incoming edge bits and the click;
a bit-controlled beam splitter re-programming the tap; and the click
broadcast on the outgoing edge bits. All of it is drawable — and
type-checked — as a string diagram with a feedback loop.

## 3. Measured separations (the existing notebooks)

| protocol | resource | measured |
|---|---|---|
| decohered walk, vanilla GNN, MapNN | classical message passing | exactly blind on every 1-WL-equivalent pair; separates the 1-WL-distinguishable control |
| passive coherent, driven | interference | separates 2C3/C6 (3.5e-2), 2C6/C12 (6.3e-3), decalin/bicyclopentyl (1.0e-2) at 2e4–7e5 shots; *exactly* blind on rook vs Shrikhande (gauge equivalence of the transfer matrices, checked at every photon order) |
| active, click re-programs its own cell | measurement, no broadcast | still exactly blind (7e-18): strong regularity absorbs one marked vertex — the cellular-algebra ceiling, observed |
| active, click broadcast one hop | measurement + classical message passing | separates rook vs Shrikhande at 1.7e-3 (twelve ticks, ~9e6 post-selected pairs at 5σ), exactly relabel-invariant; the kick-off ablation returns to exact zero |

Every number is exact-simulator output validated at machine precision
against `CMap.step.to_path()` and the doubled quimb contraction of
`CMap.unroll`.

## 4. Advantage over previous models

- **Against classical MPNNs and MapNNs**: strictly beyond 1-WL already
  passively, with the bound observed rather than assumed.
- **Against quantum walk invariants**: the passive model shares their
  cellular-algebra ceiling — and knows it, with exact-zero
  measurements — while the active model demonstrably steps past it on
  the smallest 3-WL-hard pair.
- **Against GBS graph kernels** (Brádler et al.; Schuld et al.): those
  are fixed-feature kernels of a single static interferometer; a
  QMapNN is recurrent, local, and hybrid, and its active statistics
  are not matchings-based invariants of one matrix.
- **Against the subset-register QGNN** (Raj et al.): no `O(N^j)`
  register, no initialisation oracle (the graph enters only through
  physical wiring; their essential ingredient is a classical
  preprocessing that already solves ≤4-vertex isomorphism), no proven
  ceiling at tuple 3-WL — our measured separation sits precisely
  beyond their guarantee — and no deliberate confinement to a
  classically-simulable subspace: the passive sector is efficiently
  computable (permanents), the active full drive is not (KLM), so the
  model has a genuine quantum-advantage regime. What their design
  offers that ours does not yet: a *proved* equivalence to a named WL
  level at every generic parameter, and trainability guarantees.
  Closing that gap is Experiments E1 and E5 below.
- **Against equivariant quantum graph circuits** (Mernyei et al.):
  EQGCs prove universality over bounded graphs but with qubit
  registers scaling with the graph and no locality; a QMapNN is
  message-passing-local, size-transferable, and photonic.

## 5. Proposed experiments

**E1 — The broadcast-radius ladder (the central claim).** Conjecture:
the WL level of the active photonic QMapNN climbs with the classical
broadcast radius `r` — marking `N_r(c)` writes `r`-hop induced
structure into the optics. Measure separations on the Cai–Fürer–
Immerman families CFI(K_3), CFI(K_4), CFI(K_5) and the BREC benchmark
(the same instruments as Raj et al., enabling a direct table) as a
function of `r` and of the number of post-selected photons. The
two-photon branch simulator scales to every BREC pair (≤ 25 nodes,
minutes each); deliverable: a `distinguished/total` table beside their
k-WL rows, with the `r`-versus-level scaling law, whatever it is.

**E2 — Where quantum walks failed.** The generalized-quadrangle
line-graph pairs that refuted the Emms conjecture (arXiv:1511.01962)
and larger same-parameter SRG families (25, 26, 29 vertices). A
separation there would put the active model past every published
walk-based invariant; a failure would locate its own ceiling — either
result is a finding.

**E3 — Trained cells.** Make the couplers, taps and kicks trainable
(differentiable contraction, optyx#21) and evaluate on the standard
expressivity benchmarks (CSL, EXP, BREC) and on QM9 HOMO–LUMO — the
task where Raj et al. report 0.398 → 0.235 eV as `j` rises — with the
matched `bit`-wire classical CMap at equal parameter count as the
ablation that isolates what the quantum wires buy.

**E4 — More photons, deeper feed-forward.** Three-and-more-photon
post-selected sectors with compounding kicks, exact by the same
branching simulator; measure how the margin grows with photon number
and record depth, toward the fully driven regime that is no longer
classically simulable.

**E5 — Theory obligations.** (i) Prove the passive ceiling: driven
linear-optical statistics of a graph CMap factor through the cellular
algebra — the Smith argument extended to this encoding. (ii) Prove
the single-mark blindness for same-parameter SRGs. (iii) Bound the
one-hop active protocol relative to individualisation-refinement, and
look for the analogue of the Raj et al. sandwich — a named level the
`r`-hop protocol provably reaches at generic optics.

**E6 — Hardware feasibility.** The measured margins give shot budgets
(2e4 for the easiest pair, 9e6 post-selected pairs for the SRG one) —
loop-based time-multiplexed photonics runs one physical cell reused
across time bins, which is *exactly* the stateful-channel reading of a
CMap box; estimate loss and detector-efficiency tolerances at those
budgets, and the feed-forward latency the broadcast needs.

## 6. Potential

The pattern the measurements suggest is that expressivity lives in the
*composite*: interference supplies invariants classical aggregation
cannot, classical broadcast supplies the individualisation that
symmetric quantum dynamics cannot, and the typed channel category is
what lets the two be wired into one machine and reasoned about as one
diagram. If E1's scaling law holds, QMapNNs climb the WL hierarchy by
turning a *physical* knob — broadcast radius and photon number —
rather than by enlarging a register or an initialisation oracle, on
hardware whose native operations are exactly the model's primitives.

## References

- K. Xu, W. Hu, J. Leskovec, S. Jegelka, *How Powerful are Graph
  Neural Networks?*, ICLR 2019, arXiv:1810.00826.
- C. Morris et al., *Weisfeiler and Leman Go Neural*, AAAI 2019,
  arXiv:1810.02244.
- J. Cai, M. Fürer, N. Immerman, *An optimal lower bound on the number
  of variables for graph identification*, Combinatorica 12 (1992).
- D. Emms, E. Hancock, S. Severini, R. Wilson, *A matrix representation
  of graphs and its spectrum as a graph invariant*, Electron. J.
  Combin. 13 (2006), quant-ph/0701033.
- J. K. Gamble, M. Friesen, D. Zhou, R. Joynt, S. N. Coppersmith,
  *Two-particle quantum walks applied to the graph isomorphism
  problem*, PRA 81, 052313 (2010).
- J. Smith, *On the limitations of graph invariants inspired by quantum
  walks*, Electron. Notes Discrete Math. 38 (2011); *Cellular algebras
  and graph invariants based on quantum walks*, arXiv:1103.0262.
- K. Brádler, S. Friedland, J. Izaac, N. Killoran, D. Su, *Graph
  isomorphism and Gaussian boson sampling*, Special Matrices 9 (2021).
- M. Schuld, K. Brádler, R. Israel, D. Su, B. Gupt, *Measuring the
  similarity of graphs with a Gaussian boson sampler*, PRA 101,
  032314 (2020).
- E. Knill, R. Laflamme, G. J. Milburn, *A scheme for efficient quantum
  computation with linear optics*, Nature 409 (2001).
- P. Mernyei, K. Meichanetzidis, İ. İ. Ceylan, *Equivariant Quantum
  Graph Circuits*, ICML 2022, arXiv:2112.05261.
- S. Raj, B. Coyle, L. Monbroussou, A. J. Ferreira-Martins,
  R. M. S. Farias, E. Kashefi, *Scalable Message-Passing Quantum Graph
  Neural Networks in the Weisfeiler-Leman Hierarchy*,
  arXiv:2606.26873.
- Y. Wang, M. Zhang, *Towards Better Evaluation of GNN Expressiveness
  with BREC Dataset*, arXiv:2304.07702.
