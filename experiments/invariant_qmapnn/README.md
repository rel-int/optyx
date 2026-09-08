# The permutation-invariant linear-optical QMapNN

Does a QMapNN whose cells commute with every permutation of their
incident edges still climb past 1-WL? This experiment imposes full
`S_d`-invariance at every node of a passive photonic QMapNN, so that the
network is an exact *graph* invariant by construction — independent of
the rotation system, i.e. of the order in which each vertex lists its
edges — and measures whether the 1-FWL-blind pairs of the ladder
dataset of `experiments/kwl_ladder` (PR #69) still separate. Everything
is exact, local and small: no training, one cell family, seven pairs
and one sanity pair, `T <= 4`.

## The model

A graph `G = (V, E)` becomes one `optyx.interaction.CMap` with one cell
per node (`cells.py`). The wiring is the handout's:

- **one dart pair per edge**: an edge of the `CMap` pairs a port of each
  endpoint, and every paired port is both read and written at every
  tick, the write landing in the memory its partner reads at the next
  tick. So a `CMap` edge *is* the dart pair — `u`'s out-dart is `v`'s
  in-dart one tick later and vice versa — and the handout's deviation
  from one mode per edge is native to `optyx.interaction`, not a
  redesign: nothing in the code is oriented or scheduled;
- **one memory mode per node**, the box's internal memory, a feedback
  wire from the cell to itself;
- **one drive mode per node**, the box's unpaired port, read from the
  environment (a fresh photon at every tick) and written back to it as
  the **output mode**, discarded at every tick except the last.

A cell of degree `d` is a passive unitary on `d + 2` modes — inputs the
drive, the `d` in-darts and the memory; outputs the output, the `d`
out-darts and the memory — so the interpretation is a CPTP map, unitary
plus discards, as PR #69 required.

### The invariant cell

`S_d` acts on the darts, simultaneously on the in- and out-darts, and
trivially on memory and drive. By Schur the permutation representation
on `C^d` is the trivial representation plus the standard one, each with
multiplicity one, so the commutant inside `U(d + 2)` is exactly

    U(W, psi) = B^dag . [ W (+) e^{i psi} I_{d-1} ] . B

where `B` is a fixed balanced multiport on the darts sending the
symmetric vector `(1, ..., 1) / sqrt(d)` to the first dart — the real
Householder reflection, the identity at `d = 1` — extended by the
identity on memory and drive; `W` in `U(3)` acts on the symmetric dart,
the memory and the drive, its third column routed to the output; and
`e^{i psi}` is one phase on the `d - 1` non-symmetric darts. Ten real
parameters per node, independent of the degree, shared by every node
and every tick: `invariant_cell(d, W, psi)`. The Schur claim is
executable — `test_cell_is_unitary_and_commutes_with_dart_permutations`
conjugates the cell by fifty random dart permutations at `d = 1, 2, 3,
5` and finds it unchanged to `1e-15` (**V1**).

Two regimes and two contrasts run:

- **canonical** (R1): `psi = pi`, `W` one Haar-random element of
  `U(3)` drawn once from seed 0;
- **grover** (R1, reference): `psi = pi`, `W` diagonal, so the darts
  carry the Grover coin `2J/d - I` and the drive never enters the graph:
  the readout is graph-independent and every separation must be an
  exact zero;
- **seed-01 … seed-20** (R2): `W` Haar-random and `psi` uniform on
  `[0, 2 pi)`, one seed each, reported by max and median;
- **ladder**: the port-symmetric interferometer of PR #69's passive
  cell without its tap, in the same layout — see the tension below;
- **generic**: one Haar-random unitary of `U(d + 2)` per degree, shared
  by the nodes of that degree, which does read the rotation system.

### Two tensions with the handout

1. **The ladder's cells were already port-symmetric.** The handout's
   premise is that "QMapNN cells so far read the rotation system". They
   do not: the `coupler` of `beyond_1wl` and of PR #69's `models.py` has
   every dart coupled identically to the drive, the memory and every
   other dart, and its docstring says so — "port-symmetric so the
   readout is a graph invariant". By the Schur classification above it
   is therefore *in* the invariant family, with `(W, psi)` depending on
   the degree through the `1/sqrt(d)` and `1/d` scalings. So the
   handout's contrast row, "the sigma-sensitive passive cell from the
   ladder", cannot fail V2, and the table below confirms it does not.
   The V2 contrast is measured instead on the `generic` cell, a
   Haar-random unitary per degree, which is the natural passive cell
   *outside* the commutant; the ladder cell is kept as a second
   invariant row with degree-dependent parameters.
2. **The dart pair is the `CMap`'s own semantics.** See the wiring
   above: no single-writer orientation had to be chosen, and the cost is
   `2|E|` dart wires as the handout expected.
3. **Isolated vertices.** `extension-43` has two isolated vertices on
   each side, and a cell of degree zero has no dart for the Householder
   reflection to act on; the handout's degree-independent parameter
   count presumes `d >= 1`. An isolated vertex gets the degree-one cell
   with its dart left *open* — read from the environment, i.e. the
   vacuum, and written back to it, i.e. discarded — the `d -> 0` limit
   of the ansatz, still unitary plus discards. The engine injects and
   reads the drives only, never a dangling dart.

## The engine

`engine.py` defaults to PR #69's methodology, not Fock-space
truncation. The homogeneous drive — a fresh photon in every drive at
every tick — is evaluated through the two-herald post-selected
certificate: attenuating the sources and post-selecting on exactly two
heralds leaves the uniform average, over unordered pairs of distinct
injection slots `(node, tick)`, of the exact two-photon dynamics. Since
only the last tick is read here, the slots of *every* tick are in the
ensemble, where PR #69 — which read every tick — took the two heralds
in the same pulse; the same-pulse pairs are a subset of this ensemble.

- **The step matrix is assembled from the functor image.**
  `assemble(cmap)` glues the path matrix of every box's channel by the
  read and write routing of the `CMap` — the port a box writes is sent
  to the slot its partner reads — and equals `cmap.step.to_path()`
  exactly (`test_assembled_step_matches_the_functor_image`, on a path
  and on a shuffled 4-cycle).
- **Spacetime transfer.** `transfer` unrolls the step into one matrix
  from every wire read at the first tick to the outputs of every tick
  and the wires kept at the end; `amplitudes` gives every injection
  slot its single-photon amplitude over that space. Distinct slots give
  orthonormal rows, so two heralded photons are two bosons in
  orthogonal states and their counts are permanents of two-by-two
  submatrices: `|a_i b_j + a_j b_i|^2` for one photon in each of two
  columns, half of that for two in one.
- **Grounded in optyx's own contraction.** On the triangle at `T = 2`
  and on a shuffled path at `T = 2` and `T = 3`, the driven `CMap` is
  contracted directly — `Create` on every drive slot, `Discard` on the
  earlier outputs, `Measure` on the last — and matched entry by entry
  against the certificate for three slot pairs each (same tick,
  different ticks, same drive twice): worst discrepancy `3.3e-16`, mass
  `1 - 1e-15`. The triangle case is in CI
  (`test_two_photon_certificate_matches_the_contraction`).
- **Exact zeros.** Entries at or below `1e-15` are reported as exact
  zeros, distinct from small numbers.

Two single-particle certificates share the same readout, for H4, and a
fourth, `distinguishable`, replaces the bosonic pair statistics by those
of two photons that cannot interfere (see the section on distinguishable
photons below):
**one-photon**, the one-herald ensemble, uniform over slots; and
**coherent**, a coherent state of the same amplitude in every slot with
two photons injected on average, whose outputs are coherent states of
the *coherent sum* of the slot amplitudes and whose counts are Poisson.
Both are functions of the single-particle transfer matrix alone. On the
coherent rows `1 - mass` is the Poisson weight beyond two photons, not
an error.

## The readout

At the last tick the `N` output modes are counted and everything else
is discarded. The graph-level statistic is the multiset over nodes of
the per-node count distribution `P(n_v = 1), P(n_v = 2)`, sorted
outcome by outcome as PR #69's per-cell trajectory multiset, plus the
distribution of the total count (`bins`). The score of a pair is PR
#69's `separation` — the largest difference between the two statistics
bin by bin — so the numbers are comparable with its tables, and the L1
distance the handout asks for is recorded beside it in the CSV.

## The 2-FWL ceiling is a theorem

**Claim.** Every binned two-photon counting statistic of the passive
invariant model is equal on two 2-FWL-equivalent (`C^3`-equivalent)
graphs. So the 2-FWL rung is not probed empirically beyond one
sanity row: its exact zero is a prediction, and a nonzero value there
would falsify the engine, never the theorem.

*Sketch.*

1. **Walks.** The step matrix `S` is built `S_N`-equivariantly from the
   graph: its block at node `v` is the same unitary `U` for every node
   of the same degree — here for every node — and its off-diagonal
   routing is the adjacency. An entry of `S^k`, and hence of the
   spacetime transfer matrix `M`, is therefore a sum over *decorated
   walks* of length `k` in `G`: edge steps between adjacent nodes
   weighted by dart-to-dart entries of `U`, memory self-loops, and a
   drive stub at the start and an output stub at the end. Each entry of
   `M` is a weighted homomorphism count of a decorated path, pinned at
   its two ends.
2. **Necklaces.** A two-photon probability is
   `|per M[{i, j}, {k, l}]|^2 = |M_ik M_jl + M_il M_jk|^2`, a signed sum
   of products of four walks. Expanding, every term glues at most four
   walks along their endpoint modes into disjoint closed chains: the
   diagonal terms `|M_ik|^2 |M_jl|^2` give two closed 2-walk cycles, the
   cross terms one closed 4-walk cycle. Every such pattern is a cycle of
   paths — series-parallel, of treewidth at most 2.
3. **Binning pins one node.** The per-node distribution `P(n_v = c)` sums
   the probabilities over the pairs with `v` in the required positions,
   pinning the node `v` on the necklace; the total-count distribution
   sums over all measured pairs and pins nothing. Sorting over nodes then
   takes a multiset of pinned counts, which is determined by the stable
   2-FWL colouring since pinned treewidth-`<= 2` homomorphism counts are
   2-FWL invariants of the pinned node's colour class.
4. **Dvořák / Dell–Grohe–Rattan.** Homomorphism counts from graphs of
   treewidth at most `k` are exactly the invariants of `C^{k+1}`, i.e. of
   `k`-FWL (Dvořák 2010; Dell, Grohe, Rattan 2018). With `k = 2` every
   statistic the certificate bins is a 2-FWL invariant.

**The lower edge.** 1-WL invariants are exactly the homomorphism counts
from *trees* (treewidth 1), and the necklace patterns are genuinely
cyclic — the closed 2-walk `|M_ik|^2` is already a cycle — which is why
the certificate can, and below does, exceed 1-WL. So the two-photon
invariant passive model sits strictly inside the treewidth-2 window:
above trees, at most `C^3`. It lives inside the passive family of PR #69,
whose ceiling was measured as the cellular algebra; the theorem locates
it exactly.

**Caution.** The theorem covers the *two-photon certificate*. The
`p`-photon sector expands into necklaces of up to `2p` walks whose
terminal multigraphs can have treewidth `Theta(p)`, so higher
post-selected sectors are not covered and may climb — which sharpens,
rather than contradicts, the E4 remark of
`docs/quantum_map_neural_networks.md` and PR #69's "more photons in the
post-selected sector" reading of the ceiling. There is also nothing
smaller to run on that rung: rook `4x4` vs Shrikhande, `SRG(16, 6, 2, 2)`,
is the smallest strongly regular parameter set admitting
non-isomorphic graphs, so `str-00` is the cheapest 2-FWL-blind instance
among simple graphs.

## Results

Separations, the largest bin-wise difference between the two graphs'
statistics; "exact 0" is an entry at or below `1e-15`; a blank cell was
pruned — the Grover point is exactly zero on both controls, so it does
not climb; the rook/Shrikhande row runs at `T = 2` for the canonical
cell (and for the ladder and generic cells and seven seeds before the
row was restricted, all in the CSV). The last column is PR #69's
passive model at `T = 8` on the same pairs, under its own every-tick
readout.

| pair | rung | canonical T=2 | canonical T=3 | canonical T=4 | grover T=4 | ladder T=4 | generic T=4 | #69 passive T=8 |
|---|---|---|---|---|---|---|---|---|
| control-c6-vs-p6 | control | 9.9e-03 | 1.3e-02 | 4.7e-03 | exact 0 | 1.9e-03 | 1.7e-02 | 2.8e-02 |
| control-k15-vs-p6 | control | 3.0e-02 | 2.4e-02 | 7.2e-03 | exact 0 | 2.6e-03 | 2.2e-02 | 1.0e-02 |
| regular-24 | 1fwl-blind | 1.8e-04 | 8.3e-03 | 8.8e-03 |  | 1.5e-04 | 4.4e-03 | 3.0e-03 |
| regular-47 | 1fwl-blind | 9.7e-05 | 2.8e-04 | 2.4e-03 |  | 1.5e-04 | 3.8e-03 | 3.3e-03 |
| extension-09 | 1fwl-blind | 5.3e-05 | 3.0e-03 | 1.5e-03 |  | 2.7e-04 | 4.7e-03 | 3.3e-03 |
| extension-43 | 1fwl-blind | 5.3e-05 | 3.3e-03 | 5.6e-05 |  | 1.9e-04 | 3.0e-03 | 4.1e-03 |
| classic-2c3-vs-c6 | 1fwl-blind | 1.1e-03 | 3.4e-02 | 2.2e-02 |  | 1.1e-03 | 4.1e-03 | 9.3e-03 |
| str-00 | 2fwl-blind | exact 0 |  |  |  |  |  | exact 0 |

The seed ensemble, twenty draws of `(W, psi)`, at `T = 4` and `T = 2`;
"separating" counts the seeds above `1e-15`:

| pair | rung | seeds | separating | max | median | min |
|---|---|---|---|---|---|---|
| control-c6-vs-p6 | control | 20 | 20 | 1.1e-02 | 1.9e-03 | 1.6e-05 |
| control-k15-vs-p6 | control | 20 | 20 | 1.4e-02 | 5.0e-03 | 2.0e-05 |
| regular-24 | 1fwl-blind | 20 | 20 | 9.3e-03 | 9.8e-04 | 7.4e-06 |
| regular-47 | 1fwl-blind | 20 | 20 | 3.5e-03 | 2.5e-04 | 1.7e-06 |
| extension-09 | 1fwl-blind | 20 | 20 | 1.2e-02 | 8.4e-04 | 6.6e-06 |
| extension-43 | 1fwl-blind | 20 | 20 | 1.7e-02 | 1.2e-03 | 7.0e-06 |
| classic-2c3-vs-c6 | 1fwl-blind | 20 | 20 | 1.5e-02 | 2.7e-03 | 2.7e-05 |

| pair | rung | seeds | separating | max | median | min |
|---|---|---|---|---|---|---|
| control-c6-vs-p6 | control | 20 | 20 | 2.4e-02 | 2.1e-03 | 2.4e-06 |
| control-k15-vs-p6 | control | 20 | 20 | 7.5e-02 | 5.7e-03 | 7.1e-06 |
| regular-24 | 1fwl-blind | 20 | 20 | 3.2e-04 | 1.4e-05 | 2.1e-11 |
| regular-47 | 1fwl-blind | 20 | 20 | 3.4e-04 | 3.9e-06 | 8.9e-12 |
| extension-09 | 1fwl-blind | 20 | 20 | 3.9e-04 | 5.8e-06 | 3.4e-12 |
| extension-43 | 1fwl-blind | 20 | 20 | 3.9e-04 | 5.8e-06 | 3.4e-12 |
| classic-2c3-vs-c6 | 1fwl-blind | 20 | 20 | 2.0e-03 | 8.5e-05 | 1.3e-10 |
| str-00 | 2fwl-blind | 7 | 0 | exact 0 | exact 0 | exact 0 |

The three certificates at `T = 4` — two heralded photons, one heralded
photon, and a coherent-state drive of two photons on average — for the
canonical cell, then for the ladder and generic contrasts:

| pair | rung | two-photon | one-photon | coherent |
|---|---|---|---|---|
| control-c6-vs-p6 | control | 4.7e-03 | 3.8e-03 | 3.8e-02 |
| control-k15-vs-p6 | control | 7.2e-03 | 4.2e-03 | 1.2e-01 |
| regular-24 | 1fwl-blind | 8.8e-03 | 6.8e-03 | exact 0 |
| regular-47 | 1fwl-blind | 2.4e-03 | 1.8e-03 | exact 0 |
| extension-09 | 1fwl-blind | 1.5e-03 | 1.2e-03 | exact 0 |
| extension-43 | 1fwl-blind | 5.6e-05 | 8.3e-05 | exact 0 |
| classic-2c3-vs-c6 | 1fwl-blind | 2.2e-02 | 1.8e-02 | exact 0 |

| pair | rung | two-photon | one-photon | coherent |
|---|---|---|---|---|
| control-c6-vs-p6 | control | 1.9e-03 | 2.0e-03 | 1.9e-02 |
| control-k15-vs-p6 | control | 2.6e-03 | 2.7e-03 | 1.7e-02 |
| regular-24 | 1fwl-blind | 1.5e-04 | 1.4e-04 | exact 0 |
| regular-47 | 1fwl-blind | 1.5e-04 | 1.5e-04 | exact 0 |
| extension-09 | 1fwl-blind | 2.7e-04 | 2.5e-04 | exact 0 |
| extension-43 | 1fwl-blind | 1.9e-04 | 1.8e-04 | exact 0 |
| classic-2c3-vs-c6 | 1fwl-blind | 1.1e-03 | 1.2e-03 | exact 0 |

| pair | rung | two-photon | one-photon | coherent |
|---|---|---|---|---|
| control-c6-vs-p6 | control | 1.7e-02 | 1.0e-02 | 4.4e-02 |
| control-k15-vs-p6 | control | 2.2e-02 | 1.2e-02 | 1.6e-01 |
| regular-24 | 1fwl-blind | 4.4e-03 | 2.4e-03 | 3.3e-02 |
| regular-47 | 1fwl-blind | 3.8e-03 | 2.9e-03 | 1.6e-02 |
| extension-09 | 1fwl-blind | 4.7e-03 | 2.4e-03 | 4.4e-02 |
| extension-43 | 1fwl-blind | 3.0e-03 | 1.8e-03 | 3.9e-02 |
| classic-2c3-vs-c6 | 1fwl-blind | 4.1e-03 | 3.7e-03 | 4.8e-02 |

At `T = 2` for the canonical cell:

| pair | rung | two-photon | one-photon | coherent |
|---|---|---|---|---|
| control-c6-vs-p6 | control | 9.9e-03 | 7.7e-03 | 9.0e-03 |
| control-k15-vs-p6 | control | 3.0e-02 | 2.3e-02 | 7.2e-02 |
| regular-24 | 1fwl-blind | 1.8e-04 | exact 0 | exact 0 |
| regular-47 | 1fwl-blind | 9.7e-05 | exact 0 | exact 0 |
| extension-09 | 1fwl-blind | 5.3e-05 | exact 0 | exact 0 |
| extension-43 | 1fwl-blind | 5.3e-05 | exact 0 | exact 0 |
| classic-2c3-vs-c6 | 1fwl-blind | 1.1e-03 | exact 0 | exact 0 |
| str-00 | 2fwl-blind | exact 0 | exact 0 | exact 0 |

**V2, the rotation-system table**: for each graph of a pair, the largest
separation between its sorted rotation system and three shuffled ones,
maximised over the two graphs, at `T = 4`:

| pair | rung | canonical | ladder | generic |
|---|---|---|---|---|
| control-c6-vs-p6 | control | exact 0 | exact 0 | 9.9e-03 |
| control-k15-vs-p6 | control | exact 0 | exact 0 | 9.9e-03 |
| regular-24 | 1fwl-blind | exact 0 | exact 0 | 2.5e-02 |
| regular-47 | 1fwl-blind | exact 0 | exact 0 | 4.9e-03 |
| extension-09 | 1fwl-blind | exact 0 | exact 0 | 7.2e-03 |
| extension-43 | 1fwl-blind | exact 0 | exact 0 | 6.5e-03 |
| classic-2c3-vs-c6 | 1fwl-blind | exact 0 | exact 0 | 5.3e-03 |

**V3, relabelling invariance**: a graph against a random relabelling of
itself at `T = 4`:

| pair | rung | canonical | generic | ladder | seed-01 |
|---|---|---|---|---|---|
| control-c6-vs-p6 | control | exact 0 | 2.6e-03 | exact 0 | exact 0 |
| control-k15-vs-p6 | control | exact 0 | exact 0 | exact 0 | exact 0 |
| regular-24 | 1fwl-blind | exact 0 | 2.7e-03 | exact 0 | exact 0 |
| regular-47 | 1fwl-blind | exact 0 | 3.4e-03 | exact 0 | exact 0 |
| extension-09 | 1fwl-blind | exact 0 | 6.7e-03 | exact 0 | exact 0 |
| extension-43 | 1fwl-blind | exact 0 | 4.4e-03 | exact 0 | exact 0 |
| classic-2c3-vs-c6 | 1fwl-blind | exact 0 | exact 0 | exact 0 | exact 0 |

## Findings against the hypotheses

**H1 — the invariant cell separates every control and every 1-FWL-blind
pair: confirmed.** The canonical cell is nonzero on all seven pairs at
every `T` in `{2, 3, 4}`, from `5.3e-5` at `T = 2` on the extension
pairs to `3.4e-2` at `T = 3` on 2C3 vs C6. The cost of invariance,
against PR #69's passive column at `T = 8`, is not an order of
magnitude: at `T = 4` the canonical cell sits between `5.6e-5`
(extension-43, where its `T = 3` value is `3.3e-3`) and `2.2e-2`, PR
#69's passive column between `3.0e-3` and `9.3e-3`. The margins are not
monotone in `T` — extension-43 rises then falls, 2C3 vs C6 peaks at
`T = 3` — as one expects from a readout taken at a single tick of a
unitary walk. Note that the ladder cell, the invariant cell PR #69
already had, lands an order below the canonical one on the same
readout (`1.5e-4` to `1.1e-3`): a degree-dependent `(W, psi)` is not a
better one.

**H2 — rook vs Shrikhande is an exact zero: confirmed, as the theorem
predicts.** `str-00` at `T = 2` gives `5.2e-18` for the canonical cell,
`2.1e-17` or below for every seed that ran before the sanity row was
restricted to the canonical cell, and `5.6e-17` for the ladder cell —
all exact zeros of the engine on a pair where the non-invariant
`generic` cell reads `1.6e-3`. That last number is not a separation of
two graphs: the generic cell's output depends on the rotation system,
so it differs between two encodings of the *same* graph by amounts of
the same order (V2 below), and a nonzero score on rook vs Shrikhande
carries no information. This is the practical reason to want
invariance: without it, no separation score means anything.

**H4 — the coherent-state drive isolates the mechanism, and more
sharply than expected.** Three certificates share the readout:

| certificate | statistic | degree in the transfer matrix `M` |
|---|---|---|
| coherent | `\|sum_slots M[slot, v]\|^2`, Poisson counts | linear amplitudes, squared once |
| one-photon | `mean_slots \|M[slot, v]\|^2` | quadratic: closed 2-walks |
| two-photon | `\|M_ik M_jl + M_il M_jk\|^2` | quartic: 4-necklaces |

The coherent drive is *exactly zero* on every 1-FWL-blind pair for both
invariant cells, at every `T`, while separating the controls at `1e-2`
to `1e-1`. This is a theorem rather than an observation: a coherent
state of the same amplitude in every slot injects the all-ones vector,
and an `S_N`-equivariant *linear* map sends a vector constant on the
colour-refinement classes to a vector constant on them, so the output
intensities are a function of the stable 1-WL colouring — the coherent
model is a linear message-passing network and inherits its 1-WL bound.
The `generic` cell escapes it (`1.6e-2` to `4.8e-2` on the same pairs)
only because it is not equivariant, i.e. by reading the rotation
system. The one-photon certificate — the intensity of *incoherent*
light, a sum of squared moduli — already exceeds 1-WL at `T >= 3`
(`8.3e-5` to `1.8e-2` for the canonical cell) but is exactly zero on
all five pairs at `T = 2`, where the two-photon certificate is not
(`5.3e-5` to `1.1e-3`). So within the fully invariant model the ladder
of mechanisms is measured: phase-locked linear drive `<=` 1-WL `<`
incoherent intensities `<` two-photon interference, with the last two
inside the treewidth-2 window of the ceiling theorem and the
two-photon permanent the only one that separates at two ticks.

**H3 — V2 passes for the invariant cells and fails for the generic one:
confirmed, with the ladder cell on the passing side.** The rotation
table is the headline: for each graph of every pair, three shuffled
rotation systems against the sorted one give a spread of at most
`1.4e-16` for the canonical cell and `1.1e-16` for the ladder cell —
exact zeros — and `4.9e-3` to `2.5e-2` for the generic cell, the same
order as its "separations". Invariance is what the ansatz bought; H1
says it cost no rung. That the ladder cell passes too is the tension
recorded above: PR #69's passive column was already a graph invariant,
by the port symmetry of its coupler, and this table is the first
measurement of that fact.

**R2 — the ensemble.** Twenty seeds of `(W, psi)`, Haar and uniform,
separate every control and every 1-FWL-blind pair at every `T`:
`420` of `420` rows nonzero. At `T = 4` the medians sit between
`2.5e-4` and `2.7e-3` and the maxima between `3.5e-3` and `1.7e-2`,
so the canonical draw is a typical member rather than a lucky one. At
`T = 2` the minima fall to `3e-12` on the extension pairs — small but
not exact zeros, the distinction §5 of the handout asks to keep — and
the seven seeds that ran the rook/Shrikhande row before it was
restricted to the canonical cell are all exact zeros, seven more
confirmations of the theorem at no design cost.

**V3 — relabelling.** A graph against a random relabelling of itself is
an exact zero for the canonical, ladder and seeded cells at `T = 4`
(table below). The generic cell fails V3 too, at `2.6e-3` to `6.7e-3`
on five of the seven pairs: the ladder's relabelling re-sorts every
vertex's neighbours, which changes the rotation system, and a cell that
reads the rotation system is not a function of the graph at all — its
two exact zeros are the pairs whose shuffles happen to be realised by
automorphisms. V3 is a corollary of V2, and only the invariant family
has either.

**V4 — probability.** Every two-photon and one-photon row conserves
probability to `5e-15` or better (`mass_error` in the CSV); the
contracted functor image on the triangle and the path sums to
`1 - 1e-15`.

## Distinguishable photons

> For the graph pairs you tested, let's see if the same photonic QMapNN
> with distinguishable photons can get the same separation

Two distinguishable photons — orthogonal internal states, say two
frequencies — never interfere: the joint probability of finding one in
each of two columns is the product of single-photon probabilities,
`|a_i|^2 |b_j|^2 + |a_j|^2 |b_i|^2`, where the bosonic certificate has
`|a_i b_j + a_j b_i|^2`; the same slot ensemble, the same readout, the
same score (`distinguishable` in `engine.py`). It is grounded in optyx's
own semantics of distinguishability: the driven `CMap` with `Create(1,
internal_states=...)` on the two slots is inflated with
`Diagram.inflate(2)` and contracted, and matches the certificate at
`7e-18` on the two-vertex path, where the two photons meet inside a cell
(`test_distinguishable_certificate_matches_the_inflated_contraction`,
which also re-derives the bosonic certificate by giving both photons the
*same* internal state), and at `2e-16` on a single vertex with a loop
edge, the smallest map where a photon comes back to meet the next one.
The inflated triangle exceeds the memory of an exact contraction.

**Same separation from three ticks on, none at two.** For the canonical
cell:

| pair | rung | bosons T=2 | distinguishable T=2 | bosons T=3 | distinguishable T=3 | bosons T=4 | distinguishable T=4 |
|---|---|---|---|---|---|---|---|
| control-c6-vs-p6 | control | 9.9e-03 | 1.1e-02 | 1.3e-02 | 1.4e-02 | 4.7e-03 | 4.9e-03 |
| control-k15-vs-p6 | control | 3.0e-02 | 3.2e-02 | 2.4e-02 | 2.6e-02 | 7.2e-03 | 7.8e-03 |
| regular-24 | 1fwl-blind | 1.8e-04 | exact 0 | 8.3e-03 | 8.5e-03 | 8.8e-03 | 9.2e-03 |
| regular-47 | 1fwl-blind | 9.7e-05 | exact 0 | 2.8e-04 | 3.0e-04 | 2.4e-03 | 2.5e-03 |
| extension-09 | 1fwl-blind | 5.3e-05 | exact 0 | 3.0e-03 | 3.1e-03 | 1.5e-03 | 1.6e-03 |
| extension-43 | 1fwl-blind | 5.3e-05 | exact 0 | 3.3e-03 | 3.3e-03 | 5.6e-05 | 1.1e-04 |
| classic-2c3-vs-c6 | 1fwl-blind | 1.1e-03 | exact 0 | 3.4e-02 | 3.6e-02 | 2.2e-02 | 2.4e-02 |
| str-00 | 2fwl-blind | exact 0 | exact 0 |  |  |  |  |

At `T = 3` and `T = 4` the distinguishable photons separate every
control and every 1-FWL-blind pair, at the same magnitude as the
indistinguishable ones and in fact slightly *above* them on every row
but one (extension-43 at `T = 4`, `1.1e-4` against `5.6e-5`): the
bosonic cross term is a small correction to a separation carried by the
classical two-walker statistics, which are already products of the
one-photon marginals that exceed 1-WL at `T >= 3`. At `T = 2` the
distinguishable certificate is an exact zero on all five 1-FWL-blind
pairs, like the one-photon one, while the bosonic certificate reads
`5.3e-5` to `1.1e-3`: two ticks is the regime where only two-photon
interference sees past colour refinement. The rook/Shrikhande row is an
exact zero for distinguishable photons too, as it must be — the
products of single-photon probabilities are 2-walk necklaces, inside
the same treewidth-2 window.

The twenty seeds with distinguishable photons, at `T = 2` and `T = 4`; the bosonic ensemble is in the results section above:

| pair | rung | seeds | separating | max | median | min |
|---|---|---|---|---|---|---|
| control-c6-vs-p6 | control | 20 | 20 | 2.5e-02 | 2.3e-03 | 2.0e-06 |
| control-k15-vs-p6 | control | 20 | 20 | 7.8e-02 | 5.7e-03 | 6.1e-06 |
| regular-24 | 1fwl-blind | 20 | 0 | exact 0 | exact 0 | exact 0 |
| regular-47 | 1fwl-blind | 20 | 0 | exact 0 | exact 0 | exact 0 |
| extension-09 | 1fwl-blind | 20 | 0 | exact 0 | exact 0 | exact 0 |
| extension-43 | 1fwl-blind | 20 | 0 | exact 0 | exact 0 | exact 0 |
| classic-2c3-vs-c6 | 1fwl-blind | 20 | 0 | exact 0 | exact 0 | exact 0 |

| pair | rung | seeds | separating | max | median | min |
|---|---|---|---|---|---|---|
| control-c6-vs-p6 | control | 20 | 20 | 1.2e-02 | 2.2e-03 | 1.9e-05 |
| control-k15-vs-p6 | control | 20 | 20 | 1.5e-02 | 5.6e-03 | 2.3e-05 |
| regular-24 | 1fwl-blind | 20 | 20 | 9.7e-03 | 1.0e-03 | 8.5e-06 |
| regular-47 | 1fwl-blind | 20 | 20 | 3.9e-03 | 2.7e-04 | 2.0e-06 |
| extension-09 | 1fwl-blind | 20 | 20 | 1.3e-02 | 8.9e-04 | 7.3e-06 |
| extension-43 | 1fwl-blind | 20 | 20 | 1.7e-02 | 1.3e-03 | 7.8e-06 |
| classic-2c3-vs-c6 | 1fwl-blind | 20 | 20 | 1.7e-02 | 2.9e-03 | 3.3e-05 |

So the answer to the question is *yes* for `T >= 3` and *no* for
`T = 2`: on these pairs the invariant QMapNN's climb past 1-WL does not
need photon indistinguishability, it needs two (or one) photons and
three ticks; what indistinguishability buys is the separation at the
shortest depth, and the `T = 2` column is the row where the quantum
statistics are the mechanism.

## Where this sits in the ceiling discussion

`docs/quantum_map_neural_networks.md` proposes, as E4, climbing with more
photons in the post-selected sector and, as E5(i), proving the passive
ceiling — that driven linear-optical statistics of a graph `CMap`
factor through the cellular algebra. PR #69 measured that ceiling: its
passive and Bell models are exact zeros from the 2-FWL rung on, and it
read them as the cellular-algebra bound of `beyond_3wl`. The invariant
family lives *inside* PR #69's passive family — its cells are a
sub-manifold of the port-symmetric interferometers, and the ladder cell
is literally one of them — so its ceiling is at most that one; the
theorem above locates it exactly, for the two-photon certificate, at
2-FWL (`C^3`) by the treewidth of the necklace patterns, and the
`p`-photon caution says where E4 can still climb: the necklaces of `2p`
walks leave the treewidth-2 window. Below the ceiling, the three
certificates order the mechanisms inside one invariant model —
phase-locked linear drive at 1-WL, incoherent intensities and two-photon
interference above it — which is the H4 row the proposal wanted for the
paper, measured here with the Poisson model of the coherent drive rather
than a Fock truncation.

What this experiment does not do: no cell is trained (E3), no photon
sector above two is run (E4), and no feed-forward is in the loop, so the
active-model results of PR #69 — the strongly regular pairs split by
classical broadcast — are untouched. The invariant cell is a passive
coin; its Grover point (`psi = pi`, `W` diagonal) is exactly zero on the
controls because the drive never enters the graph, and was pruned
there.

## Reproducing

```shell
python experiments/invariant_qmapnn/run.py --cells canonical,grover,ladder,generic \
    --certificates two-photon,one-photon,coherent
python experiments/invariant_qmapnn/run.py                     # the 20 seeds
python experiments/invariant_qmapnn/run.py --rotations --cells canonical,ladder,generic --ticks 4
python experiments/invariant_qmapnn/run.py --invariance --cells canonical,ladder,generic,seed-01 --ticks 4
python experiments/invariant_qmapnn/run.py --certificates distinguishable --cells canonical,ladder,seed-01,...,seed-20
python experiments/invariant_qmapnn/report.py                  # render the tables
```

The dataset is `experiments/kwl_ladder/data/dataset.json`, committed;
`results/results.csv` carries every row with its L1 distance, mass error
and wall time, `results/rotations.csv` the rotation spreads per graph
side. Everything ran locally; the longest row is the sixteen-vertex
sanity pair at about `140 s`, dominated by the path matrices of the
degree-six cells, and the seven small pairs take a few seconds each.
