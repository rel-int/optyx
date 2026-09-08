# TODO

> Here's a handout from Fable for a new experiment on QMapNNs. Open a new PR in optyx implementing it

The handout, verbatim:

> # Handout: does the permutation-invariant linear-optical QMapNN still climb past 1-WL?
> 
> **For:** Claude Code, working in `rel-int/optyx`
> **Branch base:** `claude/kwl-ladder-awc5sc` (PR #69) — reuse its dataset, engines, and validation discipline. Related context: PR #61 (active photonics beyond 3-WL), PR #58 (beyond-1-WL notebook), PR #16 (`optyx.interaction`).
> **Scope:** small, exact, local. No training. One new cell family, a handful of graph pairs, T ≤ 4.
> 
> ---
> 
> ## 1. Question and background
> 
> QMapNN cells so far read the rotation system σ: the node unitary treats its incident edge modes as an *ordered* list, so the model computes map invariants, not graph invariants. We now impose full S_d-invariance at every node (d = deg(v)): the cell must commute with every permutation of its incident edge modes. By Schur (permutation rep on ℂ^d = trivial ⊕ standard, each multiplicity one on the bare dart block), the invariant unitaries are completely classified — see §3. This makes the whole network an exact **graph** invariant by construction.
> 
> The question: **does this invariant subfamily still separate 1-WL-equivalent graph pairs?**
> 
> Priors from PR #69 that frame the experiment:
> - The unconstrained passive cell separates all five 1-FWL-blind pairs at T = 8 (scores 3.0e-3–9.3e-3) and is **exactly zero** on every 2-FWL-blind pair (cellular-algebra ceiling).
> - The invariant family defined below is a *subfamily* of the passive cells. Therefore the 2-FWL exact zero is inherited for free — do **not** burn compute re-establishing it beyond one confirmation pair. The open question is only whether the 1-FWL rung survives the symmetrisation.
> - Theory says it should: the invariant cell is a trainable-phase Grover-type coin, and coined walks with symmetric coins access spectral/closed-walk invariants outside 1-WL. Expected outcome: separation persists on the 1-FWL rung, possibly with smaller magnitude than the σ-sensitive cell. Either result (persists / collapses) is a finding; measure, don't assume.
> 
> New claim to *verify* rather than assume: the invariant model's output is exactly unchanged under arbitrary relabelling of darts around each vertex (i.e. change of rotation system for the same graph), which the σ-sensitive passive cell fails. This is the whole point of the ansatz.
> 
> ## 2. Architecture (fixed by the experiment spec — do not deviate)
> 
> Per graph G = (V, E), build one `optyx.interaction.CMap` with one cell per node. Modes:
> 
> - **Edge wiring — one dart pair per edge** {u,v}: two directed wires, v's out-dart into u's in-dart at the next tick and vice versa (the α-swap built into the wiring; a coined-walk layout). *Deviation from the original one-mode-per-edge spec, forced by invariance:* in the symmetric monoidal semantics every wire has one writer per tick, so a single undirected edge mode requires either an orientation or a firing schedule for the two endpoint cells — and no canonical, S_N-equivariant choice of either exists on general graphs (non-bipartite graphs admit no parity schedule; any fixed orientation breaks relabelling invariance). The dart pair removes the choice. Cost is 2|E| instead of |E| edge modes — negligible under the certificate engine. If Giovanni supplies a single-writer-per-tick semantics that is canonically defined, revert; otherwise darts. The cell's invariance group is then S_d acting *simultaneously* on the d in-darts and d out-darts (diagonal action); the commutant classification in §3 applies to that action, with B applied to in-darts and B† to out-darts.
> - **One memory mode per node**: a feedback wire from the cell to itself (stream semantics, as in PR #12/#15).
> - **One drive mode per node**: a fresh single photon injected into it at **every** time step.
> - **One output mode per node**: discarded at every step **except the last**, where it is the *only* thing measured (photon counting on the N output modes; everything else — edge modes, memory modes — is discarded at the end).
> 
> Everything is passive linear optics: each cell is a (d+2)-in / (d+2)-out mode unitary — inputs (d edge modes, memory, drive), outputs (d edge modes, memory, output). Interpretation stays physical: CPTP (unitary + discards), as PR #69 required.
> 
> Run for T time steps; T ∈ {2, 3, 4}. Start from vacuum on all edge/memory modes.
> 
> ## 3. The invariant cell — exact specification
> 
> S_d acts on the d edge modes and trivially on {memory, drive/output}. The commutant of this action inside U(d+2) is exactly:
> 
>   U(φ⃗) = B† · [ W ⊕ e^{iψ} I_{d−1} ] · B
> 
> where:
> - B is any fixed balanced multiport on the d edge modes sending the symmetric vector s = (1,…,1)/√d to port 0, extended by identity on memory and drive. Use the d-mode DFT, or a Householder reflection mapping s ↦ e₀ (Householder is numerically cleaner for arbitrary d). B is **not** trainable.
> - W ∈ U(3) acts on {symmetric port, memory, drive}; its third output column is routed to the **output mode**. Trainable: 9 real parameters.
> - e^{iψ} is a single shared phase on the d−1 non-symmetric edge ports. Trainable: 1 parameter.
> 
> Per-node parameter count: 10, degree-independent. Two parameter regimes to run:
> - **R1 (canonical / untrained):** ψ = π, W = a fixed Haar-random U(3) drawn once and *shared across all nodes and all time steps* (node-uniformity is required for graph-level invariance unless tied to invariant node data; with no node features, share everything). ψ = π on the complement with identity-like W degenerates toward the Grover coin — include the exact Grover point (ψ = π, W = diag phases) as one row for reference.
> - **R2 (random ensemble):** ~20 seeds of shared (W, ψ) drawn Haar/uniform; report per-pair max and median separation. No gradient training.
> 
> Implementation note: build `invariant_cell(d, W, psi) -> (d+2)x(d+2) ndarray`, and a one-line lemma test: for 50 random permutation matrices P (acting on the edge block, identity elsewhere), assert ‖P† U P − U‖ ≤ 1e−12. This is the Schur claim made executable.
> 
> ## 4. Graph pairs and budget
> 
> Reuse `experiments/kwl_ladder/` dataset and its measured rungs. Run, in order, pruning as in PR #69 (stop climbing when a rung fails to separate):
> 
> 1. **Controls (2):** the two control pairs from the ladder dataset (colour-refinement-distinguishable). Expected: clear separation. If zero here, the cell or readout is broken — stop and debug.
> 2. **1-FWL-blind rung (5 pairs):** the five pairs from the ladder dataset. **This is the experiment.** Include C6 vs C3⊔C3 if not already among them (d = 2 everywhere; note at d = 2 the invariant family is just phases on the symmetric/antisymmetric edge combinations — still expected to separate via spectra).
> 3. **2-FWL rung: no empirical probe — the ceiling is now a theorem, not an observation.** For the two-photon certificate the bound is provable (see §4b): every statistic the certificate computes is a linear combination of homomorphism counts from treewidth-≤2 patterns, hence equal on 2-FWL (C³)-equivalent graphs. Note also there is nothing smaller to run: rook-4×4 vs Shrikhande, SRG(16,6,2,2), is the standard minimal 3-WL(tuple)/2-FWL failure pair — the smallest SRG parameter set admitting non-isomorphic graphs — so no cheaper instance of that rung exists among simple graphs. *Optional engine-sanity row only:* rook/Shrikhande at T = 2; a nonzero value would falsify the engine (or the theorem's applicability to the implemented layout), which is precisely why it's worth one cheap row. Do not run anything else at or above this rung.
> 
> ### 4b. Ceiling theorem for the two-photon certificate (write this up in the README)
> 
> Claim: any binned two-photon counting statistic of the passive invariant model is unchanged between 2-FWL-equivalent graphs. Sketch: (i) entries of the spacetime transfer matrix M are sums over decorated walks in G (edge steps, memory self-loops, drive stubs) because the step matrix is built S_N-equivariantly from the graph; (ii) a two-photon probability is |per(M[{i,j},{k,l}])|², whose expansion is a signed sum over gluings of ≤4 walks along their endpoint modes into disjoint closed chains — "necklaces" (diagonal terms give two 2-walk cycles; cross terms one 4-walk cycle); every such pattern is a cycle of paths, i.e. series-parallel, treewidth ≤ 2; (iii) binning as a multiset over nodes pins one vertex, and pinned treewidth-≤2 homomorphism counts are determined by the stable 2-FWL colouring; (iv) by Dvořák / Dell–Grohe–Rattan, treewidth-≤k hom counts are exactly the C^{k+1} ≡ k-FWL invariants — so the certificate's statistics are 2-FWL invariants. Corollary (the lower edge): 1-WL invariants are exactly the *tree* (treewidth-1) hom counts, and the necklace patterns are genuinely cyclic, which is why the certificate can and does exceed 1-WL. So the two-photon invariant passive model sits strictly inside the treewidth-2 window: above trees, at most C³. Caution to record: the theorem covers the *two-photon certificate* — p-photon sectors expand into necklaces of up to 2p walks whose terminal multigraphs can have treewidth Θ(p), so higher post-selected sectors are *not* covered and may climb (this sharpens, rather than contradicts, PR #69's E4 remark).
> 
> Sizes stay ≤ 16 nodes except the confirmation pair; T ≤ 4. If any run exceeds the local budget, shrink T before shrinking the pair list.
> 
> ## 5. Engine and exactness
> 
> Default to the PR #69 methodology, not fresh Fock-space truncation:
> - The homogeneous every-node-every-step drive is evaluated through the **two-herald post-selected certificate**: uniform average over injection pairs of exact two-photon dynamics, computed by permanents on the spacetime transfer matrix. Generalise the assembled step matrix to this layout (edge + memory + drive/output wiring) and reassert `assembled == CMap.step.to_path()` for the new cell.
> - **Ground the engine in the functor image**: on the triangle (and one path graph), contract the driven `CMap` directly through optyx and match the certificate entry-by-entry at ≤ 1e−14, as PR #69 did. PR #69 found two engine bugs *only* through this check; keep it in CI (fast half only, in `test/`).
> - Exact zeros matter: report entries ≤ 1e−15 as exact zeros, distinct from small-but-nonzero (the 1.2e−4 vs 6.9e−18 distinction in #69's table is the scientific content).
> 
> ## 6. Readout and separation score
> 
> At the final step, photon-count statistics on the N output modes only. Graph-level invariant statistic: the **multiset over nodes of per-node output count distributions** (same shape as #69's per-cell trajectory-distribution multiset), plus the total-count distribution. Separation score between the two graphs of a pair: L1 distance between the sorted/multiset-binned statistics — reuse the ladder's binning and scoring code so numbers are comparable across PRs.
> 
> ## 7. Required validations (all measured, none assumed)
> 
> - **V1 — Cell invariance (Schur):** permutation-conjugation assert from §3.
> - **V2 — Rotation-system independence (the new property):** for each pair, evaluate the model on the graph under 3 different rotation systems (shuffle each vertex's incident-edge order). Assert score between rotations of the *same* graph is exactly 0 (≤ 1e−14). Run the σ-sensitive passive cell from the ladder on the same shuffled inputs as a contrast row — it should *fail* this (nonzero spread across rotations). This table is the headline.
> - **V3 — Relabelling invariance:** graph vs a random relabelling of itself: exact zero (as in #69).
> - **V4 — Probability conservation:** contracted total probability = 1 to 1e−12 on small graphs.
> 
> ## 8. Hypotheses (record outcomes against these)
> 
> - **H1:** invariant cells separate all controls and all five 1-FWL-blind pairs (nonzero; order-of-magnitude comparison to #69's passive column is the interesting number — quantify the "cost of invariance").
> - **H2:** the optional rook/Shrikhande sanity row, if run, is an exact zero — now a *prediction of the §4b theorem*, so a nonzero result is an engine bug (or a layout outside the theorem's hypotheses), never a discovery.
> - **H3:** V2 passes for the invariant cell and fails for the passive cell — invariance is what we bought; H1 is what we didn't pay.
> - **H4 (stretch, only if budget remains):** replace the single-photon drive with a coherent-state drive of matched mean photon number (classical-optics semantics). Prediction: the coherent model's statistics are functions of the single-particle transfer matrix only; if it separates strictly fewer pairs than the single-photon model, the gap isolates *photon statistics* (permanents) as the quantum mechanism, cleanly, inside a fully graph-invariant model. This is the row that would matter for the paper; still, it is optional here.
> 
> ## 9. Deliverables
> 
> - `experiments/invariant_qmapnn/` with: cell module, engine adaptation, runner (one CSV row per (pair, regime), resumable), `README.md` with the rendered tables (controls / 1fwl / confirmation / V2 rotation table / H4 if run), and findings written against H1–H4.
> - Fast tests in `test/test_invariant_qmapnn.py`: V1, the triangle functor-image grounding, V3 on P4.
> - A short section in the README relating results to `docs/quantum_map_neural_networks.md` E4/E5 and to PR #69's ceiling discussion: this cell family lives inside the passive family, so its ceiling is at most the cellular algebra; the experiment locates it exactly.
> - If anything in this handout conflicts with what the code makes natural, prefer the spec in §2 (it is the agreed experimental design) and note the tension in the README rather than silently redesigning.

## Work

- [x] `experiments/invariant_qmapnn/cells.py`: `invariant_cell(d, W, psi)` (§3), the contrast cells, and the functor from a graph with a rotation system to an `optyx.interaction.CMap` in the dart-pair / memory / drive / output layout (§2)
- [x] `experiments/invariant_qmapnn/engine.py`: the step matrix assembled from the functor image, checked against `CMap.step.to_path()`; the spacetime transfer over injection slots; the two-photon, one-photon and coherent-state certificates read on the final output modes (§5, §6, H4)
- [x] `experiments/invariant_qmapnn/run.py` + `report.py`: one resumable CSV row per (pair, cell, T), regimes R1 (canonical, Grover point) and R2 (20 seeds), pruning as in PR #69, the V2 rotation table and the V3 relabelling rows (§4, §7)
- [x] `test/test_invariant_qmapnn.py`: V1 Schur lemma, assembled == `CMap.step.to_path()`, the triangle and path functor-image grounding, V2 on P4 for both cells, V3 on P4, V4 (§7, §9)
- [x] Run controls, the 1-FWL rung at T = 2, 3, 4, the rook/Shrikhande sanity row at T = 2, the rotation and relabelling tables, H4; commit `results/`
- [x] `experiments/invariant_qmapnn/README.md`: rendered tables, the §4b ceiling theorem, findings against H1–H4, the relation to `docs/quantum_map_neural_networks.md` E4/E5 and PR #69, and the tensions with the handout (dart pairs are native to `CMap`; the ladder's cells are already port-symmetric)
- [x] `pflake8` clean, `coverage run -m pytest` green, `AGENTS.md`/`CONTRIBUTING.md` pointers if needed

> For the graph pairs you tested, let's see if the same photonic QMapNN with distinguishable photons can get the same separation

- [x] `engine.py`: a `distinguishable` two-herald certificate, the pair statistics as products of single-photon probabilities, grounded on the triangle against optyx's `inflate` of the driven functor image with orthogonal internal states
- [x] Run it on the same pairs, cells and ticks as the two-photon rows (canonical at T = 2, 3, 4 and the sanity row, the ladder cell, the 20 seeds); render the column beside the bosonic one
- [x] README findings: the distinguishable column against the two-photon and one-photon ones, tests and lint green


## Inherited from PR #69

The `TODO.md` of the branch this one is stacked on, kept as it was:

### PR #69 TODO

> Let's see how far we can climb up the k-WL ladder. Use Cai et al. and the BREC benchmark to construct a dataset of graphs of increasing k-WL indistinguishability (say 5 graphs for each k). Then benchmark the following models: 1. passive photonic QMapNN with a single photon per cell, 2. passive photonic QMapNN with a dual-rail Bell state input at each time step, 3. active photonic QMapNN with both classical and quantum messages, 4. Qubit QMapNN with a qubit circuit and quantum messages (only the prediction measured), 5. Qubit QMapNN with both classical and quantum messages. In each case, the model should be simply a functor turning the CMap representing the graph into an optyx.interaction.CMap. The interpretation of the cells should be physical, i.e. a CPTP map + optionally measurements and classical feedforward. You can use the modal GPUs for large graphs and possibly some tensor network approximation methods if the graph is too big.

- [x] Dataset: pairs of increasing k-WL indistinguishability from Cai-Fürer-Immerman and BREC, about five per rung, verified with our own 1-FWL / 2-FWL / 3-FWL — 2 controls, 5 + 5 + 5 measured rungs; every BREC dr and 4vtx pair measures 3-FWL-distinguishable, the 3-FWL-blind rung is CFI at n = 80..106
- [x] The graph as a CMap and the five models as functors into `optyx.interaction.CMap`, every cell a physical CPTP map (plus the relayed-flood active variant)
- [x] Exact few-photon certificates for the photonic models validated against optyx's own contraction of the functor image — which surfaced and fixed two discrepancies between the `beyond_3wl` Machine and its drawn cell (kicks on reflections; replaced rather than composed kick splitter), reported on #61
- [x] Benchmark over the ladder with the pruning rule; qubit models contract through quimb, the qubit-ff wall recorded as trimmed; Modal fan-out implemented and validated but unreachable from this sandbox (the egress proxy blocks gRPC), so all numbers are local
- [x] Committed results (CSV + report + README findings) with probability conservation at 1e-14 and exact-zero relabelling invariance
- [x] Lint and tests green (functor-image smoke tests in `test/`), draft PR #69 stacked on #61

> Once you see that the model doesn't separate, don't keep runnning it for higher ks. Focus on the ones that climb

- [x] Prune the benchmark accordingly: passive and Bell stop at the 2-FWL rung, the qubit models stop at the controls (the unitary cell measures blind even there; the classical-control cell does not contract), and the active family — the one that climbs — separates every strongly regular pair, with both one-hop and unbounded-relay broadcast exactly zero on the locally-isomorphic CFI and distance-regular pairs of the same rung, so its ceiling is local distinguishability, not a k-WL level
