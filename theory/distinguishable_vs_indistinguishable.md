# Distinguishable versus indistinguishable photons in linear-optical GNNs

**Question.** In the invariant-cell experiment of [PR #70](https://github.com/rel-int/optyx/pull/70), two *distinguishable* photons separate every 1-WL-blind pair that two *indistinguishable* photons separate, at the same magnitude, from three ticks on; at two ticks only the indistinguishable photons separate. Why does a setting that is classically simulable — realisable with classical light, one source at a time — beat message-passing GNNs, and what exactly does indistinguishability add?

**Answer.**

1. *The 1-WL barrier is crossed by particles, not by interference.* A coherent field adds the amplitudes of all sources and squares once, a walk pattern that is a tree; a particle squares per source and sums the probabilities, $\sum_s |M_{vs}|^2$, a walk pattern that is a *cycle*, and closed walks are exactly what colour refinement cannot count (Theorem 1). The source is used inside the computation and summed away, so every readout is a graph invariant (Lemma 1.1); in GNN terms the particle model is a node-marking subgraph GNN whose marking is free. Any source-resolved intensity has this, classical or quantum.
2. *A single detector never sees indistinguishability.* At any photon number, the count distribution on one output mode is a fixed bijective function of the same data for bosons and for distinguishable photons: the elementary symmetric polynomials of the single-photon probabilities at that mode. Bunching is a combinatorial factor $k!$, not graph data (Theorem 2). This is why the per-node readouts of PR #70 agree.
3. *At two photons the entire difference is one number:* the mean squared overlap $X$ of the two photons' output-restricted wavefunctions, the Hong–Ou–Mandel term, which enters only the multi-detector bins (Theorem 3). It is a 2-WL invariant like everything else in the model (Theorem 4).
4. *What $X$ buys is depth.* A closed walk is built from the segments of a necklace; with one photon the necklace has two segments, with the exchange term four. At two ticks each segment is at most one hop, so only the four-segment necklace closes a triangle — through two detectors rather than one (Proposition 5). This is the $T = 2$ column.
5. *Distinguishable photons never leave 2-WL, at any photon number and for every readout of order $\le 2$* (Theorem 6). Bosons can only leave it through bunched interference of three or more photons on several detectors — exactly the regime where their classical simulation becomes hard. If there is an expressivity gap between the two settings, it coincides with the simulability gap.

**Convention.** Notation and the model are those of [`plo_gnn_wl_theorems.md`](plo_gnn_wl_theorems.md) (the *handout*), whose Definition 2.1 is the invariant-cell QMapNN of PR #70; "1-WL" is colour refinement, "2-WL" is 2-FWL, i.e. $C^3$, i.e. the coherent algebra $\mathcal{W}(G)$ of its §1. Every numerical claim below was recomputed from PR #70's `engine.py` with its canonical cell ($\psi = \pi$, $W$ Haar from seed 0) on the pairs of `experiments/kwl_ladder/data/dataset.json`; the separations reproduce PR #70's tables to the printed digit, and the identities are checked to $10^{-16}$ (Appendix).

---

## 1. Setting

A graph with $n$ vertices, driven for $T$ ticks, gives a *spacetime transfer matrix* $M$ whose rows are the injection slots $s = (u, t)$, $u \in V$, $0 \le t < T$, and whose columns are the sinks: the output mode of every vertex at every tick, and the wires kept after the last tick. The row $a_s := M[s, \cdot]$ is the single-photon amplitude of the photon injected at $s$; since the dynamics is unitary and the slots are distinct modes, the rows are orthonormal:
$$\langle a_s | a_{s'} \rangle \;=\; \sum_{\text{all sinks } j} \overline{a_s(j)}\, a_{s'}(j) \;=\; \delta_{ss'}. \tag{1.1}$$
The readout counts the $n$ output modes $O$ of the last tick; write $p_s(v) := |a_s(v)|^2$ for $v \in O$ and $M_O$ for the $|{\rm slots}| \times n$ block of $M$ on those columns. The two-herald ensemble is uniform over unordered pairs $\{s, s'\}$ of distinct slots, and for one pair the joint probability of finding one photon in each of the sinks $i, j$ is

$$\text{bosons:}\quad |a_s(i)\, a_{s'}(j) + a_s(j)\, a_{s'}(i)|^2, \qquad
\text{distinguishable:}\quad p_s(i)\, p_{s'}(j) + p_s(j)\, p_{s'}(i), \tag{1.2}$$

halved when $i = j$. The graph-level statistic of PR #70 is the sorted multiset over $v \in O$ of the per-node distributions $P(n_v = 1), P(n_v = 2)$, together with the distribution of the total count $N = \sum_{v \in O} n_v$; the score of a pair of graphs is the largest bin-wise difference. Two single-particle certificates share the readout: **one-photon**, $\langle n_v \rangle = \operatorname{mean}_s p_s(v)$, and **coherent**, the Poisson counts of intensity $\propto |\sum_s a_s(v)|^2$.

---

## 2. Theorem 1: fields see trees, particles see cycles

**Theorem 1.** (a) The coherent readout $|\sum_s a_s(v)|^2$ is a function of the stable 1-WL colour of $v$. (b) The one-photon readout $\sum_s p_s(v)$ is not: it is the diagonal of $M_O^{\dagger} M_O$ summed over ticks, a closed walk through $v$, and it separates $C_6$ from $2C_3$ at $T \ge 3$ (handout, Theorem A; PR #70's one-photon rows).

*Proof of (a).* $\sum_s a_s = \mathbf{1}^{\top} M$ is the all-ones vector pushed through the tick unitaries, each a polynomial in the matrices $A\,f(D)$ and $f(D)$ of the handout's Lemma B1. Each of those is an $S_n$-equivariant linear map, and an equivariant linear map sends a vector constant on the colour-refinement classes to a vector constant on the classes refined once; iterating $T$ times lands in the stable colouring. So $(\mathbf{1}^\top M)_v$, and its modulus squared, is a function of the stable colour of $v$. $\square$

*Reading.* Both readouts are quadratic in $M$; what differs is the order of summation and squaring:
$$\Big|\sum_s a_s(v)\Big|^2 = \sum_{s, s'} a_s(v)\, \overline{a_{s'}(v)} \qquad\text{versus}\qquad \sum_s |a_s(v)|^2 = \sum_s a_s(v)\, \overline{a_s(v)}.$$
On the left, two walks $s \to v$ and $s' \to v$ are glued at $v$ only and their sources are free: the pattern is a path, a tree, and the count of tree patterns is what 1-WL determines (Dvořák; Dell–Grohe–Rattan). On the right the two walks share both endpoints: the pattern is a cycle, whose count 1-WL does not determine — the triangle count $\operatorname{tr} A^3$ is the first example, and $C_6$ versus $2C_3$ the first pair. In logical terms, 1-WL is the two-variable counting logic $C^2$ (Immerman–Lander; Cai–Fürer–Immerman), and counting the closed walks from $x$ needs a third variable that holds $x$ while two others step along the walk. This is the whole content of the phrase "a particle remembers its source": the Born rule squares each source's amplitude before the sources are summed, which pairs every walk with a copy of itself sharing both endpoints. Nothing quantum is needed for it. The right-hand side is the intensity at $v$ of mutually incoherent sources (handout §5.2, ii), and it is also what one gets by running the interferometer once per source with a laser and adding the recorded intensities.

**Lemma 1.1 (every readout is a graph invariant).** For every certificate — coherent, one-photon, distinguishable, bosonic — isomorphic graphs give identical outputs, and so do two rotation systems of the same graph.

*Proof.* A relabelling $\sigma$ of the vertices permutes the slots and the sinks; the cell is the same at every node and, by the Schur classification of the invariant cell, independent of the order of the darts; so $M[\sigma s, \sigma v] = M[s, v]$. Hence $p_{\sigma s}(\sigma v) = p_s(v)$ and $x_{\sigma s, \sigma s'} = x_{ss'}$, and every statistic of §1 is a multiset over $v$ of symmetric functions of the slots, or a sum over all slot pairs, which the permutation leaves unchanged. $\square$

Measured: PR #70's V3 rows, a graph against a random relabelling of itself, are exact zeros for the canonical, ladder and seeded cells, and its V2 rows are exact zeros across rotation systems, for every certificate. The one cell that fails both is `generic`, which fails because it reads the rotation system — for the bosonic certificate as much as for the distinguishable one. So the source is not a label that survives in the output: it is used inside the computation and summed away.

The GNN counterpart makes this concrete. A message-passing GNN lives in $C^2$: the state of $v$ after $k$ layers is a function of its depth-$k$ unfolding tree (Xu et al. 2019; Morris et al. 2019), and when $v$ aggregates the multiset of its neighbours' states, nothing in those states records that part of them is $v$'s own signal returning two hops later. The known way past this is to *mark* one node $u$, run the same network, read out, and aggregate over every choice of $u$: identity-aware GNNs (You et al. 2021) and node-marking subgraph GNNs (Bevilacqua et al. 2022). The sum over $u$ makes them invariant, the mark lets them count cycles, and Frasca et al. (2022) prove that node-based subgraph GNNs are bounded by 3-WL, i.e. 2-FWL — the ceiling of Theorem 6 below. The particle model is precisely this construction, linear and unitary: the drive at $u$ is the mark, the intensities at every $v$ are the read-out, the average over slots is the aggregation, and the mark costs nothing because the photon from $u$ is marked by being that photon. Its readout is the return probability of a walk, the diagonal of $P^k$ for a random walker, which is also what the *random-walk structural encoding* adds to a GNN at linear cost (Dwivedi et al. 2022). So "beyond 1-WL yet classically simulable" is no paradox: the 1-WL bound of message passing is a bound on the *pattern* a layer can count, two variables, not on computational power, and marking a node and averaging over the marks escapes it at polynomial cost. Distinguishable photons, indistinguishable photons and classical incoherent light all do.

---

## 3. Theorem 2: single-detector statistics never see indistinguishability

**Theorem 2.** Let $p$ photons be injected into $p$ distinct slots $S$ of a passive linear-optical network with single-photon amplitudes $a_s$, and let $v$ be one output mode. The distribution of the count $n_v$ is determined, for bosons as for distinguishable photons, by the elementary symmetric polynomials $e_k := e_k\big(\{p_s(v)\}_{s \in S}\big)$, $k \le p$; explicitly, the factorial moments $n_v^{(k)} := n_v (n_v - 1) \cdots (n_v - k + 1)$ satisfy
$$\big\langle n_v^{(k)} \big\rangle_{\rm bosons} = (k!)^2\, e_k, \qquad \big\langle n_v^{(k)} \big\rangle_{\rm dist.} = k!\, e_k. \tag{3.1}$$
Hence the two per-node distributions are bijective functions of each other — for every slot set, every network and every $p$ — and any readout built from single-mode marginals carries identical graph information in the two settings.

*Proof.* Write $b_s^\dagger$ for the creation operator of slot $s$ and $c_v = \sum_j U_{vj} b_j$ for the annihilation operator of the sink $v$ in the Heisenberg picture, $U$ the spacetime unitary of which $M$ is a block. On the input $|\psi\rangle = \prod_{s \in S} b_s^\dagger |0\rangle$ only the occupied modes contribute, so $c_v |\psi\rangle = \sum_{s \in S} a_s(v)\, b_s |\psi\rangle$ and
$$c_v^{\,k} |\psi\rangle \;=\; k! \sum_{R \subseteq S,\ |R| = k}\ \prod_{s \in R} a_s(v)\ \prod_{s \in S \setminus R} b_s^\dagger |0\rangle,$$
the $k!$ counting the orderings in which the $k$ distinct photons are removed. The states indexed by different $R$ are orthogonal Fock states, so
$\langle n_v^{(k)} \rangle = \langle \psi | c_v^{\dagger k} c_v^{\,k} | \psi \rangle = \| c_v^{\,k} \psi \|^2 = (k!)^2 \sum_{|R| = k} \prod_{s \in R} p_s(v) = (k!)^2\, e_k$.
For distinguishable photons $n_v$ is a sum of independent Bernoulli variables of parameters $p_s(v)$, whose $k$-th factorial moment is $k!\, e_k$. A distribution on $\{0, \dots, p\}$ is determined by its factorial moments, and $e_1, \dots, e_p$ are recovered from either side of (3.1). $\square$

For $p = 2$ and the pair $\{s, s'\}$, with $p := p_s(v)$, $q := p_{s'}(v)$:

| | $P(n_v = 2)$ | $P(n_v = 1)$ | $P(n_v = 0)$ |
|---|---|---|---|
| distinguishable | $pq$ | $p + q - 2pq$ | $1 - p - q + pq$ |
| bosons | $2pq$ | $p + q - 4pq$ | $1 - p - q + 2pq$ |

so that $P_{\rm bos}(2) = 2 P_{\rm dist}(2)$ and $P_{\rm bos}(1) = P_{\rm dist}(1) - 2 P_{\rm dist}(2)$. The bosonic row is the Hong–Ou–Mandel bunching, $2pq$ against $pq$: a factor two, independent of the graph. Where does the cross term $2\,\mathrm{Re}\big(a_s(v)\overline{a_{s'}(v)}\, a_{s'}(j)\overline{a_s(j)}\big)$ of (1.2) go when the other photon's sink $j$ is summed out? Into $2\,\mathrm{Re}\big(a_s(v)\overline{a_{s'}(v)}\, \langle a_s | a_{s'} \rangle\big) = 0$ by (1.1). Interference between two photons is a *two-detector* phenomenon; a single detector integrates it away exactly because the two photons came from orthogonal modes.

**Corollary 2.1 (the $T \ge 3$ agreement).** The per-node bins of PR #70's `two-photon` and `distinguishable` certificates are the same graph invariant in two affine coordinate systems. Measured: the identities of the table hold to $5 \cdot 10^{-17}$ on every pair and every $T$ (Appendix, column *affine*). The separations differ slightly because the score is a maximum over differently scaled bins and because the total-count bins carry one more term, the subject of the next section; on the 1-WL-blind pairs the per-node bins carry the separation from $T = 3$ on, for both certificates alike.

---

## 4. Theorem 3: at two photons, the whole difference is one overlap

**Theorem 3.** For the pair $\{s, s'\}$ let $x_{ss'} := \sum_{v \in O} a_s(v)\, \overline{a_{s'}(v)} = (M_O M_O^\dagger)_{ss'}$, the overlap of the two photons' amplitudes *restricted to the detectors*. Then the joint count distributions on $O$ differ by
$$P_{\rm bos}(n_v = 1, n_w = 1) - P_{\rm dist}(n_v = 1, n_w = 1) = 2\,\mathrm{Re}\big(a_s(v)\overline{a_{s'}(v)}\; a_{s'}(w) \overline{a_s(w)}\big) \quad (v \ne w), \tag{4.1}$$
whose sum over $v, w$ is the total-count shift
$$P_{\rm bos}(N = 2) = P_{\rm dist}(N = 2) + |x_{ss'}|^2, \qquad P_{\rm bos}(N = 1) = P_{\rm dist}(N = 1) - 2|x_{ss'}|^2, \qquad P_{\rm bos}(N = 0) = P_{\rm dist}(N = 0) + |x_{ss'}|^2. \tag{4.2}$$
Averaged over the two-herald ensemble, the bosonic readout of PR #70 is the distinguishable readout, re-expressed through the table of §3, plus the single scalar
$$X \;:=\; \operatorname{mean}_{\{s, s'\}} |x_{ss'}|^2 \;=\; \frac{\|G\|_F^2 - \sum_s G_{ss}^2}{2 \cdot \#\text{pairs}}, \qquad G := M_O M_O^\dagger. \tag{4.3}$$

*Proof.* Expanding (1.2) at $(v, w)$ gives the direct terms, which are the distinguishable probability, plus the cross term of (4.1). Summing the cross term over $v, w \in O$ gives $2\,\mathrm{Re}\big(x_{ss'} \overline{x_{ss'}}\big) = 2|x_{ss'}|^2$, halved on the diagonal to $|x_{ss'}|^2$ for $N = 2$; over $v \in O$, $w \notin O$ it gives $2\,\mathrm{Re}\big(x_{ss'} \cdot \overline{(\sum_{w \notin O} a_s(w)\overline{a_{s'}(w)})}\big) = -2|x_{ss'}|^2$ by (1.1), for $N = 1$; and the remainder for $N = 0$. $\square$

Measured: (4.2) holds to $5 \cdot 10^{-16}$ on every row (Appendix, column *total*). So the two-photon bosonic and distinguishable QMapNNs, as read out by PR #70, differ in exactly one graph invariant, $X$: the mean squared Hong–Ou–Mandel overlap of two photons on the detectors, zero if and only if the sources' output amplitudes are pairwise orthogonal on $O$. Since $X \ge 0$, bosons always bunch in the total count. In the necklace language of PR #70's ceiling theorem, $|x_{ss'}|^2$ is a closed chain of four walks, $s \to v \to s' \to w \to s$, the exchange cycle; the classical statistics are chains of two.

---

## 5. Theorem 4: the same ceiling

**Theorem 4.** Every statistic of §3 and §4 — the per-node bins of either certificate, the total-count bins of either, and $X$ — is equal on 2-WL-equivalent graphs.

*Proof.* By the handout's Lemma B1 the blocks of $M$ between vertex-indexed channels lie in $\mathcal{W}(G)$, so $M \circ \overline{M}$, its Schur powers $(M \circ \overline{M})^{\circ k}$, their row sums $(M \circ \overline{M})^{\circ k} J$ and the Gram block $G = M_O M_O^\dagger$ are in $\mathcal{W}(G)$ block by block. The one-photon marginal $\sum_s p_s(v)$ is a diagonal entry of the first, the elementary symmetric polynomials of $\{p_s(v)\}_s$ are polynomials in the power sums $\sum_s p_s(v)^k$, diagonal entries of the Schur powers; the distinguishable total-count bins are symmetric polynomials in the diagonal $G_{ss}$, i.e. polynomials in $\operatorname{tr}\, G^{\circ k}$; and $X$ is the total entry sum of $G \circ \overline{G}$ minus its trace. Diagonal entries are constant on 2-WL vertex classes, and traces and total sums are preserved by the algebraic isomorphism $\varphi$ of Fact 1.3, which the handout's Lemma B3 transports with the same cell parameters. $\square$

This is the theorem behind PR #70's rook-versus-Shrikhande row being an exact zero for the distinguishable certificate as well as the bosonic one. Both models live in the window $(1\text{-WL}, 2\text{-WL}]$ of the handout; the per-node invariants they generate coincide (Theorem 2), and the extra invariant of bosons, $X$, does not leave the window (Theorem 4). Whether $X$ separates, at *some* depth, a pair that the classical statistics separate at *no* depth is open; §6 shows that it does at fixed depth.

---

## 6. Proposition 5: what indistinguishability buys is depth

**Proposition 5.** At $T = 2$, on a $d$-regular graph, every classical statistic (one-photon, distinguishable) is a function of $d$ alone, while
$$X \;=\; \alpha \;+\; \beta\, \operatorname{tr} A^3 \;+\; \gamma\, \operatorname{tr} A^4 \tag{6.1}$$
with $\alpha, \beta, \gamma$ depending on $d$ and the cell only, $\gamma \ne 0$ whenever the drive couples to the darts, and $\beta \ne 0$ only when the drive couples to the memory.

*Proof.* Two ticks leave three histories. The photon of slot $(u, 1)$ can only go straight through, $a_{(u,1)}(v) = c_3\, \delta_{uv}$ with $c_3 = W_{do}$. The photon of slot $(u, 0)$ either hops once — drive to the symmetric dart of $u$, out along every dart with weight $d^{-1/2}$, in at the neighbour $w$ with weight $d^{-1/2}$, out through $w$'s output — or dwells one tick in the memory of $u$ and exits there:
$$a_{(u,0)}(v) \;=\; \frac{c_1}{d}\, A_{vu} \;+\; c_2\, \delta_{uv}, \qquad c_1 = W_{ds} W_{so},\quad c_2 = W_{dm} W_{mo}.$$
Then $p_{(u,0)}(v) = |c_1|^2 A_{vu}/d^2 + |c_2|^2 \delta_{uv}$, the cross term vanishing because $A$ has no diagonal; so $\sum_s p_s(v)$, every power sum $\sum_s p_s(v)^k$ and every $G_{ss} = \sum_v p_s(v)$ is a function of $d$: this is the distinguishable and one-photon exact zero on regular pairs at $T = 2$ (all five 1-WL-blind pairs, all twenty seeds, in PR #70). The overlaps are
$$x_{(u,0)(u',0)} = \frac{|c_1|^2}{d^2} (A^2)_{uu'} + \frac{c_1 \overline{c_2} + \overline{c_1} c_2}{d}\, A_{uu'} + |c_2|^2 \delta_{uu'}, \qquad x_{(u,0)(u',1)} = \frac{c_1 \overline{c_3}}{d} A_{uu'} + c_2 \overline{c_3}\, \delta_{uu'},$$
and summing $|x|^2$ over pairs, the cross-tick pairs contribute degree data only, the same-tick pairs contribute $\frac{|c_1|^4}{d^4} \sum_{u \ne u'} (A^2)_{uu'}^2 = \frac{|c_1|^4}{d^4} \big(\operatorname{tr} A^4 - n d^2\big)$, the term $\frac{2 |c_1|^2\, \mathrm{Re}(c_1 \overline{c_2})}{d^3} \sum_{u \ne u'} (A^2)_{uu'} A_{uu'} = \frac{2 |c_1|^2\, \mathrm{Re}(c_1 \overline{c_2})}{d^3} \operatorname{tr} A^3$, and degree terms. $\square$

The closed-walk counts of the dataset's small pairs, and the measured $T = 2$ separations of the canonical cell and of a *memoryless* cell ($W$ a Haar unitary of the symmetric dart and the drive, the identity on the memory), confirm (6.1) term by term — extension-09 is not regular, so there the walk counts carry degree weights $f(D)$ as in the handout's Lemma B1, with the same parity structure:

| pair | $\operatorname{tr} A^3$ | $\operatorname{tr} A^4$ | bosons, canonical | distinguishable, canonical | bosons, memoryless |
|---|---|---|---|---|---|
| classic-2c3-vs-c6 | 12 vs 0 | 36 vs 36 | $1.1 \cdot 10^{-3}$ | exact 0 | exact 0 |
| regular-24 | 24 vs 12 | 136 vs 136 | $1.8 \cdot 10^{-4}$ | exact 0 | exact 0 |
| regular-47 | 36 vs 42 | 340 vs 308 | $9.7 \cdot 10^{-5}$ | exact 0 | $1.1 \cdot 10^{-5}$ |
| extension-09 | 0 vs 0 | 112 vs 128 | $5.3 \cdot 10^{-5}$ | exact 0 | $1.2 \cdot 10^{-5}$ |

Without memory, $\beta = 0$ and exactly the pairs with different $\operatorname{tr} A^4$ separate; with memory, the triangle term separates the two pairs whose 4-walks agree. On $2C_3$ versus $C_6$ the memoryless cell is an exact zero at *every* $T$ and for every certificate: the handout's Lemma 3.2 in the numbers, since all even power sums of the two spectra agree and only the memory mixes parities inside one amplitude.

*Why the classical statistics need a third tick.* A closed walk is assembled from the segments of a necklace, and in $T$ ticks each segment has at most $T - 1$ hops. The one-photon necklace $s \to v \to s$ has two segments, so at $T = 2$ it closes walks of length at most two, which are degrees; at $T = 3$ a two-hop segment against a one-hop-plus-dwell segment closes a triangle, the mechanism of the handout's Theorem A. The exchange necklace $s \to v \to s' \to w \to s$ has four segments and closes walks of length four at $T = 2$: a triangle with a dwell for parity, or a 4-cycle. Indistinguishability lets the walk close *through two detectors* instead of one, and so halves the depth at which a given cycle becomes visible — from $2(T-1)$ to $4(T-1)$ hops around the necklace. That, and nothing about which pairs are separable in the limit, is what the $T = 2$ column measures.

---

## 7. Theorem 6: where the two settings can still differ

**Theorem 6.** For any number $p$ of distinguishable photons injected in distinct slots, every counting statistic binned per node or per 2-WL pair class is equal on 2-WL-equivalent graphs.

*Proof.* For a slot set $S$ the joint distribution of the counts on $O$ is the coefficient extraction
$P(n) = [x^{n}] \prod_{s \in S} \big(q_s + \sum_{v \in O} x_v\, p_s(v)\big)$, $q_s := 1 - \sum_{v \in O} p_s(v) = 1 - G_{ss}$,
a polynomial in the $p_s(v)$ and $q_s$ that is symmetric under permuting $S$. Averaging over $p$-subsets $S$ of the slots and marginalising to one or two pinned nodes $v, w$ leaves a symmetric polynomial in the vectors $\big(p_s(v), p_s(w), q_s\big)_s$, which by inclusion–exclusion over set partitions is a polynomial in the contracted sums $\sum_s p_s(v)^a\, p_s(w)^b\, q_s^c$. Each such sum is the $(v, w)$ entry of $\big((M_O \circ \overline{M_O})^{\circ a}\big)^{\!\dagger}\, \mathrm{diag}(q)^c\, (M_O \circ \overline{M_O})^{\circ b}$, an element of $\mathcal{W}(G)$ since $q$ is a diagonal of one. Entries pinned at one node or one pair class are constant on 2-WL classes and transported by $\varphi$. $\square$

Bosons are covered by the same argument only as far as the handout's Theorem B goes: readouts of order $\le 2$, or collision-free post-selection at any $p$. A $p$-photon probability with bunching on two or more detectors is a squared permanent of a $p \times p$ matrix, a signed sum over necklaces of $2p$ walks whose index pattern can have treewidth $\Theta(p)$ (handout §5.1), and those are not $C^3$ invariants a priori. So the only place where the two settings can separate *different* pairs is bunched interference of three or more photons on multi-detector correlations; whether it actually separates a 2-WL-equivalent pair such as rook versus Shrikhande is open, the smallest test being three heralded photons read on the two-node coincidences with collisions kept. This is the same regime in which simulating the bosonic model stops being efficient — permanents of growing order, Aaronson–Arkhipov — while the distinguishable model stays a product of one-photon marginals at every $p$. Within the passive model, then, an expressivity advantage of indistinguishable photons over distinguishable ones, if it exists, lives exactly where their simulability advantage lives, and nowhere below.

---

## 8. Summary

| | coherent drive | distinguishable, any $p$ | bosons, order-$\le 2$ readout | bosons, bunched $p \ge 3$ |
|---|---|---|---|---|
| lower bound | $\le$ 1-WL | $>$ 1-WL | $>$ 1-WL | $>$ 1-WL |
| ceiling | 1-WL | 2-WL (Thm 6) | 2-WL (handout Thm B) | open, treewidth $\Theta(p)$ patterns |
| per-node readout | 1-WL data | same as bosons (Thm 2) | same as distinguishable (Thm 2) | same as distinguishable (Thm 2) |
| extra invariant over the column to its left | — | closed walks (Thm 1) | $X$, one HOM overlap (Thm 3) | permanents |
| smallest $T$ seeing a triangle | never | 3 | 2 (Prop 5) | 2 |
| exact simulation | $O(n)$ linear algebra | products of one-photon marginals | $2 \times 2$ permanents | #P-hard in general |

The distinguishable photonic GNN separates the same graphs as the indistinguishable one, on per-node readouts, because a detector only ever measures a source-resolved intensity, and it beats GNNs because that intensity is a closed walk. Indistinguishability contributes one further invariant per pair of sources, the overlap on the detectors, which shortens the depth at which cycles become visible but stays inside the same coherent algebra.

---

## Appendix: the numbers

Canonical cell of PR #70 ($\psi = \pi$, $W$ Haar from seed 0), two-herald ensemble over all slot pairs, read on the last tick; "exact 0" is at or below $10^{-15}$. *node* and *total* are the separations restricted to the per-node and to the total-count bins; *affine* is the largest violation of the table of §3, *total id.* of (4.2), on the first graph of the pair.

| pair | $T$ | bosons | distinguishable | bosons, node | dist., node | bosons, total | dist., total | $X_G - X_H$ | affine | total id. |
|---|---|---|---|---|---|---|---|---|---|---|
| control-c6-vs-p6 | 2 | 9.9e-03 | 1.1e-02 | 7.0e-03 | 7.3e-03 | 9.9e-03 | 1.1e-02 | −3.3e-04 | 1e-17 | 2e-16 |
| control-c6-vs-p6 | 3 | 1.3e-02 | 1.4e-02 | 8.3e-03 | 8.8e-03 | 1.3e-02 | 1.4e-02 | −3.7e-04 | 2e-17 | 5e-16 |
| control-c6-vs-p6 | 4 | 4.7e-03 | 4.9e-03 | 2.6e-03 | 2.8e-03 | 4.7e-03 | 4.9e-03 | −8.4e-05 | 5e-17 | 5e-16 |
| regular-24 | 2 | 1.8e-04 | exact 0 | exact 0 | exact 0 | 1.8e-04 | exact 0 | −9.0e-05 | 7e-18 | 3e-17 |
| regular-24 | 3 | 8.3e-03 | 8.5e-03 | 1.9e-03 | 2.0e-03 | 8.3e-03 | 8.5e-03 | +9.6e-05 | 3e-17 | 4e-16 |
| regular-24 | 4 | 8.8e-03 | 9.2e-03 | 2.4e-03 | 2.4e-03 | 8.8e-03 | 9.2e-03 | −2.2e-04 | 1e-17 | 3e-16 |
| regular-47 | 2 | 9.7e-05 | exact 0 | exact 0 | exact 0 | 9.7e-05 | exact 0 | +4.9e-05 | 1e-17 | 3e-17 |
| regular-47 | 3 | 2.8e-04 | 3.0e-04 | 2.8e-04 | 3.0e-04 | 1.5e-04 | 1.6e-04 | −4.7e-06 | 2e-17 | 2e-17 |
| regular-47 | 4 | 2.4e-03 | 2.5e-03 | 1.2e-03 | 1.3e-03 | 2.4e-03 | 2.5e-03 | +3.5e-05 | 3e-17 | 7e-17 |
| extension-09 | 2 | 5.3e-05 | exact 0 | exact 0 | exact 0 | 5.3e-05 | exact 0 | −2.7e-05 | 7e-18 | 1e-16 |
| extension-09 | 3 | 3.0e-03 | 3.1e-03 | 5.4e-04 | 5.6e-04 | 3.0e-03 | 3.1e-03 | −5.1e-05 | 1e-17 | 6e-17 |
| extension-09 | 4 | 1.5e-03 | 1.6e-03 | 4.0e-04 | 4.1e-04 | 1.5e-03 | 1.6e-03 | −5.7e-05 | 3e-17 | 5e-16 |
| classic-2c3-vs-c6 | 2 | 1.1e-03 | exact 0 | exact 0 | exact 0 | 1.1e-03 | exact 0 | −5.5e-04 | 1e-17 | 6e-17 |
| classic-2c3-vs-c6 | 3 | 3.4e-02 | 3.6e-02 | 8.4e-03 | 8.8e-03 | 3.4e-02 | 3.6e-02 | +1.0e-03 | 4e-17 | 2e-16 |
| classic-2c3-vs-c6 | 4 | 2.2e-02 | 2.4e-02 | 5.4e-03 | 5.8e-03 | 2.2e-02 | 2.4e-02 | −1.1e-03 | 3e-17 | 2e-16 |

The *bosons* and *distinguishable* columns are PR #70's. At $T = 2$ on every 1-WL-blind pair the per-node bins of *both* certificates are exact zeros and the whole bosonic separation is $2X$ in the $N = 1$ bin, as (4.2) and (6.1) say. The rows were produced with PR #70's `engine.py` on a step matrix assembled directly from `cells.invariant_cell`, wires ordered drives first, then per vertex its darts and memory:

```python
slots = engine.amplitudes(step, n, T)              # one row per slot (u, t)
flat = slots.reshape(-1, slots.shape[-1])
measured = (T - 1) * n + np.arange(n)              # the last tick's outputs
pairs = list(combinations(flat, 2))
bos = engine.count_statistics(pairs, measured, engine.bosons)
dis = engine.count_statistics(pairs, measured, engine.distinguishable)
G = flat[:, measured] @ flat[:, measured].conj().T  # the Gram matrix M_O M_O^dagger
X = (np.abs(G) ** 2).sum() - (np.abs(np.diag(G)) ** 2).sum()
X /= 2 * len(pairs)
assert np.allclose(bos[0][:, 1], dis[0][:, 1] - 2 * dis[0][:, 2])   # section 3
assert np.allclose(bos[0][:, 2], 2 * dis[0][:, 2])
assert np.allclose(bos[1][2], dis[1][2] + X)                        # (4.2)
assert np.allclose(bos[1][1], dis[1][1] - 2 * X)
```

The memoryless cell of §6 is `invariant_cell(d, W, pi)` with `W` the direct sum of a Haar unitary of $U(2)$ on the symmetric dart and the drive (seed 1) and $1$ on the memory.

---

## References

- Z. Dvořák, *On recognizing graphs by numbers of homomorphisms* (2010); H. Dell, M. Grohe, G. Rattan, *Lovász meets Weisfeiler and Leman* (2018) — tree and treewidth-2 homomorphism counts as 1-WL and 2-WL invariants.
- K. Xu, W. Hu, J. Leskovec, S. Jegelka, *How powerful are graph neural networks?* (ICLR 2019); C. Morris et al., *Weisfeiler and Leman go neural* (AAAI 2019) — the 1-WL bound of message passing.
- N. Immerman, E. Lander, *Describing graphs: a first-order approach to graph canonization* (1990); J.-Y. Cai, M. Fürer, N. Immerman, *An optimal lower bound on the number of variables for graph identification* (1992) — $k$-WL as the counting logic $C^{k+1}$.
- J. You, J. Gomes-Selman, R. Ying, J. Leskovec, *Identity-aware graph neural networks* (AAAI 2021); B. Bevilacqua et al., *Equivariant subgraph aggregation networks* (ICLR 2022); F. Frasca, B. Bevilacqua, M. Bronstein, H. Maron, *Understanding and extending subgraph GNNs by rethinking their symmetries* (NeurIPS 2022) — node marking, its invariance by aggregation over marks, and its 3-WL ceiling.
- V. P. Dwivedi, A. T. Luu, T. Laurent, Y. Bengio, X. Bresson, *Graph neural networks with learnable structural and positional representations* (ICLR 2022) — random-walk return probabilities as node features.
- C. K. Hong, Z. Y. Ou, L. Mandel, *Measurement of subpicosecond time intervals between two photons by interference* (PRL 1987) — the two-photon overlap term.
- S. Aaronson, A. Arkhipov, *The computational complexity of linear optics* (2011) — hardness of bunched multi-photon statistics.
- J. K. Gamble, M. Friesen, D. Zhou, R. Joynt, S. N. Coppersmith, *Two-particle quantum walks applied to the graph isomorphism problem* (PRA 81, 052313, 2010) — non-interacting two-particle walks and strongly regular graphs; J. Smith, *Algebraic aspects of multi-particle quantum walks*, as cited in [PR #61](https://github.com/rel-int/optyx/pull/61) — non-interacting walk invariants factor through the cellular algebra.
- [PR #58](https://github.com/rel-int/optyx/pull/58), [PR #61](https://github.com/rel-int/optyx/pull/61), [PR #69](https://github.com/rel-int/optyx/pull/69), [PR #70](https://github.com/rel-int/optyx/pull/70) — the experiments; [`plo_gnn_wl_theorems.md`](plo_gnn_wl_theorems.md) — the model, Theorem A and the 2-WL ceiling.
