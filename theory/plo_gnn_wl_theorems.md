# The Weisfeiler–Leman window of passive linear-optical graph neural networks

**Claim.** Passive linear-optical GNNs (defined below, matching the invariant-cell QMapNN of the experiment handout) distinguish some pairs of non-isomorphic graphs that 1-WL cannot distinguish (Theorem A, using the memory mode), and distinguish no pair of graphs that 2-WL cannot distinguish (Theorem B, for readouts of counting-correlation order ≤ 2 — a property of the readout, not of the number of photons injected). Hence their distinguishing power lies strictly inside the window (1-WL, 2-WL].

**Convention.** Throughout, "2-WL" means the *folklore* 2-dimensional Weisfeiler–Leman refinement (2-FWL), equivalently the 3-variable counting logic C³, equivalently the tuple "3-WL" of the GNN literature, equivalently equality of coherent configurations in the sense recalled in §1. "1-WL" is ordinary colour refinement. All graphs are finite, simple, undirected; $A = A(G)$ is the adjacency matrix, $D = D(G)$ the diagonal degree matrix, $n = |V|$.

---

## 1. Preliminaries: the coherent algebra and 2-WL

**Definition 1.1 (coherent algebra).** The *coherent algebra* (cellular algebra, Weisfeiler–Leman closure) $\mathcal{W}(G) \subseteq \mathbb{C}^{V\times V}$ is the smallest set of matrices containing $A$, $I$, and the all-ones matrix $J$, closed under matrix product, Schur (entrywise) product $\circ$, conjugate transpose, and linear combinations.

**Fact 1.2 (Weisfeiler–Leman 1968; Higman).** $\mathcal{W}(G)$ has a unique basis of 0/1 matrices $\{J_R\}_{R \in \mathcal{R}}$ whose supports partition $V\times V$; the partition $\mathcal{R}$ is exactly the stable pair-colouring computed by 2-WL on $G$. Consequently **every element of $\mathcal{W}(G)$ has entries constant on each class $R$**; in particular the diagonal classes partition $V$ into the stable 2-WL vertex classes, and every element of $\mathcal{W}(G)$ has constant diagonal on each vertex class.

**Fact 1.3 (algebraic isomorphism; Weisfeiler–Leman 1968, Immerman–Lander 1990, Cai–Fürer–Immerman 1992).** Two graphs $G, H$ are 2-WL-equivalent iff there is an *algebraic isomorphism* $\varphi: \mathcal{W}(G) \to \mathcal{W}(H)$: a linear bijection with
$\varphi(XY) = \varphi(X)\varphi(Y)$, $\varphi(X \circ Y) = \varphi(X)\circ\varphi(Y)$, $\varphi(X^{\dagger}) = \varphi(X)^{\dagger}$, and $\varphi(A_G) = A_H$, $\varphi(I) = I$, $\varphi(J) = J$. Moreover $\varphi$ maps basis classes to basis classes with equal sizes and equal intersection numbers; hence it preserves traces, total entry-sums ($\operatorname{tr}(XJ)$), and the correspondence of diagonal (vertex) classes with their sizes.

**Lemma 1.4 (degree functions transport).** $D \in \mathcal{W}(G)$, since $D = (AJ)\circ I$. For any function $f:\mathbb{R}\to\mathbb{C}$, the matrix $f(D)$ (applied to diagonal entries) lies in $\mathcal{W}(G)$ and satisfies $\varphi(f(D_G)) = f(D_H)$.

*Proof.* $D = (AJ)\circ I \in \mathcal{W}(G)$ by closure. $D$ is a linear combination of diagonal class idempotents $E_c$ with coefficients the (class-constant) degrees $d_c$; $f(D) = \sum_c f(d_c)E_c \in \mathcal{W}(G)$. Since $\varphi(D_G) = (\varphi(A)\varphi(J))\circ \varphi(I) = (A_H J)\circ I = D_H$ and $\varphi$ maps $E_c$ to the matched idempotent, matched classes have equal degrees, so $\varphi(f(D_G)) = \sum_c f(d_c)\varphi(E_c) = f(D_H)$. $\square$

This lemma is what makes the $d^{-1/2}$ amplitudes of balanced multiports harmless: they are class functions.

**Fact 1.5 (Dvořák 2010; Dell–Grohe–Rattan 2018).** $G \equiv_{1\text{-WL}} H$ iff $\hom(F,G)=\hom(F,H)$ for all trees $F$; $G \equiv_{2\text{-WL}} H$ iff the same holds for all $F$ of treewidth $\le 2$. (Used only for remarks; the proofs below are algebraic.)

---

## 2. The model

**Definition 2.1 (PLO-GNN).** Fix $T \in \mathbb{N}$, an ancilla list (here: memory, drive, output), and parameters $\theta = (W \in \mathrm{U}(3),\, \psi \in \mathbb{R})$, shared across all nodes and all ticks. Given a graph $G$:

*Modes.* For each arc (ordered adjacent pair) $(u\to v)$ one optical mode; for each node $v$ one memory mode, and at each tick one drive mode and one output mode.

*Cell.* At node $v$ of degree $d$, the cell is the unitary
$$U_v \;=\; B_v^{\dagger}\,\bigl(W \oplus e^{i\psi} I_{d-1}\bigr)\,B_v$$
on (in-arcs of $v$) $\oplus$ (memory, drive) $\to$ (out-arcs of $v$) $\oplus$ (memory, output), where $B_v$ is any fixed balanced multiport sending the symmetric arc vector $s_v = d^{-1/2}\sum_{u\sim v} |u\to v\rangle$ to a designated port, and $W$ acts on {symmetric port, memory, drive/output}. By Schur's lemma this is the *general* form of a cell commuting with the diagonal action of $S_d$ on in- and out-arcs; the construction is independent of the arc ordering used inside $B_v$ (conjugating $B_v$ by a permutation conjugates $U_v$ by it, and $U_v$ commutes with it).

*Dynamics.* One tick applies all cells (they act on disjoint modes) followed by the arc reversal $\alpha: (v\to w) \mapsto (w\to v)$ on arc modes. A single photon is injected into every drive mode $(v,t)$ in an *injection schedule* $\mathcal{I} \subseteq V \times \{0,\dots,T-1\}$ that is a union of sets $V \times \{t\}$ (equivariant by fiat). Output modes are discarded at ticks $< T$.

*Readout.* At tick $T$, photon counting on the $n$ output modes. The model's output is the multiset over $v \in V$ of the per-node counting data of order $\le 2$: the mean counts $\langle n_v\rangle$ and the normally ordered pair correlations $\langle{:}n_v n_w{:}\rangle$, the latter binned by the 2-WL pair class of $(v,w)$ (any coarser binning, e.g. the total coincidence rate, is a fortiori covered).

All amplitudes are entries of one *spacetime transfer matrix* $M$, the product of the $T$ tick unitaries restricted to the relevant source and sink modes. Since photons are injected in pairwise orthogonal modes and the dynamics is passive, standard linear-optics identities give, for sources $s = (u,t) \in \mathcal{I}$:
$$\langle n_o\rangle = \sum_{s} |M_{o s}|^2, \qquad
\langle{:}n_o n_{o'}{:}\rangle = \sum_{s \ne s'} \tfrac{1}{1+\delta_{oo'}}\bigl|\mathrm{per}\,M[\{s,s'\},\{o,o'\}]\bigr|^2 . \tag{2.1}$$

---

## 3. Theorem A: separation beyond 1-WL

**Theorem A.** Let $G = C_6$ and $H = C_3 \sqcup C_3$. Then $G \equiv_{1\text{-WL}} H$, and there is a PLO-GNN whose readouts differ on $G$ and $H$. Explicitly: with the memory mode coupled, injections at every tick, and the two-phase point $\psi=\pi$, the per-node mean count satisfies $\langle n_v\rangle_H - \langle n_v\rangle_G = c\,\varepsilon^4\mu^2 + O(\varepsilon^4\mu^3, \varepsilon^5)$ with an explicit constant $c>0$, where $\varepsilon$ is the drive/output coupling and $\mu$ the memory coupling. Hence the readouts differ for all sufficiently small $\varepsilon,\mu>0$.

**Step 0 (1-WL equivalence and the even-sector coincidence).** Both graphs are 2-regular, so the uniform colouring is stable for colour refinement on each; 1-WL does not distinguish them. Their spectra are $\{2,1,1,-1,-1,-2\}$ and $\{2,2,-1,-1,-1,-1\}$: equal as multisets of absolute values, so **all even power sums agree while odd power sums differ** ($\sum\lambda^3 = 0$ vs $12$). The structural reason is that $C_6$ is the bipartite double cover of $C_3$: $A(C_6)^2 = A(2C_3)+2I$ up to relabelling, so the entire even-walk sector of the two graphs coincides. Consequently any statistic built from *parity-pure* amplitudes (all hop counts of one parity within a single amplitude) is blind on this pair, because squaring a parity-pure amplitude yields even walk data only — see Lemma 3.2. Separation must come from **parity mixing inside a single source amplitude**, which is what the memory mode provides.

**Step 1 (instance).** Take $W\in\mathrm{U}(3)$ real orthogonal with $\langle s|W|s\rangle = \tau$, drive/output couplings $\langle s|W|d\rangle = \langle o|W|s\rangle = \varepsilon$, memory couplings $\langle m|W|s\rangle = -\langle s|W|m\rangle = \mu$, $\langle m|W|m\rangle = 1-O(\mu^2)$, and $\eta := e^{i\psi} = -1$. The arc-block of the cell is $\tau P_{\rm sym} + \eta P_\perp$ with entries $c^v_{uw} = \eta\delta_{uw} + (\tau-\eta)/d_v$; at $d_v=2$, $\eta=-1$, $\tau = 1-O(\varepsilon^2+\mu^2)$ this is the non-backtracking (Grover) coin up to $O(\varepsilon^2+\mu^2)$.

**Step 2 (node-space recursion).** With $S_k(v)$ the total inflow to $v$ and $m_k(v)$ the memory amplitude, expanding the cell gives (cf. Lemma B1)
$$S_{k+1} = (\tau-\eta)AD^{-1}S_k + \eta\tau S_{k-1} - \mu\,AD^{-1/2}m_k + \varepsilon\,AD^{-1/2}\sigma_k,\qquad m_{k+1} = \mu\,D^{-1/2}S_k + (1-O(\mu^2))\,m_k,$$
with $\sigma_k$ the injection at tick $k$, and output amplitude $\varepsilon\,d_v^{-1/2}S_{k}(v)$ read at tick $T$. On 2-regular graphs at leading order the hop dynamics is the Chebyshev recursion $S_{k+1} = AS_k - S_{k-1}$. A photon injected at $(u,t)$ therefore reaches $v$'s output with amplitude
$$M_{v,(u,t)} = \tfrac{\varepsilon^2}{2}\Bigl[\,q_h(A) \;-\; \mu^2 \!\!\sum_{a+b = h-1}\!\! q_a(A)\,\tfrac{1}{2}A\,q_b(A) \;+\; O(\mu^3)\Bigr]_{vu} + O(\varepsilon^3),\qquad h := T-t-2, \tag{3.3}$$
where $q_h$ is the degree-$h$ Chebyshev-type polynomial of the hop recursion and the $\mu^2$ term collects paths with exactly one dwell (enter memory at some point, wait one tick, re-emit symmetrically — the factor $\tfrac12 A$ — and continue). The bracket mixes degrees $h$ and $h-1$: **parity mixing within one source.**

**Lemma 3.2 (parity purity $\Rightarrow$ blindness).** If $\mu = 0$, every amplitude $M_{v,(u,t)}$ is a polynomial in $A$ of fixed parity, and then *all* order-$\le 2$ statistics of §2 coincide on $G$ and $H$: means reduce to $(q_h(A)^2)_{vv}$; per-pair exchange terms factorise as $(q_h(A)^2)_{vw}(q_{h'}(A)^2)_{vw}$; fully aggregated exchange terms are $|(q_hq_{h'}(A))_{uu'}|^2$ summed, i.e. $\operatorname{tr}[(q_hq_{h'})(A)^2]$ — in every case even polynomials evaluated on spectra with equal absolute values, on vertex-transitive graphs. So without memory the model is blind on this pair at every readout order. $\square$

**Step 3 (the separating mean count).** By (2.1), $\langle n_v\rangle = \sum_{(u,t)}|M_{v,(u,t)}|^2$. Inserting (3.3), the $\mu^0$ terms are $\tfrac{\varepsilon^4}{4}(q_h(A)^2)_{vv}$ — even, equal on $G$ and $H$. The $\mu^2$ cross terms are
$$-\tfrac{\varepsilon^4\mu^2}{2}\sum_{a+b=h-1}\Bigl(q_h(A)\,q_a(A)\,\tfrac12 A\,q_b(A)\Bigr)_{vv} + O(\varepsilon^4\mu^3),$$
a polynomial in $A$ of total degree $h + (h-1) + 1 = 2h$ but of **odd parity** in the relevant sense: every monomial has odd degree, since $q_h$ has parity $h$, $q_aq_b$ parity $h-1$, and one extra factor $A$. On $G = C_6$, bipartite, every odd power of $A$ has zero diagonal, so this term vanishes identically. On $H = 2C_3$, vertex-transitive with spectrum $\{2,-1,-1\}$ per component, the $v$-diagonal of a polynomial $p(A)$ equals $\tfrac13\sum_{\lambda\in\{2,-1,-1\}}p(\lambda)$; with $q_k(2) = k+1$ and $|q_k(-1)|\le 1$,
$$\Bigl(q_h q_a \tfrac12 A\, q_b\Bigr)(A)_{vv} \;\ge\; \tfrac13\bigl[(h{+}1)(a{+}1)(b{+}1) - 2\cdot\tfrac12\bigr] \;>\; 0 \quad\text{for } h\ge 2.$$
All contributions carry the same sign (the common prefactor $-\tfrac{\varepsilon^4\mu^2}{2}$ times a positive number), so no cancellation occurs across $a+b=h-1$ or across injection ticks; the parity-even remainders and all normalisation corrections are even polynomials and cancel between $G$ and $H$. Thus $\langle n_v\rangle_H - \langle n_v\rangle_G = c\,\varepsilon^4\mu^2 + O(\varepsilon^4\mu^3,\varepsilon^5)$ with $c \ne 0$ (explicitly $c<0$ with this sign convention; flipping the sign of $\mu$ or $\eta$-phases flips it, and only $c\ne 0$ is needed). The two sides are polynomials in $(\varepsilon,\mu)$ (finite $T$, finite mode count) differing in an exhibited coefficient, hence differ for all small $\varepsilon,\mu>0$. $\blacksquare$

**Physical reading.** The memory mode lets a single photon's own two histories — $h$ hops, or $h-1$ hops plus a dwell — interfere at the detector. Their relative parity turns the Born-rule product into an *odd* closed-walk count, i.e. triangle data, invisible to 1-WL (tree homomorphism counts, Fact 1.5). Two-photon exchange terms are not needed for this pair; they matter for pairs whose even sectors differ across sources but not within them. Note also that square-then-sum, $\sum_s|M_{vs}|^2$, is available to *classical incoherent* light (independent thermal sources), so this particular separation is a wave-optics resource; the strictly nonclassical contribution of Fock inputs appears only at order 2, as the single-source exclusion term — see §5.2.

---

## 4. Theorem B: the 2-WL ceiling

**Theorem B.** Let $G \equiv_{2\text{-WL}} H$. Then every PLO-GNN of Definition 2.1 produces identical readouts on $G$ and $H$ — identical multisets over vertex classes of all order-$\le 2$ counting statistics (2.1), for every parameter value, every $T$, and every equivariant injection schedule.

The proof has three lemmas: the one-photon transfer matrix lives in the coherent algebra (B1); order-$\le 2$ statistics are generated from it by operations under which the coherent algebra is closed (B2); an algebraic isomorphism transports everything (B3).

**Lemma B1 (transfer blocks lie in $\mathcal{W}(G)$).** For every pair of channels $a, b \in \{\text{sym-inflow}, \text{memory}, \text{drive}, \text{output}\}$ and ticks $t \le t'$, the $V\times V$ block $M^{(b,t'),(a,t)} := \bigl(\langle w, b, t'|\,M\,|v, a, t\rangle\bigr)_{w,v}$ lies in $\mathcal{A}(G) := $ the algebra generated by $A$ and all diagonal matrices constant on 2-WL vertex classes, under matrix product and linear combination. In particular $\mathcal{A}(G) \subseteq \mathcal{W}(G)$.

*Proof.* Generalise the recursion (3.2) to full couplings. With $S_k(v)$ the symmetric inflow, $m_k(v)$ the memory amplitude, and sources $\sigma_k(v)$ from the schedule, expanding the cell $B^{\dagger}(W \oplus \eta I)B$ exactly as in Step 2 of Theorem A gives the closed linear system
$$\begin{aligned}
S_{k+1} &= (\tau-\eta)\,A D^{-1} S_k \;+\; \eta\tau\,S_{k-1} \;+\; W_{sm}\,A D^{-1/2}\, m_k \;+\; W_{sd}\, A D^{-1/2}\,\sigma_k \;+\;(\eta\text{-corrections of the same shape}),\\
m_{k+1} &= W_{ms}\,D^{-1/2} S_k + W_{mm}\, m_k + W_{md}\,\sigma_k, \qquad
(\text{output amplitude at } k{+}1) \;=\; W_{os}\,D^{-1/2}S_k + W_{om}\,m_k + W_{od}\,\sigma_k,
\end{aligned}$$
where every coefficient matrix is of the form $\gamma\, A f(D)$ or $\gamma f(D)$ with $\gamma$ a parameter scalar and $f$ a function of the degree. (The derivation is the same two lines as in §3: the cell's arc-block entries (3.1) depend only on the backtracking flag $\delta_{uw}$ and on $d_v$; summing over in-arcs turns the flag into the second-order memory term $S_{k-1}$, via the reversal $\alpha$, and the degree factors into $f(D)$'s.) Every coefficient matrix lies in $\mathcal{A}(G)$: $A \in \mathcal{A}(G)$ trivially, and $f(D) \in \mathcal{A}(G)$ because **degree is determined by the 1-WL colour, which is refined by the 2-WL vertex class** (Lemma 1.4). Iterating the system $T$ times expresses every block of $M$ as a polynomial in these coefficient matrices: a member of $\mathcal{A}(G)$. Finally $\mathcal{A}(G) \subseteq \mathcal{W}(G)$ since $\mathcal{W}(G)$ contains the generators and is closed under products. $\square$

*Remark.* The second-order recursion is the quantum analogue of the Bass–Hashimoto identity for non-backtracking walks; the two-phase (Schur-lemma) structure of the invariant cell is precisely what closes the dynamics in node space. This is where cell invariance earns its keep in the proof: a $\sigma$-dependent cell has no node-space compression, and indeed no graph-level statement could be made for it.

**Lemma B2 (order-$\le 2$ statistics are coherent-algebra data).** Every statistic in (2.1), binned per node or per 2-WL pair class, is of the form: an entry pattern of a matrix $X \in \mathcal{W}(G)$ that is *constant on 2-WL classes*, together with class sizes. Specifically:

1. *Means.* $\langle n_v \rangle = \sum_{s\in\mathcal{I}} (M \circ \bar M)_{(v,\mathrm{out},T),\,s}$. For each injection layer $V\times\{t\}$, the contribution is the $v$-diagonal entry of $\bigl(M^{(t)} \circ \bar M^{(t)}\bigr) J_c$-type products with class idempotents, where $M^{(t)}$ is the relevant block from Lemma B1. Schur products and products with $J$ and class idempotents stay in $\mathcal{W}(G)$ (Definition 1.1), so $\langle n_v\rangle$ is a diagonal entry of an element of $\mathcal{W}(G)$ — hence constant on 2-WL vertex classes (Fact 1.2).
2. *Coincidences.* Expanding $|\mathrm{per}\,M[\{s,s'\},\{o,o'\}]|^2$ gives direct terms $P_{os}P_{o's'} + P_{os'}P_{o's}$ with $P = M\circ\bar M \in \mathcal{W}(G)$ (per block), and exchange terms $2\,\mathrm{Re}\bigl[M_{os}M_{o's'}\bar M_{os'}\bar M_{o's}\bigr]$. Summing exchange terms over a source layer: $\sum_{s \in V\times\{t\}} M_{os}\bar M_{o's} = \bigl(M^{(t)} {M^{(t)}}^{\dagger}\bigr)_{oo'}$, an entry of a product of $\mathcal{W}(G)$ elements; the remaining factor likewise; their Schur product likewise. Bunched terms ($o = o'$) are the special case handled by the same closure. So every per-pair-class binned coincidence is an entry pattern (constant on pair classes, by Fact 1.2) of an element of $\mathcal{W}(G)$.

All constructions used — matrix product, Schur product, conjugate transpose, multiplication by $I, J$, class idempotents, and scalars $f(d_c)$ — are the closure operations of Definition 1.1 together with Lemma 1.4. $\square$

**Lemma B3 (transport).** Let $\varphi: \mathcal{W}(G) \to \mathcal{W}(H)$ be the algebraic isomorphism of Fact 1.3. Then $\varphi$ maps each generator used in B1–B2 to its $H$-counterpart: $A_G \mapsto A_H$, $I \mapsto I$, $J \mapsto J$, class idempotents to matched class idempotents, and $f(D_G)\mapsto f(D_H)$ (Lemma 1.4) — with the *same scalar parameters* $(\tau, \eta, W, \varepsilon)$, since parameters are graph-independent. As $\varphi$ commutes with every closure operation, it maps the element of $\mathcal{W}(G)$ representing any given statistic to the element of $\mathcal{W}(H)$ representing the same statistic. Since $\varphi$ matches classes with equal sizes and elements of coherent algebras are constant on classes, the per-class values and multiplicities agree; hence the multiset-over-vertices (resp. pair-class-binned) readouts agree. $\blacksquare$

**Corollary B.4 (strictness at the top).** The $4\times4$ rook's graph and the Shrikhande graph are 2-WL-equivalent, hence no PLO-GNN readout of order $\le 2$ separates them; since they are non-isomorphic, the model class is strictly weaker than graph isomorphism. Together with Theorem A: the distinguishing power of order-$\le2$ PLO-GNNs lies strictly inside $(1\text{-WL},\, 2\text{-WL}]$.

---

## 5. Remarks on scope and sharpness

**5.1 Why order $\le 2$.** In tensor-network terms, the coherent algebra's two products are exactly series (matrix product) and parallel (Schur product) composition: the closure of series–parallel contractions, equivalently treewidth-$\le 2$ patterns, matching Fact 1.5. Order-$\le2$ counting statistics only ever create series–parallel index patterns (Lemma B2's expansions), which is *why* the algebra route closes. Order-$p$ statistics with $p \ge 3$ and bunching produce cyclic index contractions on $\ge 6$ free indices (e.g. $\sum_{i,j,k} M_{ai}M_{bj}M_{ck}\bar M_{aj}\bar M_{bk}\bar M_{ci}$) that are not series–parallel; the corresponding patterns can reach treewidth 3 and are **not** covered by Theorem B. Collision-free $p$-photon statistics with all sources in distinct modes, by contrast, decompose into disjoint exchange cycles and remain series–parallel — so the ceiling extends to arbitrary photon number *for collision-free post-selection*, and the only currently open escape within passive optics is genuinely bunched higher-order interference.

**5.2 What is quantum here — three illuminations.** Keep the wiring and change the light. (i) *Mutually coherent* uniform illumination: fields add, then square, giving $|(M\mathbf 1)_v|^2$ — walk counts from uniform sources, 1-WL data. (ii) *Mutually incoherent classical* sources (independent thermal emitters): intensities add, giving $\langle n_v\rangle = (MNM^\dagger)_{vv}$ — closed-walk data, already beyond 1-WL (Theorem A's mechanism works verbatim, since it uses only $\sum_s|M_{vs}|^2$); this is classical partial coherence, and at order 1 single photons give exactly the same statistics. (iii) *Fock inputs*: at order 2, thermal light yields the Hanbury Brown–Twiss term $+|G^{(1)}_{vw}|^2$, photons yield $+|(MM^\dagger)_{vw}|^2 - \sum_s|M_{vs}|^2|M_{ws}|^2$; the difference is the single-source exclusion (antibunching, $g^{(2)}<1$) term, which no classical field — no state with non-negative Glauber–Sudarshan $P$-function — can produce. Both (ii) and (iii) are bounded by 2-WL by Theorem B's argument. Hence, within the passive model, the *strictly quantum* contribution to graph discrimination is exactly the exclusion term; by Spekkens' equivalence of quasiprobability negativity and generalized contextuality for prepare-and-measure fragments, it is a preparation-contextuality resource (not a Kochen–Specker or Bell witness: the readout is a single measurement context). Whether the exclusion term separates graph pairs that thermal-light order-2 statistics do not is open; it is a Schur-square datum ($M\circ\bar M$ entries) not determined by $G^{(1)}$ alone, so the natural conjecture is yes.

**5.3 Consistency.** Theorem A's instance is inside Theorem B's model class and its separating statistic is order 1; the pair $(C_6, 2C_3)$ is 2-WL-distinguishable, so there is no tension.

**5.4 What is not claimed.** No completeness at any level is claimed: the theorems bracket the model strictly above 1-WL and at most 2-WL; whether every 2-WL-distinguishable pair is separated at generic parameters is open (the natural conjecture, by analogy with generic-parameter WL-matching results, is yes for the exchange-rich readout, and the BREC 1-WL-blind rung of the accompanying experiment is the empirical probe of exactly this).

---

## References

- B. Weisfeiler, A. Leman, *The reduction of a graph to canonical form and the algebra which appears therein* (1968).
- D. G. Higman, *Coherent configurations I* (1975).
- N. Immerman, E. Lander, *Describing graphs: a first-order approach to graph canonization* (1990).
- J.-Y. Cai, M. Fürer, N. Immerman, *An optimal lower bound on the number of variables for graph identification* (1992).
- Z. Dvořák, *On recognizing graphs by numbers of homomorphisms* (2010).
- H. Dell, M. Grohe, G. Rattan, *Lovász meets Weisfeiler and Leman* (2018).
- H. Bass, *The Ihara–Selberg zeta function of a tree lattice* (1992); H. Stark, A. Terras, *Zeta functions of finite graphs and coverings* (1996) — for the non-backtracking recursion.
- M. Fürer, *On the power of combinatorial and spectral invariants* (2010) — for spectral consequences of 2-WL equivalence.
- D. Emms, S. Severini, R. Wilson, E. Hancock, *Coined quantum walks lift the cospectrality of graphs and trees* (2006) — related walk invariants, for context.
- R. W. Spekkens, *Negativity and contextuality are equivalent notions of nonclassicality* (2008); C. Ferrie, J. Emerson, *Frame representations of quantum mechanics and the necessity of negativity in quasi-probability representations* (2008) — for §5.2.
- L. Mandel, E. Wolf, *Optical Coherence and Quantum Optics* (1995) — partial coherence, HBT, antibunching and the $P$-representation.
