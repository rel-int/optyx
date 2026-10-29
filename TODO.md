> The architectures can also have non linearity in the form of feedforward from the measurement outcomes at each tick to a (possibly delayed) phase modulation on the chip. The samplers are able to simulate this behaviour for distinguishable and indistinguishable photons, I believe coherent and thermal also should be doable. Open a new PR in discopy-next/optyx porting the simulation code in the wiki to have simulation of arbitrary photonic optyx diagrams (no qubits but classical feedforward allowed) with the different sources. Then implement equilibrium propagation on them and see test if it approximates the gradient.

> Split this PR into two. First the samplers for the different light sources, then the implementation of equilibrium propagation on photonic architectures

---

The second half, equilibrium propagation, stacked on #94, the samplers.

- [x] `optyx/equilibrium.py`: the stationary means of a family of diagrams `family(theta)` (loop occupations at the cut and output counts), the reference gradient by central differences, and the equilibrium propagation estimate: the response of the means to a nudge of the sources, `q -> q + beta c`, contracted with the one-tick derivative of the means
- [x] exact one-body version for passive diagrams, independent of the source and of `p`
- [x] tests: exact for a reciprocal loop dephased at the cut, whatever the source and `p`, biased without the dephasing or the reciprocity; the sampled path runs, with feed-forward
- [x] notebook in `docs/notebooks`: equilibrium propagation against the gradient, exact for passive loops, and the sampled feed-forward grid for every source and `p`
- [ ] thermal sources in the grid, at a mean that stays under the cap
- [ ] phase-locked coherent light (fields adding up coherently over the ticks), proposed in the session, awaiting a decision
- [ ] API docs, `pflake8 optyx`, `coverage run -m pytest`, CI green

---

> claude --enable-data-mode Make a map of the sources and implement what's needed in optyx. In wiki, find the types of feedforward that converge to equilibrium and can be trained with equilibrium propagation

The map and the new sources are in #94, merged here.

- [x] `Equilibrium(fields=True)`: equilibrium propagation sampled by the `Fields` of classical light, phase-locked lasers included; the exact means with the field of a phase-locked source
- [x] `Equilibrium.propagate_fields`: equilibrium propagation on the fields of phase-locked light, exact for a symmetric mode matrix where the intensities are biased
- [WIP] @session_01XmqyhQuUzn8CQGX7CBCMw5-2026-10-29 12:00 the wiki page on the feed-forward that settles into equilibrium and can be trained, run against this branch
