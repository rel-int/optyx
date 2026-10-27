The architectures can also have non linearity in the form of feedforward from the measurement outcomes at each tick to a (possibly delayed) phase modulation on the chip. The samplers are able to simulate this behaviour for distinguishable and indistinguishable photons, I believe coherent and thermal also should be doable. Open a new PR in discopy-next/optyx porting the simulation code in the wiki to have simulation of arbitrary photonic optyx diagrams (no qubits but classical feedforward allowed) with the different sources. Then implement equilibrium propagation on them and see test if it approximates the gradient.

---

Stacked on #76, whose `optyx.sampling` is the simulation code the wiki notebooks pin: it already samples feed-forward diagrams (`Unravelling`) with `Fock`, `Coherent` and `Thermal` sources, and partially distinguishable photons in passive ones (`Interferometer`). What is missing is the last cell of that table, and equilibrium propagation.

- [x] `Unravelling(indistinguishability=p)`: colour the photons of the source as `Interferometer.colour` does, run the colours in lockstep box by box with shared classical values, a number resolving measurement recording the sum of the colours' counts; `Diagram.sample` stops refusing `indistinguishability` for diagrams with feed-forward (no qubits)
- [x] tests: colouring against the `Interferometer` on a passive diagram, and against the exact distribution of a feed-forward loop unrolled with one internal state per tick
- [x] `optyx/equilibrium.py`: the stationary means of a family of diagrams `family(theta)` (loop occupations at the cut and output counts), the reference gradient by central differences, and the equilibrium propagation estimate: the response of the means to a nudge of the sources, `q -> q + beta dC/do`, contracted with the one-tick derivative of the means
- [x] exact one-body version for passive diagrams (the Stein equation of `Interferometer.occupation`), independent of the source and of `p`
- [x] tests: exact for a reciprocal loop dephased at the cut, whatever the source and `p`, biased without the dephasing or the reciprocity; the sampled path runs, with feed-forward
- [WIP] @session_01XmqyhQuUzn8CQGX7CBCMw5-2026-10-27 13:10 notebook in `docs/notebooks`: equilibrium propagation against the gradient for every source and `p`, passive and with feed-forward
- [ ] API docs, `pflake8 optyx`, `coverage run -m pytest`
