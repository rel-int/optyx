> The architectures can also have non linearity in the form of feedforward from the measurement outcomes at each tick to a (possibly delayed) phase modulation on the chip. The samplers are able to simulate this behaviour for distinguishable and indistinguishable photons, I believe coherent and thermal also should be doable. Open a new PR in discopy-next/optyx porting the simulation code in the wiki to have simulation of arbitrary photonic optyx diagrams (no qubits but classical feedforward allowed) with the different sources. Then implement equilibrium propagation on them and see test if it approximates the gradient.

> Split this PR into two. First the samplers for the different light sources, then the implementation of equilibrium propagation on photonic architectures

---

The first half, the samplers; equilibrium propagation is the pull request stacked on this one.

- [x] `Unravelling(indistinguishability=p)`: colour the photons of the source as `Interferometer.colour` does (the shared `colouring`), run the colours in lockstep box by box with shared classical values, a detector pooling the colours' counts (`Measure.pool`, `PhotonThresholdMeasurement.pool`); `Diagram.sample` passes `indistinguishability` to it
- [x] tests: colouring against the exact distribution of a feed-forward loop with one distinguishable copy of each optical wire per tick, and against the `Interferometer` on a passive loop; refusals
- [x] sources: `Source.shift`, `Shifted`; `photonic.Dephasing`; the exact one-particle means of an `Interferometer` with dephased loop modes (`advance`, `correlations`), `from_step`, `Unravelling.tick` and `occupations`, `CQState.means`
- [ ] thermal light with feed-forward overflows `cap=8` in long runs: document the cap a thermal source needs, or test it at a lower mean
- [ ] `pflake8 optyx`, `coverage run -m pytest`, CI green

---

> claude --enable-data-mode Make a map of the sources and implement what's needed in optyx. In wiki, find the types of feedforward that converge to equilibrium and can be trained with equilibrium propagation

- [WIP] @session_01XmqyhQuUzn8CQGX7CBCMw5-2026-10-29 09:00 the map of light sources in the module docstring: Fock, coherent with or without a phase reference, thermal, squeezed; Gaussian or not, classical or not; which sampler and which exact means cover each
- [ ] `PhaseLocked` (coherent light with a fixed phase from tick to tick) and `Squeezed` sources, injected into the `Unravelling` as kets truncated at the cap
- [ ] `Fields`: trajectories of classical light (mixtures of coherent states) through any diagram of linear optics, detectors and classical feed-forward, drawing amplitudes from the source's P function and Poisson counts; no cap
- [ ] exact one-particle means with a mean field: `Interferometer.advance` and `correlations` with phase-locked sources
- [ ] tests: `Fields` against the `Unravelling` for coherent and thermal light, with feed-forward; kets against the exact distribution; means against sampling

