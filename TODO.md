> The architectures can also have non linearity in the form of feedforward from the measurement outcomes at each tick to a (possibly delayed) phase modulation on the chip. The samplers are able to simulate this behaviour for distinguishable and indistinguishable photons, I believe coherent and thermal also should be doable. Open a new PR in discopy-next/optyx porting the simulation code in the wiki to have simulation of arbitrary photonic optyx diagrams (no qubits but classical feedforward allowed) with the different sources. Then implement equilibrium propagation on them and see test if it approximates the gradient.

> Split this PR into two. First the samplers for the different light sources, then the implementation of equilibrium propagation on photonic architectures

---

The first half, the samplers; equilibrium propagation is the pull request stacked on this one.

- [x] `Unravelling(indistinguishability=p)`: colour the photons of the source as `Interferometer.colour` does (the shared `colouring`), run the colours in lockstep box by box with shared classical values, a detector pooling the colours' counts (`Measure.pool`, `PhotonThresholdMeasurement.pool`); `Diagram.sample` passes `indistinguishability` to it
- [x] tests: colouring against the exact distribution of a feed-forward loop with one distinguishable copy of each optical wire per tick, and against the `Interferometer` on a passive loop; refusals
- [x] sources: `Source.shift`, `Shifted`; `photonic.Dephasing`; the exact one-particle means of an `Interferometer` with dephased loop modes (`advance`, `correlations`), `from_step`, `Unravelling.tick` and `occupations`, `CQState.means`
- [ ] thermal light with feed-forward overflows `cap=8` in long runs: document the cap a thermal source needs, or test it at a lower mean
- [ ] `pflake8 optyx`, `coverage run -m pytest`, CI green
