Differentiable photonic sampler in JAX, stacked on #76: the JAX kernel of `optyx.sampling` moved out of #76, then benchmarked and optimised.

> That's great. Prepare a handout for another session to continue your work on the optyx PR to have jax-compatible sampling of photonic quantum recurrent networks, represented as optyx diagrams with loss and distinguishability, first passive then active.

Passive (phase 1), each point validated against the NumPy sampler and its exact tests:

- [x] progressive measurement of the external modes, so that a tick holds the loop sector times a few modes (prerequisite: fixes the memory wall and changes what the JAX kernel holds)
- [x] fixed-shape kernels: `Kernel`, the held state at one width with vacuum modes past the occupied ones, each external output one step of a `lax.scan` over the sweep, a `lax.switch` on its photon number, the chain as a `lax.scan` of `turn`, `segment_sum` marginals, `jax.random.categorical` and rank-scatter collapse, loss Kraus on an extra mode counting the lost photons
- [x] occupation indices computed by combinatorial ranking (`rank`) instead of stored tables: a rotation's partner indices are its rank plus one difference of two `binomials` rows; only the occupations of each sector are stored
- [x] rotations built from a branch-free Givens decomposition (`givens_rotation`, `Kernel.schedule`), so that gradients reach the mode matrix, loss, efficiency and indistinguishability; the diagram's own parameters are not reachable yet, filed as #81
- [x] trajectories: one compiled branch per photon number up to the cap n* + |q|, `lax.map` over trajectories (`batch_size` vmaps chunks), colours with private photons as a padded batch of single-photon walks compacted every tick, overflow of either cap flagged and raised
- [x] gradients: `Kernel.log_prob` replays recorded outcomes (equal to the sampled log probabilities), score-function estimates agree with finite differences within 1σ, exact differentiable `Kernel.distribution` for small systems checked against finite differences of the NumPy one

---

> Let's finish the full jaxable implementation of the sampler

> How can we benchmark the jax sampler against the standard one? How do we make it more efficient?

> The PR has way too many lines. Let's move the jax optimisation to a separate PR on a differentiable photonic sampler in JAX. On this PR the sampler should be implemented generally and compatible with the extension., but the aim of this PR is to extend the sampler to the full generality of optyx diagrams

Measured so far, one CPU core: on a 2+2-mode lossy network the kernel runs 15x the NumPy ticks per second, but `vmap` batching drops it to 0.4-0.7x since every photon-number `lax.switch` becomes a select of all branches; one rotation costs about as much as NumPy's on sectors of 10^3 to 2x10^5 states; on the loopback geometry a tick is 2-3x NumPy's.

- [ ] benchmark suite as in DisCoPy: `pytest-benchmark` cases over a grid of loop size, external modes, photons per tick, loss and indistinguishability, plus the loopback geometry; compile time, steady-state ticks per second, peak memory and throughput against shots reported separately, one thread and all cores, the cap set to the photon number NumPy reached; distributions compared, not samples; a `benchmark` workflow on a label
- [ ] cap from the tail of the loop photon number (Lemma 2 of the wiki plan) rather than mean + 6 sqrt(mean) + 2, which sets the number of compiled branches and the buffer size
- [ ] precompute in NumPy the prefix sums of the occupation tables that XLA spends seconds constant-folding
- [ ] batching: a switch-free mode on the space of every photon number up to the cap, with a slack mode, for GPU and small caps; bucketing trajectories by photon number for large caps
- [ ] rotations as batched small matmuls on the blocks a rotation mixes, found by rank, instead of one gather of states x (photons + 1)
- [ ] buffers sized for the photons held rather than the cap, through the bucketing above
- [ ] the general sampler of #76 under JAX: classical configurations as `lax.switch`, once #76 lands
