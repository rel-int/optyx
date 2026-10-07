Great, go ahead with the implementation in a new optyx PR. Is it the same method as the recent ORCA paper? For your theorem on the lossy setting, you need to say how the simulation time/memory is related to loss, can't just say can be simulated exactly like this. Open a new PR in wiki with a short document outlining the plan for the paper and key results. Make sure that the comparison and advancement with respect to ORCA is well argued.

---

Sampler for the stationary output stream of a passive recurrent linear-optical
network: an (L+x)-mode interferometer whose first L outputs are fed back into
its first L inputs one tick later, with a product Fock state injected into the
other x inputs at every tick.

- [x] `optyx/recurrent.py`: Fock sectors and creation operators on them, one tick as a pure trajectory with a definite loop photon number, loop loss unravelled by sampling the photons lost, detector efficiency as binomial thinning
- [x] certified burn-in depth from the singular values of the loop block (the stationary boson sampling bound) and the exact stationary mean loop photon number
- [x] exact joint distribution of a window by enumerating trajectories
- [x] tests against permanents of the unrolled interferometer, with and without loss
- [x] API docs entry
- [ ] notebook in `docs/notebooks`: memory and time per tick against loss, and the loopback experiment geometry (numbers so far in the wiki paper plan)
- [ ] validate against `Diagram.fix` of `armand/minimal-sbs-certificate` once feedback is on main
- [x] measure each external mode as soon as no later rotation touches it, so that a tick holds the loop sector times a few external modes rather than the whole (L+x)-mode sector (the lossless loopback geometry, L=5, x=20, runs out of 14 GB)

---

> Implement the full sampler with a distinguishability parameter as a method of any optyx diagram with feedback loops and discards

- [x] `Recurrent` takes the injection per tick, records only the visible outputs, and samples partially distinguishable photons by colouring them
- [x] `Recurrent.from_diagram`: one step of a closed channel diagram with feedback, dilated (discards and loss as environment outputs) to a path matrix, completed to a unitary
- [x] `channel.Diagram.sample(ticks, tol, indistinguishability, seed)`, with tests against the unrolled diagram's exact distribution

---

> That's great. Prepare a handout for another session to continue your work on the optyx PR to have jax-compatible sampling of photonic quantum recurrent networks, represented as optyx diagrams with loss and distinguishability, first passive then active.

Passive (phase 1), each point validated against the NumPy sampler and its exact tests:

- [x] progressive measurement of the external modes, so that a tick holds the loop sector times a few modes (prerequisite: fixes the memory wall and changes what the JAX kernel holds)
- [WIP] @session_01AotUfRNtbiTmw5DToMZ91K-2026-10-07 11:00 fixed-shape kernels: per-sector rotation groups stacked over rotations, interferometer as one `lax.scan`, measurement by static sector-to-pattern maps, `segment_sum`, `jax.random.categorical` and mask-then-gather collapse, loss Kraus as static index shifts
- [WIP] @session_01AotUfRNtbiTmw5DToMZ91K-2026-10-07 11:00 occupation indices computed by combinatorial ranking instead of stored tables
- [WIP] @session_01AotUfRNtbiTmw5DToMZ91K-2026-10-07 11:00 rotations built from the diagram's parameters (or a branch-free Givens decomposition), so that gradients reach the phases
- [WIP] @session_01AotUfRNtbiTmw5DToMZ91K-2026-10-07 11:00 trajectories: one compiled step per loop photon number up to the cap n*, `vmap` over trajectories bucketed by photon number, colours with private photons as a padded batch of single-photon vectors
- [WIP] @session_01AotUfRNtbiTmw5DToMZ91K-2026-10-07 11:00 gradients: log-probability replay of a recorded trajectory, score-function estimator, exact differentiable `distribution` for small systems, checked against finite differences

Active (phase 2):

- [ ] layer-by-layer interpreter of `one_step`: number resolving measurements sample and collapse, classical boxes compute on sampled values, classically controlled boxes (`BitControlledGate`, `BitControlledPhaseShift`, `ClassicalFunction`, `core.control.ControlledPhaseShift`) pick their matrix, `lax.switch` over the finite set of configurations under JAX
- [ ] no burn-in certificate or cost bound for active diagrams: require an explicit burn-in, guard the photon cap, refuse postselection

---

> Let's merge the PRs into one adding a sampling module in optyx as file. We need to find the right names for classes so that this acts like a backend for sampling diagrams on modes.

> Let's finish the full jaxable implementation of the sampler

Names agreed in session: `optyx/sampling.py`, `Interferometer` (was `Recurrent`), `FockState` (was `LoopState`), `Sweep`; `channel.Diagram.sample` stays the only entry point, no sampler class.

- [x] merge #78 into this PR and `main` (feedback, #57) into it
- [x] move `optyx/recurrent.py` to `optyx/sampling.py` with the names above
