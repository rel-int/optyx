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

Passive (phase 1): the JAX kernel moved to its own pull request, stacked on this one, together with its optimisation; the progressive measurement it builds on stays here.

- [x] progressive measurement of the external modes, so that a tick holds the loop sector times a few modes (prerequisite: fixes the memory wall and changes what the JAX kernel holds)

Active (phase 2):

- [x] layer-by-layer interpreter of `one_step`: `Unravelling` samples number resolving measurements and collapses, classical boxes sample their outputs from their inputs, classically controlled boxes take the slice of their control values (the JAX half goes to the JAX pull request)
- [x] no burn-in certificate or cost bound for active diagrams: `Diagram.sample` requires an explicit `burn_in`, the photon cap raises when amplitude passes it, postselection raises at the end of a tick that is not trace preserving

---

> Let's merge the PRs into one adding a sampling module in optyx as file. We need to find the right names for classes so that this acts like a backend for sampling diagrams on modes.

> Let's finish the full jaxable implementation of the sampler

Names agreed in session: `optyx/sampling.py`, `Interferometer` (was `Recurrent`), `FockState` (was `LoopState`), `Sweep`; `channel.Diagram.sample` stays the only entry point, no sampler class.

- [x] merge #78 into this PR and `main` (feedback, #57) into it
- [x] move `optyx/recurrent.py` to `optyx/sampling.py` with the names above
- [x] the JAX kernel, now moved to its own pull request

---

> The PR has way too many lines. Let's move the jax optimisation to a separate PR on a differentiable photonic sampler in JAX. On this PR the sampler should be implemented generally and compatible with the extension., but the aim of this PR is to extend the sampler to the full generality of optyx diagrams

A closed diagram with feedback is sampled as a quantum trajectory: one tick runs :meth:`one_step` layer by layer on a pure state of the quantum wires and definite values of the classical wires. A box contracts its Kraus map into the state, its environment and its classical outputs are measured in the number basis, sampled and collapsed, and a classical box samples its outputs from its classical inputs. Each tick is then a fixed sequence of tensor contractions, measurements and choices on classical values, which the JAX pull request can compile as it stands. Passive diagrams keep the `Interferometer`, whose ticks are exponentially cheaper and whose burn-in is certified.

- [x] merge `main`, now squashed, so that the diff is the sampler alone
- [x] move the JAX kernel to its own pull request, stacked on this one
- [x] use `channel.Diagram.dilate` from `main` rather than `sampling.dilation`
- [x] general sampler: `Unravelling`, the trajectory of any closed `channel.Diagram` with feedback, its initial loop state from `boundary()`, qubits, measurements, classical boxes and classical control in the loop, non-linear boxes such as a Kerr phase, internal states by inflation (with `Channel.inflate` now inflating the environment); linear optics through the Givens decomposition of its mode matrix, anything else one box of its Kraus map at a time; postselection refused, the photon cap guarded
- [x] `Diagram.sample` dispatches: passive diagrams to the `Interferometer`, the others to the `Unravelling` with an explicit burn-in
- [x] tests of the general sampler against the exact distribution of the unrolled diagram: feed-forward inside a loop, a qubit memory, a classical function, internal states with loss, a Kerr phase, and the passive diagram


---

> Let's extend the What is a channel? notebook with the sampling story, we should now be able to sample from these circuits and check that the outputs agree with the tensor network model

- [WIP] @session_01AotUfRNtbiTmw5DToMZ91K-2026-10-09 09:00 a sampling section in `docs/notebooks/what_is_a_channel.md`: sample the circuits of the earlier sections (boson sampling, distinguishable photons, feedforward, the delay loop, the universal setup) and compare the frequencies with the tensor network's distribution
