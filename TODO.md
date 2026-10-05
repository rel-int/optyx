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
- [ ] measure each external mode as soon as no later rotation touches it, so that a tick holds the loop sector times a few external modes rather than the whole (L+x)-mode sector (the lossless loopback geometry, L=5, x=20, runs out of 14 GB)
