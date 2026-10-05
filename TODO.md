Great, go ahead with the implementation in a new optyx PR. Is it the same method as the recent ORCA paper? For your theorem on the lossy setting, you need to say how the simulation time/memory is related to loss, can't just say can be simulated exactly like this. Open a new PR in wiki with a short document outlining the plan for the paper and key results. Make sure that the comparison and advancement with respect to ORCA is well argued.

---

Sampler for the stationary output stream of a passive recurrent linear-optical
network: an (L+x)-mode interferometer whose first L outputs are fed back into
its first L inputs one tick later, with a product Fock state injected into the
other x inputs at every tick.

- [WIP] @session_01WTPb48z2J6m377hUPTvFTz-2026-10-05 08:30 `optyx/recurrent.py`: Fock sectors and creation operators on them, one tick as a pure trajectory with a definite loop photon number, loop loss unravelled by sampling the photons lost, detector efficiency as binomial thinning
- [WIP] @session_01WTPb48z2J6m377hUPTvFTz-2026-10-05 08:30 certified burn-in depth from the singular values of the loop block (the stationary boson sampling bound) and the exact stationary mean loop photon number
- [WIP] @session_01WTPb48z2J6m377hUPTvFTz-2026-10-05 08:30 exact joint distribution of a window by enumerating trajectories
- [WIP] @session_01WTPb48z2J6m377hUPTvFTz-2026-10-05 08:30 tests against permanents of the unrolled interferometer, with and without loss
- [WIP] @session_01WTPb48z2J6m377hUPTvFTz-2026-10-05 08:30 API docs entry
