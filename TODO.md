# TODO

> Check out this handout and the relevant PRs in wiki and optyx. Analyse it before running the experiments and let me know what you think.

> go ahead, start with step 1

The handout is `handout-sampler-validation.md` (sampler validation of the ideal sample counts of the
expressivity section, rel-int/wiki#25 and rel-int/optyx#88). Step 1 of the agreed plan: the classical
recurrent samplers, checked against `photon_noise`, with error bars that hold for count products, and
the autocorrelation of consecutive ticks, exact and sampled.

- [x] `recurrent.py`: the device run tick by tick (loop state x <- Ax + Bs, fields y = Cx + Ds, Poisson counts) for coherent and thermal light, and distinguishable photons as classical particles
- [x] exact two-time moments: Cov(prod_S n(t), prod_S n(t+k)) from the two-time path matrices, and the integrated autocorrelation time, up to order 3 (order 4 needs the tables of 8 distinct detectors, too large: sampled only)
- [x] error bars for count products: the standard error of the sample variance from the spread of independent batches of chains, pulls only where the counts support them
- [x] Level 1 on the small pairs: means, variances and pulls against `photon_noise`, per source, brightness and order
- [x] tests in `test_recurrent.py`, README entry, results gzipped in `results/`

> yes, start with the photon feasibility checks

Photon feasibility, before planning the photon validation: indistinguishable photons need the trajectory
sampler of rel-int/optyx#76.

- [WIP] @session_016ArmVH3FaxVkZdf9whHi3e-2026-10-23 11:35 can `Diagram.sample` drive a source that emits a photon with probability w per tick
- [ ] Regular_47 as an optyx diagram, its unitary against `brecphot.network` to 1e-12
- [ ] the sampler's speed at w = 0.05 on Regular_47 (0.66 photons in the loop), and the budget it allows
