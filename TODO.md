# TODO

> Check out this handout and the relevant PRs in wiki and optyx. Analyse it before running the experiments and let me know what you think.

> go ahead, start with step 1

The handout is `handout-sampler-validation.md` (sampler validation of the ideal sample counts of the
expressivity section, rel-int/wiki#25 and rel-int/optyx#88). Step 1 of the agreed plan: the classical
recurrent samplers, checked against `photon_noise`, with error bars that hold for count products, and
the autocorrelation of consecutive ticks, exact and sampled.

- [WIP] @session_016ArmVH3FaxVkZdf9whHi3e-2026-10-23 12:00 `recurrent.py`: the device run tick by tick (loop state x <- Ax + Bs, fields y = Cx + Ds, Poisson counts) for coherent and thermal light, and distinguishable photons as classical particles
- [ ] exact two-time moments: Cov(prod_S n(t), prod_S n(t+k)) from the two-time path matrices, and the integrated autocorrelation time
- [ ] error bars for count products: the standard error of the sample variance from the sampled fourth moment, pulls only where the counts support them
- [ ] Level 1 on the small pairs: means, variances and pulls against `photon_noise`, per source, brightness and order
- [ ] tests in `test_recurrent.py`, README entry, results gzipped in `results/`
