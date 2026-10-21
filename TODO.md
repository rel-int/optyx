# TODO

> Run the pilot on the 49-pair subset using modal

The pilot of the forward-pass test of rel-int/wiki#26: BREC's protocol with a draw of
the input light as the forward pass and the elementary symmetric polynomials of the
single-tick intensities as the embedding, on the 49-pair subset, four cells, the
sources coherent-w1, coherent-w100, coherent-inf and thermal-inf, budgets 1e3 to 1e6.

- [x] `single_tick.py`: the readout, the samplers, the copies, BREC's decisions, the exact anchors
- [x] `tick_app.py`, `launch_tick.py`, `report_tick.py`: the Modal jobs, the launcher, the tables
- [x] validation on small graphs (on a CPU container, the workspace's GPUs being taken): invariance under relabelling, sampled against exact, false positives, power, timings
- [WIP] @session_01FqeJsXRvgZoRe2x8CLkyG8-2026-10-21 08:25 the run on the subset and its tables under `results/`
- [ ] the wiki page `forward-pass-test.md` updated with the pilot's results
