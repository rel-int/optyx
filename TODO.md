# TODO

> Let's see how far we can climb up the k-WL ladder. Use Cai et al. and the BREC benchmark to construct a dataset of graphs of increasing k-WL indistinguishability (say 5 graphs for each k). Then benchmark the following models: 1. passive photonic QMapNN with a single photon per cell, 2. passive photonic QMapNN

(The prompt arrived truncated after "2. passive photonic QMapNN"; asked for the full list, answer was "no preference", so the ladder is run as: passive one-photon, passive two-photon, active feed-forward with broadcast radius 0, 1 and 2 — the E1 design of `docs/quantum_map_neural_networks.md`, which subsumes the readings of the truncated list.)

- [WIP] @01AdC8wUqUgpSYLjK56Dvhy9-2026-08-22 06:25 Dataset: pairs of increasing k-WL indistinguishability from the Cai-Fürer-Immerman construction and the BREC benchmark, about five per rung, with the claimed WL level verified by our own 1-WL / 2-FWL (/ 3-FWL where feasible) implementations
- [WIP] @01AdC8wUqUgpSYLjK56Dvhy9-2026-08-22 06:25 Extract the beyond_3wl simulator (cell-wise step-matrix assembly, branching two-photon machine) into an importable module under `experiments/kwl_ladder/`
- [WIP] @01AdC8wUqUgpSYLjK56Dvhy9-2026-08-22 06:25 Benchmark the model ladder on every pair: passive one-photon, passive two-photon, active r=0 (own cell), active r=1 (one hop), active r=2 (two hops), trimming the expensive cells with an explicit log rather than silently
- [WIP] @01AdC8wUqUgpSYLjK56Dvhy9-2026-08-22 06:25 Committed results (CSV + report) with validation: probability conservation and relabelling invariance of every separation, exact zeros stated as measured
- [WIP] @01AdC8wUqUgpSYLjK56Dvhy9-2026-08-22 06:25 Lint and tests green (a fast smoke test in `test/`), draft PR stacked on #61
