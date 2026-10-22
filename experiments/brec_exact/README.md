# Exact coincidence profiles of photonic graph networks on BREC

The scripts behind the wiki page *BREC's test on the exact profiles, and the
loss sweep* (`content/photonic-loops/brec-exact-column.md` in `rel-int/wiki`).
They compute the exact order-N coincidence rates of the recurrent photonic
graph network of the paper for four light sources, indistinguishable and
distinguishable single photons, random-phase coherent light and thermal
light, and run two tests on them: BREC's own Hotelling protocol on relabelled
copies, and the exact rule of the draft.

- `brecphot.py`: the network of a graph, its path matrix, the rates of every
  source as one Gram-tensor computation per set partition and permutation of
  the photons, the sorted profiles, BREC's T² test and the exact gap.
- `coherent_gpu.py`: the same rates in torch on a GPU in complex128, and the
  variance of every coincidence of coherent light at any brightness from a
  phase Monte Carlo of the intensities.
- `test_brecphot.py`: independent checks on small graphs, permanents, brute
  force sums over slot subsets, a phase Monte Carlo, relabelling invariance.
- `modal_app.py`, `launch.py`, `report.py`: the CPU jobs on Modal (relabelled
  copies, loss sweep, the aggregation of the pilot).
- `gpu_app.py`, `launch_gpu.py`, `report_full.py`: the GPU jobs for the full
  benchmark with coherent light at several brightnesses.
- `launch_full_test.py`: BREC's test on the stored profiles of the full run,
  without relabelled copies, which on an exact embedding reduce the statistic
  to the projected gap over the regularisation.
- `gnn_bench.py`, `gnn_app.py`, `launch_gnn.py`, `report_gnn.py`: the comparison
  with probabilistic graph networks: samplers for Gaussian noise at every step
  or at initialisation, linear or tanh, with a squared readout; the coherent
  sampler at any brightness; exact moments of the linear models by Wick's
  theorem; the paper's batch-permutation protocol with profile and type
  features; BREC's test on sampled copies; and the GPU jobs and tables.
- `photon_noise.py`, `test_photon_noise.py`, `noise_app.py`, `launch_noise.py`,
  `report_noise.py`: the exact noise of every source at any brightness. The
  variance of an order-N coincidence is a sum of normally ordered moments of
  up to 2N photons on the multiset of detectors, each a sum over set
  partitions and permutations of products of Gram sums over the slots; the
  enumeration is done once per multiplicity pattern (4140 partitions times
  40320 permutations for eight photons, 118,053 distinct products) and
  cached in `noise_tables/`, which the GPU job combines with the source's
  coefficients: single photons, Fock states, coherent and thermal light
  share the tables. Checked against brute-force sums of permanents up to
  eight photons and against the coherent Monte Carlo.
- `write_page.py`: fills the wiki page from the aggregated results.

`results/` holds the per-pair result files of each run, gzipped, and the
tables built from them: `raw.json.gz` (the pilot's relabelled copies and loss
sweep, `tables_pilot.md`), `raw_full.json.gz` (full BREC on the GPU with
BREC's test from the profiles, `tables_full.md`), `raw_gnn_v4_subset_partial.json.gz`
(the sampling comparison with the probabilistic networks on 43 of the 49
subset pairs, stopped before the six Distance_regular pairs,
`tables_gnn_v4_subset_partial.md`) and `raw_ideal_full.json.gz` (the exact
Wick profiles and ideal sample counts of the two linear networks on all 400
pairs, `tables_ideal.md`). `report.py tables`, `report_full.py tables`,
`report_gnn.py tables` and `report_ideal.py` regenerate the tables from
decompressed copies without Modal. The profiles and noise moments behind
them, tens of gigabytes, stay on the volume. `diag_protocol.py` and
`diag_sorting.py` are the local diagnoses of the protocol's sample
efficiency against the ideal count on synthetic batches: the loss is the
order-statistics bias of sorting noisy estimates at high orders, and neither
a studentised nor a split-sample profile recovers it.

The BREC data is read from a clone of `GraphPKU/BREC` at
`/home/user/graphpku/brec/customize/Data/raw`; results live on the Modal
volume `photonic-brec-rates` under `column-pilot/`, `loss-sweep/` and
`brec-full-coherent/`.

```shell
uv venv && uv pip install numpy 'modal[api-proxy-support]' matplotlib torch
modal deploy modal_app.py && modal deploy gpu_app.py
python launch.py column && python launch.py sweep && python launch.py test
python launch_gpu.py
python report.py fetch && python report.py tables
python report_full.py fetch && python report_full.py tables
```
