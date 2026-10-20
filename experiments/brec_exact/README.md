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
- `write_page.py`: fills the wiki page from the aggregated results.

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
