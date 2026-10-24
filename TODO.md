> Implement the Gaussian squeezed light and run it on the full benchmark up to order 4 where it’s efficient (as for photons) and order 3 otherwise. Before running it full make sure that the implemntation is as efficient as possible. Use modal for this.

> Sorry, first implement it and benchmark it on a representative subset (the 15 pair subset from the handouts), report the results and then we’ll decide whether to run it on full BREC

- [x] Exact rates and variances of phase-locked squeezed vacuum by Wick's theorem on the Gram matrices G = W W^dag and T = W W^T (`squeezed.py`)
- [x] Checks: matching counts, The Walrus up to order 4 with variances, thermal light against `photon_noise`, the real Gaussian amplitude against `real_noise`, relabelling (`test_squeezed.py`)
- [x] Efficiency: meet-in-the-middle products over the monomials, one multiply-add per monomial and set; GPU against CPU to 1e-11
- [x] Modal app and the 16-pair subset of `launch_kmode.py`, four cells, order 4 (`squeezed_app.py`, `launch_squeezed.py`, `report_squeezed.py`, `results/tables_squeezed.md`)
- [x] Full BREC at the orders of the photons' exact-noise run (4 up to 63 vertices, 3 up to 110, 2 above)

> Great, then run also the higher orders on squeezed

- [x] Squeezed light at the orders of the reach run of the other sources (4 up to 96 vertices, 3 above)
