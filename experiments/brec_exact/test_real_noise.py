"""Brute-force checks of real_noise: the tables against a Monte Carlo of the
phase-locked sources on a small random interferometer."""
import sys
import time

import numpy as np

sys.path.insert(0, ".")
import real_noise as rn  # noqa: E402

rng = np.random.default_rng(1)
W = rng.normal(size=(4, 5)) + 1j * rng.normal(size=(4, 5))
W /= 3
w = 0.7
for dist in rn.CUMULANTS:
    for detectors in ([0], [0, 1], [0, 0], [0, 1, 2], [0, 0, 1], [0, 1, 2, 3], [0, 0, 1, 2], [0, 0, 1, 1]):
        exact = rn.cpu_moment(W, detectors, rn.CUMULANTS[dist]) * w ** len(detectors)
        mc, err = rn.monte_carlo(W, detectors, dist, w, 400000)
        pull = (mc - exact) / err
        flag = "" if abs(pull) < 4 else "  <-- MISMATCH"
        print(f"{dist:14s} {str(detectors):14s} exact {exact:.5e} mc {mc:.5e} +- {err:.1e} pull {pull:+.2f}{flag}")
for N, doubled in ((4, ()), (4, (0, 1)), (3, (0, 1, 2))):
    t0 = time.time()
    patterns, keys, weights = rn.tables_real(N, doubled)
    print(f"N={N} doubled={doubled}: {len(patterns)} patterns, {keys.shape[0]} keys, {len(weights)} shapes, {time.time() - t0:.1f}s")
