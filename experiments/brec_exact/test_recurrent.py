"""Checks of recurrent.py on a 4-cycle with a random cell: the two-time
moments against closed forms at order 1, and the samplers of the device
against the exact rates, variances and autocovariances of photon_noise."""
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
import brecphot as bp  # noqa: E402
import recurrent as rc  # noqa: E402

GAMMA, W_COHERENT = 0.6, 0.7


def cycle(n):
    adj = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        adj[i, (i + 1) % n] = adj[(i + 1) % n, i] = 1
    return adj


def device():
    return rc.Device(bp.network(cycle(4), bp.cell(7), GAMMA))


def test_two_time_order_one():
    """Cov(n_a(t), n_b(t + k)) from the Gram sum G = sum_m W[a, m]
    conj(W[b, m + k]) over the slots and the diagonal sum
    P = sum_m |W[a, m]|^2 |W[b, m + k]|^2: (|G|^2 + (2 lambda(2) - 2) P) w^2
    for thermal light, coherent light and single photons, the two photons
    of distinct slots interfering, and -P w^2 for distinguishable photons."""
    d = device()
    n, W, w = d.n, d.W, 0.4
    for k in (1, 3):
        early, late = W[[0]], W[[2]]
        shifted = np.concatenate([late[:, k * n:], np.zeros((1, k * n))], axis=1)
        G = (early * shifted.conj()).sum()
        P = (np.abs(early) ** 2 * np.abs(shifted) ** 2).sum()
        expected = {"thermal": abs(G) ** 2, "coherent": abs(G) ** 2 - P, "photons p=0": -P,
                    "photons p=1": abs(G) ** 2 - 2 * P}
        for source, value in expected.items():
            F = w * rc.distinct_moments(W[[0]][None], source)[0], w * rc.distinct_moments(W[[2]][None], source)[0]
            joint = w ** 2 * rc.distinct_moments(rc.two_time_rows(W, n, (0, 2), k)[[0, 3]][None], source)[0]
            assert np.isclose(joint - F[0] * F[1], w ** 2 * value, atol=1e-14), (source, k)


def test_distinct_moments_are_rates():
    d = device()
    for source in bp.SOURCES:
        tensor = bp.rate_tensors(d.W, 3, (source,))[source]
        assert np.isclose(rc.distinct_moments(d.W[[0, 1, 3]][None], source)[0], tensor[0, 1, 3])


def test_autocovariance_vanishes():
    d = device()
    cov = rc.autocovariances(d.W, d.n, [(0, 1)], 60, "thermal", 1.0)
    var = rc.rates_variances(d.W, 2, "thermal", 1.0)[2][1][0]
    assert abs(cov[-1, 0]) < 1e-8 * var


def test_samplers():
    """Sample means, variances and lag-one autocovariances within five
    standard errors of the exact ones, for every sampled source."""
    torch.manual_seed(0)
    d = device()
    nmax, chains, ticks, block = 2, 1024, 384, 32
    for source, w in (("coherent", W_COHERENT), ("thermal", 0.5), ("photons p=0", 0.5)):
        exact = rc.rates_variances(d.W, nmax, source, w)
        tracked = {N: list(range(len(exact[N][0]))) for N in exact}
        moments = rc.Moments(d.n, {N: F for N, (F, _) in exact.items()}, chains, 16, block, 2, tracked)
        for y in d.counts(source, w, chains, ticks, seed=3):
            moments.add(y)
        sums = moments.result()
        for N, (F, var) in exact.items():
            s = rc.summary(F, var, sums[N], block)
            sets = [tuple(rc.set_index(d.n, N)[:, i]) for i in tracked[N]]
            cov = rc.autocovariances(d.W, d.n, sets, 60, source, w)
            tau = rc.tau_exact(cov, var)
            assert np.all(np.abs(s["mean"] - F) < 5 * np.sqrt(tau * var / s["T"])), (source, N)
            assert np.all(np.abs(s["sample_var"] - var) < 5 * s["var_se"]), (source, N)
            lag_se = np.sqrt(np.maximum(s["m4"], var ** 2) / s["T"])
            assert np.all(np.abs(s["lagged"][0] - cov[0]) < 5 * lag_se), (source, N)
            assert np.all(np.abs(s["tau_blocks"] - tau) < 0.3), (source, N)


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print(name, "ok")
