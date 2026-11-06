"""Checks of squeezed.py: the matching counts, Gaussian moments against The
Walrus on a small lossy network, thermal light against photon_noise and the
phase-locked real Gaussian against real_noise, and relabelling invariance."""
from itertools import combinations

import numpy as np

import brecphot as bp
import photon_noise as pn
import real_noise as rn
import squeezed as sq


def double_factorial(k):
    return 1 if k <= 1 else k * double_factorial(k - 2)


def test_counts():
    for N in range(1, 5):
        for size in range(N + 1):
            for d in combinations(range(N), size):
                K = N + size
                keys, counts, j = sq.tables(N, d)
                assert counts.sum() == double_factorial(2 * K - 1)
                assert np.all(j <= K // 2)


def small_network(seed=100, gamma=0.8):
    adj = np.zeros((5, 5), dtype=np.int8)
    for a, b in ((0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 2)):
        adj[a, b] = adj[b, a] = 1
    return bp.path_matrix(bp.network(adj, bp.cell(seed), gamma))


def test_walrus():
    """E[prod n_a^{k_a}] of the Gaussian output state, from its covariance,
    with The Walrus's photon_number_moment, against the hafnian tables."""
    from thewalrus.quantum import photon_number_moment
    from thewalrus.symplectic import squeezing, passive_transformation
    W = small_network()
    W = W[:4, :12]                          # 4 detectors, 12 slots: still a sub-block of a unitary
    n, slots = W.shape
    w = 0.3
    r = np.arcsinh(np.sqrt(w))
    S = squeezing(np.full(slots, r))         # symplectic matrix, phase 0 on every slot
    cov = S @ S.T                            # vacuum covariance is the identity at hbar = 2
    mu = np.zeros(2 * slots)
    mu_out, cov_out = passive_transformation(mu, cov, W)
    mom = sq.all_moments(W, 4)
    for N in (1, 2, 3, 4):
        rates, var = sq.combine(mom, N, w)[N]
        for S, F, V in zip(combinations(range(n), N), rates, var):
            idx = {a: 1 for a in S}
            F_ref = photon_number_moment(mu_out, cov_out, idx)
            V_ref = photon_number_moment(mu_out, cov_out, {a: 2 for a in S}) - F_ref ** 2
            assert np.isclose(F, F_ref, rtol=1e-9), (S, F, F_ref)
            assert np.isclose(V, V_ref, rtol=1e-9), (S, V, V_ref)


def test_thermal_and_real():
    """j = 0 is thermal light (photon_noise), weight one the real Gaussian
    amplitude (real_noise), on every multiplicity pattern up to order 3."""
    import torch
    W = small_network()
    G, T = sq.gram(W)
    Wt = torch.as_tensor(W, dtype=torch.complex128)
    for N in (1, 2, 3):
        mom = sq.moments(G, T, N)
        thermal = pn.moments_all(Wt, N, {"thermal": bp.LAMBDA["thermal"]}, "cpu")
        for d, m in mom.items():
            assert np.allclose(m[0], thermal[d]["thermal"], rtol=1e-10, atol=1e-14)
            if N + len(d) > 4:                      # real_noise tabulates cumulants up to 8 legs
                continue
            for S, total in zip(combinations(range(W.shape[0]), N), m.sum(axis=0)):
                dets = sorted(list(S) + [S[i] for i in d])
                assert np.isclose(total, rn.cpu_moment(W, dets, rn.CUMULANTS["gaussian real"]), rtol=1e-10)


def test_relabelling():
    adj = bp.adjacency(bp.load_brec(BREC)["Strongly_regular"][0][0])
    perm = np.random.default_rng(0).permutation(adj.shape[0])
    profiles = []
    for a in (adj, bp.relabel(adj, perm)):
        mom = sq.all_moments(bp.path_matrix(bp.network(a, bp.cell(100), 0.9)), 3)
        profiles.append({N: np.sort(r) for N, (r, v) in sq.combine(mom, 3, 0.1).items()})
    for N in (1, 2, 3):
        assert np.max(np.abs(profiles[0][N] - profiles[1][N])) < 1e-13 * profiles[0][N].max()


BREC = "/home/user/graphpku/brec/customize/Data/raw"

if __name__ == "__main__":
    for t in (test_counts, test_walrus, test_thermal_and_real, test_relabelling):
        t()
        print(t.__name__, "ok", flush=True)
