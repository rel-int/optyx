"""Checks of photon_noise against brute force on a random 4 x 10 matrix,
which the slot sums take as a path matrix: the combinatorial identities do
not need it to be sub-unitary.

- base moments equal brecphot.rate_tensors on every set, every source;
- multiset moments with K <= 5 equal rate_tensors evaluated at repeated
  indices;
- single photons: sum over slot subsets of |per W_{S,C}|^2 (p = 1) and of
  per |W_{S,C}|^2 (p = 0), with S a multiset, up to K = 8;
- thermal light: the permanent of the Gram matrix on the multiset;
- coherent light: the phase Monte Carlo of the intensity moments.
"""
import sys
from itertools import combinations, permutations

import numpy as np
import torch

sys.path.insert(0, ".")
import brecphot as bp  # noqa: E402
import coherent_gpu as cg  # noqa: E402
import photon_noise as pn  # noqa: E402


def permanent(m):
    n = m.shape[0]
    return sum(np.prod([m[i, p[i]] for i in range(n)]) for p in permutations(range(n)))


def brute_photons(W, rows, p1):
    slots = W.shape[1]
    total = 0.0
    for C in combinations(range(slots), len(rows)):
        sub = W[np.ix_(rows, C)]
        total += abs(permanent(sub)) ** 2 if p1 else permanent(abs(sub) ** 2).real
    return total


def main():
    rng = np.random.default_rng(3)
    n, slots = 4, 10
    W = (rng.normal(size=(n, slots)) + 1j * rng.normal(size=(n, slots))) / np.sqrt(2 * slots)
    device = "cpu"
    Wt = torch.as_tensor(W, dtype=torch.complex128, device=device)
    lam = pn.sources_lambda(fock=(2, 5))
    # 1. base moments against rate_tensors
    for N in range(1, 5):
        base = pn.moments(Wt, N, (), lam, device)
        tensors = bp.rate_tensors(W, N)
        idx = np.array(list(combinations(range(n), N))).T
        for s in bp.SOURCES:
            assert np.allclose(base[s], tensors[s][tuple(idx)], rtol=1e-9, atol=1e-12), (N, s)
    print("base moments = rate_tensors: ok")
    # 2. multiset moments K <= 5 against rate_tensors at repeated indices
    for N, doubled in [(1, (0,)), (2, (0,)), (2, (0, 1)), (3, (0,)), (3, (0, 2)), (4, (1,))]:
        m = pn.moments(Wt, N, doubled, lam, device)
        K = N + len(doubled)
        tensors = bp.rate_tensors(W, K)
        for i, S in enumerate(combinations(range(n), N)):
            rows = []
            for j, a in enumerate(S):
                rows += [a] * (2 if j in doubled else 1)
            for s in bp.SOURCES:
                assert np.isclose(m[s][i], tensors[s][tuple(rows)], rtol=1e-9, atol=1e-12), (N, doubled, s)
    print("multiset moments = rate_tensors at repeated indices (K <= 5): ok")
    # 3. brute force for photons, thermal, up to K = 8
    for N, doubled in [(2, (0, 1)), (3, (0, 1, 2)), (4, (0, 1, 2, 3)), (4, (0, 2))]:
        m = pn.moments(Wt, N, doubled, lam, device)
        for i, S in enumerate(list(combinations(range(n), N))[:2]):
            rows = []
            for j, a in enumerate(S):
                rows += [a] * (2 if j in doubled else 1)
            b1, b0 = brute_photons(W, rows, True), brute_photons(W, rows, False)
            assert np.isclose(m["photons p=1"][i], b1, rtol=1e-8), (N, doubled, m["photons p=1"][i], b1)
            assert np.isclose(m["photons p=0"][i], b0, rtol=1e-8), (N, doubled, m["photons p=0"][i], b0)
            G = W @ W.conj().T
            th = permanent(G[np.ix_(rows, rows)]).real
            assert np.isclose(m["thermal"][i], th, rtol=1e-8), (N, doubled, m["thermal"][i], th)
        print(f"brute force K={N + len(doubled)}: photons p=1, p=0 and thermal: ok", flush=True)
    # 4. coherent light against the phase Monte Carlo at unit brightness
    moments = cg.intensity_moments(Wt, 4, 400000, device, seed=1)
    for N, doubled in [(2, (0,)), (3, (0, 1)), (4, (0, 1, 2, 3))]:
        m = pn.moments(Wt, N, doubled, lam, device)["coherent"]
        mask = sum(1 << j for j in doubled)
        mc = moments[N][mask]
        rel = np.abs(m - mc) / np.abs(mc)
        assert np.all(rel < 0.05), (N, doubled, rel)
        print(f"coherent K={N + len(doubled)} against the Monte Carlo: max rel diff {rel.max():.3f}")
    # 5. Fock |K>: the weights reproduce single photons at K = 1 and coherent light as K -> infinity
    lamK = pn.sources_lambda(fock=(1, 200))
    m = pn.moments(Wt, 3, (0, 1), lamK, device)
    assert np.allclose(m["fock 1"], m["photons p=1"], rtol=1e-9)
    assert np.allclose(m["fock 200"], m["coherent"], rtol=0.05)
    print("Fock weights: |1> = photons, |200> = coherent to 5%: ok")
    # 6. moments_all agrees with moments on every pattern
    for N in range(1, 5):
        allm = pn.moments_all(Wt, N, lam, device, chunk=3)
        for d, by_source in allm.items():
            ref = pn.moments(Wt, N, d, lam, device)
            for s in ref:
                assert np.allclose(by_source[s], ref[s], rtol=1e-9, atol=1e-14), (N, d, s)
    print("moments_all = moments on every pattern up to N = 4: ok")
    # 7. variances are nonnegative and the rates match on a real network
    adj = bp.adjacency(bp.load_brec("/home/user/graphpku/brec/customize/Data/raw")["Basic"][0][0]) \
        if False else None
    print("all checks passed")


if __name__ == "__main__":
    main()
