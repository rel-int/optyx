"""Independent checks of the profile code on small graphs."""
import os
import sys
import time
from itertools import combinations, permutations

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import brecphot as bp  # noqa: E402

BREC = "/home/user/graphpku/brec/customize/Data/raw"


def permanent(m):
    n = m.shape[0]
    return sum(np.prod([m[i, p[i]] for i in range(n)]) for p in permutations(range(n)))


def cycle(n):
    adj = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        adj[i, (i + 1) % n] = adj[(i + 1) % n, i] = 1
    return adj


def two_triangles():
    adj = np.zeros((6, 6), dtype=np.int8)
    for block in ([0, 1, 2], [3, 4, 5]):
        for i in block:
            for j in block:
                if i != j:
                    adj[i, j] = 1
    return adj


def brute(W, S, source):
    """Sum over slot subsets of the permanent formulas, for photons."""
    slots = W.shape[1]
    total = 0.0
    for C in combinations(range(slots), len(S)):
        sub = W[np.ix_(S, C)]
        if source == "photons p=1":
            total += abs(permanent(sub)) ** 2
        else:
            total += permanent(abs(sub) ** 2).real
    return total


def coherent_mc(W, S, w, samples, rng):
    slots = W.shape[1]
    acc = 0.0
    for _ in range(samples):
        alpha = np.sqrt(w) * np.exp(2j * np.pi * rng.random(slots))
        beta = W @ alpha
        counts = rng.poisson(abs(beta) ** 2)
        acc += np.prod(counts[list(S)])
    return acc / samples


def main():
    gamma, w = 0.3, 0.2
    cells = bp.cell(7)
    adj = cycle(6)
    W = bp.path_matrix(bp.network(adj, cells, gamma))
    n, slots = W.shape
    print("slots", slots)
    K = W @ W.conj().T
    tensors = {N: bp.rate_tensors(W, N) for N in (1, 2, 3)}
    for S in [(0,), (0, 1), (0, 3), (0, 1, 2), (0, 2, 4)]:
        N = len(S)
        th = tensors[N]["thermal"][S]
        assert np.isclose(th, permanent(K[np.ix_(S, S)]).real, rtol=1e-10), (S, th)
        if N <= 2:
            p1 = brute(W, S, "photons p=1")
            p0 = brute(W, S, "photons p=0")
            assert np.isclose(tensors[N]["photons p=1"][S], p1, rtol=1e-9), (S, tensors[N]["photons p=1"][S], p1)
            assert np.isclose(tensors[N]["photons p=0"][S], p0, rtol=1e-9), (S, tensors[N]["photons p=0"][S], p0)
    print("thermal = permanent, photons = sums of permanents: ok")
    # order 3 brute force for photons on a few sets (slots choose 3 permanents)
    t0 = time.time()
    S = (0, 1, 2)
    print("order-3 brute, p=1:", brute(W, S, "photons p=1"), tensors[3]["photons p=1"][S],
          "p=0:", brute(W, S, "photons p=0"), tensors[3]["photons p=0"][S], f"({time.time()-t0:.1f}s)")
    # coherent light by Monte Carlo
    rng = np.random.default_rng(0)
    for S in [(0, 1), (0, 3), (0, 1, 2)]:
        N = len(S)
        exact = tensors[N]["coherent"][S] * w ** N
        mc = coherent_mc(W, S, w, 200000, rng)
        print(f"coherent {S}: exact {exact:.6g} MC {mc:.6g} ratio {mc/exact:.4f}")
    # relabelling invariance and a separation: C6 against 2C3
    prof_g, _ = bp.network_profiles(adj, cells, gamma, w, 3)
    perm = np.random.default_rng(1).permutation(6)
    prof_gp, _ = bp.network_profiles(bp.relabel(adj, perm), cells, gamma, w, 3)
    prof_h, _ = bp.network_profiles(two_triangles(), cells, gamma, w, 3)
    for N in (1, 2, 3):
        rel = np.max(np.abs(prof_g[N] - prof_gp[N]) / np.max(prof_g[N]))
        print(f"order {N}: relabel rel diff {rel:.2e}; gap C6 vs 2C3 per source",
              [f"{bp.gap(prof_g[N][i], prof_h[N][i]):.3g}" for i in range(4)])
    # timing on BREC sizes
    data = bp.load_brec(BREC)
    for cat, idx, nmax in [("Strongly_regular", 0, 4), ("4-vertex_condition", 0, 3), ("CFI", 76, 3)]:
        g6 = data[cat][idx][0]
        adj = bp.adjacency(g6)
        t0 = time.time()
        prof, slots = bp.network_profiles(adj, bp.cell(100), 0.9, 0.05, nmax)
        print(f"{cat}_{idx}: n={adj.shape[0]} slots={slots} nmax={nmax} {time.time()-t0:.1f}s",
              {N: prof[N].shape for N in prof})
    # cached run comparison: Basic_0 seed 100, order 1 of G per source divided by w
    adj = bp.adjacency(data["Basic"][0][0])
    cached = [0.7721762542368246, 0.7731683879637361, 0.7731683879637361, 0.7721762542368249,
              0.7743190558799391, 0.7744494552631195, 0.7734467921461317, 0.7734467921461317,
              0.7749242411690676, 0.7744494552631196]
    W = bp.path_matrix(bp.network(adj, bp.cell(100), 0.9))
    mine = bp.rate_tensors(W, 1)["photons p=1"]
    print("Basic_0 order 1, cached sorted:", np.round(np.sort(cached), 5))
    print("Basic_0 order 1, mine   sorted:", np.round(np.sort(mine), 5))


if __name__ == "__main__":
    main()
