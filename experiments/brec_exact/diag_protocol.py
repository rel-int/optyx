"""Where does the protocol lose against the ideal test?  Synthetic batches
from exact coherent moments on one pair, several statistics."""
import sys
from itertools import combinations
from math import log
import numpy as np
sys.path.insert(0, ".")
import brecphot as bp

data = bp.load_brec("/home/user/graphpku/brec/customize/Data/raw")
cat, i = sys.argv[1], int(sys.argv[2])
w, nmax, seed, B = 100.0, int(sys.argv[3]) if len(sys.argv) > 3 else 3, 100, 16
g6, h6 = data[cat][i]
rng = np.random.default_rng(0)

def moments(adj):
    W = bp.path_matrix(bp.network(adj, bp.cell(seed), 0.9))
    n = W.shape[0]
    out = {}
    for N in range(1, nmax + 1):
        idx = np.array(list(combinations(range(n), N))).T
        mean = bp.rate_tensors(W, N, ("coherent",))["coherent"][tuple(idx)] * w ** N
        # phase Monte Carlo of the intensity moments at unit brightness
        K = 20000
        acc = np.zeros((2 ** N, idx.shape[1]))
        for start in range(0, K, 2000):
            ph = np.exp(2j * np.pi * rng.random((2000, W.shape[1])))
            J = np.abs(ph @ W.T) ** 2
            for mask in range(2 ** N):
                prod = np.ones((2000, idx.shape[1]))
                for k in range(N):
                    prod = prod * (J ** 2 if mask >> k & 1 else J)[:, idx[k]]
                acc[mask] += prod.sum(0)
        acc /= K
        second = sum(w ** (N + bin(m).count("1")) * acc[m] for m in range(2 ** N))
        var = np.maximum(second - (w ** N * acc[0]) ** 2, 0)
        out[N] = (mean, var)
    return out

mg, mh = moments(bp.adjacency(g6)), moments(bp.adjacency(h6))
print(f"{cat}_{i}: n={bp.adjacency(g6).shape[0]}")
ideal = {}
for N in range(1, nmax + 1):
    d2 = 0.0
    for M in range(1, N + 1):
        fg, fh = np.sort(mg[M][0]), np.sort(mh[M][0])
        vg, vh = mg[M][1][np.argsort(mg[M][0])], mh[M][1][np.argsort(mh[M][0])]
        d2 += np.sum((fg - fh) ** 2 / (vg + vh))
    ideal[N] = 4 * log(20) / d2 if d2 > 0 else np.inf
    gap = {M: f"{np.max(np.abs(np.sort(mg[M][0]) - np.sort(mh[M][0]))) / np.sort(mg[M][0]).max():.1e}" for M in range(1, N + 1)}
print("ideal diagonal counts, cumulative orders:", {N: f"{v:.1e}" for N, v in ideal.items()}, "rel gaps", gap)

def draw(m, T):
    """B batch profiles (sorted sqrt of estimates) and the raw estimates."""
    out = {}
    for N in range(1, nmax + 1):
        mean, var = m[N]
        est = np.maximum(mean[None, :] + rng.normal(size=(B, mean.shape[0])) * np.sqrt(var / (T / B)), 0)
        out[N] = est
    return out

def stats_for(eg, eh, kind, splits=499):
    """p-value of one statistic by permutation."""
    allf = np.concatenate([eg, eh])            # (2B, d)
    m = B
    masks = np.zeros((splits + 1, 2 * m)); masks[0, :m] = 1
    for r in range(1, splits + 1):
        masks[r, rng.choice(2 * m, m, replace=False)] = 1
    a, b_ = masks / m, (1 - masks) / m
    ma, mb = a @ allf, b_ @ allf
    diff = ma - mb
    if kind == "euclid":
        d = np.sum(diff ** 2, axis=1)
    else:
        sq = allf * allf
        va = np.maximum(a @ sq - ma ** 2, 0); vb = np.maximum(b_ @ sq - mb ** 2, 0)
        v = (va + vb) / 2
        if kind == "student-raw":
            d = np.sum(diff ** 2 / (v + 1e-300), axis=1)
        elif kind == "student-fit":
            mm = (ma + mb) / 2
            d = np.empty(v.shape[0])
            for r in range(v.shape[0]):
                X = np.stack([np.ones_like(mm[r]), mm[r], mm[r] ** 2], 1)
                coef, *_ = np.linalg.lstsq(X, v[r], rcond=None)
                fit = np.maximum(X @ coef, 0) + 1e-6 * v[r].mean() + 1e-300
                d[r] = np.sum(diff[r] ** 2 / fit)
    return (1 + np.sum(d[1:] >= d[0])) / (splits + 1)

def trial(T, variant):
    eg, eh = draw(mg, T), draw(mh, T)
    def prof(est): return np.sort(np.sqrt(est), axis=1)
    def logprof(est): return np.sort(np.log(est + 1e-300), axis=1)
    ps = []
    if variant == "protocol":            # per order, euclid on sqrt profiles, Bonferroni over orders
        ps = [stats_for(prof(eg[N]), prof(eh[N]), "euclid") for N in range(1, nmax + 1)]
        return min(ps) <= 0.05 / len(ps)
    if variant == "student-per-order":
        ps = [stats_for(prof(eg[N]), prof(eh[N]), "student-fit") for N in range(1, nmax + 1)]
        return min(ps) <= 0.05 / len(ps)
    if variant == "student-raw-per-order":
        ps = [stats_for(prof(eg[N]), prof(eh[N]), "student-raw") for N in range(1, nmax + 1)]
        return min(ps) <= 0.05 / len(ps)
    if variant == "student-cumulative":   # all orders in one vector
        F_g = np.concatenate([prof(eg[N]) for N in range(1, nmax + 1)], 1)
        F_h = np.concatenate([prof(eh[N]) for N in range(1, nmax + 1)], 1)
        return stats_for(F_g, F_h, "student-raw") <= 0.05
    if variant == "log-cumulative":
        F_g = np.concatenate([logprof(eg[N]) for N in range(1, nmax + 1)], 1)
        F_h = np.concatenate([logprof(eh[N]) for N in range(1, nmax + 1)], 1)
        return stats_for(F_g, F_h, "student-raw") <= 0.05
    if variant == "oracle":               # known variances of the sorted entries, cumulative, z-test
        d2 = 0.0
        for N in range(1, nmax + 1):
            fg, fh = np.sort(eg[N].mean(0)), np.sort(eh[N].mean(0))
            vg = mg[N][1][np.argsort(mg[N][0])] / T; vh = mh[N][1][np.argsort(mh[N][0])] / T
            d2 += np.sum((fg - fh) ** 2 / (vg + vh))
        # under the null d2 ~ chi2 with d dof; reject at the 95% quantile
        dof = sum(mg[N][0].shape[0] for N in range(1, nmax + 1))
        from scipy.stats import chi2
        return d2 > chi2.ppf(0.95, dof)

variants = ["protocol", "student-per-order", "student-raw-per-order", "student-cumulative", "log-cumulative", "oracle"]
budgets = [10 ** k for k in range(3, 11)]
print(f"{'T':>8} " + " ".join(f"{v:>22s}" for v in variants))
for T in budgets:
    row = []
    for v in variants:
        hits = sum(trial(T, v) for _ in range(6))
        row.append(f"{hits / 6:22.2f}")
    print(f"{T:8.0e} " + " ".join(row), flush=True)
