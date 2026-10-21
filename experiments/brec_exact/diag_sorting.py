"""Is the factor between the protocol and the ideal count the sorting?
The ideal count matches the sorted exact profiles entry by entry; the
protocol sorts noisy estimates, whose order statistics converge to the
exact sorted profile only once the noise is below the spacing of the
entries.  This measures the attenuation of the sorted gap under noise and
tries statistics that are linear in the estimates: the power sums of the
rates (the moments of the profile distribution), estimated without bias
by products over sub-batches."""
import sys
from itertools import combinations
from math import log
import numpy as np
from scipy.stats import chi2
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
n = bp.adjacency(g6).shape[0]
print(f"{cat}_{i}: n={n}")
for N in range(1, nmax + 1):
    fg, fh = np.sort(mg[N][0]), np.sort(mh[N][0])
    d = fg.shape[0]
    spacing = np.median(np.diff(np.unique(np.round(fg, 12))))  if np.unique(np.round(fg, 12)).size > 1 else np.inf
    d2 = np.sum((fg - fh) ** 2 / (mg[N][1][np.argsort(mg[N][0])] + mh[N][1][np.argsort(mh[N][0])]))
    print(f"order {N}: d={d}, distinct values {np.unique(np.round(fg, 12)).size}, "
          f"median spacing/level {spacing / fg.mean():.1e}, sorted gap/level {np.max(np.abs(fg - fh)) / fg.mean():.1e}, "
          f"noise/level at T=1e8: {np.sqrt(mg[N][1].mean() / 1e8) / fg.mean():.1e}, "
          f"ideal count (this order alone) {4 * log(20) / d2 if d2 > 0 else np.inf:.1e}, "
          f"dimension penalty x{1.645 * np.sqrt(2 * d) / (4 * log(20)):.0f}, "
          f"power sums G-H: " + " ".join(f"{(np.sum(fg ** k) - np.sum(fh ** k)) / np.sum(fg ** k):.1e}" for k in (1, 2, 3)))

def draw(m, T, sub=1):
    """B batches, each `sub` sub-batches of T/(B sub) samples: (B, sub, d)."""
    out = {}
    for N in range(1, nmax + 1):
        mean, var = m[N]
        out[N] = np.maximum(mean[None, None, :] + rng.normal(size=(B, sub, mean.shape[0])) * np.sqrt(var / (T / B / sub)), 0)
    return out

# attenuation of the sorted gap: E[sorted(est_G) - sorted(est_H)] against the exact one
print("attenuation of the sorted gap, |E sorted(est_G) - E sorted(est_H)| / |sorted(G) - sorted(H)|, per order:")
for T in [1e5, 1e6, 1e7, 1e8, 1e9]:
    row = []
    for N in range(1, nmax + 1):
        eg, eh = draw(mg, T)[N][:, 0], draw(mh, T)[N][:, 0]
        exact = np.sort(mg[N][0]) - np.sort(mh[N][0])
        noisy = np.sort(eg, axis=1).mean(0) - np.sort(eh, axis=1).mean(0)
        row.append(f"{np.linalg.norm(noisy) / np.linalg.norm(exact):.2f}" if np.linalg.norm(exact) > 0 else "nan")
    print(f"  T={T:.0e}: " + " ".join(row))

def perm_p(Fg, Fh, kind="student", splits=499):
    allf = np.concatenate([Fg, Fh]); m = B
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
        v = (np.maximum(a @ sq - ma ** 2, 0) + np.maximum(b_ @ sq - mb ** 2, 0)) / 2 + 1e-300
        d = np.max(diff ** 2 / v, axis=1) if kind == "max" else np.sum(diff ** 2 / v, axis=1)
    return (1 + np.sum(d[1:] >= d[0])) / (splits + 1)

def prof(est): return np.sort(np.sqrt(est[:, 0]), axis=1)

def power_sums(est, kmax=3):
    """Unbiased per-batch estimates of sum_i r_i^k, k=1..kmax, from products
    over distinct sub-batches, every k using the first k sub-batches."""
    return np.stack([np.sum(np.prod(est[:, :k, :], axis=1), axis=1) for k in range(1, kmax + 1)], 1)

def trial(T, variant):
    if variant == "protocol":
        eg, eh = draw(mg, T), draw(mh, T)
        ps = [perm_p(prof(eg[N]), prof(eh[N]), "euclid") for N in range(1, nmax + 1)]
        return min(ps) <= 0.05 / len(ps)
    if variant == "split-sorted":
        # sort each batch by its first half, read the values from the second:
        # unbiased for the exact values at the sorted positions, no order-statistics bias
        eg, eh = draw(mg, T, sub=2), draw(mh, T, sub=2)
        def split(est):
            order = np.argsort(est[:, 0], axis=1)
            return np.sqrt(np.take_along_axis(est[:, 1], order, axis=1))
        ps = [perm_p(split(eg[N]), split(eh[N]), "euclid") for N in range(1, nmax + 1)]
        return min(ps) <= 0.05 / len(ps)
    if variant == "split-student":
        eg, eh = draw(mg, T, sub=2), draw(mh, T, sub=2)
        def split(est):
            order = np.argsort(est[:, 0], axis=1)
            return np.sqrt(np.take_along_axis(est[:, 1], order, axis=1))
        ps = [perm_p(split(eg[N]), split(eh[N]), "student") for N in range(1, nmax + 1)]
        return min(ps) <= 0.05 / len(ps)
    if variant == "max-entry":
        eg, eh = draw(mg, T), draw(mh, T)
        ps = [perm_p(prof(eg[N]), prof(eh[N]), "max") for N in range(1, nmax + 1)]
        return min(ps) <= 0.05 / len(ps)
    if variant.startswith("moments"):
        kmax = int(variant[-1])
        eg, eh = draw(mg, T, sub=kmax), draw(mh, T, sub=kmax)
        Fg = np.concatenate([power_sums(eg[N], kmax) for N in range(1, nmax + 1)], 1)
        Fh = np.concatenate([power_sums(eh[N], kmax) for N in range(1, nmax + 1)], 1)
        # one studentised entry at a time, Bonferroni over the nmax * kmax features
        ps = [perm_p(Fg[:, [j]], Fh[:, [j]], "student") for j in range(Fg.shape[1])]
        return min(ps) <= 0.05 / len(ps)
    if variant == "oracle-sorted":
        eg, eh = draw(mg, T), draw(mh, T)
        d2, dof = 0.0, 0
        for N in range(1, nmax + 1):
            fg, fh = np.sort(eg[N][:, 0].mean(0)), np.sort(eh[N][:, 0].mean(0))
            vg = mg[N][1][np.argsort(mg[N][0])] / T; vh = mh[N][1][np.argsort(mh[N][0])] / T
            d2 += np.sum((fg - fh) ** 2 / (vg + vh)); dof += fg.shape[0]
        return d2 > chi2.ppf(0.95, dof)

variants = ["protocol", "split-sorted", "split-student", "max-entry", "oracle-sorted"]
budgets = [10 ** k for k in range(3, 11)]
print(f"{'T':>8} " + " ".join(f"{v:>14s}" for v in variants))
for T in budgets:
    row = []
    for v in variants:
        hits = sum(trial(T, v) for _ in range(8))
        row.append(f"{hits / 8:14.2f}")
    print(f"{T:8.0e} " + " ".join(row), flush=True)
