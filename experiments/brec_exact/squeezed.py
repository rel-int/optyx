"""Exact rates and noise of Gaussian squeezed light.

Every slot (a source and a delay) holds a single-mode squeezed vacuum of
mean photon number w = sinh^2 r, with the same squeezing phase for every
source and tick: the input is a Gaussian state, and so is the output.  With
b = W a at the detectors, its normally ordered contractions are

    <b_i^dag b_j> = w G[j, i],    G = W W^dag,
    <b_i b_j>     = mu T[i, j],   T = W W^T,      mu = sqrt(w (1 + w)),
    <b_i^dag b_j^dag> = mu T*[i, j],

and every normally ordered moment of the counts is a hafnian of them
(Wick's theorem): E[prod_a (b_a^dag)^{m_a} b_a^{m_a}] is a sum over the perfect
matchings of the 2K legs, K = sum m_a, of products of contractions.  The
loss and the unused output modes are vacuum and contribute nothing.  A
matching with j pairs of annihilators has j pairs of creators, so

    E[...] / w^K = sum_j (1 + 1/w)^j m_j,

with m_j a polynomial in the entries of G and T on the detectors of the
set, independent of w.  The global squeezing phase cancels.  The weight
(1 + 1/w)^j, not 1, is what separates squeezed light from the phase-locked
classical Gaussian amplitude of `real_noise.py` (mu = w), and j = 0 alone is
thermal light (mu = 0), a permanent of G.

Positions sharing a detector are interchangeable, so the matchings are
enumerated over multiplicity vectors, once per multiplicity pattern, and
cached as monomials in the N^2 entries of G and the N(N+1)/2 entries of T
and of T* on an N-set.  Evaluating a set then needs only those entries:
no sum over the slots of the path matrix, which is done once, in the two
n x n Gram matrices.
"""
from functools import lru_cache
from itertools import combinations

import numpy as np


def factor_ids(N):
    """Factor ids on an N-set: G[i, j] -> i N + j; T[i, j] and T*[i, j]
    (i <= j) -> N^2 + t and N^2 + N(N+1)/2 + t with t the triangular index."""
    tri = {}
    for i in range(N):
        for j in range(i, N):
            tri[(i, j)] = len(tri)
    return tri, N * N, N * N + len(tri), N * N + 2 * len(tri)


@lru_cache(maxsize=None)
def tables(N, doubled):
    """Monomials of the normally ordered moment of an N-set whose positions
    in `doubled` carry two photons.  Returns keys (n_keys, K) of factor ids
    padded with -1, counts (n_keys,) and j (n_keys,), the number of T
    factors of each key."""
    m = tuple(2 if i in doubled else 1 for i in range(N))
    K = sum(m)
    tri, t0, tc0, _ = factor_ids(N)

    @lru_cache(maxsize=None)
    def match(plain, dag):
        """dict monomial -> number of matchings of the remaining legs."""
        if not any(plain) and not any(dag):
            return {(): 1}
        out = {}
        if any(plain):
            i = next(k for k, c in enumerate(plain) if c)
            p = list(plain)
            p[i] -= 1
            options = []
            for k in range(N):
                c = p[k]
                if c:
                    q = list(p)
                    q[k] -= 1
                    options.append((c, t0 + tri[min(i, k), max(i, k)], tuple(q), dag))
                if dag[k]:
                    d = list(dag)
                    d[k] -= 1
                    options.append((dag[k], i * N + k, tuple(p), tuple(d)))   # <b_k^dag b_i> = w G[i, k]
        else:
            i = next(k for k, c in enumerate(dag) if c)
            d = list(dag)
            d[i] -= 1
            options = []
            for k in range(N):
                c = d[k]
                if c:
                    e = list(d)
                    e[k] -= 1
                    options.append((c, tc0 + tri[min(i, k), max(i, k)], plain, tuple(e)))
        for count, factor, rest_p, rest_d in options:
            for mono, c in match(rest_p, rest_d).items():
                key = tuple(sorted(mono + (factor,)))
                out[key] = out.get(key, 0) + count * c
        return out

    monos = match(m, m)
    keys = np.array(sorted(monos), dtype=np.int32).reshape(-1, K)
    counts = np.array([monos[tuple(k)] for k in keys], dtype=float)
    j = np.sum((keys >= t0) & (keys < tc0), axis=1)
    return keys, counts, j


def gram(W):
    """G = W W^dag and T = W W^T, the only sums over the slots."""
    return W @ W.conj().T, W @ W.T


def set_values(G, T, idx):
    """The factor values on a chunk of N-sets idx (N, chunk): (n_factors + 1,
    chunk), the last row ones for padding."""
    import torch
    N = idx.shape[0]
    rows = [G[idx[i], idx[j]] for i in range(N) for j in range(N)]
    tvals = [T[idx[i], idx[j]] for i in range(N) for j in range(i, N)]
    rows += tvals + [t.conj() for t in tvals]
    rows.append(torch.ones_like(rows[0]))
    return torch.stack(rows)


@lru_cache(maxsize=None)
def plan(N, doubles):
    """All the monomials of order N in one list, split in a prefix and a
    suffix of N factors each (padded with the ones row): the unique
    prefixes and suffixes as tries, level by level (parent, factor), every
    key as (prefix, suffix, output row, count), the output row being
    |doubled| (N + 1) + j and the doubled patterns listed in `doubles`."""
    nf = factor_ids(N)[3]
    K = 2 * N
    keys, rows, counts = [], [], []
    for r, d in enumerate(doubles):
        k, c, j = tables(N, d)
        padded = np.full((len(k), K), nf, dtype=np.int32)
        padded[:, :k.shape[1]] = k
        keys.append(padded)
        rows.append(r * (N + 1) + j)
        counts.append(c)
    keys, rows, counts = np.concatenate(keys), np.concatenate(rows), np.concatenate(counts)

    def trie(part):
        levels, ids = [], np.zeros(len(part), dtype=np.int64)
        for level in range(part.shape[1]):
            pairs = np.stack([ids, part[:, level]], axis=1)
            uniq, ids = np.unique(pairs, axis=0, return_inverse=True)
            levels.append(uniq)
            ids = ids.ravel()
        return levels, ids

    prefixes, pre = trie(keys[:, :N])
    suffixes, suf = trie(keys[:, N:])
    return prefixes, suffixes, pre, suf, rows, counts, len(doubles) * (N + 1)


def products(values, levels):
    """Products of the unique words of a trie on the factor values."""
    prod = values[levels[0][:, 1]]
    for level in levels[1:]:
        prod = prod[level[:, 0]] * values[level[:, 1]]
    return prod


def moments(G, T, N, doubles=None, chunk=4096, key_batch=1 << 16):
    """dict doubled -> array (N + 1, M_N) of the components m_j of
    E[prod_{a in doubled} n_a (n_a - 1) prod_{others} n_a] / w^K on every
    N-set, from numpy or torch arrays G, T (n x n).  The work per set is
    about one multiply-add per monomial, whatever its degree."""
    import torch
    G, T = (torch.as_tensor(x) for x in (G, T))
    n, device = G.shape[0], G.device
    if doubles is None:
        doubles = [d for size in range(N + 1) for d in combinations(range(N), size)]
    doubles = tuple(tuple(d) for d in doubles)
    prefixes, suffixes, pre, suf, rows, counts, n_rows = plan(N, doubles)
    as_t = lambda a, dtype=torch.long: torch.as_tensor(a, device=device, dtype=dtype)
    prefixes = [as_t(x) for x in prefixes]
    suffixes = [as_t(x) for x in suffixes]
    pre, suf, rows = as_t(pre), as_t(suf), as_t(rows)
    counts = as_t(counts, G.dtype)
    index = as_t(np.array(list(combinations(range(n), N))).T)
    M = index.shape[1]
    out = torch.zeros((n_rows, M), dtype=torch.float64, device=device)
    for start in range(0, M, chunk):
        idx = index[:, start:start + chunk]
        values = set_values(G, T, idx)
        P, S = products(values, prefixes), products(values, suffixes)
        acc = torch.zeros((n_rows, idx.shape[1]), dtype=G.dtype, device=device)
        for k in range(0, len(rows), key_batch):
            sl = slice(k, k + key_batch)
            acc.index_add_(0, rows[sl], counts[sl, None] * P[pre[sl]] * S[suf[sl]])
        out[:, start:start + idx.shape[1]] = acc.real
    out = out.cpu().numpy()
    return {d: out[r * (N + 1):(r + 1) * (N + 1)] for r, d in enumerate(doubles)}


def all_moments(W, nmax, **kw):
    """moments[(N, doubled)] -> array (N + 1, M_N), for every order up to nmax."""
    import torch
    G, T = gram(torch.as_tensor(W) if isinstance(W, np.ndarray) else W)
    out = {}
    for N in range(1, nmax + 1):
        for d, m in moments(G, T, N, **kw).items():
            out[(N, d)] = m
    return out


def weight(w, J, kind="squeezed"):
    """(mu / w)^(2j) for j < J: squeezed (mu^2 = w (1 + w)), the classical
    real Gaussian (mu = w) or thermal light (mu = 0)."""
    j = np.arange(J)
    if kind == "squeezed":
        return (1 + 1 / w) ** j
    if kind == "gaussian real":
        return np.ones(J)
    if kind == "thermal":
        return (j == 0).astype(float)
    raise ValueError(kind)


def combine(mom, nmax, w, kind="squeezed"):
    """Rates and variances at brightness w, per order: dict N -> (F, Var)."""
    out = {}
    for N in range(1, nmax + 1):
        rates = w ** N * (weight(w, N + 1, kind) @ mom[(N, ())])
        second = np.zeros_like(rates)
        for (M, d), m in mom.items():
            if M == N:
                second += w ** (N + len(d)) * (weight(w, N + 1, kind) @ m)
        out[N] = (rates, np.maximum(second - rates ** 2, 0.0))
    return out


def ideal_counts(mom_g, mom_h, nmax, w, kind="squeezed", beta=0.05):
    """Cumulative ideal sample counts over orders 1..N, as photon_noise."""
    rg, rh = combine(mom_g, nmax, w, kind), combine(mom_h, nmax, w, kind)
    out, d2 = {}, 0.0
    for N in range(1, nmax + 1):
        (fg, vg), (fh, vh) = rg[N], rh[N]
        og, oh = np.argsort(fg), np.argsort(fh)
        noise = vg[og] + vh[oh]
        diff = fg[og] - fh[oh]
        mask = noise > 0
        d2 += float(np.sum(diff[mask] ** 2 / noise[mask]))
        out[N] = 4 * np.log(1 / beta) / d2 if d2 > 0 else float("inf")
    return out


def gaps(mom_g, mom_h, nmax, w, kind="squeezed"):
    """Largest difference of the sorted rates over the largest rate, per order."""
    out = {}
    for N in range(1, nmax + 1):
        fg = np.sort(w ** N * (weight(w, N + 1, kind) @ mom_g[(N, ())]))
        fh = np.sort(w ** N * (weight(w, N + 1, kind) @ mom_h[(N, ())]))
        out[N] = float(np.max(np.abs(fg - fh)) / max(fg.max(), fh.max()))
    return out
