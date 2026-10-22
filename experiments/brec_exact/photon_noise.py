"""Exact noise of the coincidences of every source at any brightness.

The variance of prod_{a in S} n_a needs E[prod n_a^2] = sum over T subset of
S of the normally ordered moments E[prod_{a in T} n_a (n_a - 1) prod_{a in
S - T} n_a], i.e. the rates of Eq. (sources) of the paper evaluated on the
multiset of detectors that lists each a in T twice.  A moment of K = N + |T|
photons is a sum over set partitions tau of the K photons and permutations
rho, of prod_b g(|b|) omega(rho) prod_b G(P_b, Q_b), with G(P, Q) the Gram
sum over the slots of the path matrix with the rows of the photons of b and
of their images under rho.  With repeated detectors the Gram sums only see
positions in S, so every term is a product of at most K sums over at most N
positions, which are computed once per pair on all N-sets and combined with
coefficients that depend on the source alone: single photons, Fock states
|K>, coherent and thermal light share the tables and differ by g.

The enumeration over (tau, rho) is done once per (N, multiplicity pattern)
and cached: for K = 8 it is 4140 partitions times 40320 permutations, done
with numpy over the permutations of each partition.
"""
import json
import os
from functools import lru_cache
from itertools import combinations, permutations
from math import comb, factorial

import numpy as np

import brecphot as bp

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "noise_tables")


def lam_fock(K):
    """Normalised factorial moments of |K>: binom(K, m) / K^m."""
    return lambda m: comb(K, m) / K ** m if m <= K else 0.0


def sources_lambda(fock=()):
    out = dict(bp.LAMBDA)
    for K in fock:
        out[f"fock {K}"] = lam_fock(K)
    return out


def block_weights(lam, kmax):
    return np.array([0.0] + [bp.block_weight(lam, k) for k in range(1, kmax + 1)])


def enumerate_terms(positions):
    """For photons at the given positions (a list of length K with values
    in range(N)), the distinct products of Gram sums and, for every set
    partition shape, the number of (tau, rho) giving each product.

    Returns patterns: list of (p, q) multiplicity tuples over positions;
    keys: int array (n_keys, K) of pattern ids padded with -1; weights:
    dict (sizes, interfering) -> array (n_keys,) with `sizes` the sorted
    block sizes of tau and `interfering` True for all rho, False for the
    identity only, so that a source's coefficient is sum over sizes of
    prod g(size) times the weight."""
    K = len(positions)
    N = max(positions) + 1
    pos = np.array(positions)
    perms = np.array(list(permutations(range(K))), dtype=np.int16)     # rho
    inverse = np.argsort(perms, axis=1).astype(np.int16)               # rho^-1
    identity = np.all(perms == np.arange(K), axis=1)
    pattern_id = {}
    patterns = []
    keys_by_shape = {}
    for tau in bp.set_partitions(list(range(K))):
        sizes = tuple(sorted(len(b) for b in tau))
        ids = np.empty((perms.shape[0], len(tau)), dtype=np.int32)
        for j, b in enumerate(tau):
            p = tuple(np.bincount(pos[b], minlength=N))
            q_pos = pos[inverse[:, b]]                                  # (perms, |b|)
            # multiplicity vector of the positions of rho^-1(b), per permutation
            q = np.zeros((perms.shape[0], N), dtype=np.int8)
            for col in range(len(b)):
                np.add.at(q, (np.arange(perms.shape[0]), q_pos[:, col]), 1)
            qs, inv = np.unique(q, axis=0, return_inverse=True)
            local = np.empty(qs.shape[0], dtype=np.int32)
            for r, qq in enumerate(qs):
                key = (p, tuple(int(x) for x in qq))
                if key not in pattern_id:
                    pattern_id[key] = len(patterns)
                    patterns.append(key)
                local[r] = pattern_id[key]
            ids[:, j] = local[inv.ravel()]
        ids.sort(axis=1)
        for interfering, mask in ((True, np.ones(perms.shape[0], bool)), (False, identity)):
            rows = ids[mask]
            if rows.shape[0] == 0:
                continue
            uniq, counts = np.unique(rows, axis=0, return_counts=True)
            store = keys_by_shape.setdefault((sizes, interfering), {})
            for row, c in zip(map(tuple, uniq), counts):
                store[row] = store.get(row, 0) + int(c)
    all_keys = sorted({row for store in keys_by_shape.values() for row in store})
    key_index = {row: i for i, row in enumerate(all_keys)}
    keys = -np.ones((len(all_keys), K), dtype=np.int32)
    for row, i in key_index.items():
        keys[i, :len(row)] = row
    weights = {}
    for shape, store in keys_by_shape.items():
        w = np.zeros(len(all_keys))
        for row, c in store.items():
            w[key_index[row]] = c
        weights[shape] = w
    return patterns, keys, weights


@lru_cache(maxsize=None)
def tables(N, doubled):
    """Cached enumeration for an N-set whose positions in `doubled` (a tuple
    of position indices) are listed twice."""
    positions = []
    for j in range(N):
        positions += [j] * (2 if j in doubled else 1)
    os.makedirs(CACHE, exist_ok=True)
    path = f"{CACHE}/N{N}_" + "".join(map(str, doubled)) + ".npz"
    if os.path.exists(path):
        with np.load(path) as z:
            patterns = [(tuple(p), tuple(q)) for p, q in json.loads(str(z["patterns"]))]
            shapes = json.loads(str(z["shapes"]))
            weights = {(tuple(sizes), bool(inter)): np.asarray(row, dtype=float)
                       for (sizes, inter), row in zip(shapes, z["weights"])}
            return patterns, z["keys"], weights
    patterns, keys, weights = enumerate_terms(positions)
    np.savez(path, patterns=json.dumps([[list(p), list(q)] for p, q in patterns]),
             keys=keys, shapes=json.dumps([[list(k[0]), k[1]] for k in weights]),
             weights=np.array([np.asarray(v, dtype=float) for v in weights.values()]))
    return patterns, keys, weights


def coefficients(weights, lam_by_source, K):
    """Per source, the coefficient of every key: sum over block shapes of
    prod g(size) times the count, interference included or identity only."""
    out = {}
    for source, lam in lam_by_source.items():
        g = block_weights(lam, K)
        interfering = source != "photons p=0"
        coef = None
        for (sizes, inter), w in weights.items():
            if inter != interfering:
                continue
            term = np.prod([g[s] for s in sizes]) * w
            coef = term if coef is None else coef + term
        out[source] = coef
    return out


def gram_on_sets(W, Wc, pattern, idx):
    """G(P, Q) on the sets idx (N, chunk): sum over slots of the product
    over positions of W^p conj(W)^q at the set's detector."""
    p, q = pattern
    import torch
    total = None
    for j, (pj, qj) in enumerate(zip(p, q)):
        if pj == 0 and qj == 0:
            continue
        rows = W[idx[j]]                                    # (chunk, slots)
        factor = torch.ones_like(rows)
        for _ in range(pj):
            factor = factor * rows
        if qj:
            rowsc = Wc[idx[j]]
            for _ in range(qj):
                factor = factor * rowsc
        total = factor if total is None else total * factor
    return total.sum(dim=1)


def moments(W, N, doubled, lam_by_source, device, chunk=20000, key_batch=512):
    """E[prod_{a in doubled} n_a(n_a - 1) prod_{others} n_a] / w^K on every
    N-set, per source: dict source -> array (M_N,)."""
    import torch
    n = W.shape[0]
    K = N + len(doubled)
    patterns, keys, weights = tables(N, tuple(doubled))
    coef = coefficients(weights, lam_by_source, K)
    Wc = W.conj()
    index = torch.as_tensor(np.array(list(combinations(range(n), N))).T, device=device)
    M = index.shape[1]
    out = {s: np.zeros(M) for s in coef}
    keys_t = torch.as_tensor(keys, device=device, dtype=torch.long)
    coef_t = {s: torch.as_tensor(c, device=device, dtype=torch.complex128) for s, c in coef.items()}
    blocks = (keys >= 0).sum(axis=1)
    for start in range(0, M, chunk):
        idx = index[:, start:start + chunk]
        size = idx.shape[1]
        values = torch.stack([gram_on_sets(W, Wc, pat, idx) for pat in patterns])   # (patterns, size)
        values = torch.cat([values, torch.ones((1, size), dtype=values.dtype, device=device)])
        padded = torch.where(keys_t >= 0, keys_t, torch.full_like(keys_t, len(patterns)))
        acc = {s: torch.zeros(size, dtype=torch.complex128, device=device) for s in coef}
        for kstart in range(0, keys.shape[0], key_batch):
            ids = padded[kstart:kstart + key_batch]                       # (kb, K)
            prod = values[ids[:, 0]]
            for col in range(1, ids.shape[1]):
                prod = prod * values[ids[:, col]]
            for s in coef:
                acc[s] += coef_t[s][kstart:kstart + key_batch] @ prod
        for s in coef:
            out[s][start:start + size] = acc[s].real.cpu().numpy()
    return out


def rates_and_variances(W, nmax, lam_by_source, brightness, device, fock=()):
    """For every order N <= nmax: the rates F_S and the variances
    Var[prod n_a] of every N-set, per source, at the given brightness
    (Fock sources at their own brightness K), as arrays (M_N,)."""
    out = {}
    for N in range(1, nmax + 1):
        base = moments(W, N, (), lam_by_source, device)
        second = {s: np.zeros_like(v) for s, v in base.items()}
        for size in range(0, N + 1):
            for doubled in combinations(range(N), size):
                m = base if size == 0 else moments(W, N, doubled, lam_by_source, device)
                for s in m:
                    w = float(s.split()[1]) if s.startswith("fock") else brightness
                    second[s] += w ** (N + size) * m[s]
        rates, var = {}, {}
        for s in base:
            w = float(s.split()[1]) if s.startswith("fock") else brightness
            rates[s] = w ** N * base[s]
            var[s] = np.maximum(second[s] - rates[s] ** 2, 0.0)
        out[N] = (rates, var)
    return out


def ideal_counts(rv_g, rv_h, nmax, beta=0.05):
    """Cumulative ideal sample counts over orders 1..N, per source."""
    sources = rv_g[1][0].keys()
    out = {s: {} for s in sources}
    for s in sources:
        d2 = 0.0
        for N in range(1, nmax + 1):
            fg, vg = rv_g[N][0][s], rv_g[N][1][s]
            fh, vh = rv_h[N][0][s], rv_h[N][1][s]
            og, oh = np.argsort(fg), np.argsort(fh)
            noise = vg[og] + vh[oh]
            diff = fg[og] - fh[oh]
            mask = noise > 0
            d2 += float(np.sum(diff[mask] ** 2 / noise[mask]))
            out[s][N] = 4 * np.log(1 / beta) / d2 if d2 > 0 else float("inf")
    return out


def moments_all(W, N, lam_by_source, device, chunk=4000, key_batch=1024):
    """All multiplicity patterns of order N at once: dict doubled -> source
    -> array (M_N,).  The Gram sums of every pattern of every table are
    computed once per chunk of sets, from the gathered rows of W and their
    conjugates, and the products over the keys of every table follow."""
    import torch
    n = W.shape[0]
    doubles = [d for size in range(0, N + 1) for d in combinations(range(N), size)]
    loaded = {d: tables(N, d) for d in doubles}
    union = {}
    for d, (patterns, keys, weights) in loaded.items():
        for pat in patterns:
            union.setdefault(pat, len(union))
    remapped = {}
    for d, (patterns, keys, weights) in loaded.items():
        local = np.array([union[pat] for pat in patterns] + [len(union)])
        ids = np.where(keys >= 0, keys, len(patterns))
        remapped[d] = (torch.as_tensor(local[ids], device=device, dtype=torch.long),
                       {s: torch.as_tensor(c, device=device, dtype=torch.complex128)
                        for s, c in coefficients(weights, lam_by_source, N + len(d)).items()})
    patterns = sorted(union, key=union.get)
    Wc = W.conj()
    index = torch.as_tensor(np.array(list(combinations(range(n), N))).T, device=device)
    M = index.shape[1]
    out = {d: {s: np.zeros(M) for s in remapped[d][1]} for d in doubles}
    for start in range(0, M, chunk):
        idx = index[:, start:start + chunk]
        size = idx.shape[1]
        rows = [W[idx[j]] for j in range(N)]
        rowsc = [Wc[idx[j]] for j in range(N)]
        powered = {}

        def power(j, p, q):
            key = (j, p, q)
            if key not in powered:
                f = None
                for _ in range(p):
                    f = rows[j] if f is None else f * rows[j]
                for _ in range(q):
                    f = rowsc[j] if f is None else f * rowsc[j]
                powered[key] = f
            return powered[key]

        values = torch.empty((len(patterns) + 1, size), dtype=torch.complex128, device=device)
        for i, (p, q) in enumerate(patterns):
            total = None
            for j in range(N):
                if p[j] == 0 and q[j] == 0:
                    continue
                f = power(j, p[j], q[j])
                total = f if total is None else total * f
            values[i] = total.sum(dim=1)
            if len(powered) > 24:
                powered.clear()
        values[len(patterns)] = 1.0
        for d in doubles:
            ids, coef = remapped[d]
            acc = {s: torch.zeros(size, dtype=torch.complex128, device=device) for s in coef}
            for kstart in range(0, ids.shape[0], key_batch):
                batch = ids[kstart:kstart + key_batch]
                prod = values[batch[:, 0]]
                for col in range(1, batch.shape[1]):
                    prod = prod * values[batch[:, col]]
                for s in coef:
                    acc[s] += coef[s][kstart:kstart + key_batch] @ prod
            for s in coef:
                out[d][s][start:start + size] = acc[s].real.cpu().numpy()
        del rows, rowsc, powered, values
    return out
