"""Exact noise of phase-locked classical sources: coherent states whose real
amplitude is drawn independently per slot from a symmetric distribution.

This is the linear network driven by real noise, read through the chip's
path matrix: E[prod_a |beta_a|^2] with beta = W alpha and alpha real, a sum
over the set partitions of the 2K legs of a K-photon moment (one W leg and
one conj(W) leg per photon) into blocks of even size, weighted by the
cumulants of the amplitude and the Gram sum of each block, which may hold
unequal numbers of conjugated and unconjugated legs.  Positions sharing a
detector are interchangeable, so the enumeration runs over multisets of
block exponents with a multinomial count, not over the labelled legs.
"""
import json
import os
from functools import lru_cache
from itertools import product
from math import factorial

import numpy as np

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "noise_tables")

# cumulants of the real amplitude normalised by w^(k/2), with w = E[alpha^2]
CUMULANTS = {
    "uniform": {2: 1.0, 4: -6 / 5, 6: 48 / 7, 8: -432 / 5},         # alpha ~ U[-a, a], w = a^2 / 3
    "rademacher": {2: 1.0, 4: -2.0, 6: 16.0, 8: -272.0},              # alpha = +-sqrt(w)
    "gaussian real": {2: 1.0, 4: 0.0, 6: 0.0, 8: 0.0},                # alpha ~ N(0, w)
}


def sample_amplitude(dist, w, size, rng):
    if dist == "uniform":
        return rng.uniform(-np.sqrt(3 * w), np.sqrt(3 * w), size)
    if dist == "rademacher":
        return np.sqrt(w) * rng.choice([-1.0, 1.0], size)
    if dist == "gaussian real":
        return rng.normal(0.0, np.sqrt(w), size)
    raise ValueError(dist)


def enumerate_real_terms(positions):
    """Positions: list of length K with values in range(N).  Returns
    patterns (p, q), keys (n_keys, K) of pattern ids padded with -1, and
    weights: dict sizes -> array (n_keys,) with the number of leg
    partitions of that block-size shape giving each product."""
    K = len(positions)
    N = max(positions) + 1
    m = tuple(int(x) for x in np.bincount(np.array(positions), minlength=N))
    blocks = []
    for p in product(*[range(mj + 1) for mj in m]):
        for q in product(*[range(mj + 1) for mj in m]):
            size = sum(p) + sum(q)
            if size >= 2 and size % 2 == 0:
                blocks.append((p, q))
    blocks.sort(key=lambda b: (-(sum(b[0]) + sum(b[1])), b))
    pattern_id = {b: i for i, b in enumerate(blocks)}
    terms = {}

    def search(rem_p, rem_q, start, chosen):
        if not any(rem_p) and not any(rem_q):
            terms[tuple(sorted(chosen))] = True
            return
        for i in range(start, len(blocks)):
            p, q = blocks[i]
            if all(pi <= ri for pi, ri in zip(p, rem_p)) and all(qi <= ri for qi, ri in zip(q, rem_q)):
                search(tuple(r - x for r, x in zip(rem_p, p)), tuple(r - x for r, x in zip(rem_q, q)), i, chosen + [i])

    search(m, m, 0, [])
    all_keys = sorted(terms)
    keys = -np.ones((len(all_keys), K), dtype=np.int32)
    by_shape = {}
    total = 1
    for mj in m:
        total *= factorial(mj) ** 2
    for row, ids in enumerate(all_keys):
        keys[row, :len(ids)] = ids
        count = total
        for i in ids:
            p, q = blocks[i]
            for pj, qj in zip(p, q):
                count //= factorial(pj) * factorial(qj)
        for i in set(ids):
            count //= factorial(ids.count(i))
        sizes = tuple(sorted(sum(blocks[i][0]) + sum(blocks[i][1]) for i in ids))
        by_shape.setdefault(sizes, np.zeros(len(all_keys)))[row] = count
    return blocks, keys, by_shape


@lru_cache(maxsize=None)
def tables_real(N, doubled):
    positions = []
    for j in range(N):
        positions += [j] * (2 if j in doubled else 1)
    os.makedirs(CACHE, exist_ok=True)
    path = f"{CACHE}/real_N{N}_" + "".join(map(str, doubled)) + ".npz"
    if os.path.exists(path):
        with np.load(path) as z:
            patterns = [(tuple(p), tuple(q)) for p, q in json.loads(str(z["patterns"]))]
            shapes = json.loads(str(z["shapes"]))
            weights = {tuple(sizes): np.asarray(row, dtype=float) for sizes, row in zip(shapes, z["weights"])}
            return patterns, z["keys"], weights
    patterns, keys, weights = enumerate_real_terms(positions)
    np.savez(path, patterns=json.dumps([[list(p), list(q)] for p, q in patterns]), keys=keys,
             shapes=json.dumps([list(k) for k in weights]),
             weights=np.array([np.asarray(v, dtype=float) for v in weights.values()]))
    return patterns, keys, weights


def coefficients_real(weights, cum_by_source, K):
    """Per source, the coefficient of every key: the product of the
    normalised cumulants of its block sizes times the count."""
    out = {}
    for source, cum in cum_by_source.items():
        coef = None
        for sizes, w in weights.items():
            term = np.prod([cum[s] for s in sizes]) * w
            coef = term if coef is None else coef + term
        out[source] = coef
    return out


def cpu_moment(W, detectors, cum):
    """E[prod_{a in detectors} |beta_a|^2] / w^K on one multiset of
    detectors, straight from the tables, with numpy."""
    uniq = sorted(set(detectors))
    positions = [uniq.index(a) for a in detectors]
    doubled = tuple(j for j in range(len(uniq)) if positions.count(j) == 2)
    patterns, keys, weights = tables_real(len(uniq), doubled)
    coef = coefficients_real(weights, {"s": cum}, len(detectors))["s"]
    rows = W[uniq]
    values = np.array([np.sum(np.prod([rows[j] ** pj * np.conj(rows[j]) ** qj for j, (pj, qj) in enumerate(zip(p, q))], axis=0))
                       for p, q in patterns] + [1.0])
    ids = np.where(keys >= 0, keys, len(patterns))
    return float(np.real(coef @ np.prod(values[ids], axis=1)))


def monte_carlo(W, detectors, dist, w, samples, seed=0):
    rng = np.random.default_rng(seed)
    alpha = sample_amplitude(dist, w, (samples, W.shape[1]), rng)
    beta = alpha @ W.T
    intensity = np.abs(beta) ** 2
    prod = np.prod(intensity[:, detectors], axis=1)
    return prod.mean(), prod.std() / np.sqrt(samples)


def exact_rates(W, N, cum_by_source, device, chunk=4000, key_batch=1024):
    """E[prod_{a in S} |beta_a|^2] / w^N on every N-set, per source."""
    import photon_noise as pn
    return pn.moments_all(W, N, cum_by_source, device, chunk, key_batch,
                          tables_fn=tables_real, coef_fn=coefficients_real, doubles=[()])[()]


def mc_moments(W, nmax, dist, samples, device, batch=2000, chunk=20000, seed=0):
    """Normally ordered moments from samples of the intensities at w = 1:
    dict (N, doubled) -> array (M_N,) of E[prod_{doubled} I_a^2 prod I_a],
    plus the standard error of the plain rates, dict N -> array (M_N,)."""
    import torch
    from itertools import combinations
    n, slots = W.shape
    Wt = torch.as_tensor(W.T, device=device, dtype=torch.complex64)
    gen = torch.Generator(device=device).manual_seed(seed)
    indices = {N: torch.as_tensor(np.array(list(combinations(range(n), N))).T, device=device) for N in range(1, nmax + 1)}
    doubles = {N: [d for size in range(0, N + 1) for d in combinations(range(N), size)] for N in indices}
    acc = {(N, d): torch.zeros(indices[N].shape[1], dtype=torch.float64, device=device) for N in indices for d in doubles[N]}
    sq = {N: torch.zeros(indices[N].shape[1], dtype=torch.float64, device=device) for N in indices}
    done = 0
    while done < samples:
        B = min(batch, samples - done)
        if dist == "uniform":
            alpha = (torch.rand((B, slots), generator=gen, device=device) * 2 - 1) * np.sqrt(3.0)
        elif dist == "rademacher":
            alpha = torch.where(torch.rand((B, slots), generator=gen, device=device) < 0.5, -1.0, 1.0)
        elif dist == "gaussian real":
            alpha = torch.randn((B, slots), generator=gen, device=device)
        else:
            raise ValueError(dist)
        intensity = (alpha.to(torch.complex64) @ Wt).abs() ** 2                   # (B, n)
        for N, index in indices.items():
            for start in range(0, index.shape[1], chunk):
                idx = index[:, start:start + chunk]
                rows = [intensity[:, idx[j]] for j in range(N)]
                for d in doubles[N]:
                    prod = None
                    for j in range(N):
                        f = rows[j] * rows[j] if j in d else rows[j]
                        prod = f if prod is None else prod * f
                    acc[(N, d)][start:start + idx.shape[1]] += prod.sum(dim=0).double()
                    if not d:
                        sq[N][start:start + idx.shape[1]] += (prod * prod).sum(dim=0).double()
        done += B
    out = {k: (v / samples).cpu().numpy() for k, v in acc.items()}
    err = {N: np.sqrt(np.maximum((sq[N] / samples).cpu().numpy() - out[(N, ())] ** 2, 0.0) / samples) for N in indices}
    return out, err


def combine_real(rates, mc, nmax, w):
    """Rates (exact) and variances (Monte Carlo) at brightness w, per order."""
    out = {}
    for N in range(1, nmax + 1):
        r = w ** N * rates[N]
        second = np.zeros_like(r)
        for (M, d), m in mc.items():
            if M == N:
                second += w ** (N + len(d)) * m
        out[N] = (r, np.maximum(second - r ** 2, 0.0))
    return out
