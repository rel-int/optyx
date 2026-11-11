"""The forward-pass test: BREC's protocol with a draw of the input light as
the forward pass and a symmetric polynomial of the single-tick intensities
as the invariant embedding.

A copy is a relabelled graph with its own stream of draws.  For every draw
the intensity vector is ``I = |W alpha|^2`` with ``alpha`` one random phase
per slot (coherent light) or one complex Gaussian amplitude per slot
(thermal light), read either as intensities (the camera limit, infinite
brightness) or as Poisson counts at brightness ``w``.  The embedding of a
draw is ``e_N((I / m)^k) / C(n, N)`` for ``N, k = 1..4``, the elementary
symmetric polynomials of the per-detector powers normalised by the exact
mean intensity ``m``, and the embedding of a copy is its mean over the
draws.  Hotelling's T^2 over 32 copies of G and of H, with 32 more copies
of G for the reliability check, decides as in BREC.

The exact anchors: for thermal light the phase average of a product of
intensities is a permanent of the coherence matrix on the multiset of
detectors (Wick), for coherent light it is the constant term of a Laurent
polynomial in one variable per slot, read off by a dynamic programme over
the slots.
"""

from itertools import combinations, permutations, product
from math import comb, factorial

import numpy as np

import brecphot as bp

NMAX = 4
KMAX = 4
BUDGETS = (1000, 10000, 100000, 1000000)
SOURCES = ("coherent-w1", "coherent-w100", "coherent-inf", "thermal-inf")
PATH_TOL = 1e-4
VERSION = 1


def parse_source(source):
    """'coherent-w100' -> ('coherent', 100.0); 'thermal-inf' -> ('thermal', None)."""
    light, _, brightness = source.partition("-")
    return light, None if brightness == "inf" else float(brightness[1:])


def mean_intensity(W, brightness):
    """The exact mean intensity per detector at unit amplitude per slot."""
    return float(np.mean(np.sum(np.abs(W) ** 2, axis=1))) * (brightness or 1.0)


def symmetric_features(I, scale, nmax=NMAX, kmax=KMAX):
    """e_N((I / scale)^k) / C(n, N) for N, k = 1..nmax, kmax: (chunk, kmax * nmax),
    row-major in (k, N), by the recursion over the detectors."""
    import torch
    x = I / scale
    n = x.shape[1]
    powers = torch.stack([x ** k for k in range(1, kmax + 1)], dim=1)  # (chunk, kmax, n)
    e = torch.zeros((x.shape[0], kmax, nmax + 1), dtype=x.dtype, device=x.device)
    e[:, :, 0] = 1
    for j in range(n):
        e[:, :, 1:] = e[:, :, 1:] + powers[:, :, j:j + 1] * e[:, :, :-1].clone()
    norm = torch.tensor([comb(n, N) for N in range(1, nmax + 1)], dtype=x.dtype, device=x.device)
    return (e[:, :, 1:] / norm).reshape(x.shape[0], kmax * nmax)


def feature_names(nmax=NMAX, kmax=KMAX):
    return [f"e{N}(I^{k})" for k in range(1, kmax + 1) for N in range(1, nmax + 1)]


class Draws:
    """Streams chunks of intensity vectors (chunk, n) of one source through
    one path matrix, from its own generator."""

    def __init__(self, W, source, seed, device, chunk=8192):
        import torch
        self.light, self.brightness = parse_source(source)
        self.W = torch.as_tensor(W, dtype=torch.complex64, device=device)
        self.device, self.chunk = device, chunk
        self.gen = torch.Generator(device=device).manual_seed(int(seed))

    def chunks(self, total):
        import torch
        done = 0
        slots = self.W.shape[1]
        while done < total:
            k = min(self.chunk, total - done)
            if self.light == "coherent":
                phases = torch.rand((k, slots), generator=self.gen, device=self.device)
                alpha = torch.polar(torch.ones_like(phases), 2 * torch.pi * phases)
            else:
                alpha = torch.view_as_complex(torch.randn(
                    (k, slots, 2), generator=self.gen, device=self.device)) / 2 ** 0.5
            intensity = (alpha @ self.W.T).abs() ** 2
            if self.brightness is not None:
                intensity = torch.poisson(self.brightness * intensity, generator=self.gen)
            yield intensity.to(torch.float64)
            done += k


def copy_embeddings(W, source, seed, budgets, device, chunk=8192):
    """The embedding of one copy at every budget: dict R -> (features,) mean
    over the first R draws, and the per-feature variance at the largest."""
    import torch
    scale = mean_intensity(W, parse_source(source)[1])
    total = max(budgets)
    running = torch.zeros(KMAX * NMAX, dtype=torch.float64, device=device)
    running_sq = torch.zeros_like(running)
    seen, out = 0, {}
    pending = sorted(budgets)
    draws = Draws(W, source, seed, device, chunk=min(chunk, min(budgets)))
    for I in draws.chunks(total):
        f = symmetric_features(I, scale)
        running += f.sum(dim=0)
        running_sq += (f * f).sum(dim=0)
        seen += f.shape[0]
        while pending and seen >= pending[0]:
            out[pending.pop(0)] = (running / seen).cpu().numpy()
        if seen >= total:
            break
    variance = (running_sq / seen - (running / seen) ** 2).clamp(min=0).cpu().numpy()
    return out, variance


def copies_of(adj, cells, gamma, pair, copies=bp.NUM_RELABEL):
    """The path matrices of 32 copies of G, 32 of H and 32 more of G, copy 0
    of each side being the unrelabelled graph, with the same permutations as
    `launch.copy_tags` so that the copies match the exact column's."""
    rng = np.random.default_rng(sum(map(ord, pair)) + 1)
    out = {}
    for side, count in (("G", 2 * copies), ("H", copies)):
        for k in range(count):
            perm = None if k == 0 else rng.permutation(adj[side].shape[0])
            relabelled = adj[side] if perm is None else bp.relabel(adj[side], perm)
            out[f"{side}{k}"] = bp.path_matrix(bp.network(relabelled, cells, gamma), tol=PATH_TOL)
    return out


def hotelling_decisions(X, Y, Z):
    """BREC's decisions from the (dim, copies) embeddings of G, H and G':
    the test verbatim, with the regularisation at 1e-7, and scaled to the
    noise variance of the reliability differences; the features are first
    scaled to unit root-mean-square as in `gnn_bench.sampled_hotelling`."""
    scale = np.sqrt(np.mean(np.concatenate([X, Y, Z], axis=1) ** 2, axis=1, keepdims=True))
    scale[scale == 0] = 1
    X, Y, Z = X / scale, Y / scale, Z / scale
    noise = float(np.mean(np.diag(np.atleast_2d(np.cov(X - Z)))))
    out = {"noise_variance": noise,
           "gap_over_noise": (np.abs(X.mean(1) - Y.mean(1)) / np.sqrt(
               np.maximum(np.diag(np.atleast_2d(np.cov(X - Z))), 1e-300))).tolist()}
    for label, eps in (("verbatim", bp.EPSILON_MATRIX), ("scaled", bp.EPSILON_MATRIX * noise)):
        t2, t2r = bp.hotelling(X, Y, eps), bp.hotelling(X, Z, eps)
        separated, reliable = bp.brec_decision(t2, t2r)
        out[label] = {"T2_pair": t2, "T2_reliability": t2r,
                      "separated": separated, "reliable": reliable}
    return out


def forward_pass_test(matrices, source, seed, budgets, device, copies=bp.NUM_RELABEL):
    """The whole test for one pair, cell and source: the embeddings of every
    copy at every budget and BREC's decisions at every budget."""
    embeddings, variances = {}, {}
    for k, (tag, W) in enumerate(matrices.items()):
        embeddings[tag], variances[tag] = copy_embeddings(
            W, source, 1000 * seed + k, budgets, device)
    decisions = {}
    for R in sorted(budgets):
        X = np.array([embeddings[f"G{k}"][R] for k in range(copies)]).T
        Y = np.array([embeddings[f"H{k}"][R] for k in range(copies)]).T
        Z = np.array([embeddings[f"G{k}"][R] for k in range(copies, 2 * copies)]).T
        decisions[str(R)] = hotelling_decisions(X, Y, Z)
    return {"decisions": decisions,
            "embeddings": {tag: {str(R): v.tolist() for R, v in e.items()}
                           for tag, e in embeddings.items()},
            "draw_variance": {tag: v.tolist() for tag, v in variances.items()}}


# ------------------------------------------------------------ exact anchors

def coherent_moments(W, sets, k):
    """E[prod_{x in S} |f_x|^{2k}] for every S in `sets`, all of one size N,
    under independent uniform phases: the constant term of the Laurent
    polynomial, by a dynamic programme over the slots whose state counts the
    legs of every factor already placed, balanced at every slot."""
    T, N = len(sets), len(sets[0])
    idx = np.array(sets)
    legs = list(product(range(k + 1), repeat=N))
    balanced = [(j, jp) for j in legs for jp in legs if sum(j) == sum(jp) > 0]
    state = np.zeros((T,) + (k + 1,) * (2 * N), dtype=complex)
    state[(slice(None),) + (0,) * (2 * N)] = 1
    invfact = np.array([1 / factorial(m) for m in range(k + 1)])
    for s in range(W.shape[1]):
        w = W[idx, s]
        pw = np.stack([w ** m * invfact[m] for m in range(k + 1)], axis=-1)
        pwc = np.stack([np.conj(w) ** m * invfact[m] for m in range(k + 1)], axis=-1)
        new = state.copy()
        for j, jp in balanced:
            weight = np.prod([pw[:, x, j[x]] for x in range(N)], axis=0) \
                * np.prod([pwc[:, x, jp[x]] for x in range(N)], axis=0)
            src = (slice(None),) + tuple(slice(0, k + 1 - m) for m in j + jp)
            dst = (slice(None),) + tuple(slice(m, k + 1) for m in j + jp)
            new[dst] += state[src] * weight.reshape((T,) + (1,) * (2 * N))
        state = new
    return state[(slice(None),) + (k,) * (2 * N)].real * factorial(k) ** (2 * N)


def permanents(M):
    """Permanents of a batch of m x m matrices (B, m, m), Ryser's formula over
    a Gray code, in numpy or torch."""
    B, m, _ = M.shape
    total, rowsum, mask = 0 * M[:, 0, 0], 0 * M[:, 0, :], 0
    for g in range(1, 2 ** m):
        bit = (g & -g).bit_length() - 1
        rowsum = rowsum - M[:, :, bit] if mask & (1 << bit) else rowsum + M[:, :, bit]
        mask ^= 1 << bit
        total = total + (-1) ** bin(mask).count("1") * rowsum.prod(1)
    return ((-1) ** m * total).real


def thermal_moments(Gamma, sets, k):
    """E[prod_{x in S} |f_x|^{2k}] under complex Gaussian amplitudes."""
    idx = np.repeat(np.array(sets), k, axis=1)
    return permanents(Gamma[idx[:, :, None], idx[:, None, :]])


def exact_readout(W, light, N, k, device="cpu"):
    """The exact embedding e_N((I / m)^k) / C(n, N) of one graph at infinite
    brightness, as a number."""
    n = W.shape[0]
    sets = list(combinations(range(n), N))
    scale = mean_intensity(W, None) ** (k * N)
    if light == "thermal":
        import torch
        Gamma = torch.as_tensor(W @ W.conj().T, device=device)
        idx = torch.as_tensor(np.repeat(np.array(sets), k, axis=1), device=device)
        total = 0.0
        for start in range(0, len(sets), 65536):
            block = idx[start:start + 65536]
            total += float(permanents(Gamma[block[:, :, None], block[:, None, :]]).sum())
        return total / scale / len(sets)
    return float(coherent_moments(W, sets, k).sum()) / scale / len(sets)


def exact_anchors(matrices, n_coherent=20, device="cpu"):
    """Relative gaps of the exact readouts between G0 and H0: thermal light at
    N <= 4, k <= 3 on every pair; coherent light at N = 3, k <= 2 on pairs
    with at most `n_coherent` vertices."""
    out = {}
    wg, wh = matrices["G0"], matrices["H0"]
    for light, cases in (("thermal", [(N, k) for N in (3, 4) for k in (1, 2, 3)]),
                         ("coherent", [(3, 1), (3, 2)] if wg.shape[0] <= n_coherent else [])):
        for N, k in cases:
            g, h = exact_readout(wg, light, N, k, device), exact_readout(wh, light, N, k, device)
            out[f"{light} e{N}(I^{k})"] = {"G": g, "H": h, "relative_gap": (g - h) / (0.5 * (g + h))}
    return out


def self_check():
    """Closed forms at N = 1, Ryser against brute force, the torch features
    against a direct product on a random draw."""
    import torch
    rng = np.random.default_rng(0)
    W = rng.normal(size=(5, 7)) + 1j * rng.normal(size=(5, 7))
    ones = [(a,) for a in range(5)]
    p2, p4 = (np.abs(W) ** 2).sum(1), (np.abs(W) ** 4).sum(1)
    assert np.allclose(coherent_moments(W, ones, 1), p2)
    assert np.allclose(coherent_moments(W, ones, 2), 2 * p2 ** 2 - p4)
    M = rng.normal(size=(1, 5, 5))
    assert np.isclose(permanents(M)[0], sum(
        np.prod([M[0, i, p[i]] for i in range(5)]) for p in permutations(range(5))))
    I = torch.as_tensor(rng.random((3, 6)), dtype=torch.float64)
    f = symmetric_features(I, 1.0, nmax=3, kmax=2).numpy()
    for k in (1, 2):
        for N in (1, 2, 3):
            direct = np.mean([np.prod([I[0, a].item() ** k for a in S]) for S in combinations(range(6), N)])
            assert np.isclose(f[0, (k - 1) * 3 + N - 1], direct)
    return True
