"""Exact coincidence profiles of the recurrent photonic graph network, and
BREC's test on them.

The network of a graph (Sec. "Photonic graph invariants" of the draft):
one chip per vertex with one port mode per incident edge, one memory mode
and one local mode carrying a source and a detector; every chip applies the
same cell, a Fourier transform on its ports followed by a fixed U(3) on the
uniform port mode, the memory and the local mode, the other Fourier modes
reflected with phase -1; every fed-back mode has transmissivity gamma.

The order-N coincidence rate of a set S of detectors is, for all four
sources (Eq. "sources" of the draft),

    F_S = w^N sum_{c_1..c_N} lambda(c) sum_{rho in S_N} omega(rho)
              prod_i W[a_i, c_i] conj(W[a_i, c_rho(i)])

with W the path matrix (detectors x slots, a slot being a source and a
delay).  The collision weight lambda(c) = prod_k lambda(m_k) over the
multiplicities of the slots is resolved by Moebius inversion over set
partitions of the N photons: a partition tau into blocks of sizes k gets
the weight prod_B g(|B|), g(k) = sum_{sigma <= 1} prod_{b in sigma}
lambda(|b|) mu(sigma, 1), and the sum over slot tuples constant on the
blocks of tau factorises into one generalised Gram sum per block,

    G(P, Q)[a] = sum_s prod_{i in P} W[a_i, s] prod_{j in Q} conj(W[a_j, s]),

with P = B and Q = rho^{-1}(B).  Every G is a dense tensor over the
distinct positions of P and Q, computed with one matrix product over the
slots, so the full n^N tensor of rates costs O(n^N * slots) for fixed N.
"""

from itertools import combinations, permutations
from math import factorial, log

import numpy as np

SOURCES = ("photons p=1", "photons p=0", "coherent", "thermal")
W = 0.05  # the brightness of the BREC runs, photons per source and tick

# BREC's test constants (GraphPKU/BREC, test_BREC.py)
NUM_RELABEL = 32
OUTPUT_DIM = 16
EPSILON_MATRIX = 1e-7
EPSILON_CMP = 1e-6
THRESHOLD = 72.34

CATEGORIES = (("Basic", "basic"), ("Regular", "regular"),
              ("Strongly_regular", "str"), ("Extension", "extension"),
              ("CFI", "cfi"), ("4-vertex_condition", "4vtx"),
              ("Distance_regular", "dr"))

# The pairs of the paper's 46-pair subset, read off the cached run.
SUBSET = {
    "Basic": [0, 6, 12, 18, 24, 30, 36, 42, 48, 54],
    "Regular": [0, 10, 20, 30, 40, 47],
    "Strongly_regular": [0, 5, 10, 20, 30, 40],
    "Extension": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90],
    "CFI": [5, 11, 12, 17, 18, 31, 57, 73, 76],
    "4-vertex_condition": [0, 10],
    "Distance_regular": [1, 2, 4, 5, 18, 19],
}


# ---------------------------------------------------------------- dataset

def load_brec(path):
    """BREC as a dict from category to a list of pairs of graph6 strings."""
    result = {}
    for category, filename in CATEGORIES:
        entries = np.asarray(
            np.load(f"{path}/{filename}.npy", allow_pickle=True)).reshape(-1)
        strings = [x.decode() if isinstance(x, bytes) else str(x)
                   for x in entries]
        result[category] = list(zip(strings[0::2], strings[1::2]))
    return result


def adjacency(graph6):
    """The adjacency matrix of a graph6 string, without networkx."""
    data = [ord(c) - 63 for c in graph6.strip()]
    if data[0] == 63:
        n = (data[1] << 12) | (data[2] << 6) | data[3]
        data = data[4:]
    else:
        n, data = data[0], data[1:]
    bits = []
    for d in data:
        bits.extend((d >> k) & 1 for k in range(5, -1, -1))
    adj = np.zeros((n, n), dtype=np.int8)
    k = 0
    for j in range(1, n):
        for i in range(j):
            adj[i, j] = adj[j, i] = bits[k]
            k += 1
    return adj


def relabel(adj, perm):
    """The adjacency of the graph with vertex i renamed perm[i]."""
    inv = np.argsort(perm)
    return adj[np.ix_(inv, inv)]


# ------------------------------------------------------------ the network

def haar_unitary(rng, d):
    z = (rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))) / np.sqrt(2)
    q, r = np.linalg.qr(z)
    return q * (np.diag(r) / np.abs(np.diag(r)))


def cell(seed):
    """A Haar-random U(3) on (uniform port mode, memory, local), rotated on
    its inputs so that its local-to-local entry vanishes, and a U(2) on
    (memory, local) for isolated vertices, built the same way."""
    rng = np.random.default_rng(seed)
    out = []
    for d in (3, 2):
        u = haar_unitary(rng, d)
        a, b = u[-1, 0], u[-1, -1]
        norm = np.sqrt(abs(a) ** 2 + abs(b) ** 2)
        v = np.eye(d, dtype=complex)
        v[0, 0], v[-1, 0] = np.conj(a) / norm, np.conj(b) / norm
        v[0, -1], v[-1, -1] = b / norm, -a / norm
        u = u @ v
        assert abs(u[-1, -1]) < 1e-14
        out.append(u)
    return out


def network(adj, cells, gamma):
    """The blocks (A, B, C, D) of one tick, with the loss folded into A and
    B, and the ordering of the loop modes.

    Loop modes: one per arc (v, u) in the order of the arcs, then one memory
    mode per vertex.  External modes: one local mode per vertex.
    """
    n = adj.shape[0]
    arcs = [(v, u) for v in range(n) for u in range(n) if adj[v, u]]
    arc_index = {arc: i for i, arc in enumerate(arcs)}
    L, M = len(arcs) + n, n
    u_hat = np.zeros((L + M, L + M), dtype=complex)
    uc3, uc2 = cells
    for v in range(n):
        ports = [arc_index[(v, u)] for u in range(n) if adj[v, u]]
        d = len(ports)
        modes = ports + [len(arcs) + v, L + v]
        if d == 0:
            u_hat[np.ix_(modes, modes)] = uc2
            continue
        fourier = np.exp(2j * np.pi * np.outer(range(d), range(d)) / d)
        fourier /= np.sqrt(d)
        middle = -np.eye(d + 2, dtype=complex)
        sym = [0, d, d + 1]
        middle[np.ix_(sym, sym)] = uc3
        basis = np.eye(d + 2, dtype=complex)
        basis[:d, :d] = fourier
        u_hat[np.ix_(modes, modes)] = basis.conj().T @ middle @ basis
    # the permutation sends the output of arc (v, u) to the input of (u, v)
    pi = np.zeros((L, L))
    for (v, u), i in arc_index.items():
        pi[arc_index[(u, v)], i] = 1
    pi[len(arcs):, len(arcs):] = np.eye(n)
    A = np.sqrt(gamma) * pi @ u_hat[:L, :L]
    B = np.sqrt(gamma) * pi @ u_hat[:L, L:]
    C = u_hat[L:, :L]
    D = u_hat[L:, L:]
    return A, B, C, D


def path_matrix(blocks, tol=1e-9, max_delay=100000):
    """W = [D | CB | CAB | CA^2B | ...], truncated when the Frobenius norm of
    a column block falls below tol times that of the first."""
    A, B, C, D = blocks
    columns = [D]
    x = B
    first = np.linalg.norm(C @ x)
    for _ in range(max_delay):
        y = C @ x
        columns.append(y)
        if np.linalg.norm(y) < tol * first:
            break
        x = A @ x
    return np.concatenate(columns, axis=1)


# ---------------------------------------------------------- the sources

def set_partitions(items):
    if not items:
        yield []
        return
    first, rest = items[0], items[1:]
    for smaller in set_partitions(rest):
        for i, block in enumerate(smaller):
            yield smaller[:i] + [[first] + block] + smaller[i + 1:]
        yield [[first]] + smaller


def block_weight(lam, k):
    """g(k) = sum over partitions sigma of a k-set of prod lam(|b|) times
    mu(sigma, top) = (-1)^(|sigma|-1) (|sigma|-1)!."""
    total = 0.0
    for sigma in set_partitions(list(range(k))):
        weight = (-1) ** (len(sigma) - 1) * factorial(len(sigma) - 1)
        for b in sigma:
            weight *= lam(len(b))
        total += weight
    return total


LAMBDA = {
    "photons p=1": lambda m: 1.0 if m == 1 else 0.0,
    "photons p=0": lambda m: 1.0 if m == 1 else 0.0,
    "coherent": lambda m: 1.0 / factorial(m),
    "thermal": lambda m: 1.0,
}


def omega(source, rho):
    """The interference weight of a permutation of the photons."""
    if source == "photons p=0":
        return 1.0 if all(i == r for i, r in enumerate(rho)) else 0.0
    return 1.0


# --------------------------------------------------------- the profiles

def gram_tensor(W, Wc, P, Q):
    """G(P, Q) as a tensor over the sorted distinct positions of P and Q."""
    positions = sorted(set(P) | set(Q))
    factors = []
    for j in positions:
        p, q = P.count(j), Q.count(j)
        m = np.ones_like(W)
        for _ in range(p):
            m = m * W
        for _ in range(q):
            m = m * Wc
        factors.append(m)
    k = len(positions)
    n = W.shape[0]
    if k == 1:
        return positions, factors[0].sum(axis=1)
    if k == 2:
        return positions, factors[0] @ factors[1].T
    if k == 3:
        left = (factors[0][:, None, :] * factors[1][None, :, :]).reshape(n * n, -1)
        return positions, (left @ factors[2].T).reshape(n, n, n)
    if k == 4:
        left = (factors[0][:, None, :] * factors[1][None, :, :]).reshape(n * n, -1)
        right = (factors[2][:, None, :] * factors[3][None, :, :]).reshape(n * n, -1)
        return positions, (left @ right.T).reshape(n, n, n, n)
    if k == 5:
        left = (factors[0][:, None, :] * factors[1][None, :, :]).reshape(n * n, -1)
        right = (factors[2][:, None, None, :] * factors[3][None, :, None, :]
                 * factors[4][None, None, :, :]).reshape(n ** 3, -1)
        return positions, (left @ right.T).reshape((n,) * 5)
    raise NotImplementedError(k)


def rate_tensors(W, N, sources=SOURCES):
    """The n^N tensors of order-N rates divided by w^N, one per source."""
    n = W.shape[0]
    Wc = W.conj()
    letters = "abcdefgh"[:N]
    # group (tau, rho) by the product of Gram sums it gives
    products = {}
    for tau in set_partitions(list(range(N))):
        g = {s: np.prod([block_weight(LAMBDA[s], len(b)) for b in tau])
             for s in sources}
        if all(abs(v) < 1e-300 for v in g.values()):
            continue
        for rho in permutations(range(N)):
            inverse = [rho.index(i) for i in range(N)]
            key = tuple(sorted(
                (tuple(sorted(b)), tuple(sorted(inverse[i] for i in b)))
                for b in tau))
            coef = products.setdefault(key, dict.fromkeys(sources, 0.0))
            for s in sources:
                coef[s] += g[s] * omega(s, rho)
    cache = {}
    total = {s: np.zeros((n,) * N, dtype=complex) for s in sources}
    for key, coef in products.items():
        if all(abs(v) < 1e-300 for v in coef.values()):
            continue
        operands, subscripts = [], []
        for P, Q in key:
            if (P, Q) not in cache:
                if (Q, P) in cache:
                    pos, t = cache[(Q, P)]
                    cache[(P, Q)] = (pos, t.conj())
                else:
                    cache[(P, Q)] = gram_tensor(W, Wc, P, Q)
            pos, t = cache[(P, Q)]
            operands.append(t)
            subscripts.append("".join(letters[j] for j in pos))
        term = np.einsum(",".join(subscripts) + "->" + letters, *operands)
        for s in sources:
            if coef[s]:
                total[s] += coef[s] * term
    return {s: t.real for s, t in total.items()}


def profiles(W, nmax, w, sources=SOURCES):
    """The sorted order-N profiles, N = 1..nmax, as arrays (sources, M_N)."""
    n = W.shape[0]
    out = {}
    for N in range(1, nmax + 1):
        tensors = rate_tensors(W, N, sources)
        idx = np.array(list(combinations(range(n), N))).T
        rows = [np.sort(tensors[s][tuple(idx)] * w ** N) for s in sources]
        out[N] = np.array(rows)
    return out


def network_profiles(adj, cells, gamma, w, nmax, sources=SOURCES):
    W = path_matrix(network(adj, cells, gamma))
    return profiles(W, nmax, w, sources), W.shape[1]


# -------------------------------------------------------------- the test

def gap(fg, fh):
    """The Hellinger-type gap of Eq. (gap) between two sorted profiles."""
    return float(np.sqrt(np.sum((np.sqrt(fg) - np.sqrt(fh)) ** 2)))


def ideal_samples(fg, fh, beta=0.05):
    """Samples per network for the optimal test knowing the rates, with
    Poisson noise on every set: T = 4 ln(1/beta) / D^2, D^2 = sum of
    (F_G - F_H)^2 / (F_G + F_H) over the sorted entries."""
    denominator = fg + fh
    mask = denominator > 0
    d2 = float(np.sum((fg[mask] - fh[mask]) ** 2 / denominator[mask]))
    return 4 * log(1 / beta) / d2 if d2 > 0 else float("inf")


def features(profile, w, projection):
    """BREC-shaped output: the profiles of every order, each divided by w^N,
    concatenated and projected to OUTPUT_DIM dimensions."""
    vector = np.concatenate([profile[N] / w ** N for N in sorted(profile)])
    return projection @ vector


def hotelling(X, Y, epsilon):
    """BREC's T2_calculation: X, Y are (dim, copies); paired differences,
    their covariance regularised by epsilon on the diagonal."""
    D = X - Y
    d_mean = D.mean(axis=1, keepdims=True)
    S = np.cov(D)
    S = np.nan_to_num(S)
    inv = np.linalg.inv(S + epsilon * np.eye(S.shape[0]))
    return float(abs(np.asarray(d_mean.T @ inv @ d_mean).ravel()[0]))


def brec_decision(t2_pair, t2_reliability, threshold=THRESHOLD):
    """BREC's rule: separated when the pair statistic exceeds the threshold
    and is not within EPSILON_CMP of the reliability one; unreliable when
    the reliability statistic itself exceeds the threshold."""
    reliable = t2_reliability < threshold
    separated = (t2_pair > threshold
                 and abs(t2_pair - t2_reliability) > EPSILON_CMP)
    return bool(separated and reliable), bool(reliable)
VERSION = 2


# ------------------------------------------------- chips with K source modes

def cell_k(seed, local_modes):
    """Haar-random cells for a chip with `local_modes` source modes per
    vertex: a U(K + 2) on (uniform port mode, memory, local_1..K) and a
    U(K + 1) on (memory, local_1..K) for isolated vertices, each rotated on
    its inputs so that the detected output, local_1, has no amplitude from
    any of the K source inputs.  For K = 1 this is `cell`."""
    if local_modes == 1:
        return cell(seed)
    rng = np.random.default_rng(seed)
    out = []
    K = local_modes
    for d in (K + 2, K + 1):
        u = haar_unitary(rng, d)
        det = d - K                                   # row of the detected local output
        mixed = [0] + list(range(d - K, d))           # inputs rotated: first mode and the K locals
        r = u[det, mixed]
        port = np.conj(r) / np.linalg.norm(r)         # new first direction, as in `cell`
        # complete to an orthonormal basis of the span: the K source directions
        basis = np.linalg.qr(np.concatenate([port[:, None], rng.normal(size=(K + 1, K)) + 0j], axis=1))[0]
        basis[:, 0] = port
        basis[:, 1:] -= port[:, None] * (np.conj(port) @ basis[:, 1:])
        basis[:, 1:] = np.linalg.qr(basis[:, 1:])[0]
        v = np.eye(d, dtype=complex)
        v[np.ix_(mixed, mixed)] = basis
        u = u @ v
        assert np.all(np.abs(u[det, d - K:]) < 1e-12)
        out.append(u)
    return out


def network_k(adj, cells, gamma, local_modes):
    """The blocks (A, B, C, D) of one tick for a chip with K source modes per
    vertex and one detector per vertex, on the detected local output; the
    other K - 1 local outputs are lost.  External inputs are ordered
    (vertex, source mode).  For K = 1 this is `network`."""
    K = local_modes
    if K == 1:
        return network(adj, cells, gamma)
    n = adj.shape[0]
    arcs = [(v, u) for v in range(n) for u in range(n) if adj[v, u]]
    arc_index = {arc: i for i, arc in enumerate(arcs)}
    L, M = len(arcs) + n, K * n
    u_hat = np.zeros((L + M, L + M), dtype=complex)
    uc_big, uc_small = cells
    for v in range(n):
        ports = [arc_index[(v, u)] for u in range(n) if adj[v, u]]
        d = len(ports)
        locals_ = [L + K * v + j for j in range(K)]
        modes = ports + [len(arcs) + v] + locals_
        if d == 0:
            u_hat[np.ix_(modes, modes)] = uc_small
            continue
        fourier = np.exp(2j * np.pi * np.outer(range(d), range(d)) / d) / np.sqrt(d)
        middle = -np.eye(d + 1 + K, dtype=complex)
        sym = [0] + list(range(d, d + 1 + K))
        middle[np.ix_(sym, sym)] = uc_big
        basis = np.eye(d + 1 + K, dtype=complex)
        basis[:d, :d] = fourier
        u_hat[np.ix_(modes, modes)] = basis.conj().T @ middle @ basis
    pi = np.zeros((L, L))
    for (v, u), i in arc_index.items():
        pi[arc_index[(u, v)], i] = 1
    pi[len(arcs):, len(arcs):] = np.eye(n)
    detected = [L + K * v for v in range(n)]
    A = np.sqrt(gamma) * pi @ u_hat[:L, :L]
    B = np.sqrt(gamma) * pi @ u_hat[:L, L:]
    C = u_hat[detected, :L]
    D = u_hat[np.ix_(detected, range(L, L + M))]
    return A, B, C, D


def symmetric_columns(W, n, local_modes):
    """The path matrix of the one-mode placement: the K source columns of
    every vertex and delay averaged into the symmetric mode."""
    K = local_modes
    delays = W.shape[1] // (K * n)
    return W.reshape(n, delays, n, K).sum(axis=3).reshape(n, delays * n) / np.sqrt(K)


def single_columns(W, n, local_modes, j=0):
    """The path matrix seen by a source in local mode j alone."""
    K = local_modes
    delays = W.shape[1] // (K * n)
    return np.ascontiguousarray(W.reshape(n, delays, n, K)[:, :, :, j].reshape(n, delays * n))
