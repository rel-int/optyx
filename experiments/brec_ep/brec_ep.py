"""BREC's own test on the photonic graph network, trained by equilibrium
propagation.

The network of a graph is the one of ``experiments/brec_exact`` (rel-int/optyx
#91): one chip per vertex with a port mode per incident edge, a memory mode and
a local mode with a source and a detector; every chip applies the same cell, a
Fourier transform on its ports and a U(3) on the uniform port mode, the memory
and the local mode, the other Fourier modes reflected with phase -1; every
fed-back mode has transmissivity ``gamma``.  Here the cell is symmetric, so
the network is reciprocal.

The light is incoherent from tick to tick, so the mean count at the detector
of vertex ``a`` is ``sum_b T[a, b] q_b``, with ``q_b`` the mean number of
photons the source of ``b`` emits a tick and

    T[a, b] = sum_k |W[a, (b, k)]|^2,

the intensity transfer from the source of ``b`` to the detector of ``a``
summed over the delays ``k`` of the path matrix ``W``.  Reciprocity makes ``T``
symmetric.

Each vertex is a unit of the electronic loop of ``optyx.hopfield``: it holds
``z_v``, sets its source to ``q_v = q (1 + r_v) / 2`` with ``r_v = tanh z_v``,
and integrates ``z <- (1 - a) z + a (g m + c)`` on its rescaled count
``m = T (1 + r)``.  The fixed points are the critical points of

    E(r) = sum_v (r_v artanh r_v + ln(1 - r_v^2) / 2)
           - g (1 + r)^T T (1 + r) / 2 - c sum_v r_v,

an equivariant Hopfield energy, unique for ``g <= 1`` since ``T`` is
substochastic.  The embedding of the graph is ``f(G) = V sum_v cheb(r_v)``,
the Chebyshev polynomials of degrees 1 to 16, so ``f`` is invariant under
relabelling.

Training is BREC's: a Siamese pair of networks with shared parameters, one on
``G`` and one on ``H``, and the loss ``max(0, cos(f(G), f(H)))``.  The two
networks settle together and are nudged together by the loss, so equilibrium
propagation gives the gradient on the cell ``theta``, the gain ``g`` and the
offset ``c`` from the two nudged equilibria; the readout ``V`` is trained by
its exact gradient at the free equilibrium.
"""

from functools import lru_cache

import numpy as np
import scipy.sparse as sp

# BREC's constants (GraphPKU/BREC, base/test_BREC.py)
NUM_RELABEL = 32
SAMPLE_NUM = 400
OUTPUT_DIM = 16
EPOCH = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
LOSS_THRESHOLD = 0.2
THRESHOLD = 72.34
EPSILON_CMP = 1e-6
PARTS = {"Basic": (0, 60), "Regular": (60, 160), "Extension": (160, 260),
         "CFI": (260, 360), "4-Vertex_Condition": (360, 380),
         "Distance_Regular": (380, 400)}

GAMMA = 0.9
LEAK = 0.5
GAIN = 0.99
DEGREES = 16


# ---------------------------------------------------------------- dataset

def load(path):
    """BREC's 51,200 graph6 strings, as fixed-width bytes (no pickle)."""
    return np.load(path, allow_pickle=False)


def adjacency(graph6):
    """The adjacency matrix of a graph6 string."""
    if isinstance(graph6, bytes):
        graph6 = graph6.decode()
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


def copies(data, pair):
    """The 32 relabelled copies of a pair, and of its reliability pair."""
    test = data[pair * 2 * NUM_RELABEL:(pair + 1) * 2 * NUM_RELABEL]
    start = (pair + SAMPLE_NUM) * 2 * NUM_RELABEL
    reliability = data[start:start + 2 * NUM_RELABEL]
    return test, reliability


# ---------------------------------------------------------------- the cell

def rotation(angles):
    """A real rotation of three modes from three angles."""
    a, b, c = angles
    rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0],
                   [0, 0, 1]])
    ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0],
                   [-np.sin(b), 0, np.cos(b)]])
    rx = np.array([[1, 0, 0], [0, np.cos(c), -np.sin(c)],
                   [0, np.sin(c), np.cos(c)]])
    return rz @ ry @ rx


def cell(theta):
    """The symmetric unitary ``O diag(exp(i phi)) O^T`` of the six phases."""
    o = rotation(theta[:3])
    return o @ np.diag(np.exp(1j * np.asarray(theta[3:6]))) @ o.T


SWAP = np.array([[0, 1], [1, 0]], dtype=complex)  # isolated vertices


@lru_cache(maxsize=4096)
def _structure(key):
    adj = np.frombuffer(key[1], dtype=np.int8).reshape(key[0], key[0])
    n = adj.shape[0]
    arcs = [(v, u) for v in range(n) for u in range(n) if adj[v, u]]
    index = {arc: i for i, arc in enumerate(arcs)}
    blocks = []
    for v in range(n):
        ports = [index[(v, u)] for u in range(n) if adj[v, u]]
        blocks.append((ports, len(arcs) + v, len(arcs) + n + v))
    reverse = np.array([index[(u, v)] for (v, u) in arcs] +
                       list(range(len(arcs), len(arcs) + n)))
    return n, len(arcs), blocks, reverse


def network(adj, theta, gamma=GAMMA):
    """The sparse blocks (A, B, C, D) of one tick of the network."""
    n, arcs, blocks, reverse = _structure(
        (adj.shape[0], np.ascontiguousarray(adj, dtype=np.int8).tobytes()))
    loop = arcs + n
    u3 = cell(theta)
    rows, cols, vals = [], [], []
    for ports, memory, local in blocks:
        d = len(ports)
        modes = ports + [memory, local]
        if d == 0:
            unitary = SWAP
        else:
            fourier = np.exp(2j * np.pi * np.outer(range(d), range(d)) / d)
            fourier /= np.sqrt(d)
            middle = -np.eye(d + 2, dtype=complex)
            sym = [0, d, d + 1]
            middle[np.ix_(sym, sym)] = u3
            basis = np.eye(d + 2, dtype=complex)
            basis[:d, :d] = fourier
            unitary = basis.conj().T @ middle @ basis
        r, c = np.meshgrid(modes, modes, indexing="ij")
        rows.append(r.ravel())
        cols.append(c.ravel())
        vals.append(unitary.ravel())
    full = sp.csr_matrix((np.concatenate(vals),
                          (np.concatenate(rows), np.concatenate(cols))),
                         shape=(loop + n, loop + n))
    swap = sp.csr_matrix((np.ones(loop), (reverse, np.arange(loop))),
                         shape=(loop, loop))
    a = np.sqrt(gamma) * (swap @ full[:loop, :loop])
    b = np.sqrt(gamma) * (swap @ full[:loop, loop:])
    return a.tocsr(), b.toarray(), full[loop:, :loop].tocsr(), \
        full[loop:, loop:].toarray()


def couplings(adj, theta, gamma=GAMMA, tol=1e-9):
    """``T[a, b] = sum_k |W[a, (b, k)]|^2``, summed until the power of a
    delay falls below ``tol`` times that of the first."""
    a, b, c, d = network(adj, theta, gamma)
    total = np.abs(d) ** 2
    x = b
    first = None
    for _ in range(100000):
        y = c @ x
        power = np.abs(y) ** 2
        total += power
        norm = power.sum()
        first = norm if first is None else first
        if first == 0 or norm < tol * first:
            break
        x = a @ x
    return total


# ------------------------------------------------------------ the network

def chebyshev(r):
    """The Chebyshev polynomials of degrees 1 to DEGREES at ``r``, and their
    derivatives."""
    t = [np.ones_like(r), r]
    dt = [np.zeros_like(r), np.ones_like(r)]
    for _ in range(DEGREES - 1):
        t.append(2 * r * t[-1] - t[-2])
        dt.append(2 * t[-2] + 2 * r * dt[-1] - dt[-2])
    return np.stack(t[1:], -1), np.stack(dt[1:], -1)


class Model:
    """The parameters: the cell's six phases, the gain, the offset and the
    readout."""
    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        self.theta = rng.uniform(0, 2 * np.pi, 6)
        self.gain, self.offset = GAIN, 0.
        self.readout = rng.normal(0, 1 / np.sqrt(DEGREES),
                                  (OUTPUT_DIM, DEGREES))

    def parameters(self):
        return np.r_[self.theta, self.gain, self.offset,
                     self.readout.ravel()]

    def assign(self, params):
        self.theta, self.gain, self.offset = params[:6], params[6], params[7]
        self.gain = float(np.clip(self.gain, 0., 1.))
        self.readout = params[8:].reshape(OUTPUT_DIM, DEGREES)

    def embed(self, r):
        return self.readout @ chebyshev(r)[0].sum(0)


def settle(model, ts, nudge=None, beta=0., steps=20000, tol=None,
           dtype=np.float64):
    """The equilibria of the networks of the couplings ``ts``, settled
    together from rest, nudged by ``-beta`` times the gradient of ``nudge``,
    until no state moves by more than ``tol`` in a step (by default the
    resolution of ``dtype``) or after ``steps``."""
    tol = 4 * np.finfo(dtype).eps if tol is None else tol
    zs = [np.zeros(len(t), dtype) for t in ts]
    ts = [t.astype(dtype) for t in ts]
    gain, offset = dtype(model.gain), dtype(model.offset)
    for _ in range(steps):
        rs = [np.tanh(z) for z in zs]
        forces = [gain * (t @ (1 + r)) + offset for t, r in zip(ts, rs)]
        if beta:
            for force, grad in zip(forces, nudge(model, rs)):
                force -= beta * grad
        new = [(1 - LEAK) * z + LEAK * f for z, f in zip(zs, forces)]
        moved = max(float(np.abs(n - z).max()) for n, z in zip(new, zs))
        zs = new
        if moved <= tol:
            break
    return [np.tanh(z) for z in zs]


def cosine(model, rs):
    """BREC's loss on the pair ``rs = [r_G, r_H]`` and its gradients in the
    two states and in the readout."""
    feats = [chebyshev(r) for r in rs]
    fs = [model.readout @ p.sum(0) for p, _ in feats]
    norms = [np.linalg.norm(f) for f in fs]
    cos = fs[0] @ fs[1] / (norms[0] * norms[1])
    if cos <= 0:
        return 0., [np.zeros_like(r) for r in rs], np.zeros_like(model.readout)
    dfs = [fs[1] / (norms[0] * norms[1]) - cos * fs[0] / norms[0] ** 2,
           fs[0] / (norms[0] * norms[1]) - cos * fs[1] / norms[1] ** 2]
    dstates = [dp @ (model.readout.T @ d) for d, (_, dp) in zip(dfs, feats)]
    dreadout = sum(np.outer(d, p.sum(0)) for d, (p, _) in zip(dfs, feats))
    return cos, dstates, dreadout


def nudge_cosine(model, rs):
    return cosine(model, rs)[1]


def gradient(model, adjs, beta=1e-3, step=1e-5):
    """The EP estimate of the gradient of BREC's loss on one pair."""
    ts = [couplings(adj, model.theta) for adj in adjs]
    free = settle(model, ts)
    loss, _, dreadout = cosine(model, free)
    if loss == 0:
        return loss, np.zeros_like(model.parameters())
    plus = settle(model, ts, nudge_cosine, beta)
    minus = settle(model, ts, nudge_cosine, -beta)
    splus, sminus = [1 + r for r in plus], [1 + r for r in minus]
    dtheta = np.zeros(6)
    for i, e in enumerate(np.eye(6)):
        for adj, sp_, sm in zip(adjs, splus, sminus):
            dt = (couplings(adj, model.theta + step * e)
                  - couplings(adj, model.theta - step * e)) / (2 * step)
            dtheta[i] += -model.gain / (4 * beta) * (sp_ @ dt @ sp_
                                                     - sm @ dt @ sm)
    dgain = sum(-1 / (4 * beta) * (sp_ @ t @ sp_ - sm @ t @ sm)
                for t, sp_, sm in zip(ts, splus, sminus))
    doffset = sum(-1 / (2 * beta) * (p.sum() - m.sum())
                  for p, m in zip(plus, minus))
    return loss, np.r_[dtheta, dgain, doffset, dreadout.ravel()]


def loss_value(model, adjs):
    """BREC's loss at the free equilibrium, for checking the estimate."""
    ts = [couplings(adj, model.theta) for adj in adjs]
    return cosine(model, settle(model, ts))[0]


# ------------------------------------------------------------ BREC's test

def train(model, test, epochs=EPOCH, rate=LEARNING_RATE,
          decay=WEIGHT_DECAY):
    """BREC's training on the copies of one pair: Adam with weight decay on
    the relabelled pairs, batch of eight pairs (sixteen graphs), stopping
    early when the loss falls below LOSS_THRESHOLD.

    The networks are exactly equivariant, so every relabelled pair has the
    same loss and gradient up to rounding: a batch is represented by its
    first pair, which costs one pair instead of eight.
    """
    adjs = [adjacency(test[0]), adjacency(test[1])]
    params = model.parameters()
    first, second = np.zeros_like(params), np.zeros_like(params)
    history, time = [], 0
    for _ in range(epochs):
        losses = []
        for _ in range(NUM_RELABEL // 8):
            time += 1
            loss, grad = gradient(model, adjs)
            grad = grad + decay * params
            first = .9 * first + .1 * grad
            second = .999 * second + .001 * grad ** 2
            params = params - rate * (first / (1 - .9 ** time)) / (
                np.sqrt(second / (1 - .999 ** time)) + 1e-8)
            model.assign(params)
            params = model.parameters()
            losses.append(loss)
        history.append(float(np.mean(losses)))
        if history[-1] < LOSS_THRESHOLD:
            break
    return history


def embeddings(model, graphs, dtype=np.float32):
    """The embeddings of the graphs, each computed from its own relabelled
    adjacency, the dynamics and readout in ``dtype``."""
    out = []
    for g in graphs:
        t = couplings(adjacency(g), model.theta)
        r = settle(model, [t], dtype=dtype)[0]
        out.append(model.embed(r.astype(np.float64)).astype(dtype))
    return np.array(out, dtype=np.float64)


def t_square(model, graphs, dtype=np.float32):
    """BREC's T^2 of the 32 relabelled pairs: the mean difference of the
    embeddings against the pseudo-inverse of their covariance."""
    f = embeddings(model, graphs, dtype)
    d = (f[0::2] - f[1::2]).T
    mean = d.mean(1, keepdims=True)
    return float(mean.T @ np.linalg.pinv(np.cov(d)) @ mean)


def decide(t2_test, t2_reliability, threshold=THRESHOLD):
    """BREC's decision: separated, and reliable."""
    separated = t2_test > threshold and not np.isclose(
        t2_test, t2_reliability, atol=EPSILON_CMP, rtol=0)
    return bool(separated), bool(t2_reliability < threshold)


def run_pair(data, pair, seed=0, epochs=EPOCH, rate=1e-3,
             dtype=np.float32):
    """
    BREC's protocol on one pair: train, then test and check. The units
    start centred, the offset cancelling the mean count, and the gain at
    GAIN, near the critical gain ``1 / lambda_max(T)`` where the equilibrium
    responds most to the couplings; the rate is the largest of BREC's grid.
    """
    test, reliability = copies(data, pair)
    model = Model(seed)
    model.offset = -model.gain * float(
        couplings(adjacency(test[0]), model.theta).sum(1).mean())
    before = [t_square(model, test, dtype), t_square(model, reliability,
                                                     dtype)]
    history = train(model, test, epochs, rate) if epochs else []
    after = [t_square(model, test, dtype), t_square(model, reliability,
                                                    dtype)]
    return dict(pair=pair, seed=seed, history=history,
                untrained=dict(zip(("test", "reliability"), before),
                               decision=decide(*before)),
                trained=dict(zip(("test", "reliability"), after),
                             decision=decide(*after)),
                vertices=int(adjacency(test[0]).shape[0]))
