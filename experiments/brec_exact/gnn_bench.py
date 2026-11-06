"""Probabilistic graph networks against the photonic architecture, under the
same label-free tests.

Models, all permutation-equivariant with weights shared across vertices,
untrained, one scalar readout per vertex, ``y_v = (w . x_v)^2`` so that
every N-set product is nonnegative as a coincidence is:

* ``step-lin``, ``step-tanh``: a recurrent network driven by fresh Gaussian
  noise at every tick, x_v <- act(W_s x_v + W_n mean_{u~v} x_u + xi_v), with
  the linear map a contraction of factor rho so that it forgets in about
  1/(1-rho) ticks; run to stationarity and sampled once per tick, as the
  photonic device is.
* ``init-lin``, ``init-tanh``: Gaussian noise at initialisation only,
  unrolled to depth L with the same update and no further noise, one
  forward pass per sample (random node initialisation).
* ``coherent``: the photonic network with random-phase coherent light at
  brightness w, sampled: phases, one matrix-vector product, Poisson counts.
* ``loop-coherent``, ``loop-thermal``: the same chip run tick by tick, with
  random-phase coherent or thermal light at brightness w and measurement
  feedforward: the count n_v(t) of the detector of chip v sets the phase
  kappa pi n_v(t) / nbar on the memory mode of chip v, `delay` ticks later,
  with nbar the mean count per detector of the linear chip.  The rule is the
  same at every vertex, so the network stays equivariant; kappa = 0 is the
  linear chip, the path-matrix ``coherent`` model read tick by tick.

The linear models are Gaussian processes, so their moments are exact by
Wick's theorem: E[prod_{a in S} z_a^2] is the hafnian of the covariance of
z on the multiset S + S.
"""

from itertools import combinations

import numpy as np

import brecphot as bp

MODELS = ("step-lin", "step-tanh", "init-lin", "init-tanh", "coherent", "loop-coherent", "loop-thermal")
WIDTH = 8
DEPTH = 4
RHO = 0.95
BURN_IN = 100
BATCHES = 16
SPLITS = 1999
ALPHA = 0.05


# ------------------------------------------------------------- the models

def neighbour_mean(adj):
    deg = adj.sum(axis=1, keepdims=True).astype(float)
    return adj / np.maximum(deg, 1)


def cell_weights(seed, width=WIDTH, rho=RHO):
    """An orthogonal feature map Q, drawn by QR of a Gaussian matrix, and
    the readout w.  The update is x_v <- rho Q (x_v + mean_{u~v} x_u) / 2:
    the lazy walk operator (I + A_mean) / 2 has spectrum in [0, 1] and Q
    preserves norms, so every direction decays at most at rate rho and the
    slow modes exactly at rho, a memory of 1 / (1 - rho) ticks whatever the
    graph."""
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.normal(size=(width, width)))
    q = q * np.sign(np.diag(r))
    w = rng.normal(size=width)
    return rho * q, w / np.linalg.norm(w)


class Sampler:
    """Streams chunks of readouts y (chunk, n) for one model on one graph."""

    def __init__(self, model, adj, seed, device, noise_seed=0, brightness=1.0,
                 gamma=0.9, width=WIDTH, depth=DEPTH, chunk=1024, kappa=0.0, delay=1):
        """`seed` draws the cell or the weights, shared by every graph and
        copy of one experiment; `noise_seed` draws the light or the noise."""
        import torch
        self.model, self.device, self.chunk = model, device, chunk
        self.n = adj.shape[0]
        self.gen = torch.Generator(device=device).manual_seed(
            seed * 7919 + 104729 * noise_seed + 1)
        if model == "coherent":
            # a tail of the path matrix below 1e-4 in amplitude moves a rate
            # by 1e-8 relative, far below any sampling noise
            W = bp.path_matrix(bp.network(adj, bp.cell(seed), gamma), tol=1e-4)
            self.W = torch.as_tensor(W, dtype=torch.complex64, device=device)
            self.brightness = brightness
            return
        if model.startswith("loop"):
            A, B, C, D = bp.network(adj, bp.cell(seed), gamma)
            U = np.block([[A, B], [C, D]])
            self.UT = torch.as_tensor(U.T, dtype=torch.complex64, device=device)
            self.loop = A.shape[0]
            self.memory = torch.arange(self.loop - self.n, self.loop, device=device)
            W = bp.path_matrix((A, B, C, D))
            self.nbar = float(brightness * np.mean(np.sum(np.abs(W) ** 2, axis=1)))
            self.brightness, self.kappa, self.delay = brightness, kappa, delay
            self.state = None
            return
        q, w = cell_weights(seed, width)
        lazy = (np.eye(self.n) + neighbour_mean(adj)) / 2
        self.L = torch.as_tensor(lazy, dtype=torch.float32, device=device)
        self.q = torch.as_tensor(q, dtype=torch.float32, device=device)
        self.w = torch.as_tensor(w, dtype=torch.float32, device=device)
        self.width, self.depth = width, depth
        self.tanh = model.endswith("tanh")
        # noise of variance 1 - rho^2 per step gives the linear stationary
        # pre-activation unit variance, so that tanh works in its natural
        # regime rather than saturated; the linear models are scale-free
        self.noise_scale = float(np.sqrt(1 - RHO ** 2))
        self.state = None

    def state_dict(self):
        out = {"gen": self.gen.get_state().cpu().numpy()}
        if getattr(self, "state", None) is not None:
            out["state"] = self.state.cpu().numpy()
        if getattr(self, "history", None) is not None:
            out["history"] = self.history.cpu().numpy()
        return out

    def load_state_dict(self, state, device):
        import torch
        self.gen.set_state(torch.as_tensor(state["gen"]).to(torch.uint8).cpu())
        if "state" in state:
            self.state = torch.as_tensor(state["state"], device=device)
        if "history" in state:
            self.history = torch.as_tensor(state["history"], device=device)

    def light(self, k):
        """One pulse per source and stream: (k, n) amplitudes of mean |a|^2 = w."""
        import torch
        if self.model == "loop-coherent":
            phases = torch.rand((k, self.n), generator=self.gen, device=self.device)
            return torch.exp(2j * np.pi * phases).to(torch.complex64) * np.sqrt(self.brightness)
        re, im = (torch.randn((k, self.n), generator=self.gen, device=self.device) for _ in range(2))
        return (re + 1j * im).to(torch.complex64) * np.sqrt(self.brightness / 2)

    def tick(self):
        """One tick of every stream: interfere, count, and feed the counts of
        `delay` ticks ago forward as phases on the memory modes."""
        import torch
        x = torch.cat([self.state, self.light(self.state.shape[0])], dim=1) @ self.UT
        self.state, field = x[:, :self.loop], x[:, self.loop:]
        counts = torch.poisson((field.abs() ** 2).float(), generator=self.gen)
        if self.kappa:
            self.history = torch.cat([self.history[1:], counts[None]])
            phase = torch.exp(1j * (self.kappa * np.pi / self.nbar) * self.history[0]).to(torch.complex64)
            self.state[:, self.memory] = self.state[:, self.memory] * phase
        return counts

    def step(self, x, noise):
        """One update on a batch of states x (chunk, n, width)."""
        import torch
        pre = torch.einsum("vu,buw->bvw", self.L, x) @ self.q.T
        if noise is not None:
            pre = pre + noise
        return torch.tanh(pre) if self.tanh else pre

    def readout(self, x):
        return (x @ self.w) ** 2

    def chunks(self, total):
        """Yield readout chunks until `total` samples have been produced."""
        import torch
        done = 0
        if self.model == "coherent":
            while done < total:
                k = min(self.chunk, total - done)
                phases = torch.rand((k, self.W.shape[1]), generator=self.gen, device=self.device)
                alpha = torch.exp(2j * np.pi * phases).to(torch.complex64) * np.sqrt(self.brightness)
                intensity = (alpha @ self.W.T).abs() ** 2
                yield torch.poisson(intensity, generator=self.gen).to(torch.float64)
                done += k
            return
        if self.model.startswith("loop"):
            # `chunk` independent streams from the vacuum, advanced in
            # lockstep; stream i feeds batch i mod B, so batches are
            # independent runs on the same schedule, stationary or not
            if self.state is None:
                self.state = torch.zeros((self.chunk, self.loop), dtype=torch.complex64, device=self.device)
                self.history = torch.zeros((self.delay, self.chunk, self.n), device=self.device)
                for _ in range(BURN_IN):
                    self.tick()
            while done < total:
                k = min(self.chunk, total - done)
                yield self.tick()[:k].to(torch.float64)
                done += k
            return
        if self.model.startswith("init"):
            while done < total:
                k = min(self.chunk, total - done)
                x = torch.randn((k, self.n, self.width), generator=self.gen, device=self.device)
                for _ in range(self.depth):
                    x = self.step(x, None)
                yield self.readout(x).to(torch.float64)
                done += k
            return
        # stationary stream: `chunk` independent streams advanced in lockstep;
        # consecutive samples of one stream are consecutive ticks
        if self.state is None:
            self.state = torch.zeros((self.chunk, self.n, self.width), device=self.device)
            for _ in range(BURN_IN):
                self.state = self.step(self.state, self.noise_scale * torch.randn(
                    self.state.shape, generator=self.gen, device=self.device))
        while done < total:
            k = min(self.chunk, total - done)
            self.state = self.step(self.state, self.noise_scale * torch.randn(
                self.state.shape, generator=self.gen, device=self.device))
            yield self.readout(self.state[:k]).to(torch.float64)
            done += k


# ----------------------------------------------------- batch accumulators

class Accumulator:
    """Per-batch mean and variance of every N-set product, N = 1..nmax; the
    samples are split into `batches` consecutive groups."""

    def __init__(self, n, nmax, total=0, device="cpu", batches=BATCHES):
        import torch
        self.index = {N: torch.as_tensor(np.array(list(combinations(range(n), N))).T,
                                         device=device) for N in range(1, nmax + 1)}
        self.sum = {N: torch.zeros((batches, self.index[N].shape[1]), dtype=torch.float64,
                                   device=device) for N in self.index}
        self.sq = {N: torch.zeros_like(self.sum[N]) for N in self.index}
        self.count = torch.zeros(batches, dtype=torch.float64, device=device)
        self.per_batch = -(-total // batches)
        self.seen = 0
        self.batches = batches

    def add(self, y, stationary=False):
        """Add a chunk; in a stationary stream row i of every chunk is stream
        i, and stream i always feeds batch i mod B, so that batches are
        independent runs and the permutation test's exchangeability holds."""
        import torch
        k = y.shape[0]
        if stationary:
            batch = torch.arange(k, device=y.device) % self.batches
        else:
            batch = torch.arange(self.seen, self.seen + k, device=y.device) % self.batches
        one_hot = torch.nn.functional.one_hot(batch, self.batches).to(torch.float64)  # (k, B)
        for N, idx in self.index.items():
            prod = y[:, idx[0]]
            for i in range(1, N):
                prod = prod * y[:, idx[i]]
            self.sum[N] += one_hot.T @ prod
            self.sq[N] += one_hot.T @ (prod * prod)
        self.count += one_hot.sum(dim=0)
        self.seen += k

    def state_dict(self):
        return {**{f"sum{N}": self.sum[N].cpu().numpy() for N in self.sum},
                **{f"sq{N}": self.sq[N].cpu().numpy() for N in self.sq},
                "count": self.count.cpu().numpy(), "seen": np.array(self.seen)}

    def load_state_dict(self, state, device):
        import torch
        for N in self.sum:
            self.sum[N] = torch.as_tensor(state[f"sum{N}"], device=device)
            self.sq[N] = torch.as_tensor(state[f"sq{N}"], device=device)
        self.count = torch.as_tensor(state["count"], device=device)
        self.seen = int(state["seen"])

    def stats(self):
        """Dict N -> (mean (B, M_N), variance (B, M_N)) as numpy arrays."""
        out = {}
        c = self.count[:, None]
        for N in self.index:
            mean = self.sum[N] / c
            var = (self.sq[N] / c - mean ** 2).clamp(min=0)
            out[N] = (mean.cpu().numpy(), var.cpu().numpy())
        return out, self.count.cpu().numpy()


def chunk_size(n, nmax):
    """Samples per chunk such that a chunk of N-set products is about 2 GB."""
    from math import comb
    return int(max(32, min(32768, 2.5e8 // comb(n, nmax))) // BATCHES * BATCHES)


def sampled_statistics(model, adj, seed, total, nmax, device, noise_seed=0, **kw):
    sampler = Sampler(model, adj, seed, device, noise_seed=noise_seed,
                      chunk=chunk_size(adj.shape[0], nmax), **kw)
    acc = Accumulator(adj.shape[0], nmax, total, device)
    stationary = model.startswith(("step", "loop"))
    for y in sampler.chunks(total):
        acc.add(y, stationary)
    return acc.stats()


# ------------------------------------------------------ exact linear models

def perfect_matchings(k):
    """All perfect matchings of range(k), k even, as lists of pairs."""
    if k == 0:
        return [[]]
    out = []
    for j in range(1, k):
        rest = [i for i in range(1, k) if i != j]
        for m in perfect_matchings(k - 2):
            out.append([(0, j)] + [(rest[a], rest[b]) for a, b in m])
    return out


def hafnians_of_doubled_sets(C, sets):
    """E[prod_{a in S} z_a^2] for every set S (rows of `sets`), z Gaussian
    with covariance C, by Wick's theorem: the hafnian of C on the multiset
    S + S, summed over the perfect matchings of the 2N doubled indices."""
    N = sets.shape[1]
    doubled = np.repeat(sets, 2, axis=1)  # (M, 2N): a, a, b, b, ...
    total = np.zeros(sets.shape[0])
    for matching in perfect_matchings(2 * N):
        term = np.ones(sets.shape[0])
        for i, j in matching:
            term = term * C[doubled[:, i], doubled[:, j]]
        total += term
    return total


def linear_covariance(model, adj, seed, width=WIDTH, depth=DEPTH):
    """Covariance of the pre-readout z_v = w . x_v across vertices."""
    from scipy.linalg import solve_discrete_lyapunov
    n = adj.shape[0]
    q, w = cell_weights(seed, width)
    M = np.kron((np.eye(n) + neighbour_mean(adj)) / 2, q)
    if model == "step-lin":
        # the step noise has variance 1 - rho^2, see Sampler.noise_scale
        sigma = solve_discrete_lyapunov(M, (1 - RHO ** 2) * np.eye(n * width))
    else:
        P = np.linalg.matrix_power(M, depth)
        sigma = P @ P.T
    proj = np.kron(np.eye(n), w[None, :])
    return proj @ sigma @ proj.T


def exact_linear_profiles(model, adj, seed, nmax):
    """Sorted profiles of E[prod_{a in S} z_a^2] for the linear models."""
    C = linear_covariance(model, adj, seed)
    n = C.shape[0]
    out = {}
    for N in range(1, nmax + 1):
        sets = np.array(list(combinations(range(n), N)))
        out[N] = np.sort(hafnians_of_doubled_sets(C, sets))
    return out


# ------------------------------------------------------------ the protocol

def levels(values, errors, factor=10.0):
    """Rank of the level of every entry: a new level starts at every gap
    wider than `factor` combined standard errors in the sorted values."""
    order = np.argsort(values)
    v, e = values[order], errors[order]
    rank = np.zeros(len(values), dtype=int)
    current = 0
    for i in range(1, len(values)):
        if v[i] - v[i - 1] > factor * np.sqrt(e[i] ** 2 + e[i - 1] ** 2):
            current += 1
        rank[order[i]] = current
    return rank


def batch_features(stats, counts, index, nmax):
    """The feature vectors of every batch: profiles per order, and type
    means and spreads per order >= 2, keyed by type.  The type of a pair is
    its order-2 level with the levels of its two vertices; the type of an
    N-set is the sorted tuple of the codes of its pairs."""
    features = []
    B = counts.shape[0]
    idx = {N: np.asarray(index[N].cpu().numpy() if hasattr(index[N], "cpu") else index[N])
           for N in index}
    n = int(idx[1].max()) + 1
    for b in range(B):
        feats = {}
        mean1, var1 = stats[1][0][b], stats[1][1][b]
        colour = levels(mean1, np.sqrt(var1 / counts[b]))
        pair_code = None
        if nmax >= 2:
            mean2, var2 = stats[2][0][b], stats[2][1][b]
            rank2 = levels(mean2, np.sqrt(var2 / counts[b]))
            a, c = idx[2]
            lo, hi = np.minimum(colour[a], colour[c]), np.maximum(colour[a], colour[c])
            codes = np.stack([rank2, lo, hi], axis=1)
            _, pair_code_of_set = np.unique(codes, axis=0, return_inverse=True)
            pair_code = np.zeros((n, n), dtype=np.int64)
            pair_code[a, c] = pair_code[c, a] = pair_code_of_set.reshape(-1)
        for N in range(1, nmax + 1):
            mean = stats[N][0][b]
            feats[("profile", N)] = np.sort(np.sqrt(np.maximum(mean, 0)))
            if N == 1:
                continue
            sets = idx[N].T  # (M, N)
            if N == 2:
                keys = pair_code[sets[:, 0], sets[:, 1]][:, None]
            else:
                pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
                keys = np.sort(np.stack([pair_code[sets[:, i], sets[:, j]] for i, j in pairs], axis=1), axis=1)
            uniq, type_id = np.unique(keys, axis=0, return_inverse=True)
            type_id = type_id.reshape(-1)
            count = np.bincount(type_id, minlength=len(uniq)).astype(float)
            total = np.bincount(type_id, weights=mean, minlength=len(uniq))
            total_sq = np.bincount(type_id, weights=mean * mean, minlength=len(uniq))
            type_mean = total / count
            type_var = np.maximum(total_sq / count - type_mean ** 2, 0)
            poisson = np.bincount(type_id, weights=np.maximum(mean, 0), minlength=len(uniq)) / count / counts[b]
            feats[("type-mean", N)] = {tuple(k): float(v) for k, v in zip(uniq, type_mean)}
            feats[("type-spread", N)] = {tuple(k): float(max(v - p, 0.0))
                                         for k, v, p in zip(uniq, type_var, poisson)}
        features.append(feats)
    return features


def _matrix(feature, kind):
    """The feature as a dense (2m, d) matrix; type features are aligned on
    the union of their keys, a type absent from a batch counting as zero."""
    if kind == "profile":
        return np.array(feature)
    keys = sorted(set().union(*(f.keys() for f in feature)))
    return np.array([[f.get(k, 0.0) for k in keys] for f in feature])


def _distances(F, masks, kind):
    """The distance between the two halves of every split (rows of `masks`,
    1 for the first half): Euclidean between the half means for a profile,
    variance-normalised for type features, with the variances within the
    halves recomputed for every split."""
    m = masks.sum(axis=1, keepdims=True)
    a, b = masks / m, (1 - masks) / m
    mean_a, mean_b = a @ F, b @ F
    diff = mean_a - mean_b
    if kind == "profile":
        return np.linalg.norm(diff, axis=1)
    sq = F * F
    var_a = np.maximum(a @ sq - mean_a ** 2, 0)
    var_b = np.maximum(b @ sq - mean_b ** 2, 0)
    if kind == "studentised":
        # the variance of a profile entry is a smooth function of its mean:
        # fit v = c0 + c1 m + c2 m^2 over the entries of every split and
        # weight each squared difference by the fitted variance, which is
        # the optimal diagonal statistic up to the quality of the fit
        m = (mean_a + mean_b) / 2
        v = (var_a + var_b) / 2
        fitted = np.empty_like(v)
        for r in range(v.shape[0]):
            X = np.stack([np.ones_like(m[r]), m[r], m[r] ** 2], axis=1)
            coef, *_ = np.linalg.lstsq(X, v[r], rcond=None)
            fitted[r] = np.maximum(X @ coef, 0)
        floor = 1e-6 * np.mean(v, axis=1, keepdims=True) + 1e-300
        return np.sum(diff ** 2 / (fitted + floor), axis=1)
    within = (var_a + var_b) / 2 + 1e-12 * (mean_a ** 2 + mean_b ** 2) + 1e-300
    return np.sum(diff ** 2 / within, axis=1)


def permutation_test(features_g, features_h, nmax, splits=SPLITS, alpha=ALPHA, seed=0):
    """Fig. alg:test: the p-value of every feature vector over random
    splits of the 2m batches, and the decision at alpha / J."""
    rng = np.random.default_rng(seed)
    allf = features_g + features_h
    m = len(features_g)
    masks = np.zeros((splits + 1, 2 * m))
    masks[0, :m] = 1
    for r in range(1, splits + 1):
        masks[r, rng.choice(2 * m, m, replace=False)] = 1
    names = [("profile", N) for N in range(1, nmax + 1)]
    names += [("studentised", N) for N in range(1, nmax + 1)]
    names += [(kind, N) for N in range(2, nmax + 1) for kind in ("type-mean", "type-spread")]
    pvalues = {}
    for name in names:
        kind = name[0]
        source = ("profile", name[1]) if kind == "studentised" else name
        F = _matrix([f[source] for f in allf], "profile" if kind == "studentised" else kind)
        step = max(1, int(2e8 // max(F.shape[1], 1)))
        d = np.concatenate([_distances(F, masks[r:r + step], kind)
                            for r in range(0, masks.shape[0], step)])
        pvalues[f"{kind}-{name[1]}"] = float((1 + np.sum(d[1:] >= d[0])) / (splits + 1))
    J = len(names)
    old = {k: v for k, v in pvalues.items() if not k.startswith("studentised")}
    return {"pvalues": pvalues, "J": J, "J_old": len(old),
            "separated": bool(min(pvalues.values()) <= alpha / J),
            "separated_old": bool(min(old.values()) <= alpha / len(old))}


# --------------------------------------------------- sampled Hotelling test

def sampled_hotelling(model, adj_g, adj_h, seed, total, nmax, device, copies=bp.NUM_RELABEL, **kw):
    """BREC's test with profiles estimated from `total` samples per copy:
    32 copies of G, 32 of H and 32 more of G for the reliability check."""
    rng = np.random.default_rng(seed + 11)
    n = adj_g.shape[0]
    dim = sum(len(list(combinations(range(n), N))) for N in range(1, nmax + 1))
    projection = rng.normal(size=(bp.OUTPUT_DIM, dim)) / np.sqrt(dim)

    def profile_of(adj, k):
        stats, _ = sampled_statistics(model, bp.relabel(adj, rng.permutation(n)) if k else adj,
                                      seed, total, nmax, device, noise_seed=1 + k, **kw)
        vector = np.concatenate([np.sort(stats[N][0].mean(axis=0)) for N in range(1, nmax + 1)])
        return projection @ vector

    X = np.array([profile_of(adj_g, k) for k in range(copies)]).T
    Y = np.array([profile_of(adj_h, copies + k) for k in range(copies)]).T
    Z = np.array([profile_of(adj_g, 2 * copies + k) for k in range(copies)]).T
    # BREC's outputs are of order one; the profiles are scaled to the same
    # size so that the verbatim regularisation means the same for every model
    scale = np.sqrt(np.mean(np.concatenate([X, Y, Z], axis=1) ** 2))
    X, Y, Z = X / scale, Y / scale, Z / scale
    noise = float(np.mean(np.diag(np.cov(X - Z))))
    out = {"noise_variance": noise}
    for label, eps in (("verbatim", bp.EPSILON_MATRIX), ("scaled", bp.EPSILON_MATRIX * noise)):
        t2 = bp.hotelling(X, Y, eps)
        t2r = bp.hotelling(X, Z, eps)
        separated, reliable = bp.brec_decision(t2, t2r)
        out[label] = {"T2_pair": t2, "T2_reliability": t2r, "separated": separated,
                      "reliable": reliable}
    return out
