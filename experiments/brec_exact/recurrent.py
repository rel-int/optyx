"""Samples of the device run tick by tick, and the exact moments of
consecutive ticks.

The device is the network of `brecphot.network`: at tick t the loop modes
hold the field x_t, the sources emit s_t, the detectors see the field
y_t = C x_t + D s_t and the loop keeps x_{t+1} = A x_t + B s_t.  Started
from the vacuum and run for as many ticks as the path matrix has delays,
the field y_t is the stationary one, sum_k W_k s_{t-k}, up to the tolerance
of `brecphot.path_matrix`.

* Coherent light: s_v = sqrt(w) exp(i phi) with a fresh phase per pulse.
* Thermal light: s_v complex Gaussian with E|s_v|^2 = w.
  For both, the counts are Poisson of the intensities |y_t|^2, since the
  Glauber-Sudarshan function of either source is a distribution of
  coherent amplitudes.
* Distinguishable photons: every source emits a photon with probability w,
  which no other photon interferes with; the one emitted by v at tick t is
  counted by detector a at tick t + k with probability |W[a, k n + v]|^2,
  the amplitude of that path in the unrolled network, and lost otherwise.

Consecutive ticks are distinct output modes of the same unrolled
interferometer.  Index the slots by their age at tick t + k: detector b at
tick t + k has the row W[b], detector a at tick t has W[a] delayed by k
ticks, i.e. k n zero columns in front.  The two-time moment
E[prod_{a in S} n_a(t) prod_{b in S} n_b(t + k)] is then the order-2N rate
of Eq. (sources) on these 2N distinct rows, for every source including
single photons, and the integrated autocorrelation time of prod_S n_a is
tau = 1 + 2 sum_{k >= 1} Cov_k / Var.
"""
from itertools import combinations

import numpy as np

import brecphot as bp
import photon_noise as pn

SAMPLED = ("coherent", "thermal", "photons p=0")


def delays(W, n):
    """The number of ticks the path matrix remembers."""
    return W.shape[1] // n


def set_index(n, N):
    """The N-sets of n detectors, as an array (N, M_N)."""
    return np.array(list(combinations(range(n), N))).T


# ------------------------------------------------------------- the device

class Device:
    """The blocks of one tick and the path matrix of a network, as torch
    tensors on `device`."""

    def __init__(self, blocks, device="cpu"):
        import torch
        self.n = blocks[3].shape[0]
        self.W = bp.path_matrix(blocks)
        self.A, self.B, self.C, self.D = (
            torch.as_tensor(m, dtype=torch.complex128, device=device) for m in blocks)
        self.device = device

    def counts(self, source, w, chains, ticks, seed=0, burn_in=None):
        """Yield the counts (chains, n) of `chains` independent runs of the
        device, tick after tick, after `burn_in` ticks (by default as many
        as the path matrix has delays)."""
        import torch
        gen = torch.Generator(device=self.device).manual_seed(seed)
        burn_in = delays(self.W, self.n) if burn_in is None else burn_in
        steps = (self.photons if source == "photons p=0" else self.fields)(
            source, w, chains, gen)
        for t in range(burn_in + ticks):
            y = next(steps)
            if t >= burn_in:
                yield y

    def fields(self, source, w, chains, gen):
        import torch
        x = torch.zeros((chains, self.A.shape[0]), dtype=torch.complex128, device=self.device)
        shape = (chains, self.n)
        while True:
            if source == "coherent":
                phase = torch.rand(shape, generator=gen, device=self.device, dtype=torch.float64)
                s = np.sqrt(w) * torch.exp(2j * np.pi * phase)
            elif source == "thermal":
                re, im = (torch.randn(shape, generator=gen, device=self.device, dtype=torch.float64)
                          for _ in range(2))
                s = np.sqrt(w / 2) * torch.complex(re, im)
            else:
                raise ValueError(source)
            y = x @ self.C.T + s @ self.D.T
            x = x @ self.A.T + s @ self.B.T
            yield torch.poisson(y.abs() ** 2, generator=gen)

    def photons(self, source, w, chains, gen):
        import torch
        n, K = self.n, delays(self.W, self.n)
        # fate of a photon of vertex v: index k n + a for (delay k, detector a)
        prob = np.abs(self.W.reshape(n, K, n)) ** 2                    # (a, k, v)
        fate = torch.as_tensor(np.cumsum(prob.transpose(2, 1, 0).reshape(n, K * n), axis=1),
                               device=self.device)
        future = torch.zeros((K, chains, n), dtype=torch.float64, device=self.device)
        t = 0
        while True:
            emitted = torch.rand((chains, n), generator=gen, device=self.device) < w
            for v in range(n):
                chain = torch.nonzero(emitted[:, v]).ravel()
                u = torch.rand(chain.shape, generator=gen, device=self.device, dtype=torch.float64)
                idx = torch.searchsorted(fate[v], u)
                kept = idx < K * n
                k, a = idx[kept] // n, idx[kept] % n
                future.index_put_(((t + k) % K, chain[kept], a),
                                  torch.ones_like(u[kept]), accumulate=True)
            y = future[t % K].clone()
            future[t % K] = 0
            t += 1
            yield y


# ------------------------------------------------------------ statistics

class Moments:
    """Sums over the ticks of every N-set product X = prod_{a in S} n_a,
    centred at its exact rate F, per batch of chains: powers 1 to 4 of
    X - F, the squares of the sums over time blocks of `block` ticks, and
    the lagged products sum_t (X_t - F)(X_{t+k} - F), k = 1..lags, of the
    sets in `tracked`."""

    def __init__(self, n, rates, chains, batches, block, lags, tracked, device="cpu"):
        import torch
        self.device, self.block, self.lags = device, block, lags
        self.index = {N: torch.as_tensor(set_index(n, N), device=device) for N in rates}
        self.rates = {N: torch.as_tensor(F, device=device) for N, F in rates.items()}
        self.tracked = {N: torch.as_tensor(np.asarray(s, dtype=np.int64), device=device)
                        for N, s in tracked.items()}
        self.batch = torch.nn.functional.one_hot(
            torch.arange(chains, device=device) % batches, batches).to(torch.float64)
        zeros = lambda N: torch.zeros((batches, len(rates[N])), dtype=torch.float64, device=device)
        self.power = {N: [zeros(N) for _ in range(4)] for N in rates}
        self.blocks = {N: zeros(N) for N in rates}
        self.current = {N: torch.zeros((chains, len(rates[N])), dtype=torch.float64, device=device)
                        for N in rates}
        self.lagged = {N: torch.zeros((lags, len(s)), dtype=torch.float64, device=device)
                       for N, s in tracked.items()}
        self.history = {N: torch.zeros((lags, chains, len(s)), dtype=torch.float64, device=device)
                        for N, s in tracked.items()}
        self.ticks = 0
        self.chains = chains

    def add(self, y):
        self.ticks += 1
        for N, idx in self.index.items():
            x = y[:, idx[0]]
            for i in range(1, N):
                x = x * y[:, idx[i]]
            x = x - self.rates[N]
            p = x
            for k in range(4):
                self.power[N][k] += self.batch.T @ p
                p = p * x
            self.current[N] += x
            if self.ticks % self.block == 0:
                self.blocks[N] += self.batch.T @ self.current[N] ** 2
                self.current[N].zero_()
            xs = x[:, self.tracked[N]]
            ring = (self.ticks - 1 - np.arange(1, self.lags + 1)) % self.lags
            self.lagged[N] += (self.history[N][ring] * xs).sum(dim=1)
            self.history[N][(self.ticks - 1) % self.lags] = xs

    def result(self):
        """Per order: the per-batch power sums (4, batches, M), the block
        sums (batches, M), the lagged sums (lags, tracked) and the number of
        terms behind each, as numpy arrays."""
        out = {}
        blocks = self.ticks // self.block
        for N in self.index:
            out[N] = {
                "power": np.stack([p.cpu().numpy() for p in self.power[N]]),
                "blocks": self.blocks[N].cpu().numpy(),
                "lagged": self.lagged[N].cpu().numpy(),
                "per_batch": self.ticks * self.batch.sum(dim=0).cpu().numpy(),
                "block_terms": blocks * self.batch.sum(dim=0).cpu().numpy(),
                "lag_terms": np.array([(self.ticks - k) * self.chains for k in range(1, self.lags + 1)]),
            }
        return out


def summary(F, var, stats, block):
    """From the sums of `Moments` and the exact rates and variances of one
    order: the sample mean and variance of every set, the standard error of
    the variance from the spread of the batches, the integrated
    autocorrelation time from the time blocks, and the lagged
    autocovariances of the tracked sets."""
    power, terms = stats["power"], stats["per_batch"][:, None]
    T = terms.sum()
    s1 = power[0].sum(0) / T
    mean = F + s1
    sample_var = power[1].sum(0) / T - s1 ** 2
    batch_mean = power[0] / terms
    batch_var = power[1] / terms - batch_mean ** 2
    B = power.shape[1]
    var_se = batch_var.std(axis=0, ddof=1) / np.sqrt(B)
    m4 = (power[3].sum(0) - 4 * s1 * power[2].sum(0) + 6 * s1 ** 2 * power[1].sum(0)
          - 3 * s1 ** 4 * T) / T
    tau_blocks = (stats["blocks"].sum(0) / stats["block_terms"].sum()
                  - block ** 2 * s1 ** 2) / (block * np.maximum(sample_var, 1e-300))
    lagged = stats["lagged"] / stats["lag_terms"][:, None]
    return {"T": float(T), "rate": F, "var": var, "mean": mean, "sample_var": sample_var,
            "var_se": var_se, "m4": m4, "tau_blocks": tau_blocks, "lagged": lagged}


# ---------------------------------------------------------- exact moments

def rates_variances(W, nmax, source, w, device="cpu"):
    """The exact rates and variances of every N-set, N <= nmax, from
    `photon_noise`: dict N -> (F, Var), arrays (M_N,)."""
    import torch
    W = torch.as_tensor(W, dtype=torch.complex128, device=device)
    out = pn.rates_and_variances(W, nmax, {source: bp.LAMBDA[source]}, w, device)
    return {N: (r[source], v[source]) for N, (r, v) in out.items()}


def distinct_moments(R, source, device="cpu", chunk=256):
    """The order-K rate divided by w^K of K distinct detectors with rows R
    (batch, K, slots) of a path matrix, for one source."""
    import torch
    batch, K, _ = R.shape
    patterns, keys, weights = pn.tables(K, ())
    coef = torch.as_tensor(pn.coefficients(weights, {source: bp.LAMBDA[source]}, K)[source],
                           dtype=torch.complex128, device=device)
    ids = torch.as_tensor(np.where(keys >= 0, keys, len(patterns)), device=device)
    out = np.zeros(batch)
    for start in range(0, batch, chunk):
        rows = torch.as_tensor(R[start:start + chunk], dtype=torch.complex128, device=device)
        values = torch.ones((len(patterns) + 1, rows.shape[0]), dtype=torch.complex128, device=device)
        for i, (p, q) in enumerate(patterns):
            factor = torch.ones_like(rows[:, 0])
            for j in range(K):
                if p[j]:
                    factor = factor * rows[:, j]
                if q[j]:
                    factor = factor * rows[:, j].conj()
            values[i] = factor.sum(dim=1)
        prod = values[ids[:, 0]]
        for col in range(1, K):
            prod = prod * values[ids[:, col]]
        out[start:start + rows.shape[0]] = (coef @ prod).real.cpu().numpy()
    return out


def two_time_rows(W, n, S, k):
    """The rows of the detectors of S at tick t, then at tick t + k, over
    the slots indexed by their age at tick t + k."""
    later = W[list(S)]
    earlier = np.concatenate([np.zeros((len(S), k * n), dtype=W.dtype), later], axis=1)
    later = np.concatenate([later, np.zeros((len(S), k * n), dtype=W.dtype)], axis=1)
    return np.concatenate([earlier, later])


def autocovariances(W, n, sets, lags, source, w, device="cpu"):
    """Cov(X_t, X_{t+k}) of X = prod_{a in S} n_a, for every set of `sets`
    and k = 1..lags: array (lags, len(sets))."""
    N = len(sets[0])
    F = w ** N * distinct_moments(np.stack([W[list(S)] for S in sets]), source, device)
    joint = [distinct_moments(np.stack([two_time_rows(W, n, S, k) for S in sets]), source, device)
             for k in range(1, lags + 1)]
    return w ** (2 * N) * np.array(joint) - F ** 2


def tau_exact(cov, var):
    """The integrated autocorrelation time 1 + 2 sum_k Cov_k / Var."""
    return 1 + 2 * cov.sum(axis=0) / var
