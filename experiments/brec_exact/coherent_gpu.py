"""Coherent light on the GPU: exact profiles in double precision and the
noise of every coincidence at any brightness from a phase Monte Carlo.

Coherent light with a random phase per pulse is a mixture of coherent
states, so the device is a classical linear system: the output fields are
beta = W alpha with alpha_c = sqrt(w) e^{i phi_c}, and the counts of one
tick are independent Poisson variables with means |beta_a|^2.  Two things
follow.

* The exact rates F_S = E_phi prod_{a in S} |beta_a|^2 are the polynomials
  of ``brecphot.rate_tensors``; the same Gram-tensor algebra runs on the
  GPU in complex128, which is what the 1e-10 relative gaps need.

* The variance of a coincidence product at brightness w is a polynomial in
  w whose coefficients are phase averages of monomials in the unit-
  brightness intensities J_a = |(W e^{i phi})_a|^2.  With Poisson counts,
  E[n_a^2 | beta] = I_a^2 + I_a and I = w J, so

      E[prod_{a in S} n_a^2] = sum_{T subset S} w^(N+|T|) E[prod_{a in T} J_a^2 prod_{a in S-T} J_a],

  and Var = that minus (w^N E prod J_a)^2.  The moments M_T are estimated
  from K random phase draws, in single precision, which is enough for a
  noise level; the gap itself stays exact.  As w grows the Poisson terms
  fade and the variance tends to the intensity floor w^(2N) (E prod J^2 -
  (E prod J)^2), the regime of a bright laser read by photodiodes.
"""

import time
from itertools import combinations

import numpy as np

import brecphot as bp


def to_torch(W, device):
    import torch
    return torch.as_tensor(np.ascontiguousarray(W), dtype=torch.complex128,
                           device=device)


def gram_tensor_torch(W, Wc, P, Q):
    import torch
    positions = sorted(set(P) | set(Q))
    factors = []
    for j in positions:
        m = torch.ones_like(W)
        for _ in range(P.count(j)):
            m = m * W
        for _ in range(Q.count(j)):
            m = m * Wc
        factors.append(m)
    n = W.shape[0]
    k = len(positions)
    if k == 1:
        return positions, factors[0].sum(dim=1)
    if k == 2:
        return positions, factors[0] @ factors[1].T
    if k not in (3, 4):
        raise NotImplementedError(k)
    # the products over pairs of positions are n^2 x slots; sum over the
    # slots in chunks so that they never exceed a few GB on the card
    slots = W.shape[1]
    chunk = max(1, int(2 ** 28 // (n * n)))
    shape = (n, n, n) if k == 3 else (n, n, n, n)
    total = torch.zeros(n * n, n ** (k - 2), dtype=W.dtype, device=W.device)
    for start in range(0, slots, chunk):
        sl = slice(start, start + chunk)
        left = (factors[0][:, None, sl] * factors[1][None, :, sl]).reshape(n * n, -1)
        if k == 3:
            right = factors[2][:, sl]
        else:
            right = (factors[2][:, None, sl] * factors[3][None, :, sl]).reshape(n * n, -1)
        total += left @ right.T
        del left, right
    return positions, total.reshape(shape)


def rate_tensors_torch(W, N, sources=bp.SOURCES):
    """``brecphot.rate_tensors`` with torch tensors on W's device."""
    import torch
    from itertools import permutations
    n = W.shape[0]
    Wc = W.conj()
    letters = "abcdefgh"[:N]
    products = {}
    for tau in bp.set_partitions(list(range(N))):
        g = {s: float(np.prod([bp.block_weight(bp.LAMBDA[s], len(b)) for b in tau]))
             for s in sources}
        if all(abs(v) < 1e-300 for v in g.values()):
            continue
        for rho in permutations(range(N)):
            inverse = [rho.index(i) for i in range(N)]
            key = tuple(sorted(
                (tuple(sorted(b)), tuple(sorted(inverse[i] for i in b))) for b in tau))
            coef = products.setdefault(key, dict.fromkeys(sources, 0.0))
            for s in sources:
                coef[s] += g[s] * bp.omega(s, rho)
    cache = {}
    total = {s: torch.zeros((n,) * N, dtype=torch.complex128, device=W.device)
             for s in sources}
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
                    cache[(P, Q)] = gram_tensor_torch(W, Wc, P, Q)
            pos, t = cache[(P, Q)]
            operands.append(t)
            subscripts.append("".join(letters[j] for j in pos))
        term = torch.einsum(",".join(subscripts) + "->" + letters, *operands)
        for s in sources:
            if coef[s]:
                total[s] += coef[s] * term
    return {s: t.real for s, t in total.items()}


def profiles_torch(W, nmax, w, sources=bp.SOURCES):
    """Sorted profiles as numpy arrays (sources, M_N), and the index of the
    N-set behind every sorted entry, which the noise needs."""
    import torch
    n = W.shape[0]
    out, order = {}, {}
    for N in range(1, nmax + 1):
        tensors = rate_tensors_torch(W, N, sources)
        idx = torch.as_tensor(np.array(list(combinations(range(n), N))).T,
                              device=W.device)
        rows, ranks = [], []
        for s in sources:
            values = tensors[s][tuple(idx)] * w ** N
            sorted_values, rank = torch.sort(values)
            rows.append(sorted_values.cpu().numpy())
            ranks.append(rank.cpu().numpy())
        out[N] = np.array(rows)
        order[N] = np.array(ranks)
        del tensors
    return out, order


def intensity_moments(W, nmax, samples, device, batch=2048, seed=0):
    """E[prod_{a in T} J_a^2 prod_{a in S-T} J_a] for every N-set S and
    every T subset of S, at unit brightness, by a phase Monte Carlo in single
    precision: a dict N -> array (2^N, M_N) indexed by the bitmask of T."""
    import torch
    n = W.shape[0]
    W32 = W.to(torch.complex64)
    gen = torch.Generator(device=device).manual_seed(seed)
    index = {N: torch.as_tensor(np.array(list(combinations(range(n), N))).T,
                                device=device) for N in range(1, nmax + 1)}
    sums = {N: torch.zeros((2 ** N, index[N].shape[1]), dtype=torch.float64,
                           device=device) for N in range(1, nmax + 1)}
    done = 0
    while done < samples:
        k = min(batch, samples - done)
        phases = torch.rand((W32.shape[1], k), generator=gen, device=device)
        alpha = torch.exp(2j * np.pi * phases).to(torch.complex64)
        J = (W32 @ alpha).abs() ** 2            # (n, k) unit-brightness intensities
        J2 = J * J
        for N in range(1, nmax + 1):
            idx = index[N]
            for mask in range(2 ** N):
                prod = torch.ones((idx.shape[1], k), device=device)
                for i in range(N):
                    prod = prod * (J2 if mask >> i & 1 else J)[idx[i]]
                sums[N][mask] += prod.sum(dim=1).to(torch.float64)
        done += k
    return {N: (s / samples).cpu().numpy() for N, s in sums.items()}


def coherent_variance(moments, N, w):
    """Var[prod_{a in S} n_a] at brightness w for every N-set, from the
    unit-brightness moments; the mean part uses the Monte Carlo mean so that
    the cancellation is consistent."""
    second = np.zeros(moments[N].shape[1])
    for mask in range(2 ** N):
        second += w ** (N + bin(mask).count("1")) * moments[N][mask]
    mean = w ** N * moments[N][0]
    return np.maximum(second - mean ** 2, 0.0), mean


def ideal_samples_at(profile_g, order_g, moments_g, profile_h, order_h,
                     moments_h, nmax, w, beta=0.05):
    """Samples per network for the optimal diagonal test at brightness w:
    the exact gap of the sorted profiles over the variance of the sets behind
    the sorted entries, cumulated over orders 1..nmax."""
    d2 = 0.0
    for N in range(1, nmax + 1):
        var_g, _ = coherent_variance(moments_g, N, w)
        var_h, _ = coherent_variance(moments_h, N, w)
        fg = profile_g[N] * (w / bp.W) ** N
        fh = profile_h[N] * (w / bp.W) ** N
        noise = var_g[order_g[N]] + var_h[order_h[N]]
        mask = noise > 0
        d2 += float(np.sum((fg[mask] - fh[mask]) ** 2 / noise[mask]))
    return 4 * np.log(1 / beta) / d2 if d2 > 0 else float("inf")


def benchmark(adj, cells, gamma, nmax, device, repeat=1):
    """Wall time of the profiles on numpy (CPU) and torch (GPU)."""
    import torch
    W = bp.path_matrix(bp.network(adj, cells, gamma))
    t0 = time.time()
    cpu = bp.profiles(W, nmax, bp.W)
    t_cpu = time.time() - t0
    Wt = to_torch(W, device)
    torch.cuda.synchronize() if device != "cpu" else None
    t0 = time.time()
    for _ in range(repeat):
        gpu, _ = profiles_torch(Wt, nmax, bp.W)
    torch.cuda.synchronize() if device != "cpu" else None
    t_gpu = (time.time() - t0) / repeat
    agreement = max(float(np.max(np.abs(cpu[N] - gpu[N]) / np.max(cpu[N])))
                    for N in cpu)
    return {"n": int(adj.shape[0]), "slots": int(W.shape[1]), "nmax": nmax,
            "cpu_seconds": t_cpu, "gpu_seconds": t_gpu,
            "speedup": t_cpu / t_gpu, "max_rel_disagreement": agreement}
