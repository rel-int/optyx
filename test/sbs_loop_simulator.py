"""
sbs_loop_simulator.py - independent exact simulator for the looped optical channel.

    E(rho) = Tr_x [ Uhat ( rho (x) |q><q| ) Uhat^dag ]     (partial trace, no measurement)

Modes 0..L-1 are looped, modes L..M-1 external. Convention a_j^dag -> sum_i U_ij a_i^dag,
so U[:L,:L] = U_ll and U[:L,L:] = U_xl is the ENTERING block of section 4.

Uhat is never built from permanents: U is factored into two-mode Givens gates and each
gate is applied sparsely in Fock space.

This simulator deliberately does not use Optyx: it is the independent numerical
reference against which the production implementation is checked.
"""
import numpy as np
from math import comb, factorial, sqrt


# ---------------------------------------------------------------- Fock bases
def compositions(M, N):
    """all occupation tuples of length M summing to N"""
    if M == 1:
        return [(N,)]
    out = []
    for first in range(N + 1):
        for rest in compositions(M - 1, N - first):
            out.append((first,) + rest)
    return out


class FockSpace:
    def __init__(self, M, Nmax):
        self.M, self.Nmax = M, Nmax
        self.basis = {N: compositions(M, N) for N in range(Nmax + 1)}
        self.index = {N: {b: i for i, b in enumerate(bs)} for N, bs in self.basis.items()}
        self.dim = {N: len(bs) for N, bs in self.basis.items()}


# ------------------------------------------------- two-mode gate in Fock space
_bs_cache = {}


def two_mode_block(g, m):
    """B[p, n0] = <p, m-p| ghat |n0, m-n0>, convention a_j^dag -> sum_i g_ij c_i^dag"""
    key = (g.tobytes(), m)
    if key in _bs_cache:
        return _bs_cache[key]
    g00, g01, g10, g11 = g[0, 0], g[0, 1], g[1, 0], g[1, 1]
    B = np.zeros((m + 1, m + 1), dtype=complex)
    for n0 in range(m + 1):
        n1 = m - n0
        pref = 1.0 / sqrt(factorial(n0) * factorial(n1))
        for r in range(n0 + 1):
            c1 = comb(n0, r) * (g00 ** r) * (g10 ** (n0 - r))
            for s in range(n1 + 1):
                p = r + s
                c = c1 * comb(n1, s) * (g01 ** s) * (g11 ** (n1 - s))
                B[p, n0] += c * sqrt(factorial(p) * factorial(m - p))
        B[:, n0] *= pref
    _bs_cache[key] = B
    return B


_gate_cache = {}


def gate_matrix(fs, N, p, q, g, tag):
    """sparse matrix of a two-mode gate on modes (p,q), sector N, cached"""
    from scipy.sparse import coo_matrix
    key = (tag, N, p, q)
    if key in _gate_cache:
        return _gate_cache[key]
    basis, idx = fs.basis[N], fs.index[N]
    rows, cols, vals = [], [], []
    blocks = {}
    for i, occ in enumerate(basis):
        n0, n1 = occ[p], occ[q]
        m = n0 + n1
        if m not in blocks:
            blocks[m] = two_mode_block(g, m)
        col = blocks[m][:, n0]
        base = list(occ)
        for pp in range(m + 1):
            if col[pp] == 0:
                continue
            base[p], base[q] = pp, m - pp
            rows.append(idx[tuple(base)])
            cols.append(i)
            vals.append(col[pp])
    Mx = coo_matrix((vals, (rows, cols)), shape=(len(basis), len(basis)), dtype=complex).tocsr()
    _gate_cache[key] = Mx
    return Mx


# ------------------------------------------------------- Givens factorisation
def givens_factor(U, tol=1e-14):
    """returns (gates, diag) with U = G_1^dag ... G_n^dag D, gates=[(p,q,g2),...] in order 1..n"""
    M = U.shape[0]
    W = U.astype(complex).copy()
    gates = []
    for col in range(M - 1):
        for row in range(M - 1, col, -1):
            a, b = W[row - 1, col], W[row, col]
            if abs(b) < tol:
                continue
            r = sqrt(abs(a) ** 2 + abs(b) ** 2)
            g2 = np.array([[np.conj(a) / r, np.conj(b) / r],
                           [-b / r, a / r]], dtype=complex)
            W[[row - 1, row], :] = g2 @ W[[row - 1, row], :]
            gates.append((row - 1, row, g2))
    d = np.diag(W).copy()
    off = np.abs(W - np.diag(d)).max()
    assert off < 1e-10, f"factorisation left off-diagonal {off:.2e}"
    return gates, d


class Interferometer:
    def __init__(self, U, fs):
        self.U, self.fs, self.M = U, fs, U.shape[0]
        self.gates, self.diag = givens_factor(U)
        self.tag = U.tobytes()
        self._ph = {}          # per-sector diagonal phases, see apply()

    def _phases(self, N):
        """Diagonal phase vector of sector N, cached.

        It depends only on (this unitary, N), never on the state, but was being
        rebuilt on every call: a Python loop over the whole sector basis, run
        once per eigenvector and per channel iteration. Caching it is the single
        change; the values are identical, which `validate_multimode.py` and the
        T1 to T9 suite both re-check.
        """
        ph = self._ph.get(N)
        if ph is None:
            basis = self.fs.basis[N]
            ph = np.array([np.prod([self.diag[i] ** occ[i] for i in range(self.M)])
                           for occ in basis])
            self._ph[N] = ph
        return ph

    def apply(self, vec, N):
        """Uhat |vec>, sector N.  U = G_1^dag ... G_n^dag D, so apply D first then G_n^dag ... G_1^dag"""
        v = vec * self._phases(N)
        for gi in range(len(self.gates) - 1, -1, -1):
            p, q, g2 = self.gates[gi]
            v = gate_matrix(self.fs, N, p, q, g2.conj().T, (self.tag, gi)) @ v
        return v


# ------------------------------------------------------------- the channel
class LoopChannel:
    """rho is a dict n -> matrix over the n-photon subspace of the L loop modes"""

    def __init__(self, U, L, q, Ncut):
        self.U, self.L, self.q = U, L, tuple(q)
        self.M = U.shape[0]
        self.x = self.M - L
        self.Q = sum(q)
        self.Ncut = Ncut
        self.fs = FockSpace(self.M, Ncut + self.Q)
        self.itf = Interferometer(U, self.fs)
        self.lb = {n: compositions(L, n) for n in range(Ncut + self.Q + 1)}
        self.li = {n: {b: i for i, b in enumerate(bs)} for n, bs in self.lb.items()}
        self._grp = {}

    def groups(self, N):
        """[(m, basis_idx array, loop_pos array)] grouping the N-sector by external occupation"""
        if N in self._grp:
            return self._grp[N]
        g = {}
        for i, occ in enumerate(self.fs.basis[N]):
            g.setdefault(occ[self.L:], []).append((i, self.li[N - sum(occ[self.L:])][occ[:self.L]]))
        out = []
        for ext, items in g.items():
            m = N - sum(ext)
            idx = np.array([a for a, _ in items])
            pos = np.array([b for _, b in items])
            out.append((m, idx, pos, len(self.lb[m])))
        self._grp[N] = out
        return out

    def _embed(self, loop_occ, N):
        return self.fs.index[N][tuple(loop_occ) + self.q]

    def apply(self, rho):
        out = {}
        for n, block in rho.items():
            if block.shape[0] == 0 or np.abs(block).max() < 1e-18:
                continue
            N = n + self.Q
            w, V = np.linalg.eigh((block + block.conj().T) / 2)
            for lam, col in zip(w, V.T):
                if abs(lam) < 1e-16:
                    continue
                vec = np.zeros(self.fs.dim[N], dtype=complex)
                for a, occ in enumerate(self.lb[n]):
                    if col[a] != 0:
                        vec[self._embed(occ, N)] = col[a]
                psi = self.itf.apply(vec, N)
                for (m, idx, pos, d) in self.groups(N):
                    if m > self.Ncut:
                        continue
                    vals = psi[idx]
                    if not np.any(vals):
                        continue
                    sub = np.zeros(d, dtype=complex)
                    sub[pos] = vals
                    if m not in out:
                        out[m] = np.zeros((d, d), dtype=complex)
                    out[m] += lam * np.outer(sub, sub.conj())
        return out

    def detected(self, rho):
        """photon-number distribution of the TOTAL detected count at the next step"""
        dist = {}
        for n, block in rho.items():
            if block.shape[0] == 0 or np.abs(block).max() < 1e-18:
                continue
            N = n + self.Q
            w, V = np.linalg.eigh((block + block.conj().T) / 2)
            for lam, col in zip(w, V.T):
                if abs(lam) < 1e-16:
                    continue
                vec = np.zeros(self.fs.dim[N], dtype=complex)
                for a, occ in enumerate(self.lb[n]):
                    if col[a] != 0:
                        vec[self._embed(occ, N)] = col[a]
                psi = self.itf.apply(vec, N)
                p = np.abs(psi) ** 2
                for i, occ in enumerate(self.fs.basis[N]):
                    if p[i] == 0.0:
                        continue
                    k2 = occ[self.L:]
                    dist[k2] = dist.get(k2, 0.0) + lam * p[i]
        return dist

    def vacuum(self):
        return {0: np.ones((1, 1), dtype=complex)}


def mass(rho):
    return sum(float(np.real(np.trace(b))) for b in rho.values())


def mean_n(rho):
    return sum(n * float(np.real(np.trace(b))) for n, b in rho.items())


def tracedist(r1, r2):
    keys = set(r1) | set(r2)
    tot = 0.0
    for n in keys:
        d = max(len(r1.get(n, [])), len(r2.get(n, [])))
        if d == 0:
            continue
        A = np.zeros((d, d), dtype=complex)
        if n in r1:
            A[:r1[n].shape[0], :r1[n].shape[0]] += r1[n]
        if n in r2:
            A[:r2[n].shape[0], :r2[n].shape[0]] -= r2[n]
        tot += np.sum(np.abs(np.linalg.eigvalsh((A + A.conj().T) / 2)))
    return tot


def dtv(d1, d2):
    keys = set(d1) | set(d2)
    return 0.5 * sum(abs(d1.get(k, 0.0) - d2.get(k, 0.0)) for k in keys)


def stationary(ch, iters, tol=1e-15):
    r = ch.vacuum()
    for _ in range(iters):
        r2 = ch.apply(r)
        if tracedist(r, r2) < tol:
            return r2
        r = r2
    return r


# --------------------------------------------------------- U with prescribed U_ll
def haar(n, rng):
    z = (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))) / sqrt(2)
    Qm, R = np.linalg.qr(z)
    return Qm * (np.diag(R) / np.abs(np.diag(R)))


def cs_unitary(L, x, cos_vals, rng):
    """CS construction. r = min(L,x) singular values of U_ll are cos_vals, the other L-r are 1."""
    M = L + x
    r = min(L, x)
    assert len(cos_vals) == r
    C = np.diag(cos_vals)
    S = np.diag(np.sqrt(1 - np.array(cos_vals, dtype=float) ** 2))
    mid = np.eye(M, dtype=complex)
    k = L - r                      # loop modes untouched (only when L > x)
    mid[k:L, k:L] = C
    mid[k:L, L:L + r] = -S
    mid[L:L + r, k:L] = S
    mid[L:L + r, L:L + r] = C
    Q = np.eye(M, dtype=complex)
    Q[:L, :L] = haar(L, rng)
    Q[L:, L:] = haar(x, rng)
    R = np.eye(M, dtype=complex)
    R[:L, :L] = haar(L, rng)
    R[L:, L:] = haar(x, rng)
    return Q @ mid @ R


def Zmat(U, L, kmax=400):
    """Z = sum_i w_i^dag w_i, w_i = U_ll^i U_xl  (doc 1, section 1.5)"""
    Ull, Uxl = U[:L, :L], U[:L, L:]
    Z = np.zeros((U.shape[0] - L, U.shape[0] - L), dtype=complex)
    W = Uxl.copy()
    for _ in range(kmax):
        Z += W.conj().T @ W
        W = Ull @ W
    return Z
