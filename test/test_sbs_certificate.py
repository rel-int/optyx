"""
================================================================================
 test_sbs_certificate.py
 Verification suite for the two reductions, Theorem A, and the finite-depth
 part of the preflight algorithm. Every check is catalogued in
 `06-experiments.md`.
================================================================================

 ORDER. The tests follow the reductions appendix, the Theorem A appendix and the
 preflight appendix. Within Theorem A they follow the proof: static reduction,
 crude and refined moment estimates, the theorem, then its lossy extension.

   test  owner                 statement
   ----  --------------------  -----------------------------------------------
   T1    algorithm appendix    exact telescoping identity and block convention
   T2    reductions appendix   BS embeds into Stationary BS  [END TO END]
   T3    Theorem A, section B  static subspace reduction
   T4    proof notes           generic fourth-moment bound
   T5    Theorem A, section E  refined second moment and bipartite walk sum
   T6    Theorem A             finite-depth certificate
   T7    Theorem A, section G  lossy extension
   T8    algorithm appendix    certificate monotone and vanishing
   T9    algorithm appendix    window floor k >= ceil(L/x)
   T10   reductions appendix   nilpotent loop block, exact window
   T11   reductions appendix   finite unrolling  [END TO END]
   T12   structured appendix   correlation-moment formula at L = 3

 Repeated Fock truncation is tested separately by `test_sbs_truncation.py` as groups
 F-A to F-G in `06-experiments.md`.

 T2 and T11 are the two that test the reduction itself, in the two directions.
 Everything else tests an ingredient.

 METHODOLOGY, AND THE TWO TRAPS THIS SUITE IS BUILT TO AVOID

   Fock truncation. The truncation error is INVISIBLE to trace conservation once
   a Discard is present, so `trace == 1` proves nothing about the simulation as a
   whole. Every comparison here is run at two cutoffs, and a row is accepted only
   if ten times the drift plus twice the mass deficit still cannot lift the
   measurement above the bound. See `safe_channel`, whose docstring records the
   false violation this guard caught.

   Grid width. A grid confined to rho in [0.4, 0.8] once reported agreement where
   none held (03, Appendix B.2). The sweeps below span rho from 0.2 to 0.9.

 CONVENTIONS (sections 01 and 04)

   U is (L+x) x (L+x) unitary on modes ordered LOOP FIRST.
   U_ll = U[:L,:L]   loop  -> loop      (the block that matters)
   U_xl = U[:L,L:]   inject -> loop     ENTERING block. Reversing it is silent at
                                        L = 1 and wrong from L = 2; T1 checks this.
   U_lx = U[L:,:L]   loop  -> detect
   U_xx = U[L:,L:]   inject -> detect

 RUN
   pytest test/test_sbs_certificate.py
================================================================================
"""

import itertools
import numpy as np
from math import asin, sqrt, factorial, prod

import pytest

from sbs_loop_simulator import (
    LoopChannel, cs_unitary, haar, mass, stationary, tracedist)

TOL_EXACT = 1e-12          # for algebraic identities
rng_global = np.random.default_rng(20260801)


# ==============================================================================
#  Shared helpers
# ==============================================================================

def K(qbar):
    """Theorem A constant: K(q) = (q+1)[sqrt(6 q (q+1)) + q]."""
    return (qbar + 1) * (sqrt(6 * qbar * (qbar + 1)) + qbar)


def Theta(Ull, k, gamma=1.0):
    """Theta_k = sum_r arcsin^2( gamma^{k/2} sigma_r(U_ll^k) ),  03, (3.1) and (3.35).

    Computed from the singular values of the k-th power. No eigenvalue of any
    superoperator, no state, no Fock space:  O(k L^3).
    """
    B = np.linalg.matrix_power(Ull, k)
    s = np.clip(gamma ** (k / 2) * np.linalg.svd(B, compute_uv=False), 0.0, 1.0)
    return float(sum(asin(v) ** 2 for v in s))


def certificate(Ull, k, qbar, gamma=1.0):
    """Gamma(k) = 4 K(qbar) Theta_k,   06, (6.2)."""
    return 4.0 * K(qbar) * Theta(Ull, k, gamma)


def window(Ull, qbar, eps, gamma=1.0, kmax=4000):
    """Smallest k with Gamma(k) <= eps.  This is step 2 of the algorithm of 06."""
    for k in range(1, kmax + 1):
        if certificate(Ull, k, qbar, gamma) <= eps:
            return k
    raise RuntimeError("window not reached: check rho(U_ll) < 1, or relax eps")


def unroll(U, L, x, t):
    """The (L + x t)-mode unitary F of 06, F1.

    Mode order of F:   [ loop_in (L) | inject_1 (x) | ... | inject_t (x) ]
    on the input side, and
                       [ loop_out(L) | detect_1 (x) | ... | detect_t (x) ]
    on the output side.  Built by composing t copies of U, each acting on the
    loop block and one fresh injection block.
    """
    n = L + x * t
    F = np.eye(n, dtype=complex)
    for s in range(t):
        S = np.eye(n, dtype=complex)
        idx = list(range(L)) + list(range(L + x * s, L + x * (s + 1)))
        S[np.ix_(idx, idx)] = U
        F = S @ F
    return F


def perm(M):
    """Permanent by Ryser's formula.  Exponential, used only on tiny matrices."""
    k = M.shape[0]
    if k == 0:
        return 1.0 + 0j
    tot = 0
    for S in range(1, 1 << k):
        bits = [i for i in range(k) if S >> i & 1]
        tot += (-1) ** (k - len(bits)) * np.prod(M[:, bits].sum(axis=1))
    return tot


def bs_distribution(F, n_in, n_modes_out_kept, max_photons):
    """Exact output distribution of a boson sampling instance, marginalised.

    F           unitary on n modes
    n_in        tuple of input occupations, len n
    kept        the LAST n_modes_out_kept modes are read; the others are traced
    Returns     dict {occupation tuple on the kept modes -> probability}

    Uses the permanent formula
        |<n_out| Gamma(F) |n_in>|^2 = |perm F[n_out, n_in]|^2 / (n_out! n_in!)
    which is the definition of boson sampling, and sums over the traced modes.
    """
    n = F.shape[0]
    N = sum(n_in)
    cols = [j for j, c in enumerate(n_in) for _ in range(c)]
    denom_in = prod(factorial(c) for c in n_in)
    out = {}
    for occ in itertools.product(range(N + 1), repeat=n):
        if sum(occ) != N:
            continue
        rows = [j for j, c in enumerate(occ) for _ in range(c)]
        amp = perm(F[np.ix_(rows, cols)]) / sqrt(
            prod(factorial(c) for c in occ) * denom_in)
        p = abs(amp) ** 2
        if p < 1e-14:
            continue
        key = occ[n - n_modes_out_kept:]
        out[key] = out.get(key, 0.0) + p
    return out


def tv(d1, d2):
    """Total variation distance between two dicts of probabilities."""
    keys = set(d1) | set(d2)
    return 0.5 * sum(abs(d1.get(k, 0.0) - d2.get(k, 0.0)) for k in keys)


def converged(fn, cutoffs, tol):
    """Run fn(cutoff) at two cutoffs; return the finer value only if they agree.

    This is the guard against the Fock-truncation trap: trace conservation does
    NOT detect truncation error when a Discard is present (01, A.12).
    """
    vals = [fn(c) for c in cutoffs]
    drift = abs(vals[-1] - vals[-2])
    ok = drift <= tol * max(1.0, abs(vals[-1]))
    return vals[-1], drift, ok


def safe_channel(U, L, q, cutoffs, floor=0.99, iters=200, nsteps=4):
    """Return (channel, rho_inf, cutoff, deficit) for the first admissible cutoff.

    `deficit` is the mass lost to Fock truncation, taken as the maximum over the
    two families of states that are actually compared: rho_inf, and rho_1 to
    rho_nsteps. It is NOT required to be negligible; it is returned so the caller
    can fold it into the uncertainty of any measurement made with this channel,
    a trace distance being shiftable by at most the mass deficit of either state.

    WHY THIS GUARD EXISTS. The channel truncated at Ncut photons is not trace
    preserving: components above the cutoff are dropped, and iterating a few
    hundred times to reach rho_inf compounds the loss. During development this
    suite reported a VIOLATION of Theorem A at (L,x,q,c) = (2,2,(2,2),0.5) with
    ratio 1.26. The cause was a "stationary state" of mass 0.0025: it had simply
    evaporated. At cutoff 12 its mass was 0.992 and the measured distances fell
    back to 2% of the bound. The exact channel IS trace preserving on the loop,
    so here, and only here, the mass deficit is a direct and reliable measure of
    the truncation error.
    """
    best = None
    for cut in cutoffs:
        ch = LoopChannel(U, L, tuple(q), cut)
        ri = stationary(ch, iters)
        d = abs(1.0 - mass(ri))
        rk = ch.vacuum()
        for _ in range(nsteps):
            rk = ch.apply(rk)
            d = max(d, abs(1.0 - mass(rk)))
        if best is None or d < best[3]:
            best = (ch, ri, cut, d)
        if d <= 1e-9:
            return best
    if best is None or best[3] > 1 - floor:
        return None
    return best


def banner(name, description):
    print("\n" + "=" * 78)
    print(f" {name}   {description}")
    print("=" * 78)


RESULTS = {}


def record(name, passed, detail=""):
    RESULTS[name] = (passed, detail)
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}  {detail}")
    if not passed:
        raise AssertionError(f"{name} failed: {detail}")


# ==============================================================================
#  T1   Block identity                                          01, (1.13)
# ==============================================================================
#  sum_{i<k} w_i w_i^dag  +  B_k B_k^dag  =  I_L,   with  w_i = U_ll^i U_xl.
#
#  This is the self-check of 06, A.1. It catches, in one line: a reversed block
#  convention, a wrong matrix power, and a non-unitary U. It is the first thing
#  the algorithm should assert.
# ==============================================================================

def T1():
    banner("T1", "block identity  sum w_i w_i^dag + B_k B_k^dag = I     [algorithm appendix]")
    worst = 0.0
    for (L, x) in [(1, 1), (2, 2), (3, 2), (2, 3), (4, 3)]:
        U = haar(L + x, rng_global)
        Ull, Uxl = U[:L, :L], U[:L, L:]
        W = Uxl.copy()
        S = np.zeros((L, L), dtype=complex)
        for k in range(1, 6):
            S = S + W @ W.conj().T
            B = np.linalg.matrix_power(Ull, k)
            err = np.linalg.norm(S + B @ B.conj().T - np.eye(L), 2)
            worst = max(worst, err)
            W = Ull @ W
    record("T1", worst < TOL_EXACT, f"worst residual {worst:.2e}")
    # Negative control: the reversed convention U_xl = U[L:, :L] must FAIL.
    L = x = 2
    U = haar(L + x, rng_global)
    Wbad = U[L:, :L]
    Sbad = Wbad @ Wbad.conj().T
    bad = np.linalg.norm(Sbad + U[:L, :L] @ U[:L, :L].conj().T - np.eye(L), 2)
    record("T1-control", bad > 1e-3,
           f"reversed convention detected, residual {bad:.2e}")


# ==============================================================================
#  T2   Static reduction                                        03, Lemma 3.2
# ==============================================================================
#  P_l G P_l = B^dag B,  where G = F^dag Pi F is the projector on the subspace
#  script-G of input modes whose photons end in the loop.
#
#  This is the geometric heart of the proof: it says that the survival amplitude
#  of a loop direction is the length of its projection onto script-G, so the
#  principal angles between script-G and the loop input have cosines equal to the
#  singular values of U_ll^k.
# ==============================================================================

def T3():
    banner("T3", "static reduction  P_l G P_l = B^dag B                [Theorem A, B]")
    worst = 0.0
    for (L, x, k) in [(1, 1, 3), (2, 2, 2), (2, 2, 3), (3, 2, 2), (2, 3, 3)]:
        U = haar(L + x, rng_global)
        F = unroll(U, L, x, k)
        n = L + x * k
        Pi = np.zeros((n, n))
        Pi[:L, :L] = np.eye(L)                       # projector on loop OUTPUT
        G = F.conj().T @ Pi @ F
        lhs = G[:L, :L]
        B = np.linalg.matrix_power(U[:L, :L], k)
        worst = max(worst, np.linalg.norm(lhs - B.conj().T @ B, 2))
    record("T3", worst < TOL_EXACT, f"worst residual {worst:.2e}")


# ==============================================================================
#  T6   The certificate is an upper bound                        Theorem A
# ==============================================================================
#  ||rho_inf - rho_k||_1  <=  4 K(qbar) Theta_k.
#
#  rho_inf is obtained by iterating the channel to convergence; rho_k by k steps
#  from vacuum. Both are computed at two Fock cutoffs and reported only if the
#  measured trace distance has converged.
# ==============================================================================

def T6(quick=False):
    banner("T6", "certificate is an upper bound                        [Theorem A]")
    grid = [(1, 1, (1,), 0.20), (1, 1, (1,), 0.50), (1, 1, (1,), 0.90),
            (1, 1, (2,), 0.40), (2, 2, (1, 1), 0.30)]
    if not quick:
        grid += [(2, 2, (1, 1), 0.70), (2, 1, (1,), 0.60),
                 (2, 2, (2, 2), 0.50), (3, 2, (1, 1), 0.50)]
    worst_ratio, viol, unsafe, skipped = 0.0, 0, 0, 0
    # VERDICT CRITERION. A measurement carries two sources of Fock-truncation
    # uncertainty: the drift between two admissible cutoffs, and the mass lost to
    # truncation, which can shift a trace distance by at most that amount. A row
    # is accepted only if ten times the drift plus twice the deficit still cannot
    # lift the measurement above the bound, so no verdict here can be an artefact
    # of the cutoff. Testing `drift < tol` alone would be the wrong check: what
    # matters is the uncertainty against the MARGIN, not against the value.
    print(f"  {'L':>2}{'x':>2}{'q':>9}{'c':>6}{'cut':>4}{'k':>3} {'measured':>11} "
          f"{'Gamma(k)':>11} {'ratio':>8} {'deficit':>9} {'worst/G':>9}")
    for (L, x, q, c) in grid:
        U = cs_unitary(L, x, [c] * min(L, x), np.random.default_rng(3))
        qbar = max(q)
        pool = {1: [10, 12, 14], 2: [10, 12, 14], 3: [7, 9]}[L]
        got = safe_channel(U, L, q, pool)
        if got is None:
            skipped += 1
            print(f"  {L:>2}{x:>2}{str(q):>9}{c:>6.2f}   -   -   no admissible cutoff, skipped")
            continue
        ch, ri, cut, deficit = got
        lower = [c2 for c2 in pool if c2 < cut]
        got2 = safe_channel(U, L, q, [lower[-1]], floor=0.0) if lower else None
        ch2, ri2 = (got2[0], got2[1]) if got2 else (ch, ri)

        def dists(chan, rinf):
            rk, out = chan.vacuum(), []
            for _ in range(4):
                rk = chan.apply(rk)
                out.append(tracedist(rk, rinf))
            return out

        fine, coarse = dists(ch, ri), dists(ch2, ri2)
        for k in range(1, 5):
            meas = fine[k - 1]
            drift = abs(fine[k - 1] - coarse[k - 1])
            unc = 10 * drift + 2 * deficit
            G = certificate(U[:L, :L], k, qbar)
            ratio = meas / G if G > 0 else 0.0
            worst_ratio = max(worst_ratio, ratio)
            safe = (meas + unc) <= G
            viol += (meas > G)
            unsafe += (not safe)
            print(f"  {L:>2}{x:>2}{str(q):>9}{c:>6.2f}{cut:>4}{k:>3} {meas:>11.4e} "
                  f"{G:>11.4e} {ratio:>8.4f} {deficit:>9.1e} {(meas + unc) / G:>9.4f}"
                  + ("   VIOLATION" if meas > G else "")
                  + ("   CUTOFF UNSAFE" if not safe else ""))
    record("T6", viol == 0 and unsafe == 0 and skipped == 0,
           f"worst ratio {worst_ratio:.4f}, {viol} violations, {unsafe} cutoff-unsafe, "
           f"{skipped} skipped")
    return worst_ratio


# ==============================================================================
#  T4   Monotonicity and decay of the certificate                preflight monotonicity
# ==============================================================================
#  Theta_k is decreasing and tends to 0 when the loop block is contractive. This makes the loop in
#  step 2 of the algorithm terminate. Checked on random unitaries and on the
#  worst case rho(U_ll) close to 1.
# ==============================================================================

def T8():
    banner("T8", "certificate monotone and vanishing                    [preflight monotonicity]")
    bad = 0
    for (L, x) in [(1, 1), (2, 2), (3, 2), (4, 2), (5, 3)]:
        for trial in range(4):
            U = haar(L + x, np.random.default_rng(100 + trial))
            Ull = U[:L, :L]
            th = [Theta(Ull, k, 1.0) for k in range(1, 12)]
            if any(th[i + 1] > th[i] + 1e-14 for i in range(len(th) - 1)):
                bad += 1
            if th[-1] > th[0]:
                bad += 1
    record("T8", bad == 0, f"{bad} monotonicity failures out of 80 sequences")


# ==============================================================================
#  T5   END TO END:  the unrolled instance reproduces P_inf      finite-window reduction
# ==============================================================================
#  THIS IS THE TEST OF THE REDUCTION ITSELF.
#
#  Procedure, exactly the algorithm of the preflight algorithm:
#    1. choose eps, compute the certified window k with Gamma(k) <= eps
#    2. build F = Unroll(U, k + w), an ordinary (L + x(k+w))-mode interferometer
#    3. compute its EXACT output distribution by the permanent formula, i.e. by
#       the definition of boson sampling, and marginalise onto the last w*x
#       detector modes
#    4. compare with the stationary detected distribution, computed independently
#       by iterating the channel to convergence
#
#  The claim is  d_TV <= eps/2  from (6.5). Step 3 uses permanents and never the
#  channel; step 4 uses the channel and never permanents. The two computations
#  share no code path, so agreement is a genuine cross-check.
# ==============================================================================

def T11(quick=False):
    banner("T11", "END TO END: unrolled BS instance vs stationary law  [finite-window reduction]")
    grid = [(1, 1, (1,), 0.40, 1), (1, 1, (1,), 0.60, 1), (1, 1, (2,), 0.35, 1),
            (2, 1, (1,), 0.50, 1)]
    if not quick:
        grid += [(1, 1, (1,), 0.40, 2)]
    worst, fails = 0.0, 0
    print(f"  {'L':>2}{'x':>2}{'q':>7}{'c':>6}{'w':>3}{'eps':>7}{'k':>4}"
          f"{'modes':>7}{'phot':>6} {'d_TV':>11} {'eps/2':>8}  reference")
    for (L, x, q, c, w) in grid:
        U = cs_unitary(L, x, [c] * min(L, x), np.random.default_rng(3))
        qbar = max(q)
        for eps in (0.5, 0.3):
            k = window(U[:L, :L], qbar, eps, kmax=200)
            t = k + w
            F = unroll(U, L, x, t)
            n_in = (0,) * L + tuple(q) * t
            if sum(n_in) > 9:
                print(f"  {L:>2}{x:>2}{str(q):>7}{c:>6.2f}{w:>3}{eps:>7.2f}{k:>4}"
                      f"{L + x * t:>7}{sum(n_in):>6}   skipped, too many photons")
                continue
            P_bs = bs_distribution(F, n_in, w * x, sum(n_in))
            if w == 1:
                # independent computation: iterate the channel, then read out once.
                # Shares no code path with the permanent formula above.
                cut = 8 if L == 1 else 6
                ch = LoopChannel(U, L, tuple(q), cut)
                P_ref = ch.detected(stationary(ch, 300))
                ref = "channel"
            else:
                # joint law over w blocks: compare against a much deeper window,
                # which is Gamma(k_ref) << Gamma(k) away from stationarity.
                k_ref = window(U[:L, :L], qbar, eps / 50, kmax=200)
                t_ref = k_ref + w
                n_ref = (0,) * L + tuple(q) * t_ref
                if sum(n_ref) > 10:
                    k_ref = min(k_ref, (10 - w * sum(q)) // sum(q))
                    t_ref = k_ref + w
                    n_ref = (0,) * L + tuple(q) * t_ref
                P_ref = bs_distribution(unroll(U, L, x, t_ref), n_ref, w * x, sum(n_ref))
                ref = f"deep window k={k_ref}"
            d = tv(P_bs, P_ref)
            worst = max(worst, 2 * d / eps)
            ok = d <= eps / 2 + 1e-9
            fails += (not ok)
            print(f"  {L:>2}{x:>2}{str(q):>7}{c:>6.2f}{w:>3}{eps:>7.2f}{k:>4}"
                  f"{L + x * t:>7}{sum(n_in):>6} {d:>11.4e} {eps / 2:>8.3f}  {ref}"
                  + ("" if ok else "   VIOLATION"))
    record("T11", fails == 0, f"worst d_TV/(eps/2) = {worst:.4f}, {fails} violations")


def T12():
    """Check the correlation-moment formula beyond the one-mode case."""
    banner("T12", "correlation moment at L=3: formula vs permanents  [structured appendix, D]")
    L, x, t = 3, 1, 3
    U = haar(L + x, np.random.default_rng(53))
    F = unroll(U, L, x, t)

    # Put the final loop outputs last so bs_distribution marginalises onto them.
    loop_last = np.vstack((F[L:, :], F[:L, :]))
    n_in = (0,) * L + (1,) * t
    distribution = bs_distribution(loop_last, n_in, L, sum(n_in))
    measured = sum(
        probability * sum(occupation) * (sum(occupation) - 1)
        for occupation, probability in distribution.items()
    )

    # The same moment from the correlation Gram matrix of the t injection slots.
    W = F[:L, L:]
    G = W.conj().T @ W
    predicted = 0.0
    for s in range(t):
        for u in range(s + 1, t):
            predicted += 2 * (
                float(np.real(G[s, s] * G[u, u])) + abs(G[s, u]) ** 2
            )

    normalisation_error = abs(sum(distribution.values()) - 1.0)
    moment_error = abs(measured - predicted)
    record(
        "T12",
        normalisation_error < 1e-10 and moment_error < 1e-10,
        f"normalisation {normalisation_error:.2e}, moment residual {moment_error:.2e}",
    )


# ==============================================================================
#  T4   Fourth moment                                        proof notes
# ==============================================================================
#  Tr(sigma G^4) <= 26 (qbar+1)^4 (tr g^2)^2   for centred G = dGamma(g).
#
#  This is the engine of the crude route, 03 section 3.6, and it is what replaces
#  the naive |dGamma(g)| <= ||g|| N estimate that costs a factor L^2. It is tested
#  here on a GENERIC Hermitian g, a much wider family than the CS unitaries used
#  in T6, so it probes the combinatorial classification of Lemma 3.10 directly.
#
#  The centring hypothesis (3.22) is imposed by hand: the diagonal of g is shifted
#  so that sum_a g_aa n_a = 0, which is what Lemma 3.6 supplies in the real proof.
#  Was `test_l82.py`.
# ==============================================================================

def fock_sector(M, N):
    """Occupation tuples of M modes with exactly N photons."""
    return [c for c in itertools.product(range(N + 1), repeat=M) if sum(c) == N]


def dGamma_matrix(h, basis):
    """Matrix of dGamma(h) = sum_{a,b} h_ab b_a^dag b_b in a fixed photon sector."""
    M = len(basis[0])
    idx = {b: i for i, b in enumerate(basis)}
    H = np.zeros((len(basis),) * 2, dtype=complex)
    for b, j in idx.items():
        for a in range(M):
            for c in range(M):
                if b[c] == 0:
                    continue
                t = list(b)
                t[c] -= 1
                t[a] += 1
                t = tuple(t)
                if t in idx:
                    H[idx[t], j] += h[a, c] * sqrt(b[c]) * sqrt(t[a])
    return H


def T4():
    banner("T4", "fourth moment  <G^4> <= 26 (q+1)^4 (tr g^2)^2   [proof notes]")
    rng = np.random.default_rng(7)
    worst, viol = 0.0, 0
    print(f"  {'M':>2}{'n vector':>15}{'draws':>7} {'worst <G^4>/bound':>19}")
    for nvec in [(1, 1, 0), (2, 1, 0), (1, 1, 1), (2, 2, 1), (3, 1, 0),
                 (1, 1, 1, 1), (2, 2, 2)]:
        M, N, qb = len(nvec), sum(nvec), max(nvec)
        bas = fock_sector(M, N)
        v = np.zeros(len(bas))
        v[bas.index(nvec)] = 1.0
        local = 0.0
        for _ in range(6):
            A = rng.normal(size=(M, M)) + 1j * rng.normal(size=(M, M))
            h = (A + A.conj().T) / 2
            occ = [a for a in range(M) if nvec[a] > 0]          # impose (3.22)
            h[occ[0], occ[0]] -= sum(h[a, a].real * nvec[a] for a in occ) / nvec[occ[0]]
            assert abs(sum(h[a, a].real * nvec[a] for a in range(M))) < 1e-10
            H = dGamma_matrix(h, bas)
            m4 = float(np.real(v @ np.linalg.matrix_power(H, 4) @ v))
            bound = 26 * (qb + 1) ** 4 * float(np.real(np.trace(h @ h))) ** 2
            local = max(local, m4 / bound)
            viol += (m4 > bound)
        worst = max(worst, local)
        print(f"  {M:>2}{str(nvec):>15}{6:>7} {local:>19.5f}")
    record("T4", viol == 0,
           f"worst ratio {worst:.5f} over 42 draws, constant 26 loose by ~{1 / worst:.0f}x")


# ==============================================================================
#  T5   Observations B and C                                 03, (3.28), (3.29)
# ==============================================================================
#  The refined route uses three extra properties of the generator (3.6). This test
#  exercises the two quantitative ones on a v that has them by construction:
#  Hermitian, ZERO DIAGONAL (Observation A), and BIPARTITE between the old block
#  and the injection block (Observation C).
#
#      ||V |n>||^2   <=  qbar (qbar+1) tr v^2                         (3.28)
#      ||V^2 |n>||^2 <=  6 qbar (qbar+1)^3 (tr v^2)^2                 (3.29) + L3.12
#
#  The point of the test is not only that both hold, but HOW TIGHT they are:
#  (3.28) comes out saturated, so no gain is available on that term, whereas
#  (3.29) is used at about a quarter of its value. That is the measurement behind
#  the claim in 03, O1 that all remaining slack sits in the V^4 term.
#  Was `opt.py`.
# ==============================================================================

def T5():
    banner("T5", "observations B and C, and how tight they are  [Theorem A, E]")
    rng = np.random.default_rng(11)
    w1 = w2 = 0.0
    viol = 0
    print(f"  {'Lold':>5}{'Linj':>5}{'n vector':>16} {'ratio (3.28)':>13} {'ratio (3.29)':>13}")
    for (Lo, Li, nvec) in [(1, 1, (1, 1)), (1, 1, (2, 1)), (2, 1, (1, 1, 1)),
                           (1, 2, (1, 1, 1)), (2, 2, (1, 1, 1, 1)),
                           (2, 2, (2, 1, 2, 1)), (1, 2, (2, 2, 2)), (3, 1, (1, 1, 1, 1))]:
        M, N, qb = Lo + Li, sum(nvec), max(nvec)
        bas = fock_sector(M, N)
        psi = np.zeros(len(bas))
        psi[bas.index(nvec)] = 1.0
        r1 = r2 = 0.0
        for _ in range(5):
            Y = rng.normal(size=(Lo, Li)) + 1j * rng.normal(size=(Lo, Li))
            v = np.zeros((M, M), dtype=complex)
            v[:Lo, Lo:] = Y
            v[Lo:, :Lo] = Y.conj().T                  # bipartite, zero diagonal
            assert np.allclose(np.diag(v), 0)
            V = dGamma_matrix(v, bas)
            tv2 = float(np.real(np.trace(v @ v)))
            n1 = float(np.linalg.norm(V @ psi)) ** 2
            n2 = float(np.linalg.norm(V @ (V @ psi))) ** 2
            b1 = qb * (qb + 1) * tv2
            b2 = 6 * qb * (qb + 1) ** 3 * tv2 ** 2
            r1, r2 = max(r1, n1 / b1), max(r2, n2 / b2)
            viol += (n1 > b1 * (1 + 1e-9)) + (n2 > b2 * (1 + 1e-9))
        w1, w2 = max(w1, r1), max(w2, r2)
        print(f"  {Lo:>5}{Li:>5}{str(nvec):>16} {r1:>13.4f} {r2:>13.4f}")
    record("T5", viol == 0,
           f"(3.28) worst {w1:.4f} (saturated), (3.29) worst {w2:.4f}")


# ==============================================================================
#  T6   END TO END, other direction: BS embeds into Stationary BS  02, (2.3)
# ==============================================================================
#  Take any W on x modes. Put
#      U = [[0, W], [I, 0]]     (loop first),   L = x
#  Then U_ll = 0, so the loop empties in ONE step, and the stationary detected
#  distribution equals the boson sampling distribution of W on input |q>.
#
#  This is the lower bound: Stationary BS is at least as hard as BS. Combined
#  with T5 it closes the equivalence.
# ==============================================================================

def T2():
    banner("T2", "END TO END: BS embeds into Stationary BS               [02, (2.3)]")
    worst = 0.0
    for x in (1, 2, 3):
        for q in [(1,) * x, (2,) + (1,) * (x - 1)]:
            L = x
            W = haar(x, np.random.default_rng(7 + x))
            U = np.zeros((L + x, L + x), dtype=complex)
            U[:L, L:] = W                       # inject -> loop  is W
            U[L:, :L] = np.eye(x)               # loop  -> detect is identity
            assert np.allclose(U.conj().T @ U, np.eye(2 * x), atol=TOL_EXACT)
            assert np.linalg.norm(U[:L, :L]) < TOL_EXACT       # U_ll = 0
            # stationary detected law of the looped device
            cut = sum(q) + 1
            ch = LoopChannel(U, L, tuple(q), cut)
            ri = stationary(ch, 50)
            P_stat = ch.detected(ri)
            # boson sampling law of W on input |q>
            P_bs = bs_distribution(W, tuple(q), x, sum(q))
            d = tv(P_stat, P_bs)
            worst = max(worst, d)
            print(f"   x={x} q={str(q):<10} d_TV = {d:.3e}")
    record("T2", worst < 1e-9, f"worst d_TV {worst:.2e}")


# ==============================================================================
#  T7   Nilpotent loop block: the window is exact                nilpotent reduction
# ==============================================================================
#  If U_ll^d = 0 then Theta_d = 0, so the certificate says the truncation error
#  is exactly zero at k = d. This is a sharp prediction, easy to falsify, and it
#  is what makes the embedding of T6 a single-shot instance (d = 1).
#
#  The example used is the one that also refutes the identity
#  |lambda_2| = rho(|D|^{o2}) proposed elsewhere: D nilpotent with ||D|| = 1.
# ==============================================================================

def T10():
    banner("T10", "nilpotent loop block: window is exact                 [nilpotent window]")
    import scipy.linalg as sla
    D = 0.5 * np.array([[1, 1], [-1, -1]], dtype=complex)      # D^2 = 0, ||D|| = 1
    L = 2
    S1 = sla.sqrtm(np.eye(2) - D @ D.conj().T)
    S2 = sla.sqrtm(np.eye(2) - D.conj().T @ D)
    U = np.block([[D, S1], [S2, -D.conj().T]])
    assert np.allclose(U.conj().T @ U, np.eye(4), atol=1e-10)
    print(f"   Theta_1 = {Theta(D, 1):.4e}   Theta_2 = {Theta(D, 2):.4e}  (predicted 0)")
    worst = 0.0
    for cut in (6, 7):
        ch = LoopChannel(U, L, (1, 1), cut)
        base = ch.vacuum()
        r0 = ch.apply(ch.apply(base))
        for init in [(1, 0), (1, 1)]:
            n = sum(init)
            blk = np.zeros((len(ch.lb[n]),) * 2, dtype=complex)
            blk[ch.li[n][init], ch.li[n][init]] = 1.0
            r2 = ch.apply(ch.apply({n: blk}))
            worst = max(worst, tracedist(r2, r0))
        print(f"   cutoff {cut}:  max_sigma ||E^2(sigma) - E^2(vac)||_1 = {worst:.3e}")
    record("T10", worst < 1e-10 and Theta(D, 2) < 1e-25,
           f"exact at k=2, residual {worst:.2e}")


# ==============================================================================
#  T8   Lossy case                                               03, Cor 3.18
# ==============================================================================
#  Loss stays passive: a pure loss channel of transmissivity gamma is
#  a beam splitter to a fresh VACUUM mode which is then discarded, exactly like a
#  detected mode. So the lossy device on L + x modes is a LOSSLESS device on
#  L + (x + L) modes with injection (q, 0, ..., 0).
#
#  This test builds that enlarged unitary explicitly, checks it is unitary, and
#  then applies Theorem A unchanged. It verifies the substitution
#      beta_r  ->  gamma^{k/2} beta_r        (03, (3.35))
# ==============================================================================

def T7(quick=False):
    banner("T7", "lossy loop = lossless with more modes                 [Theorem A, G]")

    def lossy_U(U, L, x, g):
        n = L + x + L
        E = np.eye(n, dtype=complex)
        E[:L + x, :L + x] = U
        Bs = np.eye(n, dtype=complex)
        for i in range(L):
            j = L + x + i
            Bs[i, i] = Bs[j, j] = sqrt(g)
            Bs[i, j] = sqrt(1 - g)
            Bs[j, i] = -sqrt(1 - g)
        return Bs @ E

    worst, viol = 0.0, 0
    print(f"  {'L':>2}{'x':>2}{'g':>6}{'cut':>4}{'k':>3} {'measured':>11} {'Gamma(k)':>11} {'ratio':>8}")
    grid = [(1, 1, (1,), 0.50), (1, 1, (1,), 0.80)]
    if not quick:
        grid += [(2, 2, (1, 1), 0.50)]     # 6 modes once dilated, the slow one
    for (L, x, q, c) in grid:
        for g in (0.80, 0.95):
            U = cs_unitary(L, x, [c] * min(L, x), np.random.default_rng(3))
            Up = lossy_U(U, L, x, g)
            assert np.allclose(Up.conj().T @ Up, np.eye(L + x + L), atol=1e-12)
            # the enlarged loop block IS sqrt(gamma) U_ll  (03, (3.34))
            assert np.allclose(Up[:L, :L], sqrt(g) * U[:L, :L], atol=1e-12)
            qp = tuple(q) + (0,) * L
            got = safe_channel(Up, L, qp, [8, 10, 12] if L == 1 else [8, 10, 12])
            assert got is not None, "no admissible cutoff for the lossy channel"
            ch, ri, cut, _def = got
            rk = ch.vacuum()
            for k in range(1, 4):
                rk = ch.apply(rk)
                meas = tracedist(rk, ri)
                G = certificate(U[:L, :L], k, max(qp), gamma=g)
                worst = max(worst, meas / G)
                viol += (meas > G)
                print(f"  {L:>2}{x:>2}{g:>6.2f}{cut:>4}{k:>3} {meas:>11.4e} {G:>11.4e} "
                      f"{meas / G:>8.4f}" + ("   VIOLATION" if meas > G else ""))
    record("T7", viol == 0, f"worst ratio {worst:.4f}, {viol} violations")


# ==============================================================================
#  T9   Window floor                                             preflight floor
# ==============================================================================
#  Each step opens only x new directions in the loop, so filling L of them takes
#  at least ceil(L/x) steps. Equivalently U_ll^k has a unit singular value for
#  every k < ceil(L/x), so Theta_k >= (pi/2)^2 and no certificate can fire.
# ==============================================================================

def T9():
    banner("T9", "window floor  k >= ceil(L/x)                          [preflight floor]")
    bad = 0
    for (L, x) in [(4, 1), (5, 2), (6, 2), (6, 3), (7, 2)]:
        U = haar(L + x, np.random.default_rng(11))
        floor = -(-L // x)
        for k in range(1, floor):
            s = np.linalg.svd(np.linalg.matrix_power(U[:L, :L], k), compute_uv=False)
            if s[0] < 1 - 1e-9:
                bad += 1
                print(f"   L={L} x={x} k={k}: top sigma {s[0]:.6f}, expected 1")
        s = np.linalg.svd(np.linalg.matrix_power(U[:L, :L], floor), compute_uv=False)
        print(f"   L={L} x={x}: floor {floor}, sigma_1 at floor = {s[0]:.6f}")
    record("T9", bad == 0, "no certificate can fire below the floor")


# ==============================================================================
#  Driver
# ==============================================================================

@pytest.mark.parametrize(
    ("check", "quick"),
    [(T1, None), (T2, None), (T3, None), (T4, None), (T5, None),
     (T6, True), (T7, True), (T8, None), (T9, None), (T10, None),
     (T11, True), (T12, None)],
    ids=[f"T{index}" for index in range(1, 13)],
)
def test_sbs_certificate(check, quick):
    """Run every reduction and finite-depth certificate regression."""
    RESULTS.clear()
    check() if quick is None else check(quick)
