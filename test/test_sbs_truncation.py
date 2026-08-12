"""Numerically verify the certified Fock truncation bounds.

 Covers the repeated-truncation bounds in `appendix_algorithm.md`, and nothing
 else. It is the one bound family not covered by
 `test_sbs_certificate.py`. Its seven checks F-A to F-G are catalogued in
 `06-experiments.md`.


QUESTION. The preflight never builds a density matrix: it reads Theta_k
off U_ll^k, an L x L object. But anyone who wants to compute rho_k or rho_inf
classically must truncate the loop Fock space at N_max photons, and the
dimension C(N_max+L, L) is what explodes. Can the truncation be certified from a
polynomial-time analysis of U alone?

CLAIM. Write rho~_m for the trajectory computed with the loop
truncated at N_max, i.e. rho~_{m+1} = Pi E(rho~_m) with Pi dropping components
above N_max.

 (i)   rho~_m <= rho_m in the psd order.
       Because passive Fock inputs make every rho_m block diagonal in loop photon number, Pi
       commutes with it and truncation only DROPS blocks; E is positive and
       monotone, so the inequality propagates.
 (ii)  ||rho_m - rho~_m||_1 = 1 - tr(rho~_m).
       By (i) the difference is positive, so its trace norm is its trace. The
       truncation error therefore CERTIFIES ITSELF at runtime, for free.
 (iii) ||rho~_k - rho_inf||_1 <= Gamma(k) + (1 - tr rho~_k).
 (iv)  A priori, 1 - tr(rho~_k) <= k * lambda / (N_max + 1), with
           lambda = sum_a q_a Z_aa,   Z = U_xl^dag U_xl + U_ll^dag Z U_ll
       the stationary mean loop occupation, computable in O(L^3). This is
       Markov's inequality applied at every projection.

WHAT IS TESTED
 A  lambda from the Stein equation against the simulated <n>_inf
 B  every transient mean is at most the stationary mean lambda
 C  the measured tail against the rigorous Markov bound lambda/(N+1)
 D  (i) and (ii): the deficit is exactly the trace-norm error
 E  the cumulative deficit against k lambda/(N_max+1)
 F  the normalised total error against Gamma(k)+2 Delta_N(k)
 G  the necessary lower bound against the exact stationary tail
"""
import numpy as np
from math import asin, sqrt
import pytest
import scipy.linalg as sla

from sbs_loop_simulator import (
    LoopChannel, cs_unitary, mass, mean_n, stationary, tracedist)


def K(q): return (q + 1) * (sqrt(6 * q * (q + 1)) + q)


def Gamma(Ull, k, qbar):
    B = np.linalg.matrix_power(Ull, k)
    s = np.clip(np.linalg.svd(B, compute_uv=False), 0, 1)
    return 4 * K(qbar) * float(sum(asin(v) ** 2 for v in s))


def lam_stein(U, L, q):
    """lambda = <n>_inf = sum_a q_a Z_aa,  computed in O(L^3) with no Fock space.

    Z = sum_{i>=0} w_i^dag w_i  with w_i = U_ll^i U_xl, so  Z = U_xl^dag Y U_xl
    where Y is the L x L solution of the Stein equation  Y = I + U_ll^dag Y U_ll.
    Note Z is x by x while Y is L by L: solving for Z directly is a dimension
    error whenever x != L.
    """
    Ull, Uxl = U[:L, :L], U[:L, L:]
    Y = sla.solve_discrete_lyapunov(Ull.conj().T, np.eye(L))
    Z = Uxl.conj().T @ Y @ Uxl
    return float(np.real(sum(q[a] * Z[a, a] for a in range(len(q)))))


def occ(rho):
    """Loop photon-number distribution of a block-diagonal state."""
    return {n: float(np.real(np.trace(b))) for n, b in rho.items()}


def require(condition, group, case):
    """Raise at the first invalid bound, with the physical case attached."""
    if not condition:
        raise AssertionError(f"{group} failed for {case}")


QUICK = True


CASES = [(1, 1, (1,), 0.50), (1, 1, (1,), 0.80), (1, 1, (2,), 0.60),
         (2, 2, (1, 1), 0.60), (2, 1, (1,), 0.70)]


@pytest.mark.parametrize(("L", "x", "q", "c"), CASES)
def test_sbs_fock_truncation(L, x, q, c):
    case = f"L={L}, x={x}, q={q}, c={c}"
    U = cs_unitary(L, x, [c] * min(L, x), np.random.default_rng(3))
    qbar, Ull = max(q), U[:L, :L]
    lam = lam_stein(U, L, q)
    NREF = (14 if L == 1 else 10) if QUICK else (18 if L == 1 else 13)
    ch_ref = LoopChannel(U, L, q, NREF)
    ri = stationary(ch_ref, 120 if QUICK else 250)
    print(f"\n=== L={L} x={x} q={q} c={c}  rho(Ull)={max(abs(np.linalg.eigvals(Ull))):.3f} ===")

    # ---- A  Stein mean against the simulation -------------------------------
    sim = mean_n(ri)
    mean_rtol = 5e-3 if QUICK else 1e-4
    okA = abs(sim - lam) < mean_rtol * max(1.0, lam)
    print(f"  A  lambda(Stein) = {lam:.6f}   <n>_inf(sim) = {sim:.6f}   "
          f"[{'ok' if okA else 'MISMATCH'}]")
    require(okA, "F-A stationary mean", case)

    # ---- B  every transient mean is bounded by the stationary mean ----------
    d = occ(ri)
    transient_means = []
    power = np.eye(L, dtype=complex)
    Uxl = U[:L, L:]
    for _ in range(20):
        entering = power @ Uxl
        transient_means.append(float(np.real(sum(
            q[a] * np.vdot(entering[:, a], entering[:, a])
            for a in range(x)
        ))) + (transient_means[-1] if transient_means else 0.0))
        power = Ull @ power
    okB = all(value <= lam * (1 + mean_rtol) + 1e-10
              for value in transient_means)
    print(f"  B  max transient mean = {max(transient_means):.6f}   "
          f"lambda = {lam:.6f}   [{'ok' if okB else 'VIOLATION'}]")
    require(okB, "F-B transient mean", case)

    # ---- C  measured tail against Markov ------------------------------------
    print("  C  N :  P[n>N] measured   lambda/(N+1)    ratio")
    okC = True
    for N in (2, 4, 6, 8):
        tail = sum(p for n, p in d.items() if n > N)
        bound = min(1.0, lam / (N + 1))
        okC &= (tail <= bound * (1 + 1e-9))
        print(f"     {N} : {tail:>15.3e}  {bound:>14.3e}  {tail / bound:>8.4f}"
              + ("" if tail <= bound * (1 + 1e-9) else "   VIOLATION"))
    require(okC, "F-C photon-number tail", case)

    # ---- D and E  truncated trajectory --------------------------------------
    print("  D  Nmax  k   1-tr(rho~)   ||rho_k-rho~_k||_1   identity (ii)")
    okD = True
    for Nmax in (3, 5):
        ch = LoopChannel(U, L, q, Nmax)
        rk, rex = ch.vacuum(), ch_ref.vacuum()
        for k in range(1, 4):
            rk, rex = ch.apply(rk), ch_ref.apply(rex)
            deficit = 1.0 - mass(rk)
            gap = tracedist(rex, rk)
            ok = abs(gap - deficit) < 1e-9 + 1e-6 * max(1.0, deficit)
            okD &= ok
            print(f"     {Nmax:>3} {k:>2} {deficit:>12.4e} {gap:>18.4e}   "
                  f"{'exact' if ok else 'BROKEN'}")
    require(okD, "F-D repeated projection", case)

    trajectories = {}
    for Nmax in (3, 5, 7):
        ch = LoopChannel(U, L, q, Nmax)
        rk = ch.vacuum()
        trajectories[Nmax] = []
        for _ in range(3):
            rk = ch.apply(rk)
            trajectories[Nmax].append(rk)

    print("  E  Nmax  k   deficit       k lambda/(N+1)   ok")
    okE = True
    for Nmax, states in trajectories.items():
        for k, rk in enumerate(states, start=1):
            deficit = max(0.0, 1.0 - mass(rk))
            bound = min(1.0, k * lam / (Nmax + 1))
            good = deficit <= bound * (1 + 1e-9) + 1e-12
            okE &= good
            print(f"     {Nmax:>3} {k:>2} {deficit:>12.4e} {bound:>18.4e}  "
                  f"{'ok' if good else 'VIOLATION'}")
    require(okE, "F-E cumulative Markov bound", case)

    print("  F  Nmax  k   normalised error   Gamma(k)+2 Delta_N(k)  ok")
    okF = True
    for Nmax, states in trajectories.items():
        for k, rk in enumerate(states, start=1):
            trace = mass(rk)
            normalised = {n: block / trace for n, block in rk.items()}
            err = tracedist(normalised, ri)
            delta = min(1.0, k * lam / (Nmax + 1))
            bound = Gamma(Ull, k, qbar) + 2 * delta
            good = err <= bound * (1 + 1e-9)
            okF &= good
            print(f"     {Nmax:>3} {k:>2} {err:>18.4e} {bound:>24.4e}  "
                  f"{'ok' if good else 'VIOLATION'}")
    require(okF, "F-F normalised total error", case)

    print("  G  R   exact 2P[n>R]   necessary lower bound   ok")
    okG = True
    second_moment_bound = 2 * lam ** 2 + lam
    measured_second_moment = sum(p * n ** 2 for n, p in d.items())
    okG &= measured_second_moment <= second_moment_bound * (1 + mean_rtol)
    print(f"     second moment {measured_second_moment:.6f} <= "
          f"{second_moment_bound:.6f}")
    for R in (0, 1, 2, 4):
        exact = 2 * sum(p for n, p in d.items() if n > R)
        lower = (0.0 if lam == 0 else
                 2 * max(0.0, lam - R) ** 2 / second_moment_bound)
        good = exact + 1e-10 >= lower
        okG &= good
        print(f"     {R:>1} {exact:>16.4e} {lower:>23.4e}  "
              f"{'ok' if good else 'VIOLATION'}")
    require(okG, "F-G necessary lower bound", case)
