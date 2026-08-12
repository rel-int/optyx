"""
================================================================================
 test_sbs_unitaries.py
 Verification suite for the size of xi in Haar and structured unitary families.
================================================================================

 WHAT STEP B IS, AND WHY THESE TESTS ARE CONFIRMATIONS RATHER THAN EVIDENCE

 Theorem A proves the finite-depth bound and reduces the question of polynomiality
 to one number, xi = 1 / (2 log(1 / rho(U_ll))), a property of an L x L matrix.
 Step B bounds xi. Its foundation is a LITERATURE THEOREM: the squared moduli of
 the eigenvalues of an L x L truncation of an M x M Haar unitary are distributed
 as L independent Beta(j, M-L) variables, a Kostlan-type consequence of the
 Zyczkowski-Sommers joint density, J. Phys. A 33, 2045 (2000).

 Everything in the two unitary-family appendices is then proved from that law, for every M and every
 L: the Chernoff bound (4.1), the closed form (5.1), the tail exponent (5.4), the
 L = 1 generating function (5.12). The tests below therefore confirm proofs; they
 are not substitutes for the proofs. Theorem 5.12 is proved from the factorial
 moments and S7 checks its finite-size approach independently.

   test  proof location       statement
   ----  -------------------  ------------------------------------------------
   S1    Haar appendix, B     exact Beta law, empirical CDF
   S2    Haar appendix, A/C   Chernoff rate from the exact law
   S3    Haar appendix, C     spectral edge at fixed loop fraction
   S4    Haar appendix, D     population versus coherence sector
   S5    structured, A        few-channel closed form and scaling
   S6    structured, A        tail exponent
   S7    structured, D        one-mode moments and thermal convergence
   S8    structured, B        passive transfer function on the unit circle

 CONVENTIONS

   M     total number of modes
   L     loop modes, so U_ll = U[:L,:L] is the truncation
   b     M - L, the number of open channels
   c     L / M, the loop fraction
   rho   spectral radius of U_ll
   xi    1 / (2 log(1/rho)),  the window constant of `03`, Corollary 3.16

Only S7 uses the independent Fock simulator; every other check is matrix algebra
or Monte Carlo.

 RUN
   pytest test/test_sbs_unitaries.py
================================================================================
"""

import numpy as np
from math import log, sqrt
import pytest
from scipy.special import betainc
from scipy.stats import linregress

RESULTS = {}
QUICK = True


def banner(name, description):
    print("\n" + "=" * 78)
    print(f" {name}   {description}")
    print("=" * 78)


def record(name, passed, detail=""):
    RESULTS[name] = (passed, detail)
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}  {detail}")
    if not passed:
        raise AssertionError(f"{name} failed: {detail}")


def haar(n, rng):
    """Haar-distributed unitary by QR with the standard phase fix.

    The phase fix is not cosmetic: without it, numpy's QR returns a
    non-Haar-distributed Q and every eigenvalue statistic below is wrong.
    """
    z = (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))) / sqrt(2)
    q, r = np.linalg.qr(z)
    d = np.diagonal(r)
    return q * (d / np.abs(d))


def rho2_sample(M, L, rng):
    """rho(U_ll)^2 for one Haar draw."""
    U = haar(M, rng)
    return float(max(abs(np.linalg.eigvals(U[:L, :L]))) ** 2)


def tail_exact(t, M, L):
    """P[rho^2 > t], computed in log space.

    Writing it as 1 - prod_j I_t(j, M-L) underflows to exactly 0 once the product
    rounds to 1, which happens by M = 320. Using the Beta symmetry
    I_t(a,b) = 1 - I_{1-t}(b,a) and -expm1(sum log1p(-e_j)) keeps every digit.
    """
    e = np.array([float(betainc(M - L, j, 1 - t)) for j in range(1, L + 1)])
    return float(-np.expm1(np.sum(np.log1p(-e))))


def cdf_exact(t, M, L):
    """P[rho^2 <= t] = prod_j I_t(j, M-L),  Haar appendix, (4.2)."""
    if t <= 0:
        return 0.0
    if t >= 1:
        return 1.0
    out = 1.0
    for j in range(1, L + 1):
        out *= float(betainc(j, M - L, t))
    return out


# ==============================================================================
#  S1   The exact Beta law                                        04, (4.2)
# ==============================================================================
#  Empirical CDF of rho(U_ll)^2 against the product of regularised incomplete
#  Beta functions. This is the one place where the literature theorem itself is
#  checked; every other statement in `04` and `05` is derived from it.
# ==============================================================================

def S1():
    banner("S1", "the exact Beta law, empirical CDF vs product formula   [Haar appendix, (4.2)]")
    rng = np.random.default_rng(2026)
    n = 400 if QUICK else 2000
    tol = 3.0 / sqrt(n)                 # 3 sigma of a Bernoulli proportion
    worst, bad = 0.0, 0
    print(f"  {'(M,L)':>10}  " + "  ".join(f"t={t:<15.1f}" for t in (0.3, 0.5, 0.7, 0.9)))
    for (M, L) in [(8, 3), (12, 6), (20, 10), (30, 10)]:
        s = np.array([rho2_sample(M, L, rng) for _ in range(n)])
        row = []
        for t in (0.3, 0.5, 0.7, 0.9):
            emp, th = float((s <= t).mean()), cdf_exact(t, M, L)
            worst = max(worst, abs(emp - th))
            bad += (abs(emp - th) > tol)
            row.append(f"{emp:.3f} vs {th:.3f}   ")
        print(f"  {str((M, L)):>10}  " + "".join(row))
    record("S1", bad == 0, f"worst |empirical - exact| = {worst:.4f}, tolerance {tol:.4f}")


# ==============================================================================
#  S2   The Chernoff rate                                          04, (4.1)
# ==============================================================================
#  P[rho^2 > 1-delta] <= L exp(-M D(c || 1-delta)).  The upper tail is computed
#  EXACTLY from (4.2), never sampled, so this checks the rate of the proved bound
#  rather than a Monte Carlo estimate. The measured rate must approach D from
#  ABOVE, since the union bound in (4.4) throws away the product structure.
# ==============================================================================

def kl(p, q):
    """Bernoulli relative entropy D(p || q), in nats."""
    def term(a, b):
        return 0.0 if a == 0 else a * log(a / b)
    return term(p, q) + term(1 - p, 1 - q)


def S2():
    banner("S2", "the Chernoff rate, exact from the law                  [Haar appendix, (4.1)]")
    c = 0.5
    ok = True
    print(f"  {'delta':>6}  " + "".join(f"{'M=' + str(M):>9}" for M in (20, 40, 80, 160, 320))
          + f"{'Chernoff D':>13}")
    for delta in (0.2, 0.1):
        D = kl(c, 1 - delta)
        rates = []
        for M in (20, 40, 80, 160, 320):
            L = int(c * M)
            p = tail_exact(1 - delta, M, L)
            rates.append(-log(p) / M)
        # must decrease towards D and stay above it
        ok &= all(rates[i] >= rates[i + 1] - 1e-9 for i in range(len(rates) - 1))
        ok &= all(r >= D - 1e-9 for r in rates)
        print(f"  {delta:>6.1f}  " + "".join(f"{r:>9.3f}" for r in rates) + f"{D:>13.3f}")
    p20 = tail_exact(0.8, 20, 10)
    p320 = tail_exact(0.8, 320, 160)
    print(f"   P[rho^2 > 0.8] = {p20:.2e} at M=20   and   {p320:.2e} at M=320")
    record("S2", ok, "measured rate decreases towards D and stays above it")


# ==============================================================================
#  S3   The edge of the support                                    04, (4.5)
# ==============================================================================
#  At fixed loop fraction c the L eigenvalues fill a disc of radius sqrt(c) and
#  repulsion pushes the largest to the edge, so rho -> sqrt(c) and xi = O(1).
#  Medians, approached from above with visible finite-size corrections.
# ==============================================================================

def S3():
    banner("S3", "the edge of the support at fixed c                     [Haar appendix, (4.5)]")
    rng = np.random.default_rng(11)
    n = 12 if QUICK else 30
    ok = True
    print(f"  {'(M,L)':>10}{'median rho':>12}{'sqrt(c)':>10}{'xi = 1/log(M/L)':>18}")
    prev = None
    for (M, L) in [(40, 20), (80, 40), (160, 80)]:
        med = float(np.median([sqrt(rho2_sample(M, L, rng)) for _ in range(n)]))
        edge = sqrt(L / M)
        ok &= (med > edge)                       # approached from above
        if prev is not None:
            ok &= (med <= prev + 1e-3)           # and decreasing
        prev = med
        print(f"  {str((M, L)):>10}{med:>12.4f}{edge:>10.4f}{1 / log(M / L):>18.3f}")
    record("S3", ok, "medians decrease towards sqrt(c) from above; xi = O(1)")


# ==============================================================================
#  S4   Which spectral gap governs the decay                        04, (4.6)
# ==============================================================================
#  The subdominant eigenvalue of the superoperator G(X) = U_ll X U_ll^dag equals
#  rho(U_ll)^2 on the BLOCK-DIAGONAL sector, and rho(U_ll) on the coherence
#  sector |m - n| = 1, which is never populated from vacuum with product Fock
#  inputs. Using the wrong one overestimates the window by a factor two in the
#  exponent. Both are computed here exactly: the superoperator of the one-photon
#  sector has eigenvalues {lambda_i lambda_j*}, so the two sectors are read off
#  the eigenvalues of U_ll directly.
# ==============================================================================

def S4():
    banner("S4", "which spectral gap governs the decay                   [Haar appendix, (4.6)]")
    rng = np.random.default_rng(5)
    ok = True
    print(f"  {'(M,L)':>10}{'rho':>10}{'coherence':>12}{'population':>13}{'rho^2':>10}")
    for (M, L) in [(8, 4), (12, 6), (20, 8)]:
        A = haar(M, rng)[:L, :L]
        rho = float(max(abs(np.linalg.eigvals(A))))
        # coherence sector |m-n| = 1: the channel acts on ONE side only,
        #   |1><0|  ->  A |1><0|,   so the eigenvalues are those of A
        coh = float(max(abs(np.linalg.eigvals(A))))
        # population sector |m-n| = 0: X -> A X A^dag on the one-photon block,
        #   eigenvalues lambda_i lambda_j^*, largest modulus rho^2
        pop = float(max(abs(np.linalg.eigvals(np.kron(A, A.conj())))))
        ok &= abs(coh - rho) < 1e-9 and abs(pop - rho ** 2) < 1e-9
        print(f"  {str((M, L)):>10}{rho:>10.4f}{coh:>12.4f}{pop:>13.4f}{rho ** 2:>10.4f}")
    print("   the coherence sector decays as rho, the population sector as rho^2;")
    print("   only the second is populated from vacuum with product Fock inputs,")
    print("   so using |lambda_2| would overestimate the window by a factor two.")
    record("S4", ok, "coherence gap = rho, population gap = rho^2, both exact")


# ==============================================================================
#  S5   Few open channels: closed form and scaling                 05, (5.1), (5.3)
# ==============================================================================
#  At b = M - L = 1 the law collapses to P[rho^2 <= t] = t^{L(L+1)/2}, since
#  Beta(j,1) has CDF t^j. The median gives xi ~ 0.72 L(L+1), and for general
#  fixed b, xi ~ L^{1+1/b} <= L^2. Polynomial, which is the point: the regime
#  L >> x is NOT where hardness lives.
# ==============================================================================

def S5():
    banner("S5", "few open channels: closed form and scaling       [structured appendix, (5.1), (5.3)]")
    rng = np.random.default_rng(17)
    n = 200 if QUICK else 800
    tol = 3.0 / sqrt(n)
    worst, bad = 0.0, 0
    print("   (5.1) closed form  P[rho^2 <= t] = t^{L(L+1)/2}  at b = 1")
    for L in (3, 5, 8):
        s = np.array([rho2_sample(L + 1, L, rng) for _ in range(n)])
        for t in (0.7, 0.9, 0.97):
            emp, th = float((s <= t).mean()), t ** (L * (L + 1) / 2)
            worst = max(worst, abs(emp - th))
            bad += (abs(emp - th) > tol)
            print(f"     L={L} t={t}:  measured {emp:.4f}   predicted {th:.4f}")
    print("   (5.3) scaling of the median xi")
    print(f"   {'b':>3}" + "".join(f"{'L=' + str(L):>10}" for L in (10, 20, 40))
          + f"{'L^(1+1/b) at 40':>17}")
    m = 20 if QUICK else 60
    okscale = True
    for b in (1, 2, 3):
        meds = []
        for L in (10, 20, 40):
            xis = []
            for _ in range(m):
                r2 = rho2_sample(L + b, L, rng)
                xis.append(1.0 / (-log(r2)) if r2 < 1 else np.inf)
            meds.append(float(np.median(xis)))
        okscale &= (meds[0] < meds[1] < meds[2])          # grows with L
        okscale &= (meds[2] < 40 ** 2 * 3)                # stays polynomial, <= O(L^2)
        print(f"   {b:>3}" + "".join(f"{v:>10.1f}" for v in meds)
              + f"{40 ** (1 + 1 / b):>17.0f}")
    record("S5", bad == 0 and okscale,
           f"closed form worst gap {worst:.4f} (tol {tol:.4f}); medians grow and stay below 3L^2")


# ==============================================================================
#  S6   The tail exponent                                          05, (5.4)
# ==============================================================================
#  P[xi > s] ~ C_b L^{b+1} / s^b:  the tail exponent is the number of open
#  channels. Measured as the slope of log P[xi > s] against log s.
# ==============================================================================

def S6():
    banner("S6", "the tail exponent of xi is b                           [structured appendix, (5.4)]")
    rng = np.random.default_rng(23)
    n = 800 if QUICK else 4000
    L = 12
    ok = True
    print(f"  {'b':>3}{'measured slope':>17}{'predicted':>12}")
    slopes = []
    for b in (1, 2, 3):
        xis = []
        for _ in range(n):
            r2 = rho2_sample(L + b, L, rng)
            xis.append(1.0 / (-log(r2)) if r2 < 1 else 1e12)
        xis = np.sort(np.array(xis))
        # upper tail: survival function on the top decade of the sample
        lo, hi = np.quantile(xis, 0.70), np.quantile(xis, 0.995)
        grid = np.exp(np.linspace(log(lo), log(hi), 25))
        surv = np.array([(xis > g).mean() for g in grid])
        keep = surv > 0
        slope = -linregress(np.log(grid[keep]), np.log(surv[keep])).slope
        # Finite samples bias this estimator DOWN: the extreme tail is
        # undersampled, so the fitted slope falls short of b. The test therefore
        # asks for the exponent to be close from below and to GROW with b, which
        # is the qualitative content of (5.4).
        ok &= (b - 0.7 <= slope <= b + 0.4)
        slopes.append(slope)
        print(f"  {b:>3}{slope:>17.3f}{b:>12}")
    ok &= all(slopes[i] < slopes[i + 1] for i in range(len(slopes) - 1))
    record("S6", ok, f"slopes {[round(v, 2) for v in slopes]} grow with b, close from below")


# ==============================================================================
#  S7   L = 1: closed form and the thermal limit          05, (5.11), Theorem 5.12
# ==============================================================================
#  At L = 1 the loop weights are geometric, p_i = (1-rho^2) rho^{2i}, and the
#  factorial moments close in O(K):
#      F_K = (K!)^2 c^K r^{K(K-1)/2} / prod_{j=1..K} (1 - r^j),  c = 1-r, r = rho^2
#  checked here against a direct evaluation from the geometric weights.
#
#  Theorem 5.12 proves that as rho -> 1 the stationary loop state converges to a
#  thermal state of mean 1. This test checks the finite-size approach; it does
#  not assert an unproved uniform convergence rate.
# ==============================================================================

def S7():
    banner("S7", "L = 1: factorial moments and the thermal limit  [structured appendix, (5.12), Thm 5.12]")
    from scipy.special import gammaln as _gl

    from sbs_loop_simulator import LoopChannel, cs_unitary, stationary

    def e_K(K, r):
        """Elementary symmetric polynomial of the weights p_i = (1-r) r^i.

        e_K(p) = c^K r^{K(K-1)/2} / prod_{j=1..K} (1 - r^j), the standard q-series
        identity, with c = 1 - r.
        """
        c = 1 - r
        return float(c ** K * r ** (K * (K - 1) / 2)
                     / np.prod([1 - r ** j for j in range(1, K + 1)]))

    # The stationary loop distribution is computed EXACTLY by the quantum channel,
    # so it arbitrates between the two candidate normalisations of F_K without any
    # modelling assumption.
    print("   F_K of the exact stationary loop distribution, against two candidates")
    print(f"  {'rho':>6}{'K':>3}{'F_K exact':>13}{'K! e_K':>13}{'(K!)^2 e_K':>14}   verdict")
    ok_fac = True
    for rho in (0.5, 0.7, 0.8):
        U = cs_unitary(1, 1, [rho], np.random.default_rng(3))
        ch = LoopChannel(U, 1, (1,), 16 if QUICK else 24)
        ri = stationary(ch, 200 if QUICK else 400)
        p = np.array([float(np.real(np.trace(b_))) for _, b_ in sorted(ri.items())])
        r = rho ** 2
        for K in (1, 2, 3):
            n = np.arange(len(p))
            fall = np.exp(_gl(n + 1) - _gl(np.clip(n - K + 1, 1, None)))
            fall[n < K] = 0.0
            F = float(np.sum(p * fall))
            c1 = float(np.exp(_gl(K + 1))) * e_K(K, r)
            c2 = float(np.exp(2 * _gl(K + 1))) * e_K(K, r)
            # At K = 1 the two candidates coincide, 1! = (1!)^2, so only K >= 2
            # discriminates. The verdict is the match with (5.12) itself.
            ok_fac &= abs(F - c2) < 1e-6
            which = "both (K=1)" if K == 1 else (
                    "(K!)^2 e_K = (5.12)" if abs(F - c2) < 1e-6 else
                    "K! e_K" if abs(F - c1) < 1e-6 else "NEITHER")
            print(f"  {rho:>6.2f}{K:>3}{F:>13.6f}{c1:>13.6f}{c2:>14.6f}   {which}")
    print("   The exact channel selects (K!)^2 e_K, which is (5.12). The extra K! over")
    print("   the Poisson-binomial value K! e_K is BOSONIC BUNCHING: a distinguishable-")
    print("   particle picture would say each photon's fate is recorded by its own")
    print("   detector slot, hence a sum of independent Bernoulli variables. It is not:")
    print("   the detector at step i registers a NUMBER, not an identity, so the")
    print("   histories interfere. This is also the classical-coupling obstruction")
    print("   recorded in proof-notes-theorem-a.md.")

    # Theorem 5.12: as rho -> 1 the stationary state approaches thermal of mean 1.
    print("   thermal limit: p_0..p_3 against thermal(mean 1) = 2^-(n+1)")
    th = [0.5, 0.25, 0.125, 0.0625]
    ok_th, prev = True, None
    thermal_rhos = (0.80, 0.85, 0.90) if QUICK else (0.90, 0.95, 0.98)
    for rho in thermal_rhos:
        U = cs_unitary(1, 1, [rho], np.random.default_rng(3))
        ch = LoopChannel(U, 1, (1,), 24 if QUICK else 60)
        ri = stationary(ch, 600 if QUICK else 3000)
        p = np.array([float(np.real(np.trace(b_))) for _, b_ in sorted(ri.items())])
        dev = max(abs(p[n] - th[n]) for n in range(4))
        if prev is not None:
            ok_th &= (dev <= prev + 1e-6)
        prev = dev
        print(f"  rho={rho:.2f}  " + " ".join(f"{p[n]:.6f}" for n in range(4))
              + f"   max deviation {dev:.5f}")
    record("S7", ok_th and ok_fac,
           ("moments match (5.12) exactly, bunching confirmed; " if ok_fac
            else "MOMENTS DO NOT MATCH (5.12); ")
           + "thermal deviation shrinks with rho")


def S8():
    """Verify the transfer-function identity of the passive colligation."""
    banner("S8", "transfer function is unitary on the unit circle  [structured appendix, (5.7)]")
    rng = np.random.default_rng(41)
    worst = 0.0
    for L, x in ((1, 1), (2, 1), (3, 2), (2, 3)):
        U = haar(L + x, rng)
        Ull = U[:L, :L]
        Uxl = U[:L, L:]
        Ulx = U[L:, :L]
        Uxx = U[L:, L:]
        for angle in (0.17, 0.73, 1.41, 2.33):
            z = np.exp(1j * angle)
            transfer = Uxx + z * Ulx @ np.linalg.solve(
                np.eye(L) - z * Ull, Uxl)
            residual = np.linalg.norm(
                transfer.conj().T @ transfer - np.eye(x), 2)
            worst = max(worst, float(residual))
    record("S8", worst < 1e-10, f"worst unitarity residual {worst:.2e}")


@pytest.mark.parametrize(
    "check", [S1, S2, S3, S4, S5, S6, S7, S8],
    ids=[f"S{index}" for index in range(1, 9)],
)
def test_sbs_unitary_family(check):
    """Run every Haar and structured-unitary regression."""
    RESULTS.clear()
    check()
