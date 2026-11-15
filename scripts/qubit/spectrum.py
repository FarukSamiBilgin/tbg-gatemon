"""Charge-basis transmon spectrum (M6_BRIEF.md Sec. 3, AT-1 / AT-2).

Hamiltonian, in frequency units (GHz, i.e. everything already divided by h):

    H = 4 E_C (n - n_g)^2  -  (E_J/2) (|n><n+1| + h.c.)

on the truncated charge ladder n = -N .. +N.  Exact diagonalisation; N >= 35
per the brief, with an N = 35 vs N = 50 convergence check in the acceptance
suite.

PHYSICS FLAG (logged, not acted on in this milestone): the Hamiltonian uses a
pure -E_J cos(phi) potential, while material.json:cpr.harmonics_by_tau carries
a non-sinusoidal (skewed) CPR for the TBG link.  Including the higher CPR
harmonics is a logged future task, not part of M6 milestone 1.

Charge dispersion eps1 = |f01(n_g = 1/2) - f01(n_g = 0)| is exponentially
small.  Below ~1 mHz the float64 eigensolver is at its absolute-accuracy
floor and produces fake non-monotonicity (M6_BRIEF.md Sec. 1), so an
extended-precision path (`eps1_mp`, mpmath Sturm-sequence bisection on the
symmetric tridiagonal matrix) and the asymptotic perturbative formula
(`eps1_perturbative`) are provided.
"""
import math

import numpy as np

from .inputs import ANHARM_MIN_GHZ, EPS1_MAX_HZ, E, H, N_CHARGE


def hamiltonian(EJ, EC, ng=0.0, N=N_CHARGE):
    """Charge-basis transmon Hamiltonian, GHz units, shape (2N+1, 2N+1)."""
    n = np.arange(-N, N + 1)
    Hm = np.diag(4.0 * EC * (n - ng) ** 2)
    off = -EJ / 2.0 * np.ones(2 * N)
    Hm += np.diag(off, 1) + np.diag(off, -1)
    return Hm


def transmon(EJ, EC, ng=0.0, N=N_CHARGE, nlev=4):
    """Lowest `nlev` eigenvalues (GHz) at offset charge `ng`."""
    return np.linalg.eigvalsh(hamiltonian(EJ, EC, ng, N))[:nlev]


def levels(EJ, EC, N=N_CHARGE):
    """(f01, anharmonicity, eps1) in GHz.

    anharmonicity = f12 - f01 (negative for a transmon);
    eps1 = charge dispersion of f01 between n_g = 0 and n_g = 1/2.
    """
    ev0 = transmon(EJ, EC, 0.0, N)
    ev5 = transmon(EJ, EC, 0.5, N)
    f01, f12 = ev0[1] - ev0[0], ev0[2] - ev0[1]
    eps1 = abs((ev5[1] - ev5[0]) - f01)
    return f01, f12 - f01, eps1


def n01_element(EJ, EC, N=N_CHARGE):
    """|<0| n |1>| at n_g = 0 -- the charge matrix element that sets g."""
    n = np.arange(-N, N + 1)
    w, v = np.linalg.eigh(hamiltonian(EJ, EC, 0.0, N))
    return abs(v[:, 0] @ (n * v[:, 1]))


# ------------------------------------------------------------- asymptotics ---
def f01_asymptotic(EJ, EC):
    """Standard transmon asymptote sqrt(8 E_J E_C) - E_C."""
    return np.sqrt(8 * EJ * EC) - EC


def n01_asymptotic(EJ, EC):
    """Harmonic-oscillator asymptote (E_J/8E_C)^(1/4) / sqrt(2)."""
    return (EJ / (8 * EC)) ** 0.25 / np.sqrt(2)


def eps1_perturbative(EJ, EC):
    """Koch-type asymptotic charge dispersion of the m = 1 level, in GHz.

        eps_m = E_C * 2^(4m+5)/m! * sqrt(2/pi) * (E_J/2E_C)^(m/2+3/4)
                * exp(-sqrt(8 E_J/E_C))

    Asymptotic only -- roughly +12% at E_J/E_C = 100.  Used as an
    order-of-magnitude cross-check on `eps1_mp`, never as an AT value.
    """
    m = 1
    return (EC * 2.0 ** (4 * m + 5) / math.factorial(m) * np.sqrt(2 / np.pi)
            * (EJ / (2 * EC)) ** (m / 2 + 0.75) * np.exp(-np.sqrt(8 * EJ / EC)))


# ------------------------------------------------- extended precision (mpmath)
def _sturm_count(d, e, x, tiny):
    """Number of eigenvalues strictly below x for a symmetric tridiagonal
    matrix with diagonal `d` and off-diagonal `e` (LDL^T Sturm sequence)."""
    cnt = 0
    q = d[0] - x
    if q < 0:
        cnt += 1
    for i in range(1, len(d)):
        if q == 0:
            q = tiny
        q = d[i] - x - e[i - 1] ** 2 / q
        if q < 0:
            cnt += 1
    return cnt


def _eig_bisect(d, e, k, lo, hi, dps):
    """k-th eigenvalue (0-based, ascending) by Sturm bisection."""
    import mpmath as mp
    tol = mp.mpf(10) ** (-(dps - 8))
    tiny = mp.mpf(10) ** (-(dps - 5))
    for _ in range(16 * dps):
        mid = (lo + hi) / 2
        if _sturm_count(d, e, mid, tiny) <= k:
            lo = mid
        else:
            hi = mid
        if hi - lo <= tol * max(mp.mpf(1), abs(lo)):
            break
    return (lo + hi) / 2


def _levels_mp(EJ, EC, ng, N, dps, nlev=3):
    import mpmath as mp
    mp.mp.dps = dps
    EJm, ECm, ngm = mp.mpf(repr(EJ)), mp.mpf(repr(EC)), mp.mpf(repr(ng))
    d = [4 * ECm * (mp.mpf(n) - ngm) ** 2 for n in range(-N, N + 1)]
    e = [-EJm / 2] * (2 * N)
    lo = -2 * EJm - 100
    hi = 4 * ECm * (N + 1) ** 2 + 100
    return [_eig_bisect(d, e, k, lo, hi, dps) for k in range(nlev)]


def eps1_mp(EJ, EC, N=N_CHARGE, dps=60):
    """Charge dispersion eps1 in GHz at `dps` decimal digits (mpmath).

    Required whenever eps1 falls below ~1 mHz: the float64 eigensolver's
    absolute accuracy is set by ||H|| ~ 4 E_C N^2, so tiny dispersions there
    are numerical floor, not physics (M6_BRIEF.md Sec. 1).  Returns a Python
    float; use `eps1_mp_raw` for the mpmath value.
    """
    return float(eps1_mp_raw(EJ, EC, N, dps))


def eps1_mp_raw(EJ, EC, N=N_CHARGE, dps=60):
    a = _levels_mp(EJ, EC, 0.0, N, dps)
    b = _levels_mp(EJ, EC, 0.5, N, dps)
    return abs((b[1] - b[0]) - (a[1] - a[0]))


# ------------------------------------------------------------- design scan ---
def design_scan(EJ, ratios, N=N_CHARGE):
    """{ratio: {EC_GHz, C_fF, f01_GHz, anharm_GHz, eps1_Hz}} for E_J/E_C ladder."""
    scan = {}
    for r in ratios:
        EC = EJ / r
        C_fF = E ** 2 / (2 * H * EC * 1e9) * 1e15     # C_Sigma from E_C = e^2/2C
        f01, anh, eps1 = levels(EJ, EC, N)
        scan[r] = dict(EC_GHz=EC, C_fF=C_fF, f01_GHz=f01, anharm_GHz=anh,
                       eps1_Hz=eps1 * 1e9)
    return scan


def select_design(scan, eps1_max_Hz=EPS1_MAX_HZ, anharm_min_GHz=ANHARM_MIN_GHZ):
    """Smallest E_J/E_C meeting both scan criteria (M6_BRIEF.md Sec. 1)."""
    return min(r for r in scan
               if scan[r]["eps1_Hz"] < eps1_max_Hz
               and abs(scan[r]["anharm_GHz"]) >= anharm_min_GHz)
