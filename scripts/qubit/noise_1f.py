"""P4 -- 1/f gate noise done properly: Ramsey/echo filter-function integrals.

Replaces the quasistatic estimates in ``qubit/dome.py``
(``Tphi_first_order`` / ``Tphi_second_order``, kept untouched as the
milestone-1 reference) with the standard dephasing theory of

    G. Ithier, E. Collin, P. Joyez, P. J. Meeson, D. Vion, D. Esteve,
    F. Chiarello, A. Shnirman, Y. Makhlin, J. Schriefl, G. Schoen,
    "Decoherence in a superconducting quantum bit circuit",
    Phys. Rev. B 72, 134519 (2005).

===============================================================================
1.  NOISE CONVENTION  (stated once, used everywhere)
===============================================================================

The noisy parameter is the filling factor ``nu`` (electrons per moire cell),
set by the gate.  Its fluctuation ``dnu(t)`` is a stationary Gaussian process
with a 1/f spectrum.  We use *Ithier's* normalisation throughout:

    S_nu(w) = A^2 / |w|          [two-sided, angular frequency w in rad/s]

    <dnu^2> = Int_{-inf}^{+inf} S_nu(w) dw = 2 A^2 ln(w_uv / w_ir)

Equivalently, as a one-sided ordinary-frequency density,

    S_nu^(1)(f) = 2 A^2 / f      [f in Hz],   <dnu^2> = 2 A^2 ln(f_uv/f_ir)

so ``A`` is the rms filling-fraction noise per e-fold of bandwidth divided by
sqrt(2).  If you prefer the "S(f) = A_f^2/f, one-sided" convention then
A_f = sqrt(2) A.  Every T2 below is *parametric in A*; A is never fixed.

Infrared cutoff.  1/f noise has no finite variance, so the free-induction
(Ramsey) first-order result carries an explicit log of the IR cutoff.  The
cutoff is a *measurement* convention: it is set by the total averaging time of
the experiment, w_ir ~ 2 pi / T_total.  It is exposed everywhere as the keyword
``w_ir`` in **rad/s**, with default ``W_IR_DEFAULT = 2 pi * 1 Hz`` (i.e. a
1 s total acquisition).  ``f_ir_sensitivity()`` quantifies the dependence:
T2* moves only as 1/sqrt(ln), ~ -35%/+30% over six decades of f_ir.

Ultraviolet cutoff.  It enters only the second-order (quadratic) channel, and
only through the variance of the *fast* components, which self-average into a
frequency shift rather than a decay.  The exact evaluator sets w_uv to the
Nyquist frequency of its own time grid (w_uv = pi M / t); T2 is stable to
<0.3% between M = 48 and M = 128 (see the ``M_GRID`` note below).

===============================================================================
2.  WHAT IS COMPUTED
===============================================================================

Qubit frequency vs filling (all from ``qubit/dome.py``, which reads
material.json):   w01(nu) = 2 pi f01(nu),
    D1 = df01/dnu   [GHz per unit nu]     -> dw/dnu   = 2 pi * 1e9 * D1
    D2 = d2f01/dnu2 [GHz per unit nu^2]   -> d2w/dnu2 = 2 pi * 1e9 * D2

FIRST ORDER (off-apex, D1 != 0).  Gaussian, exact:

    chi(t) = (1/2) <phi^2>,  phi = (dw/dnu) Int_0^t s(t') dnu(t') dt'

  Ramsey  (s = +1):
    chi_R(t) = (dw/dnu)^2 A^2 t^2 * I_R(w_ir t / 2)
    I_R(a)   = Int_a^inf sin^2(u)/u^3 du
             = (1-cos 2a)/(4a^2) + sin(2a)/(2a) - Ci(2a)
             -> ln(1/(w_ir t)) + 3/2 - gamma_E      (gamma_E = 0.5772...)
    i.e. exactly the Ithier form  chi = (2 pi D1)^2 A^2 t^2 [ln(1/(w_ir t)) + O(1)]
    with the O(1) constant pinned to 3/2 - gamma_E = 0.92278.

  Echo (one pi pulse at t/2, s = +1 then -1):
    chi_E(t) = (dw/dnu)^2 A^2 t^2 * I_E(w_ir t / 4)
    I_E(a)   = Int_a^inf sin^4(u)/u^3 du  ->  ln 2      (the standard ln 2)
  The IR divergence cancels: the echo filter function ~ f^2 at low f.

SECOND ORDER (at the apex, D1 = 0 by symmetry, D2 dominant).  The phase is a
*quadratic* form in a Gaussian process, so the decay is non-Gaussian:

    phi(t) = (1/2)(d2w/dnu2) Int_0^t s(t') dnu(t')^2 dt'
    <e^{i phi}> = det(I - i (d2w/dnu2) K_t S)^{-1/2}
    chi_2(t) = -ln|<e^{i phi}>| = (1/4) Sum_k ln(1 + (d2w/dnu2 * mu_k)^2)

with K_t the covariance operator of dnu on [0, t] and S = diag(s).  This is
Ithier's exact quadratic-coupling expression (their Sec. II); ``chi_second_
order`` evaluates it directly by discretising K_t (no long-time or static
approximation).  Two analytic limits of the same expression are provided as
brackets and are what the literature usually quotes:

  * ``chi_second_order_static`` -- keep only the dominant (quasistatic)
    eigenvalue mu_1 = sigma_qs^2(t) t with sigma_qs^2(t) = 2 A^2 ln(1/(w_ir t)):
        chi = (1/4) ln[1 + (d2w/dnu2 * sigma_qs^2(t) * t)^2]
    This is the 1/f-log-dressed version of the textbook static-Gaussian
    quadratic-coupling result and it *under*estimates chi (many modes matter).

  * ``chi_second_order_asymptotic`` -- the long-time (Szego/Kac) limit of the
    determinant,
        chi -> (t/4 pi) Int_{w_ir}^{w_uv} ln(1 + (pi |d2w/dnu2| A^2 / w)^2) dw
             -> (pi/4) |d2w/dnu2| A^2 t
    the "chi ~ pi * D2 * A^2 * t" linear-in-t, non-Gaussian scaling.  Note it
    is sequence-independent: **echo buys nothing at second order asymptotically**
    (the measured gain at chi = 1 is only ~1.4x, see the P4 report).

CROSSOVER (general nu).  Both orders contribute; they are combined as
independent rates,

    chi_total = chi_first_order + chi_second_order

APPROXIMATION STATED: this drops the linear-quadratic cross term.  For Gaussian
dnu the leading cross contribution vanishes (odd moments vanish), so the error
is higher order; ``chi_exact_combined`` evaluates the exact linear+quadratic
Gaussian characteristic function and is used in tests/test_m6_p4.py to bound
the additivity error on the grid.

T2 is defined throughout by  chi(T2) = 1  (1/e of the coherence envelope).
The same definition is applied to Ramsey and echo so the gain map is
convention-free.

===============================================================================
3.  LITERATURE NOTE ON A  (claim discipline: A stays free)
===============================================================================

See ``LITERATURE_NOTE``.  Short version: no 1/f filling-noise amplitude has
ever been measured for a TBG junction, the translation from reported charge /
gate 1/f magnitudes in other platforms is not a measurement, and therefore A is
swept over 1e-6 .. 1e-2 and every number in this deliverable is quoted
parametrically in A.  No absolute TBG coherence claim is made anywhere.
"""
import json
import pathlib

import numpy as np
from scipy.optimize import brentq
from scipy.special import sici

from . import dome
from .inputs import NU_OPT, ROOT

NU_OPT_APEX = NU_OPT        # src: material.json:sc_dome.nu_opt (= -2.4)

# ----------------------------------------------------------------- constants --
GAMMA_E = float(np.euler_gamma)
LN2 = float(np.log(2.0))
RAMSEY_O1 = 1.5 - GAMMA_E          # = 0.9227843... the O(1) next to ln(1/w_ir t)

F_IR_DEFAULT_HZ = 1.0              # IR cutoff convention: 1 s total acquisition
W_IR_DEFAULT = 2.0 * np.pi * F_IR_DEFAULT_HZ      # rad/s

M_GRID = 96
# Time-grid points for the exact second-order determinant.  Convergence is
# ~1/M: at the apex, A = 1e-4, measured against M = 256,
#     chi_2(t = 2e-5 s):  M=48 -0.09%,  M=64 -0.06%,  M=96 -0.03%
#     chi_2(t = 1e-4 s):  M=48 -0.79%,  M=64 -0.55%,  M=96 -0.31%
# i.e. <0.1% on T2 itself (which sits at chi = 1).  M is also kept below 128
# on purpose: this machine's OpenBLAS switches a 128x128 complex LU to 20
# threads and takes 88 ms per matrix versus 0.05 ms at M = 64 -- a 1700x
# cliff.  Raise M (and expect the cliff) if you want the extra 0.3%.
T_MIN_S = 1e-13                    # lower end of the chi(T2) = 1 root search
# Upper end: the acquisition window itself, T_total = 2 pi / w_ir.  A T2 longer
# than the window is not a prediction -- the 1/f description with that IR
# cutoff no longer applies -- so such points are returned as NaN, not as a
# number.  (It also keeps the exact-determinant grid sane: w_uv = pi M / t.)

LITERATURE_NOTE = (
    "A is the 1/f amplitude of the filling-fraction noise in the convention "
    "S_nu(w) = A^2/|w| (two-sided, angular), i.e. <dnu^2> = 2 A^2 ln(w_uv/w_ir); "
    "A is a FREE PARAMETER and is swept over 1e-6..1e-2 in units of nu "
    "(electrons per moire cell). It has never been measured for a TBG junction. "
    "For orientation only, and explicitly NOT as a calibration: 1/f charge noise "
    "in conventional superconducting circuits is quoted around 1e-4..1e-3 e "
    "(rms per e-fold at 1 Hz) and 1/f flux noise around 1 uPhi0 [Astafiev et al., "
    "PRL 93, 267007 (2004); Ithier et al., PRB 72, 134519 (2005)]. Neither maps "
    "onto dnu for a gated 2D-material device: here dnu is produced by gate-voltage "
    "noise, charge traps in the hBN/substrate dielectric stack, and moire-domain "
    "rearrangement, and electrostatic-gate 1/f in 2D-material devices is strongly "
    "device- and fabrication-dependent. Every T2 in this deliverable is therefore "
    "reported as a function of A; no absolute TBG coherence number is claimed."
)


# ============================================================ filter functions =
def _P(a, b):
    """Int_a^inf (1 - cos(b u)) / u^3 du, closed form (a > 0).

    = (1-cos ba)/(2a^2) + (b/2) sin(ba)/a - (b^2/2) Ci(ba)

    NUMERICS: (1 - cos x) is evaluated as 2 sin^2(x/2).  The naive form loses
    all significance for x ~ 1e-8 (1 - cos x ~ 5e-17 in float64), which showed
    up directly as an 8% error in the echo factor at t = 10 ns.
    """
    a = np.asarray(a, dtype=float)
    ba = b * a
    ci = sici(ba)[1]
    return (np.sin(0.5 * ba) / a) ** 2 + 0.5 * b * np.sin(ba) / a \
        - 0.5 * b ** 2 * ci


def ramsey_log_integral(t, w_ir=W_IR_DEFAULT):
    """I_R = Int_{w_ir t/2}^inf sin^2(u)/u^3 du  -- the Ramsey 1/f log factor.

    Exact (no small-argument expansion).  Asymptotically ln(1/(w_ir t)) + 3/2
    - gamma_E; see ``ramsey_log_asymptotic``.
    """
    return 0.5 * _P(0.5 * w_ir * np.asarray(t, dtype=float), 2.0)


def ramsey_log_asymptotic(t, w_ir=W_IR_DEFAULT):
    """ln(1/(w_ir t)) + 3/2 - gamma_E -- the quoted Ithier log factor."""
    return np.log(1.0 / (w_ir * np.asarray(t, dtype=float))) + RAMSEY_O1


def echo_log_integral(t, w_ir=W_IR_DEFAULT):
    """I_E = Int_{w_ir t/4}^inf sin^4(u)/u^3 du  -- the echo 1/f factor -> ln 2.

    sin^4 u = [4(1-cos 2u) - (1-cos 4u)]/8, so I_E = [4 P(a,2) - P(a,4)]/8.
    """
    a = 0.25 * w_ir * np.asarray(t, dtype=float)
    return (4.0 * _P(a, 2.0) - _P(a, 4.0)) / 8.0


# ================================================================ first order =
def dw_dnu(D1_GHz):
    """dw01/dnu in rad/s per unit filling.  D1 from dome.D1 (GHz per nu)."""
    return 2.0 * np.pi * 1e9 * np.asarray(D1_GHz, dtype=float)


def d2w_dnu2(D2_GHz):
    """d2w01/dnu2 in rad/s per unit filling^2.  D2 from dome.D2 (GHz per nu^2)."""
    return 2.0 * np.pi * 1e9 * np.asarray(D2_GHz, dtype=float)


def chi_first_order(t, D1_GHz, A, sequence="ramsey", w_ir=W_IR_DEFAULT,
                    asymptotic=False):
    """First-order coherence exponent chi(t) = -ln|f_z(t)|.  Vectorised.

    Ramsey: chi = (2 pi D1)^2 A^2 t^2 [ln(1/(w_ir t)) + 3/2 - gamma_E]
    Echo:   chi = (2 pi D1)^2 A^2 t^2 ln 2
    """
    t = np.asarray(t, dtype=float)
    pref = (dw_dnu(D1_GHz) * A * t) ** 2
    if sequence == "ramsey":
        fac = ramsey_log_asymptotic(t, w_ir) if asymptotic \
            else ramsey_log_integral(t, w_ir)
    elif sequence == "echo":
        fac = LN2 if asymptotic else echo_log_integral(t, w_ir)
    else:
        raise ValueError(sequence)
    return pref * np.maximum(fac, 0.0)


# =============================================================== second order =
def _covariance(tau, A, w_ir, w_uv):
    """<dnu(0) dnu(tau)> = 2 A^2 [Ci(w_uv |tau|) - Ci(w_ir |tau|)]."""
    tau = np.abs(np.asarray(tau, dtype=float))
    out = np.empty_like(tau)
    z = tau == 0.0
    out[z] = 2.0 * A ** 2 * np.log(w_uv / w_ir)
    tn = tau[~z]
    out[~z] = 2.0 * A ** 2 * (sici(w_uv * tn)[1] - sici(w_ir * tn)[1])
    return out


def _kernel(M, dt, A, w_ir):
    """Toeplitz covariance matrix K_ij = C(|i-j| dt) on the M-point grid.

    Only M distinct lags exist, so ``sici`` is called M times, not M^2 -- this
    is what makes the exact determinant cheap enough for the dense (nu, A) map.
    """
    return _kernel_batch(np.array([M * dt]), M, A, w_ir)[0]


def _kernel_batch(ts, M, A, w_ir):
    """Stack of Toeplitz covariance matrices, shape (len(ts), M, M).

    PERFORMANCE (the reason this is batched): every LAPACK entry carries a
    fixed ~15 ms thread-pool overhead on this machine, which dwarfs the ~0.2 ms
    of actual work in a 128x128 complex LU.  Evaluating a whole batch of times
    in ONE stacked ``slogdet`` call amortises that overhead ~50x and is what
    makes the (nu, A) map a ~2 min job instead of a ~40 min one.
    """
    ts = np.atleast_1d(np.asarray(ts, dtype=float))
    dt = ts / M
    w_uv = np.pi / dt
    if np.any(w_uv <= w_ir):
        raise ValueError(f"time grid too coarse: min w_uv={w_uv.min():g} "
                         f"<= w_ir={w_ir:g}")
    lag = np.arange(M)[None, :] * dt[:, None]              # (B, M)
    c = np.empty_like(lag)
    c[:, 0] = 2.0 * A ** 2 * np.log(w_uv / w_ir)
    c[:, 1:] = 2.0 * A ** 2 * (sici(w_uv[:, None] * lag[:, 1:])[1]
                               - sici(w_ir * lag[:, 1:])[1])
    idx = np.arange(M)
    return c[:, np.abs(idx[:, None] - idx[None, :])]        # (B, M, M)


def _weight(ti, t, sequence):
    s = np.ones_like(ti)
    if sequence == "echo":
        s[ti > 0.5 * t] = -1.0
    elif sequence != "ramsey":
        raise ValueError(sequence)
    return s


def chi_second_order_batch(ts, D2_GHz, A, sequence="ramsey",
                           w_ir=W_IR_DEFAULT, M=M_GRID):
    """Exact second-order exponent (1/2) Re ln det(I - i d2w K_t S), vectorised.

    Ithier's quadratic-coupling result evaluated without a static or long-time
    approximation, for a whole array of measurement times in one LAPACK call.

    The pi-pulse weight s(t') = sign(t/2 - t') is the SAME vector on every
    time grid (the grid is relative: t_i = (i+1/2) t / M), so only the kernel
    and the dt prefactor depend on t.
    """
    ts = np.atleast_1d(np.asarray(ts, dtype=float))
    d2 = float(d2w_dnu2(D2_GHz))
    if d2 == 0.0 or A == 0.0:
        return np.zeros_like(ts)
    s = _weight((np.arange(M) + 0.5) / M, 1.0, sequence)
    K = _kernel_batch(ts, M, A, w_ir)
    Mx = np.eye(M)[None] - 1j * (d2 * ts / M)[:, None, None] \
        * (K * s[None, None, :])
    return 0.5 * np.real(np.linalg.slogdet(Mx)[1])


def chi_second_order(t, D2_GHz, A, sequence="ramsey", w_ir=W_IR_DEFAULT,
                     M=M_GRID):
    """Scalar form of ``chi_second_order_batch``."""
    return float(chi_second_order_batch(float(t), D2_GHz, A, sequence,
                                        w_ir, M)[0])


def chi_second_order_static(t, D2_GHz, A, w_ir=W_IR_DEFAULT):
    """Leading-eigenvalue (quasistatic) limit -- a *lower* bound on chi_2.

    chi = (1/4) ln[1 + (d2w/dnu2 * sigma_qs^2(t) * t)^2],
    sigma_qs^2(t) = 2 A^2 [ln(1/(w_ir t)) + 3/2 - gamma_E].
    """
    t = np.asarray(t, dtype=float)
    sig2 = 2.0 * A ** 2 * np.maximum(ramsey_log_asymptotic(t, w_ir), 0.0)
    return 0.25 * np.log1p((np.abs(d2w_dnu2(D2_GHz)) * sig2 * t) ** 2)


def chi_second_order_asymptotic(t, D2_GHz, A, w_ir=W_IR_DEFAULT, w_uv=None):
    """Long-time (Szego/Kac) limit of the determinant:

        chi -> (t/4 pi) Int_{w_ir}^{w_uv} ln(1 + (b/w)^2) dw,  b = pi |d2w| A^2
             -> (pi/4) |d2w/dnu2| A^2 t   for w_ir << b << w_uv,

    i.e. the linear-in-t "chi ~ pi D2 A^2 t" non-Gaussian scaling.  Sequence
    independent -> echo has no asymptotic second-order advantage.

    VALIDITY: the linear regime needs w_ir << 1/t << b.  Measured against the
    exact determinant at the apex, A = 1e-4 (b = 3.3e4 rad/s, w_ir = 2 pi):
    ratio exact/asymptote = 2.75 (t = 0.1 ms), 1.95 (1 ms), 1.06 (10 ms), then
    it drops below 1 once w_ir t ~ 1.  At the chi = 1 crossing -- which is what
    sets T2 -- the asymptote is therefore ~3x optimistic; the P4 maps use the
    exact determinant, and this function is a bracket, not the production path.
    """
    t = np.asarray(t, dtype=float)
    b = np.pi * np.abs(d2w_dnu2(D2_GHz)) * A ** 2
    if w_uv is None:
        return 0.25 * b * t                     # exact w_ir->0, w_uv->inf limit

    def prim(w):        # Int ln(1+(b/w)^2) dw = w ln(1+b^2/w^2) + 2b arctan(w/b)
        return w * np.log1p((b / w) ** 2) + 2.0 * b * np.arctan(w / b)
    return t / (4.0 * np.pi) * (prim(w_uv) - prim(w_ir))


# ==================================================================== combine =
def chi_total(t, D1_GHz, D2_GHz, A, sequence="ramsey", w_ir=W_IR_DEFAULT,
              M=M_GRID):
    """chi_1 + chi_2 -- first and second order combined as independent rates.

    APPROXIMATION: the linear-quadratic cross term is dropped.  It vanishes at
    leading order for Gaussian dnu (odd moments vanish); ``chi_exact_combined``
    bounds the residual.
    """
    return (float(chi_first_order(t, D1_GHz, A, sequence, w_ir))
            + chi_second_order(t, D2_GHz, A, sequence, w_ir, M))


def chi_exact_combined_batch(ts, D1_GHz, D2_GHz, A, sequence="ramsey",
                             w_ir=W_IR_DEFAULT, M=M_GRID):
    """Exact linear+quadratic Gaussian characteristic function, vectorised.

    phi = a.x + x^T B x with a_i = dw * s_i dt, B = (d2w/2) diag(s) dt:
        <e^{i phi}> = det(I - 2 i Sigma B)^{-1/2}
                      exp(-(1/2) a^T (I - 2 i Sigma B)^{-1} Sigma a)
    chi = (1/2) Re ln det(I - i d2w K S) + (1/2) Re[a^T (I - i d2w K S)^{-1} K a]

    Unlike ``chi_total`` this keeps the linear-quadratic mixing.  The two agree
    to <0.1% wherever one order dominates, but the resolvent SUPPRESSES the
    first-order term where the two are comparable, so plain rate addition is
    conservative there (see the crossover band diagnostic in the P4 export).
    """
    ts = np.atleast_1d(np.asarray(ts, dtype=float))
    d1 = float(dw_dnu(D1_GHz))
    d2 = float(d2w_dnu2(D2_GHz))
    if d2 == 0.0 or A == 0.0:
        return np.asarray(chi_first_order(ts, D1_GHz, A, sequence, w_ir),
                          dtype=float)
    dt = ts / M
    s = _weight((np.arange(M) + 0.5) / M, 1.0, sequence)
    K = _kernel_batch(ts, M, A, w_ir)
    Mx = np.eye(M)[None] - 1j * (d2 * dt)[:, None, None] * (K * s[None, None, :])
    chi = 0.5 * np.real(np.linalg.slogdet(Mx)[1])
    if d1 != 0.0:
        rhs = (K @ s)[:, :, None]                       # (B, M, 1)
        v = np.linalg.solve(Mx, rhs)[:, :, 0]           # (B, M)
        chi = chi + 0.5 * (d1 * dt) ** 2 * np.real(v @ s)
    return chi


def chi_exact_combined(t, D1_GHz, D2_GHz, A, sequence="ramsey",
                       w_ir=W_IR_DEFAULT, M=M_GRID):
    """Scalar form of ``chi_exact_combined_batch``."""
    return float(chi_exact_combined_batch(float(t), D1_GHz, D2_GHz, A,
                                          sequence, w_ir, M)[0])


# ========================================================================= T2 =
def chi_grid(ts, D1_GHz, D2_GHz, A, sequence="ramsey", w_ir=W_IR_DEFAULT,
             M=M_GRID, exact=False):
    """chi on an array of times -- one LAPACK call for the whole array.

    ``exact=False`` (default, and what the P4 maps report) is the prescribed
    independent-rate sum chi_1 + chi_2; ``exact=True`` keeps the
    linear-quadratic mixing.
    """
    ts = np.atleast_1d(np.asarray(ts, dtype=float))
    if exact:
        return chi_exact_combined_batch(ts, D1_GHz, D2_GHz, A, sequence,
                                        w_ir, M)
    return (np.asarray(chi_first_order(ts, D1_GHz, A, sequence, w_ir))
            + chi_second_order_batch(ts, D2_GHz, A, sequence, w_ir, M))


N_SCAN = (48, 48)          # batched bracket-and-refine rounds


def T2(D1_GHz, D2_GHz, A, sequence="ramsey", w_ir=W_IR_DEFAULT, M=M_GRID,
       bracket=None, chi_fn=None, n_scan=N_SCAN, exact=False):
    """Solve chi(T2) = 1 for T2 in seconds.

    chi is monotone increasing in t, so the root is unique.  It is found by
    batched bracket-and-refine on ln t (``n_scan`` points per round, one
    stacked determinant per round) followed by linear interpolation; after two
    rounds the bracket is ~1e-2 wide in ln t and the interpolation error on
    this smooth function is < 1e-5 relative.  ``chi_fn`` (a scalar callable)
    switches to scipy.brentq instead and is used by the single-channel helpers.

    Returns NaN if T2 would exceed the acquisition window 2 pi / w_ir (see the
    T_MIN_S note): outside that window the 1/f model with this IR cutoff is not
    applicable and no number is reported.
    """
    bracket = bracket or (T_MIN_S, 2.0 * np.pi / w_ir)
    lo, hi = np.log(bracket[0]), np.log(bracket[1])
    if chi_fn is not None:
        f = lambda lt: chi_fn(np.exp(lt)) - 1.0        # noqa: E731
        if f(lo) > 0 or f(hi) < 0:
            return np.nan
        return float(np.exp(brentq(f, lo, hi, xtol=1e-6, rtol=1e-12)))
    for k, n in enumerate(n_scan):
        lts = np.linspace(lo, hi, n)
        c = chi_grid(np.exp(lts), D1_GHz, D2_GHz, A, sequence, w_ir, M, exact)
        if k == 0 and (c[0] > 1.0 or c[-1] < 1.0):
            return np.nan
        j = int(np.searchsorted(c, 1.0))
        j = min(max(j, 1), n - 1)
        lo, hi = lts[j - 1], lts[j]
        c0, c1 = c[j - 1], c[j]
    return float(np.exp(lo + (hi - lo) * (1.0 - c0) / (c1 - c0)))


def T2_first_order_only(D1_GHz, A, sequence="ramsey", w_ir=W_IR_DEFAULT):
    """T2 from the first-order channel alone (closed form, no determinant)."""
    return T2(D1_GHz, 0.0, A, sequence, w_ir)


def T2_second_order_only(D2_GHz, A, sequence="ramsey", w_ir=W_IR_DEFAULT,
                         M=M_GRID):
    """T2 from the second-order channel alone."""
    return T2(0.0, D2_GHz, A, sequence, w_ir, M)


def f_ir_sensitivity(D1_GHz, D2_GHz, A, sequence="ramsey",
                     f_ir_Hz=(1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3), M=M_GRID):
    """T2 vs the IR-cutoff convention -- the ln sensitivity, quantified."""
    return {f"{f:.0e}": T2(D1_GHz, D2_GHz, A, sequence, 2 * np.pi * f, M)
            for f in f_ir_Hz}


# ====================================================== quasistatic reference =
def quasistatic_first_order(D1_GHz, dnu_rms):
    """dome.Tphi_first_order, re-exported for the comparison table.

    # src: qubit/dome.py:Tphi_first_order  (1 / (2 pi |D1| dnu_rms))
    """
    return dome.Tphi_first_order(D1_GHz, dnu_rms)


def quasistatic_second_order(D2_GHz, dnu_rms):
    """# src: qubit/dome.py:Tphi_second_order  (1 / (pi |D2| dnu_rms^2))."""
    return dome.Tphi_second_order(D2_GHz, dnu_rms)


# =========================================================== grids and export =
NU_MIN, NU_MAX, NU_STEP = -2.85, -1.95, 0.025      # 37 points, hits -2.5/-2.4
A_DECADES = (-6.0, -2.0)
A_POINTS = 25                                       # >= 25, hits 1e-4 and 1e-3

REF_NU = (-2.4, -2.5)
REF_A = (1e-4, 1e-3)

OUTDIR = ROOT / "out" / "m6"
ROWDIR = OUTDIR / "p4_rows"


def nu_grid():
    n = int(round((NU_MAX - NU_MIN) / NU_STEP)) + 1
    return NU_MIN + NU_STEP * np.arange(n)


def A_grid():
    return np.logspace(A_DECADES[0], A_DECADES[1], A_POINTS)


def dome_derivatives(nu, EC_GHz):
    """(f01, D1, D2) at nu.  # src: qubit/dome.py (reads material.json:sc_dome)."""
    return (dome.f01_of(nu, EC_GHz), dome.D1(nu, EC_GHz), dome.D2(nu, EC_GHz))


ROW_KEYS = ("T2_ramsey", "T2_echo", "T2_first_only", "T2_second_only",
            "T2_ramsey_exact", "T2_echo_exact")


def _row(nu, EC_GHz, As, w_ir, M):
    f01, d1, d2 = dome_derivatives(nu, EC_GHz)
    out = dict(nu=nu, f01_GHz=f01, D1_GHz=d1, D2_GHz=d2,
               **{k: np.empty(len(As)) for k in ROW_KEYS})
    for j, A in enumerate(As):
        out["T2_ramsey"][j] = T2(d1, d2, A, "ramsey", w_ir, M)
        out["T2_echo"][j] = T2(d1, d2, A, "echo", w_ir, M)
        out["T2_first_only"][j] = (T2_first_order_only(d1, A, "ramsey", w_ir)
                                   if d1 != 0.0 else np.inf)
        out["T2_second_only"][j] = T2_second_order_only(d2, A, "ramsey", w_ir, M)
        out["T2_ramsey_exact"][j] = T2(d1, d2, A, "ramsey", w_ir, M, exact=True)
        out["T2_echo_exact"][j] = T2(d1, d2, A, "echo", w_ir, M, exact=True)
    return out


def build_maps(EC_GHz, w_ir=W_IR_DEFAULT, M=M_GRID, rowdir=ROWDIR, verbose=True):
    """T2*(nu, A) and T2echo(nu, A) with per-nu-row checkpoint/resume.

    Row blocks are written to ``out/m6/p4_rows/row_<i>.npz``; re-running skips
    completed rows.  Deterministic: no RNG anywhere in P4.
    """
    rowdir = pathlib.Path(rowdir)
    rowdir.mkdir(parents=True, exist_ok=True)
    nus, As = nu_grid(), A_grid()
    keys = ROW_KEYS
    rows = []
    for i, nu in enumerate(nus):
        fp = rowdir / f"row_{i:03d}.npz"
        if fp.exists():
            z = np.load(fp)
            if (z["n_A"] == len(As) and float(z["w_ir"]) == w_ir
                    and all(k in z.files for k in keys)):
                rows.append({k: z[k] for k in keys}
                            | {k: float(z[k]) for k in
                               ("nu", "f01_GHz", "D1_GHz", "D2_GHz")})
                continue
        r = _row(float(nu), EC_GHz, As, w_ir, M)
        np.savez(fp, n_A=len(As), w_ir=w_ir, M=M, **r)
        rows.append(r)
        if verbose:
            print(f"row {i:3d}/{len(nus)-1}  nu={nu:+.3f}  D1={r['D1_GHz']:+9.3f}"
                  f"  T2R(1e-4)={r['T2_ramsey'][12]*1e6:12.4g} us"
                  f"  T2E(1e-4)={r['T2_echo'][12]*1e6:12.4g} us", flush=True)
    maps = {k: np.array([r[k] for r in rows]) for k in keys}
    maps["gain_echo_over_ramsey"] = maps["T2_echo"] / maps["T2_ramsey"]
    # How wrong is the prescribed independent-rate addition?  (>1 means the
    # additive map is conservative -- it under-reports T2.)
    maps["additivity_error_ramsey"] = (maps["T2_ramsey_exact"]
                                       / maps["T2_ramsey"] - 1.0)
    return dict(nu=nus, A=As,
                f01_GHz=np.array([r["f01_GHz"] for r in rows]),
                D1_GHz=np.array([r["D1_GHz"] for r in rows]),
                D2_GHz=np.array([r["D2_GHz"] for r in rows]),
                w_ir=w_ir, M=M, **maps)


def reference_rows(EC_GHz, w_ir=W_IR_DEFAULT, M=M_GRID):
    """The four (nu, A) comparison rows: quasistatic vs Ramsey vs echo."""
    with open(ROOT / "qubit_targets.json", encoding="utf-8") as fh:
        tgt = json.load(fh)["gate_noise"]        # src: qubit_targets.json:gate_noise
    out = []
    for nu in REF_NU:
        f01, d1, d2 = dome_derivatives(nu, EC_GHz)
        for A in REF_A:
            key = f"{A:.0e}".replace("e-0", "e-0")
            qs_ref = (tgt["Tphi_apex_us"].get(f"{A:.0e}")
                      if abs(d1) < 1e-9 else tgt["Tphi_off_us"].get(f"{A:.0e}"))
            qs = (quasistatic_second_order(d2, A) if abs(d1) < 1e-9
                  else quasistatic_first_order(d1, A))
            tr = T2(d1, d2, A, "ramsey", w_ir, M)
            te = T2(d1, d2, A, "echo", w_ir, M)
            trx = T2(d1, d2, A, "ramsey", w_ir, M, exact=True)
            out.append(dict(
                T2_ramsey_exact_us=trx * 1e6,
                nu=nu, A=A, f01_GHz=f01, D1_GHz=d1, D2_GHz=d2,
                quasistatic_us=qs * 1e6,
                quasistatic_reference_us=qs_ref,
                T2_ramsey_us=tr * 1e6, T2_echo_us=te * 1e6,
                ramsey_over_quasistatic=tr / qs,
                echo_over_ramsey=te / tr,
                w_ir_T2_ramsey=w_ir * tr,
                dominant_order=("second" if abs(d1) < 1e-9 else "first"),
                _key=key))
    return out


def _round(x, sig=6):
    """Round to `sig` significant digits; non-finite entries become JSON null.

    NaN marks 'T2 exceeds the acquisition window 2 pi / w_ir', i.e. outside the
    validity of the 1/f model with this IR cutoff -- it is a flag, not a gap.
    """
    a = np.asarray(x, dtype=float)
    out = np.zeros_like(a)
    fin = np.isfinite(a)
    m = fin & (a != 0)
    scale = np.zeros_like(a)
    scale[m] = 10.0 ** (sig - 1 - np.floor(np.log10(np.abs(a[m]))))
    out[m] = np.round(a[m] * scale[m]) / scale[m]
    return [None if not f else float(v) for f, v in zip(fin, out)]


def _additivity_report(mp, tol=0.01):
    """Where does chi_1 + chi_2 stop being a good approximation?

    Measured as (T2_exact / T2_additive - 1) on the production grid.  The
    failure is driven by A, not by nu: it is invisible up to A ~ 1e-4 and only
    becomes large where a *modest* second-order share (chi_2/chi_1 ~ 0.1) is
    already enough for the resolvent (I - i d2w K S)^-1 to suppress the
    first-order term.  It sits near, but not at, the apex (D1 = 0 there makes
    the two identical).
    """
    e = mp["additivity_error_ramsey"]
    ok = np.isfinite(e)
    per_A, safe_A = {}, None
    for j, A in enumerate(mp["A"]):
        col = e[:, j][np.isfinite(e[:, j])]
        m = float(np.max(np.abs(col))) if col.size else float("nan")
        per_A[f"{A:.3e}"] = m
        if m < tol:
            safe_A = float(A)
    i, j = np.unravel_index(np.nanargmax(np.where(ok, np.abs(e), -np.inf)),
                            e.shape)
    return dict(
        definition="T2_exact / T2_additive - 1 (positive = additive is "
                   "conservative, i.e. under-reports T2)",
        fraction_within_1pct=float((np.abs(e[ok]) < tol).mean()),
        fraction_within_5pct=float((np.abs(e[ok]) < 0.05).mean()),
        max_abs_error_all=float(np.nanmax(np.abs(e[ok]))),
        max_abs_error_by_A=per_A,
        largest_A_with_max_error_below_1pct=safe_A,
        worst_point=dict(nu=float(mp["nu"][i]), A=float(mp["A"][j]),
                         T2_additive_us=float(mp["T2_ramsey"][i, j] * 1e6),
                         T2_exact_us=float(mp["T2_ramsey_exact"][i, j] * 1e6)),
        note="Independent-rate addition is the prescribed and reported map. It "
             "is exact at the apex (D1 = 0) and far off-apex (chi_2 -> 0), and "
             "conservative in between. Use the T2_*_exact arrays in the npz for "
             "A above the threshold quoted here.")


def export(EC_GHz, w_ir=W_IR_DEFAULT, M=M_GRID, outdir=OUTDIR, verbose=True):
    """Run the maps and write out/m6/p4_t2_maps.{npz,json}."""
    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    mp = build_maps(EC_GHz, w_ir, M, outdir / "p4_rows", verbose)
    np.savez_compressed(outdir / "p4_t2_maps.npz", **mp)
    refs = reference_rows(EC_GHz, w_ir, M)
    doc = dict(
        deliverable="M6 P4 -- 1/f gate noise via Ramsey/echo filter functions",
        theory=("Ithier et al., Phys. Rev. B 72, 134519 (2005): first-order "
                "Gaussian filter-function integrals (Ramsey log / echo ln2) and "
                "the exact quadratic-coupling determinant at second order."),
        params=dict(
            noise_convention="S_nu(w) = A^2/|w| two-sided angular; "
                             "<dnu^2> = 2 A^2 ln(w_uv/w_ir); "
                             "one-sided ordinary-frequency equivalent 2 A^2 / f",
            A_units="filling factor nu (electrons per moire cell), free parameter",
            w_ir_rad_per_s=w_ir, f_ir_Hz=w_ir / (2 * np.pi),
            w_ir_convention="w_ir = 2 pi / T_total_acquisition; default 1 Hz",
            w_uv_convention="Nyquist of the exact-determinant time grid, "
                            "w_uv = pi M / t (second order only; <0.1% effect)",
            M_time_grid=M,
            T2_definition="chi(T2) = 1, same definition for Ramsey and echo",
            ramsey_O1_constant=RAMSEY_O1,
            echo_first_order_factor=LN2,
            crossover="maps_s uses chi_total = chi_1 + chi_2 (independent "
                      "rates, as prescribed; the linear-quadratic cross term "
                      "is dropped). The exact linear+quadratic Gaussian result "
                      "is carried alongside in the npz as T2_*_exact; see "
                      "additivity_diagnostic below.",
            EC_GHz=EC_GHz,
            deterministic="no RNG in P4",
        ),
        literature_note=LITERATURE_NOTE,
        claim_discipline=("Design study. A is free; every T2 is parametric in A. "
                          "No absolute TBG coherence claim is made."),
        grids=dict(nu=_round(mp["nu"], 6), A=_round(mp["A"], 6)),
        dome=dict(f01_GHz=_round(mp["f01_GHz"]), D1_GHz=_round(mp["D1_GHz"]),
                  D2_GHz=_round(mp["D2_GHz"])),
        maps_s=dict(
            T2_ramsey=[_round(r) for r in mp["T2_ramsey"]],
            T2_echo=[_round(r) for r in mp["T2_echo"]],
            gain_echo_over_ramsey=[_round(r) for r in
                                   mp["gain_echo_over_ramsey"]],
        ),
        reference_rows=[{k: v for k, v in r.items() if not k.startswith("_")}
                        for r in refs],
        additivity_diagnostic=_additivity_report(mp),
        npz="out/m6/p4_t2_maps.npz (dense maps incl. per-order breakdown)",
    )
    fp = outdir / "p4_t2_maps.json"
    with open(fp, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1)
    if verbose:
        print(f"wrote {fp} ({fp.stat().st_size/1024:.1f} kB)")
    return mp, doc


def main():
    import qubit
    EC = qubit.design_point()["EC_GHz"]     # src: recomputed design scan (AT-1)
    export(EC)


if __name__ == "__main__":
    main()
