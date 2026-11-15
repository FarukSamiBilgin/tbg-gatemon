"""P5 -- two-cavity chip feasibility, geometry concept and control-line budget.

M6_BRIEF.md Sec. 4, deliverable P5.  Three things happen here, in order:

1.  The AT-5 readout-window scan is REDONE with the P2 voltage division in the
    loop and with the shared C_Sigma budget enforced, i.e. the P2 mandate
    ("the readout-window scan must be re-run with beta in the loop before P5
    commits to a geometry", paper/decisions.md 2026-07-27 M6 P2, impact 2).
2.  The readout defect needs its own photonic-crystal region.  Its lattice
    constant follows from Maxwell scale invariance of photonics.json, and the
    two lattices' gap placements are computed against each other.
3.  The control line through the readout port is budgeted: Rabi rate per input
    power, and the Lorentzian leakage to the 145 GHz drive mode.

NOTHING in qubit/cavity.py, qubit/inputs.py, qubit/coupling.py is modified;
this module composes them.  Deterministic, closed form, no RNG.

=============================================================================
A.  THE COUPLING USED HERE  (P2, pinned)
=============================================================================

    g/h = 2 * beta_r * e * E_zpf(f_r, V_r) * d_eff_r * n01 / h            (P2)

    beta_r = C_c,readout / C_Sigma                       (Koch divider)
    C_c,X  = kappa_geo * eps0 * eps_eff * d_eff,X        (self-cap branch)

with the SHARED budget of the single qubit island (P2 Sec. 1):

    C_c,drive + C_c,readout  <=  C_Sigma - C_J = 11.99971 fF             (BUD)

Both electrodes are patterned in one process step, so when (BUD) is violated
they are rescaled by the SAME factor s = cap/(C_c,d + C_c,r) -- P2's
minimal-assumption allocation (coupling.two_cavity_allocation).  Under that
rescale the parameters kappa_geo and eps_eff CANCEL out of the ratio, and

    beta_r|saturated = beta_max * d_eff_r / (d_eff_drive + d_eff_r)

becomes a pure geometric number: the oversubscribed corner of the design is
parameter-free.  That is the one robust statement available here.

=============================================================================
B.  THE MEASURED LINEWIDTH FLOOR  (new P5 finding)
=============================================================================

The scan grid inherited from the brief treats kappa_r as a free knob at
1 and 3 MHz.  It is not free: the readout defect is the SAME L3 design scaled
by Maxwell invariance, so its internal quality factor is the measured one,

    Q_i = 14611.7   (photonics.json:ladder.Q_i_anchor.value)

Q_rad is purely geometric -> scale invariant; Q_mat = 1/tan_delta is
frequency independent to the extent tan_delta is.  Hence

    kappa_r  =  f_r (1/Q_i + 1/Q_w)  >=  kappa_int = f_r / Q_i

and at f_r = 44.6-56.6 GHz that floor is 3.05-3.87 MHz.  Both grid values
(1 and 3 MHz) sit AT or BELOW the floor.  Every row is therefore tagged
``kappa_achievable``.

=============================================================================
C.  THE ANALYTIC FEASIBILITY CEILING ON kappa_r  (CORRECTED 2026-07-27)
=============================================================================

Eliminating g between the two binding criteria (chi/kappa >= 1 and
T1_Purcell >= T1_min) gives a g-independent, d_eff-independent ceiling.  With
the CORRECTED dispersive shift (Delta = f01 - f_r, cavity.chi_dispersive_koch):

    |chi| = g^2 |alpha| / |Delta (Delta + alpha)|   >= kappa
    T1    = Delta^2 / (g^2 2 pi kappa)              >= T1_min
    ==>   kappa^2  <=  |alpha| |Delta| / ( |Delta + alpha| * 2 pi T1_min )  (CEIL')

with Delta = f01 - f_r signed (positive BELOW the qubit).  Two structural
consequences of the correction, both of which invert the previous reading:

 1. MONOTONICITY INVERTS above the qubit.  For f_r = f01 + D the ceiling is
    |alpha| D / ((D + |alpha|) 2 pi T1_min): INCREASING in D, saturating at
    sqrt(|alpha| / (2 pi T1_min)) = 2.9428 MHz as D -> infinity.  The stored
    (pre-correction) form decreased in D and was widest at the smallest
    detuning; that reading is withdrawn.
 2. SIDE SWAP.  The old formula, with (D + alpha) = (D - |alpha|), is the
    ceiling BELOW the qubit and beyond the 1-2 line (f_r < f12).  The sign
    error had computed the below-qubit ceiling and booked it above the qubit.

The saturation value 2.9428 MHz lies below kappa_int(f_r) for every f_r in the
mandated Delta_r = 4-16 GHz window (min 3.0521 MHz at 44.60 GHz), so the
ENTIRE f_r > f01 readout window is empty -- see `above_f01_emptiness_proof`.
The design therefore moves BELOW f01, where (CEIL') is much wider because
|Delta + alpha| = |f12 - f_r| can be small.

The readout port efficiency

    eta = kappa_ext/kappa = 1 - kappa_int/kappa

is bounded above by 1 - kappa_int/kappa_max, where kappa_max is the ceiling
after the two dispersivity caps of section E are also applied.

=============================================================================
D.  FREE PARAMETERS (design study -- ranges, no invented precision)
=============================================================================

kappa_geo   2.0   (1.0-4.0)     P2, pickup shape factor
eps_eff     6.34  (6.34-11.68)  P2, air/Si average .. fully embedded
T1_min      30 us               M6_BRIEF.md Sec. 2 feasibility criterion
d_eff_r     50/100/200 um       mandated coarse grid; 5-200 um corrected fine grid
Delta_r     4-16 GHz            mandated (above f01; now provably empty)
f_r         33.089-42.811 GHz   gap-protection window, corrected fine scan
Q_i         14611.7 measured; optimistic ceiling 1/(1/Q_rad + 1/Q_mat)
sigma,b     ladder fit, 2 points (N=3,4), R2 trivially 1.0 -> DESIGN ESTIMATE

=============================================================================
E.  THE chi SIGN-CONVENTION CORRECTION  (2026-07-27, reference-inherited)
=============================================================================

`out/verify/chi_convention_ruling.md` adjudicated a defect that this module
inherited from the frozen contract: `cavity.chi_dispersive` was called
everywhere with Delta = f_r - f01, while the Koch derivation requires
Delta = f01 - f_r.  Consequences carried here:

  * every PRODUCTION chi is now `cavity.chi_dispersive_koch(g, f01, f_r, alpha)`
    and is cross-checked at the same point against the exact two-pole truth
    standard `cavity.chi_two_pole(g, f01, f12, f_r, r)`, r = (n12/n01)^2;
  * the stored recommended point (f_r = 44.5966 GHz, chi/kappa = 1.0092)
    FAILS: its true chi/kappa is 0.424 (Koch) / 0.466 (two-pole).  It is
    retained verbatim in the `reference_convention_scan` export block, never
    deleted, mirroring the AT-6 small-frequency precedent;
  * a NEW constraint is imposed, because the corrected optimum sits close to
    the 1-2 line where the design's 0-1 dispersivity test is not the binding
    one:

        (g12 / Delta_12)^2 <= 0.01,   g12 = g n12/n01,  Delta_12 = f_r - f12  (C7)

    The ruling's own candidate optimum f_r = 39.2589 GHz violates it
    (0.01858); with C7 enforced that whole neighbourhood is not merely
    marginal but INFEASIBLE (its kappa ceiling, 1.881 MHz, falls below
    kappa_int = 2.687 MHz there).
  * the recommendation is additionally screened on the truth standard,
    |chi_two_pole|/kappa >= 1 (C8, module flag ``TWO_POLE_SCREEN``), so no
    exported operating point depends on the -9 % Koch residual.
"""
import json

import numpy as np

from . import cavity as _cav
from . import coupling as _cpl
from .inputs import (D_EFF_DRIVE_M, E, EJ_GHZ, EPS0, EPS_C, F_CAV_GHZ, H, HBAR,
                     N_CHARGE, PHOTONICS, Q_I, ROOT, V_MODE_M3)

SEED = 0
# No stochastic step anywhere in P5 (closed-form algebra + deterministic grid
# sweeps).  The seed is logged for the repo's determinism policy only.

# --------------------------------------------------------- measured inputs ---
A_UM = PHOTONICS["geometry"]["a_um"]
# src: photonics.json:geometry.a_um = 586.74 um (cavity-line rescale)
A_UM_PRECISE = PHOTONICS["geometry"]["a_um_precise"]
# src: photonics.json:geometry.a_um_precise = 586.735322 um
R_OVER_A = PHOTONICS["geometry"]["r_over_a"]        # src: photonics.json:geometry.r_over_a
T_OVER_A = PHOTONICS["geometry"]["t_over_a"]        # src: photonics.json:geometry.t_over_a
GAP_GHZ = tuple(PHOTONICS["modes"]["cavity_line_gap_GHz"])
# src: photonics.json:modes.cavity_line_gap_GHz = [137.5, 177.9] at a = 586.74
NU0 = PHOTONICS["modes"]["nu0_a_over_lambda_precise"]
# src: photonics.json:modes.nu0_a_over_lambda_precise = 0.283785063521672
V_MODE_A3 = PHOTONICS["modes"]["V_mode_a3"]         # src: photonics.json:modes.V_mode_a3
OUTER_ROWS = PHOTONICS["geometry"]["cavity_detail"]["outer_mirror_rows"]
# src: photonics.json:geometry.cavity_detail.outer_mirror_rows = 8
CHIP_SCALE_RULE = PHOTONICS["geometry"]["chip_scale"]
# src: photonics.json:geometry.chip_scale (">=15x15 periods (~9x9 mm)")
CHIP_PERIODS_MIN = 15                # the numeral inside that design rule

LADDER_SIGMA = PHOTONICS["ladder"]["exponential_fit"]["sigma"]
LADDER_B = PHOTONICS["ladder"]["exponential_fit"]["b"]
LADDER_USABLE_N = tuple(PHOTONICS["ladder"]["exponential_fit"]["usable_N"])
# src: photonics.json:ladder.exponential_fit {sigma = 1.43925, b = 5.31077,
#      usable_N = [3, 4], n_points = 2, trivial_r2_warning = true}
Q_RAD_BAND = tuple(PHOTONICS["modes"]["Q_rad_intrinsic"])
# src: photonics.json:modes.Q_rad_intrinsic = [27000, 30000]
Q_MAT = PHOTONICS["modes"]["Q_mat"]                 # src: photonics.json:modes.Q_mat = 52075
TAN_DELTA = PHOTONICS["geometry"]["tan_delta"]      # src: photonics.json:geometry.tan_delta
C_LIGHT = 299792458.0                               # exact SI definition

# ------------------------------------------------------- scan design choices --
DETUNINGS_GHZ = _cav.DETUNINGS_GHZ                   # 4..16 GHz, step 1 (mandated)
D_EFF_READOUT_M = (50e-6, 100e-6, 200e-6)            # mandated coarse grid
KAPPA_READOUT_MHZ = (1.0, 3.0)                       # mandated coarse grid
CHI_OVER_KAPPA_MIN = _cav.CHI_OVER_KAPPA_MIN         # 1.0
T1_PURCELL_MIN_S = _cav.T1_PURCELL_MIN_S             # 30 us
G_OVER_DELTA_SQ_MAX = _cav.G_OVER_DELTA_SQ_MAX       # 0.01
T1_MARGIN_TARGET = 1.2                               # P5 recommendation criterion

# --------------------------------------- chi-convention correction (Sec. E) ---
G12_OVER_DELTA12_SQ_MAX = 0.01
# (C7) 1-2-line dispersivity cap, same numerical value the design already
# imposes on the 0-1 line.  src: out/verify/chi_convention_ruling.md Sec. 3(d).

TWO_POLE_SCREEN = True
# (C8) additionally require |chi_two_pole|/kappa >= 1 at the recommended point,
# so nothing exported rests on the -9 % residual of the single Koch formula.
# One-line reversible: set False to recover the Koch-only optimum, which this
# module also exports under `koch_only_optimum`.

POLE_GUARD_GHZ = 0.05
# Grid points closer than this to either transmon line (f01, f12) are dropped
# from the corrected fine scan.  Both bands are provably infeasible -- as
# f_r -> f01 the 0-1 dispersivity cap forces g -> 0, as f_r -> f12 the C7 cap
# does -- so the guard removes only singular arithmetic, never candidates.
# `gap_window_envelope` evaluates the analytic ceiling inside the guard bands
# to make that verifiable.

# fine grid used only to locate the recommended point once the coarse grid
# turns out to be empty under the measured linewidth floor.
# DELTA_FINE_GHZ/D_EFF_FINE_UM/KAPPA_FINE_MHZ are the REFERENCE-CONVENTION grids
# and are retained unchanged so `reference_convention_scan` reproduces the
# superseded artifact bit-for-bit.
DELTA_FINE_GHZ = np.arange(4.0, 16.0 + 1e-9, 0.25)
D_EFF_FINE_UM = np.arange(40.0, 200.0 + 1e-9, 0.5)
KAPPA_FINE_MHZ = np.arange(2.0, 12.0 + 1e-9, 0.05)

# corrected fine grid: f_r swept directly (not Delta_r), over the
# gap-protection window; d_eff extended DOWN to 5 um because the corrected
# optimum needs a much smaller electrode than the superseded one.
F_R_FINE_STEP_GHZ = 0.01
D_EFF_CORRECTED_FINE_UM = np.arange(5.0, 200.0 + 1e-9, 0.5)
KAPPA_CORRECTED_FINE_MHZ = np.arange(2.0, 12.0 + 1e-9, 0.05)
FINE_BLOCK = 10          # f_r values per vectorised block (memory bound only)

# corrected coarse grid inside the gap-protection window (dumped row by row).
# Deliberately coarse: the scientifically load-bearing object at this level is
# `gap_window_envelope`, which is the EXACT closed-form feasibility map at
# 0.05 GHz resolution; these rows only exhibit the beta/budget machinery.
F_R_COARSE_STEP_GHZ = 1.0
D_EFF_COARSE_M = (30e-6, 60e-6, 100e-6, 200e-6)
KAPPA_COARSE_MHZ = (2.0, 3.0, 4.0, 6.0)


# ============================================ 0. chi convention + n12/n01 =====
def ladder_elements(EJ_GHz=EJ_GHZ, EC_GHz=None, N=N_CHARGE):
    """(n01, n12, n12/n01, r = (n12/n01)^2) at the design point.

    Recomputed from `cavity.charge_matrix_elements` (i.e. from
    `spectrum.hamiltonian`), NOT read from an artifact; the value is
    cross-checked against out/m6/p6_gates.json:params.ladder_ratio in
    `build_export`.
    """
    if EC_GHz is None:
        import qubit as _q
        EC_GHz = _q.design_point()["EC_GHz"]
    n01, n12 = _cav.charge_matrix_elements(EJ_GHz, EC_GHz, N)
    ratio = n12 / n01
    return n01, n12, ratio, ratio ** 2


_LADDER_CACHE = {}


def _ladder(n_ratio=None):
    """(n12/n01, r) -- cached design-point value unless overridden."""
    if n_ratio is not None:
        return float(n_ratio), float(n_ratio) ** 2
    if "v" not in _LADDER_CACHE:
        _, _, ratio, r = ladder_elements()
        _LADDER_CACHE["v"] = (ratio, r)
    return _LADDER_CACHE["v"]


def gap_protection_window_GHz(f01_GHz, gap_GHz=GAP_GHZ, f_cav_GHz=F_CAV_GHZ):
    """[f_r_lo, f_r_hi] over which the READOUT lattice's TE gap covers f01.

    The readout lattice is the measured L3 lattice Maxwell-rescaled to f_r, so
    its gap edges are gap_GHz * f_r / f_cav (`scaled_gap_GHz`).  Requiring
    gap_lo(f_r) <= f01 <= gap_hi(f_r) inverts to

        f01 * f_cav / gap_hi  <=  f_r  <=  f01 * f_cav / gap_lo .

    Inside this window the readout mirror region is a photonic bandgap AT THE
    QUBIT FREQUENCY, which is what the missing-T1-channel radiation finding
    asks for; outside it the qubit's in-plane radiative channel is unprotected.
    Recomputed here from photonics.json; cross-checked in `build_export`
    against out/m6/explore/missing_t1_channels.json:verdict.question_c_retune.
    """
    return (f01_GHz * f_cav_GHz / gap_GHz[1], f01_GHz * f_cav_GHz / gap_GHz[0])


# =========================================================== 1. readout scan ==
def kappa_intrinsic_MHz(f_r_GHz, Q_i=Q_I):
    """Internal-loss linewidth floor of the scaled L3 defect, in MHz."""
    return f_r_GHz * 1e3 / Q_i


def q_i_ceiling(Q_rad=None, Q_mat=Q_MAT):
    """Optimistic Q_i: vertical-radiation ceiling in parallel with Q_mat."""
    if Q_rad is None:
        Q_rad = 0.5 * (Q_RAD_BAND[0] + Q_RAD_BAND[1])
    return 1.0 / (1.0 / Q_rad + 1.0 / Q_mat)


def kappa_ceiling_MHz(Delta_r_GHz, alpha_GHz, T1_min_s=T1_PURCELL_MIN_S):
    """(CEIL'): the g-independent upper bound on kappa_r, in MHz.  CORRECTED.

        kappa^2 <= |alpha| |Delta| / (|Delta + alpha| 2 pi T1_min),
        Delta = f01 - f_r = -Delta_r        (all frequencies in Hz)

    Argument stays Delta_r = f_r - f01 (signed) for continuity with the scan
    rows; the sign flip to the Koch detuning happens here, once.

    Above f01 (Delta_r > 0) this reads |alpha| D / ((D + |alpha|) 2 pi T1_min):
    INCREASING in D, supremum sqrt(|alpha|/(2 pi T1_min)) = 2.9428 MHz.  The
    superseded form had (D - |alpha|) and decreased in D -- see module Sec. C.

    src: out/verify/chi_convention_ruling.md Sec. 3(c).
    """
    Dl = -np.asarray(Delta_r_GHz, dtype=float) * 1e9        # f01 - f_r
    a = alpha_GHz * 1e9
    return np.sqrt(abs(a) * np.abs(Dl)
                   / (np.abs(Dl + a) * 2 * np.pi * T1_min_s)) / 1e6


def kappa_ceiling_asymptote_MHz(alpha_GHz, T1_min_s=T1_PURCELL_MIN_S):
    """sup over D>0 of (CEIL') above f01 = sqrt(|alpha|/(2 pi T1_min)), MHz."""
    return float(np.sqrt(abs(alpha_GHz) * 1e9 / (2 * np.pi * T1_min_s)) / 1e6)


def _two_pole_kernel(f01_GHz, f12_GHz, f_r_GHz, r):
    """K2 with chi_two_pole = g^2 K2, in 1/GHz."""
    return 1.0 / (f01_GHz - f_r_GHz) - 0.5 * r / (f12_GHz - f_r_GHz)


def kappa_ceiling_two_pole_MHz(Delta_r_GHz, alpha_GHz, r,
                               T1_min_s=T1_PURCELL_MIN_S, f01_GHz=None):
    """(CEIL') rebuilt on the exact two-pole chi instead of the Koch single.

        |chi| = g^2 |K2| >= kappa,   T1 = Delta^2/(g^2 2 pi kappa) >= T1_min
        ==>  kappa^2 <= |K2| Delta^2 / (2 pi T1_min)

    Only the ratio Delta_r/alpha matters, so f01 is needed merely to place the
    two poles; any f01 gives the same answer at fixed Delta_r.
    """
    if f01_GHz is None:
        f01_GHz = 0.0
    Dr = np.asarray(Delta_r_GHz, dtype=float)
    f_r = f01_GHz + Dr
    K2 = _two_pole_kernel(f01_GHz, f01_GHz + alpha_GHz, f_r, r)   # 1/GHz
    Dl = -Dr                                                      # f01 - f_r
    return np.sqrt(np.abs(K2) * 1e-9 * (Dl * 1e9) ** 2
                   / (2 * np.pi * T1_min_s)) / 1e6


def _beta_readout(d_eff_r_m, kappa_geo=_cpl.KAPPA_GEO, eps_eff=_cpl.EPS_EFF,
                  d_eff_d_m=D_EFF_DRIVE_M):
    """(beta_r, beta_d, C_c_r, C_c_d, scale, oversubscription) under (BUD)."""
    cd = _cpl.c_c_selfcap(d_eff_d_m, kappa_geo, eps_eff)
    cr = _cpl.c_c_selfcap(np.asarray(d_eff_r_m, dtype=float), kappa_geo, eps_eff)
    cap = _cpl.c_c_cap()
    over = (cd + cr) / cap
    s = np.minimum(1.0, 1.0 / over)
    return (cr * s / _cpl.C_SIGMA_F, cd * s / _cpl.C_SIGMA_F,
            cr * s, cd * s, s, over)


def _evaluate(f_r, dE, K, n01, f01_GHz, alpha_GHz, chi_mode, n_ratio,
              kappa_geo, eps_eff, Q_i):
    """Vectorised core of every scan.  All inputs flat float arrays.

    ``chi_mode``:
      * ``"koch"``      -- production: cavity.chi_dispersive_koch, Delta = f01 - f_r
      * ``"reference"`` -- superseded: cavity.chi_dispersive with the inverted
                           detuning Delta = f_r - f01, retained ONLY so the
                           `reference_convention_scan` block reproduces the
                           pre-correction artifact bit-for-bit.
    The exact two-pole chi is computed in BOTH modes as a cross-check.
    """
    ratio, r = _ladder(n_ratio)
    f12 = f01_GHz + alpha_GHz
    Delta_r = f_r - f01_GHz                       # signed, f_r - f01
    Delta_l = f01_GHz - f_r                       # Koch detuning
    Delta_12 = f_r - f12

    V_r = V_MODE_M3 * (F_CAV_GHZ / f_r) ** 3
    Ez = _cav.E_zpf(f_r, V_r, EPS_C)
    beta_r, beta_d, cc_r, cc_d, s, over = _beta_readout(dE, kappa_geo, eps_eff)
    g = 2.0 * beta_r * E * Ez * dE * n01 / H / 1e9              # GHz

    if chi_mode == "koch":
        chi = _cav.chi_dispersive_koch(g, f01_GHz, f_r, alpha_GHz)
    elif chi_mode == "reference":
        chi = _cav.chi_dispersive(g, Delta_r, alpha_GHz)
    else:
        raise ValueError("chi_mode must be 'koch' or 'reference'")
    chi2 = _cav.chi_two_pole(g, f01_GHz, f12, f_r, r)           # GHz

    kap = K * 1e-3                                              # GHz
    gd2 = (g / Delta_l) ** 2
    g12d2 = (ratio * g / Delta_12) ** 2
    T1 = 1.0 / (gd2 * 2 * np.pi * kap * 1e9)                    # s
    kint = kappa_intrinsic_MHz(f_r, Q_i)
    return dict(f_r=f_r, Delta_r=Delta_r, Delta_l=Delta_l, Delta_12=Delta_12,
                dE=dE, K=K, kap=kap, g=g, chi=chi, chi2=chi2, gd2=gd2,
                g12d2=g12d2, T1=T1, kint=kint, beta_r=beta_r, beta_d=beta_d,
                cc_r=cc_r, cc_d=cc_d, s=s, over=over, ratio=ratio, r=r,
                m_chi=np.abs(chi) / kap, m_chi2=np.abs(chi2) / kap,
                m_T1=T1 / T1_PURCELL_MIN_S, eta=1.0 - kint / K)


def _masks(v, use_g12, use_two_pole, t1_margin=1.0):
    """Per-constraint boolean masks over an `_evaluate` result."""
    m = dict(
        chi=v["m_chi"] >= CHI_OVER_KAPPA_MIN,
        T1=v["m_T1"] >= t1_margin,
        disp=v["gd2"] <= G_OVER_DELTA_SQ_MAX,
        kappa=v["K"] >= v["kint"],
        disp12=(v["g12d2"] <= G12_OVER_DELTA12_SQ_MAX if use_g12
                else np.ones_like(v["gd2"], dtype=bool)),
        two_pole=(v["m_chi2"] >= CHI_OVER_KAPPA_MIN if use_two_pole
                  else np.ones_like(v["gd2"], dtype=bool)),
    )
    m["all"] = (m["chi"] & m["T1"] & m["disp"] & m["disp12"] & m["two_pole"])
    return m


def scan_rows(n01, f01_GHz, alpha_GHz, detunings_GHz=DETUNINGS_GHZ,
              d_effs_m=D_EFF_READOUT_M, kappas_MHz=KAPPA_READOUT_MHZ,
              kappa_geo=_cpl.KAPPA_GEO, eps_eff=_cpl.EPS_EFF, Q_i=Q_I,
              chi_mode="koch", n_ratio=None, use_g12=True,
              use_two_pole=None, f_r_GHz=None):
    """Vectorised readout-window scan with beta and the shared budget in.

    Returns a list of dicts, one per grid point, feasible or not; the caller
    filters.  chi is ALWAYS produced by a `qubit.cavity` function, so the scan
    can never drift from the validated formulas.

    Pass ``f_r_GHz`` instead of ``detunings_GHz`` to sweep the cavity frequency
    directly (used for the below-f01 gap-protection window, where f_r rather
    than a positive detuning is the natural coordinate).

    ``use_two_pole`` defaults to the module flag TWO_POLE_SCREEN so a row's
    ``feasible`` flag means the same thing as the recommendation's feasibility;
    the reference-convention call passes False explicitly.
    """
    if use_two_pole is None:
        use_two_pole = TWO_POLE_SCREEN
    if f_r_GHz is None:
        axis0 = f01_GHz + np.asarray(detunings_GHz, float)
    else:
        axis0 = np.asarray(f_r_GHz, float)
    F, dE, K = np.meshgrid(axis0, np.asarray(d_effs_m, float),
                           np.asarray(kappas_MHz, float), indexing="ij")
    v = _evaluate(F.ravel(), dE.ravel(), K.ravel(), n01, f01_GHz, alpha_GHz,
                  chi_mode, n_ratio, kappa_geo, eps_eff, Q_i)
    m = _masks(v, use_g12, use_two_pole)

    rows = []
    for i in range(v["f_r"].size):
        rows.append(dict(
            Delta_r_GHz=float(v["Delta_r"][i]), f_r_GHz=float(v["f_r"][i]),
            d_eff_um=float(v["dE"][i] * 1e6), kappa_MHz=float(v["K"][i]),
            beta_readout=float(v["beta_r"][i]), beta_drive=float(v["beta_d"][i]),
            correction_factor=float(2.0 * v["beta_r"][i]),
            C_c_readout_fF=float(v["cc_r"][i] * 1e15),
            C_c_drive_fF=float(v["cc_d"][i] * 1e15),
            C_c_sum_fF=float((v["cc_r"][i] + v["cc_d"][i]) * 1e15),
            budget_scale=float(v["s"][i]), oversubscription=float(v["over"][i]),
            kappa_geo_effective=float(kappa_geo * v["s"][i]),
            kappa_geo_effective_in_range=bool(
                _cpl.KAPPA_GEO_RANGE[0] <= kappa_geo * v["s"][i]
                <= _cpl.KAPPA_GEO_RANGE[1]),
            g_MHz=float(v["g"][i] * 1e3), chi_MHz=float(v["chi"][i] * 1e3),
            chi_two_pole_MHz=float(v["chi2"][i] * 1e3),
            chi_over_kappa=float(v["m_chi"][i]),
            chi_over_kappa_two_pole=float(v["m_chi2"][i]),
            T1_Purcell_us=float(v["T1"][i] * 1e6),
            g_over_Delta_sq=float(v["gd2"][i]),
            g12_over_Delta12_sq=float(v["g12d2"][i]),
            Delta_12_GHz=float(v["Delta_12"][i]),
            kappa_intrinsic_MHz=float(v["kint"][i]),
            kappa_achievable=bool(m["kappa"][i]),
            pass_chi_over_kappa=bool(m["chi"][i]),
            pass_T1_Purcell=bool(m["T1"][i]),
            pass_dispersive=bool(m["disp"][i]),
            pass_dispersive_12=bool(m["disp12"][i]),
            pass_two_pole=bool(v["m_chi2"][i] >= CHI_OVER_KAPPA_MIN),
            feasible=bool(m["all"][i]),
            feasible_achievable=bool(m["all"][i] and m["kappa"][i]),
            margin_chi_over_kappa=float(v["m_chi"][i] / CHI_OVER_KAPPA_MIN),
            margin_T1=float(v["m_T1"][i]),
            eta_readout=float(max(0.0, 1.0 - v["kint"][i] / v["K"][i])),
        ))
    return rows


def scan_beta_bracket(n01, f01_GHz, alpha_GHz, **kw):
    """Feasible-set size at the two corners of the (kappa_geo, eps_eff) box."""
    out = {}
    for tag, (kg, ee) in (
            ("lo", (_cpl.KAPPA_GEO_RANGE[0], _cpl.EPS_EFF_RANGE[0])),
            ("nominal", (_cpl.KAPPA_GEO, _cpl.EPS_EFF)),
            ("hi", (_cpl.KAPPA_GEO_RANGE[1], _cpl.EPS_EFF_RANGE[1]))):
        rows = scan_rows(n01, f01_GHz, alpha_GHz, kappa_geo=kg, eps_eff=ee, **kw)
        out[tag] = dict(
            kappa_geo=kg, eps_eff=ee,
            n_feasible=sum(r["feasible"] for r in rows),
            n_feasible_achievable=sum(r["feasible_achievable"] for r in rows),
            n_points=len(rows))
    return out


def fine_scan(n01, f01_GHz, alpha_GHz, Q_i=Q_I,
              deltas=DELTA_FINE_GHZ, d_effs_um=D_EFF_FINE_UM,
              kappas_MHz=KAPPA_FINE_MHZ, kappa_geo=_cpl.KAPPA_GEO,
              eps_eff=_cpl.EPS_EFF, t1_margin=T1_MARGIN_TARGET,
              chi_mode="koch", n_ratio=None, use_g12=True,
              use_two_pole=None, f_r_GHz=None, block=FINE_BLOCK):
    """Locate the recommended point on a fine grid under the ACHIEVABLE kappa.

    Objective (stated, not tuned, and UNCHANGED by the chi correction): among
    points that are feasible, achievable (kappa >= kappa_int) and carry a
    Purcell margin >= ``t1_margin``, maximise the readout port efficiency
    eta = 1 - kappa_int/kappa; tie-break on the weaker of the two feasibility
    margins.

    ``use_two_pole`` defaults to the module flag TWO_POLE_SCREEN (C8).  Pass
    ``f_r_GHz`` to sweep the cavity frequency directly instead of a positive
    detuning; the sweep is blocked over that axis so the flattened meshgrid
    never exceeds ``block * len(d_eff) * len(kappa)`` points at once.
    """
    if use_two_pole is None:
        use_two_pole = TWO_POLE_SCREEN
    if f_r_GHz is None:
        axis0 = f01_GHz + np.asarray(deltas, float)
        axis_label, axis_vals = "Delta_GHz", np.asarray(deltas, float)
    else:
        axis0 = np.asarray(f_r_GHz, float)
        axis_label, axis_vals = "f_r_GHz", axis0
    dE_ax = np.asarray(d_effs_um, float) * 1e-6
    K_ax = np.asarray(kappas_MHz, float)

    best_score, best = -np.inf, None
    n_ok = 0
    n_points = int(axis0.size) * int(dE_ax.size) * int(K_ax.size)
    for i0 in range(0, axis0.size, max(1, int(block))):
        F, dE, K = np.meshgrid(axis0[i0:i0 + block], dE_ax, K_ax, indexing="ij")
        v = _evaluate(F.ravel(), dE.ravel(), K.ravel(), n01, f01_GHz, alpha_GHz,
                      chi_mode, n_ratio, kappa_geo, eps_eff, Q_i)
        m = _masks(v, use_g12, use_two_pole, t1_margin)
        ok = m["all"] & m["kappa"]
        if not ok.any():
            continue
        n_ok += int(ok.sum())
        idx = np.flatnonzero(ok)
        score = (v["eta"][idx]
                 + 1e-6 * np.minimum(v["m_chi"][idx], v["m_T1"][idx] / t1_margin))
        j_local = int(np.argmax(score))
        if score[j_local] > best_score:
            best_score = float(score[j_local])
            best = _point_from(v, int(idx[j_local]), alpha_GHz, f01_GHz)
    if best is None:
        return None, dict(n_ok=0, n_points=n_points)

    stats = dict(
        n_ok=n_ok, n_points=n_points,
        grid={axis_label + "_step": float(axis_vals[1] - axis_vals[0]),
              "d_eff_um_step": float(d_effs_um[1] - d_effs_um[0]),
              "kappa_MHz_step": float(kappas_MHz[1] - kappas_MHz[0]),
              axis_label + "_range": [float(axis_vals[0]), float(axis_vals[-1])],
              "d_eff_um_range": [float(d_effs_um[0]), float(d_effs_um[-1])],
              "kappa_MHz_range": [float(kappas_MHz[0]), float(kappas_MHz[-1])],
              "n_axis0": int(axis0.size), "block": int(block)},
        constraints_applied=dict(
            chi_over_kappa_min=CHI_OVER_KAPPA_MIN,
            T1_margin=float(t1_margin),
            g_over_Delta_sq_max=G_OVER_DELTA_SQ_MAX,
            g12_over_Delta12_sq_max=(G12_OVER_DELTA12_SQ_MAX if use_g12 else None),
            two_pole_screen=bool(use_two_pole),
            kappa_floor="kappa >= f_r/Q_i"),
        chi_mode=chi_mode)
    return best, stats


def _point_from(v, j, alpha_GHz, f01_GHz):
    """Assemble the exported operating-point dict from an `_evaluate` result."""
    K, kint = float(v["K"][j]), float(v["kint"][j])
    return dict(
        Delta_r_GHz=float(v["Delta_r"][j]), f_r_GHz=float(v["f_r"][j]),
        Delta_koch_GHz=float(v["Delta_l"][j]),
        Delta_12_GHz=float(v["Delta_12"][j]),
        d_eff_um=float(v["dE"][j] * 1e6), kappa_MHz=K,
        beta_readout=float(v["beta_r"][j]), beta_drive=float(v["beta_d"][j]),
        correction_factor=float(2.0 * v["beta_r"][j]),
        C_c_readout_fF=float(v["cc_r"][j] * 1e15),
        C_c_drive_fF=float(v["cc_d"][j] * 1e15),
        C_c_sum_fF=float((v["cc_r"][j] + v["cc_d"][j]) * 1e15),
        C_c_cap_fF=float(_cpl.c_c_cap() * 1e15),
        oversubscription=float(v["over"][j]), budget_scale=float(v["s"][j]),
        g_MHz=float(v["g"][j] * 1e3),
        g12_MHz=float(v["ratio"] * v["g"][j] * 1e3),
        n12_over_n01=float(v["ratio"]), n_ratio_sq=float(v["r"]),
        chi_MHz=float(v["chi"][j] * 1e3),
        chi_two_pole_MHz=float(v["chi2"][j] * 1e3),
        chi_two_pole_over_koch=float(v["chi2"][j] / v["chi"][j]),
        chi_over_kappa=float(v["m_chi"][j]),
        chi_over_kappa_two_pole=float(v["m_chi2"][j]),
        T1_Purcell_us=float(v["T1"][j] * 1e6),
        g_over_Delta_sq=float(v["gd2"][j]),
        g12_over_Delta12_sq=float(v["g12d2"][j]),
        kappa_intrinsic_MHz=kint,
        kappa_ext_MHz=float(K - kint),
        eta_readout=float(v["eta"][j]),
        Q_w_required=float(v["f_r"][j] * 1e3 / (K - kint)),
        margin_chi_over_kappa=float(v["m_chi"][j]),
        margin_chi_over_kappa_two_pole=float(v["m_chi2"][j]),
        margin_T1=float(v["m_T1"][j]),
        margin_dispersive=float(G_OVER_DELTA_SQ_MAX / v["gd2"][j]),
        margin_dispersive_12=float(G12_OVER_DELTA12_SQ_MAX / v["g12d2"][j]),
    )


def margin_efficiency_front(f01_GHz, alpha_GHz, Delta_r_GHz, Q_i=Q_I,
                            margins=(1.00, 1.05, 1.10, 1.15, 1.20, 1.25),
                            r=None, n_ratio=None):
    """The exact margin-vs-efficiency Pareto front, closed form.

    At fixed Delta_r the two feasibility margins obey the identity

        m_chi * m_T1  =  (kappa_ceiling / kappa)^2                       (PF)

    (both are powers of g that cancel), so a SYMMETRIC margin m on both
    criteria costs kappa = kappa_ceiling/m and hence

        eta(m) = 1 - kappa_int * m / kappa_ceiling .

    The identity is UNCHANGED by the chi-convention correction -- only the
    value of kappa_ceiling moves (`kappa_ceiling_MHz`, now (CEIL')).  What the
    correction does add is that the front is no longer the whole story: the
    two dispersivity caps can bind before the (PF) line does, so
    `kappa_max_MHz` (and the `eta_ceiling` field below) is the operative
    bound, and R = kappa_ceiling/kappa_int is an upper bound on it.
    """
    r_val = float(r) if r is not None else _ladder(n_ratio)[1]
    kceil = float(kappa_ceiling_MHz(Delta_r_GHz, alpha_GHz))
    kint = kappa_intrinsic_MHz(f01_GHz + Delta_r_GHz, Q_i)
    R = kceil / kint
    kmax = float(kappa_max_MHz(Delta_r_GHz, alpha_GHz, r_val))
    return {
        "Delta_r_GHz": Delta_r_GHz,
        "f_r_GHz": float(f01_GHz + Delta_r_GHz),
        "kappa_ceiling_MHz": kceil,
        "kappa_intrinsic_MHz": kint,
        "kappa_max_all_constraints_MHz": kmax,
        "eta_ceiling": float(max(0.0, 1.0 - kint / kmax)),
        "R_max_symmetric_margin": R,
        "identity": "m_chi * m_T1 = (kappa_ceiling/kappa)^2  (g cancels)",
        "front": [dict(symmetric_margin=float(m),
                       kappa_MHz=float(kceil / m),
                       eta_readout=float(1.0 - kint * m / kceil),
                       feasible=bool(m <= R and kceil / m <= kmax))
                  for m in margins],
        "note": (
            "eta falls linearly in the margin and hits zero at m = R = "
            f"{R:.4f}. Trading the recommended point's Purcell margin for "
            "readout efficiency (or the reverse) moves along this line; nothing "
            "outside it is reachable without lifting C2 (higher Q_i) or C3 "
            "(a Purcell filter). Rows with kappa above "
            f"{kmax:.4f} MHz are additionally excluded by the dispersivity "
            "caps C7/(g/Delta)^2 and are flagged infeasible."),
    }


# ================================ 1b. corrected-window analytic diagnostics ===
def above_f01_emptiness_proof(f01_GHz, alpha_GHz, r, Q_i=Q_I,
                              T1_min_s=T1_PURCELL_MIN_S,
                              mandated_D_GHz=(4.0, 16.0), step_GHz=0.01,
                              D_max_GHz=500.0):
    """Is any f_r > f01 feasible under the corrected chi?  Closed form + sweep.

    Two independent statements:

    1. ANALYTIC, needs no grid.  Above f01 the Koch ceiling (CEIL') increases
       in D and saturates at sup = sqrt(|alpha|/(2 pi T1_min)) = 2.9428 MHz,
       while kappa_int(f01+D) = (f01+D)/Q_i increases without bound and is
       already 3.0521 MHz at the smallest mandated detuning D = 4 GHz.  So
       sup(ceiling) < min over the mandated window of kappa_int and the
       mandated window is empty with no numerics at all.
    2. SWEEP, over ALL D > 0 (not just the mandated window) and with the exact
       two-pole ceiling as well, since that one grows like sqrt(D) and does
       eventually exceed kappa_int.  Reported as max(ceiling - kappa_int) with
       the D at which it occurs.
    """
    D = np.arange(step_GHz, D_max_GHz + 1e-9, step_GHz)
    kint = kappa_intrinsic_MHz(f01_GHz + D, Q_i)
    kc_koch = kappa_ceiling_MHz(D, alpha_GHz, T1_min_s)
    kc_2p = kappa_ceiling_two_pole_MHz(D, alpha_GHz, r, T1_min_s, f01_GHz)
    sup = kappa_ceiling_asymptote_MHz(alpha_GHz, T1_min_s)
    in_win = (D >= mandated_D_GHz[0] - 1e-12) & (D <= mandated_D_GHz[1] + 1e-12)

    def _summary(kc, mask):
        d = kc[mask] - kint[mask]
        i = int(np.argmax(d))
        return dict(max_ceiling_minus_kappa_int_MHz=float(d[i]),
                    at_Delta_r_GHz=float(D[mask][i]),
                    at_f_r_GHz=float(f01_GHz + D[mask][i]),
                    ceiling_MHz=float(kc[mask][i]),
                    kappa_int_MHz=float(kint[mask][i]),
                    empty=bool(d[i] < 0.0))

    first_cross = None
    over = np.flatnonzero(kc_2p > kint)
    if over.size:
        first_cross = float(D[over[0]])
    return {
        "verdict": "EMPTY -- no f_r > f01 satisfies chi/kappa >= 1, "
                   "T1_Purcell >= 30 us and kappa >= f_r/Q_i simultaneously",
        "analytic": {
            "sup_kappa_ceiling_MHz": sup,
            "sup_formula": "sqrt(|alpha| / (2 pi T1_min)), the D -> inf limit of (CEIL')",
            "kappa_int_at_mandated_Dmin_MHz": float(
                kappa_intrinsic_MHz(f01_GHz + mandated_D_GHz[0], Q_i)),
            "mandated_window_Delta_r_GHz": list(mandated_D_GHz),
            "sup_below_min_kappa_int": bool(
                sup < kappa_intrinsic_MHz(f01_GHz + mandated_D_GHz[0], Q_i)),
            "statement": (
                "the Koch ceiling never reaches the measured internal-Q "
                "linewidth anywhere in the mandated window; no grid is needed"),
        },
        "sweep": {
            "step_GHz": step_GHz,
            "Delta_r_range_GHz": [float(D[0]), float(D[-1])],
            "koch_mandated_window": _summary(kc_koch, in_win),
            "koch_all_D": _summary(kc_koch, np.ones_like(D, dtype=bool)),
            "two_pole_mandated_window": _summary(kc_2p, in_win),
            "two_pole_first_crossing_Delta_r_GHz": first_cross,
            "two_pole_note": (
                "the exact-element ceiling grows like sqrt(D) and DOES cross "
                "kappa_int, but only far outside the mandated window and far "
                "outside the gap-protection window; inside the mandated window "
                "it stays below"),
        },
        "ceiling_table_MHz": {
            "%.1f" % d: {
                "koch": float(kappa_ceiling_MHz(d, alpha_GHz, T1_min_s)),
                "two_pole": float(kappa_ceiling_two_pole_MHz(
                    d, alpha_GHz, r, T1_min_s, f01_GHz)),
                "kappa_int": float(kappa_intrinsic_MHz(f01_GHz + d, Q_i)),
            } for d in (4.0, 8.0, 12.0, 16.0)},
        "src": "out/verify/chi_convention_ruling.md Sec. 3(c), 3(d)",
    }


def gap_window_envelope(f01_GHz, alpha_GHz, r, window_GHz, Q_i=Q_I,
                        step_GHz=0.05, t1_margin=T1_MARGIN_TARGET):
    """Analytic kappa envelope across the gap-protection window, per f_r.

    For each f_r: the C2 floor kappa_int, the C3 ceiling (CEIL'), the exact
    two-pole ceiling, the all-constraints kappa_max (`kappa_max_MHz`, which
    also carries the two dispersivity caps and the Purcell margin), and the
    resulting eta ceiling.  This is the exact feasibility map -- the fine grid
    only has to find where it is highest.
    """
    f_r = np.arange(window_GHz[0], window_GHz[1] + 1e-9, step_GHz)
    Dr = f_r - f01_GHz
    keep = (np.abs(Dr) > 1e-12) & (np.abs(f_r - (f01_GHz + alpha_GHz)) > 1e-12)
    f_r, Dr = f_r[keep], Dr[keep]
    kint = kappa_intrinsic_MHz(f_r, Q_i)
    kceil = kappa_ceiling_MHz(Dr, alpha_GHz)
    kceil2 = kappa_ceiling_two_pole_MHz(Dr, alpha_GHz, r, f01_GHz=f01_GHz)
    kmax = kappa_max_MHz(Dr, alpha_GHz, r, t1_margin)
    eta = np.where(kmax > kint, 1.0 - kint / kmax, np.nan)
    ok = kmax > kint
    i = int(np.nanargmax(np.where(ok, eta, -np.inf)))
    return {
        "window_f_r_GHz": [float(window_GHz[0]), float(window_GHz[1])],
        "step_GHz": step_GHz, "t1_margin": t1_margin,
        "rows": [dict(f_r_GHz=float(f_r[k]), Delta_r_GHz=float(Dr[k]),
                      kappa_intrinsic_MHz=float(kint[k]),
                      kappa_ceiling_koch_MHz=float(kceil[k]),
                      kappa_ceiling_two_pole_MHz=float(kceil2[k]),
                      kappa_max_MHz=float(kmax[k]),
                      eta_ceiling=(float(eta[k]) if ok[k] else None),
                      feasible=bool(ok[k]))
                 for k in range(f_r.size)],
        "n_feasible_f_r": int(ok.sum()), "n_f_r": int(f_r.size),
        "eta_ceiling_max": float(eta[i]),
        "eta_ceiling_max_at_f_r_GHz": float(f_r[i]),
        "kappa_at_eta_ceiling_max_MHz": float(kmax[i]),
        "feasible_f_r_bands_GHz": _bands(f_r, ok),
        "note": (
            "kappa_max is the closed-form largest linewidth admitting ANY g at "
            "that f_r, after chi/kappa >= 1 (Koch AND two-pole), the Purcell "
            f"margin {t1_margin}x, (g/Delta)^2 <= {G_OVER_DELTA_SQ_MAX} and "
            f"(g12/Delta12)^2 <= {G12_OVER_DELTA12_SQ_MAX}. eta_ceiling = "
            "1 - kappa_int/kappa_max is therefore the exact upper bound on the "
            "readout port efficiency at that frequency."),
    }


def _bands(x, ok):
    """Contiguous [lo, hi] runs of True in `ok`, reported in x units."""
    out, start = [], None
    for k, flag in enumerate(ok):
        if flag and start is None:
            start = k
        elif not flag and start is not None:
            out.append([float(x[start]), float(x[k - 1])])
            start = None
    if start is not None:
        out.append([float(x[start]), float(x[-1])])
    return out


def d_eff_corridor_um(Delta_r_GHz, kappa_MHz, n01, f01_GHz, alpha_GHz,
                      kappa_geo=_cpl.KAPPA_GEO, eps_eff=_cpl.EPS_EFF,
                      n_ratio=None, use_two_pole=None):
    """Closed-form [d_lo, d_hi] window on d_eff_r at fixed (Delta_r, kappa).

    CORRECTED for the chi convention.  With Delta = f01 - f_r = -Delta_r:

        chi/kappa >= 1        ->  g^2 >= kappa |Delta||Delta+alpha| / |alpha|
        |chi_2pole|/kappa >= 1->  g^2 >= kappa / |K2|                      (C8)
        T1 >= T1_min          ->  g^2 <= Delta^2 / (2 pi kappa T1_min)
        (g/Delta)^2 <= 0.01   ->  g   <= 0.1 |Delta|
        (g12/Delta12)^2<=0.01 ->  g   <= 0.1 |Delta+alpha| n01/n12          (C7)

    Below the oversubscription knee beta_r is linear in d_eff, so g = A d^2
    and every bound inverts analytically.  Returns None if the g window is
    empty.
    """
    if use_two_pole is None:
        use_two_pole = TWO_POLE_SCREEN
    ratio, r = _ladder(n_ratio)
    Dl = -Delta_r_GHz * 1e9                     # f01 - f_r, Hz
    a = alpha_GHz * 1e9
    kap = kappa_MHz * 1e6
    f_r = f01_GHz + Delta_r_GHz
    K2 = _two_pole_kernel(f01_GHz, f01_GHz + alpha_GHz, f_r, r) * 1e-9   # 1/Hz

    g_lo_koch = np.sqrt(kap * abs(Dl) * abs(Dl + a) / abs(a))
    g_lo_2p = np.sqrt(kap / abs(K2))
    g_lo = max(g_lo_koch, g_lo_2p) if use_two_pole else g_lo_koch
    g_hi_T1 = np.sqrt(Dl ** 2 / (2 * np.pi * kap * T1_PURCELL_MIN_S))
    g_hi_d01 = np.sqrt(G_OVER_DELTA_SQ_MAX) * abs(Dl)
    g_hi_d12 = np.sqrt(G12_OVER_DELTA12_SQ_MAX) * abs(Dl + a) / ratio
    g_hi = min(g_hi_T1, g_hi_d01, g_hi_d12)
    binding_hi = ("T1_Purcell" if g_hi == g_hi_T1 else
                  "dispersive_01" if g_hi == g_hi_d01 else "dispersive_12")
    if g_hi <= g_lo:
        return None
    V_r = V_MODE_M3 * (F_CAV_GHZ / f_r) ** 3
    Ez = _cav.E_zpf(f_r, V_r, EPS_C)
    # g = 2 * (k eps0 eps_eff d / C_Sigma) * e * Ez * d * n01 / h  = A d^2
    A = (2.0 * kappa_geo * EPS0 * eps_eff / _cpl.C_SIGMA_F) * E * Ez * n01 / H
    knee_m = _cpl.c_c_cap() / (kappa_geo * EPS0 * eps_eff) - D_EFF_DRIVE_M
    return dict(d_lo_um=float(np.sqrt(g_lo / A) * 1e6),
                d_hi_um=float(np.sqrt(g_hi / A) * 1e6),
                g_lo_MHz=float(g_lo / 1e6), g_hi_MHz=float(g_hi / 1e6),
                g_lo_koch_MHz=float(g_lo_koch / 1e6),
                g_lo_two_pole_MHz=float(g_lo_2p / 1e6),
                g_hi_T1_MHz=float(g_hi_T1 / 1e6),
                g_hi_dispersive_01_MHz=float(g_hi_d01 / 1e6),
                g_hi_dispersive_12_MHz=float(g_hi_d12 / 1e6),
                binding_upper_bound=binding_hi,
                binding_lower_bound=("two_pole" if use_two_pole
                                     and g_lo_2p > g_lo_koch else "koch"),
                two_pole_screen=bool(use_two_pole),
                oversubscription_knee_um=float(knee_m * 1e6))


def kappa_max_MHz(Delta_r_GHz, alpha_GHz, r, t1_margin=1.0,
                  T1_min_s=T1_PURCELL_MIN_S, use_g12=True, use_two_pole=None):
    """Largest kappa_r admitting ANY g, at fixed detuning.  Closed form.

    Eliminating g from the same five inequalities as `d_eff_corridor_um`:

        kappa <= kappa_ceiling / sqrt(t1_margin)                  (T1 vs chi)
        kappa <= 0.01 |alpha| |Delta| / |Delta+alpha|             (0-1 cap)
        kappa <= 0.01 |alpha| |Delta+alpha| / (r |Delta|)         (1-2 cap, C7)

    and, with the two-pole screen, the same three with |alpha|/(|Delta||D+a|)
    replaced by |K2|.  Together with kappa >= kappa_int this is the exact
    feasibility test at a given f_r -- no grid involved.
    """
    if use_two_pole is None:
        use_two_pole = TWO_POLE_SCREEN
    Dr = np.asarray(Delta_r_GHz, dtype=float)
    Dl = -Dr
    a = alpha_GHz
    kern = abs(a) / (np.abs(Dl) * np.abs(Dl + a))            # 1/GHz, Koch
    if use_two_pole:
        K2 = np.abs(_two_pole_kernel(0.0, a, Dr, r))         # 1/GHz
        kern = np.minimum(kern, K2)
    k_T1 = np.sqrt(kern * 1e-9 * (Dl * 1e9) ** 2
                   / (2 * np.pi * T1_min_s * t1_margin)) / 1e6
    k_d01 = G_OVER_DELTA_SQ_MAX * kern * Dl ** 2 * 1e3
    out = np.minimum(k_T1, k_d01)
    if use_g12:
        k_d12 = (G12_OVER_DELTA12_SQ_MAX * kern * (Dl + a) ** 2 / r) * 1e3
        out = np.minimum(out, k_d12)
    return out


# ============================================================== 2. geometry ==
def a_readout_um(f_r_GHz, a_um=A_UM, f_cav_GHz=F_CAV_GHZ):
    """Maxwell scaling: nu = a/lambda is invariant at fixed r/a, t/a, eps, so
    a_r = a * f_cav / f_r  exactly."""
    return a_um * f_cav_GHz / f_r_GHz


def scaled_gap_GHz(f_r_GHz, gap_GHz=GAP_GHZ, a_um=A_UM, f_cav_GHz=F_CAV_GHZ):
    """The readout lattice's fundamental TE gap = drive gap * (a/a_r)."""
    scale = a_um / a_readout_um(f_r_GHz, a_um, f_cav_GHz)
    return (gap_GHz[0] * scale, gap_GHz[1] * scale)


def air_fill_fraction(r_over_a=R_OVER_A):
    """Triangular lattice of circular holes: f_air = 2 pi r^2 / (sqrt(3) a^2)."""
    return 2.0 * np.pi * r_over_a ** 2 / np.sqrt(3.0)


def geometry_concept(f_r_GHz, alpha_GHz=None):
    """Two-lattice geometry: scaling, gap placements, concepts, footprints."""
    a_r = a_readout_um(f_r_GHz)
    a_r_precise = a_readout_um(f_r_GHz, A_UM_PRECISE)
    gap_r = scaled_gap_GHz(f_r_GHz)

    # where each frequency sits relative to the OTHER lattice's fundamental gap
    nu_readout_in_drive = float(A_UM * 1e-6 * f_r_GHz * 1e9 / C_LIGHT)
    nu_drive_in_readout = float(a_r * 1e-6 * F_CAV_GHZ * 1e9 / C_LIGHT)
    f_air = float(air_fill_fraction())
    eps_avg = float(f_air * 1.0 + (1.0 - f_air) * EPS_C)

    row_pitch_um = float(np.sqrt(3.0) / 2.0 * A_UM)
    row_pitch_r_um = float(np.sqrt(3.0) / 2.0 * a_r)
    mode_extent_um = float(V_MODE_A3 ** (1.0 / 3.0) * A_UM)
    mode_extent_r_um = float(V_MODE_A3 ** (1.0 / 3.0) * a_r)

    # concept metrics
    sep_side_by_side_um = float(OUTER_ROWS * row_pitch_um
                                + OUTER_ROWS * row_pitch_r_um)
    drive_patch_um = float(2 * OUTER_ROWS * row_pitch_um + 3 * A_UM)  # L3 = 3 holes
    readout_defect_um = float(3 * a_r)
    nested_rows_consumed = float(max(
        0.0, (drive_patch_um - readout_defect_um) / 2.0 / row_pitch_r_um))
    rows_that_fit = float((readout_defect_um - 3 * A_UM) / (2 * row_pitch_um))

    return {
        "maxwell_scaling": {
            "relation": "a_r = a * f_cav / f_r   (nu = a/lambda invariant at fixed r/a, t/a, eps_core)",
            "a_um": A_UM, "a_um_precise": A_UM_PRECISE,
            "f_cav_GHz": F_CAV_GHZ, "f_r_GHz": f_r_GHz,
            "scale_factor_a_r_over_a": F_CAV_GHZ / f_r_GHz,
            "a_r_um": a_r, "a_r_mm": a_r / 1e3,
            "a_r_um_from_precise_a": a_r_precise,
            "r_um": R_OVER_A * a_r, "t_um": T_OVER_A * a_r,
            "r_over_a": R_OVER_A, "t_over_a": T_OVER_A,
            "V_mode_readout_um3": V_MODE_A3 * a_r ** 3,
            "V_mode_scaling_check": (
                "cavity.readout_window_scan already uses V_r = V_mode (f_cav/f_r)^3, "
                "which is identically V_mode_a3 * a_r^3 -- the scan and this "
                "geometry are the same scaling, no new assumption."),
        },
        "gap_positions": {
            "drive_lattice_gap_GHz": list(GAP_GHZ),
            "readout_lattice_gap_GHz": list(gap_r),
            "f_r_in_readout_lattice": {
                "f_GHz": f_r_GHz,
                "in_gap": bool(gap_r[0] <= f_r_GHz <= gap_r[1]),
                "above_lower_edge_GHz": f_r_GHz - gap_r[0],
                "below_upper_edge_GHz": gap_r[1] - f_r_GHz,
                "note": (
                    "true by construction: Maxwell scaling maps the measured "
                    "145 GHz / [137.5, 177.9] GHz placement onto a_r rigidly, so "
                    "the readout line keeps the same +7.5 / -32.9 GHz margins "
                    "scaled by f_r/f_cav. photonics.json:tolerances "
                    "(|dr/a|<=2%, |dt/a|<=5%, |da|<=2%) also scales rigidly."),
            },
            "f_cav_lattice_gap_margins_GHz": [F_CAV_GHZ - GAP_GHZ[0],
                                              GAP_GHZ[1] - F_CAV_GHZ],
            "f_r_in_drive_lattice": {
                "f_GHz": f_r_GHz,
                "in_gap": bool(GAP_GHZ[0] <= f_r_GHz <= GAP_GHZ[1]),
                "below_lower_edge_GHz": GAP_GHZ[0] - f_r_GHz,
                "ratio_lower_edge_over_f": GAP_GHZ[0] / f_r_GHz,
                "nu_a_over_lambda": nu_readout_in_drive,
                "nu0_gap_centre": NU0,
                "verdict": (
                    "FAR BELOW the drive lattice's TE gap: at a/lambda = "
                    f"{nu_readout_in_drive:.4f} the hole lattice is deeply "
                    "subwavelength (lambda/a = "
                    f"{1.0 / nu_readout_in_drive:.1f}), so the drive lattice is "
                    "not a mirror at f_r -- it homogenises into an index-guiding "
                    f"slab of volume-average eps ~ {eps_avg:.2f} "
                    f"(air fill fraction {f_air:.4f}). The readout mode "
                    "PROPAGATES there and is guided away laterally."),
                "homogenised_eps_volume_average": eps_avg,
                "air_fill_fraction": f_air,
                "homogenisation_caveat": (
                    "volume average only; Maxwell-Garnett / full-wave effective "
                    "index would differ by a few percent. No new FDTD run."),
            },
            "f_cav_in_readout_lattice": {
                "f_GHz": F_CAV_GHZ,
                "in_fundamental_gap": bool(gap_r[0] <= F_CAV_GHZ <= gap_r[1]),
                "above_upper_edge_GHz": F_CAV_GHZ - gap_r[1],
                "ratio_f_over_upper_edge": F_CAV_GHZ / gap_r[1],
                "nu_a_over_lambda": nu_drive_in_readout,
                "verdict": (
                    f"at a_r/lambda = {nu_drive_in_readout:.4f}, i.e. "
                    f"{nu_drive_in_readout / NU0:.3f}x the fundamental cavity-line "
                    "nu, the 145 GHz drive tone is far above the readout "
                    "lattice's first TE gap and lies in its higher-order band "
                    "manifold."),
                "PARAMETRIC_CAVEAT": (
                    "whether 145 GHz lands inside a higher-order gap of the "
                    "readout lattice is NOT computed here -- it needs an MPB/FDTD "
                    "band run at a_r, which is out of P5 scope. The default "
                    "design assumption is the pessimistic one: higher-order "
                    "bands are dense, so the readout lattice does NOT confine "
                    "145 GHz. This is a parametric caveat, not a computed "
                    "result."),
            },
            "symmetry_verdict": (
                "NEITHER lattice confines the other's frequency. Each cavity "
                "needs its own mirror region, and each region is transparent "
                "(or leaky) at the other's frequency."),
        },
        "length_scales_um": {
            "row_pitch_drive": row_pitch_um,
            "row_pitch_readout": row_pitch_r_um,
            "L3_mode_linear_extent_drive": mode_extent_um,
            "L3_mode_linear_extent_readout": mode_extent_r_um,
            "mode_extent_convention": "V_mode_a3^(1/3) * a (cube-root of the measured modal volume)",
            "d_eff_drive": D_EFF_DRIVE_M * 1e6,
            "d_eff_over_a_r_at_100um": 100.0 / a_r,
            "outer_mirror_rows": OUTER_ROWS,
            "chip_rule": CHIP_SCALE_RULE,
            "chip_footprint_drive_mm": CHIP_PERIODS_MIN * A_UM / 1e3,
            "chip_footprint_readout_mm": CHIP_PERIODS_MIN * a_r / 1e3,
        },
        "concepts": [
            {
                "id": "A", "name": "side-by-side (butted lattice regions)",
                "description": (
                    "Two rectangular photonic-crystal regions sharing a straight "
                    "interface; each L3 defect buried behind its own 8 outer "
                    "mirror rows; the qubit island sits at the interface and "
                    "reaches each cavity centre through a routed trace."),
                "cavity_centre_separation_min_um": sep_side_by_side_um,
                "cavity_centre_separation_min_mm": sep_side_by_side_um / 1e3,
                "verdict": "REJECTED",
                "reason": (
                    f"the minimum centre-to-centre distance is "
                    f"{sep_side_by_side_um / 1e3:.2f} mm, so at least one "
                    "collection electrode becomes a >1 mm routed trace. P2's "
                    "divider assumes an electrically small lumped pickup "
                    "(C_c = kappa_geo eps0 eps_eff d_eff); a mm-scale trace adds "
                    "capacitance that is not in the C_Sigma budget and is a "
                    "distributed element at 40-60 GHz. The lumped beta model, "
                    "and with it the whole P2/P5 coupling chain, breaks."),
            },
            {
                "id": "B", "name": "shared interface defect (heterostructure seam)",
                "description": (
                    "A single defect straddling the seam, with drive-lattice "
                    "mirrors on one side and readout-lattice mirrors on the "
                    "other."),
                "verdict": "REJECTED",
                "reason": (
                    "confinement is one-sided at BOTH frequencies: the "
                    "readout-lattice side is transparent at 145 GHz (higher-order "
                    "bands) and the drive-lattice side is a homogenised "
                    f"eps ~ {eps_avg:.2f} slab at {f_r_GHz:.2f} GHz. A mode "
                    "bounded on only one side is not a resonator; both Q's "
                    "collapse to the open-slab radiation value."),
            },
            {
                "id": "C", "name": "nested / concentric heterostructure (RECOMMENDED)",
                "description": (
                    "The complete drive-lattice patch (its L3 defect plus its "
                    "mirror rows) is embedded at the CENTRE of the readout "
                    "lattice's L3 defect region. Both field maxima then coincide "
                    "at one point, where the island and both short collection "
                    "electrodes sit. At f_r the drive patch is a homogenised "
                    f"eps ~ {eps_avg:.2f} dielectric plug inside the readout "
                    "defect -- a perturbation of the readout mode, not a mirror."),
                "drive_patch_span_um": drive_patch_um,
                "readout_defect_span_um": readout_defect_um,
                "fits_inside_readout_defect": bool(drive_patch_um <= readout_defect_um),
                "readout_mirror_rows_consumed_per_side": nested_rows_consumed,
                "drive_mirror_rows_that_fit": rows_that_fit,
                "verdict": "VIABLE, CONDITIONAL",
                "reason": (
                    f"with the measured {OUTER_ROWS} outer mirror rows the drive "
                    f"patch spans {drive_patch_um / 1e3:.2f} mm against a "
                    f"{readout_defect_um / 1e3:.2f} mm readout L3 defect, so it "
                    f"overruns by {nested_rows_consumed:.2f} readout mirror rows "
                    "per side. Either the drive cavity is cut back to "
                    f"~{rows_that_fit:.1f} mirror rows (Q_i then unverified -- "
                    "the measured 14611 assumes 8 rows in a 24a cell), or the "
                    "readout defect is enlarged beyond L3. Both need an FDTD "
                    "run at a_r; DESIGN ESTIMATE, not a computed Q."),
                "island_placement": (
                    "island at the common centre; d_eff_drive = 20 um and "
                    f"d_eff_readout = O(60 um) are {D_EFF_DRIVE_M * 1e6 / A_UM:.4f} "
                    f"and {60.0 / a_r:.4f} of the respective lattice constants, "
                    "so both pickups stay electrically small and the P2 lumped "
                    "divider survives."),
            },
        ],
        "recommended_concept": "C",
    }


def purcell_filter_N(kappa_target_MHz, f_r_GHz, sigma=LADDER_SIGMA, b=LADDER_B,
                     Q_i=None):
    """Rows of separation N between readout cavity and port for a target kappa.

    Q_w(N) = exp(sigma N + b)   (photonics.json:ladder.exponential_fit).

    Two readings are returned:
      * ``N_naive``  from the task's stated Q = f_r/kappa relation, i.e.
        pretending the whole linewidth is the waveguide coupling;
      * ``N_with_internal`` from Q_w = f_r/(kappa - kappa_int), the physically
        correct split once the measured internal Q is honoured.
    """
    kt = float(kappa_target_MHz)
    Q_naive = f_r_GHz * 1e3 / kt
    N_naive = (np.log(Q_naive) - b) / sigma
    out = dict(kappa_target_MHz=kt, f_r_GHz=f_r_GHz,
               Q_total_naive=Q_naive, N_naive=float(N_naive),
               model="ln Q_w = sigma*N + b", sigma=sigma, b=b,
               usable_N_measured=list(LADDER_USABLE_N))
    if Q_i is not None:
        kint = kappa_intrinsic_MHz(f_r_GHz, Q_i)
        out["kappa_intrinsic_MHz"] = kint
        if kt > kint:
            Qw = f_r_GHz * 1e3 / (kt - kint)
            out["Q_w_required"] = Qw
            out["N_with_internal"] = float((np.log(Qw) - b) / sigma)
            out["eta_readout"] = 1.0 - kint / kt
        else:
            out["Q_w_required"] = None
            out["N_with_internal"] = None
            out["eta_readout"] = 0.0
            out["impossible"] = (
                f"kappa_target {kt:.4g} MHz <= kappa_intrinsic "
                f"{kint:.4g} MHz: no value of N reaches it, the cavity is "
                "already broader than the target.")
    return out


def purcell_filter_option(Delta_r_GHz, filter_bandwidth_MHz=(100.0, 200.0, 500.0)):
    """A genuine Purcell filter (bandpass element between cavity and port).

    Suppression of the Purcell rate at the qubit frequency by a filter of
    linewidth kappa_f centred on f_r:
        Gamma_P -> Gamma_P * (kappa_f / (2 Delta_r))^2       (design estimate)
    """
    D = Delta_r_GHz * 1e9
    rows = []
    for kf in filter_bandwidth_MHz:
        supp = (kf * 1e6 / (2 * D)) ** 2
        rows.append(dict(filter_bandwidth_MHz=kf, suppression=supp,
                         suppression_dB=10 * np.log10(supp),
                         T1_Purcell_gain=1.0 / supp))
    return dict(
        formula="Gamma_Purcell -> Gamma_Purcell * (kappa_f/(2*Delta_r))^2",
        Delta_r_GHz=Delta_r_GHz, rows=rows,
        status="DESIGN ESTIMATE",
        caveat=(
            "standard single-pole bandpass-filter estimate; no FDTD of a "
            "photonic-crystal filter at a_r has been run. It is the escape "
            "hatch from the (CEIL) kappa ceiling: with the Purcell constraint "
            "lifted, kappa_r is limited only by chi/kappa >= 1, and the readout "
            "port efficiency eta can be pushed toward 1."),
    )


# ========================================================== 3. control line ==
def n_drive_from_power(P_in_W, f_drive_GHz, f_r_GHz, kappa_MHz, kappa_ext_MHz):
    """Intracavity photon number of a coherent tone driving the readout port.

    Input-output steady state of  adot = -i D a - (kappa/2) a + sqrt(k_ext) a_in
    with |a_in|^2 = P_in/(hbar omega_d) photons/s:

        n = (P_in / (hbar omega_d)) * kappa_ext / (D^2 + (kappa/2)^2)

    ALL of kappa, kappa_ext and D = 2 pi (f_drive - f_r) in ANGULAR units
    (rad/s); P_in in W; the result is dimensionless.
    """
    w_d = 2 * np.pi * f_drive_GHz * 1e9
    D = 2 * np.pi * (f_drive_GHz - f_r_GHz) * 1e9
    kap = 2 * np.pi * kappa_MHz * 1e6
    kext = 2 * np.pi * kappa_ext_MHz * 1e6
    return (P_in_W / (HBAR * w_d)) * kext / (D ** 2 + (kap / 2) ** 2)


def rabi_from_power(P_in_W, g_MHz, f_drive_GHz, f_r_GHz, kappa_MHz,
                    kappa_ext_MHz):
    """Omega_R/2pi in Hz from Omega_R = 2 g sqrt(n_drive)."""
    n = n_drive_from_power(P_in_W, f_drive_GHz, f_r_GHz, kappa_MHz, kappa_ext_MHz)
    return 2.0 * g_MHz * 1e6 * np.sqrt(n)


def power_for_rabi(Omega_R_Hz, g_MHz, f_drive_GHz, f_r_GHz, kappa_MHz,
                   kappa_ext_MHz):
    """Invert rabi_from_power: P_in in W for a target Omega_R/2pi in Hz."""
    n_req = (Omega_R_Hz / (2.0 * g_MHz * 1e6)) ** 2
    w_d = 2 * np.pi * f_drive_GHz * 1e9
    D = 2 * np.pi * (f_drive_GHz - f_r_GHz) * 1e9
    kap = 2 * np.pi * kappa_MHz * 1e6
    kext = 2 * np.pi * kappa_ext_MHz * 1e6
    P = n_req * (D ** 2 + (kap / 2) ** 2) * HBAR * w_d / kext
    return P, n_req


def lorentzian_suppression_dB(f_tone_GHz, f_mode_GHz, kappa_mode_MHz):
    """Photon-number response of a mode at detuning, relative to resonance."""
    D = 2 * np.pi * (f_tone_GHz - f_mode_GHz) * 1e9
    k2 = (2 * np.pi * kappa_mode_MHz * 1e6 / 2.0) ** 2
    r = k2 / (D ** 2 + k2)
    return float(r), float(10 * np.log10(r))


def dBm(P_W):
    return float(10 * np.log10(P_W / 1e-3))


def control_line_budget(point, f01_GHz, Q_i=Q_I, f_cav_GHz=F_CAV_GHZ,
                        rabi_targets_MHz=(10.0, 50.0),
                        readout_photons=(1.0, 10.0)):
    """Rabi power budget through the readout port + cross-mode leakage."""
    f_r = point["f_r_GHz"]
    kap = point["kappa_MHz"]
    kext = point.get("kappa_ext_MHz", kap - kappa_intrinsic_MHz(f_r, Q_i))
    g = point["g_MHz"]
    kappa_drive_MHz = _cav.kappa_from_Q(f_cav_GHz, Q_i) * 1e3

    rabi = []
    for om in rabi_targets_MHz:
        P, n = power_for_rabi(om * 1e6, g, f01_GHz, f_r, kap, kext)
        rabi.append(dict(Omega_R_over_2pi_MHz=om, n_drive=float(n),
                         P_in_W=float(P), P_in_pW=float(P * 1e12),
                         P_in_dBm=dBm(P),
                         gate_time_pi_ns=float(0.5 / (om * 1e6) * 1e9)))

    read = []
    for n in readout_photons:
        # on-resonance readout tone: D = 0
        P = n * (2 * np.pi * kap * 1e6 / 2) ** 2 * HBAR * 2 * np.pi * f_r * 1e9 \
            / (2 * np.pi * kext * 1e6)
        read.append(dict(n_readout=n, P_in_W=float(P), P_in_aW=float(P * 1e18),
                         P_in_dBm=dBm(P)))

    r_ctrl, dB_ctrl = lorentzian_suppression_dB(f01_GHz, f_cav_GHz, kappa_drive_MHz)
    r_rd, dB_rd = lorentzian_suppression_dB(f_r, f_cav_GHz, kappa_drive_MHz)
    r_dr, dB_dr = lorentzian_suppression_dB(f_cav_GHz, f_r, kap)

    return {
        "formulas": {
            "n_drive": ("n = (P_in/(hbar*omega_d)) * kappa_ext / "
                        "(Delta_qr^2 + (kappa/2)^2); kappa, kappa_ext, Delta_qr "
                        "in rad/s, P_in in W, omega_d = 2 pi f_drive"),
            "rabi": "Omega_R = 2 g sqrt(n_drive); quoted as Omega_R/2pi in Hz",
            "detuning": "Delta_qr = 2 pi (f01 - f_r) = -2 pi Delta_r",
            "leakage": ("Lorentzian photon response of a mode at detuning, "
                        "(kappa/2)^2 / (Delta^2 + (kappa/2)^2)"),
        },
        "operating_point": {"f_r_GHz": f_r, "f01_GHz": f01_GHz,
                            "Delta_qr_GHz": f01_GHz - f_r,
                            "g_MHz": g, "kappa_MHz": kap,
                            "kappa_ext_MHz": kext,
                            "kappa_int_MHz": kap - kext,
                            "eta_readout": kext / kap},
        "rabi_budget": rabi,
        "readout_tone_budget": read,
        "leakage": {
            "control_tone_into_drive_cavity": {
                "f_tone_GHz": f01_GHz, "f_mode_GHz": f_cav_GHz,
                "detuning_GHz": f_cav_GHz - f01_GHz,
                "kappa_drive_MHz": kappa_drive_MHz,
                "suppression": r_ctrl, "suppression_dB": dB_ctrl,
            },
            "readout_tone_into_drive_cavity": {
                "f_tone_GHz": f_r, "f_mode_GHz": f_cav_GHz,
                "detuning_GHz": f_cav_GHz - f_r,
                "suppression": r_rd, "suppression_dB": dB_rd,
            },
            "drive_tone_into_readout_cavity": {
                "f_tone_GHz": f_cav_GHz, "f_mode_GHz": f_r,
                "detuning_GHz": f_cav_GHz - f_r,
                "kappa_readout_MHz": kap,
                "suppression": r_dr, "suppression_dB": dB_dr,
            },
            "caveat": (
                "Lorentzian mode filtering only. The port-to-far-cavity "
                "coupling coefficient (how much of a tone injected at the "
                "readout port even reaches the drive cavity's near field) is a "
                "layout quantity and is NOT modelled -- these dB figures are "
                "the on-chip frequency selectivity, not the total isolation."),
        },
    }


def pair_breaking_restatement(p3):
    """Restate P3's linear pair-breaking audit for the readout tone."""
    rs = p3["pair_breaking_audit"]["readout_sample"]
    dr = p3["pair_breaking_audit"]["drive_cavity"]
    return {
        "two_Delta_over_h_GHz": p3["pair_breaking_audit"]["two_Delta_over_h_GHz"],
        "readout_sample_47p6GHz": {
            "f_GHz": rs["f_GHz"], "hf_over_2Delta": rs["hf_over_2Delta"],
            "margin_2Delta_over_hf": rs["margin_2Delta_over_hf"],
            "sub_gap_filling_window_nu": rs["sub_gap_filling_window"],
            "window_halfwidth_nu": rs["window_halfwidth_nu"],
        },
        "drive_145GHz": {
            "f_GHz": dr["f_GHz"], "hf_over_2Delta": dr["hf_over_2Delta"],
            "margin_2Delta_over_hf": dr["margin_2Delta_over_hf"],
            "window_halfwidth_nu": dr["window_halfwidth_nu"],
        },
        "statement": (
            "LINEAR (single-photon) audit only, carried over from P3 unchanged: "
            "h f_r < 2 Delta(nu) over |nu + 2.4| < 0.4111, i.e. essentially the "
            "whole superconducting dome, with a 3.089x margin at the apex. The "
            "readout tone is therefore not pair-breaking at any power in this "
            "model."),
        "PARAMETRIC_CAVEAT": (
            "photon-ASSISTED (multi-photon) pair breaking is NOT modelled. The "
            "naive order is n_gamma >= 2 Delta / (h f_r) = "
            f"{rs['margin_2Delta_over_hf']:.3f}, i.e. a 4-photon process at the "
            "readout frequency; its rate depends on an unmodelled nonlinearity "
            "and on the intracavity photon number. Nothing here bounds it -- it "
            "is flagged, not computed."),
    }


# ================================================== 4. constraint synthesis ==
def operating_envelope(design, recommended, p2, p3, n_naive, Q_i=Q_I,
                       r=None, empty_proof=None, envelope=None, window=None):
    """The two-cavity operating envelope: every binding constraint in one place.

    C3/C4 are the CORRECTED forms (chi-convention ruling); C7/C8 are new.
    """
    alpha = design["anharm_GHz"]
    f01 = design["f01_GHz"]
    f12 = f01 + alpha
    r = _ladder()[1] if r is None else r
    ratio = np.sqrt(r)
    knee_um = (_cpl.c_c_cap() / (_cpl.KAPPA_GEO * EPS0 * _cpl.EPS_EFF)
               - D_EFF_DRIVE_M) * 1e6
    kc = {"%.1f" % D: float(kappa_ceiling_MHz(D, alpha))
          for D in (4.0, 8.0, 12.0, 16.0)}
    f_r = recommended["f_r_GHz"]
    kint_meas = kappa_intrinsic_MHz(f_r, Q_i)
    kint_opt = kappa_intrinsic_MHz(f_r, q_i_ceiling())
    kceil = float(kappa_ceiling_MHz(recommended["Delta_r_GHz"], alpha))
    dw = p3["pair_breaking_audit"]["drive_cavity"]

    return {
        "constraints": [
            {
                "id": "C1", "name": "shared C_Sigma budget (P2)",
                "statement": "C_c,drive + C_c,readout <= C_Sigma - C_J",
                "cap_fF": _cpl.c_c_cap() * 1e15,
                "C_Sigma_fF": _cpl.C_SIGMA_F * 1e15,
                "C_J_fF": _cpl.C_J_F * 1e15,
                "d_eff_readout_knee_um": knee_um,
                "meaning": (
                    f"at kappa_geo = {_cpl.KAPPA_GEO} and eps_eff = "
                    f"{_cpl.EPS_EFF:.2f}, any d_eff_readout above "
                    f"{knee_um:.1f} um oversubscribes the island and forces the "
                    "common-process rescale. The brief's 100 um sample point "
                    "sits above the knee (oversubscription "
                    f"{p2['two_cavity_budget']['oversubscription']:.4f}); the "
                    f"recommended {recommended['d_eff_um']:.1f} um sits BELOW "
                    "it, so the oversubscription finding is resolved, not "
                    "merely rescaled."),
            },
            {
                "id": "C2", "name": "measured internal-Q linewidth floor (P5, new)",
                "statement": "kappa_r >= f_r / Q_i, Q_i scale-invariant from photonics.json",
                "Q_i_measured": Q_i,
                "Q_i_optimistic_ceiling": q_i_ceiling(),
                "kappa_int_MHz_at_recommended": kint_meas,
                "kappa_int_MHz_optimistic": kint_opt,
                "meaning": (
                    "the brief's kappa_r grid {1, 3} MHz lies at or below this "
                    f"floor ({kint_meas:.3f} MHz at f_r = {f_r:.3f} GHz), so "
                    "every kappa = 1 MHz row of the naive feasible set is "
                    "physically unreachable with the measured cavity."),
            },
            {
                "id": "C3", "name": "analytic Purcell/dispersive kappa ceiling (P5; CORRECTED 2026-07-27)",
                "statement": ("kappa_r^2 <= |alpha| |Delta| / (|Delta + alpha| "
                              "2 pi T1_min), Delta = f01 - f_r"),
                "T1_min_us": T1_PURCELL_MIN_S * 1e6,
                "alpha_GHz": alpha,
                "kappa_ceiling_MHz_vs_Delta": kc,
                "kappa_ceiling_at_recommended_MHz": kceil,
                "kappa_ceiling_two_pole_at_recommended_MHz": float(
                    kappa_ceiling_two_pole_MHz(recommended["Delta_r_GHz"],
                                               alpha, r, f01_GHz=f01)),
                "supremum_above_f01_MHz": kappa_ceiling_asymptote_MHz(alpha),
                "superseded_statement": (
                    "kappa_r^2 <= |alpha| Delta_r / ((Delta_r + alpha) 2 pi T1_min) "
                    "with Delta_r = f_r - f01 -- the inverted-detuning form; it "
                    "gave 3.8249 MHz at Delta_r = 4 GHz against the corrected "
                    "2.4799 MHz, and it decreased in Delta_r where the corrected "
                    "ceiling increases"),
                "meaning": (
                    "chi/kappa >= 1 and T1_Purcell >= 30 us cannot both hold "
                    "above this kappa, for ANY g and hence any d_eff. Under the "
                    "corrected chi it INCREASES with detuning above f01 and "
                    "saturates at "
                    f"{kappa_ceiling_asymptote_MHz(alpha):.4f} MHz, which is "
                    "below kappa_int everywhere in the mandated 4-16 GHz "
                    "window -- so that window is EMPTY (see "
                    "above_f01_emptiness_proof). Below f01 the ceiling is much "
                    "wider because |Delta + alpha| = |f_r - f12| can be small, "
                    "and that is where the design now sits."),
            },
            {
                "id": "C4", "name": "readout port efficiency squeeze (C2 + C3 + C7)",
                "statement": "eta = kappa_ext/kappa <= 1 - kappa_int/kappa_max",
                "eta_max_measured_Qi": float(max(0.0, 1.0 - kint_meas / kceil)),
                "eta_max_optimistic_Qi": float(max(0.0, 1.0 - kint_opt / kceil)),
                "eta_at_recommended": recommended["eta_readout"],
                "eta_ceiling_gap_window": (
                    envelope["eta_ceiling_max"] if envelope else None),
                "eta_ceiling_gap_window_at_f_r_GHz": (
                    envelope["eta_ceiling_max_at_f_r_GHz"] if envelope else None),
                "eta_max_above_f01": 0.0,
                "superseded_value": {
                    "flag": ("REFERENCE CONVENTION -- the 'eta <= 20.2 % "
                             "anywhere, 11.5 % at the recommended point' "
                             "reading came from the inverted-detuning chi"),
                    "eta_max_measured_Qi": 0.2020,
                    "eta_at_superseded_recommended_point": 0.11532819289330343,
                },
                "meaning": (
                    "the achievable kappa window is bracketed from below by C2 "
                    "and from above by C3 together with the two dispersivity "
                    "caps. Above f01 the bracket is EMPTY (eta_max = 0). Inside "
                    "the gap-protection window below f01 it reopens, and the "
                    "efficiency ceiling there is "
                    + (f"{envelope['eta_ceiling_max']:.4f} at f_r = "
                       f"{envelope['eta_ceiling_max_at_f_r_GHz']:.3f} GHz"
                       if envelope else "computed in gap_window_scan")
                    + " -- roughly three times the superseded 11.5 % corner, "
                    "which was an artifact of the chi sign error rather than a "
                    "physical squeeze. A genuine Purcell filter still relieves "
                    "C3, and higher Q_i still relieves C2."),
            },
            {
                "id": "C5", "name": "drive-tone pair-breaking window (P3)",
                "statement": "2 Delta(nu) > h * 145 GHz only for |nu + 2.4| < 0.058571",
                "window_halfwidth_nu": dw["window_halfwidth_nu"],
                "f01_span_inside_window_GHz": dw["f01_span_inside_window_GHz"],
                "margin_2Delta_over_hf": dw["margin_2Delta_over_hf"],
                "meaning": (
                    "with the drive plane pinned at 145 GHz the qubit may only "
                    "be gated over a 0.29 GHz span of f01. Combined with C1-C4, "
                    "the readout plane is the only tunable degree of freedom "
                    "left; the 23.7-40.6 GHz tunability claim requires moving "
                    "the drive plane, not the readout plane."),
            },
            {
                "id": "C6", "name": "lattice mutual transparency (P5 geometry)",
                "statement": (
                    "f_r sits far below the drive lattice's TE gap and 145 GHz "
                    "sits far above the readout lattice's fundamental gap"),
                "meaning": (
                    "each cavity must carry its own complete mirror region; the "
                    "only topology that keeps both field maxima on one island is "
                    "the nested concept C, and that costs readout mirror rows."),
            },
            {
                "id": "C7", "name": "1-2-line dispersivity (P5, new; chi-convention ruling)",
                "statement": ("(g12/Delta_12)^2 <= 0.01 with g12 = g n12/n01 "
                              "and Delta_12 = f_r - f12"),
                "n12_over_n01": float(ratio),
                "n_ratio_sq": float(r),
                "f12_GHz": float(f12),
                "cap": G12_OVER_DELTA12_SQ_MAX,
                "value_at_recommended": recommended["g12_over_Delta12_sq"],
                "margin_at_recommended": recommended["margin_dispersive_12"],
                "value_at_ruling_candidate_f_r_39p2589": 0.018583016504363822,
                "meaning": (
                    "the corrected optimum sits between / near the two transmon "
                    "lines, where the 0-1 dispersivity test is NOT the binding "
                    "one: the exact 1-2 coupling is g12 = 1.38523 g and the 1-2 "
                    "detuning is small. Imposing the same 0.01 the design "
                    "already imposes on the 0-1 line makes the band "
                    "f_r ~ 38.15-39.44 GHz INFEASIBLE (not merely marginal): at "
                    "f_r = 39.2589 GHz -- the ruling's own candidate -- the "
                    "largest admissible kappa falls to 1.881 MHz, below "
                    "kappa_int = 2.687 MHz. The optimum therefore moves BELOW "
                    "the 1-2 line."),
            },
            {
                "id": "C8", "name": "two-pole robustness screen (P5, new; truth standard)",
                "statement": "|chi_two_pole|/kappa >= 1 as well as |chi_Koch|/kappa >= 1",
                "enabled": bool(TWO_POLE_SCREEN),
                "chi_over_kappa_koch_at_recommended": recommended["chi_over_kappa"],
                "chi_over_kappa_two_pole_at_recommended":
                    recommended["chi_over_kappa_two_pole"],
                "two_pole_over_koch_at_recommended":
                    recommended["chi_two_pole_over_koch"],
                "meaning": (
                    "the corrected single formula is exact only in the harmonic "
                    "limit (n12/n01)^2 -> 2; at E_J/E_C = 100 it overstates "
                    "|chi| by about 8 % at the recommended detuning. The ruling "
                    "designates the exact two-pole form the truth standard, so "
                    "the recommended point is required to clear chi/kappa >= 1 "
                    "under BOTH constructions. Dropping this screen (module "
                    "flag TWO_POLE_SCREEN) moves the optimum to the "
                    "koch_only_optimum block, which reaches eta = 0.3413 but "
                    "has a two-pole chi/kappa of only 0.9315."),
            },
        ],
        "envelope_statement": (
            "The two-cavity chip is feasible, but not where the superseded "
            "artifact put it. Once the P2 divider and the shared C_Sigma budget "
            f"are enforced, the naive {n_naive}-point readout window collapses; "
            "once the measured internal-Q floor C2 is added, every kappa = 1 MHz "
            "point disappears. With the CORRECTED dispersive shift (Delta = "
            "f01 - f_r) the analytic ceiling C3 inverts its monotonicity and "
            "saturates at "
            f"{kappa_ceiling_asymptote_MHz(alpha):.4f} MHz above f01, below "
            "kappa_int everywhere in the mandated 4-16 GHz window: THE ENTIRE "
            "f_r > f01 READOUT WINDOW IS EMPTY, analytically, for any g and any "
            "d_eff. What survives is a corner BELOW the qubit, inside the "
            "gap-protection window "
            f"[{window[0]:.3f}, {window[1]:.3f}] GHz over which the "
            "Maxwell-scaled readout lattice's own TE gap covers f01: "
            f"f_r ~ {recommended['f_r_GHz']:.3f} GHz "
            f"(Delta_r = {recommended['Delta_r_GHz']:.3f} GHz, i.e. "
            f"{abs(recommended['Delta_12_GHz']):.3f} GHz below the 1-2 line), "
            f"d_eff_r ~ {recommended['d_eff_um']:.1f} um, "
            f"kappa_r ~ {recommended['kappa_MHz']:.2f} MHz, at a readout port "
            f"efficiency of {recommended['eta_readout'] * 100:.1f}% -- about "
            "three times the 11.5% quoted before the correction, which was an "
            "artifact of the sign error, not a physical squeeze. That electrode "
            f"still sits below the C_Sigma oversubscription knee at "
            f"{knee_um:.1f} um. Two new constraints bind here and did not "
            "before: C7, the 1-2-line dispersivity cap, which empties the band "
            "just below f01 that the ruling had flagged only as marginal; and "
            "C8, the requirement that the exact two-pole chi clears chi/kappa "
            ">= 1 too. The remaining hard external constraint is unchanged: C5, "
            "the 145 GHz drive plane pins the qubit to a 0.29 GHz window of "
            "f01, so gate-based frequency allocation and a fixed 145 GHz drive "
            "cavity are mutually exclusive regardless of anything the readout "
            "plane does."),
        "escape_hatches": [
            "genuine Purcell filter at f_r: lifts C3, frees kappa_r and eta",
            "higher Q_i (larger cell / better vertical confinement): lowers C2, "
            f"optimistic ceiling Q_i = {q_i_ceiling():.0f} gives "
            f"kappa_int = {kint_opt:.3f} MHz",
            "drive plane moved below 147 GHz: lifts C5 entirely",
            "relaxing C7 from 0.01 to the 0.0186 the ruling's candidate needs "
            "would reopen f_r ~ 39.26 GHz, but at a 1-2-line dispersivity "
            "1.9x worse than the design allows on the 0-1 line -- an owner "
            "decision, not a default",
        ],
        "above_f01": empty_proof,
    }


# =============================================== 5. chi-convention self-check ==
def chi_convention_selfcheck(n01, f01_GHz, alpha_GHz, r,
                             g_readout_MHz=142.1265211330537,
                             f_r_GHz=44.5966287014581,
                             g_drive_MHz=181.77383656589163,
                             f_drive_GHz=F_CAV_GHZ):
    """Reproduce the ruling's Sec. 2 numerics table from this module's formulas.

    The two probe points and the two g values are the ruling's, read here as
    literals ONLY because they are the ruling's inputs, not results: the g's
    are the stored P2-pinned artifact values (p5 recommended point / p2 drive
    row) and are asserted against those artifacts in tests/test_m6_p5.py.
    Everything computed below is produced by qubit/cavity.py.
    """
    f12 = f01_GHz + alpha_GHz
    out = {}
    for tag, gM, fr, ruling in (
            ("readout_44p5966GHz", g_readout_MHz, f_r_GHz,
             dict(two_pole_MHz=-1.6090833, koch_MHz=-1.4635662,
                  reference_MHz=-3.4816279)),
            ("drive_145GHz", g_drive_MHz, f_drive_GHz,
             dict(two_pole_MHz=-0.0175154, koch_MHz=-0.0048720,
                  reference_MHz=-0.0050268))):
        g = gM * 1e-3
        two = 1e3 * _cav.chi_two_pole(g, f01_GHz, f12, fr, r)
        koch = 1e3 * _cav.chi_dispersive_koch(g, f01_GHz, fr, alpha_GHz)
        ref = 1e3 * _cav.chi_dispersive(g, fr - f01_GHz, alpha_GHz)
        out[tag] = {
            "g_MHz": gM, "f_cavity_GHz": fr,
            "chi_two_pole_MHz": two, "chi_koch_MHz": koch,
            "chi_reference_convention_MHz": ref,
            "reference_over_koch": ref / koch,
            "koch_residual_vs_two_pole": koch / two - 1.0,
            "reference_residual_vs_two_pole": ref / two - 1.0,
            "ruling_values_MHz": ruling,
            "max_reldev_vs_ruling": max(
                abs(two - ruling["two_pole_MHz"]) / abs(ruling["two_pole_MHz"]),
                abs(koch - ruling["koch_MHz"]) / abs(ruling["koch_MHz"]),
                abs(ref - ruling["reference_MHz"]) / abs(ruling["reference_MHz"])),
        }
    # harmonic-limit identity: r -> 2 collapses (TWO-POLE) onto (KOCH)
    probes = np.array([20.0, 33.5, 37.5, 39.3, 44.5966287014581, 145.0])
    g = 0.1421265211330537
    ident = np.abs(_cav.chi_two_pole(g, f01_GHz, f12, probes, 2.0)
                   / _cav.chi_dispersive_koch(g, f01_GHz, probes, alpha_GHz) - 1.0)
    out["harmonic_limit_identity"] = {
        "probe_f_r_GHz": [float(x) for x in probes],
        "max_abs_rel_difference": float(ident.max()),
        "statement": ("chi_two_pole(r = 2) == chi_dispersive_koch identically; "
                      "that identity is what fixes Delta = f01 - f_r as the "
                      "single formula's convention"),
    }
    out["pole_locations_GHz"] = {
        "koch_and_two_pole": [float(f01_GHz), float(f12)],
        "reference_convention": [float(f01_GHz), float(f01_GHz - alpha_GHz)],
        "statement": ("the corrected forms are singular only at the 0-1 and "
                      "1-2 lines; the reference convention moves the second "
                      f"pole to f01 + |alpha| = {f01_GHz - alpha_GHz:.4f} GHz, "
                      "where the transmon has no transition"),
    }
    out["r_exact"] = float(r)
    out["r_harmonic"] = 2.0
    out["one_minus_r_over_2"] = float(1.0 - r / 2.0)
    out["src"] = "out/verify/chi_convention_ruling.md"
    return out


# ================================================================== export ===
def build_export():
    import qubit as _q

    design = _q.design_point()
    n01, alpha, f01 = design["n01"], design["anharm_GHz"], design["f01_GHz"]
    f12 = f01 + alpha
    n01_ladder, n12, ratio, r = ladder_elements(EC_GHz=design["EC_GHz"])
    window = gap_protection_window_GHz(f01)

    p2 = json.loads((ROOT / "out" / "m6" / "p2_coupling.json").read_text("utf-8"))
    p3 = json.loads((ROOT / "out" / "m6" / "p3_qp.json").read_text("utf-8"))
    p6 = json.loads((ROOT / "out" / "m6" / "p6_gates.json").read_text("utf-8"))
    expl = json.loads((ROOT / "out" / "m6" / "explore"
                       / "missing_t1_channels.json").read_text("utf-8"))
    expl_win = expl["verdict"]["question_c_retune"]["gap_protection_window_f_r_GHz"]
    expl_koch = expl["verdict"]["question_c_retune"]["best_trade_at_eta_0p2"][
        "koch_convention"]

    # --- naive reference scan (cavity.py, unmodified) -------------------------
    naive = _cav.readout_window_scan(n01, f01, alpha)

    # --- REFERENCE-CONVENTION scan: the superseded artifact, preserved --------
    ref_rows = scan_rows(n01, f01, alpha, chi_mode="reference", n_ratio=ratio,
                         use_g12=False, use_two_pole=False)
    ref_rec, ref_stats = fine_scan(n01, f01, alpha, chi_mode="reference",
                                   n_ratio=ratio, use_g12=False,
                                   use_two_pole=False, block=len(DELTA_FINE_GHZ))
    ref_rec["kappa_ceiling_MHz"] = float(np.sqrt(
        abs(alpha) * 1e9 * ref_rec["Delta_r_GHz"] * 1e9
        / ((ref_rec["Delta_r_GHz"] * 1e9 + alpha * 1e9) * 2 * np.pi
           * T1_PURCELL_MIN_S)) / 1e6)
    ref_rec["kappa_ceiling_MHz_flag"] = (
        "SUPERSEDED FORMULA (CEIL) with Delta_r = f_r - f01; retained so this "
        "block reproduces the pre-correction artifact. The corrected (CEIL') "
        "value at the same detuning is in corrected_values_at_this_point.")
    ref_rec["a_r_um"] = a_readout_um(ref_rec["f_r_GHz"])

    # --- corrected scan, mandated above-f01 grid ------------------------------
    rows = scan_rows(n01, f01, alpha, n_ratio=ratio)
    rows_brief_grid = [r_ for r_ in rows if r_["d_eff_um"] in (100.0, 200.0)]
    bracket = scan_beta_bracket(n01, f01, alpha, n_ratio=ratio)

    # --- corrected scan, gap-protection window (below AND above f01) ----------
    f_r_coarse = np.arange(window[0], window[1] + 1e-9, F_R_COARSE_STEP_GHZ)
    gap_rows = scan_rows(n01, f01, alpha, f_r_GHz=f_r_coarse,
                         d_effs_m=D_EFF_COARSE_M, kappas_MHz=KAPPA_COARSE_MHZ,
                         n_ratio=ratio)
    envelope = gap_window_envelope(f01, alpha, r, window)

    f_r_fine = np.arange(window[0], window[1] + 1e-9, F_R_FINE_STEP_GHZ)
    f_r_fine = f_r_fine[(np.abs(f_r_fine - f01) > POLE_GUARD_GHZ)
                        & (np.abs(f_r_fine - f12) > POLE_GUARD_GHZ)]
    recommended, fine_stats = fine_scan(
        n01, f01, alpha, f_r_GHz=f_r_fine, n_ratio=ratio,
        d_effs_um=D_EFF_CORRECTED_FINE_UM, kappas_MHz=KAPPA_CORRECTED_FINE_MHZ)
    if recommended is None:
        raise RuntimeError("fine scan found no point; nothing to recommend")

    koch_only, koch_only_stats = fine_scan(
        n01, f01, alpha, f_r_GHz=f_r_fine, n_ratio=ratio, use_two_pole=False,
        d_effs_um=D_EFF_CORRECTED_FINE_UM, kappas_MHz=KAPPA_CORRECTED_FINE_MHZ)
    koch_only["kappa_ceiling_MHz"] = float(
        kappa_ceiling_MHz(koch_only["Delta_r_GHz"], alpha))
    koch_only["a_r_um"] = a_readout_um(koch_only["f_r_GHz"])

    recommended["margin_target_T1"] = T1_MARGIN_TARGET
    recommended["kappa_ceiling_MHz"] = float(
        kappa_ceiling_MHz(recommended["Delta_r_GHz"], alpha))
    recommended["kappa_ceiling_two_pole_MHz"] = float(
        kappa_ceiling_two_pole_MHz(recommended["Delta_r_GHz"], alpha, r,
                                   f01_GHz=f01))
    recommended["kappa_max_all_constraints_MHz"] = float(
        kappa_max_MHz(recommended["Delta_r_GHz"], alpha, r, T1_MARGIN_TARGET))
    recommended["a_r_um"] = a_readout_um(recommended["f_r_GHz"])
    recommended["chi_from_cavity_chi_dispersive_koch_MHz"] = (
        1e3 * _cav.chi_dispersive_koch(recommended["g_MHz"] * 1e-3, f01,
                                       recommended["f_r_GHz"], alpha))
    recommended["chi_from_cavity_chi_two_pole_MHz"] = (
        1e3 * _cav.chi_two_pole(recommended["g_MHz"] * 1e-3, f01, f12,
                                recommended["f_r_GHz"], r))
    recommended["chi_reference_convention_MHz"] = (
        1e3 * _cav.chi_dispersive(recommended["g_MHz"] * 1e-3,
                                  recommended["Delta_r_GHz"], alpha))
    recommended["below_1_2_line"] = bool(recommended["f_r_GHz"] < f12)
    recommended["inside_gap_protection_window"] = bool(
        window[0] <= recommended["f_r_GHz"] <= window[1])
    recommended["corridor_at_this_point"] = d_eff_corridor_um(
        recommended["Delta_r_GHz"], recommended["kappa_MHz"], n01, f01, alpha,
        n_ratio=ratio)
    # Honest-reporting addition: the eta-MAXIMISING point necessarily sits
    # where constraints bind, so its d_eff corridor is narrow.  Tabulate how
    # the corridor widens as kappa (and therefore eta) is backed off, so the
    # fabrication tolerance is a stated trade rather than a hidden fragility.
    recommended["corridor_vs_kappa"] = []
    for frac in (1.00, 0.98, 0.95, 0.90, 0.85, 0.80):
        kap = recommended["kappa_MHz"] * frac
        c = d_eff_corridor_um(recommended["Delta_r_GHz"], kap, n01, f01, alpha,
                              n_ratio=ratio)
        if c is None:
            continue
        eta_k = 1.0 - recommended["kappa_intrinsic_MHz"] / kap
        recommended["corridor_vs_kappa"].append(dict(
            kappa_MHz=float(kap), kappa_fraction=float(frac),
            eta_readout=float(eta_k),
            d_lo_um=c["d_lo_um"], d_hi_um=c["d_hi_um"],
            width_um=c["d_hi_um"] - c["d_lo_um"],
            width_pct_of_centre=200.0 * (c["d_hi_um"] - c["d_lo_um"])
            / (c["d_hi_um"] + c["d_lo_um"]),
            binding_upper_bound=c["binding_upper_bound"],
            binding_lower_bound=c["binding_lower_bound"]))
    recommended["corridor_note"] = (
        "the recommended point maximises eta, so by construction it sits where "
        "C7 (upper) and C8 (lower) both nearly bind and its d_eff window is "
        "only %.2f um wide (%.2f%% of the electrode). corridor_vs_kappa shows "
        "the trade: backing kappa off widens the window roughly linearly in "
        "the eta given up. This is a fabrication-tolerance statement, not a "
        "feasibility one."
        % (recommended["corridor_at_this_point"]["d_hi_um"]
           - recommended["corridor_at_this_point"]["d_lo_um"],
           200.0 * (recommended["corridor_at_this_point"]["d_hi_um"]
                    - recommended["corridor_at_this_point"]["d_lo_um"])
           / (recommended["corridor_at_this_point"]["d_hi_um"]
              + recommended["corridor_at_this_point"]["d_lo_um"])))
    recommended["objective"] = (
        "maximise the readout port efficiency eta = 1 - kappa_int/kappa "
        "subject to |chi_Koch|/kappa >= 1, |chi_two_pole|/kappa >= 1 (C8), "
        "(g/Delta)^2 <= 0.01 with Delta = f01 - f_r, "
        f"(g12/Delta_12)^2 <= {G12_OVER_DELTA12_SQ_MAX} (C7), "
        "kappa_r >= kappa_int(f_r) and a Purcell margin T1/30us >= "
        f"{T1_MARGIN_TARGET}; unchanged from the superseded run except for the "
        "corrected chi and the two added constraints")
    recommended["margin_efficiency_front"] = margin_efficiency_front(
        f01, alpha, recommended["Delta_r_GHz"], r=r)
    recommended["deviation_vs_exploration_precomputation"] = {
        "exploration_point": expl_koch,
        "src": ("out/m6/explore/missing_t1_channels.json:verdict."
                "question_c_retune.best_trade_at_eta_0p2.koch_convention"),
        "reproduced_here": {
            "chi_koch_MHz": 1e3 * _cav.chi_dispersive_koch(
                expl_koch["g_MHz"] * 1e-3, f01, expl_koch["f_r_GHz"], alpha),
            "chi_two_pole_MHz": 1e3 * _cav.chi_two_pole(
                expl_koch["g_MHz"] * 1e-3, f01, f12, expl_koch["f_r_GHz"], r),
            "chi_over_kappa_koch": abs(_cav.chi_dispersive_koch(
                expl_koch["g_MHz"] * 1e-3, f01, expl_koch["f_r_GHz"], alpha))
                / (expl_koch["kappa_MHz"] * 1e-3),
            "eta_readout": 1.0 - kappa_intrinsic_MHz(expl_koch["f_r_GHz"])
                / expl_koch["kappa_MHz"],
            "eta_reldev": abs(1.0 - kappa_intrinsic_MHz(expl_koch["f_r_GHz"])
                              / expl_koch["kappa_MHz"]
                              - expl_koch["eta_readout"])
                / expl_koch["eta_readout"],
            "g12_over_Delta12_sq": (ratio * expl_koch["g_MHz"] * 1e-3
                                    / (expl_koch["f_r_GHz"] - f12)) ** 2,
        },
        "verdict": (
            "the exploration point is reproduced to 1e-15 on every quantity it "
            "reports, and is then REJECTED by the new constraint C7: its "
            "(g12/Delta_12)^2 = 0.01858 is 1.86x the cap. The rejection is not "
            "marginal -- with C7 enforced the largest kappa admitting any g at "
            "f_r = 39.2589 GHz is "
            f"{float(kappa_max_MHz(expl_koch['f_r_GHz'] - f01, alpha, r, T1_MARGIN_TARGET)):.4f}"
            " MHz, below kappa_int = "
            f"{kappa_intrinsic_MHz(expl_koch['f_r_GHz']):.4f} MHz, so that f_r "
            "supports no operating point at all. This export therefore "
            "SUPERSEDES both the exploration's eta ~ 0.20 trade point and the "
            "ruling's description of it as 'marginal'."),
        "deviation_of_this_recommendation": {
            "d_f_r_GHz": recommended["f_r_GHz"] - expl_koch["f_r_GHz"],
            "d_d_eff_um": recommended["d_eff_um"] - expl_koch["d_eff_um"],
            "d_kappa_MHz": recommended["kappa_MHz"] - expl_koch["kappa_MHz"],
            "d_eta": recommended["eta_readout"] - expl_koch["eta_readout"],
            "eta_ratio": recommended["eta_readout"] / expl_koch["eta_readout"],
        },
    }
    empty_proof = above_f01_emptiness_proof(f01, alpha, r)

    geom = geometry_concept(recommended["f_r_GHz"])
    geom_brief = geometry_concept(p2["readout_sample_reference"]["f_r_GHz"])
    geom["a_r_at_brief_sample_47p6GHz_um"] = geom_brief["maxwell_scaling"]["a_r_um"]
    geom["gap_positions_at_brief_sample"] = geom_brief["gap_positions"]
    geom["purcell_filter_N"] = {
        "brief_grid": [purcell_filter_N(k, p2["readout_sample_reference"]["f_r_GHz"],
                                        Q_i=Q_I) for k in (1.0, 3.0)],
        "at_recommended": purcell_filter_N(recommended["kappa_MHz"],
                                           recommended["f_r_GHz"], Q_i=Q_I),
        "status": "DESIGN ESTIMATE",
        "caveat": (
            "the ladder fit ln Q_w = sigma N + b is measured from TWO points "
            "(N = 3, 4; photonics.json flags trivial_r2_warning), and it is "
            "applied at a_r by Maxwell scale invariance of Q_w (a purely "
            "geometric/radiative quantity). No FDTD ladder has been run at a_r."),
    }
    geom["purcell_filter_element"] = purcell_filter_option(
        recommended["Delta_r_GHz"])

    ctrl = control_line_budget(recommended, f01)
    ctrl["pair_breaking"] = pair_breaking_restatement(p3)

    n_naive = len(naive)
    env = operating_envelope(design, recommended, p2, p3, n_naive, r=r,
                             empty_proof=empty_proof, envelope=envelope,
                             window=window)

    n_corr_brief_grid = sum(r_["feasible"] for r_ in rows_brief_grid)
    n_corr_full = sum(r_["feasible"] for r_ in rows)
    n_corr_achievable = sum(r_["feasible_achievable"] for r_ in rows)
    n_ref_brief_grid = sum(r_["feasible"] for r_ in ref_rows
                           if r_["d_eff_um"] in (100.0, 200.0))
    n_ref_full = sum(r_["feasible"] for r_ in ref_rows)
    n_ref_achievable = sum(r_["feasible_achievable"] for r_ in ref_rows)
    n_gap_feas = sum(r_["feasible_achievable"] for r_ in gap_rows)

    payload = {
        "chi_convention": {
            "status": "CORRECTED 2026-07-27 (reference-inherited sign defect)",
            "ruling": "out/verify/chi_convention_ruling.md",
            "production_formula": ("cavity.chi_dispersive_koch(g, f01, f_r, "
                                   "alpha), Delta = f01 - f_r"),
            "truth_standard": ("cavity.chi_two_pole(g, f01, f12, f_r, r), "
                               "r = (n12/n01)^2 exact"),
            "superseded_formula": ("cavity.chi_dispersive(g, f_r - f01, alpha) "
                                   "-- retained, never called in production"),
            "ladder": {
                "n01_recomputed": n01_ladder, "n12_recomputed": n12,
                "n12_over_n01": ratio, "n_ratio_sq": r,
                "src": ("recomputed from qubit/spectrum.py via "
                        "cavity.charge_matrix_elements"),
                "p6_gates_ladder_ratio": p6["params"]["ladder_ratio"],
                "reldev_vs_p6": abs(ratio - p6["params"]["ladder_ratio"])
                / p6["params"]["ladder_ratio"],
                "n01_reldev_vs_design_point": abs(n01_ladder - n01) / n01,
            },
            "selfcheck": chi_convention_selfcheck(n01, f01, alpha, r),
            "impact": (
                "the superseded recommended point (f_r = 44.5966 GHz, "
                "chi/kappa = 1.0092) fails: its corrected chi/kappa is 0.424 "
                "(Koch) / 0.466 (two-pole). The whole f_r > f01 window is empty "
                "analytically. The design moves below f01 into the "
                "gap-protection window."),
        },
        "gap_protection_window": {
            "f_r_GHz": [float(window[0]), float(window[1])],
            "derivation": ("f01 lies inside the Maxwell-scaled readout "
                           "lattice's TE gap iff gap_lo f_r/f_cav <= f01 <= "
                           "gap_hi f_r/f_cav, i.e. f01 f_cav/gap_hi <= f_r <= "
                           "f01 f_cav/gap_lo"),
            "gap_GHz": list(GAP_GHZ), "f_cav_GHz": F_CAV_GHZ, "f01_GHz": f01,
            "exploration_value": expl_win,
            "reldev_vs_exploration": [abs(a - b) / b
                                      for a, b in zip(window, expl_win)],
            "src": ("recomputed from photonics.json:modes.cavity_line_gap_GHz; "
                    "cross-checked against out/m6/explore/"
                    "missing_t1_channels.json:verdict.question_c_retune"),
            "meaning": ("inside this window the readout mirror region is a "
                        "photonic bandgap AT the qubit frequency, which is what "
                        "the missing-T1-channel radiation finding asks for. The "
                        "superseded recommended point at 44.5966 GHz lies "
                        "outside it; the corrected one lies inside."),
        },
        "reference_convention_scan": {
            "flag": ("REFERENCE CONVENTION, RETAINED, NOT A DESIGN WINDOW -- "
                     "every chi below was produced with the inverted detuning "
                     "Delta = f_r - f01 (cavity.chi_dispersive). Reproduced "
                     "bit-for-bit from the superseded artifact so nothing is "
                     "deleted; superseded for every design purpose by the "
                     "corrected blocks. Mirrors the AT-6 small-frequency "
                     "precedent (paper/decisions.md 2026-07-27 M6 P3)."),
            "ruling": "out/verify/chi_convention_ruling.md Sec. 3(e)",
            "grid": {
                "Delta_r_GHz": [float(x) for x in DETUNINGS_GHZ],
                "d_eff_readout_um": [d * 1e6 for d in D_EFF_READOUT_M],
                "kappa_r_MHz": list(KAPPA_READOUT_MHZ),
                "n_points": len(ref_rows),
            },
            "counts": {
                "naive_feasible": n_naive,
                "feasible_same_grid": n_ref_brief_grid,
                "feasible_extended_grid": n_ref_full,
                "feasible_and_achievable": n_ref_achievable,
            },
            "rows": ref_rows,
            "naive_rows_for_reference": naive,
            "recommended_point": ref_rec,
            "recommended_point_search": ref_stats,
            "corrected_values_at_this_point": {
                "chi_koch_MHz": 1e3 * _cav.chi_dispersive_koch(
                    ref_rec["g_MHz"] * 1e-3, f01, ref_rec["f_r_GHz"], alpha),
                "chi_two_pole_MHz": 1e3 * _cav.chi_two_pole(
                    ref_rec["g_MHz"] * 1e-3, f01, f12, ref_rec["f_r_GHz"], r),
                "chi_over_kappa_koch": abs(_cav.chi_dispersive_koch(
                    ref_rec["g_MHz"] * 1e-3, f01, ref_rec["f_r_GHz"], alpha))
                    / (ref_rec["kappa_MHz"] * 1e-3),
                "chi_over_kappa_two_pole": abs(_cav.chi_two_pole(
                    ref_rec["g_MHz"] * 1e-3, f01, f12, ref_rec["f_r_GHz"], r))
                    / (ref_rec["kappa_MHz"] * 1e-3),
                "meets_chi_over_kappa_min": False,
                "kappa_ceiling_corrected_MHz": float(kappa_ceiling_MHz(
                    ref_rec["Delta_r_GHz"], alpha)),
                "verdict": ("FAILS chi/kappa >= 1 under both corrected "
                            "constructions; the point is withdrawn"),
            },
        },
        "above_f01_emptiness_proof": empty_proof,
        "gap_window_scan": {
            "window_f_r_GHz": [float(window[0]), float(window[1])],
            "coarse_grid": {
                "f_r_GHz_step": F_R_COARSE_STEP_GHZ,
                "d_eff_readout_um": [d * 1e6 for d in D_EFF_COARSE_M],
                "kappa_r_MHz": list(KAPPA_COARSE_MHZ),
                "n_points": len(gap_rows),
                "n_feasible_achievable": n_gap_feas,
            },
            "rows": gap_rows,
            "envelope": envelope,
            "koch_only_optimum": {
                "flag": ("TWO_POLE_SCREEN DISABLED -- reaches a higher eta but "
                         "its exact two-pole chi/kappa is below 1, so it is NOT "
                         "recommended. Exported for completeness and so the "
                         "cost of the C8 screen is explicit."),
                "point": koch_only, "search": koch_only_stats,
                "eta_cost_of_two_pole_screen": (
                    koch_only["eta_readout"] - recommended["eta_readout"]),
            },
        },
        "readout_scan_beta": {
            "grid": {
                "Delta_r_GHz": [float(x) for x in DETUNINGS_GHZ],
                "d_eff_readout_um": [d * 1e6 for d in D_EFF_READOUT_M],
                "kappa_r_MHz": list(KAPPA_READOUT_MHZ),
                "n_points": len(rows),
                "brief_grid_note": ("the naive AT-5 scan used d_eff in "
                                    "{100, 200} um only; d_eff = 50 um is a P5 "
                                    "addition"),
            },
            "chi_mode": "koch",
            "criteria": {
                "chi_over_kappa_min": CHI_OVER_KAPPA_MIN,
                "T1_Purcell_min_us": T1_PURCELL_MIN_S * 1e6,
                "g_over_Delta_sq_max": G_OVER_DELTA_SQ_MAX,
                "g12_over_Delta12_sq_max": G12_OVER_DELTA12_SQ_MAX,
                "kappa_achievable": "kappa_r >= f_r/Q_i (P5 addition)",
            },
            "counts": {
                "naive_feasible": n_naive,
                "corrected_feasible_same_grid": n_corr_brief_grid,
                "corrected_feasible_extended_grid": n_corr_full,
                "corrected_feasible_and_achievable": n_corr_achievable,
                "reference_convention_feasible_same_grid": n_ref_brief_grid,
                "reference_convention_feasible_extended_grid": n_ref_full,
                "verdict": (
                    f"under the REFERENCE convention the naive {n_naive}-point "
                    f"window became {n_ref_brief_grid} on the same grid "
                    f"({n_ref_full} with the 50 um column) and "
                    f"{n_ref_achievable} once the measured kappa floor was "
                    "enforced. Under the CORRECTED chi the same mandated grid "
                    f"gives {n_corr_brief_grid} / {n_corr_full} / "
                    f"{n_corr_achievable}: the f_r > f01 window is empty at "
                    "every stage, and `above_f01_emptiness_proof` shows it is "
                    "empty for every detuning and every d_eff, not merely on "
                    "this grid."),
            },
            "beta_bracket_counts": bracket,
            "rows": rows,
            "naive_rows_for_reference": naive,
        },
        "recommended_point": recommended,
        "recommended_point_search": fine_stats,
        "geometry": geom,
        "control_line": ctrl,
        "operating_envelope": env,
        "params": {
            "n01": n01, "f01_GHz": f01, "anharm_GHz": alpha,
            "EJ_over_EC": design["EJ_over_EC"], "EC_GHz": design["EC_GHz"],
            "C_Sigma_fF": _cpl.C_SIGMA_F * 1e15, "C_J_fF": _cpl.C_J_F * 1e15,
            "C_c_cap_fF": _cpl.c_c_cap() * 1e15,
            "kappa_geo": _cpl.KAPPA_GEO,
            "kappa_geo_range": list(_cpl.KAPPA_GEO_RANGE),
            "eps_eff": _cpl.EPS_EFF,
            "eps_eff_range": list(_cpl.EPS_EFF_RANGE),
            "d_eff_drive_um": D_EFF_DRIVE_M * 1e6,
            "f_cav_GHz": F_CAV_GHZ, "Q_i": Q_I,
            "Q_i_optimistic_ceiling": q_i_ceiling(),
            "V_mode_m3": V_MODE_M3, "eps_core": EPS_C,
            "a_um": A_UM, "gap_GHz": list(GAP_GHZ),
            "ladder_sigma": LADDER_SIGMA, "ladder_b": LADDER_B,
            "T1_Purcell_min_us": T1_PURCELL_MIN_S * 1e6,
            "T1_margin_target": T1_MARGIN_TARGET,
            "f12_GHz": f12,
            "n12_over_n01": ratio, "n_ratio_sq": r,
            "g12_over_Delta12_sq_max": G12_OVER_DELTA12_SQ_MAX,
            "two_pole_screen": bool(TWO_POLE_SCREEN),
            "pole_guard_GHz": POLE_GUARD_GHZ,
            "gap_protection_window_GHz": [float(window[0]), float(window[1])],
            "corrected_fine_grid": {
                "f_r_GHz_step": F_R_FINE_STEP_GHZ,
                "d_eff_um": [float(D_EFF_CORRECTED_FINE_UM[0]),
                             float(D_EFF_CORRECTED_FINE_UM[-1]),
                             float(D_EFF_CORRECTED_FINE_UM[1]
                                   - D_EFF_CORRECTED_FINE_UM[0])],
                "kappa_MHz": [float(KAPPA_CORRECTED_FINE_MHZ[0]),
                              float(KAPPA_CORRECTED_FINE_MHZ[-1]),
                              float(KAPPA_CORRECTED_FINE_MHZ[1]
                                    - KAPPA_CORRECTED_FINE_MHZ[0])],
                "note": "[start, stop, step]; blocked over f_r, block = %d"
                        % FINE_BLOCK,
            },
            "free_parameters": {
                "kappa_geo": "P2 pickup shape factor, 1.0-4.0, default 2.0",
                "eps_eff": "P2 pickup permittivity, 6.34-11.68, default 6.34",
                "T1_Purcell_min": "design criterion, 30 us (M6_BRIEF.md Sec. 2)",
                "T1_margin_target": "P5 recommendation criterion, 1.2x",
                "filter_bandwidth": "Purcell-filter option, 100-500 MHz, unmeasured",
                "g12_over_Delta12_sq_max": (
                    "C7 1-2-line dispersivity cap, 0.01 -- the same value the "
                    "design already imposes on the 0-1 line, not a new number"),
            },
        },
        "provenance": {
            "task": "M6 deliverable P5 -- two-cavity chip feasibility + control-line budget",
            "module": "qubit/two_cavity.py",
            "deterministic": True,
            "rng_seed": SEED,
            "rng_note": "no stochastic step: closed-form algebra + deterministic grid sweeps",
            "reads": {
                "n01/f01/anharm/EC": "recomputed by qubit.design_point()",
                "C_Sigma_fF": "qubit_targets.json:design.C_fF",
                "C_J_fF": "junction.json:contract.C_fF",
                "f_cav_GHz/Q_i/V_mode_m3/d_eff_drive_um": "qubit_targets.json:inputs",
                "a_um/r_over_a/t_over_a/eps_core/tan_delta": "photonics.json:geometry",
                "cavity_line_gap_GHz/V_mode_a3/nu0/Q_rad_intrinsic/Q_mat":
                    "photonics.json:modes",
                "outer_mirror_rows/chip_scale": "photonics.json:geometry.cavity_detail, geometry.chip_scale",
                "ladder sigma/b/usable_N": "photonics.json:ladder.exponential_fit",
                "Q_i_anchor": "photonics.json:ladder.Q_i_anchor.value",
                "beta model / C_c / budget cap": "qubit/coupling.py (P2), out/m6/p2_coupling.json",
                "pair-breaking audit": "out/m6/p3_qp.json:pair_breaking_audit",
                "chi / E_zpf / Purcell formulas": (
                    "qubit/cavity.py -- chi_dispersive_koch (production), "
                    "chi_two_pole (truth standard), chi_dispersive (reference "
                    "convention, retained unmodified)"),
                "n12/n01": ("recomputed via cavity.charge_matrix_elements from "
                            "qubit/spectrum.py; cross-checked against "
                            "out/m6/p6_gates.json:params.ladder_ratio"),
                "gap-protection window": (
                    "recomputed from photonics.json:modes.cavity_line_gap_GHz; "
                    "cross-checked against out/m6/explore/"
                    "missing_t1_channels.json:verdict.question_c_retune"),
                "chi convention": "out/verify/chi_convention_ruling.md",
            },
            "unmodified": ["qubit/inputs.py", "qubit/coupling.py",
                           "qubit/spectrum.py", "qubit/dome.py", "qubit/qp.py",
                           "qubit/qp_kernel.py"],
            "modified_this_revision": {
                "qubit/cavity.py": (
                    "ADDITIVE ONLY -- chi_dispersive_koch, chi_two_pole, "
                    "charge_matrix_elements, n_ratio_sq added; chi_dispersive "
                    "and both its call sites (drive_cavity_budget, "
                    "readout_window_scan) unchanged in behaviour, docstring "
                    "warning added. AT-4/AT-5 remain bit-faithful."),
                "qubit/two_cavity.py": (
                    "production chi repointed to the corrected formulas; "
                    "constraints C7/C8 added; scan extended below f01 into the "
                    "gap-protection window; superseded scan retained under "
                    "reference_convention_scan"),
            },
            "claim_discipline": (
                "DESIGN STUDY. Every geometry number is a Maxwell rescale of a "
                "measured photonics.json quantity or an explicitly flagged "
                "design estimate; no new FDTD/MPB run was performed for P5, and "
                "no absolute coherence claim is made."),
        },
    }
    return payload


def main():
    payload = build_export()
    outdir = ROOT / "out" / "m6"
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "p5_two_cavity.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return path, payload


if __name__ == "__main__":
    p, pay = main()
    c = pay["readout_scan_beta"]["counts"]
    print(f"wrote {p}")
    print(f"feasible: naive {c['naive_feasible']} -> corrected "
          f"{c['corrected_feasible_same_grid']} (same grid) / "
          f"{c['corrected_feasible_extended_grid']} (with 50 um) / "
          f"{c['corrected_feasible_and_achievable']} achievable")
    r = pay["recommended_point"]
    ap = pay["above_f01_emptiness_proof"]["analytic"]
    print("above f01: sup kappa_ceiling = %.4f MHz < kappa_int(Dmin) = %.4f MHz"
          " -> window EMPTY" % (ap["sup_kappa_ceiling_MHz"],
                                ap["kappa_int_at_mandated_Dmin_MHz"]))
    print("recommended: Delta_r=%.4f GHz f_r=%.5f GHz d_eff=%.1f um "
          "kappa=%.3f MHz" % (r["Delta_r_GHz"], r["f_r_GHz"], r["d_eff_um"],
                              r["kappa_MHz"]))
    print("  g=%.2f MHz chi_koch=%.4f MHz chi_2pole=%.4f MHz "
          "chi/kappa=%.4f / %.4f T1_P=%.2f us" % (
              r["g_MHz"], r["chi_MHz"], r["chi_two_pole_MHz"],
              r["chi_over_kappa"], r["chi_over_kappa_two_pole"],
              r["T1_Purcell_us"]))
    print("  (g/D)^2=%.3e (g12/D12)^2=%.5f beta_r=%.4f" % (
        r["g_over_Delta_sq"], r["g12_over_Delta12_sq"], r["beta_readout"]))
    print("  C_c sum=%.3f fF (cap %.3f) eta=%.4f Q_w=%.0f" % (
        r["C_c_sum_fF"], r["C_c_cap_fF"], r["eta_readout"], r["Q_w_required"]))
    g = pay["geometry"]["maxwell_scaling"]
    print("a_r = %.2f um (%.4f mm)" % (g["a_r_um"], g["a_r_mm"]))
    for b in pay["control_line"]["rabi_budget"]:
        print("  Omega_R/2pi=%g MHz -> P_in=%.4g W (%.2f dBm), n=%.4g"
              % (b["Omega_R_over_2pi_MHz"], b["P_in_W"], b["P_in_dBm"],
                 b["n_drive"]))
    lk = pay["control_line"]["leakage"]
    print("  leakage control->drive cavity: %.2f dB"
          % lk["control_tone_into_drive_cavity"]["suppression_dB"])
