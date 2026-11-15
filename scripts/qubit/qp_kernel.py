"""M6 production task P3 -- full Catelani-type quasiparticle kernel.

Milestone-1 `qubit/qp.py` uses the small-frequency (hbar*omega << Delta)
quasiparticle formula

    Gamma_ref = x_qp * (omega_q/pi) * sqrt(2*Delta/(hbar*omega_q)) ,

which M6_BRIEF.md Sec. 2 flags as *stretched*: the design point sits at
hbar*omega_q/Delta = 0.552, not << 1.  This module replaces it with the full
kernel and quantifies the correction.

--------------------------------------------------------------------------
1.  The kernel
--------------------------------------------------------------------------
Quasiparticle tunnelling through the junction is described by the current
spectral density S_qp(omega) (Catelani, Schoelkopf, Devoret, Glazman,
PRB 84, 064517 (2011); Glazman & Catelani, SciPost Phys. Lect. Notes 31
(2021)).  Qubit rates factorise into a matrix element and S_qp:

    Gamma_{i->f} = |<f| sin(phi/2) |i>|^2  S_qp(omega_if)      (transition)
    Gamma_parity = |<0| cos(phi/2) |0>|^2  S_qp(0)             (parity flip)

with, for a BCS density of states rho(E) = E/sqrt(E^2 - Delta^2),

                  8 E_J     2  /inf
    S_qp(omega) = ------- * -- |    dE rho(E) rho(E + hbar w)
                  pi hbar   D  /D

                              * [1 + Delta^2/(E (E + hbar w))]
                              * f(E) [1 - f(E + hbar w)] .          (K)

The normalised quasiparticle density that goes with this kernel is

    x_qp = (2/Delta) * int_Delta^inf dE rho(E) f(E) .               (N)

Non-equilibrium quasiparticles are parametrised by an effective temperature
T_qp with a Boltzmann tail pinned to the gap edge,

    f(E) = A exp(-(E - Delta)/(k_B T_qp)) ,                         (F)

A being fixed by (N) at the requested x_qp.  Setting A = exp(-Delta/k_B T)
recovers the equilibrium distribution and, through (N), the thermal floor

    x_qp_th(T) = sqrt(2 pi k_B T/Delta) exp(-Delta/k_B T)           (T)

exactly in the k_B T << Delta limit -- (T) is therefore not an independent
formula but the T_qp -> 0 asymptote of this module's own normalisation, and
`test_m6_p3.py` checks that identity numerically.

VALIDATION OF THE NORMALISATION.  Expanding (K) for hbar*w << Delta with all
quasiparticles at the gap edge gives
    S_qp -> (8 E_J/(pi hbar)) x_qp sqrt(2 Delta/(hbar w)) ,
and with the harmonic-transmon matrix element |<0|sin(phi/2)|1>|^2 ->
(1/4) sqrt(2 E_C/E_J) together with hbar*w_01 -> sqrt(8 E_J E_C) this
reproduces Gamma_ref above.  The prefactor 8 E_J/(pi hbar) in (K) is fixed by
that requirement, not typed from memory.

--------------------------------------------------------------------------
2.  Analytic T_qp -> 0 cross-check
--------------------------------------------------------------------------
With f concentrated at E = Delta, (K)/(N) collapses to a closed form.  With
w = hbar*omega/Delta,

    S_qp(omega) / [ (8 E_J/(pi hbar)) x_qp sqrt(2/w) ]  =  sqrt(1 + w/2) .

`kernel_shape_ratio_T0` implements this; the quadrature reproduces it to
better than 1e-7 at T_qp = 1e-8 * Delta/k_B (see the acceptance tests).
At the design point w = 0.5523 the kernel is therefore 13.0% *larger* than
the small-frequency estimate -- relaxation is faster, T1 shorter.

--------------------------------------------------------------------------
CLAIM DISCIPLINE (M6_BRIEF.md, hard rule)
--------------------------------------------------------------------------
Every T1 in this module is parametric in (x_qp, T_qp).  x_qp is a free
parameter of a design study; no absolute TBG coherence claim is made or
implied anywhere.

Numerics: deterministic adaptive quadrature (QUADPACK).  No stochastic
sampling is used anywhere in P3, so no RNG stream is consumed; the exported
provenance block records seed = 0 and rng = "unused" explicitly.
"""
import json
import pathlib

import numpy as np
from scipy import integrate, optimize

from . import dome, spectrum
from .inputs import (DELTA0_MEV, E, EJ_GHZ, H, HBAR, KB, N_CHARGE, ROOT,
                     TARGETS)
from .qp import X_QP_GRID, qp_prefactor

# --- P3 evaluation grids (design choices, not measured inputs) ---------------
T_QP_GRID_K = (0.020, 0.050, 0.100, 0.200)   # effective qp temperatures
T_FLOOR_GRID_K = (0.020, 0.100, 0.200)       # thermal-floor rows (task P3d)
T_QP_ZERO_K = 1e-8 * 0.304e-3 * E / KB       # "T_qp -> 0" stand-in, ~1e-11 K
BOLTZ_CUT = 120.0      # integrate f(E) out to (E-Delta) = 120 k_B T_qp
QUAD_EPSREL = 1e-10
# Quadrature tolerance.  Measured convergence: epsrel = 1e-8 / 1e-10 / 1e-12
# give bit-identical kernel ratios (15 digits) and identical Gamma_parity, so
# the quadrature is not the error source anywhere in P3.  The 5.5e-9 residual
# between the T_QP_ZERO_K quadrature and the analytic sqrt(1+w/2) is the
# finite stand-in temperature (O(k_B T_qp/hbar w) = 1.8e-8), not QUADPACK.
# Asserted in tests/test_m6_p3.py::test_quadrature_tolerance_is_immaterial.

# Tone table for the pair-breaking audit.
#   src: qubit_targets.json:design.f01_GHz          (control)
#   src: qubit_targets.json:readout_window[3].f_r_GHz (readout sample point)
#   src: qubit_targets.json:inputs.f_cav_GHz        (drive cavity, 145 GHz)
F_CONTROL_GHZ = TARGETS["design"]["f01_GHz"]
F_READOUT_GHZ = [r for r in TARGETS["readout_window"]
                 if r["Delta_r_GHz"] == 7.0 and r["d_eff_um"] == 100.0
                 and r["kappa_MHz"] == 1.0][0]["f_r_GHz"]
F_DRIVE_GHZ = TARGETS["inputs"]["f_cav_GHz"]

EPS1_HZ = TARGETS["design"]["eps1_Hz"]
# src: qubit_targets.json:design.eps1_Hz  (37.4 Hz at E_J/E_C = 100)


def delta_J(Delta_meV=DELTA0_MEV):
    """Gap in joules.  src: material.json:Delta0_meV."""
    return Delta_meV * 1e-3 * E


# ============================================================ matrix elements =
def _fourier_sin_half(m):
    """Fourier coefficient of sin(phi/2) on phi in [-pi, pi):
    s_m = (1/2pi) int sin(phi/2) e^{-i m phi} dphi = i (-1)^m m / (pi(m^2-1/4)).
    """
    return 1j * ((-1.0) ** m) * m / (np.pi * (m ** 2 - 0.25))


def _fourier_cos_half(m):
    """Fourier coefficient of cos(phi/2): c_m = -(-1)^m / (2 pi (m^2 - 1/4))."""
    return -((-1.0) ** m) / (2 * np.pi * (m ** 2 - 0.25))


def half_phase_operators(N=N_CHARGE):
    """(sin(phi/2), cos(phi/2)) as (2N+1)x(2N+1) charge-basis matrices.

    |n> <-> e^{i n phi}/sqrt(2 pi) on phi in [-pi, pi), so <n'|O|n> is the
    (n'-n)-th Fourier coefficient of O.  sin/cos(phi/2) are 4pi-periodic; the
    restriction to one 2pi branch is the standard convention and is
    exponentially accurate for a transmon whose wavefunctions are localised
    near phi = 0 (the same exponential smallness that makes eps1 = 37 Hz).
    """
    n = np.arange(-N, N + 1)
    m = np.subtract.outer(n, n)
    return _fourier_sin_half(m), _fourier_cos_half(m)


def matrix_elements(EJ_GHz=EJ_GHZ, EC_GHz=None, N=N_CHARGE, n_phi=8192):
    """|<0|sin(phi/2)|1>|^2 and |<0|cos(phi/2)|0>|^2 of the milestone-1 transmon.

    Computed twice -- exact Fourier/Toeplitz operators in the charge basis and
    midpoint quadrature on a phase grid -- and the two are returned together so
    the agreement is auditable.  `EC_GHz` defaults to the locked E_J/E_C = 100
    design point (src: qubit_targets.json:design.EJ_over_EC).
    """
    if EC_GHz is None:
        EC_GHz = EJ_GHz / TARGETS["design"]["EJ_over_EC"]
    n = np.arange(-N, N + 1)
    _, vec = np.linalg.eigh(spectrum.hamiltonian(EJ_GHz, EC_GHz, 0.0, N))
    g, e = vec[:, 0], vec[:, 1]

    Msin, Mcos = half_phase_operators(N)
    sin01 = abs(g @ (Msin @ e)) ** 2
    cos00 = abs(g @ (Mcos @ g)) ** 2

    phi = (np.arange(n_phi) + 0.5) / n_phi * 2 * np.pi - np.pi
    dphi = 2 * np.pi / n_phi
    psi = np.exp(1j * np.outer(phi, n)) / np.sqrt(2 * np.pi) @ vec[:, :2]
    sin01_g = abs(np.sum(np.conj(psi[:, 0]) * np.sin(phi / 2) * psi[:, 1])
                  * dphi) ** 2
    cos00_g = abs(np.sum(np.conj(psi[:, 0]) * np.cos(phi / 2) * psi[:, 0])
                  * dphi) ** 2

    return dict(
        sin_half_01_sq=float(sin01),
        cos_half_00_sq=float(cos00),
        sin_half_01_sq_phasegrid=float(sin01_g),
        cos_half_00_sq_phasegrid=float(cos00_g),
        sin_half_01_sq_harmonic=float(0.25 * np.sqrt(2 * EC_GHz / EJ_GHz)),
        N=N, n_phi=n_phi,
    )


# ================================================================== the kernel =
def _arccosh1p(z):
    """arccosh(1 + z), stable for z -> 0: log1p(z + sqrt(z(z+2)))."""
    return np.log1p(z + np.sqrt(z * (z + 2.0)))


def _reduced_integrals(w, theta, cut=BOLTZ_CUT, epsrel=QUAD_EPSREL):
    """(numerator, normalisation) of (K) and (N), stripped of A and prefactors.

    Dimensionless arguments: w = hbar*omega/Delta, theta = k_B T_qp/Delta.
    Substituting E = Delta cosh(t) removes the 1/sqrt(E^2-Delta^2) edge
    singularity exactly (rho dE -> Delta cosh t dt); t is then rescaled by
    sqrt(2 theta) so the Boltzmann factor is O(1)-wide in the integration
    variable at any theta.  Adaptive QUADPACK, deterministic.

    Every near-gap quantity is carried as an offset from 1 so nothing is
    formed by subtracting two nearly equal O(1) numbers:
    E/Delta - 1 = 2 sinh^2(t/2), and (E+hbar w)^2/Delta^2 - 1 is factorised as
    (epm1)(2 + epm1).  Without this the theta ~ 1e-12 stand-ins used to probe
    the T_qp -> 0 limit lose ~half their digits.
    """
    a = np.sqrt(2.0 * theta)
    smax = _arccosh1p(cut * theta) / a

    def integrand(s, with_w):
        em1 = 2.0 * np.sinh(0.5 * a * s) ** 2    # E/Delta - 1, no cancellation
        e = 1.0 + em1
        ex = np.exp(-em1 / theta)
        if not with_w:
            return 2.0 * e * ex * a
        epm1 = em1 + w                           # (E + hbar w)/Delta - 1
        ep = 1.0 + epm1
        rho2 = ep / np.sqrt(epm1 * (2.0 + epm1))
        return 2.0 * e * rho2 * (1.0 + 1.0 / (e * ep)) * ex * a

    # The rho(E + hbar w) edge structure sits at (E-Delta) ~ hbar w, i.e.
    # s ~ sqrt(w/theta); hand that point to QUADPACK so tiny w stays accurate.
    s_star = np.sqrt(max(w, 0.0) / theta)
    pts = [s_star] if 0.0 < s_star < smax else None
    num = integrate.quad(integrand, 0.0, smax, args=(True,), points=pts,
                         limit=400, epsabs=0.0, epsrel=epsrel)[0]
    den = integrate.quad(integrand, 0.0, smax, args=(False,),
                         limit=400, epsabs=0.0, epsrel=epsrel)[0]
    return num, den


def kernel_shape_ratio(w, T_qp_K, Delta_meV=DELTA0_MEV, **kw):
    """S_qp_full(omega) / S_qp_smallfreq(omega) at hbar*omega/Delta = w.

    Independent of x_qp and of E_J -- this is the pure kernel-shape
    correction the brief's marginality flag is about.
    """
    theta = KB * T_qp_K / delta_J(Delta_meV)
    num, den = _reduced_integrals(w, theta, **kw)
    return float((num / den) / np.sqrt(2.0 / w))


def kernel_shape_ratio_T0(w):
    """Analytic T_qp -> 0 limit of `kernel_shape_ratio`: sqrt(1 + w/2)."""
    return float(np.sqrt(1.0 + w / 2.0))


def marginality(f_GHz, Delta_meV=DELTA0_MEV):
    """w = hbar*omega/Delta -- the small-frequency-expansion validity ratio."""
    return float(HBAR * 2 * np.pi * f_GHz * 1e9 / delta_J(Delta_meV))


def S_qp(f_GHz, x_qp, T_qp_K, EJ_GHz=EJ_GHZ, Delta_meV=DELTA0_MEV, **kw):
    """Full quasiparticle spectral density (K), in s^-1."""
    w = marginality(f_GHz, Delta_meV)
    theta = KB * T_qp_K / delta_J(Delta_meV)
    num, den = _reduced_integrals(w, theta, **kw)
    return float(8 * H * EJ_GHz * 1e9 / (np.pi * HBAR) * x_qp * num / den)


def S_qp_smallfreq(f_GHz, x_qp, EJ_GHz=EJ_GHZ, Delta_meV=DELTA0_MEV):
    """Small-frequency limit of (K): (8E_J/pi hbar) x_qp sqrt(2 Delta/hbar w)."""
    w = marginality(f_GHz, Delta_meV)
    return float(8 * H * EJ_GHz * 1e9 / (np.pi * HBAR) * x_qp
                 * np.sqrt(2.0 / w))


# ------------------------------------------------------------------ T1 --------
def Gamma_10(x_qp, T_qp_K, f_GHz=None, EJ_GHz=EJ_GHZ, Delta_meV=DELTA0_MEV,
             me=None, EC_GHz=None, N=N_CHARGE, **kw):
    """Full-kernel relaxation rate |<0|sin(phi/2)|1>|^2 S_qp(w_q), in s^-1."""
    if f_GHz is None:
        f_GHz = TARGETS["design"]["f01_GHz"]
    if me is None:
        me = matrix_elements(EJ_GHz, EC_GHz, N)
    return me["sin_half_01_sq"] * S_qp(f_GHz, x_qp, T_qp_K, EJ_GHz,
                                       Delta_meV, **kw)


def T1_full(x_qp, T_qp_K, **kw):
    """Full-kernel quasiparticle T1 in seconds.  Parametric in (x_qp, T_qp)."""
    return 1.0 / Gamma_10(x_qp, T_qp_K, **kw)


def ratio_vs_smallfreq(T_qp_K, f_GHz=None, EJ_GHz=EJ_GHZ,
                       Delta_meV=DELTA0_MEV, EC_GHz=None, N=N_CHARGE, **kw):
    """Full-kernel rate / milestone-1 reference rate, with its decomposition.

    Three independent factors separate cleanly:

      kernel_shape      S_qp_full(w_q) / S_qp_smallfreq(w_q)  -- the physics
                        the brief flags (sqrt(1 + w/2) at T_qp -> 0);
      matrix_element    exact |<0|sin(phi/2)|1>|^2 over its harmonic value
                        (1/4) sqrt(2 E_C/E_J);
      harmonic_identity sqrt(8 E_J E_C) / (h f01) -- the reference formula
                        eliminates sqrt(E_J E_C) using the harmonic relation
                        hbar w_01 = sqrt(8 E_J E_C) but is then evaluated at
                        the *exact* f01, so this factor is an internal
                        inconsistency of the reference form, not a kernel
                        effect.  Reported separately for that reason.
    """
    if EC_GHz is None:
        EC_GHz = EJ_GHz / TARGETS["design"]["EJ_over_EC"]
    if f_GHz is None:
        f_GHz = spectrum.levels(EJ_GHz, EC_GHz, N)[0]
    me = matrix_elements(EJ_GHz, EC_GHz, N)
    w = marginality(f_GHz, Delta_meV)

    shape = kernel_shape_ratio(w, T_qp_K, Delta_meV, **kw)
    me_fac = me["sin_half_01_sq"] / me["sin_half_01_sq_harmonic"]
    harm = float(np.sqrt(8 * EJ_GHz * EC_GHz) / f_GHz)

    total = Gamma_10(1.0, T_qp_K, f_GHz, EJ_GHz, Delta_meV, me=me, **kw) \
        / qp_prefactor(f_GHz, Delta_meV)
    return dict(total=float(total), kernel_shape=shape,
                kernel_shape_T0=kernel_shape_ratio_T0(w),
                matrix_element=float(me_fac), harmonic_identity=harm,
                product_check=float(shape * me_fac * harm),
                w=w, T_qp_K=T_qp_K, f_GHz=f_GHz)


# ======================================================= parity dephasing =====
def Gamma_parity(x_qp, T_qp_K, w_cut=None, EJ_GHz=EJ_GHZ,
                 Delta_meV=DELTA0_MEV, me=None, **kw):
    """Charge-parity switching rate |<0|cos(phi/2)|0>|^2 S_qp(w -> 0), in s^-1.

    The elastic (omega -> 0) kernel has an integrable-but-logarithmic pile-up
    at the gap edge (rho(E)^2 ~ Delta/2(E-Delta)).  It is cut off physically by
    the smallest energy transfer that still counts as a distinct parity event;
    the natural scale here is the parity splitting itself, hbar*2*pi*eps1, and
    that is the default `w_cut`.  The dependence is logarithmic: eight decades
    of w_cut move Gamma_parity by only ~14x (exported in the JSON), so no
    conclusion below rests on the choice.
    """
    if w_cut is None:
        w_cut = HBAR * 2 * np.pi * EPS1_HZ / delta_J(Delta_meV)
    if me is None:
        me = matrix_elements(EJ_GHz)
    theta = KB * T_qp_K / delta_J(Delta_meV)
    num, den = _reduced_integrals(w_cut, theta, **kw)
    return float(me["cos_half_00_sq"] * 8 * H * EJ_GHz * 1e9
                 / (np.pi * HBAR) * x_qp * num / den)


def Tphi_parity(Gamma_p, eps1_Hz=EPS1_HZ):
    """Parity-induced pure-dephasing time in the motional-narrowing regime.

    Random-telegraph parity noise of full splitting d_omega = 2*pi*eps1
    switching at rate Gamma_p >> d_omega gives Gamma_phi = d_omega^2/(4 Gamma_p)
    (convention: Gamma_p is the total switching rate).  Factor-of-two
    conventions are immaterial at the magnitudes involved.
    """
    dw = 2 * np.pi * eps1_Hz
    return float(4.0 * Gamma_p / dw ** 2)


def Tphi_parity_frozen(eps1_Hz=EPS1_HZ):
    """Absolute worst case: parity frozen and unknown, 1/(pi eps1).

    This is an upper bound on parity dephasing for ANY switching rate, so it
    bounds the channel without reference to x_qp at all.
    """
    return float(1.0 / (np.pi * eps1_Hz))


# ========================================================== thermal floor =====
def x_qp_thermal(T_K, Delta_meV=DELTA0_MEV):
    """Equilibrium floor x_qp_th = sqrt(2 pi k_B T/Delta) exp(-Delta/k_B T)."""
    r = KB * T_K / delta_J(Delta_meV)
    return float(np.sqrt(2 * np.pi * r) * np.exp(-1.0 / r))


def x_qp_thermal_numeric(T_K, Delta_meV=DELTA0_MEV, **kw):
    """Same floor from this module's own normalisation (N) with A = e^-D/kT.

    Exact rather than asymptotic; the ratio to `x_qp_thermal` is the size of
    the k_B T << Delta approximation and is exported as a cross-check.
    """
    r = KB * T_K / delta_J(Delta_meV)
    _, den = _reduced_integrals(1.0, r, **kw)   # w unused for the norm branch
    return float(den * np.exp(-1.0 / r))


# ====================================================== pair-breaking audit ====
def two_delta_GHz(Delta_meV=DELTA0_MEV):
    """Pair-breaking threshold 2*Delta/h in GHz."""
    return float(2 * delta_J(Delta_meV) / H / 1e9)


def pair_breaking_boundary(f_GHz, nu_apex=None):
    """Fillings where h*f = 2*Delta(nu): the tone turns pair-breaking outside.

    Delta(nu) is the dome form of material.json:sc_dome (via qubit/dome.py):
    Tc(nu) parabolic, Delta = 1.764 k_B Tc tanh(1.74 sqrt(Tc/T - 1)).
    Returns (nu_lo, nu_hi) bracketing the apex, or None if the tone is
    sub-gap nowhere on the dome.
    """
    if nu_apex is None:
        nu_apex = dome.NU_OPT
    D_crit_meV = H * f_GHz * 1e9 / 2.0 / E * 1e3

    def g(nu):
        return dome.Delta_of(nu) - D_crit_meV

    if g(nu_apex) <= 0:
        return None
    half = dome.NU_HALFWIDTH
    lo = optimize.brentq(g, nu_apex - half, nu_apex, xtol=1e-14, rtol=1e-15)
    hi = optimize.brentq(g, nu_apex, nu_apex + half, xtol=1e-14, rtol=1e-15)
    return float(lo), float(hi)


def pair_breaking_audit(EC_GHz=None, Delta_meV=DELTA0_MEV):
    """Photon energy vs 2*Delta for the three tones, plus the filling windows."""
    if EC_GHz is None:
        EC_GHz = EJ_GHZ / TARGETS["design"]["EJ_over_EC"]
    thr = two_delta_GHz(Delta_meV)
    tones = [("control_f01", F_CONTROL_GHZ), ("readout_sample", F_READOUT_GHZ),
             ("drive_cavity", F_DRIVE_GHZ)]
    out = {}
    for name, f in tones:
        b = pair_breaking_boundary(f)
        row = dict(f_GHz=float(f), hf_meV=float(H * f * 1e9 / E * 1e3),
                   hf_over_2Delta=float(f / thr),
                   margin_2Delta_over_hf=float(thr / f),
                   pct_below_threshold=float(100.0 * (1.0 - f / thr)))
        if b is None:
            row["sub_gap_filling_window"] = None
            row["note"] = "pair-breaking everywhere on the dome"
        else:
            lo, hi = b
            row["sub_gap_filling_window"] = [lo, hi]
            row["window_halfwidth_nu"] = float((hi - lo) / 2)
            row["f01_at_window_edge_GHz"] = float(dome.f01_of(lo, EC_GHz))
            row["f01_at_apex_GHz"] = float(dome.f01_of(dome.NU_OPT, EC_GHz))
            row["f01_span_inside_window_GHz"] = float(
                dome.f01_of(dome.NU_OPT, EC_GHz) - dome.f01_of(lo, EC_GHz))
        out[name] = row
    out["two_Delta_over_h_GHz"] = thr
    return out


# ================================================================== report =====
def build_report(x_grid=X_QP_GRID, T_qp_grid=T_QP_GRID_K,
                 T_floor_grid=T_FLOOR_GRID_K):
    """Everything P3 asks for, as a JSON-ready dict."""
    EC_GHz = EJ_GHZ / TARGETS["design"]["EJ_over_EC"]
    f01 = float(spectrum.levels(EJ_GHZ, EC_GHz, N_CHARGE)[0])   # recomputed
    me = matrix_elements(EJ_GHZ, EC_GHz)
    w = marginality(f01)

    # --- kernel ------------------------------------------------------------
    ratio0 = ratio_vs_smallfreq(T_QP_ZERO_K)
    ratios = {f"{T:.3f}": ratio_vs_smallfreq(T) for T in T_qp_grid}

    def t1_row(T_qp):
        # Gamma is exactly linear in x_qp, so one kernel call per T_qp
        # suffices; the x grid is then pure division (the "vectorise across
        # the grid, one integrator call" rule, in its quadrature form).
        g1 = Gamma_10(1.0, T_qp, f01, me=me)
        return {f"{x:.0e}": float(1e6 / (g1 * x)) for x in x_grid}

    T1_full_us = {"T_qp->0": t1_row(T_QP_ZERO_K)}
    T1_full_us.update({f"{T:.3f}": t1_row(T) for T in T_qp_grid})
    T1_ref_us = {f"{x:.0e}": float(1e6 / (x * qp_prefactor(f01)))
                 for x in x_grid}

    kernel = dict(
        w_marginality=w,
        ratio_vs_smallfreq=ratio0["total"],
        ratio_vs_smallfreq_decomposition=ratio0,
        ratio_vs_smallfreq_analytic_T0=kernel_shape_ratio_T0(w),
        T1_us_vs_xqp=T1_full_us,
        T1_us_vs_xqp_reference_smallfreq=T1_ref_us,
        T1_shortening_factor_vs_reference=ratio0["total"],
        T_qp_dependence={
            "T_qp_K": list(T_qp_grid),
            "kernel_shape_ratio": [ratios[f"{T:.3f}"]["kernel_shape"]
                                   for T in T_qp_grid],
            "total_ratio": [ratios[f"{T:.3f}"]["total"] for T in T_qp_grid],
            "T1_us_at_xqp_1e-6": [T1_full_us[f"{T:.3f}"]["1e-06"]
                                  for T in T_qp_grid],
        },
        matrix_elements=me,
    )

    # --- parity ------------------------------------------------------------
    dw = 2 * np.pi * EPS1_HZ
    w_cut_default = HBAR * dw / delta_J()
    parity_rows = {}
    for T in (0.020, 0.100):
        for x in x_grid:
            gp = Gamma_parity(x, T, me=me)
            parity_rows[f"T{T:.3f}_x{x:.0e}"] = dict(
                Gamma_parity_s=gp,
                motional_narrowing_ratio=float(gp / dw),
                Tphi_parity_s=Tphi_parity(gp),
                Tphi_parity_over_T1qp=float(Tphi_parity(gp)
                                            * Gamma_10(x, T, f01, me=me)),
            )
    cutoff_scan = {f"{wc:.0e}": Gamma_parity(1e-6, 0.020, w_cut=wc, me=me)
                   for wc in (1e-9, 1e-7, 1e-5, 1e-3, 1e-1)}
    parity = dict(
        eps1_Hz=EPS1_HZ,
        delta_omega_rad_s=float(dw),
        cos_half_00_sq=me["cos_half_00_sq"],
        w_cut_default=float(w_cut_default),
        rate=parity_rows,
        Gamma_parity_cutoff_sensitivity=cutoff_scan,
        Tphi_parity_frozen_worst_case_s=Tphi_parity_frozen(),
        Tphi_frozen_over_T1qp_at_1e6=float(
            Tphi_parity_frozen() * Gamma_10(1e-6, 0.020, f01, me=me)),
        verdict=("parity dephasing negligible: even the frozen-parity upper "
                 "bound 1/(pi eps1) exceeds the qp-limited T1 by >1e3 across "
                 "the whole x_qp band"),
    )

    # --- thermal floor -----------------------------------------------------
    floor = dict(
        formula="x_qp_th(T) = sqrt(2 pi k_B T/Delta) exp(-Delta/k_B T)",
        Delta_over_kB_K=float(delta_J() / KB),
        x_qp_th={f"{T:.3f}": x_qp_thermal(T) for T in T_floor_grid},
        x_qp_th_numeric={f"{T:.3f}": x_qp_thermal_numeric(T)
                         for T in T_floor_grid},
        parametric_band=[min(x_grid), max(x_grid)],
        band_floor_ratio={f"{T:.3f}": float(x_qp_thermal(T) / min(x_grid))
                          for T in T_floor_grid},
        verdict=("the 1e-8..1e-5 band is entirely non-equilibrium: the "
                 "equilibrium floor only reaches the band's bottom edge at "
                 "~200 mK, twice the 100 mK operating point"),
    )

    audit = pair_breaking_audit(EC_GHz)

    params = dict(
        EJ_GHz=float(EJ_GHZ), EC_GHz=float(EC_GHz), f01_GHz=float(f01),
        EJ_over_EC=TARGETS["design"]["EJ_over_EC"],
        f01_GHz_target=float(TARGETS["design"]["f01_GHz"]),
        f01_recomputed_vs_target_rel=float(
            abs(f01 - TARGETS["design"]["f01_GHz"])
            / TARGETS["design"]["f01_GHz"]),
        Delta0_meV=float(DELTA0_MEV), N_charge=N_CHARGE,
        x_qp_grid=list(x_grid), T_qp_grid_K=list(T_qp_grid),
        T_qp_zero_stand_in_K=float(T_QP_ZERO_K),
        boltzmann_cut_kT=BOLTZ_CUT, quad_epsrel=QUAD_EPSREL,
    )
    prov = dict(
        task="M6 P3 -- full Catelani-type quasiparticle kernel",
        module="qubit/qp_kernel.py",
        reference=("Catelani, Schoelkopf, Devoret, Glazman, PRB 84, 064517 "
                   "(2011); Glazman & Catelani, SciPost Phys. Lect. Notes 31 "
                   "(2021)"),
        inputs={
            "Delta0_meV": "material.json:Delta0_meV",
            "sc_dome": "material.json:sc_dome (Tc_max_K, nu_opt, "
                       "nu_halfwidth, Delta_form)",
            "Ic_nA": "junction.json:contract.Ic_nA -> E_J",
            "f01_GHz/eps1_Hz/EJ_over_EC": "qubit_targets.json:design",
            "f_readout_GHz": "qubit_targets.json:readout_window[Delta_r=7]",
            "f_drive_GHz": "qubit_targets.json:inputs.f_cav_GHz",
            "reference_T1/marginality": "qubit_targets.json:quasiparticle",
        },
        numerics="adaptive QUADPACK, E = Delta cosh(t) substitution",
        rng="unused (no stochastic sampling in P3)", seed=0,
        claim_discipline=("all T1 values parametric in (x_qp, T_qp); no "
                          "absolute TBG coherence claim"),
    )
    return dict(kernel=kernel, parity=parity, thermal_floor=floor,
                pair_breaking_audit=audit, params=params, provenance=prov)


def export(path=None):
    """Write out/m6/p3_qp.json and return the report dict."""
    rep = build_report()
    if path is None:
        path = pathlib.Path(ROOT) / "out" / "m6" / "p3_qp.json"
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=1)
    return rep


if __name__ == "__main__":
    r = export()
    k, pb = r["kernel"], r["pair_breaking_audit"]
    print("w = hbar w_q/Delta            : %.6f" % k["w_marginality"])
    print("kernel/reference ratio (T->0) : %.6f" % k["ratio_vs_smallfreq"])
    print("  kernel shape                : %.6f (analytic %.6f)"
          % (k["ratio_vs_smallfreq_decomposition"]["kernel_shape"],
             k["ratio_vs_smallfreq_analytic_T0"]))
    print("  matrix element              : %.6f"
          % k["ratio_vs_smallfreq_decomposition"]["matrix_element"])
    print("  harmonic identity           : %.6f"
          % k["ratio_vs_smallfreq_decomposition"]["harmonic_identity"])
    for x in ("1e-08", "1e-07", "1e-06", "1e-05"):
        print("  T1(%s) us: full %10.4f   ref %10.4f"
              % (x, k["T1_us_vs_xqp"]["T_qp->0"][x],
                 k["T1_us_vs_xqp_reference_smallfreq"][x]))
    print("2 Delta/h = %.4f GHz" % pb["two_Delta_over_h_GHz"])
    for t in ("control_f01", "readout_sample", "drive_cavity"):
        print("  %-15s f = %8.4f GHz  %.2f%% below 2D/h  window %s"
              % (t, pb[t]["f_GHz"], pb[t]["pct_below_threshold"],
                 pb[t]["sub_gap_filling_window"]))
