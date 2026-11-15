"""Circuit-QED budget against the MEASURED photonic-crystal cavity (AT-4, AT-5).

Drive cavity: the real M2 device (f0 = 145 GHz, Q_i = 1.4612e4,
V_mode = 6.68e7 um^3, collection length d_eff = 20 um).  Everything is read
from qubit_targets.json:inputs / photonics.json -- see qubit/inputs.py.

COUPLING CONVENTION (flagged in the brief as deliverable P2, NOT pinned here):

    E_zpf = sqrt(hbar omega / (2 eps0 eps_c V_mode))
    V_zpf = E_zpf * d_eff
    g/h   = e * V_zpf * n01 / h

i.e. the full zero-point voltage drops across the junction with no
voltage-division factor beta.  Pinning beta with a lumped-circuit model of
shunt pad + collection electrode + cavity near-field is production task P2;
this milestone reproduces the reference convention exactly.
"""
import numpy as np

from . import spectrum as _spec
from .inputs import (D_EFF_DRIVE_M, EPS0, EPS_C, E, F_CAV_GHZ, H, HBAR,
                     N_CHARGE, Q_I, V_MODE_M3)

CHI_CONVENTION_RULING = "out/verify/chi_convention_ruling.md"
# The 2026-07-27 adjudication of the dispersive-shift sign convention.  Read it
# before touching any chi in this module: it fixes which detuning sign the
# single formula requires, and which of the three constructions below is the
# truth standard.

# Readout-window scan grid and feasibility thresholds (M6_BRIEF.md Sec. 2).
# Design choices, not measured inputs.
DETUNINGS_GHZ = np.arange(4.0, 16.01, 1.0)
D_EFF_READOUT_M = (100e-6, 200e-6)
KAPPA_READOUT_MHZ = (1.0, 3.0)
CHI_OVER_KAPPA_MIN = 1.0
T1_PURCELL_MIN_S = 30e-6
G_OVER_DELTA_SQ_MAX = 0.01


def E_zpf(f_GHz, V_m3, eps_c=EPS_C):
    """Zero-point electric field in V/m."""
    return np.sqrt(HBAR * 2 * np.pi * f_GHz * 1e9 / (2 * EPS0 * eps_c * V_m3))


def V_zpf(f_GHz, V_m3, d_eff_m, eps_c=EPS_C):
    """Zero-point voltage across the collection length, in V."""
    return E_zpf(f_GHz, V_m3, eps_c) * d_eff_m


def coupling_g(f_GHz, V_m3, d_eff_m, n01, eps_c=EPS_C):
    """g/h in GHz under the flagged convention g = e V_zpf n01 / h."""
    return E * V_zpf(f_GHz, V_m3, d_eff_m, eps_c) * n01 / H / 1e9


def chi_dispersive(g_GHz, Delta_GHz, alpha_GHz):
    """Dispersive shift chi = g^2 alpha / (Delta (Delta + alpha)), GHz.

    REFERENCE-CONVENTION PATH -- DO NOT USE FOR PRODUCTION NUMBERS.

    The formula itself is correct; every call site in the frozen contract
    (`m6_reference.py`, and the AT-4 / AT-5 paths below) feeds it the NEGATED
    detuning ``Delta = f_cav - f01`` instead of the Koch ``Delta = f01 - f_cav``
    that the derivation requires.  With the inverted sign the expression has a
    pole at ``f_r = f01 + |alpha|`` -- a frequency at which the transmon has NO
    transition -- and is regular at the physical 1-2 line ``f12 = f01 + alpha``.
    At the P5 readout point it overstates |chi| by a factor 2.379.

    The defect is REFERENCE-INHERITED, so this function and its two call sites
    (`drive_cavity_budget`, `readout_window_scan`) are left byte-faithful and
    are retained as the documented reference-convention path, exactly as AT-6's
    small-frequency qp ladder was retained after the full kernel superseded it.
    AT-4/AT-5 assert reproduction of the reference, not physical accuracy.

    Production code must call `chi_dispersive_koch` (corrected single formula)
    or `chi_two_pole` (the exact two-pole truth standard).

    src: out/verify/chi_convention_ruling.md Sec. 3(a), Sec. 4 item 2.
    """
    return g_GHz ** 2 * alpha_GHz / (Delta_GHz * (Delta_GHz + alpha_GHz))


def chi_dispersive_koch(g_GHz, f01_GHz, f_r_GHz, alpha_GHz):
    """Corrected single-formula dispersive shift, in GHz.

        chi = g^2 alpha / (Delta (Delta + alpha)),   Delta = f01 - f_r     (KOCH)

    i.e. Koch et al., PRA 76, 042319 (2007), Eqs. 3.10-3.13, with the detuning
    taken as QUBIT MINUS CAVITY.  Takes f01 and f_r rather than a detuning so
    the sign can never be inverted at a call site again.

    CORRECTNESS INVARIANT (the reason this signature exists): the only poles
    are at ``f_r = f01`` (the 0-1 line) and ``f_r = f01 + alpha = f12`` (the 1-2
    line).  Those are the only two frequencies at which a physical cavity pull
    can diverge.  Any rewrite that moves a pole elsewhere is wrong.

    Exact only in the harmonic limit ``(n12/n01)^2 -> 2``; against the exact
    two-pole form it is -9.04 % at the P5 readout detuning and -72 % at the
    145 GHz drive detuning (the deficit term (1 - r/2)/Delta it omits scales as
    1/Delta while this whole expression scales as 1/Delta^2).

    src: out/verify/chi_convention_ruling.md Sec. 1, Sec. 2.
    """
    Delta = f01_GHz - f_r_GHz
    return g_GHz ** 2 * alpha_GHz / (Delta * (Delta + alpha_GHz))


def chi_two_pole(g_GHz, f01_GHz, f12_GHz, f_r_GHz, n_ratio_sq):
    """Exact second-order (RWA) dispersive shift chi_01 - chi_12/2, in GHz.

        chi_ij = g_ij^2 / (f_ij - f_r),      g_ij = g * n_ij / n01
        chi    = g^2 / (f01 - f_r)  -  (r/2) g^2 / (f12 - f_r)      (TWO-POLE)

    with ``r = n_ratio_sq = (n12/n01)^2``.  The qubit couples to the cavity
    through the island charge operator, so the level-j-to-k coupling is exactly
    ``g_jk = g n_jk / n01`` -- the same operator P2 pinned for g.

    This is the TRUTH STANDARD of the chi-convention ruling.  Substituting the
    harmonic ratio ``r -> 2`` collapses it identically onto `chi_dispersive_koch`;
    that identity is what fixes the single formula's detuning convention.

    For E_J/E_C = 100 the exact ratio is r = 1.918851..., not 2, so the two
    differ by the ``(1 - r/2)/(f12 - f_r)`` deficit term.

    src: out/verify/chi_convention_ruling.md Sec. 1.
    """
    return (g_GHz ** 2 / (f01_GHz - f_r_GHz)
            - 0.5 * n_ratio_sq * g_GHz ** 2 / (f12_GHz - f_r_GHz))


def charge_matrix_elements(EJ_GHz, EC_GHz, N=N_CHARGE):
    """(n01, n12) = (|<0|n|1>|, |<1|n|2>|) at n_g = 0, Cooper-pair units.

    Recomputed from `qubit.spectrum.hamiltonian` (spectrum.py is NOT modified;
    it exposes n01 only).  n12 is what turns `chi_dispersive_koch` into
    `chi_two_pole` and what sets the 1-2-line coupling g12 = g n12/n01.
    """
    n = np.arange(-N, N + 1)
    w, v = np.linalg.eigh(_spec.hamiltonian(EJ_GHz, EC_GHz, 0.0, N))
    n01 = abs(v[:, 0] @ (n * v[:, 1]))
    n12 = abs(v[:, 1] @ (n * v[:, 2]))
    return float(n01), float(n12)


def n_ratio_sq(EJ_GHz, EC_GHz, N=N_CHARGE):
    """r = (n12/n01)^2 -- the exact matrix-element ratio in (TWO-POLE).

    Harmonic (oscillator) limit is exactly 2; at E_J/E_C = 100 it is 1.918851.
    """
    n01, n12 = charge_matrix_elements(EJ_GHz, EC_GHz, N)
    return (n12 / n01) ** 2


def kappa_from_Q(f_GHz, Q):
    """Cavity linewidth kappa = f/Q in GHz."""
    return f_GHz / Q


def purcell_rate(g_GHz, Delta_GHz, kappa_GHz):
    """Purcell relaxation rate Gamma = (g/Delta)^2 * 2 pi kappa, in s^-1."""
    return (g_GHz / Delta_GHz) ** 2 * 2 * np.pi * kappa_GHz * 1e9


def drive_cavity_budget(n01, f01_GHz, alpha_GHz, f_cav_GHz=F_CAV_GHZ,
                        V_m3=V_MODE_M3, d_eff_m=D_EFF_DRIVE_M, Q=Q_I,
                        eps_c=EPS_C):
    """Full measured-drive-cavity budget (AT-4).

    Returns E_zpf (V/m), V_zpf (V), g (MHz), chi (kHz), kappa (MHz),
    chi/kappa and the Purcell-limited T1 (us).
    """
    ez = E_zpf(f_cav_GHz, V_m3, eps_c)
    vz = ez * d_eff_m
    g = E * vz * n01 / H / 1e9
    Delta = f_cav_GHz - f01_GHz
    chi = chi_dispersive(g, Delta, alpha_GHz)
    # kappa here is the internal-Q linewidth; note purcell_rate is written with
    # the reference's 2 pi f/Q form, identical to 2 pi * kappa_from_Q.
    kappa = kappa_from_Q(f_cav_GHz, Q)
    gamma_p = (g / Delta) ** 2 * (2 * np.pi * f_cav_GHz * 1e9 / Q)
    return dict(
        E_zpf_V_per_m=ez, V_zpf_V=vz,
        g_MHz=g * 1e3, Delta_GHz=Delta,
        chi_kHz=chi * 1e6, kappa_MHz=kappa * 1e3,
        chi_over_kappa=abs(chi) / kappa,
        T1_Purcell_us=1.0 / gamma_p * 1e6,
        g_convention="g = e*V_zpf*n01/h; lumped-circuit pin = agent task",
    )


def readout_window_scan(n01, f01_GHz, alpha_GHz, f_cav_GHz=F_CAV_GHZ,
                        V_m3=V_MODE_M3, eps_c=EPS_C,
                        detunings_GHz=DETUNINGS_GHZ,
                        d_effs_m=D_EFF_READOUT_M,
                        kappas_MHz=KAPPA_READOUT_MHZ):
    """Feasible readout-cavity operating points (AT-5).

    A second photonic-crystal defect at f_r = f01 + Delta_r; its mode volume
    is scaled from the measured 145 GHz cavity as V_r = V_mode (f_cav/f_r)^3
    (fixed a/lambda, so V ~ lambda^3).  A point is feasible when
    chi/kappa >= 1, T1_Purcell >= 30 us, and (g/Delta_r)^2 <= 0.01.
    """
    rows = []
    for Dr in detunings_GHz:
        fr = f01_GHz + Dr
        Vr = V_m3 * (f_cav_GHz / fr) ** 3
        Ez_r = E_zpf(fr, Vr, eps_c)
        for de in d_effs_m:
            g = E * Ez_r * de * n01 / H / 1e9
            if (g / Dr) ** 2 > G_OVER_DELTA_SQ_MAX:
                continue
            chi = chi_dispersive(g, Dr, alpha_GHz)
            for kap_MHz in kappas_MHz:
                kap = kap_MHz * 1e-3
                T1p = 1.0 / ((g / Dr) ** 2 * 2 * np.pi * kap * 1e9)
                if abs(chi) / kap >= CHI_OVER_KAPPA_MIN and T1p >= T1_PURCELL_MIN_S:
                    rows.append(dict(
                        Delta_r_GHz=float(Dr), f_r_GHz=fr, d_eff_um=de * 1e6,
                        kappa_MHz=kap_MHz, g_MHz=g * 1e3, chi_MHz=chi * 1e3,
                        chi_over_kappa=abs(chi) / kap, T1_Purcell_us=T1p * 1e6))
    return rows


def best_readout_point(rows):
    """The reference's sample operating point: maximise the weaker of the two
    feasibility margins, min(chi/kappa, T1_Purcell/30us)."""
    return max(rows, key=lambda r: min(r["chi_over_kappa"],
                                       r["T1_Purcell_us"] / 30.0))
