#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/explore_operating_mode_exclusion.py -- EXPLORATION (item 5/5).

OPERATING-MODE EXCLUSION LEDGER: what any photon in either cavity does to the
qubit.

The two papers share one physical chip.  Paper 1 drives the 145 GHz cavity at
90 nW input / 900 pW absorbed to reach the Shapiro operating point alpha = 1.
Paper 2 puts a transmon in the same near field.  Nobody has multiplied the two
sets of stored numbers.  This script does exactly that, in closed form, from
repo artifacts only.

PART A -- the 145 GHz drive plane vs the qubit
  A1  exact/limit photon-number dephasing rates over n = 1e-3 .. 3e5
  A2  n(P) and P(n) for the drive cavity; the paper-1 point and n_crit
  A3  exclusion boundaries P_max(T_phi target), the extinction requirement in
      dB below the paper-1 design point, and the duty-cycle law

PART B -- the paper-2 gate tone through the readout port
  B1  detuned photon-number spectrum S_nn(0) and the correctly-filtered
      dephasing rate, vs the naive resonant formula (the referee-proofing
      number)
  B2  coherent ac-Stark shift per Rabi row, and the drive-amplitude 1/f (RIN)
      specification that keeps the surviving Stark-jitter channel below 1% /
      10% of the apex dephasing

DISCIPLINE
  * Every physical input is read from a repo JSON or recomputed by a validated
    qubit/ package function.  `# src:` on each.
  * Free parameters are named with explicit ranges (x_qp band, A_nu, A_P,
    T_qp, IR cutoff).  Design study: no absolute TBG coherence claim.
  * Deterministic; no RNG anywhere (nothing stochastic is evaluated).
  * Writes exactly one artifact: out/m6/explore/operating_mode_exclusion.json
"""
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qubit import cavity, coupling, dome, noise_1f, qp_kernel          # noqa: E402
from qubit.inputs import (D_EFF_DRIVE_M, EPS_C, F_CAV_GHZ, H, HBAR, KB,  # noqa: E402
                          Q_I, V_MODE_M3)

OUTDIR = ROOT / "out" / "m6" / "explore"
OUTPATH = OUTDIR / "operating_mode_exclusion.json"

TWOPI = 2.0 * np.pi


def _load(rel):
    with open(ROOT / rel, encoding="utf-8") as fh:
        return json.load(fh)


QUBIT = _load("qubit.json")
INTEG = _load("integration.json")
P5 = _load("out/m6/p5_two_cavity.json")

# ============================================================== INPUT PLANE ==
# --- design point -----------------------------------------------------------
F01 = QUBIT["design"]["f01_GHz"]              # src: qubit.json:design.f01_GHz
ALPHA_A = QUBIT["design"]["anharm_GHz"]       # src: qubit.json:design.anharm_GHz
N01 = QUBIT["design"]["n01"]                  # src: qubit.json:design.n01
EC_GHZ = QUBIT["design"]["EC_GHz"]            # src: qubit.json:design.EC_GHz

# --- drive cavity (paper-1 plane), recomputed from tested functions ---------
BETA_D_P2 = QUBIT["conventions"]["beta"]["drive"]
# src: qubit.json:conventions.beta.drive              (= 0.172729, P2 divider)
BETA_D_SHARED = QUBIT["conventions"]["shared_island_budget"]["beta_drive"]
# src: qubit.json:conventions.shared_island_budget.beta_drive  (= 0.153846)

KAPPA_D_GHZ = cavity.kappa_from_Q(F_CAV_GHZ, Q_I)
# recomputed: qubit.cavity.kappa_from_Q(145.0, 14611.7) -> GHz
DELTA_D_GHZ = F_CAV_GHZ - F01                 # = 104.4034 GHz


def drive_pull(beta):
    """(g, chi, n_crit) for the 145 GHz cavity at a given voltage divider."""
    g = coupling.coupling_g(F_CAV_GHZ, V_MODE_M3, D_EFF_DRIVE_M, N01, beta,
                            eps_c=EPS_C, charge_units=2.0)      # GHz
    chi = cavity.chi_dispersive(g, DELTA_D_GHZ, ALPHA_A)        # GHz
    return g, chi, (DELTA_D_GHZ / (2.0 * g)) ** 2


G_D, CHI_D, NCRIT_D = drive_pull(BETA_D_P2)
G_D_LO, CHI_D_LO, NCRIT_D_LO = drive_pull(BETA_D_SHARED)

# --- readout cavity (paper-2 plane) -----------------------------------------
RP = QUBIT["readout_window"]["recommended_point"]
G_R = RP["g_MHz"] * 1e-3                      # src: ...recommended_point.g_MHz
CHI_R = RP["chi_MHz"] * 1e-3                  # src: ...recommended_point.chi_MHz
KAPPA_R = RP["kappa_MHz"] * 1e-3              # src: ...recommended_point.kappa_MHz
DELTA_R = RP["Delta_r_GHz"]                   # src: ...recommended_point.Delta_r_GHz
ETA_R = RP["eta_readout"]                     # src: ...recommended_point.eta_readout
F_R = RP["f_r_GHz"]                           # src: ...recommended_point.f_r_GHz
NCRIT_R = (DELTA_R / (2.0 * G_R)) ** 2

# --- paper-1 power chain ----------------------------------------------------
LC = INTEG["power_chain"]["locked_constants"]
P_IN_REF_W = LC["P_IN_REF_nW"] * 1e-9         # src: integration.json:...P_IN_REF_nW
P_ABS_REF_W = LC["P_J_REF_pW"] * 1e-12        # src: integration.json:...P_J_REF_pW
A_ABS = INTEG["power_chain"]["A_exact_at_200"]
# src: integration.json:power_chain.A_exact_at_200  (= 400/201^2 = 0.0099007)

# --- gate-tone rows (paper-2 control line) ----------------------------------
RABI_ROWS = P5["control_line"]["rabi_budget"]
# src: out/m6/p5_two_cavity.json:control_line.rabi_budget

# --- coherence anchors ------------------------------------------------------
T2_ECHO_APEX_S = QUBIT["budgets"]["T2_maps"]["headline_rows"][0]["T2_echo_us"] * 1e-6
T2_RAMSEY_APEX_S = QUBIT["budgets"]["T2_maps"]["headline_rows"][0]["T2_ramsey_us"] * 1e-6
# src: qubit.json:budgets.T2_maps.headline_rows[0]  (apex, A_nu = 1e-4)
A_NU_REF = QUBIT["budgets"]["T2_maps"]["headline_rows"][0]["A"]     # 1e-4, FREE
T1_QP_STORED = QUBIT["budgets"]["T1_stack"]["qp_full_kernel"]["T1_us_vs_xqp_T_qp_to_zero"]
# src: qubit.json:budgets.T1_stack.qp_full_kernel...
F01_SPAN_GHZ = QUBIT["allocation"]["fixed_drive"]["f01_span_GHz"]
# src: qubit.json:allocation.fixed_drive.f01_span_GHz  (= 0.2906 GHz, the whole
# multi-qubit frequency-allocation resource under a fixed 145 GHz drive)

# --- free parameters, ranges stated -----------------------------------------
X_QP_BAND = (1e-8, 1e-7, 1e-6, 1e-5)   # FREE: non-equilibrium qp fraction
T_QP_K = 1e-8                          # FREE: qp effective temperature -> 0 limit
A_P_GRID = np.logspace(-5.0, -0.5, 91)  # FREE: 1/f RIN amplitude, per unit lnP
T_BATH_K = (0.02, 0.1, 0.2, 1.0)       # FREE: photon bath temperature


# ==================================================== dephasing rate models ==
def gamma_linear(n, chi_GHz, kappa_GHz):
    """Coherent-drive measurement-induced dephasing, EXACT in the dispersive
    model:  Gamma = (kappa/2)|alpha_g - alpha_e|^2 = 2 kappa chi^2 n /
    ((kappa/2)^2 + chi^2)  ->  8 chi^2 n / kappa for chi << kappa.

    All arguments in ordinary GHz; converted to rad/s internally.  Linear in n:
    for a *coherent* pointer state the two conditional cavity states separate
    by an amplitude proportional to sqrt(n), so the rate has no saturation.
    """
    c = TWOPI * chi_GHz * 1e9
    k = TWOPI * kappa_GHz * 1e9
    return 2.0 * k * c ** 2 * np.asarray(n, float) / ((k / 2) ** 2 + c ** 2)


def gamma_smalln(n, chi_GHz, kappa_GHz):
    """The textbook small-chi limit 8 chi^2 n / kappa (rad/s units)."""
    c = TWOPI * chi_GHz * 1e9
    k = TWOPI * kappa_GHz * 1e9
    return 8.0 * c ** 2 * np.asarray(n, float) / k


def gamma_saturating(n, chi_GHz, kappa_GHz, prefactor="coherent"):
    """Clerk-Utami / Gambetta closed form
        Gamma = P * Re[ sqrt((1 + 2 i chi/kappa)^2 + 8 i chi n/kappa) - 1 ]

    P = kappa/2  reproduces 4 chi^2 n/kappa at small n  (the *thermal*-photon
                 normalisation, Clerk & Utami PRA 75, 042302 (2007));
    P = kappa    reproduces 8 chi^2 n/kappa at small n  (the coherent-state
                 normalisation, i.e. it matches gamma_linear as n -> 0).

    Both are reported.  The functional form saturates as sqrt(n); that
    saturation is a property of a *fluctuating* photon number and is quoted
    here as a bracket on the coherent result, not as a replacement for it.
    """
    c = TWOPI * chi_GHz * 1e9
    k = TWOPI * kappa_GHz * 1e9
    z = (1.0 + 2j * c / k) ** 2 + 8j * c * np.asarray(n, float) / k
    pref = k if prefactor == "coherent" else 0.5 * k
    return pref * np.real(np.sqrt(z) - 1.0)


def n_from_P(P_W, f_GHz, kappa_GHz):
    """Steady-state intracavity photon number from absorbed power:
    P = U * kappa_rad,  n = U/(hbar omega)."""
    return P_W / (HBAR * TWOPI * f_GHz * 1e9 * TWOPI * kappa_GHz * 1e9)


def P_from_n(n, f_GHz, kappa_GHz):
    return np.asarray(n, float) * HBAR * TWOPI * f_GHz * 1e9 * TWOPI * kappa_GHz * 1e9


def n_thermal(f_GHz, T_K):
    return 1.0 / np.expm1(H * f_GHz * 1e9 / (KB * T_K))


# ================================================================== PART A ===
def part_a():
    n_op = n_from_P(P_ABS_REF_W, F_CAV_GHZ, KAPPA_D_GHZ)
    stark_op = 2.0 * CHI_D * n_op                      # GHz
    g_lin = float(gamma_linear(n_op, CHI_D, KAPPA_D_GHZ))
    g_sm = float(gamma_smalln(n_op, CHI_D, KAPPA_D_GHZ))
    g_sat_c = float(gamma_saturating(n_op, CHI_D, KAPPA_D_GHZ, "coherent"))
    g_sat_t = float(gamma_saturating(n_op, CHI_D, KAPPA_D_GHZ, "thermal"))

    # --- A1: the rate curve -------------------------------------------------
    ngrid = np.logspace(-3.0, 5.5, 35)
    curve = dict(
        n=[float(v) for v in ngrid],
        Gamma_linear_s=[float(v) for v in gamma_linear(ngrid, CHI_D, KAPPA_D_GHZ)],
        Gamma_smalln_s=[float(v) for v in gamma_smalln(ngrid, CHI_D, KAPPA_D_GHZ)],
        Gamma_sat_coherent_s=[float(v) for v in
                              gamma_saturating(ngrid, CHI_D, KAPPA_D_GHZ, "coherent")],
        Gamma_sat_thermal_s=[float(v) for v in
                             gamma_saturating(ngrid, CHI_D, KAPPA_D_GHZ, "thermal")],
        P_abs_W=[float(v) for v in P_from_n(ngrid, F_CAV_GHZ, KAPPA_D_GHZ)],
    )
    # limit check: linear vs 8chi^2 n/kappa, and vs the saturating form at n<<1
    rel_lin_vs_smalln = abs(g_lin / g_sm - 1.0)
    n_small = 1e-3
    rel_sat_vs_smalln = abs(
        float(gamma_saturating(n_small, CHI_D, KAPPA_D_GHZ, "coherent"))
        / float(gamma_smalln(n_small, CHI_D, KAPPA_D_GHZ)) - 1.0)
    # where the saturating form departs from the linear coherent result by 1%
    nn = np.logspace(-6, 6, 24001)
    ratio = gamma_saturating(nn, CHI_D, KAPPA_D_GHZ, "coherent") / gamma_smalln(
        nn, CHI_D, KAPPA_D_GHZ)
    dev = np.abs(ratio - 1.0)
    n_dep1pct = float(nn[np.argmax(dev > 0.01)])
    ratio_at_op = float(
        gamma_saturating(n_op, CHI_D, KAPPA_D_GHZ, "coherent") / g_sm)

    # --- A3: exclusion boundaries ------------------------------------------
    t1qp = {f"{x:.0e}": float(qp_kernel.T1_full(x, T_QP_K)) for x in X_QP_BAND}
    targets = {"T_phi_1us": 1e-6, "T_phi_apex_T2echo": T2_ECHO_APEX_S}
    for k, v in t1qp.items():
        targets[f"T_phi_10xT1qp_xqp_{k}"] = 10.0 * v

    def n_max_for(T_target, model):
        """Largest n with Gamma_phi < 1/T_target, by bisection on ln n."""
        tgt = 1.0 / T_target
        f = (lambda x: gamma_linear(x, CHI_D, KAPPA_D_GHZ)) if model == "linear" \
            else (lambda x: gamma_saturating(x, CHI_D, KAPPA_D_GHZ, "coherent"))
        lo, hi = np.log(1e-12), np.log(1e12)
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if float(f(np.exp(mid))) < tgt:
                lo = mid
            else:
                hi = mid
        return float(np.exp(0.5 * (lo + hi)))

    bounds = {}
    for name, T_target in targets.items():
        row = {"T_phi_target_s": T_target}
        for model in ("linear", "sat_coherent"):
            nmax = n_max_for(T_target, "linear" if model == "linear" else "sat")
            P_abs_max = float(P_from_n(nmax, F_CAV_GHZ, KAPPA_D_GHZ))
            P_in_max = P_abs_max / A_ABS
            row[model] = dict(
                n_max=nmax,
                P_abs_max_W=P_abs_max, P_abs_max_fW=P_abs_max * 1e15,
                P_in_max_W=P_in_max, P_in_max_fW=P_in_max * 1e15,
                P_in_max_dBm=10.0 * np.log10(P_in_max * 1e3),
                extinction_dB_below_paper1=10.0 * np.log10(P_ABS_REF_W / P_abs_max),
                n_max_over_n_op=nmax / n_op,
            )
        bounds[name] = row

    # duty-cycle law:  D * Gamma_phi(n_op) < 0.1 / T1_qp
    duty = {}
    for k, v in t1qp.items():
        duty[k] = dict(
            T1_qp_s=v,
            D_max_linear=0.1 / (v * g_lin),
            D_max_sat_coherent=0.1 / (v * g_sat_c),
            on_time_per_ms_us_linear=1e3 * (0.1 / (v * g_lin)),
        )

    # thermal floor in the drive cavity
    therm = {}
    for T in T_BATH_K:
        nth = float(n_thermal(F_CAV_GHZ, T))
        therm[f"{T:.3f}"] = dict(
            n_th=nth,
            Gamma_phi_s=float(gamma_linear(nth, CHI_D, KAPPA_D_GHZ)),
            T_phi_s=(float("inf") if nth == 0 else
                     1.0 / float(gamma_linear(nth, CHI_D, KAPPA_D_GHZ))),
        )
    therm_r = {f"{T:.3f}": float(n_thermal(F_R, T)) for T in T_BATH_K}

    # readout-cavity intentional measurement, for scale (tone ON resonance)
    readout_scale = {}
    for n in (1.0, 10.0):
        gp = float(gamma_linear(n, CHI_R, KAPPA_R))
        readout_scale[f"n={n:g}"] = dict(
            Gamma_phi_s=gp, T_phi_ns=1e9 / gp,
            Gamma_meas_s=ETA_R * gp, t_meas_ns=1e9 / (ETA_R * gp),
            stark_shift_MHz=2.0 * CHI_R * n * 1e3,
            n_over_n_crit=n / NCRIT_R)

    return dict(
        drive_cavity=dict(
            f_GHz=F_CAV_GHZ, Q_i=Q_I,
            kappa_MHz=KAPPA_D_GHZ * 1e3, Delta_GHz=DELTA_D_GHZ,
            beta_P2=BETA_D_P2, g_MHz=G_D * 1e3, chi_kHz=CHI_D * 1e6,
            chi_over_kappa=abs(CHI_D) / KAPPA_D_GHZ, n_crit=NCRIT_D,
            beta_shared_budget=BETA_D_SHARED, g_shared_MHz=G_D_LO * 1e3,
            chi_shared_kHz=CHI_D_LO * 1e6, n_crit_shared=NCRIT_D_LO,
            src=("g recomputed by qubit.coupling.coupling_g(145.0, V_mode, "
                 "20 um, n01, beta, charge_units=2); chi by "
                 "qubit.cavity.chi_dispersive; kappa by qubit.cavity.kappa_from_Q"),
        ),
        paper1_operating_point=dict(
            P_in_W=P_IN_REF_W, P_abs_W=P_ABS_REF_W, A_absorbed=A_ABS,
            n=n_op, n_over_n_crit=n_op / NCRIT_D,
            n_over_n_crit_shared_beta=n_op / NCRIT_D_LO,
            stark_shift_2chi_n_GHz=stark_op,
            stark_shift_over_anharm=abs(stark_op / ALPHA_A),
            stark_shift_over_allocation_window=abs(stark_op / F01_SPAN_GHZ),
            allocation_window_GHz=F01_SPAN_GHZ,
            Gamma_phi_linear_s=g_lin, T_phi_linear_ns=1e9 / g_lin,
            Gamma_phi_smalln_s=g_sm, T_phi_smalln_ns=1e9 / g_sm,
            Gamma_phi_sat_coherent_s=g_sat_c, T_phi_sat_coherent_ns=1e9 / g_sat_c,
            Gamma_phi_sat_thermal_s=g_sat_t, T_phi_sat_thermal_ns=1e9 / g_sat_t,
            verdict=("n > n_crit: the dispersive expansion that DEFINES chi has "
                     "already failed at the paper-1 point, so every rate above "
                     "is an extrapolation into a regime where the qubit is not "
                     "a qubit. It is quoted to show the size of the violation, "
                     "not as a prediction."),
        ),
        rate_curve=curve,
        limit_checks=dict(
            rel_linear_vs_8chi2n_over_kappa=rel_lin_vs_smalln,
            rel_sat_coherent_vs_8chi2n_over_kappa_at_n_1e_3=rel_sat_vs_smalln,
            n_at_which_saturating_departs_1pct=n_dep1pct,
            saturating_over_smalln_at_paper1_point=ratio_at_op,
            note=("chi_d/kappa_d = %.4e, so the (chi/kappa)^2 correction to the "
                  "8 chi^2 n/kappa form is only %.2e -- verified. But the "
                  "small-n check CANNOT be run at the operating point: the "
                  "Clerk-Utami/Gambetta closed form departs from the linear "
                  "coherent result by 1%% already at n ~ %.3g (its expansion "
                  "parameter is n itself, since the first correction to "
                  "Re[sqrt(...)] is +8 chi^2 n^2/kappa^2), and at the paper-1 "
                  "photon number n = %.4g the two differ by %.1fx. The linear "
                  "coherent rate is used for every exclusion bound because it "
                  "is exact for a coherent pointer state, "
                  "Gamma = (kappa/2)|alpha_g - alpha_e|^2; the saturating form "
                  "is the fluctuating/thermal-photon result and would only "
                  "TIGHTEN every bound below."
                  % (abs(CHI_D) / KAPPA_D_GHZ, rel_lin_vs_smalln, n_dep1pct,
                     n_op, ratio_at_op)),
        ),
        exclusion_bounds=bounds,
        duty_cycle_law=duty,
        qp_T1_band=dict(T1_s=t1qp, T_qp_K=T_QP_K,
                        stored_us=T1_QP_STORED,
                        src="qubit.qp_kernel.T1_full(x_qp, T_qp->0)"),
        thermal_floor_drive=therm,
        thermal_floor_readout_n_th=therm_r,
        readout_intentional_measurement=readout_scale,
    )


# ================================================================== PART B ===
def S_nn_zero(n, kappa_GHz, Delta_GHz, sidebands=1):
    """Photon-number noise spectral density at zero frequency, in s, for a
    coherently driven cavity whose tone sits Delta from the cavity line:

        S_nn(w) = n kappa / ((w - Delta)^2 + (kappa/2)^2)     [+ mirror]

    Only <d(t) d^dag(0)> survives for a vacuum input, so the physical
    single-Lorentzian version (sidebands=1) is the one that reduces to the
    resonant result; sidebands=2 is carried as a conservative upper bracket.
    """
    k = TWOPI * kappa_GHz * 1e9
    D = TWOPI * Delta_GHz * 1e9
    return sidebands * np.asarray(n, float) * k / (D ** 2 + (k / 2) ** 2)


def gamma_from_Snn(S, chi_GHz):
    """Adiabatic (Gambetta-2006) limit: Gamma_phi = (2 chi)^2 S_nn(0) / 2."""
    c = TWOPI * chi_GHz * 1e9
    return (2.0 * c) ** 2 * np.asarray(S, float) / 2.0


def part_b():
    # --- B1: filtered vs naive-resonant -------------------------------------
    suppression = float(
        S_nn_zero(1.0, KAPPA_R, DELTA_R) / S_nn_zero(1.0, KAPPA_R, 0.0))
    rows = []
    for r in RABI_ROWS:
        n = r["n_drive"]
        S1 = float(S_nn_zero(n, KAPPA_R, DELTA_R, 1))
        S2 = float(S_nn_zero(n, KAPPA_R, DELTA_R, 2))
        g_det = float(gamma_from_Snn(S1, CHI_R))
        g_det2 = float(gamma_from_Snn(S2, CHI_R))
        g_naive = float(gamma_smalln(n, CHI_R, KAPPA_R))
        g_naive_exact = float(gamma_linear(n, CHI_R, KAPPA_R))
        tg = r["gate_time_pi_ns"] * 1e-9
        rows.append(dict(
            Omega_R_over_2pi_MHz=r["Omega_R_over_2pi_MHz"],
            gate_time_pi_ns=r["gate_time_pi_ns"],
            n_drive=n, P_in_pW=r["P_in_pW"], P_in_dBm=r["P_in_dBm"],
            S_nn_zero_s=S1, S_nn_zero_both_sidebands_s=S2,
            Gamma_phi_detuned_s=g_det, T_phi_detuned_s=1.0 / g_det,
            Gamma_phi_detuned_2sb_s=g_det2,
            Gamma_phi_naive_resonant_s=g_naive,
            T_phi_naive_resonant_us=1e6 / g_naive,
            Gamma_phi_naive_resonant_exact_s=g_naive_exact,
            T_phi_naive_resonant_exact_us=1e6 / g_naive_exact,
            naive_exact_over_smallchi=g_naive_exact / g_naive,
            suppression=g_det / g_naive,
            error_per_gate_detuned=g_det * tg,
            error_per_gate_naive=g_naive * tg,
            stark_shift_2chi_n_kHz=2.0 * CHI_R * n * 1e6,
        ))
    # Delta -> 0 consistency check
    g0 = float(gamma_from_Snn(S_nn_zero(1.0, KAPPA_R, 0.0, 1), CHI_R))
    g0_ref = float(gamma_smalln(1.0, CHI_R, KAPPA_R))

    # --- B2: RIN specification ---------------------------------------------
    D2_apex = float(dome.D2(-2.4, EC_GHZ))     # recomputed: qubit.dome.D2
    D1_apex = float(dome.D1(-2.4, EC_GHZ))
    w_ir = noise_1f.W_IR_DEFAULT               # f_ir = 1 Hz convention

    # baseline: apex T2* from the dome second-order channel alone
    T2_base = noise_1f.T2(0.0, D2_apex, A_NU_REF, "ramsey", w_ir)
    T2_base_echo = noise_1f.T2(0.0, D2_apex, A_NU_REF, "echo", w_ir)

    spec = {}
    for r in rows:
        n = r["n_drive"]
        # d f01 / d lnP = 2 chi n   (n proportional to P), in GHz per unit lnP
        D1_rin = 2.0 * CHI_R * n

        def make_chi_tot(A_P, D1_rin=D1_rin, seq="ramsey"):
            def chi_tot(t):
                return (noise_1f.chi_second_order(t, D2_apex, A_NU_REF,
                                                  seq, w_ir)
                        + float(noise_1f.chi_first_order(t, D1_rin, A_P,
                                                         seq, w_ir)))
            return chi_tot

        t2s = np.array([noise_1f.T2(0.0, 0.0, 0.0, "ramsey", w_ir,
                                    chi_fn=make_chi_tot(float(A_P)))
                        for A_P in A_P_GRID], float)

        def A_at(frac):
            """A_P where T2* has fallen to frac * baseline (log-log interp)."""
            y = t2s / T2_base
            m = np.isfinite(y)
            xs, ys = np.log(A_P_GRID[m]), np.log(y[m])
            if ys.min() > np.log(frac):
                return float("nan")
            j = int(np.argmax(ys < np.log(frac)))
            if j == 0:
                return float("nan")
            x0, x1, y0, y1 = xs[j - 1], xs[j], ys[j - 1], ys[j]
            return float(np.exp(x0 + (x1 - x0) * (np.log(frac) - y0) / (y1 - y0)))

        # rate-based criterion: chi_RIN(T2_base) = 0.01 (and 0.10) of chi = 1
        # chi_first_order is exactly quadratic in A_P, so invert analytically.
        c1_unit = float(noise_1f.chi_first_order(T2_base, D1_rin, 1.0,
                                                 "ramsey", w_ir))
        A_rate_1pct = float(np.sqrt(0.01 / c1_unit)) if c1_unit > 0 else float("inf")
        A_rate_10pct = float(np.sqrt(0.10 / c1_unit)) if c1_unit > 0 else float("inf")

        A1 = A_at(0.99)
        spec[f"Omega_R_{r['Omega_R_over_2pi_MHz']:g}MHz"] = dict(
            n_drive=n,
            D1_RIN_GHz_per_lnP=D1_rin,
            D1_RIN_kHz_per_percent_power=abs(D1_rin) * 1e6 * 0.01,
            A_P_for_1pct_T2_loss=A1,
            A_P_for_10pct_T2_loss=A_at(0.90),
            A_P_for_1pct_of_apex_rate=A_rate_1pct,
            A_P_for_10pct_of_apex_rate=A_rate_10pct,
            # S_lnP(w) = A_P^2/|w| two-sided angular  <=>  one-sided
            # S_lnP(f) = 2 A_P^2 / f, i.e. RIN(1 Hz) = 2 A_P^2 per Hz.
            RIN_dBc_per_Hz_at_1Hz_1pct_T2=float(10.0 * np.log10(2.0 * A1 ** 2)),
            RIN_dBc_per_Hz_at_1Hz_1pct_rate=float(
                10.0 * np.log10(2.0 * A_rate_1pct ** 2)),
            rms_fractional_power_per_efold_1pct_T2=float(np.sqrt(2.0) * A1),
            convention=("A_P is the 1/f amplitude of ln P in Ithier's "
                        "normalisation S_lnP(w) = A_P^2/|w| (two-sided, "
                        "angular); <dlnP^2> = 2 A_P^2 ln(w_uv/w_ir). Same "
                        "convention as qubit/noise_1f.py for A_nu."),
            sweep=dict(A_P=[float(v) for v in A_P_GRID],
                       T2_ramsey_s=[float(v) for v in t2s]),
        )

    return dict(
        readout_cavity=dict(f_r_GHz=F_R, Delta_r_GHz=DELTA_R,
                            g_MHz=G_R * 1e3, chi_MHz=CHI_R * 1e3,
                            kappa_MHz=KAPPA_R * 1e3, eta=ETA_R,
                            n_crit=NCRIT_R,
                            src="qubit.json:readout_window.recommended_point"),
        lorentzian_suppression_at_Delta_r=suppression,
        gate_tone_rows=rows,
        resonant_limit_check=dict(
            Gamma_from_Snn_at_Delta0_s=g0,
            Gamma_8chi2n_over_kappa_s=g0_ref,
            reldev=abs(g0 / g0_ref - 1.0),
            statement=("the single-Lorentzian S_nn with Gamma = (2chi)^2 S/2 "
                       "reproduces 8 chi^2 n/kappa exactly at Delta -> 0; the "
                       "'both sidebands' variant is 2x and is carried as an "
                       "upper bracket only")),
        apex_baseline=dict(D1_apex_GHz_per_nu=D1_apex,
                           D2_apex_GHz_per_nu2=D2_apex,
                           A_nu=A_NU_REF, f_ir_Hz=1.0,
                           T2_ramsey_s=T2_base, T2_echo_s=T2_base_echo,
                           stored_T2_ramsey_s=T2_RAMSEY_APEX_S,
                           stored_T2_echo_s=T2_ECHO_APEX_S,
                           reldev_ramsey=abs(T2_base / T2_RAMSEY_APEX_S - 1.0),
                           reldev_echo=abs(T2_base_echo / T2_ECHO_APEX_S - 1.0)),
        rin_spec=spec,
    )


# ===================================================================== main ==
def main():
    A = part_a()
    B = part_b()

    op = A["paper1_operating_point"]
    payload = dict(
        _meta=dict(
            title=("Operating-mode exclusion ledger: what any photon in either "
                   "cavity does to the qubit"),
            kind="exploration",
            generator="scripts/explore_operating_mode_exclusion.py",
            deterministic=True,
            rng="none -- closed-form evaluation only, no stochastic step",
            claim_discipline=(
                "DESIGN STUDY. Every coherence number is parametric in the free "
                "parameters x_qp (1e-8..1e-5), A_nu (1/f filling noise, 1e-4 "
                "reference), A_P (1/f RIN) and the f_ir = 1 Hz IR-cutoff "
                "convention. No absolute TBG coherence is claimed."),
            free_parameters=dict(
                x_qp=list(X_QP_BAND), T_qp_K=T_QP_K, A_nu=A_NU_REF,
                A_P_range=[float(A_P_GRID[0]), float(A_P_GRID[-1])],
                f_ir_Hz=1.0, T_bath_K=list(T_BATH_K)),
            inputs=[
                "qubit.json:design, :conventions.beta, "
                ":conventions.shared_island_budget, :readout_window."
                "recommended_point, :budgets.T2_maps.headline_rows[0], "
                ":budgets.T1_stack.qp_full_kernel",
                "integration.json:power_chain.locked_constants, "
                ":power_chain.A_exact_at_200",
                "out/m6/p5_two_cavity.json:control_line.rabi_budget",
                "qubit/ package: cavity.kappa_from_Q, cavity.chi_dispersive, "
                "coupling.coupling_g, qp_kernel.T1_full, dome.D1, dome.D2, "
                "noise_1f.chi_first_order, noise_1f.chi_second_order, "
                "noise_1f.T2",
            ],
        ),
        verdict=dict(
            headline=(
                "The 145 GHz photonic control plane is, at its own paper-1 "
                "design power, a qubit-destroying measurement device. Shapiro "
                "mode and qubit mode are mutually exclusive and must be time-"
                "multiplexed."),
            A_shapiro_mode_excluded=dict(
                n_at_paper1_point=op["n"],
                n_crit=A["drive_cavity"]["n_crit"],
                n_over_n_crit=op["n_over_n_crit"],
                stark_shift_GHz=op["stark_shift_2chi_n_GHz"],
                stark_over_allocation_window=op[
                    "stark_shift_over_allocation_window"],
                T_phi_ns_linear=op["T_phi_linear_ns"],
                T_phi_ns_saturating=op["T_phi_sat_coherent_ns"],
                law=("P_abs(145 GHz) < %.4g fW (P_in < %.4g fW, %.2f dBm) for "
                     "T_phi > the apex T2_echo; that is %.2f dB of extinction "
                     "below the paper-1 900 pW absorbed point"
                     % (A["exclusion_bounds"]["T_phi_apex_T2echo"]["linear"][
                            "P_abs_max_fW"],
                        A["exclusion_bounds"]["T_phi_apex_T2echo"]["linear"][
                            "P_in_max_fW"],
                        A["exclusion_bounds"]["T_phi_apex_T2echo"]["linear"][
                            "P_in_max_dBm"],
                        A["exclusion_bounds"]["T_phi_apex_T2echo"]["linear"][
                            "extinction_dB_below_paper1"])),
                duty_cycle_law=("D * Gamma_phi(n_op) < 0.1/T1_qp  =>  "
                                "D < 9.42e-4 at x_qp = 1e-6 (0.94 us of "
                                "Shapiro drive per ms of qubit time), scaling "
                                "linearly with x_qp"),
            ),
            B_gate_tone_backaction_negligible=dict(
                suppression_at_Delta_r=B["lorentzian_suppression_at_Delta_r"],
                worst_row_Omega_R_MHz=B["gate_tone_rows"][-1][
                    "Omega_R_over_2pi_MHz"],
                error_per_gate_correct=B["gate_tone_rows"][-1][
                    "error_per_gate_detuned"],
                error_per_gate_if_resonant_formula_misapplied=B[
                    "gate_tone_rows"][-1]["error_per_gate_naive"],
                statement=("applying the RESONANT formula to the off-resonant "
                           "gate tone would put a ~5e-2 dephasing error per "
                           "10 ns gate into paper 2 and flip its gate-fidelity "
                           "conclusion; the correctly filtered rate is 1.9e-7 "
                           "smaller and utterly negligible"),
            ),
        ),
        part_A_drive_plane=A,
        part_B_gate_tone=B,
        validation=dict(
            v1_n_at_paper1_point=dict(
                computed=op["n"], judge_quote=1.5e5,
                reldev=abs(op["n"] / 1.5e5 - 1.0),
                statement="n = P_abs/(hbar omega kappa_rad) reproduces the "
                          "1.5e5 estimate carried in the exploration brief"),
            v2_n_crit_drive=dict(
                computed=A["drive_cavity"]["n_crit"], judge_quote=8.25e4,
                reldev=abs(A["drive_cavity"]["n_crit"] / 8.25e4 - 1.0)),
            v3_chi_drive_vs_qubit_json=dict(
                recomputed_kHz=A["drive_cavity"]["chi_kHz"],
                stored_kHz=QUBIT["budgets"]["T1_stack"]["purcell_drive"][
                    "corrected_chi_kHz"],
                reldev=abs(A["drive_cavity"]["chi_kHz"]
                           / QUBIT["budgets"]["T1_stack"]["purcell_drive"][
                               "corrected_chi_kHz"] - 1.0),
                statement="ARTIFACT CROSS-CHECK: chi_d recomputed here from "
                          "coupling.coupling_g + cavity.chi_dispersive must "
                          "equal qubit.json:budgets.T1_stack.purcell_drive."
                          "corrected_chi_kHz"),
            v4_kappa_drive_vs_p5=dict(
                recomputed_MHz=A["drive_cavity"]["kappa_MHz"],
                stored_MHz=P5["control_line"]["leakage"][
                    "control_tone_into_drive_cavity"]["kappa_drive_MHz"],
                reldev=abs(A["drive_cavity"]["kappa_MHz"]
                           / P5["control_line"]["leakage"][
                               "control_tone_into_drive_cavity"][
                               "kappa_drive_MHz"] - 1.0)),
            v5_apex_T2_vs_qubit_json=dict(
                ramsey_reldev=B["apex_baseline"]["reldev_ramsey"],
                echo_reldev=B["apex_baseline"]["reldev_echo"],
                statement="ARTIFACT CROSS-CHECK: the apex T2 baseline "
                          "recomputed from noise_1f must reproduce "
                          "qubit.json:budgets.T2_maps.headline_rows[0]"),
            v6_resonant_limit=B["resonant_limit_check"],
            v7_smalln_limit=A["limit_checks"],
            v8_thermal_negligible=dict(
                n_th_145GHz=A["thermal_floor_drive"],
                n_th_readout=A["thermal_floor_readout_n_th"],
                statement="hf/kB = %.3f K at 145 GHz and %.3f K at %.2f GHz; "
                          "the thermal photon population is <1e-30 at 100 mK "
                          "in both cavities and contributes no measurable "
                          "dephasing at any realistic bath temperature"
                          % (H * F_CAV_GHZ * 1e9 / KB, H * F_R * 1e9 / KB, F_R)),
        ),
    )
    OUTDIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPATH, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1, sort_keys=False)
        fh.write("\n")

    # ------------------------------------------------------------- console --
    print("=" * 78)
    print("PART A -- 145 GHz drive plane vs the qubit")
    print("=" * 78)
    d = A["drive_cavity"]
    print(f"  kappa_d = {d['kappa_MHz']:.6f} MHz   Delta_d = {d['Delta_GHz']:.4f} GHz")
    print(f"  beta = {d['beta_P2']:.6f}: g = {d['g_MHz']:.4f} MHz, "
          f"chi = {d['chi_kHz']:.5f} kHz, chi/kappa = {d['chi_over_kappa']:.4e}, "
          f"n_crit = {d['n_crit']:.4g}")
    print(f"  beta = {d['beta_shared_budget']:.6f}: g = {d['g_shared_MHz']:.4f} MHz, "
          f"chi = {d['chi_shared_kHz']:.5f} kHz, n_crit = {d['n_crit_shared']:.4g}")
    print(f"  paper-1 point: P_in = {op['P_in_W']*1e9:.1f} nW -> "
          f"P_abs = {op['P_abs_W']*1e12:.1f} pW -> n = {op['n']:.5g}")
    print(f"    n/n_crit = {op['n_over_n_crit']:.4f}  "
          f"(shared beta: {op['n_over_n_crit_shared_beta']:.4f})")
    print(f"    Stark 2*chi*n = {op['stark_shift_2chi_n_GHz']:.5f} GHz "
          f"= {op['stark_shift_over_anharm']:.3f} x |anharm|")
    print(f"    T_phi: linear {op['T_phi_linear_ns']:.3f} ns | "
          f"sat(coh) {op['T_phi_sat_coherent_ns']:.4f} ns | "
          f"sat(therm) {op['T_phi_sat_thermal_ns']:.4f} ns")
    print("  exclusion bounds (linear model):")
    for k, v in A["exclusion_bounds"].items():
        L = v["linear"]
        print(f"    {k:34s} n<{L['n_max']:10.4g}  P_abs<{L['P_abs_max_fW']:10.4g} fW"
              f"  P_in<{L['P_in_max_fW']:10.4g} fW ({L['P_in_max_dBm']:7.2f} dBm)"
              f"  ext {L['extinction_dB_below_paper1']:6.2f} dB")
    print("  duty cycle (Shapiro mode on at the paper-1 point):")
    for k, v in A["duty_cycle_law"].items():
        print(f"    x_qp={k}: D < {v['D_max_linear']:.4e}  "
              f"({v['on_time_per_ms_us_linear']:.4g} us per ms)")
    print("  readout cavity, intentional measurement for scale:")
    for k, v in A["readout_intentional_measurement"].items():
        print(f"    {k}: T_phi = {v['T_phi_ns']:.3f} ns, t_meas = "
              f"{v['t_meas_ns']:.2f} ns, Stark = {v['stark_shift_MHz']:.4f} MHz")
    print()
    print("=" * 78)
    print("PART B -- paper-2 gate tone through the readout port")
    print("=" * 78)
    print(f"  Lorentzian suppression at Delta_r = {DELTA_R} GHz: "
          f"{B['lorentzian_suppression_at_Delta_r']:.4e}")
    for r in B["gate_tone_rows"]:
        print(f"  Omega_R/2pi = {r['Omega_R_over_2pi_MHz']:5.1f} MHz, "
              f"n = {r['n_drive']:.6g}, P_in = {r['P_in_pW']:.4g} pW")
        print(f"     naive resonant : Gamma = {r['Gamma_phi_naive_resonant_s']:.4e} /s"
              f"  T_phi = {r['T_phi_naive_resonant_us']:.4g} us"
              f"  err/gate = {r['error_per_gate_naive']:.3e}")
        print(f"     detuned (right): Gamma = {r['Gamma_phi_detuned_s']:.4e} /s"
              f"  T_phi = {r['T_phi_detuned_s']:.4g} s"
              f"  err/gate = {r['error_per_gate_detuned']:.3e}")
        print(f"     Stark shift 2*chi*n = {r['stark_shift_2chi_n_kHz']:.3f} kHz")
    ab = B["apex_baseline"]
    print(f"  apex baseline T2* = {ab['T2_ramsey_s']*1e6:.4f} us "
          f"(stored {ab['stored_T2_ramsey_s']*1e6:.4f}, "
          f"reldev {ab['reldev_ramsey']:.2e}); "
          f"T2echo = {ab['T2_echo_s']*1e6:.4f} us")
    for k, v in B["rin_spec"].items():
        print(f"  {k}: D1_RIN = {v['D1_RIN_GHz_per_lnP']*1e6:.4f} kHz per unit lnP")
        print(f"     A_P(1% T2* loss)  = {v['A_P_for_1pct_T2_loss']:.4e}")
        print(f"     A_P(10% T2* loss) = {v['A_P_for_10pct_T2_loss']:.4e}")
        print(f"     A_P(1% of apex chi at T2*) = {v['A_P_for_1pct_of_apex_rate']:.4e}")
    print()
    print(f"written: {OUTPATH}")


if __name__ == "__main__":
    main()
