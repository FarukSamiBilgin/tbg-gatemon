"""EXPLORATION (M6, item 3/5) -- the two T1 channels the export does not carry.

The exported T1 stack (qubit.json:budgets.T1_stack) has exactly five entries:
qp_full_kernel, purcell_drive, purcell_readout_at_recommended, thermal_floor,
parity_bound.  It contains NO capacitor-dielectric loss and NO direct radiation
from the collection electrodes.  This script closes both gaps in closed form.

    (A)  DIELECTRIC.   Gamma_diel = omega01 * p * tan_delta, with the paper-1
         MEASURED tan_delta = 2e-5 (photonics.json:geometry.tan_delta, closed
         by photonics.json:modes.Q_mat = 52075 ~ 1/tan_delta).

    (B)  RADIATION.    The same collection electrodes that paper 1 engineered
         to couple the cavity IN are, by reciprocity, antennas that couple the
         qubit OUT at f01 = 40.5966 GHz.  f01 sits BELOW the readout lattice's
         scaled TE gap edge and the drive lattice is homogenised there, so
         there is no photonic protection at the qubit frequency.

         Gamma_rad = (2 e beta n01 / hbar)^2 * 2 hbar omega01 * R_rad
         R_rad     = 20 pi^2 (d_eff/lambda0)^2 * f_env          (short dipole)

         The Johnson/vacuum noise of the antenna radiation resistance is
         divided onto the island by the SAME Koch divider beta = C_c/C_Sigma
         that P2 pinned for g -- this is the reciprocity statement, and it is
         why the two numbers cannot be separated.

    (C)  FIX.  Retune f_r so the Maxwell-scaled readout lattice gap COVERS
         f01, and re-run the P5 readout scan inside that window.

Everything is closed form + deterministic (no RNG, no seed needed; the repo's
determinism policy is satisfied trivially -- there is no stochastic step).
Nothing in qubit/, tests/, paper/ or figs/ is modified; this script only
imports the validated package.

Writes: out/m6/explore/missing_t1_channels.json
"""
import json
import pathlib
import sys

import numpy as np
from scipy.optimize import brentq

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qubit import cavity as _cav                       # noqa: E402
from qubit import coupling as _cpl                     # noqa: E402
from qubit import qp_kernel as _qpk                    # noqa: E402
from qubit import two_cavity as _tc                    # noqa: E402
from qubit.inputs import (E, EPS0, EPS_C, F_CAV_GHZ, H, HBAR, PHOTONICS,
                          Q_I, V_MODE_M3, D_EFF_DRIVE_M)   # noqa: E402

C_LIGHT = 299792458.0            # exact SI definition
SEED = 0                         # logged only; no stochastic step anywhere


# =============================================================================
# 0.  INPUTS -- every number read from a repo artifact or the qubit package
# =============================================================================
QUBIT = json.loads((ROOT / "qubit.json").read_text(encoding="utf-8"))

F01_GHZ = QUBIT["design"]["f01_GHz"]
# src: qubit.json:design.f01_GHz = 40.5966287014581
ALPHA_GHZ = QUBIT["design"]["anharm_GHz"]
# src: qubit.json:design.anharm_GHz = -1.6323417967519944
N01 = QUBIT["design"]["n01"]
# src: qubit.json:design.n01 = 1.3047875496582912
C_SIGMA_FF = QUBIT["design"]["C_Sigma_fF"]
# src: qubit.json:design.C_Sigma_fF = 12.999713035702571

BETA_DRIVE = QUBIT["conventions"]["beta"]["drive"]
# src: qubit.json:conventions.beta.drive = 0.17272858432791746
REC = QUBIT["readout_window"]["recommended_point"]
BETA_READOUT = REC["beta_readout"]      # src: ...recommended_point.beta_readout = 0.4965946799427628
D_EFF_READOUT_M = REC["d_eff_um"] * 1e-6   # src: ...recommended_point.d_eff_um = 57.5
F_R_REC_GHZ = REC["f_r_GHz"]            # src: ...recommended_point.f_r_GHz = 44.5966287014581
KAPPA_REC_MHZ = REC["kappa_MHz"]        # src: ...recommended_point.kappa_MHz = 3.45
G_READOUT_MHZ = REC["g_MHz"]            # src: ...recommended_point.g_MHz = 142.1265211330537
ETA_REC = REC["eta_readout"]            # src: ...recommended_point.eta_readout = 0.11532819289330343

G_DRIVE_MHZ = QUBIT["budgets"]["T1_stack"]["purcell_drive"]["corrected_g_MHz"]
# src: qubit.json:budgets.T1_stack.purcell_drive.corrected_g_MHz = 181.77383656589163
T1_PURCELL_DRIVE_US = QUBIT["budgets"]["T1_stack"]["purcell_drive"]["corrected_T1_us"]
# src: ...purcell_drive.corrected_T1_us = 5290.773559483864
T1_PURCELL_READ_US = (QUBIT["budgets"]["T1_stack"]
                      ["purcell_readout_at_recommended"]["T1_Purcell_us"])
# src: ...purcell_readout_at_recommended.T1_Purcell_us = 36.54018716597618
T1_QP_1E6_US = (QUBIT["budgets"]["T1_stack"]["qp_full_kernel"]
                ["headline_T1_us_at_xqp_1e_6"])
# src: ...qp_full_kernel.headline_T1_us_at_xqp_1e_6 = 5.519965921844335

TAN_DELTA = PHOTONICS["geometry"]["tan_delta"]      # src: photonics.json:geometry.tan_delta = 2e-05
EPS_R = PHOTONICS["geometry"]["eps_core"]           # src: photonics.json:geometry.eps_core = 11.68
Q_MAT = PHOTONICS["modes"]["Q_mat"]                 # src: photonics.json:modes.Q_mat = 52075
GAP_GHZ = tuple(PHOTONICS["modes"]["cavity_line_gap_GHz"])
# src: photonics.json:modes.cavity_line_gap_GHz = [137.5, 177.9]  (at a = 586.74 um)
A_UM = PHOTONICS["geometry"]["a_um"]                # src: photonics.json:geometry.a_um = 586.74
T_OVER_A = PHOTONICS["geometry"]["t_over_a"]        # src: photonics.json:geometry.t_over_a = 0.5

EPS_HOMOG = QUBIT["two_cavity"]["geometry"]["gaps"]["f_r_in_drive_lattice"][
    "homogenised_eps_volume_average"]
# src: qubit.json:two_cavity.geometry.gaps.f_r_in_drive_lattice
#      .homogenised_eps_volume_average = 8.193152102196139

OMEGA01 = 2.0 * np.pi * F01_GHZ * 1e9               # rad/s
LAMBDA0_M = C_LIGHT / (F01_GHZ * 1e9)               # free-space wavelength at f01

T1_MIN_S = _tc.T1_PURCELL_MIN_S                     # 30 us, M6_BRIEF Sec. 2
T1_MARGIN = _tc.T1_MARGIN_TARGET                    # 1.2, P5 recommendation criterion
GD2_MAX = _tc.G_OVER_DELTA_SQ_MAX                   # 0.01, dispersive criterion

# ---- FREE PARAMETERS of this exploration (named, with ranges) ---------------
FREE = {
    "p_participation": {
        "meaning": "fraction of the qubit's electric energy stored in the lossy "
                   "dielectric of the shunt pad",
        "range": [1e-3, 1.0],
        "design_estimates": "see dielectric.participation_design_estimates",
    },
    "tan_delta": {
        "meaning": "loss tangent of that dielectric",
        "range": [1e-7, 2e-5],
        "anchor": "2e-5 MEASURED (photonics.json:geometry.tan_delta, closed by "
                  "Q_mat = 52075 ~ 1/tan_delta at 145 GHz, driven)",
    },
    "f_env": {
        "meaning": "environment enhancement of the short-dipole radiation "
                   "resistance relative to free space",
        "values": "1 (free space) | sqrt(eps_homog) = 2.862 (homogenised slab) "
                  "| (1 + n^3)/2 (dipole at a dielectric half-space, Lukosz "
                  "asymmetry, upper bound)",
    },
    "kappa_geo, eps_eff": {
        "meaning": "P2 pickup shape factor and effective permittivity; enter "
                   "only through beta(d_eff) and are inherited unchanged from "
                   "qubit/coupling.py",
        "values": [_cpl.KAPPA_GEO, _cpl.EPS_EFF],
        "ranges": [list(_cpl.KAPPA_GEO_RANGE), list(_cpl.EPS_EFF_RANGE)],
    },
}


# =============================================================================
# 1.  DIELECTRIC CHANNEL
# =============================================================================
def gamma_diel(p, tan_d, omega=OMEGA01):
    """Gamma_diel = omega01 * p * tan_delta   [s^-1]  (standard participation
    form: the loss rate of the mode is omega times the loss tangent weighted by
    the energy participation)."""
    return omega * np.asarray(p, float) * np.asarray(tan_d, float)


def dielectric_block():
    p_grid = np.logspace(-3, 0, 31)
    td_grid = np.logspace(-7, np.log10(2e-5), 31)
    P, TD = np.meshgrid(p_grid, td_grid, indexing="ij")
    G = gamma_diel(P, TD)
    T1_us = 1e6 / G

    # participation DESIGN ESTIMATES (analytic, clearly labelled)
    p_coplanar_bulk = EPS_R / (1.0 + EPS_R)
    designs = {
        "coplanar_pad_on_bulk_Si": {
            "p": float(p_coplanar_bulk),
            "basis": "conformal-map / image result for a coplanar electrode on "
                     "a semi-infinite substrate: the field energy splits "
                     "eps_r : 1 between substrate and vacuum, p = "
                     "eps_r/(1+eps_r) with eps_r = 11.68 "
                     "(photonics.json:geometry.eps_core)",
            "label": "ANALYTIC DESIGN ESTIMATE",
        },
        "membrane_interdigitated": {
            "p": 0.3,
            "basis": "suspended/undercut membrane with an interdigitated pad: "
                     "most of the field returns through air; p ~ 0.3 is the "
                     "conventional membrane-transmon figure",
            "label": "DESIGN ESTIMATE (order of magnitude)",
        },
        "vacuum_gap_pad": {
            "p": 1e-2,
            "basis": "vacuum-gap (parallel-plate over an air gap) shunt "
                     "capacitor: only the thin support dielectric participates",
            "label": "DESIGN ESTIMATE (order of magnitude)",
        },
    }
    for k, v in designs.items():
        v["Gamma_diel_s^-1_at_measured_tan_delta"] = float(
            gamma_diel(v["p"], TAN_DELTA))
        v["T1_diel_us_at_measured_tan_delta"] = float(
            1e6 / gamma_diel(v["p"], TAN_DELTA))
        v["p_times_tan_delta"] = float(v["p"] * TAN_DELTA)

    # design rules: caps on p*tan_delta
    def cap_for_T1(T1_s):
        return 1.0 / (OMEGA01 * T1_s)

    xqp_ref = (1e-8, 1e-7, 1e-6, 1e-5)
    caps = {}
    for x in xqp_ref:
        T1qp = T1_QP_1E6_US * 1e-6 * (1e-6 / x)
        caps[f"{x:.0e}"] = {
            "T1_qp_us": float(T1qp * 1e6),
            "cap_equal_contribution": float(cap_for_T1(T1qp)),
            "cap_10pct_contribution": float(0.1 * cap_for_T1(T1qp)),
            "p_max_at_measured_tan_delta_equal": float(
                cap_for_T1(T1qp) / TAN_DELTA),
            "p_max_at_measured_tan_delta_10pct": float(
                0.1 * cap_for_T1(T1qp) / TAN_DELTA),
        }

    return {
        "formula": "Gamma_diel = omega01 * p * tan_delta ; T1 = 1/Gamma",
        "omega01_rad_per_s": float(OMEGA01),
        "grid": {
            "p": p_grid.tolist(),
            "tan_delta": td_grid.tolist(),
            "T1_diel_us": T1_us.tolist(),
            "axis_order": "T1_diel_us[i_p][i_tan_delta]",
        },
        "participation_design_estimates": designs,
        "measured_tan_delta": {
            "value": TAN_DELTA,
            "src": "photonics.json:geometry.tan_delta",
            "closure": {
                "Q_mat_measured": Q_MAT,
                "one_over_tan_delta": float(1.0 / TAN_DELTA),
                "closure_pct": float(100.0 * (Q_MAT - 1.0 / TAN_DELTA)
                                     / (1.0 / TAN_DELTA)),
                "src": "photonics.json:modes.Q_mat_note (loss closure, ~4%)",
            },
            "TLS_CAVEAT": (
                "the 145 GHz Q_mat closure was measured with the cavity DRIVEN, "
                "i.e. in the TLS-saturated (high-power) limit, and at 145 GHz / "
                "~ mK it is a bulk HR-Si number. A qubit shunt pad operates at "
                "single-photon power and 40.6 GHz, where unsaturated two-level "
                "systems raise the effective loss tangent -- typically by 1-2 "
                "orders in amorphous surface oxides. tan_delta = 2e-5 is "
                "therefore an OPTIMISTIC LOWER BOUND on the loss, and the whole "
                "dielectric column of this study is a best case. The grid down "
                "to 1e-7 is included only to show what a hypothetical "
                "crystalline/vacuum-gap shunt would have to reach."),
        },
        "design_rules_p_times_tan_delta": caps,
        "design_rule_headline": (
            "p*tan_delta <= 7.10e-7 makes dielectric loss merely EQUAL to the "
            "qp headline at x_qp = 1e-6; p*tan_delta <= 7.10e-9 makes it equal "
            "to the qp channel at x_qp = 1e-8, and 7.10e-10 makes it a 10% "
            "correction there. At the MEASURED tan_delta = 2e-5 those become "
            "p <= 3.55e-2 / 3.55e-4 / 3.55e-5."),
        "reconciliation_with_the_100us_target": {
            "note": ("a p*tan_delta cap of 3.9e-8 corresponds to T1_diel = "
                     "100.5 us, i.e. a round 100 us design target, NOT to "
                     "equality with the qp channel at x_qp = 1e-8 (that cap is "
                     "7.10e-9, an order of magnitude tighter). Both are quoted "
                     "so the criterion behind each number is explicit."),
            "cap_for_T1_diel_100us": float(1.0 / (OMEGA01 * 100e-6)),
            "p_for_100us_at_measured_tan_delta": float(
                1.0 / (OMEGA01 * 100e-6) / TAN_DELTA),
        },
    }


# =============================================================================
# 2.  RADIATION CHANNEL
# =============================================================================
def R_rad_ohm(d_eff_m, lambda_m=LAMBDA0_M, f_env=1.0):
    """Short-dipole (triangular current) radiation resistance, Ohm.

    R_rad = 20 pi^2 (l/lambda)^2 * f_env.  The Hertzian (uniform-current)
    value is 4x larger, 80 pi^2 (l/lambda)^2; the triangular form is the
    correct one for an open-ended electrically-short electrode.
    """
    return 20.0 * np.pi ** 2 * (np.asarray(d_eff_m, float) / lambda_m) ** 2 * f_env


def gamma_rad(beta, d_eff_m, n01=N01, f_env=1.0, omega=OMEGA01,
              lambda_m=LAMBDA0_M):
    """Gamma_rad = (2 e beta n01/hbar)^2 * 2 hbar omega01 * R_rad   [s^-1].

    Derivation (why beta appears, and why it is the SAME beta as in g):
    the antenna's radiation resistance carries vacuum voltage noise
    S_V = 2 hbar omega R_rad at T = 0.  That EMF sits in series with the
    collection-electrode self-capacitance C_c, so the island sees
    beta = C_c/C_Sigma of it (the identical Koch divider that P2 pinned).
    With the island charge operator Q = 2 e n_hat the golden rule gives
    Gamma = |<0|Q|1>|^2 S_V(beta-divided)/hbar^2.
    """
    X = 2.0 * E * np.asarray(beta, float) * n01           # C
    return (X / HBAR) ** 2 * 2.0 * HBAR * omega * R_rad_ohm(d_eff_m, lambda_m,
                                                            f_env)


def gamma_rad_from_g(g_Hz, E_zpf_V_per_m, f_env=1.0, omega=OMEGA01,
                     lambda_m=LAMBDA0_M):
    """The d_eff-CANCELLED form.

    g_ang = (2 e beta n01) E_zpf d_eff / hbar  and R_rad ~ d_eff^2, so d_eff
    cancels exactly out of Gamma_rad:

        Gamma_rad = 40 pi^2 hbar omega01 f_env * g_ang^2
                    / (E_zpf^2 lambda0^2)

    i.e. at FIXED coupling g the radiative loss does not care how g was split
    between electrode length and voltage division -- only the cavity's
    zero-point field can change it.
    """
    g_ang = 2.0 * np.pi * np.asarray(g_Hz, float)
    return (40.0 * np.pi ** 2 * HBAR * omega * f_env * g_ang ** 2
            / (E_zpf_V_per_m ** 2 * lambda_m ** 2))


def slab_te0_neff(f_GHz, t_m, eps_slab):
    """Symmetric air-clad dielectric-slab TE0 effective index (design estimate).

    (kx t/2) tan(kx t/2) = (gamma t/2),  kx = k0 sqrt(eps - n^2),
    gamma = k0 sqrt(n^2 - 1).  Always has exactly one TE0 root.
    """
    k0 = 2.0 * np.pi * f_GHz * 1e9 / C_LIGHT
    n1 = np.sqrt(eps_slab)

    def f(n):
        kx = k0 * np.sqrt(max(n1 ** 2 - n ** 2, 0.0))
        gm = k0 * np.sqrt(max(n ** 2 - 1.0, 0.0))
        return kx * np.tan(kx * t_m / 2.0) - gm

    lo, hi = 1.0 + 1e-12, n1 - 1e-12
    return float(brentq(f, lo, hi, xtol=1e-14, rtol=1e-14))


def radiation_block(E_zpf_drive, E_zpf_read):
    n_homog = float(np.sqrt(EPS_HOMOG))
    f_envs = {
        "free_space": 1.0,
        "homogenised_slab_index": n_homog,
        "half_space_bound": float((1.0 + n_homog ** 3) / 2.0),
    }

    electrodes = {
        "drive_145GHz": dict(beta=BETA_DRIVE, d_eff_m=D_EFF_DRIVE_M,
                             g_MHz=G_DRIVE_MHZ, E_zpf=E_zpf_drive,
                             f_cav_GHz=F_CAV_GHZ),
        "readout_44p6GHz": dict(beta=BETA_READOUT, d_eff_m=D_EFF_READOUT_M,
                                g_MHz=G_READOUT_MHZ, E_zpf=E_zpf_read,
                                f_cav_GHz=F_R_REC_GHZ),
    }
    out = {}
    for name, el in electrodes.items():
        row = {
            "beta": el["beta"], "d_eff_um": el["d_eff_m"] * 1e6,
            "g_MHz": el["g_MHz"], "E_zpf_V_per_m": el["E_zpf"],
            "f_cav_GHz": el["f_cav_GHz"],
            "d_eff_over_lambda0": float(el["d_eff_m"] / LAMBDA0_M),
        }
        for tag, fe in f_envs.items():
            gr = float(gamma_rad(el["beta"], el["d_eff_m"], f_env=fe))
            gr2 = float(gamma_rad_from_g(el["g_MHz"] * 1e6, el["E_zpf"],
                                         f_env=fe))
            row[f"R_rad_ohm.{tag}"] = float(R_rad_ohm(el["d_eff_m"],
                                                      f_env=fe))
            row[f"Gamma_rad_s^-1.{tag}"] = gr
            row[f"T1_rad_us.{tag}"] = 1e6 / gr
            row[f"identity_check_from_g.{tag}_reldev"] = float(
                abs(gr2 - gr) / gr)
        out[name] = row

    totals = {}
    for tag, fe in f_envs.items():
        g_tot = sum(float(gamma_rad(el["beta"], el["d_eff_m"], f_env=fe))
                    for el in electrodes.values())
        totals[tag] = {"Gamma_rad_s^-1": g_tot, "T1_rad_us": 1e6 / g_tot,
                       "f_env": fe}

    # gap placement at f01 (this is WHY there is no protection)
    gap_r = _tc.scaled_gap_GHz(F_R_REC_GHZ)
    a_r_um = _tc.a_readout_um(F_R_REC_GHZ)
    t_drive_m = T_OVER_A * A_UM * 1e-6
    t_read_m = T_OVER_A * a_r_um * 1e-6

    return {
        "formula": ("Gamma_rad = (2 e beta n01/hbar)^2 * 2 hbar omega01 * R_rad,"
                    " R_rad = 20 pi^2 (d_eff/lambda0)^2 * f_env"),
        "lambda0_at_f01_mm": LAMBDA0_M * 1e3,
        "f_env_values": f_envs,
        "f_env_basis": (
            "free space = bare short dipole; homogenised slab = dipole embedded "
            "in an index-n medium, where R_rad scales as n (eta_0/n * (n k0)^2);"
            " half-space bound = dipole lying on a vacuum/dielectric interface, "
            "Lukosz asymmetry P_diel/P_air = n^3, giving (1+n^3)/2 relative to "
            "free space. n = sqrt(8.1932) = 2.862 is the drive lattice's "
            "volume-average index already stored in qubit.json."),
        "electrodes": out,
        "totals_both_electrodes": totals,
        "d_eff_cancellation_law": {
            "statement": ("Gamma_rad = 40 pi^2 hbar omega01 f_env g_ang^2 / "
                          "(E_zpf^2 lambda0^2): at fixed g the electrode length "
                          "and the voltage division cancel EXACTLY. The only "
                          "lever left is E_zpf, i.e. the readout mode volume."),
            "consequence": ("shrinking the electrode does not help unless g is "
                            "allowed to fall with it; the radiative channel is "
                            "locked to the readout requirement chi/kappa >= 1."),
        },
        "photonic_protection_at_f01": {
            "readout_lattice_gap_GHz": list(gap_r),
            "a_r_um": a_r_um,
            "f01_GHz": F01_GHZ,
            "f01_below_lower_edge_GHz": float(gap_r[0] - F01_GHZ),
            "f01_in_readout_gap": bool(gap_r[0] <= F01_GHZ <= gap_r[1]),
            "drive_lattice_a_over_lambda_at_f01": float(A_UM * 1e-6 / LAMBDA0_M),
            "drive_lattice_verdict": (
                "a/lambda = 0.0794 at f01: the drive lattice is deeply "
                "subwavelength and homogenises to eps ~ 8.19 (qubit.json "
                "two_cavity.geometry.gaps.f_r_in_drive_lattice) -- no gap, no "
                "protection at the qubit frequency."),
            "slab_TE0_neff_DESIGN_ESTIMATE": {
                "drive_lattice": {
                    "t_um": t_drive_m * 1e6,
                    "t_over_lambda0": float(t_drive_m / LAMBDA0_M),
                    "n_eff": slab_te0_neff(F01_GHZ, t_drive_m, EPS_HOMOG),
                },
                "readout_lattice": {
                    "t_um": t_read_m * 1e6,
                    "t_over_lambda0": float(t_read_m / LAMBDA0_M),
                    "n_eff": slab_te0_neff(F01_GHZ, t_read_m, EPS_HOMOG),
                },
                "meaning": (
                    "a 2D photonic gap acts on GUIDED (in-plane) modes only. "
                    "In the drive lattice the slab is 0.04 lambda thick at f01 "
                    "and barely guides; in the 3.25x-scaled readout lattice it "
                    "is 0.129 lambda thick and guides well, so a gap that "
                    "covered f01 there WOULD suppress the in-plane channel. "
                    "The vertical light-cone channel is NOT suppressed by any "
                    "2D gap. Splitting Gamma_rad between the two needs a "
                    "slab-LDOS/FDTD run: OUT OF SCOPE, stated, not estimated."),
            },
        },
        "drive_electrode_irreducible_floor": {
            "statement": (
                "retuning f_r can only address the READOUT electrode. The drive "
                "electrode radiates at f01 into the DRIVE lattice, whose gap is "
                "[137.5, 177.9] GHz -- a 3.4x scale mismatch that no choice of "
                "f_r changes. Its d_eff = 20 um and beta = 0.1727 are fixed by "
                "the measured 145 GHz coupling. This is therefore a floor on "
                "the whole architecture, not a design variable."),
            "T1_rad_us": {tag: 1e6 / float(gamma_rad(BETA_DRIVE, D_EFF_DRIVE_M,
                                                     f_env=fe))
                          for tag, fe in f_envs.items()},
            "removable_only_by": ("a conducting enclosure below cutoff (needs "
                                  "< c/2f01 = 3.69 mm transverse dimensions "
                                  "against a 28.6 mm chip), a ground plane, or "
                                  "a smaller-mode-volume 145 GHz cavity that "
                                  "buys the same g with less electrode"),
        },
        "mode_volume_requirement": {
            "law": ("Gamma_rad ~ g^2/E_zpf^2 ~ g^2 V_mode / omega_r: at fixed g "
                    "and f_r the ONLY lever on the readout electrode's "
                    "radiation is the readout mode volume."),
            "current_V_readout_um3": float(V_MODE_M3
                                           * (F_CAV_GHZ / F_R_REC_GHZ) ** 3 * 1e18),
            "targets": {
                f"T1_rad_readout_{t}us": {
                    "V_mode_required_um3": float(
                        V_MODE_M3 * (F_CAV_GHZ / F_R_REC_GHZ) ** 3 * 1e18
                        * (1e6 / t) / float(gamma_rad(BETA_READOUT,
                                                      D_EFF_READOUT_M, f_env=1.0))),
                    "shrink_factor_vs_Maxwell_scaled_L3": float(
                        float(gamma_rad(BETA_READOUT, D_EFF_READOUT_M,
                                        f_env=1.0)) / (1e6 / t)),
                    "V_in_units_of_(lambda_r_over_n)^3": float(
                        V_MODE_M3 * (F_CAV_GHZ / F_R_REC_GHZ) ** 3
                        * (1e6 / t) / float(gamma_rad(BETA_READOUT,
                                                      D_EFF_READOUT_M, f_env=1.0))
                        / (C_LIGHT / (F_R_REC_GHZ * 1e9) / np.sqrt(EPS_C)) ** 3),
                } for t in (10.0, 50.0, 100.0)
            },
            "verdict": ("a Maxwell-scaled L3 at 44.6 GHz has V = 2.29e9 um^3 = "
                        "0.85 (lambda/n)^3. Reaching a 50 us radiative T1 at "
                        "the stored g needs ~0.02 (lambda/n)^3, i.e. a "
                        "deeply sub-diffraction lumped/slot resonator, not a "
                        "scaled photonic-crystal defect. THIS is the concrete "
                        "readout-plane deliverable this study produces."),
        },
        "FDTD_CAVEAT": (
            "the short-dipole R_rad with a scalar f_env is a factor-~4 band, not "
            "a number: the true rate is set by the slab LDOS at f01 at the "
            "electrode position, which requires FDTD/Green-function work that is "
            "explicitly out of scope here. Everything below is quoted as a band "
            "spanning free space to the half-space bound."),
    }


# =============================================================================
# 3.  THE STACK
# =============================================================================
def stack_block(rad_totals, diel):
    x_grid = np.logspace(-8, -5, 31)
    T1_qp = T1_QP_1E6_US * 1e-6 * (1e-6 / x_grid)          # T1 ~ 1/x_qp exactly
    G_qp = 1.0 / T1_qp
    G_pr = 1.0 / (T1_PURCELL_READ_US * 1e-6)
    G_pd = 1.0 / (T1_PURCELL_DRIVE_US * 1e-6)
    G_base = G_qp + G_pr + G_pd

    scenarios = {}
    diel_cases = {
        "none": 0.0,
        "coplanar_bulk_Si_p0.921": float(gamma_diel(
            diel["participation_design_estimates"]
            ["coplanar_pad_on_bulk_Si"]["p"], TAN_DELTA)),
        "membrane_p0.30": float(gamma_diel(0.30, TAN_DELTA)),
        "vacuum_gap_p0.01": float(gamma_diel(0.01, TAN_DELTA)),
    }
    rad_cases = {"none": 0.0}
    rad_cases.update({k: v["Gamma_rad_s^-1"] for k, v in rad_totals.items()})

    for dk, dv in diel_cases.items():
        for rk, rv in rad_cases.items():
            tot = G_base + dv + rv
            scenarios[f"diel={dk} | rad={rk}"] = {
                "Gamma_diel_s^-1": dv, "Gamma_rad_s^-1": rv,
                "T1_total_us_at_xqp": {
                    "1e-08": float(1e6 / (1.0 / (T1_QP_1E6_US * 1e-6 * 100)
                                          + G_pr + G_pd + dv + rv)),
                    "1e-07": float(1e6 / (1.0 / (T1_QP_1E6_US * 1e-6 * 10)
                                          + G_pr + G_pd + dv + rv)),
                    "1e-06": float(1e6 / (1.0 / (T1_QP_1E6_US * 1e-6)
                                          + G_pr + G_pd + dv + rv)),
                    "1e-05": float(1e6 / (1.0 / (T1_QP_1E6_US * 1e-7)
                                          + G_pr + G_pd + dv + rv)),
                },
                "T1_total_us_curve": (1e6 / tot).tolist(),
            }

    # ranking at the two anchor x_qp values, worst-case and best-case radiation
    def rank(x, gd, gr):
        rows = {
            "quasiparticle": 1.0 / (T1_QP_1E6_US * 1e-6 * (1e-6 / x)),
            "purcell_readout": G_pr,
            "purcell_drive": G_pd,
            "dielectric": gd,
            "radiation": gr,
        }
        tot = sum(rows.values())
        return {
            "total_Gamma_s^-1": tot, "T1_total_us": 1e6 / tot,
            "channels": {k: {"Gamma_s^-1": v, "T1_alone_us":
                             (1e6 / v if v > 0 else None),
                             "share_pct": 100.0 * v / tot}
                         for k, v in sorted(rows.items(), key=lambda kv: -kv[1])},
            "ranking": [k for k, _ in sorted(rows.items(),
                                             key=lambda kv: -kv[1])],
        }

    gd_bulk = diel_cases["coplanar_bulk_Si_p0.921"]
    gd_vac = diel_cases["vacuum_gap_p0.01"]
    gr_lo = rad_cases["free_space"]
    gr_hi = rad_cases["half_space_bound"]
    rankings = {
        "x_qp=1e-6 | bulk-Si pad | free-space radiation":
            rank(1e-6, gd_bulk, gr_lo),
        "x_qp=1e-6 | bulk-Si pad | half-space radiation":
            rank(1e-6, gd_bulk, gr_hi),
        "x_qp=1e-6 | vacuum-gap pad | free-space radiation":
            rank(1e-6, gd_vac, gr_lo),
        "x_qp=1e-8 | bulk-Si pad | free-space radiation":
            rank(1e-8, gd_bulk, gr_lo),
        "x_qp=1e-8 | vacuum-gap pad | free-space radiation":
            rank(1e-8, gd_vac, gr_lo),
    }

    # crossing values: where each new channel equals the EXPORTED stack
    cross = {}
    for x in (1e-8, 1e-7, 1e-6, 1e-5):
        Gb = 1.0 / (T1_QP_1E6_US * 1e-6 * (1e-6 / x)) + G_pr + G_pd
        p_cross = Gb / (OMEGA01 * TAN_DELTA)
        # radiation crossing in d_eff of the READOUT electrode: below the
        # C_Sigma knee beta ~ d_eff, so Gamma_rad ~ d_eff^4 exactly.
        g_read_now = float(gamma_rad(BETA_READOUT, D_EFF_READOUT_M, f_env=1.0))
        d_cross = D_EFF_READOUT_M * (Gb / g_read_now) ** 0.25
        cross[f"{x:.0e}"] = {
            "exported_stack_Gamma_s^-1": Gb,
            "exported_stack_T1_us": 1e6 / Gb,
            "p_crossing_at_measured_tan_delta": float(p_cross),
            "p_crossing_note": ("participation at which dielectric loss alone "
                                "equals the entire exported stack"),
            "d_eff_readout_crossing_um_free_space": float(d_cross * 1e6),
            "d_eff_crossing_note": ("readout-electrode length at which "
                                    "radiation alone equals the exported "
                                    "stack; Gamma_rad ~ d_eff^4 below the "
                                    "C_Sigma knee (beta ~ d_eff)"),
            "g_readout_at_crossing_MHz": float(
                G_READOUT_MHZ * (d_cross / D_EFF_READOUT_M) ** 2),
        }

    return {
        "x_qp_grid": x_grid.tolist(),
        "baseline_channels": {
            "T1_qp_us_at_1e-6": T1_QP_1E6_US,
            "T1_purcell_readout_us": T1_PURCELL_READ_US,
            "T1_purcell_drive_us": T1_PURCELL_DRIVE_US,
            "note": ("thermal_floor and parity_bound carry no T1 rate line in "
                     "the export (n_th = 3.46e-9 and a 1/(pi eps1) dephasing "
                     "bound), so the baseline stack is qp + 2 Purcell terms, "
                     "exactly as the exported budget reads it"),
        },
        "scenarios": scenarios,
        "rankings": rankings,
        "crossings": cross,
    }


# =============================================================================
# 4.  THE FIX -- retune f_r so the scaled readout gap covers f01
# =============================================================================
def gap_window():
    """f_r such that the Maxwell-scaled readout gap brackets f01.

    gap_r(f_r) = GAP * f_r/f_cav  (photonics.json gap at a, scaled rigidly),
    so gap_lo <= f01 <= gap_hi  <=>  f01*f_cav/GAP_hi <= f_r <= f01*f_cav/GAP_lo.
    """
    lo = F01_GHZ * F_CAV_GHZ / GAP_GHZ[1]
    hi = F01_GHZ * F_CAV_GHZ / GAP_GHZ[0]
    return float(lo), float(hi)


CHI_SIGN = {"package": +1.0, "koch": -1.0}
# The detuning HANDED TO cavity.chi_dispersive.  The package call sites pass
# Delta = f_r - f01 ("package"); the textbook formula
# chi = g^2 alpha/(Delta(Delta+alpha)) is derived with Delta = f01 - f_r
# ("koch").  See chi_convention_audit().  No package file is modified: the
# validated chi_dispersive is reused verbatim, only the sign of its argument
# changes.


def scan_chunk(deltas, d_eff_um, kappa_MHz, Q_i=Q_I, convention="package"):
    """Vectorised readout scan built from the VALIDATED package primitives.

    Uses two_cavity._beta_readout (shared C_Sigma budget), cavity.E_zpf and
    cavity.chi_dispersive verbatim -- no formula is retyped here.  Adds two
    columns the P5 grid never needed because it started at Delta_r = 4 GHz:
    the 1<->2 dispersive ratio g/|Delta_chi+alpha| and |Delta_chi+alpha|/kappa,
    where Delta_chi + alpha = 0 is the cavity/|1>-|2> resonance.
    """
    sgn = CHI_SIGN[convention]
    D, dE, K = np.meshgrid(np.asarray(deltas, float),
                           np.asarray(d_eff_um, float) * 1e-6,
                           np.asarray(kappa_MHz, float), indexing="ij")
    D, dE, K = D.ravel(), dE.ravel(), K.ravel()
    f_r = F01_GHZ + D
    V_r = V_MODE_M3 * (F_CAV_GHZ / f_r) ** 3
    Ez = _cav.E_zpf(f_r, V_r, EPS_C)
    beta_r, beta_d, cc_r, cc_d, s, over = _tc._beta_readout(dE)
    g = 2.0 * beta_r * E * Ez * dE * N01 / H / 1e9              # GHz
    chi = _cav.chi_dispersive(g, sgn * D, ALPHA_GHZ)            # GHz
    kap = K * 1e-3                                              # GHz
    gd2 = (g / D) ** 2
    T1 = 1.0 / (gd2 * 2 * np.pi * kap * 1e9)
    kint = _tc.kappa_intrinsic_MHz(f_r, Q_i)
    m_chi = np.abs(chi) / kap
    m_T1 = T1 / T1_MIN_S
    eta = 1.0 - kint / K
    d12 = np.abs(sgn * D + ALPHA_GHZ)          # detuning from the 1<->2 line
    g12 = g / d12
    sep12 = d12 / kap
    Gr = gamma_rad_from_g(g * 1e9, Ez, f_env=1.0)               # readout electrode
    return dict(D=D, d=dE, K=K, f_r=f_r, Ez=Ez, beta_r=beta_r, beta_d=beta_d,
                cc_r=cc_r, cc_d=cc_d, s=s, over=over, g=g, chi=chi,
                gd2=gd2, T1=T1, kint=kint, m_chi=m_chi, m_T1=m_T1, eta=eta,
                g12=g12, sep12=sep12, Gr=Gr)


def chi_convention_audit():
    """Which detuning sign belongs in chi = g^2 alpha/(Delta(Delta+alpha))?

    Decided by an INDEPENDENT construction that uses no convention at all:

        chi = chi_01 - chi_12/2 ,  chi_ij = g_ij^2/(w_ij - w_r) ,
        g_12 = sqrt(2) g_01  (transmon ladder)

    whose two poles sit at w_r = w_01 and w_r = w_12 = w_01 + alpha, i.e. at
    the two physical transitions.  Reported here, NOT acted on: no package
    file is modified and no contract number is changed.
    """
    g = G_READOUT_MHZ * 1e-3                     # GHz
    f12 = F01_GHZ + ALPHA_GHZ

    def chi_direct(f_r, gg=g):
        return gg ** 2 / (F01_GHZ - f_r) - gg ** 2 / (f12 - f_r)

    probes = {}
    # the two pole probes are offset by +1 MHz so neither formula divides by 0;
    # 1 MHz off a pole the divergent form is ~1e3 x the regular one, which is
    # all the test needs.
    for tag, f_r in (("recommended_readout_44.597GHz", F_R_REC_GHZ),
                     ("drive_cavity_145GHz", F_CAV_GHZ),
                     ("package_pole_42.229GHz_plus_1MHz",
                      F01_GHZ - ALPHA_GHZ + 1e-3),
                     ("physical_1to2_line_38.964GHz_plus_1MHz", f12 + 1e-3)):
        gg = (G_DRIVE_MHZ * 1e-3 if "drive" in tag else g)
        D = f_r - F01_GHZ
        probes[tag] = {
            "f_r_GHz": float(f_r),
            "chi_MHz_two_pole_construction": float(chi_direct(f_r, gg) * 1e3),
            "chi_MHz_package_convention": float(
                _cav.chi_dispersive(gg, D, ALPHA_GHZ) * 1e3),
            "chi_MHz_koch_convention": float(
                _cav.chi_dispersive(gg, -D, ALPHA_GHZ) * 1e3),
        }
        p = probes[tag]
        ref = p["chi_MHz_two_pole_construction"]
        p["koch_matches_two_pole"] = bool(
            abs(p["chi_MHz_koch_convention"] - ref) <= 1e-9 * abs(ref))
        p["package_over_true"] = (p["chi_MHz_package_convention"] / ref
                                  if ref != 0 else None)

    kap_rec = KAPPA_REC_MHZ
    chi_true = probes["recommended_readout_44.597GHz"][
        "chi_MHz_two_pole_construction"]
    return {
        "status": "FLAGGED, NOT ACTED ON (exploration finding, adjacent to P5)",
        "question": ("cavity.chi_dispersive(g, Delta, alpha) = "
                     "g^2 alpha/(Delta(Delta+alpha)) is correct only for "
                     "Delta = f01 - f_r. Every call site in qubit/cavity.py and "
                     "qubit/two_cavity.py passes Delta = f_r - f01."),
        "decisive_test": (
            "the two-pole construction chi = chi_01 - chi_12/2 has poles at "
            "f_r = f01 and f_r = f12 = f01 + alpha = 38.964 GHz. The package "
            "convention instead places a pole at f_r = f01 - alpha = 42.229 "
            "GHz, where NO transmon transition exists, and shows no divergence "
            "at 38.964 GHz where the cavity is resonant with |1> -> |2>."),
        "probes": probes,
        "impact_on_the_exported_design": {
            "chi_MHz_stored": REC["chi_MHz"],
            "chi_MHz_true": float(chi_true),
            "overestimate_factor": float(abs(REC["chi_MHz"] / chi_true)),
            "chi_over_kappa_stored": REC["chi_over_kappa"],
            "chi_over_kappa_true": float(abs(chi_true) / kap_rec),
            "recommended_point_still_meets_chi_over_kappa_ge_1": bool(
                abs(chi_true) / kap_rec >= 1.0),
            "drive_cavity_error_pct": float(
                100 * abs(probes["drive_cavity_145GHz"]["package_over_true"] - 1)),
            "why_it_survived_AT4_AT5": (
                "the error is (Delta - |alpha|)/(Delta + |alpha|): 3.1% at the "
                "145 GHz drive detuning (AT-4 passes to 3%), but 2.38x at the "
                "4 GHz readout detuning, and it diverges as Delta -> |alpha|. "
                "The AT-4 tolerance is 3% and the AT-5 sample point was never "
                "cross-checked against an independent chi."),
        },
        "recommended_owner": "the P5 / two-cavity owner, not this exploration",
    }


def pick(res, strict, t1_margin=T1_MARGIN, objective="eta", eta_min=None):
    """Feasible set + argmax.  ``objective='eta'`` reproduces the P5 objective
    (maximise readout port efficiency); ``'rad'`` maximises T1_rad instead,
    optionally subject to a floor on the readout efficiency."""
    ok = ((res["m_chi"] >= 1.0) & (res["m_T1"] >= t1_margin)
          & (res["gd2"] <= GD2_MAX) & (res["K"] >= res["kint"]))
    if strict:
        ok &= (res["g12"] <= np.sqrt(GD2_MAX)) & (res["sep12"] >= 10.0)
    if eta_min is not None:
        ok &= (res["eta"] >= eta_min)
    if not ok.any():
        return None, 0
    idx = np.flatnonzero(ok)
    if objective == "rad":
        score = -res["Gr"][idx]
    else:
        score = (res["eta"][idx]
                 + 1e-6 * np.minimum(res["m_chi"][idx],
                                     res["m_T1"][idx] / t1_margin))
    j = idx[int(np.argmax(score))]
    return j, int(ok.sum())


def row_of(res, j, E_zpf_drive):
    Gr_lo = float(res["Gr"][j])
    Gr_drive = float(gamma_rad_from_g(G_DRIVE_MHZ * 1e6, E_zpf_drive, 1.0))
    n_homog = float(np.sqrt(EPS_HOMOG))
    return {
        "Delta_r_GHz": float(res["D"][j]), "f_r_GHz": float(res["f_r"][j]),
        "d_eff_um": float(res["d"][j] * 1e6), "kappa_MHz": float(res["K"][j]),
        "beta_readout": float(res["beta_r"][j]),
        "beta_drive": float(res["beta_d"][j]),
        "g_MHz": float(res["g"][j] * 1e3), "chi_MHz": float(res["chi"][j] * 1e3),
        "chi_over_kappa": float(res["m_chi"][j]),
        "T1_Purcell_us": float(res["T1"][j] * 1e6),
        "margin_T1": float(res["m_T1"][j]),
        "g_over_Delta_sq": float(res["gd2"][j]),
        "g_over_Delta12": float(res["g12"][j]),
        "Delta12_over_kappa": float(res["sep12"][j]),
        "kappa_intrinsic_MHz": float(res["kint"][j]),
        "kappa_ext_MHz": float(res["K"][j] - res["kint"][j]),
        "eta_readout": float(res["eta"][j]),
        "C_c_readout_fF": float(res["cc_r"][j] * 1e15),
        "C_c_drive_fF": float(res["cc_d"][j] * 1e15),
        "C_c_sum_fF": float((res["cc_r"][j] + res["cc_d"][j]) * 1e15),
        "C_c_cap_fF": float(_cpl.c_c_cap() * 1e15),
        "oversubscription": float(res["over"][j]),
        "budget_ok": bool(res["over"][j] <= 1.0),
        "E_zpf_readout_V_per_m": float(res["Ez"][j]),
        "Gamma_rad_readout_s^-1_free_space": Gr_lo,
        "Gamma_rad_total_s^-1_free_space": Gr_lo + Gr_drive,
        "T1_rad_total_us_free_space": 1e6 / (Gr_lo + Gr_drive),
        "T1_rad_total_us_homog_slab": 1e6 / ((Gr_lo + Gr_drive) * n_homog),
        "a_r_um": float(_tc.a_readout_um(res["f_r"][j])),
        "readout_gap_GHz": list(_tc.scaled_gap_GHz(res["f_r"][j])),
    }


def kappa_max_analytic(D, t1_margin=T1_MARGIN, convention="package"):
    """Closed-form kappa ceilings (abs-safe generalisation of two_cavity.C3).

    C3   : kappa^2 <= |alpha| |D| / (|D_chi+alpha| 2 pi T1_min)
    C6   : chi/kappa>=1 together with g <= 0.1|D_chi+alpha| forces
           kappa <= 0.01 |alpha| |D_chi+alpha| / |D|
    with D_chi = sgn * D the detuning handed to chi_dispersive, so that
    D_chi + alpha = 0 is the cavity/|1>-|2> resonance in either convention.
    """
    sgn = CHI_SIGN[convention]
    D = np.asarray(D, float) * 1e9
    a = ALPHA_GHZ * 1e9
    d12 = np.abs(sgn * D + a)
    c3 = np.sqrt(abs(a) * np.abs(D) / (d12 * 2 * np.pi * T1_MIN_S))
    c3m = c3 / np.sqrt(t1_margin)
    c6 = GD2_MAX * abs(a) * d12 / np.abs(D)
    return c3 / 1e6, c3m / 1e6, c6 / 1e6


def fix_block(E_zpf_drive):
    lo, hi = gap_window()
    D_lo, D_hi = lo - F01_GHZ, hi - F01_GHZ

    # ---- analytic efficiency ceiling across the gap-protected window --------
    Dgrid = np.concatenate([np.arange(D_lo, -0.15, 0.005),
                            np.arange(0.15, D_hi + 1e-12, 0.005)])
    kint = _tc.kappa_intrinsic_MHz(F01_GHZ + Dgrid)
    analytic = {"stored_eta_max_reference": 0.202,
                "stored_eta_at_recommended": ETA_REC}
    for conv in ("package", "koch"):
        c3, c3m, c6 = kappa_max_analytic(Dgrid, convention=conv)
        eta_c3 = 1.0 - kint / c3
        eta_c3m = 1.0 - kint / c3m
        eta_all = 1.0 - kint / np.minimum(c3m, c6)
        d12 = np.abs(CHI_SIGN[conv] * Dgrid + ALPHA_GHZ)
        ok = (np.minimum(c3m, c6) > kint) & (d12 > 1e-3)
        if not ok.any():
            analytic[conv] = {"feasible_anywhere_in_window": False}
            continue
        b3 = int(np.argmax(np.where(ok, eta_c3, -np.inf)))
        ba = int(np.argmax(np.where(ok, eta_all, -np.inf)))
        analytic[conv] = {
            "feasible_anywhere_in_window": True,
            "eta_max_no_margin": float(eta_c3[b3]),
            "Delta_at_eta_max_no_margin_GHz": float(Dgrid[b3]),
            "f_r_at_eta_max_no_margin_GHz": float(F01_GHZ + Dgrid[b3]),
            "eta_max_with_C6_and_margin": float(eta_all[ba]),
            "Delta_at_eta_max_with_C6_GHz": float(Dgrid[ba]),
            "f_r_at_eta_max_with_C6_GHz": float(F01_GHZ + Dgrid[ba]),
            "eta_max_with_margin_only": float(
                np.max(np.where(ok, eta_c3m, -np.inf))),
            "curve": {
                "Delta_GHz": Dgrid[::10].tolist(),
                "kappa_ceiling_C3_MHz": c3[::10].tolist(),
                "kappa_ceiling_C6_MHz": c6[::10].tolist(),
                "kappa_intrinsic_MHz": kint[::10].tolist(),
                "eta_max_C3": eta_c3[::10].tolist(),
                "eta_max_C3_C6_margin": eta_all[::10].tolist(),
            },
        }
    # the same ceiling evaluated AT the stored operating point, both ways
    for conv in ("package", "koch"):
        c3, c3m, c6 = kappa_max_analytic(np.array([4.0]), convention=conv)
        analytic.setdefault("at_stored_Delta_4GHz", {})[conv] = {
            "kappa_ceiling_C3_MHz": float(c3[0]),
            "kappa_intrinsic_MHz": float(_tc.kappa_intrinsic_MHz(F_R_REC_GHZ)),
            "eta_max": float(1.0 - _tc.kappa_intrinsic_MHz(F_R_REC_GHZ) / c3[0]),
            "kappa_stored_MHz": KAPPA_REC_MHZ,
            "stored_point_feasible": bool(KAPPA_REC_MHZ <= c3[0]),
        }

    # ---- numerical scan inside the window (chunked over Delta) --------------
    d_grid = np.arange(5.0, 200.0 + 1e-9, 0.5)
    k_grid = np.arange(2.0, 12.0 + 1e-9, 0.02)
    branches = {
        "positive_small_detuning": np.arange(0.15, D_hi + 1e-12, 0.01),
        "negative_detuning": np.arange(D_lo, -0.15 + 1e-12, 0.01),
    }
    scan = {}
    for conv in ("package", "koch"):
        for bname, dl in branches.items():
            for strict in (False, True):
                eta_floors = (0.0, ETA_REC, 0.2, 0.3, 0.4)
                best_eta, s_eta, n_ok_tot = None, -np.inf, 0
                pareto = {f"{e:.4f}": [None, -np.inf] for e in eta_floors}
                for d0 in np.array_split(dl, max(1, len(dl) // 8)):
                    res = scan_chunk(d0, d_grid, k_grid, convention=conv)
                    j, n_ok = pick(res, strict)
                    n_ok_tot += n_ok
                    if j is None:
                        continue
                    if res["eta"][j] > s_eta:
                        s_eta = res["eta"][j]
                        best_eta = row_of(res, j, E_zpf_drive)
                    for e in eta_floors:
                        jr, _ = pick(res, strict, objective="rad", eta_min=e)
                        if jr is not None and -res["Gr"][jr] > pareto[f"{e:.4f}"][1]:
                            pareto[f"{e:.4f}"] = [row_of(res, jr, E_zpf_drive),
                                                  -res["Gr"][jr]]
                tag = "strict_C6" if strict else "P5_constraints"
                scan[f"{conv}|{bname}|{tag}"] = {
                    "n_feasible_grid_points": n_ok_tot,
                    "best_point_max_eta": best_eta,
                    "radiation_efficiency_pareto": {
                        f"eta_min={k}": (None if v[0] is None else {
                            "Delta_r_GHz": v[0]["Delta_r_GHz"],
                            "f_r_GHz": v[0]["f_r_GHz"],
                            "d_eff_um": v[0]["d_eff_um"],
                            "kappa_MHz": v[0]["kappa_MHz"],
                            "g_MHz": v[0]["g_MHz"],
                            "eta_readout": v[0]["eta_readout"],
                            "T1_rad_total_us_free_space":
                                v[0]["T1_rad_total_us_free_space"],
                            "T1_rad_total_us_homog_slab":
                                v[0]["T1_rad_total_us_homog_slab"]})
                        for k, v in pareto.items()},
                    "grid": {"Delta_GHz": [float(dl[0]), float(dl[-1]), 0.01],
                             "d_eff_um": [5.0, 200.0, 0.5],
                             "kappa_MHz": [2.0, 12.0, 0.02]},
                }
    return {
        "chi_convention_audit": chi_convention_audit(),
        "gap_protection_window": {
            "condition": ("GAP_lo * f_r/f_cav <= f01 <= GAP_hi * f_r/f_cav with "
                          "GAP = [137.5, 177.9] GHz at a = 586.74 um"),
            "f_r_GHz_window": [lo, hi],
            "Delta_r_GHz_window": [D_lo, D_hi],
            "current_f_r_GHz": F_R_REC_GHZ,
            "current_f_r_in_window": bool(lo <= F_R_REC_GHZ <= hi),
            "f01_below_current_gap_edge_GHz": float(
                _tc.scaled_gap_GHz(F_R_REC_GHZ)[0] - F01_GHZ),
        },
        "analytic_efficiency_ceiling": analytic,
        "scan": scan,
        "new_constraint_C6": {
            "statement": ("g <= 0.1|Delta_chi + alpha| AND |Delta_chi + alpha| "
                          ">= 10 kappa, i.e. the cavity must stay dispersive "
                          "with respect to the |1>-|2> transition as well as "
                          "|0>-|1>. The P5 window started at Delta_r = 4 GHz, "
                          "2.37 GHz from that line, so it never binds there. "
                          "Inside the gap-protection window the cavity "
                          "approaches the 1<->2 line, chi diverges and the "
                          "dispersive expansion fails; without C6 the scan "
                          "reports a spurious eta -> 1 optimum a few tens of "
                          "MHz from the |1>-|2> resonance."),
            "binding": True,
        },
    }


# =============================================================================
# 5.  VALIDATION
# =============================================================================
def validation_block(E_zpf_drive, E_zpf_read):
    v = {}

    # (V1) reproduce the STORED recommended point with the package's own scan
    pt, stats = _tc.fine_scan(N01, F01_GHZ, ALPHA_GHZ)
    v["V1_recommended_point_reproduced"] = {
        "source": "qubit.two_cavity.fine_scan on its stored default grid",
        "eta_recomputed": pt["eta_readout"], "eta_stored": ETA_REC,
        "eta_reldev": abs(pt["eta_readout"] - ETA_REC) / ETA_REC,
        "kappa_MHz_recomputed": pt["kappa_MHz"], "kappa_MHz_stored": KAPPA_REC_MHZ,
        "g_MHz_recomputed": pt["g_MHz"], "g_MHz_stored": G_READOUT_MHZ,
        "T1_Purcell_us_recomputed": pt["T1_Purcell_us"],
        "T1_Purcell_us_stored": T1_PURCELL_READ_US,
        "n_ok": stats["n_ok"], "n_ok_stored": REC["search"]["n_ok"],
        "PASS": bool(abs(pt["eta_readout"] - ETA_REC) / ETA_REC < 1e-9
                     and abs(pt["g_MHz"] - G_READOUT_MHZ) / G_READOUT_MHZ < 1e-9),
    }

    # (V1b) my own scan_chunk must reproduce fine_scan bit-for-bit on that grid
    res = scan_chunk(np.array([4.0, 4.25]), _tc.D_EFF_FINE_UM, _tc.KAPPA_FINE_MHZ)
    j, _ = pick(res, strict=False)
    v["V1b_local_scan_matches_package"] = {
        "g_MHz_local": float(res["g"][j] * 1e3), "g_MHz_package": pt["g_MHz"],
        "eta_local": float(res["eta"][j]), "eta_package": pt["eta_readout"],
        "reldev_g": float(abs(res["g"][j] * 1e3 - pt["g_MHz"]) / pt["g_MHz"]),
        "PASS": bool(abs(res["g"][j] * 1e3 - pt["g_MHz"]) / pt["g_MHz"] < 1e-12),
    }

    # (V2) qp kernel headline
    T1qp = _qpk.T1_full(1e-6, _qpk.T_QP_ZERO_K) * 1e6
    v["V2_qp_headline"] = {
        "T1_us_recomputed": float(T1qp), "T1_us_stored": T1_QP_1E6_US,
        "reldev": float(abs(T1qp - T1_QP_1E6_US) / T1_QP_1E6_US),
        "PASS": bool(abs(T1qp - T1_QP_1E6_US) / T1_QP_1E6_US < 1e-9),
    }

    # (V3) the question's own anchor: T1_diel = 1.96 us at p = 0.1, 2e-5
    t1d = 1e6 / gamma_diel(0.1, TAN_DELTA)
    v["V3_dielectric_anchor"] = {
        "T1_diel_us_at_p0.1_tan2e-5": float(t1d), "expected_us": 1.96,
        "reldev": float(abs(t1d - 1.96) / 1.96),
        "PASS": bool(abs(t1d - 1.96) / 1.96 < 0.01),
    }

    # (V4) R_rad against the textbook short dipole at l = 0.1 lambda
    r01 = float(R_rad_ohm(0.1 * LAMBDA0_M))
    v["V4_R_rad_textbook"] = {
        "l_over_lambda": 0.1,
        "R_rad_ohm_computed": r01,
        "R_rad_ohm_textbook_short_dipole_20pi2": float(20 * np.pi ** 2 * 0.01),
        "R_rad_ohm_hertzian_80pi2": float(80 * np.pi ** 2 * 0.01),
        "ratio_hertzian_over_short": 4.0,
        "note": ("textbook short dipole (triangular current) at l = lambda/10 is "
                 "1.974 Ohm; the Hertzian uniform-current value is 7.896 Ohm. "
                 "The 4x is the (1/2)^2 current-distribution factor."),
        "PASS": bool(abs(r01 - 1.9739208802178716) < 1e-9),
    }

    # (V5) Gamma_rad -> 0 as d_eff -> 0 (with beta ~ d_eff, expect d^4)
    ds = np.array([57.5e-6, 5.75e-6, 0.575e-6, 0.0575e-6, 0.0])
    betas = np.array([_cpl.beta_selfcap(d) for d in ds])
    grs = gamma_rad(betas, ds)
    v["V5_radiation_limit"] = {
        "d_eff_um": (ds * 1e6).tolist(),
        "beta": betas.tolist(),
        "Gamma_rad_s^-1": grs.tolist(),
        "successive_ratios": (grs[:-2] / grs[1:-1]).tolist(),
        "expected_ratio_per_decade": 1e4,
        "Gamma_at_zero": float(grs[-1]),
        "PASS": bool(grs[-1] == 0.0
                     and abs(grs[0] / grs[1] - 1e4) / 1e4 < 1e-9),
    }

    # (V6) the d_eff-cancellation identity, checked against the direct form
    direct = float(gamma_rad(BETA_READOUT, D_EFF_READOUT_M, f_env=1.0))
    viag = float(gamma_rad_from_g(G_READOUT_MHZ * 1e6, E_zpf_read, f_env=1.0))
    v["V6_d_eff_cancellation_identity"] = {
        "Gamma_direct_s^-1": direct, "Gamma_from_g_s^-1": viag,
        "reldev": float(abs(direct - viag) / direct),
        "PASS": bool(abs(direct - viag) / direct < 1e-9),
    }

    # (V7) E_zpf of the drive cavity against the stored AT-4 budget
    ref = _cav.drive_cavity_budget(N01, F01_GHZ, ALPHA_GHZ)
    v["V7_E_zpf_drive"] = {
        "E_zpf_V_per_m": E_zpf_drive,
        "E_zpf_mV_per_m": E_zpf_drive * 1e3,
        "brief_value_mV_per_m": 83.4,
        "reldev_vs_brief": float(abs(E_zpf_drive * 1e3 - 83.4) / 83.4),
        "g_ref_MHz_from_package": ref["g_MHz"],
        "g_corrected_MHz": ref["g_MHz"] * 2 * BETA_DRIVE,
        "g_corrected_stored": G_DRIVE_MHZ,
        "PASS": bool(abs(ref["g_MHz"] * 2 * BETA_DRIVE - G_DRIVE_MHZ)
                     / G_DRIVE_MHZ < 1e-9),
    }

    # (V8) gap edge: f01 sits 1.69 GHz below the stored readout gap edge
    edge = _tc.scaled_gap_GHz(F_R_REC_GHZ)[0]
    stored_edge = (QUBIT["two_cavity"]["geometry"]["gaps"]
                   ["readout_lattice_gap_GHz"][0])
    v["V8_readout_gap_edge"] = {
        "edge_GHz_recomputed": float(edge), "edge_GHz_stored": stored_edge,
        "f01_GHz": F01_GHZ, "f01_below_edge_GHz": float(edge - F01_GHZ),
        "PASS": bool(abs(edge - stored_edge) < 1e-9),
    }
    v["ALL_PASS"] = all(x["PASS"] for x in v.values() if isinstance(x, dict))
    return v


# =============================================================================
def main():
    E_zpf_drive = float(_cav.E_zpf(F_CAV_GHZ, V_MODE_M3, EPS_C))
    V_r = V_MODE_M3 * (F_CAV_GHZ / F_R_REC_GHZ) ** 3
    E_zpf_read = float(_cav.E_zpf(F_R_REC_GHZ, V_r, EPS_C))

    diel = dielectric_block()
    rad = radiation_block(E_zpf_drive, E_zpf_read)
    stack = stack_block(rad["totals_both_electrodes"], diel)
    fix = fix_block(E_zpf_drive)
    val = validation_block(E_zpf_drive, E_zpf_read)

    rk = stack["rankings"]["x_qp=1e-6 | bulk-Si pad | free-space radiation"]
    rk_vac = stack["rankings"]["x_qp=1e-6 | vacuum-gap pad | free-space radiation"]
    par_pkg = (fix["scan"]["package|positive_small_detuning|strict_C6"]
               ["radiation_efficiency_pareto"]["eta_min=0.2000"])
    par_koch = (fix["scan"]["koch|negative_detuning|strict_C6"]
                ["radiation_efficiency_pareto"]["eta_min=0.2000"])
    verdict = {
        "question_a_dielectric": {
            "answer": "YES -- it dominates, by more than an order of magnitude",
            "T1_diel_us_at_the_analytic_bulk_Si_participation": (
                diel["participation_design_estimates"]
                ["coplanar_pad_on_bulk_Si"]["T1_diel_us_at_measured_tan_delta"]),
            "vs_qp_headline_us": T1_QP_1E6_US,
            "factor_below_qp_headline": float(
                T1_QP_1E6_US / diel["participation_design_estimates"]
                ["coplanar_pad_on_bulk_Si"]["T1_diel_us_at_measured_tan_delta"]),
            "deliverable": ("p <= 3.55e-2 at the measured tan_delta merely TIES "
                            "the qp headline; a vacuum-gap / suspended shunt "
                            "(p ~ 1e-2 -> 19.6 us) is the first design that "
                            "leaves the qp channel dominant. This mirrors "
                            "paper 1's d_eff finding: the shunt pad is a "
                            "designed element, not a trivial fab step, and the "
                            "brief's 'shunt pad, trivial fab' line does not "
                            "survive this channel."),
        },
        "question_b_radiation": {
            "answer": "YES -- it dominates at the stored operating point",
            "T1_rad_us_band": {k: v["T1_rad_us"] for k, v
                               in rad["totals_both_electrodes"].items()},
            "dominant_electrode": "readout (57.5 um, beta = 0.4966)",
            "irreducible_drive_floor_us": rad["drive_electrode_irreducible_floor"][
                "T1_rad_us"],
        },
        "question_c_retune": {
            "gap_protection_window_f_r_GHz": fix["gap_protection_window"][
                "f_r_GHz_window"],
            "current_f_r_in_window": fix["gap_protection_window"][
                "current_f_r_in_window"],
            "answer": ("PARTIALLY. The retune buys a ~12x longer radiative T1 "
                       "AT A HIGHER readout efficiency (eta 0.115 -> 0.20), and "
                       "it widens the kappa ceiling, but it does not close the "
                       "channel: a 2D photonic gap suppresses only the in-plane "
                       "guided continuum, the vertical light cone survives, and "
                       "the drive electrode's floor is untouched by any f_r."),
            "best_trade_at_eta_0p2": {
                "package_convention": par_pkg,
                "koch_convention": par_koch,
                "robustness": ("both conventions land on eta ~ 0.20 with "
                               "T1_rad ~ 5 us; the retune conclusion does not "
                               "depend on the chi sign audit, only its location "
                               "(f_r above vs below f01) does."),
            },
            "eta_ceiling": {
                "stored": ETA_REC,
                "stored_max_reference": 0.202,
                "retuned_no_margin": {
                    c: fix["analytic_efficiency_ceiling"][c].get(
                        "eta_max_no_margin") for c in ("package", "koch")},
                "retuned_with_C6_and_margin": {
                    c: fix["analytic_efficiency_ceiling"][c].get(
                        "eta_max_with_C6_and_margin")
                    for c in ("package", "koch")},
            },
        },
        "headline_ranking_x_qp_1e-6": rk["ranking"],
        "T1_total_us_bulk_Si_pad": rk["T1_total_us"],
        "T1_total_us_vacuum_gap_pad": rk_vac["T1_total_us"],
        "T1_total_us_exported_stack_only": stack["crossings"]["1e-06"][
            "exported_stack_T1_us"],
        "bottom_line": (
            "the exported T1 stack (4.75 us at x_qp = 1e-6) is missing its two "
            "largest lines. With the analytic bulk-Si participation and the "
            "free-space short-dipole estimate the stack collapses to 0.134 us, "
            "and the qp channel -- the export's headline unknown -- contributes "
            "2.4% of the total rate. Both new channels come with a concrete, "
            "quantified design fix and neither is speculative physics: one uses "
            "the paper-1 MEASURED tan_delta, the other the paper-1 MEASURED "
            "electrode geometry and the P2-PINNED beta."),
    }

    payload = {
        "verdict": verdict,
        "_meta": {
            "title": ("EXPLORATION: the two missing T1 channels -- shunt-pad "
                      "dielectric participation and the collection electrode "
                      "as an antenna at f01"),
            "date": "2026-07-27",
            "generator": "scripts/explore_missing_t1_channels.py",
            "status": "EXPLORATION -- not part of the M6 contract; qubit.json "
                      "unchanged; no existing artifact modified",
            "deterministic": True,
            "rng": f"none (no stochastic step); seed {SEED} logged by policy",
            "claim_discipline": ("design study. Every number is parametric in "
                                 "the participation p, the loss tangent, the "
                                 "radiation environment factor f_env and x_qp. "
                                 "No absolute TBG coherence value is claimed."),
        },
        "inputs": {
            "f01_GHz": F01_GHZ, "anharm_GHz": ALPHA_GHZ, "n01": N01,
            "C_Sigma_fF": C_SIGMA_FF, "omega01_rad_per_s": float(OMEGA01),
            "lambda0_mm": LAMBDA0_M * 1e3,
            "beta_drive": BETA_DRIVE, "beta_readout": BETA_READOUT,
            "d_eff_drive_um": D_EFF_DRIVE_M * 1e6,
            "d_eff_readout_um": D_EFF_READOUT_M * 1e6,
            "g_drive_MHz": G_DRIVE_MHZ, "g_readout_MHz": G_READOUT_MHZ,
            "E_zpf_drive_V_per_m": E_zpf_drive,
            "E_zpf_readout_V_per_m": E_zpf_read,
            "f_cav_GHz": F_CAV_GHZ, "f_r_recommended_GHz": F_R_REC_GHZ,
            "kappa_recommended_MHz": KAPPA_REC_MHZ, "eta_recommended": ETA_REC,
            "T1_qp_us_at_1e-6": T1_QP_1E6_US,
            "T1_purcell_readout_us": T1_PURCELL_READ_US,
            "T1_purcell_drive_us": T1_PURCELL_DRIVE_US,
            "tan_delta_measured": TAN_DELTA, "eps_r": EPS_R, "Q_mat": Q_MAT,
            "gap_GHz_at_a": list(GAP_GHZ), "a_um": A_UM, "t_over_a": T_OVER_A,
            "eps_homogenised": EPS_HOMOG, "Q_i": Q_I,
            "V_mode_m3": V_MODE_M3, "eps_core": EPS_C,
            "sources": {
                "f01/alpha/n01/C_Sigma/beta/g/eta/T1": "qubit.json",
                "tan_delta/eps_core/Q_mat/gap/a/t_over_a": "photonics.json",
                "eps_homogenised": ("qubit.json:two_cavity.geometry.gaps."
                                    "f_r_in_drive_lattice."
                                    "homogenised_eps_volume_average"),
                "Q_i/V_mode/f_cav/d_eff_drive": "qubit_targets.json via qubit.inputs",
                "T1_qp": "qubit.qp_kernel.T1_full (recomputed, see V2)",
            },
        },
        "free_parameters": FREE,
        "validation": val,
        "dielectric": diel,
        "radiation": rad,
        "stack": stack,
        "fix_retune_f_r": fix,
    }

    outdir = ROOT / "out" / "m6" / "explore"
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "missing_t1_channels.json"
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return path, payload


if __name__ == "__main__":
    p, pay = main()
    v = pay["validation"]
    print("VALIDATION:", "ALL PASS" if v["ALL_PASS"] else "FAILURES")
    for k, r in v.items():
        if isinstance(r, dict):
            print(f"  {k}: {'PASS' if r['PASS'] else 'FAIL'}")
    d = pay["dielectric"]["participation_design_estimates"]
    print("\nDIELECTRIC (tan_delta = 2e-5 measured):")
    for k, r in d.items():
        print(f"  p = {r['p']:.4f} ({k}): T1_diel = "
              f"{r['T1_diel_us_at_measured_tan_delta']:.4g} us")
    print("\nRADIATION:")
    for k, r in pay["radiation"]["electrodes"].items():
        print(f"  {k}: d_eff = {r['d_eff_um']:.1f} um, beta = {r['beta']:.4f}, "
              f"R_rad = {r['R_rad_ohm.free_space']*1e3:.4g} mOhm, "
              f"T1_rad = {r['T1_rad_us.free_space']:.4g} us (free space)")
    for k, r in pay["radiation"]["totals_both_electrodes"].items():
        print(f"  TOTAL {k}: T1_rad = {r['T1_rad_us']:.4g} us")
    print("\nRANKING at x_qp = 1e-6 (bulk-Si pad, free-space radiation):")
    rk = pay["stack"]["rankings"]["x_qp=1e-6 | bulk-Si pad | free-space radiation"]
    for k, c in rk["channels"].items():
        print(f"  {k:18s} Gamma = {c['Gamma_s^-1']:.4g} s^-1  "
              f"({c['share_pct']:.2f}%)")
    print(f"  T1_total = {rk['T1_total_us']:.4g} us")
    au = pay["fix_retune_f_r"]["chi_convention_audit"]
    imp = au["impact_on_the_exported_design"]
    print("\nCHI CONVENTION AUDIT (flagged, not acted on):")
    print(f"  chi stored {imp['chi_MHz_stored']:.4f} MHz vs true "
          f"{imp['chi_MHz_true']:.4f} MHz -> overestimate "
          f"{imp['overestimate_factor']:.3f}x")
    print(f"  chi/kappa stored {imp['chi_over_kappa_stored']:.4f} -> true "
          f"{imp['chi_over_kappa_true']:.4f} "
          f"(meets >=1: {imp['recommended_point_still_meets_chi_over_kappa_ge_1']})")
    print("\nFIX -- gap window:", pay["fix_retune_f_r"]["gap_protection_window"]
          ["f_r_GHz_window"])
    a = pay["fix_retune_f_r"]["analytic_efficiency_ceiling"]
    for conv in ("package", "koch"):
        c = a[conv]
        if not c.get("feasible_anywhere_in_window"):
            print(f"  [{conv}] no feasible point in the window")
            continue
        print(f"  [{conv}] eta_max (no margin) = {c['eta_max_no_margin']:.4f} at "
              f"f_r = {c['f_r_at_eta_max_no_margin_GHz']:.4f} GHz | "
              f"eta_max (C6+margin) = {c['eta_max_with_C6_and_margin']:.4f} at "
              f"f_r = {c['f_r_at_eta_max_with_C6_GHz']:.4f} GHz")
    for conv, r in a["at_stored_Delta_4GHz"].items():
        print(f"  [{conv}] at the STORED Delta = 4 GHz: kappa_ceiling = "
              f"{r['kappa_ceiling_C3_MHz']:.4f} MHz vs kappa_int = "
              f"{r['kappa_intrinsic_MHz']:.4f} MHz -> stored point feasible: "
              f"{r['stored_point_feasible']}")
    for k, s in pay["fix_retune_f_r"]["scan"].items():
        b = s["best_point_max_eta"]
        if b is None:
            print(f"  {k}: EMPTY (no feasible point)")
            continue
        print(f"  {k} [max eta]: Delta = {b['Delta_r_GHz']:.3f} GHz, f_r = "
              f"{b['f_r_GHz']:.4f}, d = {b['d_eff_um']:.1f} um, kappa = "
              f"{b['kappa_MHz']:.3f} MHz, g = {b['g_MHz']:.2f} MHz, "
              f"eta = {b['eta_readout']:.4f}, T1_rad = "
              f"{b['T1_rad_total_us_free_space']:.4g} us")
        for kk, v in s["radiation_efficiency_pareto"].items():
            if v is not None:
                print(f"      pareto {kk}: eta = {v['eta_readout']:.3f}, "
                      f"g = {v['g_MHz']:.2f} MHz, T1_rad = "
                      f"{v['T1_rad_total_us_free_space']:.3g} us")
    print("\nMODE-VOLUME DELIVERABLE:")
    mv = pay["radiation"]["mode_volume_requirement"]
    print(f"  current V_readout = {mv['current_V_readout_um3']:.4g} um^3")
    for k, v in mv["targets"].items():
        print(f"  {k}: V <= {v['V_mode_required_um3']:.4g} um^3 "
              f"(shrink {v['shrink_factor_vs_Maxwell_scaled_L3']:.4g}x, "
              f"{v['V_in_units_of_(lambda_r_over_n)^3']:.4g} (lam/n)^3)")
    fl = pay["radiation"]["drive_electrode_irreducible_floor"]["T1_rad_us"]
    print(f"  IRREDUCIBLE drive-electrode radiative floor: "
          + ", ".join(f"{k} {v:.4g} us" for k, v in fl.items()))
    print(f"\nwrote {p}")
