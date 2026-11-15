"""P2 -- pinning the coupling-g convention with a lumped-circuit model.

M6_BRIEF.md Sec. 4, deliverable P2.  ``qubit/cavity.py`` deliberately ships the
*reference* convention

    g_ref/h = e * V_zpf * n01 / h ,      V_zpf = E_zpf * d_eff            (REF)

which contains two unpinned assumptions.  This module pins both, and touches
nothing in ``cavity.py``: the reference path stays bit-faithful (AT-4/AT-5),
and the corrected numbers are produced here as an explicit multiplicative
correction on top of it.

=============================================================================
1.  NETWORK TOPOLOGY
=============================================================================

The photonic-crystal mode is a *dielectric* mode: it has no galvanic node.
What the qubit sees is the near field integrated along the paper-1 collection
electrode (out/qj_closure.json: the bare 100 nm vdW stack under-couples; an
engineered collection structure with d_eff ~ 20-25 um is required).  An
electrically small pickup structure (d_eff = 20 um vs a = 587 um lattice
constant, d_eff/a = 0.034) is, to leading order, a Thevenin source:

    EMF   V_oc = integral E.dl  =  E_zpf * d_eff        (the REF numerator)
    source impedance  Z_s = 1/(j w C_c),  C_c = collection-electrode
                                          self-capacitance ("antenna
                                          capacitance" of the pickup)

That EMF is *not* delivered to the junction undivided.  It drives the qubit
island through C_c, and the island already carries the shunt pad C_s and the
junction's own C_J:

        cavity near-field mode (E_zpf)
        collection electrode, length d_eff
                                              island node
      .----------.        C_c                     |
      |  ~  V_oc |-------| |--------o-------------o-------------o
      |          |                  |                           |
      '----------'                 ===  C_s                    ===  C_J
        Thevenin EMF             (shunt pad)                     |
        = E_zpf * d_eff              |             x  E_J  (junction)
                                     |                           |
      ------------------------------ o ------------------------- o
                                    GND

    Node capacitances to ground, as seen from the island:
        C_Sigma  =  C_c + C_s + C_J          <-- the transmon's TOTAL C
    and C_Sigma is *locked* at 13.0 fF by the E_J/E_C = 100 design
    (qubit_targets.json:design.C_fF).  This is the key structural point:
    the collection electrode does not add capacitance on top of the design,
    it SPENDS part of a fixed 13.0 fF budget.  Hence the hard cap

        C_c  <=  C_Sigma - C_J  =  13.0 - 1.0  =  12.0 fF
        beta <=  beta_max = 1 - C_J/C_Sigma = 0.9231.

=============================================================================
2.  VOLTAGE DIVISION  beta  (symbolic)
=============================================================================

At the qubit frequency the network is purely capacitive (all reactances
1/jwC share the same 1/jw), so the divider is frequency independent:

    V_junction   =  V_oc * Z_island / (Z_s + Z_island)
                 =  V_oc * (1/(C_s + C_J)) / (1/C_c + 1/(C_s + C_J))
                 =  V_oc * C_c / (C_c + C_s + C_J)

    ==>   beta  ==  V_junction / (E_zpf * d_eff)  =  C_c / C_Sigma .   (B)

This is exactly the Koch et al. (PRA 76, 042319 (2007), Sec. III) transmon
divider beta = C_g/C_Sigma, with the "gate capacitor" C_g realised here as the
photonic collection electrode.  The reference convention (REF) is the beta = 1
limit, i.e. the whole vacuum gap voltage across the junction -- attainable
only if C_s = C_J = 0.

=============================================================================
3.  THE CHARGE FACTOR:  2e  vs  e   (the factor-2 audit)
=============================================================================

qubit/spectrum.py diagonalises   H = 4 E_C (n - n_g)^2 - (E_J/2)(|n><n+1|+h.c.)
so n is the COOPER-PAIR number on the island and the island charge operator is
Q = 2e n.  With a divided gate voltage V_j = beta V on that node,

    H_charge = (2e n - C_c V)^2 / (2 C_Sigma)  =  4 E_C (n - n_g)^2 ,
        E_C = e^2/(2 C_Sigma),   n_g = C_c V/(2e),
    cross term:  -8 E_C n n_g = -2 e (C_c/C_Sigma) V n = -2 e beta V n .

Therefore the exact coupling operator is

    H_int = 2 beta e V_zpf n^      ==>   g/h = 2 beta e V_zpf n01 / h    (P2)

(equivalently h g = 2 e beta V_rms^0 <0|n|1>, Koch Eq. 3.3).  The reference
used a bare e, i.e. it implicitly treated n as counting SINGLE electrons while
feeding it the Cooper-pair matrix element n01 = 1.3048 from the same charge
ladder.  VERDICT: the 2 is REQUIRED; the reference convention is low by
exactly a factor 2 at fixed beta.  The two errors act in opposite directions:

    g_P2 / g_ref  =  2 * beta          (the correction factor)

and it passes through unity at C_c = C_Sigma/2 = 6.5 fF.

=============================================================================
4.  GEOMETRIC ESTIMATE OF C_c  (design study -- parametric, not precise)
=============================================================================

Two branches, both with named free parameters (defaults + ranges below):

(a) SELF-CAPACITANCE branch (default).  A planar, electrically small pickup
    of tip-to-tip length d_eff has self-capacitance set by the only length in
    the problem:
        C_c = kappa_geo * eps0 * eps_eff * d_eff
    kappa_geo is the O(1) shape factor (thin-strip dipole:
    kappa_geo = pi/[ln(d_eff/a_w) - 1] ~ 2-3 for aspect ratios 4-20;
    coplanar-strip K(k')/K(k) forms give the same 1-4 band).

(b) ENGINEERED-OVERLAP branch (upper bound).  Deliberately add an overlap
    capacitor between the pickup arm and a counter-electrode:
        C_c = eps0 * eps_pp * (w_c * L_ov) / g_c
    With sub-micron gaps this blows past the 12.0 fF cap, so this branch
    simply saturates beta -> beta_max.  It is the design knob that buys the
    ceiling correction 2*beta_max = 1.846.

Both branches are clamped by the budget cap C_c <= C_Sigma - C_J.
No number here is claimed to better than the stated range.
"""
import json

from .cavity import E_zpf, chi_dispersive
from .inputs import (D_EFF_DRIVE_M, EPS0, EPS_C, E, F_CAV_GHZ, H, JUNCTION,
                     ROOT, TARGETS, V_MODE_M3)

# ------------------------------------------------------- measured / locked ---
C_SIGMA_F = TARGETS["design"]["C_fF"] * 1e-15
# src: qubit_targets.json:design.C_fF = 12.999713 fF  (== the brief's 13.0 fF;
# locked by E_C = e^2/2C_Sigma at E_J/E_C = 100).

C_J_F = JUNCTION["contract"]["C_fF"] * 1e-15
# src: junction.json:contract.C_fF = 1.0 fF  (v0.3 wide-junction area estimate;
# the contract itself flags C_fF as an area-scaling estimate, not a measurement)

# ------------------------------------------------- free geometric parameters --
# Design-study parameters.  NOT measured; each carries an explicit range.
KAPPA_GEO = 2.0
KAPPA_GEO_RANGE = (1.0, 4.0)
# O(1) shape factor of the planar pickup, C_c = kappa_geo eps0 eps_eff d_eff.

EPS_EFF = (1.0 + EPS_C) / 2.0          # = 6.34
EPS_EFF_RANGE = ((1.0 + EPS_C) / 2.0, EPS_C)   # = (6.34, 11.68)
# Effective permittivity seen by a planar structure: surface of the HR-Si slab
# (air/Si average) up to fully embedded in the slab.
# src: photonics.json:geometry.eps_core = 11.68

W_C_M = 10e-6
W_C_RANGE_M = (2e-6, 50e-6)
G_C_M = 0.5e-6
G_C_RANGE_M = (0.1e-6, 2.0e-6)
# Overlap-branch electrode width and gap.  Both are >> lithographic floor and
# << the lattice constant a = 586.74 um (photonics.json:geometry.a_um), so the
# pickup stays a lumped element inside the L3 defect.

D_EFF_READOUT_M = 100e-6
# Readout-window sample point d_eff (M6_BRIEF.md Sec. 2 / AT-5 sample row).


# ------------------------------------------------------------ C_c estimates --
def c_c_selfcap(d_eff_m, kappa_geo=KAPPA_GEO, eps_eff=EPS_EFF):
    """Self-capacitance branch: C_c = kappa_geo eps0 eps_eff d_eff  [F]."""
    return kappa_geo * EPS0 * eps_eff * d_eff_m


def c_c_overlap(w_c_m=W_C_M, g_c_m=G_C_M, L_ov_m=D_EFF_DRIVE_M, eps_pp=EPS_C):
    """Engineered-overlap branch: C_c = eps0 eps_pp w_c L_ov / g_c  [F]."""
    return EPS0 * eps_pp * w_c_m * L_ov_m / g_c_m


def c_c_cap(C_Sigma_F=C_SIGMA_F, C_J_F=C_J_F):
    """Hard budget cap on C_c: the island total is locked at C_Sigma."""
    return C_Sigma_F - C_J_F


def beta_from_Cc(C_c_F, C_Sigma_F=C_SIGMA_F, C_J_F=C_J_F):
    """Voltage division beta = C_c/C_Sigma, clamped by the budget cap (B)."""
    return min(C_c_F, c_c_cap(C_Sigma_F, C_J_F)) / C_Sigma_F


def beta_max(C_Sigma_F=C_SIGMA_F, C_J_F=C_J_F):
    """beta ceiling: the whole shunt realised as collection electrode."""
    return 1.0 - C_J_F / C_Sigma_F


def beta_selfcap(d_eff_m, kappa_geo=KAPPA_GEO, eps_eff=EPS_EFF,
                 C_Sigma_F=C_SIGMA_F, C_J_F=C_J_F):
    """beta for the default (self-capacitance) branch."""
    return beta_from_Cc(c_c_selfcap(d_eff_m, kappa_geo, eps_eff),
                        C_Sigma_F, C_J_F)


def beta_bracket(d_eff_m, C_Sigma_F=C_SIGMA_F, C_J_F=C_J_F):
    """(beta_lo, beta_default, beta_hi) over KAPPA_GEO_RANGE x EPS_EFF_RANGE.

    Monotone in both parameters, so the corners bracket the interval.
    """
    lo = beta_selfcap(d_eff_m, KAPPA_GEO_RANGE[0], EPS_EFF_RANGE[0],
                      C_Sigma_F, C_J_F)
    hi = beta_selfcap(d_eff_m, KAPPA_GEO_RANGE[1], EPS_EFF_RANGE[1],
                      C_Sigma_F, C_J_F)
    return lo, beta_selfcap(d_eff_m, C_Sigma_F=C_Sigma_F, C_J_F=C_J_F), hi


# ------------------------------------------------------- coupling operator ---
def correction_factor(beta, charge_units=2.0):
    """g_P2/g_ref = charge_units * beta.  charge_units = 2 (Cooper pair)."""
    return charge_units * beta


def coupling_g(f_GHz, V_m3, d_eff_m, n01, beta, eps_c=EPS_C, charge_units=2.0):
    """g/h in GHz from the pinned operator  H_int = 2 beta e V_zpf n^  (P2).

    ``charge_units=1, beta=1`` reproduces cavity.coupling_g exactly (the
    reference convention) -- that continuity limit is an acceptance test.
    """
    V = E_zpf(f_GHz, V_m3, eps_c) * d_eff_m
    return charge_units * beta * E * V * n01 / H / 1e9


def correct_budget(row, beta, charge_units=2.0):
    """Apply the multiplicative correction to a reference cavity budget row.

    g -> c g,  chi -> c^2 chi,  chi/kappa -> c^2 (kappa unchanged),
    T1_Purcell -> T1/c^2,  (g/Delta)^2 -> c^2 (g/Delta)^2.
    """
    c = correction_factor(beta, charge_units)
    out = dict(row)
    out["beta"] = beta
    out["correction_factor"] = c
    for k in ("g_MHz",):
        if k in row:
            out[k] = row[k] * c
    for k in ("chi_kHz", "chi_MHz", "chi_over_kappa"):
        if k in row:
            out[k] = row[k] * c ** 2
    if "T1_Purcell_us" in row:
        out["T1_Purcell_us"] = row["T1_Purcell_us"] / c ** 2
    return out


# ------------------------------------------------------------- two-cavity ----
def two_cavity_allocation(d_eff_drive_m=D_EFF_DRIVE_M,
                          d_eff_readout_m=D_EFF_READOUT_M,
                          kappa_geo=KAPPA_GEO, eps_eff=EPS_EFF,
                          C_Sigma_F=C_SIGMA_F, C_J_F=C_J_F):
    """Both collection electrodes on the SAME island share one C_Sigma budget.

    C_c,drive + C_c,readout <= C_Sigma - C_J.  If the unconstrained
    self-capacitance estimates over-subscribe the budget, both are scaled by a
    common factor (same fabrication process -> same kappa_geo), which is the
    minimal-assumption allocation.
    """
    cd = c_c_selfcap(d_eff_drive_m, kappa_geo, eps_eff)
    cr = c_c_selfcap(d_eff_readout_m, kappa_geo, eps_eff)
    cap = c_c_cap(C_Sigma_F, C_J_F)
    s = min(1.0, cap / (cd + cr))
    return dict(
        C_c_drive_fF_unconstrained=cd * 1e15,
        C_c_readout_fF_unconstrained=cr * 1e15,
        budget_cap_fF=cap * 1e15,
        oversubscription=(cd + cr) / cap,
        scale_applied=s,
        kappa_geo_effective=kappa_geo * s,
        C_c_drive_fF=cd * s * 1e15,
        C_c_readout_fF=cr * s * 1e15,
        beta_drive=cd * s / C_Sigma_F,
        beta_readout=cr * s / C_Sigma_F,
        correction_drive=2.0 * cd * s / C_Sigma_F,
        correction_readout=2.0 * cr * s / C_Sigma_F,
    )


# ------------------------------------------------------------------ export ---
def build_export(n01, f01_GHz, alpha_GHz, drive_ref, readout_ref):
    """Assemble the out/m6/p2_coupling.json payload."""
    bd_lo, bd, bd_hi = beta_bracket(D_EFF_DRIVE_M)
    br_lo, br, br_hi = beta_bracket(D_EFF_READOUT_M)
    bmax = beta_max()
    cc_ovl = c_c_overlap()

    drive = correct_budget(drive_ref, bd)
    drive["g_ref_MHz"] = drive_ref["g_MHz"]
    drive["C_c_fF"] = c_c_selfcap(D_EFF_DRIVE_M) * 1e15
    drive["d_eff_um"] = D_EFF_DRIVE_M * 1e6
    drive["beta_range"] = [bd_lo, bd_hi]
    drive["correction_range"] = [2 * bd_lo, 2 * bd_hi]
    drive["g_MHz_range"] = [drive_ref["g_MHz"] * 2 * bd_lo,
                            drive_ref["g_MHz"] * 2 * bd_hi]
    # independent recomputation from the operator, not from the scaling
    drive["g_MHz_from_operator"] = 1e3 * coupling_g(
        F_CAV_GHZ, V_MODE_M3, D_EFF_DRIVE_M, n01, bd)
    drive["chi_kHz_from_formula"] = 1e6 * chi_dispersive(
        drive["g_MHz_from_operator"] * 1e-3, F_CAV_GHZ - f01_GHz, alpha_GHz)

    read = correct_budget(readout_ref, br)
    read["g_ref_MHz"] = readout_ref["g_MHz"]
    read["C_c_fF"] = c_c_selfcap(D_EFF_READOUT_M) * 1e15
    read["beta_range"] = [br_lo, br_hi]
    read["correction_range"] = [2 * br_lo, 2 * br_hi]
    read["g_MHz_range"] = [readout_ref["g_MHz"] * 2 * br_lo,
                           readout_ref["g_MHz"] * 2 * br_hi]
    Dr = readout_ref["Delta_r_GHz"]
    read["chi_MHz_from_formula"] = 1e3 * chi_dispersive(
        read["g_MHz"] * 1e-3, Dr, alpha_GHz)
    read["g_over_Delta_sq"] = (read["g_MHz"] * 1e-3 / Dr) ** 2

    return {
        "model": {
            "topology": (
                "Thevenin near-field pickup: EMF V_oc = E_zpf*d_eff in series "
                "with the collection-electrode self-capacitance C_c, driving "
                "the qubit island (shunt pad C_s || junction C_J). "
                "C_Sigma = C_c + C_s + C_J is LOCKED at the E_J/E_C=100 design "
                "value, so C_c spends the budget rather than adding to it. "
                "beta = V_junction/(E_zpf*d_eff) = C_c/C_Sigma "
                "(Koch et al. PRA 76, 042319 (2007) Sec. III divider)."
            ),
            "coupling_operator": "H_int = 2*beta*e*V_zpf*n_hat  (n_hat counts Cooper pairs)",
            "reference_operator": "H_int = e*V_zpf*n_hat  (qubit/cavity.py, unpinned)",
            "charge_factor_verdict": (
                "2e is correct. spectrum.py's charge ladder is the Cooper-pair "
                "number, island charge Q = 2e*n_hat, and n01 = |<0|n|1>| is the "
                "Cooper-pair matrix element; the -8 E_C n n_g cross term gives "
                "exactly 2 e beta V n. The reference's bare e is low by 2x at "
                "fixed beta."
            ),
            "params": {
                "C_Sigma_fF": C_SIGMA_F * 1e15,
                "C_J_fF": C_J_F * 1e15,
                "C_c_budget_cap_fF": c_c_cap() * 1e15,
                "kappa_geo": KAPPA_GEO,
                "eps_eff": EPS_EFF,
                "w_c_um": W_C_M * 1e6,
                "g_c_um": G_C_M * 1e6,
                "eps_pp": EPS_C,
                "C_c_selfcap_formula": "C_c = kappa_geo*eps0*eps_eff*d_eff",
                "C_c_overlap_formula": "C_c = eps0*eps_pp*w_c*L_ov/g_c",
                "C_c_overlap_fF_at_defaults": cc_ovl * 1e15,
                "C_c_overlap_saturates_cap": bool(cc_ovl > c_c_cap()),
            },
            "ranges": {
                "kappa_geo": list(KAPPA_GEO_RANGE),
                "eps_eff": list(EPS_EFF_RANGE),
                "w_c_um": [W_C_RANGE_M[0] * 1e6, W_C_RANGE_M[1] * 1e6],
                "g_c_um": [G_C_RANGE_M[0] * 1e6, G_C_RANGE_M[1] * 1e6],
                "note": ("kappa_geo and eps_eff are the only parameters that "
                         "move the default branch; C_c is monotone increasing "
                         "in both, so the corner values bracket beta."),
            },
        },
        "beta": {
            "drive": bd, "drive_range": [bd_lo, bd_hi],
            "readout_sample": br, "readout_sample_range": [br_lo, br_hi],
            "beta_max": bmax,
            "beta_breakeven": 0.5,
            "C_c_breakeven_fF": 0.5 * C_SIGMA_F * 1e15,
            "definition": "beta = C_c/C_Sigma = V_junction/(E_zpf*d_eff)",
        },
        "correction_factor": {
            "formula": "g_P2/g_ref = 2*beta",
            "drive": 2 * bd, "drive_range": [2 * bd_lo, 2 * bd_hi],
            "readout_sample": 2 * br,
            "readout_sample_range": [2 * br_lo, 2 * br_hi],
            "ceiling": 2 * bmax,
            "scaling": "chi ~ c^2, chi/kappa ~ c^2, T1_Purcell ~ 1/c^2, kappa unchanged",
        },
        "drive_cavity_reference": drive_ref,
        "drive_cavity_corrected": drive,
        "readout_sample_reference": readout_ref,
        "readout_sample_corrected": read,
        "two_cavity_budget": two_cavity_allocation(),
        "provenance": {
            "module": "qubit/coupling.py (M6 deliverable P2)",
            "deterministic": True,
            "rng_seed": None,
            "rng_note": "no stochastic step: closed-form lumped-circuit algebra",
            "reads": {
                "C_Sigma_fF": "qubit_targets.json:design.C_fF",
                "C_J_fF": "junction.json:contract.C_fF",
                "eps_core": "photonics.json:geometry.eps_core",
                "a_um": "photonics.json:geometry.a_um",
                "V_mode_m3": "qubit_targets.json:inputs.V_mode_m3",
                "d_eff_drive_um": "qubit_targets.json:inputs.d_eff_drive_um",
                "f_cav_GHz": "qubit_targets.json:inputs.f_cav_GHz",
                "Q_i": "qubit_targets.json:inputs.Q_i",
                "n01": "recomputed by qubit.design_point() (spectrum.n01_element)",
                "reference_rows": "qubit/cavity.py (AT-4 / AT-5 path, unmodified)",
                "d_eff_engineering": ("out/qj_closure.json:requirement."
                                      "d_eff_min_10x_um = 21.59 um, "
                                      "d_eff_min_90pct_um = 25.24 um"),
            },
            "unmodified": ["qubit/cavity.py", "qubit/inputs.py",
                           "qubit/spectrum.py", "qubit/dome.py", "qubit/qp.py"],
            "citation": ("Koch et al., 'Charge-insensitive qubit design derived "
                         "from the Cooper pair box', PRA 76, 042319 (2007), "
                         "Sec. III: h g = 2 e beta V_rms^0 <0|n|1>, "
                         "beta = C_g/C_Sigma."),
        },
    }


def main():
    """Recompute the P2 budgets and write out/m6/p2_coupling.json."""
    import qubit as _q

    d = _q.design_point()
    n01, alpha = d["n01"], d["anharm_GHz"]
    f01 = d["f01_GHz"]

    from . import cavity as _cav
    drive_ref = _cav.drive_cavity_budget(n01, f01, alpha)
    rows = _cav.readout_window_scan(n01, f01, alpha)
    readout_ref = next(r for r in rows
                       if r["Delta_r_GHz"] == 7.0 and r["d_eff_um"] == 100.0
                       and r["kappa_MHz"] == 1.0)

    payload = build_export(n01, f01, alpha, drive_ref, readout_ref)
    outdir = ROOT / "out" / "m6"
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "p2_coupling.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return path, payload


if __name__ == "__main__":
    p, pay = main()
    print(f"wrote {p}")
    print(f"beta_drive   = {pay['beta']['drive']:.6f}  "
          f"range {pay['beta']['drive_range'][0]:.4f}-{pay['beta']['drive_range'][1]:.4f}")
    print(f"beta_readout = {pay['beta']['readout_sample']:.6f}  "
          f"range {pay['beta']['readout_sample_range'][0]:.4f}-"
          f"{pay['beta']['readout_sample_range'][1]:.4f}")
    print(f"correction   drive {pay['correction_factor']['drive']:.4f}  "
          f"readout {pay['correction_factor']['readout_sample']:.4f}  "
          f"ceiling {pay['correction_factor']['ceiling']:.4f}")
    for k in ("drive_cavity_corrected", "readout_sample_corrected"):
        r = pay[k]
        print(f"{k}: g={r['g_MHz']:.3f} MHz  "
              f"chi={r.get('chi_kHz', r.get('chi_MHz')):.4f}  "
              f"chi/kappa={r['chi_over_kappa']:.6g}  "
              f"T1_P={r['T1_Purcell_us']:.3f} us")
    tc = pay["two_cavity_budget"]
    print(f"two-cavity: oversubscription={tc['oversubscription']:.4f} "
          f"scale={tc['scale_applied']:.4f} "
          f"corr_d={tc['correction_drive']:.4f} corr_r={tc['correction_readout']:.4f}")
