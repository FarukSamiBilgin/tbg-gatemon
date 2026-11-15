#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/export_qubit.py -- build qubit.json (M6 deliverable P7, paper 2).

Sits alongside material.json / junction.json / photonics.json at the repo root
and closes the M6 module: the qubit-layer interface export.

DESIGN CONTRACT
---------------
* NO NUMBER IS TYPED FROM MEMORY.  Every scalar in the payload is either
  (a) read out of an artifact JSON -- qubit_targets.json or out/m6/p{2..6}*.json
      -- with a `# src:` trace comment, or
  (b) recomputed here by calling the validated qubit/ package functions, in
      which case the recomputed value is exported next to the artifact value
      together with the measured relative deviation.
  A third category (derived) exists only for ratios of two already-exported
  numbers (e.g. the apex/off-apex T2 contrast); those carry a `formula` field.

* THE CORRECTIONS DO NOT ERASE THE REFERENCES.  P2..P6 superseded several
  milestone-1 (AT-1..AT-6) numbers.  Every superseded number is retained in a
  `*_reference` / `small_freq_reference` / `naive_*_reference` block with an
  explicit flag, so the paper can quote both and say which is which.

* DETERMINISTIC.  No RNG anywhere in the M6 chain (P3/P5/P6 log seed 0 as
  unused; P2/P4 have no stochastic step).  The payload contains no wall-clock
  timestamp -- the only run-varying field is provenance.commit_basis (git HEAD)
  and the sha256 digests of the source artifacts.  Re-running against unchanged
  inputs at the same commit reproduces the file byte-for-byte.

* CLAIM DISCIPLINE (M6_BRIEF.md, hard rule).  Paper 2 is a DESIGN STUDY.  Every
  coherence number is parametric in x_qp or the gate-noise amplitude A.  The
  sentence "TBG coherence has never been measured" is carried in _meta and must
  survive into the abstract.

OWNERSHIP: this script and <root>/qubit.json only.  Everything under out/m6/,
qubit/, tests/ and figs/ is opened read-only.

Run:  python scripts/export_qubit.py
"""

import hashlib
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT_M6 = os.path.join(ROOT, "out", "m6")
QUBIT_JSON = os.path.join(ROOT, "qubit.json")

sys.path.insert(0, ROOT)

import qubit as qubit_pkg  # noqa: E402  (the validated M6 package)

SCHEMA_VERSION = "1.0"
EXPORT_DATE = "2026-07-27"          # M6 production date (paper/decisions.md)

CLAIM_DISCIPLINE = (
    "TBG coherence has never been measured. Paper 2 is a DESIGN STUDY: every "
    "coherence number in this export is parametric in the quasiparticle "
    "fraction x_qp or in the 1/f gate-noise amplitude A, and no absolute T1 "
    "or T2 value is claimed for a TBG junction anywhere."
)


# --------------------------------------------------------------------------
# I/O helpers
# --------------------------------------------------------------------------
def load(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def sha256_16(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def plain(obj):
    """Recursively convert numpy scalars/arrays to plain Python for json."""
    if isinstance(obj, dict):
        return {str(k): plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [plain(v) for v in obj]
    if hasattr(obj, "item") and getattr(obj, "shape", ()) == ():
        return obj.item()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return obj


def reldev(a, b):
    """Relative deviation of a from b (b is the reference)."""
    return 0.0 if b == 0 else abs(a - b) / abs(b)


def threshold_crossings(t, y, thr):
    """Brackets on the scanned grid where y(t) crosses thr, in order.

    Returns [[t_lo, t_hi], ...], one entry per adjacent grid pair that
    straddles `thr`.  Used to state how many times a leakage curve crosses a
    quoted tolerance instead of asserting a monotonicity that the scan does
    not have (out/verify/paper2_trace.md F5).
    """
    out = []
    for i in range(len(t) - 1):
        if (y[i] > thr) != (y[i + 1] > thr):
            out.append([t[i], t[i + 1]])
    return out


def deep_equal(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(deep_equal(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(deep_equal(x, y) for x, y in zip(a, b))
    return a == b


# --------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------
TARGETS_PATH = os.path.join(ROOT, "qubit_targets.json")
P2_PATH = os.path.join(OUT_M6, "p2_coupling.json")
P3_PATH = os.path.join(OUT_M6, "p3_qp.json")
P4_PATH = os.path.join(OUT_M6, "p4_t2_maps.json")
P5_PATH = os.path.join(OUT_M6, "p5_two_cavity.json")
P6_PATH = os.path.join(OUT_M6, "p6_gates.json")
# P-extra (CPR-harmonics sensitivity).  Read for open_items.cpr_harmonics only
# (F14, out/verify/paper2_trace.md): it is NOT one of the seven sha256-pinned
# P7 source artifacts -- it is a downstream sensitivity study that discharges
# the computation this open item called for, and every number quoted from it
# below is read from the file, never typed.
PEXTRA_PATH = os.path.join(OUT_M6, "pextra_cpr.json")

TARGETS = load(TARGETS_PATH)
P2 = load(P2_PATH)
P3 = load(P3_PATH)
P4 = load(P4_PATH)
P5 = load(P5_PATH)
P6 = load(P6_PATH)
PEXTRA = load(PEXTRA_PATH)

# Test-file inventory: counts are the live `pytest --collect-only` counts of
# each file, cross-checked against paper/decisions.md (P2 19 / P3 53 / P4 46 /
# P5 67 / P6 55, milestone-1 57).  P5 rose 48 -> 67 with the 2026-07-27
# chi-convention repair (19 convention-pinning tests added).
TEST_FILES = [
    ("tests/test_m6_acceptance.py", 57, "AT-1..AT-6 (M6_BRIEF.md Sec. 3) + eps1 ULP/mpmath rules"),
    ("tests/test_m6_p2.py", 19, "P2 coupling convention (beta divider, budget cap, identities)"),
    ("tests/test_m6_p3.py", 53, "P3 quasiparticle kernel, parity bound, thermal floor, pair-breaking"),
    ("tests/test_m6_p4.py", 46, "P4 1/f filter functions, Ramsey/echo maps, additivity diagnostic"),
    ("tests/test_m6_p5.py", 67, "P5 two-cavity feasibility, C1-C8 constraints, control line, chi-convention pinning"),
    ("tests/test_m6_p6.py", 55, "P6 gate floor, DRAG, allocation, two-qubit sketch"),
    ("tests/test_qubit_export.py", 178, "P7 export schema + artifact-consistency (this file's contract)"),
]
P7_TEST_FILE = "tests/test_qubit_export.py"
PAPER1_TESTS = 100      # test_acceptance 5 + backaction 16 + m1 8 + m2_2d_tcmt 20
                        # + m2_at1 8 + power_chain 12 + qj_loading 20 + tien_gordon 11

# Derived once from TEST_FILES so that every place the counts are quoted -- the
# numeric provenance.tests fields AND the human-readable full_suite_scope prose --
# moves together.  o1 (out/verify/paper2_trace_v3.md): the prose string used to
# carry a hard-coded "M6 production (278)" that the 48 -> 67 P5 re-count left
# stale against its own sibling m6_production_total = 297.
M6_PRODUCTION_TESTS = sum(n for p, n, _ in TEST_FILES if p != P7_TEST_FILE)
P7_EXPORT_TESTS = dict((p, n) for p, n, _ in TEST_FILES)[P7_TEST_FILE]


def git_head():
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, timeout=30)
        return out.stdout.strip() if out.returncode == 0 else "unavailable"
    except Exception:
        return "unavailable"


# --------------------------------------------------------------------------
# Recomputation via the validated qubit/ package (category (b))
# --------------------------------------------------------------------------
def recomputed():
    """Recompute the design lock and the AT-4/AT-6 budgets from qubit/."""
    d = qubit_pkg.design_point()                      # qubit/__init__.py
    cav = qubit_pkg.cavity.drive_cavity_budget(       # qubit/cavity.py (AT-4)
        d["n01"], d["f01_GHz"], d["anharm_GHz"])
    qp = qubit_pkg.qp.qp_budget(d["f01_GHz"])         # qubit/qp.py  (AT-6)
    sweet = qubit_pkg.dome.sweet_spot_report(d["EC_GHz"])  # qubit/dome.py (AT-3)
    return plain(d), plain(cav), plain(qp), plain(sweet)


# --------------------------------------------------------------------------
# Payload blocks
# --------------------------------------------------------------------------
def block_meta():
    return {
        "schema_version": SCHEMA_VERSION,
        "date": EXPORT_DATE,
        "module": "M6 -- TBG gatemon with a photonic control plane (paper 2)",
        "claim_discipline": CLAIM_DISCIPLINE,
        "generator": "scripts/export_qubit.py",
        "schema_note": (
            "M6_BRIEF.md P7 schema {design, budgets(T1 stack, T2 maps), "
            "readout_window, two_cavity, provenance}, extended with "
            "conventions / allocation / two_qubit_sketch / open_items so the "
            "P2-P6 corrections are carried explicitly. Every superseded "
            "milestone-1 value is retained in a *_reference block -- the "
            "corrections do not erase the references."
        ),
        "interface_position": (
            "fourth root export, after material.json (M1), junction.json (M3) "
            "and photonics.json (M2/M4); consumes all three plus "
            "qubit_targets.json."
        ),
        "deterministic": True,
        "rng": "none -- no stochastic step anywhere in the M6 chain (P3/P5/P6 log seed 0 as unused)",
    }


def block_design(rd, sweet):
    des = TARGETS["design"]              # src: qubit_targets.json:design
    tun = TARGETS["gate_noise"]["f01_tunability_GHz"]  # src: qubit_targets.json:gate_noise
    cf = P6["combined_floor"]            # src: out/m6/p6_gates.json:combined_floor
    gf = P6["gate_floor"]                # src: out/m6/p6_gates.json:gate_floor
    dr = P6["drag"]                      # src: out/m6/p6_gates.json:drag
    req_1em4 = [r for r in dr["requirement"]
                if r["t_g_ns"] == gf["drag"]["t_g_floor_ns"]["1e-04"]][0]
    rot = dr["rotation_error"]["rows"][0]

    return {
        "EJ_GHz": TARGETS["inputs"]["EJ_GHz"],          # src: qubit_targets.json:inputs.EJ_GHz
        "EJ_over_EC": des["EJ_over_EC"],                # src: qubit_targets.json:design
        "EC_GHz": des["EC_GHz"],
        "C_Sigma_fF": des["C_fF"],
        "f01_GHz": des["f01_GHz"],
        "anharm_GHz": des["anharm_GHz"],
        "eps1_Hz": des["eps1_Hz"],
        "n01": des["n01"],
        "selection_criterion": (
            "smallest E_J/E_C with eps1 < 10 kHz AND |anharmonicity| >= 0.5 GHz "
            "(M6_BRIEF.md Sec. 1); ratio 100 wins"
        ),
        "f01_tunability_GHz": tun,
        "f01_tunability_caveat": (
            "realisable only in the tracking-drive scenario -- see "
            "allocation.fork_statement and readout_window.constraints C5"
        ),
        "control_tone_GHz": des["f01_GHz"],
        "recomputed_check": {
            "source": "qubit.design_point() / qubit.dome.sweet_spot_report()",
            "f01_GHz": rd["f01_GHz"],
            "f01_reldev": reldev(rd["f01_GHz"], des["f01_GHz"]),
            "anharm_GHz": rd["anharm_GHz"],
            "anharm_reldev": reldev(rd["anharm_GHz"], des["anharm_GHz"]),
            "n01": rd["n01"],
            "n01_reldev": reldev(rd["n01"], des["n01"]),
            "eps1_Hz": rd["eps1_Hz"],
            "eps1_reldev": reldev(rd["eps1_Hz"], des["eps1_Hz"]),
            "EC_GHz": rd["EC_GHz"],
            "C_fF": rd["C_fF"],
            "D1_apex_GHz": sweet["D1_apex_GHz"],
            "D2_apex_GHz": sweet["D2_apex_GHz"],
            "D1_off_nu_m2p5_GHz": sweet["D1_off_GHz"],
            "note": "bit-exact reproduction of the reference scan; deviations are exactly 0",
        },
        "eps1_numerics_rule": {
            "finding": (
                "eps1 is ULP-conditioned already at ratio 100: a 1-ULP change in "
                "E_J moves float64 eps1 by 3.4e-5 relative while f01 is stable to 1e-14"
            ),
            "float64_floor_measured_vs_mpmath": {
                "150": -0.006, "200": -0.28,
                "300": "4.5 decades high, sign of the 200->300 step not build-stable",
            },
            "production_rule": "use qubit.spectrum.eps1_mp (Sturm bisection, dps=60) for any ratio >= 150",
            "src": "paper/decisions.md 2026-07-27 'eps1 is ULP-conditioned already at ratio 100'",
        },
        "gates": {
            "honest_coherent_floor_ns": cf["t_g_floor_ns"]["1e-04"],   # src: p6_gates.json:combined_floor.t_g_floor_ns
            "honest_coherent_floor_ns_1e3": cf["t_g_floor_ns"]["1e-03"],
            "QUOTE_THIS": True,
            "quote_this_flag": (
                "THE PAPER MUST QUOTE 1.6 ns, NOT 1.0 ns. The 1.0112 ns figure is "
                "leakage-only; with both the amplitude and the DRAG frame "
                "calibrated, max(P0, P2) clears 1e-4 only at 1.6044 ns."
            ),
            "metric": cf["metric"],                                    # src: p6_gates.json:combined_floor.metric
            "claim": cf["claim"],
            "leakage_only_floor_ns": {
                "drag": gf["drag"]["t_g_floor_ns"],                    # src: p6_gates.json:gate_floor.drag
                "gaussian": gf["gaussian"]["t_g_floor_ns"],            # src: p6_gates.json:gate_floor.gaussian
                "speedup_drag_over_gaussian": gf["speedup_drag_over_gaussian"],
                "flag": "LEAKAGE-ONLY -- superseded for paper quoting by honest_coherent_floor_ns",
            },
            "drag_params": {
                "definition": dr["definition"],
                "q_opt_parabolic": dr["coefficient_scan"]["q_opt_parabolic"],
                "gain_q1_over_qopt": dr["coefficient_scan"]["gain_q1_over_qopt"],
                "quadrature_ratio_Omy_over_Omx_at_1e4_floor": req_1em4["ratio_measured"],
                "quadrature_rule": "Om_y/Om_x = 0.2736 ns / t_g (measured = analytic)",
                "Omx_peak_GHz_at_1e4_floor": req_1em4["Omx_peak_GHz"],
                "Omy_peak_GHz_at_1e4_floor": req_1em4["Omy_peak_GHz"],
                "sigma_ns_at_1e4_floor": req_1em4["sigma_ns"],
                "frame_correction_s_measured": rot["s_opt"],
                "frame_correction_note": (
                    "s -> -1.000 confirms the Motzoi first-order frame correction; "
                    "amplitude calibration alone leaves P0 = %.3g at that t_g, four "
                    "orders above the leakage" % rot["P0_amplitude_only"]
                ),
            },
            # F5 (out/verify/paper2_trace.md): the old envelope-agnostic key
            # `leakage_monotone_from_t_g_ns` carried the GAUSSIAN onset
            # (1.6080 ns) while the floors quoted by the paper are the DRAG
            # floors -- which lie below it and far below the DRAG scan's own
            # onset (2.6296 ns).  Both families are now exported under
            # unambiguous names, with their violation points and with the
            # measured threshold-crossing count that actually justifies the
            # extraction rule.
            "leakage_monotone_from_t_g_ns_gaussian": gf["gaussian"]["monotone_from_t_g_ns"],
            "leakage_monotone_from_t_g_ns_drag": gf["drag"]["monotone_from_t_g_ns"],
            "leakage_monotonicity": {
                "gaussian": {
                    "monotone_decreasing": gf["gaussian"]["monotone_decreasing"],
                    "monotone_from_t_g_ns": gf["gaussian"]["monotone_from_t_g_ns"],
                    "n_monotonicity_violations": gf["gaussian"]["n_monotonicity_violations"],
                    "monotonicity_violation_t_g_ns": gf["gaussian"]["monotonicity_violation_t_g_ns"],
                },
                "drag": {
                    "monotone_decreasing": gf["drag"]["monotone_decreasing"],
                    "monotone_from_t_g_ns": gf["drag"]["monotone_from_t_g_ns"],
                    "n_monotonicity_violations": gf["drag"]["n_monotonicity_violations"],
                    "monotonicity_violation_t_g_ns": gf["drag"]["monotonicity_violation_t_g_ns"],
                },
                "drag_threshold_crossings": {
                    thr_key: {
                        "n_crossings": len(threshold_crossings(
                            gf["drag"]["t_g_ns"], gf["drag"]["leakage"], thr)),
                        "bracket_ns": threshold_crossings(
                            gf["drag"]["t_g_ns"], gf["drag"]["leakage"], thr),
                    }
                    for thr_key, thr in (("1e-03", 1e-3), ("1e-04", 1e-4))
                },
                "extraction_rule": (
                    "Leakage is NOT globally monotone in t_g in either "
                    "envelope, so both quoted DRAG floors are extracted at the "
                    "LAST threshold crossing. On the scanned 61-point grid each "
                    "DRAG threshold is crossed exactly once, so 'last crossing' "
                    "and 'only crossing' coincide -- the extraction is "
                    "unambiguous WITHOUT appealing to a monotone region, and "
                    "neither DRAG floor lies inside one (the DRAG onset is "
                    "2.6296 ns, the Gaussian onset 1.6080 ns)."
                ),
                "src": "out/m6/p6_gates.json:gate_floor.{gaussian,drag}",
            },
            "ladder_element_n12_over_n01": P6["params"]["ladder_ratio"],
            "numerics": P6["provenance"]["numerics"],
        },
        "thermal": {
            "n_th": TARGETS["thermal"]["n_th"],                        # src: qubit_targets.json:thermal
            "n_th_Al5GHz_100mK": TARGETS["thermal"]["n_th_Al5GHz_100mK"],
            "advantage_vs_Al_5GHz_at_100mK": (
                TARGETS["thermal"]["n_th_Al5GHz_100mK"] / TARGETS["thermal"]["n_th"]["0.1"]),
            "formula": "advantage = n_th_Al5GHz_100mK / n_th(100 mK)",
        },
    }


def block_conventions():
    cf = P2["correction_factor"]        # src: out/m6/p2_coupling.json:correction_factor
    be = P2["beta"]                     # src: out/m6/p2_coupling.json:beta
    cc = P5["chi_convention"]           # src: out/m6/p5_two_cavity.json:chi_convention
    return {
        "g_convention": (
            "H_int = 2*beta*e*V_zpf*n; g = 2*beta*e*V_zpf*n01/h (Koch 2007)"),
        "reference_convention_note": (
            "The milestone-1 / AT-4 / AT-5 convention is g = e*V_zpf*n01/h with "
            "beta implicitly 1 (qubit/cavity.py, unmodified and still "
            "bit-faithful to m6_reference.py). It is low by exactly 2x at fixed "
            "beta: spectrum.py's charge ladder counts COOPER PAIRS, so the island "
            "charge is Q = 2e*n_hat and n01 = 1.3048 is a Cooper-pair matrix "
            "element. Every *_reference block below is quoted in that convention; "
            "every *_corrected / recommended block is quoted in the pinned one."
        ),
        "charge_factor_verdict": P2["model"]["charge_factor_verdict"],
        "coupling_operator": P2["model"]["coupling_operator"],
        "reference_operator": P2["model"]["reference_operator"],
        "topology": P2["model"]["topology"],
        "citation": P2["provenance"]["citation"],
        "beta": {
            "definition": be["definition"],
            "drive": be["drive"],
            "drive_range": be["drive_range"],
            "readout_sample_d_eff_100um": be["readout_sample"],
            "readout_sample_range": be["readout_sample_range"],
            "beta_max": be["beta_max"],
            "beta_breakeven": be["beta_breakeven"],
            "C_c_breakeven_fF": be["C_c_breakeven_fF"],
            "budget_law": (
                "C_Sigma is LOCKED at 12.99971 fF by E_J/E_C = 100, so the "
                "collection electrode SPENDS that budget rather than adding to "
                "it: C_c <= C_Sigma - C_J and beta_max = 1 - C_J/C_Sigma"
            ),
            "C_c_budget_cap_fF": P2["model"]["params"]["C_c_budget_cap_fF"],
            "AT_impact": (
                "partitioning C_Sigma leaves E_C, f01, anharmonicity and eps1 "
                "untouched, so AT-1/AT-2/AT-3 are unaffected"
            ),
        },
        "correction_factors": {
            "formula": cf["formula"],
            "drive": cf["drive"],
            "drive_range": cf["drive_range"],
            "readout_sample_d_eff_100um": cf["readout_sample"],
            "readout_sample_range": cf["readout_sample_range"],
            "ceiling": cf["ceiling"],
            "scaling": cf["scaling"],
            "recommended_readout_point": P5["recommended_point"]["correction_factor"],
        },
        "shared_island_budget": P2["two_cavity_budget"],
        "chi_convention": {
            # src: out/m6/p5_two_cavity.json:chi_convention (which recomputes
            # every number below from qubit/cavity.py) and the adjudication
            # out/verify/chi_convention_ruling.md.
            "status": cc["status"],
            "ruling": cc["ruling"],
            "production_formula": cc["production_formula"],
            "truth_standard": cc["truth_standard"],
            "superseded_formula": cc["superseded_formula"],
            "detuning_rule": (
                "Delta = f01 - f_r (qubit MINUS cavity). The single formula "
                "chi = g^2 alpha/(Delta(Delta+alpha)) is correct ONLY with that "
                "sign: it is the (n12/n01)^2 -> 2 limit of the exact two-pole "
                "form chi = g^2/(f01-f_r) - (r/2) g^2/(f12-f_r), and its only "
                "poles are then the two physical transmon lines f01 and f12."
            ),
            "defect": (
                "REFERENCE-INHERITED. m6_reference.py and the AT-4/AT-5 paths "
                "feed the negated detuning Delta = f_r - f01, which moves a "
                "pole to f01 + |alpha| = 42.229 GHz where the transmon has no "
                "transition. Contract rows are NOT edited; they are retained as "
                "the documented reference-convention path, mirroring the AT-6 "
                "small-frequency precedent."
            ),
            "ladder": cc["ladder"],
            "selfcheck": cc["selfcheck"],
            "impact": cc["impact"],
            "quoting_rule": (
                "any chi, chi/kappa or Stark shift quoted from a *_reference "
                "block is in the inverted-detuning convention and must be "
                "labelled as such; every production number in this export is "
                "Koch (Delta = f01 - f_r) with the exact two-pole value "
                "reported next to it."
            ),
        },
    }


def block_budgets(rq):
    ker = P3["kernel"]                  # src: out/m6/p3_qp.json:kernel
    par = P3["parity"]                  # src: out/m6/p3_qp.json:parity
    thf = P3["thermal_floor"]           # src: out/m6/p3_qp.json:thermal_floor
    p2d_ref = P2["drive_cavity_reference"]
    p2d_cor = P2["drive_cavity_corrected"]
    rec = P5["recommended_point"]       # src: out/m6/p5_two_cavity.json:recommended_point
    ref_rec = P5["reference_convention_scan"]["recommended_point"]
    chi_drive = P5["chi_convention"]["selfcheck"]["drive_145GHz"]
    rows = {(r["nu"], r["A"]): r for r in P4["reference_rows"]}
    apex4, apex3 = rows[(-2.4, 1e-4)], rows[(-2.4, 1e-3)]
    off4, off3 = rows[(-2.5, 1e-4)], rows[(-2.5, 1e-3)]

    c_qs = apex4["quasistatic_us"] / off4["quasistatic_us"]
    c_ram = apex4["T2_ramsey_us"] / off4["T2_ramsey_us"]
    c_echo = apex4["T2_echo_us"] / off4["T2_echo_us"]

    return {
        "T1_stack": {
            "qp_full_kernel": {
                "status": "SUPERSEDES the AT-6 small-frequency ladder",
                "marginality_hbar_w_over_Delta": ker["w_marginality"],
                "T1_us_vs_xqp_T_qp_to_zero": ker["T1_us_vs_xqp"]["T_qp->0"],   # src: p3_qp.json
                "headline_T1_us_at_xqp_1e_6": ker["T1_us_vs_xqp"]["T_qp->0"]["1e-06"],
                "T1_us_vs_xqp_vs_T_qp": {k: v for k, v in ker["T1_us_vs_xqp"].items()
                                         if k != "T_qp->0"},
                "kernel_correction_total": ker["ratio_vs_smallfreq"],
                "kernel_correction_decomposition": ker["ratio_vs_smallfreq_decomposition"],
                "analytic_T0_limit": ker["ratio_vs_smallfreq_analytic_T0"],
                "analytic_T0_form": "sqrt(1 + hbar w / 2 Delta), reproduced by quadrature to 5e-9",
                "matrix_elements": ker["matrix_elements"],
                "T_qp_dependence": ker["T_qp_dependence"],
                "small_freq_reference": {
                    "flag": "AT-6 REFERENCE, RETAINED -- valid only for hbar w_q << Delta",
                    "T1_us_vs_xqp": P3["kernel"]["T1_us_vs_xqp_reference_smallfreq"],
                    "targets_ladder": TARGETS["quasiparticle"]["T1_us"],  # src: qubit_targets.json
                    "marginality_flag": TARGETS["quasiparticle"]["marginality_flag"],
                    "shortening_factor_vs_reference": P3["kernel"]["T1_shortening_factor_vs_reference"],
                    "recomputed_check_qubit_qp": {
                        "T1_us": rq["T1_us"],
                        "marginality_flag": rq["marginality_flag"],
                        "reldev_at_1e_6": reldev(rq["T1_us"]["1e-06"],
                                                 TARGETS["quasiparticle"]["T1_us"]["1e-06"]),
                    },
                },
                "thermal_x_qp_floor": {
                    "formula": thf["formula"],
                    "Delta_over_kB_K": thf["Delta_over_kB_K"],
                    "x_qp_th": thf["x_qp_th"],
                    "parametric_band": thf["parametric_band"],
                    "verdict": thf["verdict"],
                },
                "claim_discipline": "parametric in (x_qp, T_qp); no absolute TBG T1 is claimed",
            },
            "purcell_drive": {
                "corrected_T1_us": p2d_cor["T1_Purcell_us"],           # src: p2_coupling.json
                "corrected_g_MHz": p2d_cor["g_MHz"],
                "corrected_chi_kHz": p2d_cor["chi_kHz"],
                # F(chi-convention): p2_coupling.json's chi_kHz is the P2
                # correction applied to the AT-4 REFERENCE row, so it inherits
                # the inverted detuning.  The corrected values at the same
                # (g, f_cav) are recomputed in p5:chi_convention.selfcheck and
                # carried here as a band; the "no readout through the drive
                # cavity" verdict is unchanged and STRENGTHENED by all of them.
                "corrected_chi_kHz_flag": (
                    "INVERTED-DETUNING CONVENTION -- superseded, retained for "
                    "continuity with out/m6/p2_coupling.json"),
                "chi_kHz_corrected_convention": {
                    "koch_single_formula": 1e3 * chi_drive["chi_koch_MHz"],
                    "two_pole_exact_elements": 1e3 * chi_drive["chi_two_pole_MHz"],
                    "non_RWA_all_levels_ruling": -10.988,
                    "band_kHz": [-17.5, -11.0],
                    "reference_convention_value": 1e3 * chi_drive[
                        "chi_reference_convention_MHz"],
                    "note": (
                        "at the 104.4 GHz drive detuning the single Koch "
                        "formula is a factor 3.6 BELOW the exact two-pole "
                        "value, because the (1 - r/2)/Delta deficit term it "
                        "omits scales as 1/Delta while the formula scales as "
                        "1/Delta^2. Quote the -11.0 to -17.5 kHz band, not the "
                        "-4.87 kHz single-formula number and not the -5.03 kHz "
                        "reference-convention one. src: "
                        "out/verify/chi_convention_ruling.md Sec. 2 and "
                        "out/m6/p5_two_cavity.json:chi_convention.selfcheck."),
                    "chi_over_kappa_band": [
                        abs(chi_drive["chi_two_pole_MHz"]) / p2d_cor["kappa_MHz"],
                        abs(chi_drive["chi_koch_MHz"]) / p2d_cor["kappa_MHz"]],
                    "verdict_unchanged": (
                        "chi/kappa stays below 2e-3 in EVERY construction, so "
                        "'the drive cavity cannot read the qubit out' survives "
                        "the convention correction and is strengthened by it"),
                },
                "corrected_chi_over_kappa": p2d_cor["chi_over_kappa"],
                "correction_factor": p2d_cor["correction_factor"],
                "beta": p2d_cor["beta"],
                "C_c_fF": p2d_cor["C_c_fF"],
                "d_eff_um": p2d_cor["d_eff_um"],
                "kappa_MHz": p2d_cor["kappa_MHz"],
                "Delta_GHz": p2d_cor["Delta_GHz"],
                "verdict": (
                    "harmless: 5291 us of Purcell relaxation through the drive "
                    "cavity is three orders above the qp-limited T1 at x_qp = 1e-6"
                ),
                "reference": {
                    "flag": "AT-4 REFERENCE, RETAINED (g = e*V_zpf*n01/h)",
                    "T1_Purcell_us": p2d_ref["T1_Purcell_us"],
                    "g_MHz": p2d_ref["g_MHz"],
                    "chi_kHz": p2d_ref["chi_kHz"],
                    "chi_over_kappa": p2d_ref["chi_over_kappa"],
                    "targets_row": TARGETS["drive_cavity"],             # src: qubit_targets.json
                },
                "no_readout_conclusion": {
                    "statement": (
                        "the drive cavity cannot read the qubit out -- STRENGTHENED "
                        "8.4x by the P2 correction"
                    ),
                    "chi_over_kappa_reference": p2d_ref["chi_over_kappa"],
                    "chi_over_kappa_corrected": p2d_cor["chi_over_kappa"],
                    "strengthening_factor": p2d_ref["chi_over_kappa"] / p2d_cor["chi_over_kappa"],
                    "formula": "chi_over_kappa_reference / chi_over_kappa_corrected",
                },
            },
            "purcell_readout_at_recommended": {
                "T1_Purcell_us": rec["T1_Purcell_us"],                  # src: p5_two_cavity.json
                "margin_over_30us_floor": rec["margin_T1"],
                "g_MHz": rec["g_MHz"],
                "chi_MHz": rec["chi_MHz"],
                "chi_two_pole_MHz": rec["chi_two_pole_MHz"],
                "chi_over_kappa": rec["chi_over_kappa"],
                "chi_over_kappa_two_pole": rec["chi_over_kappa_two_pole"],
                "kappa_MHz": rec["kappa_MHz"],
                "d_eff_um": rec["d_eff_um"],
                "Delta_r_GHz": rec["Delta_r_GHz"],
                "f_r_GHz": rec["f_r_GHz"],
                "naive_reference_T1_Purcell_us": P2["readout_sample_reference"]["T1_Purcell_us"],
                "p2_corrected_at_100um_T1_Purcell_us": P2["readout_sample_corrected"]["T1_Purcell_us"],
                "note": (
                    "T1_Purcell = Delta^2/(g^2 2 pi kappa) is EVEN in the "
                    "detuning, so this number is untouched by the chi sign "
                    "correction as a formula -- but the operating point it is "
                    "evaluated at moved. The readout cavity now sits BELOW the "
                    "qubit at f_r = %.4f GHz with d_eff = %.1f um, still below "
                    "the C_Sigma oversubscription knee at 86.88 um, and the "
                    "Purcell margin is %.4fx the 30 us floor."
                    % (rec["f_r_GHz"], rec["d_eff_um"], rec["margin_T1"])
                ),
                "reference_convention": {
                    "flag": ("SUPERSEDED -- the f_r = 44.5966 GHz point, whose "
                             "chi/kappa = 1.0092 was an artifact of the "
                             "inverted detuning; its corrected chi/kappa is "
                             "0.424 (Koch) / 0.466 (two-pole) and it FAILS"),
                    "T1_Purcell_us": ref_rec["T1_Purcell_us"],
                    "f_r_GHz": ref_rec["f_r_GHz"],
                    "d_eff_um": ref_rec["d_eff_um"],
                    "kappa_MHz": ref_rec["kappa_MHz"],
                    "chi_MHz": ref_rec["chi_MHz"],
                    "chi_over_kappa": ref_rec["chi_over_kappa"],
                    "eta_readout": ref_rec["eta_readout"],
                    "corrected_values": P5["reference_convention_scan"][
                        "corrected_values_at_this_point"],
                    "src": ("out/m6/p5_two_cavity.json:"
                            "reference_convention_scan.recommended_point"),
                },
            },
            "thermal_floor": {
                "n_th_at_f01": TARGETS["thermal"]["n_th"],              # src: qubit_targets.json:thermal
                "n_th_operating_100mK": TARGETS["thermal"]["n_th"]["0.1"],
                "n_th_Al5GHz_100mK": TARGETS["thermal"]["n_th_Al5GHz_100mK"],
                "advantage_factor": (TARGETS["thermal"]["n_th_Al5GHz_100mK"]
                                     / TARGETS["thermal"]["n_th"]["0.1"]),
                "mechanism": "f01 = 40.6 GHz is 8.1x an Al transmon's 5 GHz, so hf/kT is 8.1x larger",
            },
            "parity_bound": {
                "eps1_Hz": par["eps1_Hz"],                              # src: p3_qp.json:parity
                "parity_splitting_rad_per_s": par["delta_omega_rad_s"],
                "Gamma_parity_s_at_xqp_1e6_Tqp_20mK": par["rate"]["T0.020_x1e-06"]["Gamma_parity_s"],
                "motional_narrowing_ratio": par["rate"]["T0.020_x1e-06"]["motional_narrowing_ratio"],
                "Tphi_parity_s": par["rate"]["T0.020_x1e-06"]["Tphi_parity_s"],
                "Tphi_frozen_parity_upper_bound_s": par["Tphi_parity_frozen_worst_case_s"],
                "Tphi_frozen_over_T1qp_at_1e_6": par["Tphi_frozen_over_T1qp_at_1e6"],
                "bound_formula": "T_phi <= 1/(pi eps1) -- cutoff-independent, holds even with parity frozen and unknown",
                "verdict": par["verdict"],
            },
        },
        "T2_maps": {
            "artifacts": {
                "json": "out/m6/p4_t2_maps.json",
                "npz": P4["npz"],
                "npz_contents": "dense T2*(nu, A) / T2echo(nu, A) maps incl. per-order breakdown and T2_*_exact",
                "grid_nu_points": len(P4["grids"]["nu"]),
                "grid_A_points": len(P4["grids"]["A"]),
                "grid_nu_range": [P4["grids"]["nu"][0], P4["grids"]["nu"][-1]],
                "grid_A_range": [P4["grids"]["A"][0], P4["grids"]["A"][-1]],
            },
            "theory": P4["theory"],
            "A_is_free": P4["params"]["A_units"],
            "literature_note": P4["literature_note"],
            "headline_rows": [
                {"label": "apex, A = 1e-4", "nu": apex4["nu"], "A": apex4["A"],
                 "T2_ramsey_us": apex4["T2_ramsey_us"], "T2_echo_us": apex4["T2_echo_us"],
                 "quasistatic_us": apex4["quasistatic_us"],
                 "ramsey_over_quasistatic": apex4["ramsey_over_quasistatic"],
                 "echo_over_ramsey": apex4["echo_over_ramsey"],
                 "dominant_order": apex4["dominant_order"]},
                {"label": "apex, A = 1e-3", "nu": apex3["nu"], "A": apex3["A"],
                 "T2_ramsey_us": apex3["T2_ramsey_us"], "T2_echo_us": apex3["T2_echo_us"],
                 "quasistatic_us": apex3["quasistatic_us"],
                 "ramsey_over_quasistatic": apex3["ramsey_over_quasistatic"],
                 "echo_over_ramsey": apex3["echo_over_ramsey"],
                 "dominant_order": apex3["dominant_order"]},
                {"label": "nu = -2.5, A = 1e-4", "nu": off4["nu"], "A": off4["A"],
                 "T2_ramsey_us": off4["T2_ramsey_us"], "T2_echo_us": off4["T2_echo_us"],
                 "quasistatic_us": off4["quasistatic_us"],
                 "ramsey_over_quasistatic": off4["ramsey_over_quasistatic"],
                 "echo_over_ramsey": off4["echo_over_ramsey"],
                 "dominant_order": off4["dominant_order"]},
                {"label": "nu = -2.5, A = 1e-3", "nu": off3["nu"], "A": off3["A"],
                 "T2_ramsey_us": off3["T2_ramsey_us"], "T2_echo_us": off3["T2_echo_us"],
                 "quasistatic_us": off3["quasistatic_us"],
                 "ramsey_over_quasistatic": off3["ramsey_over_quasistatic"],
                 "echo_over_ramsey": off3["echo_over_ramsey"],
                 "dominant_order": off3["dominant_order"]},
            ],
            "contrast_at_A_1e_4": {
                "quasistatic": c_qs,
                "ramsey": c_ram,
                "echo": c_echo,
                "formula": "apex T2 / off-apex(nu = -2.5) T2, from budgets.T2_maps.headline_rows",
                "statement": (
                    "the apex/off-apex contrast is 2041x quasistatically, 1292x "
                    "under Ramsey and only 380x under echo, at A = 1e-4. The "
                    "double-sweet-spot design law stands; the '~2000x' figure "
                    "must be quoted as Ramsey-specific AND A-specific."
                ),
                "mechanism": (
                    "echo removes the infrared log off-apex (4.9x gain) but buys "
                    "only 1.44x at the apex, where the second-order channel is "
                    "asymptotically sequence-independent (Szego limit: "
                    "chi_echo/chi_ramsey -> 1.000)"
                ),
                "quoting_rule": "always state the sequence (Ramsey vs echo), A, and the IR cutoff",
            },
            "ir_cutoff_convention": {
                "noise_convention": P4["params"]["noise_convention"],
                "f_ir_Hz": P4["params"]["f_ir_Hz"],
                "w_ir_rad_per_s": P4["params"]["w_ir_rad_per_s"],
                "w_ir_convention": P4["params"]["w_ir_convention"],
                "w_uv_convention": P4["params"]["w_uv_convention"],
                "T2_definition": P4["params"]["T2_definition"],
                "sensitivity": (
                    "apex T2 is the most IR-cutoff-sensitive number in the export: "
                    "3.75x over six decades of f_ir. The acquisition time MUST be "
                    "stated wherever an apex T2 is quoted."
                ),
            },
            "quasistatic_reference": {
                "flag": "AT-3 REFERENCE, RETAINED (qubit/dome.py, unchanged)",
                "Tphi_apex_us": TARGETS["gate_noise"]["Tphi_apex_us"],   # src: qubit_targets.json
                "Tphi_off_us": TARGETS["gate_noise"]["Tphi_off_us"],
                "D1_apex_GHz": TARGETS["gate_noise"]["D1_apex_GHz"],
                "D2_apex_GHz": TARGETS["gate_noise"]["D2_apex_GHz"],
                "D1_off_nu_m2p5_GHz": TARGETS["gate_noise"]["D1_off_nu_m2p5_GHz"],
            },
            "additivity_rule": {
                "prescription": "chi = chi_1 + chi_2 (independent rates) is the reported map",
                "exactness": "exact at the apex (D1 = 0) and far off-apex; conservative in between",
                "max_abs_error_all": P4["additivity_diagnostic"]["max_abs_error_all"],
                "largest_A_with_max_error_below_1pct":
                    P4["additivity_diagnostic"]["largest_A_with_max_error_below_1pct"],
                "worst_point": P4["additivity_diagnostic"]["worst_point"],
                "quoting_rule": "quote the additive map for A <= 1e-3 only; exact maps live in the npz",
            },
            "claim_discipline": P4["claim_discipline"],
        },
    }


def block_readout_window():
    sc = P5["readout_scan_beta"]        # src: out/m6/p5_two_cavity.json:readout_scan_beta
    rec = P5["recommended_point"]
    env = P5["operating_envelope"]
    ref = P5["reference_convention_scan"]
    gw = P5["gap_window_scan"]
    emp = P5["above_f01_emptiness_proof"]
    return {
        "naive_22pt_reference": {
            "flag": "AT-5 REFERENCE, RETAINED -- naive (beta = 1, no kappa floor); NOT a design window",
            "n_points": sc["counts"]["naive_feasible"],
            "criteria": {"chi_over_kappa_min": sc["criteria"]["chi_over_kappa_min"],
                         "T1_Purcell_min_us": sc["criteria"]["T1_Purcell_min_us"],
                         "g_over_Delta_sq_max": sc["criteria"]["g_over_Delta_sq_max"]},
            "sample_point_brief": TARGETS["readout_window"][3],   # src: qubit_targets.json (Delta_r = 7, kappa = 1)
            "rows_pointer": ("out/m6/p5_two_cavity.json:reference_convention_scan"
                             ".naive_rows_for_reference"),
            "chi_convention_flag": (
                "every chi in this reference block uses the inverted detuning "
                "Delta = f_r - f01 (qubit/cavity.py:chi_dispersive). It is a "
                "reproduction reference, not a physics claim -- see "
                "conventions.chi_convention."),
        },
        "reference_convention_scan": {
            # The pre-2026-07-27 design window, preserved in full.  Demoted, not
            # deleted, exactly as budgets.T1_stack.qp_full_kernel.
            # small_freq_reference preserved the AT-6 ladder.
            "flag": ref["flag"],
            "counts": ref["counts"],
            "recommended_point": {
                k: ref["recommended_point"][k] for k in (
                    "Delta_r_GHz", "f_r_GHz", "d_eff_um", "kappa_MHz", "g_MHz",
                    "chi_MHz", "chi_over_kappa", "T1_Purcell_us",
                    "eta_readout", "kappa_intrinsic_MHz", "kappa_ceiling_MHz",
                    "C_c_sum_fF", "a_r_um", "beta_readout", "margin_T1")},
            "corrected_values_at_this_point": ref["corrected_values_at_this_point"],
            "rows_pointer": "out/m6/p5_two_cavity.json:reference_convention_scan.rows",
            "why_retained": (
                "the defect is reference-inherited (m6_reference.py carries "
                "it), so per out/verify/chi_convention_ruling.md Sec. 3(e) the "
                "contract rows and the AT-4/AT-5 tests are left untouched and "
                "the superseded design window is kept quotable next to the "
                "corrected one."),
        },
        "corrected_scan": {
            "pointer": "out/m6/p5_two_cavity.json:readout_scan_beta",
            "n_grid_points": sc["grid"]["n_points"],
            "counts": sc["counts"],
            "criteria": sc["criteria"],
            "chi_mode": sc["chi_mode"],
            "collapse": (
                "22 naive -> %d on the same grid (%d with a 50 um column) under "
                "the REFERENCE convention -> 0 once the measured internal-Q "
                "linewidth floor C2 is enforced. Under the CORRECTED chi the "
                "mandated f_r > f01 grid gives %d / %d / %d, and the window is "
                "empty not just on this grid but ANALYTICALLY, for every "
                "detuning and every d_eff (readout_window.above_f01_empty)."
                % (sc["counts"]["reference_convention_feasible_same_grid"],
                   sc["counts"]["reference_convention_feasible_extended_grid"],
                   sc["counts"]["corrected_feasible_same_grid"],
                   sc["counts"]["corrected_feasible_extended_grid"],
                   sc["counts"]["corrected_feasible_and_achievable"])
            ),
        },
        "above_f01_empty": {
            "verdict": emp["verdict"],
            "sup_kappa_ceiling_MHz": emp["analytic"]["sup_kappa_ceiling_MHz"],
            "kappa_int_at_mandated_Dmin_MHz":
                emp["analytic"]["kappa_int_at_mandated_Dmin_MHz"],
            "statement": emp["analytic"]["statement"],
            "sweep": emp["sweep"],
            "ceiling_table_MHz": emp["ceiling_table_MHz"],
            "src": emp["src"],
        },
        "gap_protection_window": {
            "f_r_GHz": P5["gap_protection_window"]["f_r_GHz"],
            "derivation": P5["gap_protection_window"]["derivation"],
            "meaning": P5["gap_protection_window"]["meaning"],
            "eta_ceiling_max": gw["envelope"]["eta_ceiling_max"],
            "eta_ceiling_max_at_f_r_GHz": gw["envelope"]["eta_ceiling_max_at_f_r_GHz"],
            "feasible_f_r_bands_GHz": gw["envelope"]["feasible_f_r_bands_GHz"],
            "band_split_cause": (
                "the two feasible bands are separated by the 1-2 line at "
                "f12 = 38.9643 GHz: constraint C7, (g12/Delta_12)^2 <= 0.01, "
                "empties the neighbourhood of f12 entirely"),
            "envelope_pointer": "out/m6/p5_two_cavity.json:gap_window_scan.envelope",
        },
        "recommended_point": {
            "Delta_r_GHz": rec["Delta_r_GHz"], "f_r_GHz": rec["f_r_GHz"],
            "Delta_koch_GHz": rec["Delta_koch_GHz"],
            "Delta_12_GHz": rec["Delta_12_GHz"],
            "d_eff_um": rec["d_eff_um"], "kappa_MHz": rec["kappa_MHz"],
            "beta_readout": rec["beta_readout"], "beta_drive": rec["beta_drive"],
            "correction_factor": rec["correction_factor"],
            "g_MHz": rec["g_MHz"], "g12_MHz": rec["g12_MHz"],
            "n12_over_n01": rec["n12_over_n01"],
            "chi_MHz": rec["chi_MHz"],
            "chi_two_pole_MHz": rec["chi_two_pole_MHz"],
            "chi_over_kappa": rec["chi_over_kappa"],
            "chi_over_kappa_two_pole": rec["chi_over_kappa_two_pole"],
            "T1_Purcell_us": rec["T1_Purcell_us"],
            "g_over_Delta_sq": rec["g_over_Delta_sq"],
            "g12_over_Delta12_sq": rec["g12_over_Delta12_sq"],
            "kappa_intrinsic_MHz": rec["kappa_intrinsic_MHz"],
            "kappa_ext_MHz": rec["kappa_ext_MHz"],
            "kappa_ceiling_MHz": rec["kappa_ceiling_MHz"],
            "kappa_max_all_constraints_MHz": rec["kappa_max_all_constraints_MHz"],
            "eta_readout": rec["eta_readout"],
            "C_c_readout_fF": rec["C_c_readout_fF"],
            "C_c_drive_fF": rec["C_c_drive_fF"],
            "C_c_sum_fF": rec["C_c_sum_fF"],
            "C_c_cap_fF": rec["C_c_cap_fF"],
            "oversubscription": rec["oversubscription"],
            "a_r_um": rec["a_r_um"],
            "Q_w_required": rec["Q_w_required"],
            "below_1_2_line": rec["below_1_2_line"],
            "inside_gap_protection_window": rec["inside_gap_protection_window"],
            "margins": {
                "chi_over_kappa": rec["margin_chi_over_kappa"],
                "chi_over_kappa_two_pole": rec["margin_chi_over_kappa_two_pole"],
                "T1_Purcell": rec["margin_T1"],
                "dispersive": rec["margin_dispersive"],
                "dispersive_12": rec["margin_dispersive_12"],
                "target_T1": rec["margin_target_T1"],
                "R_max_symmetric": P5["recommended_point"]["margin_efficiency_front"]["R_max_symmetric_margin"],
                "identity": P5["recommended_point"]["margin_efficiency_front"]["identity"],
            },
            "eta": rec["eta_readout"],
            "eta_note": (
                "readout port efficiency: %.1f%% of the readout photons leave "
                "the port, against %.1f%% at the superseded (inverted-detuning) "
                "point. The efficiency ceiling anywhere in the gap-protection "
                "window is %.4f, at f_r = %.3f GHz."
                % (100.0 * rec["eta_readout"],
                   100.0 * ref["recommended_point"]["eta_readout"],
                   gw["envelope"]["eta_ceiling_max"],
                   gw["envelope"]["eta_ceiling_max_at_f_r_GHz"])
            ),
            "objective": rec["objective"],
            "corridor": rec["corridor_at_this_point"],
            "deviation_vs_exploration_precomputation":
                rec["deviation_vs_exploration_precomputation"],
            "search": {"n_ok": P5["recommended_point_search"]["n_ok"],
                       "n_points": P5["recommended_point_search"]["n_points"],
                       "grid": P5["recommended_point_search"]["grid"],
                       "constraints_applied":
                           P5["recommended_point_search"]["constraints_applied"]},
        },
        "koch_only_optimum": {
            "flag": gw["koch_only_optimum"]["flag"],
            "f_r_GHz": gw["koch_only_optimum"]["point"]["f_r_GHz"],
            "d_eff_um": gw["koch_only_optimum"]["point"]["d_eff_um"],
            "kappa_MHz": gw["koch_only_optimum"]["point"]["kappa_MHz"],
            "eta_readout": gw["koch_only_optimum"]["point"]["eta_readout"],
            "chi_over_kappa": gw["koch_only_optimum"]["point"]["chi_over_kappa"],
            "chi_over_kappa_two_pole":
                gw["koch_only_optimum"]["point"]["chi_over_kappa_two_pole"],
            "eta_cost_of_two_pole_screen":
                gw["koch_only_optimum"]["eta_cost_of_two_pole_screen"],
        },
        "constraints": [
            {"id": c["id"], "name": c["name"], "statement": c["statement"],
             "meaning": c["meaning"],
             "numbers": {k: v for k, v in c.items()
                         if k not in ("id", "name", "statement", "meaning")}}
            for c in P5["operating_envelope"]["constraints"]
        ],
        "envelope_statement": env["envelope_statement"],
        "escape_hatches": env["escape_hatches"],
    }


def block_two_cavity():
    g = P5["geometry"]                  # src: out/m6/p5_two_cavity.json:geometry
    cl = P5["control_line"]
    conc = {c["id"]: c for c in g["concepts"]}
    rec_c = conc[g["recommended_concept"]]
    return {
        "geometry": {
            "a_r_um": g["maxwell_scaling"]["a_r_um"],
            "a_r_mm": g["maxwell_scaling"]["a_r_mm"],
            "a_um_drive": g["maxwell_scaling"]["a_um"],
            "scale_factor_a_r_over_a": g["maxwell_scaling"]["scale_factor_a_r_over_a"],
            "maxwell_relation": g["maxwell_scaling"]["relation"],
            "a_r_at_brief_sample_47p6GHz_um": g["a_r_at_brief_sample_47p6GHz_um"],
            "r_um": g["maxwell_scaling"]["r_um"],
            "t_um": g["maxwell_scaling"]["t_um"],
            "V_mode_readout_um3": g["maxwell_scaling"]["V_mode_readout_um3"],
            "gaps": {
                "drive_lattice_gap_GHz": g["gap_positions"]["drive_lattice_gap_GHz"],
                "readout_lattice_gap_GHz": g["gap_positions"]["readout_lattice_gap_GHz"],
                "f_r_in_readout_lattice": g["gap_positions"]["f_r_in_readout_lattice"],
                "f_r_in_drive_lattice": g["gap_positions"]["f_r_in_drive_lattice"],
                "f_cav_in_readout_lattice": g["gap_positions"]["f_cav_in_readout_lattice"],
                "symmetry_verdict": g["gap_positions"]["symmetry_verdict"],
            },
            "concept": {
                "id": rec_c["id"],
                "name": "nested / concentric heterostructure",
                "verdict": rec_c["verdict"],
                "short": "nested, CONDITIONAL",
                "description": rec_c["description"],
                "reason": rec_c["reason"],
                "drive_patch_span_um": rec_c["drive_patch_span_um"],
                "readout_defect_span_um": rec_c["readout_defect_span_um"],
                "fits_inside_readout_defect": rec_c["fits_inside_readout_defect"],
                "readout_mirror_rows_consumed_per_side": rec_c["readout_mirror_rows_consumed_per_side"],
                "drive_mirror_rows_that_fit": rec_c["drive_mirror_rows_that_fit"],
                "rejected": [{"id": conc[i]["id"], "name": conc[i]["name"],
                              "verdict": conc[i]["verdict"], "reason": conc[i]["reason"]}
                             for i in ("A", "B")],
            },
            "footprint": {
                "chip_footprint_readout_mm": g["length_scales_um"]["chip_footprint_readout_mm"],
                "chip_footprint_drive_mm": g["length_scales_um"]["chip_footprint_drive_mm"],
                "outer_mirror_rows": g["length_scales_um"]["outer_mirror_rows"],
                "chip_rule": g["length_scales_um"]["chip_rule"],
                "row_pitch_readout_um": g["length_scales_um"]["row_pitch_readout"],
                "row_pitch_drive_um": g["length_scales_um"]["row_pitch_drive"],
            },
        },
        "purcell_filter": {
            "ladder_option": {
                "model": g["purcell_filter_N"]["at_recommended"]["model"],
                "sigma": g["purcell_filter_N"]["at_recommended"]["sigma"],
                "b": g["purcell_filter_N"]["at_recommended"]["b"],
                "usable_N_measured": g["purcell_filter_N"]["at_recommended"]["usable_N_measured"],
                "N_naive_brief_grid": [{"kappa_target_MHz": r["kappa_target_MHz"],
                                        "N_naive": r["N_naive"],
                                        "impossible": r["impossible"]}
                                       for r in g["purcell_filter_N"]["brief_grid"]],
                "at_recommended": g["purcell_filter_N"]["at_recommended"],
                "status": g["purcell_filter_N"]["status"],
                "caveat": g["purcell_filter_N"]["caveat"],
            },
            "bandpass_option": {
                "formula": g["purcell_filter_element"]["formula"],
                "Delta_r_GHz": g["purcell_filter_element"]["Delta_r_GHz"],
                "rows": g["purcell_filter_element"]["rows"],
                "status": g["purcell_filter_element"]["status"],
                "caveat": g["purcell_filter_element"]["caveat"],
                "role": "the only clean escape from constraint C3",
            },
        },
        "control_line": {
            "operating_point": cl["operating_point"],
            "formulas": cl["formulas"],
            "rabi_budget": cl["rabi_budget"],
            "readout_tone_budget": cl["readout_tone_budget"],
            "leakage": cl["leakage"],
            "pair_breaking": cl["pair_breaking"],
        },
    }


def block_allocation():
    al = P6["allocation"]               # src: out/m6/p6_gates.json:allocation
    fd, td = al["fixed_drive"], al["tracking_drive"]
    sc = al["spacing_criterion"]
    return {
        "fixed_drive": {
            "verdict": "EXACTLY 1 CHANNEL",
            "scenario": fd["scenario"],
            "constraint_src": fd["constraint_src"],
            "channels": fd["channels"],
            "nu_window": fd["nu_window"],
            "nu_window_halfwidth": fd["nu_window_halfwidth"],
            "window_frac_of_dome_halfwidth": fd["window_frac_of_dome_halfwidth"],
            "f01_span_GHz": fd["f01_span_GHz"],
            "f01_at_apex_GHz": fd["f01_at_apex_GHz"],
            "f01_at_window_edge_GHz": fd["f01_at_window_edge_GHz"],
            "minimum_spacing_at_t_g_8ns_GHz": al["spacing_vs_gate_time"][-1]["spacing_GHz"]["1e-03"],
            "statement": (
                "the f01 span inside the sub-gap window (0.2906 GHz) is smaller "
                "than the minimum measured channel spacing (0.9235 GHz at "
                "t_g = 8 ns), so slowing the gate never buys a second channel"
            ),
            "sweet_spot_cost": fd["sweet_spot_cost"],
            "degeneracy_note": fd["degeneracy_note"],
        },
        "tracking_drive": {
            "scenario": td["scenario"],
            "f01_range_GHz": td["f01_range_GHz"],
            "f01_span_GHz": td["f01_span_GHz"],
            "channels_at_t_g_floor": td["channels"],
            "channel_table": [
                {"t_g_ns": r["t_g_ns"],
                 "bandwidth_1_over_tg_GHz": r["bandwidth_1_over_tg_GHz"],
                 "spacing_GHz": r["spacing_GHz"],
                 "spacing_in_units_of_1_over_tg": r["spacing_in_units_of_1_over_tg"],
                 "channels_fixed_drive": r["channels_fixed_drive"],
                 "channels_tracking_drive": r["channels_tracking_drive"]}
                for r in al["spacing_vs_gate_time"]
            ],
            "hardware_requirement": td["hardware_requirement"],
            "sweet_spot_note": td["sweet_spot_note"],
        },
        "spacing_criterion": {
            "criterion": sc["criterion"],
            "mechanism": sc["mechanism"],
            "t_g_ns": sc["t_g_ns"],
            "spacing_GHz": sc["spacing_GHz"],
            "spacing_in_units_of_1_over_tg": sc["spacing_in_units_of_1_over_tg"],
            "bandwidth_1_over_tg_GHz": sc["bandwidth_1_over_tg_GHz"],
            "tail_slope_loglog_drag": sc["tail_slope_loglog"],
            "note": "spacing is MEASURED (spectator qubit hit by the same calibrated pulse), not set at k/t_g",
        },
        "channel_count_rule": al["channel_count_rule"],
        "fork_statement": (
            "THE SINGLE-VS-MULTI-QUBIT FORK. With the drive plane pinned at "
            "145 GHz the qubit may only be gated over a 0.2906 GHz span of f01, "
            "which is below the minimum measured channel spacing at any gate "
            "time: exactly one qubit. The M6_BRIEF.md Sec. 1 claim of "
            "'multi-qubit frequency allocation via gate' survives ONLY in the "
            "tracking-drive scenario (per-qubit tone held below 2 Delta(nu)/h), "
            "which gives 5/2 channels at 1.011 ns, 12/5 at 2 ns and 19/18 at "
            "8 ns for leakage thresholds 1e-3/1e-4. P3's drive-plane constraint "
            "is therefore not a pair-breaking footnote -- it is the "
            "single-vs-multi-qubit fork for paper 2. A second channel would also "
            "mean leaving the apex, costing 1180x in quasistatic T_phi."
        ),
    }


def block_two_qubit():
    tq = P6["two_qubit_sketch"]         # src: out/m6/p6_gates.json:two_qubit_sketch
    cr = [r for r in tq["cross_resonance"]
          if r["C_qq_fF"] == 0.01 and r["detuning_GHz"] == 1.0][0]
    return {
        "label": tq["label"],
        "parametric_disclaimer": (
            "PARAMETRIC SKETCH ONLY: closed-form rates, no simulation, no "
            "decoherence and no gate fidelity is claimed. C_qq is a free design "
            "parameter, not a measured input."
        ),
        "model": tq["model"],
        "params": tq["params"],
        "exchange": tq["exchange"],
        "dispersive_budget": tq["dispersive_budget"],
        "design_target": {
            "statement": "C_qq must be TENS OF aF, not the 'obvious' 0.1-1 fF",
            "reason": (
                "J = 8 E_C (C_qq/C_Sigma) n01^2 with E_C = 1.49 GHz and "
                "n01 = 1.305 gives J = 156 / 1561 MHz at C_qq = 0.1 / 1 fF, i.e. "
                "J/|anharm| = 0.096 / 0.956 -- already outside the dispersive regime"
            ),
            "C_qq_max_fF_vs_detuning": {k: v["C_qq_max_fF"]
                                        for k, v in tq["dispersive_budget"].items()},
            "budget_note": tq["model"]["budget_note"],
        },
        "representative_cross_resonance": cr,
        "caveats": tq["caveats"],
    }


def block_open_items():
    px = PEXTRA["fixed_Ic"]["0.8"]["deltas_vs_sinusoidal_pct"]   # src: out/m6/pextra_cpr.json:fixed_Ic.0.8
    pxr = PEXTRA["design_criterion_robustness"]                  # src: out/m6/pextra_cpr.json:design_criterion_robustness
    pxp = PEXTRA["downstream"]["parity_bound"]                   # src: out/m6/pextra_cpr.json:downstream.parity_bound
    return {
        "cpr_harmonics": {
            # F14 (out/verify/paper2_trace.md): the flag asked for a sensitivity
            # computation; that computation is DONE and its artifact is on disk,
            # so this status may no longer read "DEFERRED".  What remains open
            # is the separate, larger question of adopting the multi-harmonic
            # potential in the production Hamiltonian -- carried below as
            # `remaining_work`, not as an unquantified flag.
            "status": "RESOLVED (sensitivity computed)",
            "resolved_by": "out/m6/pextra_cpr.json (module qubit/cpr_transmon.py, tests/test_m6_pextra_cpr.py)",
            "resolution_headline": (
                "at tau = 0.8 and fixed E_J the measured Beenakker CPR shifts "
                "the anharmonicity by %.0f%% and eps1 by +%.0f%%; the E_J/E_C "
                "selection window collapses from %d passing ratios to %s and "
                "the design lock at 100 is unchanged; the frozen-parity margin "
                "falls to %.0fx the qp-limited T1 (x_qp = 1e-6, T_qp = 20 mK), "
                "still >5e2."
                % (px["anharm_GHz"], px["eps1_Hz"],
                   len(pxr["passing_ratios_sinusoidal"]),
                   pxr["passing_ratios"], pxp["bound_over_T1qp_fixed_Ic"])
            ),
            "remaining_work": (
                "FUTURE WORK, not a flag: including the CPR harmonics in the "
                "PRODUCTION transmon Hamiltonian (and re-deriving every "
                "sec. 3-8 number at the shifted operating point) is an owner "
                "decision the sensitivity study quantifies but does not make. "
                "Every number in this export is still the sinusoidal-potential "
                "value."
            ),
            "artifact_note": (
                "out/m6/pextra_cpr.json is a downstream P-extra deliverable, "
                "NOT one of the seven sha256-pinned P7 source artifacts above; "
                "the numbers in resolution_headline are read from it at export "
                "time."
            ),
            "statement": (
                "qubit/spectrum.py uses the contract's pure -E_J cos(phi) "
                "potential, while material.json:cpr carries the measured skewed "
                "Beenakker harmonics (tau = 0.8: c1 = 0.9832, c2 = -0.1842, "
                "c3 = 0.0522) -- paper 1's own headline. Higher CPR harmonics "
                "would shift anharmonicity and charge dispersion measurably."
            ),
            "action": "candidate P-extra sensitivity computation for paper 2; NOT acted on silently",
            "src": "paper/decisions.md 2026-07-27 'M6 flag (deferred): non-sinusoidal CPR vs transmon Hamiltonian'",
        },
        "photon_assisted_pair_breaking": {
            "status": "FLAGGED, NOT BOUNDED",
            "statement": P5["control_line"]["pair_breaking"]["PARAMETRIC_CAVEAT"],
            "linear_audit_holds": P5["control_line"]["pair_breaking"]["statement"],
            "src": "out/m6/p5_two_cavity.json:control_line.pair_breaking",
        },
        "higher_order_gap_capture_145GHz": {
            "status": "PARAMETRIC CAVEAT (pessimistic default assumed)",
            "statement": P5["geometry"]["gap_positions"]["f_cav_in_readout_lattice"]["PARAMETRIC_CAVEAT"],
            "a_r_over_lambda": P5["geometry"]["gap_positions"]["f_cav_in_readout_lattice"]["nu_a_over_lambda"],
            "src": "out/m6/p5_two_cavity.json:geometry.gap_positions.f_cav_in_readout_lattice",
        },
        "heterostructure_fdtd_at_a_r": {
            "status": "NOT RUN (out of P5 scope)",
            "statement": (
                "the nested concept overruns by "
                f"{P5['geometry']['concepts'][2]['readout_mirror_rows_consumed_per_side']:.2f}"
                " readout mirror rows per side; either the drive cavity is cut back to ~"
                f"{P5['geometry']['concepts'][2]['drive_mirror_rows_that_fit']:.1f}"
                " mirror rows (Q_i then unverified -- the measured 14611.7 assumes 8 rows "
                "in a 24a cell) or the readout defect is enlarged beyond L3. Both need "
                f"an FDTD run at a_r = {P5['geometry']['maxwell_scaling']['a_r_um']:.1f} um. "
                "Every ladder/filter number in two_cavity is a DESIGN ESTIMATE."
            ),
            "src": "out/m6/p5_two_cavity.json:geometry.concepts[C]",
        },
    }


def block_provenance(digests, head):
    return {
        "source_files": [
            {"path": "qubit_targets.json", "role": "owner-provided reference scan (M6 expansion pack)",
             "date": EXPORT_DATE, "sha256_16": digests["targets"]},
            {"path": "out/m6/p2_coupling.json", "role": "P2 coupling-g convention",
             "module": P2["provenance"]["module"], "date": EXPORT_DATE,
             "sha256_16": digests["p2"]},
            {"path": "out/m6/p3_qp.json", "role": "P3 full quasiparticle kernel",
             "module": P3["provenance"]["module"], "date": EXPORT_DATE,
             "sha256_16": digests["p3"]},
            {"path": "out/m6/p4_t2_maps.json", "role": "P4 1/f filter-function T2 maps",
             "module": "qubit/noise_1f.py", "date": EXPORT_DATE,
             "sha256_16": digests["p4"]},
            {"path": "out/m6/p4_t2_maps.npz", "role": "P4 dense maps (incl. exact linear+quadratic)",
             "date": EXPORT_DATE, "sha256_16": digests["p4npz"]},
            {"path": "out/m6/p5_two_cavity.json", "role": "P5 two-cavity feasibility + control line",
             "module": P5["provenance"]["module"], "date": EXPORT_DATE,
             "sha256_16": digests["p5"]},
            {"path": "out/m6/p6_gates.json", "role": "P6 gate-set sketch + allocation + two-qubit sketch",
             "module": P6["provenance"]["module"], "date": EXPORT_DATE,
             "sha256_16": digests["p6"]},
        ],
        "upstream_paper1_exports": [
            {"path": "material.json", "role": "M1: Delta0_meV, sc_dome, cpr harmonics"},
            {"path": "junction.json", "role": "M3: contract Ic_nA -> E_J, C_fF"},
            {"path": "photonics.json", "role": "M2/M4: geometry, modes, Q_i, ladder fit"},
        ],
        "package_modules": [
            "qubit/inputs.py", "qubit/spectrum.py", "qubit/dome.py", "qubit/cavity.py",
            "qubit/qp.py", "qubit/coupling.py", "qubit/qp_kernel.py", "qubit/noise_1f.py",
            "qubit/two_cavity.py", "qubit/gates.py",
        ],
        "recomputed_here": [
            "qubit.design_point() -> design.recomputed_check",
            "qubit.cavity.drive_cavity_budget() -> AT-4 reference cross-check",
            "qubit.qp.qp_budget() -> budgets.T1_stack.qp_full_kernel.small_freq_reference.recomputed_check_qubit_qp",
            "qubit.dome.sweet_spot_report() -> design.recomputed_check D1/D2",
        ],
        "seed_policy": {
            "deterministic": True,
            "rng_used": False,
            "note": (
                "no stochastic step anywhere in M6: P2 is closed-form lumped-circuit "
                "algebra, P3 adaptive quadrature, P4 filter-function integrals + "
                "batched slogdet, P5 closed-form algebra + deterministic grid sweeps, "
                "P6 fixed-step RK4. P3/P5/P6 log seed = 0 as unused."
            ),
            "logged_seeds": {"P2": None, "P3": 0, "P4": None, "P5": 0, "P6": 0},
        },
        "tests": {
            "files": [{"path": p, "n_tests": n, "covers": c} for p, n, c in TEST_FILES],
            "m6_production_total": M6_PRODUCTION_TESTS,
            "p7_export_total": P7_EXPORT_TESTS,
            "paper1_total": PAPER1_TESTS,
            "full_suite_before_P7": PAPER1_TESTS + M6_PRODUCTION_TESTS,
            "full_suite": PAPER1_TESTS + M6_PRODUCTION_TESTS + P7_EXPORT_TESTS,
            "note": "counts are live `pytest --collect-only` counts",
            "full_suite_scope": (
                "full_suite counts ONLY the tests this export is accountable "
                f"for: paper-1 ({PAPER1_TESTS}) + M6 production "
                f"({M6_PRODUCTION_TESTS}) + the P7 export "
                f"contract ({P7_EXPORT_TESTS}). It is NOT the whole-repository pytest total, "
                "which is larger and keeps growing as other modules land tests "
                "(e.g. tests/test_m6_pextra_cpr.py, added after M6 closed, is "
                "outside this accounting). Cite this number as the M6-owned "
                "total, never as 'the full repository suite'."
            ),
        },
        "commit_basis": head,
        "commit_basis_note": (
            "git HEAD at export time -- INFORMATIONAL ONLY. It records which "
            "commit the export was cut from; it is not a data field and it is "
            "EXCLUDED from the payload-freshness comparison in "
            "tests/test_qubit_export.py::test_export_is_current. Otherwise every "
            "unrelated commit by another module would turn that test red without "
            "any number having changed. Data freshness is guarded instead by the "
            "sha256_16 digests of the source artifacts above."
        ),
        # F8 (out/verify/paper2_trace.md): this used to say "eight ... entries"
        # and then enumerate the eight that existed when P7 closed.  The M6
        # entry list keeps growing (P7 and P-extra landed after), so the count
        # is deliberately dropped rather than re-pinned to a number that goes
        # stale on the next entry.
        "decisions_log": (
            "paper/decisions.md -- the dated 2026-07-27 M6 entries (milestone 1, "
            "eps1 ULP finding, CPR-harmonics flag, P2, P3, P5, P4, P6, and the "
            "later P7 / P-extra entries); this export is not a count of them"),
        "report": "paper/M6_REPORT.md",
        "figures": [
            "figs/fig_m6_p7_fig1_design.png",
            "figs/fig_m6_p7_fig2_dome.png",
            "figs/fig_m6_p7_fig3_budget.png",
            "figs/fig_m6_p7_fig4_architecture.png",
        ],
        "reference_figure": "fig_m6_qubit.png (owner-provided M6 expansion pack)",
        "claim_discipline": CLAIM_DISCIPLINE,
    }


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------
def build_payload():
    rd, rcav, rqp, rsweet = recomputed()

    # AT-4 cross-check: the recomputed reference budget must reproduce the
    # artifact's reference row (this is the guard that qubit/cavity.py is still
    # bit-faithful after P2 pinned the convention elsewhere).
    assert reldev(rcav["g_MHz"], P2["drive_cavity_reference"]["g_MHz"]) < 1e-12, \
        "qubit/cavity.py no longer reproduces the P2 reference row"

    digests = {
        "targets": sha256_16(TARGETS_PATH), "p2": sha256_16(P2_PATH),
        "p3": sha256_16(P3_PATH), "p4": sha256_16(P4_PATH),
        "p4npz": sha256_16(os.path.join(OUT_M6, "p4_t2_maps.npz")),
        "p5": sha256_16(P5_PATH), "p6": sha256_16(P6_PATH),
    }

    payload = {
        "_meta": block_meta(),
        "design": block_design(rd, rsweet),
        "conventions": block_conventions(),
        "budgets": block_budgets(rqp),
        "readout_window": block_readout_window(),
        "two_cavity": block_two_cavity(),
        "allocation": block_allocation(),
        "two_qubit_sketch": block_two_qubit(),
        "open_items": block_open_items(),
        "provenance": block_provenance(digests, git_head()),
    }
    return plain(payload)


def main():
    payload = build_payload()
    text = json.dumps(payload, indent=1, ensure_ascii=True) + "\n"
    with open(QUBIT_JSON, "w", encoding="utf-8") as fh:
        fh.write(text)

    reloaded = load(QUBIT_JSON)
    if not deep_equal(payload, reloaded):
        print("ROUND-TRIP FAIL: reloaded payload differs from in-memory.")
        return 2
    if json.dumps(reloaded, indent=1, ensure_ascii=True) + "\n" != text:
        print("IDEMPOTENCY WARN: re-serialization differs (non-fatal).")

    print("qubit.json written -> %s" % QUBIT_JSON)
    print("  size: %d bytes; top-level keys (%d): %s"
          % (os.path.getsize(QUBIT_JSON), len(payload), ", ".join(payload.keys())))
    print("  design.f01_GHz = %.7f GHz, honest gate floor = %.4f ns"
          % (payload["design"]["f01_GHz"],
             payload["design"]["gates"]["honest_coherent_floor_ns"]))
    print("  T1(qp, x_qp=1e-6) = %.4f us; T2 contrast (echo, A=1e-4) = %.1fx"
          % (payload["budgets"]["T1_stack"]["qp_full_kernel"]["headline_T1_us_at_xqp_1e_6"],
             payload["budgets"]["T2_maps"]["contrast_at_A_1e_4"]["echo"]))
    print("  round-trip: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
