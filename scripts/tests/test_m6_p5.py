"""M6 deliverable P5 -- two-cavity chip feasibility (qubit/two_cavity.py).

Deterministic; no RNG.  Every physical number is read from the JSONs / prior
out/m6 artefacts or recomputed from qubit/, never typed.

Test groups, mirroring the P5 task statement:
  (i)   the scan is internally consistent -- chi always comes from a
        cavity.py formula, T1 from the Purcell definition, g from the P2
        operator;
  (ii)  the shared C_Sigma budget is respected at EVERY point, feasible or not;
  (iii) the recommended point passes all criteria with a Purcell margin
        >= 1.2x -- and the honest-reporting requirement that the coarse grid is
        empty once the measured kappa floor is enforced;
  (iv)  a_r is the exact Maxwell relation;
  (v)   the Rabi formula is dimensionally sane: P_in in W -> Omega_R in Hz,
        scaling as sqrt(P);
  (vi)  CONVENTION PINNING (added 2026-07-27 with the chi-convention repair,
        out/verify/chi_convention_ruling.md): the corrected single formula, the
        two-pole truth standard, the emptiness of the f_r > f01 window, the new
        1-2-line constraint, and the preservation of the superseded scan.

WHY SOME TESTS STILL ASSERT THE OLD NUMBERS.  The chi sign defect is
REFERENCE-INHERITED: `m6_reference.py` and the AT-4/AT-5 contract rows carry
it.  Per the ruling's Sec. 3(e) disposition (mirroring the AT-6 small-frequency
precedent) those rows are NOT edited -- they are retained as the documented
reference-convention path.  Every test below that exercises the reference
convention therefore keeps passing against the `reference_convention_scan`
block of the artifact, which reproduces the superseded numbers bit-for-bit;
every test that exercises a DESIGN decision is re-derived against the corrected
blocks.  A test that asserts a reference number is asserting reproduction, not
physical accuracy, and says so.
"""
import json
import pathlib

import numpy as np
import pytest

import qubit
from qubit import cavity, coupling, two_cavity as tc
from qubit.inputs import (D_EFF_DRIVE_M, E, EJ_GHZ, EPS0, EPS_C, F_CAV_GHZ, H,
                          HBAR, PHOTONICS, Q_I, V_MODE_M3)

ROOT = pathlib.Path(__file__).resolve().parents[1]
EXPORT = ROOT / "out" / "m6" / "p5_two_cavity.json"
RULING = ROOT / "out" / "verify" / "chi_convention_ruling.md"

REL = 1e-12         # algebraic identities: float64 round-trip only
LOOSE = 1e-9        # identities that pass through a JSON text round-trip

# The ruling's Sec. 2 numerics table, quoted to the precision it states.
# src: out/verify/chi_convention_ruling.md Sec. 2.
RULING_READOUT = dict(g_MHz=142.1265211330537, f_r_GHz=44.5966287014581,
                      two_pole_MHz=-1.6090833, koch_MHz=-1.4635662,
                      reference_MHz=-3.4816279, reference_over_koch=2.378866)
RULING_DRIVE = dict(g_MHz=181.77383656589163, f_cav_GHz=145.0,
                    two_pole_MHz=-0.0175154, koch_MHz=-0.0048720,
                    reference_MHz=-0.0050268, reference_over_koch=1.031767)
RULING_TOL = 1e-5   # the ruling prints 7 significant figures


@pytest.fixture(scope="session")
def design():
    return qubit.design_point()


@pytest.fixture(scope="session")
def payload():
    """Regenerate in-process so no test ever trusts a stale file."""
    _, pay = tc.main()
    return pay


@pytest.fixture(scope="session")
def on_disk():
    assert EXPORT.exists(), f"{EXPORT} not written"
    with open(EXPORT, encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="session")
def rows(design):
    return tc.scan_rows(design["n01"], design["f01_GHz"], design["anharm_GHz"])


@pytest.fixture(scope="session")
def ref_rows(design):
    """The superseded reference-convention rows, regenerated in-process."""
    return tc.scan_rows(design["n01"], design["f01_GHz"], design["anharm_GHz"],
                        chi_mode="reference", use_g12=False, use_two_pole=False)


@pytest.fixture(scope="session")
def r_exact(design):
    return tc.ladder_elements(EC_GHz=design["EC_GHz"])[3]


# =============================================================== (i) scan =====
def test_chi_is_exactly_cavity_chi_dispersive_koch(rows, design):
    """chi in every PRODUCTION scan row comes from cavity.chi_dispersive_koch.

    Repointed from `cavity.chi_dispersive` by the sanctioned convention
    correction (out/verify/chi_convention_ruling.md Sec. 4 item 3): the formula
    is unchanged, the detuning fed to it is now f01 - f_r.
    """
    alpha, f01 = design["anharm_GHz"], design["f01_GHz"]
    for r in rows:
        expect = 1e3 * cavity.chi_dispersive_koch(r["g_MHz"] * 1e-3, f01,
                                                  r["f_r_GHz"], alpha)
        assert r["chi_MHz"] == pytest.approx(expect, rel=REL), r


def test_reference_rows_still_come_from_the_unmodified_chi_dispersive(
        ref_rows, design):
    """The reference-convention path is byte-faithful and NOT repointed.

    This is the AT-6 precedent applied to chi: the contract row is retained as
    the documented reference, so it must keep reproducing the old formula with
    the old (inverted) detuning exactly.
    """
    alpha = design["anharm_GHz"]
    for r in ref_rows:
        expect = 1e3 * cavity.chi_dispersive(r["g_MHz"] * 1e-3,
                                             r["Delta_r_GHz"], alpha)
        assert r["chi_MHz"] == pytest.approx(expect, rel=REL), r


def test_two_pole_cross_check_present_in_every_row(rows, design, r_exact):
    """Every production row carries the truth-standard chi next to the Koch one."""
    alpha, f01 = design["anharm_GHz"], design["f01_GHz"]
    f12 = f01 + alpha
    for r in rows:
        expect = 1e3 * cavity.chi_two_pole(r["g_MHz"] * 1e-3, f01, f12,
                                           r["f_r_GHz"], r_exact)
        assert r["chi_two_pole_MHz"] == pytest.approx(expect, rel=REL), r
        assert r["chi_over_kappa_two_pole"] == pytest.approx(
            abs(r["chi_two_pole_MHz"]) / r["kappa_MHz"], rel=REL)


def test_g_is_the_P2_operator(rows, design):
    """g = 2 beta e E_zpf d_eff n01 / h, recomputed independently per row."""
    n01 = design["n01"]
    for r in rows:
        V_r = V_MODE_M3 * (F_CAV_GHZ / r["f_r_GHz"]) ** 3
        expect = 1e3 * coupling.coupling_g(r["f_r_GHz"], V_r,
                                           r["d_eff_um"] * 1e-6, n01,
                                           r["beta_readout"])
        assert r["g_MHz"] == pytest.approx(expect, rel=REL), r


def test_T1_purcell_matches_the_cavity_definition(rows):
    for r in rows:
        expect = 1e6 / cavity.purcell_rate(r["g_MHz"] * 1e-3,
                                           r["Delta_r_GHz"],
                                           r["kappa_MHz"] * 1e-3)
        assert r["T1_Purcell_us"] == pytest.approx(expect, rel=REL), r


def test_chi_over_kappa_and_margins_are_self_consistent(rows):
    for r in rows:
        assert r["chi_over_kappa"] == pytest.approx(
            abs(r["chi_MHz"]) / r["kappa_MHz"], rel=REL)
        assert r["margin_T1"] == pytest.approx(r["T1_Purcell_us"] / 30.0, rel=REL)
        assert r["g_over_Delta_sq"] == pytest.approx(
            (r["g_MHz"] * 1e-3 / r["Delta_r_GHz"]) ** 2, rel=REL)


def test_feasible_flag_is_the_conjunction_of_the_three_criteria(rows):
    for r in rows:
        assert r["feasible"] == (r["pass_chi_over_kappa"]
                                 and r["pass_T1_Purcell"]
                                 and r["pass_dispersive"])
        assert r["feasible_achievable"] == (r["feasible"] and r["kappa_achievable"])


def test_f_r_and_grid_are_what_the_grid_says(rows, design):
    f01 = design["f01_GHz"]
    seen = set()
    for r in rows:
        assert r["f_r_GHz"] == pytest.approx(f01 + r["Delta_r_GHz"], rel=REL)
        seen.add((r["Delta_r_GHz"], round(r["d_eff_um"], 6), r["kappa_MHz"]))
    assert len(seen) == len(rows) == (len(tc.DETUNINGS_GHZ)
                                      * len(tc.D_EFF_READOUT_M)
                                      * len(tc.KAPPA_READOUT_MHZ))


def test_beta_one_charge_one_limit_reproduces_the_naive_scan(design):
    """Continuity: setting the correction factor to 1 must give cavity.py back.

    The P2 operator is g = 2 beta e V n01/h; at 2 beta = 1 it is the reference
    g = e V n01/h, so the corrected scan must reproduce the naive rows exactly.

    Run in ``chi_mode="reference"`` because `cavity.readout_window_scan` is
    itself the AT-5 reference path and carries the inverted detuning; this test
    checks the COUPLING continuity limit, which the chi convention does not
    touch (g, beta and T1_Purcell are convention-even).
    """
    n01, f01 = design["n01"], design["f01_GHz"]
    alpha = design["anharm_GHz"]
    naive = {(r["Delta_r_GHz"], r["d_eff_um"], r["kappa_MHz"]): r
             for r in cavity.readout_window_scan(n01, f01, alpha)}
    # kappa_geo chosen so that beta_r = 0.5 exactly at d_eff = 100 um, with no
    # budget clamp: C_c = 0.5 C_Sigma  ->  kappa_geo = 0.5 C_Sigma/(eps0 eps 100um)
    d = 100e-6
    kg = 0.5 * coupling.C_SIGMA_F / (EPS0 * coupling.EPS_EFF * d)
    got = {(r["Delta_r_GHz"], r["d_eff_um"], r["kappa_MHz"]): r
           for r in tc.scan_rows(n01, f01, alpha, d_effs_m=(d,), kappa_geo=kg,
                                 chi_mode="reference", use_g12=False,
                                 use_two_pole=False)}
    assert got, "no rows produced"
    for key, r in got.items():
        assert r["correction_factor"] == pytest.approx(1.0, rel=REL)
        ref = naive.get(key)
        if ref is None:          # naive scan pre-filters on (g/Delta)^2
            continue
        assert r["g_MHz"] == pytest.approx(ref["g_MHz"], rel=REL)
        assert r["chi_MHz"] == pytest.approx(ref["chi_MHz"], rel=REL)
        assert r["T1_Purcell_us"] == pytest.approx(ref["T1_Purcell_us"], rel=REL)


def test_correction_factor_is_two_beta(rows):
    for r in rows:
        assert r["correction_factor"] == pytest.approx(2 * r["beta_readout"],
                                                       rel=REL)


def test_corrected_feasible_set_is_smaller_than_the_naive_22(payload):
    """The headline of the P2 mandate: beta in the loop shrinks the window."""
    c = payload["readout_scan_beta"]["counts"]
    assert c["naive_feasible"] == 22
    assert c["corrected_feasible_same_grid"] < c["naive_feasible"]
    # and the 50 um column can only add points, never remove them
    assert (c["corrected_feasible_extended_grid"]
            >= c["corrected_feasible_same_grid"])
    # the reference-convention counts are retained next to the corrected ones
    assert c["reference_convention_feasible_same_grid"] == 12
    assert c["reference_convention_feasible_extended_grid"] == 14


def test_no_200um_row_survives(rows):
    """The brief's 200 um readout electrode is killed by the budget + 2 beta."""
    assert not any(r["feasible"] for r in rows if r["d_eff_um"] == 200.0)


# ============================================================ (ii) budget =====
def test_budget_respected_at_every_point_feasible_or_not(rows):
    cap = coupling.c_c_cap() * 1e15
    for r in rows:
        assert r["C_c_sum_fF"] <= cap * (1 + 1e-12), r
        assert r["C_c_sum_fF"] == pytest.approx(
            r["C_c_readout_fF"] + r["C_c_drive_fF"], rel=REL)


def test_budget_respected_at_every_feasible_point_over_the_whole_param_box(design):
    """(ii) again, at both corners of the (kappa_geo, eps_eff) box."""
    cap = coupling.c_c_cap() * 1e15
    n01, f01, alpha = design["n01"], design["f01_GHz"], design["anharm_GHz"]
    for kg in coupling.KAPPA_GEO_RANGE + (coupling.KAPPA_GEO,):
        for ee in coupling.EPS_EFF_RANGE + (coupling.EPS_EFF,):
            for r in tc.scan_rows(n01, f01, alpha, kappa_geo=kg, eps_eff=ee):
                assert r["C_c_sum_fF"] <= cap * (1 + 1e-12), (kg, ee, r)


def test_beta_readout_plus_beta_drive_never_exceeds_beta_max(rows):
    bmax = coupling.beta_max()
    for r in rows:
        assert r["beta_readout"] + r["beta_drive"] <= bmax * (1 + 1e-12), r


def test_saturated_beta_is_the_parameter_free_geometric_ratio(design):
    """Under oversubscription kappa_geo and eps_eff cancel out of beta_r."""
    n01, f01, alpha = design["n01"], design["f01_GHz"], design["anharm_GHz"]
    bmax = coupling.beta_max()
    for kg, ee in ((2.0, 6.34), (4.0, 11.68), (3.0, 8.0)):
        for r in tc.scan_rows(n01, f01, alpha, kappa_geo=kg, eps_eff=ee):
            if r["oversubscription"] > 1.0:
                d = r["d_eff_um"]
                expect = bmax * d / (D_EFF_DRIVE_M * 1e6 + d)
                assert r["beta_readout"] == pytest.approx(expect, rel=1e-12), r


def test_budget_scale_is_the_common_process_rescale(rows):
    for r in rows:
        assert r["budget_scale"] == pytest.approx(
            min(1.0, 1.0 / r["oversubscription"]), rel=REL)


def test_recommended_point_is_below_the_oversubscription_knee(payload):
    """The P5 resolution of the P2 oversubscription finding."""
    r = payload["recommended_point"]
    assert r["oversubscription"] < 1.0
    assert r["budget_scale"] == pytest.approx(1.0, rel=REL)
    assert r["C_c_sum_fF"] < r["C_c_cap_fF"]


# ==================================================== (iii) recommendation ====
def test_recommended_point_passes_all_criteria_with_purcell_margin(payload):
    r = payload["recommended_point"]
    assert r["chi_over_kappa"] >= cavity.CHI_OVER_KAPPA_MIN
    assert r["g_over_Delta_sq"] <= cavity.G_OVER_DELTA_SQ_MAX
    assert r["T1_Purcell_us"] >= cavity.T1_PURCELL_MIN_S * 1e6
    assert r["margin_T1"] >= tc.T1_MARGIN_TARGET, (
        "no >=1.2x Purcell-margin point found; the honest fallback must be "
        "reported explicitly rather than fabricated")
    assert r["margin_T1"] == pytest.approx(
        r["T1_Purcell_us"] / (cavity.T1_PURCELL_MIN_S * 1e6), rel=REL)


def test_recommended_point_is_kappa_achievable(payload):
    """kappa_r >= f_r/Q_i: the P5 linewidth floor."""
    r = payload["recommended_point"]
    assert r["kappa_MHz"] >= r["kappa_intrinsic_MHz"]
    assert r["kappa_intrinsic_MHz"] == pytest.approx(
        r["f_r_GHz"] * 1e3 / Q_I, rel=REL)
    assert r["kappa_ext_MHz"] == pytest.approx(
        r["kappa_MHz"] - r["kappa_intrinsic_MHz"], rel=REL)
    assert 0.0 < r["eta_readout"] < 1.0


def test_coarse_grid_has_no_achievable_point_honest_report(payload, rows):
    """Honest reporting: the mandated {1,3} MHz grid is entirely unreachable."""
    assert payload["readout_scan_beta"]["counts"][
        "corrected_feasible_and_achievable"] == 0
    assert not any(r["feasible_achievable"] for r in rows)
    for r in rows:
        if r["feasible"]:
            assert r["kappa_MHz"] < r["kappa_intrinsic_MHz"], (
                "a feasible coarse-grid row that IS achievable would "
                "contradict the reported count of 0")


def test_recommended_chi_recomputed_from_cavity_module(payload, design):
    r = payload["recommended_point"]
    assert r["chi_from_cavity_chi_dispersive_koch_MHz"] == pytest.approx(
        r["chi_MHz"], rel=LOOSE)
    assert r["chi_from_cavity_chi_two_pole_MHz"] == pytest.approx(
        r["chi_two_pole_MHz"], rel=LOOSE)


def test_kappa_ceiling_identity_is_g_independent(design):
    """(CEIL'): the CORRECTED analytic ceiling really does bound the numerics.

    Rewritten with `cavity.chi_dispersive_koch` and Delta = f01 - f_r; probes
    both sides of the qubit, since the corrected ceiling is a two-sided object.
    """
    alpha, f01 = design["anharm_GHz"], design["f01_GHz"]
    g = np.logspace(-3, 0.5, 400)              # 1 MHz .. 3.2 GHz, in GHz
    for Dr in (4.0, 7.0, 10.0, 16.0, -3.0, -3.5, -5.0, -7.0):
        f_r = f01 + Dr
        kmax = float(tc.kappa_ceiling_MHz(Dr, alpha))
        chi = cavity.chi_dispersive_koch(g, f01, f_r, alpha)
        Dl = f01 - f_r
        for factor, expect_any in ((1.02, False), (0.98, True)):
            kap = kmax * factor * 1e-3
            T1 = 1.0 / ((g / Dl) ** 2 * 2 * np.pi * kap * 1e9)
            hit = np.any((np.abs(chi) / kap >= 1.0)
                         & (T1 >= cavity.T1_PURCELL_MIN_S))
            assert bool(hit) is expect_any, (Dr, factor)


def test_kappa_ceiling_monotonicity_inverted_above_f01(design):
    """CORRECTED: above f01 the ceiling INCREASES with detuning and saturates.

    The superseded (inverted-detuning) form decreased in Delta_r -- the
    "widest at the smallest detuning" reading in the old module Sec. C.  That
    reading is withdrawn; see out/verify/chi_convention_ruling.md Sec. 3(c).
    """
    alpha = design["anharm_GHz"]
    k = tc.kappa_ceiling_MHz(tc.DETUNINGS_GHZ, alpha)
    assert np.all(np.diff(k) > 0), "corrected ceiling must increase with D"
    sup = tc.kappa_ceiling_asymptote_MHz(alpha)
    assert np.all(k < sup)
    assert float(tc.kappa_ceiling_MHz(1e6, alpha)) == pytest.approx(sup, rel=1e-6)
    assert sup == pytest.approx(2.9427609811682283, rel=1e-9)
    # and the corrected value at the mandated minimum detuning
    assert float(tc.kappa_ceiling_MHz(4.0, alpha)) == pytest.approx(2.4799,
                                                                   abs=1e-4)


def test_below_f01_ceiling_is_the_superseded_formula(design):
    """Side swap: the OLD (CEIL) is the correct ceiling BELOW f01, past f12."""
    alpha = design["anharm_GHz"]
    a = abs(alpha)
    for D in (2.0, 3.5, 6.0):          # f_r = f01 - D, D > |alpha|
        old_form = float(np.sqrt(a * 1e9 * D * 1e9
                                 / ((D * 1e9 - a * 1e9) * 2 * np.pi
                                    * tc.T1_PURCELL_MIN_S)) / 1e6)
        assert float(tc.kappa_ceiling_MHz(-D, alpha)) == pytest.approx(
            old_form, rel=REL)


def test_margin_efficiency_front_identity(payload, design):
    """m_chi * m_T1 = (kappa_ceiling/kappa)^2, exactly, at the recommended pt."""
    r = payload["recommended_point"]
    fr = r["margin_efficiency_front"]
    lhs = r["margin_chi_over_kappa"] * r["margin_T1"]
    rhs = (fr["kappa_ceiling_MHz"] / r["kappa_MHz"]) ** 2
    assert lhs == pytest.approx(rhs, rel=1e-10)
    assert fr["R_max_symmetric_margin"] == pytest.approx(
        fr["kappa_ceiling_MHz"] / fr["kappa_intrinsic_MHz"], rel=REL)
    # eta falls linearly in the symmetric margin and vanishes at m = R
    for row in fr["front"]:
        assert row["eta_readout"] == pytest.approx(
            1.0 - row["symmetric_margin"] / fr["R_max_symmetric_margin"],
            rel=1e-10)


def test_d_eff_corridor_brackets_the_recommended_electrode(payload, design):
    r = payload["recommended_point"]
    corr = r["corridor_at_this_point"]
    assert corr["d_lo_um"] <= r["d_eff_um"] <= corr["d_hi_um"]
    assert corr["d_hi_um"] < corr["oversubscription_knee_um"], (
        "the corridor must sit below the C_Sigma knee for the closed form "
        "(g ~ d_eff^2) to be the right inversion")


# ================================================== (iv) Maxwell scaling ======
def test_a_r_is_the_exact_maxwell_relation(payload):
    g = payload["geometry"]["maxwell_scaling"]
    a = PHOTONICS["geometry"]["a_um"]
    f_r = g["f_r_GHz"]
    assert g["a_um"] == a
    assert g["a_r_um"] == pytest.approx(a * F_CAV_GHZ / f_r, rel=REL)
    assert g["scale_factor_a_r_over_a"] == pytest.approx(F_CAV_GHZ / f_r, rel=REL)
    # nu = a/lambda invariant: a_r * f_r / c == a * f_cav / c
    c = 299792458.0
    assert (g["a_r_um"] * 1e-6 * f_r * 1e9 / c
            == pytest.approx(a * 1e-6 * F_CAV_GHZ * 1e9 / c, rel=REL))


def test_a_r_at_the_brief_sample_is_about_1p8mm(payload):
    """The brief predicts a_r ~ 1.8 mm at the 47.6 GHz sample point."""
    a_r = payload["geometry"]["a_r_at_brief_sample_47p6GHz_um"]
    assert a_r == pytest.approx(
        PHOTONICS["geometry"]["a_um"] * F_CAV_GHZ / 47.5966287014581, rel=REL)
    assert 1.75e3 < a_r < 1.85e3


def test_mode_volume_scaling_agrees_with_cavity_module(payload):
    """V_r = V_mode (f_cav/f_r)^3 (cavity.py) == V_mode_a3 * a_r^3 (geometry)."""
    g = payload["geometry"]["maxwell_scaling"]
    f_r = g["f_r_GHz"]
    from_cavity = V_MODE_M3 * (F_CAV_GHZ / f_r) ** 3 * 1e18   # m^3 -> um^3
    # V_MODE_M3 is the 3-sig-fig targets value; V_mode_a3*a^3 is the photonics
    # one, 0.011% apart (qubit/inputs.py note), so compare loosely.
    assert g["V_mode_readout_um3"] == pytest.approx(from_cavity, rel=2e-4)


def test_gap_edges_scale_by_a_over_a_r(payload):
    gp = payload["geometry"]["gap_positions"]
    g = payload["geometry"]["maxwell_scaling"]
    scale = g["a_um"] / g["a_r_um"]
    drive = PHOTONICS["modes"]["cavity_line_gap_GHz"]
    for got, ref in zip(gp["readout_lattice_gap_GHz"], drive):
        assert got == pytest.approx(ref * scale, rel=REL)


def test_readout_line_sits_in_its_own_gap_and_below_the_drive_gap(payload):
    gp = payload["geometry"]["gap_positions"]
    assert gp["f_r_in_readout_lattice"]["in_gap"] is True
    assert gp["f_r_in_drive_lattice"]["in_gap"] is False
    assert gp["f_r_in_drive_lattice"]["below_lower_edge_GHz"] > 0
    # deeply subwavelength there: a/lambda well under the gap nu
    assert (gp["f_r_in_drive_lattice"]["nu_a_over_lambda"]
            < 0.5 * PHOTONICS["modes"]["nu0_a_over_lambda_precise"])


def test_drive_line_is_above_the_readout_lattice_fundamental_gap(payload):
    gp = payload["geometry"]["gap_positions"]
    d = gp["f_cav_in_readout_lattice"]
    assert d["in_fundamental_gap"] is False
    assert d["above_upper_edge_GHz"] > 0
    assert "PARAMETRIC_CAVEAT" in d
    assert "not computed" in d["PARAMETRIC_CAVEAT"].lower()


def test_air_fill_fraction_and_homogenised_eps(payload):
    f = tc.air_fill_fraction(0.3)
    assert f == pytest.approx(2 * np.pi * 0.09 / np.sqrt(3.0), rel=REL)
    eps = payload["geometry"]["gap_positions"]["f_r_in_drive_lattice"][
        "homogenised_eps_volume_average"]
    assert eps == pytest.approx(f + (1 - f) * EPS_C, rel=REL)
    assert 1.0 < eps < EPS_C


def test_purcell_filter_N_inverts_the_measured_ladder_fit(payload):
    fit = PHOTONICS["ladder"]["exponential_fit"]
    for entry in (payload["geometry"]["purcell_filter_N"]["brief_grid"]
                  + [payload["geometry"]["purcell_filter_N"]["at_recommended"]]):
        Q = entry["Q_total_naive"]
        assert Q == pytest.approx(entry["f_r_GHz"] * 1e3
                                  / entry["kappa_target_MHz"], rel=REL)
        # round trip through the fit
        assert np.exp(fit["sigma"] * entry["N_naive"] + fit["b"]) == \
            pytest.approx(Q, rel=1e-10)


def test_purcell_filter_N_flags_the_unreachable_brief_targets(payload):
    """kappa = 1 and 3 MHz are below the intrinsic floor -> no N reaches them."""
    for entry in payload["geometry"]["purcell_filter_N"]["brief_grid"]:
        assert entry["N_with_internal"] is None
        assert "impossible" in entry
    rec = payload["geometry"]["purcell_filter_N"]["at_recommended"]
    assert rec["N_with_internal"] is not None
    assert np.exp(PHOTONICS["ladder"]["exponential_fit"]["sigma"]
                  * rec["N_with_internal"]
                  + PHOTONICS["ladder"]["exponential_fit"]["b"]) == \
        pytest.approx(rec["Q_w_required"], rel=1e-10)


def test_geometry_concept_selection(payload):
    g = payload["geometry"]
    ids = {c["id"]: c for c in g["concepts"]}
    assert set(ids) == {"A", "B", "C"}
    assert ids["A"]["verdict"] == "REJECTED"
    assert ids["B"]["verdict"] == "REJECTED"
    assert ids["C"]["verdict"].startswith("VIABLE")
    assert g["recommended_concept"] == "C"
    # the rejection of A is quantitative, not rhetorical
    assert ids["A"]["cavity_centre_separation_min_mm"] > 1.0


# ================================================ (v) Rabi / control line =====
def test_rabi_scales_as_sqrt_power_and_returns_Hz(payload):
    r = payload["recommended_point"]
    kw = dict(g_MHz=r["g_MHz"], f_drive_GHz=payload["params"]["f01_GHz"],
              f_r_GHz=r["f_r_GHz"], kappa_MHz=r["kappa_MHz"],
              kappa_ext_MHz=r["kappa_ext_MHz"])
    P0 = 1e-12                                  # 1 pW
    om0 = tc.rabi_from_power(P0, **kw)
    assert 1e3 < om0 < 1e12, "Omega_R/2pi should be a sane Hz-scale number"
    for factor in (4.0, 100.0, 1e-4):
        om = tc.rabi_from_power(P0 * factor, **kw)
        assert om == pytest.approx(om0 * np.sqrt(factor), rel=1e-12)


def test_power_for_rabi_is_the_exact_inverse(payload):
    r = payload["recommended_point"]
    kw = dict(g_MHz=r["g_MHz"], f_drive_GHz=payload["params"]["f01_GHz"],
              f_r_GHz=r["f_r_GHz"], kappa_MHz=r["kappa_MHz"],
              kappa_ext_MHz=r["kappa_ext_MHz"])
    for target in (1e6, 1e7, 5e7):
        P, n = tc.power_for_rabi(target, **kw)
        assert tc.rabi_from_power(P, **kw) == pytest.approx(target, rel=1e-12)
        assert tc.n_drive_from_power(P, kw["f_drive_GHz"], kw["f_r_GHz"],
                                     kw["kappa_MHz"], kw["kappa_ext_MHz"]) == \
            pytest.approx(n, rel=1e-12)


def test_n_drive_formula_from_first_principles(payload):
    """n = (P/(hbar w_d)) k_ext / (D^2 + (k/2)^2), all rates in rad/s."""
    r = payload["recommended_point"]
    f01 = payload["params"]["f01_GHz"]
    P = 3.7e-12
    w_d = 2 * np.pi * f01 * 1e9
    D = 2 * np.pi * (f01 - r["f_r_GHz"]) * 1e9
    kap = 2 * np.pi * r["kappa_MHz"] * 1e6
    kext = 2 * np.pi * r["kappa_ext_MHz"] * 1e6
    expect = (P / (HBAR * w_d)) * kext / (D ** 2 + (kap / 2) ** 2)
    assert tc.n_drive_from_power(P, f01, r["f_r_GHz"], r["kappa_MHz"],
                                 r["kappa_ext_MHz"]) == \
        pytest.approx(expect, rel=REL)
    # linear in power, and the detuning is the qubit-readout one
    assert D / (2 * np.pi * 1e9) == pytest.approx(-r["Delta_r_GHz"], rel=REL)


def test_rabi_budget_rows_reproduce_their_own_powers(payload):
    r = payload["recommended_point"]
    cl = payload["control_line"]
    f01 = payload["params"]["f01_GHz"]
    for row in cl["rabi_budget"]:
        got = tc.rabi_from_power(row["P_in_W"], r["g_MHz"], f01, r["f_r_GHz"],
                                 r["kappa_MHz"], r["kappa_ext_MHz"])
        assert got == pytest.approx(row["Omega_R_over_2pi_MHz"] * 1e6, rel=LOOSE)
        assert row["P_in_dBm"] == pytest.approx(
            10 * np.log10(row["P_in_W"] / 1e-3), rel=LOOSE)
        assert row["P_in_pW"] == pytest.approx(row["P_in_W"] * 1e12, rel=LOOSE)
    # 5x the Rabi rate costs 25x the power
    a, b = cl["rabi_budget"][0], cl["rabi_budget"][1]
    ratio = b["Omega_R_over_2pi_MHz"] / a["Omega_R_over_2pi_MHz"]
    assert b["P_in_W"] / a["P_in_W"] == pytest.approx(ratio ** 2, rel=1e-9)


def test_rabi_stays_dispersive_and_below_the_anharmonicity(payload):
    """Sanity: the quoted Rabi rates are small vs |alpha| and n_drive << 1."""
    alpha = abs(payload["params"]["anharm_GHz"]) * 1e3      # MHz
    for row in payload["control_line"]["rabi_budget"]:
        assert row["Omega_R_over_2pi_MHz"] < alpha
        assert row["n_drive"] < 1.0


def test_leakage_lorentzian_is_the_measured_drive_cavity(payload):
    lk = payload["control_line"]["leakage"]["control_tone_into_drive_cavity"]
    f01 = payload["params"]["f01_GHz"]
    kappa_drive_MHz = F_CAV_GHZ / Q_I * 1e3
    assert lk["kappa_drive_MHz"] == pytest.approx(kappa_drive_MHz, rel=REL)
    D = 2 * np.pi * (f01 - F_CAV_GHZ) * 1e9
    k2 = (2 * np.pi * kappa_drive_MHz * 1e6 / 2) ** 2
    assert lk["suppression"] == pytest.approx(k2 / (D ** 2 + k2), rel=REL)
    assert lk["suppression_dB"] == pytest.approx(
        10 * np.log10(lk["suppression"]), rel=LOOSE)
    assert lk["suppression_dB"] < -60.0


def test_all_cross_mode_leakages_are_strongly_suppressed(payload):
    lk = payload["control_line"]["leakage"]
    for key in ("control_tone_into_drive_cavity",
                "readout_tone_into_drive_cavity",
                "drive_tone_into_readout_cavity"):
        assert lk[key]["suppression_dB"] < -60.0
        assert 0.0 < lk[key]["suppression"] < 1.0


def test_pair_breaking_restatement_matches_p3_exactly(payload):
    p3 = json.loads((ROOT / "out" / "m6" / "p3_qp.json").read_text("utf-8"))
    pb = payload["control_line"]["pair_breaking"]
    ref = p3["pair_breaking_audit"]["readout_sample"]
    assert pb["readout_sample_47p6GHz"]["margin_2Delta_over_hf"] == \
        pytest.approx(ref["margin_2Delta_over_hf"], rel=REL)
    assert pb["readout_sample_47p6GHz"]["window_halfwidth_nu"] == \
        pytest.approx(ref["window_halfwidth_nu"], rel=REL)
    assert pb["two_Delta_over_h_GHz"] == pytest.approx(
        p3["pair_breaking_audit"]["two_Delta_over_h_GHz"], rel=REL)
    assert ref["window_halfwidth_nu"] > 0.41
    assert "PARAMETRIC_CAVEAT" in pb


# ==================================================== export / provenance =====
def test_export_has_the_mandated_sections(on_disk):
    for key in ("readout_scan_beta", "recommended_point", "geometry",
                "control_line", "operating_envelope", "params", "provenance"):
        assert key in on_disk, key
    for key in ("a_r_um",):
        assert key in on_disk["geometry"]["maxwell_scaling"]
    assert "gap_positions" in on_disk["geometry"]
    assert "purcell_filter_N" in on_disk["geometry"]
    assert on_disk["provenance"]["deterministic"] is True
    assert on_disk["provenance"]["rng_seed"] == tc.SEED


def test_export_is_byte_stable_on_regeneration(payload, on_disk):
    assert json.dumps(payload, sort_keys=True) == \
        json.dumps(on_disk, sort_keys=True)


def test_no_existing_qubit_module_was_modified(payload):
    """P5 composes; it never rewrites the validated numerics.

    cavity.py moved from `unmodified` to `modified_this_revision` because the
    chi-convention repair ADDED functions to it.  The guarantee that matters is
    unchanged and is asserted directly below: the reference-convention path
    (chi_dispersive and its two call sites, the AT-4/AT-5 numerics) still
    produces exactly what it did.
    """
    assert set(payload["provenance"]["unmodified"]) >= {
        "qubit/inputs.py", "qubit/coupling.py", "qubit/spectrum.py"}
    assert "qubit/cavity.py" in payload["provenance"]["modified_this_revision"]
    assert "ADDITIVE ONLY" in \
        payload["provenance"]["modified_this_revision"]["qubit/cavity.py"]
    # cavity.py's own grid must still be the brief's (no 50 um column there)
    assert cavity.D_EFF_READOUT_M == (100e-6, 200e-6)
    assert tuple(cavity.KAPPA_READOUT_MHZ) == (1.0, 3.0)


def test_cavity_reference_path_is_still_bit_faithful(design):
    """AT-4/AT-5 numerics are untouched by the additive cavity.py change."""
    n01, f01 = design["n01"], design["f01_GHz"]
    alpha = design["anharm_GHz"]
    p2 = json.loads((ROOT / "out" / "m6" / "p2_coupling.json").read_text("utf-8"))
    cav = cavity.drive_cavity_budget(n01, f01, alpha)
    for k in ("g_MHz", "chi_kHz", "chi_over_kappa", "T1_Purcell_us",
              "E_zpf_V_per_m", "kappa_MHz"):
        assert cav[k] == pytest.approx(p2["drive_cavity_reference"][k], rel=REL)
    assert len(cavity.readout_window_scan(n01, f01, alpha)) == 22


def test_operating_envelope_lists_every_binding_constraint(on_disk):
    env = on_disk["operating_envelope"]
    ids = {c["id"] for c in env["constraints"]}
    # C7 (1-2-line dispersivity) and C8 (two-pole robustness screen) added by
    # the sanctioned chi-convention correction.
    assert ids == {"C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"}
    assert "envelope_statement" in env and len(env["envelope_statement"]) > 200
    c5 = next(c for c in env["constraints"] if c["id"] == "C5")
    p3 = json.loads((ROOT / "out" / "m6" / "p3_qp.json").read_text("utf-8"))
    assert c5["window_halfwidth_nu"] == pytest.approx(
        p3["pair_breaking_audit"]["drive_cavity"]["window_halfwidth_nu"],
        rel=LOOSE)
    c1 = next(c for c in env["constraints"] if c["id"] == "C1")
    assert c1["cap_fF"] == pytest.approx(coupling.c_c_cap() * 1e15, rel=LOOSE)


def test_free_parameters_all_carry_ranges(on_disk):
    fp = on_disk["params"]["free_parameters"]
    for k, v in fp.items():
        assert "-" in v or "us" in v or "x" in v, (k, v)
    assert on_disk["params"]["kappa_geo_range"] == [1.0, 4.0]
    assert on_disk["params"]["eps_eff_range"][0] < \
        on_disk["params"]["eps_eff_range"][1]


def test_run_is_deterministic(design):
    a = tc.scan_rows(design["n01"], design["f01_GHz"], design["anharm_GHz"])
    b = tc.scan_rows(design["n01"], design["f01_GHz"], design["anharm_GHz"])
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    p1, _ = tc.fine_scan(design["n01"], design["f01_GHz"], design["anharm_GHz"])
    p2, _ = tc.fine_scan(design["n01"], design["f01_GHz"], design["anharm_GHz"])
    assert json.dumps(p1, sort_keys=True) == json.dumps(p2, sort_keys=True)


# ====================================== (vi) chi-convention pinning tests =====
# Added 2026-07-27 with the sanctioned convention correction.
# src: out/verify/chi_convention_ruling.md
def test_the_ruling_file_exists():
    """The tests below pin numbers to an adjudication document; it must be there."""
    assert RULING.exists(), RULING
    assert cavity.CHI_CONVENTION_RULING == "out/verify/chi_convention_ruling.md"


def test_koch_equals_two_pole_in_the_harmonic_limit(design):
    """(i) chi_two_pole(r -> 2) == chi_dispersive_koch, to 1e-12.

    This identity is the whole argument for Delta = f01 - f_r: the exact
    two-pole form collapses onto the single formula ONLY with that sign.
    """
    f01, alpha = design["f01_GHz"], design["anharm_GHz"]
    f12 = f01 + alpha
    g = np.array([0.01, 0.0289904627489091, 0.1421265211330537, 0.5])
    for f_r in (15.0, 20.0, 33.5, 37.43887668190708, 39.3, 41.5,
                44.5966287014581, 60.0, 145.0):
        a = cavity.chi_two_pole(g, f01, f12, f_r, 2.0)
        b = cavity.chi_dispersive_koch(g, f01, f_r, alpha)
        assert np.allclose(a, b, rtol=1e-12, atol=0.0), f_r
    # and NOT equal to the reference-convention value anywhere off the poles
    ref = cavity.chi_dispersive(g[2], 4.0, alpha)
    koch = cavity.chi_dispersive_koch(g[2], f01, f01 + 4.0, alpha)
    assert abs(ref / koch - 1.0) > 1.0


def test_chi_two_pole_matches_the_ruling_at_both_operating_points(design, r_exact):
    """(ii) the ruling's Sec. 2 table, reproduced from qubit/cavity.py."""
    f01, alpha = design["f01_GHz"], design["anharm_GHz"]
    f12 = f01 + alpha
    for row, f_c in ((RULING_READOUT, RULING_READOUT["f_r_GHz"]),
                     (RULING_DRIVE, RULING_DRIVE["f_cav_GHz"])):
        g = row["g_MHz"] * 1e-3
        two = 1e3 * cavity.chi_two_pole(g, f01, f12, f_c, r_exact)
        koch = 1e3 * cavity.chi_dispersive_koch(g, f01, f_c, alpha)
        ref = 1e3 * cavity.chi_dispersive(g, f_c - f01, alpha)
        assert two == pytest.approx(row["two_pole_MHz"], rel=RULING_TOL)
        assert koch == pytest.approx(row["koch_MHz"], rel=RULING_TOL)
        assert ref == pytest.approx(row["reference_MHz"], rel=RULING_TOL)
        assert ref / koch == pytest.approx(row["reference_over_koch"],
                                           rel=RULING_TOL)


def test_ladder_ratio_is_recomputed_and_matches_p6(design, r_exact):
    """r = (n12/n01)^2 comes from spectrum.py, and agrees with p6_gates.json."""
    p6 = json.loads((ROOT / "out" / "m6" / "p6_gates.json").read_text("utf-8"))
    n01, n12, ratio, r = tc.ladder_elements(EC_GHz=design["EC_GHz"])
    assert n01 == pytest.approx(design["n01"], rel=REL)
    assert ratio == pytest.approx(p6["params"]["ladder_ratio"], rel=REL)
    assert n12 == pytest.approx(p6["params"]["n12"], rel=REL)
    assert r == pytest.approx(ratio ** 2, rel=REL) == r_exact
    assert 1.9 < r < 2.0, "exact ratio must sit just below the harmonic 2"
    assert r == pytest.approx(1.918851037132424, rel=1e-12)


def test_pole_locations_are_the_two_transmon_lines(design, r_exact):
    """The correctness invariant: chi is singular ONLY at f01 and f12."""
    f01, alpha = design["f01_GHz"], design["anharm_GHz"]
    f12 = f01 + alpha
    g = 0.1
    for f_pole in (f01, f12):
        near = f_pole + 1e-6
        far = f_pole + 1.0
        for fn in (cavity.chi_dispersive_koch,):
            assert abs(fn(g, f01, near, alpha)) > 1e4 * abs(fn(g, f01, far, alpha))
        assert abs(cavity.chi_two_pole(g, f01, f12, near, r_exact)) > \
            1e4 * abs(cavity.chi_two_pole(g, f01, f12, far, r_exact))
    # the reference convention instead blows up at f01 + |alpha| = 42.229 GHz,
    # where the transmon has NO transition -- and is finite at the physical
    # 1-2 line
    spurious = f01 - alpha
    assert spurious == pytest.approx(42.228970498210096, rel=1e-9)
    near = abs(cavity.chi_dispersive(g, spurious + 1e-6 - f01, alpha))
    far = abs(cavity.chi_dispersive(g, spurious + 1.0 - f01, alpha))
    assert near > 1e4 * far
    assert np.isfinite(cavity.chi_dispersive(g, f12 - f01, alpha))
    assert abs(cavity.chi_dispersive(g, f12 + 1e-6 - f01, alpha)) < 1.0


def test_selfcheck_block_reproduces_the_ruling(on_disk):
    sc = on_disk["chi_convention"]["selfcheck"]
    for key in ("readout_44p5966GHz", "drive_145GHz"):
        assert sc[key]["max_reldev_vs_ruling"] < RULING_TOL, key
    assert sc["harmonic_limit_identity"]["max_abs_rel_difference"] < 1e-12
    assert sc["r_exact"] == pytest.approx(1.918851037132424, rel=1e-12)
    assert sc["one_minus_r_over_2"] == pytest.approx(0.040574481433788,
                                                     rel=1e-9)
    # the ruling's headline residuals
    assert sc["readout_44p5966GHz"]["koch_residual_vs_two_pole"] == \
        pytest.approx(-0.0904, abs=5e-4)
    assert sc["readout_44p5966GHz"]["reference_residual_vs_two_pole"] == \
        pytest.approx(1.164, abs=5e-3)


def test_f_r_above_f01_window_is_empty_under_the_corrected_chi(design, on_disk):
    """(iii) sup(kappa_ceiling) < kappa_int everywhere above the qubit."""
    alpha, f01 = design["anharm_GHz"], design["f01_GHz"]
    sup = tc.kappa_ceiling_asymptote_MHz(alpha)
    kint_min = tc.kappa_intrinsic_MHz(f01 + tc.DETUNINGS_GHZ[0], Q_I)
    assert sup < kint_min, (sup, kint_min)

    proof = on_disk["above_f01_emptiness_proof"]
    assert proof["analytic"]["sup_below_min_kappa_int"] is True
    assert proof["analytic"]["sup_kappa_ceiling_MHz"] == pytest.approx(sup,
                                                                       rel=REL)
    for key in ("koch_mandated_window", "koch_all_D", "two_pole_mandated_window"):
        s = proof["sweep"][key]
        assert s["empty"] is True, key
        assert s["max_ceiling_minus_kappa_int_MHz"] < 0.0, key
    # the ruling's own two headline sweep numbers
    assert proof["sweep"]["koch_all_D"]["max_ceiling_minus_kappa_int_MHz"] == \
        pytest.approx(-0.5647, abs=1e-3)
    assert proof["sweep"]["two_pole_mandated_window"][
        "max_ceiling_minus_kappa_int_MHz"] == pytest.approx(-0.386, abs=2e-3)
    # and the numerical scan agrees: no achievable row above f01, on any grid
    assert on_disk["readout_scan_beta"]["counts"][
        "corrected_feasible_and_achievable"] == 0
    for row in on_disk["gap_window_scan"]["rows"]:
        if row["f_r_GHz"] > f01:
            assert not row["feasible_achievable"], row


def test_recommended_point_satisfies_every_constraint_including_g12(payload,
                                                                    design):
    """(iv) all of C1-C8 at the recommended point, recomputed from scratch."""
    r = payload["recommended_point"]
    f01, alpha = design["f01_GHz"], design["anharm_GHz"]
    f12 = f01 + alpha
    ratio = r["n12_over_n01"]
    g = r["g_MHz"] * 1e-3

    # C1 shared C_Sigma budget
    assert r["C_c_sum_fF"] <= r["C_c_cap_fF"] * (1 + 1e-12)
    assert r["oversubscription"] < 1.0
    # C2 measured internal-Q linewidth floor
    assert r["kappa_MHz"] >= r["kappa_intrinsic_MHz"]
    assert r["kappa_intrinsic_MHz"] == pytest.approx(r["f_r_GHz"] * 1e3 / Q_I,
                                                     rel=REL)
    # C3 analytic ceiling, corrected form
    assert r["kappa_MHz"] <= r["kappa_ceiling_MHz"]
    assert r["kappa_ceiling_MHz"] == pytest.approx(
        float(tc.kappa_ceiling_MHz(r["Delta_r_GHz"], alpha)), rel=LOOSE)
    # chi/kappa >= 1 under the production formula, recomputed
    chi = cavity.chi_dispersive_koch(g, f01, r["f_r_GHz"], alpha)
    assert abs(chi) / (r["kappa_MHz"] * 1e-3) >= 1.0
    assert abs(chi) / (r["kappa_MHz"] * 1e-3) == pytest.approx(
        r["chi_over_kappa"], rel=LOOSE)
    # T1 Purcell with the corrected (even) detuning
    Dl = f01 - r["f_r_GHz"]
    T1 = 1.0 / ((g / Dl) ** 2 * 2 * np.pi * r["kappa_MHz"] * 1e6)
    assert T1 * 1e6 == pytest.approx(r["T1_Purcell_us"], rel=LOOSE)
    assert T1 >= cavity.T1_PURCELL_MIN_S * tc.T1_MARGIN_TARGET
    # 0-1 dispersivity
    assert (g / Dl) ** 2 <= cavity.G_OVER_DELTA_SQ_MAX
    assert (g / Dl) ** 2 == pytest.approx(r["g_over_Delta_sq"], rel=LOOSE)
    # C7 1-2-line dispersivity -- the NEW constraint
    g12 = ratio * g
    D12 = r["f_r_GHz"] - f12
    assert (g12 / D12) ** 2 <= tc.G12_OVER_DELTA12_SQ_MAX
    assert (g12 / D12) ** 2 == pytest.approx(r["g12_over_Delta12_sq"], rel=LOOSE)
    assert r["g12_MHz"] == pytest.approx(ratio * r["g_MHz"], rel=LOOSE)
    # C8 two-pole robustness screen
    chi2 = cavity.chi_two_pole(g, f01, f12, r["f_r_GHz"], r["n_ratio_sq"])
    assert abs(chi2) / (r["kappa_MHz"] * 1e-3) >= 1.0
    assert abs(chi2) / (r["kappa_MHz"] * 1e-3) == pytest.approx(
        r["chi_over_kappa_two_pole"], rel=LOOSE)
    # and the point is where the correction says it should be
    assert r["f_r_GHz"] < f01, "corrected optimum must be BELOW the qubit"
    assert r["below_1_2_line"] is True
    assert r["inside_gap_protection_window"] is True


def test_gap_protection_window_is_recomputed_not_read(design, on_disk):
    """The scan window is the readout lattice's own gap covering f01."""
    f01 = design["f01_GHz"]
    lo, hi = tc.gap_protection_window_GHz(f01)
    gap = PHOTONICS["modes"]["cavity_line_gap_GHz"]
    assert lo == pytest.approx(f01 * F_CAV_GHZ / gap[1], rel=REL)
    assert hi == pytest.approx(f01 * F_CAV_GHZ / gap[0], rel=REL)
    # f01 sits inside the Maxwell-scaled gap at both edges, by construction
    for f_r in (lo, hi, 0.5 * (lo + hi)):
        g_lo, g_hi = tc.scaled_gap_GHz(f_r)
        assert g_lo <= f01 * (1 + 1e-12) and f01 <= g_hi * (1 + 1e-12)
    # exploration cross-check
    blk = on_disk["gap_protection_window"]
    assert max(blk["reldev_vs_exploration"]) < 1e-12
    assert blk["f_r_GHz"] == pytest.approx([lo, hi], rel=REL)


def test_superseded_recommended_point_is_retained_and_fails(on_disk):
    """The 44.5966 GHz point is preserved verbatim AND marked as failing."""
    blk = on_disk["reference_convention_scan"]
    assert "REFERENCE CONVENTION" in blk["flag"]
    assert "NOT A DESIGN WINDOW" in blk["flag"]
    ref = blk["recommended_point"]
    assert ref["f_r_GHz"] == pytest.approx(44.5966287014581, rel=REL)
    assert ref["d_eff_um"] == pytest.approx(57.5, rel=REL)
    assert ref["kappa_MHz"] == pytest.approx(3.449999999999995, rel=REL)
    assert ref["chi_MHz"] == pytest.approx(-3.4816278995319045, rel=REL)
    assert ref["chi_over_kappa"] == pytest.approx(1.0091675071106985, rel=REL)
    assert ref["T1_Purcell_us"] == pytest.approx(36.54018716597618, rel=REL)
    assert ref["eta_readout"] == pytest.approx(0.11532819289330343, rel=REL)
    cv = blk["corrected_values_at_this_point"]
    assert cv["chi_over_kappa_koch"] == pytest.approx(0.4242, abs=1e-3)
    assert cv["chi_over_kappa_two_pole"] == pytest.approx(0.4664, abs=1e-3)
    assert cv["meets_chi_over_kappa_min"] is False


def test_c7_makes_the_ruling_candidate_infeasible(design, r_exact, on_disk):
    """f_r = 39.2589 GHz is not marginal under C7 -- it is empty."""
    f01, alpha = design["f01_GHz"], design["anharm_GHz"]
    f_r = 39.258876681907815          # src: exploration koch_convention point
    kmax = float(tc.kappa_max_MHz(f_r - f01, alpha, r_exact,
                                  tc.T1_MARGIN_TARGET))
    kint = tc.kappa_intrinsic_MHz(f_r, Q_I)
    assert kmax < kint, (kmax, kint)
    assert kmax == pytest.approx(1.811, abs=2e-3)
    assert kint == pytest.approx(2.6868, abs=1e-3)
    # the envelope agrees, and reports the killed band
    env = on_disk["gap_window_scan"]["envelope"]
    dead = [row for row in env["rows"] if not row["feasible"]
            and row["f_r_GHz"] < f01]
    assert dead, "the C7-killed band must appear in the envelope"
    assert min(r_["f_r_GHz"] for r_ in dead) < f_r < max(
        r_["f_r_GHz"] for r_ in dead)
    assert len(env["feasible_f_r_bands_GHz"]) >= 2, \
        "the window must be split in two by the 1-2 line"


def test_two_pole_screen_costs_efficiency_and_is_reversible(on_disk):
    """C8 is a real, quantified constraint, not a relabelling."""
    ko = on_disk["gap_window_scan"]["koch_only_optimum"]
    rec = on_disk["recommended_point"]
    assert ko["point"]["eta_readout"] > rec["eta_readout"]
    assert ko["point"]["chi_over_kappa_two_pole"] < 1.0
    assert ko["eta_cost_of_two_pole_screen"] == pytest.approx(
        ko["point"]["eta_readout"] - rec["eta_readout"], rel=LOOSE)
    assert tc.TWO_POLE_SCREEN is True
    assert rec["chi_over_kappa_two_pole"] >= 1.0


def test_kappa_max_is_the_exact_feasibility_test(design, r_exact):
    """kappa_max_MHz brackets the grid scan: no feasible row exceeds it."""
    n01, f01 = design["n01"], design["f01_GHz"]
    alpha = design["anharm_GHz"]
    lo, hi = tc.gap_protection_window_GHz(f01)
    f_r = np.arange(lo, hi, 0.37)                 # off-grid probe frequencies
    f_r = f_r[(np.abs(f_r - f01) > 0.05)
              & (np.abs(f_r - (f01 + alpha)) > 0.05)]
    rows = tc.scan_rows(n01, f01, alpha, f_r_GHz=f_r,
                        d_effs_m=tuple(np.arange(10.0, 160.0, 5.0) * 1e-6),
                        kappas_MHz=tuple(np.arange(2.0, 8.0, 0.1)))
    n_checked = 0
    for row in rows:
        if row["feasible"] and row["margin_T1"] >= tc.T1_MARGIN_TARGET:
            kmax = float(tc.kappa_max_MHz(row["Delta_r_GHz"], alpha, r_exact,
                                          tc.T1_MARGIN_TARGET))
            assert row["kappa_MHz"] <= kmax * (1 + 1e-9), row
            n_checked += 1
    assert n_checked > 50, "probe grid produced too few feasible rows to test"


def test_d_eff_corridor_reports_which_bound_binds(payload, design):
    corr = payload["recommended_point"]["corridor_at_this_point"]
    assert corr["binding_upper_bound"] in ("T1_Purcell", "dispersive_01",
                                           "dispersive_12")
    assert corr["two_pole_screen"] is True
    assert corr["g_lo_MHz"] == pytest.approx(
        max(corr["g_lo_koch_MHz"], corr["g_lo_two_pole_MHz"]), rel=REL)
    assert corr["g_hi_MHz"] == pytest.approx(
        min(corr["g_hi_T1_MHz"], corr["g_hi_dispersive_01_MHz"],
            corr["g_hi_dispersive_12_MHz"]), rel=REL)


def test_corrected_fine_scan_is_deterministic_and_blocking_invariant(design):
    """Blocking over f_r is a memory device only: it must not move the answer."""
    n01, f01 = design["n01"], design["f01_GHz"]
    alpha = design["anharm_GHz"]
    lo, hi = tc.gap_protection_window_GHz(f01)
    f_r = np.arange(lo, 39.0, 0.05)
    f_r = f_r[np.abs(f_r - (f01 + alpha)) > tc.POLE_GUARD_GHZ]
    kw = dict(f_r_GHz=f_r, d_effs_um=np.arange(20.0, 100.0, 1.0),
              kappas_MHz=np.arange(2.0, 6.0, 0.05))
    a, sa = tc.fine_scan(n01, f01, alpha, block=7, **kw)
    b, sb = tc.fine_scan(n01, f01, alpha, block=1000, **kw)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    assert sa["n_ok"] == sb["n_ok"] and sa["n_points"] == sb["n_points"]
