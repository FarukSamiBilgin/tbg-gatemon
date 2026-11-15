"""M6 deliverable P2 -- coupling-convention pin (qubit/coupling.py).

Deterministic; no RNG.  Every physical number is read from the JSONs or
recomputed from qubit/, never typed.  The reference path (qubit/cavity.py) is
used as the beta = 1, single-electron-charge limit and must be reproduced
bit-exactly by the P2 operator in that limit -- that is the continuity test.
"""
import json
import pathlib

import pytest

import qubit
from qubit import cavity, coupling
from qubit.inputs import (D_EFF_DRIVE_M, EPS0, EPS_C, F_CAV_GHZ, JUNCTION,
                          TARGETS, V_MODE_M3)

ROOT = pathlib.Path(__file__).resolve().parents[1]
EXPORT = ROOT / "out" / "m6" / "p2_coupling.json"

REL = 1e-12          # algebraic identities: float64 round-trip only
LOOSE = 1e-9         # identities that pass through a JSON text round-trip


@pytest.fixture(scope="session")
def design():
    return qubit.design_point()


@pytest.fixture(scope="session")
def payload():
    """Regenerate the export in-process, so the test never trusts a stale file."""
    _, pay = coupling.main()
    return pay


@pytest.fixture(scope="session")
def on_disk():
    assert EXPORT.exists(), f"{EXPORT} not written"
    with open(EXPORT, encoding="utf-8") as fh:
        return json.load(fh)


# ------------------------------------------------------------------ (i) ------
def test_beta_one_reference_charge_recovers_reference_g(design):
    """Continuity: beta -> 1 AND charge_units -> 1 is exactly the reference."""
    n01 = design["n01"]
    g_ref = cavity.coupling_g(F_CAV_GHZ, V_MODE_M3, D_EFF_DRIVE_M, n01)
    g_p2 = coupling.coupling_g(F_CAV_GHZ, V_MODE_M3, D_EFF_DRIVE_M, n01,
                               beta=1.0, charge_units=1.0)
    assert g_p2 == pytest.approx(g_ref, rel=REL)
    # ... and it also reproduces the AT-4 budget row's g.
    row = cavity.drive_cavity_budget(n01, design["f01_GHz"],
                                     design["anharm_GHz"])
    assert g_p2 * 1e3 == pytest.approx(row["g_MHz"], rel=REL)


def test_beta_one_cooper_pair_charge_is_exactly_twice_reference(design):
    """The 2e-vs-e audit, isolated from the voltage division."""
    n01 = design["n01"]
    g_ref = cavity.coupling_g(F_CAV_GHZ, V_MODE_M3, D_EFF_DRIVE_M, n01)
    g_2e = coupling.coupling_g(F_CAV_GHZ, V_MODE_M3, D_EFF_DRIVE_M, n01,
                               beta=1.0)
    assert g_2e == pytest.approx(2.0 * g_ref, rel=REL)


def test_operator_is_linear_in_beta(design):
    """H_int = 2 beta e V_zpf n^  =>  g is exactly proportional to beta."""
    n01 = design["n01"]
    g1 = coupling.coupling_g(F_CAV_GHZ, V_MODE_M3, D_EFF_DRIVE_M, n01, 1.0)
    for b in (0.05, 0.1727, 0.5, 0.8636, 0.9231):
        gb = coupling.coupling_g(F_CAV_GHZ, V_MODE_M3, D_EFF_DRIVE_M, n01, b)
        assert gb == pytest.approx(b * g1, rel=REL)


# ----------------------------------------------------------------- (ii) ------
def test_beta_in_unit_interval():
    """beta in (0, 1] everywhere in the declared parameter box."""
    kg_lo, kg_hi = coupling.KAPPA_GEO_RANGE
    ee_lo, ee_hi = coupling.EPS_EFF_RANGE
    d_effs = (D_EFF_DRIVE_M, coupling.D_EFF_READOUT_M, 25.24e-6, 200e-6)
    for d in d_effs:
        for kg in (kg_lo, coupling.KAPPA_GEO, kg_hi):
            for ee in (ee_lo, coupling.EPS_EFF, ee_hi):
                b = coupling.beta_selfcap(d, kg, ee)
                assert 0.0 < b <= 1.0, (d, kg, ee, b)
    # the engineered-overlap branch cannot escape the interval either
    assert 0.0 < coupling.beta_from_Cc(coupling.c_c_overlap()) <= 1.0


def test_beta_capped_by_capacitance_budget():
    """C_c <= C_Sigma - C_J  =>  beta <= beta_max = 1 - C_J/C_Sigma."""
    C_S = TARGETS["design"]["C_fF"] * 1e-15
    C_J = JUNCTION["contract"]["C_fF"] * 1e-15
    bmax = coupling.beta_max()
    assert bmax == pytest.approx(1.0 - C_J / C_S, rel=REL)
    assert 0.0 < bmax < 1.0
    # an absurdly large electrode saturates, never exceeds
    assert coupling.beta_from_Cc(1e-12) == pytest.approx(bmax, rel=REL)
    assert coupling.beta_selfcap(1e-3) == pytest.approx(bmax, rel=REL)


def test_beta_equals_the_divider_algebra():
    """beta = C_c/(C_c + C_s + C_J) with C_c + C_s + C_J == C_Sigma."""
    C_S = TARGETS["design"]["C_fF"] * 1e-15
    C_J = JUNCTION["contract"]["C_fF"] * 1e-15
    C_c = coupling.c_c_selfcap(D_EFF_DRIVE_M)
    C_s = C_S - C_c - C_J                    # shunt pad = remainder of budget
    assert C_s > 0.0
    assert coupling.beta_selfcap(D_EFF_DRIVE_M) == pytest.approx(
        C_c / (C_c + C_s + C_J), rel=REL)


def test_c_c_geometry_formulas():
    """C_c estimates match their stated closed forms and the trace sources."""
    assert coupling.EPS_EFF == pytest.approx((1.0 + EPS_C) / 2.0, rel=REL)
    assert coupling.c_c_selfcap(D_EFF_DRIVE_M) == pytest.approx(
        coupling.KAPPA_GEO * EPS0 * coupling.EPS_EFF * D_EFF_DRIVE_M, rel=REL)
    assert coupling.c_c_overlap() == pytest.approx(
        EPS0 * EPS_C * coupling.W_C_M * D_EFF_DRIVE_M / coupling.G_C_M, rel=REL)
    # C_c is monotone increasing in d_eff, kappa_geo and eps_eff
    assert (coupling.c_c_selfcap(D_EFF_DRIVE_M)
            < coupling.c_c_selfcap(coupling.D_EFF_READOUT_M))


def test_beta_bracket_orders_and_contains_default():
    for d in (D_EFF_DRIVE_M, coupling.D_EFF_READOUT_M):
        lo, mid, hi = coupling.beta_bracket(d)
        assert 0.0 < lo <= mid <= hi <= coupling.beta_max()


# ---------------------------------------------------------------- (iii) ------
def test_export_exists_and_is_deterministic(payload, on_disk):
    """Regenerating the export reproduces the file on disk exactly."""
    assert json.dumps(payload, sort_keys=True) == json.dumps(on_disk,
                                                             sort_keys=True)
    assert on_disk["provenance"]["deterministic"] is True


def test_export_g_matches_the_operator(payload):
    """Exported corrected g == 2 beta e V_zpf n01 / h, recomputed from scratch."""
    d = payload["drive_cavity_corrected"]
    assert d["g_MHz"] == pytest.approx(d["g_MHz_from_operator"], rel=LOOSE)
    assert d["g_MHz"] == pytest.approx(
        d["g_ref_MHz"] * payload["correction_factor"]["drive"], rel=LOOSE)
    r = payload["readout_sample_corrected"]
    assert r["g_MHz"] == pytest.approx(
        r["g_ref_MHz"] * payload["correction_factor"]["readout_sample"],
        rel=LOOSE)


def test_export_chi_from_g_via_cavity_formula(payload, design):
    """chi is the SAME dispersive formula as qubit/cavity.py, evaluated at g_P2."""
    alpha = design["anharm_GHz"]
    d = payload["drive_cavity_corrected"]
    Delta = d["Delta_GHz"]
    chi = cavity.chi_dispersive(d["g_MHz"] * 1e-3, Delta, alpha) * 1e6
    assert d["chi_kHz"] == pytest.approx(chi, rel=LOOSE)
    assert d["chi_kHz_from_formula"] == pytest.approx(chi, rel=LOOSE)
    assert d["chi_over_kappa"] == pytest.approx(abs(chi * 1e-3)
                                                / d["kappa_MHz"], rel=LOOSE)

    r = payload["readout_sample_corrected"]
    chir = cavity.chi_dispersive(r["g_MHz"] * 1e-3, r["Delta_r_GHz"],
                                 alpha) * 1e3
    assert r["chi_MHz"] == pytest.approx(chir, rel=LOOSE)
    assert r["chi_MHz_from_formula"] == pytest.approx(chir, rel=LOOSE)
    assert r["chi_over_kappa"] == pytest.approx(abs(chir) / r["kappa_MHz"],
                                                rel=LOOSE)


def test_export_purcell_from_g(payload):
    """T1_Purcell = 1/[(g/Delta)^2 * 2 pi kappa], same form as cavity.py."""
    d = payload["drive_cavity_corrected"]
    T1 = 1.0 / cavity.purcell_rate(d["g_MHz"] * 1e-3, d["Delta_GHz"],
                                   d["kappa_MHz"] * 1e-3) * 1e6
    assert d["T1_Purcell_us"] == pytest.approx(T1, rel=LOOSE)
    r = payload["readout_sample_corrected"]
    T1r = 1.0 / cavity.purcell_rate(r["g_MHz"] * 1e-3, r["Delta_r_GHz"],
                                    r["kappa_MHz"] * 1e-3) * 1e6
    assert r["T1_Purcell_us"] == pytest.approx(T1r, rel=LOOSE)


def test_export_reference_rows_are_the_unmodified_at4_at5_rows(payload, design):
    """The stored reference rows are exactly what qubit/cavity.py produces."""
    n01, f01, alpha = design["n01"], design["f01_GHz"], design["anharm_GHz"]
    ref = cavity.drive_cavity_budget(n01, f01, alpha)
    for k, v in payload["drive_cavity_reference"].items():
        if isinstance(v, (int, float)):
            assert ref[k] == pytest.approx(v, rel=LOOSE), k
    rows = cavity.readout_window_scan(n01, f01, alpha)
    row = next(x for x in rows if x["Delta_r_GHz"] == 7.0
               and x["d_eff_um"] == 100.0 and x["kappa_MHz"] == 1.0)
    for k, v in payload["readout_sample_reference"].items():
        assert row[k] == pytest.approx(v, rel=LOOSE), k


def test_export_traces_to_json_inputs(payload):
    p = payload["model"]["params"]
    assert p["C_Sigma_fF"] == pytest.approx(TARGETS["design"]["C_fF"], rel=REL)
    assert p["C_J_fF"] == pytest.approx(JUNCTION["contract"]["C_fF"], rel=REL)
    assert p["eps_pp"] == pytest.approx(EPS_C, rel=REL)
    assert p["C_c_budget_cap_fF"] == pytest.approx(
        TARGETS["design"]["C_fF"] - JUNCTION["contract"]["C_fF"], rel=1e-9)


# ----------------------------------------------------------------- (iv) ------
@pytest.mark.parametrize("corr_key, ref_key, cor_key, chi_key", [
    ("drive", "drive_cavity_reference", "drive_cavity_corrected", "chi_kHz"),
    ("readout_sample", "readout_sample_reference",
     "readout_sample_corrected", "chi_MHz"),
])
def test_correction_applied_multiplicatively(payload, corr_key, ref_key,
                                             cor_key, chi_key):
    """g ~ c, chi ~ c^2, chi/kappa ~ c^2, T1_Purcell ~ 1/c^2, kappa fixed."""
    c = payload["correction_factor"][corr_key]
    ref, cor = payload[ref_key], payload[cor_key]
    assert cor["g_MHz"] == pytest.approx(ref["g_MHz"] * c, rel=LOOSE)
    assert cor[chi_key] == pytest.approx(ref[chi_key] * c ** 2, rel=LOOSE)
    assert cor["chi_over_kappa"] == pytest.approx(
        ref["chi_over_kappa"] * c ** 2, rel=LOOSE)
    assert cor["T1_Purcell_us"] == pytest.approx(
        ref["T1_Purcell_us"] / c ** 2, rel=LOOSE)
    assert cor["kappa_MHz"] == pytest.approx(ref["kappa_MHz"], rel=LOOSE)
    # and the self-consistency of the two ratios: (g'/g)^2 == chi'/chi
    assert (cor["g_MHz"] / ref["g_MHz"]) ** 2 == pytest.approx(
        cor[chi_key] / ref[chi_key], rel=LOOSE)
    assert (cor["g_MHz"] / ref["g_MHz"]) ** 2 == pytest.approx(
        ref["T1_Purcell_us"] / cor["T1_Purcell_us"], rel=LOOSE)


def test_correction_factor_is_two_beta(payload):
    assert payload["correction_factor"]["drive"] == pytest.approx(
        2.0 * payload["beta"]["drive"], rel=REL)
    assert payload["correction_factor"]["readout_sample"] == pytest.approx(
        2.0 * payload["beta"]["readout_sample"], rel=REL)
    assert payload["correction_factor"]["ceiling"] == pytest.approx(
        2.0 * coupling.beta_max(), rel=REL)
    # break-even: the correction passes through 1 at C_c = C_Sigma/2
    assert coupling.correction_factor(0.5) == pytest.approx(1.0, rel=REL)


def test_readout_sample_still_meets_the_at5_criteria(payload):
    """The corrected readout point must not fall out of the feasible window."""
    r = payload["readout_sample_corrected"]
    assert abs(r["chi_MHz"]) / r["kappa_MHz"] >= cavity.CHI_OVER_KAPPA_MIN
    assert r["T1_Purcell_us"] * 1e-6 >= cavity.T1_PURCELL_MIN_S
    assert r["g_over_Delta_sq"] <= cavity.G_OVER_DELTA_SQ_MAX


def test_two_cavity_budget_is_respected(payload):
    """Both collection electrodes share one C_Sigma; the sum must fit."""
    tc = payload["two_cavity_budget"]
    assert tc["C_c_drive_fF"] + tc["C_c_readout_fF"] == pytest.approx(
        tc["budget_cap_fF"], rel=1e-9)
    assert tc["oversubscription"] > 1.0      # the flagged design finding
    assert 0.0 < tc["scale_applied"] < 1.0
    assert tc["correction_drive"] < payload["correction_factor"]["drive"]
    assert tc["correction_readout"] < payload["correction_factor"]["readout_sample"]
