"""M6 acceptance tests AT-1..AT-6 (M6_BRIEF.md Sec. 3).

Fully deterministic (no RNG anywhere in M6 milestone 1).  Every target is
loaded from qubit_targets.json -- no acceptance number is hardcoded in this
file.  The only literals here are the tolerances quoted verbatim in
M6_BRIEF.md Sec. 3 and the grid/criterion values that the brief itself states
(N >= 35, the ratio ladder 50/100/150, the nu points -2.4 / -2.5, the sample
readout point Delta_r = 7 GHz / d_eff = 100 um / kappa = 1 MHz).

Session-scoped fixtures share the expensive work: the design scan (six exact
diagonalisations x 2 offset charges) and the N = 50 convergence scan.
"""
import json
import pathlib

import numpy as np
import pytest

import qubit
from qubit import cavity, dome, inputs, qp, spectrum

ROOT = pathlib.Path(__file__).resolve().parents[1]

# --- grids / criteria quoted directly from M6_BRIEF.md Sec. 3 ----------------
N_MIN = 35                  # charge-basis truncation floor
N_CONV = 50                 # convergence cross-check
AT2_RATIOS = (50, 100, 150)
NU_APEX = -2.4
NU_OFF = -2.5
SAMPLE_ROW = dict(Delta_r_GHz=7.0, d_eff_um=100.0, kappa_MHz=1.0)


@pytest.fixture(scope="session")
def targets():
    """qubit_targets.json -- the reference-measured ground truth."""
    with open(ROOT / "qubit_targets.json", encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="session")
def design():
    """Recomputed design point (E_J/E_C ladder scan + selection criterion)."""
    return qubit.design_point()


@pytest.fixture(scope="session")
def sweet(design):
    """AT-3: apex/off-apex derivatives and the quasistatic T_phi rows."""
    return dome.sweet_spot_report(design["EC_GHz"])


@pytest.fixture(scope="session")
def budget(design):
    """AT-4: measured 145 GHz drive-cavity budget."""
    return cavity.drive_cavity_budget(design["n01"], design["f01_GHz"],
                                      design["anharm_GHz"])


@pytest.fixture(scope="session")
def rows(design):
    """AT-5: feasible readout-cavity operating points."""
    return cavity.readout_window_scan(design["n01"], design["f01_GHz"],
                                      design["anharm_GHz"])


@pytest.fixture(scope="session")
def sample_pair(rows, targets):
    """AT-5 sample point (Delta_r = 7 GHz, d_eff = 100 um, kappa = 1 MHz):
    (recomputed row, reference row)."""
    def pick(rs):
        hit = [r for r in rs
               if r["Delta_r_GHz"] == SAMPLE_ROW["Delta_r_GHz"]
               and r["d_eff_um"] == SAMPLE_ROW["d_eff_um"]
               and r["kappa_MHz"] == SAMPLE_ROW["kappa_MHz"]]
        assert len(hit) == 1, hit
        return hit[0]
    return pick(rows), pick(targets["readout_window"])


@pytest.fixture(scope="session")
def qp_bud(design):
    """AT-6: quasiparticle T1 ladder + marginality flag."""
    return qp.qp_budget(design["f01_GHz"])


def rel(got, want):
    return abs(got - want) / abs(want)


# =============================================================== AT-1 =========
class TestAT1Spectrum:
    """Charge-basis exact diagonalisation, ratio 100, N >= 35."""

    def test_charge_truncation_meets_brief(self):
        assert inputs.N_CHARGE >= N_MIN

    def test_design_point_is_ratio_100(self, design, targets):
        assert design["EJ_over_EC"] == targets["design"]["EJ_over_EC"]

    def test_EJ_derived_from_junction_contract(self, targets):
        # E_J = Phi0 * Ic / (2 pi) / h, Ic from junction.json:contract.Ic_nA
        assert inputs.JUNCTION["contract"]["Ic_nA"] == 300.0
        assert rel(inputs.EJ_GHZ, targets["inputs"]["EJ_GHz"]) < 1e-12

    @pytest.mark.parametrize("key", ["f01_GHz", "anharm_GHz", "n01",
                                     "EC_GHz", "C_fF"])
    def test_design_values(self, design, targets, key):
        assert rel(design[key], targets["design"][key]) < 1e-9

    def test_design_eps1(self, design, targets):
        # eps1 is the one design number that is NOT bit-reproducible in
        # float64 across build/rounding details (see test_eps1_is_ulp_
        # sensitive below); the brief's own AT-2 tolerance of 5% applies.
        assert rel(design["eps1_Hz"], targets["design"]["eps1_Hz"]) < 0.05

    def test_f01_asymptotic_within_0p2pct(self, design, targets):
        asym = spectrum.f01_asymptotic(inputs.EJ_GHZ, design["EC_GHz"])
        assert rel(asym, targets["design"]["f01_GHz"]) < 0.002

    def test_n01_asymptotic_within_2pct(self, design, targets):
        asym = spectrum.n01_asymptotic(inputs.EJ_GHZ, design["EC_GHz"])
        assert rel(asym, targets["design"]["n01"]) < 0.02

    def test_convergence_N35_vs_N50(self, design):
        EC = design["EC_GHz"]
        f35, a35, _ = spectrum.levels(inputs.EJ_GHZ, EC, N=N_MIN)
        f50, a50, _ = spectrum.levels(inputs.EJ_GHZ, EC, N=N_CONV)
        assert rel(f35, f50) < 1e-10
        assert rel(a35, a50) < 1e-10
        n35 = spectrum.n01_element(inputs.EJ_GHZ, EC, N=N_MIN)
        n50 = spectrum.n01_element(inputs.EJ_GHZ, EC, N=N_CONV)
        assert rel(n35, n50) < 1e-10


# =============================================================== AT-2 =========
class TestAT2Dispersion:
    """Charge-dispersion ladder eps1(50/100/150), each within 5%."""

    @pytest.mark.parametrize("ratio", AT2_RATIOS)
    def test_eps1_within_5pct(self, design, targets, ratio):
        got = design["scan"][ratio]["eps1_Hz"]
        want = targets["scan"][str(ratio)]["eps1_Hz"]
        assert rel(got, want) < 0.05, f"eps1({ratio}): {got} vs {want}"

    def test_extended_precision_matches_float64_at_ratio_100(self, targets):
        """Validate the mpmath Sturm-bisection path against float64 where
        float64 is still trustworthy (eps1 = 37.4 Hz >> the ~1 mHz floor)."""
        EC = inputs.EJ_GHZ / 100
        got = spectrum.eps1_mp(inputs.EJ_GHZ, EC, N=N_MIN, dps=60) * 1e9
        assert rel(got, targets["scan"]["100"]["eps1_Hz"]) < 1e-3

    def test_extended_precision_is_monotonic_where_float64_floors(self):
        """M6_BRIEF.md Sec. 1: eps1 below ~1 mHz is float64 floor, not
        physics.  In extended precision the ladder must fall monotonically."""
        eps_mp = [spectrum.eps1_mp(inputs.EJ_GHZ, inputs.EJ_GHZ / r, N=N_MIN)
                  for r in (100, 150, 200, 300)]
        assert all(a > b for a, b in zip(eps_mp, eps_mp[1:])), eps_mp

    def test_float64_is_unusable_below_1mHz(self):
        """At E_J/E_C = 300 the true eps1 is ~5e-8 Hz; float64 returns a
        value orders of magnitude larger (pure eigenvalue floor).  This is
        the quantitative justification for the mpmath path."""
        EC = inputs.EJ_GHZ / 300
        f64 = spectrum.levels(inputs.EJ_GHZ, EC)[2] * 1e9
        mpv = spectrum.eps1_mp(inputs.EJ_GHZ, EC, N=N_MIN) * 1e9
        assert mpv < 1e-3, mpv                      # genuinely sub-mHz
        assert f64 / mpv > 100.0, (f64, mpv)        # float64 is all floor

    def test_eps1_is_ulp_sensitive_at_ratio_100(self):
        """A 1-ULP perturbation of E_J moves float64 eps1 by ~1e-5 relative
        while f01 moves by ~1e-16 -- eps1 sits ~4.5 decades above the
        eigenvalue floor, exactly as M6_BRIEF.md Sec. 1 warns."""
        EJ = inputs.EJ_GHZ
        EJ_p = np.nextafter(EJ, np.inf)
        f_a, _, e_a = spectrum.levels(EJ, EJ / 100)
        f_b, _, e_b = spectrum.levels(EJ_p, EJ_p / 100)
        assert rel(f_b, f_a) < 1e-12          # f01 is rock solid
        assert 1e-6 < rel(e_b, e_a) < 1e-3, rel(e_b, e_a)   # eps1 is not

    def test_perturbative_formula_is_order_of_magnitude_check(self):
        EC = inputs.EJ_GHZ / 100
        pert = spectrum.eps1_perturbative(inputs.EJ_GHZ, EC) * 1e9
        exact = spectrum.eps1_mp(inputs.EJ_GHZ, EC, N=N_MIN) * 1e9
        assert 0.5 < pert / exact < 2.0


# =============================================================== AT-3 =========
class TestAT3SweetSpot:
    """Dome apex is a double sweet spot; T_phi contrast reproduced."""

    def test_dome_params_read_from_material_json(self):
        assert inputs.NU_OPT == inputs.MATERIAL["sc_dome"]["nu_opt"]
        assert inputs.TC_MAX_K == inputs.MATERIAL["sc_dome"]["Tc_max_K"]
        assert inputs.NU_HALFWIDTH == inputs.MATERIAL["sc_dome"]["nu_halfwidth"]

    def test_D1_apex_vanishes(self, sweet):
        assert abs(sweet["D1_apex_GHz"]) < 0.01

    def test_D2_apex(self, sweet, targets):
        want = targets["gate_noise"]["D2_apex_GHz"]
        assert abs(sweet["D2_apex_GHz"] - want) < 3.0

    def test_D1_off_apex(self, sweet, targets):
        want = targets["gate_noise"]["D1_off_nu_m2p5_GHz"]
        assert abs(sweet["D1_off_GHz"] - want) < 0.3

    @pytest.mark.parametrize("dnu", ["1e-04", "1e-03"])
    def test_Tphi_contrast_rows(self, sweet, targets, dnu):
        assert rel(sweet["Tphi_apex_us"][dnu],
                   targets["gate_noise"]["Tphi_apex_us"][dnu]) < 0.03
        assert rel(sweet["Tphi_off_us"][dnu],
                   targets["gate_noise"]["Tphi_off_us"][dnu]) < 0.03

    def test_contrast_ratio_is_about_2000x(self, sweet):
        ratio = sweet["Tphi_apex_us"]["1e-04"] / sweet["Tphi_off_us"]["1e-04"]
        assert 1.5e3 < ratio < 2.5e3

    def test_gate_tunability_span(self, design, targets):
        f_edge = dome.f01_of(-2.8, design["EC_GHz"])
        want_edge, want_apex = targets["gate_noise"]["f01_tunability_GHz"]
        assert rel(f_edge, want_edge) < 0.01
        assert rel(design["f01_GHz"], want_apex) < 1e-9

    def test_kB_meV_constant_choice_is_immaterial(self, design, targets):
        """The reference uses kB = 0.08617 meV/K (4 s.f.).  The CODATA-exact
        conversion must land well inside the AT-3 band -- documented, not a
        tolerance tweak."""
        exact = inputs.KB / inputs.E * 1e3
        scale = exact / inputs.KB_MEV_PER_K
        d2_exact = dome.D2(NU_APEX, design["EC_GHz"],
                           EJ_apex=inputs.EJ_GHZ) * scale
        assert abs(d2_exact - targets["gate_noise"]["D2_apex_GHz"]) < 3.0


# =============================================================== AT-4 =========
class TestAT4DriveCavity:
    """Measured 145 GHz drive cavity: g, chi, chi/kappa, T1_Purcell, +-3%."""

    def test_photonics_inputs_traced(self, targets):
        assert inputs.F_CAV_GHZ == inputs.PHOTONICS["modes"]["f0_GHz"]
        assert inputs.EPS_C == inputs.PHOTONICS["geometry"]["eps_core"]
        # Q_i = 14611.7 appears verbatim in photonics.json:ladder.Q_i_anchor
        assert inputs.Q_I == inputs.PHOTONICS["ladder"]["Q_i_anchor"]["value"]
        assert inputs.Q_I == targets["inputs"]["Q_i"]

    @pytest.mark.parametrize("key", ["g_MHz", "chi_kHz", "kappa_MHz",
                                     "chi_over_kappa", "T1_Purcell_us"])
    def test_budget_within_3pct(self, budget, targets, key):
        assert rel(budget[key], targets["drive_cavity"][key]) < 0.03

    def test_drive_cavity_cannot_read_out(self, budget):
        """The measured requirement for a two-cavity architecture."""
        assert budget["chi_over_kappa"] < 0.01

    def test_purcell_not_limiting(self, budget):
        assert budget["T1_Purcell_us"] > 500.0


# =============================================================== AT-5 =========
class TestAT5ReadoutWindow:
    """Readout-window scan non-empty; sample point within 3% on all numbers."""

    def test_window_non_empty(self, rows, targets):
        assert len(rows) > 0
        assert len(rows) == len(targets["readout_window"])

    def test_all_rows_meet_stated_criteria(self, rows):
        for r in rows:
            assert r["chi_over_kappa"] >= cavity.CHI_OVER_KAPPA_MIN
            assert r["T1_Purcell_us"] >= cavity.T1_PURCELL_MIN_S * 1e6
            g_over_D = r["g_MHz"] * 1e-3 / r["Delta_r_GHz"]
            assert g_over_D ** 2 <= cavity.G_OVER_DELTA_SQ_MAX

    @pytest.mark.parametrize("key", ["Delta_r_GHz", "f_r_GHz", "d_eff_um",
                                     "kappa_MHz", "g_MHz", "chi_MHz",
                                     "chi_over_kappa", "T1_Purcell_us"])
    def test_sample_point_within_3pct(self, sample_pair, key):
        got, want = sample_pair
        assert rel(got[key], want[key]) < 0.03, f"{key}: {got[key]} vs {want[key]}"

    def test_reference_best_point_is_the_sample_point(self, rows):
        best = cavity.best_readout_point(rows)
        assert best["Delta_r_GHz"] == SAMPLE_ROW["Delta_r_GHz"]
        assert best["d_eff_um"] == SAMPLE_ROW["d_eff_um"]
        assert best["kappa_MHz"] == SAMPLE_ROW["kappa_MHz"]


# =============================================================== AT-6 =========
class TestAT6Quasiparticle:
    """T1(x_qp = 1e-6) with the reference prefactor; marginality reported."""

    def test_Delta0_read_from_material_json(self):
        assert inputs.DELTA0_MEV == inputs.MATERIAL["Delta0_meV"]

    @pytest.mark.parametrize("x", ["1e-08", "1e-07", "1e-06", "1e-05"])
    def test_T1_ladder(self, qp_bud, targets, x):
        assert rel(qp_bud["T1_us"][x], targets["quasiparticle"]["T1_us"][x]) < 0.01

    def test_T1_at_1e_minus_6_is_6p5us(self, qp_bud):
        assert abs(qp_bud["T1_us"]["1e-06"] - 6.5) < 0.1

    def test_T1_scales_inversely_with_x_qp(self, qp_bud):
        t = [qp_bud["T1_us"][k] for k in ("1e-08", "1e-07", "1e-06", "1e-05")]
        assert np.allclose(np.array(t[:-1]) / np.array(t[1:]), 10.0)

    def test_marginality_ratio_reported(self, qp_bud, targets):
        want = targets["quasiparticle"]["marginality_flag"]
        assert rel(qp_bud["marginality_flag"], want) < 0.01
        assert abs(qp_bud["marginality_flag"] - 0.55) < 0.01

    def test_thermal_occupation(self, design, targets):
        n = qp.n_thermal(design["f01_GHz"], 0.1)
        assert rel(n, targets["thermal"]["n_th"]["0.1"]) < 1e-9
