"""M6 deliverable P-extra -- non-sinusoidal CPR in the transmon (qubit/cpr_transmon.py).

Resolves the deferred flag in paper/decisions.md (2026-07-27, "M6 flag
(deferred): non-sinusoidal CPR vs transmon Hamiltonian").

Deterministic; no RNG.  Every physical number is read from material.json /
qubit_targets.json or recomputed from qubit/, never typed.  The pure-sinusoid
limit must reproduce qubit/spectrum.py -- that is the continuity test, and it
holds BITWISE (the two Hamiltonian builders emit identical float64 matrices).

Test map
    (i)   single-harmonic limit == spectrum.py to 1e-12  (asserted bitwise too)
    (ii)  the measured harmonics obey max_phi sum c_m sin(m phi) = 1
    (iii) N convergence, N = 35 vs 60
    (iv)  the on-disk export is bit-consistent with an in-process rerun
    (v)   the headline deltas, recomputed here, match the export
"""
import json
import pathlib

import numpy as np
import pytest

from qubit import cpr_transmon as ct
from qubit import dome, spectrum
from qubit.inputs import EJ_GHZ, MATERIAL, N_CHARGE, TARGETS

ROOT = pathlib.Path(__file__).resolve().parents[1]
EXPORT = ROOT / "out" / "m6" / "pextra_cpr.json"

TIGHT = 1e-12        # the continuity limit: float64 round-trip only
NORM_TOL = 1e-6      # M1 normalized on an 8192-point grid; residual ~4e-8
EPS1_F64_TOL = 1e-3  # float64 eps1 is ULP-conditioned (decisions.md 2026-07-27)

EC = TARGETS["design"]["EC_GHz"]
TAU = str(MATERIAL["cpr"]["tau_nominal"])


@pytest.fixture(scope="session")
def payload():
    """Regenerate in-process, so no test ever trusts a stale file."""
    return ct.build_export()


@pytest.fixture(scope="session")
def on_disk():
    assert EXPORT.exists(), f"{EXPORT} not written -- run python -m qubit.cpr_transmon"
    with open(EXPORT, encoding="utf-8") as fh:
        return json.load(fh)


# ----------------------------------------------------------------- (i) -------
def test_single_harmonic_hamiltonian_is_bitwise_spectrum_py():
    """c = [1.0] must rebuild spectrum.hamiltonian exactly, not merely closely."""
    for ng in (0.0, 0.5):
        for N in (N_CHARGE, 60):
            a = ct.hamiltonian(EJ_GHZ, EC, ct.C_SINUSOIDAL, ng, N)
            b = spectrum.hamiltonian(EJ_GHZ, EC, ng, N)
            assert np.array_equal(a, b), f"ng={ng} N={N}: matrices differ"


def test_single_harmonic_levels_match_spectrum_py():
    """(i) f01 / anharmonicity / eps1 / n01 reproduce spectrum.py to 1e-12."""
    f01, anh, eps1 = ct.levels(EJ_GHZ, EC, ct.C_SINUSOIDAL, N_CHARGE)
    g01, ganh, geps1 = spectrum.levels(EJ_GHZ, EC, N_CHARGE)
    assert f01 == pytest.approx(g01, rel=TIGHT)
    assert anh == pytest.approx(ganh, rel=TIGHT)
    assert eps1 == pytest.approx(geps1, rel=TIGHT)
    assert (ct.n01_element(EJ_GHZ, EC, ct.C_SINUSOIDAL, N_CHARGE)
            == pytest.approx(spectrum.n01_element(EJ_GHZ, EC, N_CHARGE),
                             rel=TIGHT))
    # and against the frozen design row that the rest of M6 is built on
    d = TARGETS["design"]
    assert f01 == pytest.approx(d["f01_GHz"], rel=TIGHT)
    assert anh == pytest.approx(d["anharm_GHz"], rel=TIGHT)
    assert eps1 * 1e9 == pytest.approx(d["eps1_Hz"], rel=TIGHT)


def test_single_harmonic_dome_derivatives_match_dome_py():
    """The sweet-spot stencil reduces to dome.D1/D2 in the sinusoidal limit."""
    assert (ct.D2(dome.NU_OPT, EC, ct.C_SINUSOIDAL)
            == pytest.approx(dome.D2(dome.NU_OPT, EC), rel=TIGHT))
    assert (ct.D1(-2.5, EC, ct.C_SINUSOIDAL)
            == pytest.approx(dome.D1(-2.5, EC), rel=TIGHT))
    assert ct.D1(dome.NU_OPT, EC, ct.C_SINUSOIDAL) == 0.0


def test_potential_reduces_to_minus_cos():
    """U(phi)/E_J = -cos(phi) for c = [1.0]; and U' = the CPR, for every tau."""
    phi = np.linspace(-np.pi, np.pi, 4001)
    assert np.allclose(ct.potential(phi, ct.C_SINUSOIDAL), -np.cos(phi),
                       rtol=0, atol=1e-15)
    for tau in ct.taus_available():
        c = ct.harmonics(tau)
        h = 1e-6
        num = (ct.potential(phi + h, c) - ct.potential(phi - h, c)) / (2 * h)
        assert np.allclose(num, ct.cpr_current(phi, c), rtol=0, atol=1e-9)


# ---------------------------------------------------------------- (ii) -------
def test_measured_harmonics_are_max_normalized():
    """(ii) max_phi sum_m c_m sin(m phi) = 1 -- Ic is the TRUE critical current."""
    for tau in ct.taus_available():
        mx = ct.cpr_max(ct.harmonics(tau))
        assert abs(mx - 1.0) < NORM_TOL, f"tau={tau}: max = {mx!r}"
    assert abs(ct.cpr_max(ct.C_SINUSOIDAL) - 1.0) < NORM_TOL


def test_skew_lowers_zero_phase_slope_and_keeps_one_well():
    """A2 = sum m c_m < 1 (weaker at phi=0) and U has a single minimum at 0."""
    prev = 1.0
    for tau in ct.taus_available():          # ascending transparency
        c = ct.harmonics(tau)
        a2 = ct.A2(c)
        assert 0.0 < a2 < 1.0
        assert a2 < prev, "A2 must fall monotonically with transparency"
        prev = a2
        assert ct.cpr_argmax(c) > 0.5, "forward skew: peak beyond phi = pi/2"
        mins = ct.potential_minima(c)
        assert len(mins) == 1 and abs(mins[0]) < 1e-9
    assert ct.A2(ct.C_SINUSOIDAL) == 1.0
    assert ct.A4(ct.C_SINUSOIDAL) == 1.0


# --------------------------------------------------------------- (iii) -------
def test_N_convergence_35_to_60():
    """(iii) N = 35 is converged; f01/anharm/n01 to 1e-9 relative, eps1 to 1e-3.

    eps1 gets the loose bound on purpose: it is 9 decades below f01, so its
    float64 value is ULP-conditioned (paper/decisions.md, 2026-07-27).
    """
    for tau in ct.taus_available():
        c = ct.harmonics(tau)
        f35, a35, e35 = ct.levels(EJ_GHZ, EC, c, 35)
        f60, a60, e60 = ct.levels(EJ_GHZ, EC, c, 60)
        assert f60 == pytest.approx(f35, rel=1e-9)
        assert a60 == pytest.approx(a35, rel=1e-9)
        assert e60 == pytest.approx(e35, rel=EPS1_F64_TOL)
        assert (ct.n01_element(EJ_GHZ, EC, c, 60)
                == pytest.approx(ct.n01_element(EJ_GHZ, EC, c, 35), rel=1e-9))


def test_extended_precision_confirms_float64_eps1():
    """The banded mpmath path agrees with float64 to better than 1e-3 relative.

    Every eps1 here is 1-320 Hz, i.e. 4.5-5.5 decades above the float64
    eigenvalue floor, so no row needs the mpmath value -- but it is exported,
    so it is checked.
    """
    for c in (ct.C_SINUSOIDAL, ct.harmonics(TAU)):
        f64 = ct.levels(EJ_GHZ, EC, c, N_CHARGE)[2] * 1e9
        mp = ct.eps1_mp(EJ_GHZ, EC, c, N_CHARGE, ct.MP_DPS)
        assert mp == pytest.approx(f64, rel=EPS1_F64_TOL)
        assert mp > 1e-3, "below 1 mHz the float64 comparison would be meaningless"


def test_retune_hits_the_design_f01():
    """Bisection on a monotone f01(E_J) lands on the design frequency."""
    for tau in ct.taus_available():
        c = ct.harmonics(tau)
        lam = ct.retune_EJ(EC, c, ct.F01_DESIGN_GHZ, N_CHARGE)
        assert lam > 1.0, "skewed CPR is weaker at phi = 0, so E_J must go UP"
        f01 = ct.levels(EJ_GHZ * lam, EC, c, N_CHARGE)[0]
        assert f01 == pytest.approx(ct.F01_DESIGN_GHZ, rel=1e-12)


# ---------------------------------------------------------------- (iv) -------
def test_export_is_bit_consistent_with_a_fresh_run(payload, on_disk):
    """(iv) every float in the on-disk export re-derives bit-for-bit."""
    def walk(a, b, path=""):
        assert type(a) is type(b), f"{path}: {type(a)} vs {type(b)}"
        if isinstance(a, dict):
            assert a.keys() == b.keys(), f"{path}: key sets differ"
            for k in a:
                walk(a[k], b[k], f"{path}.{k}")
        elif isinstance(a, list):
            assert len(a) == len(b), f"{path}: length {len(a)} vs {len(b)}"
            for i, (x, y) in enumerate(zip(a, b)):
                walk(x, y, f"{path}[{i}]")
        elif isinstance(a, float):
            assert a == b, f"{path}: {a!r} != {b!r}"
        else:
            assert a == b, f"{path}: {a!r} != {b!r}"

    walk(json.loads(json.dumps(payload)), on_disk)


def test_export_records_the_normalization_and_the_bitwise_limit(payload):
    nc = payload["normalization_check"]["rows"]
    for tau in ct.taus_available():
        assert nc[tau]["abs_dev_from_1"] < NORM_TOL
        assert nc[tau]["potential_minima_over_pi"] == pytest.approx([0.0],
                                                                    abs=1e-9)
    x = payload["reference_sinusoidal"]["cross_check_spectrum_py"]
    assert x["bitwise_identical"] is True
    assert x["max_abs_matrix_diff"] == 0.0


# ----------------------------------------------------------------- (v) -------
def _recompute_fixed_Ic(tau):
    c = ct.harmonics(tau)
    f01, anh, _ = ct.levels(EJ_GHZ, EC, c, N_CHARGE)
    return f01, anh, ct.n01_element(EJ_GHZ, EC, c, N_CHARGE)


def test_headline_fixed_Ic_deltas_match_export(payload):
    """(v) the tau = 0.8 fixed-Ic headline, recomputed independently."""
    ref = payload["reference_sinusoidal"]
    r = payload["fixed_Ic"][TAU]
    f01, anh, n01 = _recompute_fixed_Ic(TAU)
    assert r["f01_GHz"] == pytest.approx(f01, rel=TIGHT)
    assert r["anharm_GHz"] == pytest.approx(anh, rel=TIGHT)
    assert r["n01"] == pytest.approx(n01, rel=TIGHT)

    d = r["deltas_vs_sinusoidal_pct"]
    assert d["f01_GHz"] == pytest.approx(
        100 * (f01 - ref["f01_GHz"]) / ref["f01_GHz"], rel=TIGHT)
    assert d["anharm_GHz"] == pytest.approx(
        100 * (anh - ref["anharm_GHz"]) / ref["anharm_GHz"], rel=TIGHT)
    assert d["n01"] == pytest.approx(
        100 * (n01 - ref["n01"]) / ref["n01"], rel=TIGHT)

    # the physics, stated as bands so the test is about the effect, not a hash
    assert -14.0 < d["f01_GHz"] < -12.0, "f01 falls by ~13% at fixed Ic"
    assert -58.0 < d["anharm_GHz"] < -54.0, "|anharmonicity| roughly halves"
    assert -8.0 < d["n01"] < -6.0, "n01 falls by ~7%"
    assert 150.0 < d["eps1_mp_Hz"] < 220.0, "eps1 roughly triples"
    assert r["eps1_mp_Hz"] < 1e4, "still passes the 10 kHz design criterion"
    assert abs(r["anharm_GHz"]) >= 0.5, "still passes the 0.5 GHz criterion"


def test_headline_retuned_deltas_match_export(payload):
    """(v) retuning E_J restores f01 and n01 but NOT the anharmonicity."""
    ref = payload["reference_sinusoidal"]
    r = payload["retuned"][TAU]
    lam = ct.retune_EJ(EC, ct.harmonics(TAU), ct.F01_DESIGN_GHZ, N_CHARGE)
    assert r["lambda_EJ"] == pytest.approx(lam, rel=1e-12)
    assert abs(r["f01_residual_GHz"]) < 1e-9
    d = r["deltas_vs_sinusoidal_pct"]
    assert abs(d["f01_GHz"]) < 1e-9
    assert abs(d["n01"]) < 0.05, "n01 ~ sqrt(f01/8 E_C): restored with f01"
    assert -58.0 < d["anharm_GHz"] < -54.0, "anharmonicity does NOT tune away"
    assert d["eps1_mp_Hz"] < -85.0, "the stiffer junction suppresses eps1"


def test_sweet_spot_delta_matches_export(payload):
    """(v) D2(-2.4) with the measured harmonics, recomputed."""
    sw = payload["sweet_spot"]
    d2 = ct.D2(dome.NU_OPT, EC, ct.harmonics(TAU))
    row = sw["rows"][TAU]
    assert row["D2_multi_GHz"] == pytest.approx(d2, rel=TIGHT)
    assert sw["D2_sinusoidal_GHz"] == pytest.approx(sw["D2_dome_py_GHz"],
                                                    rel=TIGHT)
    assert row["delta_pct"] == pytest.approx(
        100 * (d2 - sw["D2_sinusoidal_GHz"]) / sw["D2_sinusoidal_GHz"],
        rel=TIGHT)
    assert -16.0 < row["delta_pct"] < -13.0, "|D2| drops ~15%"
    assert d2 < 0.0, "the apex stays a maximum of f01(nu)"
    assert row["D1_apex_multi_GHz"] == 0.0, "first-order sweet spot survives"
    assert row["Tphi_apex_gain_factor"] > 1.0, "flatter apex -> longer Tphi"


def test_downstream_scalings_match_export(payload):
    """(v) g ~ n01, t_g ~ 1/|alpha|, parity bound ~ 1/eps1."""
    ref = payload["reference_sinusoidal"]
    dn = payload["downstream"]

    g = dn["g_scaling"]
    assert g["g_delta_fixed_Ic_pct"] == pytest.approx(
        payload["fixed_Ic"][TAU]["deltas_vs_sinusoidal_pct"]["n01"], rel=TIGHT)

    gf = dn["gate_floor_scaling"]
    assert gf["t_g_stretch_fixed_Ic"] == pytest.approx(
        abs(ref["anharm_GHz"] / payload["fixed_Ic"][TAU]["anharm_GHz"]),
        rel=TIGHT)
    assert 2.2 < gf["t_g_stretch_fixed_Ic"] < 2.4
    for k, v in gf["p6_reference_ns"].items():
        assert gf["scaled_fixed_Ic_ns"][k] == pytest.approx(
            v * gf["t_g_stretch_fixed_Ic"], rel=TIGHT)

    pb = dn["parity_bound"]
    assert pb["bound_fixed_Ic_ms"] == pytest.approx(
        1e3 / (np.pi * payload["fixed_Ic"][TAU]["eps1_mp_Hz"]), rel=TIGHT)
    assert pb["bound_sinusoidal_ms"] > pb["bound_fixed_Ic_ms"] > 1.0
    # P3's ">1e3" margin does not survive the measured CPR, but ">5e2" does
    assert 500.0 < pb["bound_over_T1qp_fixed_Ic"] < 1000.0
    assert pb["bound_over_T1qp_sinusoidal"] > 1000.0


def test_design_lock_survives_but_the_window_collapses(payload):
    """The selected E_J/E_C is unchanged; every looser ratio now fails."""
    d = payload["design_criterion_robustness"]
    assert d["selected"] == d["selected_sinusoidal"] == 100
    assert d["passing_ratios"] == [100]
    assert d["passing_ratios_sinusoidal"] == [100, 150, 200, 300]
    assert d["verdict"] == "design lock unchanged"


def test_truncation_diagnostic_is_honest(payload):
    """material.json's 5 harmonics: fine at tau <= 0.8, NOT converged at 0.95."""
    tr = payload["harmonic_truncation"]
    for tau in ct.taus_available():
        assert tr[tau]["material_nH"] == 5
        assert tr[tau]["regen_matches_material_max_abs"] < 1e-15, (
            "the diagnostic generator must reproduce material.json exactly")
    nominal = tr[TAU]["material_vs_converged_pct"]
    assert abs(nominal["f01_GHz"]) < 0.5
    assert abs(nominal["n01"]) < 0.5
    assert abs(nominal["anharm_GHz"]) < 10.0
    high = tr["0.95"]["material_vs_converged_pct"]
    assert abs(high["anharm_GHz"]) > 50.0, (
        "tau=0.95 is truncation-limited -- this must stay flagged")
    assert abs(tr["0.95"]["converged"]["anharm_GHz"]) < 0.5, (
        "converged tau=0.95 would FAIL the 0.5 GHz anharmonicity criterion")


def test_provenance_declares_the_untouched_modules(payload):
    p = payload["provenance"]
    assert p["deterministic"] is True
    assert p["rng_seed"] is None
    for mod in ("qubit/spectrum.py", "qubit/inputs.py", "qubit/dome.py"):
        assert mod in p["unmodified"]
    assert "material.json:cpr.harmonics_by_tau" in p["reads"].values()
