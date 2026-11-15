"""M6 deliverable P4 -- 1/f gate noise via Ramsey/echo filter functions.

Validates ``qubit/noise_1f.py`` against (a) the analytic filter-function
integrals, (b) the exact Gaussian linear+quadratic characteristic function,
and (c) the quasistatic reference rows in qubit_targets.json:gate_noise.

Fully deterministic -- P4 uses no RNG.  Every dome number comes from
``qubit.dome`` (which reads material.json:sc_dome); the only literals here are
the nu points and tolerance brackets stated in the test names/docstrings.

CLAIM DISCIPLINE: every T2 asserted below is parametric in the free 1/f
amplitude A.  No absolute TBG coherence value is asserted anywhere.
"""
import json
import pathlib

import numpy as np
import pytest
from scipy.integrate import quad

import qubit
from qubit import dome, noise_1f as nf

ROOT = pathlib.Path(__file__).resolve().parents[1]

NU_APEX = -2.4          # dome apex (material.json:sc_dome.nu_opt)
NU_OFF = -2.5           # the brief's off-apex comparison point
A_REF = 1e-4            # the reference 1/f amplitude of qubit_targets.json
W_IR = nf.W_IR_DEFAULT  # 2 pi * 1 Hz

# --- stated brackets ---------------------------------------------------------
# (i)  Ramsey/quasistatic ratio is 1/sqrt(I_R) with I_R = ln(1/(w_ir T2)) + 0.923.
#      At f_ir = 1 Hz and T2* ~ 20 ns this is sqrt(16.7) = 4.09, so the honest
#      bracket is a factor of 5, not 3; a factor of 3 would require
#      f_ir > ~10 kHz.  Both the bracket and the exact sqrt-log identity are
#      asserted below.
QS_RATIO_BRACKET = (2.0, 5.0)
SQRT_LOG_TOL = 0.01
# (iv) measured power-law exponents d ln T2 / d ln A on the production grid
EXP_OFF_BRACKET = (-1.10, -0.99)
EXP_APEX_BRACKET = (-2.40, -2.00)


# ================================================================== fixtures ==
@pytest.fixture(scope="session")
def EC():
    return qubit.design_point()["EC_GHz"]


@pytest.fixture(scope="session")
def targets():
    with open(ROOT / "qubit_targets.json", encoding="utf-8") as fh:
        return json.load(fh)["gate_noise"]


@pytest.fixture(scope="session")
def apex(EC):
    """(D1, D2) at the dome apex -- from qubit.dome, never typed."""
    return dome.D1(NU_APEX, EC), dome.D2(NU_APEX, EC)


@pytest.fixture(scope="session")
def off(EC):
    return dome.D1(NU_OFF, EC), dome.D2(NU_OFF, EC)


@pytest.fixture(scope="session")
def maps(EC):
    """Production maps from out/m6/p4_t2_maps.npz, or a coarse fallback."""
    fp = ROOT / "out" / "m6" / "p4_t2_maps.npz"
    if fp.exists():
        z = np.load(fp)
        return {k: z[k] for k in z.files}
    nus = np.array([-2.75, -2.6, -2.5, NU_APEX, -2.3, -2.15])
    As = np.logspace(-5, -3, 5)
    d1 = np.array([dome.D1(n, EC) for n in nus])
    d2 = np.array([dome.D2(n, EC) for n in nus])
    tr = np.array([[nf.T2(a, b, A, "ramsey") for A in As] for a, b in zip(d1, d2)])
    te = np.array([[nf.T2(a, b, A, "echo") for A in As] for a, b in zip(d1, d2)])
    return dict(nu=nus, A=As, D1_GHz=d1, D2_GHz=d2, T2_ramsey=tr, T2_echo=te,
                gain_echo_over_ramsey=te / tr)


# ========================================== filter functions / conventions ====
class TestFilterFunctions:
    """The analytic filter-function integrals, against direct quadrature."""

    @pytest.mark.parametrize("t", [1e-10, 1e-8, 1e-5, 1e-3])
    def test_ramsey_log_integral_matches_quadrature(self, t):
        num = quad(lambda u: np.sin(u) ** 2 / u ** 3, W_IR * t / 2, np.inf,
                   limit=800)[0]
        assert abs(nf.ramsey_log_integral(t) / num - 1) < 1e-7

    @pytest.mark.parametrize("t", [1e-10, 1e-8, 1e-5, 1e-3])
    def test_echo_log_integral_matches_quadrature_and_ln2(self, t):
        num = quad(lambda u: np.sin(u) ** 4 / u ** 3, W_IR * t / 4, np.inf,
                   limit=800)[0]
        assert abs(nf.echo_log_integral(t) / num - 1) < 1e-6
        # the standard ln 2 echo factor, recovered for w_ir t << 1
        assert abs(nf.echo_log_integral(t) - nf.LN2) < 1e-5

    def test_ramsey_O1_constant_is_three_halves_minus_gamma(self):
        assert abs(nf.RAMSEY_O1 - (1.5 - np.euler_gamma)) < 1e-15
        # exact integral -> ln(1/(w_ir t)) + 3/2 - gamma_E
        for t in (1e-9, 1e-6):
            assert abs(nf.ramsey_log_integral(t)
                       - nf.ramsey_log_asymptotic(t)) < 1e-7

    def test_first_order_closed_form_matches_exact_gaussian(self, off):
        """chi_1 closed form vs the exact linear-Gaussian characteristic fn."""
        D1, _ = off
        for t in (2e-8, 5e-8):
            for seq in ("ramsey", "echo"):
                cf = float(nf.chi_first_order(t, D1, A_REF, seq))
                ex = nf.chi_exact_combined(t, D1, 0.0, A_REF, seq, M=256)
                assert abs(cf / ex - 1) < 2e-3, (seq, t, cf, ex)


# ====================================================================== (i) ===
class TestOffApexFirstOrder:
    """(i) Off-apex first-order T2* vs the quasistatic reference."""

    def test_dome_inputs_are_read_not_typed(self, off, targets):
        D1, _ = off
        assert abs(D1 - targets["D1_off_nu_m2p5_GHz"]) < 0.3

    def test_quasistatic_row_reproduced(self, off, targets):
        D1, _ = off
        qs = nf.quasistatic_first_order(D1, A_REF) * 1e6
        assert abs(qs / targets["Tphi_off_us"]["1e-04"] - 1) < 0.03

    def test_ramsey_within_stated_sqrt_log_bracket(self, off):
        """Stated bracket: quasistatic / T2* lies in (2, 5)."""
        D1, D2 = off
        t2 = nf.T2(D1, D2, A_REF, "ramsey")
        qs = nf.quasistatic_first_order(D1, A_REF)
        r = qs / t2
        assert QS_RATIO_BRACKET[0] < r < QS_RATIO_BRACKET[1], r

    def test_ratio_is_exactly_sqrt_of_the_log_factor(self, off):
        """The bracket above is not a fudge: qs/T2* = sqrt(I_R(T2*)) exactly."""
        D1, D2 = off
        for A in (1e-5, 1e-4, 1e-3):
            t2 = nf.T2(D1, D2, A, "ramsey")
            qs = nf.quasistatic_first_order(D1, A)
            assert abs(qs / t2 / np.sqrt(nf.ramsey_log_integral(t2)) - 1) \
                < SQRT_LOG_TOL

    def test_echo_ratio_is_exactly_sqrt_ln2(self, off):
        D1, D2 = off
        for A in (1e-5, 1e-4, 1e-3):
            t2 = nf.T2(D1, D2, A, "echo")
            qs = nf.quasistatic_first_order(D1, A)
            assert abs(qs / t2 / np.sqrt(nf.LN2) - 1) < SQRT_LOG_TOL


# ===================================================================== (ii) ===
class TestEchoBeatsRamsey:
    """(ii) Echo longer than Ramsey everywhere off-apex (and never shorter)."""

    def test_gain_above_one_everywhere_off_apex(self, maps):
        g = maps["gain_echo_over_ramsey"]
        off_apex = np.abs(maps["D1_GHz"]) > 1e-9
        sub = g[off_apex]
        assert np.all(np.isfinite(sub))
        assert sub.min() > 1.0, sub.min()

    def test_gain_never_below_one_anywhere(self, maps):
        g = maps["gain_echo_over_ramsey"]
        assert np.nanmin(g) > 1.0, np.nanmin(g)

    def test_off_apex_gain_is_the_sqrt_log_ratio(self, off):
        """Off-apex the gain is sqrt(I_R/ln2) -- a factor of ~4-5, not ~2."""
        D1, D2 = off
        tr = nf.T2(D1, D2, A_REF, "ramsey")
        te = nf.T2(D1, D2, A_REF, "echo")
        assert abs(te / tr
                   - np.sqrt(nf.ramsey_log_integral(tr) / nf.LN2)) < 0.05


# ==================================================================== (iii) ===
class TestApexSecondOrderOnly:
    """(iii) At the apex the first-order channel vanishes identically."""

    def test_D1_apex_is_exactly_zero(self, apex):
        assert apex[0] == 0.0

    def test_chi_first_order_vanishes_at_apex(self, apex):
        D1, _ = apex
        for t in (1e-6, 1e-4, 1e-2):
            for seq in ("ramsey", "echo"):
                assert float(nf.chi_first_order(t, D1, A_REF, seq)) == 0.0

    def test_T2_at_apex_equals_second_order_only(self, apex):
        D1, D2 = apex
        for seq in ("ramsey", "echo"):
            full = nf.T2(D1, D2, A_REF, seq)
            second = nf.T2_second_order_only(D2, A_REF, seq)
            assert abs(full / second - 1) < 1e-6

    def test_second_order_scaling_vs_quasistatic_is_O1_log(self, apex, targets):
        """T_phi ~ 1/(|d2w/dnu2| A^2) with an O(1)-log prefactor in [0.15, 0.7],
        and the quasistatic reference is longer by a factor 3..12."""
        _, D2 = apex
        d2w = abs(nf.d2w_dnu2(D2))
        for A in (1e-5, 1e-4, 1e-3):
            t2 = nf.T2(0.0, D2, A, "ramsey")
            p = t2 * d2w * A ** 2
            assert 0.15 < p < 0.70, (A, p)
        t2 = nf.T2(0.0, D2, A_REF, "ramsey")
        qs = nf.quasistatic_second_order(D2, A_REF)
        assert abs(qs * 1e6 / targets["Tphi_apex_us"]["1e-04"] - 1) < 0.03
        assert 3.0 < qs / t2 < 12.0, qs / t2


# ===================================================================== (iv) ===
class TestPowerLaws:
    """(iv) T2 ~ 1/A off-apex, ~1/A^2 (log-corrected) at the apex."""

    @staticmethod
    def _slopes(D1, D2, seq, As):
        t = np.array([nf.T2(D1, D2, A, seq) for A in As])
        return np.diff(np.log(t)) / np.diff(np.log(As))

    @pytest.mark.parametrize("seq", ["ramsey", "echo"])
    def test_off_apex_exponent_is_minus_one(self, off, seq):
        s = self._slopes(*off, seq, np.logspace(-5, -3, 5))
        assert EXP_OFF_BRACKET[0] < s.min() and s.max() < EXP_OFF_BRACKET[1], s

    @pytest.mark.parametrize("seq", ["ramsey", "echo"])
    def test_apex_exponent_is_minus_two_like(self, apex, seq):
        s = self._slopes(*apex, seq, np.logspace(-5, -3, 5))
        assert EXP_APEX_BRACKET[0] < s.min() and s.max() < EXP_APEX_BRACKET[1], s

    def test_apex_exponent_is_steeper_than_off_apex(self, apex, off):
        As = np.logspace(-5, -3, 5)
        assert (self._slopes(*apex, "ramsey", As).max()
                < self._slopes(*off, "ramsey", As).min())


# ====================================================================== (v) ===
class TestMapsMonotoneInA:
    """(v) T2 falls monotonically with A at every nu, both sequences."""

    @pytest.mark.parametrize("key", ["T2_ramsey", "T2_echo"])
    def test_monotone_decreasing_in_A(self, maps, key):
        m = maps[key]
        assert m.shape == (len(maps["nu"]), len(maps["A"]))
        for i in range(m.shape[0]):
            row = m[i]
            ok = np.isfinite(row)
            assert ok.sum() >= 2
            d = np.diff(row[ok])
            assert np.all(d < 0), (maps["nu"][i], row)

    def test_nan_only_where_T2_exceeds_the_acquisition_window(self, maps):
        """NaNs are the honest 'outside the 1/f model window' flag, not gaps."""
        for key in ("T2_ramsey", "T2_echo"):
            bad = ~np.isfinite(maps[key])
            if bad.any():
                # NaNs must sit at the small-A end of a row (longest T2)
                assert np.all(np.argmax(np.isfinite(maps[key]), axis=1)[
                    bad.any(axis=1)] > 0)


# ==================================== numerics: convergence + approximations ==
class TestNumerics:

    def test_second_order_determinant_M_convergence(self, apex):
        _, D2 = apex
        vals = [nf.chi_second_order(1e-4, D2, A_REF, "ramsey", M=m)
                for m in (48, 64, 96, 128)]
        assert max(abs(v / vals[-1] - 1) for v in vals) < 0.01, vals
        assert vals == sorted(vals)              # monotone convergence from below

    def test_T2_is_insensitive_to_the_time_grid(self, apex):
        """The quantity that matters is T2, not chi in the far tail."""
        _, D2 = apex
        a = nf.T2(0.0, D2, A_REF, "ramsey", M=64)
        b = nf.T2(0.0, D2, A_REF, "ramsey", M=128)
        assert abs(a / b - 1) < 3e-3, (a, b)

    def test_static_limit_is_a_lower_bound_on_the_exact_determinant(self, apex):
        _, D2 = apex
        for t in (2e-5, 1e-4, 1e-3):
            st = float(nf.chi_second_order_static(t, D2, A_REF))
            ex = nf.chi_second_order(t, D2, A_REF, "ramsey")
            assert st < ex, (t, st, ex)

    def test_long_time_limit_approaches_the_szego_asymptote(self, apex):
        """chi_2 -> (pi/4)|d2w|A^2 t: the linear-in-t 'pi D2 A^2 t' scaling."""
        _, D2 = apex
        ts = np.array([1e-4, 1e-3, 1e-2])
        r = [nf.chi_second_order(t, D2, A_REF, "ramsey")
             / float(nf.chi_second_order_asymptotic(t, D2, A_REF)) for t in ts]
        assert all(x > 1.0 for x in r), r          # exact exceeds the asymptote
        assert r[0] > r[1] > r[2], r               # and converges down to it
        assert r[-1] < 2.0, r

    def test_echo_gain_collapses_at_long_times_at_the_apex(self, apex):
        """The asymptote is sequence-independent: echo buys nothing at second
        order once t >> T2 (quantified, because it is a paper claim)."""
        _, D2 = apex
        r = nf.chi_second_order(1e-2, D2, A_REF, "echo") \
            / nf.chi_second_order(1e-2, D2, A_REF, "ramsey")
        assert abs(r - 1.0) < 0.01, r

    def test_independent_rate_addition_is_exact_where_one_order_dominates(
            self, apex, off):
        """chi_1 + chi_2 vs the exact linear+quadratic Gaussian result."""
        for D1, D2 in (apex, off):
            for A in (1e-5, 1e-4):
                t = nf.T2(D1, D2, A, "ramsey")
                add = nf.chi_total(t, D1, D2, A, "ramsey")
                ex = nf.chi_exact_combined(t, D1, D2, A, "ramsey")
                assert abs(add / ex - 1) < 2e-3, (D1, A, add, ex)

    def test_independent_rate_addition_is_conservative_in_the_crossover(
            self, EC):
        """Near (not at) the apex and at large A the resolvent suppresses the
        first-order term, so plain addition OVER-estimates chi -- i.e. it
        under-reports T2.  Measured, not assumed: up to +173% in T2 at
        A = 1e-2.  It never errs the other way."""
        worst = 0.0
        for nu in (-2.45, -2.425, -2.41):
            D1, D2 = dome.D1(nu, EC), dome.D2(nu, EC)
            for A in (1e-3, 1e-2):
                t = nf.T2(D1, D2, A, "ramsey")
                add = nf.chi_total(t, D1, D2, A, "ramsey")
                ex = nf.chi_exact_combined(t, D1, D2, A, "ramsey")
                assert add >= ex * (1 - 1e-9), (nu, A, add, ex)
                worst = max(worst, add / ex - 1)
        assert worst > 0.05, worst      # the effect is real, not round-off

    def test_additivity_error_is_negligible_at_the_reference_amplitude(
            self, maps):
        """A <= 1e-4 (the qubit_targets.json reference row): < 0.1% in T2."""
        if "additivity_error_ramsey" not in maps:
            pytest.skip("needs the production npz")
        e, A = maps["additivity_error_ramsey"], maps["A"]
        sub = e[:, A <= 1e-4]
        assert np.nanmax(np.abs(sub)) < 1e-3, np.nanmax(np.abs(sub))

    def test_w_ir_sensitivity_is_only_logarithmic(self, off):
        """Six decades of IR cutoff move T2* by well under a factor of 2."""
        D1, D2 = off
        s = nf.f_ir_sensitivity(D1, D2, A_REF, "ramsey")
        v = np.array([s[k] for k in s])
        assert np.all(np.diff(v) > 0)              # larger f_ir -> longer T2*
        assert v.max() / v.min() < 2.0, v.max() / v.min()

    def test_echo_first_order_is_w_ir_independent(self, off):
        D1, _ = off
        a = nf.T2_first_order_only(D1, A_REF, "echo", 2 * np.pi * 1e-3)
        b = nf.T2_first_order_only(D1, A_REF, "echo", 2 * np.pi * 1e3)
        assert abs(a / b - 1) < 1e-4

    def test_covariance_rejects_a_grid_coarser_than_the_IR_cutoff(self):
        with pytest.raises(ValueError):
            nf._kernel(8, 1.0, 1e-4, W_IR)


# ======================================================================= I/O ==
@pytest.fixture(scope="session")
def doc():
    fp = ROOT / "out" / "m6" / "p4_t2_maps.json"
    if not fp.exists():
        pytest.skip("run `python -m qubit.noise_1f` to build the P4 export")
    with open(fp, encoding="utf-8") as fh:
        return json.load(fh), fp


class TestExport:
    """out/m6/p4_t2_maps.{json,npz} -- run `python -m qubit.noise_1f` first."""

    def test_json_is_small(self, doc):
        assert doc[1].stat().st_size < 400_000

    def test_params_and_literature_note_present(self, doc):
        d = doc[0]
        p = d["params"]
        assert p["f_ir_Hz"] == 1.0
        assert "A^2/|w|" in p["noise_convention"]
        assert "free parameter" in p["A_units"]
        assert "chi(T2) = 1" in p["T2_definition"]
        assert "FREE PARAMETER" in d["literature_note"]
        assert "no absolute" in d["claim_discipline"].lower()

    def test_additivity_diagnostic_reported(self, doc):
        a = doc[0]["additivity_diagnostic"]
        assert a["fraction_within_1pct"] > 0.75
        assert a["largest_A_with_max_error_below_1pct"] >= 1e-4
        assert a["max_abs_error_all"] > 0.05      # the caveat is real

    def test_reference_rows_present_and_consistent(self, doc):
        rows = doc[0]["reference_rows"]
        assert {(r["nu"], r["A"]) for r in rows} == {
            (nu, A) for nu in (-2.4, -2.5) for A in (1e-4, 1e-3)}
        for r in rows:
            assert abs(r["quasistatic_us"] / r["quasistatic_reference_us"] - 1) \
                < 0.03
            assert r["T2_echo_us"] > r["T2_ramsey_us"] > 0

    def test_grid_shapes(self, doc):
        d = doc[0]
        nn, na = len(d["grids"]["nu"]), len(d["grids"]["A"])
        assert na >= 25 and nn >= 25
        assert d["grids"]["nu"][0] == -2.85 and d["grids"]["nu"][-1] == -1.95
        assert d["grids"]["A"][0] == 1e-6 and d["grids"]["A"][-1] == 1e-2
        for k in ("T2_ramsey", "T2_echo", "gain_echo_over_ramsey"):
            assert len(d["maps_s"][k]) == nn
            assert all(len(r) == na for r in d["maps_s"][k])
