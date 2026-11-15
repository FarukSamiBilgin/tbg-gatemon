"""M6 production task P3 -- full quasiparticle kernel (qubit/qp_kernel.py).

Four required acceptances (task P3.3), plus the cross-checks that make each of
them meaningful:

  (i)   small-frequency limit -- the kernel collapses onto the milestone-1
        reference prefactor when hbar*w/Delta -> 0 (and, for the *total* rate,
        when E_J/E_C -> infinity as well, since the reference form also uses
        the harmonic matrix element and the harmonic hbar*w_01 identity);
  (ii)  T1 scales exactly as 1/x_qp;
  (iii) the 20 mK equilibrium quasiparticle floor lies far below the bottom of
        the 1e-8..1e-5 parametric band -- computed, then asserted;
  (iv)  the pair-breaking filling boundary matches an independent analytic
        inversion of the material.json dome formula.

Everything here is deterministic (adaptive quadrature + exact
diagonalisation); no RNG is consumed anywhere in P3.

CLAIM DISCIPLINE: every T1 asserted below is parametric in (x_qp, T_qp).
`test_no_absolute_coherence_claim` enforces that the exported artefact never
carries a bare T1 that is not keyed by an x_qp value.
"""
import json

import numpy as np
import pytest

from qubit import dome, inputs, qp, qp_kernel, spectrum

# --- values quoted by M6_BRIEF.md Sec. 2 / the P3 task statement -------------
W_DESIGN = 0.5523            # hbar*w_q/Delta, the marginality flag
X_BAND = (1e-8, 1e-5)        # parametric x_qp band
T_FLOOR_K = 0.020            # required thermal-floor row
F_DRIVE_GHZ = 145.0          # drive-cavity tone under audit


@pytest.fixture(scope="session")
def targets():
    with open(inputs.ROOT / "qubit_targets.json", encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="session")
def report():
    """The exported P3 artefact, recomputed in-process."""
    return qp_kernel.build_report()


@pytest.fixture(scope="session")
def EC():
    return inputs.EJ_GHZ / 100


@pytest.fixture(scope="session")
def f01(EC):
    return float(spectrum.levels(inputs.EJ_GHZ, EC, inputs.N_CHARGE)[0])


def rel(a, b):
    return abs(a - b) / abs(b)


# =========================================================== (i) small-freq ===
class TestSmallFrequencyLimit:
    """The kernel must reduce to the reference prefactor for hbar*w << Delta."""

    @pytest.mark.parametrize("w", [1e-3, 1e-4, 1e-5])
    def test_kernel_shape_ratio_goes_to_unity(self, w):
        """S_qp_full/S_qp_smallfreq -> 1 as w -> 0, with k_B T_qp << hbar*w
        (the quasiparticles must sit within hbar*w of the gap edge for the
        small-frequency expansion to be the relevant limit at all)."""
        T_qp = 1e-4 * w * qp_kernel.delta_J() / inputs.KB
        r = qp_kernel.kernel_shape_ratio(w, T_qp)
        assert abs(r - 1.0) < 1e-3, (w, r)

    @pytest.mark.parametrize("w", [1e-3, 1e-2, 1e-1, 0.5523, 1.0, 3.0])
    def test_quadrature_matches_analytic_T0_form(self, w):
        """sqrt(1 + w/2) is the closed-form T_qp -> 0 limit of the kernel;
        the quadrature must reproduce it over the whole w range, not only
        where the small-frequency expansion holds.  "T_qp -> 0" means
        k_B T_qp << hbar*omega, so the stand-in temperature is scaled with w
        (the leading finite-T_qp correction is O(k_B T_qp/hbar w))."""
        T_qp = 1e-9 * w * qp_kernel.delta_J() / inputs.KB
        r = qp_kernel.kernel_shape_ratio(w, T_qp)
        assert rel(r, qp_kernel.kernel_shape_ratio_T0(w)) < 1e-6, (w, r)

    def test_S_qp_reduces_to_the_smallfreq_form(self):
        f_small = 1e-4 * F_DRIVE_GHZ          # w = 5e-7, deep in the limit
        T_qp = qp_kernel.T_QP_ZERO_K
        full = qp_kernel.S_qp(f_small, 1e-6, T_qp)
        approx = qp_kernel.S_qp_smallfreq(f_small, 1e-6)
        assert rel(full, approx) < 1e-4, (full, approx)

    @pytest.mark.parametrize("ratio,w_target,tol",
                             [(1e4, 1e-3, 5e-3), (1e6, 1e-5, 5e-4)])
    def test_total_rate_recovers_reference_prefactor(self, ratio, w_target,
                                                     tol):
        """Full-kernel Gamma_10 / qp.qp_prefactor -> 1 in the joint limit
        hbar*w_q/Delta -> 0 AND E_J/E_C -> infinity.  Both limits are needed:
        the reference form additionally assumes the harmonic matrix element
        (1/4)sqrt(2E_C/E_J) and the harmonic identity hbar*w_01 =
        sqrt(8 E_J E_C), neither of which involves w."""
        # pick E_J so that hbar*sqrt(8 E_J E_C)/Delta = w_target at this ratio
        f_target = w_target * qp_kernel.delta_J() / inputs.HBAR / (2 * np.pi)
        EJ = f_target / 1e9 * np.sqrt(ratio / 8.0)
        EC = EJ / ratio
        N = 80
        f01 = float(spectrum.levels(EJ, EC, N)[0])
        assert rel(qp_kernel.marginality(f01), w_target) < 0.05
        d = qp_kernel.ratio_vs_smallfreq(qp_kernel.T_QP_ZERO_K, f_GHz=f01,
                                         EJ_GHz=EJ, EC_GHz=EC, N=N)
        assert abs(d["total"] - 1.0) < tol, d

    def test_design_point_marginality_is_the_brief_flag(self, f01, targets):
        w = qp_kernel.marginality(f01)
        assert abs(w - W_DESIGN) < 5e-4
        assert rel(w, targets["quasiparticle"]["marginality_flag"]) < 1e-6

    def test_kernel_is_13pct_larger_at_the_design_point(self, report):
        """The measured consequence of the marginality flag."""
        k = report["kernel"]["ratio_vs_smallfreq_decomposition"]
        assert rel(k["kernel_shape"], np.sqrt(1 + W_DESIGN / 2)) < 1e-4
        assert 1.12 < k["kernel_shape"] < 1.14

    def test_ratio_decomposition_is_exact(self, report):
        d = report["kernel"]["ratio_vs_smallfreq_decomposition"]
        assert rel(d["product_check"], d["total"]) < 1e-12


# ============================================================ matrix element ==
class TestMatrixElements:
    """|<0|sin(phi/2)|1>|^2 computed two independent ways."""

    def test_fourier_and_phasegrid_agree(self, report):
        m = report["kernel"]["matrix_elements"]
        assert rel(m["sin_half_01_sq"], m["sin_half_01_sq_phasegrid"]) < 1e-12
        assert rel(m["cos_half_00_sq"], m["cos_half_00_sq_phasegrid"]) < 1e-12

    def test_close_to_harmonic_value(self, report):
        m = report["kernel"]["matrix_elements"]
        assert rel(m["sin_half_01_sq"], m["sin_half_01_sq_harmonic"]) < 1e-3

    def test_charge_truncation_converged(self, EC):
        a = qp_kernel.matrix_elements(inputs.EJ_GHZ, EC, N=35)
        b = qp_kernel.matrix_elements(inputs.EJ_GHZ, EC, N=60)
        assert rel(a["sin_half_01_sq"], b["sin_half_01_sq"]) < 1e-12
        assert rel(a["cos_half_00_sq"], b["cos_half_00_sq"]) < 1e-12

    def test_cos_half_diagonal_is_near_unity(self, report):
        """|<0|cos(phi/2)|0>|^2 ~ 1 - phi_zpf^2/4: the parity channel is
        NOT matrix-element suppressed, unlike relaxation."""
        assert 0.9 < report["kernel"]["matrix_elements"]["cos_half_00_sq"] < 1.0


# ================================================================ (ii) 1/x ====
class TestT1ScalesInverselyWithXqp:

    @pytest.mark.parametrize("T_qp", [0.020, 0.100, 0.200])
    def test_decade_ladder(self, T_qp, f01):
        t = [qp_kernel.T1_full(x, T_qp, f_GHz=f01)
             for x in (1e-8, 1e-7, 1e-6, 1e-5)]
        assert np.allclose(np.array(t[:-1]) / np.array(t[1:]), 10.0, rtol=1e-12)

    def test_exported_table_is_inverse_linear(self, report):
        for row in report["kernel"]["T1_us_vs_xqp"].values():
            t = [row[k] for k in ("1e-08", "1e-07", "1e-06", "1e-05")]
            assert np.allclose(np.array(t[:-1]) / np.array(t[1:]), 10.0,
                               rtol=1e-12)

    def test_full_kernel_shortens_T1_vs_reference(self, report, targets):
        full = report["kernel"]["T1_us_vs_xqp"]["T_qp->0"]
        for k in ("1e-08", "1e-07", "1e-06", "1e-05"):
            ref = targets["quasiparticle"]["T1_us"][k]
            assert full[k] < ref
            assert rel(full[k], ref / report["kernel"]["ratio_vs_smallfreq"]) \
                < 1e-9

    def test_T_qp_dependence_is_monotone_and_weak(self, report):
        d = report["kernel"]["T_qp_dependence"]
        r = d["kernel_shape_ratio"]
        assert all(a > b for a, b in zip(r, r[1:]))     # falls with T_qp
        assert (max(r) - min(r)) / max(r) < 0.05        # by less than 5%


# ========================================================= (iii) thermal floor =
class TestThermalFloor:

    def test_20mK_floor_is_far_below_the_band(self, report):
        x = report["thermal_floor"]["x_qp_th"]["0.020"]
        assert x == pytest.approx(qp_kernel.x_qp_thermal(T_FLOOR_K), rel=1e-12)
        assert x < 1e-8 * 1e-30, x       # 70 decades of headroom, measured

    @pytest.mark.parametrize("T,lo,hi", [(0.020, 1e-79, 1e-76),
                                         (0.100, 1e-17, 1e-15),
                                         (0.200, 1e-9, 1e-7)])
    def test_floor_rows(self, T, lo, hi):
        assert lo < qp_kernel.x_qp_thermal(T) < hi

    def test_band_is_entirely_non_equilibrium_at_the_operating_point(self):
        """100 mK is the operating temperature (material.json:cpr.model);
        the floor there is 8 decades below the bottom of the band."""
        assert qp_kernel.x_qp_thermal(inputs.T_OP_K) / X_BAND[0] < 1e-7

    def test_floor_reaches_the_band_edge_only_near_200mK(self):
        assert qp_kernel.x_qp_thermal(0.200) > X_BAND[0]
        assert qp_kernel.x_qp_thermal(0.150) < X_BAND[0]

    def test_asymptotic_formula_matches_the_kernel_normalisation(self, report):
        """x_qp_th is the k_B T << Delta limit of this module's own
        normalisation integral -- not an independent formula.  The residual is
        the size of that approximation."""
        a = report["thermal_floor"]["x_qp_th"]
        b = report["thermal_floor"]["x_qp_th_numeric"]
        assert rel(a["0.020"], b["0.020"]) < 3e-3
        assert rel(a["0.100"], b["0.100"]) < 2e-2
        assert rel(a["0.200"], b["0.200"]) < 3e-2

    def test_Delta_over_kB(self, report):
        want = qp_kernel.delta_J() / inputs.KB
        assert rel(report["thermal_floor"]["Delta_over_kB_K"], want) < 1e-12


# ===================================================== (iv) pair breaking =====
def _analytic_boundary(f_GHz, T=inputs.T_OP_K):
    """Independent inversion of material.json:sc_dome.Delta_form.

    Delta(nu) = 1.764 k_B Tc(nu) tanh(1.74 sqrt(Tc/T - 1)) = h f / 2
    is inverted for Tc by fixed-point iteration on the tanh (which is
    saturated, so this converges in a handful of steps), then
    Tc(nu) = Tc_max [1 - ((nu - nu_opt)/halfwidth)^2] is inverted in closed
    form.  Shares no code path with `qp_kernel.pair_breaking_boundary`, which
    uses Brent bisection on nu.
    """
    D_crit = inputs.H * f_GHz * 1e9 / 2.0 / inputs.E * 1e3        # meV
    Tc = D_crit / (inputs.BCS_GAP_RATIO * inputs.KB_MEV_PER_K)    # tanh -> 1
    for _ in range(60):
        Tc = D_crit / (inputs.BCS_GAP_RATIO * inputs.KB_MEV_PER_K
                       * np.tanh(inputs.BCS_TANH_COEF
                                 * np.sqrt(Tc / T - 1.0)))
    if Tc > inputs.TC_MAX_K:
        return None
    d = inputs.NU_HALFWIDTH * np.sqrt(1.0 - Tc / inputs.TC_MAX_K)
    return inputs.NU_OPT - d, inputs.NU_OPT + d


class TestPairBreakingAudit:

    def test_threshold(self, report):
        want = 2 * qp_kernel.delta_J() / inputs.H / 1e9
        assert rel(report["pair_breaking_audit"]["two_Delta_over_h_GHz"],
                   want) < 1e-12
        assert abs(want - 147.0) < 0.05          # M6_BRIEF.md Sec. 1

    def test_drive_tone_sits_1p4pct_below_threshold(self, report):
        row = report["pair_breaking_audit"]["drive_cavity"]
        assert row["f_GHz"] == F_DRIVE_GHZ
        assert 1.0 < row["pct_below_threshold"] < 1.8
        assert row["margin_2Delta_over_hf"] < 1.02

    def test_control_and_readout_have_3x_margin(self, report):
        a = report["pair_breaking_audit"]
        assert a["control_f01"]["margin_2Delta_over_hf"] > 3.5
        assert a["readout_sample"]["margin_2Delta_over_hf"] > 3.0

    @pytest.mark.parametrize("tone", ["control_f01", "readout_sample",
                                      "drive_cavity"])
    def test_boundary_matches_analytic_inversion(self, report, tone):
        got = report["pair_breaking_audit"][tone]["sub_gap_filling_window"]
        want = _analytic_boundary(report["pair_breaking_audit"][tone]["f_GHz"])
        assert want is not None
        assert abs(got[0] - want[0]) < 1e-9, (got, want)
        assert abs(got[1] - want[1]) < 1e-9, (got, want)

    def test_boundary_is_a_root_of_the_dome_form(self, report):
        lo, hi = report["pair_breaking_audit"]["drive_cavity"][
            "sub_gap_filling_window"]
        D_crit = inputs.H * F_DRIVE_GHZ * 1e9 / 2.0 / inputs.E * 1e3
        assert abs(dome.Delta_of(lo) - D_crit) < 1e-12
        assert abs(dome.Delta_of(hi) - D_crit) < 1e-12

    def test_drive_window_is_narrow(self, report):
        """The design constraint: keeping the 145 GHz drive sub-gap confines
        the gate to |nu + 2.4| < ~0.06, i.e. a small fraction of the +-0.5
        dome halfwidth that the tunability claim spans."""
        row = report["pair_breaking_audit"]["drive_cavity"]
        assert 0.055 < row["window_halfwidth_nu"] < 0.062
        assert row["window_halfwidth_nu"] / inputs.NU_HALFWIDTH < 0.15
        assert row["f01_span_inside_window_GHz"] < 0.5

    def test_145GHz_outside_window_is_pair_breaking(self, report):
        lo, hi = report["pair_breaking_audit"]["drive_cavity"][
            "sub_gap_filling_window"]
        for nu in (lo - 0.01, hi + 0.01, -2.6, -2.2):
            assert 2 * dome.Delta_of(nu) * 1e-3 * inputs.E / inputs.H \
                < F_DRIVE_GHZ * 1e9
        for nu in (inputs.NU_OPT, lo + 0.005, hi - 0.005):
            assert 2 * dome.Delta_of(nu) * 1e-3 * inputs.E / inputs.H \
                > F_DRIVE_GHZ * 1e9


# ========================================================= parity dephasing ===
class TestParityDephasing:

    def test_deep_in_motional_narrowing_everywhere_in_the_band(self, report):
        for row in report["parity"]["rate"].values():
            assert row["motional_narrowing_ratio"] > 1e3, row

    def test_Tphi_parity_dwarfs_the_qp_T1(self, report):
        for row in report["parity"]["rate"].values():
            assert row["Tphi_parity_over_T1qp"] > 1e5, row

    def test_frozen_parity_upper_bound_still_negligible(self, report):
        """Bound that holds for ANY switching rate: 1/(pi eps1) = 8.5 ms."""
        p = report["parity"]
        assert rel(p["Tphi_parity_frozen_worst_case_s"],
                   1.0 / (np.pi * qp_kernel.EPS1_HZ)) < 1e-12
        assert p["Tphi_parity_frozen_worst_case_s"] > 5e-3
        assert p["Tphi_frozen_over_T1qp_at_1e6"] > 1e3

    def test_cutoff_dependence_is_only_logarithmic(self, report):
        s = report["parity"]["Gamma_parity_cutoff_sensitivity"]
        v = [s[k] for k in ("1e-09", "1e-07", "1e-05", "1e-03", "1e-01")]
        assert all(a > b for a, b in zip(v, v[1:]))   # falls with the cutoff
        assert v[0] / v[-1] < 60.0                    # over 8 decades of cutoff

    def test_eps1_traced_to_targets(self, targets):
        assert qp_kernel.EPS1_HZ == targets["design"]["eps1_Hz"]


# ================================================= numerics / provenance ======
class TestNumericsAndProvenance:

    def test_quadrature_tolerance_is_immaterial(self, f01):
        w = qp_kernel.marginality(f01)
        a = qp_kernel.kernel_shape_ratio(w, 0.020, epsrel=1e-8)
        b = qp_kernel.kernel_shape_ratio(w, 0.020, epsrel=1e-11)
        assert rel(a, b) < 1e-10

    def test_boltzmann_cutoff_is_immaterial(self, f01):
        w = qp_kernel.marginality(f01)
        a = qp_kernel.kernel_shape_ratio(w, 0.200, cut=120.0)
        b = qp_kernel.kernel_shape_ratio(w, 0.200, cut=400.0)
        assert rel(a, b) < 1e-8

    def test_recomputed_f01_matches_the_milestone1_target(self, report):
        assert report["params"]["f01_recomputed_vs_target_rel"] < 1e-9

    def test_reference_module_is_untouched(self, targets, f01):
        """qp.py must still reproduce qubit_targets.json:quasiparticle."""
        b = qp.qp_budget(targets["design"]["f01_GHz"])
        for k, v in targets["quasiparticle"]["T1_us"].items():
            assert rel(b["T1_us"][k], v) < 1e-9

    def test_export_round_trip(self, tmp_path, report):
        p = tmp_path / "p3_qp.json"
        qp_kernel.export(p)
        with open(p, encoding="utf-8") as fh:
            d = json.load(fh)
        for k in ("kernel", "parity", "thermal_floor", "pair_breaking_audit",
                  "params", "provenance"):
            assert k in d
        for k in ("ratio_vs_smallfreq", "T1_us_vs_xqp", "T_qp_dependence"):
            assert k in d["kernel"]
        assert d["provenance"]["seed"] == 0
        assert d["provenance"]["rng"].startswith("unused")
        assert rel(d["kernel"]["ratio_vs_smallfreq"],
                   report["kernel"]["ratio_vs_smallfreq"]) < 1e-12

    def test_no_absolute_coherence_claim(self, report):
        """Claim discipline: every exported T1 is keyed by an x_qp value."""
        for row in report["kernel"]["T1_us_vs_xqp"].values():
            assert set(row) == {"1e-08", "1e-07", "1e-06", "1e-05"}
        assert "x_qp" in report["provenance"]["claim_discipline"]
