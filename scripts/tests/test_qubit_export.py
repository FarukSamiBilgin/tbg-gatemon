"""M6 deliverable P7: contract tests for qubit.json and scripts/export_qubit.py.

What this file guarantees:

  (i)   the P7 schema keys of M6_BRIEF.md Sec. 4 are present (plus the four
        extension blocks that carry the P2-P6 corrections);
  (ii)  every headline number in qubit.json is bit-identical to the artifact it
        was read from -- both files are loaded and compared, so a silently
        edited export or a regenerated artifact fails here;
  (iii) the claim-discipline sentence "TBG coherence has never been measured"
        is present in _meta and echoed in provenance;
  (iv)  no absolute T1/T2 claim escapes a parametric context: every T1/T2 block
        carries its parametric key (x_qp / T_qp / A) or an explicit
        claim_discipline string, and the P2 correction factors in the export
        match out/m6/p2_coupling.json exactly.

Nothing here recomputes physics -- that is what tests/test_m6_*.py do.  This
file only checks that the export is a faithful, complete, self-consistent view
of artifacts that are already validated.
"""
import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
QUBIT_JSON = ROOT / "qubit.json"
OUT_M6 = ROOT / "out" / "m6"

CLAIM_SENTENCE = "TBG coherence has never been measured"


def _load(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def Q():
    assert QUBIT_JSON.exists(), "qubit.json missing -- run scripts/export_qubit.py"
    return _load(QUBIT_JSON)


@pytest.fixture(scope="module")
def targets():
    return _load(ROOT / "qubit_targets.json")


@pytest.fixture(scope="module")
def p2():
    return _load(OUT_M6 / "p2_coupling.json")


@pytest.fixture(scope="module")
def p3():
    return _load(OUT_M6 / "p3_qp.json")


@pytest.fixture(scope="module")
def p4():
    return _load(OUT_M6 / "p4_t2_maps.json")


@pytest.fixture(scope="module")
def p5():
    return _load(OUT_M6 / "p5_two_cavity.json")


@pytest.fixture(scope="module")
def p6():
    return _load(OUT_M6 / "p6_gates.json")


@pytest.fixture(scope="module")
def p4_rows(p4):
    return {(r["nu"], r["A"]): r for r in p4["reference_rows"]}


# ======================================================== (i) schema ==========
class TestSchema:
    @pytest.mark.parametrize("key", [
        "_meta", "design", "conventions", "budgets", "readout_window",
        "two_cavity", "allocation", "two_qubit_sketch", "open_items",
        "provenance",
    ])
    def test_top_level_key(self, Q, key):
        assert key in Q

    def test_brief_P7_keys_are_all_present(self, Q):
        """M6_BRIEF.md P7: {design, budgets(T1 stack, T2 maps), readout_window,
        two_cavity, provenance}."""
        for k in ("design", "budgets", "readout_window", "two_cavity", "provenance"):
            assert k in Q
        assert "T1_stack" in Q["budgets"] and "T2_maps" in Q["budgets"]

    @pytest.mark.parametrize("key", [
        "schema_version", "date", "claim_discipline", "generator",
        "deterministic", "rng",
    ])
    def test_meta_keys(self, Q, key):
        assert key in Q["_meta"]

    def test_generator_is_this_repos_script(self, Q):
        assert Q["_meta"]["generator"] == "scripts/export_qubit.py"
        assert (ROOT / "scripts" / "export_qubit.py").exists()

    @pytest.mark.parametrize("key", [
        "qp_full_kernel", "purcell_drive", "purcell_readout_at_recommended",
        "thermal_floor", "parity_bound",
    ])
    def test_T1_stack_keys(self, Q, key):
        assert key in Q["budgets"]["T1_stack"]

    @pytest.mark.parametrize("key", [
        "artifacts", "headline_rows", "contrast_at_A_1e_4",
        "ir_cutoff_convention", "quasistatic_reference",
    ])
    def test_T2_maps_keys(self, Q, key):
        assert key in Q["budgets"]["T2_maps"]

    @pytest.mark.parametrize("key", [
        "naive_22pt_reference", "corrected_scan", "recommended_point",
        "constraints", "envelope_statement",
    ])
    def test_readout_window_keys(self, Q, key):
        assert key in Q["readout_window"]

    @pytest.mark.parametrize("key", ["geometry", "purcell_filter", "control_line"])
    def test_two_cavity_keys(self, Q, key):
        assert key in Q["two_cavity"]

    @pytest.mark.parametrize("key", ["fixed_drive", "tracking_drive", "fork_statement"])
    def test_allocation_keys(self, Q, key):
        assert key in Q["allocation"]

    @pytest.mark.parametrize("key", [
        "g_convention", "reference_convention_note", "correction_factors", "beta",
    ])
    def test_conventions_keys(self, Q, key):
        assert key in Q["conventions"]

    @pytest.mark.parametrize("key", [
        "cpr_harmonics", "photon_assisted_pair_breaking",
        "higher_order_gap_capture_145GHz", "heterostructure_fdtd_at_a_r",
    ])
    def test_open_items_keys(self, Q, key):
        assert key in Q["open_items"]

    @pytest.mark.parametrize("key", [
        "source_files", "seed_policy", "tests", "commit_basis", "figures",
    ])
    def test_provenance_keys(self, Q, key):
        assert key in Q["provenance"]

    def test_round_trip(self, Q):
        """The file on disk reloads to an identical object (json round-trip)."""
        again = json.loads(json.dumps(Q))
        assert again == Q

    def test_no_nan_or_inf_anywhere(self, Q):
        """json.load turns NaN/Infinity into floats; the export must have none."""
        bad = []

        def walk(o, path):
            if isinstance(o, dict):
                for k, v in o.items():
                    walk(v, path + "/" + str(k))
            elif isinstance(o, list):
                for i, v in enumerate(o):
                    walk(v, "%s[%d]" % (path, i))
            elif isinstance(o, float) and (o != o or o in (float("inf"), float("-inf"))):
                bad.append(path)

        walk(Q, "")
        assert bad == [], "non-finite values at %s" % bad


# ============================================ (ii) headline numbers ===========
class TestDesignMatchesTargets:
    @pytest.mark.parametrize("key,src_key", [
        ("EJ_over_EC", "EJ_over_EC"), ("EC_GHz", "EC_GHz"),
        ("C_Sigma_fF", "C_fF"), ("f01_GHz", "f01_GHz"),
        ("anharm_GHz", "anharm_GHz"), ("eps1_Hz", "eps1_Hz"), ("n01", "n01"),
    ])
    def test_design_row_is_bit_identical(self, Q, targets, key, src_key):
        assert Q["design"][key] == targets["design"][src_key]

    def test_EJ_from_targets_inputs(self, Q, targets):
        assert Q["design"]["EJ_GHz"] == targets["inputs"]["EJ_GHz"]

    def test_tunability_span(self, Q, targets):
        assert Q["design"]["f01_tunability_GHz"] == targets["gate_noise"]["f01_tunability_GHz"]

    def test_recompute_reproduces_the_reference_scan(self, Q):
        rc = Q["design"]["recomputed_check"]
        assert rc["f01_reldev"] == 0.0
        assert rc["anharm_reldev"] == 0.0
        assert rc["eps1_reldev"] == 0.0
        assert rc["n01_reldev"] < 1e-14
        assert rc["D1_apex_GHz"] == 0.0

    def test_thermal_block(self, Q, targets):
        assert Q["design"]["thermal"]["n_th"] == targets["thermal"]["n_th"]
        assert (Q["design"]["thermal"]["n_th_Al5GHz_100mK"]
                == targets["thermal"]["n_th_Al5GHz_100mK"])
        assert Q["design"]["thermal"]["advantage_vs_Al_5GHz_at_100mK"] == pytest.approx(
            targets["thermal"]["n_th_Al5GHz_100mK"] / targets["thermal"]["n_th"]["0.1"],
            rel=0, abs=0)


class TestGateNumbersMatchP6:
    def test_honest_coherent_floor(self, Q, p6):
        assert (Q["design"]["gates"]["honest_coherent_floor_ns"]
                == p6["combined_floor"]["t_g_floor_ns"]["1e-04"])
        assert (Q["design"]["gates"]["honest_coherent_floor_ns_1e3"]
                == p6["combined_floor"]["t_g_floor_ns"]["1e-03"])

    def test_honest_floor_is_1p60_ns_and_flagged(self, Q):
        g = Q["design"]["gates"]
        assert g["honest_coherent_floor_ns"] == pytest.approx(1.6044397958489334, rel=1e-12)
        assert g["QUOTE_THIS"] is True
        assert "1.6 ns" in g["quote_this_flag"] and "NOT 1.0 ns" in g["quote_this_flag"]

    def test_leakage_only_floors(self, Q, p6):
        lo = Q["design"]["gates"]["leakage_only_floor_ns"]
        assert lo["drag"] == p6["gate_floor"]["drag"]["t_g_floor_ns"]
        assert lo["gaussian"] == p6["gate_floor"]["gaussian"]["t_g_floor_ns"]
        assert lo["speedup_drag_over_gaussian"] == p6["gate_floor"]["speedup_drag_over_gaussian"]
        assert "LEAKAGE-ONLY" in lo["flag"]

    def test_leakage_only_floor_is_not_quoted_as_the_headline(self, Q):
        """1.0112 ns must never be the exported headline floor."""
        g = Q["design"]["gates"]
        assert g["honest_coherent_floor_ns"] > g["leakage_only_floor_ns"]["drag"]["1e-04"]

    def test_drag_params(self, Q, p6):
        d = Q["design"]["gates"]["drag_params"]
        cs = p6["drag"]["coefficient_scan"]
        assert d["q_opt_parabolic"] == cs["q_opt_parabolic"]
        assert d["gain_q1_over_qopt"] == cs["gain_q1_over_qopt"]
        req = [r for r in p6["drag"]["requirement"]
               if r["t_g_ns"] == p6["gate_floor"]["drag"]["t_g_floor_ns"]["1e-04"]][0]
        assert d["quadrature_ratio_Omy_over_Omx_at_1e4_floor"] == req["ratio_measured"]
        assert d["Omx_peak_GHz_at_1e4_floor"] == req["Omx_peak_GHz"]

    def test_ladder_element(self, Q, p6):
        assert (Q["design"]["gates"]["ladder_element_n12_over_n01"]
                == p6["params"]["ladder_ratio"])


class TestConventionsMatchP2:
    def test_g_convention_string_is_the_pinned_one(self, Q):
        assert Q["conventions"]["g_convention"] == (
            "H_int = 2*beta*e*V_zpf*n; g = 2*beta*e*V_zpf*n01/h (Koch 2007)")

    def test_reference_convention_is_retained_not_erased(self, Q):
        note = Q["conventions"]["reference_convention_note"]
        assert "e*V_zpf*n01/h" in note
        assert "2x" in note

    @pytest.mark.parametrize("key", [
        "formula", "drive", "drive_range", "ceiling", "scaling",
    ])
    def test_correction_factor_fields_match_p2(self, Q, p2, key):
        assert Q["conventions"]["correction_factors"][key] == p2["correction_factor"][key]

    def test_readout_sample_correction_matches_p2(self, Q, p2):
        cf = Q["conventions"]["correction_factors"]
        assert cf["readout_sample_d_eff_100um"] == p2["correction_factor"]["readout_sample"]
        assert cf["readout_sample_range"] == p2["correction_factor"]["readout_sample_range"]

    def test_correction_is_exactly_two_beta(self, Q):
        cf = Q["conventions"]["correction_factors"]
        be = Q["conventions"]["beta"]
        assert cf["drive"] == pytest.approx(2 * be["drive"], rel=1e-15)
        assert cf["readout_sample_d_eff_100um"] == pytest.approx(
            2 * be["readout_sample_d_eff_100um"], rel=1e-15)
        assert cf["ceiling"] == pytest.approx(2 * be["beta_max"], rel=1e-15)

    def test_beta_block_matches_p2(self, Q, p2):
        be = Q["conventions"]["beta"]
        for k, src in [("drive", "drive"), ("beta_max", "beta_max"),
                       ("beta_breakeven", "beta_breakeven"),
                       ("C_c_breakeven_fF", "C_c_breakeven_fF")]:
            assert be[k] == p2["beta"][src]

    def test_shared_island_budget_matches_p2(self, Q, p2):
        assert Q["conventions"]["shared_island_budget"] == p2["two_cavity_budget"]

    def test_recommended_correction_matches_p5(self, Q, p5):
        assert (Q["conventions"]["correction_factors"]["recommended_readout_point"]
                == p5["recommended_point"]["correction_factor"])


class TestT1StackMatchesArtifacts:
    def test_qp_ladder_matches_p3(self, Q, p3):
        blk = Q["budgets"]["T1_stack"]["qp_full_kernel"]
        assert blk["T1_us_vs_xqp_T_qp_to_zero"] == p3["kernel"]["T1_us_vs_xqp"]["T_qp->0"]
        assert blk["headline_T1_us_at_xqp_1e_6"] == p3["kernel"]["T1_us_vs_xqp"]["T_qp->0"]["1e-06"]

    def test_headline_qp_T1_is_5p52_us(self, Q):
        blk = Q["budgets"]["T1_stack"]["qp_full_kernel"]
        assert blk["headline_T1_us_at_xqp_1e_6"] == pytest.approx(5.519965921844335, rel=1e-12)

    def test_kernel_correction_matches_p3(self, Q, p3):
        blk = Q["budgets"]["T1_stack"]["qp_full_kernel"]
        assert blk["kernel_correction_total"] == p3["kernel"]["ratio_vs_smallfreq"]
        assert blk["kernel_correction_total"] == pytest.approx(1.1724913601591385, rel=1e-12)
        assert (blk["kernel_correction_decomposition"]
                == p3["kernel"]["ratio_vs_smallfreq_decomposition"])
        assert blk["marginality_hbar_w_over_Delta"] == p3["kernel"]["w_marginality"]

    def test_small_freq_reference_is_retained(self, Q, p3, targets):
        ref = Q["budgets"]["T1_stack"]["qp_full_kernel"]["small_freq_reference"]
        assert ref["T1_us_vs_xqp"] == p3["kernel"]["T1_us_vs_xqp_reference_smallfreq"]
        assert ref["targets_ladder"] == targets["quasiparticle"]["T1_us"]
        assert ref["marginality_flag"] == targets["quasiparticle"]["marginality_flag"]
        assert "RETAINED" in ref["flag"]

    def test_correction_relates_the_two_ladders(self, Q):
        blk = Q["budgets"]["T1_stack"]["qp_full_kernel"]
        ref = blk["small_freq_reference"]["T1_us_vs_xqp"]["1e-06"]
        full = blk["T1_us_vs_xqp_T_qp_to_zero"]["1e-06"]
        assert ref / full == pytest.approx(blk["kernel_correction_total"], rel=1e-12)

    def test_qubit_qp_module_still_reproduces_the_reference(self, Q):
        rc = Q["budgets"]["T1_stack"]["qp_full_kernel"]["small_freq_reference"]["recomputed_check_qubit_qp"]
        assert rc["reldev_at_1e_6"] < 1e-14

    def test_thermal_x_qp_floor_matches_p3(self, Q, p3):
        blk = Q["budgets"]["T1_stack"]["qp_full_kernel"]["thermal_x_qp_floor"]
        assert blk["x_qp_th"] == p3["thermal_floor"]["x_qp_th"]
        assert blk["Delta_over_kB_K"] == p3["thermal_floor"]["Delta_over_kB_K"]

    @pytest.mark.parametrize("export_key,p2_key", [
        ("corrected_T1_us", "T1_Purcell_us"),
        ("corrected_g_MHz", "g_MHz"),
        ("corrected_chi_kHz", "chi_kHz"),
        ("corrected_chi_over_kappa", "chi_over_kappa"),
        ("correction_factor", "correction_factor"),
        ("beta", "beta"),
        ("C_c_fF", "C_c_fF"),
        ("d_eff_um", "d_eff_um"),
        ("kappa_MHz", "kappa_MHz"),
    ])
    def test_purcell_drive_corrected_matches_p2(self, Q, p2, export_key, p2_key):
        pd = Q["budgets"]["T1_stack"]["purcell_drive"]
        assert pd[export_key] == p2["drive_cavity_corrected"][p2_key]

    def test_purcell_drive_is_5291_us(self, Q):
        assert Q["budgets"]["T1_stack"]["purcell_drive"]["corrected_T1_us"] == pytest.approx(
            5290.773559483864, rel=1e-12)

    def test_purcell_drive_reference_matches_at4(self, Q, p2, targets):
        ref = Q["budgets"]["T1_stack"]["purcell_drive"]["reference"]
        assert ref["T1_Purcell_us"] == p2["drive_cavity_reference"]["T1_Purcell_us"]
        assert ref["g_MHz"] == p2["drive_cavity_reference"]["g_MHz"]
        assert ref["targets_row"] == targets["drive_cavity"]

    def test_no_readout_conclusion_strengthening_is_8p4x(self, Q):
        nr = Q["budgets"]["T1_stack"]["purcell_drive"]["no_readout_conclusion"]
        assert nr["strengthening_factor"] == pytest.approx(
            nr["chi_over_kappa_reference"] / nr["chi_over_kappa_corrected"], rel=1e-15)
        assert 8.3 < nr["strengthening_factor"] < 8.5

    def test_purcell_readout_matches_p5(self, Q, p5):
        pr = Q["budgets"]["T1_stack"]["purcell_readout_at_recommended"]
        rec = p5["recommended_point"]
        assert pr["T1_Purcell_us"] == rec["T1_Purcell_us"]
        # Repointed at the corrected artifact by the sanctioned chi-convention
        # correction (out/verify/chi_convention_ruling.md; paper/decisions.md
        # 2026-07-27 M6 chi-convention entry): the recommended point moved from
        # f_r = 44.5966 GHz to 37.4389 GHz. The superseded 36.54018716597618 us
        # is retained and asserted in the reference_convention sub-block below.
        assert pr["T1_Purcell_us"] == pytest.approx(36.05848325289382, rel=1e-12)
        assert pr["margin_over_30us_floor"] == rec["margin_T1"]
        assert pr["reference_convention"]["T1_Purcell_us"] == pytest.approx(
            36.54018716597618, rel=1e-12)

    def test_parity_bound_matches_p3(self, Q, p3):
        pb = Q["budgets"]["T1_stack"]["parity_bound"]
        assert pb["eps1_Hz"] == p3["parity"]["eps1_Hz"]
        assert (pb["Tphi_frozen_parity_upper_bound_s"]
                == p3["parity"]["Tphi_parity_frozen_worst_case_s"])
        assert (pb["Tphi_frozen_over_T1qp_at_1e_6"]
                == p3["parity"]["Tphi_frozen_over_T1qp_at_1e6"])
        assert (pb["Gamma_parity_s_at_xqp_1e6_Tqp_20mK"]
                == p3["parity"]["rate"]["T0.020_x1e-06"]["Gamma_parity_s"])


class TestT2MapsMatchP4:
    def test_pointer_to_artifacts(self, Q, p4):
        art = Q["budgets"]["T2_maps"]["artifacts"]
        assert art["json"] == "out/m6/p4_t2_maps.json"
        assert art["npz"] == p4["npz"]
        assert (OUT_M6 / "p4_t2_maps.npz").exists()
        assert art["grid_nu_points"] == len(p4["grids"]["nu"])
        assert art["grid_A_points"] == len(p4["grids"]["A"])

    @pytest.mark.parametrize("nu,A,label", [
        (-2.4, 1e-4, "apex, A = 1e-4"),
        (-2.4, 1e-3, "apex, A = 1e-3"),
        (-2.5, 1e-4, "nu = -2.5, A = 1e-4"),
        (-2.5, 1e-3, "nu = -2.5, A = 1e-3"),
    ])
    def test_headline_row_is_bit_identical(self, Q, p4_rows, nu, A, label):
        row = [r for r in Q["budgets"]["T2_maps"]["headline_rows"]
               if r["label"] == label][0]
        src = p4_rows[(nu, A)]
        for k in ("T2_ramsey_us", "T2_echo_us", "quasistatic_us",
                  "ramsey_over_quasistatic", "echo_over_ramsey", "dominant_order"):
            assert row[k] == src[k], k

    def test_contrast_numbers(self, Q, p4_rows):
        c = Q["budgets"]["T2_maps"]["contrast_at_A_1e_4"]
        apex, off = p4_rows[(-2.4, 1e-4)], p4_rows[(-2.5, 1e-4)]
        assert c["quasistatic"] == pytest.approx(
            apex["quasistatic_us"] / off["quasistatic_us"], rel=1e-15)
        assert c["ramsey"] == pytest.approx(
            apex["T2_ramsey_us"] / off["T2_ramsey_us"], rel=1e-15)
        assert c["echo"] == pytest.approx(
            apex["T2_echo_us"] / off["T2_echo_us"], rel=1e-15)

    def test_contrast_is_2041_1292_380(self, Q):
        c = Q["budgets"]["T2_maps"]["contrast_at_A_1e_4"]
        assert round(c["quasistatic"]) == 2041
        assert round(c["ramsey"]) == 1292
        assert round(c["echo"]) == 380

    def test_contrast_statement_carries_the_quoting_rule(self, Q):
        c = Q["budgets"]["T2_maps"]["contrast_at_A_1e_4"]
        assert "2041" in c["statement"] and "1292" in c["statement"] and "380" in c["statement"]
        assert "Ramsey-specific" in c["statement"]
        assert "A = 1e-4" in c["statement"]

    def test_ir_cutoff_convention_matches_p4(self, Q, p4):
        ir = Q["budgets"]["T2_maps"]["ir_cutoff_convention"]
        assert ir["f_ir_Hz"] == p4["params"]["f_ir_Hz"] == 1.0
        assert ir["w_ir_rad_per_s"] == p4["params"]["w_ir_rad_per_s"]
        assert ir["noise_convention"] == p4["params"]["noise_convention"]
        assert ir["T2_definition"] == p4["params"]["T2_definition"]

    def test_quasistatic_reference_retained(self, Q, targets):
        qs = Q["budgets"]["T2_maps"]["quasistatic_reference"]
        assert qs["Tphi_apex_us"] == targets["gate_noise"]["Tphi_apex_us"]
        assert qs["Tphi_off_us"] == targets["gate_noise"]["Tphi_off_us"]
        assert qs["D2_apex_GHz"] == targets["gate_noise"]["D2_apex_GHz"]
        assert "RETAINED" in qs["flag"]

    def test_additivity_rule_matches_p4(self, Q, p4):
        ad = Q["budgets"]["T2_maps"]["additivity_rule"]
        d = p4["additivity_diagnostic"]
        assert ad["max_abs_error_all"] == d["max_abs_error_all"]
        assert ad["largest_A_with_max_error_below_1pct"] == d["largest_A_with_max_error_below_1pct"]
        assert ad["worst_point"] == d["worst_point"]


class TestReadoutWindowMatchesArtifacts:
    def test_naive_reference_count_and_flag(self, Q, p5, targets):
        nv = Q["readout_window"]["naive_22pt_reference"]
        assert nv["n_points"] == p5["readout_scan_beta"]["counts"]["naive_feasible"] == 22
        assert nv["n_points"] == len(targets["readout_window"])
        assert "REFERENCE" in nv["flag"] and "NOT a design window" in nv["flag"]

    def test_naive_sample_point_is_the_brief_row(self, Q, targets):
        s = Q["readout_window"]["naive_22pt_reference"]["sample_point_brief"]
        assert s["Delta_r_GHz"] == 7.0 and s["kappa_MHz"] == 1.0
        assert s == targets["readout_window"][3]

    def test_corrected_counts_match_p5(self, Q, p5):
        assert (Q["readout_window"]["corrected_scan"]["counts"]
                == p5["readout_scan_beta"]["counts"])

    @pytest.mark.parametrize("key", [
        "Delta_r_GHz", "f_r_GHz", "d_eff_um", "kappa_MHz", "g_MHz", "chi_MHz",
        "chi_over_kappa", "T1_Purcell_us", "eta_readout", "C_c_sum_fF", "a_r_um",
        "kappa_intrinsic_MHz", "kappa_ceiling_MHz", "beta_readout",
    ])
    def test_recommended_point_is_bit_identical(self, Q, p5, key):
        assert Q["readout_window"]["recommended_point"][key] == p5["recommended_point"][key]

    def test_recommended_point_headline_numbers(self, Q):
        r = Q["readout_window"]["recommended_point"]
        # Constants repointed at the corrected artifact by the sanctioned
        # chi-convention correction (out/verify/chi_convention_ruling.md;
        # paper/decisions.md 2026-07-27 M6 chi-convention entry). The
        # superseded pair (36.54018716597618 us, 0.11532819289330343) is
        # retained verbatim in readout_window.reference_convention_scan and is
        # asserted there.
        assert r["T1_Purcell_us"] == pytest.approx(36.05848325289382, rel=1e-12)
        assert r["eta"] == pytest.approx(0.3074991055500481, rel=1e-12)
        assert r["chi_over_kappa"] > 1.0
        assert r["chi_over_kappa_two_pole"] > 1.0
        assert r["C_c_sum_fF"] < r["C_c_cap_fF"]
        ref = Q["readout_window"]["reference_convention_scan"]["recommended_point"]
        assert ref["T1_Purcell_us"] == pytest.approx(36.54018716597618, rel=1e-12)
        assert ref["eta_readout"] == pytest.approx(0.11532819289330343, rel=1e-12)

    def test_margins(self, Q, p5):
        m = Q["readout_window"]["recommended_point"]["margins"]
        rec = p5["recommended_point"]
        assert m["chi_over_kappa"] == rec["margin_chi_over_kappa"]
        assert m["T1_Purcell"] == rec["margin_T1"]
        assert m["dispersive"] == rec["margin_dispersive"]
        assert m["R_max_symmetric"] == rec["margin_efficiency_front"]["R_max_symmetric_margin"]

    def test_constraints_C1_to_C6_present_and_faithful(self, Q, p5):
        ids = [c["id"] for c in Q["readout_window"]["constraints"]]
        # Extended to C8 by the sanctioned chi-convention correction
        # (out/verify/chi_convention_ruling.md; paper/decisions.md 2026-07-27
        # M6 chi-convention entry): C7 = 1-2-line dispersivity
        # (g12/Delta_12)^2 <= 0.01, C8 = two-pole robustness screen.
        assert ids == ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"]
        src = {c["id"]: c for c in p5["operating_envelope"]["constraints"]}
        for c in Q["readout_window"]["constraints"]:
            assert c["statement"] == src[c["id"]]["statement"]
            assert c["meaning"] == src[c["id"]]["meaning"]

    def test_C2_kappa_floor_makes_the_brief_grid_unreachable(self, Q):
        c2 = [c for c in Q["readout_window"]["constraints"] if c["id"] == "C2"][0]
        # Threshold repointed by the sanctioned chi-convention correction
        # (out/verify/chi_convention_ruling.md; paper/decisions.md 2026-07-27
        # M6 chi-convention entry): the recommended f_r moved from 44.5966 GHz
        # to 37.4389 GHz, so kappa_int = f_r/Q_i moved from 3.05 to 2.56 MHz.
        # The assertion still says what it always said -- the floor is above the
        # brief's kappa_r = 1 MHz grid row and comparable to its 3 MHz row.
        assert c2["numbers"]["kappa_int_MHz_at_recommended"] > 2.5
        assert c2["numbers"]["kappa_int_MHz_at_recommended"] > 1.0

    def test_envelope_statement_matches_p5(self, Q, p5):
        assert (Q["readout_window"]["envelope_statement"]
                == p5["operating_envelope"]["envelope_statement"])


class TestTwoCavityMatchesP5:
    def test_geometry_scale(self, Q, p5):
        g = Q["two_cavity"]["geometry"]
        ms = p5["geometry"]["maxwell_scaling"]
        assert g["a_r_um"] == ms["a_r_um"]
        # Repointed by the sanctioned chi-convention correction
        # (out/verify/chi_convention_ruling.md; paper/decisions.md 2026-07-27
        # M6 chi-convention entry): a_r = a*f_cav/f_r is unchanged as a
        # relation, but f_r moved 44.5966 -> 37.4389 GHz, so a_r moved
        # 1907.7069831787167 -> 2272.4319621778322 um.
        assert g["a_r_um"] == pytest.approx(2272.4319621778322, rel=1e-12)
        assert g["scale_factor_a_r_over_a"] == ms["scale_factor_a_r_over_a"]

    def test_gaps(self, Q, p5):
        gp = p5["geometry"]["gap_positions"]
        g = Q["two_cavity"]["geometry"]["gaps"]
        assert g["drive_lattice_gap_GHz"] == gp["drive_lattice_gap_GHz"]
        assert g["readout_lattice_gap_GHz"] == gp["readout_lattice_gap_GHz"]
        assert g["symmetry_verdict"] == gp["symmetry_verdict"]

    def test_concept_is_nested_conditional(self, Q, p5):
        c = Q["two_cavity"]["geometry"]["concept"]
        assert c["id"] == p5["geometry"]["recommended_concept"] == "C"
        assert "nested" in c["name"]
        assert c["verdict"] == "VIABLE, CONDITIONAL"
        assert [r["verdict"] for r in c["rejected"]] == ["REJECTED", "REJECTED"]

    def test_footprint(self, Q, p5):
        f = Q["two_cavity"]["geometry"]["footprint"]
        ls = p5["geometry"]["length_scales_um"]
        assert f["chip_footprint_readout_mm"] == ls["chip_footprint_readout_mm"]
        assert f["outer_mirror_rows"] == ls["outer_mirror_rows"]

    def test_purcell_filter_blocks(self, Q, p5):
        pf = Q["two_cavity"]["purcell_filter"]
        assert (pf["ladder_option"]["at_recommended"]
                == p5["geometry"]["purcell_filter_N"]["at_recommended"])
        assert pf["bandpass_option"]["rows"] == p5["geometry"]["purcell_filter_element"]["rows"]
        assert pf["ladder_option"]["status"] == "DESIGN ESTIMATE"

    def test_control_line(self, Q, p5):
        cl = Q["two_cavity"]["control_line"]
        assert cl["rabi_budget"] == p5["control_line"]["rabi_budget"]
        assert cl["leakage"] == p5["control_line"]["leakage"]
        assert cl["operating_point"] == p5["control_line"]["operating_point"]

    def test_rabi_power_headline(self, Q):
        rows = {r["Omega_R_over_2pi_MHz"]: r for r in Q["two_cavity"]["control_line"]["rabi_budget"]}
        # Repointed by the sanctioned chi-convention correction
        # (out/verify/chi_convention_ruling.md; paper/decisions.md 2026-07-27
        # M6 chi-convention entry). The Rabi formula is untouched; the
        # operating point it is evaluated at moved (larger g, larger
        # kappa_ext), so the required power fell 8.4116 -> 3.1132 pW at 10 MHz.
        assert rows[10.0]["P_in_pW"] == pytest.approx(3.1132177037589295, rel=1e-12)
        assert rows[50.0]["P_in_pW"] == pytest.approx(77.83044259397323, rel=1e-12)
        # the 25x scaling between the two targets is convention-independent
        assert rows[50.0]["P_in_pW"] / rows[10.0]["P_in_pW"] == pytest.approx(
            25.0, rel=1e-9)


class TestAllocationMatchesP6:
    def test_fixed_drive_is_exactly_one_channel(self, Q, p6):
        fd = Q["allocation"]["fixed_drive"]
        assert fd["verdict"] == "EXACTLY 1 CHANNEL"
        assert fd["channels"] == p6["allocation"]["fixed_drive"]["channels"]
        assert set(fd["channels"].values()) == {1}

    def test_fixed_drive_window_matches_p3_via_p6(self, Q, p3, p6):
        fd = Q["allocation"]["fixed_drive"]
        pb = p3["pair_breaking_audit"]["drive_cavity"]
        assert fd["f01_span_GHz"] == pb["f01_span_inside_window_GHz"]
        assert fd["nu_window_halfwidth"] == pb["window_halfwidth_nu"]
        assert fd["f01_span_GHz"] == p6["allocation"]["fixed_drive"]["f01_span_GHz"]

    def test_span_is_below_the_minimum_spacing(self, Q):
        fd = Q["allocation"]["fixed_drive"]
        assert fd["f01_span_GHz"] < fd["minimum_spacing_at_t_g_8ns_GHz"]

    def test_channel_table_matches_p6(self, Q, p6):
        assert (Q["allocation"]["tracking_drive"]["channel_table"]
                == [{k: r[k] for k in ("t_g_ns", "bandwidth_1_over_tg_GHz",
                                       "spacing_GHz", "spacing_in_units_of_1_over_tg",
                                       "channels_fixed_drive", "channels_tracking_drive")}
                    for r in p6["allocation"]["spacing_vs_gate_time"]])

    def test_spacing_criterion_matches_p6(self, Q, p6):
        sc = Q["allocation"]["spacing_criterion"]
        assert sc["spacing_GHz"] == p6["allocation"]["spacing_criterion"]["spacing_GHz"]
        assert sc["spacing_GHz"]["1e-03"] == pytest.approx(4.212005705701001, rel=1e-12)
        assert sc["spacing_GHz"]["1e-04"] == pytest.approx(11.952149911016534, rel=1e-12)

    def test_fork_statement_names_both_branches(self, Q):
        s = Q["allocation"]["fork_statement"]
        assert "exactly one qubit" in s
        assert "tracking-drive" in s
        assert "145 GHz" in s

    def test_sweet_spot_cost_matches_p6(self, Q, p6):
        assert (Q["allocation"]["fixed_drive"]["sweet_spot_cost"]
                == p6["allocation"]["fixed_drive"]["sweet_spot_cost"])


class TestTwoQubitSketchMatchesP6:
    def test_label_and_disclaimer(self, Q, p6):
        tq = Q["two_qubit_sketch"]
        assert tq["label"] == p6["two_qubit_sketch"]["label"]
        assert "SKETCH" in tq["label"]
        assert "no gate fidelity is claimed" in tq["parametric_disclaimer"]

    def test_exchange_and_budget_bit_identical(self, Q, p6):
        assert Q["two_qubit_sketch"]["exchange"] == p6["two_qubit_sketch"]["exchange"]
        assert (Q["two_qubit_sketch"]["dispersive_budget"]
                == p6["two_qubit_sketch"]["dispersive_budget"])

    def test_design_target_is_tens_of_aF(self, Q):
        dt = Q["two_qubit_sketch"]["design_target"]
        assert "TENS OF aF" in dt["statement"]
        for v in dt["C_qq_max_fF_vs_detuning"].values():
            assert v < 0.2      # i.e. below 200 aF at every scanned detuning

    def test_representative_cross_resonance(self, Q):
        cr = Q["two_qubit_sketch"]["representative_cross_resonance"]
        assert cr["C_qq_fF"] == 0.01 and cr["detuning_GHz"] == 1.0
        assert cr["nu_ZX_MHz"] == pytest.approx(36.3713980415353, rel=1e-12)
        assert cr["t_CNOT_ns"] == pytest.approx(6.873532870925273, rel=1e-12)


# =============================================== (iii) claim discipline =======
class TestClaimDiscipline:
    def test_meta_sentence_present(self, Q):
        assert CLAIM_SENTENCE in Q["_meta"]["claim_discipline"]

    def test_meta_sentence_says_design_study(self, Q):
        cd = Q["_meta"]["claim_discipline"]
        assert "DESIGN STUDY" in cd
        assert "x_qp" in cd and "A" in cd

    def test_sentence_echoed_in_provenance(self, Q):
        assert CLAIM_SENTENCE in Q["provenance"]["claim_discipline"]

    def test_p4_claim_discipline_carried_through(self, Q, p4):
        assert Q["budgets"]["T2_maps"]["claim_discipline"] == p4["claim_discipline"]

    def test_qp_block_is_declared_parametric(self, Q):
        cd = Q["budgets"]["T1_stack"]["qp_full_kernel"]["claim_discipline"]
        assert "parametric" in cd and "x_qp" in cd

    def test_gate_floor_is_declared_coherent_only(self, Q):
        assert "coherent errors only" in Q["design"]["gates"]["claim"]


# ============================ (iv) no absolute T1/T2 outside parametric =======
class TestNoAbsoluteCoherenceClaim:
    """Pragmatic implementation (per the P7 spec): every block that carries a
    T1/T2 number must also carry its parametric key, and the P2 correction
    factors must match the artifact exactly -- the two things that would have
    to be wrong for an absolute claim to slip into the paper."""

    def test_every_qp_T1_is_keyed_by_x_qp(self, Q):
        blk = Q["budgets"]["T1_stack"]["qp_full_kernel"]
        assert set(blk["T1_us_vs_xqp_T_qp_to_zero"]) == {"1e-08", "1e-07", "1e-06", "1e-05"}
        for T_qp, ladder in blk["T1_us_vs_xqp_vs_T_qp"].items():
            assert set(ladder) == {"1e-08", "1e-07", "1e-06", "1e-05"}, T_qp

    def test_every_T2_headline_row_carries_A(self, Q):
        for row in Q["budgets"]["T2_maps"]["headline_rows"]:
            assert "A" in row and row["A"] > 0
            assert "nu" in row

    def test_T2_contrast_is_labelled_with_A_and_sequence(self, Q):
        c = Q["budgets"]["T2_maps"]["contrast_at_A_1e_4"]
        assert set(("quasistatic", "ramsey", "echo")) <= set(c)
        assert "IR cutoff" in c["quoting_rule"]

    def test_A_is_declared_free(self, Q):
        assert "free parameter" in Q["budgets"]["T2_maps"]["A_is_free"]

    def test_correction_factors_match_p2_exactly(self, Q, p2):
        """The (iv) guard named in the P7 spec."""
        cf = Q["conventions"]["correction_factors"]
        assert cf["drive"] == p2["correction_factor"]["drive"]
        assert cf["readout_sample_d_eff_100um"] == p2["correction_factor"]["readout_sample"]
        assert cf["ceiling"] == p2["correction_factor"]["ceiling"]

    def test_no_bare_T1_or_T2_assignment_string_in_the_export(self, Q):
        """Textual guard: the strings 'T1 =' / 'T2 =' may appear only inside a
        parametric or explicitly-flagged context."""
        text = json.dumps(Q)
        offenders = []
        for needle in ("T1 = ", "T2 = ", "T1=", "T2="):
            start = 0
            while True:
                i = text.find(needle, start)
                if i < 0:
                    break
                ctx = text[max(0, i - 400):i + 400]
                parametric = any(t in ctx for t in (
                    "x_qp", "parametric", "A =", "Purcell", "REFERENCE", "reference"))
                if not parametric:
                    offenders.append(text[max(0, i - 80):i + 80])
                start = i + 1
        assert offenders == [], offenders


# ==================================================== provenance / freshness ==
class TestProvenance:
    def test_source_files_exist(self, Q):
        for f in Q["provenance"]["source_files"]:
            assert (ROOT / f["path"]).exists(), f["path"]

    def test_upstream_exports_exist(self, Q):
        for f in Q["provenance"]["upstream_paper1_exports"]:
            assert (ROOT / f["path"]).exists(), f["path"]

    def test_package_modules_exist(self, Q):
        for m in Q["provenance"]["package_modules"]:
            assert (ROOT / m).exists(), m

    def test_test_files_exist_and_counts_sum(self, Q):
        t = Q["provenance"]["tests"]
        for f in t["files"]:
            assert (ROOT / f["path"]).exists(), f["path"]
        prod = sum(f["n_tests"] for f in t["files"]
                   if f["path"] != "tests/test_qubit_export.py")
        # Counts repointed by the sanctioned chi-convention correction
        # (out/verify/chi_convention_ruling.md; paper/decisions.md 2026-07-27
        # M6 chi-convention entry): tests/test_m6_p5.py grew 48 -> 67 with the
        # 19 convention-pinning tests, so M6 production is 278 -> 297.
        assert t["m6_production_total"] == prod == 297
        assert t["full_suite_before_P7"] == prod + t["paper1_total"] == 397
        assert t["full_suite"] == 397 + t["p7_export_total"]

    def test_p7_test_count_is_this_files_own_count(self, Q, request):
        """The exported count must be this file's live collection count."""
        n = len([i for i in request.session.items
                 if "test_qubit_export.py" in i.nodeid])
        exported = Q["provenance"]["tests"]["p7_export_total"]
        if n < exported:
            pytest.skip("partial selection (%d of %d collected)" % (n, exported))
        assert exported == n

    def test_seed_policy_is_deterministic(self, Q):
        sp = Q["provenance"]["seed_policy"]
        assert sp["deterministic"] is True
        assert sp["rng_used"] is False
        assert Q["_meta"]["deterministic"] is True

    def test_digests_are_current(self, Q):
        """Every recorded sha256_16 matches the file on disk -- if an artifact
        was regenerated, the export is stale and must be re-run."""
        import hashlib
        for f in Q["provenance"]["source_files"]:
            h = hashlib.sha256()
            with open(ROOT / f["path"], "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 16), b""):
                    h.update(chunk)
            assert h.hexdigest()[:16] == f["sha256_16"], f["path"]

    def test_figure_naming_convention(self, Q):
        figs = Q["provenance"]["figures"]
        assert figs == [
            "figs/fig_m6_p7_fig1_design.png",
            "figs/fig_m6_p7_fig2_dome.png",
            "figs/fig_m6_p7_fig3_budget.png",
            "figs/fig_m6_p7_fig4_architecture.png",
        ]

    def test_export_is_current(self, Q):
        """Rebuild the payload in memory and compare: qubit.json must be what
        scripts/export_qubit.py produces right now.

        provenance.commit_basis is EXCLUDED from the comparison on purpose. It
        records git HEAD at export time, so any commit by any other module --
        touching none of this export's inputs -- would otherwise turn this test
        red with no number having changed. Data freshness is guarded by
        test_digests_are_current (sha256 of every source artifact); the commit
        field itself is validated by test_commit_basis_is_a_real_commit.
        """
        import copy
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "export_qubit_mod", ROOT / "scripts" / "export_qubit.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        fresh, disk = copy.deepcopy(mod.build_payload()), copy.deepcopy(Q)
        for payload in (fresh, disk):
            payload["provenance"].pop("commit_basis", None)
        assert fresh == disk

    def test_commit_basis_is_a_real_commit(self, Q):
        """The recorded commit_basis must be a commit that exists in this repo
        (or the honest sentinel written when git is unavailable)."""
        import subprocess
        sha = Q["provenance"]["commit_basis"]
        if sha == "unavailable":
            pytest.skip("git was unavailable at export time (sentinel recorded)")
        assert len(sha) == 40 and all(c in "0123456789abcdef" for c in sha), sha
        try:
            out = subprocess.run(["git", "cat-file", "-e", sha + "^{commit}"],
                                 cwd=ROOT, capture_output=True, timeout=30)
        except (OSError, subprocess.SubprocessError):
            pytest.skip("git not runnable in this environment")
        assert out.returncode == 0, "commit_basis %s is not a commit in this repo" % sha

    def test_commit_basis_is_documented_as_informational(self, Q):
        note = Q["provenance"]["commit_basis_note"]
        assert "INFORMATIONAL ONLY" in note
        assert "EXCLUDED" in note
