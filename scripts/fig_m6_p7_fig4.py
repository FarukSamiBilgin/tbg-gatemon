"""M6 P7 paper-2 Figure 4 -- two-cavity architecture + kappa_r envelope.

(a) Schematic of the nested/concentric two-cavity concept (concept C,
"VIABLE, CONDITIONAL" in out/m6/p5_two_cavity.json:geometry.concepts) --
readout lattice patch, embedded drive patch, island with two collection
electrodes at the common centre. Geometry numbers are Maxwell rescales /
direct reads from out/m6/p5_two_cavity.json; the drawing itself is a schematic
(not to a single consistent internal scale for every sub-feature) and is
labelled "concept, conditional" per that artifact's own verdict.

(b) The kappa_r operating window vs f_r, REDRAWN 2026-07-27 for the corrected
dispersive shift (out/verify/chi_convention_ruling.md): C2 (measured
internal-Q floor, kappa_r >= f_r/Q_i), the CORRECTED C3 ceiling
kappa^2 <= |alpha||Delta|/(|Delta+alpha| 2 pi T1_min) with Delta = f01 - f_r,
and the all-constraints kappa_max (C3 with the 1.2x Purcell margin plus the
two dispersivity caps and the two-pole screen).  The allowed band is shaded;
the mandated f_r > f01 window is drawn as an explicitly EMPTY reference-
convention region; the new recommended point is starred and the superseded
one is marked.

(c) Readout efficiency eta vs symmetric margin m: eta(m) = 1 - m/R with
R = kappa_ceiling/kappa_intrinsic at the recommended point's Delta_r, nominal
and optimistic-Q_i variants, front points and recommended point marked, with
the superseded (reference-convention) front shown for comparison.

Never invented: every number is read from out/m6/p5_two_cavity.json.

Output: figs/fig_m6_p7_fig4_architecture.png, dpi=200, Agg backend.
"""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
P5_PATH = ROOT / "out" / "m6" / "p5_two_cavity.json"
OUT_PNG = ROOT / "figs" / "fig_m6_p7_fig4_architecture.png"
OUT_CAPTION = ROOT / "figs" / "fig_m6_p7_fig4_architecture.caption.txt"

plt.rcParams["font.size"] = 10.5


def main():
    p5 = json.loads(P5_PATH.read_text())
    geom = p5["geometry"]
    rec = p5["recommended_point"]
    front = rec["margin_efficiency_front"]["front"]
    constraints = {c["id"]: c for c in p5["operating_envelope"]["constraints"]}
    c2 = constraints["C2"]
    c3 = constraints["C3"]
    env_rows = p5["gap_window_scan"]["envelope"]["rows"]
    emp = p5["above_f01_emptiness_proof"]
    ref_rec = p5["reference_convention_scan"]["recommended_point"]
    koch_only = p5["gap_window_scan"]["koch_only_optimum"]["point"]
    f01 = p5["params"]["f01_GHz"]          # src: p5:params.f01_GHz
    f12 = p5["params"]["f12_GHz"]          # src: p5:params.f12_GHz
    window = p5["gap_protection_window"]["f_r_GHz"]

    a_r_um = geom["maxwell_scaling"]["a_r_um"]  # src: p5:geometry.maxwell_scaling.a_r_um
    drive_span_um = geom["concepts"][2]["drive_patch_span_um"]
    readout_span_um = geom["concepts"][2]["readout_defect_span_um"]
    overrun_rows = geom["concepts"][2]["readout_mirror_rows_consumed_per_side"]
    footprint_mm = geom["length_scales_um"]["chip_footprint_readout_mm"]
    concept_verdict = geom["concepts"][2]["verdict"]
    d_eff_drive_um = p5["params"]["d_eff_drive_um"]
    d_eff_readout_um = rec["d_eff_um"]

    fig = plt.figure(figsize=(15.5, 5.4), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.0, 1.0])
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[0, 2])

    # ---------------------------------------------------------- panel (a) ---
    footprint_side = 1.0  # normalised drawing units == chip_footprint_readout_mm
    readout_side = footprint_side * (readout_span_um / (footprint_mm * 1e3))
    drive_side = footprint_side * (drive_span_um / (footprint_mm * 1e3))
    # true-to-scale ratios of the three spans (readout defect < drive patch <
    # outer footprint); the drive patch legitimately overruns the readout
    # defect (rows_consumed_per_side in the artifact), so drive_side >
    # readout_side here.

    axA.add_patch(mpatches.Rectangle((-footprint_side / 2, -footprint_side / 2),
                                       footprint_side, footprint_side,
                                       fc="#dceefb", ec="0.3", lw=1.3, zorder=1))
    axA.add_patch(mpatches.Rectangle((-readout_side / 2, -readout_side / 2),
                                       readout_side, readout_side,
                                       fc="#a9d6f5", ec="0.25", lw=1.3, zorder=2,
                                       label="readout L3 defect"))
    axA.add_patch(mpatches.Rectangle((-drive_side / 2, -drive_side / 2),
                                       drive_side, drive_side,
                                       fc="#f7c99e", ec="0.2", lw=1.3, zorder=3,
                                       label="drive patch (L3 + mirrors)"))
    axA.plot(0, 0, "o", color="k", ms=7, zorder=5)
    axA.annotate("island (2 electrodes)", xy=(0, 0), xytext=(0.0, 0.235), fontsize=7.6,
                  ha="center", zorder=6)
    axA.plot([0, 0.012], [0, drive_side / 2 * 0.7], color="firebrick", lw=1.6, zorder=5)
    axA.plot([0, -0.012], [0, readout_side / 2 * 0.6], color="darkblue", lw=1.6, zorder=5)
    axA.annotate(f"drive electrode\n$d_{{eff}}$={d_eff_drive_um:.0f} um",
                  xy=(0.012, drive_side / 2 * 0.7), xytext=(0.04, drive_side / 2 * 0.85),
                  fontsize=6.8, color="firebrick", ha="left")
    axA.annotate(f"readout electrode\n$d_{{eff}}$={d_eff_readout_um:.1f} um",
                  xy=(-0.012, readout_side / 2 * 0.6), xytext=(-0.05, readout_side / 2 * 1.7),
                  fontsize=6.8, color="darkblue", ha="right")

    axA.annotate("", xy=(-drive_side / 2, -footprint_side * 0.38),
                  xytext=(drive_side / 2, -footprint_side * 0.38),
                  arrowprops=dict(arrowstyle="<->", color="0.2", lw=1.0))
    axA.text(0, -footprint_side * 0.41, f"drive patch span {drive_span_um/1e3:.2f} mm",
              ha="center", fontsize=7.4)

    axA.annotate("", xy=(-readout_side / 2, -footprint_side * 0.30),
                  xytext=(readout_side / 2, -footprint_side * 0.30),
                  arrowprops=dict(arrowstyle="<->", color="0.15", lw=1.0))
    axA.text(0, -footprint_side * 0.33, f"readout defect span {readout_span_um/1e3:.2f} mm",
              ha="center", fontsize=7.4)

    axA.annotate("", xy=(-footprint_side / 2, footprint_side * 0.52),
                  xytext=(footprint_side / 2, footprint_side * 0.52),
                  arrowprops=dict(arrowstyle="<->", color="0.35", lw=1.0))
    axA.text(0, footprint_side * 0.56, f"readout chip footprint {footprint_mm:.1f} x {footprint_mm:.1f} mm",
              ha="center", fontsize=7.4)

    axA.text(0, -footprint_side * 0.62,
              f"$a_r$ = {a_r_um:.0f} um (Maxwell-scaled from photonics.json)\n"
              f"concept C (nested/concentric): {concept_verdict}\n"
              "SCHEMATIC -- CONCEPT, CONDITIONAL (no FDTD run at $a_r$; island/\n"
              "electrodes not to scale, see caption)",
              ha="center", fontsize=7.0, style="italic", color="0.25")

    axA.set_xlim(-footprint_side * 0.7, footprint_side * 0.7)
    axA.set_ylim(-footprint_side * 0.75, footprint_side * 0.68)
    axA.set_aspect("equal")
    axA.axis("off")
    axA.set_title("(a) nested/concentric two-cavity concept (schematic)")
    axA.legend(loc="upper right", fontsize=7.3, framealpha=0.9)

    # ---------------------------------------------------------- panel (b) ---
    # Gap-protection window: the dense, closed-form envelope (src:
    # p5:gap_window_scan.envelope.rows).  Above f01: the four tabulated
    # detunings of the emptiness proof (src:
    # p5:above_f01_emptiness_proof.ceiling_table_MHz).
    f_env = np.array([r["f_r_GHz"] for r in env_rows])
    k_int = np.array([r["kappa_intrinsic_MHz"] for r in env_rows])
    k_ceil = np.array([r["kappa_ceiling_koch_MHz"] for r in env_rows])
    k_max = np.array([r["kappa_max_MHz"] for r in env_rows])
    feas = np.array([r["feasible"] for r in env_rows])

    tab = emp["ceiling_table_MHz"]
    d_tab = sorted(float(k) for k in tab)
    f_tab = np.array([f01 + d for d in d_tab])
    k_int_tab = np.array([tab["%.1f" % d]["kappa_int"] for d in d_tab])
    k_ceil_tab = np.array([tab["%.1f" % d]["koch"] for d in d_tab])

    Y_TOP = 8.0
    # mandated (superseded) f_r > f01 window, drawn as provably empty
    axB.axvspan(f_tab[0], f_tab[-1], color="0.82", alpha=0.55, zorder=0)
    axB.text(0.5 * (f_tab[0] + f_tab[-1]), Y_TOP * 0.935,
             "reference-convention window\nEMPTY under corrected $\\chi$",
             ha="center", va="top", fontsize=7.2, color="0.25", style="italic")

    axB.plot(f_env, k_int, "-", color="C0", lw=2.0,
             label=r"C2 floor: $\kappa_r \geq f_r/Q_i$")
    axB.plot(f_tab, k_int_tab, "-o", color="C0", lw=2.0, ms=4)
    axB.plot(f_env, k_ceil, "-", color="C3", lw=1.7,
             label=r"C3 ceiling (corrected): $\Delta=f_{01}-f_r$")
    axB.plot(f_tab, k_ceil_tab, "-s", color="C3", lw=1.7, ms=5)
    axB.plot(f_env, k_max, "--", color="C1", lw=1.9,
             label=r"$\kappa_{max}$: C3 $\times$1.2 margin + C7 + C8")
    axB.fill_between(f_env, k_int, np.maximum(k_max, k_int),
                     where=feas, color="seagreen", alpha=0.22,
                     label=r"allowed $\kappa_r$ band")

    # feasible-f_r lobes as an unambiguous bar (the upper lobe is only 0.35 GHz
    # wide and is otherwise lost against the pole)
    bands = p5["gap_window_scan"]["envelope"]["feasible_f_r_bands_GHz"]
    for k, (b0, b1) in enumerate(bands):
        axB.plot([b0, b1], [0.16, 0.16], "-", color="seagreen", lw=6,
                 solid_capstyle="butt", zorder=4,
                 label="feasible $f_r$ lobes" if k == 0 else None)
    axB.annotate("2nd lobe %.2f-%.2f GHz" % tuple(bands[-1]),
                 xy=(0.5 * (bands[-1][0] + bands[-1][1]), 0.16),
                 xytext=(f01 + 1.1, 0.75), fontsize=6.6, color="seagreen",
                 ha="left", va="bottom",
                 arrowprops=dict(arrowstyle="->", color="seagreen", lw=0.9))

    for x, lab in ((f12, "$f_{12}$"), (f01, "$f_{01}$")):
        axB.axvline(x, color="0.35", ls=":", lw=1.1)
        axB.text(x, Y_TOP * 0.055, lab, fontsize=8, ha="center",
                 backgroundcolor="white")
    axB.annotate("", xy=(window[0], Y_TOP * 0.885),
                 xytext=(window[1], Y_TOP * 0.885),
                 arrowprops=dict(arrowstyle="<->", color="darkgreen", lw=1.1))
    axB.text(0.5 * (window[0] + window[1]), Y_TOP * 0.90,
             "gap-protection window", ha="center", fontsize=7.2,
             color="darkgreen")

    axB.plot(ref_rec["f_r_GHz"], ref_rec["kappa_MHz"], "X", color="0.45",
             ms=11, mec="k", mew=0.8, zorder=5,
             label="superseded point (fails)")
    axB.plot(rec["f_r_GHz"], rec["kappa_MHz"], "*", color="gold", ms=20,
             mec="k", mew=1.0, zorder=6, label="recommended point")

    axB.set_xlim(f_env[0] - 0.5, f_tab[-1] + 0.7)
    axB.set_ylim(0.0, Y_TOP)
    axB.set_xlabel(r"$f_r$ (GHz)")
    axB.set_ylabel(r"$\kappa_r$ (MHz)")
    axB.set_title(r"(b) $\kappa_r$ window vs $f_r$ (corrected $\chi$)")
    axB.grid(alpha=0.3)
    axB.legend(loc="upper left", fontsize=6.9, framealpha=0.92)

    # ---------------------------------------------------------- panel (c) ---
    R_nominal = rec["margin_efficiency_front"]["R_max_symmetric_margin"]
    eta_max_opt = c2["kappa_int_MHz_optimistic"]
    kappa_ceiling_rec = rec["kappa_ceiling_MHz"]
    R_optimistic = kappa_ceiling_rec / eta_max_opt

    m_curve = np.linspace(1.0, R_nominal, 100)
    eta_curve = 1.0 - m_curve / R_nominal
    m_curve_opt = np.linspace(1.0, R_optimistic, 100)
    eta_curve_opt = 1.0 - m_curve_opt / R_optimistic

    axC.plot(m_curve, eta_curve, "-", color="C0", lw=2.0,
              label=f"nominal $Q_i$: $\\eta=1-m/R$, $R$={R_nominal:.4f}")
    axC.plot(m_curve_opt, eta_curve_opt, "--", color="C1", lw=2.0,
              label=f"optimistic $Q_i$: $R$={R_optimistic:.4f}")
    front_m = [f["symmetric_margin"] for f in front]
    front_eta = [f["eta_readout"] for f in front]
    axC.scatter(front_m, front_eta, marker="o", s=45, color="C0", edgecolor="k", zorder=4)

    # superseded (reference-convention) front, for comparison
    R_ref = ref_rec["kappa_ceiling_MHz"] / ref_rec["kappa_intrinsic_MHz"]
    m_ref_curve = np.linspace(1.0, R_ref, 60)
    axC.plot(m_ref_curve, 1.0 - m_ref_curve / R_ref, ":", color="0.45", lw=1.8,
             label=f"superseded convention: $R$={R_ref:.4f}")

    m_rec = rec["kappa_ceiling_MHz"] / rec["kappa_MHz"]
    eta_rec = rec["eta_readout"]
    axC.plot(m_rec, eta_rec, "*", color="gold", ms=20, mec="k", mew=1.0, zorder=5,
              label="recommended point")
    axC.plot(ref_rec["kappa_ceiling_MHz"] / ref_rec["kappa_MHz"],
             ref_rec["eta_readout"], "X", color="0.45", ms=11, mec="k",
             mew=0.8, zorder=5, label="superseded point (fails)")
    eta_ceil = p5["gap_window_scan"]["envelope"]["eta_ceiling_max"]
    axC.axhline(eta_ceil, color="darkgreen", ls="-.", lw=1.2)
    axC.text(R_nominal * 0.80, eta_ceil + 0.010,
             f"$\\eta$ ceiling in the gap window = {eta_ceil:.4f}",
             fontsize=7.2, color="darkgreen", ha="left")
    axC.plot(koch_only["kappa_ceiling_MHz"] / koch_only["kappa_MHz"],
             koch_only["eta_readout"], "o", color="C3", ms=8, mec="k",
             mew=0.8, zorder=5,
             label=f"C8 dropped: $\\eta$={koch_only['eta_readout']:.4f}, "
                   f"$\\chi_{{2p}}/\\kappa$="
                   f"{koch_only['chi_over_kappa_two_pole']:.4f}")
    axC.axvline(R_nominal, color="0.5", ls=":", lw=1.0)
    axC.annotate(f"$R$={R_nominal:.4f}\n($\\eta$=0)", xy=(R_nominal, 0.0),
                  xytext=(R_nominal * 0.99, 0.05), fontsize=7.5, ha="right")

    axC.set_xlabel(r"symmetric margin $m$")
    axC.set_ylabel(r"readout port efficiency $\eta$")
    axC.set_title("(c) readout efficiency vs symmetric margin")
    axC.grid(alpha=0.3)
    axC.legend(loc="upper right", fontsize=6.8, framealpha=0.92)

    fig.suptitle(
        "M6 P7 Fig. 4 -- two-cavity architecture and readout envelope "
        "(paper 2, design study; corrected $\\chi$ convention)",
        fontsize=10.5,
    )

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=200)

    caption = f"""Figure 4 (paper 2) -- two-cavity architecture, kappa_r envelope, readout efficiency.

(a) Schematic of concept C, the nested/concentric two-cavity chip -- the ONLY
surviving concept out of three considered (side-by-side and shared-seam are
REJECTED), src: out/m6/p5_two_cavity.json:geometry.concepts. Numbers shown:
readout chip footprint {footprint_mm:.2f} x {footprint_mm:.2f} mm (src:
geometry.length_scales_um.chip_footprint_readout_mm); drive patch (its L3
defect plus its 8 measured outer mirror rows) spans {drive_span_um/1e3:.2f} mm
against a {readout_span_um/1e3:.2f} mm readout L3 defect (src:
geometry.concepts[2].drive_patch_span_um / .readout_defect_span_um) -- the
drive patch OVERRUNS the readout defect by {overrun_rows:.2f} readout mirror rows per side
(src: geometry.concepts[2].readout_mirror_rows_consumed_per_side); a_r =
{a_r_um:.1f} um is the Maxwell-rescaled readout lattice constant at the
recommended f_r (src: geometry.maxwell_scaling.a_r_um). Collection-electrode
lengths: drive d_eff = {d_eff_drive_um:.0f} um, readout d_eff =
{d_eff_readout_um:.1f} um (src: params.d_eff_drive_um,
recommended_point.d_eff_um). The three squares (outer footprint, readout
defect, drive patch) are drawn to the TRUE relative scale of their printed
spans; the island and its two collection-electrode stubs are schematic only
(not to scale -- both d_eff are microns against a millimetre-scale patch) and
exist to show topology (electrodes reach the common centre), not geometry.
Verdict quoted
verbatim from the artifact: "{concept_verdict}" -- no FDTD/MPB run has been
performed at a_r (out of P5 scope).

(b) The kappa_r operating window vs f_r, REDRAWN for the corrected dispersive
shift (out/verify/chi_convention_ruling.md; the production chi is now
qubit/cavity.py:chi_dispersive_koch with Delta = f01 - f_r). C2 floor (blue):
the measured, scale-invariant internal-Q linewidth kappa_r >= f_r/Q_i, src:
gap_window_scan.envelope.rows[*].kappa_intrinsic_MHz inside the gap-protection
window and above_f01_emptiness_proof.ceiling_table_MHz[*].kappa_int above f01
(Q_i = {p5['params']['Q_i']}). C3 ceiling (red): the CORRECTED analytic bound
kappa_r^2 <= |alpha|*|Delta|/(|Delta+alpha|*2*pi*T1_min), Delta = f01 - f_r,
src: gap_window_scan.envelope.rows[*].kappa_ceiling_koch_MHz and
above_f01_emptiness_proof.ceiling_table_MHz[*].koch. Orange dashed:
kappa_max, the largest kappa admitting ANY g once the 1.2x Purcell margin,
(g/Delta)^2 <= 0.01, C7 ((g12/Delta_12)^2 <= 0.01) and C8 (the exact two-pole
chi/kappa >= 1) are all imposed, src:
gap_window_scan.envelope.rows[*].kappa_max_MHz. Shaded green: the allowed
band, i.e. where kappa_max > kappa_int; it splits into
{len(p5['gap_window_scan']['envelope']['feasible_f_r_bands_GHz'])} lobes
separated by the 1-2 line f12 = {f12:.4f} GHz, which C7 empties (src:
envelope.feasible_f_r_bands_GHz). Grey band, {f_tab[0]:.2f}-{f_tab[-1]:.2f}
GHz: the mandated Delta_r = 4-16 GHz reference-convention window, which is
EMPTY under the corrected chi -- the C3 ceiling saturates at
{emp['analytic']['sup_kappa_ceiling_MHz']:.4f} MHz while kappa_int is already
{emp['analytic']['kappa_int_at_mandated_Dmin_MHz']:.4f} MHz at its smallest
detuning, and a {emp['sweep']['step_GHz']} GHz sweep over all Delta_r > 0 gives
max(ceiling - kappa_int) =
{emp['sweep']['koch_all_D']['max_ceiling_minus_kappa_int_MHz']:.4f} MHz (src:
above_f01_emptiness_proof). Grey X: the SUPERSEDED recommended point, f_r =
{ref_rec['f_r_GHz']:.4f} GHz, kappa_r = {ref_rec['kappa_MHz']:.3f} MHz, whose
corrected chi/kappa is only
{p5['reference_convention_scan']['corrected_values_at_this_point']['chi_over_kappa_koch']:.4f}
(Koch) / {p5['reference_convention_scan']['corrected_values_at_this_point']['chi_over_kappa_two_pole']:.4f}
(two-pole) -- it fails. Gold star: the corrected recommended point, f_r =
{rec['f_r_GHz']:.4f} GHz, kappa_r = {rec['kappa_MHz']:.3f} MHz, d_eff =
{rec['d_eff_um']:.1f} um, eta = {rec['eta_readout']:.4f} (src:
recommended_point).

(c) Readout port efficiency eta = 1 - kappa_int/kappa_ceiling vs the
symmetric margin m (m_chi = m_T1 = m along this front), src:
out/m6/p5_two_cavity.json:recommended_point.margin_efficiency_front. The
identity m_chi*m_T1 = (kappa_ceiling/kappa)^2 is unchanged by the convention
correction; only the value of kappa_ceiling moves. Solid: nominal Q_i,
eta(m) = 1 - m/R with R = {R_nominal:.4f} = the largest symmetric margin
reachable anywhere (eta = 0 there), src:
recommended_point.margin_efficiency_front.R_max_symmetric_margin. Circles:
the 6 tabulated front points (m = 1.00-1.25). Dashed: the optimistic-Q_i
variant, R = {R_optimistic:.4f} (derived from
operating_envelope.constraints[1].kappa_int_MHz_optimistic =
{eta_max_opt:.4f} MHz and recommended_point.kappa_ceiling_MHz =
{kappa_ceiling_rec:.4f} MHz, both src: out/m6/p5_two_cavity.json). Dotted
grey: the SUPERSEDED reference-convention front, R = {R_ref:.4f} -- the
"largest symmetric margin available anywhere is 1.2532" reading is withdrawn.
Green dash-dot: the exact eta ceiling anywhere in the gap-protection window,
{eta_ceil:.4f} at f_r =
{p5['gap_window_scan']['envelope']['eta_ceiling_max_at_f_r_GHz']:.3f} GHz (src:
gap_window_scan.envelope). Gold star: the recommended point at its own
(m, eta) = ({m_rec:.4f}, {eta_rec:.4f}) (m computed as
kappa_ceiling_MHz/kappa_MHz at that point; it does not sit exactly on the
symmetric front because the recommended point was chosen by the
T1-margin-target objective, not the symmetric-margin objective -- src:
recommended_point.objective). Red circle: the optimum reached if the C8
two-pole screen is dropped, eta = {koch_only['eta_readout']:.4f} but with an
exact two-pole chi/kappa of only
{koch_only['chi_over_kappa_two_pole']:.4f} < 1 -- NOT recommended (src:
gap_window_scan.koch_only_optimum).

CONVENTION NOTE. Panels (b) and (c) are drawn from the corrected chi. Panel
(a) is convention-independent (pure geometry), but its a_r follows the
recommended f_r and therefore moved from 1907.71 um to {a_r_um:.2f} um.

src: out/m6/p5_two_cavity.json (geometry, gap_window_scan,
above_f01_emptiness_proof, reference_convention_scan, recommended_point,
operating_envelope.constraints, params). Deterministic; no RNG (rng_seed=0,
closed-form algebra + deterministic grid sweeps per provenance.rng_note).
"""
    OUT_CAPTION.write_text(caption)

    print("inputs:")
    print(f"  {P5_PATH}")
    print("rng seeds: 0 (out/m6/p5_two_cavity.json:provenance.rng_seed; no stochastic step)")
    print(f"output: {OUT_PNG}  ({OUT_PNG.stat().st_size} bytes, dpi=200)")
    print(f"output: {OUT_CAPTION}")


if __name__ == "__main__":
    main()
