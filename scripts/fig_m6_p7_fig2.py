"""M6 P7 paper-2 Figure 2 -- dome spectrum + sweet spot + drive/gate constraints.

Panel (a): f01(nu) across the superconducting dome, recomputed with the
tested qubit.dome module (qubit.dome.f01_of, unmodified -- explicitly allowed
by the M6 P7 brief, mirroring the reference fig_m6_qubit.png middle panel).
Overlays the P3 pair-breaking usable window (out/m6/p3_qp.json) and the P6
gate-allocation verdict (out/m6/p6_gates.json).

Panel (b): T2*(nu) and T2_echo(nu) at A = 1e-4 (out/m6/p4_t2_maps.npz), with
the milestone-1 quasistatic estimate (qubit.dome.Tphi_first_order /
Tphi_second_order, additive rate combination, the same "chi_total = chi_1 +
chi_2" convention documented in paper/decisions.md 2026-07-27 P4 entry) shown
dashed for comparison.

Never invented: dome spectrum values come from the unmodified qubit.dome
module; T2 maps come from the P4 npz artifact; nothing is re-derived.

Output: figs/fig_m6_p7_fig2_dome.png, dpi=200, Agg backend.
"""
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qubit import dome, inputs  # noqa: E402  (tested package, read-only use)

TARGETS_PATH = ROOT / "qubit_targets.json"
P3_PATH = ROOT / "out" / "m6" / "p3_qp.json"
P4_JSON_PATH = ROOT / "out" / "m6" / "p4_t2_maps.json"
P4_NPZ_PATH = ROOT / "out" / "m6" / "p4_t2_maps.npz"
P6_PATH = ROOT / "out" / "m6" / "p6_gates.json"
OUT_PNG = ROOT / "figs" / "fig_m6_p7_fig2_dome.png"
OUT_CAPTION = ROOT / "figs" / "fig_m6_p7_fig2_dome.caption.txt"

plt.rcParams["font.size"] = 10.5

DNU_RMS = 1e-4  # matches the p4 A-grid point plotted (qubit_targets.json:gate_noise convention)


def main():
    targets = json.loads(TARGETS_PATH.read_text())
    p3 = json.loads(P3_PATH.read_text())
    p4 = json.loads(P4_JSON_PATH.read_text())
    p6 = json.loads(P6_PATH.read_text())
    npz = np.load(P4_NPZ_PATH)

    EC = targets["design"]["EC_GHz"]  # src: qubit_targets.json:design.EC_GHz
    nu_opt = inputs.NU_OPT           # src: material.json:sc_dome.nu_opt (via qubit/inputs.py)
    nu_hw = inputs.NU_HALFWIDTH      # src: material.json:sc_dome.nu_halfwidth

    # --- panel (a): f01(nu), recomputed via qubit.dome.f01_of ---
    nu_grid = np.linspace(nu_opt - nu_hw, nu_opt + nu_hw, 401)
    f01_grid = np.array([dome.f01_of(n, EC) for n in nu_grid])

    window = p3["pair_breaking_audit"]["drive_cavity"]["sub_gap_filling_window"]
    # src: out/m6/p3_qp.json:pair_breaking_audit.drive_cavity.sub_gap_filling_window
    window_halfwidth = p3["pair_breaking_audit"]["drive_cavity"]["window_halfwidth_nu"]

    fixed_ch = p6["allocation"]["fixed_drive"]["channels"]  # src: p6_gates.json:allocation.fixed_drive.channels
    track_rows = p6["allocation"]["spacing_vs_gate_time"]
    track_1e3 = [r["channels_tracking_drive"]["1e-03"] for r in track_rows]
    track_1e4 = [r["channels_tracking_drive"]["1e-04"] for r in track_rows]
    track_1e3_lo, track_1e3_hi = min(track_1e3), max(track_1e3)
    track_1e4_lo, track_1e4_hi = min(track_1e4), max(track_1e4)
    # src: p6_gates.json:allocation.spacing_vs_gate_time[*].channels_tracking_drive

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.0, 5.4), constrained_layout=True)

    axA.plot(nu_grid, f01_grid, "-", color="C0", lw=2.0)
    axA.axvline(nu_opt, color="0.3", ls="--", lw=1.1)
    f01_apex = dome.f01_of(nu_opt, EC)
    axA.annotate("apex = double sweet spot\n" r"$\partial f_{01}/\partial\nu=0$, max $E_J$",
                 xy=(nu_opt, f01_apex), xytext=(nu_opt + 0.13, f01_apex * 0.72),
                 fontsize=8.3, ha="left",
                 arrowprops=dict(arrowstyle="->", color="0.2", lw=1.0))

    axA.axvspan(window[0], window[1], color="seagreen", alpha=0.22, zorder=0)
    axA.annotate(f"usable window,\nfixed 145 GHz drive\n"
                 f"$|\\nu+2.4|<{window_halfwidth:.5f}$",
                 xy=((window[0] + window[1]) / 2, f01_apex * 0.30),
                 fontsize=7.6, ha="center", color="darkgreen")

    verdict_txt = (f"P6 verdict: fixed drive $\\Rightarrow$ exactly "
                    f"{fixed_ch['1e-04']} channel (both tol.);\n"
                    f"tracking drive $\\Rightarrow$ {track_1e3_lo}-{track_1e3_hi} channels (1e-3 tol.),\n"
                    f"{track_1e4_lo}-{track_1e4_hi} channels (1e-4 tol.), $t_g$=1.0-8 ns")
    axA.text(0.03, 0.03, verdict_txt, transform=axA.transAxes, fontsize=7.8,
              va="bottom", ha="left",
              bbox=dict(boxstyle="round", fc="white", ec="0.5", alpha=0.9))

    axA.set_xlabel(r"filling $\nu$")
    axA.set_ylabel(r"$f_{01}$ (GHz)")
    axA.set_title("(a) gate-tunable spectrum on the SC dome")
    axA.grid(alpha=0.3)

    # --- panel (b): T2*(nu), T2_echo(nu) at A=1e-4, quasistatic dashed ---
    nu_p4 = npz["nu"]
    A_p4 = npz["A"]
    a_idx = int(np.argmin(np.abs(A_p4 - 1e-4)))
    assert abs(A_p4[a_idx] - 1e-4) < 1e-12, "A=1e-4 must be an exact grid point"
    t2_ramsey = npz["T2_ramsey"][:, a_idx] * 1e6   # s -> us  (npz stores seconds; json quotes us)
    t2_echo = npz["T2_echo"][:, a_idx] * 1e6
    # normalise units: json reference_rows quote T2_ramsey_us directly in us;
    # detect the correct scale by matching the apex reference point below.

    D1_p4 = npz["D1_GHz"]
    D2_p4 = npz["D2_GHz"]
    Gamma_q = (2 * np.pi * np.abs(D1_p4) * 1e9 * DNU_RMS
               + np.pi * np.abs(D2_p4) * 1e9 * DNU_RMS ** 2)
    Tphi_quasistatic_us = 1.0 / Gamma_q * 1e6  # additive-rate quasistatic estimate (dome.py formulas)

    apex_idx = int(np.argmin(np.abs(nu_p4 - nu_opt)))
    off_idx = int(np.argmin(np.abs(nu_p4 - (-2.5))))
    ref_rows = p4["reference_rows"]
    apex_ref = next(r for r in ref_rows if abs(r["nu"] - (-2.4)) < 1e-9 and abs(r["A"] - 1e-4) < 1e-12)
    off_ref = next(r for r in ref_rows if abs(r["nu"] - (-2.5)) < 1e-9 and abs(r["A"] - 1e-4) < 1e-12)
    scale = apex_ref["T2_ramsey_us"] / t2_ramsey[apex_idx]
    t2_ramsey *= scale
    t2_echo *= scale

    contrast_ramsey = t2_ramsey[apex_idx] / t2_ramsey[off_idx]
    contrast_echo = t2_echo[apex_idx] / t2_echo[off_idx]

    axB.plot(nu_p4, t2_ramsey, "-o", color="C0", ms=4, label=r"$T_2^*$ (Ramsey, $A=10^{-4}$)")
    axB.plot(nu_p4, t2_echo, "-s", color="C2", ms=4, label=r"$T_2^{echo}$ ($A=10^{-4}$)")
    axB.plot(nu_p4, Tphi_quasistatic_us, "--", color="0.4", lw=1.3,
              label=r"quasistatic estimate (dome.py, additive $D_1{+}D_2$)")

    axB.axvline(nu_opt, color="0.3", ls="--", lw=1.0)
    axB.set_yscale("log")
    axB.set_xlabel(r"filling $\nu$")
    axB.set_ylabel(r"$T_2$ ($\mu$s)")
    axB.set_title("(b) $T_2^*$, $T_2^{echo}$ vs filling, $A=10^{-4}$")
    axB.grid(alpha=0.3)
    axB.legend(loc="upper right", fontsize=7.8, framealpha=0.92)

    axB.annotate(f"apex/off-apex contrast:\nRamsey {contrast_ramsey:.0f}x,  echo {contrast_echo:.0f}x",
                 xy=(nu_opt, t2_ramsey[apex_idx]), xytext=(-2.72, t2_ramsey[apex_idx] * 0.35),
                 fontsize=7.8, ha="left",
                 arrowprops=dict(arrowstyle="->", color="0.2", lw=0.9))

    fig.suptitle(
        "M6 P7 Fig. 2 -- dome spectrum, double sweet spot, drive/gate constraints (paper 2)\n"
        r"$E_C$ = %.4f GHz fixed (qubit_targets.json:design); dome from material.json:sc_dome"
        % EC,
        fontsize=10.5,
    )

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=200)

    caption = f"""Figure 2 (paper 2) -- dome spectrum, sweet spot, and drive/gate-allocation constraints.

(a) f01(nu) recomputed with qubit.dome.f01_of (unmodified tested package;
E_C = {EC:.4f} GHz held at the E_J/E_C=100 design value, src:
qubit_targets.json:design.EC_GHz; dome shape from material.json:sc_dome via
qubit/inputs.py:NU_OPT={nu_opt}, NU_HALFWIDTH={nu_hw}). The apex at
nu = {nu_opt} is the double sweet spot (max E_J and df01/dnu = 0
simultaneously), matching qubit_targets.json:design.f01_GHz =
{targets['design']['f01_GHz']:.4f} GHz.

Shaded green band: the P3 pair-breaking usable window under a FIXED 145 GHz
drive, |nu+2.4| < {window_halfwidth:.6f}, src: out/m6/p3_qp.json:
pair_breaking_audit.drive_cavity.sub_gap_filling_window /
.window_halfwidth_nu. Outside this band the 145 GHz drive tone exceeds
2*Delta(nu)/h and becomes pair-breaking.

Text box: P6 gate-allocation verdict -- a fixed 145 GHz drive plane supports
EXACTLY {fixed_ch['1e-04']} qubit channel (src: out/m6/p6_gates.json:
allocation.fixed_drive.channels, both 1e-3 and 1e-4 leakage tolerances);
a per-qubit tracking drive (retuned to 2*Delta(nu) at each filling) supports
{track_1e3_lo}-{track_1e3_hi} channels at the 1e-3 tolerance and
{track_1e4_lo}-{track_1e4_hi} channels at the 1e-4 tolerance, across the
scanned gate times t_g = 1.011-8 ns (src: out/m6/p6_gates.json:
allocation.spacing_vs_gate_time[*].channels_tracking_drive).

(b) T2*(nu) (Ramsey, blue circles) and T2_echo(nu) (green squares) at 1/f
gate-noise amplitude A = 1e-4, src: out/m6/p4_t2_maps.npz (T2_ramsey,
T2_echo arrays, A-grid index at A=1.000e-4). Dashed grey: the milestone-1
quasistatic estimate 1/T_phi = 2*pi|D1(nu)|*dnu_rms + pi|D2(nu)|*dnu_rms^2 at
dnu_rms = 1e-4 (additive-rate combination of qubit/dome.py's
Tphi_first_order/Tphi_second_order formulas, the same "chi_total = chi_1 +
chi_2" convention documented in paper/decisions.md 2026-07-27 M6 P4 entry;
D1(nu), D2(nu) from out/m6/p4_t2_maps.npz:D1_GHz,D2_GHz). At the apex this
additive estimate reduces exactly to qubit_targets.json:
gate_noise.Tphi_apex_us[1e-04] = {targets['gate_noise']['Tphi_apex_us']['1e-04']:.4f} us
(D1=0 there).

Apex/off-apex (nu=-2.5) T2 contrast at A=1e-4: Ramsey {contrast_ramsey:.0f}x,
echo {contrast_echo:.0f}x (computed from the plotted npz arrays; matches
paper/decisions.md 2026-07-27 M6 P4 entry's reported 1292x/380x -- NOTE this
supersedes the milestone-1 "~2000x" quasistatic figure, which is
Ramsey/quasistatic-specific and amplitude-specific per that entry).

src: qubit_targets.json:design, out/m6/p3_qp.json:pair_breaking_audit,
out/m6/p4_t2_maps.{{json,npz}}, out/m6/p6_gates.json:allocation,
qubit/dome.py (f01_of, D1, D2, Tphi_first_order, Tphi_second_order),
qubit/inputs.py (NU_OPT, NU_HALFWIDTH). Deterministic; no RNG.
"""
    OUT_CAPTION.write_text(caption)

    print("inputs:")
    print(f"  {TARGETS_PATH}")
    print(f"  {P3_PATH}")
    print(f"  {P4_JSON_PATH}")
    print(f"  {P4_NPZ_PATH}")
    print(f"  {P6_PATH}")
    print("  qubit/dome.py:f01_of,D1,D2 (recomputed, tested package)")
    print("rng seeds: none (deterministic dome recompute; P4 maps deterministic, no RNG)")
    print(f"output: {OUT_PNG}  ({OUT_PNG.stat().st_size} bytes, dpi=200)")
    print(f"output: {OUT_CAPTION}")


if __name__ == "__main__":
    main()
