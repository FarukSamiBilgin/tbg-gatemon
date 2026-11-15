"""M6 P7 paper-2 Figure 3 -- T1/T2 coherence budget.

(a) T1 vs x_qp: full Catelani-type kernel (out/m6/p3_qp.json, solid) vs the
small-frequency reference formula (qubit_targets.json:quasiparticle, dashed);
Purcell-limited T1 floors from the P2-corrected drive cavity
(out/m6/p2_coupling.json) and the P5 recommended readout point
(out/m6/p5_two_cavity.json); thermal quasiparticle-density floors at
100/200 mK (out/m6/p3_qp.json:thermal_floor).

(b) T2 vs 1/f amplitude A at the apex and at nu=-2.5, Ramsey and echo
(out/m6/p4_t2_maps.npz), with the frozen-parity dephasing bound
(out/m6/p3_qp.json:parity) and the P4 additive-map validity note.

Never invented: every curve/marker is read directly from the named JSON/NPZ
artifacts; no physics is recomputed in this script.

Output: figs/fig_m6_p7_fig3_budget.png, dpi=200, Agg backend.
"""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
TARGETS_PATH = ROOT / "qubit_targets.json"
P2_PATH = ROOT / "out" / "m6" / "p2_coupling.json"
P3_PATH = ROOT / "out" / "m6" / "p3_qp.json"
P4_JSON_PATH = ROOT / "out" / "m6" / "p4_t2_maps.json"
P4_NPZ_PATH = ROOT / "out" / "m6" / "p4_t2_maps.npz"
P5_PATH = ROOT / "out" / "m6" / "p5_two_cavity.json"
OUT_PNG = ROOT / "figs" / "fig_m6_p7_fig3_budget.png"
OUT_CAPTION = ROOT / "figs" / "fig_m6_p7_fig3_budget.caption.txt"

plt.rcParams["font.size"] = 10.5


def main():
    targets = json.loads(TARGETS_PATH.read_text())
    p2 = json.loads(P2_PATH.read_text())
    p3 = json.loads(P3_PATH.read_text())
    p4 = json.loads(P4_JSON_PATH.read_text())
    p5 = json.loads(P5_PATH.read_text())
    npz = np.load(P4_NPZ_PATH)

    # --- panel (a) data ---
    x_qp_full = sorted(float(x) for x in p3["kernel"]["T1_us_vs_xqp"]["T_qp->0"].keys())
    t1_full = [p3["kernel"]["T1_us_vs_xqp"]["T_qp->0"][f"{x:.0e}"] for x in x_qp_full]
    # src: out/m6/p3_qp.json:kernel.T1_us_vs_xqp["T_qp->0"] (full Catelani kernel, T_qp->0)

    x_qp_ref = sorted(float(x) for x in targets["quasiparticle"]["T1_us"].keys())
    t1_ref = [targets["quasiparticle"]["T1_us"][f"{x:.0e}"] for x in x_qp_ref]
    # src: qubit_targets.json:quasiparticle.T1_us (small-frequency reference formula)

    t1_purcell_drive = p2["drive_cavity_corrected"]["T1_Purcell_us"]
    # src: out/m6/p2_coupling.json:drive_cavity_corrected.T1_Purcell_us (P2 beta-corrected)
    t1_purcell_readout = p5["recommended_point"]["T1_Purcell_us"]
    # src: out/m6/p5_two_cavity.json:recommended_point.T1_Purcell_us

    x_qp_th = p3["thermal_floor"]["x_qp_th"]
    # src: out/m6/p3_qp.json:thermal_floor.x_qp_th
    x_qp_th_100 = x_qp_th["0.100"]
    x_qp_th_200 = x_qp_th["0.200"]

    # --- panel (b) data ---
    nu_p4 = npz["nu"]
    A_p4 = npz["A"]
    apex_idx = int(np.argmin(np.abs(nu_p4 - (-2.4))))
    off_idx = int(np.argmin(np.abs(nu_p4 - (-2.5))))
    t2_ramsey_apex = npz["T2_ramsey"][apex_idx, :] * 1e6
    t2_echo_apex = npz["T2_echo"][apex_idx, :] * 1e6
    t2_ramsey_off = npz["T2_ramsey"][off_idx, :] * 1e6
    t2_echo_off = npz["T2_echo"][off_idx, :] * 1e6

    parity_bound_us = p3["parity"]["Tphi_parity_frozen_worst_case_s"] * 1e6
    # src: out/m6/p3_qp.json:parity.Tphi_parity_frozen_worst_case_s (frozen-parity 1/(pi*eps1) bound)

    A_additive_max = 1e-3  # validity note, src: paper/decisions.md 2026-07-27 M6 P4 entry
    # ("Quote the additive map for A <= 1e-3 only.")

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.0, 5.4), constrained_layout=True)

    # --- panel (a) ---
    axA.plot(x_qp_full, t1_full, "-o", color="C0", ms=6,
              label="full Catelani kernel ($T_{qp}\\to0$)")
    axA.plot(x_qp_ref, t1_ref, "--o", color="C0", ms=5, alpha=0.55, mfc="white",
              label="small-frequency reference")
    axA.axhline(t1_purcell_drive, color="C3", ls="-", lw=1.4,
                 label=f"Purcell, drive cavity (P2-corr.), {t1_purcell_drive:.0f} $\\mu$s")
    axA.axhline(t1_purcell_readout, color="C1", ls="-", lw=1.4,
                 label=f"Purcell, readout (P5 rec.), {t1_purcell_readout:.1f} $\\mu$s")

    for x_th, label, color in ((x_qp_th_200, "200 mK", "purple"), (x_qp_th_100, "100 mK", "indigo")):
        if x_th >= min(x_qp_full):
            axA.axvline(x_th, color=color, ls=":", lw=1.2, alpha=0.8)
            axA.annotate(label, xy=(x_th, t1_full[-1] * 0.5), rotation=90, fontsize=7.5,
                          color=color, ha="right", va="bottom")
        else:
            axA.annotate(f"{label} floor:\n$x_{{qp,th}}$={x_th:.1e}\n(off-scale)",
                          xy=(x_qp_full[0], t1_full[0]),
                          xytext=(x_qp_full[0] * 1.3, t1_full[0] * 0.35),
                          fontsize=6.8, color=color, ha="left")

    axA.set_xscale("log")
    axA.set_yscale("log")
    axA.set_xlabel(r"$x_{qp}$")
    axA.set_ylabel(r"$T_1$ ($\mu$s)")
    axA.set_title("(a) $T_1$ budget: quasiparticle kernel + Purcell floors")
    axA.grid(alpha=0.3, which="both")
    axA.legend(loc="upper right", fontsize=7.3, framealpha=0.92)

    # --- panel (b) ---
    axB.plot(A_p4, t2_ramsey_apex, "-o", color="C0", ms=4, label=r"apex ($\nu$=-2.4), Ramsey")
    axB.plot(A_p4, t2_echo_apex, "-s", color="C2", ms=4, label=r"apex ($\nu$=-2.4), echo")
    axB.plot(A_p4, t2_ramsey_off, "--o", color="C0", ms=4, mfc="white", alpha=0.7,
              label=r"$\nu$=-2.5, Ramsey")
    axB.plot(A_p4, t2_echo_off, "--s", color="C2", ms=4, mfc="white", alpha=0.7,
              label=r"$\nu$=-2.5, echo")
    axB.axhline(parity_bound_us, color="0.25", ls="-.", lw=1.3,
                 label=f"frozen-parity bound, {parity_bound_us:.3e} $\\mu$s")
    axB.axvline(A_additive_max, color="crimson", ls=":", lw=1.2)
    axB.annotate("additive-map\nvalidity: $A\\leq10^{-3}$", xy=(A_additive_max, t2_ramsey_off[0]),
                  xytext=(A_additive_max * 1.6, t2_ramsey_off[0]), fontsize=7.3, color="crimson")

    axB.set_xscale("log")
    axB.set_yscale("log")
    axB.set_xlabel(r"1/f amplitude $A$ (filling units)")
    axB.set_ylabel(r"$T_2$ ($\mu$s)")
    axB.set_title("(b) $T_2$ vs gate-noise amplitude, apex and off-apex")
    axB.grid(alpha=0.3, which="both")
    axB.legend(loc="lower left", fontsize=7.3, framealpha=0.92)

    fig.suptitle(
        "M6 P7 Fig. 3 -- $T_1$/$T_2$ coherence budget (paper 2, design study --\n"
        "parametric in $x_{qp}$ and gate-noise $A$; no absolute TBG coherence claim)",
        fontsize=10.5,
    )

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=200)

    caption = f"""Figure 3 (paper 2) -- T1/T2 coherence budget.

(a) T1 vs normalised quasiparticle density x_qp. Solid blue: the full
Catelani-type kernel at T_qp -> 0, src: out/m6/p3_qp.json:
kernel.T1_us_vs_xqp["T_qp->0"] = {dict(zip(x_qp_full, t1_full))}. Dashed
(open markers): the small-frequency reference formula, src:
qubit_targets.json:quasiparticle.T1_us = {dict(zip(x_qp_ref, t1_ref))}
(the full kernel is {p3['kernel']['ratio_vs_smallfreq']:.4f}x shorter at the
design point, marginality hbar*w_q/Delta = {p3['kernel']['w_marginality']:.4f},
src: out/m6/p3_qp.json:kernel.ratio_vs_smallfreq / w_marginality).

Purcell-limited T1 floors (horizontal lines): drive cavity, P2 beta-corrected,
{t1_purcell_drive:.1f} us (src: out/m6/p2_coupling.json:
drive_cavity_corrected.T1_Purcell_us); readout, P5 recommended point,
{t1_purcell_readout:.2f} us (src: out/m6/p5_two_cavity.json:
recommended_point.T1_Purcell_us).

Thermal quasiparticle-density floor x_qp_th(T) = sqrt(2 pi kB T/Delta)
exp(-Delta/kB T): at 200 mK, x_qp_th = {x_qp_th_200:.4e} (inside the plotted
range); at 100 mK, x_qp_th = {x_qp_th_100:.4e} (far below the plotted range,
annotated off-scale). src: out/m6/p3_qp.json:thermal_floor.x_qp_th. Per that
artifact's verdict, the parametric band 1e-8..1e-5 is entirely
non-equilibrium at the 100 mK operating point.

(b) T2 vs 1/f gate-noise amplitude A, at the apex (nu=-2.4, solid) and at
nu=-2.5 (dashed, open markers), Ramsey (blue circles) and echo (green
squares), src: out/m6/p4_t2_maps.npz (T2_ramsey, T2_echo). Dash-dot black:
the frozen-parity dephasing bound 1/(pi*eps1) = {parity_bound_us:.3e} us, src:
out/m6/p3_qp.json:parity.Tphi_parity_frozen_worst_case_s. Vertical red dotted
line: validity limit for the additive (chi_1+chi_2) T2 map, A <= 1e-3, src:
paper/decisions.md 2026-07-27 M6 P4 entry ("Quote the additive map for
A <= 1e-3 only" -- the exact linear+quadratic determinant, carried as
T2_*_exact in the npz, should be used above this amplitude).

src: out/m6/p3_qp.json (kernel, thermal_floor, parity), qubit_targets.json:
quasiparticle, out/m6/p2_coupling.json:drive_cavity_corrected,
out/m6/p5_two_cavity.json:recommended_point, out/m6/p4_t2_maps.npz.
Deterministic; no RNG (P3/P4 are closed-form/quadrature, no stochastic step).
"""
    OUT_CAPTION.write_text(caption)

    print("inputs:")
    print(f"  {TARGETS_PATH}")
    print(f"  {P2_PATH}")
    print(f"  {P3_PATH}")
    print(f"  {P4_JSON_PATH}")
    print(f"  {P4_NPZ_PATH}")
    print(f"  {P5_PATH}")
    print("rng seeds: none (all sources deterministic, no RNG)")
    print(f"output: {OUT_PNG}  ({OUT_PNG.stat().st_size} bytes, dpi=200)")
    print(f"output: {OUT_CAPTION}")


if __name__ == "__main__":
    main()
