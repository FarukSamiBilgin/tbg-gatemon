"""M6 P7 paper-2 Figure 1 -- design scan: f01, |anharm|, eps1 vs E_J/E_C.

Reads qubit_targets.json:scan (the float64 design-scan artifact) and recomputes
the extended-precision charge dispersion eps1_mp at ratios 150/200/300 with
qubit/spectrum.py:eps1_mp (mpmath Sturm-sequence bisection, dps=60, N=35) --
this is the SAME tested package function that paper/decisions.md's
"eps1 is ULP-conditioned already at ratio 100" entry (2026-07-27) names as the
"production rule ... for any ratio >= 150"; nothing is re-derived, only the
package's own extended-precision path is invoked (the same allowance used for
the qubit.dome recompute in Figure 2). No qubit/ source file is modified.

Never invented: every number is either read from qubit_targets.json or
produced by calling the unmodified qubit.spectrum module.

Output: figs/fig_m6_p7_fig1_design.png, dpi=200, Agg backend.
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

from qubit import spectrum, inputs  # noqa: E402  (tested package, read-only use)

TARGETS_PATH = ROOT / "qubit_targets.json"
OUT_PNG = ROOT / "figs" / "fig_m6_p7_fig1_design.png"
OUT_CAPTION = ROOT / "figs" / "fig_m6_p7_fig1_design.caption.txt"

plt.rcParams["font.size"] = 10.5

MP_RATIOS = (150, 200, 300)  # decisions.md 2026-07-27: float64 floor >= ~1mHz


def main():
    targets = json.loads(TARGETS_PATH.read_text())
    scan = targets["scan"]  # src: qubit_targets.json:scan
    design_ratio = targets["design"]["EJ_over_EC"]  # src: qubit_targets.json:design.EJ_over_EC

    ratios = sorted(int(r) for r in scan.keys())
    f01 = [scan[str(r)]["f01_GHz"] for r in ratios]
    anharm = [abs(scan[str(r)]["anharm_GHz"]) for r in ratios]
    eps1_f64 = [scan[str(r)]["eps1_Hz"] for r in ratios]

    # Extended-precision correction at the ULP-sensitive ratios (recompute via
    # the tested qubit.spectrum.eps1_mp path; EJ_GHZ is the package's own
    # derived value, src: qubit/inputs.py:EJ_GHZ <- junction.json:contract.Ic_nA).
    EJ = inputs.EJ_GHZ
    eps1_mp_vals = {}
    for r in MP_RATIOS:
        EC = EJ / r
        eps1_mp_vals[r] = spectrum.eps1_mp(EJ, EC, N=inputs.N_CHARGE) * 1e9  # GHz -> Hz

    eps1_max_hz = inputs.EPS1_MAX_HZ  # src: qubit/inputs.py:EPS1_MAX_HZ (design criterion, 10 kHz)

    fig, ax = plt.subplots(figsize=(7.6, 5.4), constrained_layout=True)

    l1, = ax.plot(ratios, f01, "-o", color="C0", ms=7, label=r"$f_{01}$ (GHz)")
    l2, = ax.plot(ratios, anharm, "-s", color="C1", ms=7, label=r"$|\alpha|$ (GHz)")

    ax.axvline(design_ratio, color="0.3", ls="--", lw=1.2, zorder=1)
    ax.annotate(f"design lock\n$E_J/E_C$={design_ratio}",
                xy=(design_ratio, max(f01) * 0.55), xytext=(design_ratio + 12, max(f01) * 0.62),
                fontsize=8.5, ha="left",
                arrowprops=dict(arrowstyle="->", color="0.3", lw=1.0))

    ax.set_xlabel(r"$E_J/E_C$")
    ax.set_ylabel("GHz")
    ax.grid(alpha=0.3)

    # --- eps1, log right axis ---
    ax2 = ax.twinx()
    l3, = ax2.plot(ratios, eps1_f64, "--^", color="C3", ms=8,
                    label=r"$\epsilon_1$ float64 (Hz)")
    mp_r = sorted(eps1_mp_vals.keys())
    mp_v = [eps1_mp_vals[r] for r in mp_r]
    l4, = ax2.plot(mp_r, mp_v, "-D", color="darkred", ms=7, mfc="none", mew=1.6,
                    label=r"$\epsilon_1$ extended precision (Hz, mpmath dps=60)")
    ax2.set_yscale("log")
    ax2.set_ylabel(r"charge dispersion $\epsilon_1$ (Hz)", color="C3")
    ax2.tick_params(axis="y", colors="C3")

    l5 = ax2.axhline(eps1_max_hz, color="crimson", ls=":", lw=1.3)
    ax2.annotate(f"scan criterion: {eps1_max_hz/1e3:.0f} kHz",
                 xy=(ratios[0], eps1_max_hz), xytext=(ratios[0], eps1_max_hz * 3),
                 fontsize=8, color="crimson")

    # float64 floor region, ratios >= 200 (decisions.md 2026-07-27 ULP finding)
    ax.axvspan(200, ratios[-1] + 10, color="0.6", alpha=0.18, hatch="//", zorder=0)
    ax.annotate("float64 floor\n(ratio $\\geq$ 200):\nnon-monotone,\nuse eps1_mp",
                xy=(250, max(f01) * 0.18), fontsize=7.8, ha="center", color="0.25")

    ax.set_xlim(0, ratios[-1] + 20)
    ax.set_ylim(0, max(f01) * 1.25)

    lines = [l1, l2, l3, l4]
    ax.legend(lines, [l.get_label() for l in lines], loc="upper right", fontsize=8.3,
              framealpha=0.92)

    fig.suptitle(
        "M6 P7 Fig. 1 -- TBG gatemon design scan (paper 2)\n"
        r"$f_{01}$, $|\alpha|$, $\epsilon_1$ vs $E_J/E_C$; float64 (qubit_targets.json:scan) "
        "vs mpmath-corrected (qubit.spectrum.eps1_mp)",
        fontsize=10.5,
    )

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=200)

    caption = f"""Figure 1 (paper 2) -- TBG gatemon design scan: f01, |anharm|, eps1 vs E_J/E_C.

f01 (GHz, left axis, blue circles) and |anharmonicity| (GHz, left axis, orange
squares) at E_J/E_C = {{25,50,100,150,200,300}}, src: qubit_targets.json:scan.
Design lock at E_J/E_C = {design_ratio} (dashed vertical line), src:
qubit_targets.json:design.EJ_over_EC. Reduced-unit conversion: E_J/h =
{EJ:.4f} GHz is fixed across the scan (src: qubit/inputs.py:EJ_GHZ <-
junction.json:contract.Ic_nA, v0.3 junction Ic=300 nA); E_C/h = (E_J/h)/ratio
sets C_Sigma via E_C = e^2/2C (qubit_targets.json:scan.<ratio>.C_fF).

Charge dispersion eps1 (Hz, right axis, log, red triangles dashed) is the
float64 value from qubit_targets.json:scan.<ratio>.eps1_Hz. The scan
criterion eps1 < {eps1_max_hz/1e3:.0f} kHz is marked (crimson dotted line),
src: qubit/inputs.py:EPS1_MAX_HZ (M6_BRIEF.md Sec. 1 design criterion).

mpmath-corrected series (dark-red open diamonds, solid): eps1 recomputed at
ratios 150/200/300 with qubit.spectrum.eps1_mp (Sturm-sequence bisection on
the tridiagonal charge Hamiltonian, dps=60, N=35) -- values {eps1_mp_vals[150]:.4e},
{eps1_mp_vals[200]:.4e}, {eps1_mp_vals[300]:.4e} Hz, matching the reference
figures quoted for this deliverable (src: qubit/spectrum.py:eps1_mp; policy
per paper/decisions.md 2026-07-27 "M6 finding: eps1 is ULP-conditioned already
at ratio 100" -- "Production rule: use spectrum.eps1_mp for any ratio >= 150").

Hatched grey band (ratio >= 200): float64 floor region flagged non-monotonic
and unreliable by the same decisions.md entry -- the mpmath bisection quoted
above (dps=60) is the not the float64 series and should be read there instead,
per that entry's cross-checks (ratio 150: -0.6%, ratio 200: -28%, ratio 300:
4.5 decades high and sign-unstable relative to float64).

src: qubit_targets.json:scan, qubit_targets.json:design.EJ_over_EC,
qubit/inputs.py:EJ_GHZ,EPS1_MAX_HZ, qubit/spectrum.py:eps1_mp (N=35, dps=60).
Deterministic; no RNG (mpmath bisection is a fixed-point iteration, not a
stochastic step).
"""
    OUT_CAPTION.write_text(caption)

    print("inputs:")
    print(f"  {TARGETS_PATH}")
    print("  qubit/spectrum.py:eps1_mp (recomputed at ratios 150,200,300, N=35, dps=60)")
    print("  qubit/inputs.py:EJ_GHZ, EPS1_MAX_HZ")
    print("rng seeds: none (deterministic scan + deterministic mpmath bisection)")
    print(f"output: {OUT_PNG}  ({OUT_PNG.stat().st_size} bytes, dpi=200)")
    print(f"output: {OUT_CAPTION}")


if __name__ == "__main__":
    main()
