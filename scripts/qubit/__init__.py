"""M6 qubit package -- TBG gatemon with a photonic control plane (paper 2).

A faithful port of m6_reference.py into focused modules, mirroring the style
of the paper-1 `junction/` package:

    inputs.py    every physical number, read from the four JSONs (trace policy)
    spectrum.py  charge-basis transmon: levels, n01, charge dispersion (AT-1/2)
    dome.py      Tc/Delta/E_J/f01 vs filling nu, sweet spot, quasistatic Tphi (AT-3)
    cavity.py    E_zpf/V_zpf, g, chi, kappa, Purcell, readout-window scan (AT-4/5)
    qp.py        quasiparticle T1 prefactor + thermal occupation (AT-6)

CLAIM DISCIPLINE (M6_BRIEF.md, hard rule): paper 2 is a DESIGN STUDY.  Every
coherence number here is parametric in x_qp or the gate-noise amplitude.
No absolute T1/T2 claim for TBG is made anywhere in this package.

Design lock (M6_BRIEF.md Sec. 1) is recomputed, not hardcoded: call
`design_point()`.
"""
from . import cavity, dome, inputs, qp, spectrum
from .inputs import EJ_GHZ, N_CHARGE, RATIOS

__all__ = ["inputs", "spectrum", "dome", "cavity", "qp",
           "EJ_GHZ", "N_CHARGE", "RATIOS", "design_point"]


def design_point(EJ_GHz=EJ_GHZ, ratios=RATIOS, N=N_CHARGE):
    """Run the design scan and return the selected shunt design point.

    Selection criterion (M6_BRIEF.md Sec. 1): smallest E_J/E_C with
    eps1 < 10 kHz and |anharmonicity| >= 0.5 GHz.
    """
    scan = spectrum.design_scan(EJ_GHz, ratios, N)
    r = spectrum.select_design(scan)
    d = dict(scan[r])
    d["EJ_over_EC"] = r
    d["n01"] = spectrum.n01_element(EJ_GHz, d["EC_GHz"], N)
    d["scan"] = scan
    return d
