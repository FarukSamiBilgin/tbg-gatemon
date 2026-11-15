"""M6 input plane — every physical number is READ from the paper-1 exports.

Trace policy (M6_BRIEF.md Sec. 5, mirroring paper 1): each physical input
carries a ``# src: <file>:<key>`` comment.  The ONLY numbers allowed to be
literal in this package are

  * mathematical / physical constants (e, h, hbar, k_B, eps0, Phi0),
  * the BCS gap-formula shape constants that live inside the *string*
    ``material.json:sc_dome.Delta_form``, and
  * the shunt design choice (the E_J/E_C ratio ladder that is *scanned*, and
    the two scan-criterion thresholds), which is a design decision, not a
    measured input.

Nothing else is typed from memory.
"""
import json
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------- constants --
# CODATA-2018 exact SI definitions -- mathematical/physical constants only.
E = 1.602176634e-19            # C
H = 6.62607015e-34             # J s
HBAR = H / (2 * np.pi)         # J s
KB = 1.380649e-23              # J/K
EPS0 = 8.8541878128e-12        # F/m
PHI0 = 2.067833848e-15         # Wb  (= h/2e)


def _load(name):
    with open(ROOT / name, encoding="utf-8") as fh:
        return json.load(fh)


MATERIAL = _load("material.json")
JUNCTION = _load("junction.json")
PHOTONICS = _load("photonics.json")
TARGETS = _load("qubit_targets.json")

# ------------------------------------------------------- junction / material --
IC_A = JUNCTION["contract"]["Ic_nA"] / 1e9
# src: junction.json:contract.Ic_nA  (v0.3 wide junction, 300 nA)
# NOTE (numerics, not style): division by the exactly-representable 1e9 is
# correctly rounded and reproduces the literal 300e-9 bit-for-bit, whereas
# `300.0 * 1e-9` lands 1 ULP away.  That 1-ULP shift in E_J moves the float64
# charge dispersion eps1 at E_J/E_C = 100 by 3.4e-5 relative (37.4263 ->
# 37.4276 Hz) -- direct evidence for the brief's Sec. 1 warning that eps1 sits
# only ~4.5 decades above the eigenvalue floor.  Use spectrum.eps1_mp when it
# matters.

EJ_GHZ = PHI0 * IC_A / (2 * np.pi) / H / 1e9
# E_J/h = Phi0*Ic/(2 pi h).  Derived, never typed: 149.005... GHz.

DELTA0_MEV = MATERIAL["Delta0_meV"]
# src: material.json:Delta0_meV  (dome-apex gap, 0.304 meV)

# --- superconducting dome, from material.json:sc_dome -------------------------
TC_MAX_K = MATERIAL["sc_dome"]["Tc_max_K"]        # src: material.json:sc_dome.Tc_max_K
NU_OPT = MATERIAL["sc_dome"]["nu_opt"]            # src: material.json:sc_dome.nu_opt
NU_HALFWIDTH = MATERIAL["sc_dome"]["nu_halfwidth"]  # src: material.json:sc_dome.nu_halfwidth
DELTA_FORM = MATERIAL["sc_dome"]["Delta_form"]
# src: material.json:sc_dome.Delta_form
#   = "1.764*kB*Tc*tanh(1.74*sqrt(Tc/T-1))"
# The two shape constants below are the BCS numbers carried *inside* that
# string; they are formula structure, not measured inputs.
BCS_GAP_RATIO = 1.764      # 2*Delta(0)/(kB Tc) / 2, i.e. Delta(0) = 1.764 kB Tc
BCS_TANH_COEF = 1.74       # Delta(T) ~= Delta(0) tanh(1.74 sqrt(Tc/T - 1))

KB_MEV_PER_K = 0.08617
# k_B expressed in meV/K.  m6_reference.py uses this 4-significant-figure
# value; the CODATA-exact conversion KB/E*1e3 = 0.0861733266... shifts
# D2(-2.4) from -168.8423 to -168.8455 GHz/nu^2 (2e-5 relative), i.e. 0.1% of
# the AT-3 tolerance band of +-3.  The reference literal is retained so the
# port is bit-faithful; see the cross-check in tests/test_m6_acceptance.py.

T_OP_K = 0.1
# Operating temperature, 100 mK.  src: material.json:cpr.model
#   ("Beenakker short junction, T=0.1 K"), consistent with
#   junction.json:contract.Gamma_noise_100mK.

# --------------------------------------------------------------- photonics ---
F_CAV_GHZ = TARGETS["inputs"]["f_cav_GHz"]
# src: qubit_targets.json:inputs.f_cav_GHz ; cross-checks photonics.json:modes.f0_GHz

Q_I = TARGETS["inputs"]["Q_i"]
# src: qubit_targets.json:inputs.Q_i = 14611.7.  Provenance in the photonic
# module: photonics.json:ladder.Q_i_anchor.value carries the same 14611.7
# (photonics.json:modes.Q_i_insitu_24a stores the truncated integer 14611).

V_MODE_M3 = TARGETS["inputs"]["V_mode_m3"]
# src: qubit_targets.json:inputs.V_mode_m3 = 6.68e-11 m^3.  This is the
# 3-significant-figure form of photonics.json:modes.V_mode_um3 = 66792789.7
# um^3 = 6.67928e-11 m^3 (0.011% apart -> 0.005% in g, since g ~ V^-1/2).
# The targets value is used so the port reproduces the reference exactly.

D_EFF_DRIVE_M = TARGETS["inputs"]["d_eff_drive_um"] * 1e-6
# src: qubit_targets.json:inputs.d_eff_drive_um  (collection length, 20 um)

EPS_C = PHOTONICS["geometry"]["eps_core"]
# src: photonics.json:geometry.eps_core  (HR-Si slab, 11.68)

# ------------------------------------------------------------ design choice --
# Not measured inputs: the shunt-pad design ladder that is scanned, and the
# two scan-acceptance thresholds (M6_BRIEF.md Sec. 1).
RATIOS = (25, 50, 100, 150, 200, 300)
EPS1_MAX_HZ = 1e4          # criterion: charge dispersion below 10 kHz
ANHARM_MIN_GHZ = 0.5       # criterion: |anharmonicity| at least 0.5 GHz

N_CHARGE = 35              # charge-basis truncation (M6_BRIEF.md Sec. 5: N >= 35)

__all__ = [
    "ROOT", "E", "H", "HBAR", "KB", "EPS0", "PHI0",
    "MATERIAL", "JUNCTION", "PHOTONICS", "TARGETS",
    "IC_A", "EJ_GHZ", "DELTA0_MEV",
    "TC_MAX_K", "NU_OPT", "NU_HALFWIDTH", "DELTA_FORM",
    "BCS_GAP_RATIO", "BCS_TANH_COEF", "KB_MEV_PER_K", "T_OP_K",
    "F_CAV_GHZ", "Q_I", "V_MODE_M3", "D_EFF_DRIVE_M", "EPS_C",
    "RATIOS", "EPS1_MAX_HZ", "ANHARM_MIN_GHZ", "N_CHARGE",
]
