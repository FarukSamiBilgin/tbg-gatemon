#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M6 exploration item 1 -- junction-level ABS physics.

    PART A  2 E_A(phi) pair-breaking audit of the 145 GHz drive plane
    PART B  ABS-weighted quasiparticle matrix elements

MOTIVATION
----------
paper/decisions.md 2026-07-27 (M6 P3) and qubit.json:allocation.fixed_drive
carry the single-vs-multi-qubit fork of paper 2:

    "the 145 GHz drive tone sits 1.370% below threshold [2 Delta/h = 147.014
     GHz] ... 2 Delta(nu) < h * 145 GHz for |nu + 2.4| > 0.058571"

That audit (qubit/qp_kernel.py:pair_breaking_audit) is a BULK-GAP audit: it
compares h*f_d against 2*Delta, the energy needed to break a pair in the
*leads*.  A Josephson junction of transparency tau does not have a 2*Delta
threshold at the junction itself.  Its Andreev doublet sits at

    E_A(phi) = Delta * sqrt(1 - tau sin^2(phi/2)),                        (A)

so the pair-transition energy across the junction is 2*E_A(phi) <= 2*Delta,
with equality only at phi = 0.  material.json:cpr is the Beenakker short-
junction CPR at tau = 0.8, i.e. exactly the current that follows from (A)
(cross-checked here: `abs consistency` block).  The junction's own threshold is
therefore phase dependent, and in a transmon phi is a quantum variable with
zero-point spread sigma_phi ~ (2 E_C/E_J)^(1/4) = 0.376 rad.

PART A asks the question the bulk audit cannot: what fraction of the qubit
ground-state phase distribution already sits where 2 E_A(phi) < h * 145 GHz?

PART B is a different physical channel: the OPERATOR, not the levels.
qubit/qp_kernel.py evaluates the Catelani-type rates with the tunnel-limit
matrix elements |<0|sin(phi/2)|1>|^2 and |<0|cos(phi/2)|0>|^2.  At high
transparency the quasiparticle-ABS vertex acquires the coherence-factor
weight (1 - tau sin^2(phi/2))^(-p) = (Delta/E_A(phi))^(2p); PART B measures
how much the two exported brackets move under it.

CLAIM DISCIPLINE (M6_BRIEF.md Sec. "Claim discipline", hard rule)
-----------------------------------------------------------------
This is a THRESHOLD + WEIGHT audit, in the same register as P3's own
photon-assisted pair-breaking flag (qubit.json:open_items.
photon_assisted_pair_breaking): it locates a channel and weights it with the
ground-state phase distribution.  It does NOT compute a rate.  The rate is
quoted only parametrically,

    Gamma_ABS = W * n_drive * gamma_0,

with gamma_0 [s^-1] an explicitly FREE per-photon vertex rate; the script
reports the break-even gamma_0 at which this channel would equal the exported
qp-limited relaxation rate.  No absolute TBG coherence claim is made anywhere.
The weight (1 - tau sin^2(phi/2))^(-p) is a MODEL CHOICE (Catelani-Glazman ABS
form); both p = 1/4 and p = 1/2 and both normalisation conventions are quoted
as a band, never a single number.

NUMERICS / PROVENANCE
---------------------
* Deterministic.  No RNG is consumed anywhere -- exact diagonalisation on
  fixed grids, closed-form thresholds and brentq on monotone functions.
  Seconds of compute.
* No file under qubit/, tests/, paper/ or figs/ is read-write; the tested
  package is imported and used read-only.
* Every physical number is read from material.json / qubit.json /
  qubit_targets.json / photonics.json / out/m6/*.json or recomputed by the
  validated qubit/ package.  Each carries a `# src:` trace.

Writes: out/m6/explore/abs_pair_breaking_audit.json
"""
import json
import pathlib
import sys

import numpy as np
from scipy import optimize

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qubit import cpr_transmon, dome, qp_kernel, spectrum   # noqa: E402
from qubit.inputs import (DELTA0_MEV, E, EJ_GHZ, H, HBAR, N_CHARGE,  # noqa: E402
                          TARGETS)

OUT = ROOT / "out" / "m6" / "explore" / "abs_pair_breaking_audit.json"

# ---------------------------------------------------------------- inputs -----
EC_GHZ = EJ_GHZ / TARGETS["design"]["EJ_over_EC"]
# src: qubit_targets.json:design.EJ_over_EC = 100 ; EJ_GHZ = Phi0*Ic/(2 pi h)
#      with Ic from junction.json:contract.Ic_nA (qubit/inputs.py)

N_PHI = 8192          # phase grid, identical to qp_kernel.matrix_elements default
N_PHI_CONV = 32768    # convergence check only
N_CONV = 50           # charge truncation convergence check (AT-1 uses 35 vs 50)

F_DRIVE_GHZ = TARGETS["inputs"]["f_cav_GHz"]
# src: qubit_targets.json:inputs.f_cav_GHz = 145.0 (photonics.json drive mode)

TWO_DELTA_GHZ = qp_kernel.two_delta_GHz()
# src: qubit/qp_kernel.py:two_delta_GHz() = 2*Delta0/h, Delta0 from
#      material.json:Delta0_meV = 0.304 -> 147.01374591876302 GHz

TAUS_MATERIAL = tuple(float(t) for t in cpr_transmon.taus_available())
# src: material.json:cpr.harmonics_by_tau keys -> (0.5, 0.8, 0.95)
TAU_NOMINAL = float(cpr_transmon.TAU_NOMINAL)
# src: material.json:cpr.tau_nominal = 0.8

W_TARGETS = (1e-3, 1e-2, 1e-1)     # audit thresholds requested by the task
W_TARGETS_SUPP = (0.5, 0.9)        # supplementary: where the weight saturates
P_EXPONENTS = (0.25, 0.5)          # ABS coherence-factor exponents (model choice)


def _load(rel):
    with open(ROOT / rel, encoding="utf-8") as fh:
        return json.load(fh)


QUBIT_JSON = _load("qubit.json")
PHOTONICS = _load("photonics.json")
P3 = _load("out/m6/p3_qp.json")
PEXTRA = _load("out/m6/pextra_cpr.json")

T1_STACK = QUBIT_JSON["budgets"]["T1_stack"]["qp_full_kernel"]
ME_STORED = T1_STACK["matrix_elements"]
# src: qubit.json:budgets.T1_stack.qp_full_kernel.matrix_elements
T1_LADDER_US = T1_STACK["T1_us_vs_xqp_T_qp_to_zero"]
# src: qubit.json:budgets.T1_stack.qp_full_kernel.T1_us_vs_xqp_T_qp_to_zero
KERNEL_CORRECTION = T1_STACK["kernel_correction_total"]
# src: ... .kernel_correction_total = 1.1724913601591385

ALLOC = QUBIT_JSON["allocation"]["fixed_drive"]
# src: qubit.json:allocation.fixed_drive  (bulk window, f01 span, fork verdict)
NU_WINDOW_HALFWIDTH_BULK = ALLOC["nu_window_halfwidth"]
# src: qubit.json:allocation.fixed_drive.nu_window_halfwidth = 0.05857106400995393

RABI_BUDGET = QUBIT_JSON["two_cavity"]["control_line"]["rabi_budget"]
# src: qubit.json:two_cavity.control_line.rabi_budget[*].n_drive
N_DRIVE_RANGE = (min(r["n_drive"] for r in RABI_BUDGET),
                 max(r["n_drive"] for r in RABI_BUDGET))

A_UM_DRIVE = PHOTONICS["geometry"]["a_um_precise"]
# src: photonics.json:geometry.a_um_precise = 586.735322 (145 GHz cavity line)


# ============================================================== phase grid ====
PHI = (np.arange(N_PHI) + 0.5) / N_PHI * 2 * np.pi - np.pi
DPHI = 2 * np.pi / N_PHI
# Identical construction to qubit/qp_kernel.py:matrix_elements (midpoint rule on
# phi in [-pi, pi)); reusing it is what makes the regression check meaningful.


def wavefunctions(Hm, n_phi=N_PHI, N=N_CHARGE):
    """(psi0, psi1) on the midpoint phase grid from a charge-basis H.

    |n> <-> e^{i n phi}/sqrt(2 pi); this is the exact Fourier-transform path of
    qp_kernel.matrix_elements, not a re-derivation.
    """
    n = np.arange(-N, N + 1)
    phi = (np.arange(n_phi) + 0.5) / n_phi * 2 * np.pi - np.pi
    _, vec = np.linalg.eigh(Hm)
    psi = np.exp(1j * np.outer(phi, n)) / np.sqrt(2 * np.pi) @ vec[:, :2]
    return phi, 2 * np.pi / n_phi, psi


def brackets(phi, dphi, psi, weight=None):
    """|<0|sin(phi/2) W|1>|^2, |<0|cos(phi/2) W|0>|^2 and <0|W|0>."""
    g, e = psi[:, 0], psi[:, 1]
    w = np.ones_like(phi) if weight is None else weight
    s01 = abs(np.sum(np.conj(g) * np.sin(phi / 2) * w * e) * dphi) ** 2
    c00 = abs(np.sum(np.conj(g) * np.cos(phi / 2) * w * g) * dphi) ** 2
    w00 = float(np.real(np.sum(np.conj(g) * w * g) * dphi))
    return float(s01), float(c00), w00


def _digits(a, b):
    """Number of agreeing significant decimal digits (capped at 16)."""
    if a == b:
        return 16.0
    return float(min(16.0, -np.log10(abs(a - b) / abs(b))))


# ================================================== PART A: threshold weight ==
def f_2EA_max_GHz(nu, gap_convention="dome"):
    """2*Delta(nu)/h in GHz -- the phi = 0 (bulk) pair-breaking threshold.

    "dome"   : 2*dome.Delta_of(nu)/h, exactly what
               qubit/qp_kernel.py:pair_breaking_boundary compares against, so
               the stored bulk window is reproduced bit-for-bit.
    "pinned" : rescaled so 2*Delta(apex)/h == qp_kernel.two_delta_GHz().
               dome.Delta_of(-2.4) = 0.3040076 meV vs material.json:Delta0_meV
               = 0.304 -- a 2.5e-5 relative difference, carried as a
               sensitivity, never as a second answer.
    """
    d = dome.Delta_of(nu)
    if gap_convention == "dome":
        return float(2.0 * d * 1e-3 * E / H / 1e9)
    return float(TWO_DELTA_GHZ * d / dome.Delta_of(dome.NU_OPT))


def phi_threshold(f_d_GHz, nu, tau, gap_convention="dome"):
    """Smallest |phi| with 2 E_A(phi, nu) < h f_d, from eq. (A).

    2 Delta(nu) sqrt(1 - tau sin^2(phi/2)) < h f_d
      <=>  sin^2(phi/2) > (1 - r^2)/tau,   r = h f_d / (2 Delta(nu)).

    Returns 0.0 when the tone is above the bulk gap (whole grid sub-threshold),
    and NaN when no phase satisfies it (the ABS channel is closed at this tau).
    """
    r = f_d_GHz / f_2EA_max_GHz(nu, gap_convention)
    if r >= 1.0:
        return 0.0
    arg = (1.0 - r * r) / tau
    if arg >= 1.0:
        return float("nan")
    return float(2.0 * np.arcsin(np.sqrt(arg)))


class PhaseDist:
    """|psi0(phi)|^2 with a grid sum and a continuous (interpolated-CDF) tail."""

    def __init__(self, phi, dphi, psi0):
        self.phi, self.dphi = phi, dphi
        self.p = np.abs(psi0) ** 2
        self.norm = float(self.p.sum() * dphi)
        self.edges = np.concatenate(([phi[0] - dphi / 2], phi + dphi / 2))
        self.cdf = np.concatenate(([0.0], np.cumsum(self.p) * dphi))
        self.sigma = float(np.sqrt(np.sum(self.p * phi ** 2) * dphi))

    def W_grid(self, phi_thr):
        if phi_thr is None or np.isnan(phi_thr):
            return 0.0
        return float(np.sum(self.p[np.abs(self.phi) > phi_thr]) * self.dphi)

    def W(self, phi_thr):
        """Continuous, strictly monotone version -- what brentq is run on."""
        if phi_thr is None or np.isnan(phi_thr):
            return 0.0
        if phi_thr <= 0.0:
            return self.norm
        inside = (np.interp(phi_thr, self.edges, self.cdf)
                  - np.interp(-phi_thr, self.edges, self.cdf))
        return float(max(0.0, self.norm - inside))


def tau_min_open(f_d_GHz, nu=None, gap_convention="dome"):
    """Smallest tau for which the ABS channel exists at all: tau = 1 - r^2."""
    nu = dome.NU_OPT if nu is None else nu
    r = f_d_GHz / f_2EA_max_GHz(nu, gap_convention)
    return float(max(0.0, 1.0 - r * r))


def f_safe_GHz(dist, tau, W_target, nu=None, gap_convention="dome"):
    """Drive frequency at which W(apex) = W_target (brentq, monotone in f)."""
    nu = dome.NU_OPT if nu is None else nu
    f_hi = f_2EA_max_GHz(nu, gap_convention) * (1.0 - 1e-12)
    f_lo = f_2EA_max_GHz(nu, gap_convention) * np.sqrt(max(1e-30, 1.0 - tau))
    f_lo = f_lo * (1.0 + 1e-9)
    if dist.W(phi_threshold(f_lo, nu, tau, gap_convention)) > W_target:
        return None
    return float(optimize.brentq(
        lambda f: dist.W(phi_threshold(f, nu, tau, gap_convention)) - W_target,
        f_lo, f_hi, xtol=1e-13, rtol=8.9e-16, maxiter=300))


def nu_at_W(dist, tau, W_target, f_d_GHz=None, gap_convention="dome"):
    """Filling (upper branch) where W(nu) = W_target, or None if unreachable."""
    f_d_GHz = F_DRIVE_GHZ if f_d_GHz is None else f_d_GHz
    apex = dome.NU_OPT
    # Upper bracket = the BULK edge (2 Delta(nu) = h f_d), where W is exactly 1
    # by construction.  Going further up the dome is pointless and eventually
    # hits Delta(nu) = 0 (dome.Delta_of returns 0 once Tc <= T_op).
    lo = apex
    hi = float(optimize.brentq(
        lambda nu: f_2EA_max_GHz(nu, gap_convention) - f_d_GHz,
        apex, apex + dome.NU_HALFWIDTH * 0.9, xtol=1e-15, rtol=8.9e-16))

    def g(nu):
        return dist.W(phi_threshold(f_d_GHz, nu, tau, gap_convention)) - W_target

    if g(lo) > 0.0 or g(hi) < 0.0:
        return None
    return float(optimize.brentq(g, lo, hi, xtol=1e-14, rtol=8.9e-16))


# ============================================== ABS <-> stored CPR consistency =
def abs_cpr_consistency(tau, npts=400001):
    """Check that eq. (A) generates material.json:cpr.harmonics_by_tau[tau].

    Beenakker short junction: I(phi) ~ -dE_A/dphi ~ tau sin(phi) /
    (2 sqrt(1 - tau sin^2(phi/2))), normalised to max = 1 (junction/cpr.py
    convention, restated in qubit/cpr_transmon.py Sec. 1).  The residual is the
    5-harmonic truncation of material.json, not a disagreement.
    """
    ph = np.linspace(0.0, 2.0 * np.pi, npts)
    i_abs = tau * np.sin(ph) / (2.0 * np.sqrt(1.0 - tau * np.sin(ph / 2) ** 2))
    i_abs = i_abs / i_abs.max()
    i_har = cpr_transmon.cpr_current(ph, cpr_transmon.harmonics(tau))
    return dict(max_abs_residual=float(np.abs(i_abs - i_har).max()),
                rms_residual=float(np.sqrt(np.mean((i_abs - i_har) ** 2))),
                argmax_over_pi_from_EA=float(ph[np.argmax(i_abs)] / np.pi),
                argmax_over_pi_from_harmonics=float(
                    cpr_transmon.cpr_argmax(cpr_transmon.harmonics(tau))),
                n_harmonics_material=int(len(cpr_transmon.harmonics(tau))),
                npts=int(npts))


# ==================================================================== main ====
def build():
    # ---------------------------------------------------------- bases --------
    H_cos = spectrum.hamiltonian(EJ_GHZ, EC_GHZ, 0.0, N_CHARGE)
    phi, dphi, psi_cos = wavefunctions(H_cos)
    assert np.array_equal(phi, PHI)
    dist = {"cosine": PhaseDist(phi, dphi, psi_cos[:, 0])}
    psi = {"cosine": psi_cos}
    for tau in TAUS_MATERIAL:
        c = cpr_transmon.harmonics(tau)
        _, _, pc = wavefunctions(
            cpr_transmon.hamiltonian(EJ_GHZ, EC_GHZ, c, 0.0, N_CHARGE))
        key = "cpr_tau%s" % tau
        psi[key] = pc
        dist[key] = PhaseDist(phi, dphi, pc[:, 0])

    # ------------------------------------------------ regression check -------
    s01, c00, _ = brackets(phi, dphi, psi_cos)
    me_pkg = qp_kernel.matrix_elements(EJ_GHZ, EC_GHZ, N_CHARGE, N_PHI)
    regression = dict(
        sin_half_01_sq_here=s01, cos_half_00_sq_here=c00,
        sin_half_01_sq_stored=ME_STORED["sin_half_01_sq"],
        cos_half_00_sq_stored=ME_STORED["cos_half_00_sq"],
        sin_half_01_sq_package=me_pkg["sin_half_01_sq"],
        cos_half_00_sq_package=me_pkg["cos_half_00_sq"],
        digits_sin=_digits(s01, ME_STORED["sin_half_01_sq"]),
        digits_cos=_digits(c00, ME_STORED["cos_half_00_sq"]),
        digits_sin_vs_p3=_digits(
            s01, P3["kernel"]["matrix_elements"]["sin_half_01_sq"]),
        source="qubit.json:budgets.T1_stack.qp_full_kernel.matrix_elements",
        note=("the phase-grid path is used here; qubit.json also stores the "
              "charge-basis Toeplitz path -- the two differ in the 16th digit "
              "(4e-17 absolute) in the package's own export"),
    )

    # ------------------------------------------------ bulk cross-check -------
    bulk_root = optimize.brentq(
        lambda nu: f_2EA_max_GHz(nu, "dome") - F_DRIVE_GHZ,
        dome.NU_OPT, dome.NU_OPT + dome.NU_HALFWIDTH,
        xtol=1e-15, rtol=8.9e-16)
    bulk = dict(
        two_Delta_over_h_GHz=TWO_DELTA_GHZ,
        two_Delta_over_h_GHz_dome=f_2EA_max_GHz(dome.NU_OPT, "dome"),
        f_drive_GHz=F_DRIVE_GHZ,
        margin_2Delta_over_hf=float(TWO_DELTA_GHZ / F_DRIVE_GHZ),
        pct_below_threshold=float(100.0 * (1.0 - F_DRIVE_GHZ / TWO_DELTA_GHZ)),
        nu_window_halfwidth_recomputed=float(bulk_root - dome.NU_OPT),
        nu_window_halfwidth_stored=NU_WINDOW_HALFWIDTH_BULK,
        nu_window_halfwidth_reldev=float(
            abs((bulk_root - dome.NU_OPT) - NU_WINDOW_HALFWIDTH_BULK)
            / NU_WINDOW_HALFWIDTH_BULK),
        src="qubit.json:allocation.fixed_drive + qubit/qp_kernel.py",
    )

    # ------------------------------------------------ PART A rows ------------
    sigma_harmonic = float((2.0 * EC_GHZ / EJ_GHZ) ** 0.25)
    rows = {}
    for tau in TAUS_MATERIAL:
        pt = phi_threshold(F_DRIVE_GHZ, dome.NU_OPT, tau, "dome")
        pt_pin = phi_threshold(F_DRIVE_GHZ, dome.NU_OPT, tau, "pinned")
        row = dict(
            tau=tau,
            phi_threshold_rad=pt,
            phi_threshold_rad_pinned_gap=pt_pin,
            phi_threshold_over_sigma_harmonic=float(pt / sigma_harmonic),
            E_A_at_phi0_over_Delta=1.0,
            E_A_at_pi_over_Delta=float(np.sqrt(1.0 - tau)),
            tau_min_channel_open=tau_min_open(F_DRIVE_GHZ),
        )
        for bname, key in (("cosine", "cosine"), ("exact_cpr", "cpr_tau%s" % tau)):
            d = dist[key]
            row[bname] = dict(
                sigma_phi_rad=d.sigma,
                phi_threshold_over_sigma=float(pt / d.sigma),
                W_apex=d.W(pt),
                W_apex_grid_sum=d.W_grid(pt),
                W_apex_pinned_gap=d.W(pt_pin),
                nu_at_W={},
                f_safe_GHz={},
            )
            for wt in W_TARGETS:
                nu_w = nu_at_W(d, tau, wt)
                row[bname]["nu_at_W"]["%.0e" % wt] = (
                    None if nu_w is None else float(nu_w - dome.NU_OPT))
                fs = f_safe_GHz(d, tau, wt)
                row[bname]["f_safe_GHz"]["%.0e" % wt] = fs
                if fs is not None:
                    row[bname].setdefault("f_safe_margin", {})[
                        "%.0e" % wt] = float(TWO_DELTA_GHZ / fs)
                    row[bname].setdefault("f_safe_pct_below_2Delta", {})[
                        "%.0e" % wt] = float(100.0 * (1.0 - fs / TWO_DELTA_GHZ))
            row[bname]["nu_at_W_supplementary"] = {}
            for wt in W_TARGETS_SUPP:
                nu_w = nu_at_W(d, tau, wt)
                row[bname]["nu_at_W_supplementary"]["%.2f" % wt] = (
                    None if nu_w is None else float(nu_w - dome.NU_OPT))
            empty = [w for w in W_TARGETS
                     if row[bname]["nu_at_W"]["%.0e" % w] is None]
            row[bname]["nu_window_verdict"] = (
                "EMPTY at W < {%s}: W already equals %.6f AT THE APEX, so no "
                "filling inside the stored bulk window |nu+2.4| < %.6f meets "
                "any of the three audit thresholds. W rises monotonically to 1 "
                "at the bulk edge."
                % (", ".join("%.0e" % w for w in empty),
                   row[bname]["W_apex"], NU_WINDOW_HALFWIDTH_BULK)
                if len(empty) == len(W_TARGETS) else
                "partially populated: %s" % row[bname]["nu_at_W"])
        row["W_apex_band"] = [min(row["cosine"]["W_apex"],
                                  row["exact_cpr"]["W_apex"]),
                              max(row["cosine"]["W_apex"],
                                  row["exact_cpr"]["W_apex"])]
        row["W_apex_basis_spread_pct"] = float(
            100.0 * (row["exact_cpr"]["W_apex"] / row["cosine"]["W_apex"] - 1.0))
        rows[str(tau)] = row

    # W(nu) profile at the nominal tau, both bases
    nu_offsets = (0.0, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05,
                  NU_WINDOW_HALFWIDTH_BULK, 0.08, 0.10)
    profile = {"nu_offset": list(map(float, nu_offsets)),
               "f_2Delta_of_nu_GHz": [f_2EA_max_GHz(dome.NU_OPT + d, "dome")
                                      for d in nu_offsets]}
    for bname, key in (("cosine", "cosine"),
                       ("exact_cpr", "cpr_tau%s" % TAU_NOMINAL)):
        profile[bname + "_W"] = [
            dist[key].W(phi_threshold(F_DRIVE_GHZ, dome.NU_OPT + d,
                                      TAU_NOMINAL, "dome"))
            for d in nu_offsets]

    # tau trend on a fine grid (cosine basis: the material file only resolves 3 taus)
    tau_fine = [0.01, 0.02, 0.0272, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5,
                0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    trend = dict(tau=tau_fine,
                 W_apex_cosine=[dist["cosine"].W(
                     phi_threshold(F_DRIVE_GHZ, dome.NU_OPT, t, "dome"))
                     for t in tau_fine],
                 phi_threshold_rad=[phi_threshold(F_DRIVE_GHZ, dome.NU_OPT,
                                                  t, "dome")
                                    for t in tau_fine])
    trend["monotone_nondecreasing"] = bool(
        all(b >= a - 1e-15 for a, b in zip(trend["W_apex_cosine"][:-1],
                                           trend["W_apex_cosine"][1:])))
    trend["tau_min_channel_open"] = tau_min_open(F_DRIVE_GHZ)

    # other tones: is this a drive-plane-only effect, as the bulk audit says?
    tones = {}
    for name, f in (("control_f01", TARGETS["design"]["f01_GHz"]),
                    ("readout_recommended",
                     QUBIT_JSON["two_cavity"]["control_line"]
                     ["operating_point"]["f_r_GHz"]),
                    ("readout_sample_47p6", qp_kernel.F_READOUT_GHZ),
                    ("drive_cavity_145", F_DRIVE_GHZ)):
        e = {}
        for tau in TAUS_MATERIAL:
            pt = phi_threshold(f, dome.NU_OPT, tau, "dome")
            e[str(tau)] = dict(
                channel_open=bool(not np.isnan(pt)),
                phi_threshold_rad=(None if np.isnan(pt) else pt),
                W_apex_cosine=dist["cosine"].W(pt),
                tau_min_channel_open=tau_min_open(f))
        tones[name] = dict(f_GHz=float(f),
                           hf_over_2Delta=float(f / TWO_DELTA_GHZ), rows=e)

    # ------------------------------------------- parametric rate statement ---
    T1_QP_US = T1_LADDER_US["1e-06"]
    gamma_qp = 1.0 / (T1_QP_US * 1e-6)
    omega_d = 2 * np.pi * F_DRIVE_GHZ * 1e9
    gamma_rows = {}
    for tau in TAUS_MATERIAL:
        band = rows[str(tau)]["W_apex_band"]
        ent = {}
        for lab, Wv in (("W_lo", band[0]), ("W_hi", band[1])):
            ent[lab] = {}
            for nlab, nd in (("n_drive_lo", N_DRIVE_RANGE[0]),
                             ("n_drive_hi", N_DRIVE_RANGE[1])):
                pref = Wv * nd
                ent[lab][nlab] = dict(
                    W_times_n_drive=float(pref),
                    gamma0_breakeven_s=float(gamma_qp / pref),
                    gamma0_breakeven_over_omega_drive=float(
                        gamma_qp / pref / omega_d))
        gamma_rows[str(tau)] = ent
    gamma_block = dict(
        law="Gamma_ABS = W(tau, nu, f_d) * n_drive * gamma_0",
        free_parameter=dict(
            name="gamma_0",
            meaning=("per-drive-photon, per-unit-sub-threshold-weight ABS "
                     "pair-breaking vertex rate, s^-1"),
            status="FREE -- not computed here; this block is a break-even map"),
        n_drive_range=list(map(float, N_DRIVE_RANGE)),
        n_drive_src=("qubit.json:two_cavity.control_line.rabi_budget[*]."
                     "n_drive (10 and 50 MHz Rabi)"),
        n_drive_caveat=("that budget is the 44.6 GHz READOUT-port control "
                        "line; the intracavity photon number of the 145 GHz "
                        "drive mode is not exported anywhere in qubit.json, "
                        "so this range is used only as the photon-number "
                        "SCALE. Gamma_ABS is linear in n_drive -- rescale "
                        "freely."),
        reference_rate=dict(
            T1_qp_us_at_xqp_1e6=T1_QP_US,
            Gamma_qp_s=float(gamma_qp),
            src="qubit.json:budgets.T1_stack.qp_full_kernel."
                "T1_us_vs_xqp_T_qp_to_zero['1e-06']"),
        omega_drive_rad_s=float(omega_d),
        rows=gamma_rows,
        reading=("gamma0_breakeven_over_omega_drive is the fraction of the "
                 "drive frequency the unknown vertex rate would have to reach "
                 "before this channel matched the exported qp-limited "
                 "relaxation rate. Small numbers mean the channel is cheap to "
                 "excite -- a weak flag, not a computed rate."),
    )

    # ------------------------------- drive-plane redesign consequence --------
    fs_nom = rows[str(TAU_NOMINAL)]["cosine"]["f_safe_GHz"]["1e-03"]
    fs_nom_cpr = rows[str(TAU_NOMINAL)]["exact_cpr"]["f_safe_GHz"]["1e-03"]
    redesign = dict(
        rule=("Maxwell scale invariance at fixed r/a, t/a and dispersionless "
              "eps_core: a -> a * f_old/f_new reproduces the same mode"),
        a_um_current=float(A_UM_DRIVE),
        f_current_GHz=float(F_DRIVE_GHZ),
        f_safe_GHz_W1em3_cosine=fs_nom,
        f_safe_GHz_W1em3_exact_cpr=fs_nom_cpr,
        a_um_required_cosine=(None if fs_nom is None else
                              float(A_UM_DRIVE * F_DRIVE_GHZ / fs_nom)),
        a_um_required_exact_cpr=(None if fs_nom_cpr is None else
                                 float(A_UM_DRIVE * F_DRIVE_GHZ / fs_nom_cpr)),
        a_growth_pct_band=[
            float(100.0 * (F_DRIVE_GHZ / fs_nom - 1.0)),
            float(100.0 * (F_DRIVE_GHZ / fs_nom_cpr - 1.0))],
        caveat=("Q_i and V_mode/lambda^3 are scale invariant for a lossless "
                "dielectric, but material.json/photonics.json fix tan_delta "
                "and the fabrication tolerance budget in ABSOLUTE units; a "
                "re-scaled lattice needs its own tolerance run. Not done "
                "here -- this is the size of the required move, not a design."),
        src="photonics.json:geometry.a_um_precise / a_provenance",
    )

    nomrow = rows[str(TAU_NOMINAL)]
    verdict_A = dict(
        headline=(
            "THE 1.370%% BULK MARGIN IS NOT A JUNCTION MARGIN. At the measured "
            "tau = %.2f the Andreev pair-transition energy 2 E_A(phi) drops "
            "below h * 145 GHz for |phi| > %.6f rad, which is only %.3f "
            "sigma_phi of the E_J/E_C = 100 ground state (sigma_phi = %.6f "
            "rad). The ground-state phase weight already sitting above the "
            "junction threshold AT THE DOME APEX is W = %.4f (cosine basis) to "
            "%.4f (exact-CPR eigenbasis) -- a third of the wavefunction, not a "
            "1.4%% margin."
            % (TAU_NOMINAL, nomrow["phi_threshold_rad"],
               nomrow["cosine"]["phi_threshold_over_sigma"],
               dist["cosine"].sigma, nomrow["cosine"]["W_apex"],
               nomrow["exact_cpr"]["W_apex"])),
        window_verdict=(
            "the ABS-corrected sub-threshold filling window is EMPTY at every "
            "audit threshold {1e-3, 1e-2, 0.1}: W(apex) = %.4f exceeds all "
            "three. The stored bulk window |nu+2.4| < %.6f is the region where "
            "W rises from %.4f to 1, not a region where the drive is safe."
            % (nomrow["cosine"]["W_apex"], NU_WINDOW_HALFWIDTH_BULK,
               nomrow["cosine"]["W_apex"])),
        safe_drive_rule=(
            "ABS-corrected safe-drive rule (replaces the bulk margin "
            "2 Delta/h f_d = 1.0139): to hold W(apex) < 1e-3 the drive plane "
            "must sit at f_d <= %.3f GHz (cosine) / %.3f GHz (exact CPR), i.e. "
            "%.2f-%.2f%% below 2 Delta/h -- margin 1.182-1.210, not 1.014."
            % (nomrow["cosine"]["f_safe_GHz"]["1e-03"],
               nomrow["exact_cpr"]["f_safe_GHz"]["1e-03"],
               nomrow["cosine"]["f_safe_pct_below_2Delta"]["1e-03"],
               nomrow["exact_cpr"]["f_safe_pct_below_2Delta"]["1e-03"])),
        scope=(
            "DRIVE-PLANE ONLY, exactly as the bulk audit found. The control "
            "(40.597 GHz) and readout (44.597 / 47.597 GHz) tones do not open "
            "the ABS channel at all for tau <= 0.8 (they need tau >= 0.924 / "
            "0.908 / 0.895); at tau = 0.95 their W is 1.7e-11 / 5.4e-11 / "
            "1.2e-10, at the charge-basis tail floor. The C5 fork's STRUCTURE "
            "survives; its NUMBER does not."),
        fork_consequence=(
            "qubit.json:allocation.fixed_drive already reads 'EXACTLY 1 "
            "CHANNEL'. The ABS audit does not add a channel -- it removes the "
            "last one: at 145 GHz there is no ABS-quiet filling at all, so the "
            "fixed-drive scenario is not 'one clean qubit', it is 'one qubit "
            "with an open, unbounded junction-level pair-breaking channel'. "
            "The tracking-drive scenario is unaffected only if each per-qubit "
            "tone also respects f_d < f_safe(tau, nu), which is a STRICTER "
            "constraint than h f_d < 2 Delta(nu)."),
        claim_limit=(
            "W is a WEIGHT, not a rate. It says the channel is open and how "
            "much ground-state phase density is on the open side; it does not "
            "say how fast. See parametric_rate for the free-gamma_0 "
            "break-even map, and note that this audit is the ABS analogue of "
            "the already-flagged qubit.json:open_items."
            "photon_assisted_pair_breaking -- same register, different "
            "mechanism."),
    )

    part_A = dict(
        verdict=verdict_A,
        question=("is the 1.370% bulk margin of the 145 GHz drive an artifact "
                  "of the BULK gap? 2 E_A(phi) = 2 Delta sqrt(1 - tau "
                  "sin^2(phi/2)) is the junction's own pair-transition energy"),
        definition=dict(
            E_A="E_A(phi, nu) = Delta(nu) sqrt(1 - tau sin^2(phi/2))",
            W="W(tau, nu, f_d) = int dphi |psi0(phi)|^2 [2 E_A(phi,nu) < h f_d]",
            interpretation=("ground-state phase weight sitting where the drive "
                            "photon is ABOVE the local Andreev pair-transition "
                            "energy; a threshold+weight audit, NOT a rate"),
            gap_convention_primary="dome (reproduces the stored bulk window)",
        ),
        regression_check=regression,
        bulk_reference=bulk,
        sigma_phi_harmonic_rad=sigma_harmonic,
        sigma_phi_cosine_basis_rad=dist["cosine"].sigma,
        rows_by_tau=rows,
        W_vs_nu_at_tau_nominal=profile,
        tau_trend=trend,
        tones=tones,
        parametric_rate=gamma_block,
        drive_plane_redesign=redesign,
        abs_cpr_consistency={str(t): abs_cpr_consistency(t)
                             for t in TAUS_MATERIAL},
    )

    # ================================ PART B: ABS-weighted qp matrix elements =
    partB_rows = {}
    for tau in TAUS_MATERIAL:
        wgt = {p: (1.0 - tau * np.sin(phi / 2) ** 2) ** (-p)
               for p in P_EXPONENTS}
        ent = {}
        for bname, key in (("cosine", "cosine"),
                           ("exact_cpr", "cpr_tau%s" % tau)):
            s0, c0, _ = brackets(phi, dphi, psi[key])
            sub = dict(sin_half_01_sq_unweighted=s0,
                       cos_half_00_sq_unweighted=c0,
                       sin01_unweighted_over_stored=float(
                           s0 / ME_STORED["sin_half_01_sq"]),
                       cos00_unweighted_over_stored=float(
                           c0 / ME_STORED["cos_half_00_sq"]))
            for p in P_EXPONENTS:
                s, c, w00 = brackets(phi, dphi, psi[key], wgt[p])
                sub["p%.2f" % p] = dict(
                    weight_expectation_0W0=w00,
                    sin_half_01_sq_A=s, cos_half_00_sq_A=c,
                    sin_half_01_sq_B=float(s / w00 ** 2),
                    cos_half_00_sq_B=float(c / w00 ** 2),
                    ratio_sin_A=float(s / s0), ratio_sin_B=float(s / s0 / w00 ** 2),
                    ratio_cos_A=float(c / c0), ratio_cos_B=float(c / c0 / w00 ** 2),
                    ratio_sin_A_vs_stored=float(
                        s / ME_STORED["sin_half_01_sq"]),
                    ratio_sin_B_vs_stored=float(
                        s / w00 ** 2 / ME_STORED["sin_half_01_sq"]),
                    analytic_leading_order_ratio_A=float(
                        1.0 + 1.5 * p * tau * np.sqrt(2 * EC_GHZ / EJ_GHZ)),
                )
            ent[bname] = sub
        partB_rows[str(tau)] = ent

    # --- T1 rescale (primary: cosine basis, so the exported kernel still holds)
    nom = partB_rows[str(TAU_NOMINAL)]
    ratios_primary = {}
    for p in P_EXPONENTS:
        k = "p%.2f" % p
        ratios_primary["%s_convA" % k] = nom["cosine"][k]["ratio_sin_A"]
        ratios_primary["%s_convB" % k] = nom["cosine"][k]["ratio_sin_B"]
    band = (min(ratios_primary.values()), max(ratios_primary.values()))
    t1_rescaled = {}
    for lab, R in (("lower_bound_R_%.6f" % band[1], band[1]),
                   ("upper_bound_R_%.6f" % band[0], band[0])):
        t1_rescaled[lab] = {k: float(v / R) for k, v in T1_LADDER_US.items()}

    ratios_cpr = {}
    for p in P_EXPONENTS:
        k = "p%.2f" % p
        ratios_cpr["%s_convA" % k] = nom["exact_cpr"][k]["ratio_sin_A_vs_stored"]
        ratios_cpr["%s_convB" % k] = nom["exact_cpr"][k]["ratio_sin_B_vs_stored"]
    band_cpr = (min(ratios_cpr.values()), max(ratios_cpr.values()))

    # kernel-shape factor at the CPR-basis f01, so the two effects can be
    # combined by the reader without re-deriving anything.
    f01_cpr = PEXTRA["fixed_Ic"][str(TAU_NOMINAL)]["f01_GHz"]
    # src: out/m6/pextra_cpr.json:fixed_Ic["0.8"].f01_GHz = 35.26207337857237
    f01_ref = float(spectrum.levels(EJ_GHZ, EC_GHZ, N_CHARGE)[0])
    w_ref = qp_kernel.marginality(f01_ref)
    w_cpr = qp_kernel.marginality(f01_cpr)
    S_ratio = float((np.sqrt(2.0 / w_cpr) * qp_kernel.kernel_shape_ratio_T0(w_cpr))
                    / (np.sqrt(2.0 / w_ref) * qp_kernel.kernel_shape_ratio_T0(w_ref)))

    verdict_B = dict(
        headline=(
            "THE P3 BRACKETS SURVIVE, BUT ONLY AS A LOWER BOUND. In the design "
            "(cosine) eigenbasis the ABS coherence-factor weight raises "
            "|<0|sin(phi/2)|1>|^2 from the exported %.11e by a factor %.4f "
            "(p=1/4, normalised) to %.4f (p=1/2, bare) at tau = 0.8. The "
            "parity bracket |<0|cos(phi/2)|0>|^2 moves by at most %.2f%%. "
            "Relaxation is FASTER: the exported T1 ladder shortens by 3.0-8.7%%."
            % (ME_STORED["sin_half_01_sq"], band[0], band[1],
               100.0 * (nom["cosine"]["p0.50"]["ratio_cos_A"] - 1.0))),
        T1_statement=(
            "T1(x_qp = 1e-6, T_qp -> 0) = %.4f us -> %.4f .. %.4f us; the "
            "total correction against the milestone-1 small-frequency form "
            "grows from the booked x%.6f to x%.4f .. x%.4f."
            % (T1_LADDER_US["1e-06"], T1_LADDER_US["1e-06"] / band[1],
               T1_LADDER_US["1e-06"] / band[0], KERNEL_CORRECTION,
               KERNEL_CORRECTION * band[0], KERNEL_CORRECTION * band[1])),
        larger_sibling_effect=(
            "A BIGGER effect hides next to it, and it is not the weight: the "
            "BARE bracket evaluated in the exact-CPR eigenbasis is %.4f times "
            "the exported cosine-basis value, because the skewed CPR softens "
            "the well (A2 = 0.733) and widens sigma_phi from %.4f to %.4f rad. "
            "Weight and basis together give %.4f .. %.4f. P-extra computed the "
            "CPR spectrum but never the qp matrix element, so this number is "
            "new -- and it cannot be applied to the exported ladder alone, "
            "because the same basis moves f01 to 35.262 GHz."
            % (nom["exact_cpr"]["sin01_unweighted_over_stored"],
               dist["cosine"].sigma, dist["cpr_tau%s" % TAU_NOMINAL].sigma,
               band_cpr[0], band_cpr[1])),
        claim_limit=(
            "the exponent p is a MODEL CHOICE. Both p = 1/4 and p = 1/2 and "
            "both normalisation conventions are quoted; nothing here is a "
            "single number. Everything stays parametric in x_qp."),
    )

    part_B = dict(
        verdict=verdict_B,
        model=dict(
            weight="(1 - tau sin^2(phi/2))^(-p) = (Delta/E_A(phi))^(2p)",
            p_values=list(P_EXPONENTS),
            status="MODEL CHOICE, not a measured input",
            citation=("Catelani, Schoelkopf, Devoret, Glazman, PRB 84, 064517 "
                      "(2011); Glazman & Catelani, SciPost Phys. Lect. Notes "
                      "31 (2021) -- the high-transparency ABS vertex carries "
                      "coherence factors ~ Delta/E_A(phi); the exponent p is "
                      "left open and both 1/4 and 1/2 are quoted"),
            convention_A=("bare, anchored at the potential minimum: weight(0) "
                          "= 1, so the tau -> 0 limit returns the exported "
                          "tunnel-limit bracket exactly"),
            convention_B=("ground-state normalised: weight / <0|weight|0>, "
                          "which removes the trivial inflation from weight >= "
                          "1 everywhere and isolates the SHAPE correlation"),
        ),
        rows_by_tau=partB_rows,
        stored_reference=dict(
            sin_half_01_sq=ME_STORED["sin_half_01_sq"],
            cos_half_00_sq=ME_STORED["cos_half_00_sq"],
            kernel_correction_already_booked=KERNEL_CORRECTION,
            src="qubit.json:budgets.T1_stack.qp_full_kernel"),
        headline_tau_nominal=dict(
            tau=TAU_NOMINAL,
            ratios_cosine_basis=ratios_primary,
            ratio_band_cosine_basis=list(band),
            T1_us_rescaled_cosine_basis=t1_rescaled,
            T1_us_stored=T1_LADDER_US,
            total_correction_vs_smallfreq_band=[
                float(KERNEL_CORRECTION * band[0]),
                float(KERNEL_CORRECTION * band[1])],
        ),
        exact_cpr_basis=dict(
            bare_element_ratio_vs_stored=float(
                nom["exact_cpr"]["sin01_unweighted_over_stored"]),
            weighted_ratio_band_vs_stored=list(band_cpr),
            f01_GHz_cpr_basis=float(f01_cpr),
            f01_GHz_design=f01_ref,
            marginality_w_design=float(w_ref),
            marginality_w_cpr_basis=float(w_cpr),
            S_qp_ratio_T0_cpr_over_design=S_ratio,
            caveat=("the exact-CPR eigenbasis is NOT a free ride: it also "
                    "moves f01 to 35.262 GHz (out/m6/pextra_cpr.json, the "
                    "deferred qubit.json:open_items.cpr_harmonics flag), which "
                    "moves S_qp by the factor above. These CPR-basis numbers "
                    "therefore cannot be applied to the exported T1 ladder "
                    "without also adopting the P-extra spectrum. Quoted here "
                    "because the qp MATRIX ELEMENT in the CPR basis is a "
                    "number P-extra never computed."),
        ),
    )

    # ============================================================ validation ==
    conv_phi, conv_dphi, conv_psi = wavefunctions(H_cos, N_PHI_CONV)
    d_conv = PhaseDist(conv_phi, conv_dphi, conv_psi[:, 0])
    H_N = spectrum.hamiltonian(EJ_GHZ, EC_GHZ, 0.0, N_CONV)
    _, _, psi_N = wavefunctions(H_N, N_PHI, N_CONV)
    d_N = PhaseDist(phi, dphi, psi_N[:, 0])
    pt_nom = phi_threshold(F_DRIVE_GHZ, dome.NU_OPT, TAU_NOMINAL, "dome")
    wgt_nom = (1.0 - TAU_NOMINAL * np.sin(phi / 2) ** 2) ** (-0.5)
    wgt_conv = (1.0 - TAU_NOMINAL * np.sin(conv_phi / 2) ** 2) ** (-0.5)
    s_nom, _, _ = brackets(phi, dphi, psi_cos, wgt_nom)
    s_conv, _, _ = brackets(conv_phi, conv_dphi, conv_psi, wgt_conv)

    validation = dict(
        i_regression_brackets=dict(
            digits_sin=regression["digits_sin"],
            digits_cos=regression["digits_cos"],
            pass_ge_10_digits=bool(regression["digits_sin"] >= 10.0
                                   and regression["digits_cos"] >= 10.0)),
        ii_W_to_zero_as_tau_to_zero=dict(
            tau_min_channel_open=trend["tau_min_channel_open"],
            W_at_tau_0p01=trend["W_apex_cosine"][0],
            W_at_tau_0p02=trend["W_apex_cosine"][1],
            statement=("W is identically 0 for tau <= 1 - (f_d/f_2Delta)^2 = "
                       "%.6f: below that transparency no phase satisfies "
                       "2 E_A(phi) < h f_d" % trend["tau_min_channel_open"]),
            pass_=bool(trend["W_apex_cosine"][0] == 0.0
                       and trend["W_apex_cosine"][1] == 0.0)),
        iii_basis_consistency=dict(
            W_apex_cosine=rows[str(TAU_NOMINAL)]["cosine"]["W_apex"],
            W_apex_exact_cpr=rows[str(TAU_NOMINAL)]["exact_cpr"]["W_apex"],
            spread_pct=rows[str(TAU_NOMINAL)]["W_apex_basis_spread_pct"],
            band=rows[str(TAU_NOMINAL)]["W_apex_band"],
            pass_same_order=bool(
                0.5 < rows[str(TAU_NOMINAL)]["exact_cpr"]["W_apex"]
                / rows[str(TAU_NOMINAL)]["cosine"]["W_apex"] < 2.0)),
        iv_tau_monotone=dict(monotone=trend["monotone_nondecreasing"],
                             W=[rows[str(t)]["cosine"]["W_apex"]
                                for t in TAUS_MATERIAL]),
        v_bulk_window_reproduced=dict(
            recomputed=bulk["nu_window_halfwidth_recomputed"],
            stored=bulk["nu_window_halfwidth_stored"],
            reldev=bulk["nu_window_halfwidth_reldev"],
            pass_lt_1e_12=bool(bulk["nu_window_halfwidth_reldev"] < 1e-12)),
        vi_abs_generates_stored_cpr=dict(
            max_abs_residual={str(t): abs_cpr_consistency(t)["max_abs_residual"]
                              for t in TAUS_MATERIAL},
            statement=("eq. (A) differentiated gives material.json's stored "
                       "Beenakker harmonics; the residual is the documented "
                       "5-harmonic truncation (qubit/cpr_transmon.py Sec. 4), "
                       "not a model mismatch")),
        vii_grid_and_truncation_convergence=dict(
            W_apex_n_phi_8192=dist["cosine"].W(pt_nom),
            W_apex_n_phi_32768=d_conv.W(pt_nom),
            W_apex_reldev_grid=float(
                abs(d_conv.W(pt_nom) - dist["cosine"].W(pt_nom))
                / dist["cosine"].W(pt_nom)),
            W_apex_N35=dist["cosine"].W(pt_nom),
            W_apex_N50=d_N.W(pt_nom),
            W_apex_reldev_N=float(abs(d_N.W(pt_nom) - dist["cosine"].W(pt_nom))
                                  / dist["cosine"].W(pt_nom)),
            weighted_sin01_n_phi_8192=s_nom,
            weighted_sin01_n_phi_32768=s_conv,
            weighted_sin01_reldev=float(abs(s_conv - s_nom) / s_nom)),
        viii_wavefunction_tail_floor=dict(
            p0_at_minus_pi=float(dist["cosine"].p[0]),
            note=("|psi0|^2 at the branch edge is 4.4e-12, so W values below "
                  "~1e-10 are at the charge-basis tail floor and are reported "
                  "as bounds, not measurements")),
    )

    params = dict(
        EJ_GHz=float(EJ_GHZ), EC_GHz=float(EC_GHZ),
        EJ_over_EC=TARGETS["design"]["EJ_over_EC"],
        N_charge=N_CHARGE, N_charge_conv=N_CONV,
        n_phi=N_PHI, n_phi_conv=N_PHI_CONV,
        f_drive_GHz=float(F_DRIVE_GHZ),
        two_Delta_over_h_GHz=float(TWO_DELTA_GHZ),
        Delta0_meV=float(DELTA0_MEV),
        Delta_apex_dome_meV=float(dome.Delta_of(dome.NU_OPT)),
        nu_apex=float(dome.NU_OPT), nu_halfwidth=float(dome.NU_HALFWIDTH),
        taus_material=list(TAUS_MATERIAL), tau_nominal=TAU_NOMINAL,
        W_targets=list(W_TARGETS), p_exponents=list(P_EXPONENTS),
        n_drive_range=list(map(float, N_DRIVE_RANGE)),
        hbar_over_h=float(HBAR / H),
    )

    provenance = dict(
        task="M6 exploration item 1/5 -- ABS pair-breaking audit + ABS-weighted "
             "qp matrix elements",
        script="scripts/explore_abs_pair_breaking_audit.py",
        deterministic=True, rng="unused (no stochastic step anywhere)",
        seed=None,
        reads={
            "EJ/EC/N_charge": "qubit/inputs.py (junction.json:contract.Ic_nA, "
                              "qubit_targets.json:design.EJ_over_EC)",
            "Delta0_meV / sc_dome": "material.json (via qubit/dome.py)",
            "cpr.harmonics_by_tau / tau_nominal": "material.json:cpr",
            "f_cav_GHz": "qubit_targets.json:inputs.f_cav_GHz",
            "two_Delta_over_h": "qubit/qp_kernel.py:two_delta_GHz()",
            "matrix_elements / T1 ladder / kernel_correction":
                "qubit.json:budgets.T1_stack.qp_full_kernel",
            "bulk nu window / fork": "qubit.json:allocation.fixed_drive",
            "n_drive": "qubit.json:two_cavity.control_line.rabi_budget",
            "a_um_precise": "photonics.json:geometry.a_um_precise",
            "f01 in the CPR basis": "out/m6/pextra_cpr.json:fixed_Ic['0.8']",
            "p3 matrix elements": "out/m6/p3_qp.json:kernel.matrix_elements",
        },
        package_functions_used=[
            "qubit.spectrum.hamiltonian", "qubit.spectrum.levels",
            "qubit.cpr_transmon.hamiltonian", "qubit.cpr_transmon.harmonics",
            "qubit.cpr_transmon.cpr_current", "qubit.cpr_transmon.cpr_argmax",
            "qubit.dome.Delta_of", "qubit.qp_kernel.matrix_elements",
            "qubit.qp_kernel.two_delta_GHz", "qubit.qp_kernel.marginality",
            "qubit.qp_kernel.kernel_shape_ratio_T0",
        ],
        modified_files="none -- qubit/, tests/, paper/, figs/ untouched",
        claim_discipline=(
            "threshold + weight audit only. Gamma_ABS is quoted parametrically "
            "with a FREE gamma_0; the ABS coherence-factor exponent p is a "
            "model choice quoted as a band over p and over two normalisation "
            "conventions and two eigenbases. No absolute TBG coherence claim."),
    )

    return dict(_meta=dict(
        title="Junction-level ABS physics: 2 E_A(phi) pair-breaking audit of "
              "the 145 GHz drive + ABS-weighted quasiparticle matrix elements",
        module="M6 exploration (paper 2), item 1/5",
        status="EXPLORATION -- not a contract change; nothing in qubit/ moved"),
        part_A_threshold_weight=part_A, part_B_qp_operator=part_B,
        validation=validation, params=params, provenance=provenance)


def main():
    payload = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)

    A, B = payload["part_A_threshold_weight"], payload["part_B_qp_operator"]
    r = A["regression_check"]
    print("regression: sin01 %.11e (%.1f digits)  cos00 %.11e (%.1f digits)"
          % (r["sin_half_01_sq_here"], r["digits_sin"],
             r["cos_half_00_sq_here"], r["digits_cos"]))
    print("bulk: 2D/h = %.5f GHz, 145 GHz is %.4f%% below; window +-%.9f "
          "(stored %.9f)"
          % (A["bulk_reference"]["two_Delta_over_h_GHz"],
             A["bulk_reference"]["pct_below_threshold"],
             A["bulk_reference"]["nu_window_halfwidth_recomputed"],
             A["bulk_reference"]["nu_window_halfwidth_stored"]))
    print("sigma_phi = %.6f rad (harmonic %.6f)"
          % (A["sigma_phi_cosine_basis_rad"], A["sigma_phi_harmonic_rad"]))
    print("--- PART A: sub-threshold weight at the apex, f_d = 145 GHz ---")
    for t, row in A["rows_by_tau"].items():
        print("  tau=%-5s phi_thr=%.6f rad (%.3f sigma)  W = %.6f (cos) / "
              "%.6f (exact CPR)  f_safe(1e-3) = %.4f / %.4f GHz"
              % (t, row["phi_threshold_rad"],
                 row["cosine"]["phi_threshold_over_sigma"],
                 row["cosine"]["W_apex"], row["exact_cpr"]["W_apex"],
                 row["cosine"]["f_safe_GHz"]["1e-03"],
                 row["exact_cpr"]["f_safe_GHz"]["1e-03"]))
    print("  W < {1e-3,1e-2,1e-1} nu-windows at tau=0.8 (cosine): %s"
          % A["rows_by_tau"][str(TAU_NOMINAL)]["cosine"]["nu_at_W"])
    print("  drive-plane redesign: a %.3f -> %.3f / %.3f um (+%.2f/%.2f%%)"
          % (A["drive_plane_redesign"]["a_um_current"],
             A["drive_plane_redesign"]["a_um_required_cosine"],
             A["drive_plane_redesign"]["a_um_required_exact_cpr"],
             A["drive_plane_redesign"]["a_growth_pct_band"][0],
             A["drive_plane_redesign"]["a_growth_pct_band"][1]))
    print("--- PART B: ABS-weighted qp brackets, tau = %s ---" % TAU_NOMINAL)
    h = B["headline_tau_nominal"]
    print("  cosine-basis ratios: %s" % h["ratios_cosine_basis"])
    print("  T1(x_qp=1e-6): stored %.4f us -> %.4f .. %.4f us"
          % (h["T1_us_stored"]["1e-06"],
             h["T1_us_stored"]["1e-06"] / h["ratio_band_cosine_basis"][1],
             h["T1_us_stored"]["1e-06"] / h["ratio_band_cosine_basis"][0]))
    print("  exact-CPR basis bare element ratio vs stored: %.6f; weighted "
          "band %.6f .. %.6f"
          % (B["exact_cpr_basis"]["bare_element_ratio_vs_stored"],
             B["exact_cpr_basis"]["weighted_ratio_band_vs_stored"][0],
             B["exact_cpr_basis"]["weighted_ratio_band_vs_stored"][1]))
    print("--- VERDICTS ---")
    for k in ("headline", "window_verdict", "safe_drive_rule", "scope"):
        print("  A/%s: %s" % (k, A["verdict"][k]))
    for k in ("headline", "T1_statement", "larger_sibling_effect"):
        print("  B/%s: %s" % (k, B["verdict"][k]))
    print("  gamma_0 break-even (tau=0.8): %.3e .. %.3e s^-1 "
          "(%.2e .. %.2e of omega_drive)"
          % (A["parametric_rate"]["rows"]["0.8"]["W_hi"]["n_drive_hi"]
             ["gamma0_breakeven_s"],
             A["parametric_rate"]["rows"]["0.8"]["W_lo"]["n_drive_lo"]
             ["gamma0_breakeven_s"],
             A["parametric_rate"]["rows"]["0.8"]["W_hi"]["n_drive_hi"]
             ["gamma0_breakeven_over_omega_drive"],
             A["parametric_rate"]["rows"]["0.8"]["W_lo"]["n_drive_lo"]
             ["gamma0_breakeven_over_omega_drive"]))
    print("validation: %s" % {k: (v.get("pass_") or v.get("pass_ge_10_digits")
                                  or v.get("pass_lt_1e_12")
                                  or v.get("pass_same_order")
                                  or v.get("monotone"))
                              for k, v in payload["validation"].items()})
    print("wrote %s" % OUT)
    return payload


if __name__ == "__main__":
    main()
