"""P-extra -- the measured non-sinusoidal CPR inside the charge-basis transmon.

Resolves the M6 deferred flag (paper/decisions.md, 2026-07-27 "M6 flag
(deferred): non-sinusoidal CPR vs transmon Hamiltonian").  ``qubit/spectrum.py``
is NOT touched: it keeps the contract's pure -E_J cos(phi) potential and stays
bit-faithful for AT-1..AT-6.  This module adds the multi-harmonic potential
alongside it and measures the sensitivity.

=============================================================================
1.  NORMALIZATION CHAIN  (the part that is easy to get wrong)
=============================================================================

material.json:cpr carries the Beenakker short-junction harmonics under the M3
convention (material.json:cpr.model = "Beenakker short junction, T=0.1 K,
normalized max=1", and junction/cpr.py:normalize_cpr):

    I(phi) = Ic * sum_{m=1..M} c_m sin(m phi),
    with the c_m rescaled so that     max_phi sum_m c_m sin(m phi) = 1.        (N)

(N) is what makes Ic the TRUE critical current: max_phi I(phi) = Ic.  It is
asserted at import-time cost zero by `normalization_check()` and is an
acceptance test here -- the harmonics in material.json are ALREADY renormalized
(measured residual |max - 1| < 5e-8, set by the 8192-point grid on which M1
performed the rescaling).

The Josephson energy follows by integrating I dphi with the Josephson relation
V = (Phi0/2pi) dphi/dt:

    U(phi) = (Phi0/2pi) integral I(phi') dphi'
           = -E_J sum_m (c_m/m) cos(m phi) + const,                           (U)
    E_J = Phi0*Ic/(2 pi) = hbar*Ic/(2e),

i.e. E_J is fixed by the SAME Ic that fixes the c_m through (N).  E_J is taken
straight from qubit/inputs.py:EJ_GHZ (= Phi0*Ic/(2 pi h), Ic from
junction.json:contract.Ic_nA) -- no new number is introduced anywhere here.

Setting c = [1.0] in (U) returns exactly -E_J cos(phi), so `hamiltonian` with
c = [1.0] is bit-identical to `spectrum.hamiltonian` (asserted in the tests).

Two invariants of (U) that explain every number below:

    A2 = sum_m m   c_m  = U''(0)/E_J        (curvature -> E_J_eff = A2 * E_J)
    A4 = sum_m m^3 c_m  = -U''''(0)/E_J     (quartic -> anharmonicity)

For a forward-skewed CPR the peak sits at phi > pi/2, so after normalization
(N) the zero-phase slope A2 < 1: at FIXED Ic a skewed junction is a WEAKER
junction at phi = 0.  Measured A2 = 0.8538 / 0.7330 / 0.6742 for
tau = 0.5 / 0.8 / 0.95.

=============================================================================
2.  CHARGE BASIS
=============================================================================

cos(m phi) = 1/2 sum_n (|n><n+m| + |n+m><n|), so (U) is a symmetric BANDED
matrix of bandwidth M (M = 5 for material.json) with the m-th off-diagonal
equal to -E_J c_m/(2m):

    H = 4 E_C (n - n_g)^2  -  sum_m (E_J c_m / 2m) (|n><n+m| + h.c.)

The pure-cosine case M = 1 is the tridiagonal spectrum.py matrix.  Because the
matrix is no longer tridiagonal, spectrum.eps1_mp's Sturm-sequence bisection
does not apply; the extended-precision path here is mpmath's symmetric
Householder+QL eigensolver (`eps1_mp`), which is 1.5 s per (N=35, dps=50)
matrix -- cheap enough that it is run for every exported row rather than only
below 1 mHz.

=============================================================================
3.  WHAT IS HELD FIXED  (the comparison is meaningless without this)
=============================================================================

(a) FIXED-Ic  -- the physical scenario.  Ic (junction.json) and C_Sigma
    (qubit_targets.json:design.C_fF, a pad-geometry number the CPR cannot
    touch) are both held.  Hence E_J and E_C are both exactly the contract
    values and the whole spectrum moves.  This is what the device does.

(b) RETUNED   -- what an experimentalist does next.  E_J -> lambda * E_J with
    lambda solved so f01 returns to the design 40.5966287014581 GHz, E_C held.
    Reported because it isolates the part of the shift that CANNOT be tuned
    away.  Caveat stated in the export: lambda > 1, and E_J(nu) is already
    MAXIMAL at the dome apex, so lambda is a fabrication knob (wider junction
    -> larger Ic) or a shunt knob (smaller C_Sigma -> larger E_C, not done
    here), not a gate knob.

=============================================================================
4.  HARMONIC-TRUNCATION DIAGNOSTIC
=============================================================================

material.json stores M = 5 harmonics.  A2 converges quickly but A4 does not
(m^3 c_m is alternating and only weakly decaying), so the truncation is checked
by REGENERATING the Beenakker CPR with more harmonics under the identical
normalization rule (m1/cpr.py:beenakker_harmonics, the very code that produced
material.json -- its nH=5 output reproduces material.json to 2.8e-17).  This is
a diagnostic, not an input: every exported headline number comes from the
5-harmonic material.json list.
"""
import json

import numpy as np

from . import dome, spectrum
from .inputs import (EJ_GHZ, IC_A, MATERIAL, N_CHARGE, ROOT, TARGETS,
                     ANHARM_MIN_GHZ, EPS1_MAX_HZ, RATIOS)

# ------------------------------------------------------------ measured CPR ---
CPR_MODEL = MATERIAL["cpr"]["model"]
# src: material.json:cpr.model = "Beenakker short junction, T=0.1 K, normalized max=1"

TAU_NOMINAL = MATERIAL["cpr"]["tau_nominal"]
# src: material.json:cpr.tau_nominal = 0.8

HARMONICS_BY_TAU = MATERIAL["cpr"]["harmonics_by_tau"]
# src: material.json:cpr.harmonics_by_tau  -- keys "0.5", "0.8", "0.95",
# each a list c_1..c_5 already normalized per (N).

C_SINUSOIDAL = [1.0]
# The contract CPR: I = Ic sin(phi), U = -E_J cos(phi).  Not a measurement.

EC_DESIGN_GHZ = TARGETS["design"]["EC_GHz"]
# src: qubit_targets.json:design.EC_GHz  (E_J/E_C = 100 design lock)

F01_DESIGN_GHZ = TARGETS["design"]["f01_GHz"]
# src: qubit_targets.json:design.f01_GHz = 40.5966287014581 (retune target)

N_CONV = 60
# Convergence check ceiling required by the P-extra task (N >= 35 -> 60).

MP_DPS = 50
# mpmath working precision for the banded extended-precision path.  eps1/||H||
# ~ 1e-11 here, so 50 digits leaves ~39 spare.


def harmonics(tau):
    """c_1..c_M for transparency `tau` from material.json:cpr.harmonics_by_tau."""
    return np.asarray(HARMONICS_BY_TAU[str(tau)], float)


def taus_available():
    """Transparency rows present in material.json, ascending."""
    return sorted(HARMONICS_BY_TAU, key=float)


# ------------------------------------------------------- normalization (N) ---
def cpr_current(phi, c):
    """I(phi)/Ic = sum_m c_m sin(m phi)  (junction/cpr.py convention)."""
    phi = np.asarray(phi, float)
    f = np.zeros_like(phi)
    for m, cm in enumerate(c, start=1):
        if cm != 0.0:
            f = f + cm * np.sin(m * phi)
    return f


def cpr_max(c, npts=400001):
    """max_phi sum_m c_m sin(m phi) -- must equal 1 under convention (N)."""
    phi = np.linspace(0.0, 2.0 * np.pi, npts)
    return float(cpr_current(phi, c).max())


def cpr_argmax(c, npts=400001):
    """phi at which the CPR peaks, in units of pi (pi/2 for a pure sine)."""
    phi = np.linspace(0.0, 2.0 * np.pi, npts)
    return float(phi[np.argmax(cpr_current(phi, c))] / np.pi)


def A2(c):
    """U''(0)/E_J = sum_m m c_m  -- the effective-E_J reduction factor."""
    return float(sum(m * cm for m, cm in enumerate(c, start=1)))


def A4(c):
    """-U''''(0)/E_J = sum_m m^3 c_m  (quartic; slow to converge in M)."""
    return float(sum(m ** 3 * cm for m, cm in enumerate(c, start=1)))


def potential(phi, c):
    """U(phi)/E_J = -sum_m (c_m/m) cos(m phi)   [eq. (U)]."""
    phi = np.asarray(phi, float)
    u = np.zeros_like(phi)
    for m, cm in enumerate(c, start=1):
        if cm != 0.0:
            u = u - cm / m * np.cos(m * phi)
    return u


def potential_minima(c, npts=200001):
    """Interior local minima of U on (-pi, pi], in units of pi.

    A second minimum would break the single-well transmon picture; measured
    result for every material.json row is the single minimum at phi = 0.
    """
    phi = np.linspace(-np.pi, np.pi, npts)
    u = potential(phi, c)
    turn = np.diff(np.sign(np.diff(u)))
    return (phi[1:-1][turn > 0] / np.pi).tolist()


def normalization_check(cs=None):
    """Assert-ready record that material.json's harmonics obey (N)."""
    if cs is None:
        cs = {t: harmonics(t) for t in taus_available()}
    out = {}
    for tau, c in cs.items():
        mx = cpr_max(c)
        out[str(tau)] = dict(cpr_max=mx, abs_dev_from_1=abs(mx - 1.0),
                             argmax_over_pi=cpr_argmax(c),
                             A2=A2(c), A4=A4(c),
                             potential_minima_over_pi=potential_minima(c))
    return out


# ---------------------------------------------------------- charge basis -----
def hamiltonian(EJ, EC, c=C_SINUSOIDAL, ng=0.0, N=N_CHARGE):
    """Banded charge-basis Hamiltonian in GHz, shape (2N+1, 2N+1).

    c = [1.0] reproduces spectrum.hamiltonian bit-for-bit.
    """
    n = np.arange(-N, N + 1)
    Hm = np.diag(4.0 * EC * (n - ng) ** 2)
    for m, cm in enumerate(c, start=1):
        if cm == 0.0 or m > 2 * N:
            continue
        off = -EJ * cm / (2.0 * m) * np.ones(2 * N + 1 - m)
        Hm = Hm + np.diag(off, m) + np.diag(off, -m)
    return Hm


def transmon(EJ, EC, c=C_SINUSOIDAL, ng=0.0, N=N_CHARGE, nlev=4):
    """Lowest `nlev` eigenvalues (GHz) at offset charge `ng`."""
    return np.linalg.eigvalsh(hamiltonian(EJ, EC, c, ng, N))[:nlev]


def levels(EJ, EC, c=C_SINUSOIDAL, N=N_CHARGE):
    """(f01, anharmonicity, eps1) in GHz -- signature mirrors spectrum.levels."""
    ev0 = transmon(EJ, EC, c, 0.0, N)
    ev5 = transmon(EJ, EC, c, 0.5, N)
    f01, f12 = ev0[1] - ev0[0], ev0[2] - ev0[1]
    eps1 = abs((ev5[1] - ev5[0]) - f01)
    return f01, f12 - f01, eps1


def n01_element(EJ, EC, c=C_SINUSOIDAL, N=N_CHARGE):
    """|<0|n|1>| at n_g = 0 -- the Cooper-pair matrix element that sets g."""
    n = np.arange(-N, N + 1)
    _, v = np.linalg.eigh(hamiltonian(EJ, EC, c, 0.0, N))
    return abs(v[:, 0] @ (n * v[:, 1]))


def row(EJ, EC, c=C_SINUSOIDAL, N=N_CHARGE, with_mp=True, dps=MP_DPS):
    """Full design row for one CPR: f01, anharm, n01, eps1 (float64 + mpmath)."""
    f01, anh, eps1 = levels(EJ, EC, c, N)
    d = dict(EJ_GHz=float(EJ), EC_GHz=float(EC), EJ_over_EC=float(EJ / EC),
             f01_GHz=float(f01), anharm_GHz=float(anh),
             n01=float(n01_element(EJ, EC, c, N)),
             eps1_Hz=float(eps1 * 1e9), N=int(N))
    if with_mp:
        d["eps1_mp_Hz"] = eps1_mp(EJ, EC, c, N, dps)
        d["eps1_float64_reldev"] = (d["eps1_Hz"] - d["eps1_mp_Hz"]) / d["eps1_mp_Hz"]
    return d


# ------------------------------------------- extended precision (banded) -----
def _hamiltonian_mp(EJ, EC, c, ng, N):
    import mpmath as mp
    EJm, ECm, ngm = mp.mpf(repr(EJ)), mp.mpf(repr(EC)), mp.mpf(repr(ng))
    A = mp.zeros(2 * N + 1)
    for i, n in enumerate(range(-N, N + 1)):
        A[i, i] = 4 * ECm * (mp.mpf(n) - ngm) ** 2
    for m, cm in enumerate(c, start=1):
        if cm == 0.0 or m > 2 * N:
            continue
        v = -EJm * mp.mpf(repr(float(cm))) / (2 * m)
        for i in range(2 * N + 1 - m):
            A[i, i + m] = v
            A[i + m, i] = v
    return A


def _levels_mp(EJ, EC, c, ng, N, dps, nlev=3):
    import mpmath as mp
    mp.mp.dps = dps
    ev = mp.eigsy(_hamiltonian_mp(EJ, EC, c, ng, N), eigvals_only=True)
    return [ev[k] for k in range(nlev)]


def eps1_mp(EJ, EC, c=C_SINUSOIDAL, N=N_CHARGE, dps=MP_DPS):
    """Charge dispersion in Hz at `dps` digits, for the BANDED multi-harmonic
    matrix (mpmath symmetric eigensolver -- spectrum.eps1_mp's Sturm bisection
    is tridiagonal-only and cannot be reused)."""
    a = _levels_mp(EJ, EC, c, 0.0, N, dps)
    b = _levels_mp(EJ, EC, c, 0.5, N, dps)
    return float(abs((b[1] - b[0]) - (a[1] - a[0])) * 10 ** 9)


# ---------------------------------------------------------------- retune -----
def retune_EJ(EC, c, f01_target=F01_DESIGN_GHZ, N=N_CHARGE,
              bracket=(0.25, 4.0), iters=200, tol=1e-15):
    """lambda such that f01(lambda*E_J_apex, EC, c) = f01_target (bisection).

    f01 is monotone increasing in E_J at fixed E_C, so plain bisection on the
    bracket is exact to the float64 resolution of lambda.
    """
    lo, hi = bracket
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if levels(EJ_GHZ * mid, EC, c, N)[0] < f01_target:
            lo = mid
        else:
            hi = mid
        if hi - lo <= tol * max(1.0, lo):
            break
    return 0.5 * (lo + hi)


# ------------------------------------------------------------- sweet spot ----
def f01_of_nu(nu, EC, c=C_SINUSOIDAL, EJ_apex=EJ_GHZ, N=N_CHARGE):
    """f01(nu) with the multi-harmonic potential; mirrors dome.f01_of.

    The dome enters ONLY through E_J(nu) = E_J_apex * Delta(nu)/Delta0
    (dome.EJ_of).  material.json:nu_table carries a `cpr_harmonics` entry on
    every superconducting row and all nine are bit-identical to
    cpr.harmonics_by_tau["0.8"], so the measured harmonics do NOT resolve any
    nu dependence: they are held fixed across the dome here, which is exactly
    what the material file says.
    """
    EJ = dome.EJ_of(nu, EJ_apex)
    if EJ <= 0:
        return np.nan
    return levels(EJ, EC, c, N)[0]


def D1(nu, EC, c=C_SINUSOIDAL, dnu=dome.DNU, **kw):
    """df01/dnu, GHz per unit filling (same central difference as dome.D1)."""
    return (f01_of_nu(nu + dnu, EC, c, **kw)
            - f01_of_nu(nu - dnu, EC, c, **kw)) / (2 * dnu)


def D2(nu, EC, c=C_SINUSOIDAL, dnu=dome.DNU, **kw):
    """d^2 f01/dnu^2, GHz per unit filling^2 (same stencil as dome.D2)."""
    return (f01_of_nu(nu + dnu, EC, c, **kw)
            - 2 * f01_of_nu(nu, EC, c, **kw)
            + f01_of_nu(nu - dnu, EC, c, **kw)) / dnu ** 2


# --------------------------------------------------- truncation diagnostic ---
def truncation_scan(tau, nH_list=(5, 6, 8, 12, 20, 40), EC=None, N=N_CHARGE):
    """Regenerate the Beenakker CPR at increasing harmonic count.

    Uses m1/cpr.py:beenakker_harmonics -- the generator that produced
    material.json -- with the identical max=1 normalization applied to each
    truncation.  DIAGNOSTIC ONLY: no exported headline number uses it.
    """
    from m1.cpr import beenakker_harmonics
    EC = EC_DESIGN_GHZ if EC is None else EC
    ref = harmonics(tau)
    scan = []
    for nH in nH_list:
        c = beenakker_harmonics(float(tau), nH=nH)
        f01, anh, eps1 = levels(EJ_GHZ, EC, c, N)
        scan.append(dict(nH=int(nH), A2=A2(c), A4=A4(c), cpr_max=cpr_max(c),
                         f01_GHz=float(f01), anharm_GHz=float(anh),
                         n01=float(n01_element(EJ_GHZ, EC, c, N)),
                         eps1_Hz=float(eps1 * 1e9)))
    c5 = beenakker_harmonics(float(tau), nH=5)
    return dict(rows=scan,
                regen_matches_material_max_abs=float(np.max(np.abs(c5 - ref))),
                converged=scan[-1], material_nH=int(len(ref)))


# ------------------------------------------------------------- percentages ---
def pct(new, ref):
    """Signed percentage change of `new` relative to `ref`."""
    return 100.0 * (new - ref) / ref


def _read_m6(name):
    """Load a sibling M6 export from out/m6/ (returns None if absent)."""
    p = ROOT / "out" / "m6" / name
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


# ----------------------------------------------------------------- export ----
def build_export(N=N_CHARGE):
    """Assemble the out/m6/pextra_cpr.json payload (pure float64 + mpmath)."""
    EC = EC_DESIGN_GHZ
    taus = taus_available()

    # ---- reference: the contract's pure sinusoid, same E_J, same E_C --------
    ref = row(EJ_GHZ, EC, C_SINUSOIDAL, N)
    ref_sp = spectrum.levels(EJ_GHZ, EC, N)
    ref["cross_check_spectrum_py"] = dict(
        f01_GHz=float(ref_sp[0]), anharm_GHz=float(ref_sp[1]),
        eps1_Hz=float(ref_sp[2] * 1e9),
        n01=float(spectrum.n01_element(EJ_GHZ, EC, N)),
        max_abs_matrix_diff=float(np.max(np.abs(
            hamiltonian(EJ_GHZ, EC, C_SINUSOIDAL, 0.0, N)
            - spectrum.hamiltonian(EJ_GHZ, EC, 0.0, N)))),
        bitwise_identical=bool(np.array_equal(
            hamiltonian(EJ_GHZ, EC, C_SINUSOIDAL, 0.0, N),
            spectrum.hamiltonian(EJ_GHZ, EC, 0.0, N))),
    )

    keys = ("f01_GHz", "anharm_GHz", "n01", "eps1_Hz", "eps1_mp_Hz")

    fixed_Ic, retuned = {}, {}
    for tau in taus:
        c = harmonics(tau)
        r = row(EJ_GHZ, EC, c, N)
        r["A2"] = A2(c)
        r["A4"] = A4(c)
        r["deltas_vs_sinusoidal_pct"] = {k: pct(r[k], ref[k]) for k in keys}
        fixed_Ic[str(tau)] = r

        lam = retune_EJ(EC, c, F01_DESIGN_GHZ, N)
        rr = row(EJ_GHZ * lam, EC, c, N)
        rr["lambda_EJ"] = float(lam)
        rr["Ic_required_nA"] = float(IC_A * lam * 1e9)
        rr["f01_residual_GHz"] = float(rr["f01_GHz"] - F01_DESIGN_GHZ)
        rr["deltas_vs_sinusoidal_pct"] = {k: pct(rr[k], ref[k]) for k in keys}
        retuned[str(tau)] = rr

    # ---- N convergence -----------------------------------------------------
    conv = {}
    for tau in taus:
        c = harmonics(tau)
        rows = {}
        for Nc in (35, 45, N_CONV):
            f01, anh, eps1 = levels(EJ_GHZ, EC, c, Nc)
            rows[str(Nc)] = dict(f01_GHz=float(f01), anharm_GHz=float(anh),
                                 eps1_Hz=float(eps1 * 1e9),
                                 n01=float(n01_element(EJ_GHZ, EC, c, Nc)))
        base = rows[str(N_CHARGE)]
        top = rows[str(N_CONV)]
        rows["max_abs_dev_35_vs_60"] = {
            k: abs(top[k] - base[k]) for k in base}
        rows["max_rel_dev_35_vs_60"] = {
            k: abs(top[k] - base[k]) / abs(base[k]) for k in base}
        conv[str(tau)] = rows

    # ---- design-criterion robustness (does the E_J/E_C = 100 lock survive?)--
    ladder = {}
    c_nom = harmonics(TAU_NOMINAL)
    for r_ratio in RATIOS:
        ECr = EJ_GHZ / r_ratio
        f01, anh, eps1 = levels(EJ_GHZ, ECr, c_nom, N)
        ladder[str(r_ratio)] = dict(EC_GHz=float(ECr), f01_GHz=float(f01),
                                    anharm_GHz=float(anh),
                                    eps1_Hz=float(eps1 * 1e9))
    passing = [r_ratio for r_ratio in RATIOS
               if ladder[str(r_ratio)]["eps1_Hz"] < EPS1_MAX_HZ
               and abs(ladder[str(r_ratio)]["anharm_GHz"]) >= ANHARM_MIN_GHZ]
    scan_sin = spectrum.design_scan(EJ_GHZ, RATIOS, N)
    passing_sin = [r_ratio for r_ratio in RATIOS
                   if scan_sin[r_ratio]["eps1_Hz"] < EPS1_MAX_HZ
                   and abs(scan_sin[r_ratio]["anharm_GHz"]) >= ANHARM_MIN_GHZ]

    # ---- sweet spot --------------------------------------------------------
    d2_sin = D2(dome.NU_OPT, EC, C_SINUSOIDAL)
    d1_sin_off = D1(-2.5, EC, C_SINUSOIDAL)
    sweet = dict(
        nu_apex=float(dome.NU_OPT),
        dnu_stencil=float(dome.DNU),
        D2_sinusoidal_GHz=float(d2_sin),
        D2_dome_py_GHz=float(dome.D2(dome.NU_OPT, EC)),
        D1_apex_sinusoidal_GHz=float(D1(dome.NU_OPT, EC, C_SINUSOIDAL)),
        harmonics_vs_nu=(
            "material.json:nu_table carries cpr_harmonics on all 9 "
            "superconducting rows and every one is bit-identical to "
            "cpr.harmonics_by_tau['0.8']; the file therefore does NOT resolve "
            "tau (or the harmonics) vs nu. Held fixed at the tau=0.8 row "
            "across the dome, as the file states."),
        rows={},
    )
    for tau in taus:
        c = harmonics(tau)
        d2 = D2(dome.NU_OPT, EC, c)
        d1a = D1(dome.NU_OPT, EC, c)
        d1o = D1(-2.5, EC, c)
        lam = retuned[str(tau)]["lambda_EJ"]
        d2r = D2(dome.NU_OPT, EC, c, EJ_apex=EJ_GHZ * lam)
        sweet["rows"][str(tau)] = dict(
            D2_multi_GHz=float(d2), delta_pct=float(pct(d2, d2_sin)),
            D1_apex_multi_GHz=float(d1a),
            D1_off_nu_m2p5_multi_GHz=float(d1o),
            D1_off_delta_pct=float(pct(d1o, d1_sin_off)),
            Tphi_apex_gain_factor=float(abs(d2_sin / d2)),
            Tphi_apex_us_dnu1em4=float(
                dome.Tphi_second_order(d2, 1e-4) * 1e6),
            D2_retuned_GHz=float(d2r),
            D2_retuned_delta_pct=float(pct(d2r, d2_sin)),
        )

    # ---- truncation diagnostic --------------------------------------------
    trunc = {str(t): truncation_scan(t, EC=EC, N=N) for t in taus}
    for t in taus:
        conv_row = trunc[t]["converged"]
        m5 = fixed_Ic[t]
        trunc[t]["material_vs_converged_pct"] = {
            k: pct(m5[k], conv_row[k])
            for k in ("f01_GHz", "anharm_GHz", "n01", "eps1_Hz")}

    # ---- downstream --------------------------------------------------------
    nom = str(TAU_NOMINAL)
    dn_fixed, dn_ret = fixed_Ic[nom], retuned[nom]
    g_fixed_pct = pct(dn_fixed["n01"], ref["n01"])
    g_ret_pct = pct(dn_ret["n01"], ref["n01"])
    alpha_ratio_fixed = abs(ref["anharm_GHz"] / dn_fixed["anharm_GHz"])
    alpha_ratio_ret = abs(ref["anharm_GHz"] / dn_ret["anharm_GHz"])
    # src: out/m6/p6_gates.json:gate_floor.*.t_g_floor_ns["1e-04"] and
    #      combined_floor.t_g_floor_ns["1e-04"]  (P6 export, read not typed)
    p6 = _read_m6("p6_gates.json")
    tg_ref = {} if p6 is None else {
        "gaussian_1e-4_ns": p6["gate_floor"]["gaussian"]["t_g_floor_ns"]["1e-04"],
        "drag_1e-4_ns": p6["gate_floor"]["drag"]["t_g_floor_ns"]["1e-04"],
        "combined_1e-4_ns": p6["combined_floor"]["t_g_floor_ns"]["1e-04"],
    }
    # src: out/m6/p3_qp.json:parity.Tphi_parity_frozen_worst_case_s and
    # parity.Tphi_frozen_over_T1qp_at_1e6 -- their ratio IS P3's own qp-limited
    # T1 at x_qp = 1e-6, T = 100 mK (derived, never typed).
    p3 = _read_m6("p3_qp.json")
    if p3 is None:
        t1_qp_us = None
    else:
        t1_qp_us = 1e6 * (p3["parity"]["Tphi_parity_frozen_worst_case_s"]
                          / p3["parity"]["Tphi_frozen_over_T1qp_at_1e6"])
    parity_ref_s = 1.0 / (np.pi * ref["eps1_mp_Hz"])
    parity_fix_s = 1.0 / (np.pi * dn_fixed["eps1_mp_Hz"])
    parity_ret_s = 1.0 / (np.pi * dn_ret["eps1_mp_Hz"])
    downstream = dict(
        tau=nom,
        g_scaling=dict(
            law="g = 2 beta e V_zpf n01 / h  (qubit/coupling.py) -> g ~ n01 exactly",
            n01_sinusoidal=ref["n01"],
            n01_fixed_Ic=dn_fixed["n01"], g_delta_fixed_Ic_pct=float(g_fixed_pct),
            n01_retuned=dn_ret["n01"], g_delta_retuned_pct=float(g_ret_pct),
            note=("V_zpf depends on the CAVITY frequency, not on f01, and beta "
                  "is pure capacitance, so the g shift is exactly the n01 "
                  "shift. Under retuning n01 is recovered to 1e-4 relative "
                  "because n01 ~ sqrt(f01/8E_C) and both are restored."),
            chi_note=("chi ~ g^2 alpha /(Delta (Delta+alpha)) also moves through "
                      "Delta = f_cav - f01 and alpha; not re-derived here, P2 "
                      "owns the chi budget."),
        ),
        gate_floor_scaling=dict(
            law="leakage-limited t_g ~ 1/|anharmonicity| (no re-simulation)",
            anharm_sinusoidal_GHz=ref["anharm_GHz"],
            anharm_fixed_Ic_GHz=dn_fixed["anharm_GHz"],
            anharm_retuned_GHz=dn_ret["anharm_GHz"],
            t_g_stretch_fixed_Ic=float(alpha_ratio_fixed),
            t_g_stretch_retuned=float(alpha_ratio_ret),
            p6_reference_ns=tg_ref,
            scaled_fixed_Ic_ns={k: v * alpha_ratio_fixed
                                for k, v in tg_ref.items()},
            scaled_retuned_ns={k: v * alpha_ratio_ret for k, v in tg_ref.items()},
        ),
        parity_bound=dict(
            law="T_phi <= 1/(pi eps1)  (qubit/qp_kernel.py:Tphi_parity_frozen)",
            eps1_sinusoidal_Hz=ref["eps1_mp_Hz"],
            eps1_fixed_Ic_Hz=dn_fixed["eps1_mp_Hz"],
            eps1_retuned_Hz=dn_ret["eps1_mp_Hz"],
            bound_sinusoidal_ms=float(parity_ref_s * 1e3),
            bound_fixed_Ic_ms=float(parity_fix_s * 1e3),
            bound_retuned_ms=float(parity_ret_s * 1e3),
            bound_delta_fixed_Ic_pct=float(pct(parity_fix_s, parity_ref_s)),
            bound_delta_retuned_pct=float(pct(parity_ret_s, parity_ref_s)),
            T1_qp_us_at_x1em6=t1_qp_us,
            bound_over_T1qp_sinusoidal=(None if t1_qp_us is None else
                                        parity_ref_s * 1e6 / t1_qp_us),
            bound_over_T1qp_fixed_Ic=(None if t1_qp_us is None else
                                      parity_fix_s * 1e6 / t1_qp_us),
            bound_over_T1qp_retuned=(None if t1_qp_us is None else
                                     parity_ret_s * 1e6 / t1_qp_us),
            p3_stated_ratio=(None if p3 is None else
                             p3["parity"]["Tphi_frozen_over_T1qp_at_1e6"]),
            verdict_ref=("P3's verdict was 'the frozen-parity bound exceeds "
                         "the qp-limited T1 by >1e3 across the whole x_qp "
                         "band'; see bound_over_T1qp_* for whether it "
                         "survives the CPR harmonics."),
        ),
    )

    return {
        "normalization_check": {
            "convention": ("max_phi sum_m c_m sin(m phi) = 1, so Ic is the "
                           "TRUE critical current (M3_BRIEF.md Sec. 2, "
                           "junction/cpr.py:normalize_cpr)"),
            "cpr_model": CPR_MODEL,
            "grid_points": 400001,
            "rows": normalization_check(),
            "sinusoidal": dict(cpr_max=cpr_max(C_SINUSOIDAL),
                               argmax_over_pi=cpr_argmax(C_SINUSOIDAL),
                               A2=A2(C_SINUSOIDAL), A4=A4(C_SINUSOIDAL)),
            "energy_chain": ("U(phi) = -E_J sum_m (c_m/m) cos(m phi), "
                             "E_J = Phi0 Ic/(2 pi) = hbar Ic/(2e), same Ic as "
                             "normalizes the c_m; charge basis coupling "
                             "|n> -> |n+-m| is -E_J c_m/(2m)"),
        },
        "reference_sinusoidal": ref,
        "fixed_Ic": fixed_Ic,
        "retuned": retuned,
        "N_convergence": conv,
        "design_criterion_robustness": {
            "ladder_tau_nominal": ladder,
            "criteria": {"eps1_max_Hz": EPS1_MAX_HZ,
                         "anharm_min_GHz": ANHARM_MIN_GHZ},
            "passing_ratios": passing,
            "passing_ratios_sinusoidal": passing_sin,
            "selected": min(passing) if passing else None,
            "selected_sinusoidal": TARGETS["design"]["EJ_over_EC"],
            "verdict": ("design lock unchanged" if passing
                        and min(passing) == TARGETS["design"]["EJ_over_EC"]
                        else "design lock MOVES"),
            "window_note": ("the SELECTED ratio is unchanged, but the window "
                            "of acceptable ratios collapses: the higher "
                            "harmonics cut |anharmonicity| so every ratio "
                            ">= 150 now fails the 0.5 GHz criterion."),
        },
        "sweet_spot": sweet,
        "harmonic_truncation": trunc,
        "downstream": downstream,
        "params": {
            "EJ_GHz": float(EJ_GHZ),
            "Ic_nA": float(IC_A * 1e9),
            "EC_GHz": float(EC),
            "EJ_over_EC": float(EJ_GHZ / EC),
            "f01_target_GHz": float(F01_DESIGN_GHZ),
            "N_charge": int(N),
            "N_convergence_max": int(N_CONV),
            "mp_dps": int(MP_DPS),
            "dnu_stencil": float(dome.DNU),
            "taus": taus,
            "tau_nominal": TAU_NOMINAL,
            "harmonics_by_tau": {t: harmonics(t).tolist() for t in taus},
            "held_fixed_fixed_Ic": "Ic (-> E_J) and C_Sigma (-> E_C)",
            "held_fixed_retuned": ("E_C; E_J scaled by lambda so f01 returns "
                                   "to the design value"),
            "retune_caveat": ("lambda > 1 for every tau and E_J(nu) is already "
                              "MAXIMAL at the dome apex, so the retune is a "
                              "fabrication knob (larger Ic / wider junction) "
                              "or a shunt knob (smaller C_Sigma), not a gate "
                              "knob. Reported to isolate the untunable part "
                              "of the shift."),
        },
        "provenance": {
            "module": "qubit/cpr_transmon.py (M6 deliverable P-extra)",
            "resolves": ("paper/decisions.md 2026-07-27 'M6 flag (deferred): "
                         "non-sinusoidal CPR vs transmon Hamiltonian'"),
            "deterministic": True,
            "rng_seed": None,
            "rng_note": ("no stochastic step: exact diagonalisation on fixed "
                         "grids; bisection on a monotone function"),
            "reads": {
                "harmonics_by_tau": "material.json:cpr.harmonics_by_tau",
                "cpr_model": "material.json:cpr.model",
                "tau_nominal": "material.json:cpr.tau_nominal",
                "nu_harmonics": "material.json:nu_table[*].cpr_harmonics",
                "Ic_nA": "junction.json:contract.Ic_nA (via qubit/inputs.py)",
                "EC_GHz/f01_GHz/EJ_over_EC": "qubit_targets.json:design",
                "dome": "material.json:sc_dome (via qubit/dome.py)",
                "t_g_floor_ns": "out/m6/p6_gates.json:gate_floor/combined_floor",
                "T1_qp": "out/m6/p3_qp.json:parity.rate",
                "truncation_generator": ("m1/cpr.py:beenakker_harmonics -- "
                                         "diagnostic only, its nH=5 output "
                                         "reproduces material.json to 2.8e-17"),
            },
            "unmodified": ["qubit/spectrum.py", "qubit/inputs.py",
                           "qubit/dome.py", "qubit/cavity.py", "qubit/qp.py",
                           "qubit/coupling.py", "qubit/qp_kernel.py",
                           "m1/cpr.py", "junction/cpr.py"],
            "tests": "tests/test_m6_pextra_cpr.py",
        },
    }


def main():
    """Recompute P-extra and write out/m6/pextra_cpr.json."""
    payload = build_export()
    outdir = ROOT / "out" / "m6"
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "pextra_cpr.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return path, payload


if __name__ == "__main__":
    p, pay = main()
    print(f"wrote {p}")
    ref = pay["reference_sinusoidal"]
    print(f"sinusoidal ref: f01={ref['f01_GHz']:.6f} anh={ref['anharm_GHz']:.6f} "
          f"n01={ref['n01']:.7f} eps1={ref['eps1_mp_Hz']:.4f} Hz (mp)")
    for scen in ("fixed_Ic", "retuned"):
        print(f"--- {scen} ---")
        for t, r in pay[scen].items():
            d = r["deltas_vs_sinusoidal_pct"]
            print(f"  tau={t}: f01={r['f01_GHz']:.6f} ({d['f01_GHz']:+.3f}%)  "
                  f"anh={r['anharm_GHz']:.6f} ({d['anharm_GHz']:+.3f}%)  "
                  f"n01={r['n01']:.7f} ({d['n01']:+.4f}%)  "
                  f"eps1={r['eps1_mp_Hz']:.4f} Hz ({d['eps1_mp_Hz']:+.2f}%)")
    sw = pay["sweet_spot"]
    print(f"D2 sinusoidal = {sw['D2_sinusoidal_GHz']:.6f}")
    for t, r in sw["rows"].items():
        print(f"  tau={t}: D2={r['D2_multi_GHz']:.6f} ({r['delta_pct']:+.3f}%)  "
              f"retuned {r['D2_retuned_GHz']:.6f} ({r['D2_retuned_delta_pct']:+.3f}%)")
