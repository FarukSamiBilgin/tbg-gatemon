"""M6 exploration 2/5 -- E_J 1/f noise: the missing THIRD CLAUSE of the
double-sweet-spot law.

QUESTION
--------
The dome apex nu = -2.4 zeroes df01/dnu (first clause) while maximising E_J
(second clause).  But  D_EJ = df01/dlnE_J  never vanishes anywhere on the dome.
P4 (qubit/noise_1f.py, out/m6/p4_t2_maps.json) treated ONLY filling noise, whose
first-order coupling vanishes at the apex by construction, so the entire P4 T2
map is blind to the one 1/f channel that bypasses the dome symmetry: junction
transparency / moire-domain / Tc-strain drift, all of which funnel into dlnE_J.

This script quantifies, parametrically in a free amplitude A_EJ:
  (1) D_EJ by central finite difference on qubit.spectrum.levels, with an
      FD/truncation/N convergence table and the harmonic sanity check
      sqrt(8 E_J E_C)/2.
  (2) The EXACT chain-rule identity  D1(nu) = D_EJ(E_J(nu)) * dlnE_J/dnu(nu),
      which is what makes the apex a sweet spot at all -- and which shows the
      apex kills only the SECOND factor.
  (3) T2 from the E_J channel alone (first order, Ramsey and echo) over
      A_EJ in logspace(-8, -4, 25).
  (4) The combined apex budget chi_tot = chi_2(gate, A) + chi_1(E_J, A_EJ)
      on the full (A, A_EJ) grid -> the T2 phase diagram, plus the same at
      nu = -2.5 -> the collapse of the 1292x / 380x sweet-spot contrast.
  (5) Crossovers: A_EJ where the E_J channel matches the gate-noise apex rate;
      where the apex T2 falls below the P3 quasiparticle 2*T1 ladder; where the
      contrast falls to 100x / 10x / 2x.

CONVENTIONS (identical to P4, so every number is directly comparable)
---------------------------------------------------------------------
  gate noise :  S_nu(w)          = A^2    / |w|     (two-sided, angular)
  E_J noise  :  S_dlnEJ(w)       = A_EJ^2 / |w|     (two-sided, angular)
                i.e. FRACTIONAL critical-current noise, since E_J ~ Ic;
                one-sided ordinary-frequency equivalent S^(1)(f) = 2 A_EJ^2 / f,
                so sqrt(S_Ic(1 Hz))/Ic = sqrt(2) * A_EJ  Hz^-1/2.
  f_ir = 1 Hz (1 s acquisition);  T2 defined by chi(T2) = 1;  M = 96.

FREE PARAMETERS (design-study discipline -- nothing here is a measurement)
  A     in [1e-6, 1e-2]  : P4's filling-noise amplitude grid (25 log points).
  A_EJ  in [1e-8, 1e-4]  : fractional 1/f E_J (= Ic) noise, 25 log points.
                           NEVER measured for a TBG junction.  For ORIENTATION
                           ONLY and explicitly NOT as a calibration, 1/f
                           critical-current noise in conventional Josephson
                           junctions is usually quoted in the
                           1e-6 .. 1e-5 Hz^-1/2 range for sqrt(S_Ic(1Hz))/Ic
                           and scales roughly as 1/sqrt(junction area)
                           [Van Harlingen et al., PRB 70, 064517 (2004)].
                           No absolute TBG coherence number is claimed.
  h = 1e-5 (relative FD step on E_J), convergence-tested over 4 decades.

ADDITIVITY.  Gate-filling noise and E_J noise are two INDEPENDENT stationary
processes, so their coherence exponents add exactly (the characteristic
functions multiply).  This is a stronger statement than P4's within-channel
chi_1 + chi_2 rule.  At the apex D1 = 0, so the apex map here is exact under
the exported additivity rule; the nu = -2.5 column inherits P4's caveat and is
only quoted for A <= 1e-3 (qubit.json:budgets.T2_maps.additivity_rule).

Deterministic: no RNG anywhere (no seed needed).
Writes ONLY out/m6/explore/ej_1f_third_clause.json.
"""
import json
import pathlib
import sys
import time

import numpy as np
from scipy.optimize import brentq

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import qubit                                            # noqa: E402
from qubit import dome, noise_1f as nz, spectrum        # noqa: E402
from qubit.inputs import EJ_GHZ, N_CHARGE, ROOT         # noqa: E402

# ------------------------------------------------------------------ free params
H_FD = 1e-5                      # relative central-difference step on E_J
A_EJ_DECADES = (-8.0, -4.0)      # free parameter range (see module docstring)
A_EJ_POINTS = 25
NU_OFF = -2.5                    # the P4 off-apex reference filling
SEQUENCES = ("ramsey", "echo")

OUTDIR = ROOT / "out" / "m6" / "explore"
OUTFILE = OUTDIR / "ej_1f_third_clause.json"


# --------------------------------------------------------------- two-channel T2
def chi_two_channel_batch(ts, D1_gate, D2_gate, A, D_EJ, A_EJ,
                          sequence, w_ir, M):
    """chi_gate(t; A) + chi_EJ(t; A_EJ) on an array of times.

    chi_gate = chi_first_order(D1_gate, A) + chi_second_order(D2_gate, A)
               (P4's prescribed independent-rate rule; exact at the apex)
    chi_EJ   = chi_first_order(D_EJ, A_EJ)   -- a genuinely independent process,
               so this addition is exact.
    Every term is qubit/noise_1f.py verbatim; nothing is re-derived here.
    """
    ts = np.atleast_1d(np.asarray(ts, dtype=float))
    chi = np.zeros_like(ts)
    if A != 0.0:
        if D1_gate != 0.0:
            chi = chi + np.asarray(
                nz.chi_first_order(ts, D1_gate, A, sequence, w_ir), dtype=float)
        if D2_gate != 0.0:
            chi = chi + nz.chi_second_order_batch(ts, D2_gate, A, sequence,
                                                  w_ir, M)
    if A_EJ != 0.0 and D_EJ != 0.0:
        chi = chi + np.asarray(
            nz.chi_first_order(ts, D_EJ, A_EJ, sequence, w_ir), dtype=float)
    return chi


def T2_two_channel(D1_gate, D2_gate, A, D_EJ, A_EJ, sequence="ramsey",
                   w_ir=nz.W_IR_DEFAULT, M=nz.M_GRID, n_scan=nz.N_SCAN):
    """Solve chi_gate + chi_EJ = 1 for T2 (s).

    Identical batched bracket-and-refine algorithm as qubit.noise_1f.T2 (same
    bracket, same n_scan, same linear interpolation, same NaN policy: a T2
    longer than the acquisition window 2 pi / w_ir is not reported).
    """
    lo, hi = np.log(nz.T_MIN_S), np.log(2.0 * np.pi / w_ir)
    c0 = c1 = None
    for k, n in enumerate(n_scan):
        lts = np.linspace(lo, hi, n)
        c = chi_two_channel_batch(np.exp(lts), D1_gate, D2_gate, A,
                                  D_EJ, A_EJ, sequence, w_ir, M)
        if k == 0 and (c[0] > 1.0 or c[-1] < 1.0):
            return np.nan
        j = int(np.searchsorted(c, 1.0))
        j = min(max(j, 1), n - 1)
        lo, hi = lts[j - 1], lts[j]
        c0, c1 = c[j - 1], c[j]
    return float(np.exp(lo + (hi - lo) * (1.0 - c0) / (c1 - c0)))


def invert_for_A_EJ(target_T2_s, D1_gate, D2_gate, A, D_EJ, sequence,
                    w_ir=nz.W_IR_DEFAULT, M=nz.M_GRID,
                    lo=1e-12, hi=1e-1):
    """A_EJ such that T2_two_channel(...) == target_T2_s.  NaN if unbracketed."""
    def f(la):
        t2 = T2_two_channel(D1_gate, D2_gate, A, D_EJ, float(np.exp(la)),
                            sequence, w_ir, M)
        if not np.isfinite(t2):
            return 1.0          # T2 > window  =>  above target
        return np.log(t2 / target_T2_s)
    a, b = np.log(lo), np.log(hi)
    fa, fb = f(a), f(b)
    if not (fa > 0 > fb):
        return float("nan")
    return float(np.exp(brentq(f, a, b, xtol=1e-8, rtol=1e-12)))


def _round(x, sig=6):
    a = np.asarray(x, dtype=float)
    out = np.zeros_like(a)
    fin = np.isfinite(a)
    m = fin & (a != 0)
    sc = np.zeros_like(a)
    sc[m] = 10.0 ** (sig - 1 - np.floor(np.log10(np.abs(a[m]))))
    out[m] = np.round(a[m] * sc[m]) / sc[m]
    if a.ndim == 0:
        return None if not fin else float(out)
    return [None if not f else float(v) for f, v in zip(fin.ravel(), out.ravel())]


def _round2d(M2):
    return [_round(row) for row in np.asarray(M2, dtype=float)]


def _sanitize(o):
    """NaN/Inf -> None so the artifact is strict, parseable JSON."""
    if isinstance(o, dict):
        return {k: _sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_sanitize(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def _cross_log(x, y, target):
    """log-log interpolate x at which the monotone-decreasing y hits target."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(y) & (y > 0)
    if ok.sum() < 2:
        return float("nan")
    lx, ly = np.log(x[ok]), np.log(y[ok])
    lt = np.log(target)
    if ly[0] < lt or ly[-1] > lt:
        return float("nan")          # target outside the sampled range
    return float(np.exp(np.interp(-lt, -ly, lx)))


# ==============================================================================
def main():
    t_start = time.time()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- inputs --
    # src: qubit/inputs.py:EJ_GHZ  <- junction.json:contract.Ic_nA (300 nA)
    EJ = float(EJ_GHZ)
    dp = qubit.design_point()          # src: recomputed design scan (AT-1)
    EC = float(dp["EC_GHz"])           # src: E_J/E_C = 100 lock -> 1.490050532 GHz
    f01_apex = float(dp["f01_GHz"])

    with open(ROOT / "qubit.json", encoding="utf-8") as fh:
        QJ = json.load(fh)             # src: qubit.json (paper-2 interface)
    T2M = QJ["budgets"]["T2_maps"]
    STORED = {r["label"]: r for r in T2M["headline_rows"]}
    stored_contrast = T2M["contrast_at_A_1e_4"]
    stored_qs = T2M["quasistatic_reference"]
    T1_LADDER = QJ["budgets"]["T1_stack"]["qp_full_kernel"][
        "T1_us_vs_xqp_T_qp_to_zero"]      # src: qubit.json (P3 full qp kernel)
    w_ir = float(T2M["ir_cutoff_convention"]["w_ir_rad_per_s"])   # src: qubit.json
    M = int(nz.M_GRID)                 # src: qubit/noise_1f.py:M_GRID

    # ------------------------------------------- (1)  D_EJ = df01 / dln E_J --
    def dEJ_of(EJv, h=H_FD, N=N_CHARGE):
        fp = spectrum.levels(EJv * (1.0 + h), EC, N)[0]
        fm = spectrum.levels(EJv * (1.0 - h), EC, N)[0]
        return (fp - fm) / (2.0 * h)

    def dEJ_4pt(EJv, h=H_FD, N=N_CHARGE):
        f1p = spectrum.levels(EJv * (1 + h), EC, N)[0]
        f1m = spectrum.levels(EJv * (1 - h), EC, N)[0]
        f2p = spectrum.levels(EJv * (1 + 2 * h), EC, N)[0]
        f2m = spectrum.levels(EJv * (1 - 2 * h), EC, N)[0]
        return (8.0 * (f1p - f1m) - (f2p - f2m)) / (12.0 * h)

    D_EJ = float(dEJ_of(EJ))
    D_EJ_harmonic = float(np.sqrt(8.0 * EJ * EC) / 2.0)

    fd_table = {}
    for h in (1e-3, 1e-4, 1e-5, 1e-6, 1e-7):
        fd_table[f"{h:.0e}"] = dict(two_point=float(dEJ_of(EJ, h)),
                                    four_point=float(dEJ_4pt(EJ, h)))
    N_table = {str(N): float(dEJ_of(EJ, H_FD, N)) for N in (35, 50, 60)}

    # second derivative in ln E_J (to bound the E_J second-order channel).
    # Evaluated on the exact log grid E_J*exp(+-h2) so the second difference is
    # d2 f01/d(ln E_J)^2 with no O(h) contamination; h2 = 1e-3 keeps the
    # eigenvalue round-off (~1e-13 rel) out of the 1/h2^2 amplification.
    H2_FD = 1e-3
    fpp = spectrum.levels(EJ * np.exp(H2_FD), EC)[0]
    fmm = spectrum.levels(EJ * np.exp(-H2_FD), EC)[0]
    f00 = spectrum.levels(EJ, EC)[0]
    D2_EJ = float((fpp - 2 * f00 + fmm) / H2_FD ** 2)

    # ------------------------------- (2)  chain rule: D1(nu) = D_EJ * dlnEJ/dnu
    dnu = dome.DNU                     # src: qubit/dome.py:DNU = 1e-3
    D1_apex = float(dome.D1(-2.4, EC))
    D2_apex = float(dome.D2(-2.4, EC))
    D1_off = float(dome.D1(NU_OFF, EC))
    D2_off = float(dome.D2(NU_OFF, EC))

    def dlnEJ_dnu(nu):
        return float((np.log(dome.EJ_of(nu + dnu)) - np.log(dome.EJ_of(nu - dnu)))
                     / (2.0 * dnu))

    chain = {}
    for nu in (-2.4, NU_OFF, -2.6, -2.3):
        EJn = float(dome.EJ_of(nu))
        s = dlnEJ_dnu(nu)
        dej = float(dEJ_of(EJn))
        d1d = float(dome.D1(nu, EC))
        chain[f"{nu:+.2f}"] = dict(
            EJ_GHz=EJn, dlnEJ_dnu=s, D_EJ_local_GHz=dej,
            product_GHz=dej * s, D1_direct_GHz=d1d,
            abs_dev_GHz=abs(dej * s - d1d),
            reldev=(abs(dej * s - d1d) / abs(d1d)) if abs(d1d) > 1e-9 else None)

    # ---------------------------------------- VALIDATION vs stored artifacts --
    apex_row = STORED["apex, A = 1e-4"]
    off_row = STORED["nu = -2.5, A = 1e-4"]
    repro = dict(
        apex_T2_ramsey_us=dict(
            recomputed=nz.T2(0.0, D2_apex, 1e-4, "ramsey", w_ir, M) * 1e6,
            stored=apex_row["T2_ramsey_us"]),
        apex_T2_echo_us=dict(
            recomputed=nz.T2(0.0, D2_apex, 1e-4, "echo", w_ir, M) * 1e6,
            stored=apex_row["T2_echo_us"]),
        off_T2_ramsey_us=dict(
            recomputed=nz.T2(D1_off, D2_off, 1e-4, "ramsey", w_ir, M) * 1e6,
            stored=off_row["T2_ramsey_us"]),
        off_T2_echo_us=dict(
            recomputed=nz.T2(D1_off, D2_off, 1e-4, "echo", w_ir, M) * 1e6,
            stored=off_row["T2_echo_us"]),
        D2_apex_GHz=dict(recomputed=D2_apex, stored=stored_qs["D2_apex_GHz"]),
        D1_off_GHz=dict(recomputed=D1_off,
                        stored=stored_qs["D1_off_nu_m2p5_GHz"]),
        chain_rule_D1_off_GHz=dict(recomputed=chain[f"{NU_OFF:+.2f}"]["product_GHz"],
                                   stored=stored_qs["D1_off_nu_m2p5_GHz"]),
    )
    for k, v in repro.items():
        v["reldev"] = abs(v["recomputed"] - v["stored"]) / abs(v["stored"])

    # solver self-consistency: two-channel solver must reduce to noise_1f.T2
    solver_check = {}
    for seq in SEQUENCES:
        a = T2_two_channel(0.0, D2_apex, 1e-4, D_EJ, 0.0, seq, w_ir, M)
        b = nz.T2(0.0, D2_apex, 1e-4, seq, w_ir, M)
        c = T2_two_channel(0.0, 0.0, 0.0, D_EJ, 1e-6, seq, w_ir, M)
        d = nz.T2_first_order_only(D_EJ, 1e-6, seq, w_ir)
        solver_check[seq] = dict(
            gate_only_reldev=abs(a - b) / b, EJ_only_reldev=abs(c - d) / d)

    # -------------------------------- (3)  E_J channel alone: T2_EJ(A_EJ) ----
    A_EJ_grid = np.logspace(*A_EJ_DECADES, A_EJ_POINTS)
    T2_EJ = {seq: np.array([nz.T2_first_order_only(D_EJ, a, seq, w_ir)
                            for a in A_EJ_grid]) for seq in SEQUENCES}
    scaling = {}
    for seq in SEQUENCES:
        lg = np.log(T2_EJ[seq])
        la = np.log(A_EJ_grid)
        slope = np.gradient(lg, la)
        scaling[seq] = dict(
            local_slope_min=float(slope.min()), local_slope_max=float(slope.max()),
            endpoint_slope=float((lg[-1] - lg[0]) / (la[-1] - la[0])),
            product_T2_times_A_EJ_min=float((T2_EJ[seq] * A_EJ_grid).min()),
            product_T2_times_A_EJ_max=float((T2_EJ[seq] * A_EJ_grid).max()))
    echo_over_ramsey_EJ = T2_EJ["echo"] / T2_EJ["ramsey"]

    # bound on the E_J SECOND-order channel (never used; shown to be negligible)
    i1e6 = int(np.argmin(np.abs(np.log(A_EJ_grid) - np.log(1e-6))))
    t_ref = float(T2_EJ["ramsey"][i1e6])
    chi1_ref = float(nz.chi_first_order(t_ref, D_EJ, 1e-6, "ramsey", w_ir))
    chi2_ref = float(nz.chi_second_order(t_ref, D2_EJ, 1e-6, "ramsey", w_ir, M))
    second_order_EJ_bound = dict(
        d2f01_dlnEJ2_GHz=D2_EJ, t_eval_s=t_ref, A_EJ=1e-6,
        chi_first=chi1_ref, chi_second=chi2_ref,
        ratio_second_over_first=chi2_ref / chi1_ref,
        verdict="the E_J second-order channel is negligible: unlike the gate "
                "channel there is no symmetry that removes the first order, so "
                "first order always dominates")

    # ----------------------- (4)  combined (A, A_EJ) maps at apex and off-apex
    A_grid = nz.A_grid()               # src: qubit/noise_1f.py (P4's own A grid)
    nA, nE = len(A_grid), len(A_EJ_grid)
    maps = {}
    for seq in SEQUENCES:
        apex = np.empty((nA, nE))
        off = np.empty((nA, nE))
        for i, A in enumerate(A_grid):
            for j, aej in enumerate(A_EJ_grid):
                apex[i, j] = T2_two_channel(0.0, D2_apex, A, D_EJ, aej,
                                            seq, w_ir, M)
                off[i, j] = T2_two_channel(D1_off, D2_off, A, D_EJ, aej,
                                           seq, w_ir, M)
        maps[seq] = dict(apex=apex, off=off, contrast=apex / off)
        print(f"  map {seq}: done  ({time.time() - t_start:.1f} s)", flush=True)

    # baseline (A_EJ = 0) columns, for the degradation factor
    base = {}
    for seq in SEQUENCES:
        base[seq] = dict(
            apex=np.array([nz.T2(0.0, D2_apex, A, seq, w_ir, M) for A in A_grid]),
            off=np.array([nz.T2(D1_off, D2_off, A, seq, w_ir, M) for A in A_grid]))

    # ----------------------------------------------- (5)  crossover extraction
    iA = int(np.argmin(np.abs(A_grid - 1e-4)))
    A_ref = float(A_grid[iA])
    apex0 = {seq: float(base[seq]["apex"][iA]) for seq in SEQUENCES}
    off0 = {seq: float(base[seq]["off"][iA]) for seq in SEQUENCES}

    cross = {}
    for seq in SEQUENCES:
        # A_EJ at which the E_J channel ALONE gives the P4 apex headline T2
        a_match = invert_for_A_EJ(apex0[seq], 0.0, 0.0, 0.0, D_EJ, seq, w_ir, M)
        # A_EJ at which the COMBINED apex T2 is x times the P4 headline
        deg = {}
        for frac in (0.9, 0.5, 0.1, 0.01):
            deg[f"{frac:g}"] = invert_for_A_EJ(frac * apex0[seq], 0.0, D2_apex,
                                               A_ref, D_EJ, seq, w_ir, M)
        # A_EJ at which the apex T2 crosses 2*T1_qp (T2 <= 2 T1)
        t1x = {}
        for k, t1_us in T1_LADDER.items():
            tgt = 2.0 * t1_us * 1e-6
            t1x[k] = dict(
                two_T1_us=2.0 * t1_us,
                already_below_at_A_EJ_0=bool(apex0[seq] < tgt),
                A_EJ_crossing=invert_for_A_EJ(tgt, 0.0, D2_apex, A_ref,
                                              D_EJ, seq, w_ir, M))
        # A_EJ at which the apex/off-apex contrast falls to a target
        con = {}
        for target in (100.0, 10.0, 2.0):
            def g(la, target=target, seq=seq):
                aej = float(np.exp(la))
                a = T2_two_channel(0.0, D2_apex, A_ref, D_EJ, aej, seq, w_ir, M)
                o = T2_two_channel(D1_off, D2_off, A_ref, D_EJ, aej, seq, w_ir, M)
                return np.log((a / o) / target)
            lo, hi = np.log(1e-12), np.log(1e-1)
            con[f"{target:g}x"] = (
                float(np.exp(brentq(g, lo, hi, xtol=1e-8, rtol=1e-12)))
                if g(lo) > 0 > g(hi) else float("nan"))
        cross[seq] = dict(
            A_EJ_matching_P4_apex_headline=a_match,
            apex_T2_headline_us=apex0[seq] * 1e6,
            offapex_T2_headline_us=off0[seq] * 1e6,
            A_EJ_for_apex_T2_fraction_of_headline=deg,
            A_EJ_crossing_2T1_qp_ladder=t1x,
            A_EJ_for_contrast=con)

    # ------------------------------------- experimental discriminator (NEW) --
    # At the apex the two channels have DIFFERENT echo gains: the gate channel
    # is second order there (Szego limit -> echo buys ~1.44x) while the E_J
    # channel is first order (echo removes the IR log -> ~4.2x).  The apex
    # echo/Ramsey RATIO is therefore an amplitude-free discriminator between
    # them: it needs no knowledge of A or A_EJ, only two measurements.
    disc = {}
    for A_lab, A_val in (("1e-4", A_ref),
                         ("1e-3", float(A_grid[int(np.argmin(
                             np.abs(np.log(A_grid) - np.log(1e-3))))]))):
        i = int(np.argmin(np.abs(np.log(A_grid) - np.log(A_val))))
        ratio = maps["echo"]["apex"][i] / maps["ramsey"]["apex"][i]
        base_ratio = base["echo"]["apex"][i] / base["ramsey"]["apex"][i]
        disc[A_lab] = dict(
            A=A_val, A_EJ=_round(A_EJ_grid),
            apex_echo_over_ramsey=_round(ratio),
            limit_pure_gate_noise=float(base_ratio),
            limit_pure_EJ_noise=float(echo_over_ramsey_EJ[-1]),
            A_EJ_at_ratio_2=_cross_log(A_EJ_grid, 1.0 / ratio, 1.0 / 2.0)
            if np.all(np.diff(ratio) > 0) else None)
    discriminator = dict(
        observable="apex echo gain  T2_echo(-2.4) / T2_ramsey(-2.4)",
        why="the gate channel is SECOND order at the apex (asymptotically "
            "sequence-independent, measured gain 1.44x) while the E_J channel is "
            "FIRST order (echo removes the infrared log, gain ~4.2x). The ratio "
            "is amplitude-free: it discriminates the two 1/f channels without "
            "knowing A or A_EJ.",
        pure_gate_noise_limit=float(base["echo"]["apex"][iA]
                                    / base["ramsey"]["apex"][iA]),
        pure_EJ_noise_limit=float(echo_over_ramsey_EJ.mean()),
        pure_EJ_noise_limit_range=[float(echo_over_ramsey_EJ.min()),
                                   float(echo_over_ramsey_EJ.max())],
        table=disc,
        falsifiable_prediction="if the measured apex echo gain is ~1.4x the "
            "dephasing is filling noise and the double sweet spot is doing its "
            "job; if it is ~4x the apex is limited by E_J (critical-current) "
            "noise, which no gate bias can remove")

    # ---------------------------------------------- the survival ridge (NEW) --
    ridge = {}
    for seq in SEQUENCES:
        rows = []
        for i, A in enumerate(A_grid):
            c = maps[seq]["contrast"][i]
            rows.append(dict(
                A=float(A),
                contrast_at_A_EJ_0=float(base[seq]["apex"][i]
                                         / base[seq]["off"][i]),
                A_EJ_contrast_100x=_cross_log(A_EJ_grid, c, 100.0),
                A_EJ_contrast_10x=_cross_log(A_EJ_grid, c, 10.0),
                A_EJ_contrast_2x=_cross_log(A_EJ_grid, c, 2.0),
                A_EJ_apex_halved=_cross_log(
                    A_EJ_grid, maps[seq]["apex"][i] / base[seq]["apex"][i], 0.5)))
        ridge[seq] = rows

    # equivalent apex offset in filling: |D2| * |dnu| * A  ==  D_EJ * A_EJ
    dnu_equiv = {f"{a:.0e}": float(D_EJ * a / (abs(D2_apex) * A_ref))
                 for a in (1e-8, 1e-7, 1e-6, 1e-5, 1e-4)}

    # The ridge is proportional: A_EJ_crit ~ k * A, because both channels are
    # linear in their own amplitude at fixed t.  Report k and the unit-free
    # equivalent apex offset dnu_eq = D_EJ k / |D2_apex| that goes with it.
    ridge_law = {}
    for seq in SEQUENCES:
        for key in ("A_EJ_contrast_100x", "A_EJ_contrast_10x",
                    "A_EJ_contrast_2x", "A_EJ_apex_halved"):
            pts = [(r["A"], r[key]) for r in ridge[seq]
                   if r[key] is not None and np.isfinite(r[key])]
            if len(pts) < 2:
                continue
            Av = np.array([p[0] for p in pts])
            Ev = np.array([p[1] for p in pts])
            ks = Ev / Av
            p_fit = float(np.polyfit(np.log(Av), np.log(Ev), 1)[0])
            ridge_law[f"{seq}:{key}"] = dict(
                n_A_points=int(ks.size), k_median=float(np.median(ks)),
                k_min=float(ks.min()), k_max=float(ks.max()),
                loglog_exponent_dlnA_EJ_dlnA=p_fit,
                dnu_eq_at_k_median=float(D_EJ * np.median(ks) / abs(D2_apex)),
                dnu_eq_in_dome_halfwidths=float(
                    D_EJ * np.median(ks) / abs(D2_apex) / 0.5))

    # Closed form for the contrast ridge.  Once the E_J channel dominates the
    # apex AND the gate first-order channel dominates off-apex, both T2s come
    # from the SAME first-order Gaussian filter function, so the logs cancel
    # under echo and the contrast is just the ratio of the two sensitivities:
    #     C = |D1(nu_off)| A / (D_EJ A_EJ)   =>   A_EJ_crit/A = |D1|/(D_EJ C)
    contrast_law = dict(
        formula="C_EJ_limited = |D1(nu_off)| * A / (D_EJ * A_EJ)",
        derivation="in the E_J-limited apex regime both apex and off-apex "
                   "dephasing are FIRST-order Gaussian with the same filter "
                   "function, so the IR logs cancel exactly under echo and "
                   "the contrast reduces to the ratio of sensitivities",
        D1_off_over_D_EJ=abs(D1_off) / D_EJ,
        predicted_k={f"{C:g}x": abs(D1_off) / (D_EJ * C)
                     for C in (100.0, 10.0, 2.0)},
        measured_k_echo={f"{C:g}x": ridge_law.get(
            f"echo:A_EJ_contrast_{int(C)}x", {}).get("k_median")
            for C in (100.0, 10.0, 2.0)},
        measured_k_ramsey={f"{C:g}x": ridge_law.get(
            f"ramsey:A_EJ_contrast_{int(C)}x", {}).get("k_median")
            for C in (100.0, 10.0, 2.0)},
        note="the echo k matches the closed form to <1% at C = 100 and C = 10 "
             "and degrades at C = 2, where neither channel is yet dominant. The "
             "Ramsey k sits ~9% high because the apex and off-apex Ramsey logs "
             "are evaluated at very different T2 and do not cancel.")

    # cross-check: map interpolation vs the independent brentq root solves
    map_vs_root = {}
    for seq in SEQUENCES:
        r = ridge[seq][iA]
        pairs = [("apex_halved", r["A_EJ_apex_halved"],
                  cross[seq]["A_EJ_for_apex_T2_fraction_of_headline"]["0.5"]),
                 ("contrast_100x", r["A_EJ_contrast_100x"],
                  cross[seq]["A_EJ_for_contrast"]["100x"]),
                 ("contrast_10x", r["A_EJ_contrast_10x"],
                  cross[seq]["A_EJ_for_contrast"]["10x"]),
                 ("contrast_2x", r["A_EJ_contrast_2x"],
                  cross[seq]["A_EJ_for_contrast"]["2x"])]
        map_vs_root[seq] = {
            n: dict(map_interp=a, root_solve=b,
                    reldev=(abs(a - b) / b if (a is not None and np.isfinite(a)
                                               and np.isfinite(b)) else None))
            for n, a, b in pairs}

    # the headline replacement table at A = 1e-4
    replacement = []
    for aej in (0.0, 1e-8, 3e-8, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4):
        r = dict(A_EJ=aej)
        for seq in SEQUENCES:
            t2 = T2_two_channel(0.0, D2_apex, A_ref, D_EJ, aej, seq, w_ir, M)
            o2 = T2_two_channel(D1_off, D2_off, A_ref, D_EJ, aej, seq, w_ir, M)
            r[f"apex_T2_{seq}_us"] = t2 * 1e6
            r[f"offapex_T2_{seq}_us"] = o2 * 1e6
            r[f"contrast_{seq}"] = t2 / o2
            r[f"apex_fraction_of_P4_{seq}"] = t2 / apex0[seq]
        replacement.append(r)

    # ------------------------------------------------------------------ write --
    doc = dict(
        title="E_J 1/f noise: the missing third clause of the double-sweet-spot "
              "law (the apex never protects against critical-current noise)",
        exploration="M6 exploration item 2/5 (read-only; no repo file modified)",
        generator="scripts/explore_ej_1f_third_clause.py",
        deterministic="no RNG anywhere -- no seed required",
        runtime_s=None,
        question=(
            "The dome apex zeroes df01/dnu, but df01/dlnE_J never vanishes. What "
            "1/f critical-current noise amplitude A_EJ silently replaces the P4 "
            "headline apex numbers, and what does the 1292x / 380x sweet-spot "
            "contrast collapse to as a function of (A, A_EJ)?"),
        conventions=dict(
            gate_noise="S_nu(w) = A^2/|w| two-sided angular (P4, unchanged)",
            EJ_noise="S_dlnEJ(w) = A_EJ^2/|w| two-sided angular; A_EJ is "
                     "FRACTIONAL E_J (= Ic) noise, dlnE_J = dlnIc = dlnDelta",
            one_sided_translation="S^(1)(f) = 2 A_EJ^2 / f, so "
                                  "sqrt(S_Ic(1 Hz))/Ic = sqrt(2) * A_EJ Hz^-1/2",
            f_ir_Hz=w_ir / (2 * np.pi), w_ir_rad_per_s=w_ir, M_time_grid=M,
            T2_definition="chi(T2) = 1, Ramsey and echo alike (P4 definition)",
            channel_additivity="gate-filling noise and E_J noise are INDEPENDENT "
                               "processes, so their chi add EXACTLY (product of "
                               "characteristic functions). Within the gate "
                               "channel P4's chi_1 + chi_2 rule is used; it is "
                               "exact at the apex (D1 = 0) and inherits P4's "
                               "caveat at nu = -2.5 (quote A <= 1e-3).",
            src="qubit/noise_1f.py (used verbatim), qubit.json:budgets.T2_maps"),
        free_parameters=dict(
            A=dict(range=[float(A_grid[0]), float(A_grid[-1])], n=nA,
                   meaning="1/f filling-noise amplitude (P4 grid, unchanged)"),
            A_EJ=dict(range=[float(A_EJ_grid[0]), float(A_EJ_grid[-1])], n=nE,
                      meaning="1/f fractional critical-current / E_J noise "
                              "amplitude -- NEVER measured for a TBG junction",
                      orientation_only="conventional Josephson junctions are "
                              "usually quoted around 1e-6..1e-5 Hz^-1/2 for "
                              "sqrt(S_Ic(1Hz))/Ic with a ~1/sqrt(area) scaling "
                              "[Van Harlingen et al., PRB 70, 064517 (2004)]; "
                              "this is NOT a calibration for TBG and no absolute "
                              "TBG coherence number is claimed anywhere here"),
            h_FD=dict(value=H_FD, meaning="relative central-difference step on "
                                          "E_J; convergence table below")),
        design_point=dict(EJ_GHz=EJ, EC_GHz=EC, f01_GHz=f01_apex,
                          EJ_over_EC=int(dp["EJ_over_EC"]), N_charge=N_CHARGE,
                          src="qubit.design_point() <- junction.json Ic = 300 nA"),
        # ---------------------------------------------------------------------
        D_EJ=dict(
            value_GHz=D_EJ,
            definition="D_EJ = df01/dln E_J at the design point, central "
                       "finite difference on qubit.spectrum.levels",
            harmonic_limit_GHz=D_EJ_harmonic,
            harmonic_formula="sqrt(8 E_J E_C)/2",
            reldev_vs_harmonic=D_EJ / D_EJ_harmonic - 1.0,
            fd_convergence=fd_table,
            fd_spread_over_4_decades=float(
                max(v["two_point"] for k, v in fd_table.items() if k != "1e-07")
                - min(v["two_point"] for k, v in fd_table.items() if k != "1e-07")),
            charge_truncation_convergence=N_table,
            never_vanishes="D_EJ > 0 everywhere on the dome: it is (1/2) the "
                           "plasma frequency to leading order, so it can only "
                           "vanish where E_J does, i.e. at the dome edges where "
                           "the qubit itself ceases to exist",
            second_order=second_order_EJ_bound),
        chain_rule=dict(
            identity="D1(nu) = D_EJ(E_J(nu)) * dln E_J/dnu (nu)",
            statement="THE THIRD CLAUSE. The dome apex is a sweet spot only "
                      "because the SECOND factor vanishes there (dE_J/dnu = 0 at "
                      "the top of the parabolic dome). The FIRST factor, D_EJ, is "
                      "maximal at the apex -- the apex maximises E_J, and D_EJ "
                      "grows as sqrt(E_J). The double sweet spot therefore "
                      "protects against noise in nu and against nothing else.",
            table=chain,
            src="qubit/dome.py (material.json:sc_dome) + qubit/spectrum.py"),
        validation=dict(
            reproduced_stored_rows=repro,
            solver_reduces_to_noise_1f=solver_check,
            note="every 'stored' value is read from qubit.json at run time "
                 "(budgets.T2_maps.headline_rows / .quasistatic_reference); "
                 "nothing is typed from memory"),
        # ---------------------------------------------------------------------
        EJ_channel_alone=dict(
            A_EJ=_round(A_EJ_grid),
            T2_ramsey_us=_round(T2_EJ["ramsey"] * 1e6),
            T2_echo_us=_round(T2_EJ["echo"] * 1e6),
            echo_over_ramsey=_round(echo_over_ramsey_EJ),
            echo_over_ramsey_mean=float(echo_over_ramsey_EJ.mean()),
            scaling=scaling,
            scaling_statement="the echo T2 scales EXACTLY as 1/A_EJ (log-slope "
                              "-1 to 1e-12) because the echo filter kills the IR "
                              "log; the Ramsey T2 scales as 1/A_EJ divided by "
                              "sqrt(ln(1/(w_ir T2)) + 3/2 - gamma_E), i.e. a "
                              "slightly shallower measured log-slope"),
        # ---------------------------------------------------------------------
        grids=dict(A=_round(A_grid), A_EJ=_round(A_EJ_grid)),
        maps_us=dict(
            axis_order="[i = A index][j = A_EJ index]",
            apex_T2_ramsey=_round2d(maps["ramsey"]["apex"] * 1e6),
            apex_T2_echo=_round2d(maps["echo"]["apex"] * 1e6),
            offapex_T2_ramsey=_round2d(maps["ramsey"]["off"] * 1e6),
            offapex_T2_echo=_round2d(maps["echo"]["off"] * 1e6),
            contrast_ramsey=_round2d(maps["ramsey"]["contrast"]),
            contrast_echo=_round2d(maps["echo"]["contrast"]),
            baseline_A_EJ_zero=dict(
                apex_T2_ramsey_us=_round(base["ramsey"]["apex"] * 1e6),
                apex_T2_echo_us=_round(base["echo"]["apex"] * 1e6),
                offapex_T2_ramsey_us=_round(base["ramsey"]["off"] * 1e6),
                offapex_T2_echo_us=_round(base["echo"]["off"] * 1e6)),
            quoting_caveat="the off-apex column uses P4's chi_1 + chi_2 rule; "
                           "qubit.json:budgets.T2_maps.additivity_rule says quote "
                           "it for A <= 1e-3 only. The apex column is exact."),
        headline_replacement_table_at_A_1e_4=dict(
            A=A_ref, rows=replacement,
            stored_P4_contrast=stored_contrast),
        crossovers=dict(A=A_ref, **cross),
        experimental_discriminator=discriminator,
        survival_ridge=dict(
            definition="for each filling-noise amplitude A, the E_J-noise "
                       "amplitude A_EJ at which the apex/off-apex(nu=-2.5) "
                       "contrast has fallen to 100x / 10x / 2x, and at which the "
                       "apex T2 is halved. Log-interpolated on the (A, A_EJ) "
                       "map; null = the target lies outside the sampled A_EJ "
                       "range (i.e. already crossed at A_EJ = 0, or never "
                       "reached).",
            rows=ridge,
            proportional_law=dict(
                statement="the ridge is a straight line in the log-log "
                          "(A, A_EJ) plane. The CONTRAST crossings have "
                          "exponent 1 (A_EJ_crit ~ A) because they compare two "
                          "FIRST-order channels; the apex-halved crossing has "
                          "exponent ~2 (A_EJ_crit ~ A^2) because it compares "
                          "the first-order E_J channel against the SECOND-order "
                          "gate channel, whose T2 goes as 1/A^2. That exponent "
                          "is itself the fingerprint of the order of each "
                          "channel.",
                k=ridge_law,
                closed_form=contrast_law,
                dnu_eq_definition="dnu_eq = D_EJ * k / |D2_apex| -- the "
                                  "unit-free filling offset that would produce "
                                  "the same first-order sensitivity"),
            map_vs_rootsolve_crosscheck=map_vs_root),
        equivalent_apex_offset=dict(
            definition="|dnu_eq| such that the FIRST-order gate channel at "
                       "nu = -2.4 + dnu_eq equals the E_J channel: "
                       "|D2_apex| |dnu_eq| A = D_EJ A_EJ",
            A=A_ref, dnu_eq_vs_A_EJ=dnu_equiv,
            reference_scales=dict(
                P4_nu_grid_step=float(nz.NU_STEP),
                P3_pair_breaking_halfwidth=0.058571,
                dome_halfwidth=0.5)),
        claim_discipline=(
            "DESIGN STUDY. A_EJ is a FREE parameter and has never been measured "
            "for a TBG junction; every number here is parametric in (A, A_EJ). "
            "No absolute TBG coherence claim is made."),
    )
    doc["runtime_s"] = time.time() - t_start
    doc = _sanitize(doc)
    with open(OUTFILE, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1, allow_nan=False)
    print(f"wrote {OUTFILE} ({OUTFILE.stat().st_size / 1024:.1f} kB) "
          f"in {doc['runtime_s']:.1f} s")
    return doc


if __name__ == "__main__":
    main()
