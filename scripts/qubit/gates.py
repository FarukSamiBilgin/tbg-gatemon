"""M6 production task P6 -- gate-set sketch for the TBG gatemon.

Three questions, in order of how much of the answer is *simulated* rather
than quoted:

  (a) SINGLE-QUBIT GATE TIME FLOOR.  Solved, not quoted.  A three-level
      time-dependent Schrodinger equation is integrated in the rotating frame
      with a deterministic fixed-step RK4, for a truncated-Gaussian pi pulse
      whose amplitude is re-calibrated at every gate time, with and without
      first-order DRAG.  Leakage L(t_g) = P_2 at the end of the pulse (start
      in |0>) is scanned over t_g = 0.5 .. 20 ns and the t_g at which
      L = 1e-3 and L = 1e-4 is extracted by log-log interpolation on a
      refined bracket.

  (b) DRAG REQUIREMENT.  The optimal first-order DRAG coefficient is *scanned*
      (not assumed to be 1), and the derivative-quadrature amplitude that the
      control electronics must deliver at the measured floor is reported both
      as a measured peak and against its closed-form scaling.

      Also reported here, because leaving it out would make the floor in (a)
      read as a gate error: amplitude calibration alone leaves a residual P_0
      four orders of magnitude ABOVE the leakage (the |2> level Stark-shifts
      the 0-1 transition, tilting the rotation axis).  `rotation_error`
      measures it, `combined_floor` re-runs the t_g scan with both knobs
      (amplitude and DRAG frame coefficient) calibrated and reports the gate
      time at which max(P_0, P_2) clears each threshold.  That combined number
      is the honest engineering floor; the (a) numbers are leakage only.

  (c) FREQUENCY ALLOCATION.  f01(nu) across the SC dome (qubit/dome.py),
      overlaid with the P3 pair-breaking constraint
      (out/m6/p3_qp.json:pair_breaking_audit).  The channel spacing is *not*
      a rule of thumb: a spectator qubit detuned by df is driven with the same
      calibrated pulse in the same 3-level integrator, and the spacing
      criterion is the detuning beyond which its excitation stays below a
      stated tolerance.

  (d) TWO-QUBIT GATES.  Parametric SKETCH ONLY -- closed-form rates for a
      capacitively coupled pair, no simulation, no fidelity.  Every number in
      section (d) is labelled `sketch` in the export.

CLAIM DISCIPLINE (M6_BRIEF.md Sec. "Claim discipline for paper 2").  Every
error number in this module is a *coherent leakage/crosstalk* number computed
from a closed-system Schrodinger equation.  No decoherence (T1, T2, quasi-
particles, 1/f gate noise) is folded in anywhere.  These are therefore FLOORS
on the gate time, not fidelities, and the module never exports a quantity
named or framed as a gate fidelity.

DETERMINISM.  Fixed-step RK4 throughout, step count set by an explicit
steps-per-period floor (`SPP`, well above the house minimum of 100) and
verified by a step-halving convergence check that is exported.  No RNG is
constructed or consumed anywhere in P6.

Model (rotating frame at the drive frequency, RWA, three levels; Motzoi,
Gambetta, Rebentrost, Wilhelm, PRL 103, 110501 (2009)):

    H(t)/hbar = [[ 0,          Om*(t)/2,        0             ],
                 [ Om(t)/2,    d1,              lam*Om*(t)/2  ],
                 [ 0,          lam*Om(t)/2,     d2            ]]

    Om(t) = Om_x(t) + i Om_y(t),   Om_x = E(t),   Om_y = -q E'(t)/delta
    d1 = 2 pi df,   d2 = 2 d1 + delta,   delta = 2 pi * anharmonicity

`lam` is the ladder ratio |<1|n|2>| / |<0|n|1>|, taken from the *exact*
charge-basis eigenvectors of qubit/spectrum.py rather than the harmonic
sqrt(2): at E_J/E_C = 100 it is 1.3852, 2.05% below sqrt(2).
"""
import json
import math
import pathlib

import numpy as np

from . import dome, spectrum
from .inputs import EJ_GHZ, N_CHARGE, NU_HALFWIDTH, NU_OPT, ROOT, TARGETS

# ------------------------------------------------------------- numerics ------
SPP = 200.0          # steps per period of the fastest frame frequency
                     # (house rule: >= 100; 2x margin taken)
N_STEPS_MIN = 2000   # floor on the RK4 step count for any single pulse
GOLDEN_ITERS = 24    # amplitude-calibration golden-section iterations
SIGMA_RATIO = 4.0    # Gaussian truncated at t_g = 4 sigma (+-2 sigma)

# --------------------------------------------------------------- design ------
EJ_OVER_EC = TARGETS["design"]["EJ_over_EC"]
# src: qubit_targets.json:design.EJ_over_EC
EC_GHZ = EJ_GHZ / EJ_OVER_EC
# E_J from junction.json:contract.Ic_nA via qubit/inputs.py; E_C from the
# locked design ratio.  Never typed.

LEAK_THRESHOLDS = (1e-3, 1e-4)   # task-specified leakage levels
TG_SCAN = (0.5, 20.0)            # task-specified gate-time scan window, ns


# ============================================================ level data =====
def ladder(EJ=EJ_GHZ, EC=EC_GHZ, N=N_CHARGE):
    """Exact 3-level data from the charge-basis eigenproblem.

    Returns f01, f12, anharm (GHz), n01, n12 and the ladder ratio n12/n01
    that sets the |1>-|2> drive matrix element relative to |0>-|1>.
    """
    n = np.arange(-N, N + 1)
    w, v = np.linalg.eigh(spectrum.hamiltonian(EJ, EC, 0.0, N))
    n01 = abs(float(v[:, 0] @ (n * v[:, 1])))
    n12 = abs(float(v[:, 1] @ (n * v[:, 2])))
    f01, f12 = float(w[1] - w[0]), float(w[2] - w[1])
    return dict(f01_GHz=f01, f12_GHz=f12, anharm_GHz=f12 - f01,
                n01=n01, n12=n12, ladder_ratio=n12 / n01,
                ladder_ratio_over_sqrt2=n12 / n01 / math.sqrt(2.0))


_LAD = ladder()
DELTA_ANG = 2.0 * np.pi * _LAD["anharm_GHz"]      # rad/ns, negative
LAM = _LAD["ladder_ratio"]


# ============================================================== envelope =====
def envelope(t, t_g, amp, sigma_ratio=SIGMA_RATIO):
    """Truncated, lifted Gaussian envelope and its exact derivative.

    E(t) = amp * (G(t) - c) / (1 - c),  G = exp(-(t - t_g/2)^2 / 2 sigma^2),
    sigma = t_g / sigma_ratio, c = G(0) = G(t_g).  The lift makes the pulse
    start and end at exactly zero, which is what makes the end-of-pulse
    leakage a well-defined gate property rather than a truncation artefact.

    `t`, `t_g`, `amp` broadcast together.  Units: rad/ns for E, rad/ns^2 for
    the derivative, ns for t.
    """
    sig = t_g / sigma_ratio
    c = np.exp(-(t_g ** 2) / (8.0 * sig ** 2))
    u = t - t_g / 2.0
    gauss = np.exp(-(u ** 2) / (2.0 * sig ** 2))
    norm = amp / (1.0 - c)
    return norm * (gauss - c), norm * (-u / sig ** 2) * gauss


def shape_area(t_g, sigma_ratio=SIGMA_RATIO, n=200001):
    """Integral of the unit-amplitude envelope over the pulse (rad-ns/rad)."""
    t = np.linspace(0.0, t_g, n)
    E, _ = envelope(t, t_g, 1.0, sigma_ratio)
    return float(np.trapezoid(E, t))


def pi_amplitude(t_g, sigma_ratio=SIGMA_RATIO):
    """Amplitude whose pulse area is exactly pi (ideal two-level pi pulse).

    Starting point for the calibration; the 3-level system needs a few percent
    more because population parked in |2> mid-pulse is missing from the
    |0>-|1> Rabi.
    """
    return np.pi / shape_area(t_g, sigma_ratio)


def n_steps_for(t_g, detuning_GHz=0.0, amp=0.0, lam=LAM, delta=DELTA_ANG):
    """Fixed step count from an explicit steps-per-period floor.

    The fastest angular frequency the integrator has to resolve is the largest
    of the frame frequencies |d1|, |d2| and the peak drive rate lam*|Om|.
    """
    d1 = 2.0 * np.pi * np.abs(detuning_GHz)
    d2 = np.abs(2.0 * 2.0 * np.pi * np.asarray(detuning_GHz) + delta)
    w_max = np.maximum(np.maximum(d1, d2), lam * np.abs(amp))
    w_max = np.maximum(w_max, abs(delta))
    n = np.ceil(SPP * np.asarray(t_g) * w_max / (2.0 * np.pi))
    return int(max(N_STEPS_MIN, np.max(n)))


# ============================================================ propagator =====
def propagate(t_g, amp, drag_q=0.0, detuning_GHz=0.0, n_steps=None,
              lam=LAM, delta=DELTA_ANG, sigma_ratio=SIGMA_RATIO, psi0=None,
              stark_s=0.0):
    """Fixed-step RK4 of the 3-level rotating-frame TDSE, vectorised.

    All of `t_g`, `amp`, `drag_q`, `detuning_GHz` broadcast to a common batch
    shape (B,) and are integrated in ONE call -- each batch member gets its own
    dt = t_g/n_steps but they share the step count, so the whole bias grid is a
    single integrator call (house rule).

    Returns the final state, shape (B, 3), complex.
    """
    t_g, amp, drag_q, det, stark = np.broadcast_arrays(
        np.asarray(t_g, float), np.asarray(amp, float),
        np.asarray(drag_q, float), np.asarray(detuning_GHz, float),
        np.asarray(stark_s, float))
    t_g = np.atleast_1d(t_g).ravel()
    amp = np.atleast_1d(amp).ravel()
    drag_q = np.atleast_1d(drag_q).ravel()
    det = np.atleast_1d(det).ravel()
    stark = np.atleast_1d(stark).ravel()
    B = t_g.size

    if n_steps is None:
        n_steps = n_steps_for(t_g, det, amp, lam, delta)

    # Pre-tabulate the complex drive on the RK4 stage grid (t, t+dt/2, t+dt):
    # half-step resolution over the whole pulse, shape (2*n_steps+1, B).
    dt = t_g / n_steps
    k = np.arange(2 * n_steps + 1)[:, None]
    tt = 0.5 * k * dt[None, :]
    E, Ed = envelope(tt, t_g[None, :], amp[None, :], sigma_ratio)
    Om = E - 1j * (drag_q[None, :] * Ed / delta)     # Om_x + i Om_y
    Omc = np.conj(Om)

    # Optional first-order DRAG frame (Stark) correction: the |2> level pulls
    # the 0-1 transition during the pulse, so an on-resonance drive is
    # effectively detuned.  Under THIS code's sign convention (the correction
    # is subtracted from d1 below), Motzoi et al. Eq. (6) reads
    #     d_drive(t) = -s * (lam^2 - 4) * Om_x(t)^2 / (4 delta),
    # so s = -1 is the Motzoi first-order frame correction (matches the
    # rotation_error() definition string and the measured s_opt -> -1.000).
    # s = 0 reproduces the uncorrected gate exactly.
    d1 = 2.0 * np.pi * det[None, :] - (
        stark[None, :] * (lam ** 2 - 4.0) * E ** 2 / (4.0 * delta))
    d2 = 2.0 * d1 + delta

    if psi0 is None:
        psi = np.zeros((B, 3), complex)
        psi[:, 0] = 1.0
    else:
        psi = np.broadcast_to(np.asarray(psi0, complex), (B, 3)).copy()

    def deriv(j, ps):
        """-i H(t_j) psi at half-step index j."""
        om, omc = Om[j], Omc[j]
        out = np.empty_like(ps)
        out[:, 0] = 0.5 * omc * ps[:, 1]
        out[:, 1] = 0.5 * om * ps[:, 0] + d1[j] * ps[:, 1] + 0.5 * lam * omc * ps[:, 2]
        out[:, 2] = 0.5 * lam * om * ps[:, 1] + d2[j] * ps[:, 2]
        return -1j * out

    h = dt[:, None]
    for s in range(n_steps):
        j = 2 * s
        k1 = deriv(j, psi)
        k2 = deriv(j + 1, psi + 0.5 * h * k1)
        k3 = deriv(j + 1, psi + 0.5 * h * k2)
        k4 = deriv(j + 2, psi + h * k3)
        psi = psi + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return psi


def populations(psi):
    """|amplitude|^2, shape (B, 3)."""
    return np.abs(psi) ** 2


# =========================================================== calibration =====
def calibrate_amplitude(t_g, drag_q=0.0, n_steps=None, iters=GOLDEN_ITERS,
                        span=(0.70, 1.45), **kw):
    """Golden-section search for the amplitude that empties |0>.

    This is the numerical stand-in for a Rabi amplitude calibration: minimise
    P_0 at the end of the pulse over the amplitude, bracketed around the ideal
    two-level pi-area value.  Deterministic (fixed iteration count, no RNG),
    and vectorised over the batch.
    """
    t_g = np.atleast_1d(np.asarray(t_g, float))
    drag_q = np.broadcast_to(np.asarray(drag_q, float), t_g.shape)
    a0 = np.array([pi_amplitude(x) for x in t_g.ravel()]).reshape(t_g.shape)
    lo, hi = span[0] * a0, span[1] * a0
    if n_steps is None:
        n_steps = n_steps_for(t_g, 0.0, hi, kw.get("lam", LAM),
                              kw.get("delta", DELTA_ANG))

    def P0(a):
        return populations(propagate(t_g, a, drag_q, 0.0, n_steps, **kw))[:, 0]

    inv_phi = (math.sqrt(5.0) - 1.0) / 2.0
    x1 = hi - inv_phi * (hi - lo)
    x2 = lo + inv_phi * (hi - lo)
    f1, f2 = P0(x1), P0(x2)
    for _ in range(iters):
        # Standard golden section with point reuse: exactly ONE propagation
        # per iteration, vectorised over the batch.  The reused point differs
        # per batch member, so the new abscissa is assembled with np.where and
        # then evaluated in a single integrator call.
        left = f1 < f2                       # minimum lies in [lo, x2]
        new_hi = np.where(left, x2, hi)
        new_lo = np.where(left, lo, x1)
        new_x1 = np.where(left, new_hi - inv_phi * (new_hi - new_lo), x2)
        new_x2 = np.where(left, x1, new_lo + inv_phi * (new_hi - new_lo))
        probe = np.where(left, new_x1, new_x2)
        fp = P0(probe)
        new_f1 = np.where(left, fp, f2)      # left: fresh x1; right: reused x2
        new_f2 = np.where(left, f1, fp)      # left: reused x1; right: fresh x2
        lo, hi, x1, x2, f1, f2 = new_lo, new_hi, new_x1, new_x2, new_f1, new_f2
    return 0.5 * (lo + hi)


# ============================================================== leakage ======
def leakage_scan(t_g, drag_q=0.0, n_steps=None, calibrate=True, **kw):
    """Leakage L = P_2(t_g) starting from |0>, for a batch of gate times.

    Returns dict of arrays: t_g_ns, amp_rad_per_ns, leakage, P0, P1, norm.
    """
    t_g = np.atleast_1d(np.asarray(t_g, float))
    drag_q = np.broadcast_to(np.asarray(drag_q, float), t_g.shape).copy()
    amp = (calibrate_amplitude(t_g, drag_q, **kw) if calibrate
           else np.array([pi_amplitude(x) for x in t_g]))
    if n_steps is None:
        n_steps = n_steps_for(t_g, 0.0, amp, kw.get("lam", LAM),
                              kw.get("delta", DELTA_ANG))
    P = populations(propagate(t_g, amp, drag_q, 0.0, n_steps, **kw))
    return dict(t_g_ns=t_g, amp_rad_per_ns=amp, drag_q=drag_q,
                leakage=P[:, 2], P0=P[:, 0], P1=P[:, 1], norm=P.sum(axis=1),
                n_steps=n_steps)


def _log_cross(x, y, target):
    """Largest x at which the monotone-decreasing y crosses `target`.

    Log-log linear interpolation between the bracketing samples.  Taking the
    LAST crossing rather than the first makes the answer a genuine floor even
    if y is not perfectly monotone.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    idx = np.nonzero(y >= target)[0]
    if idx.size == 0 or idx[-1] == x.size - 1:
        return None
    i = idx[-1]
    lx0, lx1 = math.log(x[i]), math.log(x[i + 1])
    ly0, ly1 = math.log(y[i]), math.log(y[i + 1])
    return float(math.exp(lx0 + (math.log(target) - ly0) * (lx1 - lx0)
                          / (ly1 - ly0)))


def gate_floor(thresholds=LEAK_THRESHOLDS, t_span=TG_SCAN, n_coarse=61,
               n_refine=41, **kw):
    """t_g at which leakage reaches each threshold, Gaussian vs DRAG.

    Coarse log scan over `t_span`, then one refined log scan inside the
    bracketing interval of every (mode, threshold) crossing.  Both scans are
    single vectorised integrator calls.
    """
    tg = np.geomspace(t_span[0], t_span[1], n_coarse)
    modes = {"gaussian": 0.0, "drag": 1.0}

    # one integrator call for both modes
    tg_all = np.concatenate([tg, tg])
    q_all = np.concatenate([np.zeros(n_coarse), np.ones(n_coarse)])
    sc = leakage_scan(tg_all, q_all, **kw)
    coarse = {m: dict(t_g_ns=tg,
                      leakage=sc["leakage"][i * n_coarse:(i + 1) * n_coarse],
                      amp_rad_per_ns=sc["amp_rad_per_ns"][i * n_coarse:(i + 1) * n_coarse],
                      P1=sc["P1"][i * n_coarse:(i + 1) * n_coarse],
                      norm=sc["norm"][i * n_coarse:(i + 1) * n_coarse])
              for i, m in enumerate(modes)}

    # refined bracket per (mode, threshold)
    floors, refine_tg, refine_q, keys = {}, [], [], []
    for m, q in modes.items():
        y = coarse[m]["leakage"]
        for thr in thresholds:
            idx = np.nonzero(y >= thr)[0]
            if idx.size == 0 or idx[-1] == tg.size - 1:
                floors[(m, thr)] = None
                continue
            i = idx[-1]
            sub = np.geomspace(tg[i], tg[i + 1], n_refine)
            refine_tg.append(sub)
            refine_q.append(np.full(n_refine, q))
            keys.append((m, thr))
    if keys:
        rt = np.concatenate(refine_tg)
        rq = np.concatenate(refine_q)
        rs = leakage_scan(rt, rq, **kw)
        for j, key in enumerate(keys):
            s = slice(j * n_refine, (j + 1) * n_refine)
            floors[key] = _log_cross(rt[s], rs["leakage"][s], key[1])

    out = {}
    for m in modes:
        y = coarse[m]["leakage"]
        # Leakage is not monotone everywhere: around t_g ~ 1.4 ns the |2>
        # amplitude picked up on the rising edge interferes with the amplitude
        # picked up on the falling edge and the curve wiggles.  Rather than
        # assert monotonicity globally, MEASURE where it starts and export it,
        # so the floor extraction (which takes the LAST crossing) can be
        # checked to sit inside the monotone region.
        viol = np.nonzero(np.diff(y) >= 0.0)[0]
        mono_from = float(tg[viol[-1] + 1]) if viol.size else float(tg[0])
        fl = {f"{t:.0e}": floors[(m, t)] for t in thresholds}
        out[m] = dict(
            t_g_ns=[float(x) for x in coarse[m]["t_g_ns"]],
            leakage=[float(x) for x in y],
            amp_rad_per_ns=[float(x) for x in coarse[m]["amp_rad_per_ns"]],
            P1_end=[float(x) for x in coarse[m]["P1"]],
            norm_min=float(coarse[m]["norm"].min()),
            norm_max_dev=float(np.max(np.abs(coarse[m]["norm"] - 1.0))),
            monotone_decreasing=bool(viol.size == 0),
            monotone_from_t_g_ns=mono_from,
            n_monotonicity_violations=int(viol.size),
            monotonicity_violation_t_g_ns=[float(tg[i]) for i in viol],
            t_g_floor_ns=fl,
        )
    out["speedup_drag_over_gaussian"] = {
        f"{t:.0e}": (None if not (floors.get(("gaussian", t))
                                  and floors.get(("drag", t)))
                     else floors[("gaussian", t)] / floors[("drag", t)])
        for t in thresholds}
    return out


def convergence_check(t_g=2.0, drag_q=(0.0, 1.0), factors=(1, 2, 4), **kw):
    """Step-halving check: leakage vs RK4 step count at fixed amplitude."""
    q = np.asarray(drag_q, float)
    tg = np.full(q.shape, float(t_g))
    amp = calibrate_amplitude(tg, q, **kw)
    base = n_steps_for(tg, 0.0, amp)
    rows = {}
    for f in factors:
        P = populations(propagate(tg, amp, q, 0.0, base * f, **kw))
        rows[str(base * f)] = dict(
            leakage=[float(x) for x in P[:, 2]],
            norm_dev=[float(abs(x - 1.0)) for x in P.sum(axis=1)])
    ref = rows[str(base * factors[-1])]["leakage"]
    rows["rel_change_vs_finest"] = {
        str(base * f): [abs(rows[str(base * f)]["leakage"][i] - ref[i])
                        / max(ref[i], 1e-300) for i in range(q.size)]
        for f in factors[:-1]}
    rows["drag_q"] = [float(x) for x in q]
    rows["t_g_ns"] = float(t_g)
    rows["steps_per_period_floor"] = SPP
    return rows


# ================================================================= DRAG ======
def drag_coefficient_scan(t_g, q_span=(0.0, 2.0), n=81, **kw):
    """Leakage vs DRAG coefficient q at fixed t_g; returns the measured optimum.

    q is defined by Om_y(t) = -q * E'(t) / delta, so the textbook first-order
    DRAG solution is q = 1.  The optimum is *measured*, not assumed.
    """
    q = np.linspace(q_span[0], q_span[1], n)
    tg = np.full(n, float(t_g))
    sc = leakage_scan(tg, q, **kw)
    y = sc["leakage"]
    i = int(np.argmin(y))
    # parabolic refinement in q on the three points around the minimum
    if 0 < i < n - 1:
        y0, y1, y2 = math.log(y[i - 1]), math.log(y[i]), math.log(y[i + 1])
        denom = y0 - 2.0 * y1 + y2
        shift = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
        q_opt = float(q[i] + shift * (q[1] - q[0]))
    else:
        q_opt = float(q[i])
    return dict(t_g_ns=float(t_g), q=[float(x) for x in q],
                leakage=[float(x) for x in y],
                q_opt_grid=float(q[i]), q_opt_parabolic=q_opt,
                leakage_at_q_opt_grid=float(y[i]),
                leakage_at_q1=float(np.interp(1.0, q, y)),
                gain_q1_over_qopt=float(np.interp(1.0, q, y) / y[i]))


def calibrate_two_knob(t_g, drag_q=1.0, s_span=(-3.0, 3.0), n_s=61, rounds=2,
                       **kw):
    """Calibrate BOTH knobs: pulse amplitude and the DRAG frame coefficient s.

    Amplitude alone cannot make a good gate.  The |2> level Stark-shifts the
    0-1 transition during the pulse, so a nominally resonant drive is
    effectively detuned and the rotation axis tilts off the equator; the
    residual P_0 that leaves is far larger than the leakage.  `s` is the
    first-order DRAG frame correction that removes it.

    The s scan is one integrator call over the OUTER product of the t_g batch
    and the s grid, alternated with an amplitude re-calibration.  Deterministic
    (fixed grids, fixed round count, no RNG).  Returns (amp, s_opt).
    """
    tg = np.atleast_1d(np.asarray(t_g, float))
    q = np.broadcast_to(np.asarray(drag_q, float), tg.shape)
    amp = calibrate_amplitude(tg, q, **kw)
    s_grid = np.linspace(s_span[0], s_span[1], n_s)
    s_opt = np.zeros_like(tg)
    for _ in range(rounds):
        TG, Q, A = np.repeat(tg, n_s), np.repeat(q, n_s), np.repeat(amp, n_s)
        S = np.tile(s_grid, tg.size)
        P = populations(propagate(TG, A, Q, 0.0, n_steps_for(TG, 0.0, A),
                                  stark_s=S, **kw))
        P0 = P[:, 0].reshape(tg.size, n_s)
        j = np.argmin(P0, axis=1)
        s_opt = s_grid[j].copy()
        for i, jj in enumerate(j):          # parabolic refinement in s
            if 0 < jj < n_s - 1:
                y0, y1, y2 = (math.log(P0[i, jj - 1]), math.log(P0[i, jj]),
                              math.log(P0[i, jj + 1]))
                den = y0 - 2.0 * y1 + y2
                if den != 0:
                    s_opt[i] += 0.5 * (y0 - y2) / den * (s_grid[1] - s_grid[0])
        amp = calibrate_amplitude(tg, q, stark_s=s_opt, **kw)
    return amp, s_opt


def rotation_error(t_g_list, drag_q=1.0, **kw):
    """What the leakage-only floor does NOT capture, measured.

    Reported so that the exported gate-time floor is read for what it is: a
    LEAKAGE floor, not a gate error.
    """
    tg = np.atleast_1d(np.asarray(t_g_list, float))
    q = np.full(tg.shape, float(drag_q))
    amp0 = calibrate_amplitude(tg, q, **kw)
    P_before = populations(propagate(tg, amp0, q, 0.0,
                                     n_steps_for(tg, 0.0, amp0), **kw))
    amp, s_opt = calibrate_two_knob(tg, q, **kw)
    P_after = populations(propagate(tg, amp, q, 0.0,
                                    n_steps_for(tg, 0.0, amp),
                                    stark_s=s_opt, **kw))
    return dict(
        # SIGN CONVENTION (paper2_trace.md F6): `propagate` implements
        #   d1 = 2*pi*det - s (lam^2 - 4) E^2 / (4 delta)
        # i.e. the frame shift enters with a MINUS sign, so s = -1 -- not
        # s = +1 -- is the Motzoi first-order frame correction under this
        # code's own convention.  The string below states the implemented
        # sign; it previously read "s = 1", which contradicted the code.
        definition="d_drive(t) = -s (lam^2 - 4) Om_x(t)^2 / (4 delta) "
                   "(the sign qubit/gates.py:propagate implements); "
                   "s = -1 is the Motzoi first-order frame correction",
        drag_q=float(drag_q),
        rows=[dict(t_g_ns=float(tg[i]),
                   s_opt=float(s_opt[i]),
                   P0_amplitude_only=float(P_before[i, 0]),
                   leakage_amplitude_only=float(P_before[i, 2]),
                   P0_with_frame_correction=float(P_after[i, 0]),
                   leakage_with_frame_correction=float(P_after[i, 2]),
                   P0_improvement=float(P_before[i, 0] / P_after[i, 0]),
                   leakage_change=float(P_after[i, 2] / P_before[i, 2]))
              for i in range(tg.size)],
        note=("the exported t_g floor is a LEAKAGE floor; the dominant "
              "coherent error at that t_g is this rotation error, which is "
              "removed by a frame correction and not by a slower gate"))


def combined_floor(t_span=(0.8, 8.0), n=41, drag_q=1.0,
                   thresholds=LEAK_THRESHOLDS, **kw):
    """Gate time at which BOTH coherent errors clear each threshold.

    Error metric = max(P_0, P_2) at the end of a pi pulse whose amplitude AND
    DRAG frame coefficient are both calibrated.  This is the honest engineering
    gate time; the headline `gate_floor` numbers are leakage only and are
    optimistic wherever the rotation error dominates.

    Still coherent-only: no T1, T2, quasiparticle or 1/f contribution.
    """
    tg = np.geomspace(t_span[0], t_span[1], n)
    q = np.full(tg.shape, float(drag_q))
    amp, s_opt = calibrate_two_knob(tg, q, **kw)
    P = populations(propagate(tg, amp, q, 0.0, n_steps_for(tg, 0.0, amp),
                              stark_s=s_opt, **kw))
    err = np.maximum(P[:, 0], P[:, 2])
    return dict(
        metric="max(P0, P2) after calibrating both amplitude and DRAG frame s",
        drag_q=float(drag_q),
        t_g_ns=[float(x) for x in tg],
        P0=[float(x) for x in P[:, 0]],
        leakage=[float(x) for x in P[:, 2]],
        combined_error=[float(x) for x in err],
        s_opt=[float(x) for x in s_opt],
        amp_rad_per_ns=[float(x) for x in amp],
        norm_max_dev=float(np.max(np.abs(P.sum(axis=1) - 1.0))),
        t_g_floor_ns={f"{t:.0e}": _log_cross(tg, err, t) for t in thresholds},
        claim="coherent errors only -- no decoherence, not a fidelity")


def drag_requirement(t_g, drag_q=1.0, **kw):
    """Derivative-quadrature amplitude the electronics must deliver at `t_g`.

    Measured peaks plus the closed-form ratio
        max|Om_y| / max|Om_x| = 1 / ((1 - e^-2) * sigma * sqrt(e) * |delta|),
    sigma = t_g / 4, delta = 2 pi * anharmonicity.
    """
    tg = np.atleast_1d(np.asarray(t_g, float))
    q = np.broadcast_to(np.asarray(drag_q, float), tg.shape)
    amp = calibrate_amplitude(tg, q, **kw)
    rows = []
    for i, x in enumerate(tg):
        t = np.linspace(0.0, x, 20001)
        E, Ed = envelope(t, x, amp[i])
        oy = -q[i] * Ed / DELTA_ANG
        sig = x / SIGMA_RATIO
        c = math.exp(-2.0)
        analytic = 1.0 / ((1.0 - c) * sig * math.sqrt(math.e) * abs(DELTA_ANG))
        rows.append(dict(
            t_g_ns=float(x), drag_q=float(q[i]),
            Omx_peak_rad_per_ns=float(np.max(np.abs(E))),
            Omx_peak_GHz=float(np.max(np.abs(E)) / (2.0 * np.pi)),
            Omy_peak_rad_per_ns=float(np.max(np.abs(oy))),
            Omy_peak_GHz=float(np.max(np.abs(oy)) / (2.0 * np.pi)),
            ratio_measured=float(np.max(np.abs(oy)) / np.max(np.abs(E))),
            ratio_analytic=float(analytic * q[i]),
            amp_over_pi_area=float(amp[i] / pi_amplitude(x)),
            sigma_ns=float(sig)))
    return rows


# ======================================================= crosstalk / map =====
def spectator_scan(t_g, detunings_GHz, drag_q=1.0, amp=None, **kw):
    """Unwanted excitation of a spectator qubit detuned by df.

    The spectator starts in |0> and is hit by the SAME calibrated pulse that
    performs the pi gate on the target (drive frequency = target f01).  The
    reported error is P_1 + P_2 at the end -- population that actually left the
    spectator's ground state.  The accompanying AC-Stark phase is reported
    separately because it is deterministic and frame-correctable.
    """
    df = np.atleast_1d(np.asarray(detunings_GHz, float))
    if amp is None:
        amp = float(calibrate_amplitude(np.array([float(t_g)]),
                                        np.array([float(drag_q)]), **kw)[0])
    tg = np.full(df.shape, float(t_g))
    qq = np.full(df.shape, float(drag_q))
    aa = np.full(df.shape, float(amp))
    n_steps = n_steps_for(tg, df, aa)
    psi = propagate(tg, aa, qq, df, n_steps, **kw)
    P = populations(psi)
    phase = np.unwrap(np.angle(psi[:, 0]))
    return dict(detuning_GHz=df, error=P[:, 1] + P[:, 2], P1=P[:, 1],
                P2=P[:, 2], stark_phase_rad=phase, amp_rad_per_ns=float(amp),
                n_steps=int(n_steps), norm=P.sum(axis=1))


def _last_crossing(x, y, tol):
    """Largest x at which y still exceeds tol (log-linear interpolation).

    The LAST crossing, not the first: the spectator error is not monotone in
    detuning -- it carries the side lobes of the pulse spectrum -- so only the
    outermost crossing is a safe spacing.
    """
    idx = np.nonzero(y >= tol)[0]
    if idx.size == 0:
        return float(x[0])
    if idx[-1] == x.size - 1:
        return None                      # never falls below tol on this grid
    i = idx[-1]
    return float(np.interp(math.log(tol),
                           [math.log(y[i + 1]), math.log(y[i])],
                           [x[i + 1], x[i]]))


def min_spacing(t_g, tolerances=(1e-3, 1e-4), df_max=30.0, n=721, drag_q=1.0,
                compare_gaussian=True, df_min=0.02, **kw):
    """Smallest qubit-qubit spacing whose spectator error stays under `tol`.

    The spacing is MEASURED, not assumed to be a multiple of the 1/t_g gate
    bandwidth; the ratio to 1/t_g is reported so the criterion can be read
    either way.

    `compare_gaussian` repeats the scan with the DRAG quadrature switched off.
    That comparison matters: Om_y(t) = -q E'(t)/delta is *discontinuous at the
    pulse edges* for a truncated Gaussian (E'(0) != 0), so DRAG buys a factor
    ~2 in gate time at the cost of a 1/detuning spectral tail, whereas the
    plain Gaussian's edge discontinuity is only in the derivative (1/detuning^2).
    Whichever tail dominates is what actually sets the channel spacing.
    """
    # LOG-spaced detunings: the crossing can land anywhere from ~0.1 to ~15
    # GHz depending on t_g and tolerance, and a linear grid resolves the small
    # ones far too coarsely (the long-t_g rows were grid-quantised at 0.05 GHz
    # before this change).
    df = np.geomspace(df_min, df_max, n)
    sc = spectator_scan(t_g, df, drag_q, **kw)
    out = {f"{t:.0e}": _last_crossing(df, sc["error"], t) for t in tolerances}

    def tail_slope(err):
        """log-log slope of the crosstalk tail over the outer decade."""
        sel = df > df_max / 4.0
        return float(np.polyfit(np.log(df[sel]), np.log(err[sel]), 1)[0])

    res = dict(t_g_ns=float(t_g), drag_q=float(drag_q),
               bandwidth_1_over_tg_GHz=1.0 / float(t_g),
               df_max_scanned_GHz=float(df_max), n_detunings=int(n),
               detuning_GHz=[float(x) for x in df],
               error=[float(x) for x in sc["error"]],
               stark_phase_rad=[float(x) for x in sc["stark_phase_rad"]],
               norm_max_dev=float(np.max(np.abs(sc["norm"] - 1.0))),
               spacing_GHz=out,
               spacing_in_units_of_1_over_tg={
                   k: (None if v is None else v * float(t_g))
                   for k, v in out.items()},
               tail_slope_loglog=tail_slope(sc["error"]),
               criterion=("spacing = largest detuning at which a spectator "
                          "qubit, hit by the same calibrated pulse, still "
                          "leaves its ground state with probability >= tol"),
               mechanism=("far-detuning crosstalk is set by the pulse-edge "
                          "discontinuities, not by the Gaussian core. The "
                          "DRAG quadrature Om_y = -q E'/delta has a STEP at "
                          "t = 0, t_g (amplitude tail ~1/df, error ~1/df^2); "
                          "the envelope itself has only a kink (error "
                          "~1/df^4). Measured slopes are in "
                          "tail_slope_loglog and confirm both exponents."))
    if compare_gaussian:
        sc0 = spectator_scan(t_g, df, 0.0, **kw)
        res["gaussian_comparison"] = dict(
            drag_q=0.0,
            error=[float(x) for x in sc0["error"]],
            tail_slope_loglog=tail_slope(sc0["error"]),
            spacing_GHz={f"{t:.0e}": _last_crossing(df, sc0["error"], t)
                         for t in tolerances})
    return res


def spacing_tradeoff(t_g_list, tolerances=(1e-3, 1e-4), df_max=30.0, n=721,
                     drag_q=1.0, **kw):
    """Channel spacing vs gate time -- the speed/allocation trade curve."""
    rows = []
    for tg in t_g_list:
        s = min_spacing(tg, tolerances, df_max, n, drag_q,
                        compare_gaussian=False, **kw)
        rows.append(dict(t_g_ns=s["t_g_ns"],
                         bandwidth_1_over_tg_GHz=s["bandwidth_1_over_tg_GHz"],
                         spacing_GHz=s["spacing_GHz"],
                         spacing_in_units_of_1_over_tg=
                         s["spacing_in_units_of_1_over_tg"]))
    return rows


def f01_map(n_nu=241, EC=EC_GHZ):
    """f01(nu) across the SC dome -- the frequency-allocation axis.

    Reuses qubit/dome.py (unmodified) for Tc(nu) -> Delta(nu) -> E_J(nu).
    """
    half = NU_HALFWIDTH
    nu = np.linspace(NU_OPT - half, NU_OPT + half, n_nu)
    f = np.array([dome.f01_of(float(x), EC) for x in nu])
    ok = np.isfinite(f)
    return dict(nu=[float(x) for x in nu],
                f01_GHz=[(float(v) if np.isfinite(v) else None) for v in f],
                nu_supported=[float(x) for x in nu[ok]],
                f01_min_GHz=float(np.nanmin(f)), f01_max_GHz=float(np.nanmax(f)),
                nu_apex=float(NU_OPT), nu_halfwidth=float(half))


def allocation(t_g, p3, spacing, tradeoff=None, EC=EC_GHZ):
    """Channel counts for the fixed-drive and tracking-drive scenarios.

    `p3` is out/m6/p3_qp.json:pair_breaking_audit -- the fixed-drive window and
    the f01 span inside it are READ, never recomputed from memory.
    """
    drv = p3["drive_cavity"]
    win_half = float(drv["window_halfwidth_nu"])
    span_fixed = float(drv["f01_span_inside_window_GHz"])

    tun = TARGETS["gate_noise"]["f01_tunability_GHz"]
    # src: qubit_targets.json:gate_noise.f01_tunability_GHz
    span_track = float(tun[1] - tun[0])

    def channels(span, d):
        return None if d is None else int(math.floor(span / d)) + 1

    # coherence cost of leaving the apex: D1 and quasistatic T_phi at the
    # edge of the fixed-drive window (dome.py, unmodified)
    nu_edge = NU_OPT - win_half
    d1_edge = float(dome.D1(nu_edge, EC))
    d2_apex = float(dome.D2(NU_OPT, EC))
    dnu_rms = 1e-4
    tphi_edge = float(dome.Tphi_first_order(d1_edge, dnu_rms)) if d1_edge else None
    tphi_apex = float(dome.Tphi_second_order(d2_apex, dnu_rms))

    fixed = dict(
        scenario="fixed 145 GHz drive tone (photonics.json drive mode)",
        constraint_src="out/m6/p3_qp.json:pair_breaking_audit.drive_cavity",
        nu_window=[float(x) for x in drv["sub_gap_filling_window"]],
        nu_window_halfwidth=win_half,
        window_frac_of_dome_halfwidth=win_half / float(NU_HALFWIDTH),
        f01_span_GHz=span_fixed,
        f01_at_apex_GHz=float(drv["f01_at_apex_GHz"]),
        f01_at_window_edge_GHz=float(drv["f01_at_window_edge_GHz"]),
        channels={k: channels(span_fixed, v)
                  for k, v in spacing["spacing_GHz"].items()},
        degeneracy_note=("f01(nu) is even about the apex, so every f01 inside "
                         "the window except the apex value is realised at two "
                         "fillings; the span, not the window width, is the "
                         "allocation resource"),
        sweet_spot_cost=dict(
            nu_edge=float(nu_edge), D1_edge_GHz_per_nu=d1_edge,
            D2_apex_GHz_per_nu2=d2_apex, dnu_rms=dnu_rms,
            Tphi_apex_us=tphi_apex * 1e6,
            Tphi_edge_us=None if tphi_edge is None else tphi_edge * 1e6,
            contrast=None if tphi_edge is None else tphi_apex / tphi_edge),
    )
    tracking = dict(
        scenario="per-qubit drive tone tracking 2*Delta(nu)",
        f01_range_GHz=[float(tun[0]), float(tun[1])],
        f01_span_GHz=span_track,
        channels={k: channels(span_track, v)
                  for k, v in spacing["spacing_GHz"].items()},
        hardware_requirement=("one drive tone per filling, each held below "
                              "2*Delta(nu)/h; the fixed 145 GHz photonic drive "
                              "mode cannot serve this scenario"),
        sweet_spot_note=("only one qubit can sit at nu = -2.4; every other "
                         "channel pays the first-order gate-noise penalty of "
                         "qubit_targets.json:gate_noise"),
    )
    if tradeoff:
        for row in tradeoff:
            row["channels_fixed_drive"] = {
                k: channels(span_fixed, v) for k, v in row["spacing_GHz"].items()}
            row["channels_tracking_drive"] = {
                k: channels(span_track, v) for k, v in row["spacing_GHz"].items()}

    return dict(spacing_criterion=spacing, spacing_vs_gate_time=tradeoff,
                fixed_drive=fixed, tracking_drive=tracking,
                channel_count_rule="N = floor(f01_span / spacing) + 1",
                f01_map=f01_map(EC=EC))


# ====================================================== two-qubit sketch =====
def two_qubit_sketch(C_qq_fF=(0.001, 0.01, 0.1, 0.5, 1.0),
                     detunings_GHz=(0.29, 1.0, 2.0), Omega_GHz=None,
                     EC=EC_GHZ, n01=None, lam=LAM, anharm=None):
    """PARAMETRIC SKETCH ONLY -- no simulation, no fidelity.

    Two transmons sharing a coupling capacitance C_qq.  Eliminating the
    coupler to first order in C_qq/C_Sigma gives the cross charging term
    H_int/h = 8 E_C (C_qq/C_Sigma) n_1 n_2, so in the qubit subspace

        J = 8 E_C (C_qq / C_Sigma) n01^2                    [exchange rate]

    iSWAP  (resonant, full excitation transfer)  t = 1/(2 J)
    sqrt(iSWAP)                                  t = 1/(4 J)
    CZ via the |11>-|20> resonance (matrix element lam*J)   t = 1/(2 lam J)
    cross resonance  nu_ZX = (J Om / D) * (alpha / (alpha + D)),
                     t_CNOT ~ 1/(4 nu_ZX)

    C_Sigma is LOCKED at the E_J/E_C = 100 design value, so -- exactly as P2
    found for the collection electrode -- C_qq SPENDS the C_Sigma budget
    rather than adding to it.
    """
    C_sigma = float(TARGETS["design"]["C_fF"])
    # src: qubit_targets.json:design.C_fF (13.0 fF, locked by E_J/E_C = 100)
    if n01 is None:
        n01 = float(TARGETS["design"]["n01"])
        # src: qubit_targets.json:design.n01
    if anharm is None:
        anharm = float(TARGETS["design"]["anharm_GHz"])
        # src: qubit_targets.json:design.anharm_GHz

    rows = []
    for c in C_qq_fF:
        ratio = c / C_sigma
        J = 8.0 * EC * ratio * n01 ** 2          # GHz
        rows.append(dict(
            C_qq_fF=float(c), C_qq_over_C_Sigma=float(ratio),
            J_MHz=float(J * 1e3),
            t_iSWAP_ns=float(1.0 / (2.0 * J)),
            t_sqrt_iSWAP_ns=float(1.0 / (4.0 * J)),
            t_CZ_11_20_ns=float(1.0 / (2.0 * lam * J)),
            J_over_anharm=float(J / abs(anharm)),
            dispersive_ok_vs_detuning={
                f"{d:g}": bool(J / d <= 0.1) for d in detunings_GHz}))

    cr = []
    if Omega_GHz is not None:
        for c in C_qq_fF:
            J = 8.0 * EC * (c / C_sigma) * n01 ** 2
            for d in detunings_GHz:
                nu_zx = (J * Omega_GHz / d) * (anharm / (anharm + d))
                cr.append(dict(
                    C_qq_fF=float(c), detuning_GHz=float(d),
                    J_MHz=float(J * 1e3), Omega_GHz=float(Omega_GHz),
                    nu_ZX_MHz=float(abs(nu_zx) * 1e3),
                    t_CNOT_ns=float(1.0 / (4.0 * abs(nu_zx))),
                    valid_dispersive=bool(J / d <= 0.1),
                    valid_weak_drive=bool(Omega_GHz / d <= 0.5),
                    straddling=bool(0.0 < d < abs(anharm))))

    # C_qq that keeps J/detuning <= 0.1 (the dispersive-validity budget)
    cap = {}
    for d in detunings_GHz:
        cap[f"{d:g}"] = dict(
            C_qq_max_fF=float(0.1 * d * C_sigma / (8.0 * EC * n01 ** 2)),
            J_at_cap_MHz=float(0.1 * d * 1e3),
            t_iSWAP_at_cap_ns=float(1.0 / (2.0 * 0.1 * d)))

    return dict(
        label="SKETCH -- parametric closed-form rates, not simulated",
        model=dict(
            H_int="H_int/h = 8 E_C (C_qq/C_Sigma) n_1 n_2 "
                  "(first order in C_qq/C_Sigma)",
            J="J = 8 E_C (C_qq/C_Sigma) n01^2",
            CR="nu_ZX = (J Om / D)(alpha/(alpha + D)); "
               "t_CNOT ~ 1/(4 nu_ZX)  [Magesan & Gambetta, PRA 101, 052308]",
            budget_note="C_Sigma is locked at 13.0 fF by E_J/E_C = 100, so "
                        "C_qq spends the same budget that P2 found the "
                        "collection electrodes spend"),
        params=dict(C_Sigma_fF=C_sigma, EC_GHz=float(EC), n01=float(n01),
                    ladder_ratio=float(lam), anharm_GHz=float(anharm),
                    C_qq_fF_range=[float(min(C_qq_fF)), float(max(C_qq_fF))],
                    detunings_GHz=[float(d) for d in detunings_GHz]),
        exchange=rows, cross_resonance=cr, dispersive_budget=cap,
        caveats=["no simulation: closed-form rates only",
                 "no decoherence and no fidelity is claimed",
                 "C_qq is a free design parameter, not a measured input",
                 "CR rows with valid_dispersive = false are outside the "
                 "perturbative regime the formula was derived in"])


# =============================================================== report ======
def build_report():
    """Assemble the full P6 artefact."""
    floors = gate_floor()
    conv = convergence_check()

    tg_drag_4 = floors["drag"]["t_g_floor_ns"]["1e-04"]
    tg_drag_3 = floors["drag"]["t_g_floor_ns"]["1e-03"]
    tg_gauss_4 = floors["gaussian"]["t_g_floor_ns"]["1e-04"]

    q_scan = drag_coefficient_scan(tg_drag_4)
    req = drag_requirement([tg_drag_3, tg_drag_4, tg_gauss_4], 1.0)
    req_opt = drag_requirement([tg_drag_4], q_scan["q_opt_parabolic"])
    rot = rotation_error([tg_drag_4, 2.0, 5.0])
    comb = combined_floor()

    spacing = min_spacing(tg_drag_4)
    trade = spacing_tradeoff([tg_drag_4, 2.0, 4.0, 8.0])

    with open(pathlib.Path(ROOT) / "out" / "m6" / "p3_qp.json",
              encoding="utf-8") as fh:
        p3 = json.load(fh)["pair_breaking_audit"]

    alloc = allocation(tg_drag_4, p3, spacing, trade)

    # drive amplitude available for a cross-resonance tone: the calibrated
    # peak Rabi rate at the DRAG 1e-4 floor
    omega_GHz = req[1]["Omx_peak_GHz"]
    sketch = two_qubit_sketch(Omega_GHz=omega_GHz)

    params = dict(
        EJ_GHz=float(EJ_GHZ), EC_GHz=float(EC_GHZ), EJ_over_EC=EJ_OVER_EC,
        f01_GHz=_LAD["f01_GHz"], anharm_GHz=_LAD["anharm_GHz"],
        anharm_GHz_target=float(TARGETS["design"]["anharm_GHz"]),
        f01_GHz_target=float(TARGETS["design"]["f01_GHz"]),
        n01=_LAD["n01"], n12=_LAD["n12"], ladder_ratio=_LAD["ladder_ratio"],
        ladder_ratio_over_sqrt2=_LAD["ladder_ratio_over_sqrt2"],
        delta_rad_per_ns=float(DELTA_ANG), N_charge=N_CHARGE,
        sigma_ratio=SIGMA_RATIO, steps_per_period_floor=SPP,
        n_steps_min=N_STEPS_MIN, golden_iters=GOLDEN_ITERS,
        t_g_scan_ns=list(TG_SCAN), leak_thresholds=list(LEAK_THRESHOLDS),
        nu_apex=float(NU_OPT), nu_halfwidth=float(NU_HALFWIDTH),
    )
    prov = dict(
        task="M6 P6 -- gate-set sketch (single-qubit floor, DRAG, frequency "
             "allocation, two-qubit parametric sketch)",
        module="qubit/gates.py",
        reference=("Motzoi, Gambetta, Rebentrost, Wilhelm, PRL 103, 110501 "
                   "(2009); Gambetta et al., PRA 83, 012308 (2011); "
                   "Magesan & Gambetta, PRA 101, 052308 (2020); "
                   "Koch et al., PRA 76, 042319 (2007)"),
        inputs={
            "EJ_over_EC/f01/anharm/n01/C_fF": "qubit_targets.json:design",
            "Ic_nA": "junction.json:contract.Ic_nA -> E_J (qubit/inputs.py)",
            "f01_tunability_GHz": "qubit_targets.json:gate_noise",
            "sc_dome": "material.json:sc_dome via qubit/dome.py",
            "fixed-drive window / f01 span":
                "out/m6/p3_qp.json:pair_breaking_audit.drive_cavity",
        },
        numerics=("fixed-step RK4, 3-level rotating-frame TDSE; step count "
                  "from a %g steps-per-period floor with a %g-step minimum; "
                  "amplitude calibrated by %d-iteration golden section; "
                  "vectorised -- one integrator call per scan"
                  % (SPP, N_STEPS_MIN, GOLDEN_ITERS)),
        rng="unused (no stochastic sampling in P6)", seed=0,
        claim_discipline=(
            "LEAKAGE-ONLY FLOOR. Every error here is coherent leakage or "
            "coherent crosstalk from a closed-system Schrodinger equation. No "
            "T1, T2, quasiparticle or 1/f contribution is folded in, and no "
            "gate fidelity is claimed. Section two_qubit_sketch is parametric "
            "closed form only."),
    )
    return dict(gate_floor=floors, combined_floor=comb, convergence=conv,
                drag=dict(coefficient_scan=q_scan, requirement=req,
                          requirement_at_q_opt=req_opt, rotation_error=rot,
                          definition="Om_y(t) = -q E'(t)/delta; "
                                     "first-order DRAG is q = 1"),
                allocation=alloc, two_qubit_sketch=sketch,
                params=params, provenance=prov)


def export(path=None):
    """Write out/m6/p6_gates.json and return the report dict."""
    rep = build_report()
    if path is None:
        path = pathlib.Path(ROOT) / "out" / "m6" / "p6_gates.json"
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=1)
    return rep


if __name__ == "__main__":
    r = export()
    p, gf = r["params"], r["gate_floor"]
    print("anharm = %.4f GHz   delta = %.4f rad/ns   ladder n12/n01 = %.4f "
          "(%.4f x sqrt2)" % (p["anharm_GHz"], p["delta_rad_per_ns"],
                              p["ladder_ratio"], p["ladder_ratio_over_sqrt2"]))
    print("\n-- single-qubit gate-time floor (leakage only) --")
    for thr in ("1e-03", "1e-04"):
        print("  L = %s :  gaussian %7.4f ns   drag %7.4f ns   speedup %.2fx"
              % (thr, gf["gaussian"]["t_g_floor_ns"][thr],
                 gf["drag"]["t_g_floor_ns"][thr],
                 gf["speedup_drag_over_gaussian"][thr]))
    q = r["drag"]["coefficient_scan"]
    print("\n-- DRAG --")
    print("  q_opt = %.4f (grid %.4f), leakage %.3e; q = 1 gives %.3e (%.2fx)"
          % (q["q_opt_parabolic"], q["q_opt_grid"], q["leakage_at_q_opt_grid"],
             q["leakage_at_q1"], q["gain_q1_over_qopt"]))
    for row in r["drag"]["requirement"]:
        print("  t_g %6.3f ns: Om_x %6.3f GHz  Om_y %6.4f GHz  ratio %.4f "
              "(analytic %.4f)" % (row["t_g_ns"], row["Omx_peak_GHz"],
                                   row["Omy_peak_GHz"], row["ratio_measured"],
                                   row["ratio_analytic"]))
    print("\n-- what the leakage floor does not capture --")
    for row in r["drag"]["rotation_error"]["rows"]:
        print("  t_g %6.3f ns: P0 %.3e -> %.3e (s_opt %+.3f, %.0fx); "
              "leakage %.3e -> %.3e"
              % (row["t_g_ns"], row["P0_amplitude_only"],
                 row["P0_with_frame_correction"], row["s_opt"],
                 row["P0_improvement"], row["leakage_amplitude_only"],
                 row["leakage_with_frame_correction"]))
    a = r["allocation"]
    sp = a["spacing_criterion"]
    print("\n-- allocation (spacing measured at t_g = %.4f ns) --" % sp["t_g_ns"])
    for k, v in sp["spacing_GHz"].items():
        u = sp["spacing_in_units_of_1_over_tg"][k]
        print("  crosstalk tol %s -> spacing %s GHz (%s / t_g);  gaussian-only %s"
              % (k, "n/a" if v is None else "%.4f" % v,
                 "n/a" if u is None else "%.2f" % u,
                 sp["gaussian_comparison"]["spacing_GHz"][k]))
    for name in ("fixed_drive", "tracking_drive"):
        print("  %-14s span %8.4f GHz  channels %s"
              % (name, a[name]["f01_span_GHz"], a[name]["channels"]))
    print("  sweet-spot cost at window edge: T_phi %.4g us -> %.4g us (%.4g x)"
          % (a["fixed_drive"]["sweet_spot_cost"]["Tphi_apex_us"],
             a["fixed_drive"]["sweet_spot_cost"]["Tphi_edge_us"],
             a["fixed_drive"]["sweet_spot_cost"]["contrast"]))
    print("  spacing vs gate time:")
    for row in a["spacing_vs_gate_time"]:
        print("    t_g %6.3f ns -> %s GHz | channels fixed %s  tracking %s"
              % (row["t_g_ns"], row["spacing_GHz"],
                 row["channels_fixed_drive"], row["channels_tracking_drive"]))
    print("\n-- combined coherent floor (amplitude + DRAG frame calibrated) --")
    print("  max(P0,P2) reaches %s" % r["combined_floor"]["t_g_floor_ns"])
    print("\n-- two-qubit SKETCH (parametric, not simulated) --")
    for row in r["two_qubit_sketch"]["exchange"]:
        print("  C_qq %6.3f fF: J %9.3f MHz  iSWAP %8.3f ns  CZ %8.3f ns  "
              "J/|anharm| %.3f"
              % (row["C_qq_fF"], row["J_MHz"], row["t_iSWAP_ns"],
                 row["t_CZ_11_20_ns"], row["J_over_anharm"]))
    print("  dispersive budget (J/detuning <= 0.1):")
    for k, v in r["two_qubit_sketch"]["dispersive_budget"].items():
        print("    detuning %5s GHz -> C_qq <= %8.5f fF (J %6.1f MHz, "
              "iSWAP %7.3f ns)"
              % (k, v["C_qq_max_fF"], v["J_at_cap_MHz"], v["t_iSWAP_at_cap_ns"]))
    print("  cross-resonance (Om = %.3f GHz):"
          % r["two_qubit_sketch"]["cross_resonance"][0]["Omega_GHz"])
    for row in r["two_qubit_sketch"]["cross_resonance"]:
        if row["valid_dispersive"]:
            print("    C_qq %6.3f fF  D %4.2f GHz: nu_ZX %7.3f MHz  "
                  "t_CNOT %8.2f ns  straddling=%s"
                  % (row["C_qq_fF"], row["detuning_GHz"], row["nu_ZX_MHz"],
                     row["t_CNOT_ns"], row["straddling"]))
