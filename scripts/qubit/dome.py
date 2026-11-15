"""Gate-tunable spectrum on the superconducting dome (AT-3).

The TBG-specific noise channel: the filling nu is set by a gate, the dome
sets Tc(nu) -> Delta(nu) -> Ic(nu) -> E_J(nu), so gate-voltage noise is
frequency noise on the qubit.  The dome apex nu = -2.4 is a DOUBLE sweet
spot: E_J is maximal there AND df01/dnu vanishes by symmetry.

All dome parameters are read from material.json:sc_dome; the BCS shape
constants come from the Delta_form string carried in that same block
(see qubit/inputs.py for the trace policy).
"""
import numpy as np

from . import spectrum
from .inputs import (BCS_GAP_RATIO, BCS_TANH_COEF, DELTA0_MEV, EJ_GHZ,
                     KB_MEV_PER_K, NU_HALFWIDTH, NU_OPT, N_CHARGE, TC_MAX_K,
                     T_OP_K)

DNU = 1e-3   # finite-difference step for D1/D2 (matches m6_reference.py)


def Tc_of(nu):
    """Dome Tc(nu) in K.  src: material.json:sc_dome (parabolic dome)."""
    return TC_MAX_K * max(0.0, 1.0 - ((nu - NU_OPT) / NU_HALFWIDTH) ** 2)


def Delta_of(nu, T=T_OP_K):
    """Gap Delta(nu, T) in meV.

    src: material.json:sc_dome.Delta_form
        Delta = 1.764 * kB * Tc * tanh(1.74 * sqrt(Tc/T - 1))
    Returns 0 where the dome Tc has fallen below the operating temperature.
    """
    Tc = Tc_of(nu)
    if Tc <= T:
        return 0.0
    return (BCS_GAP_RATIO * KB_MEV_PER_K * Tc
            * np.tanh(BCS_TANH_COEF * np.sqrt(Tc / T - 1.0)))


def EJ_of(nu, EJ_apex=EJ_GHZ):
    """E_J(nu)/h in GHz, scaled linearly with the gap (Ic ~ Delta)."""
    return EJ_apex * Delta_of(nu) / DELTA0_MEV


def f01_of(nu, EC, EJ_apex=EJ_GHZ, N=N_CHARGE):
    """Qubit frequency f01(nu) in GHz; NaN outside the dome."""
    EJ = EJ_of(nu, EJ_apex)
    if EJ <= 0:
        return np.nan
    return spectrum.levels(EJ, EC, N)[0]


def D1(nu, EC, dnu=DNU, **kw):
    """df01/dnu in GHz per unit filling (central difference)."""
    return (f01_of(nu + dnu, EC, **kw) - f01_of(nu - dnu, EC, **kw)) / (2 * dnu)


def D2(nu, EC, dnu=DNU, **kw):
    """d^2 f01/dnu^2 in GHz per unit filling^2 (central difference)."""
    return (f01_of(nu + dnu, EC, **kw) - 2 * f01_of(nu, EC, **kw)
            + f01_of(nu - dnu, EC, **kw)) / dnu ** 2


# --------------------------------------------------- quasistatic dephasing ---
def Tphi_second_order(D2_val, dnu_rms):
    """Quasistatic T_phi at a first-order sweet spot: 1/(pi |D2| dnu_rms^2)."""
    return 1.0 / (np.pi * abs(D2_val) * 1e9 * dnu_rms ** 2)


def Tphi_first_order(D1_val, dnu_rms):
    """Quasistatic T_phi off the sweet spot: 1/(2 pi |D1| dnu_rms)."""
    return 1.0 / (2 * np.pi * abs(D1_val) * 1e9 * dnu_rms)


def sweet_spot_report(EC, nu_apex=NU_OPT, nu_off=-2.5, dnu_rms=(1e-4, 1e-3),
                      **kw):
    """D1/D2 at the apex, D1 off-apex, and the quasistatic T_phi contrast."""
    d1a, d2a = D1(nu_apex, EC, **kw), D2(nu_apex, EC, **kw)
    d1o = D1(nu_off, EC, **kw)
    return dict(
        D1_apex_GHz=d1a, D2_apex_GHz=d2a, D1_off_GHz=d1o,
        Tphi_apex_us={f"{d:.0e}": Tphi_second_order(d2a, d) * 1e6
                      for d in dnu_rms},
        Tphi_off_us={f"{d:.0e}": Tphi_first_order(d1o, d) * 1e6
                     for d in dnu_rms},
    )
