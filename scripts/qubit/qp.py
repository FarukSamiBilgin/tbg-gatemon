"""Quasiparticle relaxation and thermal occupation (AT-6).

Small-frequency (hbar omega_q << Delta) quasiparticle T1 for a
Josephson-junction qubit:

    Gamma_qp = x_qp * (omega_q / pi) * sqrt(2 Delta / (hbar omega_q))
    T1 = 1 / Gamma_qp

MARGINALITY FLAG (M6_BRIEF.md Sec. 2, deliverable P3): at the design point
hbar*omega_q/Delta = 0.55, so the small-frequency expansion is stretched.
The full Catelani-type spectral kernel is production task P3; this milestone
reproduces the reference prefactor and *reports* the marginality ratio.

x_qp (normalised quasiparticle density) is a free parameter -- paper 2 is a
DESIGN STUDY and makes no absolute T1 claim.
"""
import numpy as np

from .inputs import DELTA0_MEV, E, H, HBAR, KB

X_QP_GRID = (1e-8, 1e-7, 1e-6, 1e-5)   # parametric sweep, a design choice


def _delta_J(Delta_meV=DELTA0_MEV):
    """Gap in joules.  src: material.json:Delta0_meV."""
    return Delta_meV * 1e-3 * E


def qp_prefactor(f01_GHz, Delta_meV=DELTA0_MEV):
    """Gamma_qp / x_qp in s^-1: (omega_q/pi) sqrt(2 Delta / (hbar omega_q))."""
    wq = 2 * np.pi * f01_GHz * 1e9
    return (wq / np.pi) * np.sqrt(2 * _delta_J(Delta_meV) / (HBAR * wq))


def T1_qp(x_qp, f01_GHz, Delta_meV=DELTA0_MEV):
    """Quasiparticle-limited T1 in seconds at normalised density x_qp."""
    return 1.0 / (x_qp * qp_prefactor(f01_GHz, Delta_meV))


def marginality(f01_GHz, Delta_meV=DELTA0_MEV):
    """hbar*omega_q/Delta -- the small-frequency-expansion validity ratio."""
    return HBAR * 2 * np.pi * f01_GHz * 1e9 / _delta_J(Delta_meV)


def qp_budget(f01_GHz, x_grid=X_QP_GRID, Delta_meV=DELTA0_MEV):
    return dict(T1_us={f"{x:.0e}": T1_qp(x, f01_GHz, Delta_meV) * 1e6
                       for x in x_grid},
                marginality_flag=marginality(f01_GHz, Delta_meV))


# --------------------------------------------------------------- thermal -----
def n_thermal(f_GHz, T_K):
    """Boltzmann occupation exp(-h f / kB T) of the qubit excited state."""
    return float(np.exp(-H * f_GHz * 1e9 / (KB * T_K)))
