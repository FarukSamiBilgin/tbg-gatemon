"""M6 exploration 4/5 -- the cavities the qubits already touch are two-qubit buses.

QUESTION
--------
qubit.json:two_qubit_sketch demands a DIRECT island-island capacitance
C_qq <= 0.0186 fF (18.6 aF) at a parking detuning D = 0.29 GHz.  That is the
least manufacturable number in the whole paper-2 design.  But the chip already
contains two shared electromagnetic modes that couple any two islands sitting
on them:

  * the READOUT cavity at f_r = 44.597 GHz (recommended point, Delta_r = 4 GHz,
    g = 142.13 MHz, kappa = 3.45 MHz) -> J_ro = g1 g2 (1/D1 + 1/D2)/2,
  * the measured 145 GHz DRIVE cavity (Delta_d = 104.40 GHz,
    g_d = 181.77 MHz at beta_drive = 0.17273) -> J_dr = g_d^2/Delta_d.

Is either cavity bus a viable replacement for the direct capacitor, given that
the ONLY on/off knob is the 0.2906 GHz f01 span of the fixed-drive filling
window, and that f01(nu) is EVEN about the dome apex (so two symmetric
parkings are exactly degenerate)?

WHAT IS COMPUTED (all deterministic, no RNG anywhere)
-----------------------------------------------------
1.  A 41 x 41 (nu1, nu2) grid over the fixed-drive window
    [-2.458571, -2.341429]: EJ(nu) (dome), f01(nu) and n01(nu) (charge-basis
    exact diagonalisation at the LOCKED E_C = 1.490051 GHz), D1/D2(nu), and
    T2_echo / T2_ramsey(nu) from the P4 filter-function integrals.
2.  Readout bus: J_ro(nu1,nu2), residual swap P = J^2/(J^2+(d12/2)^2), static
    ZZ from a 4th-order dispersive formula AND from exact diagonalisation of
    the two-transmon + cavity Hamiltonian, a bring-to-resonance iSWAP error
    budget, and the Pareto front of (isolation vs parked-qubit coherence).
3.  Drive bus: the same quantities at the beta_drive bracket
    [0.15385 (shared island budget), 0.17273 (unconstrained)].
4.  N_ops = T1_qp(x_qp, T_qp = 0.1 K) / t_iSWAP over x_qp = 1e-8..1e-5, for
    both buses and for every stored direct-C_qq row.
5.  A QUANTIFIED version of the measurement-conflict statement: how far apart
    the two joint dispersive shifts can be pushed inside the window, against
    the cavity linewidth.

FREE PARAMETERS (named, with ranges -- design-study discipline, no absolute
TBG coherence claim anywhere in this file)
    A_1F        1/f filling-noise amplitude, nominal 1e-4, range 1e-6..1e-2
    X_QP_GRID   quasiparticle fraction, 1e-8..1e-5 (non-equilibrium band)
    T_QP_K      effective qp temperature, 0.1 K (the operating point)
    BETA_DRIVE  voltage divider of the drive electrode, 0.15385..0.17273
    DNU_SET     rms gate-setting error on the filling, 1e-5..1e-3
    NLEV / NCAV Hilbert-space truncation, 3/5 with a 4/8 convergence check

Writes: out/m6/explore/cavity_bus_two_qubit.json
Owns nothing else.  qubit/, tests/, paper/, figs/ untouched.
"""
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qubit import coupling, dome, inputs, noise_1f, qp_kernel, spectrum  # noqa: E402

OUT = ROOT / "out" / "m6" / "explore" / "cavity_bus_two_qubit.json"

# =============================================================== inputs ======
QJ = json.loads((ROOT / "qubit.json").read_text(encoding="utf-8"))

DESIGN = QJ["design"]
EC_GHZ = DESIGN["EC_GHz"]              # src: qubit.json:design.EC_GHz (LOCKED)
C_SIGMA_FF = DESIGN["C_Sigma_fF"]      # src: qubit.json:design.C_Sigma_fF
ALPHA_GHZ = DESIGN["anharm_GHz"]       # src: qubit.json:design.anharm_GHz
N01_DESIGN = DESIGN["n01"]             # src: qubit.json:design.n01
F01_DESIGN = DESIGN["f01_GHz"]         # src: qubit.json:design.f01_GHz

RP = QJ["readout_window"]["recommended_point"]
F_R_GHZ = RP["f_r_GHz"]                # src: qubit.json:readout_window.recommended_point.f_r_GHz
DELTA_R_GHZ = RP["Delta_r_GHz"]        # src: ...recommended_point.Delta_r_GHz
D_EFF_R_M = RP["d_eff_um"] * 1e-6      # src: ...recommended_point.d_eff_um
BETA_R = RP["beta_readout"]            # src: ...recommended_point.beta_readout
KAPPA_R_MHZ = RP["kappa_MHz"]          # src: ...recommended_point.kappa_MHz
KAPPA_R_INT_MHZ = RP["kappa_intrinsic_MHz"]   # src: ...kappa_intrinsic_MHz (C2 floor)
G_R_STORED_MHZ = RP["g_MHz"]           # src: ...recommended_point.g_MHz
CHI_R_STORED_MHZ = RP["chi_MHz"]       # src: ...recommended_point.chi_MHz
T1P_R_STORED_US = RP["T1_Purcell_us"]  # src: ...recommended_point.T1_Purcell_us
C_C_READOUT_FF = RP["C_c_readout_fF"]  # src: ...recommended_point.C_c_readout_fF
C_C_DRIVE_FF = RP["C_c_drive_fF"]      # src: ...recommended_point.C_c_drive_fF
ETA_READOUT = RP["eta_readout"]        # src: ...recommended_point.eta_readout

PD = QJ["budgets"]["T1_stack"]["purcell_drive"]
DELTA_D_GHZ = PD["Delta_GHz"]          # src: qubit.json:budgets.T1_stack.purcell_drive.Delta_GHz
KAPPA_D_MHZ = PD["kappa_MHz"]          # src: ...purcell_drive.kappa_MHz
G_D_STORED_MHZ = PD["corrected_g_MHz"]         # src: ...corrected_g_MHz
T1P_D_STORED_US = PD["corrected_T1_us"]        # src: ...corrected_T1_us

BETA_DRIVE = QJ["conventions"]["beta"]["drive"]                       # 0.172729
BETA_DRIVE_SHARED = QJ["conventions"]["shared_island_budget"]["beta_drive"]   # 0.153846
C_C_CAP_FF = QJ["conventions"]["beta"]["C_c_budget_cap_fF"]           # 11.99971 fF
C_J_FF = QJ["conventions"]["beta"]["budget_law"]                      # string, for provenance

FD = QJ["allocation"]["fixed_drive"]
NU_WINDOW = FD["nu_window"]            # src: qubit.json:allocation.fixed_drive.nu_window
F01_SPAN_GHZ = FD["f01_span_GHz"]      # src: ...fixed_drive.f01_span_GHz = 0.2906162
D1_EDGE_STORED = FD["sweet_spot_cost"]["D1_edge_GHz_per_nu"]          # 9.958083
MIN_SPACING_8NS_GHZ = FD["minimum_spacing_at_t_g_8ns_GHz"]            # 0.9234687

TQS = QJ["two_qubit_sketch"]
LADDER_RATIO = TQS["params"]["ladder_ratio"]           # src: n12/n01 = 1.3852260
CQQ_ROWS = TQS["exchange"]                             # stored direct-C_qq rows
CQQ_BUDGET = TQS["dispersive_budget"]                  # stored dispersive caps

GATE_FLOOR_NS = 1.6044                 # src: paper/decisions.md 2026-07-27 M6 P6
# ("THE PAPER MUST QUOTE 1.6 ns"); also qubit.json two_qubit/gate discussion.

# --------------------------------------------------------- free parameters --
A_1F = 1e-4                    # 1/f filling-noise amplitude (FREE, 1e-6..1e-2)
A_1F_RANGE = (1e-6, 1e-2)
A_1F_SENS = (1e-5, 1e-4, 1e-3)
X_QP_GRID = (1e-8, 1e-7, 1e-6, 1e-5)   # FREE, non-equilibrium band
T_QP_K = 0.1                   # operating temperature (material.json / inputs.T_OP_K)
DNU_SET = (1e-5, 1e-4, 1e-3)   # FREE: rms gate-setting error on the filling
NGRID = 41                     # nu points per axis (41 x 41 pairs)
NLEV, NCAV = 3, 5              # truncation; 4/8 convergence check below

J_OVER_D_DISPERSIVE = 0.1      # the two_qubit_sketch dispersive criterion

# ============================================== 1-D tables over the window ===
nu_edge_lo, nu_edge_hi = NU_WINDOW
nu_grid = np.linspace(nu_edge_lo, nu_edge_hi, NGRID)
# f01(nu) is even about the apex, so |nu + 2.4| is the only physical coordinate.
d_grid = np.abs(nu_grid - inputs.NU_OPT)

EJ_nu = np.array([dome.EJ_of(v) for v in nu_grid])
f01_nu = np.array([dome.f01_of(v, EC_GHZ) for v in nu_grid])
n01_nu = np.array([spectrum.n01_element(EJ_nu[i], EC_GHZ) for i in range(NGRID)])
D1_nu = np.array([dome.D1(v, EC_GHZ) for v in nu_grid])
D2_nu = np.array([dome.D2(v, EC_GHZ) for v in nu_grid])
alpha_nu = np.array([spectrum.levels(EJ_nu[i], EC_GHZ)[1] for i in range(NGRID)])

# T2 from the P4 filter-function integrals (echo and Ramsey), A free.
T2_echo_nu = np.array([noise_1f.T2(D1_nu[i], D2_nu[i], A_1F, "echo")
                       for i in range(NGRID)])
T2_ram_nu = np.array([noise_1f.T2(D1_nu[i], D2_nu[i], A_1F, "ramsey")
                      for i in range(NGRID)])
T2_echo_A = {f"{a:.0e}": np.array([noise_1f.T2(D1_nu[i], D2_nu[i], a, "echo")
                                   for i in range(NGRID)])
             for a in A_1F_SENS}

i_apex = int(np.argmin(d_grid))
i_edge_hi = NGRID - 1

# ================================================= couplings across the grid =
V_MODE_R_M3 = inputs.V_MODE_M3 * (inputs.F_CAV_GHZ / F_R_GHZ) ** 3
# src: qubit/cavity.py:readout_window_scan -- V_r = V_mode (f_cav/f_r)^3.

g_r_nu = np.array([coupling.coupling_g(F_R_GHZ, V_MODE_R_M3, D_EFF_R_M,
                                       n01_nu[i], BETA_R) for i in range(NGRID)])   # GHz
g_d_nu = np.array([coupling.coupling_g(inputs.F_CAV_GHZ, inputs.V_MODE_M3,
                                       inputs.D_EFF_DRIVE_M, n01_nu[i], BETA_DRIVE)
                   for i in range(NGRID)])                                          # GHz
g_d_shared_nu = g_d_nu * (BETA_DRIVE_SHARED / BETA_DRIVE)

Delta_r_nu = F_R_GHZ - f01_nu               # readout-cavity detuning, GHz
Delta_d_nu = inputs.F_CAV_GHZ - f01_nu      # drive-cavity detuning, GHz


def bus_J(g_a, g_b, Da, Db):
    """Cavity-mediated exchange J = g_a g_b (1/Da + 1/Db)/2, same units as g."""
    return 0.5 * g_a * g_b * (1.0 / Da + 1.0 / Db)


# 41 x 41 meshgrid -> flat arrays -> vectorised evaluation (one pass).
I1, I2 = np.meshgrid(np.arange(NGRID), np.arange(NGRID), indexing="ij")
i1, i2 = I1.ravel(), I2.ravel()

J_ro = bus_J(g_r_nu[i1], g_r_nu[i2], Delta_r_nu[i1], Delta_r_nu[i2])          # GHz
J_dr = bus_J(g_d_nu[i1], g_d_nu[i2], Delta_d_nu[i1], Delta_d_nu[i2])          # GHz
J_dr_sh = bus_J(g_d_shared_nu[i1], g_d_shared_nu[i2],
                Delta_d_nu[i1], Delta_d_nu[i2])                               # GHz
d12 = f01_nu[i1] - f01_nu[i2]                                                 # GHz


def residual_swap(J, delta):
    """Max population transferred by an always-on exchange at detuning delta."""
    return J ** 2 / (J ** 2 + (delta / 2.0) ** 2)


def zz_perturbative(J, delta, a1, a2, r=LADDER_RATIO):
    """4th-order static ZZ, zeta = E11 - E10 - E01 + E00, in the same units as J.

    Derived for the |11> <-> {|20>, |02>} repulsion with the MEASURED ladder
    element r = |<1|n|2>|/|<0|n|1>| (= 1.38523) rather than the harmonic sqrt2:

        zeta = (r J)^2 [ 1/(delta - a2) - 1/(delta + a1) ]

    which reduces to the textbook 2 J^2 (a1 + a2)/((delta - a2)(delta + a1))
    at r = sqrt(2), i.e. |zeta| -> 4 J^2/|alpha| at delta = 0.
    """
    return (r * J) ** 2 * (1.0 / (delta - a2) - 1.0 / (delta + a1))


P_ro = residual_swap(J_ro, d12)
P_dr = residual_swap(J_dr, d12)
P_dr_sh = residual_swap(J_dr_sh, d12)
ZZ_ro = zz_perturbative(J_ro, d12, alpha_nu[i1], alpha_nu[i2])
ZZ_dr = zz_perturbative(J_dr, d12, alpha_nu[i1], alpha_nu[i2])

t_iswap_ro = 1.0 / (2.0 * J_ro * 1e9)       # s ; convention from two_qubit_sketch
t_iswap_dr = 1.0 / (2.0 * J_dr * 1e9)
t_iswap_dr_sh = 1.0 / (2.0 * J_dr_sh * 1e9)

# ============================================= exact diagonalisation check ===
def n_matrix(EJ, EC, nlev, N=inputs.N_CHARGE):
    """<m| n_hat |m'> in the lowest `nlev` transmon eigenstates, at n_g = 0.

    Built from qubit/spectrum.py's charge-basis Hamiltonian, so the ladder is
    the measured one (n12/n01 = 1.38523, not the harmonic sqrt2).
    """
    n = np.arange(-N, N + 1)
    _, v = np.linalg.eigh(spectrum.hamiltonian(EJ, EC, 0.0, N))
    v = v[:, :nlev]
    return v.T @ (n[:, None] * v)


def transmon_levels(EJ, EC, nlev, N=inputs.N_CHARGE):
    """Lowest `nlev` transmon eigenvalues referred to the ground state (GHz)."""
    w = np.linalg.eigvalsh(spectrum.hamiltonian(EJ, EC, 0.0, N))[:nlev]
    return w - w[0]


def two_qubit_cavity_spectrum(E1, E2, b1, b2, g1, g2, f_c, ncav, rwa=True):
    """Eigen-decomposition of two transmons + one cavity mode.

    H/h = sum_i E_i(n) + f_c a^dag a + sum_i g_i (b_i^dag a + b_i a^dag)
    with b_i the normalised charge lowering operator (b[0,1] = 1).
    """
    n1, n2 = len(E1), len(E2)
    a = np.diag(np.sqrt(np.arange(1, ncav)), 1)
    I1_, I2_, Ic = np.eye(n1), np.eye(n2), np.eye(ncav)

    def kron3(x, y, z):
        return np.kron(np.kron(x, y), z)

    H = (kron3(np.diag(E1), I2_, Ic) + kron3(I1_, np.diag(E2), Ic)
         + kron3(I1_, I2_, f_c * (a.T @ a)))
    for b, g, which in ((b1, g1, 0), (b2, g2, 1)):
        bd = b.T
        if rwa:
            op_q_up, op_c_dn = bd, a
            op_q_dn, op_c_up = b, a.T
            if which == 0:
                H += g * (kron3(op_q_up, I2_, op_c_dn)
                          + kron3(op_q_dn, I2_, op_c_up))
            else:
                H += g * (kron3(I1_, op_q_up, op_c_dn)
                          + kron3(I1_, op_q_dn, op_c_up))
        else:
            xq, xc = b + bd, a + a.T
            if which == 0:
                H += g * kron3(xq, I2_, xc)
            else:
                H += g * kron3(I1_, xq, xc)
    w, v = np.linalg.eigh(H)
    return w, v, (n1, n2, ncav)


def _idx(n1, n2, m, dims):
    _, N2, NC = dims
    return (n1 * N2 + n2) * NC + m


def zz_direct_exchange(f1, f2, alpha1, alpha2, EJ1, EJ2, J, nlev=4):
    """Static ZZ for a DIRECT exchange coupling J (no cavity in the model).

    This is the clean test of `zz_perturbative`: the 4th-order formula should
    reproduce it to <1% because J/|alpha| ~ 3e-3.
    """
    E1 = transmon_levels(EJ1, EC_GHZ, nlev)
    E2 = transmon_levels(EJ2, EC_GHZ, nlev)
    b1 = n_matrix(EJ1, EC_GHZ, nlev)
    b1 = np.triu(b1) / b1[0, 1]
    b2 = n_matrix(EJ2, EC_GHZ, nlev)
    b2 = np.triu(b2) / b2[0, 1]
    I = np.eye(nlev)
    H = np.kron(np.diag(E1), I) + np.kron(I, np.diag(E2))
    H += J * (np.kron(b1.T, b2) + np.kron(b1, b2.T))
    w, v = np.linalg.eigh(H)
    p2 = np.abs(v) ** 2
    k00 = int(np.argmax(p2[0 * nlev + 0, :]))
    k11 = int(np.argmax(p2[1 * nlev + 1, :]))
    wt = p2[1 * nlev + 0, :] + p2[0 * nlev + 1, :]
    ks = np.argsort(wt)[-2:]
    return float(w[k11] - w[ks].sum() + w[k00])


def zz_exact(f1, f2, alpha1, alpha2, EJ1, EJ2, g1, g2, f_c, nlev, ncav,
             rwa=True):
    """Exact static ZZ (GHz) from the two-transmon + cavity diagonalisation.

    The two singly-excited qubit levels are identified as the pair with the
    largest weight on span{|100>, |010>} and their energies are SUMMED, which
    stays well defined through the exact degeneracy at delta12 = 0.
    """
    E1 = np.array([0.0, f1, 2 * f1 + alpha1][:nlev]
                  + list(transmon_levels(EJ1, EC_GHZ, nlev)[3:]))
    E2 = np.array([0.0, f2, 2 * f2 + alpha2][:nlev]
                  + list(transmon_levels(EJ2, EC_GHZ, nlev)[3:]))
    b1 = n_matrix(EJ1, EC_GHZ, nlev)
    b2 = n_matrix(EJ2, EC_GHZ, nlev)
    b1 = np.triu(b1) / b1[0, 1]
    b2 = np.triu(b2) / b2[0, 1]
    w, v, dims = two_qubit_cavity_spectrum(E1, E2, b1, b2, g1, g2, f_c, ncav,
                                           rwa)
    p2 = np.abs(v) ** 2
    k00 = int(np.argmax(p2[_idx(0, 0, 0, dims), :]))
    k11 = int(np.argmax(p2[_idx(1, 1, 0, dims), :]))
    wt = p2[_idx(1, 0, 0, dims), :] + p2[_idx(0, 1, 0, dims), :]
    ks = np.argsort(wt)[-2:]
    return float(w[k11] - w[ks].sum() + w[k00])


# ZZ sweep: qubit 1 pinned at the apex, qubit 2 walked out to the window edge.
zz_rows = []
for j in range(NGRID):
    f1, f2 = f01_nu[i_apex], f01_nu[j]
    a1, a2 = alpha_nu[i_apex], alpha_nu[j]
    g1, g2 = g_r_nu[i_apex], g_r_nu[j]
    Jp = bus_J(g1, g2, Delta_r_nu[i_apex], Delta_r_nu[j])
    dz = f1 - f2
    zz_rows.append(dict(
        nu2=float(nu_grid[j]), delta12_GHz=float(dz), J_pert_GHz=float(Jp),
        zz_pert_GHz=float(zz_perturbative(Jp, dz, a1, a2)),
        zz_pert_harmonic_GHz=float(zz_perturbative(Jp, dz, a1, a2, r=np.sqrt(2))),
        zz_exact_GHz=zz_exact(f1, f2, a1, a2, EJ_nu[i_apex], EJ_nu[j],
                              g1, g2, F_R_GHZ, NLEV, NCAV),
    ))

# Truncation convergence and RWA check at the largest split.
jz = NGRID - 1
zz_conv = dict(
    nlev3_ncav5=zz_rows[jz]["zz_exact_GHz"],
    nlev4_ncav8=zz_exact(f01_nu[i_apex], f01_nu[jz], alpha_nu[i_apex],
                         alpha_nu[jz], EJ_nu[i_apex], EJ_nu[jz],
                         g_r_nu[i_apex], g_r_nu[jz], F_R_GHZ, 4, 8),
    nlev4_ncav8_noRWA=zz_exact(f01_nu[i_apex], f01_nu[jz], alpha_nu[i_apex],
                               alpha_nu[jz], EJ_nu[i_apex], EJ_nu[jz],
                               g_r_nu[i_apex], g_r_nu[jz], F_R_GHZ, 4, 8,
                               rwa=False),
)

# Exact J from the degenerate single-excitation splitting (validation).
def J_exact_degenerate(idx, nlev=NLEV, ncav=NCAV):
    f, al, EJ, g = f01_nu[idx], alpha_nu[idx], EJ_nu[idx], g_r_nu[idx]
    E = np.array([0.0, f, 2 * f + al])
    b = n_matrix(EJ, EC_GHZ, nlev)
    b = np.triu(b) / b[0, 1]
    w, v, dims = two_qubit_cavity_spectrum(E, E, b, b, g, g, F_R_GHZ, ncav)
    p2 = np.abs(v) ** 2
    wt = p2[_idx(1, 0, 0, dims), :] + p2[_idx(0, 1, 0, dims), :]
    ks = np.sort(np.argsort(wt)[-2:])
    return 0.5 * float(w[ks[1]] - w[ks[0]])


J_exact_apex = J_exact_degenerate(i_apex)
J_pert_apex = bus_J(g_r_nu[i_apex], g_r_nu[i_apex],
                    Delta_r_nu[i_apex], Delta_r_nu[i_apex])

# --- WHY the cavity ZZ is smaller than the pure-exchange ZZ (Delta_r sweep) ---
# At FIXED effective J the cavity-mediated cross-Kerr partially cancels the
# exchange-mediated ZZ; the deficit dies as 1/Delta_r.  Measured on this grid.
J_fix = float(bus_J(g_r_nu[i_apex], g_r_nu[i_edge_hi],
                    Delta_r_nu[i_apex], Delta_r_nu[i_edge_hi]))
d12_max = float(f01_nu[i_apex] - f01_nu[i_edge_hi])
zz_pure = zz_perturbative(J_fix, d12_max, alpha_nu[i_apex], alpha_nu[i_edge_hi])
zz_vs_Delta = []
for Dr in (4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 1024.0):
    Da, Db = Dr, Dr + d12_max
    g_eq = float(np.sqrt(J_fix / (0.5 * (1.0 / Da + 1.0 / Db))))
    ze = zz_exact(f01_nu[i_apex], f01_nu[i_edge_hi], alpha_nu[i_apex],
                  alpha_nu[i_edge_hi], EJ_nu[i_apex], EJ_nu[i_edge_hi],
                  g_eq, g_eq, f01_nu[i_apex] + Dr, 4, 8)
    law = 1.0 - abs(ALPHA_GHZ) / Dr
    zz_vs_Delta.append(dict(Delta_r_GHz=Dr, g_required_MHz=g_eq * 1e3,
                            zz_exact_kHz=ze * 1e6, ratio=float(ze / zz_pure),
                            law_1_minus_alpha_over_Delta=law,
                            law_reldev=float(abs(ze / zz_pure - law) / law)))
zz_direct_check = zz_direct_exchange(f01_nu[i_apex], f01_nu[i_edge_hi],
                                     alpha_nu[i_apex], alpha_nu[i_edge_hi],
                                     EJ_nu[i_apex], EJ_nu[i_edge_hi], J_fix)


def zz_cavity_corrected(J, delta, a1, a2, Delta_bus):
    """Pure-exchange ZZ times the measured cavity-cancellation factor."""
    return zz_perturbative(J, delta, a1, a2) * (1.0 - abs(ALPHA_GHZ) / Delta_bus)


ZZ_ro_corr = zz_cavity_corrected(J_ro, d12, alpha_nu[i1], alpha_nu[i2],
                                 DELTA_R_GHZ)
ZZ_dr_corr = zz_cavity_corrected(J_dr, d12, alpha_nu[i1], alpha_nu[i2],
                                 DELTA_D_GHZ)

# ==================================================== error / depth budgets ==
T1_qp = {f"{x:.0e}": qp_kernel.T1_full(x, T_QP_K) for x in X_QP_GRID}
GAMMA_P_R = 1.0 / (T1P_R_STORED_US * 1e-6)      # readout-cavity Purcell rate
GAMMA_P_D = 1.0 / (T1P_D_STORED_US * 1e-6)      # drive-cavity Purcell rate
# BOTH cavities are always attached to every island, whichever one is used as
# the bus, so the Purcell background is the SUM.
GAMMA_P_TOT = GAMMA_P_R + GAMMA_P_D


def gate_budget(t_g, T2_1, T2_2, gamma_purcell, x_qp):
    """Leading-order INCOHERENT error contributions for one bus gate.

    NOT a fidelity: no simulation, no coherent-error term, no claim.
        relaxation  : t_g (2 Gamma_purcell + 2/T1_qp)   [both qubits]
        dephasing   : (t_g/T2_i)^2 summed              [1/f, chi(T2) = 1]
    """
    g_qp = 1.0 / T1_qp[f"{x_qp:.0e}"]
    e_p = 2.0 * gamma_purcell * t_g
    e_qp = 2.0 * g_qp * t_g
    e_phi = (t_g / T2_1) ** 2 + (t_g / T2_2) ** 2
    return dict(purcell=e_p, qp=e_qp, dephasing=e_phi,
                total=e_p + e_qp + e_phi)


def cqq_equiv_fF(J_GHz):
    """Direct capacitance that would give the same J (two_qubit_sketch model)."""
    return J_GHz * C_SIGMA_FF / (8.0 * EC_GHZ * N01_DESIGN ** 2)


# ------------------------------------------------------------- ON / OFF -----
# ON  : both islands at the apex -> f01 exactly degenerate (evenness is EXACT).
# OFF : island 1 stays at the apex, island 2 is walked to the window edge.
k_on = i_apex * NGRID + i_apex
k_off = i_apex * NGRID + i_edge_hi

on_off = {}
for tag, Jarr, Parr, gamP, t_arr in (
        ("readout_bus", J_ro, P_ro, GAMMA_P_TOT, t_iswap_ro),
        ("drive_bus", J_dr, P_dr, GAMMA_P_TOT, t_iswap_dr)):
    J_on, J_off = float(Jarr[k_on]), float(Jarr[k_off])
    t_g = float(t_arr[k_on])
    on_off[tag] = dict(
        J_on_MHz=J_on * 1e3, J_off_MHz=J_off * 1e3,
        t_iSWAP_ns=t_g * 1e9,
        t_sqrt_iSWAP_ns=t_g * 0.5e9,
        P_on=float(Parr[k_on]), P_off=float(Parr[k_off]),
        on_off_ratio=float(Parr[k_on] / Parr[k_off]),
        delta12_off_GHz=float(abs(d12[k_off])),
        T2_echo_parked_off_us=float(T2_echo_nu[i_edge_hi] * 1e6),
        T2_echo_apex_us=float(T2_echo_nu[i_apex] * 1e6),
        off_idle_in_gate_times=float(T2_echo_nu[i_edge_hi] / t_g),
        budget=gate_budget(t_g, T2_echo_nu[i_apex], T2_echo_nu[i_apex],
                           gamP, 1e-7),
        C_qq_equivalent_fF=cqq_equiv_fF(J_on),
        C_qq_equivalent_aF=cqq_equiv_fF(J_on) * 1e3,
    )

# Structural closed form: t_g/T1_Purcell = pi kappa / Delta, g cancels exactly.
purcell_floor = {
    "identity": ("t_iSWAP/T1_Purcell = [1/(2J)] (g/D)^2 2 pi kappa "
                 "= pi kappa / D  with J = g^2/D -- g cancels"),
    "readout_bus": {
        "pi_kappa_over_Delta": float(np.pi * KAPPA_R_MHZ * 1e-3 / DELTA_R_GHZ),
        "direct_t_over_T1P": float(on_off["readout_bus"]["t_iSWAP_ns"] * 1e-9
                                   * GAMMA_P_R),
        "kappa_MHz": KAPPA_R_MHZ, "Delta_GHz": DELTA_R_GHZ,
        "floor_at_C2_kappa_int": float(np.pi * KAPPA_R_INT_MHZ * 1e-3
                                       / DELTA_R_GHZ),
        "max_bus_depth_N_purcell": float(DELTA_R_GHZ
                                         / (np.pi * KAPPA_R_MHZ * 1e-3)),
    },
    "drive_bus": {
        "pi_kappa_over_Delta": float(np.pi * KAPPA_D_MHZ * 1e-3 / DELTA_D_GHZ),
        "direct_t_over_T1P": float(on_off["drive_bus"]["t_iSWAP_ns"] * 1e-9
                                   * GAMMA_P_D),
        "kappa_MHz": KAPPA_D_MHZ, "Delta_GHz": DELTA_D_GHZ,
        "max_bus_depth_N_purcell": float(DELTA_D_GHZ
                                         / (np.pi * KAPPA_D_MHZ * 1e-3)),
    },
    "absolute_Q_i_floor": {
        "derivation": ("kappa >= f_bus/Q_i (constraint C2) => the two-qubit "
                       "Purcell error of ANY cavity-bus iSWAP on this platform "
                       "is 2 pi kappa/D >= 2 pi (1 + f01/D)/Q_i, which tends to "
                       "2 pi/Q_i as the bus detuning grows. g, d_eff, beta and "
                       "C_c all cancel."),
        "Q_i_measured": inputs.Q_I,
        "two_pi_over_Qi": float(2 * np.pi / inputs.Q_I),
        "at_readout_Delta_4GHz": float(2 * np.pi * (1 + F01_DESIGN / DELTA_R_GHZ)
                                       / inputs.Q_I),
        "at_drive_Delta_104GHz": float(2 * np.pi * (1 + F01_DESIGN / DELTA_D_GHZ)
                                       / inputs.Q_I),
        "drive_over_absolute_floor": float((1 + F01_DESIGN / DELTA_D_GHZ)),
    },
}

# ------------------------- EXTENSION: spend the leftover C_c on the drive bus
# The recommended readout point uses C_c,readout = 6.4556 fF of an 11.9997 fF
# per-island cap, and the drive electrode uses only 2.2454 fF (d_eff = 20 um).
# Because g ~ beta * V_zpf ~ C_c * d_eff ~ d_eff^2, enlarging the drive
# electrode buys J_dr ~ d_eff^4 at CONSTANT Purcell error (pi kappa/Delta is
# g-independent).  Constraints: the per-island C_c cap, the readout window's
# T1_Purcell >= 30 us floor on the COMBINED Purcell, the dispersive criterion,
# and an electrical-smallness sanity bound on the lumped pickup model.
LAMBDA_EFF_DRIVE_M = 299792458.0 / (inputs.F_CAV_GHZ * 1e9
                                    * np.sqrt(inputs.EPS_C))
T1_PURCELL_FLOOR_S = 30e-6           # src: qubit/cavity.py:T1_PURCELL_MIN_S
d_scan_um = np.arange(20.0, 70.01, 0.25)
Cc_d = coupling.c_c_selfcap(d_scan_um * 1e-6)                  # F
beta_d_scan = Cc_d / (C_SIGMA_FF * 1e-15)
g_d_scan = np.array([coupling.coupling_g(inputs.F_CAV_GHZ, inputs.V_MODE_M3,
                                         d * 1e-6, N01_DESIGN, b)
                     for d, b in zip(d_scan_um, beta_d_scan)])   # GHz
J_d_scan = g_d_scan ** 2 / DELTA_D_GHZ                           # GHz
t_d_scan = 1.0 / (2.0 * J_d_scan * 1e9)                          # s
gam_d_scan = (g_d_scan / DELTA_D_GHZ) ** 2 * 2 * np.pi * KAPPA_D_MHZ * 1e6
T1P_tot_scan = 1.0 / (GAMMA_P_R + gam_d_scan)
ok_cap = (Cc_d * 1e15 + C_C_READOUT_FF) <= C_C_CAP_FF
ok_purc = T1P_tot_scan >= T1_PURCELL_FLOOR_S
ok_disp = (g_d_scan / DELTA_D_GHZ) ** 2 <= 0.01
ok_lump = (d_scan_um * 1e-6 / LAMBDA_EFF_DRIVE_M) <= 0.1
ok_all = ok_cap & ok_purc & ok_disp & ok_lump
err_scan = 2.0 * t_d_scan * (1.0 / T1P_tot_scan + 1.0 / T1_qp["1e-07"])
k_best = int(np.arange(len(d_scan_um))[ok_all][int(np.argmin(err_scan[ok_all]))])
chi_d_best = float(g_d_scan[k_best] ** 2 * ALPHA_GHZ
                   / (DELTA_D_GHZ * (DELTA_D_GHZ + ALPHA_GHZ)))

drive_upgrade = {
    "idea": ("the drive electrode is the cheapest unused resource on the "
             "island: g ~ C_c * d_eff ~ d_eff^2 so J_dr ~ d_eff^4, while the "
             "bus Purcell error pi kappa_d/Delta_d does NOT depend on g"),
    "constraints": {
        "C_c_cap_fF": C_C_CAP_FF, "C_c_readout_fF": C_C_READOUT_FF,
        "C_c_drive_headroom_fF": C_C_CAP_FF - C_C_READOUT_FF,
        "T1_Purcell_floor_us": T1_PURCELL_FLOOR_S * 1e6,
        "dispersive_g_over_Delta_sq_max": 0.01,
        "lumped_smallness_d_over_lambda_max": 0.1,
        "lambda_eff_drive_um": LAMBDA_EFF_DRIVE_M * 1e6,
        "C_c_model": "coupling.c_c_selfcap: C_c = kappa_geo eps0 eps_eff d_eff",
    },
    "binding_constraint_at_optimum": (
        [name for name, ok in (("C_c budget cap", ok_cap),
                               ("T1_Purcell(combined) >= 30 us", ok_purc),
                               ("(g/Delta)^2 <= 0.01", ok_disp),
                               ("d_eff <= lambda/10", ok_lump))
         if k_best + 1 < len(ok) and not ok[k_best + 1]] or ["grid end"]),
    "baseline_d_eff_um": 20.0,
    "optimum": {
        "d_eff_um": float(d_scan_um[k_best]),
        "C_c_drive_fF": float(Cc_d[k_best] * 1e15),
        "C_c_sum_fF": float(Cc_d[k_best] * 1e15 + C_C_READOUT_FF),
        "beta_drive": float(beta_d_scan[k_best]),
        "g_MHz": float(g_d_scan[k_best]) * 1e3,
        "J_MHz": float(J_d_scan[k_best]) * 1e3,
        "t_iSWAP_ns": float(t_d_scan[k_best]) * 1e9,
        "T1_Purcell_drive_us": float(1.0 / gam_d_scan[k_best]) * 1e6,
        "T1_Purcell_combined_us": float(T1P_tot_scan[k_best]) * 1e6,
        "g_over_Delta_sq": float((g_d_scan[k_best] / DELTA_D_GHZ) ** 2),
        "d_over_lambda": float(d_scan_um[k_best] * 1e-6 / LAMBDA_EFF_DRIVE_M),
        "chi_drive_kHz": float(chi_d_best) * 1e6,
        "chi_over_kappa_drive": float(abs(chi_d_best) * 1e3 / KAPPA_D_MHZ),
        "C_qq_equivalent_aF": cqq_equiv_fF(float(J_d_scan[k_best])) * 1e3,
        "incoherent_budget_xqp_1e-7": gate_budget(
            float(t_d_scan[k_best]), T2_echo_nu[i_apex], T2_echo_nu[i_apex],
            GAMMA_P_R + float(gam_d_scan[k_best]), 1e-7),
        "ZZ_kHz": float(abs(zz_cavity_corrected(
            float(J_d_scan[k_best]), abs(d12[k_off]), alpha_nu[i_apex],
            alpha_nu[i_edge_hi], DELTA_D_GHZ)) * 1e6),
        "P_residual_off": float(residual_swap(float(J_d_scan[k_best]),
                                              abs(d12[k_off]))),
    },
    "rows": [{"d_eff_um": float(d_scan_um[k]),
              "C_c_drive_fF": float(Cc_d[k] * 1e15),
              "J_MHz": float(J_d_scan[k]) * 1e3,
              "t_iSWAP_ns": float(t_d_scan[k]) * 1e9,
              "T1_Purcell_combined_us": float(T1P_tot_scan[k]) * 1e6,
              "budget_total_xqp_1e-7": float(err_scan[k]),
              "feasible": bool(ok_all[k])}
             for k in range(0, len(d_scan_um), 20)],
    "no_readout_verdict_survives": {
        "chi_over_kappa_at_largest_electrode": float(
            abs(chi_d_best) * 1e3 / KAPPA_D_MHZ),
        "statement": ("even with the largest drive electrode the budget "
                      "allows, chi/kappa through the 145 GHz cavity stays far "
                      "below 1: the P2/P5 'the drive cavity cannot read the "
                      "qubit out' verdict is robust to this upgrade"),
    },
    "caveats": ("d_eff_drive = 20 um is a STORED photonics input; changing it "
                "is a design proposal, not a measured value. It would require "
                "re-running P2's divider and P5's control-line budget, and it "
                "pushes the lumped pickup from lambda/30 to lambda/13 at "
                "145 GHz. Reported as a parametric extension, not a result."),
}

# --------------------------------------- both buses are on at the same time --
J_both = J_ro + J_dr
combined_bus = dict(
    J_readout_MHz=float(J_ro[k_on]) * 1e3,
    J_drive_MHz=float(J_dr[k_on]) * 1e3,
    J_total_MHz=float(J_both[k_on]) * 1e3,
    drive_contribution_pct=float(J_dr[k_on] / J_ro[k_on] * 100),
    t_iSWAP_ns=float(1.0 / (2 * J_both[k_on] * 1e9)) * 1e9,
    statement=("the two buses are NOT alternatives: every island is capacitively "
               "attached to both cavities, so J_total = J_ro + J_dr always. The "
               "drive cavity adds 6.3% to the readout bus at the stored 20 um "
               "electrode and CANNOT be switched off either."),
)

# ------------------------------------------------- isolation/coherence Pareto
# For every reachable parking split, the best (largest) worst-qubit T2_echo.
# T2_echo falls monotonically with |nu + 2.4|, so min(T2_1, T2_2) is set by the
# LARGER of the two detunings.
T2min = np.minimum(T2_echo_nu[i1], T2_echo_nu[i2])
T_IDLE_REF_S = 100e-9        # reference idle window: ~1 readout-bus iSWAP,
#                              ~62 single-qubit gates at the 1.6044 ns floor

# APEX LAW, tested EXACTLY on the grid (no binning). For any parking (d_a, d_b)
# with d_a <= d_b, moving the inner qubit to the apex leaves min(T2) = T2(d_b)
# unchanged and INCREASES the split (d_b^2 - d_a^2 -> d_b^2). The apex family
# therefore weakly dominates every other parking, for every split.
outer_idx = np.where(d_grid[i1] >= d_grid[i2], i1, i2)
apex_split_of = np.abs(f01_nu[i_apex] - f01_nu)          # split of (apex, d)
apex_T2min_of = np.minimum(T2_echo_nu[i_apex], T2_echo_nu)
viol_split = apex_split_of[outer_idx] < np.abs(d12) * (1.0 - 1e-12)
viol_T2 = apex_T2min_of[outer_idx] < T2min * (1.0 - 1e-12)
pareto_apex_law = bool((not viol_split.any()) and (not viol_T2.any()))
apex_law_rows = dict(
    n_pairs=int(len(d12)),
    n_split_violations=int(viol_split.sum()),
    n_T2_violations=int(viol_T2.sum()),
    max_split_gain_by_moving_inner_to_apex_GHz=float(
        np.max(apex_split_of[outer_idx] - np.abs(d12))),
)

# The Pareto front IS the apex family: qubit 1 pinned at nu = -2.4 (double
# sweet spot), qubit 2 walked out towards the window edge.
pareto = []
for j in range(i_apex, NGRID):
    kb = i_apex * NGRID + j
    pareto.append(dict(
        split_GHz=float(abs(d12[kb])),
        nu1=float(nu_grid[i_apex]), nu2=float(nu_grid[j]),
        d1_from_apex=float(d_grid[i_apex]), d2_from_apex=float(d_grid[j]),
        D1_parked_GHz_per_nu=float(D1_nu[j]),
        P_residual_readout=float(P_ro[kb]),
        P_residual_drive=float(P_dr[kb]),
        ZZ_readout_kHz_cavity_corrected=float(abs(ZZ_ro_corr[kb]) * 1e6),
        ZZ_drive_kHz_cavity_corrected=float(abs(ZZ_dr_corr[kb]) * 1e6),
        T2_echo_min_us=float(T2min[kb] * 1e6),
        T2_echo_min_in_readout_gate_times=float(
            T2min[kb] / t_iswap_ro[k_on]),
        # OFF-state cost over a 100 ns idle window: the coherent residual swap
        # PLUS the 1/f dephasing the detuned qubit pays for being off-apex.
        idle_dephasing_100ns=float((T_IDLE_REF_S / T2min[kb]) ** 2),
        idle_cost_readout_100ns=float(P_ro[kb]
                                      + (T_IDLE_REF_S / T2min[kb]) ** 2),
        idle_cost_drive_100ns=float(P_dr[kb]
                                    + (T_IDLE_REF_S / T2min[kb]) ** 2),
        one_qubit_at_apex=True,
    ))

_ic_r = np.array([r["idle_cost_readout_100ns"] for r in pareto])
_ic_d = np.array([r["idle_cost_drive_100ns"] for r in pareto])
off_state_cost = dict(
    reference_idle_s=T_IDLE_REF_S,
    reference_note=("100 ns = one readout-bus iSWAP = 62 single-qubit gates at "
                    "the honest 1.6044 ns floor"),
    metric="P_residual_swap + (t_idle/T2_echo_parked)^2, on the apex family",
    readout_bus=dict(
        best_total=float(_ic_r.min()),
        at_split_GHz=float(pareto[int(np.argmin(_ic_r))]["split_GHz"]),
        at_nu2=float(pareto[int(np.argmin(_ic_r))]["nu2"]),
        P_residual=float(pareto[int(np.argmin(_ic_r))]["P_residual_readout"]),
        dephasing=float(pareto[int(np.argmin(_ic_r))]["idle_dephasing_100ns"]),
        T2_echo_parked_us=float(pareto[int(np.argmin(_ic_r))]["T2_echo_min_us"]),
        at_window_edge=dict(
            P_residual=float(pareto[-1]["P_residual_readout"]),
            dephasing=float(pareto[-1]["idle_dephasing_100ns"]),
            total=float(_ic_r[-1])),
    ),
    drive_bus=dict(
        best_total=float(_ic_d.min()),
        at_split_GHz=float(pareto[int(np.argmin(_ic_d))]["split_GHz"]),
        at_nu2=float(pareto[int(np.argmin(_ic_d))]["nu2"]),
        P_residual=float(pareto[int(np.argmin(_ic_d))]["P_residual_drive"]),
        dephasing=float(pareto[int(np.argmin(_ic_d))]["idle_dephasing_100ns"]),
        T2_echo_parked_us=float(pareto[int(np.argmin(_ic_d))]["T2_echo_min_us"]),
    ),
    finding=("THE BINDING RESULT. The nu-window is not a usable on/off knob "
             "for the readout bus: suppressing the residual swap requires "
             "parking a qubit far enough off-apex that its own echo coherence "
             "collapses, and the two costs cross at a floor of "
             f"{_ic_r.min():.3f} per 100 ns of idling. The 22x weaker drive "
             f"bus reaches {_ic_d.min():.2e} at the same knob, because its "
             "residual swap is already negligible at small splits."),
)


# --------------------------------------------- parking-error sensitivity ----
# ON state = symmetric parking; a filling error dnu detunes it by |D1| dnu.
park_err = []
for d_set in DNU_SET:
    for label, idx in (("apex", i_apex), ("mid", NGRID // 2 + NGRID // 4),
                       ("edge", i_edge_hi)):
        dd = abs(D1_nu[idx]) * d_set                      # GHz of detuning error
        Jl = float(J_ro[idx * NGRID + idx])
        park_err.append(dict(
            dnu_rms=d_set, parking=label, nu=float(nu_grid[idx]),
            D1_GHz_per_nu=float(D1_nu[idx]),
            detuning_error_MHz=float(dd * 1e3),
            swap_error_readout=float(1.0 - residual_swap(Jl, dd)),
            detuning_over_J=float(dd / Jl) if Jl else None,
        ))

# ---------------------------------------------------------- N_ops vs x_qp ---
def nops_table(t_g):
    return {k: float(v / t_g) for k, v in T1_qp.items()}


nops = {
    "readout_bus": dict(t_iSWAP_ns=on_off["readout_bus"]["t_iSWAP_ns"],
                        N_ops=nops_table(on_off["readout_bus"]["t_iSWAP_ns"] * 1e-9),
                        N_purcell=purcell_floor["readout_bus"]["max_bus_depth_N_purcell"]),
    "drive_bus": dict(t_iSWAP_ns=on_off["drive_bus"]["t_iSWAP_ns"],
                      N_ops=nops_table(on_off["drive_bus"]["t_iSWAP_ns"] * 1e-9),
                      N_purcell=purcell_floor["drive_bus"]["max_bus_depth_N_purcell"]),
    "drive_bus_shared_budget_beta": dict(
        beta=BETA_DRIVE_SHARED,
        J_MHz=float(J_dr_sh[k_on]) * 1e3,
        t_iSWAP_ns=float(t_iswap_dr_sh[k_on]) * 1e9,
        N_ops=nops_table(float(t_iswap_dr_sh[k_on]))),
}
for row in CQQ_ROWS:
    nops[f"C_qq_{row['C_qq_fF']:g}fF"] = dict(
        C_qq_fF=row["C_qq_fF"], J_MHz=row["J_MHz"],
        t_iSWAP_ns=row["t_iSWAP_ns"],
        N_ops=nops_table(row["t_iSWAP_ns"] * 1e-9),
        dispersive_ok_at_0p29GHz=row["dispersive_ok_vs_detuning"]["0.29"])
cap029 = CQQ_BUDGET["0.29"]
nops["C_qq_at_dispersive_cap_0p29GHz"] = dict(
    C_qq_fF=cap029["C_qq_max_fF"], J_MHz=cap029["J_at_cap_MHz"],
    t_iSWAP_ns=cap029["t_iSWAP_at_cap_ns"],
    N_ops=nops_table(cap029["t_iSWAP_at_cap_ns"] * 1e-9),
    budget=gate_budget(cap029["t_iSWAP_at_cap_ns"] * 1e-9,
                       T2_echo_nu[i_apex], T2_echo_nu[i_apex],
                       GAMMA_P_R, 1e-7))

# ------------------------------------ stray direct C_qq vs the bus coupling --
# Two islands on ONE cavity sit at the SAME field maximum, so their stray
# mutual capacitance adds directly to the bus J.  For the bus to be the
# designable coupler the stray must stay below it.
J_direct = lambda C: 8.0 * EC_GHZ * (C / C_SIGMA_FF) * N01_DESIGN ** 2   # GHz
stray = dict(
    model="J_direct = 8 E_C (C_qq/C_Sigma) n01^2  (qubit.json:two_qubit_sketch.model)",
    C_qq_equal_to_readout_bus_aF=cqq_equiv_fF(float(J_ro[k_on])) * 1e3,
    C_qq_equal_to_drive_bus_aF=cqq_equiv_fF(float(J_dr[k_on])) * 1e3,
    C_qq_dispersive_cap_aF=cap029["C_qq_max_fF"] * 1e3,
    tightening_vs_cap=float(cap029["C_qq_max_fF"]
                            / cqq_equiv_fF(float(J_ro[k_on]))),
    J_total_rows=[dict(C_qq_stray_aF=c * 1e3,
                       J_direct_MHz=J_direct(c) * 1e3,
                       J_total_readout_bus_MHz=(J_direct(c)
                                                + float(J_ro[k_on])) * 1e3,
                       fractional_shift=float(J_direct(c) / J_ro[k_on]))
                  for c in (0.0001, 0.0005, 0.001, 0.003232, 0.01, 0.018576)],
    sign_note=("both cavities sit ABOVE f01 (Delta_r = +4.0 GHz, "
               "Delta_d = +104.4 GHz), so J_bus > 0 and adds to the direct "
               "capacitive J; a readout mode placed BELOW f01 would subtract "
               "and could cancel the stray -- an unexplored design knob."),
)

# ------------------------------- measurement conflict, QUANTIFIED not modelled
def chi_of(g, D, al):
    return g ** 2 * al / (D * (D + al))


chi_apex = chi_of(g_r_nu[i_apex], Delta_r_nu[i_apex], alpha_nu[i_apex])
chi_edge = chi_of(g_r_nu[i_edge_hi], Delta_r_nu[i_edge_hi], alpha_nu[i_edge_hi])
meas = dict(
    statement=("a SHARED readout cavity gives one joint dispersive shift: the "
               "pull is chi_1 n_1 + chi_2 n_2, so |01> and |10> are separated "
               "only by |chi_1 - chi_2|. Per-qubit readout needs the FDM "
               "neighbour cavity -- NOT modelled here."),
    chi_apex_MHz=float(chi_apex * 1e3),
    chi_window_edge_MHz=float(chi_edge * 1e3),
    max_dchi_MHz=float(abs(chi_apex - chi_edge) * 1e3),
    kappa_MHz=KAPPA_R_MHZ,
    dchi_over_kappa=float(abs(chi_apex - chi_edge) * 1e3 / KAPPA_R_MHZ),
    resolvable=bool(abs(chi_apex - chi_edge) * 1e3 > KAPPA_R_MHZ),
    eta_readout=ETA_READOUT,
    verdict=("the largest chi split reachable inside the fixed-drive window is "
             "smaller than the cavity linewidth, so ONE shared readout cavity "
             "measures only the total excitation number of the pair"),
)

# Spatial vs frequency addressability of the DRIVE (the fork reframing).
addressing = dict(
    f01_span_in_window_GHz=F01_SPAN_GHZ,
    measured_min_channel_spacing_at_8ns_GHz=MIN_SPACING_8NS_GHZ,
    frequency_addressable=bool(F01_SPAN_GHZ > MIN_SPACING_8NS_GHZ),
    deficit_factor=float(MIN_SPACING_8NS_GHZ / F01_SPAN_GHZ),
    statement=("the fork verdict 'EXACTLY 1 qubit' is about FREQUENCY-selective "
               "driving and it is unchanged here: 0.2906 GHz of span against a "
               "measured 0.9235 GHz minimum spacing even at t_g = 8 ns. A bus "
               "does not repair that. Two bus-coupled islands therefore need "
               "SPATIALLY separate drive ports; what the bus removes is the "
               "coupler fab tolerance, not the addressing requirement."),
)

# ============================================================== validation ===
val = {}
# (V1) J reproduces g^2/Delta at equal detunings.
val["V1_J_equal_detuning"] = dict(
    J_grid_MHz=float(J_ro[k_on]) * 1e3,
    g2_over_Delta_MHz=float(g_r_nu[i_apex] ** 2 / Delta_r_nu[i_apex]) * 1e3,
    reldev=float(abs(J_ro[k_on] - g_r_nu[i_apex] ** 2 / Delta_r_nu[i_apex])
                 / (g_r_nu[i_apex] ** 2 / Delta_r_nu[i_apex])),
    tolerance=1e-12, passed=bool(abs(J_ro[k_on]
                                     - g_r_nu[i_apex] ** 2 / Delta_r_nu[i_apex])
                                 / (g_r_nu[i_apex] ** 2 / Delta_r_nu[i_apex])
                                 < 1e-12))
# (V2) exact-diag J from the degenerate splitting vs the dispersive formula.
val["V2_J_exact_vs_dispersive"] = dict(
    J_exact_MHz=J_exact_apex * 1e3, J_pert_MHz=J_pert_apex * 1e3,
    reldev=float(abs(J_exact_apex - J_pert_apex) / J_pert_apex),
    tolerance=0.05,
    passed=bool(abs(J_exact_apex - J_pert_apex) / J_pert_apex < 0.05))
# (V3a) the 4th-order ZZ formula against a DIRECT-exchange diagonalisation
#       (no cavity): this is the clean test that the formula itself is right.
val["V3a_ZZ_formula_vs_direct_exchange_diag"] = dict(
    regime="two transmons, 4 levels each, direct exchange J, no cavity",
    J_MHz=J_fix * 1e3, delta12_GHz=d12_max,
    zz_formula_kHz=float(zz_pure) * 1e6,
    zz_diag_kHz=zz_direct_check * 1e6,
    reldev=float(abs(zz_direct_check - zz_pure) / abs(zz_pure)),
    tolerance=0.01,
    passed=bool(abs(zz_direct_check - zz_pure) / abs(zz_pure) < 0.01))
# (V3b) the FULL two-transmon + cavity diagonalisation is systematically LOWER
#       than the pure-exchange formula.  That is physics, not a numerical
#       failure: at fixed effective J the cavity-mediated cross-Kerr partially
#       cancels the exchange-mediated ZZ, and the deficit dies as 1/Delta_bus.
disp_rows = [r for r in zz_rows if abs(r["delta12_GHz"]) >= 10 * J_pert_apex]
dev_raw = [abs(r["zz_exact_GHz"] - r["zz_pert_GHz"]) / abs(r["zz_pert_GHz"])
           for r in disp_rows]
corr = 1.0 - abs(ALPHA_GHZ) / DELTA_R_GHZ
dev_corr = [abs(r["zz_exact_GHz"] - corr * r["zz_pert_GHz"])
            / abs(corr * r["zz_pert_GHz"]) for r in disp_rows]
val["V3b_ZZ_cavity_diag_vs_pure_exchange"] = dict(
    n_rows=len(disp_rows), regime="|delta12| >= 10 J, cavity included",
    max_reldev_vs_pure_exchange=float(max(dev_raw)),
    median_reldev_vs_pure_exchange=float(np.median(dev_raw)),
    cavity_cancellation_factor=corr,
    max_reldev_vs_cavity_corrected=float(max(dev_corr)),
    tolerance=0.05, passed=bool(max(dev_corr) < 0.05),
    finding=("the cavity bus produces LESS static ZZ than a direct exchange "
             "of the same strength: the measured factor at Delta_r = 4 GHz is "
             f"{corr:.4f}, i.e. a {100*(1-corr):.1f}% cancellation, and the "
             "Delta_r sweep below shows it converging to 1 as 1 - |alpha|/Delta"),
    delta_sweep_at_fixed_J=zz_vs_Delta,
    empirical_law="zeta_cavity_bus = zeta_exchange * (1 - |alpha|/Delta_bus)",
    law_max_reldev=float(max(r["law_reldev"] for r in zz_vs_Delta)),
    zz_at_zero_split_exact_kHz=abs(zz_rows[i_apex]["zz_exact_GHz"]) * 1e6,
    zz_at_zero_split_pure_exchange_kHz=abs(zz_rows[i_apex]["zz_pert_GHz"]) * 1e6,
    zz_textbook_4J2_over_alpha_kHz=float(
        4 * J_pert_apex ** 2 / abs(alpha_nu[i_apex]) * 1e6),
    truncation_convergence=zz_conv,
    truncation_reldev_3x5_vs_4x8=float(
        abs(zz_conv["nlev3_ncav5"] - zz_conv["nlev4_ncav8"])
        / abs(zz_conv["nlev4_ncav8"])),
    rwa_reldev=float(abs(zz_conv["nlev4_ncav8"] - zz_conv["nlev4_ncav8_noRWA"])
                     / abs(zz_conv["nlev4_ncav8"])))
# (V4) cross-check against the stored artifact numbers (the required check).
val["V4_artifact_cross_checks"] = dict(
    g_readout_MHz=dict(recomputed=float(g_r_nu[i_apex]) * 1e3,
                       stored=G_R_STORED_MHZ,
                       reldev=float(abs(g_r_nu[i_apex] * 1e3 - G_R_STORED_MHZ)
                                    / G_R_STORED_MHZ)),
    g_drive_MHz=dict(recomputed=float(g_d_nu[i_apex]) * 1e3,
                     stored=G_D_STORED_MHZ,
                     reldev=float(abs(g_d_nu[i_apex] * 1e3 - G_D_STORED_MHZ)
                                  / G_D_STORED_MHZ)),
    chi_readout_MHz=dict(recomputed=float(chi_apex) * 1e3,
                         stored=CHI_R_STORED_MHZ,
                         reldev=float(abs(chi_apex * 1e3 - CHI_R_STORED_MHZ)
                                      / abs(CHI_R_STORED_MHZ))),
    T2_echo_apex_us=dict(recomputed=float(T2_echo_nu[i_apex]) * 1e6,
                         stored=QJ["budgets"]["T2_maps"]["headline_rows"][0]["T2_echo_us"],
                         reldev=float(abs(T2_echo_nu[i_apex] * 1e6
                                          - QJ["budgets"]["T2_maps"]["headline_rows"][0]["T2_echo_us"])
                                      / QJ["budgets"]["T2_maps"]["headline_rows"][0]["T2_echo_us"])),
    T1_qp_1e_6_us=dict(recomputed=T1_qp["1e-06"] * 1e6,
                       stored=QJ["budgets"]["T1_stack"]["qp_full_kernel"]["T1_us_vs_xqp_vs_T_qp"]["0.100"]["1e-06"]),
    f01_span_GHz=dict(recomputed=float(f01_nu.max() - f01_nu.min()),
                      stored=F01_SPAN_GHZ,
                      reldev=float(abs((f01_nu.max() - f01_nu.min())
                                       - F01_SPAN_GHZ) / F01_SPAN_GHZ)),
    D1_window_edge=dict(recomputed=float(abs(D1_nu[i_edge_hi])),
                        stored=D1_EDGE_STORED,
                        reldev=float(abs(abs(D1_nu[i_edge_hi]) - D1_EDGE_STORED)
                                     / D1_EDGE_STORED)),
    tolerance=0.01,
)
val["V4_artifact_cross_checks"]["passed"] = bool(all(
    v["reldev"] < 0.01 for v in val["V4_artifact_cross_checks"].values()
    if isinstance(v, dict) and "reldev" in v))
# (V5) per-island C_Sigma budget: each island spends its OWN C_c.
c_spent = C_C_READOUT_FF + C_C_DRIVE_FF
val["V5_per_island_C_Sigma_budget"] = dict(
    C_c_readout_fF=C_C_READOUT_FF, C_c_drive_fF=C_C_DRIVE_FF,
    C_c_sum_per_island_fF=c_spent, C_c_cap_fF=C_C_CAP_FF,
    headroom_fF=C_C_CAP_FF - c_spent,
    oversubscription=c_spent / C_C_CAP_FF,
    n_islands=2, passed=bool(c_spent <= C_C_CAP_FF),
    statement=("each island carries its own C_Sigma = 12.99971 fF and its own "
               "pair of collection electrodes; the CAVITY is shared, the "
               "BUDGET is not. Two islands therefore spend 8.70101 fF each "
               "against an 11.99971 fF cap, exactly as the single-qubit "
               "recommended point does -- no shared-budget violation, and "
               "3.29870 fF of headroom per island remains. A direct C_qq "
               "would instead spend from BOTH budgets simultaneously."),
    budget_law=C_J_FF,
)

# ================================================================== export ===
def rnd(o, sig=7):
    if isinstance(o, dict):
        return {k: rnd(v, sig) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [rnd(v, sig) for v in o]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        x = float(o)
        if not np.isfinite(x) or x == 0.0:
            return x
        return float(f"%.{sig}g" % x)
    return o


rb, db = on_off["readout_bus"], on_off["drive_bus"]
doc = {
    "_meta": {
        "title": ("The cavities the qubits already touch are two-qubit buses: "
                  "J through the shared readout (and drive) cavity vs the "
                  "unmanufacturable tens-of-aF C_qq"),
        "module": "M6 exploration 4/5",
        "date": "2026-07-27",
        "generator": "scripts/explore_cavity_bus_two_qubit.py",
        "deterministic": True,
        "rng": "none -- no stochastic step; no seed is used or needed",
        "claim_discipline": ("DESIGN STUDY. Every coherence number is "
                             "parametric in x_qp or in the 1/f amplitude A. No "
                             "absolute TBG T1/T2 and no gate fidelity is "
                             "claimed: the error entries are leading-order "
                             "INCOHERENT BUDGET CONTRIBUTIONS, not simulated "
                             "fidelities."),
        "owns": ["scripts/explore_cavity_bus_two_qubit.py",
                 "out/m6/explore/cavity_bus_two_qubit.json"],
        "reads": ["qubit.json", "qubit/ (spectrum, dome, cavity, coupling, "
                  "noise_1f, qp_kernel, inputs)"],
    },
    "free_parameters": {
        "A_1f": {"value": A_1F, "range": list(A_1F_RANGE),
                 "units": "filling nu per e-fold, S_nu(w) = A^2/|w| two-sided",
                 "sensitivity_values": list(A_1F_SENS)},
        "x_qp": {"grid": list(X_QP_GRID), "T_qp_K": T_QP_K},
        "beta_drive": {"nominal": BETA_DRIVE, "shared_budget": BETA_DRIVE_SHARED},
        "dnu_set": {"grid": list(DNU_SET),
                    "meaning": "rms gate-setting error on the parking filling"},
        "truncation": {"nlev": NLEV, "ncav": NCAV,
                       "convergence_check": "nlev 4 / ncav 8, and non-RWA"},
        "grid": {"n_nu": NGRID, "n_pairs": NGRID ** 2,
                 "nu_window": NU_WINDOW},
        "IR_cutoff": {"f_ir_Hz": 1.0,
                      "note": "P4 convention, 1 s acquisition; T2 apex is the "
                              "most IR-sensitive number in the export"},
    },
    "inputs_used": {
        "EC_GHz": EC_GHZ, "C_Sigma_fF": C_SIGMA_FF, "alpha_GHz": ALPHA_GHZ,
        "n01_design": N01_DESIGN, "f01_design_GHz": F01_DESIGN,
        "ladder_ratio_n12_over_n01": LADDER_RATIO,
        "readout": {"f_r_GHz": F_R_GHZ, "Delta_r_GHz": DELTA_R_GHZ,
                    "d_eff_um": RP["d_eff_um"], "beta_readout": BETA_R,
                    "kappa_MHz": KAPPA_R_MHZ,
                    "kappa_intrinsic_MHz": KAPPA_R_INT_MHZ,
                    "g_MHz": G_R_STORED_MHZ,
                    "T1_Purcell_us": T1P_R_STORED_US,
                    "eta_readout": ETA_READOUT,
                    "V_mode_readout_um3": V_MODE_R_M3 * 1e18},
        "drive": {"f_cav_GHz": inputs.F_CAV_GHZ, "Delta_d_GHz": DELTA_D_GHZ,
                  "d_eff_um": inputs.D_EFF_DRIVE_M * 1e6,
                  "beta_drive": BETA_DRIVE, "kappa_MHz": KAPPA_D_MHZ,
                  "g_MHz": G_D_STORED_MHZ, "T1_Purcell_us": T1P_D_STORED_US},
        "window": {"nu_window": NU_WINDOW, "f01_span_GHz": F01_SPAN_GHZ,
                   "D1_edge_GHz_per_nu": D1_EDGE_STORED},
    },
    "grid_tables": {
        "nu": nu_grid.tolist(),
        "d_from_apex": d_grid.tolist(),
        "EJ_GHz": EJ_nu.tolist(),
        "f01_GHz": f01_nu.tolist(),
        "n01": n01_nu.tolist(),
        "alpha_GHz": alpha_nu.tolist(),
        "D1_GHz_per_nu": D1_nu.tolist(),
        "D2_GHz_per_nu2": D2_nu.tolist(),
        "T2_echo_us_A1e-4": (T2_echo_nu * 1e6).tolist(),
        "T2_ramsey_us_A1e-4": (T2_ram_nu * 1e6).tolist(),
        "T2_echo_us_vs_A": {k: (v * 1e6).tolist() for k, v in T2_echo_A.items()},
        "g_readout_MHz": (g_r_nu * 1e3).tolist(),
        "g_drive_MHz": (g_d_nu * 1e3).tolist(),
        "variation_across_window": {
            "g_readout_pct": float((g_r_nu.max() / g_r_nu.min() - 1) * 100),
            "J_readout_pct": float((J_ro.max() / J_ro.min() - 1) * 100),
            "n01_pct": float((n01_nu.max() / n01_nu.min() - 1) * 100),
            "EJ_pct": float((EJ_nu.max() / EJ_nu.min() - 1) * 100),
        },
    },
    "readout_bus": {
        "J_MHz": {"min": float(J_ro.min()) * 1e3, "max": float(J_ro.max()) * 1e3,
                  "at_apex_pair": float(J_ro[k_on]) * 1e3},
        "t_iSWAP_ns": rb["t_iSWAP_ns"],
        "t_sqrt_iSWAP_ns": rb["t_sqrt_iSWAP_ns"],
        "t_convention": ("t_iSWAP = 1/(2J), the convention already used in "
                         "qubit.json:two_qubit_sketch (J = 15.611 MHz -> "
                         "32.028 ns). NOTE: the brief's '0.8-1.0 us' drive-bus "
                         "figure corresponds instead to 1/(4J); the repo "
                         "convention is kept here and gives 1.58-1.99 us."),
        "C_qq_equivalent_aF": rb["C_qq_equivalent_aF"],
        "on_off": rb,
        "dispersive_headroom": {
            "J_over_max_split": float(J_ro[k_on] / abs(d12[k_off])),
            "criterion": J_OVER_D_DISPERSIVE,
            "margin_factor": float(J_OVER_D_DISPERSIVE
                                   / (J_ro[k_on] / abs(d12[k_off]))),
            "statement": ("J_bus/D = 0.0174 against the sketch's 0.1 dispersive "
                          "criterion: the readout bus sits 5.75x INSIDE the "
                          "budget that forced C_qq <= 18.6 aF"),
        },
        "ZZ_sweep_apex_to_edge": zz_rows,
        "ZZ_at_max_split_kHz_exact": float(abs(zz_rows[i_edge_hi]["zz_exact_GHz"])
                                           * 1e6),
        "ZZ_at_max_split_kHz_pure_exchange": float(abs(ZZ_ro[k_off]) * 1e6),
        "ZZ_at_max_split_kHz_cavity_corrected": float(abs(ZZ_ro_corr[k_off])
                                                      * 1e6),
        "ZZ_at_degeneracy_kHz_exact": float(abs(zz_rows[i_apex]["zz_exact_GHz"])
                                            * 1e6),
        "ZZ_clock_ns_at_max_split": float(
            1.0 / (2 * abs(zz_rows[i_edge_hi]["zz_exact_GHz"]) * 1e9) * 1e9),
        "ZZ_phase_per_1q_gate_rad": float(
            2 * np.pi * abs(zz_rows[i_edge_hi]["zz_exact_GHz"]) * 1e9
            * GATE_FLOOR_NS * 1e-9),
        "ZZ_phase_per_iSWAP_rad": float(
            2 * np.pi * abs(zz_rows[i_apex]["zz_exact_GHz"]) * 1e9
            * rb["t_iSWAP_ns"] * 1e-9),
    },
    "combined_bus": combined_bus,
    "drive_electrode_upgrade": drive_upgrade,
    "drive_bus": {
        "J_MHz": {"beta_0p17273": float(J_dr[k_on]) * 1e3,
                  "beta_0p15385_shared": float(J_dr_sh[k_on]) * 1e3},
        "t_iSWAP_ns": {"beta_0p17273": db["t_iSWAP_ns"],
                       "beta_0p15385_shared": float(t_iswap_dr_sh[k_on]) * 1e9},
        "C_qq_equivalent_aF": db["C_qq_equivalent_aF"],
        "on_off": db,
        "ZZ_at_max_split_kHz": float(abs(ZZ_dr[k_off]) * 1e6),
        "ZZ_at_degeneracy_kHz": float(abs(ZZ_dr[k_on]) * 1e6),
        "pair_breaking_note": ("the drive cavity is the 145 GHz mode whose "
                              "pair-breaking window DEFINES the nu range used "
                              "here; using it as a bus adds no new tone and no "
                              "new pair-breaking exposure (qubit.json:"
                              "open_items.photon_assisted_pair_breaking still "
                              "applies unchanged)."),
    },
    "purcell_structural_floor": purcell_floor,
    "isolation_coherence_pareto": {
        "front": pareto,
        "off_state_cost": off_state_cost,
        "apex_law": {
            "holds": bool(pareto_apex_law),
            "exact_grid_test": apex_law_rows,
            "criterion": ("for every one of the 1681 parkings, moving the "
                          "INNER qubit to nu = -2.4 leaves min(T2_echo) "
                          "unchanged and does not reduce the split: zero "
                          "violations means the apex family weakly dominates"),
            "statement": ("because f01(nu) is EVEN about the apex, the split "
                          "depends only on (|nu1+2.4|, |nu2+2.4|) while the "
                          "coherence cost |D1| grows monotonically with each; "
                          "so for every reachable split the Pareto-optimal "
                          "parking puts ONE qubit exactly at the apex and pays "
                          "the whole coherence bill on the other. The evenness "
                          "flagged as a liability in allocation.degeneracy_note "
                          "is what makes the ON state (both at the apex) "
                          "EXACTLY degenerate with no fine tuning."),
        },
    },
    "parking_error_sensitivity": park_err,
    "N_ops_vs_x_qp": nops,
    "stray_direct_C_qq": stray,
    "measurement_conflict": meas,
    "addressability": addressing,
    "validation": val,
}

# ------------------------------------------------------------- the verdict --
e_ro = rb["budget"]
e_dr = db["budget"]
e_cap = nops["C_qq_at_dispersive_cap_0p29GHz"]["budget"]
doc["verdict"] = {
    "headline": (
        "The readout cavity IS a two-qubit bus and it replaces the "
        "unmanufacturable capacitor: J_ro = 5.0507 MHz, t_iSWAP = 98.996 ns, "
        "equivalent to a direct C_qq of 3.2353 aF -- 5.74x BELOW the 18.576 aF "
        "dispersive cap the sketch demanded, i.e. inside the budget with room "
        "to spare and with zero fab tolerance, because it is set by a cavity "
        "that has already been measured. What the bus does NOT provide is an "
        "off switch: the 0.2906 GHz nu window is the only knob and it cannot "
        "suppress the residual swap without parking a qubit where its own echo "
        "coherence collapses."),
    "readout_bus": {
        "grade": "COUPLER VIABLE; SWITCH NOT VIABLE",
        "coupler_verdict": ("J, dispersive headroom, ZZ and the per-gate error "
                            "budget are all acceptable"),
        "switch_verdict": ("the nu-window on/off knob floors the idle cost at "
                           f"{off_state_cost['readout_bus']['best_total']:.4f} "
                           "per 100 ns; a tunable coupler or a compiled "
                           "always-on scheme is required"),
        "off_state_cost": off_state_cost["readout_bus"],
        "on_state_is_protected": ("both islands at nu = -2.4: f01 degenerate "
                                  "EXACTLY by the evenness of the dome, and "
                                  "D1 = 0 makes the resonance condition "
                                  "first-order insensitive to gate-setting "
                                  "noise (swap error 0 at dnu_rms = 1e-3, vs "
                                  "0.2001 for a symmetric parking at "
                                  "nu = -2.3707)"),
        "J_MHz": rb["J_on_MHz"], "t_iSWAP_ns": rb["t_iSWAP_ns"],
        "equivalent_C_qq_aF": rb["C_qq_equivalent_aF"],
        "off_residual_swap": rb["P_off"],
        "on_off_ratio": rb["on_off_ratio"],
        "ZZ_off_kHz_exact": float(abs(zz_rows[i_edge_hi]["zz_exact_GHz"]) * 1e6),
        "incoherent_budget_at_xqp_1e-7_A_1e-4": e_ro,
        "N_ops_purcell_limited": purcell_floor["readout_bus"]["max_bus_depth_N_purcell"],
        "N_ops_qp_limited": nops["readout_bus"]["N_ops"],
        "binding_constraint": ("Purcell through the SAME cavity that provides "
                               "the coupling: t_g/T1_P = pi kappa_r/Delta_r "
                               "with g cancelled = "
                               f"{purcell_floor['readout_bus']['pi_kappa_over_Delta']:.3e}"),
    },
    "drive_bus": {
        "grade": "NOT VIABLE at the stored 20 um drive electrode",
        "why_not": ("t_iSWAP = 1.5799 us is longer than everything: the "
                    "READOUT cavity's own Purcell background (36.54 us) runs "
                    "during the drive-bus gate and alone costs 8.7e-2, and the "
                    "quasiparticle channel adds 5.7e-2 at x_qp = 1e-7. The "
                    "isolation is excellent (P_off = 4.70e-6) but the gate is "
                    "too slow to use it."),
        "off_state_cost": off_state_cost["drive_bus"],
        "J_MHz": db["J_on_MHz"], "t_iSWAP_ns": db["t_iSWAP_ns"],
        "equivalent_C_qq_aF": db["C_qq_equivalent_aF"],
        "off_residual_swap": db["P_off"],
        "incoherent_budget_at_xqp_1e-7_A_1e-4": e_dr,
        "N_ops_purcell_limited": purcell_floor["drive_bus"]["max_bus_depth_N_purcell"],
        "N_ops_qp_limited": nops["drive_bus"]["N_ops"],
        "binding_constraint": "quasiparticle T1 over the 1.6-2.0 us gate",
    },
    "drive_bus_upgraded_electrode": {
        "grade": ("LOWEST GATE ERROR ON CHIP, but it moves a stored photonics "
                  "input (d_eff_drive 20 -> 47.25 um) and needs a P2/P5 re-run"),
        "ZZ_kHz": drive_upgrade["optimum"]["ZZ_kHz"],
        "ZZ_cost": ("static ZZ scales as J^2 so the faster bus carries more of "
                    "it: 232.3 kHz at the maximum split vs 33.9 kHz for the "
                    "readout bus, i.e. 3.5x more conditional phase per gate"),
        "d_eff_um": drive_upgrade["optimum"]["d_eff_um"],
        "J_MHz": drive_upgrade["optimum"]["J_MHz"],
        "t_iSWAP_ns": drive_upgrade["optimum"]["t_iSWAP_ns"],
        "incoherent_budget_at_xqp_1e-7_A_1e-4":
            drive_upgrade["optimum"]["incoherent_budget_xqp_1e-7"],
        "gain_over_readout_bus": float(
            e_ro["total"]
            / drive_upgrade["optimum"]["incoherent_budget_xqp_1e-7"]["total"]),
        "caveat": drive_upgrade["caveats"],
    },
    "direct_C_qq_at_cap": {
        "C_qq_fF": cap029["C_qq_max_fF"], "t_iSWAP_ns": cap029["t_iSWAP_at_cap_ns"],
        "incoherent_budget_at_xqp_1e-7_A_1e-4": e_cap,
        "ratio_bus_over_direct": float(e_ro["total"] / e_cap["total"]),
        "why": ("every term in the budget is proportional to t_g, so the bus "
                "costs exactly the gate-time ratio J_cap/J_bus = "
                f"{cap029['J_at_cap_MHz'] / rb['J_on_MHz']:.2f}x more "
                "decoherence per gate than a perfect direct capacitor at its "
                "dispersive cap -- and buys away the entire fab tolerance"),
    },
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(rnd(doc), indent=1) + "\n", encoding="utf-8")

# ------------------------------------------------------------------ stdout --
print(f"grid {NGRID}x{NGRID} over nu in [{nu_edge_lo:.6f}, {nu_edge_hi:.6f}]")
print(f"f01 span {f01_nu.max()-f01_nu.min():.6f} GHz "
      f"(stored {F01_SPAN_GHZ:.6f})")
print(f"J_ro  = {rb['J_on_MHz']:.4f} MHz  t_iSWAP = {rb['t_iSWAP_ns']:.2f} ns "
      f"| equiv C_qq = {rb['C_qq_equivalent_aF']:.3f} aF")
print(f"J_dr  = {db['J_on_MHz']:.4f} MHz  t_iSWAP = {db['t_iSWAP_ns']:.1f} ns "
      f"| equiv C_qq = {db['C_qq_equivalent_aF']:.4f} aF")
print(f"OFF residual swap: readout {rb['P_off']:.3e} drive {db['P_off']:.3e}")
print(f"ZZ at max split (exact diag): readout "
      f"{abs(zz_rows[i_edge_hi]['zz_exact_GHz'])*1e6:.3f} kHz "
      f"(pure-exchange formula {abs(ZZ_ro[k_off])*1e6:.3f} kHz, "
      f"cavity factor {corr:.4f})")
print(f"combined bus J_total = {combined_bus['J_total_MHz']:.4f} MHz "
      f"(drive adds {combined_bus['drive_contribution_pct']:.2f}%)")
o = drive_upgrade["optimum"]
print(f"drive-electrode upgrade: d_eff {o['d_eff_um']:.2f} um -> "
      f"J {o['J_MHz']:.3f} MHz, t {o['t_iSWAP_ns']:.2f} ns, "
      f"T1P_comb {o['T1_Purcell_combined_us']:.2f} us, "
      f"budget {o['incoherent_budget_xqp_1e-7']['total']:.3e}")
print(f"Purcell floor pi*kappa/Delta: readout "
      f"{purcell_floor['readout_bus']['pi_kappa_over_Delta']:.4e} "
      f"drive {purcell_floor['drive_bus']['pi_kappa_over_Delta']:.4e}")
print(f"budget @ x_qp=1e-7,A=1e-4: readout {e_ro['total']:.3e} "
      f"drive {e_dr['total']:.3e} direct-cap {e_cap['total']:.3e}")
print(f"parked-qubit T2_echo at window edge: {T2_echo_nu[i_edge_hi]*1e6:.4f} us "
      f"= {T2_echo_nu[i_edge_hi]/ (rb['t_iSWAP_ns']*1e-9):.2f} gate times")
print(f"OFF-state idle cost per 100 ns: readout best "
      f"{off_state_cost['readout_bus']['best_total']:.4f} at split "
      f"{off_state_cost['readout_bus']['at_split_GHz']:.4f} GHz | drive best "
      f"{off_state_cost['drive_bus']['best_total']:.3e}")
print(f"apex law (1681 pairs): holds={pareto_apex_law} "
      f"split_viol={apex_law_rows['n_split_violations']} "
      f"T2_viol={apex_law_rows['n_T2_violations']}")
for k, v in val.items():
    if isinstance(v, dict) and "passed" in v:
        print(f"  {k}: {'PASS' if v['passed'] else 'FAIL'}")
print(f"wrote {OUT}")
