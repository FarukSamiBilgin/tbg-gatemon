"""
M6 reference — TBG gatemon with a photonic control plane.

Inputs locked from the completed modules:
  junction v0.3 : Ic = 300 nA  -> E_J/h = 149.0 GHz, E_J/k_B = 7.15 K
  material v0.3 : Delta0 = 0.304 meV, dome Tc(nu) = 2K*[1-((nu+2.4)/0.5)^2]+
  photonics     : f_cav = 145.0 GHz, Q_i(in-situ) = 1.461e4,
                  V_mode = 6.68e7 um^3, collection length d_eff = 20 um

Reference tasks:
  1. Exact charge-basis transmon spectrum vs E_J/E_C (user hand-calc 50 included)
  2. Charge dispersion -> design-point selection criterion
  3. Gate-noise sensitivity from the SC dome (the TBG-specific channel)
  4. Purcell + dispersive budget vs the MEASURED drive cavity; readout window scan
  5. Quasiparticle T1 and thermal occupation
Exports qubit_targets.json + figure.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

E = 1.602176634e-19
H = 6.62607015e-34
HBAR = H / (2 * np.pi)
KB = 1.380649e-23
EPS0 = 8.8541878128e-12
PHI0 = 2.067833848e-15

EJ_GHZ = PHI0 * 300e-9 / (2 * np.pi) / H / 1e9        # 149.00 GHz
DELTA0_MEV = 0.304
F_CAV = 145.0                                          # GHz
Q_I = 14611.7
V_MODE = 6.68e7 * 1e-18                                # m^3
EPS_C = 11.68
D_EFF_DRIVE = 20e-6                                    # m

# ---------------- exact transmon (charge basis) ------------------------------
def transmon(EJ, EC, ng=0.0, N=35, nlev=4):
    n = np.arange(-N, N + 1)
    Hm = np.diag(4.0 * EC * (n - ng) ** 2)
    off = -EJ / 2.0 * np.ones(2 * N)
    Hm += np.diag(off, 1) + np.diag(off, -1)
    ev = np.linalg.eigvalsh(Hm)[:nlev]
    return ev

def levels(EJ, EC):
    ev0 = transmon(EJ, EC, 0.0)
    ev5 = transmon(EJ, EC, 0.5)
    f01, f12 = ev0[1] - ev0[0], ev0[2] - ev0[1]
    eps1 = abs((ev5[1] - ev5[0]) - f01)                # charge dispersion of f01
    return f01, f12 - f01, eps1

def n01_element(EJ, EC, N=35):
    n = np.arange(-N, N + 1)
    Hm = np.diag(4.0 * EC * n ** 2)
    off = -EJ / 2.0 * np.ones(2 * N)
    Hm += np.diag(off, 1) + np.diag(off, -1)
    w, v = np.linalg.eigh(Hm)
    return abs(v[:, 0] @ (n * v[:, 1]))

# ---------------- 1) design scan ---------------------------------------------
print(f"E_J/h = {EJ_GHZ:.2f} GHz (v0.3 junction)")
ratios = [25, 50, 100, 150, 200, 300]
scan = {}
print("ratio   EC(GHz)  C(fF)   f01(GHz)  anharm(GHz)  eps1")
for r in ratios:
    EC = EJ_GHZ / r
    C = E ** 2 / (2 * H * EC * 1e9) * 1e15
    f01, anh, eps1 = levels(EJ_GHZ, EC)
    scan[r] = dict(EC_GHz=EC, C_fF=C, f01_GHz=f01, anharm_GHz=anh,
                   eps1_Hz=eps1 * 1e9)
    tag = " <- user hand-calc" if r == 50 else ""
    print(f"{r:5d}  {EC:7.3f}  {C:6.1f}  {f01:8.3f}  {anh:+10.4f}  "
          f"{eps1*1e9:9.3e} Hz{tag}")

# design criterion: eps1 < 10 kHz AND |anharm| >= 0.5 GHz -> pick smallest ratio
design_r = min(r for r in ratios
               if scan[r]["eps1_Hz"] < 1e4 and abs(scan[r]["anharm_GHz"]) >= 0.5)
D = scan[design_r]
EC_D = D["EC_GHz"]
n01 = n01_element(EJ_GHZ, EC_D)
# asymptotic checks
f01_asym = np.sqrt(8 * EJ_GHZ * EC_D) - EC_D
n01_asym = (EJ_GHZ / (8 * EC_D)) ** 0.25 / np.sqrt(2)
print(f"\nDESIGN LOCK: E_J/E_C = {design_r}, C = {D['C_fF']:.1f} fF, "
      f"f01 = {D['f01_GHz']:.3f} GHz (asym {f01_asym:.3f}), "
      f"anh = {D['anharm_GHz']:+.4f} GHz, eps1 = {D['eps1_Hz']:.2e} Hz")
print(f"n01 = {n01:.4f} (asym {n01_asym:.4f})")

# ---------------- 2) thermal occupation --------------------------------------
nth = {T: float(np.exp(-H * D["f01_GHz"] * 1e9 / (KB * T)))
       for T in (0.02, 0.1, 0.2)}
nth_al_100mK = np.exp(-H * 5e9 / (KB * 0.1))
print(f"n_th(100 mK) = {nth[0.1]:.2e}  [5 GHz Al transmon at 100 mK: "
      f"{nth_al_100mK:.2e}]")

# ---------------- 3) gate-noise sweet spot from the dome ---------------------
def Tc_of(nu):
    return 2.0 * max(0.0, 1.0 - ((nu + 2.4) / 0.5) ** 2)

def Delta_of(nu, T=0.1):
    Tc = Tc_of(nu)
    return 0.0 if Tc <= T else 1.764 * 0.08617 * Tc * np.tanh(
        1.74 * np.sqrt(Tc / T - 1.0))

def f01_of(nu):
    EJ = EJ_GHZ * Delta_of(nu) / DELTA0_MEV
    if EJ <= 0:
        return np.nan
    return levels(EJ, EC_D)[0]

nus = np.round(np.arange(-2.85, -1.949, 0.01), 3)
f01_nu = np.array([f01_of(nu) for nu in nus])
dnu = 1e-3
def D1(nu): return (f01_of(nu + dnu) - f01_of(nu - dnu)) / (2 * dnu)
def D2(nu): return (f01_of(nu + dnu) - 2 * f01_of(nu) + f01_of(nu - dnu)) / dnu ** 2
D1_apex, D2_apex = D1(-2.4), D2(-2.4)
D1_off = D1(-2.5)
print(f"sweet spot: D1(-2.4) = {D1_apex:+.4f} GHz/nu (should be ~0), "
      f"D2(-2.4) = {D2_apex:+.1f} GHz/nu^2 ; D1(-2.5) = {D1_off:+.2f} GHz/nu")
# quasistatic dephasing estimates
def Tphi_apex(dnu_rms): return 1.0 / (np.pi * abs(D2_apex) * 1e9 * dnu_rms ** 2)
def Tphi_off(dnu_rms): return 1.0 / (2 * np.pi * abs(D1_off) * 1e9 * dnu_rms)
for dn in (1e-4, 1e-3):
    print(f"  dnu_rms={dn:.0e}: T_phi(apex) ~ {Tphi_apex(dn)*1e6:10.1f} us | "
          f"T_phi(nu=-2.5) ~ {Tphi_off(dn)*1e6:8.3f} us")
f01_edge = f01_of(-2.8)
print(f"gate tunability: f01({-2.4})={D['f01_GHz']:.2f} -> f01(-2.8)="
      f"{f01_edge:.2f} GHz")

# ---------------- 4) cavity budget: drive cavity (measured) ------------------
Ezpf = np.sqrt(HBAR * 2 * np.pi * F_CAV * 1e9 / (2 * EPS0 * EPS_C * V_MODE))
Vzpf = Ezpf * D_EFF_DRIVE
g_drive = E * Vzpf * n01 / H / 1e9                       # GHz (convention flagged)
Dq = F_CAV - D["f01_GHz"]
alpha = D["anharm_GHz"]
chi_drive = g_drive ** 2 * alpha / (Dq * (Dq + alpha))   # GHz
kappa_drive = F_CAV / Q_I                                # GHz
Gp_drive = (g_drive / Dq) ** 2 * (2 * np.pi * F_CAV * 1e9 / Q_I)
print(f"\ndrive cavity (measured): E_zpf={Ezpf*1e3:.1f} mV/m, "
      f"V_zpf={Vzpf*1e6:.2f} uV, g/h={g_drive*1e3:.0f} MHz")
print(f"  chi = {chi_drive*1e6:+.0f} kHz, kappa = {kappa_drive*1e3:.1f} MHz, "
      f"chi/kappa = {abs(chi_drive)/kappa_drive:.4f} -> readout via drive "
      f"cavity NOT viable")
print(f"  Purcell(drive): T1 = {1/Gp_drive*1e6:.0f} us (not limiting)")

# ---------------- readout-cavity window scan ---------------------------------
rows = []
for Dr in np.arange(4.0, 16.01, 1.0):                    # detuning GHz
    fr = D["f01_GHz"] + Dr
    Vr = V_MODE * (F_CAV / fr) ** 3                      # scaled PhC cavity
    Ez_r = np.sqrt(HBAR * 2 * np.pi * fr * 1e9 / (2 * EPS0 * EPS_C * Vr))
    for de in (100e-6, 200e-6):
        g = E * Ez_r * de * n01 / H / 1e9
        if (g / Dr) ** 2 > 0.01:
            continue
        chi = g ** 2 * alpha / (Dr * (Dr + alpha))
        for kap_MHz in (1.0, 3.0):
            kap = kap_MHz * 1e-3
            T1p = 1.0 / ((g / Dr) ** 2 * 2 * np.pi * kap * 1e9)
            if abs(chi) / kap >= 1.0 and T1p >= 30e-6:
                rows.append(dict(Delta_r_GHz=Dr, f_r_GHz=fr, d_eff_um=de*1e6,
                                 kappa_MHz=kap_MHz, g_MHz=g*1e3,
                                 chi_MHz=chi*1e3, chi_over_kappa=abs(chi)/kap,
                                 T1_Purcell_us=T1p*1e6))
print(f"readout window: {len(rows)} feasible points "
      f"(chi/kappa>=1, T1_P>=30us, (g/D)^2<=0.01)")
if rows:
    best = max(rows, key=lambda r: min(r["chi_over_kappa"], r["T1_Purcell_us"]/30))
    print("  sample:", {k: (round(v, 2) if isinstance(v, float) else v)
                        for k, v in best.items()})

# ---------------- 5) quasiparticle T1 ----------------------------------------
wq = 2 * np.pi * D["f01_GHz"] * 1e9
Delta_J = DELTA0_MEV * 1e-3 * E
pref = (wq / np.pi) * np.sqrt(2 * Delta_J / (HBAR * wq))
qp = {x: 1.0 / (x * pref) for x in (1e-8, 1e-7, 1e-6, 1e-5)}
print("quasiparticle T1: " + ", ".join(f"x={x:.0e}: {t*1e6:.1f} us"
                                       for x, t in qp.items()))
print(f"  (hbar*w_q/Delta = {HBAR*wq/Delta_J:.2f} -- formula marginal, "
      f"agent recomputes with full Catelani kernel)")

# ---------------- export ------------------------------------------------------
out = dict(
    inputs=dict(EJ_GHz=EJ_GHZ, Delta0_meV=DELTA0_MEV, f_cav_GHz=F_CAV,
                Q_i=Q_I, V_mode_m3=V_MODE, d_eff_drive_um=20.0),
    scan={str(r): scan[r] for r in ratios},
    design=dict(EJ_over_EC=design_r, EC_GHz=EC_D, C_fF=D["C_fF"],
                f01_GHz=D["f01_GHz"], anharm_GHz=D["anharm_GHz"],
                eps1_Hz=D["eps1_Hz"], n01=n01),
    thermal=dict(n_th=nth, n_th_Al5GHz_100mK=float(nth_al_100mK)),
    gate_noise=dict(D1_apex_GHz=D1_apex, D2_apex_GHz=D2_apex,
                    D1_off_nu_m2p5_GHz=D1_off,
                    Tphi_apex_us={f"{d:.0e}": Tphi_apex(d) * 1e6
                                  for d in (1e-4, 1e-3)},
                    Tphi_off_us={f"{d:.0e}": Tphi_off(d) * 1e6
                                 for d in (1e-4, 1e-3)},
                    f01_tunability_GHz=[float(f01_edge), D["f01_GHz"]]),
    drive_cavity=dict(g_MHz=g_drive * 1e3, chi_kHz=chi_drive * 1e6,
                      kappa_MHz=kappa_drive * 1e3,
                      chi_over_kappa=abs(chi_drive) / kappa_drive,
                      T1_Purcell_us=1 / Gp_drive * 1e6,
                      g_convention="g = e*V_zpf*n01/h; lumped-circuit pin = agent task"),
    readout_window=rows,
    quasiparticle=dict(T1_us={f"{x:.0e}": t * 1e6 for x, t in qp.items()},
                       marginality_flag=float(HBAR * wq / Delta_J)),
)
with open("/home/claude/qubit_targets.json", "w") as fh:
    json.dump(out, fh, indent=1)

# ---------------- figure ------------------------------------------------------
plt.rcParams.update({"font.size": 10.5, "axes.grid": True, "grid.alpha": 0.3})
fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(12.8, 4.2))

rs = np.array(ratios, float)
a1.plot(rs, [scan[r]["f01_GHz"] for r in ratios], "o-", c="C0", label="$f_{01}$ (GHz)")
a1.plot(rs, [abs(scan[r]["anharm_GHz"]) for r in ratios], "s-", c="C1",
        label=r"$|\alpha|$ (GHz)")
a1b = a1.twinx()
a1b.semilogy(rs, [scan[r]["eps1_Hz"] for r in ratios], "^--", c="C3",
             label=r"$\epsilon_1$ (Hz)")
a1b.axhline(1e4, ls=":", c="C3", lw=0.8)
a1.axvline(design_r, c="k", ls="--", lw=0.8)
a1.text(design_r + 5, 40, f"design\n$E_J/E_C$={design_r}", fontsize=9)
a1.set_xlabel(r"$E_J/E_C$"); a1.set_ylabel("GHz")
a1b.set_ylabel(r"charge dispersion $\epsilon_1$ (Hz)", color="C3")
a1.set_title("TBG gatemon design scan")
a1.legend(loc="center right", fontsize=8)

a2.plot(nus, f01_nu, c="C0", lw=1.6)
a2.axvline(-2.4, c="k", ls="--", lw=0.8)
a2.annotate("apex = double sweet spot\n$\\partial f_{01}/\\partial\\nu=0$, max $E_J$",
            xy=(-2.4, D["f01_GHz"]), xytext=(-2.32, 24),
            arrowprops=dict(arrowstyle="->", lw=0.8), fontsize=9)
a2.set_xlabel(r"filling $\nu$"); a2.set_ylabel(r"$f_{01}$ (GHz)")
a2.set_title("gate-tunable spectrum on the SC dome")

xs = list(qp.keys())
a3.loglog(xs, [qp[x] * 1e6 for x in xs], "o-", c="C2", label="quasiparticle")
a3.axhline(1 / Gp_drive * 1e6, c="C0", ls="--",
           label=f"Purcell, drive cavity ({1/Gp_drive*1e6:.0f} µs)")
if rows:
    a3.axhline(best["T1_Purcell_us"], c="C1", ls=":",
               label=f"Purcell, readout pt ({best['T1_Purcell_us']:.0f} µs)")
a3.set_xlabel(r"$x_{qp}$"); a3.set_ylabel(r"$T_1$ (µs)")
a3.set_title("relaxation budget")
a3.legend(fontsize=8)

fig.suptitle("M6 reference: TBG gatemon with a photonic control plane")
fig.tight_layout()
fig.savefig("/home/claude/fig_m6_qubit.png", dpi=160)
print("done")
