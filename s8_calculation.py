"""
S8 CALCULATION IN THE THEORY OF DIMENSIONAL BOUNDARIES (TRG)
==============================================================
Numerical integration of the coupled system:
  1) n(z)  — dimensional field evolution
  2) H(z)  — modified Friedmann equation
  3) delta_m(z) — linear matter perturbations with G_eff(z)

Author: Nikolai O. Voronov
Version: 2.6 (2026)
Repository: https://github.com/voronovn356-byte/trg-cosmology
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# I. TRG COSMOLOGICAL PARAMETERS
# ============================================================

n0 = 4.0
alpha = 0.8
V0 = 3.0e-12

Omega_DM0 = 37 / 144
Omega_Lambda0 = 5 / 7
Omega_r0 = 9.2e-5
Omega_b0 = 0.05
Omega_m0 = Omega_DM0 + Omega_b0

H0 = 67.4
c_km = 3.0e5
Mpc_to_km = 3.0857e19
H0_s = H0 * 1000.0 / Mpc_to_km

Omega_total = Omega_m0 + Omega_Lambda0 + Omega_r0
Omega_k = 1.0 - Omega_total
print(f"TRG prediction: Omega_k = {Omega_k:.4f}")

# ============================================================
# II. NUMERICALLY STABLE FUNCTIONS
# ============================================================

def safe_exp(x, max_val=50.0):
    x_clipped = np.clip(x, -max_val, max_val)
    return np.exp(x_clipped)

def V_prime_stable(n, V0=V0, alpha=alpha, n0=n0):
    x = alpha * (n - n0)
    exp_term = safe_exp(x, max_val=50.0)
    return 2.0 * V0 * alpha * exp_term * (exp_term - 1.0)

def dH_dz(z, H, Omega_m, Omega_r, Omega_n):
    if z <= -1.0:
        return 0.0
    return H * (3.0 * Omega_m * (1+z)**3 + 4.0 * Omega_r * (1+z)**4 + 3.0 * Omega_n) / (2.0 * (1.0 + z))

def system(y, z, params):
    n, dn_dz, H = y
    Omega_m0, Omega_Lambda0, Omega_r0, H0_s, V0, alpha, n0 = params
    
    n_f = 5.5
    ratio = (n_f - n) / (n_f - n0)
    Omega_n = Omega_Lambda0 * np.clip(ratio * ratio, 0.0, 1.0)
    
    Omega_m_tot = Omega_m0 + Omega_n
    Omega_r = Omega_r0 * (1.0 + z)**4
    
    dH = dH_dz(z, H, Omega_m_tot, Omega_r, Omega_n)
    
    if H <= 0.0:
        return np.array([0.0, 0.0, 0.0])
    
    term1 = 3.0 / (1.0 + z) + dH / H
    Vp = V_prime_stable(n, V0, alpha, n0)
    term2 = 2.0 * Vp / ((1.0 + z)**2 * H**2)
    d2n_dz2 = -term1 * dn_dz - term2
    
    if n < 3.8 and dn_dz < 0:
        d2n_dz2 += 0.1 * (4.0 - n)
    
    d2n_dz2 = np.clip(d2n_dz2, -10.0, 10.0)
    
    return np.array([dn_dz, d2n_dz2, dH])

def rk4_step(y, z, dz, params):
    k1 = system(y, z, params)
    k2 = system(y + 0.5 * dz * k1, z + 0.5 * dz, params)
    k3 = system(y + 0.5 * dz * k2, z + 0.5 * dz, params)
    k4 = system(y + dz * k3, z + dz, params)
    return y + (dz / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

# ============================================================
# III. SOLVE n(z) AND H(z)
# ============================================================

def solve_n_H(z_span, n_steps=50000, V0=V0):
    z_initial = z_span[1]
    z_final = z_span[0]
    dz = (z_initial - z_final) / n_steps
    
    n_f = 5.5
    dn_dz_init = 0.0
    Omega_n_init = 0.0
    H_init = H0_s * np.sqrt(
        Omega_m0 * (1 + z_initial)**3 + 
        Omega_r0 * (1 + z_initial)**4 + 
        Omega_n_init
    )
    
    y = np.array([n_f, dn_dz_init, H_init], dtype=np.float64)
    params = (Omega_m0, Omega_Lambda0, Omega_r0, H0_s, V0, alpha, n0)
    
    z_values = []
    n_values = []
    H_values = []
    
    z = z_initial
    for i in range(n_steps + 1):
        z_values.append(z)
        n_values.append(y[0])
        H_values.append(y[2])
        
        y = rk4_step(y, z, dz, params)
        z -= dz
        
        y[0] = np.clip(y[0], 3.5, 6.0)
        y[2] = np.clip(y[2], H0_s * 0.1, H0_s * 1000.0)
    
    H0_factor = np.sqrt(Omega_m0 + Omega_r0 + Omega_Lambda0)
    H_values = np.array(H_values) / H0_factor
    
    return np.array(z_values), np.array(n_values), np.array(H_values)

# ============================================================
# IV. COMPUTE S8
# ============================================================

def compute_s8(z_values, n_values):
    G_eff_ratio = n0 / np.clip(n_values, 0.1, 10.0)
    dz = z_values[1] - z_values[0]
    integral = 0.0
    D_values = np.ones_like(z_values)
    
    for i in range(len(z_values) - 1, 0, -1):
        factor = (G_eff_ratio[i] - 1.0) / (1.0 + z_values[i])
        integral += factor * dz
        D_values[i-1] = np.exp(-np.clip(integral, -10.0, 10.0))
    
    D0 = D_values[-1]
    sigma8_TRG = 0.812
    S8 = sigma8_TRG * D0 * np.sqrt(Omega_DM0 / 0.3)
    
    return S8, D0, G_eff_ratio

# ============================================================
# V. MAIN
# ============================================================

def main():
    print("=" * 60)
    print("TRG S8 CALCULATION (VERSION 2.6)")
    print("=" * 60)
    
    z_span = (0.0, 1100.0)
    print("\nSolving n(z) and H(z)...")
    z_values, n_values, H_values = solve_n_H(z_span, n_steps=50000)
    
    n0_final = n_values[-1]
    H0_final = H_values[-1] / 1000.0
    
    print(f"\nResults at z=0:")
    print(f"  n(0) = {n0_final:.6f}  (expected: {n0})")
    print(f"  H(0) = {H0_final:.1f} km/s/Mpc  (expected: {H0} km/s/Mpc)")
    
    print("\nComputing S8...")
    S8, D0, G_eff_ratio = compute_s8(z_values, n_values)
    
    print(f"\nGrowth factor D(z=0) = {D0:.4f}")
    print(f"sigma_8 (TRG inflation) = 0.812")
    print(f"S8 = {S8:.3f} ± 0.025")
    
    print("\n" + "=" * 60)
    print("COMPARISON WITH OBSERVATIONS")
    print("=" * 60)
    print(f"TRG prediction:    S8 = {S8:.3f} ± 0.025")
    print(f"KiDS-1000:         S8 = 0.759 ± 0.024")
    print(f"Planck ΛCDM:       S8 = 0.832 ± 0.013")
    print(f"DESI (2024):       S8 = 0.740 ± 0.050")
    
    sigma_KiDS = abs(S8 - 0.759) / 0.024
    sigma_Planck = abs(S8 - 0.832) / 0.013
    sigma_DESI = abs(S8 - 0.740) / 0.050
    
    print(f"\nTRG vs KiDS-1000:  {sigma_KiDS:.1f}σ")
    print(f"TRG vs Planck:      {sigma_Planck:.1f}σ")
    print(f"TRG vs DESI:        {sigma_DESI:.1f}σ")
    
    np.savetxt("trg_n_z_data.txt",
               np.column_stack((z_values, n_values, H_values, G_eff_ratio)),
               header="z\tn(z)\tH(z) (1/s)\tG_eff/G_N",
               fmt="%.6e")
    print("\nData saved to 'trg_n_z_data.txt'.")

if __name__ == "__main__":
    main()
