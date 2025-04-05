import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------
# 1. Données d'entrée et Constantes
# -----------------------------------------------------
P_kW = 8.5
rpm = 541
d_mm = 30
rB_mm = 50 # Rayon Engrenage B
rC_mm = 75 # Rayon Engrenage C
L_AB_mm = 200
L_BC_mm = 400
L_CD_mm = 350
pressure_angle_deg = 20.0 # Angle de pression supposé

# --- Conversion en unités SI ---
P = P_kW * 1000
omega = rpm * (2 * np.pi / 60)
d = d_mm / 1000
rB = rB_mm / 1000
rC = rC_mm / 1000
L_AB = L_AB_mm / 1000
L_BC = L_BC_mm / 1000
L_CD = L_CD_mm / 1000
x_A = 0.0
x_B = L_AB
x_C = x_B + L_BC
x_D = x_C + L_CD
L_total = x_D
pressure_angle_rad = np.deg2rad(pressure_angle_deg)

# -----------------------------------------------------
# 2. Calculs Préparatoires
#    *** INTERPRÉTATION: Engrenages droits standards en B et C ***
# -----------------------------------------------------
# Couple transmis
T = P / omega

# --- Forces sur l'Engrenage B (Entrée Moteur) ---
# Force tangentielle (crée le couple T)
Ft_B = T / rB  # Supposée en +y direction
# Force radiale (due à l'angle de pression)
Fr_B = abs(Ft_B) * np.tan(pressure_angle_rad) # Supposée en +z direction (séparation)

# --- Forces sur l'Engrenage C (Sortie Accessoire) ---
# Force tangentielle (s'oppose au couple T)
Ft_C = - T / rC # Supposée en -y direction
# Force radiale (due à l'angle de pression)
Fr_C = - abs(Ft_C) * np.tan(pressure_angle_rad) # Supposée en -z direction (séparation)


print("--- Calcul basé sur: Engrenages droits standards en B et C ---")
print(f"Calculs Préparatoires:")
print(f"  Couple (T): {T:.2f} Nm")
print(f"Forces en B: Ft_B (y) = {Ft_B:.2f} N, Fr_B (z) = {Fr_B:.2f} N")
print(f"Forces en C: Ft_C (y) = {Ft_C:.2f} N, Fr_C (z) = {Fr_C:.2f} N")
print("-" * 30)

# -----------------------------------------------------
# 3. Calcul des Réactions aux Paliers (A et D)
# -----------------------------------------------------
# --- Plan XY (Forces en Y => Réactions Ay, Dy; Moments Mz) ---
# Ay + Dy + Ft_B + Ft_C = 0
# (Ft_B * x_B) + (Ft_C * x_C) + (Dy * x_D) = 0
A_mat_yz = np.array([[1, 1], [0, x_D]])
B_vec_yz = np.array([[-Ft_B - Ft_C],
                   [-(Ft_B * x_B) - (Ft_C * x_C)]])
sol_yz = np.linalg.solve(A_mat_yz, B_vec_yz)
Ay = sol_yz[0, 0]
Dy = sol_yz[1, 0]

# --- Plan XZ (Forces en Z => Réactions Az, Dz; Moments My) ---
# Az + Dz + Fr_B + Fr_C = 0
# -(Fr_B * x_B) - (Fr_C * x_C) - (Dz * x_D) = 0
A_mat_zy = np.array([[1, 1], [0, x_D]])
B_vec_zy = np.array([[-Fr_B - Fr_C],
                   [(Fr_B * x_B) + (Fr_C * x_C)]])
sol_zy = np.linalg.solve(A_mat_zy, B_vec_zy)
Az = sol_zy[0, 0]
Dz = sol_zy[1, 0]

print("Réactions aux paliers:")
print(f"  Ay: {Ay:.2f} N, Dy: {Dy:.2f} N")
print(f"  Az: {Az:.2f} N, Dz: {Dz:.2f} N")
print(f"  Vérif Equilibre Y: {Ay + Ft_B + Ft_C + Dy:.2e} N (~0)")
print(f"  Vérif Equilibre Z: {Az + Fr_B + Fr_C + Dz:.2e} N (~0)")
print("-" * 30)

# -----------------------------------------------------
# 4. Calcul des Efforts Internes le long de l'arbre
# -----------------------------------------------------
x_vals = np.linspace(0, L_total, 500)
Vy = np.zeros_like(x_vals)
Vz = np.zeros_like(x_vals)
Mz = np.zeros_like(x_vals)
My = np.zeros_like(x_vals)
Tx = np.zeros_like(x_vals)

for i, x in enumerate(x_vals):
    # Efforts Tranchants
    Vy[i] = Ay
    Vz[i] = Az
    if x >= x_B:
        Vy[i] += Ft_B # Force tangentielle en B
        Vz[i] += Fr_B # Force radiale en B
    if x >= x_C:
        Vy[i] += Ft_C # Force tangentielle en C
        Vz[i] += Fr_C # Force radiale en C

    # Moments Fléchissants
    Mz[i] = Ay * x
    if x >= x_B: Mz[i] += Ft_B * (x - x_B)
    if x >= x_C: Mz[i] += Ft_C * (x - x_C)

    My[i] = - (Az * x)
    if x >= x_B: My[i] -= Fr_B * (x - x_B)
    if x >= x_C: My[i] -= Fr_C * (x - x_C)

    # Couple de Torsion
    if x > x_B and x <= x_C: Tx[i] = T
    else: Tx[i] = 0

M_res = np.sqrt(My**2 + Mz**2)

# -----------------------------------------------------
# 5. Tracer les Diagrammes des Efforts Internes
# -----------------------------------------------------
plt.figure(figsize=(12, 10))

# Vy
plt.subplot(3, 2, 1); plt.plot(x_vals, Vy, label='Vy(x)'); plt.title('Effort Tranchant $V_y$'); plt.xlabel('Position x (m)'); plt.ylabel('$V_y$ (N)'); plt.grid(True)
plt.axvline(x_A, color='k', linestyle='--', label='A'); plt.axvline(x_B, color='r', linestyle='--', label='B (Engr.)'); plt.axvline(x_C, color='g', linestyle='--', label='C (Engr.)'); plt.axvline(x_D, color='k', linestyle='--'); plt.legend()
# Vz
plt.subplot(3, 2, 2); plt.plot(x_vals, Vz, label='Vz(x)'); plt.title('Effort Tranchant $V_z$'); plt.xlabel('Position x (m)'); plt.ylabel('$V_z$ (N)'); plt.grid(True)
plt.axvline(x_A, color='k', linestyle='--'); plt.axvline(x_B, color='r', linestyle='--'); plt.axvline(x_C, color='g', linestyle='--'); plt.axvline(x_D, color='k', linestyle='--'); plt.legend()
# Mz
plt.subplot(3, 2, 3); plt.plot(x_vals, Mz, label='Mz(x)'); plt.title('Moment Fléchissant $M_z$'); plt.xlabel('Position x (m)'); plt.ylabel('$M_z$ (Nm)'); plt.grid(True)
plt.axvline(x_A, color='k', linestyle='--'); plt.axvline(x_B, color='r', linestyle='--'); plt.axvline(x_C, color='g', linestyle='--'); plt.axvline(x_D, color='k', linestyle='--'); plt.legend()
# My
plt.subplot(3, 2, 4); plt.plot(x_vals, My, label='My(x)'); plt.title('Moment Fléchissant $M_y$'); plt.xlabel('Position x (m)'); plt.ylabel('$M_y$ (Nm)'); plt.grid(True)
plt.axvline(x_A, color='k', linestyle='--'); plt.axvline(x_B, color='r', linestyle='--'); plt.axvline(x_C, color='g', linestyle='--'); plt.axvline(x_D, color='k', linestyle='--'); plt.legend()
# Tx
plt.subplot(3, 2, 5); plt.plot(x_vals, Tx, label='Tx(x)'); plt.title('Couple de Torsion $T_x$'); plt.xlabel('Position x (m)'); plt.ylabel('$T_x$ (Nm)'); plt.ylim(min(0, T*(-0.1)), max(0, T*1.1)); plt.grid(True)
plt.axvline(x_A, color='k', linestyle='--'); plt.axvline(x_B, color='r', linestyle='--'); plt.axvline(x_C, color='g', linestyle='--'); plt.axvline(x_D, color='k', linestyle='--'); plt.legend()
# M_res
plt.subplot(3, 2, 6); plt.plot(x_vals, M_res, label='$M_{res}(x)$', color='purple'); plt.title('Moment Fléchissant Résultant $M_{res}$'); plt.xlabel('Position x (m)'); plt.ylabel('$M_{res}$ (Nm)'); plt.grid(True)
plt.axvline(x_A, color='k', linestyle='--'); plt.axvline(x_B, color='r', linestyle='--'); plt.axvline(x_C, color='g', linestyle='--'); plt.axvline(x_D, color='k', linestyle='--');

# Trouver et marquer le point critique (max M_res où T > 0)
idx_B = np.searchsorted(x_vals, x_B)
idx_C = np.searchsorted(x_vals, x_C)
x_crit, M_crit = x_B, M_res[idx_B]
idx_crit_global = idx_B
if idx_B < idx_C:
    M_res_BC = M_res[idx_B:idx_C+1]
    x_vals_BC = x_vals[idx_B:idx_C+1]
    Tx_BC = Tx[idx_B:idx_C+1]
    valid_indices_local = np.where(Tx_BC > 1e-9)[0]
    if len(valid_indices_local) > 0:
        M_res_BC_T_nonzero = M_res_BC[valid_indices_local]
        idx_max_local_valid = np.argmax(M_res_BC_T_nonzero)
        idx_crit_local = valid_indices_local[idx_max_local_valid]
        idx_crit_global = idx_B + idx_crit_local
        x_crit = x_vals[idx_crit_global]
        M_crit = M_res[idx_crit_global]
        plt.plot(x_crit, M_crit, 'ro', markersize=8, label=f'Max M in BC (T>0) ({M_crit:.1f} Nm)')
    else:
        print("Avertissement: Pas de zone avec T>0 trouvée entre B et C.")
        plt.plot(x_crit, M_crit, 'ro', markersize=8, label=f'M at B ({M_crit:.1f} Nm)')
else:
     print("Avertissement: Problème indices B>=C.")
     plt.plot(x_crit, M_crit, 'ro', markersize=8, label=f'M at B ({M_crit:.1f} Nm)')
plt.legend(); plt.tight_layout(); plt.show()

# -----------------------------------------------------
# 6. Identification du Point Critique et Calcul des Contraintes
# -----------------------------------------------------
My_crit = My[idx_crit_global]
Mz_crit = Mz[idx_crit_global]
T_crit = Tx[idx_crit_global]

print(f"Point Critique (max M résultant entre B et C où T>0):")
print(f"  Position x_crit: {x_crit:.3f} m")
print(f"  Moment Résultant M_crit: {M_crit:.2f} Nm (My={My_crit:.2f}, Mz={Mz_crit:.2f})")
print(f"  Couple de Torsion T_crit: {T_crit:.2f} Nm")
if abs(T_crit) < 1e-9: print("  ATTENTION: Torsion nulle au point critique identifié !")
print("-" * 30)

c = d / 2; I = np.pi * d**4 / 64; J = np.pi * d**4 / 32
sigma_x_mag = M_crit * c / I # Magnitude max contrainte normale due à M_res
tau_torsion = T_crit * c / J # Contrainte cisaillement due à T

print("Contraintes au Point Critique (en surface):")
print(f"  Sigma_x (magnitude max due à M_res): {sigma_x_mag / 1e6:.2f} MPa")
print(f"  Tau_max (due à Torsion): {tau_torsion / 1e6:.2f} MPa")
print("-" * 30)

# -----------------------------------------------------
# 7. Calcul des Contraintes Principales
# -----------------------------------------------------
sigma_x_for_principal = sigma_x_mag # Utiliser la magnitude pour le calcul des principales

if abs(tau_torsion) < 1e-9 * 1e6:
     sigma_1 = sigma_x_for_principal
     sigma_2 = 0.0
     tau_max_inplane = abs(sigma_x_for_principal / 2)
     tau_max_abs = abs(sigma_1 / 2)
     print("Note: Calcul des contraintes principales simplifié car tau_torsion ≈ 0.")
else:
     sigma_1 = (sigma_x_for_principal / 2) + np.sqrt((sigma_x_for_principal / 2)**2 + tau_torsion**2)
     sigma_2 = (sigma_x_for_principal / 2) - np.sqrt((sigma_x_for_principal / 2)**2 + tau_torsion**2)
     tau_max_inplane = np.sqrt((sigma_x_for_principal / 2)**2 + tau_torsion**2)
     tau_max_abs = max(abs(sigma_1 / 2), abs(sigma_2 / 2), tau_max_inplane)

print("Contraintes Principales au Point Critique:")
print(f"  Sigma_1: {sigma_1 / 1e6:.2f} MPa")
print(f"  Sigma_2: {sigma_2 / 1e6:.2f} MPa")
print(f"  Sigma_3: 0.00 MPa")
print(f"  Tau_max (dans le plan): {tau_max_inplane / 1e6:.2f} MPa")
print(f"  Tau_max (absolu): {tau_max_abs / 1e6:.2f} MPa")
print("-" * 30)