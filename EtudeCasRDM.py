import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------
# 1. Données d'entrée et Constantes
# -----------------------------------------------------
P_kW = 8.5  # Puissance en kW
rpm = 541   # Vitesse de rotation en tr/min
d_mm = 30   # Diamètre de l'arbre en mm
rB_mm = 50  # Rayon de l'engrenage B en mm
rC_mm = 75  # Rayon de l'engrenage C en mm

# Longueurs des segments en mm
L_AB_mm = 200
L_BC_mm = 400
L_CD_mm = 350

# --- Conversion en unités SI (m, N, rad/s, Pa) ---
P = P_kW * 1000  # W
omega = rpm * (2 * np.pi / 60)  # rad/s
d = d_mm / 1000  # m
rB = rB_mm / 1000 # m
rC = rC_mm / 1000 # m
L_AB = L_AB_mm / 1000 # m
L_BC = L_BC_mm / 1000 # m
L_CD = L_CD_mm / 1000 # m

# Positions des points clés (m)
x_A = 0.0
x_B = L_AB
x_C = x_B + L_BC
x_D = x_C + L_CD
L_total = x_D

# -----------------------------------------------------
# 2. Calculs Préparatoires
#    *** INTERPRÉTATION : Fy et Fz (labels) sont TANGENTIELLES ***
#    *** PAS DE FORCES RADIALES CONSIDÉRÉES ***
# -----------------------------------------------------
# Couple
T = P / omega

# --- Forces (Toutes tangentielles, dans le plan XY) ---
# Force tangentielle en B (correspond au label "Fy")
Fy_tangent_B = T / rB  # (+y direction, entrée moteur)

# Force tangentielle en C (correspond au label "Fz", mais agit en Y)
# Doit s'opposer au couple moteur
Fz_tangent_C = - T / rC # (-y direction, sortie accessoire)

# Forces radiales (supposées nulles selon cette interprétation)
Fr_radial_B = 0.0
Fr_radial_C = 0.0

print("--- ATTENTION: Calcul basé sur l'interprétation Fy et Fz sont TANGENTIELLES ---")
print("---            Implique AUCUNE force radiale ni flexion M_y              ---")
print(f"Calculs Préparatoires:")
print(f"  Vitesse angulaire (omega): {omega:.2f} rad/s")
print(f"  Couple (T): {T:.2f} Nm")
print(f"Forces Tangentielles:")
print(f"  Force en B (y): Fy_tangent_B = {Fy_tangent_B:.2f} N")
print(f"  Force en C (y): Fz_tangent_C = {Fz_tangent_C:.2f} N")
print("-" * 30)

# -----------------------------------------------------
# 3. Calcul des Réactions aux Paliers (A et D)
# -----------------------------------------------------
# --- Plan XY (Forces en Y, Moments autour de Z) ---
# Equations:
# Ay + Dy + Fy_tangent_B + Fz_tangent_C = 0
# (Fy_tangent_B * x_B) + (Fz_tangent_C * x_C) + (Dy * x_D) = 0  (Moment / A)

# Système matriciel: A_mat * [Ay, Dy]' = B_mat
A_mat_yz = np.array([[1, 1],
                   [0, x_D]])
B_vec_yz = np.array([[-Fy_tangent_B - Fz_tangent_C],
                   [-(Fy_tangent_B * x_B) - (Fz_tangent_C * x_C)]])

# Résolution
sol_yz = np.linalg.solve(A_mat_yz, B_vec_yz)
Ay = sol_yz[0, 0]
Dy = sol_yz[1, 0]

# --- Plan XZ (Forces en Z = 0, Moments autour de Y = 0) ---
# Puisqu'il n'y a pas de forces en Z, les réactions Az et Dz sont nulles.
Az = 0.0
Dz = 0.0

print("Réactions aux paliers:")
print(f"  Ay: {Ay:.2f} N, Dy: {Dy:.2f} N")
print(f"  Az: {Az:.2f} N, Dz: {Dz:.2f} N")

# Vérification de l'équilibre (approximatif à cause des erreurs numériques)
print(f"  Vérif Equilibre Y: Ay+Fy_tangent_B+Fz_tangent_C+Dy = {Ay + Fy_tangent_B + Fz_tangent_C + Dy:.2e} N (~0)")
print(f"  Vérif Equilibre Z: Az+Fr_radial_B+Fr_radial_C+Dz = {Az + Fr_radial_B + Fr_radial_C + Dz:.2e} N (~0)")
print("-" * 30)

# -----------------------------------------------------
# 4. Calcul des Efforts Internes le long de l'arbre
# -----------------------------------------------------
# Discrétisation de l'arbre
x_vals = np.linspace(0, L_total, 500)
Vy = np.zeros_like(x_vals)
Vz = np.zeros_like(x_vals) # Sera toujours 0
Mz = np.zeros_like(x_vals)
My = np.zeros_like(x_vals) # Sera toujours 0
Tx = np.zeros_like(x_vals)

for i, x in enumerate(x_vals):
    # --- Efforts Tranchants ---
    Vy[i] = Ay
    Vz[i] = Az # = 0
    if x >= x_B:
        Vy[i] += Fy_tangent_B
        # Vz[i] += Fr_radial_B (=0)
    if x >= x_C:
        Vy[i] += Fz_tangent_C
        # Vz[i] += Fr_radial_C (=0)

    # --- Moments Fléchissants ---
    # Mz = Somme des moments des forces Y autour de z
    Mz[i] = Ay * x
    if x >= x_B:
        Mz[i] += Fy_tangent_B * (x - x_B)
    if x >= x_C:
        Mz[i] += Fz_tangent_C * (x - x_C)

    # My = 0 car pas de forces en Z
    My[i] = 0.0

    # --- Couple de Torsion ---
    if x > x_B and x <= x_C:
        Tx[i] = T
    else:
        Tx[i] = 0 # Nul avant B et après C

# Moment fléchissant résultant (M_res = sqrt(My^2 + Mz^2) = sqrt(0 + Mz^2) = abs(Mz))
M_res = np.abs(Mz) # Simplifié car My = 0

# -----------------------------------------------------
# 5. Tracer les Diagrammes des Efforts Internes
# -----------------------------------------------------
plt.figure(figsize=(12, 10))

# --- Vy(x) ---
plt.subplot(3, 2, 1)
plt.plot(x_vals, Vy, label='Vy(x)')
plt.title('Effort Tranchant $V_y$') # Utilisation de LaTeX pour l'indice
plt.xlabel('Position x (m)')      # Label x corrigé
plt.ylabel('$V_y$ (N)')             # Label y corrigé
plt.grid(True)
plt.axvline(x_A, color='k', linestyle='--', label='A')
plt.axvline(x_B, color='r', linestyle='--', label='B (Fy Tang.)')
plt.axvline(x_C, color='g', linestyle='--', label='C (Fz Tang.)')
plt.axvline(x_D, color='k', linestyle='--')
plt.legend()

# --- Vz(x) ---
plt.subplot(3, 2, 2)
plt.plot(x_vals, Vz, label='Vz(x)')
plt.title('Effort Tranchant $V_z$ (Nul)') # Utilisation de LaTeX pour l'indice
plt.xlabel('Position x (m)')      # Label x corrigé
plt.ylabel('$V_z$ (N)')             # Label y corrigé
plt.ylim(-1, 1) # Force l'échelle autour de 0 car Vz=0
plt.grid(True)
plt.axvline(x_A, color='k', linestyle='--')
plt.axvline(x_B, color='r', linestyle='--')
plt.axvline(x_C, color='g', linestyle='--')
plt.axvline(x_D, color='k', linestyle='--')
plt.legend()

# --- Mz(x) ---
plt.subplot(3, 2, 3)
plt.plot(x_vals, Mz, label='Mz(x)')
plt.title('Moment Fléchissant $M_z$') # Utilisation de LaTeX pour l'indice
plt.xlabel('Position x (m)')      # Label x corrigé
plt.ylabel('$M_z$ (Nm)')            # Label y corrigé
plt.grid(True)
plt.axvline(x_A, color='k', linestyle='--')
plt.axvline(x_B, color='r', linestyle='--')
plt.axvline(x_C, color='g', linestyle='--')
plt.axvline(x_D, color='k', linestyle='--')
plt.legend()

# --- My(x) ---
plt.subplot(3, 2, 4)
plt.plot(x_vals, My, label='My(x)')
plt.title('Moment Fléchissant $M_y$ (Nul)') # Utilisation de LaTeX pour l'indice
plt.xlabel('Position x (m)')      # Label x corrigé
plt.ylabel('$M_y$ (Nm)')            # Label y corrigé
plt.ylim(-1, 1) # Force l'échelle autour de 0 car My=0
plt.grid(True)
plt.axvline(x_A, color='k', linestyle='--')
plt.axvline(x_B, color='r', linestyle='--')
plt.axvline(x_C, color='g', linestyle='--')
plt.axvline(x_D, color='k', linestyle='--')
plt.legend()

# --- Tx(x) ---
plt.subplot(3, 2, 5)
plt.plot(x_vals, Tx, label='Tx(x)')
plt.title('Couple de Torsion $T_x$') # Utilisation de LaTeX pour l'indice
plt.xlabel('Position x (m)')      # Label x corrigé
plt.ylabel('$T_x$ (Nm)')            # Label y corrigé
plt.ylim(min(0, T*(-0.1)), max(0, T*1.1)) # Ajuste limite y
plt.grid(True)
plt.axvline(x_A, color='k', linestyle='--')
plt.axvline(x_B, color='r', linestyle='--')
plt.axvline(x_C, color='g', linestyle='--')
plt.axvline(x_D, color='k', linestyle='--')
plt.legend()

# --- M_res(x) ---
plt.subplot(3, 2, 6)
plt.plot(x_vals, M_res, label='$M_{res}(x) = |M_z(x)|$', color='purple') # Légende corrigée avec LaTeX
plt.title('Moment Fléchissant Résultant $M_{res}$') # Utilisation de LaTeX pour l'indice
plt.xlabel('Position x (m)')      # Label x corrigé
plt.ylabel('$M_{res}$ (Nm)')         # Label y corrigé
plt.grid(True)
plt.axvline(x_A, color='k', linestyle='--')
plt.axvline(x_B, color='r', linestyle='--')
plt.axvline(x_C, color='g', linestyle='--')
plt.axvline(x_D, color='k', linestyle='--')

# Marquer le max de M (=|Mz|) dans la zone B-C où T>0
idx_B = np.searchsorted(x_vals, x_B)
idx_C = np.searchsorted(x_vals, x_C)

x_crit = x_B # Initialisation
M_crit = M_res[idx_B] # Initialisation
idx_crit_global = idx_B # Initialisation

if idx_B < idx_C:
    M_res_BC = M_res[idx_B:idx_C+1]
    x_vals_BC = x_vals[idx_B:idx_C+1]
    if len(M_res_BC) > 0:
        idx_crit_local = np.argmax(M_res_BC)
        idx_crit_global_temp = idx_B + idx_crit_local
        # Vérifier si T est non nul à ce point potentiel
        if Tx[idx_crit_global_temp] != 0:
            idx_crit_global = idx_crit_global_temp
            x_crit = x_vals[idx_crit_global]
            M_crit = M_res[idx_crit_global]
            plt.plot(x_crit, M_crit, 'ro', markersize=8, label=f'Max |Mz| in BC ({M_crit:.1f} Nm)')
        else: # Si T=0 au max de |Mz| (pile en C), chercher avant
             idx_C_strict = np.searchsorted(x_vals, x_C, side='left')
             if idx_B <= idx_C_strict:
                  M_res_BC_strict = M_res[idx_B:idx_C_strict+1]
                  x_vals_BC_strict = x_vals[idx_B:idx_C_strict+1]
                  idx_crit_local_strict = np.argmax(M_res_BC_strict)
                  idx_crit_global = idx_B + idx_crit_local_strict
                  x_crit = x_vals[idx_crit_global]
                  M_crit = M_res[idx_crit_global]
                  plt.plot(x_crit, M_crit, 'ro', markersize=8, label=f'Max |Mz| in BC ({M_crit:.1f} Nm)')
             else: # Fallback
                  idx_crit_global = idx_B
                  x_crit = x_vals[idx_crit_global]
                  M_crit = M_res[idx_crit_global]
                  plt.plot(x_crit, M_crit, 'ro', markersize=8, label=f'|Mz| at B ({M_crit:.1f} Nm)')
                  print("Avertissement: Point critique pris en B (fallback).")
    else:
        idx_crit_global = idx_B
        x_crit = x_vals[idx_crit_global]
        M_crit = M_res[idx_crit_global]
        print("Avertissement: Zone B-C trop petite.")
        plt.plot(x_crit, M_crit, 'ro', markersize=8, label=f'|Mz| at B ({M_crit:.1f} Nm)')
else:
     idx_crit_global = idx_B
     x_crit = x_vals[idx_crit_global]
     M_crit = M_res[idx_crit_global]
     print("Avertissement: Problème indices B>=C.")
     plt.plot(x_crit, M_crit, 'ro', markersize=8, label=f'|Mz| at B ({M_crit:.1f} Nm)')

plt.legend()
plt.tight_layout()
plt.show()


# -----------------------------------------------------
# 6. Identification du Point Critique et Calcul des Contraintes
# -----------------------------------------------------
# Le point critique (x_crit) est où M_res=abs(Mz) est max entre B et C (où Tx != 0)
My_crit = 0.0 # Car My(x) = 0
Mz_crit = Mz[idx_crit_global]
M_crit = abs(Mz_crit) # = M_res[idx_crit_global]
T_crit = Tx[idx_crit_global] # Devrait être égal à T

# Vérifier si T_crit est bien non nul
if abs(T_crit) < 1e-9:
    print(f"ATTENTION: Le couple de torsion T est nul ({T_crit:.2e} Nm) au point critique identifié x={x_crit:.3f}m.")

print(f"Point Critique (max Mz résultant entre B et C où T>0):")
print(f"  Position x_crit: {x_crit:.3f} m")
print(f"  Moment Résultant M_crit = |Mz|: {M_crit:.2f} Nm (My=0, Mz={Mz_crit:.2f})")
print(f"  Couple de Torsion T_crit: {T_crit:.2f} Nm")
print("-" * 30)

# Propriétés géométriques de la section
c = d / 2
I = np.pi * d**4 / 64
J = np.pi * d**4 / 32

# Calcul des contraintes au point critique
sigma_x = Mz_crit * c / I # Utiliser Mz (avec son signe) pour sigma_x
tau_torsion = T_crit * c / J

print("Contraintes au Point Critique (en surface):")
print(f"  Rayon externe c: {c*1000:.1f} mm")
print(f"  Moment quadratique I: {I:.3e} m^4")
print(f"  Moment polaire J: {J:.3e} m^4")
print(f"  Contrainte Normale de Flexion (sigma_x): {sigma_x / 1e6:.2f} MPa")
print(f"  Contrainte Cisaill. de Torsion (tau): {tau_torsion / 1e6:.2f} MPa")
print("-" * 30)

# -----------------------------------------------------
# 7. Calcul des Contraintes Principales
# -----------------------------------------------------
# Gérer le cas où T est nul au point critique
if abs(tau_torsion) < 1e-9 * 1e6: # Tolérance en MPa
     sigma_1 = sigma_x if sigma_x >= 0 else 0.0
     sigma_2 = sigma_x if sigma_x < 0 else 0.0
     tau_max_inplane = abs(sigma_x / 2)
     tau_max_abs = abs(sigma_x / 2)
     print("Note: Calcul des contraintes principales simplifié car tau_torsion ≈ 0.")
else:
     sigma_1 = (sigma_x / 2) + np.sqrt((sigma_x / 2)**2 + tau_torsion**2)
     sigma_2 = (sigma_x / 2) - np.sqrt((sigma_x / 2)**2 + tau_torsion**2)
     tau_max_inplane = np.sqrt((sigma_x / 2)**2 + tau_torsion**2)
     tau_max_abs = max(abs(sigma_1 / 2), abs(sigma_2 / 2), tau_max_inplane)

print("Contraintes Principales au Point Critique:")
print(f"  Sigma_1: {sigma_1 / 1e6:.2f} MPa")
print(f"  Sigma_2: {sigma_2 / 1e6:.2f} MPa")
print(f"  Sigma_3: 0.00 MPa")
print(f"  Tau_max (dans le plan): {tau_max_inplane / 1e6:.2f} MPa")
print(f"  Tau_max (absolu): {tau_max_abs / 1e6:.2f} MPa")
print("-" * 30)