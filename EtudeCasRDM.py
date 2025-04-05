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

# Angle de pression supposé (en degrés)
pressure_angle_deg = 20.0

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

pressure_angle_rad = np.deg2rad(pressure_angle_deg)

# -----------------------------------------------------
# 2. Calculs Préparatoires
# -----------------------------------------------------
# Couple
T = P / omega

# Forces sur les engrenages (Hypothèses de direction basées sur Fig.3 et transmission)
# Engrenage C (Sortie)
Fy_C = - T / rC  # Force tangentielle (-y direction)
Fz_C = - abs(Fy_C) * np.tan(pressure_angle_rad) # Force radiale (-z direction)

# Engrenage B (Entrée)
Ft_B = T / rB   # Force tangentielle (+y direction)
Fr_B = abs(Ft_B) * np.tan(pressure_angle_rad) # Force radiale (+z direction)

print(f"Calculs Préparatoires:")
print(f"  Vitesse angulaire (omega): {omega:.2f} rad/s")
print(f"  Couple (T): {T:.2f} Nm")
print(f"Forces en B: Ft_B (y) = {Ft_B:.2f} N, Fr_B (z) = {Fr_B:.2f} N")
print(f"Forces en C: Fy_C (y) = {Fy_C:.2f} N, Fz_C (z) = {Fz_C:.2f} N")
print("-" * 30)

# -----------------------------------------------------
# 3. Calcul des Réactions aux Paliers (A et D)
# -----------------------------------------------------
# --- Plan XY (Forces en Y, Moments autour de Z) ---
# Equations:
# Ay + Dy + Ft_B + Fy_C = 0
# (Ft_B * x_B) + (Fy_C * x_C) + (Dy * x_D) = 0  (Moment / A)

# Système matriciel: A_mat * [Ay, Dy]' = B_mat
A_mat_yz = np.array([[1, 1],
                   [0, x_D]])
B_vec_yz = np.array([[-Ft_B - Fy_C],
                   [-(Ft_B * x_B) - (Fy_C * x_C)]])

# Résolution
sol_yz = np.linalg.solve(A_mat_yz, B_vec_yz)
Ay = sol_yz[0, 0]
Dy = sol_yz[1, 0]

# --- Plan XZ (Forces en Z, Moments autour de Y) ---
# Equations:
# Az + Dz + Fr_B + Fz_C = 0
# -(Fr_B * x_B) - (Fz_C * x_C) - (Dz * x_D) = 0  (Moment / A)
# Attention aux signes pour le moment M_y

# Système matriciel: A_mat * [Az, Dz]' = B_mat
A_mat_zy = np.array([[1, 1],
                   [0, x_D]]) # La matrice A est la même structure
B_vec_zy = np.array([[-Fr_B - Fz_C],
                   [(Fr_B * x_B) + (Fz_C * x_C)]]) # Signes inversés pour B car M_y = -(Fz*x)

# Résolution
sol_zy = np.linalg.solve(A_mat_zy, B_vec_zy)
Az = sol_zy[0, 0]
Dz = sol_zy[1, 0]

print("Réactions aux paliers:")
print(f"  Ay: {Ay:.2f} N, Dy: {Dy:.2f} N")
print(f"  Az: {Az:.2f} N, Dz: {Dz:.2f} N")

# Vérification de l'équilibre (approximatif à cause des erreurs numériques)
print(f"  Vérif Equilibre Y: Ay+Ft_B+Fy_C+Dy = {Ay + Ft_B + Fy_C + Dy:.2e} N (devrait être ~0)")
print(f"  Vérif Equilibre Z: Az+Fr_B+Fz_C+Dz = {Az + Fr_B + Fz_C + Dz:.2e} N (devrait être ~0)")
print("-" * 30)

# -----------------------------------------------------
# 4. Calcul des Efforts Internes le long de l'arbre
# -----------------------------------------------------
# Discrétisation de l'arbre
x_vals = np.linspace(0, L_total, 500)
Vy = np.zeros_like(x_vals)
Vz = np.zeros_like(x_vals)
Mz = np.zeros_like(x_vals)
My = np.zeros_like(x_vals)
Tx = np.zeros_like(x_vals)

for i, x in enumerate(x_vals):
    # --- Efforts Tranchants ---
    Vy[i] = Ay
    Vz[i] = Az
    if x > x_B:
        Vy[i] += Ft_B
        Vz[i] += Fr_B
    if x > x_C:
        Vy[i] += Fy_C
        Vz[i] += Fz_C
    # (La réaction en D n'est pas incluse car on s'arrête juste avant D pour les efforts internes)

    # --- Moments Fléchissants ---
    # Mz = Somme des moments des forces Y autour de z
    Mz[i] = Ay * x
    if x > x_B:
        Mz[i] += Ft_B * (x - x_B)
    if x > x_C:
        Mz[i] += Fy_C * (x - x_C)

    # My = Somme des moments des forces Z autour de y (attention My = -Somme(Fz*bras_levier))
    My[i] = - (Az * x)
    if x > x_B:
        My[i] -= Fr_B * (x - x_B)
    if x > x_C:
        My[i] -= Fz_C * (x - x_C)

    # --- Couple de Torsion ---
    if x > x_B and x <= x_C:
        Tx[i] = T
    else:
        Tx[i] = 0 # Nul avant B et après C (le couple est "consommé" en C)

# Moment fléchissant résultant
M_res = np.sqrt(My**2 + Mz**2)

# -----------------------------------------------------
# 5. Tracer les Diagrammes des Efforts Internes
# -----------------------------------------------------
plt.figure(figsize=(12, 10))

plt.subplot(3, 2, 1)
plt.plot(x_vals, Vy, label='Vy(x)')
plt.title('Effort Tranchant Vy')
plt.xlabel('x (m)')
plt.ylabel('Vy (N)')
plt.grid(True)
plt.axvline(x_A, color='k', linestyle='--', label='A')
plt.axvline(x_B, color='r', linestyle='--', label='B')
plt.axvline(x_C, color='g', linestyle='--', label='C')
plt.axvline(x_D, color='k', linestyle='--')
plt.legend()

plt.subplot(3, 2, 2)
plt.plot(x_vals, Vz, label='Vz(x)')
plt.title('Effort Tranchant Vz')
plt.xlabel('x (m)')
plt.ylabel('Vz (N)')
plt.grid(True)
plt.axvline(x_A, color='k', linestyle='--')
plt.axvline(x_B, color='r', linestyle='--')
plt.axvline(x_C, color='g', linestyle='--')
plt.axvline(x_D, color='k', linestyle='--')
plt.legend()

plt.subplot(3, 2, 3)
plt.plot(x_vals, Mz, label='Mz(x)')
plt.title('Moment Fléchissant Mz')
plt.xlabel('x (m)')
plt.ylabel('Mz (Nm)')
plt.grid(True)
plt.axvline(x_A, color='k', linestyle='--')
plt.axvline(x_B, color='r', linestyle='--')
plt.axvline(x_C, color='g', linestyle='--')
plt.axvline(x_D, color='k', linestyle='--')
plt.legend()


plt.subplot(3, 2, 4)
plt.plot(x_vals, My, label='My(x)')
plt.title('Moment Fléchissant My')
plt.xlabel('x (m)')
plt.ylabel('My (Nm)')
plt.grid(True)
plt.axvline(x_A, color='k', linestyle='--')
plt.axvline(x_B, color='r', linestyle='--')
plt.axvline(x_C, color='g', linestyle='--')
plt.axvline(x_D, color='k', linestyle='--')
plt.legend()

plt.subplot(3, 2, 5)
plt.plot(x_vals, Tx, label='Tx(x)')
plt.title('Couple de Torsion Tx')
plt.xlabel('x (m)')
plt.ylabel('Tx (Nm)')
plt.ylim(min(Tx)-abs(T*0.1), max(Tx)+abs(T*0.1)) # Ajuste limite y pour visibilité
plt.grid(True)
plt.axvline(x_A, color='k', linestyle='--')
plt.axvline(x_B, color='r', linestyle='--')
plt.axvline(x_C, color='g', linestyle='--')
plt.axvline(x_D, color='k', linestyle='--')
plt.legend()

plt.subplot(3, 2, 6)
plt.plot(x_vals, M_res, label='M_res(x)', color='purple')
plt.title('Moment Fléchissant Résultant M')
plt.xlabel('x (m)')
plt.ylabel('M (Nm)')
plt.grid(True)
plt.axvline(x_A, color='k', linestyle='--')
plt.axvline(x_B, color='r', linestyle='--')
plt.axvline(x_C, color='g', linestyle='--')
plt.axvline(x_D, color='k', linestyle='--')

# Marquer le max de M dans la zone B-C
idx_B = np.searchsorted(x_vals, x_B)
idx_C = np.searchsorted(x_vals, x_C)
M_res_BC = M_res[idx_B:idx_C+1]
x_vals_BC = x_vals[idx_B:idx_C+1]
idx_crit_local = np.argmax(M_res_BC)
idx_crit_global = idx_B + idx_crit_local
x_crit = x_vals[idx_crit_global]
M_crit = M_res[idx_crit_global]
plt.plot(x_crit, M_crit, 'ro', markersize=8, label=f'Max M in BC ({M_crit:.1f} Nm)')
plt.legend()


plt.tight_layout()
plt.show()

# -----------------------------------------------------
# 6. Identification du Point Critique et Calcul des Contraintes
# -----------------------------------------------------
# Le point critique (x_crit) est où M_res est max entre B et C (où Tx != 0)
# Récupération des efforts internes au point critique
My_crit = My[idx_crit_global]
Mz_crit = Mz[idx_crit_global]
T_crit = Tx[idx_crit_global] # Devrait être égal à T

print(f"Point Critique (max M résultant entre B et C):")
print(f"  Position x_crit: {x_crit:.3f} m")
print(f"  Moment Résultant M_crit: {M_crit:.2f} Nm (My={My_crit:.2f}, Mz={Mz_crit:.2f})")
print(f"  Couple de Torsion T_crit: {T_crit:.2f} Nm")
print("-" * 30)

# Propriétés géométriques de la section
c = d / 2  # Rayon externe
I = np.pi * d**4 / 64  # Moment quadratique
J = np.pi * d**4 / 32  # Moment polaire

# Calcul des contraintes au point critique (en surface, c=d/2)
# Contrainte normale due à la flexion résultante M_crit
sigma_x = M_crit * c / I

# Contrainte de cisaillement due à la torsion T_crit
# Le cisaillement max tau_xy et tau_xz sont sur la surface.
# L'élément le plus critique sera là où la contrainte normale de flexion est max.
tau_torsion = T_crit * c / J # C'est le tau_max dû à la torsion (ex: tau_xy ou tau_xz)

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
# L'état de contrainte au point le plus critique en surface (là où sigma_x est max)
# est: sigma_x (calculé), sigma_y = 0, tau_xy = tau_torsion (ou tau_xz selon orientation)

# Formules pour contrainte plane
sigma_1 = (sigma_x / 2) + np.sqrt((sigma_x / 2)**2 + tau_torsion**2)
sigma_2 = (sigma_x / 2) - np.sqrt((sigma_x / 2)**2 + tau_torsion**2)

# Cisaillement max dans le plan (sigma1, sigma2)
tau_max_inplane = np.sqrt((sigma_x / 2)**2 + tau_torsion**2)

# Cisaillement max absolu (considérant sigma3 = 0)
# sigma_3 = 0
# tau_max_abs = max(abs(sigma_1 - sigma_2)/2, abs(sigma_1 - 0)/2, abs(sigma_2 - 0)/2)
tau_max_abs = max(abs(sigma_1 / 2), abs(sigma_2 / 2), tau_max_inplane)

print("Contraintes Principales au Point Critique:")
print(f"  Sigma_1: {sigma_1 / 1e6:.2f} MPa")
print(f"  Sigma_2: {sigma_2 / 1e6:.2f} MPa")
# Sigma_3 est 0 car état de contrainte plan en surface
print(f"  Sigma_3: 0.00 MPa")
print(f"  Tau_max (dans le plan): {tau_max_inplane / 1e6:.2f} MPa")
print(f"  Tau_max (absolu): {tau_max_abs / 1e6:.2f} MPa")
print("-" * 30)