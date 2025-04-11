from matplotlib.pylab import f
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------
# 1. Données et Constantes
# -----------------------------------------------
P_kW = 8.5
rpm = 541
d_mm = 30
rB_mm = 50
rC_mm = 75
L_AB_mm = 200
L_BC_mm = 400
L_CD_mm = 350

# Conversion en SI
P = P_kW * 1000  # W
omega = rpm * (2 * np.pi / 60)  # rad/s
T = P / omega  # Couple (Nm)
d = d_mm / 1000  # Diamètre (m)
c = d / 2  # Rayon (m)
I = np.pi * d**4 / 64
J = np.pi * d**4 / 32

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

# -----------------------------------------------
# 2. Forces tangentielles
# -----------------------------------------------
Ft_B_y = -T / rB  # Force tangentielle en B selon -Y
Ft_C_z = -T / rC  # Force tangentielle en C selon -Z
# -----------------------------------------------
# 3. Équilibre statique (réactions A et D)
# -----------------------------------------------
# Équations d'équilibre :
# Y: Ay + Dy + Ft_B_y = 0
# Z: Az + Dz + Ft_C_z = 0
# Moments autour de A selon Z (forces en Y): Dy*(x_D) + Ft_B_y*(x_B) = 0
# Moments autour de A selon Y (forces en Z): Dz*(x_D) + Ft_C_z*(x_C) = 0

# Résolution matricielle
A_mat = np.array([
    [1, 1, 0, 0],
    [0, 0, 1, 1],
    [0, x_D, 0, 0],
    [0, 0, 0, x_D]
])

B_vec = np.array([
    [-Ft_B_y],
    [-Ft_C_z],
    [-(Ft_B_y * x_B)],
    [-(Ft_C_z * x_C)]
])

Ay, Dy, Az, Dz = np.linalg.solve(A_mat, B_vec).flatten()

print("Réactions aux appuis :")
print(f"Ay = {Ay:.2f} N, Dy = {Dy:.2f} N")
print(f"Az = {Az:.2f} N, Dz = {Dz:.2f} N")
print("Forces aux engrenages :")
print(f"By = {Ft_B_y:.2f} N, Cz = {Ft_C_z:.2f} N")

# -----------------------------------------------
# 4. Efforts internes
# -----------------------------------------------

# Discrétisation de l'arbre
x_vals = np.linspace(0, L_total, 500)
Vy = np.zeros_like(x_vals)
Vz = np.zeros_like(x_vals)
Mz = np.zeros_like(x_vals)
My = np.zeros_like(x_vals)
Tx = np.zeros_like(x_vals)

for i, x in enumerate(x_vals):
    # Efforts tranchants
    Vy[i] = Ay + (Ft_B_y if x >= x_B else 0) + (Dy if x>= x_D else 0)
    Vz[i] = Az + (Ft_C_z if x >= x_C else 0) + (Dz if x>= x_D else 0)

    # Moments fléchissants
    My[i] = Ay * x + (Ft_B_y * (x - x_B) if x >= x_B else 0)
    Mz[i] = Az * x + (Ft_C_z * (x - x_C) if x >= x_C else 0)

    # Torsion
    Tx[i] = T if x_B < x <= x_C else 0

# Moment résultant combiné
M_res = np.sqrt(My**2 + Mz**2)

# --- Ajustement pour le tracé pour commencer à zéro ---
# L'approche ci-dessus fait que Vy[0] = Ay et Vz[0] = Az.
# Pour que le *graphique* commence à (0,0) avant le saut de la réaction Ay/Az,
# nous insérons un point (0,0) au début des données de tracé.
x_plot = np.insert(x_vals, 0, 0) # Ajoute un 0 au début de x
Vy_plot = np.insert(Vy, 0, 0)   # Ajoute un 0 au début de Vy
Vz_plot = np.insert(Vz, 0, 0)   # Ajoute un 0 au début de Vz


# -----------------------------------------------
# 5. Tracé des résultats
# -----------------------------------------------

fig, axes = plt.subplots(3, 2, figsize=(14, 12))

axes[0,0].plot(x_plot, Vy_plot)
axes[0,0].set(title='Effort tranchant Vy', xlabel='x (m)', ylabel='Vy (N)')
axes[0,0].grid()

axes[0,1].plot(x_plot, Vz_plot)
axes[0,1].set(title='Effort tranchant Vz', xlabel='x (m)', ylabel='Vz (N)')
axes[0,1].grid()

axes[1,0].plot(x_vals, My)
axes[1,0].set(title='Moment fléchissant My', xlabel='x (m)', ylabel='My (Nm)')
axes[1,0].grid()

axes[1,1].plot(x_vals, Mz)
axes[1,1].set(title='Moment fléchissant Mz', xlabel='x (m)', ylabel='Mz (Nm)')
axes[1,1].grid()

axes[2,0].plot(x_vals, Tx)
axes[2,0].set(title='Couple de torsion Tx', xlabel='x (m)', ylabel='Tx (Nm)')
axes[2,0].grid()

axes[2,1].plot(x_vals, M_res, color='purple')
axes[2,1].set(title='Moment résultant M_res', xlabel='x (m)', ylabel='M_res (Nm)')
axes[2,1].grid()

plt.tight_layout()
plt.show()



# -----------------------------------------------
# 6. Identification du Point Critique
# -----------------------------------------------
idx_zone = (x_vals >= x_B) & (x_vals <= x_C)
idx_crit = np.argmax(M_res[idx_zone])
idx_global = np.where(idx_zone)[0][0] + idx_crit
x_crit = x_vals[idx_global]
M_crit = M_res[idx_global]
T_crit = Tx[idx_global]

print("\nPoint critique :")
print(f"x = {x_crit:.3f} m, M_res = {M_crit:.2f} Nm, T = {T_crit:.2f} Nm")

# -----------------------------------------------
# 7. Contraintes au point critique
# -----------------------------------------------
sigma_max = M_crit * c / I
tau_torsion = T_crit * c / J

# Contraintes principales via Mohr
sigma_1 = sigma_max / 2 + np.sqrt((sigma_max / 2)**2 + tau_torsion**2)
sigma_2 = sigma_max / 2 - np.sqrt((sigma_max / 2)**2 + tau_torsion**2)

# Formule du manuel (Hibbeler, équation 10-30)
sigma_eq = np.sqrt(sigma_1**2 - sigma_1 * sigma_2 + sigma_2**2)

print("\nContraintes calculées (manuel Hibbeler):")
print(f"Contrainte normale (sigma): {sigma_max/1e6:.2f} MPa")
print(f"Contrainte de cisaillement (tau): {tau_torsion/1e6:.2f} MPa")
print(f"Sigma 1 (max): {sigma_1/1e6:.2f} MPa")
print(f"Sigma 2 (min): {sigma_2/1e6:.2f} MPa")
print(f"Contrainte équivalente (Von Mises - Hibbeler): {sigma_eq/1e6:.2f} MPa")

# -----------------------------------------------
# Cercle de Mohr
# -----------------------------------------------
sigma_avg = sigma_max / 2
R_mohr = np.sqrt((sigma_max / 2)**2 + tau_torsion**2)

theta = np.linspace(0, 2 * np.pi, 360)
x_mohr = sigma_avg + R_mohr * np.cos(theta)
y_mohr = R_mohr * np.sin(theta)

plt.figure(figsize=(6,6))
# Cercle de Mohr
plt.plot(x_mohr / 1e6, y_mohr / 1e6, label='Cercle de Mohr')

# Centre
plt.plot(sigma_avg / 1e6, 0, 'ko', label='Centre')

# Contraintes principales
sigma_1 = sigma_avg + R_mohr
sigma_2 = sigma_avg - R_mohr
plt.plot(np.array([sigma_1, sigma_2]) / 1e6, [0, 0], 'ro', label='Contraintes principales')

# Axes
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.xlabel('Contrainte normale σ (MPa)')
plt.ylabel('Contrainte cisaillement τ (MPa)')
plt.title('Cercle de Mohr au Point Critique')
plt.grid(True)
plt.axis('equal')
plt.legend()
plt.show()

print("\nContraintes principales (Cercle de Mohr):")
print(f"Sigma 1 = {sigma_1/1e6:.2f} MPa")
print(f"Sigma 2 = {sigma_2/1e6:.2f} MPa")
print(f"Tau max (rayon R) = {R_mohr/1e6:.2f} MPa")