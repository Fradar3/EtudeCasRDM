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
# 2. Forces tangentielles corrigées clairement
# -----------------------------------------------
Ft_B_y = T / rB  # Force tangentielle en B selon +Y
Ft_C_z = -T / rC  # Force tangentielle en C selon -Z

print("Forces corrigées clairement:")
print(f"Ft_B_y (+Y) = {Ft_B_y:.2f} N")
print(f"Ft_C_z (-Z) = {Ft_C_z:.2f} N")

# -----------------------------------------------
# 3. Équilibre statique clair (réactions A et D)
# -----------------------------------------------
# Équations d'équilibre :
# Y: Ay + Dy + Ft_B_y = 0
# Z: Az + Dz + Ft_C_z = 0
# Moments autour de A selon Z (forces en Y): Dy*(x_D) + Ft_B_y*(x_B) = 0
# Moments autour de A selon Y (forces en Z): Dz*(x_D) + Ft_C_z*(x_C) = 0

# Résolution matricielle claire
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

sol = np.linalg.solve(A_mat, B_vec)
Ay, Dy, Az, Dz = sol.flatten()

print("\nRéactions aux appuis corrigées:")
print(f"Ay = {Ay:.2f} N, Dy = {Dy:.2f} N")
print(f"Az = {Az:.2f} N, Dz = {Dz:.2f} N")

# -----------------------------------------------
# 4. Efforts internes clairement corrigés
# -----------------------------------------------
x_vals = np.linspace(0, L_total, 500)
Vy = np.zeros_like(x_vals)
Vz = np.zeros_like(x_vals)
Mz = np.zeros_like(x_vals)
My = np.zeros_like(x_vals)
Tx = np.zeros_like(x_vals)

for i, x in enumerate(x_vals):
    # Efforts tranchants
    Vy[i] = Ay
    Vz[i] = Az
    if x >= x_B:
        Vy[i] += Ft_B_y
    if x >= x_C:
        Vz[i] += Ft_C_z

    # Moments fléchissants
    Mz[i] = Ay * x
    if x >= x_B:
        Mz[i] += Ft_B_y * (x - x_B)

    My[i] = -Az * x
    if x >= x_C:
        My[i] -= Ft_C_z * (x - x_C)

    # Torsion
    if x_B < x <= x_C:
        Tx[i] = T
    else:
        Tx[i] = 0

M_res = np.sqrt(My**2 + Mz**2)
idx_B = np.argmin(np.abs(x_vals - x_B))
idx_C = np.argmin(np.abs(x_vals - x_C))

print(f"\nMoment résultant précis au point B (x={x_B} m): {M_res[idx_B]:.2f} Nm")
print(f"Moment résultant précis au point C (x={x_C} m): {M_res[idx_C]:.2f} Nm")

# -----------------------------------------------
# 5. Tracé clair et complet des résultats
# -----------------------------------------------
plt.figure(figsize=(12, 10))

plt.subplot(3, 2, 1)
plt.plot(x_vals, Vy, label='Vy(x)')
plt.title('Effort Tranchant Vy')
plt.xlabel('x (m)')
plt.ylabel('Vy (N)')
plt.grid(); plt.legend()

plt.subplot(3, 2, 2)
plt.plot(x_vals, Vz, label='Vz(x)')
plt.title('Effort Tranchant Vz')
plt.xlabel('x (m)')
plt.ylabel('Vz (N)')
plt.grid(); plt.legend()

plt.subplot(3, 2, 3)
plt.plot(x_vals, Mz, label='Mz(x)')
plt.title('Moment Fléchissant Mz')
plt.xlabel('x (m)')
plt.ylabel('Mz (Nm)')
plt.grid(); plt.legend()

plt.subplot(3, 2, 4)
plt.plot(x_vals, My, label='My(x)')
plt.title('Moment Fléchissant My')
plt.xlabel('x (m)')
plt.ylabel('My (Nm)')
plt.grid(); plt.legend()

plt.subplot(3, 2, 5)
plt.plot(x_vals, Tx, label='T(x)')
plt.title('Couple de Torsion Tx')
plt.xlabel('x (m)')
plt.ylabel('Tx (Nm)')
plt.grid(); plt.legend()

plt.subplot(3, 2, 6)
plt.plot(x_vals, M_res, label='M_res(x)', color='purple')
plt.title('Moment Résultant M_res')
plt.xlabel('x (m)')
plt.ylabel('M_res (Nm)')
plt.grid(); plt.legend()

plt.tight_layout()
plt.show()

# -----------------------------------------------
# 6. Identification du Point Critique clair
# -----------------------------------------------
idx_zone = (x_vals >= x_B) & (x_vals <= x_C)
idx_crit = np.argmax(M_res[idx_zone])
idx_global = np.where(idx_zone)[0][0] + idx_crit
x_crit = x_vals[idx_global]
M_crit = M_res[idx_global]
T_crit = Tx[idx_global]

print("\nPoint critique clairement identifié :")
print(f"x = {x_crit:.3f} m, M_res = {M_crit:.2f} Nm, T = {T_crit:.2f} Nm")

# -----------------------------------------------
# 7. Contraintes au point critique
# -----------------------------------------------
sigma_max = M_crit * c / I
tau_torsion = T_crit * c / J

sigma_eq = np.sqrt(sigma_max**2 + 3*tau_torsion**2)

print("\nContraintes clairement calculées:")
print(f"Contrainte normale (sigma): {sigma_max/1e6:.2f} MPa")
print(f"Contrainte de cisaillement (tau): {tau_torsion/1e6:.2f} MPa")
print(f"Contrainte équivalente (Von Mises): {sigma_eq/1e6:.2f} MPa")

# -----------------------------------------------
# Cercle de Mohr (Ajout clair)
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