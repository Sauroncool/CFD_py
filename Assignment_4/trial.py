import numpy as np
import matplotlib.pyplot as plt

# Parameters
γ = 1.4
R = 287.0

# Nozzle geometry
def Area(x):
    return 1.0 + 2.0 * (x - 1)**2

# Reservoir conditions
P_0 = 1.0133e5  # Pa
T0 = 300.0       # K
rho0 = P_0 / (R * T0)
Pe = 0.585 * P_0  # Exit pressure

# Grid setup
Nx = 101
L = 2.0
Δx = L / (Nx - 1)
x = np.linspace(0, L, Nx)
A = Area(x)

# Initial conditions - start with uniform flow at reservoir conditions
rho = np.ones(Nx) * rho0
u = np.zeros(Nx)
P = np.ones(Nx) * P_0

# Conservative variables [ρ*A, ρ*u*A, E*A]
U = np.zeros((Nx, 3))
for i in range(Nx):
    e = P[i] / (rho[i] * (γ - 1)) + 0.5 * u[i]**2  # specific total energy
    U[i, 0] = rho[i] * A[i]
    U[i, 1] = rho[i] * u[i] * A[i]
    U[i, 2] = rho[i] * e * A[i]

# Compute primitive variables from conservative variables
def get_primitive(U):
    rho = U[:, 0] / A
    u = U[:, 1] / U[:, 0]
    e = U[:, 2] / U[:, 0]
    P = (γ - 1) * (U[:, 2] - 0.5 * U[:, 1]**2 / U[:, 0])
    T = P / (rho * R)
    return rho, u, P, T

# van Leer Flux Vector Splitting
def van_leer_flux(U_left, U_right):
    # Left state
    rho_L = U_left[0] / A[:-1]
    u_L = U_left[1] / U_left[0]
    P_L = (γ - 1) * (U_left[2] - 0.5 * U_left[1]**2 / U_left[0])
    a_L = np.sqrt(γ * P_L / rho_L)
    M_L = u_L / a_L
    
    # Right state
    rho_R = U_right[0] / A[1:]
    u_R = U_right[1] / U_right[0]
    P_R = (γ - 1) * (U_right[2] - 0.5 * U_right[1]**2 / U_right[0])
    a_R = np.sqrt(γ * P_R / rho_R)
    M_R = u_R / a_R
    
    # Initialize fluxes
    F_p = np.zeros_like(U_left)
    F_m = np.zeros_like(U_right)
    
    # Positive flux components
    mask = M_L > -1
    F_p[mask, 0] = 0.25 * rho_L[mask] * a_L[mask] * (M_L[mask] + 1)**2 * A[:-1][mask]
    F_p[mask, 1] = F_p[mask, 0] * 2 * a_L[mask] / γ * ((γ - 1)/2 * M_L[mask] + 1)
    F_p[mask, 2] = F_p[mask, 0] * 2 * a_L[mask]**2 / (γ**2 - 1) * ((γ - 1)/2 * M_L[mask] + 1)**2
    
    # Negative flux components
    mask = M_R < 1
    F_m[mask, 0] = -0.25 * rho_R[mask] * a_R[mask] * (M_R[mask] - 1)**2 * A[1:][mask]
    F_m[mask, 1] = F_m[mask, 0] * 2 * a_R[mask] / γ * ((γ - 1)/2 * M_R[mask] - 1)
    F_m[mask, 2] = F_m[mask, 0] * 2 * a_R[mask]**2 / (γ**2 - 1) * ((γ - 1)/2 * M_R[mask] - 1)**2
    
    return F_p + F_m

# Apply boundary conditions
def apply_boundary_conditions(U):
    # Inlet (subsonic) - fixed total pressure and temperature
    rho_in = U[1, 0] / A[0]
    u_in = U[1, 1] / U[1, 0]
    P_in = (γ - 1) * (U[1, 2] - 0.5 * U[1, 1]**2 / U[1, 0])
    a_in = np.sqrt(γ * P_in / rho_in)
    M_in = u_in / a_in
    
    # Use isentropic relations for inlet
    P0_actual = P_in * (1 + 0.5*(γ-1)*M_in**2)**(γ/(γ-1))
    T0_actual = T0 * (1 + 0.5*(γ-1)*M_in**2)
    
    # Adjust to match reservoir conditions
    P[0] = P_0 / (1 + 0.5*(γ-1)*M_in**2)**(γ/(γ-1))
    T[0] = T0 / (1 + 0.5*(γ-1)*M_in**2)
    rho[0] = P[0] / (R * T[0])
    u[0] = M_in * np.sqrt(γ * R * T[0])
    
    U[0, 0] = rho[0] * A[0]
    U[0, 1] = rho[0] * u[0] * A[0]
    U[0, 2] = (P[0]/(γ-1) + 0.5*rho[0]*u[0]**2) * A[0]
    
    # Outlet (subsonic) - fixed static pressure
    P[-1] = Pe
    rho[-1] = U[-2, 0] / A[-1]
    u[-1] = U[-2, 1] / U[-2, 0]
    U[-1, 0] = rho[-1] * A[-1]
    U[-1, 1] = rho[-1] * u[-1] * A[-1]
    U[-1, 2] = (P[-1]/(γ-1) + 0.5*rho[-1]*u[-1]**2) * A[-1]

# Solve using van Leer FVS
def solve():
    CFL = 0.5
    t_final = 0.1
    t = 0.0
    
    while t < t_final:
        # Compute fluxes at cell interfaces
        F = van_leer_flux(U[:-1], U[1:])
        
        # Compute time step using CFL condition
        rho, u, P, _ = get_primitive(U)
        a = np.sqrt(γ * P / rho)
        dt = CFL * Δx / np.max(np.abs(u) + a)
        
        if t + dt > t_final:
            dt = t_final - t
        
        # Update solution
        U[1:-1] -= dt / Δx * (F[1:] - F[:-1])
        apply_boundary_conditions(U)
        
        t += dt
        
    return U

# Solve and plot results
U_final = solve()
rho, u, P, T = get_primitive(U_final)

plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.plot(x, u, label="Velocity")
plt.xlabel("x")
plt.ylabel("Velocity (m/s)")
plt.title("Flow Velocity")

plt.subplot(2, 2, 2)
plt.plot(x, P/P_0, label="Pressure")
plt.xlabel("x")
plt.ylabel("P/P0")
plt.title("Pressure Ratio")

plt.subplot(2, 2, 3)
plt.plot(x, rho/rho0, label="Density")
plt.xlabel("x")
plt.ylabel("ρ/ρ0")
plt.title("Density Ratio")

plt.subplot(2, 2, 4)
plt.plot(x, A, label="Area")
plt.xlabel("x")
plt.ylabel("Area (m²)")
plt.title("Nozzle Geometry")

plt.tight_layout()
plt.show()