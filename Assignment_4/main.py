import numpy as np
import matplotlib.pyplot as plt

# Isentropic Relations
def P0_P(γ, M):
    return (1 + 0.5 * (γ - 1) * M**2) ** (γ / (γ - 1))

def T0_T(γ, M):
    return 1 + 0.5 * (γ - 1) * M**2

def vectors(P, ρ, u, A):
    E = P / (γ - 1) + 0.5 * ρ * u**2  # Total energy
    U = np.array([ρ * A, ρ * u * A, E * A])  # Conservative variables
    return U

def extract(U, A):
    ρ = U[0, :] / A
    u = U[1, :] / U[0, :]
    E = U[2, :] / A
    P = (γ - 1) * (E - 0.5 * ρ * u**2)
    return ρ, u, E, P

def FVS(U, A, γ):
    ρ, u, E, P = extract(U, A)  # Extract primitive variables
    a = np.sqrt(γ * P / ρ)  # Speed of sound
    M = u / a  # Mach number

    F = np.array([ρ * u * A,(ρ * u**2 + P) * A,(E + P) * u * A])  # Flux vector
    
    F_plus = np.zeros((3, Nx))
    F_minus = np.zeros((3, Nx))
    
    for i in range(Nx):
        if M[i] <= -1:
            F_plus[:, i] = np.zeros(3)
            F_minus[:, i] = F[:, i]
        elif M[i] >= 1:
            F_plus[:, i] = F[:, i]
            F_minus[:, i] = np.zeros(3)
        else:
            α = 0.25 * ρ[i] * a[i] * (M[i] + 1)**2 * A[i]
            F_plus[0, i] = α
            F_plus[1, i] = α * (2 * a[i] / γ) * (1 + ((γ - 1) / 2) * M[i])
            F_plus[2, i] = α * (2 * (a[i] ** 2) / (γ**2 - 1)) * (1 + ((γ - 1) / 2) * M[i]) ** 2
            F_minus[:, i] = F[:, i] - F_plus[:, i]
    return F_plus, F_minus

# Air Properties
R = 287.0  # Specific gas constant for air
γ = 1.4  # Specific heat ratio for air

# Domain of the CD nozzle
L = 2.0  # Length of the domain
Nx = 101  # Number of grid points
Δx = L / (Nx - 1)  # Grid spacing
x = np.linspace(0, L, Nx)  # Grid points

# Reservoir conditions
P0 = 1.0133e5  # Reservoir pressure (Pa)
T0 = 300.0  # Reservoir temperature (K)
Pe = 0.585 * P0  # Outlet pressure (Pa)

# Area distribution function
def Area(x):
    return 1.0 + 2.0 * (x - 1.0) ** 2  # Area function

A = Area(x)  # Area distribution
dA_dx = np.gradient(A, Δx)  # Derivative of area function

# Initial conditions
M = np.ones(Nx) * 0.01
T = T0 / T0_T(γ, M)
P = P0 / P0_P(γ, M)
ρ = P / (R * T)
u = np.sqrt(γ * R * T) * M
a = np.sqrt(γ * R * T)  # Speed of sound

U = vectors(P, ρ, u, A)  # Conservative variables

# Courant number and iteration parameters
CFL = 0.8
tol = 1e-3
iter_max = 50000

for iter in range(iter_max):
    U_old = U.copy()  # Store new values for convergence check    
    
    lambda_max = np.max(np.abs(u) + a)  # Maximum wave speed
    Δt = CFL * Δx / lambda_max  # Time step size

    # Boundary conditions
    # Inlet
    u[0] = u[1]
    # T[0] as a function of u[0] and T0
    T[0] = (2*γ*R*T0 - (γ-1)*u[0]**2)/(2*γ*R)
    M[0] = u[0] / np.sqrt(γ * R * T[0])
    P[0] = P0 / P0_P(γ, M[0])
    ρ[0] = P[0] / (R * T[0])
    
    # Outlet
    P[-1] = Pe
    ρ[-1] = ρ[-2]
    u[-1] = u[-2]
    
    U = vectors(P, ρ, u, A)  # Update conservative variables

    # Compute fluxes
    F_plus, F_minus = FVS(U, A, γ)

    S = np.array([np.zeros(Nx), P * dA_dx, np.zeros(Nx)])
    U[:, 1:-1] = U[:,1:-1] - (Δt / Δx) * (F_plus[:, 1:-1] - F_plus[:, :-2]) \
                   - (Δt / Δx) * (F_minus[:, 2:] - F_minus[:, 1:-1]) \
                   + Δt * S[:, 1:-1]
        
    ρ, u, E, P = extract(U, A)
    T = P / (ρ * R)
    a = np.sqrt(γ * R * T)
    M = u / a

    # Check convergence
    error = np.max(np.abs(U - U_old))
    if iter % 100 == 0:
        print(f"Iteration {iter}, Error: {error:.6e}")
    if error < tol:
        print(f"Converged in {iter} iterations. Error: {error:.6e}")
        break
    if iter == iter_max - 1:
        print("Maximum iterations reached without convergence.")
        break

# Plot results
fig, axes = plt.subplots(3, 1, figsize=(12, 8))
for ax, data, ylabel in zip(axes, [P / P0, T / T0, M], ["Pressure Ratio", "Temperature Ratio", "Mach Number"]):
    ax.plot(x, data)
    ax.set(xlabel="x (m)", ylabel=ylabel)
    ax.grid()
plt.tight_layout()
plt.show()

from scipy.optimize import fsolve

def A_Astar(γ, M):
    return ((2/(γ+1))*(1 + 0.5*(γ-1)*M**2))**((γ+1)/(2*(γ-1))) / M

# Analytical solution (for comparison)
def solve_Mach(A_ratio, γ, M_guess):
    def eqn(M):
        return A_ratio - A_Astar(γ, M)
    return fsolve(eqn, M_guess)[0]

# Find the throat
throat_index = np.argmin(A)

M_analytical_1 = np.zeros_like(x)
M_analytical_2 = np.zeros_like(x)

# Subsonic Solution before the throat
for i in range(throat_index):
    M_analytical_1[i] = solve_Mach(A[i]/A[throat_index], γ, M_guess = 0.2)
    M_analytical_2[i] = solve_Mach(A[i]/A[throat_index], γ, M_guess = 0.2)

# Sonic Solution at the throat
M_analytical_1[throat_index] = 1.0
M_analytical_2[throat_index] = 1.0

# Supersonic Solution after the throat
for i in range(throat_index + 1, Nx):
    M_analytical_1[i] = solve_Mach(A[i]/A[throat_index], γ, M_guess = 1.8)

# Subsonic Solution after the throat
for i in range(throat_index + 1, Nx):
    M_analytical_2[i] = solve_Mach(A[i]/A[throat_index], γ, M_guess = 0.2)

P_analytical_1 = P0 / P0_P(γ, M_analytical_1)
T_analytical_1 = T0 / T0_T(γ, M_analytical_1)

P_analytical_2 = P0 / P0_P(γ, M_analytical_2)
T_analytical_2 = T0 / T0_T(γ, M_analytical_2)

# Plotting
fig, axes = plt.subplots(3, 1, figsize=(12, 12))

# Numerical vs Analytical
axes[0].plot(x, P/P0, 'b-', label='Numerical')
axes[0].plot(x, P_analytical_1/P0, 'r--', label='Analytical Supersonic')
axes[0].plot(x, P_analytical_2/P0, 'g--', label='Analytical Subsonic')
axes[0].set(ylabel='P/P0', title='Pressure Ratio')
axes[0].legend()
axes[0].grid()

axes[1].plot(x, T/T0, 'b-', label='Numerical')
axes[1].plot(x, T_analytical_1/T0, 'r--', label='Analytical Supersonic')
axes[1].plot(x, T_analytical_2/T0, 'g--', label='Analytical Subsonic')
axes[1].set(ylabel='T/T0', title='Temperature Ratio')
axes[1].grid()

axes[2].plot(x, M, 'b-', label='Numerical')
axes[2].plot(x, M_analytical_1, 'r--', label='Analytical Supersonic')
axes[2].plot(x, M_analytical_2, 'g--', label='Analytical Subsonic')
axes[2].set(ylabel='Mach Number', title='Mach Number Distribution')
axes[2].grid()

plt.tight_layout()
plt.savefig('results.png')
plt.show()