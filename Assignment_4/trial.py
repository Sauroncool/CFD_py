import matplotlib.pyplot as plt
import numpy as np


# Isentropic Relations
def P0_P(M, γ=1.4):
    return (1 + 0.5 * (γ - 1) * M**2) ** (γ / (γ - 1))


def T0_T(M, γ=1.4):
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

    F = np.array([ρ * u * A, (ρ * u**2 + P) * A, (E + P) * u * A])  # Flux vector

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
            α = 0.25 * ρ[i] * a[i] * (M[i] + 1) ** 2 * A[i]
            F_plus[0, i] = α
            F_plus[1, i] = α * (2 * a[i] / γ) * (1 + ((γ - 1) / 2) * M[i])
            F_plus[2, i] = (
                α * (2 * (a[i] ** 2) / (γ**2 - 1)) * (1 + ((γ - 1) / 2) * M[i]) ** 2
            )
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

Pe_P0 = 0.585  # Exit pressure ratio
Pe = Pe_P0*P0  # Outlet pressure (Pa)


# Area distribution function
def Area(x):
    return 1.0 + 2.0 * (x - 1.0) ** 2  # Area function


A = Area(x)  # Area distribution
dA_dx = np.gradient(A, Δx)  # Derivative of area function

# Initial conditions
M = np.ones(Nx) * 0.01
T = T0 / T0_T(M)
P = P0 / P0_P(M)
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
    T[0] = (2 * γ * R * T0 - (γ - 1) * u[0] ** 2) / (2 * γ * R)
    M[0] = u[0] / np.sqrt(γ * R * T[0])
    P[0] = P0 / P0_P(M[0])
    ρ[0] = P[0] / (R * T[0])

    # Outlet
    P[-1] = Pe
    ρ[-1] = ρ[-2]
    u[-1] = u[-2]

    U = vectors(P, ρ, u, A)  # Update conservative variables

    # Compute fluxes
    F_plus, F_minus = FVS(U, A, γ)

    S = np.array([np.zeros(Nx), P * dA_dx, np.zeros(Nx)])
    U[:, 1:-1] = (
        U[:, 1:-1]
        - (Δt / Δx) * (F_plus[:, 1:-1] - F_plus[:, :-2])
        - (Δt / Δx) * (F_minus[:, 2:] - F_minus[:, 1:-1])
        + Δt * S[:, 1:-1]
    )

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
for ax, data, ylabel in zip(
    axes, [P / P0, T / T0, M], ["Pressure Ratio", "Temperature Ratio", "Mach Number"]
):
    ax.plot(x, data)
    ax.set(xlabel="x (m)", ylabel=ylabel)
    ax.grid()
plt.tight_layout()
plt.show()

from scipy.optimize import fsolve


def A_Astar(M, γ=1.4):
    return ((2 / (γ + 1)) * (1 + 0.5 * (γ - 1) * M**2)) ** ((γ + 1) / (2 * (γ - 1))) / M


def solve_mach_A_ratio(A_ratio, M_guess):
    def eqn(M):
        return A_ratio - A_Astar(M)

    return fsolve(eqn, M_guess)[0]

# Normal Shock Relations
def after_shock_mach_number(M1, γ=1.4):
    return np.sqrt((1 + ((γ - 1) / 2) * M1**2) / (γ * M1**2 - (γ - 1) / 2))

def P2_P1(M1, γ=1.4):
    return 1 + (2 * γ / (γ + 1)) * (M1**2 - 1)

def P02_P01(M, γ=1.4):  # Total pressure ratio (Zucker and Biblarz)
    M2 = after_shock_mach_number(M)
    return P2_P1(M, γ) * ((1 + (γ - 1) / 2 * M2**2)/(1 + (γ - 1) / 2 * M**2))**(γ / (γ - 1))

def solve_mach_P0_ratio(P0_ratio,M_guess):
    def eqn(M):
        return P0_ratio - P02_P01(M)

    return fsolve(eqn, M_guess)[0]

throat_index = np.argmin(A)  # Index of the throat_index
At = A[throat_index]  # Throat Area

PeAe_P0eAe_star = (Pe_P0)*(A[-1]/At)  # PeAe/P0eAe*

def Me_squared(PeAe_P0eAe_star, γ = 1.4): #(JD Anderson)
    term_1 = -1 / (γ - 1)
    term_2 = np.sqrt((1 / (γ - 1)**2) + (2 / (γ - 1)) * (2 / (γ + 1))**((γ + 1) / (γ - 1)) * (1/PeAe_P0eAe_star)**2)
    
    return term_1 + term_2

Me = np.sqrt(Me_squared(PeAe_P0eAe_star))  # Exit Mach number
print(f"Exit Mach number:{Me:.2f}")

P0e = Pe * P0_P(Me)  # Exit total pressure

Mx = solve_mach_P0_ratio(P0e / P0, M_guess=1.8)  # Solve for Mach number just before the shock
print(f"Mach number just before the shock:{Mx:.2f}")

My = after_shock_mach_number(Mx)  # Mach number just after the shock
print(f"Mach number just after the shock:{My:.2f}")


# Solve for Mach Number before the throat
M_supersonic_case = np.zeros_like(x)
M_analytical = np.zeros_like(x)

# Subsonic Solution before the throat
for i in range(throat_index):
    M_supersonic_case[i] = solve_mach_A_ratio(A[i] / At, M_guess=0.2)

# Sonic Solution at the throat
M_supersonic_case[throat_index] = 1.0

# Supersonic Solution after the throat till the shock using Pe and P0
for i in range(throat_index + 1, Nx):
    M_supersonic_case[i] = solve_mach_A_ratio(A[i] / At, M_guess=2.0)

shock_index = np.where(M_supersonic_case > Mx)[0][0]

print(f"Shock location at x = {x[shock_index]:.2f} m")

for i in range(shock_index):
    M_analytical[i] = M_supersonic_case[i]

for i in range(shock_index, Nx):
    M_analytical[i] = solve_mach_A_ratio(A[i] / At, M_guess=0.2)  # Subsonic guess after shock

P_analytical = np.zeros_like(x)
T_analytical = np.zeros_like(x)
for i in range(shock_index):
    P_analytical[i] = P0 / P0_P(M_analytical[i])
    T_analytical[i] = T0 / T0_T(M_analytical[i])
for i in range(shock_index, Nx):
    P_analytical[i] = P0e / P0_P(M_analytical[i])
    T_analytical[i] = T0 / T0_T(M_analytical[i])

P_supersonic_case = P0 / P0_P(M_supersonic_case)
T_supersonic_case = T0 / T0_T(M_supersonic_case)

M_subsonic_case = np.zeros_like(x)
for i in range(throat_index):
    M_subsonic_case[i] = solve_mach_A_ratio(A[i] / At, M_guess=0.2)
M_subsonic_case[throat_index] = 1.0
for i in range(throat_index + 1, Nx):
    M_subsonic_case[i] = solve_mach_A_ratio(A[i] / At, M_guess=0.2)

P_subsonic_case = P0 / P0_P(M_subsonic_case)
T_subsonic_case = T0 / T0_T(M_subsonic_case)


# Plotting
fig, axes = plt.subplots(3, 1, figsize=(12, 12))

# Plot Mach number
axes[0].plot(x, M, linewidth=2)
axes[0].plot(x, M_analytical, '--', linewidth=2)
axes[0].plot(x, M_supersonic_case, 'r--', linewidth=2)
axes[0].plot(x, M_subsonic_case, 'g--', linewidth=2)
axes[0].set_ylabel("Mach Number")
axes[0].grid()
axes[0].set_title("Mach Number Distribution")

# Plot Pressure Ratio
axes[1].plot(x, P / P0, linewidth=2)
axes[1].plot(x, P_analytical / P0, '--', linewidth=2)
axes[1].plot(x, P_supersonic_case / P0, 'r--', linewidth=2)
axes[1].plot(x, P_subsonic_case / P0, 'g--', linewidth=2)
axes[1].set_ylabel("P / P0")
axes[1].grid()
axes[1].set_title("Pressure Ratio Distribution")

# Plot Temperature Ratio
axes[2].plot(x, T / T0, linewidth=2)
axes[2].plot(x, T_analytical / T0, '--', linewidth=2)
axes[2].plot(x, T_supersonic_case / T0, 'r--', linewidth=2)
axes[2].plot(x, T_subsonic_case / T0, 'g--', linewidth=2)
axes[2].set_xlabel("x (m)")
axes[2].set_ylabel("T / T0")
axes[2].grid()
axes[2].set_title("Temperature Ratio Distribution")

# Common Legend at top
labels = ["Numerical", "Analytical", "Idealized Supersonic", "Idealized Subsonic"]
fig.legend(labels, loc="upper center", ncol=3, fontsize=12, frameon=True)
plt.tight_layout(rect=[0, 0, 1, 0.96])  # leave space for the legend
plt.show()