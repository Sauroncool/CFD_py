import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt


# Isentropic Relations
def P0_P(M, γ=1.4):
    return (1 + 0.5 * (γ - 1) * M**2) ** (γ / (γ - 1))


def T0_T(M, γ=1.4):
    return 1 + 0.5 * (γ - 1) * M**2


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

γ = 1.4  # Specific heat ratio for air
R = 287.0  # Specific gas constant for air

# Reservoir conditions
P0 = 1.0133e5  # Pa
T0 = 300.0  # K

Pe_P0 = 0.585  # Exit pressure ratio

# Nozzle exit conditions
Pe = Pe_P0 * P0  # Pa

# Nozzle geometry
L = 2.0  # Length of the domain
Nx = 101  # Number of grid points
Δx = L / (Nx - 1)  # Grid spacing
x = np.linspace(0, L, Nx)  # Grid points


def Area(x):
    return 1.0 + 2.0 * (x - 1.0) ** 2  # Area function


A = Area(x)  # Area distribution

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

#shock_index = np.argmin(np.abs(M_supersonic_case - Mx))  # Index of the shock
shock_index = np.where(M_supersonic_case > Mx)[0][0]

print(f"Shock location at x = {x[shock_index]:.2f} m")

for i in range(shock_index):
    M_analytical[i] = M_supersonic_case[i]

A_star_new = A[shock_index]/A_Astar(My)
for i in range(shock_index, Nx):
    M_analytical[i] = solve_mach_A_ratio(A[i] / A_star_new, M_guess=0.2)  # Subsonic guess after shock

plt.figure(figsize=(10, 5))
plt.plot(x, M_supersonic_case, label="Fully Expanded Supersonic Solution", linestyle='--')
plt.plot(x, M_analytical, label="Analytical with Shock", linewidth=2)
plt.axvline(x[shock_index], color='red', linestyle=':', label="Shock Location")
plt.xlabel("x (m)")
plt.ylabel("Mach Number")
plt.title("Mach Number Distribution in a CD Nozzle with Normal Shock")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
