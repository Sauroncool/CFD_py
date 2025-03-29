# Compressible Relations

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

def T0_T(γ,M):
    return 1 + 0.5 * (γ - 1) * M**2

def P0_P(γ,M):
    return (1 + 0.5 * (γ - 1) * M**2)**(γ/(γ - 1))

def rho0_rho(γ,M):
    return (1 + 0.5 * (γ - 1) * M**2)**(1/(γ - 1))

def A_Astar(M):
    return (2/(γ + 1)) * ((1 + 0.5 * (γ - 1) * M**2)**((γ + 1)/(2*(γ - 1))) / M)

if __name__ == "__main__":

    γ = 1.4
    R = 287.0

    # Reservoir conditions
    P0 = 1.0133e5
    T0 = 300.0

    # Stagnation density
    ρ0 = P0/(R*T0)

    Pe = 0.585 * P0

    # Initial guess for M
    # Solve for M using fsolve
    M_solution = fsolve(lambda M: P0_P(γ, M) - P0/Pe, 0.7)
    M_e = M_solution[0]

    print("Mach number at the exit (M_e):", round(M_e, 2))
    # Calculate Te and ρe
    Te = T0/T0_T(γ, M_e)
    ρe = ρ0/rho0_rho(γ, M_e)

    print("Exit temperature (Te):", round(Te, 2), "K")
    print("Exit pressure (Pe):", round(Pe, 2), "Pa")
    print("Exit density (ρe):", round(ρe, 2), "kg/m^3")
    print("Exit velocity (ue):", round((M_e*np.sqrt(γ * R * Te)), 2), "m/s")
