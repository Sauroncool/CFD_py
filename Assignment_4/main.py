# Initial conditions
P = np.ones(Nx) * P_0
T = np.ones(Nx) * T0
rho = np.ones(Nx) * rho0
u = np.zeros(Nx)

U = np.zeros((Nx, 3))
F = np.zeros((Nx, 3))

# Compute flux
def compute_flux(U):
    rho, rho_u, E = U.T
    u = rho_u / rho
    p = (γ - 1) * (E - 0.5 * rho * u**2)
    F = np.zeros_like(U)
    F[:, 0] = rho_u
    F[:, 1] = rho_u * u + p
    F[:, 2] = u * (E + p)
    return F

# van Leer Flux Splitting
def van_leer_flux(U):
    rho, rho_u, E = U.T
    u = rho_u / rho
    p = (γ - 1) * (E - 0.5 * rho * u**2)
    a = np.sqrt(γ * p / rho)
    M = u / a
    
    F = compute_flux(U)
    Fp = 0.5 * (F + a[:, None] * U) * (1 + np.sign(M)[:, None])
    Fm = 0.5 * (F - a[:, None] * U) * (1 - np.sign(M)[:, None])
    
    return Fp, Fm

# Apply boundary conditions
def apply_boundary_conditions(U):
    rho, rho_u, E = U.T
    u = rho_u / rho
    
    # Inlet
    U[0, 1] = U[1, 1]
    M_in = u[0] / np.sqrt(γ * R * T0)
    P[0] = P_0 / P0_P(γ, M_in)
    T[0] = T0 / T0_T(γ, M_in)
    rho[0] = P[0] / (R * T[0])
    U[0, 0] = rho[0]
    U[0, 2] = P[0] / (γ - 1) + 0.5 * rho[0] * U[0, 1]**2
    
    # Outlet
    P[-1] = Pe
    rho[-1] = rho[-2]
    U[-1, 1] = U[-2, 1]
    U[-1, 0] = rho[-1]
    U[-1, 2] = P[-1] / (γ - 1) + 0.5 * rho[-1] * U[-1, 1]**2

# Solve using van Leer FVS
def solve():
    CFL = 0.5
    t_final = 0.1
    t = 0.0
    
    while t < t_final:
        Fp, Fm = van_leer_flux(U)
        
        # Compute time step using CFL condition
        u = U[:, 1] / U[:, 0]
        a = np.sqrt(γ * (γ - 1) * (U[:, 2] - 0.5 * U[:, 0] * u**2) / U[:, 0])
        dt = CFL * Δx / np.max(np.abs(u) + a)
        
        # Update solution using van Leer FVS scheme
        U[1:-1] -= dt / Δx * (Fp[1:-1] - Fm[:-2])
        apply_boundary_conditions(U)
        
        t += dt
        
    return U

U_final = solve()

plt.plot(x, U_final[:, 1] / U_final[:, 0], label="Velocity")
plt.xlabel("x")
plt.ylabel("Velocity")
plt.legend()
plt.show()
