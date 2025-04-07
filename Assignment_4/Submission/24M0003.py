import matplotlib.pyplot as plt
import numpy as np
import time
from scipy.optimize import fsolve

class GasProperties:
    """Class to store gas properties and related thermodynamic relations"""
    def __init__(self, γ=1.4, R=287.05):
        self.γ = γ          # Specific heat ratio
        self.R = R          # Specific gas constant (J/kg·K)
    
    def P0_P(self, M):
        """Total to static pressure ratio"""
        return (1 + 0.5 * (self.γ - 1) * M**2) ** (self.γ / (self.γ - 1))
    
    def T0_T(self, M):
        """Total to static temperature ratio"""
        return 1 + 0.5 * (self.γ - 1) * M**2
    
    def A_Astar(self, M):
        """Area ratio for isentropic flow"""
        return ((2 / (self.γ + 1)) * (1 + 0.5 * (self.γ - 1) * M**2)) ** ((self.γ + 1) / (2 * (self.γ - 1))) / M
    
    def after_shock_mach(self, M1):
        """Mach number after normal shock"""
        return np.sqrt((1 + ((self.γ - 1) / 2) * M1**2) / (self.γ * M1**2 - (self.γ - 1) / 2))
    
    def P02_P01(self, M1):
        """Total pressure ratio across shock"""
        term1 = (( (self.γ + 1) / 2 ) * M1**2) / (1 + ( (self.γ - 1) / 2 ) * M1**2)
        term2 = ( (2 * self.γ / (self.γ + 1)) * M1**2 - (self.γ - 1) / (self.γ + 1) )
        return (term1 ** (self.γ / (self.γ - 1))) * (term2 ** (1 / (1 - self.γ)))
    
    def Me_squared(self, PeAe_P0eAe_star):
        """Exit Mach number squared for given pressure-area ratio"""
        term_1 = -1 / (self.γ - 1)
        term_2 = np.sqrt((1 / (self.γ - 1)**2) + (2 / (self.γ - 1)) * 
                         (2 / (self.γ + 1))**((self.γ + 1) / (self.γ - 1)) * 
                         (1/PeAe_P0eAe_star)**2)
        return term_1 + term_2

class NozzleGeometry:
    """Class to handle nozzle geometry and grid generation"""
    def __init__(self, length=2.0, num_points=101):
        self.length = length
        self.num_points = num_points
        self.x = np.linspace(0, length, num_points)
        self.Δx = length / (num_points - 1)
    
    def area_distribution(self, x):
        """Area distribution function for CD nozzle"""
        return 1.0 + 2.0 * (x - 1.0) ** 2
    
    def setup_geometry(self):
        """Calculate area distribution and its derivative"""
        self.A = self.area_distribution(self.x)
        self.dA_dx = np.gradient(self.A, self.Δx)
        self.throat_index = np.argmin(self.A)
        return self.A, self.dA_dx, self.throat_index

class FlowSolver:
    """Class to handle the flow solution using Flux Vector Splitting"""
    def __init__(self, gas_props, nozzle_geom):
        self.gas = gas_props
        self.nozzle = nozzle_geom
        self.CFL = 0.95
        self.tol = 1e-3 # Convergence tolerance
        self.iter_max = 50000
    
    def create_state_vectors(self, P, ρ, u, A):
        """Create conservative variable vector"""
        E = P / (self.gas.γ - 1) + 0.5 * ρ * u**2  # Total energy
        return np.array([ρ * A, ρ * u * A, E * A])  # Conservative variables
    
    def extract_variables(self, U, A):
        """Extract primitive variables from conservative variables"""
        ρ = U[0, :] / A
        u = U[1, :] / U[0, :]
        E = U[2, :] / A
        P = (self.gas.γ - 1) * (E - 0.5 * ρ * u**2)
        T = P / (ρ * self.gas.R)
        a = np.sqrt(self.gas.γ * self.gas.R * T)
        M = u / a
        return ρ, u, E, P, T, a, M
    
    def flux_vector_splitting(self, U, A):
        """Flux Vector Splitting scheme"""
        ρ, u, E, P, _, a, M = self.extract_variables(U, A)
        
        F = np.array([ρ * u * A, (ρ * u**2 + P) * A, (E + P) * u * A])  # Flux vector
        F_plus = np.zeros((3, self.nozzle.num_points))
        F_minus = np.zeros((3, self.nozzle.num_points))
        
        for i in range(self.nozzle.num_points):
            if M[i] <= -1:
                F_plus[:, i] = np.zeros(3)
                F_minus[:, i] = F[:, i]
            elif M[i] >= 1:
                F_plus[:, i] = F[:, i]
                F_minus[:, i] = np.zeros(3)
            else:
                α = 0.25 * ρ[i] * a[i] * (M[i] + 1) ** 2 * A[i]
                F_plus[0, i] = α
                F_plus[1, i] = α * (2 * a[i] / self.gas.γ) * (1 + ((self.gas.γ - 1) / 2) * M[i])
                F_plus[2, i] = (α * (2 * (a[i] ** 2) / (self.gas.γ**2 - 1)) * 
                               (1 + ((self.gas.γ - 1) / 2) * M[i]) ** 2)
                F_minus[:, i] = F[:, i] - F_plus[:, i]
        return F_plus, F_minus
    
    def apply_boundary_conditions(self, U, P0, T0, Pe):
        """Apply inlet and outlet boundary conditions"""
        ρ, u, E, P, T, _, M = self.extract_variables(U, self.nozzle.A)
        
        # Inlet boundary (reservoir conditions)
        u[0] = u[1]
        T[0] = (2 * self.gas.γ * self.gas.R * T0 - (self.gas.γ - 1) * u[0]**2) / (2 * self.gas.γ * self.gas.R)
        M[0] = u[0] / np.sqrt(self.gas.γ * self.gas.R * T[0])
        P[0] = P0 / self.gas.P0_P(M[0])
        ρ[0] = P[0] / (self.gas.R * T[0])
        
        # Outlet boundary (fixed pressure)
        P[-1] = Pe
        ρ[-1] = ρ[-2]
        u[-1] = u[-2]
        
        return self.create_state_vectors(P, ρ, u, self.nozzle.A)
    
    def solve_flow(self, P0, T0, Pe):
        """Main solver routine"""
        # Initial conditions
        M = np.zeros_like(self.nozzle.x)
        T = T0 / self.gas.T0_T(M)
        P = P0 / self.gas.P0_P(M)
        ρ = P / (self.gas.R * T)
        u = np.sqrt(self.gas.γ * self.gas.R * T) * M
        a = np.sqrt(self.gas.γ * self.gas.R * T)
        
        U = self.create_state_vectors(P, ρ, u, self.nozzle.A)
        
        start_time = time.time()
        for iter in range(self.iter_max):
            U_old = U.copy()
            
            # Boundary conditions
            U = self.apply_boundary_conditions(U, P0, T0, Pe)
            
            # Calculate time step
            λ_max = np.max(np.abs(u) + a)
            Δt = self.CFL * self.nozzle.Δx / λ_max
            
            # Compute fluxes
            F_plus, F_minus = self.flux_vector_splitting(U, self.nozzle.A)
            
            # Source term (area change)
            S = np.array([np.zeros(self.nozzle.num_points), 
                          P * self.nozzle.dA_dx, 
                          np.zeros(self.nozzle.num_points)])
            
            # Update solution
            U[:, 1:-1] = (U[:, 1:-1] - 
                         (Δt / self.nozzle.Δx) * (F_plus[:, 1:-1] - F_plus[:, :-2]) -
                         (Δt / self.nozzle.Δx) * (F_minus[:, 2:] - F_minus[:, 1:-1]) +
                         Δt * S[:, 1:-1])
            
            # Extract the solution
            ρ, u, E, P, T, a, M = self.extract_variables(U, self.nozzle.A)
            
            # Check convergence
            error = np.linalg.norm(U - U_old)
            if iter % 100 == 0:
                print(f"Iteration {iter}, Error: {error:.6e}")
            if error < self.tol:
                print(f"Converged in {iter} iterations. Error: {error:.6e}")
                break
        
        end_time = time.time()
        print(f"Time taken: {end_time - start_time:.2f} seconds")
        
        return ρ, u, P, T, M

class AnalyticalSolution:
    """Class to compute analytical solution for comparison"""
    def __init__(self, gas_props, nozzle_geom):
        self.gas = gas_props
        self.nozzle = nozzle_geom
    
    def solve_mach_A_ratio(self, A_ratio, M_guess):
        """Solve for Mach number given area ratio"""
        def eqn(M):
            return A_ratio - self.gas.A_Astar(M)
        return fsolve(eqn, M_guess)[0]
    
    def solve_mach_P0_ratio(self, P0_ratio, M_guess):
        """Solve for Mach number given total pressure ratio"""
        def eqn(M):
            return P0_ratio - self.gas.P02_P01(M)
        return fsolve(eqn, M_guess)[0]
    
    def compute_analytical_solution(self, P0, T0, Pe_P0):
        """Compute analytical solution for given pressure ratio"""
        Pe = Pe_P0 * P0
        At = self.nozzle.A[self.nozzle.throat_index]
        PeAe_P0eAe_star = Pe_P0 * (self.nozzle.A[-1]/At)
        
        # Exit Mach number
        Me = np.sqrt(self.gas.Me_squared(PeAe_P0eAe_star))
        P0e = Pe * self.gas.P0_P(Me)
        
        # Shock location and properties
        Mx = self.solve_mach_P0_ratio(P0e / P0, M_guess=1.8)
        My = self.gas.after_shock_mach(Mx)
        
        # Initialize arrays
        M_analytical = np.zeros_like(self.nozzle.x)
        M_supersonic = np.zeros_like(self.nozzle.x)
        M_subsonic = np.zeros_like(self.nozzle.x)
        
        # Subsonic before throat
        for i in range(self.nozzle.throat_index):
            M_supersonic[i] = self.solve_mach_A_ratio(self.nozzle.A[i]/At, 0.2)
        
        # Sonic at throat
        M_supersonic[self.nozzle.throat_index] = 1.0
        
        # Supersonic after throat
        for i in range(self.nozzle.throat_index, self.nozzle.num_points):
            M_supersonic[i] = self.solve_mach_A_ratio(self.nozzle.A[i]/At, 1.8)
        
        # Find shock location
        shock_index = np.where(M_supersonic > Mx)[0][0]
        
        # Construct full solution
        M_analytical[:shock_index] = M_supersonic[:shock_index]
        
        # After shock
        A_star_new = self.nozzle.A[shock_index]/self.gas.A_Astar(My)
        for i in range(shock_index, self.nozzle.num_points):
            M_analytical[i] = self.solve_mach_A_ratio(self.nozzle.A[i]/A_star_new, 0.2)
        
        # Compute pressure and temperature distributions
        P_analytical = np.zeros_like(self.nozzle.x)
        T_analytical = np.zeros_like(self.nozzle.x)
        
        for i in range(shock_index):
            P_analytical[i] = P0 / self.gas.P0_P(M_analytical[i])
            T_analytical[i] = T0 / self.gas.T0_T(M_analytical[i])
        
        for i in range(shock_index, self.nozzle.num_points):
            P_analytical[i] = P0e / self.gas.P0_P(M_analytical[i])
            T_analytical[i] = T0 / self.gas.T0_T(M_analytical[i])
        
        P_supersonic = P0 / self.gas.P0_P(M_supersonic)
        T_supersonic = T0 / self.gas.T0_T(M_supersonic)
        
        for i in range(self.nozzle.throat_index):
            M_subsonic[i] = self.solve_mach_A_ratio(self.nozzle.A[i]/At, 0.2)
        
        M_subsonic[self.nozzle.throat_index] = 1.0
        
        for i in range(self.nozzle.throat_index, self.nozzle.num_points):
            M_subsonic[i] = self.solve_mach_A_ratio(self.nozzle.A[i]/At, 0.2)
        
        P_subsonic = P0 / self.gas.P0_P(M_subsonic)
        T_subsonic = T0 / self.gas.T0_T(M_subsonic)
        
        return (M_analytical, P_analytical, T_analytical,
                M_supersonic, P_supersonic, T_supersonic,
                M_subsonic, P_subsonic, T_subsonic,
                shock_index, Me, Mx, My)

class PostProcessor:
    """Class for post-processing and visualization"""
    @staticmethod
    def plot_solution(x, numerical, analytical, supersonic, subsonic, titles, ylabels, filename):
        """Create comparison plots"""
        fig, axes = plt.subplots(3, 1, figsize=(12, 12))
        
        # Plot Mach number
        axes[0].plot(x, numerical[0], linewidth=2)
        axes[0].plot(x, analytical[0], '--', linewidth=2)
        axes[0].plot(x, supersonic[0], 'r--', linewidth=2)
        axes[0].plot(x, subsonic[0], 'g--', linewidth=2)
        axes[0].set_ylabel(ylabels[0])
        axes[0].grid()
        axes[0].set_title(titles[0])
        
        # Plot Pressure Ratio
        axes[1].plot(x, numerical[1], linewidth=2)
        axes[1].plot(x, analytical[1], '--', linewidth=2)
        axes[1].plot(x, supersonic[1], 'r--', linewidth=2)
        axes[1].plot(x, subsonic[1], 'g--', linewidth=2)
        axes[1].set_ylabel(ylabels[1])
        axes[1].grid()
        axes[1].set_title(titles[1])
        
        # Plot Temperature Ratio
        axes[2].plot(x, numerical[2], linewidth=2)
        axes[2].plot(x, analytical[2], '--', linewidth=2)
        axes[2].plot(x, supersonic[2], 'r--', linewidth=2)
        axes[2].plot(x, subsonic[2], 'g--', linewidth=2)
        axes[2].set_xlabel("x (m)")
        axes[2].set_ylabel(ylabels[2])
        axes[2].grid()
        axes[2].set_title(titles[2])
        
        # Common Legend
        labels = ["Numerical", "Analytical", "Idealized Supersonic", "Idealized Subsonic"]
        fig.legend(labels, loc="upper center", ncol=4, fontsize=12, frameon=True)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def plot_individual_results(x, y, xlabel, ylabel, title, filename):
        """Create individual plots"""
        plt.figure(figsize=(6, 4))
        plt.plot(x, y)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid()
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

def main():
    """Main function to run the nozzle flow simulation"""
    # Initialize classes
    air = GasProperties(γ=1.4, R=287.05)
    nozzle = NozzleGeometry(length=2.0, num_points=101)
    A, dA_dx, throat_index = nozzle.setup_geometry()
    
    # Reservoir conditions
    P0 = 1.0133e5      # Reservoir pressure (Pa)
    T0 = 300.0         # Reservoir temperature (K)
    Pe_P0 = 0.585      # Exit pressure ratio
    Pe = Pe_P0 * P0    # Outlet pressure (Pa)
    
    # Solve numerically
    solver = FlowSolver(air, nozzle)
    ρ, u, P, T, M = solver.solve_flow(P0, T0, Pe)
    
    # Compute analytical solution
    analytical = AnalyticalSolution(air, nozzle)
    (M_analytical, P_analytical, T_analytical,
     M_supersonic, P_supersonic, T_supersonic,
     M_subsonic, P_subsonic, T_subsonic,
     shock_index, Me, Mx, My) = analytical.compute_analytical_solution(P0, T0, Pe_P0)

    # Print results
    print("\nNumerical Solution:")
    print(f"Exit Mach number: {M[-1]:.2f}")
    num_shock_index = None
    for i in range(1, nozzle.num_points):
        if M[i - 1] > 1 and M[i] < 1:
            num_shock_index = i
            break
    if num_shock_index:
        print(f"Mach number before the shock: {M[num_shock_index - 2]:.2f}")
        print(f"Mach number after the shock: {M[num_shock_index + 1]:.2f}")
        print(f"Shock location at x = {nozzle.x[num_shock_index]:.2f} m")
    else:
        print("No shock detected in the solution.")
    
    print("\nAnalytical Solution:")
    print(f"Exit Mach number: {Me:.2f}")
    print(f"Mach before shock: {Mx:.2f}")
    print(f"Mach after shock: {My:.2f}")
    print(f"Shock location at x = {nozzle.x[shock_index]:.2f} m")
    
    # Prepare data for plotting
    numerical_data = (M, P/P0, T/T0)
    analytical_data = (M_analytical, P_analytical/P0, T_analytical/T0)
    supersonic_data = (M_supersonic, P_supersonic/P0, T_supersonic/T0)
    subsonic_data = (M_subsonic, P_subsonic/P0, T_subsonic/T0)
    
    # Create plots
    titles = ["Mach Number Distribution", "Pressure Ratio Distribution", "Temperature Ratio Distribution"]
    ylabels = ["Mach Number", "P/P0", "T/T0"]
    PostProcessor.plot_solution(nozzle.x, numerical_data, analytical_data, 
                              supersonic_data, subsonic_data, titles, ylabels, 
                              "CD_nozzle_results.png")
    
    # Individual plots
    PostProcessor.plot_individual_results(nozzle.x, P/P0, "x (m)", "Pressure Ratio (P/P0)", 
                                        "Pressure Ratio", "pressure_ratio.png")
    PostProcessor.plot_individual_results(nozzle.x, T/T0, "x (m)", "Temperature Ratio (T/T0)", 
                                        "Temperature Ratio", "temperature_ratio.png")
    PostProcessor.plot_individual_results(nozzle.x, M, "x (m)", "Mach Number", 
                                        "Mach Number Distribution", "mach_number.png")

if __name__ == "__main__":
    main()