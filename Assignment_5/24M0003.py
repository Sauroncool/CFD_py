import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Tuple, Optional

class LidDrivenCavitySolver:
    def __init__(self, num_x: int = 31, num_y: int = 31, L: float = 1.0, H: float = 1.0,
                 Re: float = 100, U0: float = 1.0, σ_c: float = 0.4, σ_d: float = 0.6):
        """
        Initialize the lid-driven cavity flow solver.
        
        Parameters:
            num_x, num_y: Number of grid points in x and y directions
            L, H: Length and height of the cavity
            Re: Reynolds number
            U0: Lid velocity
            σ_c, σ_d: Courant numbers for convection and diffusion
        """
        self.num_x = num_x
        self.num_y = num_y
        self.L = L
        self.H = H
        self.Re = Re
        self.U0 = U0
        self.σ_c = σ_c
        self.σ_d = σ_d
        
        # Derived parameters
        self.dx = L / (num_x - 1)
        self.dy = H / (num_y - 1)
        self.ν = U0 * L / Re  # Kinematic viscosity
        
        # Initialize fields
        self.ψ = np.full((num_x, num_y), 100.0)  # Stream function
        self.ω = np.zeros((num_x, num_y))        # Vorticity
        self.u = np.zeros((num_x, num_y))        # x-velocity
        self.v = np.zeros((num_x, num_y))        # y-velocity
        
        # History tracking
        self.history = {
            'ψ': [],
            'ω': [],
            'u': [],
            'v': [],
            'time': [],
            'error': []
        }
        self.iter_count = 0
        self.time = 0.0
        
    def apply_boundary_conditions(self) -> None:
        """Apply boundary conditions to vorticity field."""
        # Lid: Top boundary
        self.ω[1:-1, -1] = -(2.0 * (self.ψ[1:-1, -2] - self.ψ[1:-1, -1])) / (self.dy ** 2) - (2.0 * self.U0) / self.dy
        # Bottom boundary
        self.ω[1:-1, 0] = -(2.0 * (self.ψ[1:-1, 1] - self.ψ[1:-1, 0])) / (self.dy ** 2)
        # Left boundary
        self.ω[0, 1:-1] = -2.0 * (self.ψ[1, 1:-1] - self.ψ[0, 1:-1]) / (self.dx ** 2)
        # Right boundary
        self.ω[-1, 1:-1] = -2.0 * (self.ψ[-2, 1:-1] - self.ψ[-1, 1:-1]) / (self.dx ** 2)
    
    def solve_stream_function(self, max_iter: int = 4000, tolerance: float = 1e-2) -> None:
        """
        Solve the stream function Poisson equation using iterative method.
        
        Parameters:
            max_iter: Maximum number of iterations
            tolerance: Convergence tolerance
        """
        β = self.dx / self.dy
        β_sq = β * β
        
        for _ in range(max_iter):
            ψ_old = np.copy(self.ψ)
            self.ψ[1:-1, 1:-1] = (
                (self.ψ[2:, 1:-1] + self.ψ[:-2, 1:-1]) +
                β_sq * (self.ψ[1:-1, 2:] + self.ψ[1:-1, :-2]) +
                self.dx * self.dx * self.ω[1:-1, 1:-1]
            ) / (2.0 * (1.0 + β_sq))
            
            if np.max(np.abs(self.ψ - ψ_old)) < tolerance:
                break
    
    def compute_velocity(self) -> None:
        """Compute velocity components from stream function."""
        # Interior points
        self.u[1:-1, 1:-1] = (self.ψ[1:-1, 2:] - self.ψ[1:-1, :-2]) / (2 * self.dy)
        self.v[1:-1, 1:-1] = -(self.ψ[2:, 1:-1] - self.ψ[:-2, 1:-1]) / (2 * self.dx)
        
        # Boundary conditions
        self.u[1:-1, 0] = 0.0
        self.u[1:-1, -1] = self.U0
        self.u[0, :] = 0.0
        self.u[-1, :] = 0.0
        
        self.v[1:-1, 0] = 0.0
        self.v[1:-1, -1] = 0.0
        self.v[0, :] = 0.0
        self.v[-1, :] = 0.0
    
    def compute_time_step(self) -> float:
        """Compute time step from CFL condition."""
        u_max = np.max(abs(self.u))
        v_max = np.max(abs(self.v))
        dt_c = self.σ_c * self.dx * self.dy / (u_max * self.dy + v_max * self.dx)
        dt_d = self.σ_d * (1.0 / (2.0 * self.ν)) * (self.dx**2 * self.dy**2) / (self.dx**2 + self.dy**2)
        return min(dt_c, dt_d)
    
    def solve_vorticity_transport(self, dt: float) -> None:
        """Solve the vorticity transport equation."""
        ω_new = np.copy(self.ω)
        I, J = self.ω.shape
        convection = np.zeros_like(self.ω)
        
        for i in range(1, I-1):
            for j in range(1, J-1):
                # Convection term - x direction
                if self.u[i, j] > 0:
                    if i >= 2:
                        dw_dx = (3 * self.ω[i, j] - 4 * self.ω[i-1, j] + self.ω[i-2, j]) / (2 * self.dx)
                    else:
                        dw_dx = (self.ω[i, j] - self.ω[i-1, j]) / self.dx
                else:
                    if i <= I - 3:
                        dw_dx = (-3 * self.ω[i, j] + 4 * self.ω[i+1, j] - self.ω[i+2, j]) / (2 * self.dx)
                    else:
                        dw_dx = (self.ω[i+1, j] - self.ω[i, j]) / self.dx
                
                # Convection term - y direction
                if self.v[i, j] > 0:
                    if j >= 2:
                        dw_dy = (3 * self.ω[i, j] - 4 * self.ω[i, j-1] + self.ω[i, j-2]) / (2 * self.dy)
                    else:
                        dw_dy = (self.ω[i, j] - self.ω[i, j-1]) / self.dy
                else:
                    if j <= J - 3:
                        dw_dy = (-3 * self.ω[i, j] + 4 * self.ω[i, j+1] - self.ω[i, j+2]) / (2 * self.dy)
                    else:
                        dw_dy = (self.ω[i, j+1] - self.ω[i, j]) / self.dy
                
                convection[i, j] = self.u[i, j] * dw_dx + self.v[i, j] * dw_dy
        
        # Diffusion term
        diffusion = (self.ω[2:, 1:-1] - 2 * self.ω[1:-1, 1:-1] + self.ω[:-2, 1:-1]) / self.dx**2 + \
                    (self.ω[1:-1, 2:] - 2 * self.ω[1:-1, 1:-1] + self.ω[1:-1, :-2]) / self.dy**2
        
        # Update vorticity
        ω_new[1:-1, 1:-1] = self.ω[1:-1, 1:-1] + dt * (self.ν * diffusion - convection[1:-1, 1:-1])
        self.ω = ω_new
    
    def record_history(self, rms_u: float, rms_v: float) -> None:
        """Record current state in history."""
        self.history['ψ'].append(np.copy(self.ψ))
        self.history['ω'].append(np.copy(self.ω))
        self.history['u'].append(np.copy(self.u))
        self.history['v'].append(np.copy(self.v))
        self.history['time'].append(self.time)
        self.history['error'].append([rms_u, rms_v])
    
    def compute_rms_velocity_error(self, u_old: np.ndarray, v_old: np.ndarray) -> Tuple[float, float]:
        """Compute RMS error in velocity fields."""
        rms_u = np.sqrt(np.sum((self.u - u_old)**2) / (self.num_x * self.num_y))
        rms_v = np.sqrt(np.sum((self.v - v_old)**2) / (self.num_x * self.num_y))
        return rms_u, rms_v
    
    def solve(self, iter_max: int = 50000, tol: float = 1e-8, 
              record_interval: int = 10) -> None:
        """
        Main solver loop.
        
        Parameters:
            iter_max: Maximum number of iterations
            tol: Convergence tolerance
            record_interval: Interval for recording history and animation
        """
        for self.iter_count in range(iter_max):
            u_old = np.copy(self.u)
            v_old = np.copy(self.v)
            
            # Solve sequence
            self.apply_boundary_conditions()
            self.compute_velocity()
            dt = self.compute_time_step()
            self.time += dt
            self.solve_vorticity_transport(dt)
            self.solve_stream_function()
            self.compute_velocity()
            
            # Compute errors
            rms_u, rms_v = self.compute_rms_velocity_error(u_old, v_old)
            
            # Record history
            if self.iter_count % record_interval == 0:
                self.record_history(rms_u, rms_v)
                print(f"Iteration {self.iter_count}: RMS u = {rms_u:.8f}, RMS v = {rms_v:.8f}, Time = {self.time:.3f} s")
            
            # Check convergence
            if rms_u < tol and rms_v < tol:
                print(f"Converged after {self.iter_count} iterations")
                self.record_history(rms_u, rms_v)  # Record final state
                break
    
    def plot_results(self) -> None:
        """Plot the results including comparisons with Ghia et al. data."""
        # Create grid
        x = np.linspace(0, self.L, self.num_x)
        y = np.linspace(0, self.H, self.num_y)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Stream function contours
        plt.figure(figsize=(8, 6))
        plt.contourf(self.ψ.T, levels=20, cmap='viridis')
        plt.colorbar(label='Stream function (ψ)')
        plt.title('Stream function (ψ) contours')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.axis('equal')
        plt.savefig('stream_function_contours.png')
        plt.show()
        
        # Streamlines
        plt.figure(figsize=(8, 6))
        plt.streamplot(X.T, Y.T, self.u.T, self.v.T, color='k', density=1.5, linewidth=1)
        plt.title('Streamlines (from ψ)')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.axis('equal')
        plt.grid(True)
        plt.savefig('streamlines.png', dpi=300)
        plt.show()
        
        # v-velocity comparison at mid-horizontal line
        try:
            ghia_data = pd.read_excel("Mid horizontal line (y velocity) Ghia Ghia.xlsx")
            x_ghia = ghia_data.iloc[:, 0].values
            v_ghia = ghia_data.iloc[:, 1].values
            
            plt.figure(figsize=(8, 6))
            plt.scatter(x_ghia, v_ghia, label='Ghia et al. (Literature)', color='red', marker='o')
            plt.plot(x, self.v[:, self.num_y//2], label='Computed v velocity', color='blue')
            plt.title('v velocity along mid-horizontal line (x-axis)')
            plt.xlabel('x')
            plt.ylabel('v velocity')
            plt.grid(True)
            plt.legend()
            plt.savefig('v_velocity_mid_horizontal_comparison.png', dpi=300)
            plt.show()
        except FileNotFoundError:
            print("Ghia data file not found, skipping comparison plot")
        
        # u-velocity comparison at mid-vertical line
        try:
            ghia_data_u = pd.read_excel("Mid vertical line (x velocity) Ghia Ghia.xlsx")
            x_ghia_u = ghia_data_u.iloc[:, 0].values
            u_ghia = ghia_data_u.iloc[:, 1].values
            
            plt.figure(figsize=(8, 6))
            plt.scatter(x_ghia_u, u_ghia, label='Ghia et al. (Literature)', color='red', marker='o')
            plt.plot(y, self.u[self.num_x//2, :], label='Computed u velocity', color='blue')
            plt.title('u velocity along mid-vertical line (y-axis)')
            plt.xlabel('u velocity')
            plt.ylabel('y')
            plt.grid(True)
            plt.legend()
            plt.savefig('u_velocity_mid_vertical_comparison.png', dpi=300)
            plt.show()
        except FileNotFoundError:
            print("Ghia data file not found, skipping comparison plot")
        
        # Plot log error history
        if len(self.history['error']) > 0:
            error_history = np.array(self.history['error'])
            plt.figure(figsize=(8, 6))
            plt.plot(np.log10(error_history[:, 0]), label='RMS u velocity')
            plt.plot(np.log10(error_history[:, 1]), label='RMS v velocity')
            plt.title('Log of RMS velocity error history')
            plt.xlabel('Iteration')
            plt.ylabel('Log10(RMS velocity)')
            plt.grid(True)
            plt.legend()
            plt.savefig('log_error_history.png', dpi=300)
            plt.show()
    
    def create_animation(self, filename: str = 'cavity_flow_animation.mp4', 
                         frame_interval: int = 2) -> None:
        """
        Create an animation of the solution evolution.
        
        Parameters:
            filename: Output filename for the animation
            frame_interval: Interval between frames in the animation
        """
        from matplotlib.animation import FuncAnimation
        
        if not self.history['ψ']:
            print("No history data available for animation")
            return
        
        # Create grid
        x = np.linspace(0, self.L, self.num_x)
        y = np.linspace(0, self.H, self.num_y)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Set up figure
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Select frames to animate
        frames = list(range(0, len(self.history['ψ']), frame_interval))
        if len(self.history['ψ']) - 1 not in frames:
            frames.append(len(self.history['ψ']) - 1)  # Include final frame
        
        def update(frame_idx):
            frame = frames[frame_idx]
            ax.clear()
            ax.contourf(X.T, Y.T, self.history['ψ'][frame].T, levels=20, cmap='viridis')
            ax.streamplot(X.T, Y.T, self.history['u'][frame].T, self.history['v'][frame].T, 
                         color='k', density=1.5, linewidth=1)
            ax.set_title(f'Streamlines and ψ contours at t = {self.history["time"][frame]:.3f} s')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.axis('equal')
            ax.grid(True)
        
        ani = FuncAnimation(fig, update, frames=len(frames), interval=1)
        ani.save(filename, writer='ffmpeg', fps=10)
        plt.close()


# Example usage
if __name__ == "__main__":
    # Create and run solver
    solver = LidDrivenCavitySolver(num_x=31, num_y=31, Re=100)
    solver.solve(iter_max=50000, tol=1e-8)
    
    # Plot results
    solver.plot_results()
    
    # Create animation (uncomment to use)
    #solver.create_animation()