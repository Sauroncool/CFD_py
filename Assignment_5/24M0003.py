import numpy as np

num_x = 31  # Number of grid points in x direction
num_y = 31  # Number of grid points in y direction
L = 1.0     # Length of the cavity
H = 1.0     # Height of the cavity

dx = L / (num_x - 1)  # Grid spacing in x direction
dy = H / (num_y - 1)  # Grid spacing in y direction

Re = 100  # Reynolds number
U0 = 1.0  # Lid velocity

ν = U0 * L / Re  # Kinematic viscosity (nu)

iter_max = 50000  # Maximum number of iterations
tol = 1e-8       # Convergence tolerance

σ_c = 0.4  # Courant number for convection (sigma_c)
σ_d = 0.6  # Courant number for diffusion (sigma_d)

# Allocate arrays
ψ = np.full((num_x, num_y), 100.0)  # Stream function (ψ)
ω = np.zeros((num_x, num_y))       # Vorticity (ω)
u = np.zeros((num_x, num_y))       # Velocity in x direction
v = np.zeros((num_x, num_y))       # Velocity in y direction

# History for animation
ψ_history = np.zeros((iter_max, num_x, num_y))  # Stream function history (ψ)
ω_history = np.zeros((iter_max, num_x, num_y))  # Vorticity history (ω)
u_history = np.zeros((iter_max, num_x, num_y))  # Velocity x-component history
v_history = np.zeros((iter_max, num_x, num_y))  # Velocity y-component history
time_history = np.zeros(iter_max)
error_history = np.zeros((iter_max,2))


def boundary_conditions(ψ, ω, dx, dy):
    # Lid: Top boundary
    ω[1:-1, -1] = -(2.0 * (ψ[1:-1, -2] - ψ[1:-1, -1])) / (dy ** 2) - (2.0 * U0) / dy
    # Bottom boundary
    ω[1:-1, 0] = -(2.0 * (ψ[1:-1, 1] - ψ[1:-1, 0])) / (dy ** 2)
    # Left boundary
    ω[0, 1:-1] = -2.0 * (ψ[1, 1:-1] - ψ[0, 1:-1]) / (dx ** 2)
    # Right boundary
    ω[-1, 1:-1] = -2.0 * (ψ[-2, 1:-1] - ψ[-1, 1:-1]) / (dx ** 2)
    return ω

# Write solve_stream_function in vectorized form
def solve_stream_function(ψ, ω, dx, dy, max_iter=4000, tolerance=1e-2):
    β = dx / dy
    β_sq = β * β

    iter = 0
    max_residual = 1.0
    β

    while max_residual > tolerance and iter < max_iter:
        ψ_old = np.copy(ψ)
        ψ[1:-1, 1:-1] = (
            (ψ[2:, 1:-1] + ψ[:-2, 1:-1]) +
            β_sq * (ψ[1:-1, 2:] + ψ[1:-1, :-2]) +
            dx * dx * ω[1:-1, 1:-1]
        ) / (2.0 * (1.0 + β_sq))

        max_residual = np.max(np.abs(ψ - ψ_old))

        iter += 1
    return ψ


def compute_velocity(ψ, dx, dy):
    # Compute velocity components from stream function (ψ)
    u[1:-1, 1:-1] = (ψ[1:-1, 2:] - ψ[1:-1, :-2]) / (2 * dy)
    v[1:-1, 1:-1] = -(ψ[2:, 1:-1] - ψ[:-2, 1:-1]) / (2 * dx)

    # Apply boundary conditions
    u[1:-1, 0] = 0.0
    u[1:-1, -1] = U0
    u[0, :] = 0.0
    u[-1, :] = 0.0

    v[1:-1, 0] = 0.0
    v[1:-1, -1] = 0.0
    v[0, :] = 0.0
    v[-1, :] = 0.0
    return u, v


def compute_time_step(u, v, dx, dy):
    # Compute time step from CFL condition using σ_c and σ_d
    u_max = np.max(abs(u))
    v_max = np.max(abs(v))
    dt_c = σ_c * dx * dy / (u_max * dy + v_max * dx)
    dt_d = σ_d * (1.0 / (2.0 * ν)) * (dx**2 * dy**2) / (dx**2 + dy**2)
    return min(dt_c, dt_d)


def vorticity_transport_equation(ψ, ω, dx, dy, dt):
    u, v = compute_velocity(ψ, dx, dy)
    ω_new = np.copy(ω)

    I, J = ω.shape

    convection = np.zeros_like(ω)

    for i in range(1, I-1):
        for j in range(1, J-1):
            # Convection term
            if u[i, j] > 0:
                if i >= 2:
                    dw_dx = (3 * ω[i, j] - 4 * ω[i-1, j] + ω[i-2, j]) / (2 * dx)
                else:
                    dw_dx = (ω[i, j] - ω[i-1, j]) / dx
            else:
                if i <= I - 3:
                    dw_dx = (-3 * ω[i, j] + 4 * ω[i+1, j] - ω[i+2, j]) / (2 * dx)
                else:
                    dw_dx = (ω[i+1, j] - ω[i, j]) / dx

            if v[i, j] > 0:
                if j >= 2:
                    dw_dy = (3 * ω[i, j] - 4 * ω[i, j-1] + ω[i, j-2]) / (2 * dy)
                else:
                    dw_dy = (ω[i, j] - ω[i, j-1]) / dy
            else:
                if j <= J - 3:
                    dw_dy = (-3 * ω[i, j] + 4 * ω[i, j+1] - ω[i, j+2]) / (2 * dy)
                else:
                    dw_dy = (ω[i, j+1] - ω[i, j]) / dy

            convection[i, j] = u[i, j] * dw_dx + v[i, j] * dw_dy

    # Diffusion term
    diffusion = (ω[2:, 1:-1] - 2 * ω[1:-1, 1:-1] + ω[:-2, 1:-1]) / dx**2 + \
                (ω[1:-1, 2:] - 2 * ω[1:-1, 1:-1] + ω[1:-1, :-2]) / dy**2

    # Update vorticity
    ω_new[1:-1, 1:-1] = ω[1:-1, 1:-1] + dt * (ν * diffusion - convection[1:-1, 1:-1])

    return ω_new



iter = 0
time = 0.0

for iter in range(iter_max):
    u_old = np.copy(u)
    v_old = np.copy(v)

    ω = boundary_conditions(ψ, ω, dx, dy)
    u, v = compute_velocity(ψ, dx, dy)
    dt = compute_time_step(u, v, dx, dy)
    time += dt
    ω = vorticity_transport_equation(ψ, ω, dx, dy, dt)
    ψ = solve_stream_function(ψ, ω, dx, dy)
    u, v = compute_velocity(ψ, dx, dy)

    ψ_history[iter] = ψ
    ω_history[iter] = ω
    u_history[iter] = u
    v_history[iter] = v
    time_history[iter] = time

    rms_u = np.sqrt(np.sum((u - u_old)**2) / (num_x * num_y))
    rms_v = np.sqrt(np.sum((v - v_old)**2) / (num_x * num_y))
    error_history[iter] = [rms_u, rms_v]

    if iter % 10 == 0:
        print(f"Iteration {iter}: RMS u = {rms_u:.8f}, RMS v = {rms_v:.8f}, Time = {time:.3f} s")

    if rms_u < tol and rms_v < tol:
        print(f"Converged after {iter} iterations")
        break


# Plotting
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.contourf(ψ.T, levels=20, cmap='viridis')
plt.colorbar(label='Stream function (ψ)')
plt.title('Stream function (ψ) contours')
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')
plt.savefig('stream_function_contours.png')
plt.show()

x = np.linspace(0, L, num_x)
y = np.linspace(0, H, num_y)
X, Y = np.meshgrid(x, y, indexing='ij')

plt.figure(figsize=(8, 6))
plt.streamplot(X.T, Y.T, u.T, v.T, color='k', density=1.5, linewidth=1)
plt.title('Streamlines (from ψ)')
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')
plt.grid(True)
plt.savefig('streamlines.png')
plt.show()

# Plot v at mid horizontal line
import pandas as pd
# Load Ghia et al. data
ghia_data = pd.read_excel("Mid horizontal line (y velocity) Ghia Ghia.xlsx")
x_ghia = ghia_data.iloc[:, 0].values  # x coordinate
v_ghia = ghia_data.iloc[:, 1].values  # v velocity

# Plotting comparison
plt.figure(figsize=(8, 6))
plt.scatter(x_ghia, v_ghia, label='Ghia et al. (Literature)', color='red', marker='o')
plt.plot(x, v[:, num_y//2], label='Computed v velocity', color='blue')
plt.title('v velocity along mid-horizontal line (x-axis)')
plt.xlabel('x')
plt.ylabel('v velocity')
plt.grid(True)
plt.legend()
plt.savefig('v_velocity_mid_horizontal_comparison.png')
plt.show()


# u velocity at mid vertical line
ghia_data_u = pd.read_excel("Mid vertical line (x velocity) Ghia Ghia.xlsx")
x_ghia_u = ghia_data_u.iloc[:, 0].values  # y coordinate
u_ghia = ghia_data_u.iloc[:, 1].values  # u velocity

# Plotting comparison
plt.figure(figsize=(8, 6))
plt.scatter(x_ghia_u, u_ghia, label='Ghia et al. (Literature)', color='red', marker='o')
plt.plot(y, u[num_x//2, :], label='Computed u velocity', color='blue')
plt.title('u velocity along mid-vertical line (y-axis)')
plt.xlabel('u velocity')
plt.ylabel('y')
plt.grid(True)
plt.legend()
plt.savefig('u_velocity_mid_vertical_comparison.png')
plt.show()

# Plot log error history
plt.figure(figsize=(8, 6))
plt.plot(np.log10(error_history[:iter, 0]), label='RMS u velocity')
plt.plot(np.log10(error_history[:iter, 1]), label='RMS v velocity')
plt.title('Log of RMS velocity error history')
plt.xlabel('Iteration')
plt.ylabel('Log10(RMS velocity)')
plt.grid(True)
plt.legend()
plt.savefig('log_error_history.png')
plt.show()


# Uncomment the following lines to create an animation of the streamlines and stream function contours
# # Animation
# from matplotlib.animation import FuncAnimation

# fig, ax = plt.subplots(figsize=(8, 6))
# frame_interval = 25
# frames = list(range(0, iter + 1, frame_interval))

# def update(frame):
#     ax.clear()
#     ax.contourf(X.T, Y.T, ψ_history[frame].T, levels=20, cmap='viridis')
#     ax.streamplot(X.T, Y.T, u_history[frame].T, v_history[frame].T, color='k', density=1.5, linewidth=1)
#     ax.set_title(f'Streamlines and ψ contours at t = {time_history[frame]:.3f} s')
#     ax.set_xlabel('x')
#     ax.set_ylabel('y')
#     ax.axis('equal')
#     ax.grid(True)

# ani = FuncAnimation(fig, update, frames=frames, interval=20)
# ani.save('streamlines_stream_function_animation.mp4', writer='ffmpeg', fps=10)
