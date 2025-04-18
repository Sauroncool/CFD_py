import numpy as np

num_x = 31 # Number of grid points in x direction
num_y = 31 # Number of grid points in y direction
L = 1.0 # Length of the cavity
H = 1.0 # Height of the cavity

dx = L / (num_x - 1) # Grid spacing in x direction
dy = H / (num_y - 1) # Grid spacing in y direction

Re = 100 # Reynolds number
U0 = 1.0 # Lid velocity

nu = U0 * L / Re # Kinematic viscosity

iter_max = 2000 # Maximum number of iterations
tol = 1e-8 # Convergence tolerance

sigma_c = 0.4 # courant number for convection
sigma_d = 0.6 # courant number for diffusion

# Allocate arrays
psi = np.full((num_x, num_y), 100.0) # Stream function
omega = np.zeros((num_x, num_y)) # Vorticity
u = np.zeros((num_x, num_y)) # Velocity in x direction
v = np.zeros((num_x, num_y)) # Velocity in y direction

psi_history = np.zeros((iter_max, num_x, num_y)) # Stream function history
omega_history = np.zeros((iter_max, num_x, num_y)) # Vorticity history
u_history = np.zeros((iter_max, num_x, num_y))  # Velocity x-component history
v_history = np.zeros((iter_max, num_x, num_y))  # Velocity y-component history

time = 0
time_history = np.zeros(iter_max)


def boundary_conditions(psi,omega,dx,dy):
    # Lid: Top boundary
    omega[1:-1, -1] = -(2.0 * (psi[1:-1, -2] - psi[1:-1, -1])) / (dy ** 2) - (2.0 * U0) / dy

    # Bottom boundary
    omega[1:-1, 0] = -(2.0 * (psi[1:-1,1]-psi[1:- 1, 0])) / (dy ** 2)

    # Left boundary
    omega[0, 1:-1] = -2.0 * (psi[1, 1:-1] - psi[0,1:-1]) / (dx ** 2)

    # Right boundary
    omega[-1, 1:-1] = -2.0 * (psi[-2, 1:-1] - psi[-1,1:-1]) / (dx ** 2)

    return psi, omega

def solve_stream_function(psi, omega, dx, dy, max_iter=1000, tolerance=1e-2):
    NX, NY = psi.shape
    beta = dx / dy
    beta_sq = beta * beta
    denom = 2.0 * (1.0 + beta_sq)

    iter = 0
    max_residual = 1.0

    while max_residual > tolerance and iter < max_iter:
        max_residual = 0.0

        for i in range(1, NX - 1):
            for j in range(1, NY - 1):
                psi_old = psi[i, j]
                psi[i, j] = (
                    (psi[i + 1, j] + psi[i - 1, j]) +
                    beta_sq * (psi[i, j + 1] + psi[i, j - 1]) +
                    dx * dx * omega[i, j]
                ) / denom

                residual = abs(psi[i, j] - psi_old)
                if residual > max_residual:
                    max_residual = residual

        iter += 1

    return psi



def compute_velocity(psi, dx, dy):
    # Compute velocity components from stream function
    u[1:-1, 1:-1] = (psi[1:-1, 2:] - psi[1:-1, :-2]) / (2 * dy)
    v[1:-1, 1:-1] = -(psi[2:, 1:-1] - psi[:-2, 1:-1]) / (2 * dx)

    # Apply boundary conditions for velocity
    u[1:-1, 0] = 0.0
    u[1:-1, -1] = U0
    u[0, :] = 0.0
    u[-1, :] = 0.0

    v[1:-1, 0] = 0.0
    v[1:-1, -1] = 0.0
    v[0, :] = 0.0
    v[-1, :] = 0.0

    return u, v

def compute_time_step(u,v, dx, dy):
    # Compute time step based on stability criteria
    u_max = np.max(abs(u))
    v_max = np.max(abs(v))
    dt_c = sigma_c * dx*dy / (u_max*dy + v_max*dx)
    dt_d = sigma_d * (1.0 / (2.0*nu)) * (dx*dx * dy*dy) / (dx*dx + dy*dy)
    return min(dt_c, dt_d)

def vorticity_transport_equation(psi,omega,dx,dy,dt):
    u,v = compute_velocity(psi, dx, dy) # Compute velocity from stream function
    omega_new = np.copy(omega) # Initialize new vorticity array

    for i in range(1, num_x - 1):
        for j in range(1, num_y - 1):
            # Compute convection term
            convection = u[i, j] * (omega[i, j + 1] - omega[i, j - 1]) / (2 * dy) + \
                        v[i, j] * (omega[i + 1, j] - omega[i - 1, j]) / (2 * dx)

            # Compute diffusion term
            diffusion = (omega[i + 1, j] - 2 * omega[i, j] + omega[i - 1, j]) / (dx ** 2) + \
                        (omega[i, j + 1] - 2 * omega[i, j] + omega[i, j - 1]) / (dy ** 2)

            # Update vorticity using explicit method
            omega_new[i, j] = omega[i, j] + dt * (nu * diffusion - convection)
            
    return omega_new


iter = 0 # Iteration counter

# Iteration loop
for iter in range(iter_max):
    u_old = np.copy(u) # Store old values of u for convergence check
    v_old = np.copy(v) # Store old values of v for convergence check

    psi, omega = boundary_conditions(psi, omega, dx, dy) # Apply boundary conditions

    u, v = compute_velocity(psi, dx, dy) # Compute velocity from stream function
    dt = compute_time_step(u, v, dx, dy) # Compute time step based on stability criteria
    omega = vorticity_transport_equation(psi, omega, dx, dy, dt) # Solve vorticity transport equation
    psi = solve_stream_function(psi, omega, dx, dy) # Solve stream function equation
    u, v = compute_velocity(psi, dx, dy) # Compute velocity from stream function

    # Add to history
    psi_history[iter] = psi
    omega_history[iter] = omega
    u_history[iter] = u
    v_history[iter] = v

    time += dt # Update time
    time_history[iter] = time # Store time history


    # Convergnece
    rms_u = np.sqrt(np.sum((u - u_old) ** 2) / (num_x * num_y))
    rms_v = np.sqrt(np.sum((v - v_old) ** 2) / (num_x * num_y))
    # Print every 10 iterations
    if iter % 10 == 0:
        print(f"Iteration {iter}, RMS u: {rms_u:.6f}, RMS v: {rms_v:.6f}")

    # Check for convergence
    if rms_u < tol and rms_v < tol:
        print(f"Converged after {iter} iterations")
        break


# Plotting

import matplotlib.pyplot as plt

# Stream-function contours
plt.figure(figsize=(8, 6))
plt.contourf(psi.T, levels=20, cmap='viridis')
plt.colorbar(label='Stream function')
plt.title('Stream function contours')
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')
plt.show()


# Create coordinate grid
x = np.linspace(0, L, num_x)
y = np.linspace(0, H, num_y)
X, Y = np.meshgrid(x, y, indexing='ij')

# Streamlines
plt.figure(figsize=(8, 6))
plt.streamplot(X.T, Y.T, u.T, v.T, color='k', density=1.5, linewidth=1)
plt.title('Streamlines')
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')
plt.grid(True)
plt.show()

# Animation of streamlines
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots(figsize=(8, 6))

frame_interval = 25
frames = list(range(0, iter + 1, frame_interval))

def update(frame):
    ax.clear()
    ax.streamplot(X.T, Y.T, u_history[frame].T, v_history[frame].T, color='k', density=1.5, linewidth=1)
    ax.set_title(f'Streamlines at t = {time_history[frame]:.3f} s')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.axis('equal')
    ax.grid(True)

ani = FuncAnimation(fig, update, frames=frames, repeat=False)
ani.save('streamlines_real_time.mp4', writer='ffmpeg', fps=10)  # or dynamic fps






    



