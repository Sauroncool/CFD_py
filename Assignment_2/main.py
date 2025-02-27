import matplotlib.pyplot as plt
import numpy as np
from grid_generation import generate_grid, plot_grid
from laplace_solver import compute_velocity, solve_laplace

# Grid parameters
nx, ny = 101, 81  # Grid points in ξ and η directions
a, b = 1, 0.5  # Semi-major and semi-minor axes of the ellipse
R_far = 20  # Far-field boundary distance

# Generate grid in computational space (ξ, η)
xi, eta, x, y = generate_grid(nx, ny, a, b, R_far)

# Solve Laplace equation
phi = solve_laplace(xi, eta)

# Compute velocity field
u_xi, v_eta = compute_velocity(phi, xi, eta)

# Convert velocities to physical space using chain rule
u = np.zeros_like(u_xi)
v = np.zeros_like(v_eta)

# Derivatives for transformation
x_xi = np.gradient(x, axis=0)
x_eta = np.gradient(x, axis=1)
y_xi = np.gradient(y, axis=0)
y_eta = np.gradient(y, axis=1)

# Apply the chain rule for transformation
for i in range(1, nx - 1):  # Avoid boundaries
    for j in range(1, ny - 1):  # Avoid boundaries
        J = x_xi[i, j] * y_eta[i, j] - x_eta[i, j] * y_xi[i, j]  # Jacobian determinant
        if J != 0:
            u[i, j] = (u_xi[i, j] * y_eta[i, j] - v_eta[i, j] * y_xi[i, j]) / J
            v[i, j] = (-u_xi[i, j] * x_eta[i, j] + v_eta[i, j] * x_xi[i, j]) / J

# Plot the grid
plot_grid(x, y)

# Potential field (φ) in physical domain
plt.subplot(1, 2, 1)
cp = plt.contourf(x, y, phi, 50, cmap="viridis")
plt.colorbar(cp)
plt.title("Potential Field (φ) in Physical Domain")
plt.xlabel("x")
plt.ylabel("y")

# Velocity field (u, v) in physical domain
plt.subplot(1, 2, 2)
plt.quiver(x, y, u, v, scale=10, color="red")
plt.title("Velocity Field (u, v) in Physical Domain")
plt.xlabel("x")
plt.ylabel("y")

plt.tight_layout()
plt.show()
