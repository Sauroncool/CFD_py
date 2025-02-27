import matplotlib.pyplot as plt
import numpy as np
from grid_generation import generate_grid, plot_grid
from laplace_solver import compute_velocity, solve_laplace

# Grid parameters
nx, ny = 101, 81  # Grid points in ξ and η directions
a, b = 1, 0.5  # Semi-major and semi-minor axes of the ellipse
# a, b = 1, 1  # Circle
R_far = 20  # Far-field boundary distance

# Generate grid in computational space (ξ, η)
xi, eta, x, y = generate_grid(nx, ny, a, b, R_far)

# Plot the grid
plot_grid(x, y)

# Solve Laplace equation
phi = solve_laplace(xi, eta, x)

# # Check boundary conditions
# print("Far-field boundary φ values (should increase linearly with x):")
# print(phi[:, -1])  # Last column should be ~ U_inf * x

# print("Ellipse boundary φ gradient (should be ~0 for no-penetration):")
# phi_grad = np.gradient(phi, axis=0)  # Normal derivative at the ellipse
# print(np.max(np.abs(phi_grad[0, :])))  # Should be close to 0

# Plot potential field
plt.figure(figsize=(10, 10))
plt.contourf(x, y, phi, 50, cmap="viridis")
plt.title("Potential Field (φ) in Physical Domain")
plt.xlabel("x")
plt.ylabel("y")
plt.colorbar(label="Potential φ")
plt.show()

# Compute velocity field
u_xi, v_eta = compute_velocity(phi, xi, eta)

# Convert velocities to physical space using chain rule
u = np.zeros_like(u_xi)
v = np.zeros_like(v_eta)

# Compute derivatives using central differences
x_xi = np.gradient(x, axis=0, edge_order=2)
x_eta = np.gradient(x, axis=1, edge_order=2)
y_xi = np.gradient(y, axis=0, edge_order=2)
y_eta = np.gradient(y, axis=1, edge_order=2)

# Compute the Jacobian determinant
J = x_xi * y_eta - x_eta * y_xi  # Jacobian determinant

# Avoid division by zero in transformation
valid_indices = J != 0
u[valid_indices] = (u_xi[valid_indices] * y_eta[valid_indices] - v_eta[valid_indices] * y_xi[valid_indices]) / J[valid_indices]
v[valid_indices] = (-u_xi[valid_indices] * x_eta[valid_indices] + v_eta[valid_indices] * x_xi[valid_indices]) / J[valid_indices]

# print("Jacobian determinant (min, max):", np.min(J), np.max(J))

# Plot velocity potential derivatives
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.contourf(x, y, np.gradient(phi, axis=0), cmap="coolwarm")
plt.colorbar()
plt.title("∂φ/∂x")

plt.subplot(1, 2, 2)
plt.contourf(x, y, np.gradient(phi, axis=1), cmap="coolwarm")
plt.colorbar()
plt.title("∂φ/∂y")
plt.savefig("potential_derivatives.png")
plt.show()

# Velocity field (u, v) in physical domain
plt.figure(figsize=(10, 10))
plt.quiver(x, y, u, v, color="red")
plt.title("Velocity Field (u, v) in Physical Domain")
plt.xlabel("x")
plt.ylabel("y")
plt.savefig("velocity_field.png")
plt.show()

# Velocity magnitude
plt.figure(figsize=(10, 10))
u_mag = np.sqrt(u**2 + v**2)
plt.contourf(x, y, u_mag, cmap="coolwarm")
plt.colorbar()
plt.title("Velocity Magnitude")
plt.xlabel("x")
plt.ylabel("y")
plt.savefig("velocity_magnitude.png")
plt.show()

# Pressure field (p) in physical domain
p = 1 - u_mag**2
plt.figure(figsize=(10, 10))
plt.contourf(x, y, p, cmap="coolwarm")
plt.colorbar()
plt.title("Pressure Field (p) in Physical Domain")
plt.xlabel("x")
plt.ylabel("y")
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.savefig("pressure_field.png")
plt.show()

# plt.figure(figsize=(10, 8))
# plt.streamplot(x, y, u, v, density=1.5, linewidth=0.7, arrowsize=1.2)
# plt.xlabel("x")
# plt.ylabel("y")
# plt.title("Streamlines")
# plt.gca().set_aspect("equal")
# plt.show()