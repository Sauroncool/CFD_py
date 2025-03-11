# Compute Streamline pattern using point jacobi method
import matplotlib.pyplot as plt
import numpy as np

# Grid size
l_x = 3.0
l_y = 4.0

Δx = 0.1
Δy = 0.1
β = Δx / Δy

nx = int(l_x / Δx) + 1
ny = int(l_y / Δy) + 1

ψ = np.ones((nx, ny)) * 100  # Initial guess

ψ1 = 100
ψ2 = 150
ψ3 = 300

# Boundary conditions
ψ[0, :] = ψ3  # Left
ψ[-1, :] = ψ3  # Right
ψ[:, -1] = ψ3  # Top
# Bottom
ψ[: int(1.1 * 1 / Δx), 0] = ψ3
ψ[int(1.1 * 1 / Δx) : int(2.0 * 1 / Δx), 0] = ψ1
ψ[int(2.0 * 1 / Δx) :, 0] = ψ3
# Middle
ψ[int(1.5 * 1 / Δx), : int(1.1 * 1 / Δy)] = ψ1
ψ[int(1.5 * 1 / Δx), int(1.1 * 1 / Δy) : int(2.0 * 1 / Δy)] = ψ2
ψ[int(1.5 * 1 / Δx), int(2.0 * 1 / Δy) :] = ψ3

# Point Jacobi method
iterations = 10000
tolerance = 1e-4
for k in range(iterations):
    ψ_old = ψ.copy()  # Store previous iteration

    # Update internal points using vectorized operations
    ψ[1:-1, 1:-1] = (1 / (2 * (1 + β**2))) * (
        β**2 * (ψ[1:-1, 2:] + ψ[1:-1, 0:-2]) + (ψ[2:, 1:-1] + ψ[0:-2, 1:-1])
    )

    # Boundary conditions
    ψ[0, :] = ψ3  # Left
    ψ[-1, :] = ψ3  # Right
    ψ[:, -1] = ψ3  # Top
    # Bottom
    ψ[: int(1.1 * 1 / Δx), 0] = ψ3
    ψ[int(1.1 * 1 / Δx) : int(2.0 * 1 / Δx), 0] = ψ1
    ψ[int(2.0 * 1 / Δx) :, 0] = ψ3
    # Middle
    ψ[int(1.5 * 1 / Δx), : int(1.1 * 1 / Δy)] = ψ1
    ψ[int(1.5 * 1 / Δx), int(1.1 * 1 / Δy) : int(2.0 * 1 / Δy)] = ψ2
    ψ[int(1.5 * 1 / Δx), int(2.0 * 1 / Δy) :] = ψ3

    # Convergence check
    error = np.linalg.norm(ψ - ψ_old) / np.linalg.norm(ψ)
    if error < tolerance:
        print(f"Converged in {k} iterations.")
        break

if error >= tolerance:
    print("Convergence not reached")

# Plot streamfunction Contours
plt.figure(figsize=(10, 8))
plt.contourf(ψ.T, cmap="viridis")
plt.colorbar(label="Streamfunction")
plt.title("Streamfunction Contours")
plt.xlabel("x")
plt.ylabel("y")
plt.savefig("streamfunction_nonsymmetric.png")
# plt.show()
plt.close()

# Compute velocity components from the streamfunction
u = np.gradient(ψ, axis=1) / Δy  # u = ∂ψ/∂y
v = -np.gradient(ψ, axis=0) / Δx  # v = -∂ψ/∂x

# Create a grid for plotting
x = np.linspace(0, l_x, nx)
y = np.linspace(0, l_y, ny)
X, Y = np.meshgrid(x, y)

# Plot streamlines
plt.figure(figsize=(10, 8))
plt.streamplot(X, Y, u.T, v.T, color="b", density=2, linewidth=1, arrowsize=0.5)
plt.contourf(X, Y, ψ.T, cmap="viridis", alpha=0.5)  # Overlay streamfunction contours
plt.colorbar(label="Streamfunction")
# y_segments = [(0, 1.0), (1.1, 1.9), (2.0, 4.0)]  # Define y segments with gaps
#
# for y_start, y_end in y_segments:
#     plt.plot([1.5,1.5], [y_start, y_end], color="k", linewidth=2)
plt.title("Streamline Patterns")
plt.xlabel("x")
plt.ylabel("y")
plt.savefig("streamline_nonsymmetric.png")
# plt.show()
plt.close()
