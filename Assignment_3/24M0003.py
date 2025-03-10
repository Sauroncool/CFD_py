# Compute Streamline pattern using point jacobi method
import matplotlib.pyplot as plt
import numpy as np

# Grid size
l_x = 1.5
l_y = 4.0

Δx = 0.1
Δy = 0.1
β = Δx / Δy

nx = int(l_x / Δx) + 1
ny = int(l_y / Δy) + 1

ψ = np.ones((nx, ny)) * 0  # Initial guess

ψ1 = 100
ψ2 = 150
ψ3 = 300

# Boundary conditions
ψ[0, :] = ψ3  # Left
ψ[:, -1] = ψ3  # Top
# Bottom
ψ[: int(1.1 * nx / l_x), 0] = ψ3
ψ[int(1.1 * nx / l_x) :, 0] = ψ1
# Right
ψ[-1, : int(1.0 * ny / l_y)] = ψ1
ψ[-1, int(1.1 * ny / l_y) : int(1.9 * ny / l_y)] = ψ2
ψ[-1, int(2.0 * ny / l_y) :] = ψ3

# Point Jacobi method
iterations = 10000
tolerance = 1e-4
error_val = []
for k in range(iterations):
    ψ_old = ψ.copy()  # Store previous iteration

    # Update internal points using vectorized operations
    ψ[1:-1, 1:-1] = (1 / (2 * (1 + β**2))) * (
        β**2 * (ψ[1:-1, 2:] + ψ[1:-1, 0:-2]) + (ψ[2:, 1:-1] + ψ[0:-2, 1:-1])
    )

    # Apply boundary conditions again (only needed outside update loop)
    ψ[0, :] = ψ3  # Left
    ψ[:, -1] = ψ3  # Top
    # Bottom
    ψ[: int(1.1 * nx / l_x), 0] = ψ3
    ψ[int(1.1 * nx / l_x) :, 0] = ψ1
    # Right
    ψ[-1, : int(1.0 * ny / l_y)] = ψ1
    ψ[-1, int(1.1 * ny / l_y) : int(1.9 * ny / l_y)] = ψ2
    ψ[-1, int(2.0 * ny / l_y) :] = ψ3

    # Convergence check
    error = np.linalg.norm(ψ - ψ_old) / np.linalg.norm(ψ)
    error_val.append(error)
    if error < tolerance:
        print(f"Converged in {k} iterations.")
        break

if error >= tolerance:
    print("Convergence not reached")

# plot a logarithmic(base 10) error plot
plt.plot(np.log10(error_val))
plt.xlabel("Iterations")
plt.ylabel("Log10(Error)")
plt.title("Convergence Plot")
plt.show()

# Mirroring ψ about the right edge
ψ = np.vstack((ψ,ψ[-2::-1, :]))

# Compute velocity components from the streamfunction
u = np.gradient(ψ, axis=1) / Δy  # u = ∂ψ/∂y
v = -np.gradient(ψ, axis=0) / Δx  # v = -∂ψ/∂ξ

# Make grid for plotting streamlines
x = np.linspace(0, 2*l_x, 2*nx-1)
y = np.linspace(0, l_y, ny)
X, Y = np.meshgrid(x, y)


# Plot streamfunction Contours
plt.figure(figsize=(10, 8))
plt.contourf(ψ.T, cmap="viridis")
plt.colorbar(label="Streamfunction")
plt.title("Streamfunction Contours")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# Plot streamlines
plt.figure(figsize=(10, 8))
plt.streamplot(X, Y, u.T, v.T, color="b", density=2, linewidth=1, arrowsize=0.5)
plt.contourf(X, Y, ψ.T, cmap="viridis", alpha=0.5)  # Overlay streamfunction contours
plt.colorbar(label="Streamfunction")
plt.title("Streamline Patterns")
plt.xlabel("x")
plt.ylabel("y")
plt.show()