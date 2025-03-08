# Compute Streamline pattern using point jacobi method
import matplotlib.pyplot as plt
import numpy as np

# Grid size
l_x = 3.0
l_y = 4.0

Δx = 0.1
Δy = 0.1

nx = int(l_x / Δx)
ny = int(l_y / Δy)

ψ = np.ones((nx, ny)) * 0  # Initial guess

ψ1 = 100
ψ2 = 150
ψ3 = 300

# Boundary conditions
ψ[0, :] = ψ3  # Left
ψ[-1, :] = ψ3  # Right
ψ[:, -1] = ψ3  # Top
# Bottom
ψ[: int(1.1 * nx / l_x), 0] = ψ3
ψ[int(1.1 * nx / l_x) : int(2.0 * nx / l_x), 0] = ψ1
ψ[int(2.0 * nx / l_x) :, 0] = ψ3
# Middle
ψ[int(1.5 * nx / l_x), : int(1.1 * ny / l_y)] = ψ1
ψ[int(1.5 * nx / l_x), int(1.1 * ny / l_y) : int(2.0 * ny / l_y)] = ψ2
ψ[int(1.5 * nx / l_x), int(2.0 * ny / l_y) :] = ψ3

# Check this below part with prof
# How to apply inlet and outlet conditions


# Jacobi method
iterations = 10000
tolerance = 1e-4
for k in range(iterations):
    ψ_old = ψ.copy()  # Store previous iteration

    # Update internal points using vectorized operations
    ψ[1:-1, 1:-1] = 0.25 * (
        ψ_old[:-2, 1:-1] + ψ_old[2:, 1:-1] + ψ_old[1:-1, :-2] + ψ_old[1:-1, 2:]
    )

    # Apply boundary conditions again (only needed outside update loop)
    ψ[0, :] = ψ3
    ψ[-1, :] = ψ3
    ψ[:, -1] = ψ3
    ψ[: int(1.1 * nx / l_x), 0] = ψ3
    ψ[int(1.1 * nx / l_x) : int(2.0 * nx / l_x), 0] = ψ1
    ψ[int(2.0 * nx / l_x) :, 0] = ψ3
    ψ[int(1.5 * nx / l_x), : int(1.1 * ny / l_y)] = ψ1
    ψ[int(1.5 * nx / l_x), int(1.1 * ny / l_y) : int(2.0 * ny / l_y)] = ψ2
    ψ[int(1.5 * nx / l_x), int(2.0 * ny / l_y) :] = ψ3

    # Convergence check
    error = np.linalg.norm(ψ - ψ_old) / np.linalg.norm(ψ)
    if error < tolerance:
        print(f"Converged in {k} iterations.")
        break

if error >= tolerance:
    print("Convergence not reached")

# Plot streamfunction Contours
plt.contourf(ψ.T, cmap="viridis")
plt.colorbar(label="Streamfunction")
plt.title("Streamfunction Contours")
plt.show()
