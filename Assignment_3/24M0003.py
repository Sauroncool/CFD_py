import csv
import os

import matplotlib.pyplot as plt
import numpy as np


def initialize_grid(l_x, l_y, Δx, Δy, ψ1, ψ2, ψ3, init_guess=200):
    """Initialize the grid and apply boundary conditions."""
    β = Δx / Δy
    nx, ny = int(l_x / Δx) + 1, int(l_y / Δy) + 1
    ψ = np.full((nx, ny), init_guess, dtype=float)

    # Apply boundary conditions
    ψ[0, :] = ψ3  # Left
    ψ[:, -1] = ψ3  # Top

    ψ[: int(1.1 / Δx), 0] = ψ3  # Bottom left
    ψ[int(1.1 / Δx) :, 0] = ψ1  # Bottom right

    ψ[-1, : int(1.1 / Δy)] = ψ1  # Right bottom
    ψ[-1, int(1.1 / Δy) : int(2.0 / Δy)] = ψ2  # Right middle
    ψ[-1, int(2.0 / Δy) :] = ψ3  # Right top

    return ψ, β, nx, ny


def point_jacobi(ψ, β, nx, ny, iterations=10000, tolerance=1e-4):
    """Perform Point Jacobi iterations to solve for the streamfunction."""
    error_val = []
    for k in range(iterations):
        ψ_old = ψ.copy()
        ψ[1:-1, 1:-1] = (1 / (2 * (1 + β**2))) * (
            β**2 * (ψ[1:-1, 2:] + ψ[1:-1, :-2]) + (ψ[2:, 1:-1] + ψ[:-2, 1:-1])
        )

        error = np.linalg.norm(ψ - ψ_old) / np.linalg.norm(ψ)
        error_val.append(error)
        if error < tolerance:
            print(f"Converged in {k} iterations.")
            return ψ, error_val, k, True

    print("Convergence not reached")
    return ψ, error_val, k, False


def save_plot(fig, filename):
    """Save a plot to the results directory."""
    fig.savefig(os.path.join("Results", filename))
    plt.close(fig)


def plot_convergence(error_val, k, filename):
    """Plot and save the convergence plot."""
    fig, ax = plt.subplots()
    ax.plot(np.log10(error_val))
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Log10(Error)")
    if max(error_val) > 1e-4:
        ax.set_title(f"Convergence Plot (Not Converged after {k} iterations)")
    ax.set_title(f"Convergence Plot (Converged in {k} iterations)")
    save_plot(fig, filename)


def compute_velocity(ψ, Δx, Δy):
    """Compute velocity components from the streamfunction."""
    u = np.gradient(ψ, axis=1) / Δy  # u = ∂ψ/∂y
    v = -np.gradient(ψ, axis=0) / Δx  # v = -∂ψ/∂x
    return u, v


def plot_streamfunction(ψ, filename):
    """Plot and save the streamfunction contours."""
    fig, ax = plt.subplots(figsize=(10, 8))
    c = ax.contourf(ψ.T, cmap="viridis")
    plt.colorbar(c, label="Streamfunction")
    ax.set_title("Streamfunction Contours")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    save_plot(fig, filename)


def plot_streamlines(X, Y, u, v, ψ, filename):
    """Plot and save the streamline patterns."""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.streamplot(X, Y, u.T, v.T, color="b", density=2, linewidth=1, arrowsize=0.5)
    c = ax.contourf(X, Y, ψ.T, cmap="viridis", alpha=0.5)
    plt.colorbar(c, label="Streamfunction")
    ax.set_title("Streamline Patterns")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    save_plot(fig, filename)


def save_csv(y, ψ, nx, ny, Δx, filename):
    """Save the streamfunction data to a CSV file."""
    with open(os.path.join("Results", filename), mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["y", "ψ(x=0)", "ψ(x=1)", "ψ(x=2)", "ψ(x=3)"])
        for i in range(ny):
            writer.writerow(
                [
                    y[i],
                    round(ψ[0, i], 4),
                    round(ψ[int(1.0 / Δx), i], 4),
                    round(ψ[int(2.0 / Δx), i], 4),
                    round(ψ[-1, i], 4),
                ]
            )


if __name__ == "__main__":
    l_x, l_y, Δx, Δy = 1.5, 4.0, 0.1, 0.1

    test_cases = [(100, 150, 300), (100, 200, 300), (100, 250, 300)]
    init_guesses = [100, 150, 200]

    for i, (ψ1, ψ2, ψ3) in enumerate(test_cases, 1):
        for j, init_guess in enumerate(init_guesses, 1):
            ψ, β, nx, ny = initialize_grid(l_x, l_y, Δx, Δy, ψ1, ψ2, ψ3, init_guess)
            print(f"Test case {i}, Initial guess {j}")
            ψ, error_val, k, flag = point_jacobi(ψ, β, nx, ny)

            filename_prefix = f"test_case_{i}_init_{j}"
            plot_convergence(error_val, k, f"{filename_prefix}_convergence.png")

            if flag:
                ψ = np.vstack((ψ, ψ[-2::-1, :]))  # Mirroring ψ about the right edge
                u, v = compute_velocity(ψ, Δx, Δy)
                x = np.linspace(0, 2 * l_x, 2 * nx - 1)
                y = np.linspace(0, l_y, ny)
                X, Y = np.meshgrid(x, y)

                plot_streamfunction(ψ, f"{filename_prefix}_streamfunction.png")
                plot_streamlines(X, Y, u, v, ψ, f"{filename_prefix}_streamlines.png")
                save_csv(y, ψ, nx, ny, Δx, f"{filename_prefix}.csv")
