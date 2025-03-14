import matplotlib.pyplot as plt
import numpy as np


def initialize_grid(l_x, l_y, Δx, Δy, ψ1, ψ2, ψ3, init_guess=200):
    """Initialize the grid and apply boundary conditions."""
    β = Δx / Δy
    nx = int(l_x / Δx) + 1
    ny = int(l_y / Δy) + 1
    ψ = np.ones((nx, ny)) * init_guess

    # Apply boundary conditions
    ψ[0, :] = ψ3  # Left
    ψ[:, -1] = ψ3  # Top
    # Bottom
    ψ[: int(1.1 * 1 / Δx), 0] = ψ3
    ψ[int(1.1 * 1 / Δx) :, 0] = ψ1
    # Right
    ψ[-1, : int(1.1 * 1 / Δy)] = ψ1
    ψ[-1, int(1.1 * 1 / Δy) : int(2.0 * 1 / Δy)] = ψ2
    ψ[-1, int(2.0 * 1 / Δy) :] = ψ3

    return ψ, β, nx, ny


def point_jacobi(ψ, β, nx, ny, iterations=10000, tolerance=1e-4):
    """Perform Point Jacobi iterations to solve for the streamfunction."""
    error_val = []
    for k in range(iterations):
        ψ_old = ψ.copy()

        # Update internal points
        ψ[1:-1, 1:-1] = (1 / (2 * (1 + β**2))) * (
            β**2 * (ψ[1:-1, 2:] + ψ[1:-1, 0:-2]) + (ψ[2:, 1:-1] + ψ[0:-2, 1:-1])
        )

        # Convergence check
        error = np.linalg.norm(ψ - ψ_old) / np.linalg.norm(ψ)
        error_val.append(error)
        if error < tolerance:
            print(f"Converged in {k} iterations.")
            flag = True
            break

    if error >= tolerance:
        print("Convergence not reached")
        flag = False

    return ψ, error_val, k, flag


def plot_convergence(error_val, k):
    """Plot and save the convergence plot."""
    plt.plot(np.log10(error_val))
    plt.xlabel("Iterations")
    plt.ylabel("Log10(Error)")
    plt.title(f"Convergence Plot (Converged in {k} iterations)")
    plt.savefig(f"convergence_{ψ1}_{ψ2}_{ψ3}_{init_guess}.png")
    # plt.show()
    plt.close()


def compute_velocity(ψ, Δx, Δy):
    """Compute velocity components from the streamfunction."""
    u = np.gradient(ψ, axis=1) / Δy  # u = ∂ψ/∂y
    v = -np.gradient(ψ, axis=0) / Δx  # v = -∂ψ/∂ξ
    return u, v


def plot_streamfunction(ψ):
    """Plot and save the streamfunction contours."""
    plt.figure(figsize=(10, 8))
    plt.contourf(ψ.T, cmap="viridis")
    plt.colorbar(label="Streamfunction")
    plt.title("Streamfunction Contours")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig(f"streamfunction_{ψ1}_{ψ2}_{ψ3}_{init_guess}.png")
    # plt.show()
    plt.close()


def plot_streamlines(X, Y, u, v, ψ):
    """Plot and save the streamline patterns."""
    plt.figure(figsize=(10, 8))
    plt.streamplot(X, Y, u.T, v.T, color="b", density=2, linewidth=1, arrowsize=0.5)
    plt.contourf(X, Y, ψ.T, cmap="viridis", alpha=0.5)
    plt.colorbar(label="Streamfunction")
    plt.title("Streamline Patterns")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig(f"Streamline_{ψ1}_{ψ2}_{ψ3}_{init_guess}.png")
    # plt.show()
    plt.close()


if __name__ == "__main__":
    l_x, l_y = 1.5, 4.0
    Δx, Δy = 0.1, 0.1

    # Test cases
    def test_case_1():
        return 100, 150, 300

    def test_case_2():
        return 100, 200, 300

    def test_case_3():
        return 100, 250, 300

    # Initial guess
    def init_guess_1():
        return 100

    def init_guess_2():
        return 150

    def init_guess_3():
        return 200

    ψ1, ψ2, ψ3 = test_case_1()
    init_guess = init_guess_1()

    ψ, β, nx, ny = initialize_grid(l_x, l_y, Δx, Δy, ψ1, ψ2, ψ3, init_guess)
    ψ, error_val, k, flag = point_jacobi(ψ, β, nx, ny)
    plot_convergence(error_val, k)

    if flag:
        # Mirroring ψ about the right edge
        ψ = np.vstack((ψ, ψ[-2::-1, :]))

        u, v = compute_velocity(ψ, Δx, Δy)
        x = np.linspace(0, 2 * l_x, 2 * nx - 1)
        y = np.linspace(0, l_y, ny)
        X, Y = np.meshgrid(x, y)

        plot_streamfunction(ψ)
        plot_streamlines(X, Y, u, v, ψ)

        # Write value of streamfunction at x = 0,1,2,3 for all y into a csv File
        import csv

        with open(f"{ψ1}_{ψ2}_{ψ3}_{init_guess}.csv", mode="w") as file:
            writer = csv.writer(file)
            writer.writerow(["y", "ψ(x=0)", "ψ(x=1)", "ψ(x=2)", "ψ(x=3)"])
            for i in range(ny):
                writer.writerow(
                    [
                        y[i],
                        round(ψ[0, i], 4),
                        round(ψ[int(1.0 * 1 / Δx), i], 4),
                        round(ψ[int(2.0 * 1 / Δx), i], 4),
                        round(ψ[-1, i], 4),
                    ]
                )
    else:
        print("Convergence not reached")
