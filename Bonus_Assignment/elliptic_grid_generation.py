# Code to improve grid generated using the TFI method by using elliptic grid generation.
import numpy as np


def coeff(x, y, ξ, η):
    dx_dξ = np.gradient(x, ξ, axis=0)
    dy_dξ = np.gradient(y, ξ, axis=0)

    dx_dη = np.gradient(x, η, axis=1)
    dy_dη = np.gradient(y, η, axis=1)

    a = dx_dη**2 + dy_dη**2
    b = dx_dξ * dx_dη + dy_dξ * dy_dη
    c = dx_dξ**2 + dy_dξ**2
    return a, b, c


def GS_iteration(x, y, ξ, η, max_iter=1000, tolerance=5e-4):
    x_new, y_new = np.copy(x), np.copy(y)
    Δξ = 1 / (len(ξ) - 1)
    Δη = 1 / (len(η) - 1)
    for iter in range(max_iter):
        a, b, c = coeff(x, y, ξ, η)
        # Interior points update
        x_new[1:-1, 1:-1] = (
            a[1:-1, 1:-1] * (x[2:, 1:-1] + x[:-2, 1:-1]) / Δξ**2
            + c[1:-1, 1:-1] * (x[1:-1, 2:] + x[1:-1, :-2]) / Δη**2
            - b[1:-1, 1:-1] * (
                x[2:, 2:] - x[2:, :-2] + x[:-2, :-2] - x[:-2, 2:]
            ) / (2 * Δξ * Δη)
        ) / (2 * (a[1:-1, 1:-1] / Δξ**2 + c[1:-1, 1:-1] / Δη**2))

        y_new[1:-1, 1:-1] = (
            a[1:-1, 1:-1] * (y[2:, 1:-1] + y[:-2, 1:-1]) / Δξ**2
            + c[1:-1, 1:-1] * (y[1:-1, 2:] + y[1:-1, :-2]) / Δη**2
            - b[1:-1, 1:-1] * (
                y[2:, 2:] - y[2:, :-2] + y[:-2, :-2] - y[:-2, 2:]
            ) / (2 * Δξ * Δη)
        ) / (2 * (a[1:-1, 1:-1] / Δξ**2 + c[1:-1, 1:-1] / Δη**2))

        # Boundary points update
        x_new[0, 1:-1] = (
        a[0, 1:-1] * (x[1, 1:-1] + x[-2, 1:-1]) / Δξ**2
        + c[0, 1:-1] * (x[0, 2:] + x[0, :-2]) / Δη**2
        - b[0, 1:-1] * (
            x[1, 2:] - x[1, :-2] + x[-2, :-2] - x[-2, 2:]
        ) / (2 * Δξ * Δη)
    ) / (2 * (a[0, 1:-1] / Δξ**2 + c[0, 1:-1] / Δη**2))

        y_new[0, 1:-1] = (
            a[0, 1:-1] * (y[1, 1:-1] + y[-2, 1:-1]) / Δξ**2
            + c[0, 1:-1] * (y[0, 2:] + y[0, :-2]) / Δη**2
            - b[0, 1:-1] * (
                y[1, 2:] - y[1, :-2] + y[-2, :-2] - y[-2, 2:]
            ) / (2 * Δξ * Δη)
        ) / (2 * (a[0, 1:-1] / Δξ**2 + c[0, 1:-1] / Δη**2))



        x_new[-1, :] = x_new[0, :]
        y_new[-1, :] = y_new[0, :]



        # Compute convergence error
        x_error = np.max(np.abs(x_new - x))
        y_error = np.max(np.abs(y_new - y))

        # x_error = np.sum(np.abs(x_new - x))
        # y_error = np.sum(np.abs(y_new - y))
        error = x_error + y_error
        if (iter + 1) % 100 == 0:
            print(f"Iteration {iter + 1}, Error: {error}")

        if error < tolerance:
            print(f"Converged in {iter + 1} iterations, Error: {error}")
            break

        if iter == max_iter - 1:
            print("Did not converge, Error: ", error)

        x[:], y[:] = x_new, y_new

    return x_new, y_new


if __name__ == "__main__":
    from airfoil import load_airfoil_data, parametric_interpolation
    from tfi import grid_generation, plot_grid

    num_xi = 101
    num_eta = 81
    R_outer = 10.0
    # Get airfoil coordinates
    filename = "naca2412.dat"

    # Load data
    x_afl_pts, y_afl_pts = load_airfoil_data(filename)

    # Perform parametric cubic spline interpolation
    x_airfoil, y_airfoil = parametric_interpolation(x_afl_pts, y_afl_pts)

    # Generate the grid
    x, y, ξ, η = grid_generation(num_xi, num_eta, x_airfoil, y_airfoil, R_outer)

    plot_grid(x, y, "grid_tfi.png")

    x, y = GS_iteration(x, y, ξ, η)

    plot_grid(x, y, "grid_elliptic.png")
