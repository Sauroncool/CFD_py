import matplotlib.pyplot as plt
import numpy as np
from airfoil import load_airfoil_data, parametric_interpolation


def grid_generation(num_xi, num_eta, x_airfoil, y_airfoil, R_outer):
    num_airfoil_pts = len(x_airfoil)

    # Generate xi and eta values
    xi = np.linspace(0, 1, num_xi)
    eta = np.linspace(0, 1, num_eta)

    # Initialize the x and y grid arrays
    x = np.zeros((num_xi, num_eta))
    y = np.zeros((num_xi, num_eta))

    # Interpolate airfoil points to match xi distribution
    x_inner = np.interp(xi, np.linspace(0, 1, num_airfoil_pts), x_airfoil)
    y_inner = np.interp(xi, np.linspace(0, 1, num_airfoil_pts), y_airfoil)

    # Compute the x and y values for the grid
    for i in range(num_xi):
        for j in range(num_eta):
            x[i, j] = (1 - eta[j]) * x_inner[i] + eta[j] * R_outer * np.cos(
                2 * np.pi * xi[i]
            )
            y[i, j] = (1 - eta[j]) * y_inner[i] + eta[j] * R_outer * np.sin(
                2 * np.pi * xi[i]
            )
    return x, y, xi, eta


def plot_grid(x, y):
    plt.figure()
    plt.plot(x, y, "b-", linewidth=0.5)
    plt.plot(x.T, y.T, "r-", linewidth=0.5)
    plt.axis("equal")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(linestyle="--", alpha=0.5)
    plt.show()


if __name__ == "__main__":
    # Define the grid size
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

    dx_dξ = np.gradient(x, ξ, axis=0)
    dy_dξ = np.gradient(y, ξ, axis=0)

    dx_dη = np.gradient(x, η, axis=1)
    dy_dη = np.gradient(y, η, axis=1)

    print(dx_dξ.shape)
    print(y.shape)

    # Plot the grid
    plot_grid(x, y)
