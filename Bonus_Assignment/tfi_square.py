import matplotlib.pyplot as plt
import numpy as np
from airfoil import load_airfoil_data, parametric_interpolation


def square_boundary(xi):
    """Generate points on a square boundary with edge length 20, starting at (10, 0) and moving counterclockwise."""
    x_outer = np.zeros_like(xi)
    y_outer = np.zeros_like(xi)

    for i, t in enumerate(xi):
        if t < 0.125:
            # Upper Right edge from (10, 0) to (10, 10)
            x_outer[i] = 10
            y_outer[i] = 10 * t / 0.125
        elif t < 0.375:
            # Top edge from (10, 10) to (-10, 10)
            x_outer[i] = 10 - 20 * (t - 0.125) / 0.25
            y_outer[i] = 10
        elif t < 0.625:
            # Left edge from (-10, 10) to (-10, -10)
            x_outer[i] = -10
            y_outer[i] = 10 - 20 * (t - 0.375) / 0.25
        elif t < 0.875:
            # Bottom edge from (-10, -10) to (10, -10)
            x_outer[i] = -10 + 20 * (t - 0.625) / 0.25
            y_outer[i] = -10
        else:
            # Lower Right edge from (10, -10) to (10, 0)
            x_outer[i] = 10
            y_outer[i] = -10 + 10 * (t - 0.875) / 0.125

    return x_outer, y_outer


def grid_generation(num_xi, num_eta, x_airfoil, y_airfoil):
    num_airfoil_pts = len(x_airfoil)
    xi = np.linspace(0, 1, num_xi)
    eta = np.linspace(0, 1, num_eta)

    # Interpolate airfoil points to match xi distribution
    x_inner = np.interp(xi, np.linspace(0, 1, num_airfoil_pts), x_airfoil)
    y_inner = np.interp(xi, np.linspace(0, 1, num_airfoil_pts), y_airfoil)

    # Generate outer square boundary
    x_outer, y_outer = square_boundary(xi)

    # plt.figure()
    # plt.plot(x_inner, y_inner, "r-", linewidth=0.5)
    # plt.plot(x_outer, y_outer, "b-", linewidth=0.5)
    # plt.grid(linestyle="--", alpha=0.5)
    # plt.axis("equal")
    # plt.xlabel("x")
    # plt.ylabel("y")
    # plt.show()

    # Initialize the x and y grid arrays
    x = np.zeros((num_xi, num_eta))
    y = np.zeros((num_xi, num_eta))

    # # print x_inner, y_inner
    # for i in range(num_xi):
    #     print(i, x_inner[i], y_inner[i])

    # Compute the x and y values for the grid
    for i in range(num_xi):
        for j in range(num_eta):
            x[i, j] = (1 - eta[j]) * x_inner[i] + eta[j] * x_outer[i]
            y[i, j] = (1 - eta[j]) * y_inner[i] + eta[j] * y_outer[i]

    return x, y


def plot_grid(x, y):
    plt.figure()
    plt.plot(x, y, "b-", linewidth=0.5)
    plt.plot(x.T, y.T, "r-", linewidth=0.5)
    plt.axis("equal")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.grid(linestyle="--", alpha=0.5)
    plt.show()


if __name__ == "__main__":
    num_xi = 303
    num_eta = 81
    filename = "naca2412.dat"

    x, y = load_airfoil_data(filename)
    x_airfoil, y_airfoil = parametric_interpolation(x, y)

    x, y = grid_generation(num_xi, num_eta, x_airfoil, y_airfoil)
    plot_grid(x, y)
