import matplotlib.pyplot as plt
import numpy as np


def generate_grid(num_xi, num_eta, a, b, R_far):
    """
    Generates grid points using Transfinite Interpolation.
    Args:
        num_xi (int): Number of grid points in the ξ-direction.
        num_eta (int): Number of grid points in the η-direction.
        a (float): Semi-major axis of the ellipse.
        b (float): Semi-minor axis of the ellipse.
        R_far (float): Radius for the far-field boundary.
    Returns:
        x (ndarray): Grid point x-coordinates.
        y (ndarray): Grid point y-coordinates.
    """
    xi = np.linspace(0, 1, num_xi)
    eta = np.linspace(0, 1, num_eta)

    x = np.zeros((num_eta, num_xi))
    y = np.zeros((num_eta, num_xi))

    for i in range(num_eta):
        for j in range(num_xi):
            # Parametric equations for the ellipse boundary
            x_inner = a * np.cos(2 * np.pi * xi[j])  # Should return a scalar
            y_inner = b * np.sin(2 * np.pi * xi[j])  # Should return a scalar

            x_outer = R_far * np.cos(2 * np.pi * xi[j])  # Should return a scalar
            y_outer = R_far * np.sin(2 * np.pi * xi[j])  # Should return a scalar

            # Transfinite interpolation
            x[i, j] = (1 - eta[i]) * x_inner + eta[i] * x_outer
            y[i, j] = (1 - eta[i]) * y_inner + eta[i] * y_outer

    xi, eta = np.meshgrid(xi, eta)
    return xi, eta, x, y


def plot_grid(x, y):
    """
    Plots the generated grid.
    Args:
        x (ndarray): x-coordinates of the grid.
        y (ndarray): y-coordinates of the grid.
    """
    plt.figure(figsize=(10, 8))
    for i in range(x.shape[0]):
        plt.plot(x[i, :], y[i, :], "b")  # ξ-direction
    for j in range(x.shape[1]):
        plt.plot(x[:, j], y[:, j], "r")  # η-direction
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Grid Generated using Transfinite Interpolation")
    plt.gca().set_aspect("equal")
    plt.savefig("grid.png")
    plt.show()


# Example usage (to be removed in final integration)
# if __name__ == "__main__":
#   nx, ny = 101, 81
#  a, b = 1, 0.5
# R_far = 20
# x, y = generate_grid(nx, ny, a, b, R_far)
# plot_grid(x, y)
