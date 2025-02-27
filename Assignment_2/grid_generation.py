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
        xi (ndarray): Computational grid (ξ-coordinates, normalized 0 to 1).
        eta (ndarray): Computational grid (η-coordinates, normalized 0 to 1).
        x (ndarray): Grid point x-coordinates.
        y (ndarray): Grid point y-coordinates.
    """
    xi = np.linspace(0, 1, num_xi)  # Computational domain in ξ-direction
    eta = np.linspace(0, 1, num_eta)  # Computational domain in η-direction

    XI, ETA = np.meshgrid(xi, eta)  # 2D computational coordinates

    # Generate theta values for the ellipse boundary
    theta = np.linspace(0, 2 * np.pi, num_xi)

    # Inner and outer boundary points
    x_inner = a * np.cos(theta)
    y_inner = b * np.sin(theta)
    x_outer = R_far * np.cos(theta)
    y_outer = R_far * np.sin(theta)

    # Transfinite Interpolation (Vectorized)
    x = (1 - ETA) * x_inner[np.newaxis, :] + ETA * x_outer[np.newaxis, :]
    y = (1 - ETA) * y_inner[np.newaxis, :] + ETA * y_outer[np.newaxis, :]

    return XI, ETA, x, y


def plot_grid(x, y):
    """
    Plots the generated grid.
    Args:
        x (ndarray): x-coordinates of the grid.
        y (ndarray): y-coordinates of the grid.
    """
    plt.figure(figsize=(10, 10))
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


# Example usage
# if __name__ == "__main__":
#     nx, ny = 101, 81  # Grid resolution
#     a, b = 1, 0.5  # Ellipse parameters
#     R_far = 20  # Far-field boundary

#     xi, eta, x, y = generate_grid(nx, ny, a, b, R_far)
#     plot_grid(x, y)

#     # Example: Mapping from computational to physical coordinates
#     i, j = 40, 50  # Sample computational index
#     print(f"At (xi, eta) = ({xi[i, j]:.3f}, {eta[i, j]:.3f}), physical coordinates are (x, y) = ({x[i, j]:.3f}, {y[i, j]:.3f})")
