# Generating the grid using Transfinite Interpolation
import matplotlib.pyplot as plt
import numpy as np


# define the grid size
def grid_generation(num_xi, num_eta, a, b, R_outer):
    # Generate xi and eta values
    ξ = np.linspace(0, 1, num_xi)
    η = np.linspace(0, 1, num_eta)

    # Initialize the x and y grid arrays
    x = np.zeros((num_xi, num_eta))
    y = np.zeros((num_xi, num_eta))

    # Compute the x and y values
    for i in range(num_xi):
        for j in range(num_eta):
            x[i, j] = np.cos(2 * np.pi * ξ[i]) * ((1 - η[j]) * a + η[j] * R_outer)
            y[i, j] = np.sin(2 * np.pi * ξ[i]) * ((1 - η[j]) * b + η[j] * R_outer)

    return x, y


# Plot the grid
def plot_grid(x, y):
    plt.figure()
    plt.plot(x, y, "b-", linewidth=0.5)
    plt.plot(x.T, y.T, "r", linewidth=0.5)
    plt.axis("equal")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(linestyle="--", alpha=0.5)
    plt.show()


if __name__ == "__main__":
    # Define the grid size
    num_xi = 101
    num_eta = 81
    a = 1.0
    b = 0.5
    R_outer = 20.0

    # Generate the grid
    x, y = grid_generation(num_xi, num_eta, a, b, R_outer)

    # Plot the grid
    plot_grid(x, y)
