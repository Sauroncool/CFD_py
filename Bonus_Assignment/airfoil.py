import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline


def load_airfoil_data(filename):
    """Loads airfoil data from a file and returns x, y coordinates as numpy arrays."""
    x, y = [], []
    with open(filename, "r") as file:
        lines = file.readlines()[1:]  # Skip the first line with airfoil name
        for line in lines:
            values = line.split()
            x.append(float(values[0]))
            y.append(float(values[1]))
    return np.array(x), np.array(y)


def parametric_interpolation(x, y, num_points=200):
    """Interpolates the airfoil as a single continuous curve using parametric cubic splines."""
    # Ensure the trailing edge is smoothly tapered
    if x[0] != x[-1] or y[0] != y[-1]:
        # Average the y-coordinates of the first and last points to ensure a smooth trailing edge
        y_te = (y[0] + y[-1]) / 2
        x = np.append(x, x[0])  # Close the loop by appending the first point
        y = np.append(y, y_te)  # Use the averaged y-coordinate for the trailing edge

    # Ensure the first and last y values are identical for periodic boundary conditions
    y[-1] = y[0]  # Force the last y value to match the first

    # Create a parameter t based on the index
    t = np.linspace(0, 1, len(x))

    # Create cubic splines for both x(t) and y(t) with periodic boundary conditions
    cs_x = CubicSpline(t, x, bc_type="periodic")
    cs_y = CubicSpline(t, y, bc_type="periodic")

    # Generate smooth parametric values
    t_new = np.linspace(0, 1, num_points)
    x_new = cs_x(t_new)
    y_new = cs_y(t_new)

    return x_new, y_new


# def parametric_interpolation(x, y, num_points=200):
#     """Interpolates the airfoil as a single continuous curve using parametric cubic splines."""
#     # Create a parameter t based on the index
#     t = np.linspace(0, 1, len(x))

#     # Create cubic splines for both x(t) and y(t)
#     cs_x = CubicSpline(t, x)
#     cs_y = CubicSpline(t, y)

#     # Generate smooth parametric values
#     t_new = np.linspace(0, 1, num_points)
#     x_new = cs_x(t_new)
#     y_new = cs_y(t_new)

#     return x_new, y_new


def plot_airfoil(x, y, x_new, y_new):
    """Plots the original and interpolated airfoil."""
    plt.figure(figsize=(10, 5))
    plt.plot(x, y, "o", label="Original Data", alpha=0.5)
    plt.plot(x_new, y_new, label="Parametric Spline Interpolation", linewidth=2)
    plt.title("Airfoil Interpolation as a Single Spline")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.legend()
    plt.savefig("interpolated_airfoil.png")
    plt.show()


if __name__ == "__main__":
    filename = "naca2412.dat"

    # Load data
    x, y = load_airfoil_data(filename)

    # Perform parametric cubic spline interpolation
    x_new, y_new = parametric_interpolation(x, y)

    # Plot the interpolated airfoil
    plot_airfoil(x, y, x_new, y_new)
