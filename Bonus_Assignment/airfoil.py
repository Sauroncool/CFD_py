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


def parametric_interpolation(x, y, num_points=500):

    # Append the last point also to the front to close the trailing edge
    x = np.append(x[-1], x)
    y = np.append(y[-1], y) 

    # Create a parameter t based on cumulative distance
    t = np.linspace(0, 1, len(x))

    # Create cubic splines for both x(t) and y(t)
    cs_x = CubicSpline(t, x, bc_type="clamped")  # Clamped boundary for smooth merging
    cs_y = CubicSpline(t, y, bc_type="clamped")

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
