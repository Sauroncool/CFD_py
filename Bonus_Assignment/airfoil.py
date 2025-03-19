import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline

# Load airfoil data
x, y = [], []

with open("naca2412.dat", "r") as file:
    lines = file.readlines()[1:]  # Skip the first line with the airfoil name
    for line in lines:
        values = line.split()
        x.append(float(values[0]))
        y.append(float(values[1]))

# Convert to numpy arrays
x = np.array(x)
y = np.array(y)

# Find the true leading edge (smallest x value)
leading_edge_idx = np.argmin(x)

# Split upper and lower surfaces
x_upper, y_upper = x[: leading_edge_idx + 1], y[: leading_edge_idx + 1]  # Upper surface
x_lower, y_lower = x[leading_edge_idx:], y[leading_edge_idx:]  # Lower surface

# Reverse upper surface to have increasing x
x_upper, y_upper = x_upper[::-1], y_upper[::-1]

# Create cubic spline interpolation
cs_upper = CubicSpline(x_upper, y_upper)
cs_lower = CubicSpline(x_lower, y_lower)

# Generate new points for smooth interpolation
x_new = np.linspace(0, 1, 200)
y_upper_new = cs_upper(x_new)
y_lower_new = cs_lower(x_new)

# Plot interpolated airfoil
plt.figure(figsize=(10, 5))
plt.plot(x, y, "o", label="Original Data", alpha=0.5)
plt.plot(x_new, y_upper_new, label="Upper Surface Interpolation")
plt.plot(x_new, y_lower_new, label="Lower Surface Interpolation")
plt.title("Interpolated Airfoil Shape")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.gca().set_aspect("equal", adjustable="box")
plt.legend()
plt.show()
