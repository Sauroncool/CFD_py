# Code for grid generation
import matplotlib.pyplot as plt
import numpy as np

# Define the grid size
num_xi = 101  # Number of points in the ξ-direction
num_eta = 81  # Number of points in the η-direction

# Define the semi-major and semi-minor axes for the ellipse
a = 1.0
b = 0.5

# Define the radius for the outer boundary (far field)
R_outer = 20.0

# Generate ξ and η values
xi = np.linspace(0, 1, num_xi)
eta = np.linspace(0, 1, num_eta)

# Initialize x and y grid arrays
x = np.zeros((num_eta, num_xi))
y = np.zeros((num_eta, num_xi))

# Compute the grid points using TFI
for i in range(num_eta):
    for j in range(num_xi):
        theta = 2 * np.pi * xi[j]

        # Inner boundary (ellipse)
        x_inner = a * np.cos(theta)
        y_inner = b * np.sin(theta)

        # Outer boundary (circle)
        x_outer = R_outer * np.cos(theta)
        y_outer = R_outer * np.sin(theta)

        # TFI mapping
        x[i, j] = (1 - eta[i]) * x_inner + eta[i] * x_outer
        y[i, j] = (1 - eta[i]) * y_inner + eta[i] * y_outer

# Plot the grid
plt.figure(figsize=(10, 8))
for i in range(num_eta):
    plt.plot(x[i, :], y[i, :], "b-", linewidth=0.5)  # Lines along ξ

for j in range(num_xi):
    plt.plot(x[:, j], y[:, j], "r-", linewidth=0.5)  # Lines along η

# Add imaginary cut line (dashed line along θ = 0)
plt.plot(
    [x[0, 0], x[-1, 0]],
    [y[0, 0], y[-1, 0]],
    "k--",
    linewidth=1.5,
    label="Imaginary Cut (θ = 0)",
)

# Highlight corresponding points a, b, c, d
# Point a and d (inner boundary start/end)
plt.scatter(x[0, 0], y[0, 0], color="purple", s=80, label="Point a/b")
plt.text(x[0, 0], y[0, 0] + 0.5, "b", fontsize=14, color="black")
plt.text(x[0, 0], y[0, 0] - 1.0, "a", fontsize=14, color="black")

# Point b and c (outer boundary start/end)
plt.scatter(x[-1, 0], y[-1, 0], color="green", s=80, label="Point d/c")
plt.text(x[-1, 0], y[-1, 0] + 0.5, "c", fontsize=14, color="black")
plt.text(x[-1, 0], y[-1, 0] - 1.0, "d", fontsize=14, color="black")

# Formatting the plot
plt.gca().set_aspect("equal")
plt.title("Transfinite Interpolation Grid (101 × 81)")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True, linestyle="--", alpha=0.5)

# Save the plot
plt.savefig("24M0003.png")
plt.show()
