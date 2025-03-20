# Code to improve grid generated using the TFI method by using elliptic grid generation.
import matplotlib.pyplot as plt
import numpy as np
from airfoil import load_airfoil_data, parametric_interpolation
from tfi import grid_generation

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


def coeff(x, y, ξ, η):
    dx_dξ = np.gradient(x, ξ, axis=0)
    dy_dξ = np.gradient(y, ξ, axis=0)

    dx_dη = np.gradient(x, η, axis=1)
    dy_dη = np.gradient(y, η, axis=1)

    a = dx_dξ**2 + dy_dξ**2
    b = dx_dξ * dx_dη + dy_dξ * dy_dη
    c = dx_dη**2 + dy_dη**2

    return a, b, c


Δξ = ξ[1] - ξ[0]
Δη = η[1] - η[0]


def GS_iterartion(x, y, ξ, η, Δξ, Δη, max_iter=1000, tol=1e-6):
    x_new, y_new = np.copy(x), np.copy(y)
    a, b, c = coeff(x, y, ξ, η)

    for i in range(1, len(ξ) - 1):
        for j in range(1, len(η) - 1):
            x_new[i, j] = (
                a[i, j] * (x[i + 1, j] + x[i - 1, j]) / Δξ**2
                + c[i, j] * (x[i, j + 1] + x[i, j - 1]) / Δη**2
                - b[i, j]
                * (
                    x[i + 1, j + 1]
                    - x[i + 1, j - 1]
                    + x[i - 1, j - 1]
                    - x[i - 1, j + 1]
                )
                / (2 * Δξ * Δη)
            ) / (2 * (a[i, j] / Δξ**2 + c[i, j] / Δη**2))

            y_new[i, j] = (
                a[i, j] * (y[i + 1, j] + y[i - 1, j]) / Δξ**2
                + c[i, j] * (y[i, j + 1] + y[i, j - 1]) / Δη**2
                - b[i, j]
                * (
                    y[i + 1, j + 1]
                    - y[i + 1, j - 1]
                    + y[i - 1, j - 1]
                    - y[i - 1, j + 1]
                )
                / (2 * Δξ * Δη)
            ) / (2 * (a[i, j] / Δξ**2 + c[i, j] / Δη**2))

    return x_new, y_new


plt.figure()
plt.plot(x, y, "b-", linewidth=0.5)
plt.plot(x.T, y.T, "r-", linewidth=0.5)
plt.axis("equal")
plt.show()


for i in range(1000):
    x, y = GS_iterartion(x, y, ξ, η, Δξ, Δη)

plt.figure()
plt.plot(x, y, "b-", linewidth=0.5)
plt.plot(x.T, y.T, "r-", linewidth=0.5)
plt.axis("equal")
plt.show()
