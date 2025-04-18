import numpy as np
import matplotlib.pyplot as plt

# Constants
NX = 31
NY = 31
RE = 100.0
L = 1.0
H = 1.0
U0 = 1.0
MAX_ITER = 5000
TOL = 1e-8
SIGMA_C = 0.4
SIGMA_D = 0.6

dx = L / (NX - 1)
dy = H / (NY - 1)
beta = dx / dy
beta2 = beta * beta

# Allocate arrays
psi = np.full((NX, NY), 100.0)
omega = np.zeros((NX, NY))
u = np.zeros((NX, NY))
v = np.zeros((NX, NY))


def initialize():
    psi[:, :] = 100.0
    omega[:, :] = 0.0


def set_boundary_conditions():
    # Lid: Top boundary
    psi[:, NY - 1] = 100.0
    omega[1:NX - 1, NY - 1] = -2.0 * (psi[1:NX - 1, NY - 2]) / (dy ** 2) - 2.0 * U0 / dy

    # Bottom boundary
    psi[:, 0] = 0.0
    omega[1:NX - 1, 0] = -2.0 * psi[1:NX - 1, 1] / (dy ** 2)

    # Left boundary
    psi[0, :] = 0.0
    omega[0, 1:NY - 1] = -2.0 * psi[1, 1:NY - 1] / (dx ** 2)

    # Right boundary
    psi[NX - 1, :] = 0.0
    omega[NX - 1, 1:NY - 1] = -2.0 * psi[NX - 2, 1:NY - 1] / (dx ** 2)


def solve_stream_function():
    for _ in range(MAX_ITER):
        max_error = 0.0
        for i in range(1, NX - 1):
            for j in range(1, NY - 1):
                psi_new = (1 - SIGMA_C) * psi[i, j] + (SIGMA_C / (2 * (1 + beta2))) * (
                    psi[i + 1, j] + psi[i - 1, j] + beta2 * (psi[i, j + 1] + psi[i, j - 1]) +
                    dx * dx * omega[i, j]
                )
                max_error = max(max_error, abs(psi_new - psi[i, j]))
                psi[i, j] = psi_new
        if max_error < TOL:
            break


def solve_vorticity():
    for _ in range(MAX_ITER):
        max_error = 0.0
        for i in range(1, NX - 1):
            for j in range(1, NY - 1):
                u_local = (psi[i, j + 1] - psi[i, j - 1]) / (2 * dy)
                v_local = -(psi[i + 1, j] - psi[i - 1, j]) / (2 * dx)

                omega_xx = (omega[i + 1, j] + omega[i - 1, j]) / dx ** 2
                omega_yy = (omega[i, j + 1] + omega[i, j - 1]) / dy ** 2

                convection = (
                    u_local * (omega[i, j + 1] - omega[i, j - 1]) / (2 * dy) +
                    v_local * (omega[i + 1, j] - omega[i - 1, j]) / (2 * dx)
                )

                omega_new = (1 - SIGMA_D) * omega[i, j] + SIGMA_D * (
                    (omega_xx + omega_yy - RE * convection) / (2 * (1 / dx ** 2 + 1 / dy ** 2))
                )

                max_error = max(max_error, abs(omega_new - omega[i, j]))
                omega[i, j] = omega_new

        if max_error < TOL:
            break


def compute_velocity():
    for i in range(1, NX - 1):
        for j in range(1, NY - 1):
            u[i, j] = (psi[i, j + 1] - psi[i, j - 1]) / (2 * dy)
            v[i, j] = -(psi[i + 1, j] - psi[i - 1, j]) / (2 * dx)

    # Boundary conditions
    u[:, 0] = 0
    u[:, NY - 1] = U0
    v[:, 0] = 0
    v[:, NY - 1] = 0

    u[0, :] = 0
    v[0, :] = 0
    u[NX - 1, :] = 0
    v[NX - 1, :] = 0


def plot_results():
    X, Y = np.meshgrid(np.linspace(0, L, NX), np.linspace(0, H, NY), indexing='ij')

    plt.figure(figsize=(12, 4))

    # Streamfunction contours
    plt.subplot(1, 3, 1)
    plt.contourf(X, Y, psi, levels=50, cmap='jet')
    plt.colorbar()
    plt.title("Stream Function")
    plt.xlabel("X")
    plt.ylabel("Y")

    # Velocity vectors
    plt.subplot(1, 3, 2)
    skip = (slice(None, None, 2), slice(None, None, 2))
    plt.quiver(X[skip], Y[skip], u[skip], v[skip])
    plt.title("Velocity Field")
    plt.xlabel("X")
    plt.ylabel("Y")

    # Vorticity contours
    plt.subplot(1, 3, 3)
    plt.contourf(X, Y, omega, levels=50, cmap='jet')
    plt.colorbar()
    plt.title("Vorticity")
    plt.xlabel("X")
    plt.ylabel("Y")

    plt.tight_layout()
    plt.show()


# Driver code
if __name__ == "__main__":
    initialize()
    for _ in range(100):  # Outer loop for coupling
        set_boundary_conditions()
        solve_stream_function()
        solve_vorticity()
    compute_velocity()
    plot_results()
