import matplotlib.pyplot as plt
from airfoil import load_airfoil_data, parametric_interpolation
from tfi import grid_generation


def elliptic_smoothing(x, y, num_iterations=5000, tol=1e-6):
    num_xi, num_eta = x.shape
    x_new, y_new = x.copy(), y.copy()

    for _ in range(num_iterations):
        x_old, y_old = x_new.copy(), y_new.copy()
        max_error = 0

        for i in range(1, num_xi - 1):
            for j in range(1, num_eta - 1):
                x_new[i, j] = 0.25 * (
                    x_old[i + 1, j]
                    + x_old[i - 1, j]
                    + x_old[i, j + 1]
                    + x_old[i, j - 1]
                )
                y_new[i, j] = 0.25 * (
                    y_old[i + 1, j]
                    + y_old[i - 1, j]
                    + y_old[i, j + 1]
                    + y_old[i, j - 1]
                )

                max_error = max(
                    max_error,
                    abs(x_new[i, j] - x_old[i, j]),
                    abs(y_new[i, j] - y_old[i, j]),
                )

        if max_error < tol:
            print(f"Converged after {_ + 1} iterations.")
            break

    return x_new, y_new


def plot_grid(x, y, title="Elliptic Grid"):
    plt.figure()
    plt.plot(x, y, "b-", linewidth=0.5)
    plt.plot(x.T, y.T, "r-", linewidth=0.5)
    plt.axis("equal")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.grid(linestyle="--", alpha=0.5)
    plt.show()


if __name__ == "__main__":
    num_xi, num_eta = 101, 81
    R_outer = 10.0
    filename = "naca2412.dat"

    x_airfoil, y_airfoil = parametric_interpolation(*load_airfoil_data(filename))
    x, y = grid_generation(num_xi, num_eta, x_airfoil, y_airfoil, R_outer)

    x_smoothed, y_smoothed = elliptic_smoothing(x, y)

    plot_grid(x, y, title="TFI Grid")
    plot_grid(x_smoothed, y_smoothed, title="Elliptic Grid")
