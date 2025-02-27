import numpy as np


def solve_laplace(xi, eta, U_inf=1.0, tol=1e-5, max_iter=10000):
    """
    Solves Laplace's equation in computational (ξ, η) space using Gauss-Seidel method.
    Args:
        xi (ndarray): ξ-coordinates from grid generation.
        eta (ndarray): η-coordinates from grid generation.
        U_inf (float): Free stream velocity.
        tol (float): Convergence tolerance.
        max_iter (int): Maximum number of iterations.
    Returns:
        phi (ndarray): Potential field.
    """
    nx, ny = xi.shape
    phi = np.zeros((nx, ny))

    # Dirichlet BC: Far-field potential φ = U_inf * ξ
    phi[:, -1] = U_inf * xi[:, -1]  # Far-field boundary (η = 1)

    # Iterative solver
    for iteration in range(max_iter):
        phi_old = phi.copy()

        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                phi[i, j] = 0.25 * (
                    phi[i + 1, j] + phi[i - 1, j] + phi[i, j + 1] + phi[i, j - 1]
                )

        # Neumann BC: No-penetration (∂φ/∂n = 0) at the ellipse boundary
        phi[0, :] = phi[1, :]  # Bottom boundary (ellipse)

        # Convergence check
        if np.max(np.abs(phi - phi_old)) < tol:
            print(f"Converged in {iteration} iterations.")
            break
    else:
        print("Did not converge within the maximum number of iterations.")

    return phi


def compute_velocity(phi, xi, eta):
    """
    Computes velocity field from the potential field in (ξ, η) space.
    Args:
        phi (ndarray): Potential field.
        xi (ndarray): ξ-coordinates.
        eta (ndarray): η-coordinates.
    Returns:
        u (ndarray): Velocity in ξ-direction.
        v (ndarray): Velocity in η-direction.
    """
    u = np.zeros_like(phi)
    v = np.zeros_like(phi)

    # Central difference for interior points
    eps = 1e-6  # Small number to avoid division by zero
    u[1:-1, 1:-1] = (phi[2:, 1:-1] - phi[:-2, 1:-1]) / (
        xi[2:, 1:-1] - xi[:-2, 1:-1] + eps
    )
    v[1:-1, 1:-1] = (phi[1:-1, 2:] - phi[1:-1, :-2]) / (
        eta[1:-1, 2:] - eta[1:-1, :-2] + eps
    )

    return u, v


# Example usage (to be removed in final integration)
if __name__ == "__main__":
    from grid_generation import generate_grid

    nx, ny = 101, 81
    a, b = 1, 0.5
    R_far = 20
    xi, eta, x, y = generate_grid(nx, ny, a, b, R_far)
    phi = solve_laplace(xi, eta)
    u, v = compute_velocity(phi, xi, eta)
