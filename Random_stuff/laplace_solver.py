# Description: Solves Laplace's equation in computational (ξ, η) space using Finite Difference method.
import numpy as np


def derivates(phi, del_xi, del_eta):
    phi_xi = (phi[1:, :] - phi[:-1, :]) / del_xi
    phi_eta = (phi[:, 1:] - phi[:, :-1]) / del_eta
    phi_xi_xi = (phi[2:, :] - 2 * phi[1:-1, :] + phi[:-2, :]) / del_xi**2
    phi_eta_eta = (phi[:, 2:] - 2 * phi[:, 1:-1] + phi[:, :-2]) / del_eta**2
    phi_xi_eta = (phi[2:, 2:] - phi[2:, :-2] - phi[:-2, 2:] + phi[:-2, :-2]) / (
        4 * del_xi * del_eta
    )
    return phi_xi, phi_eta, phi_xi_xi, phi_eta_eta, phi_xi_eta


def solve_laplace(xi, eta, num_xi, num_eta, U_inf=1.0, tol=1e-5, max_iter=10000):
    phi = np.zeros((num_xi, num_eta))

    del_xi = xi[1] - xi[0]
    del_eta = eta[1] - eta[0]

    # Solve Laplace's equation using Finite Difference method
    for _ in range(max_iter):
        phi_xi, phi_eta, phi_xi_xi, phi_eta_eta, phi_xi_eta = derivates(
            phi, del_xi, del_eta
        )
        phi_new = 0.25 * (phi_xi_xi + phi_eta_eta)

        # Check for convergence
        if np.linalg.norm(phi_new - phi) < tol:
            break

        phi = phi_new


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
