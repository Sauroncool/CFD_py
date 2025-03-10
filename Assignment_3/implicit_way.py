from scipy.sparse import diags

# Grid size
l_x = 3.0
l_y = 4.0

Δx = 0.1
Δy = 0.1
β = Δx / Δy
γ = -2 * (1 + β**2)

nx = int(l_x / Δx)
ny = int(l_y / Δy)


def create_banded_matrix(β, γ, nx, ny):
    """
    Creates a tridiagonal banded matrix A used in the BTCS method
    """
    diagonals = [β**2, 1, γ, 1, β**2]
    offsets = [-3, -1, 0, 1, 3]
    A = diags(
        diagonals, offsets, shape=((nx - 1) * (ny - 1), (nx - 1) * (ny - 1))
    ).tocsr()
    return A


A = create_banded_matrix(β, γ, nx, ny)
print(A.toarray())
