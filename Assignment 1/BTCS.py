from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import numpy as np
import matplotlib.pyplot as plt
from test_cases import test_cases

def create_banded_matrix(ν,Nx):
    """
    Creates a tridiagonal banded matrix A used in the BTCS method
    """
    diagonals = [-ν / 2, 1, ν / 2]
    offsets = [-1, 0, 1]
    A = diags(diagonals, offsets, shape=(Nx, Nx)).tocsr()
    A[0, 0] = 1
    A[-1, -1] = 1
    A[0, 1] = 0
    A[-1, -2] = 0
    return A

def BTCS(u, A, bc1, bc2): # Backward Time Central Space
    u_new = u.copy()
    u_new = spsolve(A, u)
    u_new[0] = bc1
    u_new[-1] = bc2
    return u_new

# Define the physical parameters
a = 1  # Speed of Propagataion

# Define the grid parameters
L = 1.0  # Length of the domain in the x direction
Nx = 101  # Number of grid points in the x direction
Δx = L/(Nx-1)  # Grid spacing in the x direction
#print('Grid spacing in the x direction: Δx =', Δx)

# Define the simulation parameters
sim_time = 0.35  # Total simulation time

x_values = np.linspace(0, L, Nx)

for i, test_case in enumerate(test_cases(x_values), start=1):
    u_init = np.array(test_case['initial_condition'])

    # Plot the initial condition
    plt.plot(x_values, u_init)
    plt.xlim(0, 1)
    plt.ylim(-1.5, 1.5)
    plt.grid()
    plt.title(f'Initial Condition - Test Case {i}')
    plt.savefig(f'Plots/Test_Case_{i}_Initial_Condition.png')
    plt.close()

    u = u_init.copy()  # Reset u to the initial condition for each method
    for ν in [0.5, 1.0, 1.5]:
        # print('CFL Number:', ν)
        Δt = ν * Δx / a  # time step size
        num_time_step = int(sim_time / Δt)  # Number of time steps
        # print('Number of time steps:', num_time_step)

        A = create_banded_matrix(ν, Nx)
        # Run the simulation
        for j in range(num_time_step):
            u = BTCS(u, A, test_case['boundary_conditions']['left'],
                     test_case['boundary_conditions']['right'])

        # Plot and save the results
        plt.plot(x_values, u, label=f"After {sim_time} seconds (numerically)")
        plt.title(f'BTCS{ν} - Test Case {i}')
        plt.xlim(0, 1)
        plt.ylim(-1.5, 1.5)
        plt.grid()
        plt.savefig(f'Plots/Test_Case_{i}_{ν}_BTCS.png')
        plt.close()
        u = u_init.copy()  # Reset u to the initial condition for each method

