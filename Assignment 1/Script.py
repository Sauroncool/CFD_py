from function import explicit_functions,method_names
from test_cases import test_cases
import matplotlib.pyplot as plt
import numpy as np

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

    for method_name, func in zip(method_names, explicit_functions()):
        u = u_init.copy()  # Reset u to the initial condition for each method

        for ν in [0.5, 1.0, 1.5]:
            #print('CFL Number:', ν)
            Δt = ν * Δx / a  # time step size
            num_time_step = int(sim_time / Δt)  # Number of time steps
            #print('Number of time steps:', num_time_step)

            # Run the simulation
            for j in range(num_time_step):
                u = func(u, ν, test_case['boundary_conditions']['left'],
                         test_case['boundary_conditions']['right'])

            # Plot and save the results
            plt.plot(x_values, u, label=f"After {sim_time} seconds (numerically)")
            plt.title(f'{method_name}{ν} - Test Case {i}')
            plt.xlim(0, 1)
            plt.ylim(-1.5, 1.5)
            plt.grid()
            plt.savefig(f'Plots/Test_Case_{i}_{ν}_{method_name}.png')
            plt.close()
            u = u_init.copy()  # Reset u to the initial condition for each method