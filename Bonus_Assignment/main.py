from airfoil import load_airfoil_data, parametric_interpolation, plot_airfoil
from tfi import grid_generation, plot_grid
from elliptic_grid_generation import GS_iterartion


num_xi = 101
num_eta = 81
R_outer = 10.0
# Get airfoil coordinates
filename = "naca2412.dat"

# Load data
x_afl_pts, y_afl_pts = load_airfoil_data(filename)

# Perform parametric cubic spline interpolation
x_airfoil, y_airfoil = parametric_interpolation(x_afl_pts, y_afl_pts)
plot_airfoil(x_afl_pts, y_afl_pts, x_airfoil, y_airfoil)

# Generate the grid
x, y, ξ, η = grid_generation(num_xi, num_eta, x_airfoil, y_airfoil, R_outer)
plot_grid(x, y)

Δξ = ξ[1] - ξ[0]
Δη = η[1] - η[0]

x, y = GS_iterartion(x, y, ξ, η, Δξ, Δη)
plot_grid(x, y)