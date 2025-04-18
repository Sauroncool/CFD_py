from airfoil import load_airfoil_data, parametric_interpolation, plot_airfoil
from elliptic_grid_generation import GS_iteration
from tfi_square import grid_generation, plot_grid

num_xi = 40
num_eta = 40
# Get airfoil coordinates
#filename = "naca2412.dat"
filename = "NACA63412 coordinates.csv"  # Use the new CSV format

# Load data
x_afl_pts, y_afl_pts = load_airfoil_data(filename)

# Perform parametric cubic spline interpolation
x_airfoil, y_airfoil = parametric_interpolation(x_afl_pts , y_afl_pts, num_xi)
plot_airfoil(x_afl_pts, y_afl_pts, x_airfoil, y_airfoil)

# Generate the grid
x, y, ξ, η = grid_generation(num_xi, num_eta, x_airfoil , y_airfoil) # Offset the airfoil to center it in the grid
plot_grid(x, y, "grid_tfi.png")

x, y = GS_iteration(x, y, ξ, η, max_iter=50000, tolerance=1e-6)
plot_grid(x, y, "grid_elliptic.png")

