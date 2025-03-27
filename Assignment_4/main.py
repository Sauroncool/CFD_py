# quasi-1-D Euler equations for the isentropic flow of a calorically perfect gas with γ = 1.4 through a nozzle

import numpy as np
import matplotlib.pyplot as plt

# Parameters
gamma = 1.4
R = 287.0

# Nozzle geometry
# A = 1.0 + 2.0(x − 1)2, 0 ≤ x ≤ 2
def A(x):
    return 1.0 + 2.0*(x - 1)**2

# Reservoir conditions
# p0 = 1.0133e5, T0 = 300.0
p0 = 1.0133e5
T0 = 300.0
rho0 = p0/(R*T0)#Pe/P0 = 0.585

'''At Inlet: The stagnation temperature and stagnation pressure at the inlet
boundary are assumed to be equal to the reservoir temperature and pressure.
The inlet velocity is extrapolated from the interior grid point (by setting
u1 = u2, where u1 is the velocity at the inlet boundary and u2 is the velocity at
the nearest interior grid point). Specify consistent temperature, pressure and
density from isentropic relation based on the inlet stagnation temperature
and stagnation pressure.'''

'''At Outlet: The exit pressure pe is specified using the exit-to-stagnation pres-
sure ratio, pe/p0 = 0.585. Density and velocity are extrapolated from the in-
terior domain, following the conditions ρImax = ρImax−1 and uImax = uImax−1.'''





# van-Leer Flux Vector Splitting (FVS) scheme
def fvs(U, dt, dx):
