# List of method names for titles
from numpy.f2py.capi_maps import cb_sign2map

method_names = ["FTFS", "FTCS", "FTBS", "Lax-Wendroff", "Beam Warming", "Fromm"]

# Functions
def FTFS(u, ν, bc1, bc2): # Forward Time Forward Space
    u_new = u.copy()
    u_new[0] = bc1
    u_new[1:-1] = (u[1:-1]) - ν * (u[2:] - u[1:-1])
    u_new[-1] = bc2
    return u_new

def FTCS(u, ν, bc1, bc2): # Forward Time Central Space
    u_new = u.copy()
    u_new[0] = bc1
    u_new[1:-1] = (u[1:-1]) - (ν/2) * (u[2:] - u[:-2])
    u_new[-1] = bc2
    return u_new

def FTBS(u, ν, bc1, bc2): # Forward Time Backward Space
    u_new = u.copy()
    u_new[0] = bc1
    u_new[1:-1] = (u[1:-1]) - ν * (u[1:-1] - u[:-2])
    u_new[-1] = bc2
    return u_new

def LW(u, ν, bc1, bc2): # Lax-Wendroff
    u_new = u.copy()
    u_new[0] = bc1
    u_new[1:-1] = (u[1:-1]) - (ν/2) * (u[2:] - u[:-2]) + (ν**2/2) * (u[2:] - 2*u[1:-1] + u[:-2])
    u_new[-1] = bc2
    return u_new

def BW(u,ν,bc1,bc2): # Beam Warming
    u_new = u.copy()
    u_new[0] = bc1
    u_new[1] = bc1
    u_new[2:-1] = (u[2:-1]) - (ν/2) * (3*u[2:-1] - 4*u[1:-2] + u[:-3]) + (ν**2/2) * (u[2:-1] - 2*u[1:-2] + u[:-3])
    u_new[-1] = bc2
    return u_new

def Fromm(u,ν,bc1,bc2):
    u_new = u.copy()
    u_new[0]=bc1
    u_new[1] = bc1
    u_new[2:-1] = 0.5*((u[2:-1]) - (ν/2) * (u[3:] - u[1:-2]) + (ν**2/2) * (u[3:] - 2*u[2:-1] + u[1:-2])+(u[2:-1]) - (ν/2) * (3*u[2:-1] - 4*u[1:-2] + u[:-3]) + (ν**2/2) * (u[2:-1] - 2*u[1:-2] + u[:-3]))
    u_new[-1] = bc2
    return u_new

def explicit_functions():
    return [FTFS, FTCS, FTBS, LW, BW, Fromm]