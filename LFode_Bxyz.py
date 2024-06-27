import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import mpl_toolkits.mplot3d.art3d as art3d
import pandas as pd
import numpy as np
from scipy.integrate import ode
import sys
sys.path.append(r'C:\Users\embut\Documents\REU\Code')
import Bxyz_field as Bxyz

#Parameters
I = 12**3       # Current thru the loop in Amperes
a = 2*10**3     # Current loop radius in meters

q = 1           # Charge of particle in e
m = 938*10**6   # Mass of particle in eV/c^2

initial = {'x':1.3*a,
           'y':0,
           'z':0,
           'vx':-8*10**3,
           'vy':0,
           'vz':10*2}     # Meters and m/s

d_f = 4*10**5         # meters
t_f = 10        # seconds

# This modification of scipy.integrate.ode assumes the current loop is centered at the origin.
def LFode(system,t_f,d_f,initial,args):
    # INPUTS:
    # system:   Input system of differential equations
    # t_f:      Time when the simulation stops if d_f isn't triggered (seconds)
    # d_f:      Distance from the origin when the simulation stops (meters)
    # initial:  Dictionary of initial conditions, the first 3 of which are position
    # args:     Vector of arguments for the system (q,m,B) = (ch|arge (e), mass (eV/c^2))
    # OUTPUTS:
    # output:   Solutions to the ODE system
    
    t0 = 0
    q,m = args
    q = q*1.602176634*(10**(-19))         # Charge of particle (converts e to Coulombs)
    m = m*1.78266192162790*(10**(-36))    # Converts eV to kg
    Bmag = np.linalg.norm(Bxyz.BHere(I,a,0,0,0), ord=2)     # Order of magnitude average in Teslas
    dt = (m*2*np.pi)/(10*abs(q)*Bmag*(10))
    
    keys = initial.keys()
    initial = [initial[s] for s in keys]
    keys = ['t'] + list(keys)
    y0 = [t0] + initial
    output = pd.DataFrame(columns=keys)
    output.loc[0] = y0
    
    sol = ode(system)
    sol.set_initial_value(initial,t0).set_f_params(q,m)
    
    t=0
    d = np.sqrt((initial[0])**2 + (initial[1])**2 + (initial[2])**2)
    while sol.successful() and d <= d_f and t <= t_f:
        t = t+dt
        y = sol.integrate(t)
        vector = [t]
        i = 0
        while i < len(y):
            vector.append(y[i])
            i+=1
        output.loc[len(output)] = vector
        d = np.sqrt((vector[1])**2 + (vector[2])**2 + (vector[3])**2)
    
    return output

#System of coupled ODEs from the Lorentz force law
def LorentzForce(t,inputs,q,m):
    # INPUTS:
    # t:        Time (seconds)
    # inputs:   Vector of position and velocity (6 components)
    #               position (x,y,z) is the first 3 components (meters)
    #               velocity (vx,vy,vz) is the last 3 components (meters/second)
    # q:        Charge of the particle (Coulombs)
    # m:        Mass of the particle (kg)
    # OUTPUTS:
    # v_a:      Coupled ODEs solving for velocity and acceleration (vx,vy,vz,ax,ay,az)
    
    x, y, z, vx, vy, vz = inputs
    Bx, By, Bz = Bxyz.BHere(I,a,x,y,z)      # Teslas
    v_a = [vx, vy, vz, (q/m)*(vy*Bz - vz*By), (q/m)*(vz*Bx - vx*Bz), (q/m)*(vx*By - vy*Bx)]
    
    return v_a

args = (q,m)
soln = LFode(LorentzForce,t_f,d_f,initial,args)
soln = pd.DataFrame(data=soln, columns=['t','x','y','z','vx','vy','vz'])

plt.figure(figsize=(20, 5), dpi=1200)
plt.subplot(1,3,1)
plt.plot(soln['t'], soln['x'], color='r', label='x(t)')
plt.plot(soln['t'], soln['vx'], color='g', label='$v_x(t)$')
plt.xlabel('Time (s)')
plt.gca().set_title('x-coordinates')
plt.legend(loc='lower right', frameon=True, edgecolor='k')
plt.grid()

plt.subplot(1,3,2)
plt.plot(soln['t'], soln['y'], color='r', label='y(t)')
plt.plot(soln['t'], soln['vy'], color='g', label='$v_y(t)$')
plt.xlabel('Time (s)')
plt.gca().set_title('y-coordinates')
plt.legend(loc='lower right', frameon=True, edgecolor='k')
plt.grid()

plt.subplot(1,3,3)
plt.plot(soln['t'], soln['z'], color='r', label='z(t)')
plt.plot(soln['t'], soln['vz'], color='g', label='$v_z(t)$')
plt.xlabel('Time (s)')
plt.gca().set_title('z-coordinates')
plt.legend(loc='lower right', frameon=True, edgecolor='k')
plt.grid()

plt.suptitle('Particle in a Dipole Magnetic Field', fontsize=20)

plt.show()

scale=1.5
ax = plt.figure(figsize=(10,7), dpi=1200).add_subplot(projection='3d')
ax.plot(soln['x'], soln['y'], soln['z'], color='green')
p = Circle((0, 0), a, color='blue', fill=False)
ax.add_patch(p)
art3d.pathpatch_2d_to_3d(p, z=0, zdir="z")
ax.set_xlabel("x (meters)", fontsize=15)
ax.set_ylabel("y (meters)", fontsize=15)
ax.set_zlabel("z (meters)", fontsize=15)
ax.set_xlim(-scale*a, scale*a)
ax.set_ylim(-scale*a, scale*a)
ax.set_zlim(-scale*a, scale*a)
ax.set_title('Trajectory of a Particle in a Dipole B-field', fontsize=20)
ax.tick_params(axis='x', which='major', pad=-5)
ax.tick_params(axis='y', which='major', pad=-5)
ax.tick_params(axis='z', which='major', pad=-5)
#ax.dist = 12
ax.set_box_aspect(aspect=None, zoom=0.8)

