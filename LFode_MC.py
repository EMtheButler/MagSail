# This code calculates the relativistic trajectory of a random sample of charged
# particles with thru the mangetic field of a magsail.
# It uses a Monte Carlo method to sample initial conditions within the parameter
# spaces of solar wind, solar cosmic rays, and galactic cosmic rays.

import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import random
from decimal import Decimal
from scipy.integrate import ode
import sys
sys.path.append(r'.\\')
import Bfield as Bxyz

#random.seed(0.01)
c = 299792458 # m/s
u0 = 4*np.pi*10**-7         # Permeability of free space in (m*kg)/(s^2*A^2)

###################### Input Parameters #######################################
EarthCurrent = 6199085033 #Amps
EarthCoreRadius = 2*10**6 # meters

I = round(float(10**9), 30)       # Current thru the loop in Amperes
LoopMass = round(float(10**5), 30)        # Mass of currnet loop in kg
a = round(float((LoopMass*8*10**10)/(2*np.pi*I*6*10**3)), 30)   # Current loop radius in meters
#I = round(float(10**-6), 30)
print('Loop current: ' + str("{:.0E}".format(Decimal(I))) + ' Amps')
print('Loop radius: ' + str(round(Decimal(a))) + ' m')

q = 1           # Charge of particle in e
m = 938*10**6   # Mass of particle in eV/c^2

d_f = 10*a         # meters
t_f = 0.1       # seconds

# Type of radiation 
#title = 'Solar Wind'               # Constrained to the Sun's position
#title = 'Solar Cosmic Rays'        # Constrained to the Sun's position
title = 'Galactic Cosmic Rays'     # Isotropic initial positions
NumSamples = 2000        # Number of particles sampled
NumSim = 1              # Number of times the simulation runs
CvT = 'Test'             # Control (B=0) versus test (B!=0)

InitialEnergy = 10**2 # Initial kinetic energy of particle in MeV
InitialVelocity = np.sqrt((c**2)*(1 - ((m/(m + (InitialEnergy*10**6)))**2)))    # m/s
print('Proton\'s initial velocity: ' + str(round(Decimal(InitialVelocity/c), 3)) + 'c')
print('Proton\'s initial energy: ' + str(InitialEnergy) + ' MeV')

###############################################################################

# Current loop coordinates
xx = [a*np.cos(theta) for theta in np.arange(0,2*3.14,0.01)]
yy = [a*np.sin(theta) for theta in np.arange(0,2*3.14,0.01)]
zz = [0 for theta in np.arange(0,2*3.14,0.01)]
'''
def RandNum(span):
    # This function gets a random number from a uniform distribution within the input range.
    # INPUTS:   range:      The span of the 1-dimensional parameter space [min, max]
    # OUTPUTS:  rand:       Random number within the parameter space range.
    
    minimum, maximum = span
    #span = maximum - minimum
    #choice = random.uniform(0,1)
    rand = random.uniform(minimum, maximum)#minimum + span*choice
    
    return rand
'''
def CtoV(celerity):
    # This function calculates velocity from celerity.
    # INPUTS:   celerity:     3-vector of celerity (ux, uy, uz) (m/s)
    # OUTPUTS:  velocity:     3-vector of velocity (vx, vy, vz) (m/s)
    
    ux, uy, uz = celerity
    factor = 1/np.sqrt(ux**2 + uy**2 + uz**2 + c**2)
    vx = (ux*c)*factor
    vy = (uy*c)*factor
    vz = (uz*c)*factor
    velocity = [vx, vy, vz]
    
    return velocity

def VtoC(velocity):
    # This function calculates celerity from velocity.
    # INPUTS:   velocity:     3-vector of particle velocity (vx, vy, vz) (m/s)
    # OUTPUTS:  celerity:     3-vector of particle celerity (ux, uy, uz) (m/s)
    
    if np.linalg.norm(velocity, ord=2) > c:
        raise ValueError("Velocity exceeds c")
    else:
        vx, vy, vz = velocity
        gamma = 1/np.sqrt(1 - ((vx**2 + vy**2 + vz**2)/(c**2)))
        ux = vx*gamma
        uy = vy*gamma
        uz = vz*gamma
        celerity = [ux, uy, uz]
        return celerity

def Dist(r):
    # This function finds the particle's distance from the current loop.
    # INPUTS:   r:        Coordinates of particle (meters)
    # OUTPUTS:  d_min:    Distance from current loop (meters)
    
    d = np.empty((len(xx)))
    i = 0
    while i < len(xx):
        cc = [xx[i], yy[i], zz[i]]
        d[i] = math.dist(cc,r)
        i+=1
    
    d_min = np.min(d)
    
    return d_min

def Inputs(RadiationType, NumSamples=5000):
    # This function generates an array of initial conditions spanning a chosen parameter space.
    # INPUTS:   RadiationType:  One of three types of radiation: (str)
    #                               "SolarWind",
    #                               "SCR",  (Solar Cosmic Ray)
    #                               "GCR"  (Galactic Cosmic Ray)
    #           NumSamples:     Number of samples between the bounds of the parameter space
    #                               (float, default = 5000)
    # OUTPUTS:  initial:        Array of initial conditions
    #           kinetic:        Array of initial kinetic energies of each particle (eV)

    SunAngularSize = 0.00872665     #rad
    if RadiationType=='Solar Wind':
        rho = d_f               # Initial ρ-coordinate (meters)
        theta_range = [(np.pi-SunAngularSize)/2, (np.pi+SunAngularSize)/2]
            # Initial θ spanning Sun's angular size (rad)
        vel_range = [3*10**5, 9*10**5]  # Initial velocity range (m/s)
        v_phi_range = [0, 2*np.pi]
        v_costheta_range = [-1, 1]
    elif RadiationType=='Solar Cosmic Rays':
        rho = d_f
        EarthTilt = 0.401426 #rad
        theta_range = [(np.pi-SunAngularSize)/2, (np.pi+SunAngularSize)/2]
        vel_range = [0.14*c, 0.43*c]
        v_phi_range = [0, 2*np.pi]
        v_costheta_range = [-1, 1]
    elif RadiationType=='Galactic Cosmic Rays':
        rho = d_f
        theta_range = [0, np.pi]
        vel_range = [0.43*c, 0.996*c]
        v_phi_range = [0, 2*np.pi]
        v_costheta_range = [-1, 1]
    else:
        raise ValueError('Error: Unrecognized Radiation Type')
    
    initial = pd.DataFrame(columns=['x','y','z','vx','vy','vz'])
    kinetic = []
    i = 0
    while i <= NumSamples:
        theta = random.uniform(theta_range[0], theta_range[1])
        vel =  random.uniform(vel_range[0], vel_range[1])   #InitialVelocity
        v_phi = random.uniform(v_phi_range[0], v_phi_range[1])
        v_costheta = random.uniform(v_costheta_range[0], v_costheta_range[1])

        r_Cart = [rho*np.sin(theta), 0, rho*np.cos(theta)] # Initial position (x,y,z)
        v_Cart = [vel*(np.sin(np.arccos(v_costheta)))*np.cos(v_phi),
                  vel*(np.sin(np.arccos(v_costheta)))*np.sin(v_phi),
                  vel*v_costheta] # Initial velocity (vx,vy,vz)
        
        rmag = np.linalg.norm(r_Cart, ord=2)
        vmag = np.linalg.norm(v_Cart, ord=2)
        v_dot_r =((r_Cart[0]*v_Cart[0])+(r_Cart[1]*v_Cart[1])+(r_Cart[2]*v_Cart[2]))/(rmag*vmag)
        if v_dot_r < np.arctan(-2*a/rho):
            initial.loc[i] = r_Cart + v_Cart
            gamma = 1/np.sqrt(1 - ((vel/c)**2))
            kinetic_energy = m*gamma - m
            kinetic.append(kinetic_energy)
        
        i+=1
        
    return initial, kinetic

def LFode(system,t_f,d_f,initial,args):
    # Solves the Lorentz force system of ODEs to find a particle's trajectory.
    # This function operates in the rest frame of the magsail (origin @ center).
    # INPUTS:   system:   Input system of differential equations
    #           t_f:      End condition: Max duration of simulaiton (seconds)
    #           d_f:      End condition: Distance from the origin that ends the program (meters)
    #           initial:  Dictionary of initial conditions (x,y,z,vx,vy,vz) (meters and m/s)
    #           args:     Tuple of arguments for the system (q,m) = (charge (e), mass (eV/c^2))
    # OUTPUTS:  output:   Solutions to the ODE system as a dataframe
    #           dist:     Distance of closest approach (meters)
    
    output = pd.DataFrame(columns=['t','x','y','z','vx','vy','vz','s'])
    dist = []
    angles = []

    q,m = args
    r = [initial['x'], initial['y'], initial['z']]
    v = [initial['vx'], initial['vy'], initial['vz']]
    u = VtoC(v)
    inputs = r + u
    output.loc[0] = [0] + r + v + [np.sqrt((initial['x'])**2 + (initial['y'])**2)]
    
    q = q*1.602176634*(10**(-19))         # Charge of particle (converts e to Coulombs)
    m = m*1.78266192162790*(10**(-36))    # Converts eV/c^2 to kg
    
    Bmag = np.linalg.norm(Bxyz.BHere(I,a,r[0],r[1],r[2]), ord=2) # Teslas
    if Bmag == 0:
        print(Bmag)
    dt1 = (m*2*np.pi)/(10*abs(q)*Bmag)     # Seconds
    d = Dist(r)  # Meters
    dist.append(d)
    vmag = np.linalg.norm(v, ord=2)
    if vmag == 0:
            print(vmag)
    dt2 = 0.05*d/np.linalg.norm(v, ord=2)   # Seconds
    if dt1 <= dt2 and dt1 > 10**-8:
        dt = dt1
    elif dt2 < dt1 and dt2 > 10**-8:
        dt = dt2
    else:
        dt = 10**-8
    
    d_xy = dt2*np.linalg.norm([v[0], v[1]], ord=2)
    angles.append(np.arccos(d_xy/(0.05*d))) # List of angles with the plane of the loop
    
    sol = ode(system)
    sol.set_initial_value(inputs,0).set_f_params(q,m)
    
    t=0
    i=0
    while sol.successful() and d <= d_f and t <= t_f and i < 750:
        i+=1
        t = t+dt
        y = sol.integrate(t)
        x,y,z,ux,uy,uz = y
        r = [x,y,z]
        s = np.sqrt(x**2 + y**2)        # Cylindrical radial coordinate
        v = CtoV([ux,uy,uz])
        vector = [t] + r + v + [s]
        output.loc[len(output)] = vector
        
        Bmag = np.linalg.norm(Bxyz.BHere(I,a,r[0],r[1],r[2]), ord=2) # Teslas
        if Bmag == 0:
            print(Bmag)
        dt1 = (m*2*np.pi)/(10*abs(q)*Bmag)       # Seconds
        d = Dist(r)   # Meters
        dist.append(d)
        vmag = np.linalg.norm(v, ord=2)
        if vmag == 0:
            print(vmag)
        dt2 = 0.05*d/np.linalg.norm(v, ord=2)    # Seconds
        if dt1 <= dt2 and dt1 > 10**-8:
            dt = dt1
        elif dt2 < dt1 and dt2 > 10**-8:
            dt = dt2
        else:
            dt = 10**-8

        d_xy = dt2*np.linalg.norm([v[0], v[1]], ord=2)
        angles.append(np.arccos(d_xy/(0.05*d)))
    if i > 500:
        print(i)

    index = dist.index(np.min(dist))
    dist = np.min(dist)         # Distance from loop at closest approach
    angle = angles[index]       # Angle b/w trajectory and loop plane at closest approach
    
    return output, dist, angle

def LorentzForce(t,inputs,q,m):
    # System of coupled ODEs from the Lorentz force law
    # INPUTS:   t:        Time (seconds)
    #           inputs:   Vector of position and momentum (6 components)
    #                       position (x,y,z) is the first 3 components (meters)
    #                       celerity (ux,uy,uz) is the last 3 components (m/s)
    #           q:        Charge of the particle (Coulombs)
    #           m:        Mass of the particle (kg)
    # OUTPUTS:  v_dp:     Coupled ODEs solving for velocity and proper accel.
    #                       (vx,vy,vz,dux/dt,duy/dt,duz/dt)
    
    x, y, z, ux, uy, uz = inputs
    v = CtoV([ux,uy,uz])
    vx, vy, vz = v
    Bx, By, Bz = Bxyz.BHere(I,a,x,y,z)      # Teslas
    v_du = [vx, vy, vz, (q/m)*(vy*Bz - vz*By), (q/m)*(vz*Bx - vx*Bz), (q/m)*(vx*By - vy*Bx)]
    
    return v_du

ClosestApproaches = []
# Run the simulation NumSim times, plot the first result, and print the average closest approach
for j in range(NumSim):
    print('Iteration ' + str(j+1))
    # Run simulation and add to 3D xyz and 2D rz plots
    initial_conditions, kinetic = Inputs(title, NumSamples)
    args = (q,m)
    #EarthTitle = 'Shielding from Solar Radiaiton Storms by Earth\'s Magnetosphere'

    distance = []
    angles = []

    fig1 = go.Figure()
    #plt.figure(figsize=(8, 5.5))
    i=0
    while i < len(initial_conditions):
        #print(initial_conditions.iloc[i])
        soln, dist, angle = LFode(LorentzForce,t_f,d_f,initial_conditions.iloc[i],args)

        if j == 0:
            fig1.add_traces(go.Scatter3d(x=soln['x'], y=soln['y'], z=soln['z'],
                                        mode="lines", name='Particle %s Trajectory' %(i+1), 
                                        marker=dict(color='blue')))
            fig1.add_traces(go.Scatter3d(x=[soln.loc[0,'x']], y=[soln.loc[0,'y']], 
                                        z=[soln.loc[0,'z']],
                                        name='Initial Position %s' %(i+1), 
                                        marker=dict(color='red', size=5)))
            fig1.add_traces(go.Scatter3d(x=[soln.iloc[-1,1]], y=[soln.iloc[-1,2]], 
                                         z=[soln.iloc[-1,3]],
                                         name='Final Position %s' %(i+1), 
                                         marker=dict(color='blue', size=5)))
            
            plt.plot(soln['s'], soln['z'])
            plt.xlabel('Radial Position (meters)')
            plt.ylabel('Axial Position (meters)')
            plt.title('%s MeV Proton Trajectory in Cylindrical Coordinates for I = %s A' 
                      %(round(Decimal(InitialEnergy)), "{:.0E}".format(Decimal(I))))
            
        distance.append(dist)
        angles.append(angle)
        #print(dist,i)
    
        i+=1

    if j == 0:
        fig1.add_traces(go.Scatter3d(x=xx, y=yy, z=zz, mode='lines', name='Current Loop', 
                                     marker=dict(color='red')))
        fig1.update_layout(title=dict(text='Radiation Shielding at I = 1 GA',#'%s MeV Proton at I = %s A' 
                                      #%(round(Decimal(InitialEnergy)), 
                                       # "{:.0E}".format(Decimal(I))), 
                                      font=dict(family='Times New Roman', size=50)), 
                           legend=dict(font=dict(family="Times New Roman", size=18)), 
                           height=750, width=1000)
        fig1.show()     
        plt.show()

        # 2D distance from loop plots
        plt.figure(figsize=(12, 5.5))
        plt.subplot(1,2,1)
        plt.scatter(kinetic, distance)
        plt.xlabel('Kinetic Energy (eV)')
        plt.ylabel('Closest Approach to Current Loop (meters)')
        plt.gca().set_title('Kinetic Energy v. Distance')

        plt.subplot(1,2,2)
        plt.scatter(angles, distance)
        plt.xlabel('Angle of Incidence (rad)')
        plt.ylabel('Closest Approach to Current Loop (meters)')
        plt.gca().set_title('Angle of Incidence v. Distance')

        plt.suptitle('%s MeV Proton: Closest Approach at I = %s A' 
                     %(round(Decimal(InitialEnergy)), "{:.0E}".format(Decimal(I))), fontsize=20)
        plt.show()

    ClosestApproaches.append(min(distance))

ClosestMean = np.mean(ClosestApproaches)
ClosestSTD = np.std(ClosestApproaches)
ClosestCL = 1.96    # Confidence level of 95%
ClosestPM = (ClosestCL*ClosestSTD)/np.sqrt(NumSim)      # Error bars for ClosestMean
print('Average closest approach: ' + str(round(ClosestMean)) + ' ' 
      + u"\u00B1" + ' ' + str(round(ClosestPM)) + ' m')

gamma = 1/np.sqrt(1 - (InitialVelocity**2)/(c**2))
InitialCelerity = InitialVelocity*gamma
CelerityCrit = ((q*1.602176634*(10**(-19)))*I*u0)/(2*np.pi*(m*1.78266192162790*(10**(-36))))
data = np.array([NumSim, m, q, I, a, LoopMass, CelerityCrit, InitialCelerity, ClosestMean, 
                 ClosestPM])

with open('..\\Data\\FinalData\\RadiationShielding.csv', 'a') as f_out:
    for entry in data:
        f_out.write(str(entry) + ',')
    f_out.write(CvT)
    f_out.write('\n')
