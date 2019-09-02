#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 12:00:58 2018

@author: tanthanhnhanphan
"""
"""Source-Panel method"""



import os
import math
import matplotlib.pyplot as plt
from scipy import integrate
import numpy as np


class Panel:
    """
    Contains information related to a panel.
    """
    def __init__(self, xa, ya, xb, yb):
        """
        Initializes the panel.
        
        Sets the end-points and calculates the center, length,
        and angle (with the x-axis) of the panel.
        Defines if the panel is on the lower or upper surface of the geometry.
        Initializes the source-sheet strength, tangential velocity,
        and pressure coefficient to zero.
        
        Parameters
        ----------
        xa: float
            x-coordinate of the first end-point.
        ya: float
            y-coordinate of the first end-point.
        xb: float
            x-coordinate of the second end-point.
        yb: float
            y-coordinate of the second end-point.
        """
        self.xa, self.ya = xa, ya
        self.xb, self.yb = xb, yb
        
        self.xc, self.yc = (xa + xb) / 2, (ya + yb) / 2  # control-point (center-point)
        self.length = math.sqrt((xb - xa)**2 + (yb - ya)**2)  # length of the panel
        
        # orientation of the panel (angle between x-axis and panel's normal)
        if xb - xa <= 0.0:
            self.beta = math.acos((yb - ya) / self.length)
        elif xb - xa > 0.0:
            self.beta = math.pi + math.acos(-(yb - ya) / self.length)
        
        # location of the panel
        if self.beta <= math.pi:
            self.loc = 'upper'
        else:
            self.loc = 'lower'
        
        self.sigma = 0.0  # source strength
        self.vt = 0.0  # tangential velocity
        self.cp = 0.0  # pressure coefficient


def define_panels(x, y, N=40):
    """
    Discretizes the geometry into panels using the 'cosine' method.
    
    Parameters
    ----------
    x: 1D array of floats
        x-coordinate of the points defining the geometry.
    y: 1D array of floats
        y-coordinate of the points defining the geometry.
    N: integer, optional
        Number of panels;
        default: 40.
    
    Returns
    -------
    panels: 1D Numpy array of Panel objects
        The discretization of the geometry into panels.
    """
    R = (x.max() - x.min()) / 2  # radius of the circle
    x_center = (x.max() + x.min()) / 2 # x-coord of the center
    # define x-coord of the circle points
    x_circle = x_center + R * np.cos(np.linspace(0.0, 2*math.pi, N + 1))
    
    #Projection of the x-coordinate on the surface
    x_airfoil = np.copy(x_circle)
    #Initialization of the y-coordinate array
    y_airfoil = np.empty_like(x_airfoil)
    #Extend the array
    x, y = np.append(x, x[0]), np.append(y, y[0])
    
    # computes the y-coordinate of end-points
    I = 0
    for i in range(N):
        while I < len(x) - 1:
            if (x[I] <= x_airfoil[i] <= x[I + 1]) or (x[I + 1] <= x_airfoil[i] <= x[I]):
                break
            else:
                I += 1
        a = (y[I + 1] - y[I]) / (x[I + 1] - x[I])
        b = y[I + 1] - a * x[I + 1]
        y_airfoil[i] = a * x_airfoil[i] + b
    y_airfoil[N] = y_airfoil[0]
    
    panels = np.empty(N, dtype=object)
    for i in range(N):
        panels[i] = Panel(x_airfoil[i], y_airfoil[i], x_airfoil[i + 1], y_airfoil[i + 1])
    
    return panels


"""     
def define_panels(x, y, N = 40):
    
    #Radius of the circle
    R = (x.max() - x.min()) / 2
    #x-coordinate of the center
    x_center = (x.max() + x.min()) / 2
    
    #Define the x-coordinate of the circle
    x_circle = x_center + R * np.cos(np.linspace(0.0, 2*math.pi, N + 1))
    
    #Projection of the x-coordinate on the surface
    x_airfoil = np.copy(x_circle)
    #Initialization of the y-coordinate array
    y_airfoil = np.empty_like(x_airfoil)
    #Extend the array
    x, y = np.append(x, x[0]), np.append(y, y[0])
    
    #Compute the y-coordinate of the end-points
    
    I = 0
    for i in range(N):
        while I < len(x) - 1:
            if (x[I] <= x_airfoil[i] <= x[I+1]) or (x[I+1] <= x_airfoil[i] <= x[I]):
                break
            else:
                I += 1
        a = (y[I + 1] - y[I]) / (x[I + 1] - x[I])
        b = y[I + 1] - a * x[I + 1]
        y_airfoil[i] = a * x_airfoil[i] + b
    y_airfoil[N] = y_airfoil[i]
    
    panels = np.empty(N, dtype = object)
    for i in range(N):
        panels[i] = Panel(x_airfoil[i], y_airfoil[i], x_airfoil[i+1], y_airfoil[i+1])
    
    return panels
"""

class Freestream:
    """
    Class contains information about a freestream
    """
    def __init__(self, u_inf = 1.0, AoA = 0.0):
        """
        Sets the freestream speed and angle (wrt to x-axis)
        
        Inputs:
            u_inf: freestream speed
            AoA: Angle of attack
        """
        self.u_inf = u_inf
        self.AoA = np.radians(AoA)
    
    
"""Read the geometry from a data file"""
naca_0012 = os.path.join('naca0012.dat')
with open (naca_0012, 'r') as file_name:
    x, y = np.loadtxt(file_name, dtype = float, delimiter = '\t', unpack = 'True')
    
"""Plot the NACA0012 airfoil"""
width = 10
plt.figure(figsize = (width, width))
plt.grid()
plt.xlabel('x', fontsize = 16)
plt.ylabel('y', fontsize = 16)
plt.plot(x, y, color = '#CD2305', linestyle = '-', linewidth = 2)
plt.axis('scaled', adjustable = 'box')
plt.xlim(-0.1, 1.1)
plt.ylim(-0.1, 0.1)
plt.show()

"""Plot the discretization"""

        
#Define the freestream
u_inf = 1.0
AoA = 4.0
freestream = Freestream(u_inf, AoA)


def integral(x, y, panel, dxdz, dydz):
    """
    Evaluates the contribution of a panel at one point
    
    Inputs:
        x: x-coordinate of the target point
        y: y-coordinate of the target point
        panel: Source panel which contribution is evaluated
        dxdz: Derivative of x in z-direction
        dydx: Derivative of y in z-direction
    """
    def integrand(s):
        return (((x - (panel.xa - math.sin(panel.beta) * s)) * dxdz +
                 (y - (panel.ya + math.cos(panel.beta) * s)) * dydz) /
                ((x - (panel.xa - math.sin(panel.beta) * s))**2 +
                 (y - (panel.ya + math.cos(panel.beta) * s))**2) )
    return integrate.quad(integrand, 0.0, panel.length)[0]


def build_matrix(panels):
    """
    Builds the source matrix.
    
    Parameters
    ----------
    panels: 1D array of Panel object
        The source panels.
    
    Returns
    -------
    A: 2D Numpy array of floats
        The source matrix (NxN matrix; N is the number of panels).
    """
    N = len(panels)
    A = np.empty((N, N), dtype=float)
    np.fill_diagonal(A, 0.5)
    
    for i, p_i in enumerate(panels):
        for j, p_j in enumerate(panels):
            if i != j:
                A[i, j] = 0.5 / math.pi * integral(p_i.xc, p_i.yc, p_j,
                                                   math.cos(p_i.beta),
                                                   math.sin(p_i.beta))
    
    return A

def build_rhs(panels, freestream):
    """
    Builds the RHS of the linear system.
    
    Parameters
    ----------
    panels: 1D array of Panel objects
        The source panels.
    freestream: Freestream object
        The freestream conditions.
    
    Returns
    -------
    b: 1D Numpy array of floats
        RHS of the linear system.
    """
    b = np.empty(len(panels), dtype=float)
    
    for i, panel in enumerate(panels):
        b[i] = -freestream.u_inf * math.cos(freestream.AoA - panel.beta)
    
    return b
A = build_matrix(panels)           # compute the singularity matrix
b = build_rhs(panels, freestream)

#Solve the linear system
sigma = np.linalg.solve(A, b)

for i, panel in enumerate(panels):
    panel.sigma = sigma[i]
    
def tangential_velocity(panels, freestream):
    """
    Compute the velocity on the surface of the panels
    
    Inputs:
        panels: 1D array of panel objects
        freestream: Freetsream object
    """
    N = len(panels)
    A = np.empty((N, N), dtype = float)
    np.fill_diagonal(A, 0.0)
    
    for i, p_i in enumerate(panels):
        for j, p_j in enumerate(panels):
            if i != j:
                A[i, j] = 0.5 / math.pi * integral(p_i.xc, p_i.yc, p_j, -math.sin(p_i.beta), math.cos(p_i.beta))
    
#    b = freestream.u_inf * np.sin([freestream.AoA - panel.beta] for panel in panels)
    b = freestream.u_inf * np.sin([freestream.AoA - panel.beta 
                                      for panel in panels])
    
    sigma = np.array([panel.sigma for panel in panels])
    
    vt = np.dot(A, sigma) + b
    
    for i, panel in enumerate(panels):
        panel.vt = vt[i]


tangential_velocity(panels, freestream)

def pressure_coefficient(panels, freestream):
    """
    Computes the surface pressure coefficient on the panels
    
    Inputs:
        panels: 1D array of panel objects
        freestream: Freetsream object
    """
    for panel in panels:
        panel.cp = 1.0 - (panel.vt / freestream.u_inf) **2

pressure_coefficient(panels, freestream)
    
    
                
"""Theoretical solution (from Jupyter notebook)"""
voverVsquared=np.array([0.0, 0.64, 1.01, 1.241, 1.378, 1.402, 1.411, 1.411,
                           1.399, 1.378, 1.35, 1.288, 1.228, 1.166, 1.109, 1.044,
                           0.956, 0.906, 0.0])
print(voverVsquared)
       
xtheo=np.array([0.0, 0.5, 1.25, 2.5, 5.0, 7.5, 10.0, 15.0, 20.0, 25.0, 30.0,
                   40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 95.0, 100.0])
xtheo /= 100
print(xtheo)

plt.figure(figsize = (10, 6))
plt.grid()
plt.xlabel('x', fontsize = 16)
plt.ylabel('y', fontsize = 16)
plt.title('Number of panels: {}'.format(N))
plt.plot([panel.xc for panel in panels if panel.loc == 'upper'],
         [panel.cp for panel in panels if panel.loc == 'upper'],
         label = 'upper',
         color = '#CD2305', linewidth = 1, marker ='x', markersize = 8)
plt.plot([panel.xc for panel in panels if panel.loc == 'lower'],
         [panel.cp for panel in panels if panel.loc == 'lower'],
         label = 'lower',
         color='b', linewidth=0, marker='d', markersize=6)
plt.plot(xtheo, 1 - voverVsquared, label = 'theoretical', color = 'k', linestyle = '--', linewidth = 2)
plt.legend(loc = 'best', prop={'size': 16})
plt.xlim(-0.1, 1.1)
plt.ylim(1.0, -0.6)
plt.show()

"""Check the accuracy"""
"""For a closed body, the sum of all the source strengths should be 0.
If not, it means that the body would be adding/ absorbing mass from the flow
"""
accuracy = sum([panel.sigma*panel.length for panel in panels])
print(accuracy)



"""Compute mesh grid and velocity field"""

def velocity_field(panels, freestream, X, Y):
    """
    Computes the velocity u and v
    
    Inputs:
        panels: 1D array of panel objects
        freestream: Freetsream object
        X: 2D numpy array of floats - x-coordinates of mesh points
        Y: 2D numpy array of floats - y-coordinates of mesh points
    
    Outputs:
        u, v
    """
    #freestream
    u = freestream.u_inf * math.cos(freestream.AoA) * np.ones_like(X, dtype = float)
    v = freestream.u_inf * math.sin(freestream.AoA) * np.ones_like(X, dtype = float)
    
    #vectorize the integral - to avoid the nested loops
    vec_integral = np.vectorize(integral)
    
    #Add the source panels
    for panel in panels:
        u += panel.sigma / (2 * math.pi) * vec_integral(X, Y, panel, 1.0, 0.0)
        v += panel.sigma / (2 * math.pi) * vec_integral(X, Y, panel, 0.0, 1.0)
    
    return u, v

#Generate a mesh grid
n = 10
x_start, x_end = -1.0, 2.0
y_start, y_end = -0.3, 0.3
x = np.linspace(x_start, x_end, n)
y = np.linspace(y_start, y_end, n)
X, Y = np.meshgrid(x, y)

#compute the velocity field
u, v = velocity_field(panels, freestream, X, Y)

#plot the streamlines
width = 10
plt.figure(figsize = (width, width))
plt.xlabel('x', fontsize = 16)
plt.ylabel('y', fontsize = 16)
plt.title('Streamline around a NACA0012 airfoil with ${}^o$ AoA'.format(AoA), fontsize = 16)
plt.streamplot(X, Y, u, v, density = 2, linewidth = 1, arrowsize = 1, arrowstyle = '->')
plt.fill([panel.xc for panel in panels], [panel.yc for panel in panels], color ='black', linestyle = 'solid', linewidth = 2, zorder = 2)
plt.xlim(0.9, 1.1)
plt.ylim(-0.03, 0.03)
plt.axis('scaled', adjustable = 'box')
plt.show()


"""Compute the pressure field around an airfoil"""
cp = 1 - (u**2 + v**2) / (u_inf**2)

width = 10
plt.figure(figsize = (width, width))
plt.xlabel('x', fontsize = 16)
plt.ylabel('y', fontsize = 16)
contf = plt.contourf(X, Y, cp, levels = np.linspace(-0.5, 0.5, 100), extend = 'both')
cbar = plt.colorbar(contf, orientation='horizontal', shrink=0.5, pad = 0.1, ticks=[-1.0, -0.5, 0.0, 0.5, 1.0])
cbar.set_label('$C_p$', fontsize = 16)
plt.fill([panel.xc for panel in panels], [panel.yc for panel in panels], color = 'k', linestyle = 'solid', linewidth = 2, zorder = 2)
plt.xlim(x_start, x_end)
plt.ylim(y_start, y_end)
plt.axis('scaled', adjustable = 'box')
plt.show()

 
    
    
    
    
    
      
        
        
        
        
        
        
        
        
        
        
        
        
    