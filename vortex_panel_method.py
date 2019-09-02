#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 15:42:58 2018

@author: Tan Thanh Nhan Phan
"""

"""Vortex-Panel Method"""

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import integrate
import os

"""Read the geometry from data file"""
naca0012 = os.path.join('naca0012.dat')
with open(naca0012, 'r') as file_name:
    x, y = np.loadtxt(file_name, dtype =float, delimiter = '\t',unpack = True)

"""Plot the geometry"""
"""
width = 10
plt.figure(figsize= (width, width))
plt.grid()
plt.xlabel('x', fontsize = 16)
plt.ylabel('y', fontsize = 16)
plt.plot(x, y, color = 'k', linestyle = '-', linewidth = 2)
plt.axis('scaled', adjustable = 'box')
plt.xlim(-0.1, 1.1)
plt.ylim(-0.1, 0.1)
plt.show()
"""

class Panel:
    """
    Class contains information about a panel
    
    Inputs:
        xa: x-coordinate of first end point
        ya: y-coordinate of first end point
        xb: x-coordinate of second end point
        yb: y-coordinate of second end point
    """
    def __init__(self, xa, ya, xb, yb):
        self.xa, self.ya = xa, ya
        self.xb, self.yb = xb, yb
        
        self.xc, self.yc = (xa + xb) / 2, (ya + yb) / 2
        self.length = np.sqrt((xb - xa)**2 + (yb - ya)**2)
        
        if xb - xa <= 0.0:
            self.beta = np.arccos((yb - ya) / self.length)
        elif xb - xa > 0.0:
            self.beta = np.pi + np.arccos(- (yb - ya) / self.length)
        
        if self.beta <= np.pi:
            self.loc = 'upper'
        else:
            self.loc = 'lower'
        
        self.sigma = 0.0 #Source strength
        self.vt = 0.0 #Tangential velocity
        self.cp = 0.0 #Pressure coeffcient

def define_panels(x, y, N = 40):
    """
    Discretizes the geometry into panels using cosine method
    
    Inputs:
        x: 1D array of the points defining the geometry
        y: 1D array of the points defining the geometry
        N: number of panels
    Outputs:
        panels: 1D numpy array of Panel objects
    """
    
    R = (x.max() - x.min()) / 2.0  # circle radius
    x_center = (x.max() + x.min()) / 2.0  # x-coordinate of circle center
    
    theta = np.linspace(0.0, 2.0 * np.pi, N + 1)  # array of angles
    x_circle = x_center + R * np.cos(theta)  # x-coordinates of circle
    
    x_ends = np.copy(x_circle)  # x-coordinate of panels end-points
    y_ends = np.empty_like(x_ends)  # y-coordinate of panels end-points
    
    # extend coordinates to consider closed surface
    x, y = np.append(x, x[0]), np.append(y, y[0])
    
    # compute y-coordinate of end-points by projection
    I = 0
    for i in range(N):
        while I < len(x) - 1:
            if (x[I] <= x_ends[i] <= x[I + 1]) or (x[I + 1] <= x_ends[i] <= x[I]):
                break
            else:
                I += 1
        a = (y[I + 1] - y[I]) / (x[I + 1] - x[I])
        b = y[I + 1] - a * x[I + 1]
        y_ends[i] = a * x_ends[i] + b
    y_ends[N] = y_ends[0]
    
    # create panels
    panels = np.empty(N, dtype=object)
    for i in range(N):
        panels[i] = Panel(x_ends[i], y_ends[i], x_ends[i + 1], y_ends[i + 1])
    
    return panels

#Discretize geometry into panels
panels = define_panels(x, y, N = 40)
"""
#Plot discretized geometry
width = 10
plt.figure(figsize= (width, width))
plt.grid()
plt.xlabel('x', fontsize = 16)
plt.ylabel('y', fontsize = 16)
plt.plot(x, y, color = 'k', linestyle = '-', linewidth = 2)
plt.plot(np.append([panel.xa for panel in panels], panels[0].xa), np.append([panel.ya for panel in panels], panels[0].ya), linestyle = '-', linewidth = 1, markersize = 6, marker = 'o', color = '#CD2305')
plt.axis('scaled', adjustable = 'box')
plt.xlim(-0.1, 1.1)
plt.ylim(-0.1, 0.1)
plt.show()
"""
class Freestream:
    """
    Class contains freestream conditions
    """
    def __init__(self, u_inf, AoA):
        self.u_inf = u_inf
        self.AoA = np.radians(AoA)

#Define the freestream
u_inf = 20.0
AoA = 5.0
freestream = Freestream(u_inf, AoA)

def integral(x, y, panel, dxdk, dydk):
   
    def integrand(s):
        return ((x - (panel.xa - np.sin(panel.beta) * s)) * dxdk + \
                (y - (panel.ya + np.cos(panel.beta) * s)) * dydk) / \
                ((x - (panel.xa - np.sin(panel.beta) * s)) ** 2 + \
                (y - (panel.ya + np.cos(panel.beta) * s)) **2 )
    return integrate.quad(integrand, 0.0, panel.length)[0]

"""
def source_normal(panels):
    
    A = np.empty((panels.size, panels.size), dtype = float)
    np.fill_diagonal(A, 0.5)
    #Source contribution on a panel from others
    for i, p_i in enumerate(panels):
        for j, p_j in enumerate(panels):
            if i != j:
                A[i, j] = 0.5 / np.pi * integral(p_i.xc, p_i.yc, p_j, np.cos(p_i.beta), np.sin(p_i.beta))
    
    return A
"""
def source_normal(panels):
    """
    Builds the source contribution matrix for the normal velocity.
    
    Inputs
    ----------
    panels: 1D array of Panel objects
        List of panels.
    
    Outputs
    -------
    A: 2D Numpy array of floats
        Source contribution matrix.
    """
    A = np.empty((panels.size, panels.size), dtype=float)
    # source contribution on a panel from itself
    np.fill_diagonal(A, 0.5)
    # source contribution on a panel from others
    for i, panel_i in enumerate(panels):
        for j, panel_j in enumerate(panels):
            if i != j:
                A[i, j] = 0.5 / np.pi * integral(panel_i.xc, panel_i.yc, 
                                                    panel_j,
                                                    np.cos(panel_i.beta),
                                                    np.sin(panel_i.beta))
    return A

def vortex_normal(panels):
    """
    Builds the vortex contribution matrix for the normal velocity.
    
    Parameters
    ----------
    panels: 1D array of Panel objects
        List of panels.
    
    Returns
    -------
    A: 2D Numpy array of floats
        Vortex contribution matrix.
    """
    A = np.empty((panels.size, panels.size), dtype=float)
    # vortex contribution on a panel from itself
    np.fill_diagonal(A, 0.0)
    # vortex contribution on a panel from others
    for i, panel_i in enumerate(panels):
        for j, panel_j in enumerate(panels):
            if i != j:
                A[i, j] = -0.5 / np.pi * integral(panel_i.xc, panel_i.yc, 
                                                     panel_j,
                                                     np.sin(panel_i.beta),
                                                     -np.cos(panel_i.beta))
    return A



A_source = source_normal(panels)
B_vortex = vortex_normal(panels)


print(B_vortex.shape)
def kutta_condition(A_source, B_vortex):
    """
    Builds the Kutta condition array
    
    Inputs:
        A_souce: 2D numpy array
            Source contribution matrix for the normal velocity
        B_vortex: 2D numpy array
            Vortex contribution matrix for the normal velocity
    
    Returns:
        b: 1D numpy array 
            The left-hand side of the Kutta-condition equation
    """
    b = np.empty(A_source.shape[0] + 1, dtype = float)
    """
    Matrix of the source contribution on tangential velocity
    is the same as the matrix of the vortex contribution on
    normal velocity
    """
    b[:-1] = B_vortex[0, :] +  B_vortex[-1, :]
    """
    Matrix of vortex contribution on tangential velocity
    is the opposite of matrix of source contribution on
    normal velocity
    """
    b[-1] = - np.sum(A_source[0, :] + A_source[-1, :])
    
    return b

def singularity_matrix(A_source, B_vortex):
    """
    Builds the left-hand side matrix of the matrix
    
    Inputs:
        A_source: 2D numpy array of floats
            Source contribution matrix for the normal velocity
        B_vortex: 2D numpy array of floats
            Vortex contribution matrix for the normal velocity
    """
    
    #A = np.empty((A_source.shape[0] + 1, A_source.shape[0] + 1), dtype = float)
    A = np.empty((A_source.shape[0] + 1, A_source.shape[1] + 1), dtype = float)
    #Source contribution matrix
    A[:-1, :-1] = A_source
    #Vortex contribution matrix
    A[:-1, -1] = np.sum(B_vortex, axis = 1)
    #Kutta condition array
    A[-1, :] = kutta_condition(A_source, B_vortex)
    
    return A

def freestream_rhs(panels, freestream):
    """
    Builds a right-hand side of the system
    
    Inputs:
        panels: 1D numpy array of Panel objects
        freestream: freestream object
    Outputs:
        b: rhs matrix
    """
    b = np.empty(panels.size + 1, dtype = float)
    #Freestream contribution on each panel
    for i, panel in enumerate(panels):
        b[i] = -freestream.u_inf * np.cos(freestream.AoA - panel.beta)
    
    #Freestream contribution on the Kutta condition
    b[-1] = -freestream.u_inf * (np.sin(freestream.AoA - panels[0].beta) + 
                                 np.sin(freestream.AoA - panels[-1].beta))
    
    return b


A = singularity_matrix(A_source, B_vortex)
b = freestream_rhs(panels, freestream)

#Solve the linear system
strengths = np.linalg.solve(A, b)

#Store source strength on each panel
for i, panel in enumerate(panels):
    panel.sigma = strengths[i]
    
#Store vortex strength
gamma = strengths[-1]

def tangential_velocity(panels, freestream, gamma, A_source, B_vortex):
    """
    Computes the tangential surface velocity
    
    Inputs:
        panels: 1D numpy array of Panel objects
        freestream: Freestream object
        gamma: Circulation density
        A_source: 2D numpy array of source contribution matrix for 
            normal velocity
        B_vortex: 2D numpy array of vortex contribution matrix for 
            normal velocity
    """
    A = np.empty((panels.size, panels.size + 1), dtype = float)
    """
    Matrix of the source contribution on tangential velocity
    is the same as the matrix of the vortex contribution on
    normal velocity
    """
    A[:, :-1] = B_vortex
    """
    Matrix of vortex contribution on tangential velocity
    is the opposite of matrix of source contribution on
    normal velocity
    """
    A[:, -1] = - np.sum(A_source, axis = 1)
    """Freetstream condition"""
    b = freestream.u_inf * np.sin([freestream.AoA - panel.beta for panel in panels])
    
    strengths = np.append([panel.sigma for panel in panels], gamma)
    
    tangential_velocities = np.dot(A, strengths) + b
    
    for i, panel in enumerate(panels):
        panel.vt = tangential_velocities[i]

#Compute tangential velocities
tangential_velocity(panels, freestream, gamma, A_source, B_vortex)

for panel in panels:
    panel.cp = 1.0 - (panel.vt / freestream.u_inf)**2
        
#Plot the surface Cp
        
plt.figure(figsize = (10, 6))
plt.grid()
plt.xlabel('x', fontsize = 16)
plt.ylabel('$C_p$', fontsize = 16)
#plt.plot([panel.xc for panel in panels if panel.loc = 'upper'],
#         [panel.cp for panel in panels if panel.loc = 'upper'],
#         label = 'upper surface', color ='b', linestyle ='-',
#         marker = 'o', markersize = 6)
plt.plot([panel.xc for panel in panels if panel.loc == 'upper'],
            [panel.cp for panel in panels if panel.loc == 'upper'],
            label='upper surface',color='r', linestyle='-', linewidth=2, marker='o', markersize=6)
plt.plot([panel.xc for panel in panels if panel.loc == 'lower'],
            [panel.cp for panel in panels if panel.loc == 'lower'],
            label= 'lower surface',
            color='b', linestyle='-', linewidth=1, marker='o', markersize=6)
plt.legend(loc='best', prop={'size':16})
plt.xlim(-0.1, 1.1)
plt.ylim(1.5, -8.0)
plt.title('Number of panels: {}'.format(panels.size), fontsize=16);
    
# calculate the accuracy
accuracy = sum([panel.sigma * panel.length for panel in panels])
print('sum of singularity strengths: {:0.6f}'.format(accuracy))

print(b)
#for panel in panels:
#    print(panel.vt)


#Create a mesh grid
n = 50
x_start, x_end = -1.0, 2.0
y_start, y_end = -0.3, 0.3
x = np.linspace(x_start, x_end, n)
y = np.linspace(y_start, y_end, n)
X, Y = np.meshgrid(x, y)

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
        u += panel.sigma / (2 * np.pi) * vec_integral(X, Y, panel, 1.0, 0.0) 
        v += panel.sigma / (2 * np.pi) * vec_integral(X, Y, panel, 0.0, 1.0) 
    
    u += gamma / (2 * math.pi) * vec_integral(X, Y, panel, 1.0, 0.0) 
    v -= gamma / (2 * math.pi) * vec_integral(X, Y, panel, 0.0, 1.0) 
    
    return u, v
            
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
plt.fill([panel.xa for panel in panels], [panel.ya for panel in panels], color = 'k', linestyle = 'solid', linewidth = 2, zorder = 2)
plt.xlim(x_start, x_end)
plt.ylim(y_start, y_end)
plt.axis('scaled', adjustable = 'box')
plt.show()
           
            
            
            
            
            
            
            