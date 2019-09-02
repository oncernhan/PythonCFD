#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 02:56:58 2018

@author: tanthanhnhanphan
"""

"""test"""
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import integrate
import os


mainfoil = np.genfromtxt('MainFoil_N=50.csv' , delimiter=',')
flapfoil = np.genfromtxt('FlapFoil_N=50.csv', delimiter=',')
x_main = mainfoil[:,0]
y_main = mainfoil[:,1]
x_flap = flapfoil[:,0]
y_flap = flapfoil[:,1]
width = 10
plt.figure(figsize= (width, width))
plt.xlabel('x', fontsize = 16)
plt.ylabel('y', fontsize = 16)
plt.plot(x_main,y_main,color ='k')
plt.plot(x_flap,y_flap,color ='r')
plt.axis('scaled', adjustable = 'box')
plt.xlim(-0.3, 1.4)
plt.ylim(-0.6, 0.5)
plt.show() 

x_total = np.append(x_main, x_flap)
y_total = np.append(y_main, y_flap)

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

def define_panels(x, y, N):
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
    for i in range(N) :
        while I <= len(x) - 1:
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
Na = 40
Nb = 42
panels_main = define_panels(x_main, y_main, Na)
panels_flap = define_panels(x_flap, y_flap, Nb)
panels = np.append(panels_main, panels_flap)
#print(panels.shape)
#panels = define_panels(x_total, y_total, N = 40)
#Plot discretized geometry
width = 10
plt.figure(figsize= (width, width))
plt.grid()
plt.xlabel('x', fontsize = 16)
plt.ylabel('y', fontsize = 16)
plt.plot(x_main, y_main, color = 'k', linestyle = '-', linewidth = 2)
plt.plot(x_flap, y_flap, color = 'k', linestyle = '-', linewidth = 2)
plt.plot(np.append([panel.xa for panel in panels_main], panels_main[0].xa), np.append([panel.ya for panel in panels_main], panels_main[0].ya), linestyle = '-', linewidth = 1, markersize = 6, marker = 'o', color = '#CD2305')
plt.plot(np.append([panel.xa for panel in panels_flap], panels_flap[0].xa), np.append([panel.ya for panel in panels_flap], panels_flap[0].ya), linestyle = '-', linewidth = 1, markersize = 6, marker = 'o', color = '#CD2305')
plt.axis('scaled', adjustable = 'box')
plt.xlim(-0.3, 1.4)
plt.ylim(-0.6, 0.5)
plt.show()


class Freestream:
    """
    Class contains freestream conditions
    """
    def __init__(self, u_inf, AoA):
        self.u_inf = u_inf
        self.AoA = np.radians(AoA)

#Define the freestream
u_inf = 60
AoA = -10.0
freestream = Freestream(u_inf, AoA)

def integral(x, y, panel, dxdk, dydk):
   
    def integrand(s):
        return ((x - (panel.xa - np.sin(panel.beta) * s)) * dxdk + \
                (y - (panel.ya + np.cos(panel.beta) * s)) * dydk) / \
                ((x - (panel.xa - np.sin(panel.beta) * s)) ** 2 + \
                (y - (panel.ya + np.cos(panel.beta) * s)) **2 )
    return integrate.quad(integrand, 0.0, panel.length)[0]


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

A_source_main = source_normal(panels_main)
#print("A_soA_source_main.shape)
B_vortex_main = vortex_normal(panels_main)

A_source_flap = source_normal(panels_flap)
B_vortex_flap = vortex_normal(panels_flap)

A_source = source_normal(panels)
B_vortex = vortex_normal(panels)

print("B vortex main", B_vortex_main.shape)
print("B vortex flap", B_vortex_flap.shape)

def kutta_condition(A_source, B_vortex, A_source_main, B_vortex_main, A_source_flap, B_source_flap):
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
    b = np.zeros(A_source.shape[0] + 2, dtype = float)
    #print(b)
    print("b shape",b.shape)
    """
    Matrix of the source contribution on tangential velocity
    is the same as the matrix of the vortex contribution on
    normal velocity
    """
    #b[:-2] = B_vortex[0, :] +  B_vortex[-1, :]
    #print(b[:-2])
    Na_1 = Na + 1
    #print("Na_1", b[:Na])
    b[:Na] = B_vortex_main[0, :] + B_vortex_main[-1, :]
    print("After", b[:Na])
    N = Na + Nb
    print("Hey", b[Na+1:N+1])
    """
    Matrix of vortex contribution on tangential velocity
    is the opposite of matrix of source contribution on
    normal velocity
    """
    b[-2] = - np.sum(A_source_main[0, :] + A_source_main[-1, :])
    print(b[-2])
    b[-1]  = - np.sum(A_source_flap[0, :] + A_source_flap[-1, :])
    print("b -1" , b[-1])
    return b

def kutta_condition_main(A_source, B_vortex, A_source_main, B_vortex_main):
    
    b = np.zeros(A_source.shape[0] + 2, dtype = float)
    b[:Na] = B_vortex_main[0, :]  + B_vortex_main[-1, :]
    b[-2] = - np.sum(A_source_main[0, :] + A_source_main[-1, :])
    return b

def kutta_condition_flap(A_source, B_vortec, A_source_flap, B_source_flap):
    
    b = np.zeros(A_source.shape[0] + 2, dtype = float)
    N = Na + Nb
    b[Na+1:N+1] = B_vortex_flap[0, :] + B_vortex_flap[-1, :]
    b[-1]  = - np.sum(A_source_flap[0, :] + A_source_flap[-1, :])
    
    return b

    




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
    A = np.empty((A_source.shape[0]+ 2, A_source.shape[1] + 2), dtype = float)
    print("A shape:", A.shape)
    print("A source shape", A_source.shape)
    print("B vortex shape", B_vortex.shape)
    print(A[:-2, -2].shape)
    #Source contribution matrix
    A[:-2, :-2] = A_source
    #Vortex contribution matrix
    A[:-2, -2] = np.sum(B_vortex_main)
    A[:-2, -1] = np.sum(B_vortex_flap)
    #Kutta condition array
    A[-2, :] = kutta_condition_main(A_source, B_vortex, A_source_main, B_vortex_main)
    A[-1, :] = kutta_condition_flap(A_source, B_vortex, A_source_flap, B_vortex_flap)
    
    return A

#A = singularity_matrix(A_source, B_vortex)
#b = freestream_rhs(panels, freestream)

b_main = freestream_rhs(panels_main, freestream)
b_flap = freestream_rhs(panels_flap, freestream)
b = np.append(b_main, b_flap)
A = singularity_matrix(A_source, B_vortex)

print(np.linalg.det(A))
#print(np.linalg.det(b))
#print(A[-2, :])
#print(A[-1, :])
#print(A[:-2,-2])
#print(A[:-2,-1])

#Solve the linear system
strengths = np.linalg.solve(A, b)

for i, panel in enumerate(panels):
    panel.sigma = strengths[i]
    
#Store vortex strength
gamma_a = strengths[-2]
print(gamma_a)
gamma_b = strengths[-1]
print(gamma_b)

accuracy = sum([panel.sigma * panel.length for panel in panels_flap])
print('sum of singularity strengths: {:0.6f}'.format(accuracy))

def tangential_velocity(panels, freestream, gamma_a, gamma_b, A_source_main, A_source_flap, B_vortex_main, B_vortex_flap):
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
    A = np.zeros((panels.size, panels.size + 2), dtype = float)
    """
    Matrix of the source contribution on tangential velocity
    is the same as the matrix of the vortex contribution on
    normal velocity
    """
    #A[:, :Na] = B_vortex_main
    #A[:, :-2] = B_vortex_flap
    N = Na + Nb
    #print(A[:Na, :Na].shape)
    print(A[:-2, :-2].shape)
    A[:, :-2] = B_vortex
    #print(A[:, :-2])
    """
    Matrix of vortex contribution on tangential velocity
    is the opposite of matrix of source contribution on
    normal velocity
    """
    A[:, -2] = - np.sum(A_source_main)
    A[:, -1] = - np.sum(A_source_flap)
    """Freetstream condition"""
    b = freestream.u_inf * np.sin([freestream.AoA - panel.beta for panel in panels])
    gamma = [gamma_a, gamma_b]
    
    strengths = np.append([panel.sigma for panel in panels], gamma_a)
    strengths = np.append(strengths, gamma_b)
    tangential_velocities = np.dot(A, strengths) + b
    
    for i, panel in enumerate(panels):
        panel.vt = tangential_velocities[i]
        
#Compute tangential velocities
tangential_velocity(panels, freestream, gamma_a, gamma_b, A_source_main, A_source_flap, B_vortex_main, B_vortex_flap)

for panel in panels:
    panel.cp = 1.0 - (panel.vt / freestream.u_inf)**2
    print(panel.vt)


#Plot the surface Cp      
plt.figure(figsize = (10, 6))
plt.grid()
plt.xlabel('x', fontsize = 16)
plt.ylabel('$C_p$', fontsize = 16)
#plt.plot([panel.xc for panel in panels if panel.loc = 'upper'],
#         [panel.cp for panel in panels if panel.loc = 'upper'],
#         label = 'upper surface', color ='b', linestyle ='-',
#         marker = 'o', markersize = 6)
plt.plot([panel.xc for panel in panels_main if panel.loc == 'upper'],
            [panel.cp for panel in panels_main if panel.loc == 'upper'],
            label='upper surface',color='r', linestyle='-', linewidth=2, marker='o', markersize=6)
plt.plot([panel.xc for panel in panels_main if panel.loc == 'lower'],
            [panel.cp for panel in panels_main if panel.loc == 'lower'],
            label= 'lower surface',
            color='b', linestyle='-', linewidth=1, marker='o', markersize=6)
plt.plot([panel.xc for panel in panels_flap if panel.loc == 'upper'],
            [panel.cp for panel in panels_flap if panel.loc == 'upper'],
            label='upper surface',color='r', linestyle='-', linewidth=2, marker='o', markersize=6)
plt.plot([panel.xc for panel in panels_flap if panel.loc == 'lower'],
            [panel.cp for panel in panels_flap if panel.loc == 'lower'],
            label= 'lower surface',
            color='b', linestyle='-', linewidth=1, marker='o', markersize=6)

plt.legend(loc='best', prop={'size':16})
#plt.axis('scaled', adjustable = 'box')
#plt.xlim(-1, 1)

#plt.ylim(200, -100)
plt.title('Number of panels: {}'.format(panels.size), fontsize=16)
plt.show()

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
    
    u += (gamma_a + gamma_b) / (2 * math.pi) * vec_integral(X, Y, panel, 1.0, 0.0) 
    v -= (gamma_a + gamma_b) / (2 * math.pi) * vec_integral(X, Y, panel, 0.0, 1.0) 
    
    return u, v
            
#compute the velocity field
u, v = velocity_field(panels, freestream, X, Y)

#plot the streamlines
width = 10
plt.figure(figsize = (width, width))
plt.xlabel('x', fontsize = 16)
plt.ylabel('y', fontsize = 16)
plt.title('Streamline around Willam 1973 airfoil with ${}^o$ AoA'.format(AoA), fontsize = 16)
plt.streamplot(X, Y, u, v, density = 2, linewidth = 1, arrowsize = 1, arrowstyle = '->')
plt.fill([panel.xc for panel in panels], [panel.yc for panel in panels], color ='black', linestyle = 'solid', linewidth = 2, zorder = 2)
plt.xlim(0.9, 1.1)
plt.ylim(-0.03, 0.03)
plt.axis('scaled', adjustable = 'box')
plt.show()            
            
"""Compute the pressure field around an airfoil"""
cp = 1 - (u**2 + v**2) / (u_inf**2)

width = 10
plt.figure(figsize = (width, 6))
plt.xlabel('x', fontsize = 16)
plt.ylabel('y', fontsize = 16)
contf = plt.contourf(X, Y, cp, levels = np.linspace(-0.5, 0.5, 100), extend = 'both')
cbar = plt.colorbar(contf, orientation='horizontal', shrink=0.5, pad = 0.1, ticks=[-1.0, -0.5, 0.0, 0.5, 1.0])
cbar.set_label('$C_p$', fontsize = 16)
plt.fill([panel.xa for panel in panels], [panel.ya for panel in panels], color = 'k', linestyle = 'solid', linewidth = 2, zorder = 2)
plt.title('Pressure coefficient for William 1973 airfoil', fontsize = 16)
plt.axis('scaled', adjustable = 'box')
plt.xlim(x_start, x_end)
plt.ylim(y_start, y_end)
plt.show()


mainfoil_theo = np.genfromtxt('Cp_Main_theoretical.csv' , delimiter=',')
flapfoil_theo = np.genfromtxt('Cp_Flap_theoretical.csv', delimiter=',')
x_main = mainfoil_theo[:,0]
y_main = mainfoil_theo[:,1]
x_flap = flapfoil_theo[:,0]
y_flap = flapfoil_theo[:,1]
width = 10
plt.figure(figsize= (width, 6))
plt.xlabel('x', fontsize = 16)
plt.ylabel('y', fontsize = 16)
plt.plot(x_main,y_main,color ='k', marker = 'o', markersize = 6)
plt.plot(x_flap,y_flap,color ='r', marker = 'o', markersize = 6)
plt.grid()
plt.show() 












