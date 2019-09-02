#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 09:46:57 2018

@author: tanthanhnhanphan
"""

"""Assignment: Source distribution on Airfoil"""

import numpy as np
from math import *
from matplotlib import pyplot as plt

naca0012_x = np.loadtxt(fname = "NACA0012_x.txt")
naca0012_y = np.loadtxt(fname = "NACA0012_y.txt")
naca0012_sigma = np.loadtxt(fname = "NACA0012_sigma.txt")

#X,Y = np.meshgrid(naca0012_x, naca0012_y)


N = 51
x_start, x_end = -1.0, 2.0
y_start, y_end = -0.5, 0.5
x = np.linspace(x_start, x_end, N)
y = np.linspace(y_start, y_end, N)
X, Y = np.meshgrid(x, y)

"""
width = 10.0
height = (y_end - y_start) / (x_end - x_start) * width
plt.figure(figsize=(width, height))
plt.xlabel('x', fontsize=16)
plt.ylabel('y', fontsize=16)
plt.xlim(x_start, x_end)
plt.ylim(y_start, y_end)
plt.scatter(X, Y, s=5, color='#CD2305', marker='o')
"""
            
"""Freestream"""
u_inf = 100

u_freestream = u_inf * np.ones((N, N), float)
v_freestream = np.zeros((N, N), float)
    
"""Freestream function"""
psi_freestream = u_inf * Y    

"""NACA0012"""
def source_sink(strength, xs, ys, X, Y):
    u_s = strength / (2*pi) * (X - xs) / ((X - xs)**2 + (Y - ys)**2)
    v_s = strength / (2*pi) * (Y - ys) / ((X - xs)**2 + (Y - ys)**2)
    
    return u_s, v_s
def stream_function_source_sink (strength, xs, ys, X, Y):
    psi = strength / (2*pi) *np.arctan2((Y - ys), (X- xs))
    return psi


gamma = 1.0
def vortex_function(strength, xv, yv, X, Y):
    u = strength / (2*pi) * (Y - yv) / ((X - xv)**2 + (Y - yv)**2)
    v =-strength / (2*pi) * (X - xv) / ((X - xv)**2 + (Y - yv)**2)
    return u, v



k = len(naca0012_x)
#print(k)
if k != len(naca0012_y):
    raise RuntimeError('Check dimensions')
if k != len(naca0012_sigma):
    raise RuntimeError('Check dimensions')





u_source, v_source, psi_source = np.zeros((k,N,N), float), np.zeros((k,N,N), float), np.zeros((k,N,N), float)
u_combine, v_combine, psi_combine = np.zeros((k,N,N), float), np.zeros((k,N,N), float), np.zeros((k,N,N), float)
u_total, v_total, psi_total = np.zeros((N,N), float), np.zeros((N,N), float), np.zeros((N,N), float)
#print(X.shape)
print(u_source.shape)
print(X[1])
for j in range(k):
    for i in range(N):
        u_source[j,:,:], v_source[j,:,:] = source_sink(naca0012_sigma[j], naca0012_x[j], naca0012_y[j], X, Y)
        #print(u_source.shape)
        psi_source[j,:,:] = stream_function_source_sink(naca0012_sigma[j], naca0012_x[j], naca0012_y[j], X, Y)
        u_combine[j,:,:] = u_source[j,:,:] + u_freestream[i,i]
        v_combine[j,:,:] = v_source[j,:,:] + v_freestream[i,i]
        psi_combine[j,:,:] = psi_source[j,:,:] + psi_freestream[i,i]
for j in range(k):
    u_total[:,:] = u_combine[j,:,:]
    v_total[:,:] = v_combine[j,:,:]
    psi_total[:,:] += psi_combine[j,:,:]
print(u_total.shape)
#u_source, v_source = source_sink(naca0012_sigma, naca0012_x, naca0012_y, X,Y)   
#psi_source = stream_function_source_sink(naca0012_sigma, naca0012_x, naca0012_y, X,Y) 

#u_combine = u_freestream + u_source
#v_combine = v_freestream + v_source
#psi_combine = psi_freestream + psi_source
"""
width = 10.0
height = (y_end - y_start) / (x_end - x_start) * width
plt.figure(figsize = (width, height))
plt.grid(True)
plt.xlabel('x', fontsize = 16)
plt.ylabel('y', fontsize = 16)
plt.xlim(x_start, x_end)
plt.ylim(y_start, y_end)
plt.streamplot(X, Y, u_source[:,:], v_source[:,:], density = 2, linewidth = 0.5, arrowsize = 1, arrowstyle = '->')
#plt.scatter(x_source, y_source, color = '#CD2305', s = 80, marker= 'o')
#plt.scatter(x_stag, y_stag, color = 'g', s = 40, marker = 'o')
#plt.contour(X, Y, psi_combine, levels = [-sigma/2, sigma/2], colors = '#CD2305', linewidths = 2, linestyles = 'solid')
plt.show()
"""
print(k)
u_vortex, v_vortex = np.zeros((k,N,N), float), np.zeros((k,N,N), float)
u_combine, v_combine = np.zeros((N, N), float),np.zeros((N, N), float)
for i in range(k):
    u_vortex[i,:,:], v_vortex[i,:,:] = vortex_function(gamma, naca0012_x[i], naca0012_y[i], X, Y)
    u_combine[:,:] += u_vortex[i,:,:]
    v_combine[:,:] += v_vortex[i,:,:]

u_combine += u_freestream
v_combine += v_freestream    


width = 10.0
height = (y_end - y_start) / (x_end - x_start) * width
plt.figure(figsize = (width, height))
plt.grid(True)
plt.xlabel('x', fontsize = 16)
plt.ylabel('y', fontsize = 16)
plt.xlim(x_start, x_end)
plt.ylim(y_start, y_end)
plt.streamplot(X, Y, u_combine, v_combine, density = 2, linewidth = 1, arrowsize = 1, arrowstyle ='->')
for i in range(len(naca0012_x)):
    plt.scatter(naca0012_x[i], naca0012_y[i], color = '#CD2305', s = 10, marker ='o')
plt.show()


