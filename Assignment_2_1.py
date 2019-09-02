#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 11:41:30 2018

@author: tanthanhnhanphan
"""

import numpy as np
from matplotlib import pyplot as plt
from math import *


N = 31
x_start, x_end = -4.0, 4.0
y_start, y_end = -1.5, 2.5
x = np.linspace(x_start, x_end, N)
y = np.linspace(y_start, y_end, N)
X, Y = np.meshgrid(x,y)

k = 11
gamma = 0.5
x_vortex = np.linspace(x_start, x_end, k)
y_vortex = [1.00] * k

u_vortex = np.zeros((k, N, N), float)
v_vortex = np.zeros((k, N, N), float)
psi_vortex = np.zeros((k, N, N), float)
u_vortex_total = np.zeros((N, N), float)
v_vortex_total = np.zeros((N, N), float)


def velocity_infinite_vortices(strength, a, X, Y,b):
    u_v = strength / (2*a) * np.sinh(2*pi*(Y-b)/a) / (np.cosh(2*pi*(Y-b)/a) - np.cos(2*pi*(X-a)/a))
    v_v =-strength / (2*a) * np.sin(2*pi*(X-a)/a) / (np.cosh(2*pi*(Y-b)/a) - np.cos(2*pi*(X-a)/a))
    return u_v, v_v

for k in range(len(x_vortex)):
    u_vortex[i,:,:], v_vortex[i,:,:] = velocity_infinite_vortices(gamma, x_vortex[i], X, Y, y_vortex[i])
    u_vortex_total[:,:] += u_vortex[i,:,:]
    v_vortex_total[:,:] += v_vortex[i,:,:]

width = 10.0
height = (y_end - y_start) / (x_end - x_start) * width
plt.figure(figsize = (width, height))
plt.grid(True)
plt.xlabel('x', fontsize = 16)
plt.ylabel('y', fontsize = 16)
plt.xlim(x_start, x_end)
plt.ylim(y_start, y_end)
plt.streamplot(X, Y, u_vortex_total, v_vortex_total, density = 2, linewidth = 1, arrowsize = 1, arrowstyle = '->')
for i in range(len(x_vortex)):
    plt.scatter(x_vortex[i], y_vortex[i], color = '#CD2305', s = 80, marker= 'o')
#plt.scatter(x_stag, y_stag, color = 'g', s = 40, marker = 'o')
#plt.contour(X, Y, psi_combine, levels = [-sigma/2, sigma/2], colors = '#CD2305', linewidths = 2, linestyles = 'solid')
plt.show()