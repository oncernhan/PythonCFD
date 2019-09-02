#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 17:17:52 2018

@author: tanthanhnhanphan
"""

"""Lesson 4: Vortex"""

import numpy as np
from matplotlib import pyplot as plt
from math import *


N = 50
x_start, x_end = -2.0, 2.0
y_start, y_end = -1.0, 1.0
x = np.linspace(x_start, x_end, N)
y = np.linspace(y_start, y_end, N)
X,Y = np.meshgrid(x,y)

gamma = 5.0
x_vortex, y_vortex = 0.0, 0.0

def vortex_function(strength, xv, yv, X, Y):
    u = strength / (2*pi) * (Y - yv) / ((X - xv)**2 + (Y - yv)**2)
    v =-strength / (2*pi) * (X - xv) / ((X - xv)**2 + (Y - yv)**2)
    psi = strength / (4*pi) * np.log((X - xv)**2 + (Y - yv)**2)
    return u, v, psi

u_vortex, v_vortex, psi_vortex = vortex_function(gamma, x_vortex, y_vortex, X, Y)

width = 10.0
length = (y_end - y_start) / (x_end - x_start) * width
plt.figure(figsize = (width, length))
plt.xlabel('x', fontsize = 16)
plt.ylabel('y', fontsize = 16)
plt.xlim(x_start, x_end)
plt.ylim(y_start, y_end)
plt.streamplot(X, Y, u_vortex, v_vortex, density = 2, linewidth = 1, arrowsize = 1, arrowstyle = '->')
plt.scatter(x_vortex, y_vortex, color = '#CD2305', s = 80, marker = 'o')
plt.show()


"""Vortex and Sink"""
from Source_Sink_Freestream import freestream, source_sink, stream_function_source_sink

strength_sink = -1.0
x_sink, y_sink = 0.0, 0.0
u_freestream, v_freestream, psi_freestream = freestream(N, Y)
u_sink, v_sink = source_sink(strength_sink, x_sink, y_sink, X, Y)
psi_sink = stream_function_source_sink(strength_sink, x_sink, y_sink, X, Y)

u_combine = u_sink + u_vortex + u_freestream
v_combine = v_sink + v_vortex + v_freestream


width = 10.0
length = (y_end - y_start) / (x_end - x_start) * width
plt.figure(figsize = (width, length))
plt.xlabel('x', fontsize = 16)
plt.ylabel('y', fontsize = 16)
plt.xlim(x_start, x_end)
plt.ylim(y_start, y_end)
plt.streamplot(X, Y, u_combine, v_combine, density = 2, linewidth = 1, arrowsize = 1, arrowstyle ='->')
plt.scatter(x_vortex, y_vortex, color='#CD2305', s = 80, marker = 'o')
plt.show()