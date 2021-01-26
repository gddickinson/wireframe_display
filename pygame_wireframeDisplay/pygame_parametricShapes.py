# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 13:14:54 2020

@author: g_dic
"""

from math import *
import numpy as np

#parametric surfaces    
def sphere(u, v):
    x = sin(u) * cos(v)
    y = cos(u)
    z = -sin(u) * sin(v)
    return x, y, z

def funnel(u,v):
    x = u
    y = u * sin(v)
    z = u * cos(v)       
    return x,y,z

def noseCone(u,v):
    x = u*u
    y = u * sin(v)
    z = u * cos(v)       
    return x,y,z

def helix_3D(u,v):
    x = (3+cos(u)) * cos(v)
    y = (3+cos(u)) * sin(v)
    z = sin(u) + 0.5*v
    return x,y,z
    
def klein(u, v):
    u = u * 2
    if u < pi:
        x = 3 * cos(u) * (1 + sin(u)) + (2 * (1 - cos(u) / 2)) * cos(u) * cos(v)
        z = -8 * sin(u) - 2 * (1 - cos(u) / 2) * sin(u) * cos(v)
    else:
        x = 3 * cos(u) * (1 + sin(u)) + (2 * (1 - cos(u) / 2)) * cos(v + pi)
        z = -8 * sin(u)
    y = -2 * (1 - cos(u) / 2) * sin(v)
    return x, y, z


def mobius_tube(u, v):
    sign = np.sign
    R = 1.5
    n = 3
    u = u * 2
    x = (
        1.0 * R
        + 0.125 * sin(u / 2) * pow(abs(sin(v)), 2 / n) * sign(sin(v))
        + 0.5 * cos(u / 2) * pow(abs(cos(v)), 2 / n) * sign(cos(v))
    ) * cos(u)
    y = (
        1.0 * R
        + 0.125 * sin(u / 2) * pow(abs(sin(v)), 2 / n) * sign(sin(v))
        + 0.5 * cos(u / 2) * pow(abs(cos(v)), 2 / n) * sign(cos(v))
    ) * sin(u)
    z = -0.5 * sin(u / 2) * pow(abs(cos(v)), 2 / n) * sign(cos(v)) + 0.125 * cos(
        u / 2
    ) * pow(abs(sin(v)), 2 / n) * sign(sin(v))
    return x, y, z

def torus(u,v):
    c, a = 2, 1
    x = (c + a*np.cos(u)) * np.cos(v)
    y = (c + a*np.cos(u)) * np.sin(v)
    z = a * np.sin(u)
    return x,y,z

def helix(u,v):
    b = 3
    x = u * cos(v)
    y= u * sin(v)
    z = b * v
    return x,y,z

def sineSurface(u,v):
    a =5
    x = a * sin(u)
    y = a * sin(v)
    z = a * sin(u+v)
    return x,y,z

def eightSurface(u,v):
    x = cos(u) * sin(2*v)
    y = sin(u) * sin(2*v)
    z = sin(v)
    return x,y,z
    
def ellipsoid(u,v):
    a = 5
    b = 5
    c = 20
    x = a* cos(u) * sin(v)
    y = b * sin(u) * sin(v)
    z = c * cos(v)
    return x,y,z

def hyperOctahedron(u,v):
    x = (cos(u)*cos(v))**3
    y = (sin(u)*cos(v))**3
    z = sin(v)**3
    return x,y,z

def geodesic(u,v):
    x = np.sqrt(u) * cos(v)
    y = np.sqrt(u) * sin(v)
    z = u
    return x,y,z

def cornucopia(u,v):
    a = 1
    b = 1
    x = exp(b*v) * cos(v+exp(a*v)) * cos(u) * cos(v)
    y = exp(b*v) * sin(v+exp(a*v)) * cos(u) * sin(v)
    z = exp(a*v) * sin(u)
    
    return x,y,z
