#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 01:46:52 2020
@author: apple
"""
import scipy as sp
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.stats as sts
import scipy.constants as cs
from scipy.special import eval_genlaguerre,  factorial
from numpy import dot, outer, identity, array, zeros, count_nonzero, einsum, argmax, eye, linspace
from numpy import flip, transpose, exp, ones, amax, amin, arange, meshgrid, stack, prod, average
from numpy import floor, diag, sort
from scipy import sqrt, rand, mod, log, exp
from numpy.linalg import eigh
from scipy.sparse import diags
from scipy.optimize import curve_fit, fsolve
from math import log10
import pickle


q = 1 # no. of unit of positive charge on the ion
e0 = cs.epsilon_0 # permittivity of free space
m = cs.m_p * 20 + cs.m_n * 20  # mass of the Ca+ ion

Bz = 1  # axial electric field (unit: T)

rb = 40e3 * 2 * sp.pi
Q = q * cs.e  # charge on the ion
#wz = sp.sqrt(2 * Q * V0 / (m * d0**2)) # z-axis ang.freq
wz = 500e3 * 2 * sp.pi  # trap frequency
#wc = Q * Bz / m  # cyclotron freq
wc = 730e3 * 2 * sp.pi
wr = wc / 2 # effective radial trapping frequency 
we = sp.sqrt(wc**2 / 4 - wz**2 / 2)
wp = wc + sp.sqrt(wc**2 - 2 * wz**2)  # modified cyclotron freq
wm = wc - sp.sqrt(wc**2 - 2 * wz**2)  # magnetron freq

L = 729e-9  # sideband cooling transition wavelength
LD = 393e-9 # decay sideband transition wavelength

mwz24pe_z2e2 = m * wz**2 * 4 * sp.pi * e0 / Q**2  # factor to be used for calculating ion equilibrium positions
mwr24pe_z2e2 = (Q**2 / m * wr**2 * 4 * sp.pi * e0)**(1/3)
def FactorialDiv(x, y):
    '''
    calculates x! / y! (x <= y) using stirling's approximation
    '''
    if isinstance(x, np.ndarray):
        if amin([x, y]) <= 50:
            fd = factorial(x) / factorial(y)
        else:
            fd = exp((x*(log(x)-1)+log(sqrt(2*3.14159263*x))) -
                     (y*(log(y)-1)+log(sqrt(2*3.14159263*y))))
    else:
        fd = 1
        for i in range(x + 1, y + 1):
            fd = fd * i
        fd = 1 / fd     
    return fd

def sumColumn(m, column):
    total = 0
    for row in range(len(m)):
        total += m[row][column]
    return total
