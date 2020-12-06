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
import scipy.stats as sts
import scipy.constants as cs
from scipy.special import eval_genlaguerre

q = 1 # no. of unit of positive charge on the ion
m = cs.m_p * 20 + cs.m_n * 20  # mass of the Ca+ ion

Bz = 1  # axial electric field (unit: T)

rb = 40e3 * 2 * sp.pi
Q = q * cs.e  # charge on the ion
#wz = sp.sqrt(2 * Q * V0 / (m * d0**2)) # z-axis ang.freq
wz = 187e3 * 2 * sp.pi  # trap frequency
#wc = Q * Bz / m  # cyclotron freq
wc = 715e3 * 2 * sp.pi
wp = wc + sp.sqrt(wc**2 - 2 * wz**2)  # modified cyclotron freq
wm = wc - sp.sqrt(wc**2 - 2 * wz**2)  # magnetron freq

L = 729e-9  # sideband cooling transition wavelength
LD = 397e-9 # decay sideband transition wavelength

def FactorialDiv(x, y):
    '''
    calculates x! / y!
    '''
    fd = 1
    if x < y:
        for i in range(x + 1, y + 1):
            fd = fd * i
        fd = 1 / fd
    else:
        for i in range(y + 1, x + 1):
            fd = fd * i
    return fd

def sumColumn(m, column):
    total = 0
    for row in range(len(m)):
        total += m[row][column]
    return total