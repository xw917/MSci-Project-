#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 00:20:14 2020

@author: cordelia
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

from System import *
from Base import *
from Param_Const import *

#%%
def PlotRedSb(n0):
    '''
    Plot the relative strengths of the carrier, first and second sidebands at 
    and below quantum number n0
    '''
    fig, ax = plt.subplots()
    Om_carrier, Om_red, Om_blue = [], [], []
    for i in range(2, n0):
        Om_carrier.append(EffRabiFreq(i, i, L, wz, 1))
        if i > 0:
            Om_red.append(EffRabiFreq(i, i - 1, Freq2Wav(L, - wz), wz, 1))
        if i > 1:
            Om_blue.append(EffRabiFreq(i, i - 2, Freq2Wav(L, - 2 * wz), wz, 1))
    ax.plot(Om_carrier, label = 'carrier')
    ax.plot(Om_red, label = 'red1')
    ax.plot(Om_blue, label = 'red2')
    suum = sp.array(Om_carrier)**2 + sp.array(Om_red)**2 + sp.array(Om_blue)**2
    ax.plot(suum, label = 'sum')
    ax.legend()
    ax.set_xlabel('Motional State Number (n)')
    ax.set_ylabel('Normalised Rabi Frequency')
    return ax

PlotRedSb(120)

#%%
"""
Plotting for <n> vs t
"""
N = 50 # run 50 times 
# pulse_time = [10e-6, 25e-6, 50e-6, 100e-6]
# cycle = 500

# for t in pulse_time:
    # S = np.zeros(cycle)
    # for n in range(N):
        # s = system(50, cycle, t) # n0 = 50, No. of pulses = 500
        # S = [sum(i) for i in zip(S,s.excitation())]
    # ave_S = [j/N for j in S] # average phonon number 
    # plt.plot(np.arange(0, cycle)*t, ave_S, label = 't = %s ms' %(t*10e3))
# plt.legend(title = 'pulse duration')
# plt.xlabel("time(ms)")
# plt.ylabel("<n>")
# plt.xlim(0, 0.02)

#%%
"""
Plotting for distribution of motional states 
"""
# N = 50 # run 50 times
# distribution = []
# for n in range(N):
    # s = system(50, 100, 1.5e-6) # n0 = 50, No. of pulses = 100, pulse duration = 1.5e-6s
    # distribution += s.excitation().tolist()
# print(len(distribution))

# b = np.arange(0, 50)
# plt.hist(distribution, bins = b)
