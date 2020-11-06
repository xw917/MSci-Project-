#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 00:20:14 2020

@author: cordelia
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from collections import OrderedDict
import pickle

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
Plotting for <n> vs t for different pulse lengths 
Section 1: Saving large data, DO NOT RUN
Section 2: Plotting 
"""
# Section 1
allData_n_vs_t = OrderedDict()
N = 100 # the whole process repeats 10 times
cycle = 500
pulse_time = [10e-6, 20e-6, 30e-6, 40e-6, 50e-6, 60e-6, 70e-6, 80e-6, 90e-6, 
              100e-6]

for t in pulse_time:
    data = {'states':[]}
    S = np.zeros(cycle)
    for n in range(N):
        s = system(50, cycle, t) # n0 = 50, No. of pulses = 500
        S = [sum(i) for i in zip(S,s.excitation())]
    ave_S = [j/N for j in S] # average phonon number 
    data['states'] = ave_S # storing <n> for each pulse time 
    allData_n_vs_t[t] = data

pickle.dump(allData_n_vs_t, open('data_n_vs_t_tdiff', 'wb'))

#%%
# Section 2
ave_state = pickle.load(open('data_n_vs_t_tdiff', 'rb'))
for t, n in ave_state.items():
    plt.plot(np.arange(0, 500)*t*10**3, n['states'], label = 't = %s ms' %(t*10e3))

plt.legend(title = 'pulse duration')
plt.xlabel("time(ms)")
plt.ylabel("<n>")
plt.xlim(0, 25)

#%%
"""
Plotting for <n> vs t for different fork states n = 20 ~ 50
Section 1: Saving large data, DO NOT RUN
Section 2: Plotting
"""
# Section 1
allData_n_vs_t_ndiff = OrderedDict()
N2 = np.arange(20, 51, 1) # array of different fork states from n = 20 to 50
Num = 100 # the whole process repeats 10 times 

for n in N2: 
    data = {'states':[]}
    S = np.zeros(500)
    for num in range(Num):
        s = system(n0 = n, N = 500, t_pulse = 10e-6) 
        # number of pulses set to be 500, pulse time set to be 10e-6s
        S = [sum(i) for i in zip(S,s.excitation())]
    ave_S = [j/Num for j in S] # average phonon number     
    data['states'] = ave_S # storing <n> for each pulse time 
    allData_n_vs_t_ndiff[n] = data
    # print(n)

pickle.dump(allData_n_vs_t_ndiff, open('data_n_vs_t_ndiff', 'wb'))
  
#%%
# Section 2
ave_state2 = pickle.load(open('data_n_vs_t_ndiff', 'rb'))
# print(ave_state2)
for n_fork, s in ave_state2.items():
    # print(s['states'])
    plt.plot(np.arange(0, 500)*10e-6, s['states'])

plt.xlabel("time(s)")
plt.ylabel("<n>")
  
#%%
"""
Plotting for distribution of motional states 
"""
N = 50 # run 50 times
distribution = []
for n in range(N):
    s = system(50, 500, 10e-6) # n0 = 50, No. of pulses = 100, pulse duration = 1.5e-6s
    states = s.excitation()
    distribution += states.tolist()
# print(len(distribution))

b = np.arange(0, 50)
plt.hist(distribution, bins = b)
plt.xlabel('motional state n')
