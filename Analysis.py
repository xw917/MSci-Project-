#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 11:48:25 2020

@author: cordelia
"""

import numpy as np
import scipy as sp
import timeit
import matplotlib.pyplot as plt
import random
import collections

from Base1 import *
from Param_Const1 import *
from test_file import *

# pu = pulse()
# tr = trap(pu, N = 500, n0 = 85, M = 1500)
# a, b = tr.sideband_cool_sch(pu)
# ap = twod_traj(pu, tr, a)

#%%
            
def plot_ave(pulse, trap, ToP):
    """
    Plot of average phonon state versus time
    
    Parameters
    ----------
    pulse : class, defined by the class pulse
    trap  : class, defined by the class trap
    ToP   : str, type of plot:
                 input 'a' --> plots for different pulse lengths
                 input 'b' --> plots for comparison between Matrix and Monte 
                               Carlo methods
                 input 'c' --> plots for different initial Fock states
                 input 'd' --> plots for comparison among different conditions:
                               1. No decay and off-resonant excitations
                               2. With decay but no off-resonant excitations
                               3. With decay and carrier excitations 
                                    
    """
    time_array = np.linspace(0, pulse.t * trap.N, trap.N + 1)
    # fig, ax = plt.subplots()
    all_trial_n, all_trial_n_ave = trap.sideband_cool_sch(pulse, ave = True)
    if ToP == 'a':
        plt.plot(time_array * 1e3, all_trial_n_ave, label = pulse.t)
    if ToP == 'b':
        plt.plot(time_array * 1e3, all_trial_n_ave, color = 'black', label = 'Monte Carlo')
    if ToP == 'c':
        plt.plot(time_array * 1e3, all_trial_n_ave)
    if ToP == 'd':
        if trap.no_decay == True and trap.off_resonant_excite  == False:
            plt.plot(time_array * 1e3, all_trial_n_ave, label = 'No decay and carrier excite')
        if trap.no_decay == False and trap.off_resonant_excite  == False:
            plt.plot(time_array * 1e3, all_trial_n_ave, 
                     label = 'With decay to %s sideband, no carrier excite'%(trap.sideband))
        if trap.no_decay == False and trap.off_resonant_excite  == True:
            plt.plot(time_array * 1e3, all_trial_n_ave, 
                     label = 'With decay to %s sideband and carrier excite'%(trap.sideband))
    # plt.xlabel('time / ms')
    # plt.ylabel('Phonon State')
    # plt.legend()
    
def twod_traj(pulse, trap, data):
    time_array = np.linspace(0, pulse.t * trap.N, trap.N + 1)
    entries, seqlength = data.shape
    minn, maxn = np.min(data), np.max(data)
    n_plot, n_search = np.arange(minn, maxn + 2), np.arange(minn, maxn + 2) - 0.5
    
    datapack = np.apply_along_axis(np.histogram, 0, data, bins = n_search, range = (minn, maxn))
    twod_data = np.stack(datapack[0])
#    print(twod_data.shape)
    Time_array, N_plot = np.meshgrid(time_array, n_plot)
    
    fig, ax = plt.subplots()
#    meshplot = ax.pcolormesh(Time_array, N_plot, twod_data.T)
    meshplot = ax.pcolormesh(twod_data.T[:, :20])
    ax.set_xlabel('Time')
    ax.set_ylabel('Photon state')
    plt.colorbar(meshplot)
    return twod_data.T

def twod_traj2(pulse, trap, logplot = False):
    """
    Population distribution over time. 
    
    Parameters
    ----------
    pulse : class, the applied laser pulse.
    trap  : class, the system. 
    logplot : True --> plot logplot, False --> logplot off. The default is False.

    Returns
    -------
    A 2D contour plot. 

    """
    a, b = trap.sideband_cool_sch(pulse)
    # print(a)
    m = []
    for time in range(trap.N):
        state = []
        # prob = np.zeros(trap.n0 + trap.sideband*2 + 1)
        prob = np.zeros(int(np.max(a)) + 1)
        accu_data = list(collections.Counter(sorted(a.T[time])).items())
        # print(accu_data)
        for i in range(len(accu_data)):
            prob[int(accu_data[i][0])] = accu_data[i][1]/trap.M
        m.append(prob)
    Time_array, N_plot = np.meshgrid((np.linspace(0, pulse.t * trap.N, 
                                                  trap.N + 1))*1e3, np.arange(0, np.max(a) + 2) - 0.5)
    fig, ax = plt.subplots()
    
    if logplot == True:
        import matplotlib.colors
        meshplot = ax.pcolormesh(Time_array, N_plot, np.array(m).T + 1e-34,
                                 norm = matplotlib.colors.LogNorm(vmin=1e-10), shading = 'auto')
    else:
        meshplot = ax.pcolormesh(Time_array, N_plot, np.array(m).T)
        
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Phonon state')
    plt.colorbar(meshplot)

def plot_traj(pulse, trap):
    all_trial_n, all_trial_n_ave = trap.sideband_cool_sch(pulse, ave = True)
    [trial, no_step] = all_trial_n.shape
    time_array = sp.linspace(0, pulse.t * (no_step - 1), no_step)
    fig, ax = plt.subplots()
    for i in range(trial):
        ax.plot(time_array * 1e3, all_trial_n[i, :])
    ax.set_xlabel('time / ms')
    ax.set_ylabel('Phonon State')
    return ax

def end_hist(pulse, trap):
    """
    Distribution at the end of cooling
    """
    all_trial_n, all_trial_n_ave = trap.sideband_cool_sch(pulse)
    n_max = np.amax(all_trial_n)
    hist_xar = sp.arange(n_max + 1) - 0.5
    
    # fig, ax = plt.subplots()
    plt.hist(all_trial_n[:, -1], bins = hist_xar)
    plt.xlabel('Phonon State')
    plt.ylabel('Distribution')
    # return ax

def plot_matrix_method(pulse, trap, ToP):
    """
    Plot of average phonon state versus time
    
    Parameters
    ----------
    pulse : class, defined by the class pulse
    trap  : class, defined by the class trap
    ToP   : str, type of plot, 
                 input 'a' --> plots for different pulse lengths
                 input 'b' --> plots for comparison between Matrix and Monte 
                               Carlo methods
                 input 'c' --> plots for different initial Fock states
    """
    n0, d = trap.matrix_method(pulse)
    for k in range(len(d)):
        ave_list = []
        timestep = np.arange(0, trap.N+1, 1)
        for i in range(len(d[k])):
            sum2 = 0
            for j in range(len(d[k][i])):
                sum2 += (j) * d[k][i][j]
            ave_list.append(sum2)
        if ToP == 'a':
            plt.plot(timestep * pulse.t * 1e3, ave_list, label = pulse.t)
        if ToP == 'b':
            plt.plot(timestep * pulse.t * 1e3, ave_list, label = 'Matrix')
        if ToP == 'c':
            plt.plot(timestep * pulse.t * 1e3, ave_list)
    # plt.legend()
    # plt.xlabel('time (ms)')
    # plt.ylabel('n')
    #plt.xlim(0, 10)   
    
#%%
# functions for two ions

def plot_twoD_rabi_strength(n_com, n_b, band_com, band_b, wavelength):
    
    def twoD_rabi_strength(n1, n2):
        data = []
        for n in range(band_com, n1):
           sub_data = []
           for nn in range(band_b, n2):
               Om_red1 = eff_rabi_freq2(n, n-band_com, nn, nn-band_b, freq_to_wav(wavelength, wz * (-1)), wz)
               sub_data.append(Om_red1)
           data.append(sub_data)
        return data
    
    Z = twoD_rabi_strength(n1 = n_com, n2 = n_b)
    cs = plt.contourf(Z, levels = 20)
    cs.changed()
    plt.xlabel('$n_{B}$')
    plt.ylabel('$n_{COM}$')
    plt.colorbar()
    