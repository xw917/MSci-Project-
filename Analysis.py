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
import pickle 
import matplotlib.colors
from mpl_toolkits.mplot3d import Axes3D

from Base1 import *
from Param_Const1 import *
from test_file import *
from Schrodinger import *
from N_Ions_System import *

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
        plt.plot(time_array * 1e3, all_trial_n_ave, color = 'magenta', linewidth = 3, label = 'Monte Carlo')
    if ToP == 'c':
        plt.plot(time_array * 1e3, all_trial_n_ave, color = 'b')
    if ToP == 'd':
        if trap.no_decay == True and trap.off_resonant_excite  == False:
            plt.plot(time_array * 1e3, all_trial_n_ave, label = 'Decay to carrier')
        if trap.no_decay == False and trap.off_resonant_excite  == False:
            plt.plot(time_array * 1e3, all_trial_n_ave, 
                     label = 'Decay to %s sideband'%(trap.sideband))
        if trap.no_decay == False and trap.off_resonant_excite  == True:
            plt.plot(time_array * 1e3, all_trial_n_ave, 
                     label = 'Decay to %s sideband, off-resonant excite'%(trap.sideband))
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
    
    if logplot == True:
        import matplotlib.colors
        meshplot = plt.pcolormesh(Time_array, N_plot, np.array(m).T + 1e-34,
                                 norm = matplotlib.colors.LogNorm(vmin=1e-10), shading = 'auto')
    else:
        meshplot = plt.pcolormesh(Time_array, N_plot, np.array(m).T)
        
    plt.xlabel('Time (ms)')
    plt.ylabel('Phonon state')
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
            plt.plot(timestep * pulse.t * 1e3, ave_list, color = 'black', label = 'Matrix')
        if ToP == 'c':
            plt.plot(timestep * pulse.t * 1e3, ave_list, color = 'b')
    # plt.legend()
    # plt.xlabel('time (ms)')
    # plt.ylabel('n')
    #plt.xlim(0, 10)   
    
#%%
# functions for two ions

def plot_twoD_rabi_strength(n_com, n_b, band_com, band_b, wavelength):
    sideband = band_com + band_b
    def twoD_rabi_strength(n1, n2):
        data = []
        for n in range((-1)*band_com, n1):
           sub_data = []
           for nn in range((-1)*band_b, n2):
               Om_red1 = eff_rabi_freq2(n+band_com, n, nn+band_b, nn, freq_to_wav(wavelength, wz * (-1)), wz)
               sub_data.append(Om_red1)
           data.append(sub_data)
        return data
    
    Z = twoD_rabi_strength(n1 = n_com, n2 = n_b)
    cs = plt.contourf(Z, levels = 20)
    cs.changed()
    plt.xlabel('$n_{B}$')
    plt.ylabel('$n_{COM}$')
    plt.colorbar()



def twoD_Contour(sidebands, pulse_time, states):
        
    def excitation_probability(time, state):
        data = []
        both_states = np.array([state, state])
        R = LoadAnything('ions' + str(2) + '_laser_rabi.data')
        for t in time:
            no_excited, h_non_zero, which_ion_change = laser_sb_ham_info(2)
            which_ion_chg_arr = ones((both_states.shape[1], *which_ion_change.shape),
                             dtype = 'int') * (which_ion_change - 1)
            # calculate total sideband strengths (terms in the hamiltonian matrix)
            total_sb_stren = 1
            for motion_state, sideband, mode in zip(both_states, sidebands, range(2)):
                changed_motion_stt = no_excited[:,None] * sideband + motion_state
                sb_stren_this_mode = R[which_ion_chg_arr.transpose(1,2,0),mode,changed_motion_stt[:,None],
                                    changed_motion_stt[:,None].transpose(1,0,2)]
                total_sb_stren = total_sb_stren * sb_stren_this_mode
            H = einsum('ijk,ij->kij', total_sb_stren, h_non_zero)
    
            psi_0 = eye(1, M = H.shape[-1])[0]
            # solve the schrodinger's equation
            psi_t = SchrodingerEqn(H, psi_0, array([t]), h_factor = rb/2)[:,:,0]
            # calculate excite probabilty
            probs  = (psi_t * psi_t.conj()).real
            # print(probs)
            sub_data = []
            for p in probs:
                sub_data.append(sum(p[1:]))
            data.append(sub_data)
        return data

    Z = excitation_probability(pulse_time, states)
    
    cs = plt.contourf(Z, levels = 20)
    cs.changed()
    plt.xlabel('States (n)')
    plt.ylabel(r'Pulse time ($1\times 10^{-5}$s) ')
    plt.colorbar()

def plot_cooling(trap, lasers):
    cursor = 0
    end_cursor_list = []
    all_com, all_b, all_com_full, all_b_full = [], [], [], []
    percentage_com, percentage_b = [], []
    
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx() 
    
    for i in range(len(lasers)):
        all_data, ave_data = trap.sideband_cool_sch(lasers[i])

        com, b = [i[0] for i in ave_data], [i[1] for i in ave_data]
        com_all, b_all = [i[0] for i in all_data], [i[1] for i in all_data]
        
        all_com.append(com)
        all_b.append(b)
        all_com_full.append(com_all)
        all_b_full.append(b_all)
        
        pct_com = [list(each_pulse).count(0)/trap.M for each_pulse in com_all]
        percentage_com.append(pct_com)
        pct_b = [list(each_pulse).count(0)/trap.M for each_pulse in b_all]
        percentage_b.append(pct_b)

        # plot average phonon state
        ax1.axvline(x = cursor, ls='--', color = 'grey')
        if i == 0:
            ax1.plot(np.linspace(0, (lasers[i].N)*lasers[i].t*1e3, lasers[i].N+1), com, '--', color = 'blue', label = 'com')
            ax1.plot(np.linspace(0, (lasers[i].N)*lasers[i].t*1e3, lasers[i].N+1), b, color = 'blue', label = 'breathing')
        else:
            ax1.plot(np.linspace(cursor, cursor + (lasers[i].N)*lasers[i].t*1e3, lasers[i].N+1), com, '--', color = 'blue')
            ax1.plot(np.linspace(cursor, cursor + (lasers[i].N)*lasers[i].t*1e3, lasers[i].N+1), b, color = 'blue')
        # print(cursor, cursor*lasers[i-1].t*1e3, (cursor+lasers[i].N)*lasers[i].t*1e3)

        cursor += (lasers[i].N)*lasers[i].t*1e3
        end_cursor_list.append(cursor)
        
    # plot the percentage of states in ground state
    end_point_percentage_com = [perc[-1] for perc in percentage_com]
    end_point_percentage_b = [perc[-1] for perc in percentage_b]
    
    ax2.plot(end_cursor_list, end_point_percentage_com, '^', ms = 12, color = 'r', label = 'com')
    ax2.plot(end_cursor_list, end_point_percentage_b, 'x', ms = 12, color = 'r', label = 'breathing')
        
    ax1.set_xlabel(r'time(ms)')
    ax1.set_ylabel(r'$\langle n \rangle$', color = 'blue')
    ax1.tick_params(axis='y', labelcolor = 'blue')
    ax2.set_ylabel('fraction', color = 'red')
    ax2.tick_params(axis='y', labelcolor = 'red')
    
    ax1.legend(bbox_to_anchor=(1.12,1), loc="upper left") 
    ax2.legend(bbox_to_anchor=(1.12,0.5), loc="upper left")       
    plt.show()
    return all_com, all_b, all_com_full, all_b_full, percentage_com, percentage_b

def twod_traj_N_ions(pulse, trap, all_data):
    m = []
    for time in range(pulse.N):
        state = []
        # prob = np.zeros(trap.n0 + trap.sideband*2 + 1)
        prob = np.zeros(int(np.max(all_data)) + 1)
        accu_data = list(collections.Counter(all_data[time]).items())
        # print(accu_data)
        for i in range(len(accu_data)):
            prob[int(accu_data[i][0])] = accu_data[i][1]/trap.M
        m.append(prob)
    Time_array, N_plot = np.meshgrid((np.linspace(0, pulse.t * pulse.N, 
                                                  pulse.N + 1))*1e3, np.arange(0, np.max(all_data) + 2) - 0.5)
    # fig, ax = plt.subplots()
    
    import matplotlib.colors
    meshplot = plt.pcolormesh(Time_array, N_plot, np.array(m).T + 1e-34,
                             norm = matplotlib.colors.LogNorm(vmin=1e-10), shading = 'auto')
    
    plt.xlabel('Time (ms)')
    plt.ylabel('Phonon state')
    plt.colorbar(meshplot)

"""
#%%
'''
 Find the quilibrium positions of ions. Small probability that the returned array is NOT
 the equilibrium positions. Recommend to run this part several times.
'''
N = 2  # number of ions
equi_pos, last_trial_pos = equil_pos_icc(N)
#%%
'''
 Find the normal modes associated with the determined crystal configuration
'''
norm_freq, norm_coor = icc_normal_modes(equi_pos)
'''
 Plot the ion crystal and normal modes
'''
figure = 0
while int(6 * (figure + 1) - 3 * N) < 6:
    plot_nm(equi_pos, norm_coor, norm_freq, kmode = int(figure * 6))
    figure = figure + 1
"""
#%%
def plot_cooling_3_planar(trap, lasers):
    cursor = 0
    end_cursor_list = []
    all_com, all_b, all_d, all_com_full, all_b_full, all_d_full = [], [], [], [], [], []
    percentage_com, percentage_b, percentage_d = [], [], []
    
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx() 
    
    for i in range(len(lasers)):
        all_data, ave_data = trap.sideband_cool_sch(lasers[i])

        com, b, d = [i[2] for i in ave_data], [i[0] for i in ave_data], [i[1] for i in ave_data]
        com_all, b_all, d_all = [i[2] for i in all_data], [i[0] for i in all_data], [i[1] for i in all_data]
        
        all_com.append(com)
        all_b.append(b)
        all_d.append(d)
        
        all_com_full.append(com_all)
        all_b_full.append(b_all)
        all_d_full.append(d_all)
        
        pct_com = [list(each_pulse).count(0)/trap.M for each_pulse in com_all]
        percentage_com.append(pct_com)
        pct_b = [list(each_pulse).count(0)/trap.M for each_pulse in b_all]
        percentage_b.append(pct_b)
        pct_d = [list(each_pulse).count(0)/trap.M for each_pulse in d_all]
        percentage_d.append(pct_d)

        # plot average phonon state
        ax1.axvline(x = cursor, ls='--', color = 'grey')
        if i == 0:
            ax1.plot(np.linspace(0, (lasers[i].N)*lasers[i].t*1e3, lasers[i].N+1), com, '--', color = 'blue', label = 'com')
            ax1.plot(np.linspace(0, (lasers[i].N)*lasers[i].t*1e3, lasers[i].N+1), b, color = 'blue', label = 'tilt 1')
            ax1.plot(np.linspace(0, (lasers[i].N)*lasers[i].t*1e3, lasers[i].N+1), d, ':', color = 'blue', label = 'tilt 2')
        else:
            ax1.plot(np.linspace(cursor, cursor + (lasers[i].N)*lasers[i].t*1e3, lasers[i].N+1), com, '--', color = 'blue')
            ax1.plot(np.linspace(cursor, cursor + (lasers[i].N)*lasers[i].t*1e3, lasers[i].N+1), b, color = 'blue')
            ax1.plot(np.linspace(cursor, cursor + (lasers[i].N)*lasers[i].t*1e3, lasers[i].N+1), d, ':', color = 'blue')
        # print(cursor, cursor*lasers[i-1].t*1e3, (cursor+lasers[i].N)*lasers[i].t*1e3)

        cursor += (lasers[i].N)*lasers[i].t*1e3
        end_cursor_list.append(cursor)
        
    # plot the percentage of states in ground state
    end_point_percentage_com = [perc[-1] for perc in percentage_com]
    end_point_percentage_b = [perc[-1] for perc in percentage_b]
    end_point_percentage_d = [perc[-1] for perc in percentage_d]
    
    ax2.plot(end_cursor_list, end_point_percentage_com, '^', ms = 12, color = 'r', label = 'com')
    ax2.plot(end_cursor_list, end_point_percentage_b, 'x', ms = 12, color = 'r', label = 'tilt 1')
    ax2.plot(end_cursor_list, end_point_percentage_d, 'o', ms = 12, color = 'r', label = 'tilt 2')
        
    ax1.set_xlabel(r'time(ms)')
    ax1.set_ylabel(r'$\langle n \rangle$', color = 'blue')
    ax1.tick_params(axis='y', labelcolor = 'blue')
    ax2.set_ylabel('fraction', color = 'red')
    ax2.tick_params(axis='y', labelcolor = 'red')
    
    ax1.legend(bbox_to_anchor=(1.12,1), loc="upper left") 
    ax2.legend(bbox_to_anchor=(1.12,0.5), loc="upper left")       
    plt.show()
    return all_com, all_b, all_d, all_com_full, all_b_full, all_d_full, percentage_com, percentage_b, percentage_d

def plot_cooling_4_planar(trap, lasers):
    cursor = 0
    end_cursor_list = []
    all_com, all_b, all_d, all_e, all_com_full, all_b_full, all_d_full, all_e_full = [], [], [], [], [], [], [], []
    percentage_com, percentage_b, percentage_d, percentage_e = [], [], [], []
    
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx() 
    
    for i in range(len(lasers)):
        all_data, ave_data = trap.sideband_cool_sch(lasers[i])

        com, b, d, e = [i[3] for i in ave_data], [i[0] for i in ave_data], [i[1] for i in ave_data], [i[2] for i in ave_data]
        com_all, b_all, d_all, e_all = [i[3] for i in all_data], [i[0] for i in all_data], [i[1] for i in all_data], [i[2] for i in all_data]
        
        all_com.append(com)
        all_b.append(b)
        all_d.append(d)
        all_e.append(e)
        
        all_com_full.append(com_all)
        all_b_full.append(b_all)
        all_d_full.append(d_all)
        all_e_full.append(e_all)
        
        pct_com = [list(each_pulse).count(0)/trap.M for each_pulse in com_all]
        percentage_com.append(pct_com)
        pct_b = [list(each_pulse).count(0)/trap.M for each_pulse in b_all]
        percentage_b.append(pct_b)
        pct_d = [list(each_pulse).count(0)/trap.M for each_pulse in d_all]
        percentage_d.append(pct_d)
        pct_e = [list(each_pulse).count(0)/trap.M for each_pulse in e_all]
        percentage_e.append(pct_e)

        # plot average phonon state
        ax1.axvline(x = cursor, ls='--', color = 'grey')
        if i == 0:
            ax1.plot(np.linspace(0, (lasers[i].N)*lasers[i].t*1e3, lasers[i].N+1), com, color = 'cornflowerblue', label = 'com')
            ax1.plot(np.linspace(0, (lasers[i].N)*lasers[i].t*1e3, lasers[i].N+1), b, color = 'blue', label = 'tilt')
            ax1.plot(np.linspace(0, (lasers[i].N)*lasers[i].t*1e3, lasers[i].N+1), d, ':', color = 'blue', label = 'tilt 2')
            ax1.plot(np.linspace(0, (lasers[i].N)*lasers[i].t*1e3, lasers[i].N+1), e, '--', color = 'blue', label = 'tilt 3')
        else:
            ax1.plot(np.linspace(cursor, cursor + (lasers[i].N)*lasers[i].t*1e3, lasers[i].N+1), com, color = 'cornflowerblue')
            ax1.plot(np.linspace(cursor, cursor + (lasers[i].N)*lasers[i].t*1e3, lasers[i].N+1), b, color = 'blue')
            ax1.plot(np.linspace(cursor, cursor + (lasers[i].N)*lasers[i].t*1e3, lasers[i].N+1), d, ':', color = 'blue')
            ax1.plot(np.linspace(cursor, cursor + (lasers[i].N)*lasers[i].t*1e3, lasers[i].N+1), e, '--', color = 'blue')
        # print(cursor, cursor*lasers[i-1].t*1e3, (cursor+lasers[i].N)*lasers[i].t*1e3)

        cursor += (lasers[i].N)*lasers[i].t*1e3
        end_cursor_list.append(cursor)
        
    # plot the percentage of states in ground state
    end_point_percentage_com = [perc[-1] for perc in percentage_com]
    end_point_percentage_b = [perc[-1] for perc in percentage_b]
    end_point_percentage_d = [perc[-1] for perc in percentage_d]
    end_point_percentage_e = [perc[-1] for perc in percentage_e]
    
    ax2.plot(end_cursor_list, end_point_percentage_com, '^', ms = 12, color = 'r', label = 'com')
    ax2.plot(end_cursor_list, end_point_percentage_b, 'x', ms = 12, color = 'r', label = 'tilt')
    ax2.plot(end_cursor_list, end_point_percentage_d, 'o', ms = 12, color = 'r', label = 'tilt 2')
    ax2.plot(end_cursor_list, end_point_percentage_e, '+', ms = 12, color = 'r', label = 'tilt 3')
        
    ax1.set_xlabel(r'time(ms)')
    ax1.set_ylabel(r'$\langle n \rangle$', color = 'blue')
    ax1.tick_params(axis='y', labelcolor = 'blue')
    ax2.set_ylabel('fraction', color = 'red')
    ax2.tick_params(axis='y', labelcolor = 'red')
    
    ax1.legend(bbox_to_anchor=(1.12,1), loc="upper left") 
    ax2.legend(bbox_to_anchor=(1.12,0.5), loc="upper left")       
    plt.show()
    return all_com, all_b, all_d, all_e, all_com_full, all_b_full, all_d_full, all_e_full, percentage_com, percentage_b, percentage_d, percentage_e
#%%
def sideband_stren_3_planar(sideband = np.array([0, 0, 0])):
    R = LoadAnything('3_ions_planar.data')
    #R = LoadAnything('3_ions_planar_change_basis.data')
    #print(R[0].shape)
    data_point_1, data_point_2, data_point_3 = np.zeros((8000000, 4)), np.zeros((8000000, 4)), np.zeros((8000000, 4))
    for i in range(200):
        for j in range(200):
            for k in range(200):
                data_point_1[40000*i + 200*j + k] = [i, j, k, R[0][0][i][i+sideband[0]] * R[0][1][j][j+sideband[1]] * R[0][2][k][k+sideband[2]]]
                data_point_2[40000*i + 200*j + k] = [i, j, k, R[1][0][i][i+sideband[0]] * R[1][1][j][j+sideband[1]] * R[1][2][k][k+sideband[2]]]
                data_point_3[40000*i + 200*j + k] = [i, j, k, R[2][0][i][i+sideband[0]] * R[2][1][j][j+sideband[1]] * R[2][2][k][k+sideband[2]]]

    x1, y1, z1, c1 = np.array(data_point_1).T
    x2, y2, z2, c2 = np.array(data_point_2).T
    x3, y3, z3, c3 = np.array(data_point_3).T
    
    C = (c1 + c2 + c3)/3
    
    fig = plt.figure(figsize=plt.figaspect(0.3))
    
    ax = fig.add_subplot(projection='3d')
    img = ax.scatter(x1, y1, z1, c=C, cmap=plt.hot())
    fig.colorbar(img, orientation = 'horizontal', shrink=0.6)
    ax.set_xlabel('Breathing')
    ax.set_ylabel('Breathing 2')
    ax.set_zlabel('COM')
    
    ax.set_xlim3d(0, 100)
    ax.set_ylim3d(0, 100)
    ax.set_zlim3d(0, 100)
    """
    ax = fig.add_subplot(1, 3, 2, projection='3d')    
    img = ax.scatter(x2, y2, z2, c=c2, cmap=plt.hot())
    fig.colorbar(img, orientation = 'horizontal', shrink=0.6)
    ax.set_xlabel('Breathing')
    ax.set_ylabel('Breathing 2')
    ax.set_zlabel('COM')    
    
    ax = fig.add_subplot(1, 3, 3, projection='3d')    
    img = ax.scatter(x3, y3, z3, c=c3, cmap=plt.hot())
    fig.colorbar(img, orientation = 'horizontal', shrink=0.6)
    ax.set_xlabel('Breathing')
    ax.set_ylabel('Breathing 2')
    ax.set_zlabel('COM')
    plt.show()
    """

#%%

def twoD_Contour_3(sideband = np.array([0, 0, -1]), target_sideband = np.array([0, -1, 0]), ion = 1):
    # pulse_time = np.arange(0, 20, 1)*1e-5
    # state = np.arange(270)
    modes_w, K, m =  icc_normal_modes_z(3)
    title_detuning = sum(target_sideband * modes_w - sideband * modes_w)

    frequencies = [round(i, -1) for i in modes_w * we] # round-off frequency 
    detuning = sum(target_sideband * frequencies - sideband * frequencies)
    def excite_prob_ion1(state, pulse_time):
        R = LoadAnything('3_ions_planar.data')
        current_mode = np.array([state for _ in range(3)])
        exci = np.array([[0, 1, 2] for _ in range(len(state))]).T
        total_sb_stren = rb
        for motion_state, mode, sb in zip(current_mode, range(3), sideband):
            sb_stren_this_mode = R[exci, mode, motion_state, motion_state + sb]
            total_sb_stren *= sb_stren_this_mode
        modes_w, K, m =  icc_normal_modes_z(3)
        detuning = sum(target_sideband * frequencies - sideband * frequencies)
        excite_probability = np.array([[amplitude(total_sb_stren[ion][state], detuning) * 
                              (np.sin(frequency(total_sb_stren[ion][state], detuning) * pulse_time))**2 
                              for state in range(len(total_sb_stren[ion]))] for ion in range(len(total_sb_stren))])
        if ion == 3:
            return (excite_probability[0] + excite_probability[1] + excite_probability[2])/3
        else:
            return excite_probability[ion]
    
    Z = excite_prob_ion1(np.arange(270), np.arange(0, 200, 1)*1e-6)

    cs = plt.contourf(Z, levels = 20)
    cs.changed()
    plt.ylabel('States (n)')
    plt.xlabel(r'Pulse time ($\mu$s) ')
    plt.title(r'Detuning = %.2f $\omega_{eff}$'%(abs(title_detuning)))
    plt.colorbar()

def twoD_Contour_4(sideband = np.array([0, 0, 0, -1]), target_sideband = np.array([0, 0, 0, -1]), ion = 4):
    # pulse_time = np.arange(0, 20, 1)*1e-5
    # state = np.arange(270)
    modes_w, K, m =  icc_normal_modes_z(4)
    frequencies = [round(i, -1) for i in modes_w * we] # round-off frequency 
    detuning = sum(target_sideband * frequencies - sideband * frequencies)
    def excite_prob_ion1(state, pulse_time):
        R = LoadAnything('4_ions_planar.data')
        current_mode = np.array([state for _ in range(4)])
        exci = np.array([[0, 1, 2, 3] for _ in range(len(state))]).T
        total_sb_stren = rb
        for motion_state, mode, sb in zip(current_mode, range(4), sideband):
            sb_stren_this_mode = R[exci, mode, motion_state, motion_state + sb]
            total_sb_stren *= sb_stren_this_mode
        modes_w, K, m =  icc_normal_modes_z(4)
        detuning = sum(target_sideband * frequencies - sideband * frequencies)
        excite_probability = np.array([[amplitude(total_sb_stren[ion][state], detuning) * 
                              (np.sin(frequency(total_sb_stren[ion][state], detuning) * pulse_time))**2 
                              for state in range(len(total_sb_stren[ion]))] for ion in range(len(total_sb_stren))])
        if ion == 4:
            return (excite_probability[0] + excite_probability[1] + excite_probability[2] + excite_probability[3])/4
        else:
            return excite_probability[ion]
    
    Z = excite_prob_ion1(np.arange(360), np.arange(0, 200, 1)*1e-6)
    
    cs = plt.contourf(Z, levels = 20)
    cs.changed()
    plt.ylabel('States (n)')
    plt.xlabel(r'Pulse time ($\mu$s) ')
    plt.title('Sideband = %s, Detuning = %s'%(sideband, detuning))
    plt.colorbar()

def twoD_Contour_4_diff_n(sideband = np.array([0, 0, 0, -1]), target_sideband = np.array([0, 0, 0, -1]), ion = 4, 
                          mode_vector = [20, 20, 20, 0]):
    # pulse_time = np.arange(0, 20, 1)*1e-5
    # state = np.arange(270)
    modes_w, K, m =  icc_normal_modes_z(4)
    frequencies = [round(i, -1) for i in modes_w * we] # round-off frequency 
    detuning = sum(target_sideband * frequencies - sideband * frequencies)
    def excite_prob_ion1(state, pulse_time):
        R = LoadAnything('4_ions_planar.data')
        current_mode = []
        for item in mode_vector:
            if item != 0:
                current_mode.append(np.ones(len(state))*item)
            else:
                current_mode.append(state)
        current_mode = np.array(current_mode)
        exci = np.array([[0, 1, 2, 3] for _ in range(len(state))]).T
        total_sb_stren = rb
        for motion_state, mode, sb in zip(current_mode.astype(int), range(4), sideband):
            sb_stren_this_mode = R[exci, mode, motion_state, motion_state + sb]
            total_sb_stren *= sb_stren_this_mode
        modes_w, K, m =  icc_normal_modes_z(4)
        detuning = sum(target_sideband * frequencies - sideband * frequencies)
        excite_probability = np.array([[amplitude(total_sb_stren[ion][state], detuning) * 
                              (np.sin(frequency(total_sb_stren[ion][state], detuning) * pulse_time))**2 
                              for state in range(len(total_sb_stren[ion]))] for ion in range(len(total_sb_stren))])
        if ion == 4:
            return (excite_probability[0] + excite_probability[1] + excite_probability[2] + excite_probability[3])/4
        else:
            return excite_probability[ion]
    
    Z = excite_prob_ion1(np.arange(360), np.arange(0, 200, 1)*1e-6)
    
    cs = plt.contourf(Z, levels = 20)
    cs.changed()
    plt.ylabel('States (n)')
    plt.xlabel(r'Pulse time ($\mu$s) ')
    plt.title('Sideband = %s, Detuning = %s'%(sideband, detuning))
    plt.colorbar()
#%%
def prob_amp(N = np.array([43, 42, 42, 41]), target_sideband = np.array([0, 0, 0, -1]), sb = 1):
    sidebands = [] # store all possible excitation sidebands
    if len(N) == 4:
        for i in range(-sb, sb+1, 1):
            for j in range(-sb, sb+1, 1):
                for k in range(-sb, sb+1, 1):
                    for l in range(-sb, sb+1, 1):
                        sidebands.append([i, j, k, l])
    if len(N) == 3:
        for i in range(-sb, sb+1, 1):
            for j in range(-sb, sb+1, 1):
                for k in range(-sb, sb+1, 1):
                    sidebands.append([i, j, k])
    modes_w, K, m =  icc_normal_modes_z(len(N))                   
    frequencies = [round(i, -1) for i in modes_w * we] # round-off frequency 
    detuning = [sum(target_sideband * frequencies - np.array(sideband) * frequencies) for sideband in sidebands]
    exci = np.arange(len(N))
    R = LoadAnything(str(len(N)) + '_ions_planar.data')
    A = []
    for i in range(len(sidebands)):
        total_sb_stren = rb
        for motion_state, mode, sb in zip(N, range(len(N)), sidebands[i]):
            sb_stren_this_mode = R[exci, mode, motion_state, motion_state + sb]
            total_sb_stren *= sb_stren_this_mode
        A.append([amplitude(total_sb_stren[j], detuning[i]) for j in range(len(total_sb_stren))])
    amp_ion1 = np.array([sum(a) for a in A])/len(N)
    
    sigf_sb_red, sigf_sb_blue = [], []
    sigf_detuning_red, sigf_detuning_blue = [], []
    sigf_amp_red, sigf_amp_blue = [], []
    for i in range(len(detuning)):
        if amp_ion1[i] > 0.01:
            if 1 in sidebands[i]:
                sigf_sb_blue.append(sidebands[i])
                sigf_detuning_blue.append(detuning[i])
                sigf_amp_blue.append(amp_ion1[i])
            else:
                sigf_sb_red.append(sidebands[i])
                sigf_detuning_red.append(detuning[i])
                sigf_amp_red.append(amp_ion1[i])
      
    plt.vlines(x = np.array(sigf_detuning_red)*1e-3, ymin = np.zeros(len(sigf_detuning_red)), 
               ymax = np.array(sigf_amp_red), color = 'r')   
    plt.vlines(x = np.array(sigf_detuning_blue)*1e-3, ymin = np.zeros(len(sigf_detuning_blue)), 
               ymax = np.array(sigf_amp_blue), color = 'b')   
    plt.ylim(0, 1)
    plt.xlim(-200, 200)
    plt.xlabel('Detuning (kHz)')
    plt.ylabel('Excitation probability amplitude')
    for k in range(len(sigf_sb_red)):
        if sigf_detuning_red[k] == sigf_detuning_red[k-1]:
             plt.text(sigf_detuning_red[k]*1e-3, sigf_amp_red[k]+0.05, '%s'%(sigf_sb_red[k]), rotation=60, va='center', fontsize=10)
        else:
            plt.text(sigf_detuning_red[k]*1e-3, sigf_amp_red[k]+0.15, '%s'%(sigf_sb_red[k]), rotation=60, va='center', fontsize=10)
    for k in range(len(sigf_sb_blue)):
        if sigf_detuning_blue[k] == sigf_detuning_blue[k-1]:
             plt.text(sigf_detuning_blue[k]*1e-3, sigf_amp_blue[k]+0.03, '%s'%(sigf_sb_blue[k]), rotation=60, va='center', fontsize=10)
        else:
            plt.text(sigf_detuning_blue[k]*1e-3, sigf_amp_blue[k]+0.13, '%s'%(sigf_sb_blue[k]), rotation=60, va='center', fontsize=10)
    return sigf_sb_blue, sigf_amp_blue, sigf_sb_red, sigf_amp_red
    # return
#%%
def sideband_strength(N, current_mode, sb, xmin, xmax, ion_n):
    modes_w, K, m =  icc_normal_modes_z(N)
    sidebands_red, sidebands_blue = [], [] # store all possible excitation sidebands
    if N == 3:
        for i in range(-sb, sb+1, 1):
            for j in range(-sb, sb+1, 1):
                for k in range(-sb, sb+1, 1):
                    if i < 0 or j < 0 or k < 0:
                        sidebands_red.append([i, j, k])
                    else:
                        sidebands_blue.append([i, j, k])
    if N == 4:
        for i in range(-sb, sb+1, 1):
            for j in range(-sb, sb+1, 1):
                for k in range(-sb, sb+1, 1):
                    for l in range(-sb, sb+1, 1):
                        if i < 0 or j < 0 or k < 0 or l < 0:
                            sidebands_red.append([i, j, k, l])
                        else:
                            sidebands_blue.append([i, j, k, l])
                
    selected_sideband_red = find_sideband(current_mode, sidebands_red)[0]
    selected_sideband_blue = find_sideband(current_mode, sidebands_blue)[0]
    
    R = LoadAnything(str(N) + '_ions_planar.data')
    
    sideband_stren_red_all_ion, detuning_red_all_ion = [], []
    sideband_stren_blue_all_ion, detuning_blue_all_ion = [], []
    for ion in range(N):
        sideband_stren_red, detuning_red = [], []
        sideband_stren_blue, detuning_blue = [], []
        for i in range(len(selected_sideband_blue)):
            detuning_blue.append(sum(modes_w * selected_sideband_blue[i])*we)
            total_sb_stren_blue = 1
            for mode in range(N):
                total_sb_stren_blue *= R[ion, mode, current_mode[mode], current_mode[mode] + selected_sideband_blue[i][mode]]
            sideband_stren_blue.append(total_sb_stren_blue[0])
        for j in range(len(selected_sideband_red)):  
            detuning_red.append(sum(modes_w * selected_sideband_red[j])*we)
            total_sb_stren_red = 1
            for mode in range(N):
                total_sb_stren_red *= R[ion, mode, current_mode[mode], current_mode[mode] + selected_sideband_red[j][mode]]
            sideband_stren_red.append(total_sb_stren_red[0])  
        sideband_stren_red_all_ion.append(sideband_stren_red)
        sideband_stren_blue_all_ion.append(sideband_stren_blue)
        detuning_red_all_ion.append(detuning_red)
        detuning_blue_all_ion.append(detuning_blue)
    
    sideband_in_range, detune_in_range, stren_in_range = [], [], []
    for detune in detuning_red_all_ion[ion_n]:
        if detune > xmin*1e3 and detune < xmax*1e3:
            sideband_in_range.append(sidebands_red[detuning_red_all_ion[ion_n].index(detune)])
            stren_in_range.append(sideband_stren_red[detuning_red_all_ion[ion_n].index(detune)])
            detune_in_range.append(detune)
    for detune in detuning_blue_all_ion[ion_n]:
        if detune > xmin*1e3 and detune < xmax*1e3:
            sideband_in_range.append(sidebands_blue[detuning_blue_all_ion[ion_n].index(detune)])
            stren_in_range.append(sideband_stren_blue[detuning_blue_all_ion[ion_n].index(detune)])
            detune_in_range.append(detune)
        
    if N == 3:
        fig = plt.figure(figsize=(8, 8/1.6))
        ax1 = fig.add_axes([0, 0.6, 1, 0.4])
        ax2 = fig.add_axes([0, 0, 1, 0.4])
        
        ax1.vlines(x = np.array(detuning_red_all_ion[0])*1e-3, ymin = np.zeros(len(sideband_stren_red_all_ion[0])), ymax = np.array(sideband_stren_red_all_ion[0]), color = 'r')
        ax1.vlines(x = np.array(detuning_blue_all_ion[0])*1e-3, ymin = np.zeros(len(sideband_stren_blue_all_ion[0])), ymax = np.array(sideband_stren_blue_all_ion[0]), color = 'b')
        ax1.vlines(x = detuning_blue_all_ion[0][sidebands_blue.index([0, 0, 0])], ymin = 0, ymax = sideband_stren_blue_all_ion[0][sidebands_blue.index([0, 0, 0])], color = 'black')
        ax1.set_title('ion 1')
        ax1.set_ylabel('Rabi Strengths')
        ax1.set_ylim(0, 1)
        
        ax2.vlines(x = np.array(detuning_red_all_ion[1])*1e-3, ymin = np.zeros(len(sideband_stren_red_all_ion[1])), ymax = np.array(sideband_stren_red_all_ion[1]), color = 'r')
        ax2.vlines(x = np.array(detuning_blue_all_ion[1])*1e-3, ymin = np.zeros(len(sideband_stren_blue_all_ion[1])), ymax = np.array(sideband_stren_blue_all_ion[1]), color = 'b')
        ax2.vlines(x = detuning_blue_all_ion[1][sidebands_blue.index([0, 0, 0])], ymin = 0, ymax = sideband_stren_blue_all_ion[1][sidebands_blue.index([0, 0, 0])], color = 'black')
        ax2.set_title('ion 2 and ion 3')
        ax2.set_ylabel('Rabi Strengths')
        ax2.set_xlabel('Detuning (kHz)')
        ax2.set_ylim(0, 1)
        
    if N == 4:
        plt.vlines(x = np.array(detuning_red_all_ion[0])*1e-3, ymin = np.zeros(len(sideband_stren_red_all_ion[0])), ymax = np.array(sideband_stren_red_all_ion[0]), color = 'r')
        plt.vlines(x = np.array(detuning_blue_all_ion[0])*1e-3, ymin = np.zeros(len(sideband_stren_blue_all_ion[0])), ymax = np.array(sideband_stren_blue_all_ion[0]), color = 'b')
        plt.vlines(x = detuning_blue_all_ion[0][sidebands_blue.index([0, 0, 0, 0])], ymin = 0, ymax = sideband_stren_blue_all_ion[0][sidebands_blue.index([0, 0, 0, 0])], color = 'black')
        plt.ylim(0, 0.5)
        plt.xlabel('Detuning (kHz)')
        plt.ylabel('Rabi Strengths')
        
    return sideband_in_range, detune_in_range, stren_in_range
#%%
"""
params = {
   'axes.labelsize': 18,
   'font.size': 18,
   'font.family': 'sans-serif', # Optionally change the font family to sans-serif
   'font.serif': 'Arial', # Optionally change the font to Arial
   'legend.fontsize': 18,
   'xtick.labelsize': 16,
   'ytick.labelsize': 16, 
   'figure.figsize': [8.8, 8.8/1.618] # Using the golden ratio and standard column width of a journal
} 
plt.rcParams.update(params)

laser33 = Pulse(pulse_length = 30e-6, N = 600, wavelength = 5.51317846*we)
t33 = iontrap(G = 3, T0 = 0.001, n0 = np.array([50, 50, 50]), M = 500, ore = True, thermal_state = True, 
              neglect_entanglement = True, basis = False)
lasers3 = np.array([laser33])
c3, b3, d3, c_all3, b_all3, d_all3, c_pc3, b_pc3, d_pc3= plot_cooling_3_planar(t33, lasers3)
#%%
"""
"""
t4 = iontrap(G = 2, T0 = 0.001, n0 = np.array([0, 0]), structure = '1D', M = 10, 
             ore = False, thermal_state = False, neglect_entanglement = True, basis = False)

states = []
for _ in range(100): # realisations
    state_i = [[0, 0]]
    crnt_phonon_states = np.array([[0, 0] for _ in range(2)])
    for _ in range(30): # time step
        for i in range(2):
            sb_spe_ion = t4.sideband_spectrum(modes = crnt_phonon_states[i], ion = i, draw = False)
            strong_sidebands, str_sbs_stren = sb_spe_ion[0], sb_spe_ion[1]
            index_in_str_sb = np.random.choice(strong_sidebands.shape[1], size = None,
                                               p = str_sbs_stren / np.sum(str_sbs_stren))
            crnt_phonon_states[i] = crnt_phonon_states[i] + strong_sidebands[:, index_in_str_sb]
        state_i.append(list(sum(crnt_phonon_states)))
    states.append(np.array(state_i).T)
# print(states)
mode1 = np.array([data[0] for data in states]).T
# print(mode1)
# count_mode1 = [list(collections.Counter(each_time_step).items()) for each_time_step in mode1]
# print(count_mode1)
data_array = np.array(mode1)
# Create a figure for plotting the data as a 3D histogram.
#
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#
# Create an X-Y mesh of the same dimension as the 2D data. You can
# think of this as the floor of the plot.
#
x_data, y_data = np.meshgrid( np.arange(data_array.shape[1]),
                              np.arange(data_array.shape[0]) )
#
# Flatten out the arrays so that they may be passed to "ax.bar3d".
# Basically, ax.bar3d expects three one-dimensional arrays:
# x_data, y_data, z_data. The following call boils down to picking
# one entry from each array and plotting a bar to from
# (x_data[i], y_data[i], 0) to (x_data[i], y_data[i], z_data[i]).
#
x_data = x_data.flatten()
y_data = y_data.flatten()
z_data = data_array.flatten()
ax.bar3d( x_data,
          y_data,
          np.zeros(len(z_data)),
          1, 1, z_data )
#
# Finally, display the plot.
#
plt.show()
"""
"""
mode2 = np.array([data[1] for data in states]).T
mode3 = np.array([data[2] for data in states]).T
mode4 = np.array([data[3] for data in states]).T
"""