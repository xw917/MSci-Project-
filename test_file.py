#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 20:28:36 2020

@author: cordelia
"""

import timeit
import datetime

from Base1 import *
from Param_Const1 import *

def wrapper(func, *args, **kwargs):
    def wrap():
        return func(*args, **kwargs)
    return wrap

#%%
class pulse:
    def __init__(self, pulse_length = 1e-5, wavelength = L, decay_wavelength = LD, sideband = -1):
        """
        pulse_length : (s)
        sideband: which sideband.
        """
        self.t = pulse_length
        self.L = wavelength
        self.Ld = decay_wavelength
        self.sideband = np.array(sideband)

class trap:
    def __init__(self, pulse, N = 100, n0 = 50, no_decay = False, 
                 ore = False, sideband = 2, M = 1, thermal_state = False): 
        """
        Initialisation.
        
        Parameters
        ----------
        
        pulse    : class, simulation of the laser pulse
        n0       : int/list, initial state after Doppler cooling 
        N        : int, number of pulses applied (cycle of cooling process)
        no_decay : True - when decay to other motional sidebands is not considered
                   False - when decay to other motional sidebands is considered
        ore      : True - when excitation to carrier is considered 
                   False - off resonant excitation not considered 
        sideband : int, input required if no_decay is False 
        M        : int, number of realisations, mainly used for averaging  
        thermal_state
                 : True - thermal state
                   False - Fock state
                   
        if initialise a thermal state, n0 = initial (theoretical) average motional
            quantum number
        """
        # variables defined in the trap class
        # print('Start time:', datetime.datetime.now().time()) 
        self.n0 = n0
        self.N = N
        self.alln = np.array([self.n0 for i in range(self.N + 1)])
        self.excite = False
        self.no_decay = no_decay
        self.off_resonant_excite = ore
        self.M = M
        self.tml = thermal_state
        
        if not self.tml:
            self.n = (n0 * np.ones(M)).astype('int')
        else:
            self.T0_th = nave_to_T(n0, wz)  # theoretical value of initial temperature
            self.n = Boltzmann_state(self.T0_th, wz, size = M)  # initialise array of n0
            actual_aven0, maxn0 = np.average(self.n), np.max(self.n)
            print('Initial temperature (theoretical):', self.T0_th, 'K')
            print('Initial average n   (actual)     :', actual_aven0)
            """
            if abs(actual_aven0 - n0) >= 2:
                raise Exception('Deviation in ⟨n0⟩ too large. Try again!!!')
            """
        
        # variables defined in the pulse class 
        self.L = pulse.L
        self.Ld = pulse.Ld
#        self.t = pulse.t
        
        self.sideband = sideband
            
        if isinstance(self.n0, int): # single Fock state 
            self.n_list = self.n0 * np.ones(self.M)
            self.n_list = self.n_list.astype(int)
            self.M_trial = self.M
        else: # list of initial states 
            self.n_list = self.n0.astype(int)
            self.M_trial = self.n0.size
            
        # calculate Rabi frequency matrix
        if not thermal_state:
            rab_dim = int(n0 * 2)
        else:
            rab_dim = int(maxn0 + 10)
        self.R, self.Rd = sp.zeros((rab_dim, rab_dim)), sp.zeros((rab_dim, rab_dim))
        for i in range(rab_dim):
            for j in range(i + 1):
                # matrix elements for excitation Rabi frequencies 
                self.R[i, j] = eff_rabi_freq(i, j, freq_to_wav(self.L, wz * pulse.sideband), wz)
                # matrix elements for decay Rabi frequencies
                self.Rd[i, j] = eff_rabi_freq(i, j, freq_to_wav(self.Ld, 0), wz)
        self.R = self.R + np.tril(self.R, k = -1).T
        self.Rd = self.Rd + np.tril(self.Rd, k = -1).T
        
        # print(self.R)
        
        # print('Initialisation completed.')
        # print('Current time:', datetime.datetime.now().time())
        
    def apply_one_pulse(self, pulse):
        """
        Simulate the situation when one pulse is applied. The excitation 
        probability is calculated and whether or not excitation takes place 
        is determined. 
        """
        not_in_ground_state = (self.n - 1) >= 0  # whether the motion has reached ground state
        
        # construct the corresponding array of eff.r.freq
        om_red = np.array([self.R[n, n + pulse.sideband] for n in self.n]) * rb * not_in_ground_state
        # print('one pulse')
        # print('om_red = %s'%(om_red))
        red_prob = rabi_osci(pulse.t, om_red, 1) # calculate sideband excite probability
        
        if self.off_resonant_excite:
            detune = wz * pulse.sideband # detuning
            # construct carrier strength
            om_carr = np.array([self.R[n, n] for n in self.n]) * rb
            # probability for carrier excitation
            amplitude, freq = om_carr**2 / (om_carr**2 + detune**2), np.sqrt(om_carr**2 + detune**2)
            carr_prob = rabi_osci(pulse.t, freq, amplitude)
            # determine whether: excite upon sideband(2), excite upon carrier(1) or don't excite(0)
            eon = np.array([np.random.choice(3, size = 1,
                  p = np.array([(1-red_prob[i]) + (amplitude[i]-carr_prob[i]), carr_prob[i], red_prob[i]]) / (1 + amplitude[i]))[0]
                  for i in range(self.M)])
            # excite the ion
            self.excite = (eon != 0)
            # shift the photon state
            self.n = self.n + self.excite * (1 - eon)
        else:  # if no carrier excitation
            eon = np.array([np.random.choice([0, 1], size = 1,
                                             p = [1 - rp, rp])[0] for rp in red_prob])
            # print('eon = %s'%(eon))
            # excite the ion
            self.excite = (eon != 0)
            # shift the photon state
            self.n = self.n + self.excite * (- eon)
            
    def decay(self):
        '''
        Simulate the decay process (allow decay to different sidebands), also
        assume instantaneous decay.
        '''
        # calculate sideband strength
        sideband_arr = np.arange(- self.sideband, self.sideband + 1, 1)
        # determine whether the sideband exists or not, negative sideband not exist 
        sideband_exist = np.array([((self.n + sideband) >= 0) for sideband in sideband_arr])
        # effective Rabi frequency/ free space Rabi frequency 
        sideband_strg = np.array([np.concatenate((np.zeros(abs(np.min([n - self.sideband, 0]))),
                                  self.Rd[n, np.max([n - self.sideband, 0]): n + self.sideband + 1]))
                                  for n in self.n]).T
        # print('sideband strg is %s'%(sideband_strg))
        sideband_prob = (sideband_exist * sideband_strg)**2
        sideband_prob /= np.sum(sideband_prob, axis = 0)
        # print('after normalisation: %s'%(sideband_prob))
        # choose a sideband
        heat_index = np.array([np.random.choice(sideband_arr.size, size = 1,
                                    p = sideband_prob[:,i])[0] for i in range(self.n.size)])
        # implement the decay
        self.n = self.n + sideband_arr[heat_index] * self.excite
        self.excite = np.zeros(self.n.size, dtype = bool)
        # print('decay to %s'%(self.n))
        
    def sideband_cool(self, pulse):
        """
        Simulate the cooling process, one cycle = apply a pulse + decay, full
        process is to repeat this cycle for N times.
        
        Return
        ------
        self.alln : list, the list of evolution of state 
        """     
        self.apply_one_pulse(pulse) # excitation step n -> n-1
        # print('now in state %s'%(self.n))
        if not self.no_decay:
            self.decay()
        
            
    def sideband_cool_sch(self, pulse, ave = True):
        """
        Simulate the cooling process for multiple realisations 
        
        Parameters 
        ----------
        ave : True - calculate average
              False - ignore average 
        
        Return
        ------
        all_trial_n     : array, cooling trajectory for all the trials 
        all_trial_n_ave : array, average of all the trials, if the input ave is 
                          false, then this will output "No average"
        """

        # print('Cooling starts at:', datetime.datetime.now().time())
        all_trial_n = sp.zeros((self.N + 1, self.M))
        all_trial_n[0, :] = self.n
        for i in range(self.N):
            self.sideband_cool(pulse)
            all_trial_n[i+1, :] = self.n
        if ave:
            all_trial_n_ave = np.einsum('ij->i', all_trial_n) / self.n.size
        else:
            all_trial_n_ave = "No average"
        
        # print('Cooling completed.')
        # print('Cooling ends at:', datetime.datetime.now().time())
        
        return all_trial_n.T, all_trial_n_ave
    
    def matrix_calculator(self, pulse, N0):
        """
        Calculate the matrix for excitation probability
        
        Parameters
        ----------
        N0 : int, initial state
        
        Return
        ------
        matrix : list of lists, matrix for excitation probability 
        """
        matrix = np.zeros((N0,  N0)) # matrix for probabilities
        for i in range(1, N0):
            Om_red = self.R[i, i-1] * rb
            Red_prob = rabi_osci(pulse.t, Om_red, 1) 
            matrix[i, i] = 1-Red_prob
            matrix[i-1, i] = Red_prob
        return matrix
    
    def decay_matrix_calculator(self, N0):
        """
        Calculate the matrix for decay probability
        
        Parameters
        ----------
        N0 : int, initial state
        
        Return
        ------
        decay_ matrix : list of lists, matrix for decay probability
        """
        decay_matrix = np.zeros((N0, N0))
        # decay_matrix[-1, -1] = 1
        sideband_arr = np.arange(-self.sideband, self.sideband + 1, 1)
        # print(sideband_arr)
        for n in range(0, N0): 
            # print(n)
            for sideband in sideband_arr:
                if n + sideband >= 0 and n + sideband < N0:
                    Prob_red = self.Rd[n, n + sideband]
                    decay_matrix[n, n + sideband] = Prob_red**2

        for column in range(len(decay_matrix[0])):
            s = sumColumn(decay_matrix, column)
            for row in range(len(decay_matrix)):
                decay_matrix[row, column] /= s

        return decay_matrix
        
    def matrix_method(self, pulse):
        """
        The matrix method to simulate cooling process 
 
        Return
        ------
        data : list, probability distrubution after applying N pulses
        """
        all_data = []
        if isinstance(self.n0, int):
            n0 = [self.n0]
        else: 
            n0 = self.n0
            
        for n in n0:
            N0 = n+1  
            matrix = self.matrix_calculator(pulse, N0)  
            if self.no_decay == False: 
                # decay matrix is calculated only when decay is considered 
                decay_matrix = self.decay_matrix_calculator(N0)
            
            data = []
            distr = np.zeros((N0,1))
            distr[-1] = 1
            data.append(distr)
        
            for i in range(self.N):
                # probability distribution after each pulse
                distr = np.matmul(matrix, distr) 
                # print(sum(distr))
            
                if self.no_decay == False: # case when decay is considered            
                    # probability distribution taking into account decay
                    distr = np.matmul(decay_matrix,  distr) 
                    # print(sum(distr))
                data.append(distr)
            all_data.append(data)
        return n0, all_data

