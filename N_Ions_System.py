# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 13:43:40 2020

@author: Corde
"""

from Base1 import *
from Param_Const1 import *

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
        self.sideband = sideband
        
class Trap:
    def __init__(self, pulse, Ni = 1, cooling = 'c', N = 100, n0 = 50, no_decay = True, 
                 ore = False, sideband = 2, M = 100, thermal_state = False): 
        """
        Initialisation.
        
        Parameters
        ----------
        
        pulse    : class, simulation of the laser pulse
        Ni       : int, number of ions in the trap
        cooling  : string, which mode to cool, 'c' is COM, 'b' is breathing 
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
        self.Ni, self.mode = Ni, Ni # may need to change self.mode for more than 2 ions 
        self.cooling = cooling
        self.n0, self.N, self.M = n0, N, M
        self.alln = np.array([self.n0 for i in range(self.N + 1)])
        self.no_decay, self.off_resonant_excite, self.tml = no_decay, ore, thermal_state
        self.excite, self.sideband = False, sideband
        
        if not self.tml:
            # list of states for different modes, by default, the [0] represents COM while [1] represents breathing
            self.n = (self.n0 * np.ones(self.Ni)).astype('int') 
            # list of trap frequencies for different modes, by default, the [0] represents COM while [1] represents breathing
            trap_freq = np.array([wz, np.sqrt(3)*wz])
        else:
            self.T0_th = nave_to_T(n0, wz)  # theoretical value of initial temperature
            self.n = Boltzmann_state(self.T0_th, wz, size = M)  # initialise array of n0
            actual_aven0, maxn0 = np.average(self.n), np.max(self.n)
            print('Initial temperature (theoretical):', self.T0_th, 'K')
            print('Initial average n   (actual)     :', actual_aven0)
            
            if abs(actual_aven0 - n0) >= 2:
                raise Exception('Deviation in ⟨n0⟩ too large. Try again!!!')
        
        # variables defined in the pulse class 
        self.L, self.Ld = pulse.L, pulse.Ld
        # self.t = pulse.t
        
        if isinstance(self.n0, int): # single Fock state 
            self.n_list = self.n0 * np.ones(self.M)
            self.n_list = self.n_list.astype(int)
            self.M_trial = self.M
        else: # list of initial states 
            self.n_list = self.n0.astype(int)
            self.M_trial = self.n0.size
            
        # calculate Rabi frequency matrix
        rab_dim = int(self.n0+1)
        self.R, self.Rd = [], [] # initialisation
        for mode in range(self.mode): 
            R_mode, Rd_mode = np.zeros((rab_dim, rab_dim)), np.zeros((rab_dim, rab_dim))
            w = trap_freq[mode] # choosing the resonant trap frequency for different modes 
            for i in range(rab_dim):
                for j in range(i+1):
                    R_mode[i, j] = eff_rabi_freq_single_mode(self.Ni, mode, i, j, freq_to_wav(self.L, w * pulse.sideband), w)
                    Rd_mode[i, j] = eff_rabi_freq_single_mode(self.Ni, mode, i, j, freq_to_wav(self.Ld, 0), w)
            R_mode, Rd_mode = R_mode + np.tril(R_mode, k = -1).T, Rd_mode + np.tril(Rd_mode, k = -1).T
            self.R.append(R_mode) 
            self.Rd.append(Rd_mode)
            
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

        om_red = []
        for mode in range(self.mode):
            # print('mode = %s, selectied n = %s'%(mode, self.n[mode]))
            sub_om_red = self.R[mode][self.n[mode]][self.n[mode]+pulse.sideband] * rb * not_in_ground_state[mode]
            om_red.append(sub_om_red)
        
        if self.cooling == 'c': # cooling the COM mode 
            com_prob = excitation_prob(self.n[0], om_red[0], pulse.t)
            breathing_prob = excitation_prob(self.n[1], om_red[1], pulse.t)
        
        """
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
          """
          
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