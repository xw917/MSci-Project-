# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 13:43:40 2020

@author: Corde
"""

from Base1 import *
from Param_Const1 import *
import pickle 

#%%
class pulse:
    def __init__(self, pulse_length = 5e-4, wavelength = L, decay_wavelength = LD, N = 10, sideband = [0, -2]):
        """
        Initialisation of laser.
        
        Parameters
        ----------
        pulse_length : (s)
        sideband     : list, sideband for [0]: COM mode and [1]: breathing mode 
        """
        self.t = pulse_length
        self.L = wavelength
        self.Ld = decay_wavelength
        self.N = N
        self.sideband = sideband
        
        self.R = np.zeros((500, 500))
        for i in range(100):
            for j in range(i + 1):
                # matrix elements for excitation Rabi frequencies 
                for sideband in self.sideband:
                    if sideband != 0:
                        self.R[i, j] = eff_rabi_freq(i, j, freq_to_wav(L, wz * sideband), wz)
        self.R = self.R + np.tril(self.R, k = -1).T
        
class Trap:
    def __init__(self, Ni = 2, n0 = [50, 50], M = 3, spin_motion = True): 
        """
        Initialisation of the system.
        
        Parameters
        ----------
        
        pulse    : class, simulation of the laser pulse
        Ni       : int, number of ions in the trap
        cooling  : string, which mode to cool, 'c' is COM, 'b' is breathing 
        n0       : list, initial state after Doppler cooling for COM and breathing mode
        N        : int, number of pulses applied (cycle of cooling process)
        M        : int, number of realisations, mainly used for averaging  
        spin-motion: int, 1 --> consider spin-motion entanglement
                          0 --> make approximation of outside LD regime 
                   
        if initialise a thermal state, n0 = initial (theoretical) average motional
            quantum number
        """
        # variables defined in the trap class
        # print('Start time:', datetime.datetime.now().time())
        self.Ni, self.mode = Ni, Ni # may need to change self.mode for more than 2 ions 
        self.n0, self.M = np.array(n0), M
        self.sm = spin_motion 
        self.excite = np.zeros(self.M)
        
        # list of trap frequencies for different modes, by default, the [0] represents COM while [1] represents breathing
        self.trap_freq = np.array([wz, np.sqrt(3)*wz])        
        # list of states for different modes, by default, the [0] represents COM while [1] represents breathing
        # self.n = np.multiply(self.n0, np.ones(self.Ni)).astype('int') 
        self.n = [[],[]]
        # average_number_of_state = [] 
        self.T0_th = nave_to_T(self.n0, wz)  # theoretical value of initial temperature
        for mode in range(self.mode):
            distribution_n0 = Boltzmann_state(self.T0_th[mode], self.trap_freq[mode], size = M)  # initialise array of n0
            self.n[mode] = distribution_n0
            # average_number_of_state.append(np.average(distribution_n0)) # average n 
        # self.nn, self.kk = average_number_of_state[0],  average_number_of_state[1]# n bar and k bar for COM and breathing mode respectively
        # thermal_state_probability = (self.nn**n)/((self.nn+1)**(n+1)) * (self.kk**k)/((self.kk+1)**(k+1))

        # variables defined in the pulse class 
        # self.L, self.Ld = pulse.L, pulse.Ld
        # self.t = pulse.t
        
            
        """
        Rabi strengths
        index 1: mode, [0] == COM, [1] == breathing
        index 2: rabi strengths, ['resonant_excite'], ['off_excite'], ['decay']
        index 3: sideband, [0] == carrier, [1] == 1RSB, [2] == 2RSB, [3] == 3RSB
        """
        self.all_rabi_strength = pickle.load(open('test', 'rb'))
            
        # print('Initialisation completed.')
        # print('Current time:', datetime.datetime.now().time())
        
    def apply_one_pulse(self, pulse):
        """
        Simulate the situation when one pulse is applied. The excitation 
        probability is calculated and whether or not excitation takes place 
        is determined. 
        
        !!! Need to modify to a more general form (cooling COM mode and cooling both modes simultaneously)
        """
        if self.sm:
            """Case for spin-motion entanglement (within Lamb-Dicke regime)"""
            eta_c = LambDicke(freq_to_wav(pulse.L, wz * pulse.sideband[1]), wz)/np.sqrt(2) # LD param for COM mode
            eta_b = LambDicke(freq_to_wav(pulse.L, wz * pulse.sideband[1]), wz)/np.sqrt(2*np.sqrt(3)) # LD param for breathing mode
            # omega0 = np.array([[self.R[n, n + pulse.sideband[mode]] * rb for n in self.n[mode]] for mode in range(self.Ni)])
            # omega0 = [[rb * eff_rabi_freq(n + pulse.sideband[mode], n, freq_to_wav(self.L, wz * pulse.sideband[mode]), wz) for n in self.n[mode]] for mode in range(self.Ni)]
            # print(omega0)
            for i in range(self.M):
                if not pulse.sideband[0]: # laser points to resonant frequency of breathing mode 
                    not_in_ground_state = (self.n[1][i]+pulse.sideband[1]) >= 0  # whether the motion has reached ground state
                    mode_cool = 1
                    omega0 = pulse.R[self.n[1][i], self.n[1][i] + pulse.sideband[1]] * rb * not_in_ground_state
                    ee, eg, gg = excitation_prob(self.n[1][i], omega0 * eta_b, pulse.t)
                    
                if not pulse.sideband[1]: # laser points to resonant frequency of COM mode 
                    not_in_ground_state = (self.n[0][i]+pulse.sideband[0]) >= 0  # whether the motion has reached ground state
                    mode_cool = 0
                    omega0 = pulse.R[self.n[0][i], self.n[0][i] + pulse.sideband[0]] * rb * not_in_ground_state
                    ee, eg, gg = excitation_prob(self.n[0][i], omega0[0][i] * eta_c, pulse.t)
                probability = np.array([gg, 2*eg, ee])
                eon = np.random.choice(3, size = 1, p = probability)[0]
                if eon == 1:
                    self.excite[i] = 1
                    self.n[mode_cool][i] -= 1
                if eon == 2: 
                    self.excite[i] = 1
                    self.n[mode_cool][i] -= 2

        
        if not self.sm:
            """Case when spin-motion entanglement not considered (outside Lamb-Dicke regime)"""
            n, k = self.n[0], self.n[1]        
            oc = self.all_rabi_strength[0]["off_excite"][-pulse.sideband] # rabi strength for COM mode 
            ob = self.all_rabi_strength[1]["resonant_excite"][-pulse.sideband] # rabi strength for breathing mode 
            thermal_factor = (self.nn**n)/((self.nn+1)**(n+1)) * (self.kk**k)/((self.kk+1)**(k+1))
            # Resonant excitation for breathing mode
            om_b_resonant = rb * oc[n][n] * ob[k][k-pulse.sideband] * not_in_ground_state[1]
            prob_b_resonant_ge = rabi_osci(pulse.t, om_b_resonant, 1)
            prob_b_resonant_ee = prob_b_resonant_ge**2
            
            # Off-resonant excitation for breathing mode
            # detune = self.trap_freq[1] * pulse.sideband # detuning 
            # om_b_off_res = rb * oc[n][n] * ob[k][k] * not_in_ground_state[1]
            
            # A, f = om_b_off_res**2 / (om_b_off_res**2 + detune**2), np.sqrt(om_b_off_res**2 + detune**2)
            # prob_b_off_res_ge = thermal_factor *rabi_osci(pulse.t, f, A)
            # prob_b_off_res_ee = (thermal_factor *rabi_osci(pulse.t, f, A))**2
            
            # probability = (1/(1+A))*np.array([((1 - 2*prob_b_resonant_ge - prob_b_resonant_ee) + (A - 2*prob_b_off_res_ge - prob_b_off_res_ee)), # no excitation
                          # (2*prob_b_resonant_ge), # on resonant excite one ion
                          # (prob_b_resonant_ee), # on resonant excite two ions
                          # (2*prob_b_off_res_ge), # off-resonant excite one ion
                          # (prob_b_off_res_ee)]) # off-resonant excite two ions 
                          
            probability = np.array([(1 - 2*prob_b_resonant_ge - prob_b_resonant_ee), 2*prob_b_resonant_ge, prob_b_resonant_ee])
            eon = np.random.choice(3, size = 1, p = probability)
            print(probability)
            
            self.excite = (eon != 0)
            if eon == 1:
                self.n[1] -= 1
            if eon == 2:
                self.n[1] -= 2
          
    def decay(self):
        '''
        Simulate the decay process (allow decay to different sidebands), also
        assume instantaneous decay.
        '''
        # calculate sideband strength
        # sideband_arr = np.array([np.arange(-2, 3, 1), np.arange(-2, 3, 1)]) # decay to 1RSB, 1BSB, carrier, 2RSB, 2BSB
        # determine whether the sideband exists or not, negative sideband not exist 
        # sideband_prob = []
        #"""
        for mode in range(2):
            for i in range(self.M):
                sideband_arr = np.arange(-2, 3, 1) # sidebands which can decay to
                sideband_exist = np.array([((self.n[mode][i] + sideband) >= 0) for sideband in sideband_arr]) # decay to state lower than ground state is not allowed 
                Rd = self.all_rabi_strength[mode]["decay"] 
                sideband_strg = np.concatenate((np.zeros(abs(np.min([self.n[mode][i]-2, 0]))),
                                                Rd[self.n[mode][i]][np.max([self.n[mode][i]-2, 0]) : self.n[mode][i]+2+1])) # sideband strength
                # print(sideband_strg)
                sideband_prob = np.multiply(sideband_exist, sideband_strg)**2 / np.sum(np.multiply(sideband_exist, sideband_strg)**2)
                decay_to = np.random.choice(sideband_arr.size, size = 1, p = sideband_prob)
                # print(self.excite[i])
                self.n[mode][i] += sideband_arr[decay_to[0]] * self.excite[i]

    def sideband_cool(self, pulse):
        """
        Simulate the cooling process, one cycle = apply a pulse + decay, full
        process is to repeat this cycle for N times. 
        """    
        # print(self.n)
        self.apply_one_pulse(pulse) # excitation step n -> n-1
        self.decay()
        self.excite = np.zeros(self.M)
        # print('now in state %s'%(self.n))
        
            
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
        all_trial_n, all_trial_m = sp.zeros((pulse.N + 1, self.M)), sp.zeros((pulse.N + 1, self.M))
        all_trial_n[0, :], all_trial_m[0, :] = self.n[0], self.n[1]

        for i in range(pulse.N):
            self.sideband_cool(pulse)
            all_trial_n[i+1, :] = self.n[0]
            all_trial_m[i+1, :] = self.n[1]
        if ave:
            all_trial_n_ave = np.einsum('ij->i', all_trial_n) / self.n[0].size
            all_trial_m_ave = np.einsum('ij->i', all_trial_m) / self.n[1].size
            return all_trial_n_ave, all_trial_m_ave
        else:
            return "No average"
        
        # print('Cooling completed.')
        # print('Cooling ends at:', datetime.datetime.now().time())
        
        # return all_trial_n_ave, all_trial_m_ave # , all_trial_n.T, all_trial_m.T