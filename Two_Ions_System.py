# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 13:43:40 2020

@author: Corde
"""

from Base1 import *
from Param_Const1 import *
from Schrodinger import *
import pickle 

#%%
"""
class pulse:
    def __init__(self, pulse_length = 5e-4, wavelength = L, decay_wavelength = LD, N = 10, sideband = np.array([0, -1])):

        Initialisation of laser.
        
        Parameters
        ----------
        pulse_length : (s)
        sideband     : list, sideband for [0]: COM mode and [1]: breathing mode 

        self.t = pulse_length
        self.L = wavelength
        self.Ld = decay_wavelength
        self.N = N
        self.sideband = sideband
        
        all_rabi_strength = pickle.load(open('test2', 'rb'))
        self.R_com = all_rabi_strength[-sideband[0]]
        self.R_b = all_rabi_strength[-sideband[1]]
"""       
class Trap:
    def __init__(self, Ni = 2, n0 = [50, 50], M = 10, spin_motion = False): 
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
        self.excite = [np.zeros(self.M), np.zeros(self.M)]
        
        # list of trap frequencies for different modes, by default, the [0] represents COM while [1] represents breathing
        self.trap_freq = np.array([wz, np.sqrt(3)*wz])        
        # list of states for different modes, by default, the [0] represents COM while [1] represents breathing
        self.n = [[],[]]
        # average_number_of_state = [] 
        self.T0_th = nave_to_T(self.n0, wz)  # theoretical value of initial temperature
        for mode in range(2):
            distribution_n0 = Boltzmann_state(self.T0_th[mode], self.trap_freq[mode], size = M)  # initialise array of n0
            self.n[mode] = distribution_n0
        self.n = np.array(self.n)
            # average_number_of_state.append(np.average(distribution_n0)) # average n 
        # self.nn, self.kk = average_number_of_state[0],  average_number_of_state[1]# n bar and k bar for COM and breathing mode respectively
            
        """
        Rabi strengths
        index 1: mode, [0] == COM, [1] == breathing
        index 2: rabi strengths, ['resonant_excite'], ['off_excite'], ['decay']
        index 3: sideband, [0] == carrier, [1] == 1RSB, [2] == 2RSB, [3] == 3RSB
        """
        self.all_rabi_strength = pickle.load(open('test', 'rb'))
        self.R  = LoadAnything('ions' + str(2) + '_laser_rabi.data')
        self.Rd = LoadAnything('ions' + str(2) + '_decay_rabi.data')
            
        # print('Initialisation completed.')
        # print('Current time:', datetime.datetime.now().time())
        
    def apply_one_pulse(self, pulse):
        """
        Simulate the situation when one pulse is applied. The excitation 
        probability is calculated and whether or not excitation takes place 
        is determined. 
        """
        if self.sm:
            """Case for spin-motion entanglement (within Lamb-Dicke regime)"""
            no_excited, h_non_zero, which_ion_change = laser_sb_ham_info(2)
            which_ion_chg_arr = ones((self.n.shape[1], *which_ion_change.shape),
                             dtype = 'int') * (which_ion_change - 1)
            # calculate total sideband strengths (terms in the hamiltonian matrix)
            total_sb_stren = 1
            for motion_state, sideband, mode in zip(self.n, pulse.sideband, range(2)):
                changed_motion_stt = no_excited[:,None] * sideband + motion_state
                sb_stren_this_mode = self.R[which_ion_chg_arr.transpose(1,2,0),mode,changed_motion_stt[:,None],
                                    changed_motion_stt[:,None].transpose(1,0,2)]
                total_sb_stren = total_sb_stren * sb_stren_this_mode
                H = einsum('ijk,ij->kij', total_sb_stren, h_non_zero)

            psi_0 = eye(1, M = H.shape[-1])[0]
            # solve the schrodinger's equation
            psi_t = SchrodingerEqn(H, psi_0, array([pulse.t]), h_factor = rb/2)[:,:,0]
            # calculate excite probabilty
            probs  = (psi_t * psi_t.conj()).real
            # determine the state which the system is excited to via Monte Carlo
            state_hilbertspace = array([np.random.choice(H.shape[-1], size = None,
                                         p = prob / np.sum(prob)) for prob in probs])
            # determine which ion(s) get excited
            self.excite = which_ion_excite(2)[state_hilbertspace,:].T
            # shift the motional states
            self.n = self.n + outer(pulse.sideband, no_excited[state_hilbertspace])

  
        if not self.sm:
            """Case when spin-motion entanglement not considered (outside Lamb-Dicke regime)"""
            n_all, k_all = self.n[0], self.n[1]  
            oc = self.all_rabi_strength[0]['resonant_excite'][-pulse.sideband[0]]
            ob = self.all_rabi_strength[1]['resonant_excite'][-pulse.sideband[1]]
            
            for i in range(self.M):
                n, k = n_all[i], k_all[i]
                # n, k = 31, 1
                # print(n, k)                
                if n+pulse.sideband[0] < 0 or k+pulse.sideband[1] < 0:
                    effective_rabi_eg = 0
                    p_eg= 0
                    p_ee = 0
                else:
                    effective_rabi_eg = rb * oc[n, n+pulse.sideband[0]] * ob[k, k+pulse.sideband[1]]
                    p_eg = rabi_osci(pulse.t, effective_rabi_eg, 1)
                    p_ee = p_eg**2

                probability = np.array([(1-p_eg)**2, 2*p_eg*(1-p_eg), p_eg**2])
                # print(probability)

                eon = np.random.choice(3, size = 1, p = probability)
                # print(eon)

                if eon == 1:
                    self.excite[0][i] = 1
                    self.excite[1][i] = 1
                    self.n[0][i] += pulse.sideband[0]
                    self.n[1][i] += pulse.sideband[1]
                if eon == 2:
                    self.excite[0][i] = 1
                    self.excite[1][i] = 1
                    self.n[0][i] += pulse.sideband[0]*2
                    self.n[1][i] += pulse.sideband[1]*2
                
                # print(self.excite)
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
        Rd_com = self.all_rabi_strength[0]["decay"] 
        Rd_b = self.all_rabi_strength[1]["decay"] 
        
        for mode in range(2):
            for i in range(self.M):
                sideband_arr = np.arange(-2, 3, 1) # sidebands which can decay to
                sideband_exist = np.array([((self.n[mode][i] + sideband) >= 0) for sideband in sideband_arr]) # decay to state lower than ground state is not allowed 
                # print(mode, self.n[mode][i])
                strg_com = np.concatenate((np.zeros(abs(np.min([self.n[0][i]-2, 0]))),
                                                Rd_com[self.n[0][i]][np.max([self.n[0][i]-2, 0]) : self.n[0][i]+2+1])) # sideband strength for COM mode
                strg_b = np.concatenate((np.zeros(abs(np.min([self.n[1][i]-2, 0]))),
                                                Rd_b[self.n[1][i]][np.max([self.n[1][i]-2, 0]) : self.n[1][i]+2+1])) # sideband strength for breathing mode
                sideband_strg = np.multiply(strg_com, strg_b)
                # print(sideband_strg)
                sideband_prob = np.multiply(sideband_exist, sideband_strg) / np.sum(np.multiply(sideband_exist, sideband_strg))
                # print(sideband_prob)
                decay_to = np.random.choice(sideband_arr.size, size = 1, p = sideband_prob)
                # print(self.excite[i])
                self.n[mode][i] += sideband_arr[decay_to[0]] * self.excite[mode][i]

    def sideband_cool(self, pulse):
        """
        Simulate the cooling process, one cycle = apply a pulse + decay, full
        process is to repeat this cycle for N times. 
        """    
        # print(self.n)
        self.apply_one_pulse(pulse) # excitation step n -> n-1
        # print(self.n)
        self.decay()
        self.excite = [np.zeros(self.M), np.zeros(self.M)]
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