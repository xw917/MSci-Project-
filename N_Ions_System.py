#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 00:48:01 2021

@author: apple
"""

from Schrodinger import *
from Base1 import *
from Param_Const1 import *
# from Two_Ions_System import *
from itertools import combinations

class Pulse:
    def __init__(self, pulse_length = 5e-4, wavelength = L, decay_wavelength = LD, N = 1000, sideband = np.array([0, -1])):
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
        
        all_rabi_strength = pickle.load(open('test2', 'rb'))
        self.R_com = all_rabi_strength[-sideband[0]]
        self.R_b = all_rabi_strength[-sideband[1]]
        
        
class iontrap:
    def __init__(self, G, T0, w = wz, no_decay = False, n0 = None, structure = '2D', # deleted the variable "pulse" here
                 ore = False, sideband = 2, M = 10, thermal_state = True, readout = False, neglect_entanglement = False, basis = True, basis_num = 1): 
        """
        Initialisation of a trap in which ions form a string
        
        Parameters
        ----------
        
        G        : int, number of ions in the string
        pulse    : class, simulation of the laser pulse
        w        : trap frequency
        n0       : initial photon state vector: [mode0, mode1, mode2, ...].
                   Ignoroed if thermal_state == True
        T0       : int/list, initial state after Doppler cooling 
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
        readout  : Whether to read Rabi strengths from stored data or to re-calculate.
        neglect_entanglement : True - neglect spin-motion entanglement and treat ions as independent 
                               False - take into account spin-motion entanglement 
                   
        if initialise a thermal state, n0 = initial (theoretical) average motional
            quantum number
        """
        # print('Start time:', datetime.datetime.now().time())
        self.G = G
        self.structure = structure
        
        if self.structure == '2D':
            self.modes_w, self.K, Kmatrix = icc_normal_modes_z(N = self.G)
            self.w = we
            if not basis:
                self.R = LoadAnything(str(self.G) + '_ions_planar.data')
                self.Rd = LoadAnything(str(self.G) + '_ions_planar_decay.data')
            else:
                self.R = LoadAnything(str(self.G) + '_ions_planar_change_basis_' + str(basis_num) +'.data')
                self.Rd = LoadAnything(str(self.G) + '_ions_planar_decay_change_basis_' + str(basis_num) +'.data')
                
            self.allowed_sideband = 1
            self.sidebands = LoadAnything(str(self.G) + '_ions_planar_sidebands.data')
            # self.sidebands = [[-1, 0, 0], [0, -1, 0], [0, 0, -1], [0, 0, 0]]
            # self.sidebands = [[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]]
            self.sidebands.append([0 for _ in range(self.G)])
        
        if self.structure == '1D':
            self.modes_w, self.K = string_normal_modes(self.G)  # normal frequencies and normal coordinates
            self.w = wz
            self.R  = LoadAnything('ions' + str(self.G) + '_laser_rabi.data')
            self.Rd = LoadAnything('ions' + str(self.G) + '_decay_rabi.data')
        
        # get information on the structure of on-resonant sideband hamiltonian
        self.wch_ion_ex = which_ion_excite(self.G)
        # ↑ a 2d vector which describes which ion(s) are excited in the state vector
        self.h_st = laser_sb_ham_info(self.G)
        
        self.tml = thermal_state
        self.M = M
        # self.N = pulse.N
        if not self.tml:
            self.n0 = n0
            self.n = (ones((self.M, self.n0.size)) * self.n0).T.astype('int')
        else:
            self.T0_th = T0  # value of initial temperature
            # generate random photon states for each mode
            self.n = array([Boltzmann_state(self.T0_th, wn * self.w, size = M) for wn in self.modes_w])
            
        # self.alln = np.array([self.n for i in range(self.N + 1)])
        self.excite = True * ones((self.G, self.M), dtype = bool)
        self.no_decay = no_decay
        self.off_resonant_excite = ore
        self.entg = neglect_entanglement
        
        # variables defined in the pulse class 
        # self.L = pulse.L
        # self.Ld = pulse.Ld
        self.Rb = rb
        # self.t = pulse.t
        
        
        '''
        Rabi frequency matrix has number of dimension = 4. It has the following axes that correspond
        to different variables:
        
        (which ion, which motional mode, motional state 0, motional state 1)
        '''
            
        # calculate Rabi frequency matrix
        """
        if not readout:
            if not thermal_state:
                rab_dim = int(n0 * 2)  # dimension of the 0th and 1st axes
            else:
    #            maxn0 = amax(self.n)
                maxn0 = 690
                print('max n:', maxn0)
                rab_dim = int(maxn0 + 10)  # dimension of the 0th and 1st axes
            self.R = sp.zeros((self.G, self.modes_w.size, rab_dim, rab_dim))
            self.Rd = sp.zeros((self.G, self.modes_w.size, rab_dim, rab_dim))
            for i in range(rab_dim):
                print('dimension 1:', i)
                for j in range(i + 1):
                    # matrix elements for excitation Rabi frequencies 
    #                nm_freq = self.w * self.modes_w[h]
                    nm_freq = ones((self.G, self.modes_w.size)) * self.modes_w * self.w
    #                        print(g, h, i, j)
                    self.R[:, :, i, j] = eff_rabi_freq(i, j, freq_to_wav(self.L, nm_freq * pulse.sideband),
                                                 nm_freq, normal_coor = np.abs(self.K))
                    # matrix elements for decay Rabi frequencies
                    self.Rd[:, :, i, j] = eff_rabi_freq(i, j, freq_to_wav(self.Ld, 0),
                                                 nm_freq, normal_coor = np.abs(self.K))
            self.R  = self.R + np.tril(self.R, k = -1).transpose(0,1,3,2)
            self.Rd = self.Rd + np.tril(self.Rd, k = -1).transpose(0,1,3,2)
        """
        #else:
        #self.R  = LoadAnything('ions' + str(self.G) + '_laser_rabi.data')
        #self.Rd = LoadAnything('ions' + str(self.G) + '_decay_rabi.data')
    
        
        # print('Initialisation completed.')
        # print('Current time:', datetime.datetime.now().time())
        
    def sideband_spectrum(self, modes = None, ion = 0, rang = 5.5, draw = True, incl_sb_thres = 0.06,
                          draw_width = 0.01):
        '''
        Draw the sideband spectrum to determ
        
        rang : range of frequencies to be determined(drawn). Default: 5.5 × trap frequencies
        modes: array, current mode vector. If not, give, take it to be the maximum of that of
               the self.R and self.Rd matrices
        draw:  whether to plot the spectra
        incl_sb_thres: sidebands with amplitude fraction of the maximum value of all
               sidebands of that mode are included
               
        draw_width: width of the bins in the spectrum bar plot
        '''
        if modes.any() == None:
            current_mode = self.Rd.shape[-1] * ones(self.modes_w.size)
        else:
            if not isinstance(modes, np.ndarray):
                modes = array(modes)
            current_mode = modes
        # For each phonon state, find the list of strongest sidebands and their strengths
        sbs_arr, sbs_lim  = [], []
        for mode_state, mode_ind in zip(current_mode, range(self.modes_w.size)):
            all_amp = self.Rd[ion, mode_ind, : , mode_state]
            no_red_sb = index_1st_greater_than(all_amp, amax(all_amp) * incl_sb_thres) - mode_state
            no_blu_sb = index_lst_greater_than(all_amp, amax(all_amp) * incl_sb_thres) - mode_state
            sbs_lim.append([no_red_sb, no_blu_sb])
            sbs_arr.append(arange(no_red_sb,  no_blu_sb + 1))
        # form a meshgrid to include all sidebands of sidebands
        sbs_mesh = meshgrid(*sbs_arr)
        # calculate detunings
        if draw:
            detuning = np.sum(stack(sbs_mesh, axis = -1) * self.modes_w, axis = -1).flatten()
        
        sbs_indc = stack(sbs_mesh, axis = -1)
        # sideband and sideband strengths array
        sidebands, sideband_strengths = [], []
        for state, mode_ind in zip(current_mode, range(self.modes_w.size)):
            phon_ind = sbs_indc[...,mode_ind].flatten()

            sidebands.append(phon_ind)
            sideband_strengths.append(self.Rd[ion, mode_ind, phon_ind+state, state])
        # calculate total strengths by multiplying sidebands together
        total_strengths = prod(array(sideband_strengths), axis = 0)
        # to plot the sideband spectrum
        if draw:
            fig, ax = plt.subplots()
            ax.bar(detuning, height = total_strengths, width = draw_width)
            
            detune = np.linspace(-rang, rang, 1000)
            free_space_linewidth = self.Rb**2 / (self.Rb**2 + (detune*self.w)**2)
            ax.plot(detune, free_space_linewidth, color = 'black')
            
            ax.set_ylabel('Sideband Strength', size = 14)
            ax.set_xticks(detuning)
            
#            labels = ['(' + str(m1) + ',' + str(m2) + ')' for m1, m2 in zip(sidebands[0], sidebands[1])]
            comp_sbs = [tuple(each_sb) for each_sb in list(array(sidebands).T)]
            sbs_labels = [str(comp_sb) for comp_sb in comp_sbs]
            ax.set_xticklabels(sbs_labels)
            ax.tick_params(bottom = True, labelbottom = True, top = False, labeltop = False,
                            labelrotation = 45)
            
            ax2 = ax.twiny()
            ax.set_xlim([-rang, rang])
            ax2.set_xlim(ax.get_xlim())
            
            sidebands = array(sidebands)
            return sidebands, total_strengths, fig, ax
        else:
            sidebands = array(sidebands)
            after_decay = current_mode[:,None] + sidebands
            if amin(after_decay) < 0:
                raise Exception('Negative motional state somewhere (after decay)')
            return sidebands, total_strengths
        
    def plot_sb(self, modes, sideband, ion = 0, plot = True, dotplot = False):
        '''
        Plot sideband strength for the sideband of mode.
        '''
        sideband_strengths = self.R[ion, modes, :, :].diagonal(offset = sideband,
                                   axis1 = -2, axis2 = -1)
        if plot:
            fig, ax = plt.subplots()
            for sideband_strength, mode in zip(sideband_strengths, modes):
                ax.plot(sideband_strength,
                        label = 'Sideband %i for ion %i and mode %i' % (sideband, ion + 1, mode))
                if dotplot:
                    ax.plot(sideband_strength, '.')
            ax.legend()
            return sideband_strengths, fig, ax
        else:
            return sideband_strengths
    
    def apply_one_pulse(self, pump):
        '''
        Apply one pulse of the pump laser to the trapped ion system
        
        negl_entang: bool. Whether to neglect spin-motion entanglement.
            If true, treat each ion as independent rabi oscill.
        '''
        if not self.entg:
            # set up the hamiltonian for all simulations
            H = self.laser_sideband_hamiltonian(pump.sideband)
            # set-up initial states
            psi_0 = eye(1, M = H.shape[-1])[0]
            # solve the schrodinger's equation
            psi_t = SchrodingerEqn(H, psi_0, array([pump.t]), h_factor = self.Rb / 2)[:,:,0]
            # calculate excite probabilty
            probs  = (psi_t * psi_t.conj()).real
#            print('prob:', probs)
#            print(probs[0,0])
            '''
            assume there is no off-resonant excitation since self.Rb is significantly
            smaller than trap frequency
            '''
            # determine the state which the system is excited to via Monte Carlo
            state_hilbertspace = array([np.random.choice(H.shape[-1], size = None,
                                             p = prob / np.sum(prob)) for prob in probs])
            self.ham = H
            self.psit = psi_t
            self.stt_hilb = state_hilbertspace
            # determine which ion(s) get excited
            self.excite = self.wch_ion_ex[state_hilbertspace,:].T
            # shift the motional states
            self.nbefore = self.n
            self.n = self.n + outer(pump.sideband, self.h_st[0][state_hilbertspace])
        else:            
            # determine the excitation order
            exci_orders = array([np.random.permutation(ions) for ions in ones(self.n.shape).T * arange(self.G)],
                                 dtype = int).T 
            # print(exci_orders)
            # excite each ion by order
            current_mode, current_exci = self.n, self.excite
            # print('initial %s'%(current_mode))
            frequencies = [round(i, -1) for i in self.modes_w * self.w]
            laser_sideband = round(pump.L, -1)
            # print(frequencies)
            # print('laser sideband = %s'%(laser_sideband))
            for exci in exci_orders:
                # print(exci)
                #print(current_mode)
                if not self.off_resonant_excite: # off resonant excitation not considered 
                    total_sb_stren = rb
                    for motion_state, mode, sb in zip(current_mode, range(self.modes_w.size), pump.sideband):
                        # print('motion state = %s'%(motion_state))
                        # print(motion_state + sb)
                       sb_stren_this_mode = self.R[exci, mode, motion_state, motion_state + sb]
                       total_sb_stren *= sb_stren_this_mode
                    # print('sideband_stren = %s'%(total_sb_stren))

                    # rabi oscillate to deduce propbabilities
                    # print('total sideband strength = %s'%(total_sb_stren))
                    probs = rabi_osci(pump.t, total_sb_stren, 1)
                    # print('probs = %s'%(probs))
                    # Monte Carlo to determine whether excite or not
                    excite_or_not = array([np.random.choice(arange(2), size = None, p = [1- prob, prob]) for prob in probs])
                    # update the motional states
                    current_mode = current_mode + outer(pump.sideband, excite_or_not)
                    # print('excite or not = %s'%(excite_or_not))
                    # update the excitation state matrix
                    current_exci[exci, :] = excite_or_not
                    # print('current excite = %s'%(current_exci))
                else: # consider off resonant excitation 
                    #""" find detuning for each possible excitation sideband"""
                    # print(self.sidebands)

                    if np.all(current_mode >= self.allowed_sideband):
                        sidebands = [np.delete(self.sidebands, -1, 0) for _ in range(self.M)] # excitation sidebands
                        selected_sideband = [self.sidebands for _ in range(self.M)] 
                        detune_i = [sum(np.array(sidebands_ij) * frequencies) + laser_sideband for sidebands_ij in np.delete(self.sidebands, -1, 0)]
                        detuning = [detune_i for _ in range(self.M)]                                    
                    else:
                        sidebands = find_sideband(current_mode, np.delete(self.sidebands, -1, 0))
                        selected_sideband = find_sideband(current_mode, self.sidebands)
                        detuning = [[sum(np.array(sidebands_ij) * frequencies) + laser_sideband for sidebands_ij in sidebands_i] for sidebands_i in sidebands]
                    
                    # print('sideband is = %s'%(np.array(sidebands)))
                    # print('current mode = %s'%(current_mode))
                    # print('detuning = %s'%(detuning))
                    # selected_sideband = [[sidebands[i][index] for index in sideband_index[i]] for i in range(self.M)] 
                    # print('selected sideband = %s'%(selected_sideband))
                    
                    #total_sb_stren = rb
                    #for motional_state, mode, sb
                    
                    changed_state, current_exci_i = [], []
                    for i in range(self.M):
                        sideband_stren_i = []
                        prob_amp, prob_freq = [], []
                        sigf_sb = []
                        # excite_probability_i = np.zeros(len(detuning[i]))
                        not_excite_probability_i = 0
                        for j in range(len(sidebands[i])):  
                            total_sb_stren = rb
                            # print('? %s'%(sidebands[i][j]))
                            for mode in range(self.modes_w.size):
                                # print('! %s, %s, %s, %s'%(exci[i], mode, int(current_mode[mode][i]), int(current_mode[mode][i] + sidebands[i][j][mode])))
                                total_sb_stren *= self.R[exci[i], mode, int(current_mode[mode][i]), int(current_mode[mode][i] + sidebands[i][j][mode])]
                            A = amplitude(total_sb_stren, detuning[i][j]) # calculate excitation probability amplitude
                            if A >= 0.01: # add a threshold which we accept the sideband
                                prob_amp.append(A)
                                prob_freq.append(frequency(total_sb_stren, detuning[i][j]))
                                sideband_stren_i.append(total_sb_stren)
                                sigf_sb.append(sidebands[i][j])
                        # print('sideband strength = %s'%(sideband_stren_i))
                        excite_probability_i = np.array(prob_amp) * (np.sin(np.array(prob_freq) * pump.t))**2
                        # print('excitation probability = %s'%(excite_probability_i))
                        if len(excite_probability_i) == 0:
                            not_excite_probability_i += 1
                        else:
                            for k in range(len(prob_amp)):
                                not_excite_probability_i += subtract(prob_amp[k], excite_probability_i[k])
                        # print('not excite prob = %s'%(not_excite_probability_i))
                        all_probability_i = np.append(excite_probability_i, not_excite_probability_i)
                        sigf_sb.append(np.zeros(self.G))
                        # print(sigf_sb)
                        norm_prob_i = np.array(all_probability_i)/sum(all_probability_i)
                        # print('normalised probability= %s'%(norm_prob_i))
                        excite_or_not = np.random.choice(arange(len(norm_prob_i)), size = None, p = norm_prob_i)
                        # pick out the excitation sideband for each mode
                        # print('excite or not = %s'%(excite_or_not))
                        changed_state.append(sigf_sb[excite_or_not])
                        if excite_or_not == len(sigf_sb) - 1:
                            current_exci_i.append(False)
                        else:
                            current_exci_i.append(True)
                    # print('changed state = %s'%(changed_state))
                    # print(current_mode)
                    current_mode = (current_mode.T + changed_state).T
                    # print('after = %s'%(current_mode))
                    current_exci[exci, :] = current_exci_i
                    # print('exci = %s'%(current_exci))
            # renew the system
            self.n = current_mode.astype(int)
            # print('excited to = %s'%(self.n))
            self.excite = current_exci
            # print('exci = %s'%(current_exci))
                    
        #print('n minimum (after pump):', amin(self.n))
        if amin(self.n) < 0:
            raise Exception('Negative motional state (after pump)')
                
    def decay(self):
        '''
        Determine the sideband which the system decays upon
        
        structure of array(all_decay_sb):
            (each simulation, diff. ions, sidebands for different modes)
        structure of self.excite:
            (diff ions, each simulation)
        '''
        decayed_state = []
        # print('n minimum (before decay):', amin(self.n))
        for excit, phonon_states in zip(self.excite.T, self.n.T):
            if amin(self.n) < 0:
                break
            # pick up the excited ions
            which_ions_exc = np.nonzero(excit)[0]
            # randomise the decay order (which ion decays first is random)
            np.random.shuffle(which_ions_exc)
            '''
            Decay each ion successively (to avoid negative motional states)
            '''
            crnt_phonon_states = phonon_states
            for which_ion in which_ions_exc:
                # pick out the strongest sidebands
                sb_spe_ion = self.sideband_spectrum(modes = crnt_phonon_states, ion = which_ion, draw = False)
                strong_sidebands, str_sbs_stren = sb_spe_ion[0], sb_spe_ion[1]
                # monte carlo to find the decay sideband for this ion
                index_in_str_sb = np.random.choice(strong_sidebands.shape[1], size = None,
                                                   p = str_sbs_stren / np.sum(str_sbs_stren))
                crnt_phonon_states = crnt_phonon_states + strong_sidebands[:, index_in_str_sb]
            
            decayed_state.append(crnt_phonon_states)
        
        self.n = array(decayed_state).T
        # print('decay to = %s'%(self.n))
        # de-excite all ions
        self.excite = zeros(self.excite.shape, dtype = int)

        if amin(self.n) < 0:
            raise Exception('Negative motional state (after decay)')
    
    
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
            
    def sideband_cool_sch(self, pulse, decay = True, take_average = True):
        '''
        Simulate the entire sideband cool process with the pulse
        '''
        self.no_decay = not decay
        # set up the array to store all motional states
        all_trial_n = zeros((pulse.N + 1, self.modes_w.size, self.M), dtype = int)
        all_trial_n[0,:,:] = self.n
        for i in range(pulse.N):
            self.sideband_cool(pulse)
            all_trial_n[i + 1,:,:] = self.n
            # print('pulse %i our of %i completed.' % (i, self.N))
        # take average of all trials
        if take_average:
            all_trial_n_ave = average(all_trial_n, axis = 2)
            return all_trial_n, all_trial_n_ave
        else:
            return all_trial_n
    
    def laser_sideband_hamiltonian(self, sidebands):
        '''
        N: number of ions in the string
        h_st: structure of the hamiltonian = laser_sb_ham_info(N)
        eta: lamb-dicke parameter of a single ion in the trap
        mode_vec : current mode vector
        sideband: sideband vector (e.g. [-1, -1, -1] for red sideband on all modes)
        '''
#        eta_tensor = eta * (self.K / sqrt(self.modes_w))
        
        no_excited, h_non_zero, which_ion_change = self.h_st
        '''
        How to set up the hamiltonian?
        1.Rabi frequency = Rabi frequency (n0, n1, which ion, which mode)
        2.Find the corresponding Rabi frequency using indexing
        '''
        which_ion_chg_arr = ones((self.n.shape[1], *which_ion_change.shape),
                                 dtype = 'int') * (which_ion_change - 1)
        # calculate total sideband strengths (terms in the hamiltonian matrix)
        total_sb_stren = 1
        for motion_state, sideband, mode in zip(self.n, sidebands, range(self.modes_w.size)):
            changed_motion_stt = no_excited[:,None] * sideband + motion_state
            sb_stren_this_mode = self.R[which_ion_chg_arr.transpose(1,2,0),mode,changed_motion_stt[:,None],
                                        changed_motion_stt[:,None].transpose(1,0,2)]
            total_sb_stren = total_sb_stren * sb_stren_this_mode
        # now compute the actual hamiltonian
        hamiltonian = einsum('ijk,ij->kij', total_sb_stren, h_non_zero)
        return hamiltonian
