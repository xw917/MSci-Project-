#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 00:48:01 2021

@author: apple
"""

from Schrodinger import *
from Base1 import *

class pulse:
    def __init__(self, pulse_length = 1e-5, wavelength = 729e-9, decay_wavelength = 379e-9,
                 sideband = array([-1, -1]), Rb = rb):
        """
        pulse_length : (s)
        sideband: which sideband.
        """
        self.t = pulse_length
        self.L = wavelength
        self.Ld = decay_wavelength
        self.sideband = sideband
        self.Rb = Rb
        
class iontrap:
    def __init__(self, G, pulse, T0, w = wz, N = 100, no_decay = False, n0 = None, 
                 ore = False, sideband = 2, M = 100, thermal_state = True, readout = True): 
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
                   
        if initialise a thermal state, n0 = initial (theoretical) average motional
            quantum number
        """
        # print('Start time:', datetime.datetime.now().time())
        self.G = G
        self.modes_w, self.K = string_normal_modes(self.G)  # normal frequencies and normal coordinates
        self.w = w
        
        # get information on the structure of on-resonant sideband hamiltonian
        self.wch_ion_ex = which_ion_excite(self.G)
        # ↑ a 2d vector which describes which ion(s) are excited in the state vector
        self.h_st = laser_sb_ham_info(self.G)
        
        self.tml = thermal_state
        self.M = M
        if not self.tml:
            self.n0 = n0
            self.n = (ones((self.M, self.n0.size)) * self.n0).T.astype('int')
#            self.n = array([arange(250) + 1, arange(250) + 1], dtype = 'int')
        else:
            self.T0_th = T0  # value of initial temperature
            # generate random photon states for each mode
            print('Mean phonon state:', cs.k * 1e-3 / (cs.hbar * self.w))
            self.n = array([Boltzmann_state(self.T0_th, wn * self.w, size = M) for wn in self.modes_w])
            
        self.excite = True * ones((self.G, self.M), dtype = bool)
        self.no_decay = no_decay
        self.off_resonant_excite = ore
        
        # variables defined in the pulse class 
        self.L = pulse.L
        self.Ld = pulse.Ld
        self.Rb = pulse.Rb
        self.t = pulse.t
        
        self.sideband = pulse.sideband
        
        '''
        Rabi frequency matrix has number of dimension = 4. It has the following axes that correspond
        to different variables:
        
        (which ion, which motional mode, motional state 0, motional state 1)
        '''
            
        # calculate Rabi frequency matrix
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
#                    print(np.abs(self.K), nm_freq, freq_to_wav(self.L, nm_freq * pulse.sideband))
                    self.R[:, :, i, j] = eff_rabi_freq(i, j, freq_to_wav(self.L, nm_freq * pulse.sideband),
                                                 nm_freq, normal_coor = np.abs(self.K))
                    # matrix elements for decay Rabi frequencies
                    self.Rd[:, :, i, j] = eff_rabi_freq(i, j, freq_to_wav(self.Ld, 0),
                                                 nm_freq, normal_coor = np.abs(self.K))
            self.R  = self.R + np.tril(self.R, k = -1).transpose(0,1,3,2)
            self.Rd = self.Rd + np.tril(self.Rd, k = -1).transpose(0,1,3,2)
        else:
            self.R  = LoadAnything('ions' + str(self.G) + '_laser_rabi.data')
            self.Rd = LoadAnything('ions' + str(self.G) + '_decay_rabi.data')
        
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
            current_mode = self.R.shape[-1] * ones(self.modes_w.size)
        else:
            if not isinstance(modes, np.ndarray):
                modes = array(modes)
            current_mode = modes
        # For each phonon state, find the list of strongest sidebands and their strengths
        sbs_arr, sbs_lim  = [], []
        for mode_state, mode_ind in zip(current_mode, range(self.modes_w.size)):
            all_amp = self.R[ion,mode_ind,:,mode_state]
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
            sideband_strengths.append(self.R[ion,mode_ind,phon_ind+state,state])
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
    
    def apply_one_pulse(self, pump, negl_entang = False):
        '''
        Apply one pulse of the pump laser to the trapped ion system
        
        negl_entang: bool. Whether to neglect spin-motion entanglement.
            If true, treat each ion as independent rabi oscill.
        '''
        if not negl_entang:
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
            # obtain the rabi frequencies for a modes for all ions
            total_sb_stren = 1
            for motion_state, mode, sb in zip(self.n, range(self.modes_w.size), pump.sideband):
                sb_stren_this_mode = self.R[arange(self.G)[:,None], mode, motion_state, motion_state + sb]
                total_sb_stren = total_sb_stren * sb_stren_this_mode
            # rabi oscillate to deduce probabilities
            probs, excites = rabi_osci(pump.t, total_sb_stren * self.Rb, 1).T, []
            for n_trial, prob in zip(self.n.T, probs):  # each prob corresponds to one simulation
                '''
                sometimes an independent oscil. treatment fails to see the 'sidebands' which leads to one or more
                of the motional states below zero (which vanish when being corrected treated in the
                'full hamiltonian' method). The maximum allowed no. of ion's excitation under this case is
                manually determined. Any excitation cases > than allowed no. of ions is removed
                '''
                hmny_blw_grnd_stt = int(amin(n_trial + pump.sideband * self.G))
                if hmny_blw_grnd_stt < 0:
                    # calculate the cut-off in the ion excitation array
                    ion_exc_patt_cutoff = argmax(self.h_st[0] > (self.G + hmny_blw_grnd_stt))
                    ion_exc_patt = self.wch_ion_ex[:ion_exc_patt_cutoff, :]
                else:
                    ion_exc_patt_cutoff = self.h_st[0].shape[0]
                    ion_exc_patt = self.wch_ion_ex
                ion_ex_pattern_prob = prod(ion_exc_patt * prob + np.logical_not(ion_exc_patt) * (1 - prob),
                                           axis = 1)
                # Monte Carlo to determine which ion to excite
                print(ion_exc_patt_cutoff)
                which_ion_exci = np.random.choice(arange(ion_exc_patt_cutoff), size = None,
                                                  p = ion_ex_pattern_prob / np.sum(ion_ex_pattern_prob))
                excites.append(ion_exc_patt[which_ion_exci, :])
            # change the excitation array and phonon state array
            self.excite = array(excites).T
            
            self.n = self.n + einsum('ik,j->jk', self.excite, pump.sideband)
        print('n minimum (after pump):', amin(self.n))
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
        print('n minimum (before decay):', amin(self.n))
        for excit, phonon_states in zip(self.excite.T, self.n.T):
            if amin(self.n) < 0:
                break
            # pick up the excited ions
            which_ions_exc = nonzero(excit)[0]
            # randomise the decay order
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
        # de-excite all ions
        self.excite = zeros(self.excite.shape, dtype = int)

        if amin(self.n) < 0:
            raise Exception('Negative motional state (after decay)')
    
    def sideband_cool(self, pulse, negl_entang = False):
        """
        Simulate the cooling process, one cycle = apply a pulse + decay, full
        process is to repeat this cycle for N times.
        """     
        self.apply_one_pulse(pulse, negl_entang = negl_entang) # excitation step n -> n-1

        if not self.no_decay:
            self.decay()
            
    def sideband_cool_sch(self, pumpsch, decay = True, take_average = True, negl_entang = False):
        '''
        Simulate the entire sideband cool process with
        
        pumpsch: pump scheme = (pumps groups, no. of cycle for each group)
        
        e.g. pumpsch = ([[pp0,pp1,pp2],[pp3,pp4]], [2,4])
        
        then the following pump scheme is implemented:
            [pp0,pp1,pp2] * 2 + [pp3,pp4] * 4
            = [pp0,pp1,pp2,pp0,pp1,pp2,pp3,pp4,pp3,pp4,pp3,pp4,pp3,pp4]
        '''
        pump_groups, cycles = pumpsch
        pump_groups_len = []
        # Calculate N: total number of pulses in the pump scheme
        for pump_group in pump_groups:
            pump_groups_len.append(len(pump_group))
        i, N = 0, int(dot(array(pump_groups_len), array(cycles)))
        
        self.no_decay = not decay
        # set up the array to store all motional states
        self.all_trial_n = zeros((N + 1, self.modes_w.size, self.M), dtype = int)
        self.all_trial_n[0,:,:] = self.n
        
        tarr, current_t = [0], 0
        for pump_group, no_of_cycle in zip(pump_groups, cycles):
            for cycle_no in range(no_of_cycle):
                for pulse in pump_group:
#                    self.sideband_cool(pulse, negl_entang = negl_entang)
                    self.sideband_cool(pulse, negl_entang = negl_entang)
                    self.all_trial_n[i + 1,:,:] = self.n
#                    print(self.n)
                    print('pulse %i out of %i completed.' % (i, N))
                    
                    current_t = current_t + pulse.t
                    tarr.append(current_t)
                    i = i + 1

        # take average of all trials
        if take_average:
            all_trial_n_ave = average(self.all_trial_n, axis = 2)
            return tarr, all_trial_n_ave
    
    def laser_sideband_hamiltonian(self, sidebands):
        '''
        Find the laser sideband hamiltonian matrix (Without the self.Rb/2 factor)
        
        N: number of ions in the string
        h_st: structure of the hamiltonian = laser_sb_ham_info(N)
        eta: lamb-dicke parameter of a single ion in the trap
        mode_vec : current mode vector
        sideband: sideband vector (e.g. [-1, -1, -1] for red sideband on all modes)
        '''
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
            sttv_motion_stt = no_excited[:,None] * sideband + motion_state
#            print(motion_state)
            sb_stren_this_mode = self.R[which_ion_chg_arr.transpose(1,2,0),mode,sttv_motion_stt[:,None],
                                        sttv_motion_stt[:,None].transpose(1,0,2)]
            total_sb_stren = total_sb_stren * sb_stren_this_mode
        # now compute the actual hamiltonian
        hamiltonian = einsum('ijk,ij->kij', total_sb_stren, h_non_zero)
        return hamiltonian
        