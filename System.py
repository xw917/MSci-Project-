#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 22:40:11 2020

@author: cordelia
"""

import numpy as np

from Base import *
from Param_Const import *


class system:
    def __init__(self, n0 = 50, N = 100, t_pulse = 1.5e-6*31):
        """
        The state is initialised, assuming the system has already been Doppler
        cooled. 

        Parameters
        ----------
        n0 : int, initial vibrational state. The default is 50.
        N  : int, number of pulses. The default is 100.
        t_pulse : float, duration of a single pulse. The default is 1.5e-6*31.

        """
        self.n0 = n0
        self.N = N
        self.t_pulse = t_pulse
        self.n = n0 # the vibrational state, initial at n0
        self.alln = np.array([50 for i in range(self.N)])
        
    def __repr__(self):
        return "%s initial state = %s, time pulse = %s" % ("Initialisation: ",
                                                           self.n0, 
                                                           self.t_pulse)
    
    def excitation(self):
        """
        The first excitation from state S_1/2 to state D_5/2 following with a
        repumping from state D_5/2 to P_3/2, in this simulation, we assume 
        the transition between state D_5/2 to P_3/ is 100% and the decay rate
        is infinitely large. 

        Returns
        -------
        An array of motional numbers correspond to different cycles. 

        """
        for i in range(self.N):
            if self.n0 != 0: # when motional ground state isn't reached
                # find the effective Rabi frequency for the transition:
                Om_red = EffRabiFreq(self.n0, self.n0-1, Freq2Wav(L, -wz), wz, rb)
                #print(Om_red)
                # find the excitation probability:
                Red_prob = RabiOsci(self.t_pulse, Om_red, rb)
                # Excitation takes place or not:
                eon = np.random.choice(2, 1, p = [1-Red_prob, Red_prob]) 
                # print(eon)
                if eon[0]:
                    self.n0 -= 1 # n --> n - 1
                    # print("current state = %s" % (self.n0))
                self.alln[i] = self.n0
            else: # when motional ground state is reached 
                self.alln[i] = 0      
        return self.alln

                   