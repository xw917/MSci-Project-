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
        N  : int, number of iterations. The default is 100.
        t_pulse : float, duration of a single pulse. The default is 1.5e-6*31.

        """
        self.n0 = n0
        self.N = N
        self.t_pulse = t_pulse
        self.n = n0 # the vibrational state, initial at n0
        self.alln = np.zeros(self.N)
        
    def __repr__(self):
        return "%s initial state = %s, time pulse = %s" % ("Initialisation: ",
                                                           self.n0, 
                                                           self.t_pulse)
    
    def excitation(self):
        """
        The first excitation from state S_1/2 to state D_5/2 following with a
        repumping from state D_5/2 to P_3/2

        Returns
        -------
        None.

        """
        # the first excitation 
        for i in range(self.N):
            # find the effective Rabi frequency for the transition
            Om_red = EffRabiFreq(self.n0, self.n0-1, Freq2Wav(L, -wz), wz, rb)
            # find the excitation probability 
            Red_prob = RabiOsci(self.t_pulse, Om_red, rb)
            eon = np.random.choice(2, self.N, [1-Red_prob, Red_prob]) 
            for i in range(self.N):
                if eon[i] == 1:
                    self.n0 -= 1
                    self.alln[i] = self.n0
            return self.alln
            