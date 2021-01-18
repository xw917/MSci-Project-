# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 14:46:33 2021

@author: Corde
"""
import numpy as np
from collections import OrderedDict
import pickle 
from Base1 import *
from Param_Const1 import *


all_rabi_strength = OrderedDict()

rab_dim = 500 # rabi frequency dimension = 100 + 1
trap_freq = np.array([wz, np.sqrt(3)*wz]) # choosing the resonant trap frequency for different modes 

for mode in range(2): # two modes are considered where [0]: COM and [1]: breathing 
    rabi_strength = {'resonant_excite': [], 'off_excite':[], 'decay':[]}
    Rd = np.zeros((rab_dim, rab_dim)) # initialise the matrix for decay strength
    w = trap_freq[mode]
    """calculate the matrix for resonant excitation"""
    for sideband in range(4): # rabi strength for carrier, 1 RSB, 2 RSB, 3 RSB
        R_res, R_off = np.zeros((rab_dim, rab_dim)), np.zeros((rab_dim, rab_dim)) # initialise the matrix for excitation strength
        # print('sideband = %s'%(sideband))
        for i in range(rab_dim):
            for j in range(i+1):
                # print('n = %s --> m = %s'%(i,j))
                R_res[i, j] = eff_rabi_freq_single_mode(2, mode, i, j, freq_to_wav(L, -w*sideband), w)
                if mode == 0:
                    R_off[i, j] = eff_rabi_freq_single_mode(2, mode, i, j, freq_to_wav(L, -np.sqrt(3)*wz*sideband), w)
                else:
                    R_off[i, j] = eff_rabi_freq_single_mode(2, mode, i, j, freq_to_wav(L, -wz*sideband), np.sqrt(3)*wz)
                    
        R_res, R_off = R_res + np.tril(R_res, k = -1).T, R_off + np.tril(R_off, k = -1).T
        rabi_strength['resonant_excite'].append(R_res) 
        rabi_strength['off_excite'].append(R_off)
        
        
    """calculate the matrix for decay"""   
    for i in range(rab_dim):
        for j in range(i+1):
            Rd[i, j] = eff_rabi_freq_single_mode(2, mode, i, j, freq_to_wav(LD, 0), w)      
    Rd = Rd + np.tril(Rd, k = -1).T     
    rabi_strength['decay'] = Rd 
    
    all_rabi_strength[mode] = rabi_strength

pickle.dump(all_rabi_strength, open('test','wb'))

#%%
all_rabi_strength = OrderedDict()

sideband = np.arange(0, 4, 1)
d = 500
for s in sideband:
    R = np.zeros((d, d))
    for i in range(d):
        for j in range(i + 1):
            # matrix elements for excitation Rabi frequencies 
            R[i, j] = eff_rabi_freq(i, j, freq_to_wav(L, wz * (-1)*s), wz)
    R += np.tril(R, k = -1).T
    all_rabi_strength[s] = R
    
pickle.dump(all_rabi_strength, open('test2','wb'))


