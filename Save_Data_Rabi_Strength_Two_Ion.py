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
from Schrodinger import *
#%%

all_rabi_strength = OrderedDict()

rab_dim = 500 # rabi frequency dimension = 100 + 1
trap_freq = np.array([wz, np.sqrt(3)*wz]) # choosing the resonant trap frequency for different modes 

for mode in range(2): # two modes are considered where [0]: COM and [1]: breathing 
    rabi_strength = {'resonant_excite': [], 'decay':[]}
    Rd = np.zeros((rab_dim, rab_dim)) # initialise the matrix for decay strength
    w = trap_freq[mode]
    """calculate the matrix for resonant excitation"""
    for sideband in range(4): # rabi strength for carrier, 1 RSB, 2 RSB, 3 RSB
        R_res = np.zeros((rab_dim, rab_dim)) # initialise the matrix for excitation strength
        # print('sideband = %s'%(sideband))
        for i in range(rab_dim):
            for j in range(i+1):
                # print('n = %s --> m = %s'%(i,j))
                R_res[i, j] = eff_rabi_freq_single_mode(2, mode, i, j, freq_to_wav(L, wz * -sideband), wz)
                    
        R_res = R_res + np.tril(R_res, k = -1).T
        rabi_strength['resonant_excite'].append(R_res)         
        
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

#%%
"""save data for 3 ions"""
maxn0 = 1000
G = 3 # number of ions
rab_dim = int(maxn0 + 10)  # dimension of the 0th and 1st axes

modes_w, K, m =  icc_normal_modes_z(N = 3) # normal frequencies and normal coordinates 
 
R = np.zeros((G, modes_w.size, rab_dim, rab_dim))
Rd = np.zeros((G, modes_w.size, rab_dim, rab_dim))
for i in range(rab_dim):
    for j in range(i + 1):
        # matrix elements for excitation Rabi frequencies 
        nm_freq = ones((G, modes_w.size)) * modes_w * we
        sideband = i-j
        """
        R[:, :, i, j] = eff_rabi_freq(i, j, freq_to_wav(L, nm_freq * np.array([-sideband for k in range(len(modes_w))])),
                                     nm_freq, normal_coor = np.abs(K))
        Rd[:, :, i, j] = eff_rabi_freq(i, j, freq_to_wav(LD, 0), nm_freq, normal_coor = np.abs(K))
        """
        R[:, :, i, j] = eff_rabi_freq(i, j, freq_to_wav(L, nm_freq * np.array([-1, -1, -1])),
                                     nm_freq, normal_coor = np.abs(K))
        Rd[:, :, i, j] = eff_rabi_freq(i, j, freq_to_wav(LD, 0), nm_freq, normal_coor = np.abs(K))

R  = R + np.tril(R, k = -1).transpose(0,1,3,2)
Rd = Rd + np.tril(Rd, k = -1).transpose(0,1,3,2)

pickle.dump(R, open('3_ions_planar.data', 'wb'))
pickle.dump(Rd, open('3_ions_planar_decay.data', 'wb'))

#%%
"""save data for 3 ions (change basis)"""
maxn0 = 1000
G = 3 # number of ions
rab_dim = int(maxn0 + 10)  # dimension of the 0th and 1st axes

modes_w, K_init, m =  icc_normal_modes_z(N = 3) # normal frequencies and normal coordinates
# modes_w, K = transform_basis(N = 3, tran_m = np.array([[-1, 2, 1], [1, 2, -1], [1, 1, 2]]))
# print(modes_w, K)
print(K_init)
K = []
# print((K_init.T[0]+K_init.T[1])/np.sqrt(2))
# print((K_init.T[0]-K_init.T[1])/np.sqrt(2))
K.append(K_init.T[0]*(3/2) - K_init.T[1]/2)
K.append((K_init.T[0]/2 + K_init.T[1]/2))
K.append(K_init.T[2])
# print(np.array(K).T)
K = np.array(K).T
print(K)
#%%
# K_init[:, 0] = K_init[:, 1]
# K = K_init

R = np.zeros((G, modes_w.size, rab_dim, rab_dim))
Rd = np.zeros((G, modes_w.size, rab_dim, rab_dim))
for i in range(rab_dim):
    for j in range(i + 1):
        # matrix elements for excitation Rabi frequencies 
        nm_freq = ones((G, modes_w.size)) * modes_w * we
        sideband = i-j

        R[:, :, i, j] = eff_rabi_freq(i, j, freq_to_wav(L, nm_freq * np.array([-1, -1, -1])),
                                     nm_freq, normal_coor = np.abs(K))
        Rd[:, :, i, j] = eff_rabi_freq(i, j, freq_to_wav(LD, 0), nm_freq, normal_coor = np.abs(K))

R  = R + np.tril(R, k = -1).transpose(0,1,3,2)
Rd = Rd + np.tril(Rd, k = -1).transpose(0,1,3,2)

pickle.dump(R, open('3_ions_planar_change_basis_1.data', 'wb'))
pickle.dump(Rd, open('3_ions_planar_decay_change_basis_1.data', 'wb'))


#%%
"""save data for 4 ions"""
maxn0 = 1000
G = 4 # number of ions
rab_dim = int(maxn0 + 10)  # dimension of the 0th and 1st axes

modes_w, K, m =  icc_normal_modes_z(N = 4) # normal frequencies and normal coordinates 
for i in range(len(K)):
    for j in range(len(K[i])):
        if abs(K[i][j]) < 1e-14:
            K[i][j] = 0

R = np.zeros((G, modes_w.size, rab_dim, rab_dim))
Rd = np.zeros((G, modes_w.size, rab_dim, rab_dim))
for i in range(rab_dim):
    for j in range(i + 1):
        # matrix elements for excitation Rabi frequencies 
        nm_freq = ones((G, modes_w.size)) * modes_w * we
        sideband = i-j
        R[:, :, i, j] = eff_rabi_freq(i, j, freq_to_wav(L, nm_freq * np.array([-1, -1, -1, -1])),
                                     nm_freq, normal_coor = np.abs(K))
        Rd[:, :, i, j] = eff_rabi_freq(i, j, freq_to_wav(LD, 0), nm_freq, normal_coor = np.abs(K))
R  = R + np.tril(R, k = -1).transpose(0,1,3,2)
Rd = Rd + np.tril(Rd, k = -1).transpose(0,1,3,2)

pickle.dump(R, open('4_ions_planar.data', 'wb'))
pickle.dump(Rd, open('4_ions_planar_decay.data', 'wb'))


#%%
""" save for 3 ions possible sidebands """
sidebands = [] # store all possible excitation sidebands
for i in range(-1, 2, 1):
    for j in range(-1, 2, 1):
        for k in range(-1, 2, 1):
            sidebands.append([i, j, k])
# print(sidebands)
# self.sidebands.append([0, 0, 0]) # for ion not excite
# sidebands = np.array(sidebands)
pickle.dump(sidebands, open('3_ions_planar_sidebands.data', 'wb'))
#%%
""" save for 4 ions possible sidebands """
sidebands = [] # store all possible excitation sidebands
for i in range(-1, 2, 1):
    for j in range(-1, 2, 1):
        for k in range(-1, 2, 1):
            for l in range(-1, 2, 1):
                sidebands.append([i, j, k, l])
# self.sidebands.append([0, 0, 0]) # for ion not excite
pickle.dump(sidebands, open('4_ions_planar_sidebands.data', 'wb'))

#%%
"""save data for 4 ions (small)"""
maxn0 = 0
G = 4 # number of ions
rab_dim = int(maxn0 + 3)  # dimension of the 0th and 1st axes

modes_w, K, m =  icc_normal_modes_z(N = 4) # normal frequencies and normal coordinates 
for i in range(len(K)):
    for j in range(len(K[i])):
        if abs(K[i][j]) < 1e-14:
            K[i][j] = 0

Rs = np.zeros((G, modes_w.size, rab_dim, rab_dim))
Rds = np.zeros((G, modes_w.size, rab_dim, rab_dim))
for i in range(rab_dim):
    for j in range(i + 1):
        # matrix elements for excitation Rabi frequencies 
        nm_freq = ones((G, modes_w.size)) * modes_w * we
        sideband = i-j
        Rs[:, :, i, j] = eff_rabi_freq(i, j, freq_to_wav(L, nm_freq * np.array([-1, -1, -1, -1])),
                                     nm_freq, normal_coor = np.abs(K))
        Rds[:, :, i, j] = eff_rabi_freq(i, j, freq_to_wav(LD, 0), nm_freq, normal_coor = np.abs(K))
Rs  = Rs + np.tril(Rs, k = -1).transpose(0,1,3,2)
Rds = Rds + np.tril(Rds, k = -1).transpose(0,1,3,2)

pickle.dump(Rs, open('4_ions_planar_small.data', 'wb'))
pickle.dump(Rds, open('4_ions_planar_decay_small.data', 'wb'))

#%%
"""save data for 3 ions (small)"""
maxn0 = 0
G = 3 # number of ions
rab_dim = int(maxn0 + 3)  # dimension of the 0th and 1st axes

modes_w, K, m =  icc_normal_modes_z(N = 3) # normal frequencies and normal coordinates 
for i in range(len(K)):
    for j in range(len(K[i])):
        if abs(K[i][j]) < 1e-14:
            K[i][j] = 0

Rs = np.zeros((G, modes_w.size, rab_dim, rab_dim))
Rds = np.zeros((G, modes_w.size, rab_dim, rab_dim))
for i in range(rab_dim):
    for j in range(i + 1):
        # matrix elements for excitation Rabi frequencies 
        nm_freq = ones((G, modes_w.size)) * modes_w * we
        sideband = i-j
        Rs[:, :, i, j] = eff_rabi_freq(i, j, freq_to_wav(L, nm_freq * np.array([-1, -1, -1])),
                                     nm_freq, normal_coor = np.abs(K))
        Rds[:, :, i, j] = eff_rabi_freq(i, j, freq_to_wav(LD, 0), nm_freq, normal_coor = np.abs(K))
Rs  = Rs + np.tril(Rs, k = -1).transpose(0,1,3,2)
Rds = Rds + np.tril(Rds, k = -1).transpose(0,1,3,2)

pickle.dump(Rs, open('3_ions_planar_small.data', 'wb'))
pickle.dump(Rds, open('3_ions_planar_decay_small.data', 'wb'))

    