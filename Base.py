# -*- coding: utf-8 -*-
"""
Spyder Editor

This file specifies parameters of the Penning Trap
"""
from Param_Const import *
"""
Assumptions:
    1. Monochromatic radiation (infinitely narow bandwidth)
    2. Infinitely short excited state lifetime
    3. No overlapping sidebands
    4. Ions of single species
    5. No electric and magnetic field fluctuations
    6. Ignoring spontaneous decay for Rabi Oscillation
"""

def ExciteOrNot(excit_prob):
    '''
    under a given excitation probability, does a single ion get excited or not?
    '''
    output = sts.bernoulli.rvs(excit_prob, size = 1)
    if output[0] == 1:
        EoN = True
    else:
        EoN = False
    return EoN

def RabiOsci(t, Omega, Omega_0):
    '''
    Excitation probability after addressing the ion with a detuned radiation 
    with time t under radi freq omega
    
    t: time duration of laser dressing (unit: s)
    Omega: Rabi oscillation (angular) frequency (unit: rad/s)
    v: detuning (+ve for blue and -ve for red)
    '''
    A = Omega**1 / Omega_0**1
    W = Omega / 2
#    print(2*W*t)
    return A/2 * (1 - sp.cos(2 * W * t))

def LambDicke(lamb, w):
    k_wave = 2 * sp.pi / lamb
    return sp.sqrt(cs.hbar * k_wave**2 / (2 * m * w))

def EffRabiFreq(n, m, lamb, w, om_0):
    '''
    Calculate effective Rabi Freq for a transition
    
    n, m: initial and final QHO state in the trap
    lamb: radiation wavelength
    w: trap frequency
    om_0: original Rabi frequency
    '''
    eta, n_diff = LambDicke(lamb, w), abs(n - m) 
    # eta is the LD parameter for axial motion
    factor1 = sp.exp(- eta**2 / 2) * (eta**n_diff)
    factor2 = sp.sqrt(FactorialDiv(sp.amin([n, m]), sp.amax([n, m])))
    factor3 = genlaguerre(np.amin([n, m]), n_diff)(eta**2)
    return om_0 * factor1 * factor2 * abs(factor3)

def Boltzmann_state(T, w, size = 100):
    '''
    Generate Boltzmann distributed states
    Typical initial temperature ~ 1mK, initial average quantum number of 
    vibrational mode ~ 20-50
    
    w: trap frequency
    '''
    bolt_fac = cs.hbar * w / (cs.k * T)
    return sts.planck.rvs(bolt_fac, size = size)

def Freq2Wav(lamb, dw):
    '''
    Return shifted wavelength (+ve dw for blue shift, -ve dw for red shift)
    '''
    w0 = cs.c / lamb * 2 * sp.pi
    return cs.c / (w0 + dw) * 2 * sp.pi


n0, N = 50, 100 # starts at n = 50 and does 100 times
t_pulse = 1.5e-6 * 31 # duration of a single pulse
n, alln = n0, [n0]
for i in range(N):
    # print('How far:', i)
    Om_red = EffRabiFreq(n, n-1, Freq2Wav(L, - wz), wz, rb)
    # print('Effective Rabi:', Om_red, 'Real Rabi:', rb)
    Red_prob = RabiOsci(t_pulse, Om_red, rb)
    # print('Excitation Prob:', Red_prob)
    eon = ExciteOrNot(Red_prob)
    if eon:
        n = n - 1
    # print('now:', n)
    alln.append(n)
print(alln)
    
    
