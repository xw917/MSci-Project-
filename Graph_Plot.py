#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 00:20:14 2020

@author: cordelia
"""

import numpy as np
import matplotlib.pyplot as plt

from System.py import *
from Base import *
from Param_Const import *

#%%
"""
Plotting for <n> vs t
"""
N = 100
S = np.zeros(N)
for i in range(N): # run N times
    # print(S)
    s = system(50, 100, 1.5e-6*31)
    S = [sum(i) for i in zip(S,s.excitation())]
ave_S = [j/N for j in S] # average phonon number 
#nprint(ave_S)

plt.plot(np.arange(0, N), ave_S)
plt.xlabel("Unit cycle")
plt.ylabel("<n>")

    