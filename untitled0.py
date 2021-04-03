# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 14:49:05 2021

@author: Corde
"""

# testing
import numpy as np
import numpy as np
from collections import OrderedDict
import pickle 
from Base1 import *
from Param_Const1 import *
from Schrodinger import *

def find_sideband(n, sidebands):
    sb_list = []
    for iteration in n.T:
        sub_list = []
        for sb in sidebands:
            #print(sum([i >= 0 for i in (iteration + sb)]))
            #print(len(n))
            if sum([i >= 0 for i in (iteration + sb)]) == len(n):
                sub_list.append(sb)
        sb_list.append(sub_list)    
    return sb_list

n = np.array([[ 66,   6,   2,   2,  47],
 [  0,  87,  25,  87,  5],
 [139,  39,  12,  54,  30]])

changed_state = np.array([[ 1, -1, 0], [0, 0, 0]])

# print(find_sideband(n, changed_state))

print(np.all(n >= 0))

#%%
import matplotlib.pyplot as plt
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from mpl_toolkits.mplot3d.axes3d import get_test_data
import numpy as np
fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
X, Y, Z = get_test_data(0.05)
C = np.linspace(-5, 5, Z.size).reshape(Z.shape)
scamap = plt.cm.ScalarMappable(cmap='inferno')
fcolors = scamap.to_rgba(C)
ax.plot_surface(X, Y, Z, facecolors=fcolors, cmap='inferno')
fig.colorbar(scamap)
plt.show()
#%%
fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
X = np.arange(0, 10, 1)
Y = np.arange(0, 10, 1)
Z = np.arange(0, 10, 1)


#%%
modes_w, K =  icc_normal_modes_z(N = 3) # normal frequencies and normal coordinates 
G = 3
nm_freq = ones((G, modes_w.size)) * modes_w * we
ftw = freq_to_wav(L, nm_freq * np.array([-1 for k in range(len(modes_w))]))
eta = LambDicke(ftw, nm_freq) * abs(K)

om_list = []
     
                  
#%%

A = array([[[-2, -2, -2],
       [-2, -2, -1],
       [-2, -2,  0],
       [-2, -2,  1],
       [-2, -1, -2],
       [-2, -1, -1]], [[-2, -2, -2],
       [-2, -2, -1]]])

B = np.array([-1, 0, 1])

print([[sum(j) for j in i] for i in A])
# print(A*B)
#%%

# print(final_list)


d = [-6178960.859403918, -3037368.2058141246, -3089480.3185733324, 52112.33501646062, -3089480.540830585, 52112.11275920784, 0.0, 3141592.653589793]
#print(find_smallest_detuning(detune = d, N = 2))

#%%

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = np.random.standard_normal(100)
y = np.random.standard_normal(100)
z = np.random.standard_normal(100)
c = np.random.standard_normal(100)

img = ax.scatter(x, y, z, c=c, cmap=plt.hot())
fig.colorbar(img)
plt.show()

#%%
params = {
   'axes.labelsize': 18,
   'font.size': 18,
   'font.family': 'sans-serif', # Optionally change the font family to sans-serif
   'font.serif': 'Arial', # Optionally change the font to Arial
   'legend.fontsize': 18,
   'xtick.labelsize': 16,
   'ytick.labelsize': 16, 
   'figure.figsize': [8.8, 8.8/1.618] # Using the golden ratio and standard column width of a journal
} 
plt.rcParams.update(params)

R = LoadAnything('3_ions_planar.data')
data_point_1 = np.zeros((500, 2))
data_point_2 = np.zeros((500, 2))
data_point_3 = np.zeros((500, 2))

data_point_11 = np.zeros((500, 2))
data_point_22 = np.zeros((500, 2))
data_point_33 = np.zeros((500, 2))

data_point_111 = np.zeros((500, 2))
data_point_222 = np.zeros((500, 2))
data_point_333 = np.zeros((500, 2))


for i in range(0, 501, 1):
    data_point_1[i-1] = [i, R[0][0][i][i-1] * R[0][1][0][0] * R[0][2][0][0]]
    data_point_2[i-1] = [i, R[1][0][i][i-1] * R[1][1][0][0] * R[1][2][0][0]]
    data_point_3[i-1] = [i, R[2][0][i][i-1] * R[2][1][0][0] * R[2][2][0][0]]
    
    data_point_11[i-1] = [i, R[0][0][0][0] * R[0][1][0][0] * R[0][2][i][i-1]]
    data_point_22[i-1] = [i, R[1][0][0][0] * R[1][1][0][0] * R[1][2][i][i-1]]
    data_point_33[i-1] = [i, R[2][0][0][0] * R[2][1][0][0] * R[2][2][i][i-1]]
    
    data_point_111[i-1] = [i, R[0][0][0][0] * R[0][1][i][i-1] * R[0][2][0][0]]
    data_point_222[i-1] = [i, R[1][0][0][0] * R[1][1][i][i-1] * R[1][2][0][0]]
    data_point_333[i-1] = [i, R[2][0][0][0] * R[2][1][i][i-1] * R[2][2][0][0]]
    
x = data_point_1.T[0]

y1 = data_point_1.T[1]
y2 = data_point_2.T[1]
y3 = data_point_3.T[1]

y11 = data_point_11.T[1]
y22 = data_point_22.T[1]
y33 = data_point_33.T[1]

y111 = data_point_111.T[1]
y222 = data_point_222.T[1]
y333 = data_point_333.T[1]

plt.plot(x, y1, color = 'b', label = 'breathing 1')
plt.plot(x, y2, color = 'b')
plt.plot(x, y3, color = 'b')

plt.plot(x, y11, color = 'r', label = 'COM mode')
plt.plot(x, y22, color = 'r')
plt.plot(x, y33, color = 'r')

plt.plot(x, y111, color = 'g', label = 'breathing 2')
plt.plot(x, y222, color = 'g')
plt.plot(x, y333, color = 'g')

plt.xlabel('state n')
plt.ylabel('Rabi strength')
plt.legend()

#%%
R = LoadAnything('3_ions_planar.data')
data_point_1 = np.zeros((500, 2))
data_point_2 = np.zeros((500, 2))
data_point_3 = np.zeros((500, 2))

data_point_11 = np.zeros((500, 2))
data_point_22 = np.zeros((500, 2))
data_point_33 = np.zeros((500, 2))

data_point_111 = np.zeros((500, 2))
data_point_222 = np.zeros((500, 2))
data_point_333 = np.zeros((500, 2))


for i in range(0, 501, 1):
    data_point_1[i-1] = [i, R[0][0][i][i] * R[0][1][150][150] * R[0][2][150][149]]
    data_point_2[i-1] = [i, R[1][0][i][i] * R[1][1][150][150] * R[1][2][150][149]]
    data_point_3[i-1] = [i, R[2][0][i][i] * R[2][1][150][150] * R[2][2][150][149]]
    
    data_point_11[i-1] = [i, R[0][0][40][40] * R[0][1][40][40] * R[0][2][i][i-1]]
    data_point_22[i-1] = [i, R[1][0][40][40] * R[1][1][40][40] * R[1][2][i][i-1]]
    data_point_33[i-1] = [i, R[2][0][40][40] * R[2][1][40][40] * R[2][2][i][i-1]]
    
    data_point_111[i-1] = [i, R[0][0][40][40] * R[0][1][i][i-1] * R[0][2][40][40]]
    data_point_222[i-1] = [i, R[1][0][40][40] * R[1][1][i][i-1] * R[1][2][40][40]]
    data_point_333[i-1] = [i, R[2][0][40][40] * R[2][1][i][i-1] * R[2][2][40][40]]
    
x = data_point_1.T[0]

y1 = data_point_1.T[1]
y2 = data_point_2.T[1]
y3 = data_point_3.T[1]

y11 = data_point_11.T[1]
y22 = data_point_22.T[1]
y33 = data_point_33.T[1]

y111 = data_point_111.T[1]
y222 = data_point_222.T[1]
y333 = data_point_333.T[1]

plt.plot(x, y1, color = 'b', label = 'breathing 1')
plt.plot(x, y2, color = 'b')
plt.plot(x, y3, color = 'b')

plt.plot(x, y11, color = 'r', label = 'COM mode')
plt.plot(x, y22, color = 'r')
plt.plot(x, y33, color = 'r')

plt.plot(x, y111, color = 'g', label = 'breathing 2')
plt.plot(x, y222, color = 'g')
plt.plot(x, y333, color = 'g')

plt.xlabel('state n')
plt.ylabel('Rabi strength')
plt.legend()