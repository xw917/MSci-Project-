#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 14:52:44 2021

@author: apple
"""

from Param_Const1 import *
import datetime as dt
from itertools import product, combinations

def SchrodingerEqn(H, Psi_0, t, h_factor = 1):
    '''
    Solve the Schrodinger's equation in the matrix mechanics formalism.
    
    H: 2D array. Hamiltonian WITHOUT the factor of ħ
        Hamiltonian must be Hermitian, otherwise the code won't work
    Psi_0: 1D array, must have the same dimension as H. Initial normalised state vector at t = 0.
    t: 1D array. Moments at which state vector is evaluated.
    h_factor: extra factor in the hamiltonian (in our case = free space rabi freq. / 2)
    
    No. of dimension of the Hamiltonian = 3
    
     ---- ----
    |    |    |
    |____|____|  each layer represents one simulation attempt. Each simulation attempt includes all ions
    |    |    |
    |____|____|
    '''
    if not isinstance(t, list) and not isinstance(t, np.ndarray):
        t = array([t])
    # find energy eigenstates and eigenvalues
    eigvals, eigvecs = eigh(H)
    eigvals = eigvals * h_factor
    # decompose the initial state vectors into eigen-components
    initial_coefficients = dot(transpose(eigvecs, axes = (0,2,1)).conj(), Psi_0)
    initial_cond = einsum('ijk,ik->ijk', eigvecs, initial_coefficients)
    # evaluate at the specified times
    time_dep = exp(-1j * einsum('ij,k->ijk', eigvals, t))
    Psi_t = einsum('ijk,ilj->ilk', time_dep, initial_cond)
    '''
    Shape of eigvecs             : (simulation, along an eig vector  , for diff. eig vectors)
    Shape of initial_coefficient : (simulation, for diff. eig vectors)
    Shape of time_dep            : (simulation, for diff. eig vectors, time)
    Shape of the solution Psi_t  : (simulation, along a  stt vector  , time)
    '''
    return Psi_t

def laser_sb_ham_info(N):
    '''
    The structure of the laser sideband hamiltonian (eqn 7.2 of the thesis).
    
    no_excited           : how many ions are excited in each component of the state vector
    hamiltonian_non_zero : which term in the hamiltonian is non-zero
    which_ion_change     : which ion is to be excited in each term in the hamiltonian
    '''
    ion_excite_or_not = which_ion_excite(N)
    
    ion_excite_difference = ion_excite_or_not[:, None] - ion_excite_or_not
    ion_excite_diff_abs   = np.abs(ion_excite_difference)
    hamiltonian_non_zero  = (count_nonzero(ion_excite_diff_abs, axis = 2) == 1)
    
    which_ion_change = (argmax(einsum('ijk,ij->ijk', ion_excite_diff_abs, hamiltonian_non_zero),
                               axis = 2) + 1) * hamiltonian_non_zero
                               
    #  calculate number of ions excited
    no_excited = np.sum(ion_excite_or_not, axis = 1)
    
    return (no_excited, hamiltonian_non_zero, which_ion_change)

def which_ion_excite(N):
    '''
    determine the matrix of which ion is exited (1), which one is not (0)
    '''
    ion_excite_or_not = [[0] * N]
    for i in range(1, N + 1):
        ion_excite_or_not  = ion_excite_or_not + list(get_all_comb_from(N, i))
    return array(ion_excite_or_not)

def get_all_comb_from(size, count):
    '''
    get all combinations from 'size'
    '''
    for positions in combinations(range(size), count):
        p = [0] * size
        for i in positions:
            p[i] = 1
        yield p
    
def force_on_ions_string(pos, factor = mwz24pe_z2e2):
    '''
    This returns an array that is proportional to the total forces acting on each ion in a string.
    
    This function can be used to compute the equilibrium positions of ions in the string
    
    pos: position vector
    factor = 4πϵ0 * m*wz^2 /Z^2E^2 where
        Z = no. of charge on the ion, E = elementary charge
        m = ion mass, wz = trap frequency
    '''
    N = pos.size  # number of ions
    # calculate the coulomb repulsions
    zizj = pos[:, None] - pos
    unit_zizj2 = (zizj + diag(ones(N) * sp.inf)) ** (-2) * np.sign(zizj)
    # sum over columns
    sum_unit_zizj2 = np.sum(unit_zizj2, axis = 1)
    # calculate the trapping force
    kz = factor * pos
#    print('forces:', sum_unit_zizj2 - kz)
    return sum_unit_zizj2 - kz
    
def equil_pos_string(N, factor = mwz24pe_z2e2):
    '''
    Equilibrium positions of ions in a string (along wz)
    
    This function is stable until at least N = 300
    '''
    if N == 1:
        ansatz = (4 * factor) ** (-1/3) / 2
    else:
        ansatz = (4 * factor) ** (-1/3) / 2 / (log(N) / log(10))
    left_pos  = (- divmod(N,2)[0] + divmod(N-1,2)[1] / 2) * ansatz
    right_pos = - left_pos
    trial_pos = linspace(left_pos, right_pos, N)
    # solve for the equilibrium position
    posi = fsolve(force_on_ions_string, trial_pos, args = factor, maxfev = int(1e3))
    return sort(posi)

def string_normal_modes(N, accurate = True, factor = mwz24pe_z2e2):
    '''
    For a string of ions, this function returns:
        
    1. Normal mode frequencies (as a fraction of the single-ion trapping frequency)
    2. Normal coordinates
    
    N: number of ions in the string
    
    accurate: True  : take into account all coulomb interactions
              False : try to model it as beads on a string
    factor: the same factor that occurs in function "force_on_ions_string"
    '''
    # kinetic matrix
    if not accurate:
        K_matrix = diags([1, -2, 1], offsets = [-1, 0, 1], shape = (N, N)).toarray()
    else:
        # first determin the equilibrium positions
        equil_pos = equil_pos_string(N, factor = factor)
        # set up the kinetic matrix
        # calculate the off-diagonal terms
        unit_zizj3 = (np.abs(equil_pos[:, None] - equil_pos) + diag(ones(N) * sp.inf)) ** (-3) * 2 / factor
        # calculate the diagonal terms
        K_diagonal = - np.sum(unit_zizj3, axis = 0) - 1
        K_matrix = unit_zizj3 + diag(K_diagonal)
    # eigenvalues  and eigenvectors
    eigvals, eigvecs = eigh(K_matrix, UPLO = 'U')
    # calculate normal frequencies and normal coordinates
    norm_freq, norm_coor = flip(sp.sqrt(- eigvals), axis = 0), flip(eigvecs, axis = 1)

    return (norm_freq, norm_coor)

def force_on_icc(pos, Wc = wc, Wz = wz):
    '''
    Returns an array that is proportional to the total forces acting on each ion in an
    arbitrary-configured ion coulomb crystal
    
    pos: position vector for all ions:
        pos = (x0, y0, z0, x1, y1, z1, ... , xN, yN, zN)
    '''
    factor = m * Wz**2 * 4 * sp.pi * e0 / Q**2
    we2 = Wc**2 / 4 - Wz**2 / 2
#    we2 = (312.5855e3 * 2 * sp.pi)**2
    if we2 < 0:
        raise Exception('Effective radial frequency (we) in the rotating frame is imaginary.')
    w2_wz2 = array([we2 / Wz**2, we2 / Wz**2 , 1])
    N = int(pos.size / 3)  # number of ions
    r = pos.reshape((N, 3))  # reshape the position array
    ri_rj = r[:, None] - r
    ri_rj3 = 1 / (np.sum(ri_rj ** 2, axis = 2) ** (3/2) + diag(ones(N) * sp.inf))
    # calculate Coulomb force
    F_clmb = einsum('ijk,ij->ik', ri_rj, ri_rj3)
    # calculate the total force
    F = - w2_wz2 * r + F_clmb / factor
    return F.flatten() * 1e6  # this function is not giving the exact value of forces

def equil_pos_icc(N, Wc = wc, Wz = wz, axial_ans = False):
    '''
    Equilibrium positions of the ions as an ion coulomb crystal (GENERAL)
    which minimises GLOBALLY the potential energy
    
    axial_ans: use axial initial position ansatz
    
    THIS FUNCTION IS NOT STABLE AND REQUIRES SIGNIFICANT TESTING AND IMPROVEMENT
    '''
    # calculate effective radial frequency
    # print('Axial frequency:', Wz)
    # print('Cyclotron frequency:', Wc)
    factor = m * Wz**2 * 4 * sp.pi * e0 / Q**2
    # wr = Wc / 2
    # factor = (Q**2 / m * wr**2 * 4 * sp.pi * e0)**(-1/3)
    we = sp.sqrt(Wc**2 / 4 - Wz**2 / 2)
    # print('Effective radial trapping:', we)
#    we = 312.5855e3 * 2 * sp.pi
    # ansatz is an estimate of the order of magnitude of inter-ion separation
    if N != 1:    
        ansatz = (4 * factor) ** (-1/3) / (log(N) / log(10)) / 2
    else:
        ansatz = (4 * factor) ** (-1/3) / 2
    tr, min_pot, succ_fail = 0, sp.inf, zeros(2)
    while tr < int(10 * N):
        # initialisse random positions, scaled by the ansatz
        trial_z = (rand(N,1) - 0.5) * (N * ansatz)
        if axial_ans:
            trial_xy = zeros((N, 2))
        else:
            trial_xy = (rand(N,2) - 0.5) * (N * (Wz / we) * ansatz)
        # combine as the trial positions
        trial_pos = np.concatenate((trial_xy, trial_z), axis = 1)
        # solve for the equilibrium position
        posi, infodict, ier, mesg = fsolve(force_on_icc, trial_pos.flatten(), args = (Wc, Wz),
                      maxfev = int(2e3), xtol = 1.49e-9, full_output = True)
        # calculate potential energy
        pot = V_icc(posi)
        succ_fail = succ_fail + array([ier == 1, ier != 1], dtype = 'int')
        # print('Pot. energy:', pot, 'Success?:', (ier == 1))
        # update the minimum potential energy reached
        if pot < min_pot and ier == 1:
            min_pot, tr = pot, 0
            min_pot_pos, min_trial_pos = posi, trial_pos.flatten()
            # print('===> Updated min. pot. energy:', min_pot)
        else:
            tr = tr + 1
    # print('===> Final min. pot. energy:', min_pot)
    # print('===> Successful: %i. Failed: %i.' % (succ_fail[0], succ_fail[1]))
    return min_pot_pos, min_trial_pos

def icc_normal_modes(pos, Wc = wc, Wz = wz, ret_K = True):
    '''
    Normal modes of an ion coulomb crystal.
    
    pos: equilibrium positions. 1d array: [x0,y0,z0,x1,y1,z1,...]
         Can be determined from function 'equil_pos_icc'.
    
    return: normal frequency and normal coordinates
        if norm_freq > 0: oscillatory motion
           norm_freq = 0: stable rotation of the entire crystal
           norm_freq is imaginary: exponential decay of motion
    
    ret_K: bool, whether to return the K matrix
    '''
    # determine number of ions
    mat_dim = pos.size  # dimension of K_matrix
    N = int(mat_dim / 3)
    posi = pos.reshape((N, 3))  # reshape the positions

    # trap frequencies
    factor = m * Wz**2 * 4 * sp.pi * e0 / Q**2
    we2 = Wc**2 / 4 - Wz**2 / 2
    # calculate r1-r2
    ri_rj = posi[:, None] - posi
    ri_rjri_rj = einsum('lij,lik->lijk', ri_rj, ri_rj)
    
    unit_rirj3 = 1 / (np.sum(ri_rj**2, axis = -1) ** (3/2) + diag(ones(N) * sp.inf))
    unit_rirj5 = unit_rirj3 ** (5/3)
    
    # off-diagonal sub matrices
    ri_rj2_rirj5 = einsum('ijkl,ij->ijkl', ri_rjri_rj, unit_rirj5)
    off_diag_submat = (3*ri_rj2_rirj5 - einsum('ij,kl->ijkl', unit_rirj3, diag(ones(3)))) / factor
    # diagonal sub matrices
    diag_submat = - np.sum(off_diag_submat, axis = 1) - \
                  diag(ones(3) * array([we2/Wz**2, we2/Wz**2, 1]))
    # form the overall K_matrix
    off_diag_submat[arange(N), arange(N), :, :] = diag_submat
    K_matrix = off_diag_submat.transpose(0,2,1,3).reshape(mat_dim, mat_dim)
    
    # find eigenvectors and eigenvalues
    eigvals, eigvecs = eigh(K_matrix, UPLO = 'U')
    # calculate normal frequencies and normal coordinates
    norm_freq, norm_coor = flip(sp.sqrt(- eigvals), axis = 0), flip(eigvecs, axis = 1)
    if not ret_K:
        return (norm_freq, norm_coor)
    else:
        return (norm_freq, norm_coor, K_matrix)
    
def find_equil(N):
    p = np.array([[0, 0, 1, -1], [0, -1, 1, -2, 1, 1], [0, 1, 0, -1, 1, 0, -1, 0]]) # x1, x2, ..., y1, y2, ...
    # [0, -0.7, 0.7, -0.8, 0.4, 0.4]
    def min_potential(p):
        u = p.reshape((2, N))[0]
        v = p.reshape((2, N))[1]
        eq = []
        for i in range(N):
            sum_u, sum_v = 0, 0
            for j in range(N):
                if j != i:
                    sum_u += (u[i] - u[j])/((u[i] - u[j])**2 + (v[i] - v[j])**2)**(3/2)
                    sum_v += (v[i] - v[j])/((u[i] - u[j])**2 + (v[i] - v[j])**2)**(3/2)
            eq.append(u[i] - sum_u)
            eq.append(v[i] - sum_v)
        return eq
    solution = fsolve(min_potential, p[N-2])
    return solution.reshape((2, N)).T, min_potential(solution)
    

def icc_normal_modes_z(N, l = mwr24pe_z2e2):
    position = np.array([[0, 0.62996, 0, -0.62996], 
                         [0, -0.83268, -0.72112, 0.41634, 0.72112, 0.41634], 
                         [0, 0.98549, 0.98549, 0, 0, -0.98549, -0.98549, 0]])
    posi = np.array(position[N-2]).reshape((N, 2))  # reshape the positions in the format of [[x0, y0], [x1, y1], ...]
    K_matrix = np.zeros((N, N)) # initialise the K matrix 
    """ K matrix diagonal elements: m == n """
    for n in range(N):
        ri_rj = 0
        for m in range(N):
            if m != n: ri_rj += 1/((posi[n,0] - posi[m,0])**2 + (posi[n,1] - posi[m,1])**2)**(3/2)
        K_matrix[n, n] = (wz/we)**2 - ri_rj
    """ K matrix off-diagonal elements: m != n"""
    for n in range(N):
        for m in range(N):
            if m != n: K_matrix[n, m] = 1/((posi[n,0] - posi[m,0])**2 + (posi[n,1] - posi[m,1])**2)**(3/2)
    eigvals0, eigvecs = np.linalg.eigh(K_matrix)
    norm_freq = np.sqrt(eigvals0)
    """
    # writing eigenvalues and eigenvectors in accending orders 
    eigvals = sorted(eigvals0)
    order = [eigvals.index(eigvals0[i]) for i in range(N)][::-1]
    eigvecs[:, np.arange(N)] = eigvecs[:, order]
    norm_freq = np.sqrt(eigvals)
    """
    return (norm_freq, eigvecs, K_matrix) 

    
def V_icc(pos, factor = mwz24pe_z2e2):
    '''
    calculate (not exactly) the potential energy of an icc given
    spatial positions (in the rotating frame)
    '''
    # calculate effective radial frequency
    we2 = wc**2 / 4 - wz**2 / 2
    # reshape the position
    N = int(pos.size / 3)
    positions = pos.reshape((N, 3))
    # calculate potential energy due to trapping field
    V_trap = 1/2 * np.sum(dot(positions**2, array([we2/wz**2, we2/wz**2, 1])))
    # calculate Coulomb potential
    ri_rj = np.sum((positions[:, None] - positions) ** 2, axis = 2) + diag(ones(N) * sp.inf)
    V_clmb = 1 / factor / 2 * np.sum(1 / ri_rj)
    return (V_trap + V_clmb) * 1e7

def plot_3d_pos(pos, proj = True):
    '''
    Plot arbitrary 3d positions as 2d projections
    '''
    positions = pos.reshape((int(pos.size / 3), 3))  * 1e6
    
    if proj:
        fig, ax = plt.subplots(ncols = 3, tight_layout = True)
        ax[0].plot(positions[:,0], positions[:,2], '.', markersize = 13)
        ax[1].plot(positions[:,0], positions[:,1], '.', markersize = 13)
        ax[2].plot(positions[:,1], positions[:,2], '.', markersize = 13)
        
        # change the aspect ratio of the axes
        for axe in ax:
            axe.axis('equal')
        
        ax[0].set_xlabel('x / $\mu$m')
        ax[0].set_ylabel('z / $\mu$m')
        ax[1].set_xlabel('x / $\mu$m')
        ax[1].set_ylabel('y / $\mu$m')
        ax[2].set_xlabel('y / $\mu$m')
        ax[2].set_ylabel('z / $\mu$m')
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        ax.scatter(positions[:,0], positions[:,1], positions[:,2])
        
        ax.set_xlabel('x / $\mu$m')
        ax.set_ylabel('y / $\mu$m')
        ax.set_zlabel('z / $\mu$m')
    
    return fig, ax

def plot_nm(pos, eig, freq, kmode = 0):
    '''
    3d plot of normal modes
    
    pos: 1darray = [x0,y0,z0,x1,y1,z1,...] : equilibrium positions
    eig: 2darray with each colmn as the normal mode vectors
    freq: 1darray = normal frequencies
    
    k : int. Start plotting from the k-th mode!
    '''
    A, B = 3, 2
    N = int(pos.size / 3)  # number of ions
    posi = pos.reshape((N, 3)) * 1e6
    
    fig, k, k_fig = plt.figure(tight_layout = True), kmode, 0
    for i in range(A):
        for j in range(B):
            if k >= int(3*N):
                break
            # initialise the subplots
            ax = fig.add_subplot(B, A, k_fig + 1, projection = '3d')
            ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            
            # plot the positions
            ax.scatter(posi[:,0], posi[:,1], posi[:,2])
            
            # plot the vectors
            vecs = eig[:,k].reshape((N, 3))  # reshape eigenvectors
            print('vecs = %s'%(vecs))
            
            limits = amax(array([ax.get_xlim(), ax.get_ylim(), ax.get_zlim()]))
            params = linspace(0, 1, 100)
            
            for ion in range(N):
                eivp = posi[ion,:] + outer(params, vecs[ion,:]) * limits / 2
                ax.plot(eivp[:,0], eivp[:,1], eivp[:,2])
                
            # set frequency as title of subplots
            if freq[k].imag == 0:
                ax.set_title('w = %f wz' % freq[k].real)
            else:
                if freq[k].real == 0:
                    ax.set_title('w = %f i wz' % freq[k].imag)
                else:
                    ax.set_title('w = %f + %f i wz' % (freq[k].real,  freq[k].imag))
            
            X = array(ax.get_xlim())
            Y = array(ax.get_ylim())
            Z = array(ax.get_zlim())
            max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0

            mid_x = (X.max()+X.min()) * 0.5
            mid_y = (Y.max()+Y.min()) * 0.5
            mid_z = (Z.max()+Z.min()) * 0.5
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)   
            
            k = k + 1
            k_fig = k_fig + 1
#%%

def plot_nm_axial_direction(N):
    '''
    3d plot of normal modes
    
    pos: 1darray = [x0,y0,z0,x1,y1,z1,...] : equilibrium positions
    eig: 2darray with each colmn as the normal mode vectors
    freq: 1darray = normal frequencies
    
    k : int. Start plotting from the k-th mode!
    '''
    position = np.array([[0, 0.62996, 0, -0.62996], [0, -0.83268, -0.72112, 0.41634, 0.72112, 0.41634],
               [0, 0.98549, 0.98549, 0, 0, -0.98549, -0.98549, 0]])
    pos = position[N-2]
    
    freq, K, m = icc_normal_modes_z(N)
    title_freq = (freq*we)/wz
    posi_xy = np.array(pos).reshape((N, 2))#  * 1e6   
    # print(posi_xy)
    posi = np.zeros((N, 3))
    # print(posi)
    for i in range(len(posi_xy)):
       posi[i] = np.append(posi_xy[i], 0)
    
    # freq, K  = transform_basis(N = 3, tran_m = np.array([[1, 2, -1], [-1, 1, 1], [1, -1, 1]]))
    # print(posi)
    # print(K)
    
    fig = plt.figure()    
    i = 0
    while i < N:
        if N != 4:
            ax = fig.add_subplot(1, N, i+1, projection = '3d')
        else:
            ax = fig.add_subplot(2, 2, i+1, projection = '3d')
        ax.set_title(r'$%.3f\omega_{z}$'%(title_freq[i]))
        ax.scatter(posi[:,0], posi[:,1], posi[:,2])
        j = 0
        while j < N:
            ax.quiver(posi[j, 0], posi[j, 1], posi[j, 2], posi[j, 0], posi[j, 1], K[j,i], length = 0.05, normalize = True)
            j += 1
        i += 1
        
    """
    fig = plt.figure()
    ax = fig.add_subplot(1, 3, 1, projection = '3d')
    ax.scatter(posi[:,0], posi[:,1], posi[:,2])
    ax.quiver(posi[0, 0], posi[0, 1], posi[0, 2], posi[0, 0], posi[0, 1], K[1,0], length = 0.05, normalize = True)
    ax.quiver(posi[1, 0], posi[1, 1], posi[1, 2], posi[1, 0], posi[1, 1], K[2,0], length = 0.05, normalize = True)
    ax.quiver(posi[2, 0], posi[2, 1], posi[2, 2], posi[2, 0], posi[2, 1], K[0,0], length = 0.05, normalize = True)
    ax.set_title(r'$5.42\omega_{e}$')
    
    ax = fig.add_subplot(1, 3, 2, projection = '3d')
    ax.scatter(posi[:,0], posi[:,1], posi[:,2])
    ax.quiver(posi[0, 0], posi[0, 1], posi[0, 2], posi[0, 0], posi[0, 1], K[1,1], length = 0.05, normalize = True)
    ax.quiver(posi[1, 0], posi[1, 1], posi[1, 2], posi[1, 0], posi[1, 1], K[2,1], length = 0.05, normalize = True)
    ax.quiver(posi[2, 0], posi[2, 1], posi[2, 2], posi[2, 0], posi[2, 1], K[0,1], length = 0.05, normalize = True)
    ax.set_title(r'$5.42\omega_{e}$')
  
    ax = fig.add_subplot(1, 3, 3, projection = '3d')
    ax.scatter(posi[:,0], posi[:,1], posi[:,2])
    ax.quiver(posi[0, 0], posi[0, 1], posi[0, 2], posi[0, 0], posi[0, 1], K[1,2], length = 0.05, normalize = True)
    ax.quiver(posi[1, 0], posi[1, 1], posi[1, 2], posi[1, 0], posi[1, 1], K[2,2], length = 0.05, normalize = True)
    ax.quiver(posi[2, 0], posi[2, 1], posi[2, 2], posi[2, 0], posi[2, 1], K[0,2], length = 0.05, normalize = True)
    ax.set_title(r'$5.51\omega_{e}$')
    """
#%%
"""
params = {
   'axes.labelsize': 18,
   'font.size': 18,
   'font.family': 'sans-serif', # Optionally change the font family to sans-serif
   'font.serif': 'Arial', # Optionally change the font to Arial
   'legend.fontsize': 18,
   'xtick.labelsize': 16,
   'ytick.labelsize': 16, 
   'figure.figsize': [8.8, 1.7*8.8/1.618] # Using the golden ratio and standard column width of a journal
} 
plt.rcParams.update(params)
freq, K, m = icc_normal_modes_z(4)
plot_nm_axial_direction(N = 4, eig = K, freq = freq)
"""