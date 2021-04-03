# -*- coding: utf-8 -*-
"""
Spyder Editor
This file specifies parameters of the Penning Trap
"""
from Param_Const1 import *
from Schrodinger import *

"""
Assumptions:
    1. Monochromatic radiation (infinite
    ly narow bandwidth)
    2. Infinitely short excited state lifetime
    3. No overlapping sidebands
    4. Ions of single species
    5. No electric and magnetic field fluctuations
    6. Ignoring spontaneous decay for Rabi Oscillation
"""

def rabi_osci(t, Omega, A):
    '''
    Excitation probability after addressing the ion with a detuned radiation 
    with time t under radi freq omega
    
    t: time duration of laser dressing (unit: s)
    Omega: Rabi oscillation (angular) frequency (unit: rad/s)
    v: detuning (+ve for blue and -ve for red)
    '''
    W = Omega / 2
    p = A/2 * (1 - sp.cos(2 * W * t))
    return p

def excitation_prob(n, omega, t):
    """
    calculate the probabilities for different excitations (may need to 
    modify for case of more than 2 ions)

    Parameters
    ----------
    state : int, current state
    rabi  : float, rabi frequency 
    t     : float, time
    
    Returns
    -------
    probabilities of state |e, e>, |e, g>, and |g, g>

    """
    """
    if n == 0:
        ee = 0
        eg = 0
        gg = 1
    else:
        """
    ee = ((np.sqrt(n*(n-1))/(2*n-1))*(1-np.cos(np.sqrt((2*n-1)/2)*omega*t)))**2
    eg = (np.sqrt(n/(2*(2*n-1)))*np.sin(np.sqrt((2*n-1)/2)*omega*t))**2
    gg = (1-(n/(2*n-1))*(1-np.cos(np.sqrt((2*n-1)/2)*omega*t)))**2
    
    return ee, 2*eg, gg

def LambDicke(lamb, w):
    """
    L-D parameter for a single ion of mass m in a trap of frequency w, 
    interacting with laser light of wave vector k.
    """
    k_wave = 2 * sp.pi / lamb
    return np.sqrt(cs.hbar * k_wave**2 / (2 * m * w))

def LambDicke_planar(N):
    modes_w, K, m =  icc_normal_modes_z(N)
    nm_freq = ones((N, modes_w.size)) * modes_w * we
    ftw = freq_to_wav(L, nm_freq * np.array([(-1) for _ in range(N)]))
    eta = LambDicke(ftw, nm_freq) * abs(K)
    return eta

def freq_to_wav(lamb, dw):
    '''
    Return shifted wavelength (+ve dw for blue shift, -ve dw for red shift)
    '''
    w0 = cs.c / lamb * 2 * sp.pi
    return cs.c / (w0 + dw) * 2 * sp.pi

def eff_rabi_freq(n, m, lamb, w, normal_coor = 1):
    '''
    Calculate the matrix element ⟨n|e^i*η(a+a†)|m⟩
    
    n, m: initial and final QHO state in the trap
    lamb: radiation wavelength
    w: trap frequency
    '''
    eta, n_diff = LambDicke(lamb, w) * normal_coor, abs(n - m)
#    print('frequencies:', w)
#    print('ld-parameters:', eta)
    # eta is the LD parameter for axial motion
    factor1 = sp.exp(- eta**2 / 2) * (eta**n_diff)
    factor2 = np.sqrt(FactorialDiv(sp.amin([n, m], axis = 0), sp.amax([n, m], axis = 0)))
    factor3 = eval_genlaguerre(np.amin([n, m], axis = 0), n_diff, eta**2)
#    print(factor1, factor2, factor3)
    return factor1 * factor2 * np.abs(factor3)

def simple_eff_rabi(n, m, eta):
    n_diff = abs(n - m)
    factor1 = sp.exp(- eta**2 / 2) * (eta**n_diff)
    factor2 = np.sqrt(FactorialDiv(sp.amin([n, m], axis = 0), sp.amax([n, m], axis = 0)))
    factor3 = eval_genlaguerre(np.amin([n, m], axis = 0), n_diff, eta**2)
#    print(factor1, factor2, factor3)
    return factor1 * factor2 * np.abs(factor3)
    

def eff_rabi_freq2(n1, m1, n2, m2, lamb, w):
    """
    Calculate the matrix element for two-mode (COM mode and breathing mode for 
    string)

    Parameters
    ----------
    n1, m1 : int. Initial and final states for COM mode.
    n2, m2 : int. Initial and final states for breathing mode.
    lamb : float, radiation wavelength
    w : float, trap frequency
    """
    # L-D parameter for COM mode of string
    com_eta, n_diff_com = LambDicke(lamb, w)/np.sqrt(2), abs(n1 - m1)
    # L-D parameter for breathing mode of string 
    b_eta, n_diff_b = LambDicke(lamb, w)/np.sqrt(2*np.sqrt(3)), abs(n2 - m2) 
    
    factor1_com = np.exp(-com_eta**2 / 2) * (com_eta**n_diff_com)
    factor2_com = np.sqrt(FactorialDiv(np.amin([n1, m1]), np.amax([n1, m1])))
    factor3_com = eval_genlaguerre(np.amin([n1, m1]), n_diff_com, com_eta**2)
    
    factor1_b = np.exp(-b_eta**2 / 2) * (b_eta**n_diff_b)
    factor2_b = np.sqrt(FactorialDiv(np.amin([n2, m2]), np.amax([n2, m2])))
    factor3_b = eval_genlaguerre(np.amin([n2, m2]), n_diff_b, b_eta**2)
    
    return (factor1_com*factor2_com*abs(factor3_com)) * (factor1_b*factor2_b*abs(factor3_b))

def eff_rabi_freq_single_mode(Ni, mode, n, m, lamb, w):
    '''
    Calculate the matrix element ⟨n|e^i*η(a+a†)|m⟩ for single mode. 
    
    Parameters
    ----------
    Ni: int, number of ions in the trap
    mode: int, 0 --> com mode
               1 --> breathing mode 
    n, m: initial and final QHO state in the trap
    lamb: radiation wavelength
    w: trap frequency
    
    Return
    ------
    Rabi strength of a particular mode at a specific transition
    '''
    # L-D parameters for different numbers of ion and different mode 
    LD_param = [[LambDicke(lamb, wz)], # single ion 
                [LambDicke(lamb, wz)/np.sqrt(2), LambDicke(lamb, wz)/np.sqrt(2*np.sqrt(3))]] # two ions: com, breathing 
    eta, n_diff = LD_param[Ni-1][mode], abs(n - m) 
    # eta is the LD parameter for axial motion
    factor1 = sp.exp(- eta**2 / 2) * (eta**n_diff)
    factor2 = np.sqrt(FactorialDiv(sp.amin([n, m]), sp.amax([n, m])))
    factor3 = eval_genlaguerre(np.amin([n, m]), n_diff, eta**2)
    return factor1 * factor2 * abs(factor3)
        
def nave_to_T(nave, w):
    '''
    Calculate trap temperature from average n of a Boltzmann distribution
    
    nave : float, average value of motional state number n
    w    : float, trap (angular) frequency
    '''
    return cs.hbar * w / cs.k / sp.log((1 + nave) / nave)

def T_to_nave(T, w):
    '''
    Calculate the (theoretical rather than statistical) average n of a Boltzmann distribution
    
    T : float, trap temperature
    w : float, trap frequency
    '''
    gamm = cs.hbar * w / (cs.k * T)
    return sp.exp(- gamm) / (1 - sp.exp(-gamm))

def Boltzmann_state(T, w, size = 100):
    '''
    Generate Boltzmann distributed states
    Typical initial temperature ~ 1mK, initial average quantum number of 
    vibrational mode ~ 20-50
    
    w: trap frequency
    '''
    bolt_fac = cs.hbar * w / (cs.k * T)
    return sts.planck.rvs(bolt_fac, size = size)
    # return sts.boltzmann(bolt_fac, size = size)

def BED(T, w):
    '''
    Bose-Einstein distribution. Returns the expected number of phonons in a given phonon state
    '''
    return 1 / (exp(cs.hbar * w / cs.k / T) - 1)

def PlotRedSb(n0, sb, l):
    """
    Plot the relative strengths of the carrier, first and second sidebands at 
    and below quantum number n0
    
    Parameters
    ----------
    n0 : int, maximum quantum number shown in the plot
    sb : int, number of sidebands shown in the plot
    l  : float, wavelength of sideband transitions 
    """
    Om_list = [[] for _ in range(sb+1)]
    # Om_list[0] = carrier, Om_list[1] = red 1, Om_list[2] = red 2, ...
    
    for i in range(0, n0+1):
        Om_list[0].append(eff_rabi_freq(i, i, l, wz))
        for j in range(1, sb+1):
            if i > j:
                Om_list[j].append(eff_rabi_freq(i, i - j, freq_to_wav(l, - wz), wz))
    # return Om_list
    plt.plot(Om_list[0], '--', color = 'black', label = 'carrier')
    for Om in range(1, len(Om_list)):
        plt.plot(Om_list[Om], label = '%s RSB'%(Om))
    plt.legend()
    plt.xlabel('Motional State Number (n)')
    plt.ylabel('Normalised Rabi Strength')
    plt.title('Transition wavelength = %s nm'%(int(l*1e9)))
    
def StoreAnything(filename, data_to_store):
    '''
    Store anything under the filename
    '''
    filee = open(filename, 'wb')
    pickle.dump(data_to_store, filee)
    filee.close()
    print('file closed')
    
def LoadAnything(filename):
    return pickle.load(open(filename, 'rb'))

def index_1st_greater_than(array, value):
    '''
    Find the index in an array of the first element larger than a value
    '''
    return argmax(array >= value)

def index_lst_greater_than(array, value):
    '''
    Find the index in an array of the last element larger than a value
    '''
    return array.size - (argmax(flip(array, axis = 0) >= value) + 1)

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

def division(n, d):
    return n / d if d else 0

def subtract(a, b):
    return a - b if a else 1

def amplitude(a, b):
    return division(a**2, (a**2 + b**2))

def frequency(a, b):
    return np.sqrt(a**2 + b**2)/2

def find_smallest_detuning_index(detune, N):
    list1 = [ele**2 for ele in detune]
    # print(list1)
    sorted_list1 = sort(list1)
    # print(sorted_list1)
    final_list = [list1.index(ele) for ele in sorted_list1[:N+1]] 
    # print(final_list)
    return [list1.index(ele) for ele in sorted_list1[:N+1]] 

def detuning_func(sidebands, frequency, laser_sideband):
    return sum(np.array(sidebands) * frequency) + laser_sideband

def transform_basis(N, tran_m):
    modes_w, K, matrix =  icc_normal_modes_z(N = N) # normal frequencies and normal coordinates
    eigvals0, eigvecs = np.linalg.eig(np.matmul(np.linalg.inv(tran_m), np.matmul(matrix, tran_m)))
    eigvals = sorted(eigvals0)
    order = [eigvals.index(eigvals0[i]) for i in range(N)][::-1]
    eigvecs[:, np.arange(N)] = eigvecs[:, order]
    norm_freq = np.sqrt(eigvals)
    return (norm_freq, eigvecs)

