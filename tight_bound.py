#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 05 10:00:00 2024

@author: juliezhu 
"""

import sys
import numpy as np
from tqdm import tqdm
from math import gamma
from copy import deepcopy
import matplotlib.pyplot as plt
from tools.kernel_tools import *
from tools.quadrature_tools import *
from tools.dataset_tools import get_synthetic_uniform_dataset

directory = 'real_approx/'

L = 2

def mc_bound(x_y, d, M_R, M_S, sigma):
    c2 = x_y**2/(2*sigma**2) #choose sigma s.t. c=1
    bound = (L * c2/np.sqrt(gamma(d/2)) * (c2/(2*M_R-1))**(2*M_R-1))\
            + 8/M_S * ((4*M_R+d)/(d-1) * c2)**2 * np.exp(4*(4*M_R+d)*c2/(d-1))
    return 2*bound

def omc_bound(x_y, d, M_R, M_S, sigma):
    c2 = x_y**2/(2*sigma**2)
    bound = (L * c2/np.sqrt(gamma(d/2)) * (c2/(2*M_R-1))**(2*M_R-1))\
            + 2/M_S * ((4*M_R+d)/(d-1) * c2)**4 * np.exp(4*(4*M_R+d)*c2/(d-1))
    return 2*bound



def main(name, d, M_R):

    approx_type = 'SR-OMC'
    nsamples = 100
    repeated = 20
    N_list = np.arange(1, 46, 8)

    print('start dataset {}'.format(name))
    if name == 'Synthetic':
        X = get_synthetic_uniform_dataset(nsamples, d, 1)
        Y = -deepcopy(X)
    else:
        pass
    print('dataset size: ', X.shape)
    d = X.shape[1]
    sigma = np.sqrt(d)
    print('sigma: ', sigma)

    print('start to calculate exact kernel: ...')
    K_exact = kernel(X, Y, sigma, 'exact', None, M_R)

    print('calculate error: ...')
    err = np.zeros((len(N_list), repeated))
    for j, nn in enumerate(N_list):
        for r in range(repeated):
            K = kernel(X, Y, sigma, approx_type, nn, M_R)
            err[j, r] = np.linalg.norm(np.diag(K) - np.diag(K_exact))**2/nsamples # expected squared error
    

    upper_bound = [omc_bound(2, d, M_R, d*N, sigma) for N in N_list]
    err_m = np.average(err, axis=1)
    std = np.sqrt(np.var(err, axis=1))
    bins = M_R * d * N_list
    plt.loglog(bins, upper_bound,  color = 'black', linewidth = 2, label = 'bound')
    plt.loglog(bins, err_m,  color = 'green', marker='*', markersize=10, linewidth = 2, label = approx_type)
    plt.fill_between(bins, err_m-std, err_m+std, color = 'green', alpha = .2)
    plt.ylabel('Expected squared error', fontsize=22)
    plt.xlabel('Number of features', fontsize=22)
    plt.title('Practical error and upper bound', fontsize=22)
    plt.legend(loc='upper right', fontsize=18)
    plt.grid(which='major', color='#DDDDDD', linewidth=0.8)
    plt.grid(which='minor', color='silver', linestyle=':', linewidth=0.5)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.savefig('plottings/upper_bound.png', format='png', dpi=200)
    plt.show()

    print('upper bound: ', upper_bound)
    print('exact error: ', err_m)

    print('Finished!')

if __name__ == "__main__":
    dataset = sys.argv[1]
    d = sys.argv[2]
    M_R = sys.argv[3]
    main(dataset, int(d), int(M_R))