#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 15:36:17 2024

@author: juliezhu
"""


import numpy as np
import pickle as pkl
from tqdm import tqdm
from tools.function_tools import *
from tools.quadrature_tools import *
from tools.kernel_tools import *
from tools.visualization_tools import *
from tools.dataset_tools import *

color_list = {'SR-MC': 'y',
              'SR-SOMC': 'red',
              'SR-OMC': 'lightcoral',
              'SR-OKQ-MC': 'darkgrey',
              'SR-OKQ-SOMC': 'blue',
              'SR-OKQ-OMC': 'cornflowerblue',
              'RFF': 'green',
              'ORF': 'gold', 
              'QMC': 'darkorange', 
              'GQ': 'turquoise',
              'SSR': 'purple'}


marker_list = {'SR-MC': 'P',
              'SR-SOMC': 'P',
              'SR-OMC': '^',
              'SR-OKQ-MC': 'darkgrey',
              'SR-OKQ-SOMC': 'v',
              'SR-OKQ-OMC': 'cornflowerblue',
              'RFF': '*',
              'ORF': 's', 
              'QMC': 'o', 
              'GQ': 'P',
              'SSR': 'D'}



compute_K = True
directory = 'real_approx/'
#### Change the dataset here
#### For now, we use a synthetic Gausssian dataset
d = 16
N_d = 100
repeated = 25
# dataset = get_synthetic_Gaussian_dataset(N_d,d)

#### Define the Gaussian kernel (be aware of the bandwidth)
sigma = 2*np.sqrt(d)
my_asymptotic_kernel = Gaussian_kernel(sigma)


#### Define the SR quadrature that we will use, along with the corresponding empirical kernels
## The radial quadrature (shared between several SR quadratures):
alpha = d/2-1
N_R = 2
N_S_ = 16
N_S = N_S_*d
N = 2*N_R*N_S

approx_types = ['RFF', 'ORF', 'QMC', 'SSR', 'SR-OMC', 'SR-OKQ-SOMC']
## number of features vs. relative error in matrix spectral norm
n_model = len(approx_types)

D_list = np.array([0.1*i for i in range(2,21,2)])
max_err_list = np.zeros((n_model,len(D_list)))
mean_err_list = np.zeros((n_model,len(D_list)))

if compute_K:
    for i in tqdm(range(len(D_list))):
        cache_error_max = np.zeros((n_model, repeated))
        cache_error_mean = np.zeros((n_model, repeated))
        D = D_list[i]
        # generate random data with norm D
        dataset = get_synthetic_uniform_dataset(N_d,d,D)

        ground_truth = kernel(dataset, dataset, sigma, 'exact', None, None)

        for j in range(len(approx_types)):
            approx_type = approx_types[j]
            for k in range(repeated):
                cache_error_max[j, k] = np.max(np.abs(kernel(dataset, dataset, sigma, approx_type, 2*N_S_, N_R)-ground_truth))
                cache_error_mean[j, k] = np.mean(np.abs(kernel(dataset, dataset, sigma, approx_type, 2*N_S_, N_R)-ground_truth))
            max_err_list[j,i] = np.average(cache_error_max[j, :])
            mean_err_list[j,i] = np.average(cache_error_mean[j, :])

    with open(directory + 'dataset_matrix/diameter_vs_err_d={:d}'.format(d), 'wb') as f:
        pkl.dump(max_err_list, f)
        pkl.dump(mean_err_list, f)
        print('saved error')
else:
    with open(directory + 'dataset_matrix/diameter_vs_err_d={:d}'.format(d), 'rb') as f:
        max_err_list = pkl.load(f)
        mean_err_list = pkl.load(f)
    print('load error done!')



## plot
for i in range(n_model):
    approx_type = approx_types[i]
    result = max_err_list[i,:]
    plt.plot(2*D_list, result, color = color_list[approx_type], linewidth = 0.5, marker = marker_list[approx_type], markersize=5, label = approx_type)
plt.title(r'Maximum error v.s. Dataset diameter, d: {:d}, $\sigma$: {:.2f}, $N_R$: {:d}, $N_S$: {:d}'.format(d, sigma, N_R, N_S), fontsize=12)
# plt.xlim([0,4])
plt.xlabel('Region diameter (D)', fontsize=12)
plt.ylabel('Estimated mean error', fontsize=12)
plt.grid(which='major', color='#DDDDDD', linewidth=0.8)
plt.grid(which='minor', color='silver', linestyle=':', linewidth=0.5)
plt.legend(fontsize=12)
plt.savefig('plottings/diameter_max_err_d={:d}.png'.format(d), format='png', dpi=200)
plt.show()

plt.clf()

for i in range(n_model):
    approx_type = approx_types[i]
    result = mean_err_list[i,:]
    plt.plot(2*D_list, result, color = color_list[approx_type], linewidth = 0.5, marker = marker_list[approx_type], markersize=5, label = approx_type)
plt.title(r'Average error v.s. Dataset diameter, d: {:d}, $\sigma$: {:.2f}, $N_R$: {:d}, $N_S$: {:d}'.format(d, sigma, N_R, N_S), fontsize=12)
# plt.xlim([2*D_list[0],6])
plt.xlabel('Region diameter (D)', fontsize=12)
plt.ylabel('Estimated mean error', fontsize=12)
plt.grid(which='major', color='#DDDDDD', linewidth=0.8)
plt.grid(which='minor', color='silver', linestyle=':', linewidth=0.5)
plt.legend(fontsize=12)
plt.savefig('plottings/diameter_mean_err_d={:d}.png'.format(d), format='png', dpi=200)
plt.show()



