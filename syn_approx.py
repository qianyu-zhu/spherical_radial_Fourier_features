#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 15:36:17 2024

@author: juliezhu
"""
import numpy as np
from tools.function_tools import *
from tools.quadrature_tools import *
from tools.kernel_tools import *
from tools.visualization_tools import *
from tools.dataset_tools import *
from tqdm import tqdm
from tools.SSR_tools import *

color_list = {'SR-MC': 'y',
              'SR-SymOrt': 'red',
              'SR-Ort': 'lightcoral',
              'SR-OKQ-MC': 'darkgrey',
              'SR-OKQ-SymOrt': 'blue',
              'SR-OKQ-Ort': 'cornflowerblue',
              'RFF': 'green',
              'ORF': 'gold', 
              'QMC': 'darkorange', 
              'GQ': 'turquoise',
              'SSR': 'purple'}

def relative_errors(X, Y, approx_types, sigma, start_deg=1, max_deg=2,
               step=1, shift=1, runs=3):
    # need an exact kernel
    K_exact = kernel(X, Y, sigma, 'exact')
    steps = int(np.ceil(max_deg/step)+1)
    errs = {}
    for approx_type in tqdm(approx_types):
        errs[approx_type] = np.zeros((steps, runs))
        if approx_type == 'exact':
            continue
        for j, deg in enumerate(np.arange(start_deg, max_deg+step, step)):
            nn = deg + shift
            for r in range(runs):
                K = kernel(X, Y, sigma, approx_type, nn)
                errs[approx_type][j, r] = kernel_approximation_err(K_exact, K, 'rel f')
    return errs

## number of features vs. relative error in matrix spectral norm
def main():
    approx_types = ['SR-Ort', 'SR-OKQ-SymOrt', 'RFF', 'ORF', 'QMC', 'GQ', 'SSR']

    d = 8
    N_d = 25

    dataset = get_synthetic_Gaussian_dataset(N_d,d)
    name = 'synthetic dataset, d = {:d}, N_d = {:d}'.format(d,N_d)

    # start_deg, max_deg, runs, shift, step, nsamples, delimiter
    params = [0, 7, 10, 0, 1]
    print('start dataset {}'.format(name))
    start_deg, max_deg, runs, shift, step = params
    print('dataset size: ', dataset.shape)

    #### Define the Gaussian kernel (be aware of the bandwidth)
    sigma = 2*d**(1/4) # theoretical gaurantee: >=2*d**(1/4)
    print('sigma: ', sigma)


    print('calculate error')
    errs = relative_errors(dataset, dataset, approx_types, sigma, start_deg,
                                    max_deg, step, shift, runs)
    
    for approx_type in approx_types:
        err = errs[approx_type]
        err_m = np.average(err, axis=1)
        std = np.sqrt(np.var(err, axis=1))
        bins = ((d+1)*2+1) * 2*np.arange(start_deg, max_deg+step, step)
        print(approx_type, err_m)
        plt.loglog(bins, err_m,  color = color_list[approx_type], linewidth = 1)
        plt.fill_between(bins, err_m-std, err_m+std, color = color_list[approx_type], alpha = .1)
        # plt.plot(bins, err_m+std,  color = color_list[approx_type], linestyle='dashed', linewidth = 0.5)
        # plt.plot(bins, err_m-std,  color = color_list[approx_type], linestyle='dashed', linewidth = 0.5)
        plt.scatter(bins, err_m, color = color_list[approx_type], marker='s', s=10, label = approx_type)
    plt.ylabel('relative error (log)')
    plt.xlabel('number of features')
    plt.xlim([bins[0], bins[-1]])
    plt.title('kernel approximation on {}'.format(name))
    plt.legend(loc='upper right')
    plt.show()

    print('Finished!')


if __name__ == "__main__":
    main()