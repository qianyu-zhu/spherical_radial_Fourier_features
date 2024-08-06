#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 12:26:17 2023

@author: ayoubbelhadji
"""

import numpy as np
# import matplotlib.pyplot as plt
# from scipy.sparse import diags
# import math
# from scipy import linalg
from tools.SSR_tools import *
from tools.quadrature_tools import kernel_SR_noOKQ_MC, kernel_SR_noOKQ_Ort, kernel_SR_noOKQ_SymOrt, \
            kernel_RFF, kernel_ORF, kernel_QMC, kernel_GQ, kernel_SR_OKQ_MC, kernel_SR_OKQ_Ort, kernel_SR_OKQ_SymOrt

## Kernels

MAPPING = {'SR-MC': kernel_SR_noOKQ_MC,
           'SR-OMC': kernel_SR_noOKQ_Ort,
           'SR-SOMC': kernel_SR_noOKQ_SymOrt,
           'SR-OKQ-MC': kernel_SR_OKQ_MC,
           'SR-OKQ-OMC': kernel_SR_OKQ_Ort,
           'SR-OKQ-SOMC': kernel_SR_OKQ_SymOrt,
            'RFF': kernel_RFF,
            'ORF': kernel_ORF, 
            'QMC': kernel_QMC, 
            'GQ': kernel_GQ}


def Gaussian_kernel(sigma):
    def kernel_aux(x):
        log_output = -((np.linalg.norm(x))**2)/(2*(sigma*sigma))
        return np.exp(log_output)
    return kernel_aux


def empirical_kernel(w_list,n_list,sigma):
    def empirical_kernel_aux(x):
        tmp_list = [w*np.cos(np.dot(n,x)/sigma) for (n,w) in zip(n_list,w_list)]
        return sum(tmp_list)
    return empirical_kernel_aux
    


def get_kernel_matrix(my_kernel,dataset):
    N,d = dataset.shape
    kernel_matrix = np.zeros((N,N))
    for i in list(range(N)):
        for j in list(range(N)):
            kernel_matrix[i,j] = my_kernel(dataset[i,:]-dataset[j,:])
            
    return kernel_matrix


def delta_kernels_using_dataset(kernel_1,kernel_2,dataset,mode):
    K_1 = get_kernel_matrix(kernel_1,dataset)
    K_2 = get_kernel_matrix(kernel_2,dataset)
    delta = np.linalg.norm(K_1-K_2)
    delta2 = np.linalg.norm(K_1-K_2, 2)
    if mode =='abs f':
        return delta
    elif mode =='rel f':
        return delta/np.linalg.norm(K_1)
    elif mode =='abs 2':
        return delta2
    elif mode =='rel 2':
        return delta2/np.linalg.norm(K_1, 2)


def delta_kernels_using_dataset_ssr(kernel_1, K_2, dataset, mode):
    K_1 = get_kernel_matrix(kernel_1,dataset)
    delta = np.linalg.norm(K_1-K_2, 2)
    if mode =='abs f':
        return delta
    elif mode =='rel f':
        return delta/np.linalg.norm(K_1, 2)
    

def kernel_approximation_err(K_1, K_2, mode):
    delta = np.linalg.norm(K_1-K_2)
    
    if mode =='abs f':
        return delta
    elif mode =='rel f':
        return delta/np.linalg.norm(K_1)

    delta2 = np.linalg.norm(K_1-K_2, 2)
    if mode =='abs 2':
        return delta2
    elif mode =='rel 2':
        return delta2/np.linalg.norm(K_1, 2)
    

## error w.r.t. dataset diameter
def diameter_error(ground_truth, cache_error, mode='max'):
    assert len(ground_truth)== len(cache_error), 'mismatch of dimension'
    ground_truth, cache_error = np.array(ground_truth), np.array(cache_error)
    if mode=='max':
        return np.max(np.abs(ground_truth-cache_error))
    elif mode=='mean':
        return np.average(np.abs(ground_truth-cache_error))
    

def kernel_Gaussian(X, Y, sigma):
    """
    X, Y: size d*N or 1*N, each row is a sample
    sigma: kernel bandwidth
    return: Kernel Gram matrix
    """
    M, N = np.shape(X)[0], np.shape(Y)[0]
    K = np.zeros((M,N))
    for i in range(M):
        for j in range(N):
            K[i,j] = np.linalg.norm(X[i,:] - Y[j,:])
    K = np.exp(-K**2/(2*(sigma**2))) # if wanna use sth other than Gaussian, change this line.
    return K


def kernel(X, Y, sigma, approx_type, nn=None, N_R = 3):
    """
    X, Y: input data, N*d
    nn: feature parameter
    n_list: matrix of nodes, row
    w_list: vector of weights
    approx type: from ['exact', 'RFF', 'ORF', 'QMC', 'GQ', 'SSR', 'SR-OKQ', 'SR']
    """
    _, d = X.shape
    if approx_type == 'exact':
        return kernel_Gaussian(X, Y, sigma)
    elif approx_type == 'SSR':
        return kernel_SSR(X, Y, nn, sigma)
    else:
        mapping_f = MAPPING[approx_type]
        weights, nodes = mapping_f(nn, d, N_R)
        return empirical_kernel(X, Y, weights, nodes, sigma)


def empirical_kernel(X, Y, w_list, n_list, sigma):

    X_dot_n = X@n_list.T
    Y_dot_n = Y@n_list.T
    w_matrix = np.diag(np.concatenate((w_list,w_list),axis=0))
    embeddingX = np.concatenate((np.cos(X_dot_n/sigma), np.sin(X_dot_n/sigma)), axis=1) #size=(N,2D)
    embeddingY = np.concatenate((np.cos(Y_dot_n/sigma), np.sin(Y_dot_n/sigma)), axis=1)
    return embeddingX @ w_matrix @ embeddingY.T  #size=(N,N)
