#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 12:26:17 2023

@author: ayoubbelhadji
"""
import numpy as np
from mlxtend.data import loadlocal_mnist


PARAMS = {'Powerplant': [1, 5, 500, 0, 1, 550, 8500],
          'LETTER': [1, 5, 500, 0, 1, 550, 10000],
          'USPS': [1, 5, 500, 0, 1, 550, 4500],
          'MNIST': [1, 5, 100, 0, 1, 550, 50000],
          'CIFAR100': [1, 5, 50, 0, 1, 550, 50000],
          'LEUKEMIA': [1, 5, 10, 0, 1, 10, 38]
          }
DIMS = [4, 16, 256, 784, 3072, 7129]


## Function manipulation tools

## synthetic dataset

def get_synthetic_Gaussian_dataset(N,d):
    
    mean = np.zeros((d))
    cov = np.eye((d))
    
    dataset = np.random.multivariate_normal(mean, cov, N)
    dataset /= dataset.max()
    return dataset



def get_synthetic_uniform_dataset(N,d,D):

    mean = np.zeros((d))
    cov = np.eye((d))

    dataset = np.random.multivariate_normal(mean, cov, N)
    norm = np.linalg.norm(dataset, axis=1)
    uniform = np.diag(D/norm)@dataset
    return uniform



## real-world datset

def pack_dataset(data, labels, n):
    xtest = data[n:]
    ytest = labels[n:]
    xtrain = data[:n]
    ytrain = labels[:n]
    d = xtrain.shape[1]
    return [xtrain, ytrain, xtest, ytest], d


def load_dataset(name, n, path):
    path = path + '%s/' % name
    data = np.load(path+'data.npy')
    labels = np.load(path+'labels.npy')
    return pack_dataset(data, labels, n)


def sample_dataset(nsamples, dataset, random = False):
    xtrain = dataset[0]
    if random:
        N = xtrain.shape[0]
        idx = np.arange(N)
        np.random.shuffle(idx)
        pick = np.random.choice(idx, nsamples, False)
        sampled = xtrain[pick, :]
        return sampled
    else:
        sampled = xtrain[:nsamples, :]
        return sampled



def scale_data(xtrain, xtest):
    xtrain = xtrain.astype('float')
    xtest = xtest.astype('float')
    xtrain /= xtrain.max()
    xtest /= xtest.max()
    return xtrain, xtest

def load_mnist(temp_dir='./datasets/MNIST/'):
    xtrain, ytrain = loadlocal_mnist(
            images_path=temp_dir+'train-images.idx3-ubyte', 
            labels_path=temp_dir+'train-labels.idx1-ubyte')
    xtest, ytest = loadlocal_mnist(
            images_path=temp_dir+'t10k-images.idx3-ubyte', 
            labels_path=temp_dir+'t10k-labels.idx1-ubyte')
    print(xtrain.shape, ytrain.shape, xtest.shape, ytest.shape)
    return (xtrain, ytrain, xtest, ytest)

def make_dataset(dataset_name, params=None, path='./datasets/'):
    if params is None:
        params = PARAMS[dataset_name]
    start_deg, max_deg, runs, shift, step, NSAMPLES, delimiter = params

    if dataset_name == 'MNIST':
        xtrain, ytrain, xtest, ytest = load_mnist()
        d = xtrain.shape[1]
    else:
        dataset, d = load_dataset(dataset_name, delimiter, path)
        xtrain, ytrain, xtest, ytest = dataset

    xtrain, xtest = scale_data(xtrain, xtest)
    dataset = [xtrain, ytrain, xtest, ytest]
    return dataset, params

