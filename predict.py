#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 09:00:00 2024

@author: juliezhu
"""

import numpy as np
import pickle as pkl
from sklearn import svm
from tqdm import tqdm
from approx import color_list
from tools.kernel_tools import *
from tools.dataset_tools import *
from tools.function_tools import *
from tools.quadrature_tools import *
from tools.visualization_tools import *
from sklearn.metrics import accuracy_score
from tools.dataset_tools import make_dataset, sample_dataset

N_R = 3
compute_err = True
directory = 'real_predict/'
# plt.rcParams['font.family'] = 'Times New Roman'

color_list = {'SR-MC': 'y',
              'SR-SOMC': 'red',
              'SR-OMC': 'lightcoral',
              'SR-OKQ-MC': 'darkgrey',
              'SR-OKQ-SOMC': 'blue',
              'SR-OKQ-OMC': 'cornflowerblue',
              'ORF': 'green', 
              'QMC': 'darkorange', 
              'SSR': 'purple'}

def get_scores(dataset_name, approx_type, N_list, clf, repeated, scores, N_R):
    dataset, _ = make_dataset(dataset_name)
    xtrain, ytrain, xtest, ytest = dataset
    d = xtrain.shape[1]
    n1 = xtrain.shape[0]
    n2 = xtest.shape[0]
    print('start training for {}, {}'.format(dataset_name, approx_type))
    print('training size: {:d}*{:d}, testing size: {:d}*{:d}'.format(n1, d, n2, d))
    sigma = 2*(d)**(1/4)
    print('sigma: {:.2f}'.format(sigma))

    scores[approx_type] = np.zeros((len(N_list), repeated))
    for k, nn in enumerate(tqdm(N_list)):
        if approx_type == 'exact' and k >=1:
            continue
        for i in range(repeated):
            if approx_type == 'exact' and i >=1:
                continue
            precomputed = kernel(xtrain, xtrain, sigma, approx_type, nn, N_R)
            precomputed_test = kernel(xtest, xtrain, sigma, approx_type, nn, N_R)
            clf.fit(precomputed, ytrain)
            if dataset_name in ['Powerplant']:
                predict = clf.predict(precomputed_test)
                scores[approx_type][k][i] = np.mean(np.abs(predict - ytest)**2)
                # scores[approx_type][k][i] = clf.score(precomputed_test, ytest)
            else:
                predict = clf.predict(precomputed_test)
                scores[approx_type][k][i] = accuracy_score(predict, ytest)
    return scores

def main():
    scores = {}
    repeated = 20

    dataset_name, d = 'Powerplant', 4
    # dataset_name, d = 'LETTER', 16

    if dataset_name in ['Powerplant']:
        clf = svm.SVR(kernel='precomputed')
    else: 
        clf = svm.SVC(kernel='precomputed')

    approx_types = ['exact', 'SSR', 'SR-OMC'] #add 'ORF', 'QMC', 'exact' 'SR-Ort', 'SR-OKQ-Ort', , 'SR-SymOrt', 'SR-OKQ-SOMC'
    N_list1 = np.arange(1, 16, 3)  #np.arange(1, 11, 2)
    N_list2 = np.arange(1, 16, 3)
    bins1 = N_R * d * N_list1
    bins2 = ((d+1) * 2 ) * N_list2

    if compute_err:
        for approx_type in tqdm(approx_types):
            if approx_type == 'SSR':
                bins = bins2
                N_list = N_list2
            else:
                bins = bins1
                N_list = N_list1
            scores = get_scores(dataset_name, approx_type, N_list, clf, repeated, scores, N_R)
        with open(directory + 'scores_' + dataset_name, 'wb') as f:
            pkl.dump(scores, f)
            print('saved scores')
    else:
        with open(directory + 'scores_' + dataset_name, 'rb') as f:
            scores = pkl.load(f)
            print('load scores')

    for approx_type in approx_types:
        if approx_type not in ['SSR','SR-OMC','exact']:
            continue
        if approx_type == 'SSR':
            bins = bins2
            N_list = N_list2
        else:
            bins = bins1
            N_list = N_list1
        if approx_type == 'exact':
            print('exact', scores[approx_type])
            plt.loglog(bins, [scores[approx_type][0][0]]*len(bins), color = 'black', linestyle='dashed', linewidth = 2, label = 'Exact')
            continue
        average_error = np.average(scores[approx_type], axis = 1)
        std = np.sqrt(np.var(scores[approx_type], axis = 1))
        # if approx_type == 'QMC':
        #     std[0] = 1600
        print(approx_type, average_error)
        plt.loglog(bins, average_error, color = color_list[approx_type], marker='*', markersize=10, linewidth = 2, label = approx_type)
        # plt.plot(bins, average_error+std,  color = color_list[approx_type], linestyle='dashed', linewidth = 0.5)
        # plt.plot(bins, average_error-std,  color = color_list[approx_type], linestyle='dashed', linewidth = 0.5)
        print(approx_type, std)
        plt.fill_between(bins, average_error-std, average_error+std, color = color_list[approx_type], alpha = .2)

    plt.title('Prediction accuracy on dataset: {}'.format(dataset_name), fontsize=16)
    plt.xlim([max(bins1[0], bins2[0]), min(bins1[-1], bins2[-1])])
    plt.xlabel('Number of features', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.legend(loc='upper right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.grid(which='major', color='#DDDDDD', linewidth=0.8)
    plt.grid(which='minor', color='silver', linestyle=':', linewidth=0.5)
    plt.savefig('plottings/real_predic_' + dataset_name + '.png', format='png', dpi=200)
    plt.show()
    return


if __name__ == "__main__":
    main()