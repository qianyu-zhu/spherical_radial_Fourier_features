#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 12:26:17 2023

@author: ayoubbelhadji
"""

import numpy as np
import matplotlib.pyplot as plt

## Visualization tools



def plot_this_2D_function(function,a,b,M,filename):
    # Generate x and y values
    x = np.linspace(a, b, M)
    y = np.linspace(a, b, M)
    X, Y = np.meshgrid(x, y)  # Create a grid of x and y values

    # Calculate corresponding z values using the function
    Z = np.zeros((M,M))
    for m_1 in list(range(M)):
        for m_2 in list(range(M)):
            theta = np.array((X[m_1,m_2],Y[m_1,m_2]))
            #print(theta)
            #print(function(theta))
            Z[m_1,m_2] = function(theta)
    #Z = function((X, Y)

    # Create the 3D plot
    f = plt.figure(figsize=(10, 8))
    plt.imshow(Z, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower', cmap='viridis')
    plt.colorbar()
    
    #plt.xlabel('X')
    #plt.ylabel('Y')
    f.savefig(filename+'.pdf', bbox_inches='tight')
    #plt.title('3D Heatmap Plot of the Three-Dimensional Function')
    plt.show()
    
    
def plot_this_function(function,a,b,M,legend):
    I = np.linspace(a,b,M)
    e_list = []
    for i in I:
        e_list.append(function(i))
    plt.plot(I,e_list)
    plt.title(legend)
    plt.show()
    #return e_list
    
    
def plot_delta_functions(function_1,function_2,a,b,M,legend):
    I = np.linspace(a,b,M)
    e_list = []
    for i in I:
        e_list.append(np.abs(function_1(i)-function_2(i)))
    plt.plot(I,e_list)
    plt.title(legend)
    plt.show()
    #return e_list
    
