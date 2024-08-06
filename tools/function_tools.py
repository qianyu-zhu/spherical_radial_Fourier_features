#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 12:26:17 2023

@author: ayoubbelhadji
"""

import numpy as np

## Function manipulation tools

def get_delta_functions(function_1,function_2):
    def delta_function_aux(x):
        return np.abs(function_1(x) - function_2(x))
    return delta_function_aux
    

    

def section_2D_function(function,v):
    def section_2D_function_aux(t):
        x = t*v 
        return function(x)
    return section_2D_function_aux
    

