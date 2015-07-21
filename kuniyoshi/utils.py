#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import time

import matplotlib.pyplot as plt
import sklearn.preprocessing


def binarize(X, binary_values=(0,1), **kwargs):
    X_binary = sklearn.preprocessing.binarize(X=X, **kwargs)
    X_binary[X_binary == 0] = binary_values[0]
    X_binary[X_binary == 1] = binary_values[1]
    return X_binary


def print_header(string, bar_length=40):
    print('-' * bar_length)
    print(string)
    print('-' * bar_length)


def print_train_data(X, y, target_names):
    print_header('train_data')

    print('target_names:', target_names)

    print('X.shape:', X.shape)
    print('X:', X, sep='\n')

    print('y.shape:', y.shape)
    print('y:', y, sep='\n')


def print_params(**kwargs):
    print_header('parameters')

    for k, v in kwargs.iteritems():
        print('{}: {}'.format(k, v))
