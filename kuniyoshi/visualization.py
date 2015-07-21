#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function


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
