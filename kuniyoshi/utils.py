#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sklearn.preprocessing


def binarize(X, binary_values=(0,1), **kwargs):
    X_binary = sklearn.preprocessing.binarize(X=X, **kwargs)
    X_binary[X_binary == 0] = binary_values[0]
    X_binary[X_binary == 1] = binary_values[1]
    return X_binary
