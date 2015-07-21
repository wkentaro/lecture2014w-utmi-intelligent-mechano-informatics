#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import time

import numpy as np


class Hopfield(object):
    def __init__(self, mode='vector'):
        self.mode = mode
        self.weight_ = None

    def fit(self, X, y, **kwargs):
        n_sample = len(X)
        if self.mode == 'vector':
            self._fit_vector(X, y, **kwargs)

    def _fit_vector(self, X, y, watch_weight=False):
        W = np.zeros((X.shape[1], X.shape[1]))
        for x in X:
            if watch_weight:
                print('W[0:10, 0:10]:')
                print(W[0:10, 0:10])
                time.sleep(0.1)
            x = np.atleast_2d(x).reshape((-1, 1))
            W += np.dot(x, x.T)
            for i in range(len(W)):
                W[i][i] = 0
        self.weight_ = W

    def recall(self, x, n_times):
        x = np.atleast_2d(x).reshape((-1, 1))
        W = self.weight_
        for _ in xrange(n_times):
            x = np.dot(W, x)
            x = np.sign(x)
        return x
