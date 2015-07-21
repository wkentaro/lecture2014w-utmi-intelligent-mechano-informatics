#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import time
import collections

import numpy as np


class Hopfield(object):
    def __init__(self, mode='vector'):
        self.mode = mode
        self.weight_ = None

    def fit(self, X, y, **kwargs):
        n_sample = len(X)
        if self.mode == 'vector':
            self._fit_vector(X, y, **kwargs)
        elif self.mode == 'hebbian':
            self._fit_hebbian(X, y, **kwargs)
        else:
            raise ValueError('unknown fit mode')

    def _fit_vector(self, X, y, watch_weight=False):
        W = np.zeros((X.shape[1], X.shape[1]))
        for x in X:
            if watch_weight:
                print('W[0:10, 0:10]:')
                print(W[0:10, 0:10])
                time.sleep(0.1)
            x = np.atleast_2d(x).reshape((-1, 1))
            W += np.dot(x, x.T)
        self.weight_ = W

    def _fit_hebbian(self, X, y, watch_weight=False):
        W = np.zeros((X.shape[1], X.shape[1]))
        count = collections.defaultdict(int)
        for x, yi in zip(X, y):
            if watch_weight:
                print('W[0:10, 0:10]:')
                print(W[0:10, 0:10])
            count[yi] += 1
            n = count[yi]
            for i, xi in enumerate(x):
                for j, xj in enumerate(x):
                    delta = ((n-1) / n) * W[i][j] + (1/n) * xi * xj
                    W[i][j] += delta
        self.weight_ = W

    def recall(self, x, n_times):
        x = np.atleast_2d(x).reshape((-1, 1))
        W = self.weight_
        for _ in xrange(n_times):
            x = np.dot(W, x)
            x = np.sign(x)
        return x
