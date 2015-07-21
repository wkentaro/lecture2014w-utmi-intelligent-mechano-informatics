#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.datasets.base import Bunch


def load_alphabet():
    C = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1],
        [0, 1, 1, 1, 1],
        [0, 1, 1, 1, 1],
        [0, 0, 0, 0, 0],
        ])
    H = np.array([
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        ])
    I = np.array([
        [1, 0, 0, 0, 1],
        [1, 1, 0, 1, 1],
        [1, 1, 0, 1, 1],
        [1, 1, 0, 1, 1],
        [1, 0, 0, 0, 1],
        ])
    L = np.array([
        [0, 1, 1, 1, 1],
        [0, 1, 1, 1, 1],
        [0, 1, 1, 1, 1],
        [0, 1, 1, 1, 1],
        [0, 0, 0, 0, 0],
        ])
    X = np.array([
        [0, 1, 1, 1, 0],
        [1, 0, 1, 0, 1],
        [1, 1, 0, 1, 1],
        [1, 0, 1, 0, 1],
        [0, 1, 1, 1, 0],
        ])
    T = np.array([
        [0, 0, 0, 0, 0],
        [1, 1, 0, 1, 1],
        [1, 1, 0, 1, 1],
        [1, 1, 0, 1, 1],
        [1, 1, 0, 1, 1],
        ])
    dataset = Bunch(
                target_names=list('chiltx'),
                target=np.array(list('chiltx')),
                data=np.array(
                        map(lambda x:x.reshape(-1).astype(float),
                            [C, H, I, L, T, X])
                        ),
                images=np.array([C, H, I, L, T, X]),
                image_shape=(5, 5),
                )
    return dataset


def create_train_data(data, target, target_names, n_sample):
    X, y = [], []
    for t in target_names:
        X.append(data[target == t])
        y.append(target[target == t])
    X = np.vstack(X)
    y = np.hstack(y)
    p = np.random.randint(0, len(X), n_sample)
    X, y = X[p], y[p]
    return X, y
