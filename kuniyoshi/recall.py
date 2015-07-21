#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from skimage.util import random_noise

from utils import binarize
from visualize import compare_origin_noise_recall


def recall_with_noise(clf, X, noise_amount=0.05):
    X = X.astype(float)
    X_noise = random_noise(X, mode='s&p', amount=noise_amount)
    X_noise = binarize(X_noise, binary_values=(-1,1))
    X_recall = []
    for x in X_noise:
        recall = clf.recall(x=x, n_times=10).reshape(-1)
        recall[recall < 0] = -1
        recall[recall >= 0] = 1
        X_recall.append(recall)
    X_recall = np.array(X_recall)
    return X, X_noise, X_recall


def get_recalling_performance(X, X_recall):
    # similarity
    similarities = np.array([np.linalg.norm(o-r) for o, r in zip(X, X_recall)])
    similarities /= X[0].size
    similarities = 1. - similarities
    # accuracy
    mask = (similarities == 1)
    accurate = np.copy(similarities)
    accurate[mask], accurate[~mask] = 1, 0
    return similarities, accurate


def view_recalling_result(X, X_noise, X_recall, accurate, **kwargs):
    # get params
    n_label = kwargs['n_label']
    img_shape = kwargs['img_shape']

    h, w = img_shape
    X = X.reshape((-1, h, w)).astype(int)
    X_noise = X_noise.reshape((-1, h, w)).astype(int)
    X_recall = X_recall.reshape((-1, h, w)).astype(int)
    for org, noise, recall, a in zip(X, X_noise, X_recall, accurate):
        if a == 1:
            save_dir = 'accurate_{}'.format(n_label)
        elif a == 0:
            save_dir = 'wrong_{}'.format(n_label)
        else:
            raise ValueError('unnexpected accuracy value')
        compare_origin_noise_recall(org, noise, recall, save_dir)
