#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division

import numpy as np
from skimage.util import random_noise
from skimage.transform import resize
import matplotlib.pyplot as plt
import click

from utils import binarize
from fit import fit_hopfield
from visualization import (
    print_header,
    print_params,
    compare_origin_noise_recall,
    )


@click.command()
@click.option('--n-sample', default=100, type=int)
@click.option('--n-label', default=2, type=int, help='number of labels')
@click.option('--noise-amount', default=0.05, type=float, help='noise amount')
@click.option('--fit-mode', default='vector', type=str, help='fit mode')
@click.option('--save-fig', is_flag=True)
def main(n_sample, n_label, noise_amount, fit_mode, save_fig):
    # parameters
    params = {
        'n_sample': n_sample,
        'n_label': n_label,
        'noise_amount': noise_amount,
        'fit_mode': fit_mode,
        }
    print_params(**params)

    # fit hopfield
    print_header('fitting hopfield')
    hf, X, y, target_names, params = fit_hopfield(params)
    print_params(**params)

    # recall
    X_noise = random_noise(X.astype(float), mode='s&p', amount=noise_amount)
    X_noise = binarize(X_noise, binary_values=(-1,1))
    X_recall = []
    for x in X_noise:
        recall = hf.recall(x=x, n_times=10).reshape(-1)
        recall[recall < 0] = -1
        recall[recall >= 0] = 1
        X_recall.append(recall)
    X_recall = np.array(X_recall)

    print_header('result')
    # similarity
    similarity = np.array([np.linalg.norm(o-r) for o, r in zip(X, X_recall)])
    similarity /= X[0].size
    similarity = 1. - similarity
    print('similarity:', np.mean(similarity))
    # accuracy
    mask = (similarity == 1)
    accuracy = np.copy(similarity)
    accuracy[mask], accuracy[~mask] = 1, 0
    print('accuracy:', accuracy.sum() / len(accuracy))

    # compare 3 images & save
    if save_fig:
        h, w = img_shape
        X = X.reshape((-1, h, w)).astype(int)
        X_noise = X_noise.reshape((-1, h, w)).astype(int)
        X_recall = X_recall.reshape((-1, h, w)).astype(int)
        for org, noise, recall, a in zip(X, X_noise, X_recall, accuracy):
            if a == 1:
                save_dir = 'accurate_{}'.format(n_label)
            elif a == 0:
                save_dir = 'wrong_{}'.format(n_label)
            else:
                raise ValueError('unnexpected accuracy value')
            compare_origin_noise_recall(org, noise, recall, save_dir)


if __name__ == '__main__':
    main()
