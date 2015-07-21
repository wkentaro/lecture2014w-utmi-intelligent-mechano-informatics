#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division

import numpy as np
from skimage.util import random_noise
from skimage.transform import resize
import matplotlib.pyplot as plt
import click

from hopfield import Hopfield
from utils import binarize
from data import load_alphabet, create_train_data
from visualization import (
    print_header,
    print_train_data,
    print_params,
    compare_origin_noise_recall,
    )


@click.command()
@click.option('--n-label', default=2, type=int, help='number of labels')
@click.option('--noise-amount', default=0.05, type=float, help='noise amount')
@click.option('--fit-mode', default='vector', type=str, help='fit mode')
def main(n_label, noise_amount, fit_mode):
    # load dataset
    dataset = load_alphabet()

    # parameters
    img_shape = dataset.image_shape
    print_params(n_label=n_label, img_shape=img_shape,
                 noise_amount=noise_amount, fit_mode=fit_mode)

    # transform data
    dataset.data = binarize(dataset.data, binary_values=(-1,1))

    # create train data
    target_names = dataset.target_names[:n_label]
    X, y = create_train_data(data=dataset.data,
                             target=dataset.target,
                             target_names=target_names,
                             n_sample=100)
    print_train_data(X, y, target_names)

    # fit hopfield
    hf = Hopfield(mode=fit_mode)
    hf.fit(X, y, watch_weight=False)

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

    # accuracy
    accuracy = np.array([np.linalg.norm(o-r) for o, r in zip(X, X_recall)])
    mask = (accuracy == 0)
    accuracy[mask], accuracy[~mask] = 1, 0
    print('accuracy:', accuracy.sum() / len(accuracy))

    # compare 3 images
    mask = (accuracy == 1)
    for origin, noise, recall in zip(X[mask], X_noise[mask], X_recall[mask]):
        origin, noise, recall = map(lambda x:x.reshape(img_shape).astype(int),
                                    [origin, noise, recall])
        compare_origin_noise_recall(origin, noise, recall,
                                    save_dir='accurate_{}'.format(n_label))

    mask = ~mask
    for origin, noise, recall in zip(X[mask], X_noise[mask], X_recall[mask]):
        origin, noise, recall = map(lambda x:x.reshape(img_shape).astype(int),
                                    [origin, noise, recall])
        compare_origin_noise_recall(origin, noise, recall,
                                    save_dir='wrong_{}'.format(n_label))


if __name__ == '__main__':
    main()
