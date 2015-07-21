#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import time

import matplotlib.pyplot as plt


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


def compare_origin_noise_recall(origin, noise, recall, save_dir):
    fig_data = [('original', origin), ('noise', noise), ('recall', recall)]
    for i, (title, img) in enumerate(fig_data):
        plt.subplot(131 + i)
        plt.imshow(img, cmap='gray')
        plt.title(title)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    filename = 'fig_{}.png'.format(time.time())
    filename = os.path.join(save_dir, filename)
    plt.savefig(filename)


def save_comparing_figs(X, X_noise, X_recall, img_shape, save_dir):
    for origin, noise, recall in zip(X[mask], X_noise[mask], X_recall[mask]):
        origin, noise, recall = map(lambda x:x.reshape(img_shape).astype(int),
                                    [origin, noise, recall])
        compare_origin_noise_recall(origin, noise, recall, save_dir)
