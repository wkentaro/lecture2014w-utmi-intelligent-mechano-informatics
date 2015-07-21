#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time

import matplotlib.pyplot as plt


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
