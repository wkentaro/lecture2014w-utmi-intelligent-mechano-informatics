#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import matplotlib.pyplot as plt

from data import load_alphabet


save_dir = 'alphabet_images'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

dataset = load_alphabet()

for target, img in zip(dataset.target, dataset.images):
    plt.imshow(img, cmap='gray')
    filename = '{}.png'.format(target)
    filename = os.path.join(save_dir, filename)
    plt.savefig(filename)
    plt.cla()
