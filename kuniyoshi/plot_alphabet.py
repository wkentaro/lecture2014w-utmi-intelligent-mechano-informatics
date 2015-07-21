#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import matplotlib.pyplot as plt

from data import load_alphabet


dataset = load_alphabet()
images = []
for t in dataset.target_names:
    images.append(dataset.images[dataset.target == t][0])
for i, (label, img) in enumerate(zip(dataset.target_names, images)):
    plt.subplot(2, len(dataset.target_names)/2, i+1)
    plt.imshow(img, cmap='gray')
    plt.title(label.upper())
    plt.axis('off')
plt.savefig('alphabet_images.png')
