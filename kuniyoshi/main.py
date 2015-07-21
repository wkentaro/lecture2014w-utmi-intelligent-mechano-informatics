#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division

import numpy as np
from skimage.util import random_noise
from skimage.transform import resize
import matplotlib.pyplot as plt

from hopfield import Hopfield
from utils import binarize
from data import load_alphabet, create_train_data
from visualization import print_header, print_train_data, print_params


def main():
    # load dataset
    dataset = load_alphabet()

    # parameters
    n_label = 2
    img_shape = dataset.image_shape
    print_params(n_label=n_label, img_shape=img_shape)

    # transform data
    dataset.data = binarize(dataset.data, binary_values=(-1,1))

    # create train data
    target_names = dataset.target_names[:n_label]
    X, y = create_train_data(data=dataset.data,
                             target=dataset.target,
                             target_names=target_names,
                             n_sample=10)
    print_train_data(X, y, target_names)

    # fit hopfield
    hf = Hopfield()
    hf.fit(X, y, watch_weight=False)

    # recall
    index = np.random.randint(0, len(X))
    print_header('input')
    img = X[index].reshape(img_shape).astype(int)
    img[img == -1] = 0
    print(img)
    plt.subplot(131)
    plt.imshow(img, cmap='gray')
    print_header('input_with_noise')
    input_ = X[index].reshape(img_shape)
    input_ = random_noise(X[index].astype(float), mode='s&p', amount=0.1)
    input_ = binarize(input_, binary_values=(-1,1))
    img = input_.reshape(img_shape).astype(int)
    img[img == -1] = 0
    print(img)
    plt.subplot(132)
    plt.imshow(img, cmap='gray')
    print_header('output')
    ret = hf.recall(x=input_, n_times=10)
    img = ret.reshape(img_shape).astype(int)
    img[img == -1] = 0
    print(img)
    plt.subplot(133)
    plt.imshow(img, cmap='gray')
    plt.show()


if __name__ == '__main__':
    main()
