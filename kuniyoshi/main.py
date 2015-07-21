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
from visualization import print_header, print_train_data, print_params


@click.command()
@click.option('--n-label', default=2, type=int, help='number of labels')
def main(n_label):
    # load dataset
    dataset = load_alphabet()

    # parameters
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
    origin = X[index]
    noise = random_noise(origin.astype(float), mode='s&p', amount=0.1)
    noise = binarize(noise, binary_values=(-1,1))
    recall = hf.recall(x=noise, n_times=10)
    imgs = map(lambda x:x.reshape(img_shape).astype(int),
               [origin, noise, recall])
    for i, img in enumerate(imgs):
        plt.subplot(131 + i)
        plt.imshow(img, cmap='gray')
        print(img.astype(int))
    print('norm:', np.linalg.norm(imgs[0] - imgs[2]))
    plt.show()


if __name__ == '__main__':
    main()
