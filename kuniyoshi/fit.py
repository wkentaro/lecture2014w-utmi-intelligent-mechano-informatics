#!/usr/bin/env python
# -*- coding: utf-8 -*-

from hopfield import Hopfield
from data import load_alphabet, create_train_data
from utils import binarize
from visualization import print_train_data


def fit_hopfield(params):
    # get params
    n_label = params['n_label']
    n_sample = params['n_sample']
    fit_mode = params['fit_mode']

    # load dataset
    dataset = load_alphabet()

    # transform data
    dataset.data = binarize(dataset.data, binary_values=(-1,1))

    # create train data
    target_names = dataset.target_names[:n_label]
    X, y = create_train_data(data=dataset.data,
                             target=dataset.target,
                             target_names=target_names,
                             n_sample=n_sample)
    print_train_data(X, y, target_names)

    # fit hopfield
    hf = Hopfield(mode=fit_mode)
    hf.fit(X, y, watch_weight=False)

    # set params
    params['img_shape'] = dataset.image_shape

    return hf, X, y, target_names, params