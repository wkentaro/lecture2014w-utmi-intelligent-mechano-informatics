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
from recall import (
    recall_with_noise,
    get_recalling_performance,
    view_recalling_result,
    )
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
    print('\n.. fitting hopfield\n')
    hf, X, y, target_names, params = fit_hopfield(params)
    print_params(**params)

    # recall
    print('\n.. recalling\n')
    X, X_noise, X_recall = recall_with_noise(clf=hf, X=X,
                                             noise_amount=noise_amount)

    print_header('result')
    similarities, accurate = get_recalling_performance(X, X_recall)
    print('similarity:', np.mean(similarities))
    print('accuracy:', np.mean(accurate))

    # compare 3 images & save
    if save_fig:
        print('\n.. view recalling result\n')
        view_recalling_result(X, X_noise, X_recall,
                              accurate=accurate, **params)


if __name__ == '__main__':
    main()
