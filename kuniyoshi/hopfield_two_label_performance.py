#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division
import itertools

import numpy as np
import click
import matplotlib.pyplot as plt

from utils import print_header, print_params
from fit import fit_hopfield
from recall import (
    recall_with_noise,
    get_recalling_performance,
    view_recalling_result,
    )


def _two_label_performance(target_names, params):
    # get_params
    noise_amount = params['noise_amount']

    # set params
    params['target_names'] = target_names
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

    similarity = np.mean(similarities)
    accuracy = np.mean(accurate)
    return similarity, accuracy


def hopfield_two_label_performance(
        n_sample,
        noise_amount,
        fit_mode,
        save_fig,
        ):

    # parameters
    params = {
        'n_sample': n_sample,
        'noise_amount': noise_amount,
        'fit_mode': fit_mode,
        }
    print_params(**params)

    labels = []
    similarities = []
    accuracies = []
    for target_names in itertools.combinations('chilot', 2):
        similarity, accuracy = _two_label_performance(target_names, params)
        labels.append(','.join(target_names))
        similarities.append(similarity)
        accuracies.append(accuracy)

    print('labels:', labels)
    print('similarities:', similarities)
    print('accuracies:', accuracies)

    fig, ax = plt.subplots()
    ind = np.arange(len(labels))
    width = 0.35
    ax.bar(ind, similarities, width, label='similarities', color='r')
    ax.bar(ind+width, accuracies, width, label='accuracies', color='b')
    ax.set_xlabel('two labels')
    ax.set_ylabel('performance')
    ax.set_xticks(ind+width)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    plt.legend(loc='lower right')
    # plt.show()
    plt.savefig('two_label_performance.png')


@click.command()
@click.option('--n-sample', default=200, type=int)
@click.option('--noise-amount', default=0.05, type=float, help='noise amount')
@click.option('--fit-mode', default='vector', type=str, help='fit mode')
@click.option('--save-fig', is_flag=True)
def main(**kwargs):
    hopfield_two_label_performance(**kwargs)


if __name__ == '__main__':
    main()