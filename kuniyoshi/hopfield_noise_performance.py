#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import click
import numpy as np
import matplotlib.pyplot as plt

from hopfield_single_performance import hopfield_single_performance


def main():
    # static params
    n_sample = 200
    fit_mode = 'hebbian'
    save_fig = True

    for n_label, c in zip([2, 4], ['rb', 'gy']):
        # dynamic params: noise_amount
        noise_amounts = np.arange(0.0, 0.51, 0.05)
        similarities = []
        accuracies = []
        for noise_amount in noise_amounts:
            similarity, accuracy = hopfield_single_performance(n_sample,
                                                            n_label,
                                                            noise_amount,
                                                            fit_mode,
                                                            save_fig)
            similarities.append(similarity)
            accuracies.append(accuracy)

        print('noise_amounts:', noise_amounts)
        print('similarities:', similarities)
        print('accuracies:', accuracies)

        plt.plot(noise_amounts * 100, similarities,
                 label='similarities (n_label: {})'.format(n_label),
                 c=c[0], marker='o')
        plt.plot(noise_amounts * 100, accuracies,
                 label='accuracies (n_label: {})'.format(n_label),
                 c=c[1], marker='o')
    plt.xlabel('noise amount [%]')
    plt.ylabel('performance')
    plt.xticks(noise_amounts * 100)
    plt.ylim(0, 1)
    plt.legend(loc='lower left')
    plt.savefig('noise_performance.png')
    # plt.show()


if __name__ == '__main__':
    main()