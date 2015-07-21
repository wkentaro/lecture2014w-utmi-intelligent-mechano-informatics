#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import matplotlib.pyplot as plt

from hopfield_single_performance import hopfield_single_performance


# static params
n_sample = 100
noise_amount = 0.05
fit_mode='hebbian'
save_fig = False

# dynamic params: n_label
n_labels = []
similarities = []
accuracies = []
for n_label in range(2, 7):
    similarity, accuracy = hopfield_single_performance(n_sample,
                                                       n_label,
                                                       noise_amount,
                                                       fit_mode,
                                                       save_fig)
    n_labels.append(n_label)
    similarities.append(similarity)
    accuracies.append(accuracy)

print('n_labels:', n_labels)
print('similarities:', similarities)
print('accuracies:', accuracies)

plt.plot(n_labels, similarities, label='similarities', c='r', marker='o')
plt.plot(n_labels, accuracies, label='accuracies', c='b', marker='o')
plt.xlabel('number of labels')
plt.ylabel('number performance')
plt.xticks(n_labels)
plt.ylim(0, 1)
plt.legend(loc='lower left')
plt.savefig('hopfield_labels_performance.png')
# plt.show()
