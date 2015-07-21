#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.datasets import load_digits, load_iris, fetch_mldata
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

from svm import dataset_fixed_cov, SVM


class AdaBoost(object):
    def __init__(self, classifiers):
        self.clfs = classifiers

    def fit(self, X, y):
        clfs = self.clfs
        M = len(clfs)
        N = len(X)

        for i, clf in enumerate(clfs):
            print('fitting:', i)
            clf.fit(X, y)

        w = np.ones(N) / N
        alpha = np.zeros(M)
        scores = [(i, clf.score(X, y)) for i, clf in enumerate(clfs)]
        for m, score in sorted(scores, key=lambda x:x[1], reverse=True):
            print('updating weight:', m)
            print('score:', score) 
            I = (clfs[m].predict(X) != y).astype(int)
            J = (w * I).sum()
            print('J:', J)
            epsilon = J / w.sum()
            print('epsion:', epsilon)
            alpha[m] = np.log((1. - epsilon) / epsilon)
            w = w * np.exp(alpha[m] * I)
            w /= w.sum()

        self.alpha = alpha

    def predict(self, X):
        clfs = self.clfs
        alpha = self.alpha
        M = len(clfs)

        y_pred = np.zeros(len(X))
        for m in range(M):
            y_pred += alpha[m] * clfs[m].predict(X)
        y_pred = np.sign(y_pred)

        return y_pred

    def score(self, X, y):
        y_pred = self.predict(X)
        score = accuracy_score(y, y_pred)
        return score


def test_adaboost():
    X, y = dataset_fixed_cov(n=300)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

    clfs = []
    for i in range(10):
        clfs.append(SVM(iterations=i+1, learning_rate=0.01))
    ada = AdaBoost(classifiers=clfs)
    ada.fit(X, y)
    print('alpha:', ada.alpha)
    for i, clf in enumerate(clfs):
        print('score:', i, clf.score(X_test, y_test))
    score = ada.score(X_test, y_test)
    print('score:', score)


if __name__ == '__main__':
    test_adaboost()
