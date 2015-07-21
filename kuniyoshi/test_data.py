#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from data import load_alphabet


def test_load_alphabet():
    dataset = load_alphabet()
    # number of labels
    assert len(np.unique(dataset.target)) == len(dataset.target_names)
    # number of data
    assert len(dataset.data) == len(dataset.target)