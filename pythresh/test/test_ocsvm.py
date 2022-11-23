# -*- coding: utf-8 -*-
from __future__ import division, print_function

import sys
import unittest
from os.path import dirname as up

# noinspection PyProtectedMember
import numpy as np
from numpy.testing import (assert_allclose, assert_array_less, assert_equal,
                           assert_raises)
from pyod.models.knn import KNN
from pyod.utils.data import generate_data

from pythresh.thresholds.ocsvm import OCSVM

# temporary solution for relative imports in case pythresh is not installed
# if pythresh is installed, no need to use the following line

path = up(up(up(__file__)))
sys.path.append(path)


class TestOCSVM(unittest.TestCase):
    def setUp(self):
        self.n_train = 200
        self.n_test = 100
        self.contamination = 0.1
        self.X_train, self.X_test, self.y_train, self.y_test = generate_data(
            n_train=self.n_train, n_test=self.n_test,
            contamination=self.contamination, random_state=42)

        self.clf = KNN()
        self.clf.fit(self.X_train)

        self.scores = self.clf.decision_scores_

        self.models = ['poly', 'sgd']
        self.degree = ['auto', 2, 3, 4, 7, 10]
        self.gamma = ['auto', 0.1, 0.5, 0.9]
        self.criterion = ['aic', 'bic']
        self.nu = [0.1, 0.5, 0.9]
        self.tol = [1e-1, 1e-3, 1e-5]

    def test_prediction_labels(self):

        for model in self.models:
            for degree in self.degree:
                for gamma in self.gamma:
                    for criterion in self.criterion:
                        for nu in self.nu:
                            for tol in self.tol:

                                self.thres = OCSVM(model=model, degree=degree,
                                                   gamma=gamma, criterion=criterion,
                                                   nu=nu, tol=tol)

                                pred_labels = self.thres.eval(self.scores)
                                assert (self.thres.thresh_ is None)

                                assert_equal(pred_labels.shape,
                                             self.y_train.shape)

                                if (not np.all(pred_labels == 0)) & (not np.all(pred_labels == 1)):

                                    assert (pred_labels.min() == 0)
                                    assert (pred_labels.max() == 1)
