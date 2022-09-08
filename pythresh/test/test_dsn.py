# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function

from os.path import dirname as up
import sys

import unittest
# noinspection PyProtectedMember
import numpy as np
from numpy.testing import assert_allclose
from numpy.testing import assert_array_less
from numpy.testing import assert_equal
from numpy.testing import assert_raises

# temporary solution for relative imports in case pythresh is not installed
# if pythresh is installed, no need to use the following line

path = up(up(up(__file__)))
sys.path.append(path)

from pythresh.thresholds.dsn import DSN

from pyod.models.knn import KNN
from pyod.utils.data import generate_data


class TestDSN(unittest.TestCase):
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
        self.metrics = ['JS', 'WS', 'ENG', 'BHT', 'HLL', 'HI', 'LK',
                        'LP', 'MAH', 'TMT', 'RES', 'KS']

    def test_prediction_labels(self):

        for metric in self.metrics:

            self.thres = DSN(metric=metric)
            pred_labels = self.thres.eval(self.scores)
            assert (self.thres.thresh_ != None)
    
            assert_equal(pred_labels.shape, self.y_train.shape)

            if not np.all((pred_labels==0)|(pred_labels==1)):
            
                assert (pred_labels.min() == 0)
                assert (pred_labels.max() == 1)


    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
