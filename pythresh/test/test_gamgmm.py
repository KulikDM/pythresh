import os
import sys
import unittest
from itertools import product
from os.path import dirname as up

# noinspection PyProtectedMember
import numpy as np
from joblib import Parallel, delayed
from numpy.testing import assert_equal
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.pca import PCA
from pyod.utils.data import generate_data

from pythresh.thresholds.gamgmm import GAMGMM

# temporary solution for relative imports in case pythresh is not installed
# if pythresh is installed, no need to use the following line

path = up(up(up(__file__)))
sys.path.append(path)

sys.stdout = open(os.devnull, 'w')


def tester(y_train, scores, verbose, n_contam, draws,
           p0, phigh, high_gamma, K):

    thres = GAMGMM(n_contaminations=n_contam, n_draws=draws,
                   p0=p0, phigh=phigh, high_gamma=high_gamma,
                   K=K, skip=True, verbose=verbose)

    pred_labels = thres.eval(scores)
    assert (thres.thresh_ is not None)

    assert_equal(pred_labels.shape, y_train.shape)

    if (not np.all(pred_labels == 0)) & (not np.all(pred_labels == 1)):

        assert (pred_labels.min() == 0)
        assert (pred_labels.max() == 1)


class TestGAMGMM(unittest.TestCase):
    def setUp(self):
        self.n_train = 200
        self.n_test = 100
        self.contamination = 0.1
        self.X_train, self.X_test, self.y_train, self.y_test = generate_data(
            n_train=self.n_train, n_test=self.n_test,
            contamination=self.contamination, random_state=42)

        clf = KNN()
        clf.fit(self.X_train)

        scores = clf.decision_scores_

        clfs = [KNN(), PCA(), IForest()]

        multiple_scores = []
        for clf in clfs:
            clf.fit(self.X_train)
            multiple_scores.append(clf.decision_scores_)

        multiple_scores = np.vstack(multiple_scores).T

        self.all_scores = [scores, multiple_scores]

        self.n_contaminations = [200, 1000, 10000]

        self.n_draws = [10, 50, 100]

        self.p0 = [0.005, 0.01, 0.05]

        self.phigh = [0.005, 0.01, 0.05]

        self.high_gamma = [0.15, 0.25, 0.5]

        self.K = [10, 50, 100]

        self.verbose = [True, False]

    def test_prediction_labels(self):

        # Create an iterable of all the loop variables
        all_loop_variables = [self.all_scores, self.verbose, self.n_contaminations,
                              self.n_draws, self.p0, self.phigh, self.high_gamma, self.K]

        # Get all combinations of loop variables
        all_combinations = list(product(*all_loop_variables))

        Parallel(n_jobs=-1)(delayed(tester)(self.y_train, *args)
                            for args in all_combinations)
