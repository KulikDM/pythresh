import os
import sys
import unittest
import warnings
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
warnings.simplefilter('ignore')


def tester(y_train, scores, skip, verbose):

    thres = GAMGMM(skip=skip, steps=10, verbose=verbose)

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

        self.skip = [True, False]

        self.verbose = [True, False]

    def test_prediction_labels(self):

        # Create an iterable of all the loop variables
        all_loop_variables = [[self.y_train], self.all_scores,
                              self.skip, self.verbose]

        # Get all combinations of loop variables
        all_combinations = list(product(*all_loop_variables))

        Parallel(n_jobs=-1)(delayed(tester)(*args)
                            for args in all_combinations)
