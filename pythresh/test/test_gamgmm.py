import os
import sys
import unittest
import warnings
from os.path import dirname as up

# noinspection PyProtectedMember
import numpy as np
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

        multiple_scores = [
            clf.fit(self.X_train).decision_scores_ for clf in clfs]

        multiple_scores = np.vstack(multiple_scores).T

        self.all_scores = [scores, multiple_scores]

        self.all_scores = [scores, multiple_scores]

        self.skip = [True, False]

        self.verbose = [True, False]

    def test_prediction_labels(self):

        for scores in self.all_scores:
            for skip in self.skip:
                for verbose in self.verbose:

                    self.thres = GAMGMM(skip=skip, steps=10, verbose=verbose)

                    pred_labels = self.thres.eval(scores)
                    assert (self.thres.thresh_ is not None)
                    assert (self.thres.dscores_ is not None)

                    assert (self.thres.dscores_.min() == 0)
                    assert (self.thres.dscores_.max() == 1)

                    assert_equal(pred_labels.shape, self.y_train.shape)

                    if (not np.all(pred_labels == 0)) & (not np.all(pred_labels == 1)):

                        assert (pred_labels.min() == 0)
                        assert (pred_labels.max() == 1)
