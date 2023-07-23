import sys
import unittest
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

        self.all_scores = [scores, multiple_scores]

        self.n_contaminations = [200, 1000, 10000]

        self.n_draws = [10, 50, 100]

        self.p0 = [0.005, 0.01, 0.05]

        self.phigh = [0.005, 0.01, 0.05]

        self.high_gamma = [0.15, 0.25, 0.5]

        self.K = [10, 50, 100]

        self.verbose = [True, False]

    def test_prediction_labels(self):

        for scores in self.all_scores:

            for verbose in self.verbose:

                for n_contam in self.n_contaminations:

                    for draws in self.n_draws:

                        for p0 in self.p0:

                            for phigh in self.phigh:

                                for high_gamma in self.high_gamma:

                                    for K in self.K:

                                        self.thres = GAMGMM(n_contaminations=n_contam, n_draws=draws,
                                                            p0=p0, phigh=phigh, high_gamma=high_gamma,
                                                            K=K, skip=True, verbose=verbose)

                                        pred_labels = self.thres.eval(scores)
                                        assert (self.thres.thresh_ is not None)

                                        assert_equal(
                                            pred_labels.shape, self.y_train.shape)

                                        if (not np.all(pred_labels == 0)) & (not np.all(pred_labels == 1)):

                                            assert (pred_labels.min() == 0)
                                            assert (pred_labels.max() == 1)
