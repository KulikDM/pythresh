import sys
import unittest
from itertools import product
from os.path import dirname as up

# noinspection
import numpy as np
from numpy.testing import assert_equal
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.pca import PCA
from pyod.utils.data import generate_data

from pythresh.thresholds.mixmod import MIXMOD

# temporary solution for relative imports in case pythresh is not installed
# if pythresh is installed, no need to use the following line

path = up(up(up(__file__)))
sys.path.append(path)


class TestMIXMOD(unittest.TestCase):
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

        self.methods = ['mean', 'ks']

        self.tol = [1e-3, 1e-5, 1e-8, 1e-12]

        self.max_iter = [50, 100, 250, 500]

    def test_prediction_labels(self):

        params = product(self.all_scores, self.methods,
                         self.tol, self.max_iter)

        for scores, method, tol, max_iter in params:

            self.thres = MIXMOD(method=method, tol=tol, max_iter=max_iter)
            pred_labels = self.thres.eval(scores)

            assert (self.thres.thresh_ is not None)
            assert (self.thres.dscores_ is not None)
            assert (self.thres.mixture_ is not None)

            assert (self.thres.dscores_.min() == 0)
            assert (self.thres.dscores_.max() == 1)

            assert (self.thres.mixture_.components is not None)
            assert (self.thres.mixture_.weights is not None)
            assert (self.thres.mixture_.params is not None)

            nscores = self.thres.dscores_ + 1

            assert (callable(self.thres.mixture_.loglikelihood) and
                    (_ := self.thres.mixture_.loglikelihood(nscores))
                    is not None)

            assert (callable(self.thres.mixture_.pdf) and
                    (_ := self.thres.mixture_.pdf(nscores))
                    is not None)

            assert (callable(self.thres.mixture_.posterior) and
                    (_ := self.thres.mixture_.posterior(nscores))
                    is not None)

            assert_equal(pred_labels.shape, self.y_train.shape)

            if (not np.all(pred_labels == 0)) & (not np.all(pred_labels == 1)):

                assert (pred_labels.min() == 0)
                assert (pred_labels.max() == 1)
