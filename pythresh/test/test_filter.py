import sys
import unittest
from itertools import product
from os.path import dirname as up

# noinspection PyProtectedMember
import joblib
import numpy as np
from numpy.testing import assert_equal
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.pca import PCA
from pyod.utils.data import generate_data

from pythresh.thresholds.filter import FILTER

# temporary solution for relative imports in case pythresh is not installed
# if pythresh is installed, no need to use the following line

path = up(up(up(__file__)))
sys.path.append(path)


class TestFilter(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.n_train = 200
        cls.n_test = 100
        cls.contamination = 0.1
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = generate_data(
            n_train=cls.n_train, n_test=cls.n_test,
            contamination=cls.contamination, random_state=42)

        cls.clfs = [KNN(), PCA(), IForest()]
        cls.single_score = cls.clfs[0].fit(cls.X_train).decision_scores_
        cls.multiple_scores = np.vstack([
            clf.fit(cls.X_train).decision_scores_ for clf in cls.clfs
        ]).T
        cls.all_scores = [cls.single_score, cls.multiple_scores]

        cls.methods = ['gaussian', 'savgol', 'hilbert', 'wiener', 'medfilt',
                       'decimate', 'detrend', 'resample']

        cls.sigmas = ['auto', int(cls.n_train**0.6), int(cls.n_train**0.75)]

        cls.params = list(product(cls.all_scores, cls.methods, cls.sigmas))

    def setUp(self):
        self.thres = FILTER()

    def check_labels(self, labels, scores_shape):
        self.assertEqual(labels.shape, scores_shape[:1])
        self.assertIn(labels.min(), [0, 1])
        self.assertIn(labels.max(), [0, 1])

    def check_fitted_attributes(self, thres):
        self.assertTrue(thres.__sklearn_is_fitted__())
        self.assertIsNotNone(thres.labels_)
        self.assertIsNotNone(thres.thresh_)

    def test_eval(self):
        for scores, method, sigma in self.params:
            thres = FILTER(method=method, sigma=sigma)
            pred_labels = thres.eval(scores)

            self.assertIsNotNone(thres.thresh_)
            self.assertIsNotNone(thres.dscores_)
            self.assertGreaterEqual(thres.dscores_.min(), 0)
            self.assertLessEqual(thres.dscores_.max(), 1)
            self.check_labels(pred_labels, scores.shape)

    def test_fit(self):
        for scores in self.all_scores:
            self.thres.fit(scores)
            self.check_fitted_attributes(self.thres)
            self.check_labels(self.thres.labels_, scores.shape)

    def test_predict(self):
        for scores in self.all_scores:
            self.thres.fit(scores)
            pred_labels = self.thres.predict(scores)
            self.check_fitted_attributes(self.thres)
            self.check_labels(pred_labels, scores.shape)
            assert_equal(self.thres.labels_, pred_labels)

    def test_test_data(self):
        for scores, test_scores in zip(self.all_scores, [
            self.clfs[0].fit(self.X_train).decision_function(self.X_test),
            np.vstack([clf.fit(self.X_train).decision_function(self.X_test)
                      for clf in self.clfs]).T
        ]):
            self.thres.fit(scores)
            pred_labels = self.thres.predict(test_scores)
            self.check_fitted_attributes(self.thres)
            self.check_labels(pred_labels, test_scores.shape)

    def test_save_and_load(self):
        for scores in self.all_scores:
            self.thres.fit(scores)
            joblib.dump(self.thres, 'model.pkl')
            loaded_thres = joblib.load('model.pkl')

            assert_equal(self.thres.predict(scores),
                         loaded_thres.predict(scores))
