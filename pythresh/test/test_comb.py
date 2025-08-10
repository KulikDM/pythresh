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

from pythresh.thresholds.comb import COMB

# temporary solution for relative imports in case pythresh is not installed
# if pythresh is installed, no need to use the following line

path = up(up(up(__file__)))
sys.path.append(path)


class TestCOMB(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.n_train = 200
        cls.n_test = 100
        cls.contamination = 0.1
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = generate_data(
            n_train=cls.n_train, n_test=cls.n_test,
            contamination=cls.contamination, random_state=42)

        cls.clfs = [KNN(), PCA(random_state=1234), IForest(random_state=1234)]
        cls.single_score = cls.clfs[0].fit(cls.X_train).decision_scores_
        cls.multiple_scores = np.vstack([
            clf.fit(cls.X_train).decision_scores_ for clf in cls.clfs
        ]).T
        cls.all_scores = [cls.single_score, cls.multiple_scores]

        cls.methods = ['mean', 'median', 'mode', 'bagged', 'stacked']
        cls.max_contams = [0.25, 1/3, 0.5, 2/3]

        cls.params = list(
            product(cls.all_scores, cls.methods, cls.max_contams))

    def setUp(self):
        self.thres = COMB()

    def check_labels(self, labels, scores_shape):
        self.assertEqual(labels.shape, scores_shape[:1])
        self.assertIn(labels.min(), [0, 1])
        self.assertIn(labels.max(), [0, 1])

    def check_fitted_attributes(self, thres):
        self.assertTrue(thres.__sklearn_is_fitted__())
        self.assertIsNotNone(thres.labels_)

    def test_eval(self):
        for scores, method, max_contam in self.params:
            thres = COMB(method=method, max_contam=max_contam)
            pred_labels = thres.eval(scores)

            self.assertIsNotNone(thres.dscores_)
            self.assertIsNotNone(thres.confidence_interval_)
            self.assertGreaterEqual(thres.dscores_.min(), 0)
            self.assertLessEqual(thres.dscores_.max(), 1)
            self.check_labels(pred_labels, scores.shape)

    def test_fit(self):
        for scores, method, max_contam in self.params:
            thres = COMB(method=method, max_contam=max_contam)
            thres.fit(scores)
            self.check_fitted_attributes(thres)
            self.check_labels(thres.labels_, scores.shape)

    def test_predict(self):
        for scores, method, max_contam in self.params:
            thres = COMB(method=method, max_contam=max_contam)
            thres.fit(scores)
            pred_labels = thres.predict(scores)
            self.check_fitted_attributes(thres)
            self.check_labels(pred_labels, scores.shape)
            assert_equal(thres.labels_, pred_labels)

    def test_test_data(self):
        all_test_scores = [
            self.clfs[0].fit(self.X_train).decision_function(self.X_test),
            np.vstack([clf.fit(self.X_train).decision_function(self.X_test)
                      for clf in self.clfs]).T
        ]

        for scores, method, max_contam in self.params:
            test_scores = all_test_scores[0] if scores.ndim == 1 else all_test_scores[1]

            thres = COMB(method=method, max_contam=max_contam)
            thres.fit(scores)
            pred_labels = thres.predict(test_scores)

            self.check_fitted_attributes(thres)
            self.check_labels(pred_labels, test_scores.shape)

    def test_save_and_load(self):
        for scores in self.all_scores:
            self.thres.fit(scores)
            joblib.dump(self.thres, 'model.pkl')
            loaded_thres = joblib.load('model.pkl')

            assert_equal(self.thres.predict(scores),
                         loaded_thres.predict(scores))
