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

from pythresh.thresholds.zscore import ZSCORE

# temporary solution for relative imports in case pythresh is not installed
# if pythresh is installed, no need to use the following line

path = up(up(up(__file__)))
sys.path.append(path)


class TestZSCORE(unittest.TestCase):
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

        self.thres = ZSCORE()

    def test_prediction_labels(self):

        for scores in self.all_scores:

            pred_labels = self.thres.eval(scores)
            assert (self.thres.thresh_ is not None)
            assert (self.thres.dscores_ is not None)

            assert (self.thres.dscores_.min() == 0)
            assert (self.thres.dscores_.max() == 1)

            assert_equal(pred_labels.shape, self.y_train.shape)

            assert (pred_labels.min() == 0)
            assert (pred_labels.max() == 1)

    def test_factor_adjustment(self):
        """Test the effect of the factor on ZSCORE thresholding."""
        for scores in self.all_scores:
            # Test with default factor (1)
            thres_default = ZSCORE(factor=1)
            pred_labels_default = thres_default.eval(scores)
            default_outliers = np.sum(pred_labels_default)

            # Test with a higher factor
            thres_high = ZSCORE(factor=2)
            pred_labels_high = thres_high.eval(scores)
            high_outliers = np.sum(pred_labels_high)

            # Test with a lower factor
            thres_low = ZSCORE(factor=0.5)
            pred_labels_low = thres_low.eval(scores)
            low_outliers = np.sum(pred_labels_low)

            # Assertions on the number of outliers
            self.assertLessEqual(high_outliers, default_outliers,
                                 'Higher factor should reduce the number of outliers.')
            self.assertGreaterEqual(low_outliers, default_outliers,
                                    'Lower factor should increase the number of outliers.')

            # Assertions on predictions being binary
            for pred_labels in [pred_labels_default, pred_labels_high, pred_labels_low]:
                self.assertTrue(np.array_equal(np.unique(pred_labels), [0, 1]),
                                'Predictions should only contain 0 and 1.')
