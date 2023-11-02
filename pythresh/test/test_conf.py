import sys
import unittest
from itertools import product
from os.path import dirname as up

# noinspection PyProtectedMember
from pyod.models.iforest import IForest
from pyod.utils.data import generate_data

from pythresh.thresholds.filter import FILTER
from pythresh.thresholds.ocsvm import OCSVM
from pythresh.utils.conf import CONF

# temporary solution for relative imports in case pythresh is not installed
# if pythresh is installed, no need to use the following line

path = up(up(up(__file__)))
sys.path.append(path)


class TestCONF(unittest.TestCase):
    def setUp(self):
        self.n_train = 200
        self.n_test = 100
        self.contamination = 0.1
        self.X_train, self.X_test, self.y_train, self.y_test = generate_data(
            n_train=self.n_train, n_test=self.n_test,
            contamination=self.contamination, random_state=42)

        self.clf = IForest()

        self.thres = [FILTER(), OCSVM()]

        self.alphas = [0.05, 0.1, 0.2]

        self.splits = [0.2, 0.5, 0.8]

        self.n_tests = [10, 100, 1000]

    def test_prediction_labels(self):

        params = product(self.thres,
                         self.alphas,
                         self.splits,
                         self.n_tests)

        for thres, alpha, split, n_test in params:

            confidence = CONF(self.clf, thres, alpha=alpha,
                              split=split, n_test=n_test)
            uncertains = confidence.eval(self.X_train)

            assert (isinstance(uncertains, list))
            assert (len(uncertains) <= len(self.X_train))

            if len(uncertains) > 0:

                assert (min(uncertains) > 0)
                assert (max(uncertains) < len(self.X_train))
                assert (len(set(uncertains)) == len(uncertains))
