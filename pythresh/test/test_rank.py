import sys
import unittest
from os.path import dirname as up

# noinspection PyProtectedMember
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.pca import PCA
from pyod.utils.data import generate_data

from pythresh.thresholds.filter import FILTER
from pythresh.utils.rank import RANK

# temporary solution for relative imports in case pythresh is not installed
# if pythresh is installed, no need to use the following line

path = up(up(up(__file__)))
sys.path.append(path)


class TestAUCP(unittest.TestCase):
    def setUp(self):
        self.n_train = 200
        self.n_test = 100
        self.contamination = 0.1
        self.X_train, self.X_test, self.y_train, self.y_test = generate_data(
            n_train=self.n_train, n_test=self.n_test,
            contamination=self.contamination, random_state=42)

        self.clfs = [KNN(), PCA(), IForest()]

        self.thres = FILTER()

        self.weights = [[0.5, 0.25, 0.25],
                        [0.25, 0.5, 0.25],
                        [0.25, 0.25, 0.5]]

    def test_prediction_labels(self):

        for weights in self.weights:

            ranker = RANK(self.clfs, self.thres, weights=weights)
            rankings = ranker.eval(self.X_train)

            cdf_rank = ranker.cdf_rank_
            clust_rank = ranker.clust_rank_
            mode_rank = ranker.mode_rank_

            assert (cdf_rank is not None)
            assert (clust_rank is not None)
            assert (mode_rank is not None)
            assert (rankings is not None)

            len_clf = len(self.clfs)

            assert (len(cdf_rank) == len_clf)
            assert (len(clust_rank) == len_clf)
            assert (len(mode_rank) == len_clf)
            assert (len(rankings) == len_clf)
