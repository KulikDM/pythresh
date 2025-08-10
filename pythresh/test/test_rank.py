import sys
import unittest
from itertools import product
from os.path import dirname as up

# noinspection PyProtectedMember
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.pca import PCA
from pyod.utils.data import generate_data

from pythresh.thresholds.filter import FILTER
from pythresh.thresholds.iqr import IQR
from pythresh.thresholds.ocsvm import OCSVM
from pythresh.utils.rank import RANK

# temporary solution for relative imports in case pythresh is not installed
# if pythresh is installed, no need to use the following line

path = up(up(up(__file__)))
sys.path.append(path)


class TestRANK(unittest.TestCase):
    def setUp(self):
        self.n_train = 200
        self.n_test = 100
        self.contamination = 0.1
        self.X_train, self.X_test, self.y_train, self.y_test = generate_data(
            n_train=self.n_train, n_test=self.n_test,
            contamination=self.contamination, random_state=42)

        self.clfs = [KNN(), PCA(random_state=1234), IForest(random_state=1234)]

        self.thres = [FILTER(), self.contamination,
                      [FILTER(), IQR(), OCSVM()]]

        self.method = ['model', 'native']

        self.weights = [[0.5, 0.25, 0.25],
                        [0.25, 0.5, 0.25],
                        [0.25, 0.25, 0.5],
                        None]

    def test_prediction_labels(self):

        params = product(self.thres,
                         self.method,
                         self.weights)

        for thres, method, weights in params:

            ranker = RANK(self.clfs, thres, method=method, weights=weights)
            rankings = ranker.eval(self.X_train)

            cdf_rank = ranker.cdf_rank_
            clust_rank = ranker.clust_rank_
            consensus_rank = ranker.consensus_rank_

            assert (cdf_rank is not None)
            assert (clust_rank is not None)
            assert (consensus_rank is not None)
            assert (rankings is not None)

            n_clfs = len(self.clfs)
            n_thres = len(thres) if isinstance(thres, list) else 1
            len_models = n_clfs * n_thres

            assert (len(cdf_rank) == len_models)
            assert (len(clust_rank) == len_models)
            assert (len(consensus_rank) == len_models)
            assert (len(rankings) == len_models)

            assert (len(set(rankings)) == len_models)
