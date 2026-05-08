from itertools import product

import pytest
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.pca import PCA
from utils import (
    generate_train_test_data,
)

from pythresh.thresholds.filter import FILTER
from pythresh.thresholds.iqr import IQR
from pythresh.thresholds.ocsvm import OCSVM
from pythresh.utils.rank import RANK

# -----------------------
# Param grid
# -----------------------

threshes = [FILTER(), 0.1,
            [FILTER(), IQR(), OCSVM()]]
methods = ['model', 'native']
weights = [[0.5, 0.25, 0.25],
           [0.25, 0.5, 0.25],
           [0.25, 0.25, 0.5],
           None]

param_grid = list(product(threshes, methods, weights))


# -----------------------
# Fixtures
# -----------------------


@pytest.fixture(scope="module")
def data():
    return generate_train_test_data()

@pytest.fixture(scope="module")
def clfs():
    return [
        KNN(),
        PCA(random_state=1234),
        IForest(random_state=1234),
    ]


# -----------------------
# Prediction Labels
# -----------------------

@pytest.mark.parametrize("thres,method,weights", param_grid)
def test_prediction_labels(data, clfs, thres, method, weights):
    X_train, _, _, _ = data

    ranker = RANK(clfs, thres, method=method, weights=weights)
    rankings = ranker.eval(X_train)

    cdf_rank = ranker.cdf_rank_
    clust_rank = ranker.clust_rank_
    consensus_rank = ranker.consensus_rank_

    assert (cdf_rank is not None)
    assert (clust_rank is not None)
    assert (consensus_rank is not None)
    assert (rankings is not None)

    n_clfs = len(clfs)
    n_thres = len(thres) if isinstance(thres, list) else 1
    len_models = n_clfs * n_thres

    assert (len(cdf_rank) == len_models)
    assert (len(clust_rank) == len_models)
    assert (len(consensus_rank) == len_models)
    assert (len(rankings) == len_models)

    assert (len(set(rankings)) == len_models)
