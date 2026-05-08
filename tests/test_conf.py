from itertools import product

import pytest
from utils import (
    build_scores,
    generate_train_test_data,
)

from pythresh.thresholds.filter import FILTER
from pythresh.thresholds.ocsvm import OCSVM
from pythresh.utils.conf import CONF

# -----------------------
# Param grid
# -----------------------


threshes = [FILTER(), OCSVM()]
alphas = [0.05, 0.1, 0.2]
splits = [0.2, 0.5, 0.8]
n_tests = [10, 100, 1000]

param_grid = list(product(threshes, alphas, splits, n_tests))


# -----------------------
# Fixtures
# -----------------------

@pytest.fixture(scope="module")
def data():
    return generate_train_test_data()

@pytest.fixture(scope="module")
def scores(data):
    X_train, _, _, _ = data
    return build_scores(X_train)


# -----------------------
# Prediction Labels
# -----------------------

@pytest.mark.parametrize("thres,alpha,split,n_test", param_grid)
def test_prediction_labels(scores, thres, alpha, split, n_test):
    s = scores[0]

    confidence = CONF(
        thresh=thres,
        alpha=alpha,
        split=split,
        n_test=n_test
    )
    uncertains = confidence.eval(s)

    assert (isinstance(uncertains, list))
    assert (len(uncertains) <= len(s))

    if len(uncertains) > 0:

        assert (min(uncertains) > 0)
        assert (max(uncertains) < len(s))
        assert (len(set(uncertains)) == len(uncertains))
