from itertools import product

import numpy as np
import pytest
from utils import (
    build_scores,
    check_labels,
    check_scores_normalized,
    generate_train_test_data,
)

from pythresh.thresholds.dsn import DSN

# -----------------------
# Param grid
# -----------------------


metrics = ["JS", "MAH"]
score_cases = [
    ("single", 0),
    ("multi", 1),
]

param_grid = list(product(metrics, score_cases))


# -----------------------
# Fixtures
# -----------------------


@pytest.fixture(scope="module")
def data():
    return generate_train_test_data(n_train=1000)


@pytest.fixture(scope="module")
def scores(data):
    X_train, _, _, _ = data
    return build_scores(X_train)


# -----------------------
# Eval
# -----------------------


@pytest.mark.parametrize("metric,score_case", param_grid)
def test_prediction_labels(scores, metric, score_case):
    _, idx = score_case
    s = scores[idx]

    thres = DSN(metric=metric)
    pred_labels = thres.eval(s)
    dscores = thres.dscores_

    assert thres.thresh_ is not None
    check_scores_normalized(dscores)
    check_labels(pred_labels, s.shape)

    if not (np.all(pred_labels == 0) or np.all(pred_labels == 1)):
        assert pred_labels.min() == 0
        assert pred_labels.max() == 1
