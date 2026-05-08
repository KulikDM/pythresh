from itertools import product

import joblib
import pytest
from numpy.testing import assert_equal
from utils import (
    build_scores,
    build_test_scores,
    check_fitted_attributes,
    check_labels,
    check_predict_consistency,
    check_scores_normalized,
    generate_train_test_data,
)

from pythresh.thresholds.meta import META

# -----------------------
# Param grid
# -----------------------

methods = ['LIN', 'GNB', 'GNBC', 'GNBM']
score_cases = [
    ("single", 0),
    ("multi", 1),
]

param_grid = list(product(methods, score_cases))


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
# Eval
# -----------------------

@pytest.mark.parametrize("method,score_case", param_grid)
def test_eval(scores, method, score_case):
    _, idx = score_case
    s = scores[idx]
    thres = META(method=method)

    labels = thres.eval(s)

    assert thres.thresh_ is None
    check_scores_normalized(thres.dscores_)
    check_labels(labels, s.shape)


# -----------------------
# Fit
# -----------------------

@pytest.mark.parametrize("method,score_case", param_grid)
def test_fit(scores, method, score_case):
    _, idx = score_case
    s = scores[idx]
    thres = META(method=method)

    thres.fit(s)

    check_fitted_attributes(thres, expect_thresh=False)
    check_labels(thres.labels_, s.shape)


# -----------------------
# Predict
# -----------------------

@pytest.mark.parametrize("method,score_case", param_grid)
def test_predict(scores, method, score_case):
    _, idx = score_case
    s = scores[idx]
    thres = META(method=method)

    thres.fit(s)
    pred = check_predict_consistency(thres, s)

    check_fitted_attributes(thres, expect_thresh=False)
    check_labels(pred, s.shape)


# -----------------------
# Train/Test
# -----------------------

@pytest.mark.parametrize("method,score_case", param_grid)
def test_test_data(data, scores, method, score_case):
    _, idx = score_case

    X_train, X_test, _, _ = data
    test_scores = build_test_scores(X_train, X_test)

    train_s = scores[idx]
    test_s = test_scores[idx]

    thres = META(method=method)
    thres.fit(train_s)

    pred = thres.predict(test_s)

    check_fitted_attributes(thres, expect_thresh=False)
    check_labels(pred, test_s.shape)


# -----------------------
# Save / Load
# -----------------------

@pytest.mark.parametrize("score_case", score_cases)
def test_save_and_load(tmp_path, scores, score_case):
    _, idx = score_case
    s = scores[idx]

    thres = META()
    thres.fit(s)

    file = tmp_path / "model.pkl"
    joblib.dump(thres, file)

    loaded = joblib.load(file)

    assert_equal(thres.predict(s), loaded.predict(s))
