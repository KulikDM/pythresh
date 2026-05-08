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

from pythresh.thresholds.filter import FILTER

# -----------------------
# Param grid
# -----------------------

methods = ["gaussian", "savgol", "hilbert", "wiener",
           "medfilt", "decimate", "detrend", "resample"]

sigmas = ["auto", int(200**0.6), int(200**0.75)]
score_cases = [
    ("single", 0),
    ("multi", 1),
]

param_grid = list(product(methods, sigmas, score_cases))


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

@pytest.mark.parametrize("method,sigma,score_case", param_grid)
def test_eval(scores, method, sigma, score_case):
    _, idx = score_case
    s = scores[idx]

    thres = FILTER(
        method=method,
        sigma=sigma,
    )
    labels = thres.eval(s)

    assert thres.thresh_ is not None
    check_scores_normalized(thres.dscores_)
    check_labels(labels, s.shape)


# -----------------------
# Fit
# -----------------------

@pytest.mark.parametrize("score_case", score_cases)
def test_fit(scores, score_case):
    _, idx = score_case
    s = scores[idx]

    thres = FILTER()
    thres.fit(s)

    check_fitted_attributes(thres, expect_thresh=True)
    check_labels(thres.labels_, s.shape)


# -----------------------
# Predict
# -----------------------

@pytest.mark.parametrize("score_case", score_cases)
def test_predict(scores, score_case):
    _, idx = score_case
    s = scores[idx]

    thres = FILTER()
    thres.fit(s)

    pred = check_predict_consistency(thres, s)

    check_fitted_attributes(thres, expect_thresh=True)
    check_labels(pred, s.shape)


# -----------------------
# Train/Test
# -----------------------

@pytest.mark.parametrize("score_case", score_cases)
def test_test_data(data, scores, score_case):
    _, idx = score_case

    X_train, X_test, _, _ = data
    test_scores = build_test_scores(X_train, X_test)

    train_s = scores[idx]
    test_s = test_scores[idx]

    thres = FILTER()
    thres.fit(train_s)

    pred = thres.predict(test_s)

    check_fitted_attributes(thres, expect_thresh=True)
    check_labels(pred, test_s.shape)


# -----------------------
# Save / Load
# -----------------------

@pytest.mark.parametrize("score_case", score_cases)
def test_save_and_load(tmp_path, scores, score_case):
    _, idx = score_case
    s = scores[idx]

    thres = FILTER()
    thres.fit(s)

    file = tmp_path / "model.pkl"
    joblib.dump(thres, file)

    loaded = joblib.load(file)

    assert_equal(thres.predict(s), loaded.predict(s))
