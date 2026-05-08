import warnings
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

from pythresh.thresholds.comb import COMB

# -----------------------
# Param grid
# -----------------------

methods = ["mean", "median", "mode", "bagged", "stacked"]
max_contams = [0.25, 0.5, 0.01]
fallbacks = ["ignore", "warn", "raise"]
score_cases = [
    ("single", 0),
    ("multi", 1),
]

param_grid = list(product(methods, max_contams, score_cases))


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


@pytest.mark.parametrize("method,max_contam,score_case", param_grid)
def test_eval(scores, method, max_contam, score_case):
    _, idx = score_case
    s = scores[idx]
    thres = COMB(
        method=method,
        max_contam=max_contam,
    )

    labels = thres.eval(s)

    check_scores_normalized(thres.dscores_)
    check_labels(labels, s.shape)


# -----------------------
# Fit
# -----------------------


@pytest.mark.parametrize("method,max_contam,score_case", param_grid)
def test_fit(scores, method, max_contam, score_case):
    _, idx = score_case
    s = scores[idx]
    thres = COMB(
        method=method,
        max_contam=max_contam,
    )

    thres.fit(s)

    is_thresh = thres.thresh_ is not None
    check_fitted_attributes(thres, expect_thresh=is_thresh)
    check_labels(thres.labels_, s.shape)


# -----------------------
# Predict
# -----------------------


@pytest.mark.parametrize("method,max_contam,score_case", param_grid)
def test_predict(scores, method, max_contam, score_case):
    _, idx = score_case
    s = scores[idx]
    thres = COMB(
        method=method,
        max_contam=max_contam,
    )

    thres.fit(s)
    pred = check_predict_consistency(thres, s)

    is_thresh = thres.thresh_ is not None
    check_fitted_attributes(thres, expect_thresh=is_thresh)
    check_labels(pred, s.shape)


# -----------------------
# Train/Test
# -----------------------


@pytest.mark.parametrize("method,max_contam,score_case", param_grid)
def test_test_data(data, scores, method, max_contam, score_case):
    _, idx = score_case

    X_train, X_test, _, _ = data
    test_scores = build_test_scores(X_train, X_test)

    train_s = scores[idx]
    test_s = test_scores[idx]

    thres = COMB(
        method=method,
        max_contam=max_contam,
    )
    thres.fit(train_s)

    pred = thres.predict(test_s)

    is_thresh = thres.thresh_ is not None
    check_fitted_attributes(thres, expect_thresh=is_thresh)
    check_labels(pred, test_s.shape)


# -----------------------
# Save / Load
# -----------------------


@pytest.mark.parametrize("score_case", score_cases)
def test_save_and_load(tmp_path, scores, score_case):
    _, idx = score_case
    s = scores[idx]

    thres = COMB()
    thres.fit(s)

    file = tmp_path / "model.pkl"
    joblib.dump(thres, file)

    loaded = joblib.load(file)

    assert_equal(thres.predict(s), loaded.predict(s))


# -----------------------
# Test Fallbacks
# -----------------------


@pytest.mark.parametrize("fallback", fallbacks)
def test_fit_threshold_fallback_behavior(scores, fallback):
    s = scores[0]

    thres = COMB(max_contam=0.01, fallback=fallback)

    trigger_data = s

    if fallback == "ignore":
        # should run silently
        thres.fit(trigger_data)

    elif fallback == "warn":
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            thres.fit(trigger_data)

            assert len(w) >= 1
            assert any("outside the range" in str(wi.message) for wi in w)

    elif fallback == "raise":
        with pytest.raises(ValueError, match="outside the range"):
            thres.fit(trigger_data)
