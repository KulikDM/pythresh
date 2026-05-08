from itertools import product

import joblib
import numpy as np
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

from pythresh.thresholds.regr import REGR

# -----------------------
# Param grid
# -----------------------

methods = ["siegel", "theil"]
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

    thres = REGR(method=method)
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

    thres = REGR()
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

    thres = REGR()
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

    thres = REGR()
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

    thres = REGR()
    thres.fit(s)

    file = tmp_path / "model.pkl"
    joblib.dump(thres, file)

    loaded = joblib.load(file)

    assert_equal(thres.predict(s), loaded.predict(s))


# -----------------------
# Memory Fallback
# -----------------------


def test_eval_memory_error_fallback(monkeypatch):
    import scipy.stats as stats

    def boom(*args, **kwargs):
        raise MemoryError()

    monkeypatch.setattr(stats, "siegelslopes", boom, raising=True)

    assert stats.siegelslopes is boom

    scores = np.random.rand(1000).astype(float)

    thres = REGR(method="siegel")
    labels = thres.eval(scores)

    eps = np.finfo(scores.dtype).eps

    assert thres.thresh_ == 1.0 + eps
    assert labels.shape == scores.shape
    assert_equal(labels, 0)
