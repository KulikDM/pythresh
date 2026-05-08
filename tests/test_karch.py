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

from pythresh.thresholds.karch import KARCH

# -----------------------
# Param grid
# -----------------------

methods = ["simple", "complex"]
ndims = list(range(1, 5))
score_cases = [
    ("single", 0),
    ("multi", 1),
]

param_grid = list(product(methods, ndims, score_cases))


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

@pytest.mark.parametrize("method,ndim,score_case", param_grid)
def test_eval(scores, method, ndim, score_case):
    _, idx = score_case
    s = scores[idx]

    thres = KARCH(method=method, ndim=ndim)
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

    thres = KARCH()
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

    thres = KARCH()
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

    thres = KARCH()
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

    thres = KARCH()
    thres.fit(s)

    file = tmp_path / "model.pkl"
    joblib.dump(thres, file)

    loaded = joblib.load(file)

    assert_equal(thres.predict(s), loaded.predict(s))

# -----------------------
# Memory Error
# -----------------------

def test_eval_memory_error_fallback(monkeypatch):
    def boom(*args, **kwargs):
        raise MemoryError()

    scores = np.random.rand(1000)

    th_simple = KARCH(method='simple')
    labels_simple = th_simple.eval(scores)

    monkeypatch.setattr(np, "dot", boom)

    th_complex = KARCH(method='complex')
    labels = th_complex.eval(scores)

    assert labels.shape == scores.shape
    assert_equal(labels, labels_simple)
