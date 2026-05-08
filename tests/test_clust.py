import importlib
from itertools import product

import joblib
import pytest
import sklearn
from numpy.testing import assert_equal
from utils import (
    build_scores,
    build_test_scores,
    check_fitted_attributes,
    check_labels,
    generate_train_test_data,
)

import pythresh.thresholds.clust as clust_module
from pythresh.thresholds.clust import CLUST

# -----------------------
# Param grid
# -----------------------

methods = ["agg", "birch", "bang", "bgm", "bsas", "dbscan", "ema", "hdbscan", "kmeans", "mbsas", "mshift", "optics", "somsc", "spec", "xmeans"]

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

    thres = CLUST(method=method)
    labels = thres.eval(s)

    assert thres.thresh_ is None
    assert thres.dscores_ is None
    check_labels(labels, s.shape)


# -----------------------
# Fit
# -----------------------


@pytest.mark.parametrize("method,score_case", param_grid)
def test_fit(scores, method, score_case):
    _, idx = score_case
    s = scores[idx]

    thres = CLUST(method=method)
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

    thres = CLUST(method=method)
    thres.fit(s)

    pred = thres.predict(s)

    check_fitted_attributes(thres, expect_thresh=False)
    check_labels(pred, s.shape)

    # NOTE: original test did NOT enforce equality
    # so we keep that behavior


# -----------------------
# Train/Test generalization
# -----------------------


@pytest.mark.parametrize("method,score_case", param_grid)
def test_test_data(data, scores, method, score_case):
    _, idx = score_case

    X_train, X_test, _, _ = data
    test_scores = build_test_scores(X_train, X_test)

    train_s = scores[idx]
    test_s = test_scores[idx]

    thres = CLUST(method=method)
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

    thres = CLUST()
    thres.fit(s)

    file = tmp_path / "model.pkl"
    joblib.dump(thres, file)

    loaded = joblib.load(file)

    assert_equal(thres.predict(s), loaded.predict(s))


# -----------------------
# HDBScan Missing Import
# -----------------------


def test_hdbscan_missing_raises(monkeypatch):

    monkeypatch.setattr(sklearn, "__version__", "1.2.0")

    importlib.reload(clust_module)

    assert clust_module.HDBSCAN is None

    from pythresh.thresholds.clust import CLUST

    with pytest.raises(ImportError, match="hdbscan"):
        CLUST(method="hdbscan")
