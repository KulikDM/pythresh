import numpy as np
from numpy.testing import assert_equal
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.pca import PCA
from pyod.utils.data import generate_data

# -----------------------
# Data / Scores
# -----------------------

def generate_train_test_data(
    n_train=200,
    n_test=100,
    contamination=0.1,
    random_state=42,
):
    return generate_data(
        n_train=n_train,
        n_test=n_test,
        contamination=contamination,
        random_state=random_state,
    )


def build_clfs():
    return [
        KNN(),
        PCA(random_state=1234),
        IForest(random_state=1234),
    ]


def build_scores(X_train):
    clfs = build_clfs()

    single_score = clfs[0].fit(X_train).decision_scores_

    multiple_scores = np.vstack([
        clf.fit(X_train).decision_scores_ for clf in clfs
    ]).T

    return [single_score, multiple_scores]


def build_test_scores(X_train, X_test):
    clfs = build_clfs()

    single = clfs[0].fit(X_train).decision_function(X_test)

    multi = np.vstack([
        clf.fit(X_train).decision_function(X_test)
        for clf in clfs
    ]).T

    return [single, multi]


# -----------------------
# Assertions
# -----------------------

def check_labels(labels, scores_shape):
    assert labels.shape == scores_shape[:1]
    assert labels.min() in (0, 1)
    assert labels.max() in (0, 1)


def check_scores_normalized(dscores):
    assert dscores is not None
    assert 0 <= dscores.min() <= 1
    assert 0 <= dscores.max() <= 1


def check_fitted_attributes(thres, expect_thresh=True):
    assert thres.__sklearn_is_fitted__()
    assert thres.labels_ is not None

    if expect_thresh:
        assert thres.thresh_ is not None
    else:
        assert thres.thresh_ is None


def check_predict_consistency(thres, scores):
    pred = thres.predict(scores)
    assert_equal(thres.labels_, pred)
    return pred
