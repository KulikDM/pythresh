import numpy as np
import scipy.stats as stats
from scipy.special import ndtr
from sklearn.decomposition import TruncatedSVD
from sklearn.utils import check_array


def normalize(data):

    return ((data - data.min(axis=0)) /
            (data.max(axis=0) - data.min(axis=0)))


def cut(decision, limit):

    labels = np.zeros(len(decision), dtype=int)

    labels[decision >= limit] = 1

    return labels


def gen_kde(data, lower, upper, size):

    # Create a KDE of the data
    kde = stats.gaussian_kde(data)
    dat_range = np.linspace(lower, upper, size)

    return kde(dat_range), dat_range


def gen_cdf(data, lower, upper, size):

    # Create a KDE & CDF of the data
    kde = stats.gaussian_kde(data)
    dat_range = np.linspace(lower, upper, size)
    cdf = np.array(tuple(ndtr(np.ravel(item - kde.dataset) / kde.factor).mean()
                         for item in dat_range))

    return cdf, dat_range


def check_scores(decision, random_state=1234):

    # Check decision scores dimensionality and pre-process
    if (np.asarray(decision).ndim == 2) & (np.atleast_2d(decision).shape[1] > 1):

        decision = check_array(decision, ensure_2d=True)
        decision = decompose(decision, random_state).squeeze()

    else:

        decision = check_array(decision, ensure_2d=False).squeeze()

    return decision


def decompose(data, random_state=1234):

    # Decompose decision scores to 1D array for thresholding
    decomp = TruncatedSVD(n_components=1, random_state=random_state)

    data = decomp.fit_transform(normalize(data))

    return data
