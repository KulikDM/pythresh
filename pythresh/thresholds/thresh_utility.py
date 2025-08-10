import numpy as np
import scipy.stats as stats
from scipy.interpolate import interp1d
from scipy.special import ndtr
from sklearn.decomposition import TruncatedSVD
from sklearn.utils import check_array


def get_min_max(data):

    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)

    return min_val, max_val


def normalize(data, min_val=None, max_val=None):

    if min_val is None or max_val is None:
        min_val, max_val = get_min_max(data)

    normed = (data - min_val) / (max_val - min_val)

    return normed


def cut(decision, limit):

    labels = np.zeros(len(decision), dtype=int)

    labels[decision >= limit] = 1

    return labels


def gen_interp(x, y):

    interpolator = interp1d(x, y, kind='cubic',
                            fill_value='extrapolate')

    return interpolator


def gen_kde(data, lower, upper, size):

    insize = min(size, 5000)

    # Create a KDE of the data
    kde = stats.gaussian_kde(data)
    dat_range = np.linspace(lower, upper, insize)
    dat_eval = np.linspace(lower, upper, size)

    # Use interpolation for fast KDE upsampling
    if size > insize:
        interpolator = gen_interp(dat_range, kde(dat_range))
        return interpolator(dat_eval), dat_eval

    return kde(dat_eval), dat_eval


def gen_cdf(data, lower, upper, size):

    insize = min(size, 5000)

    # Create a KDE & CDF of the data
    kde = stats.gaussian_kde(data)
    dat_range = np.linspace(lower, upper, insize)
    dat_eval = np.linspace(lower, upper, size)

    cdf = np.array(tuple(ndtr(np.ravel(item - kde.dataset) / kde.factor).mean()
                         for item in dat_range))

    # Use interpolation for fast CDF upsampling
    if size > insize:
        interpolator = gen_interp(dat_range, cdf)
        return interpolator(dat_eval), dat_eval

    return cdf, dat_eval


def check_scores(decision, decomp=None, min_val=None, max_val=None, random_state=1234):

    # Check decision scores dimensionality and pre-process
    if (np.asarray(decision).ndim == 2) & (np.atleast_2d(decision).shape[1] > 1):

        decision = check_array(decision, ensure_2d=True)
        decision = normalize(decision, min_val, max_val)
        decision, decomp = decompose(decision, decomp, random_state)

    else:
        decision = check_array(decision, ensure_2d=False)

    return decision.squeeze(), decomp


def decompose(data, decomp=None, random_state=1234):

    # Decompose decision scores to 1D array for thresholding
    if decomp is None:
        decomp = TruncatedSVD(n_components=1, random_state=random_state)
        data = decomp.fit_transform(data)
    else:
        data = decomp.transform(data)

    return data, decomp
