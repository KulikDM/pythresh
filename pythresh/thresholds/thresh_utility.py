import numpy as np
import scipy.stats as stats
from scipy.special import ndtr


def normalize(data):

    return ((data - data.min()) / (data.max() - data.min()))


def cut(decision, limit):

    labels = np.zeros(len(decision), dtype=int)

    labels[decision>=limit] = 1

    return labels


def gen_kde(data, lower, upper, size):

    # Create a KDE of the data
    kde = stats.gaussian_kde(data)
    dat_range = np.linspace(lower,upper,size)

    return kde(dat_range), dat_range

def gen_cdf(data, lower, upper, size):

    # Create a KDE & CDF of the data
    kde = stats.gaussian_kde(data)
    dat_range = np.linspace(lower,upper,size)
    cdf = np.array(tuple(ndtr(np.ravel(item - kde.dataset) / kde.factor).mean()
            for item in dat_range))

    return cdf, dat_range
