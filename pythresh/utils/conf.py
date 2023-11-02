import numpy as np
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.utils import check_array


class CONF():
    """CONF class for calculating the confidence of thresholding.

    Use the CONF class for evaluating the confidence of thresholding methods
    based on confidence-interval bounds to find datpoints that lie within the
    bounds and therefore are difficult to allocate whether they are true inliers
    or outliers for the selected confidence level

    Parameters
    ----------

    od_model : {pyod.model class}
        The outlier detection model

    thresh : {pythresh.threshold class}
        The thresholding method

    alpha : float, optional (default=0.05)
        Confidence level corresponding to the t-Student distribution map to sample

    split : float, optional (default=0.5)
        The test size thresholding test

    n_test : int, optional (default=100)
        The number of thresholding tests to build the confidence region

    random_state : int, optional (default=1234)
        Random seed for the random number generators of the thresholders. Can also
        be set to None.

    Attributes
    ----------

    cdf_rank_ : list of tuples shape (2, n_od_models) of cdf based rankings

    Notes
    -----

    The `CONF` class is designed for evaluating the confidence of thresholding methods within
    the context of outlier detection. It assesses the confidence of thresholding, a critical step
    in the outlier detection process. By sampling and testing different thresholds evaluated by the
    selected thresholding method, the class provides a confidence region for the selected threshold
    method. After building the confidence region, uncertain data points are identified. These are
    data points that lie within the confidence-interval bounds and may be challenging to classify
    as outliers or inliers.

    Examples
    --------

    .. code:: python

        # Import libraries
        from pyod.models.knn import KNN
        from pythresh.thresholds.filter import FILTER
        from pythresh.utils.conf import CONF

        # Initialize models
        clfs = KNN()
        thres = FILTER()

        # Get indices of datapoint outside of confidence bounds
        confidence = CONF(clfs, thres)
        uncertains = confidence.eval(X)
    """

    def __init__(self, od_model, thresh, alpha=0.05, split=0.5, n_test=100, random_state=1234):

        self.od_model = od_model
        self.thresh = thresh
        self.alpha = alpha
        self.split = split
        self.n_test = n_test
        self.random_state = random_state

    def eval(self, X):
        """Outlier detection and thresholding method confidence interval bounds.

        Parameters
        ----------
        X : np.array or list of input data of shape
            (n_samples, 1) or (n_samples, n_features)

        Returns
        -------
        rankings : list of indices of all datapoints that lie within the
            confidence-interval bounds and can be classified as "uncertain"
            datapoints
        """

        X = check_array(X, ensure_2d=True)

        # Fit model and threshold
        self.od_model.fit(X)

        scores = self.od_model.decision_scores_
        scores = ((scores - scores.min()) / (scores.max() - scores.min()))

        labels = self.thresh.eval(scores)

        # Initialize setup for tests
        boundings = []
        bound = self.thresh.thresh_
        index = np.arange(len(scores))

        for _ in range(self.n_test):

            if bound:

                thr_test = self._valid_thresh(scores, labels)

            else:

                thr_test = self._invalid_thresh(scores, labels, index)

            boundings.append(thr_test)

            self.random_state = self.random_state + \
                1 if self.random_state else self.random_state

        # Compute the confidence interval and identify uncertain data points
        if bound:

            n = len(boundings) - 1
            t_crit = stats.t.ppf(1-self.alpha, df=n)

            ci = t_crit * np.std(boundings)/(np.sqrt(n))

            uncertain_indices = np.where(
                (scores >= bound-ci) & (scores <= bound+ci))[0]

        else:

            boundings = np.vstack(boundings).T
            count = np.count_nonzero(~np.isnan(boundings), axis=1)

            cnf = np.nansum(boundings, axis=1)/np.maximum(count, 1)

            uncertain_indices = np.where(((labels == 1) & (cnf < 1-self.alpha) & (cnf > 0)) |
                                         ((labels == 0) & (cnf > self.alpha) & (cnf < 1)))[0]

        return uncertain_indices.tolist()

    def _valid_thresh(self, scores, labels):
        """Thresholding test for non-classification type thresholders."""

        # Split data and threshold
        _, sco_split, _, _ = train_test_split(scores, labels,
                                              test_size=self.split,
                                              stratify=labels,
                                              random_state=self.random_state)

        _ = self.thresh.eval(sco_split)

        return self.thresh.thresh_

    def _invalid_thresh(self, scores, labels, index):
        """Thresholding test for classification type thresholders."""

        # Split data and threshold
        info = np.zeros(len(scores)) * np.nan

        _, sco_split, _, _, _, ind = train_test_split(scores, labels, index,
                                                      test_size=self.split,
                                                      stratify=labels,
                                                      random_state=self.random_state)

        lbls = self.thresh.eval(sco_split)

        info[ind] = lbls

        return info
