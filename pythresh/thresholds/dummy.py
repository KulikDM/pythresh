import numpy as np

from .base import BaseThresholder
from .thresh_utility import cut


class DUMMY(BaseThresholder):
    r"""DUMMY class for dummy thresholder.

       Use the DUMMY thresholder to threshold based on a given contamination
       level. This is useful for benchmarking.

       Parameters
       ----------

       contam : float in (0., 1.0) or None, optional (default=None)
            The amount of contamination of the data set, i.e.
            the proportion of outliers in the data set. Used when fitting to
            define the threshold on the decision function. Default None sets
            no outliers to exist in the training data.

       fallback : str ('ignore', 'warn', 'raise'), optional (default='warn')
            The action to take for thresholders when their criterion are
            not met. In these cases when set to 'ignore' on eval and fit
            all train data is set to inliers and the threshold is set to
            max of the train scores + eps. Passing 'warn' will do the same as
            'ignore' but also produce a warning. If 'raise', the thresholder
            raises a ValueError.

       random_state : int, optional (default=1234)
            Random seed for the random number generators of the thresholders. Can also
            be set to None.

       Attributes
       ----------

       thresh_ : threshold value that separates inliers from outliers

       dscores_ : 1D array of decomposed decision scores


    """

    def __init__(self, contam=None, fallback='warn', random_state=1234):

        super().__init__(fallback=fallback)

        self.contam = 0 if contam is None else contam
        self.random_state = random_state
        np.random.seed(random_state)

    def eval(self, decision):
        """Outlier/inlier evaluation process for decision scores.

        Parameters
        ----------
        decision : np.array or list of shape (n_samples)
                   or np.array of shape (n_samples, n_detectors)
                   which are the decision scores from a
                   outlier detection.

        Returns
        -------
        outlier_labels : numpy array of shape (n_samples,)
            For each observation, tells whether or not
            it should be considered as an outlier according to the
            fitted model. 0 stands for inliers and 1 for outliers.
        """

        decision = self._data_setup(decision)

        eps = np.finfo(decision.dtype).eps
        perc = (1 - self.contam)*100
        limit = np.percentile(decision, perc) + eps

        self._check_threshold(limit)

        self.thresh_ = limit

        return cut(decision, limit)
