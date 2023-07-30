import numpy as np
import scipy.stats as stats

from .base import BaseThresholder
from .thresh_utility import check_scores, cut, normalize


class MAD(BaseThresholder):
    r"""MAD class for Median Absolute Deviation thresholder.

       Use the median absolute deviation to evaluate a non-parametric
       means to threshold scores generated by the decision_scores
       where outliers are set to any value beyond the mean plus the
       median absolute deviation over the standard deviation.
       See :cite:`archana2015mad` for details.

       Parameters
       ----------

       random_state : int, optional (default=1234)
            Random seed for the random number generators of the thresholders. Can also
            be set to None.

       Attributes
       ----------

       thresh_ : threshold value that separates inliers from outliers

       dscores_ : 1D array of decomposed decision scores

       Notes
       -----

       The median absolute deviation is defined as:

       .. math::

          MAD = med\lvert x - med(x)\rvert \mathrm{.}

       And the threshold is set such that:

       .. math::

          \mathrm{lim} = \bar{x} + \frac{MAD}{\sigma} \mathrm{,}

       where :math:`\bar{x}` and :math:`\sigma` are the mean and
       standard deviation of the scores respectively

    """

    def __init__(self, random_state=1234):

        self.random_state = random_state

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

        decision = check_scores(decision, random_state=self.random_state)

        decision = normalize(decision)

        self.dscores_ = decision

        # Set limit
        mean = np.mean(decision)
        limit = mean + \
            stats.median_abs_deviation(decision, scale=np.std(decision))

        self.thresh_ = limit

        return cut(decision, limit)
