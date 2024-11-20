import numpy as np
import scipy.stats as stats

from .base import BaseThresholder
from .thresh_utility import check_scores, normalize


class ZSCORE(BaseThresholder):
    r"""ZSCORE class for ZSCORE thresholder.

       Use the zscore to evaluate a non-parametric means to threshold
       scores generated by the decision_scores where outliers are set
       to any value beyond a zscore of one.
       See :cite:`bagdonavicius2020zscore` for details.

       Parameters
       ----------
       factor : int, optional (default=1)
            The factor to multiply the zscore by to set the threshold.
            The default is 1.
       random_state : int, optional (default=1234)
            Random seed for the random number generators of the thresholders. Can also
            be set to None.

       Attributes
       ----------

       thresh_ : threshold value that separates inliers from outliers

       dscores_ : 1D array of decomposed decision scores

       Notes
       -----

       The z-score can be calculated as follows:

       .. math::

           Z = \frac{x-\bar{x}}{\sigma} \mathrm{,}

       where :math:`\bar{x}` and :math:`\sigma` are the mean and the
       standard deviation of the decision scores respectively. The threshold
       is set that any value beyond an absolute z-score of 1 is considered
       and outlier.

    """

    def __init__(self, factor=1, random_state=1234):

        self.factor = factor
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

        # Get the zscore of the decision scores
        zscore = stats.zscore(decision)

        # Set the limit to where the zscore is greater than the factor
        labels = np.zeros(len(decision), dtype=int)
        mask = np.where(zscore >= self.factor)
        labels[mask] = 1

        self.thresh_ = np.min(labels[labels == 1])

        return labels
