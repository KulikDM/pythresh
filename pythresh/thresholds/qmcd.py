import inspect

import numpy as np
import scipy.stats as stats

from .base import BaseThresholder
from .thresh_utility import cut


class QMCD(BaseThresholder):
    """QMCD class for Quasi-Monte Carlo Discrepancy thresholder.

       Use the quasi-Monte Carlo discrepancy to evaluate a non-parametric means
       to threshold scores generated by the decision_scores where outliers
       are set to any value beyond and percentile or quantile of one minus the
       discrepancy. See :cite:`iouchtchenko2019qmcd` for details.

       Parameters
       ----------

       method : {'CD', 'WD', 'MD', 'L2-star'}, optional (default='WD')
            Type of discrepancy

            - 'CD':      Centered Discrepancy
            - 'WD':      Wrap-around Discrepancy
            - 'MD':      Mix between CD/WD
            - 'L2-star': L2-star discrepancy

       lim : {'Q', 'P'}, optional (default='P')
            Filtering method to threshold scores using 1 - discrepancy

            - 'Q': Use quantile limiting
            - 'P': Use percentile limiting

       random_state : int, optional (default=1234)
            Random seed for the random number generators of the thresholders. Can also
            be set to None.

       Attributes
       ----------

       thresh_ : threshold value that separates inliers from outliers

       dscores_ : 1D array of decomposed decision scores

       Notes
       -----

       For the QMCD method it is assumed that the decision scores are pseudo-random
       values within a distribution :math:`M`. "Quasi-random" sequences, which are
       numbers that are better equidistributed for :math:`M` than pseudo-random numbers
       are used to calculate the decision scores discrepancy value.

       The discrepancy value is a uniformity criterion which is used to assess the space
       filling of a number of samples in a hypercube. It quantifies the distance between
       the continuous uniform distribution on a hypercube and the discrete uniform distribution
       on distinct sample points. Therefore, lower values mean better coverage of the parameter
       space.

       The QMCD method utilizes the discrepancy value by assuming that when it is at its lowest
       value (0) the "quasi-random" generated sequences and the decision scores are equally
       equidistributed across :math:`M`. Outliers are assumed to solely raise the discrepancy
       value. And therefore, the contamination of the dataset can be set as one minus the
       discrepancy.
    """

    def __init__(self, method='WD', lim='P', random_state=1234):

        super().__init__()
        self.method = method
        self.lim = lim
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

        # Get the quasi Monte-Carlo discrepancy of the labels
        disc = stats.qmc.discrepancy(
            decision.reshape(-1, 1), method=self.method)

        # Set the limit to either the quantile or percentile of 1-discrepancy
        if self.lim == 'Q':

            limit = np.quantile(decision, 1.0-disc)

        elif self.lim == 'P':

            arg_map = {'old': 'interpolation', 'new': 'method'}
            arg_name = (arg_map['new'] if 'method' in
                        inspect.signature(np.percentile).parameters
                        else arg_map['old'])

            limit = np.percentile(decision, (1.0-disc) *
                                  100, **{arg_name: 'midpoint'})

        self.thresh_ = limit

        return cut(decision, limit)
