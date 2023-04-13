import numpy as np
import scipy.stats as stats
from sklearn.utils import check_array

from .base import BaseThresholder
from .thresh_utility import cut, normalize


class BOOT(BaseThresholder):
    r"""BOOT class for Bootstrapping thresholder.

       Use a bootstrapping based method to find a non-parametric means
       to threshold scores generated by the decision_scores where outliers
       are set to any value beyond the mean of the confidence intervals.
       See :cite:`martin2006boot` for details

       Parameters
       ----------
       random_state : int, optional (default=1234)
            Random seed for bootstrapping a confidence interval. Can also be set to None.

       Attributes
       ----------

       thresh_ : threshold value that separates inliers from outliers

       Notes
       -----

       The two sided bias-corrected and accelerated bootstrap confidence interval
       is calculated with a confidence level of 0.95. The statistic calculating
       the confidence interval is the standard deviation of the decision
       scores, with the statistic treating corresponding elements of the
       samples in the decision scores as paired

       The returned upper and lower confidence intervals are used to threshold
       the decision scores. Outliers are set to any value above the mean of the
       upper and lower confidence intervals.

       Examples
       --------
       The effects of randomness can affect the thresholder's output perfomance
       signicantly. Therefore, to alleviate the effects of randomness on the
       thresholder a combined model can be used with different random_state values.
       E.g.

       .. code:: python

            # train the KNN detector
            from pyod.models.knn import KNN
            from pythresh.thresholds.comb import COMB
            from pythresh.thresholds.boot import BOOT

            clf = KNN()
            clf.fit(X_train)

            # get outlier scores
            decision_scores = clf.decision_scores_  # raw outlier scores

            # get outlier labels with combined model
            thres = COMB(thresholders = [BOOT(random_state=1234),
            BOOT(random_state=42), BOOT(random_state=9685),
            BOOT(random_state=111222)])
            labels = thres.eval(decision_scores)

    """

    def __init__(self, random_state=1234):
        self.random_state = random_state

    def eval(self, decision):
        """Outlier/inlier evaluation process for decision scores.

        Parameters
        ----------
        decision : np.array or list of shape (n_samples)
                   which are the decision scores from a
                   outlier detection.

        Returns
        -------
        outlier_labels : numpy array of shape (n_samples,)
            For each observation, tells whether or not
            it should be considered as an outlier according to the
            fitted model. 0 stands for inliers and 1 for outliers.
        """

        decision = check_array(decision, ensure_2d=False)

        decision = normalize(decision)

        limit1, limit2 = stats.bootstrap(
            decision.reshape(1, -1),
            np.std,
            paired=True,
            random_state=self.random_state
        ).confidence_interval

        self.thresh_ = (limit1+limit2)/2

        return cut(decision, (limit1+limit2)/2)
