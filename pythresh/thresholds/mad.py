import numpy as np
import scipy.stats as stats
from sklearn.utils import check_array
from .base import BaseThresholder
from .thresh_utility import normalize, cut


class MAD(BaseThresholder):
    """MAD class for Median Absolute Deviation thresholder.

       Use the median absolute deviation to evaluate a non-parametric
       means to threshold scores generated by the decision_scores
       where outliers are set to any value beyond 3 times the median
       absolute deviation

       Paramaters
       ----------

       Attributes
       ----------

       eval_: numpy array of binary labels of the training data. 0 stands
           for inliers and 1 for outliers/anomalies.

    """

    def __init__(self):

        super(MAD, self).__init__()

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

        # Set limit equal to 3 median abolute deviations
        limit = 3*stats.median_abs_deviation(decision, scale='normal')

        self.thresh_ = limit

        return cut(decision, limit)

