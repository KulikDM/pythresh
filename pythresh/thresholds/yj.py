import numpy as np
import scipy.stats as stats
from sklearn.utils import check_array
from .base import BaseThresholder
from .thresh_utility import normalize, cut, gen_kde


class YJ(BaseThresholder):
    """YJ class for Yeo-Johnson transformation thresholder.

       Use the Yeo-Johnson transformation to evaluate
       a non-parametric means to threshold scores generated by the
       decision_scores where outliers are set to any value beyond the
       max value in the YJ transformed data

       Paramaters
       ----------

       Attributes
       ----------

       eval_: numpy array of binary labels of the training data. 0 stands
           for inliers and 1 for outliers/anomalies.

    """

    def __init__(self):

        super(YJ, self).__init__()

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

        # Generate KDE
        val, _ = gen_kde(decision,0,1,len(decision)*3)

        # Use Yeo-Johnson transformation to reshape distribution
        # iterate to get average transformation
        mean_s = np.zeros(len(val))
        for i in range(50):
            scores = stats.yeojohnson(val)[0]
            mean_s += scores
        mean_s = mean_s/50

        # Set limit to the max value from the transformation
        limit = np.max(mean_s)

        self.thresh_ = limit

        return cut(decision, limit)

