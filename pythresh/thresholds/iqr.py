import numpy as np
from sklearn.utils import check_array
from .base import BaseThresholder
from .thresh_utility import normalize, cut


class IQR(BaseThresholder):
    """IQR class for Inter-Qaurtile Region thresholder.

       Use the inter-quartile region to evaluate a non-parametric
       means to threshold scores generated by the decision_scores
       where outliers are set to any value beyond the third quartile
       plus 1.5 times the inter-quartile region

       Paramaters
       ----------

       Attributes
       ----------

       eval_: numpy array of binary labels of the training data. 0 stands
           for inliers and 1 for outliers/anomalies.

    """

    def __init__(self):

        super(IQR, self).__init__()

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

        try:
            # First quartile (Q1)
            P1 = np.percentile(decision, 25, interpolation='midpoint')
  
            # Third quartile (Q3)
            P3 = np.percentile(decision, 75, interpolation='midpoint')
        except TypeError:
            # First quartile (Q1)
            P1 = np.percentile(decision, 25, method='midpoint')
  
            # Third quartile (Q3)
            P3 = np.percentile(decision, 75, method='midpoint')

        # Calculate IQR and generate limit
        iqr = abs(P3-P1)
        limit = P3 + 1.5*iqr

        self.thresh_ = limit

        return cut(decision, limit)
