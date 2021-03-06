import numpy as np
from sklearn.utils import check_array
from .base import BaseThresholder
from .thresh_utility import normalize


class CLF(BaseThresholder):
    """CLF class for Trained Classifier thresholder.

       Use the trained linear classifier to evaluate a non-parametric means
       to threshold scores generated by the decision_scores where outliers
       are set to any value beyond 0. See :cite:`aggarwal2017clf` for details.
       
       Paramaters
       ----------

       Attributes
       ----------

       thres_ : threshold value that seperates inliers from outliers

    """

    def __init__(self):

        self.m = 4.0581548062264075
        self.c = -1.5357998356223497

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

        # Calculate expected y
        pred = self.m*decision + self.c

        # Determine labels
        pred[pred>0] = 1
        pred[pred<=0] = 0

        self.thresh_ = None

        return pred.astype(int)
