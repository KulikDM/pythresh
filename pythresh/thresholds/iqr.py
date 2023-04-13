import inspect

import numpy as np
from sklearn.utils import check_array

from .base import BaseThresholder
from .thresh_utility import cut, normalize


class IQR(BaseThresholder):
    r"""IQR class for Inter-Qaurtile Region thresholder.

       Use the inter-quartile region to evaluate a non-parametric
       means to threshold scores generated by the decision_scores
       where outliers are set to any value beyond the third quartile
       plus 1.5 times the inter-quartile region.
       See :cite:`bardet2015iqr` for details.

       Parameters
       ----------

       Attributes
       ----------

       thresh_ : threshold value that separates inliers from outliers

       Notes
       -----

       The inter-quartile region is given as:

       .. math::

           IQR = \lvert Q_3-Q_1 \rvert

       where :math:`Q_1` and :math:`Q_3` are the first and third quartile
       respectively. The threshold for the decision scores is set as:

       .. math::

           t = Q_3 + 1.5 IQR

    """

    def __init__(self):

        super().__init__()

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

        arg_map = {'old': 'interpolation', 'new': 'method'}
        arg_name = (arg_map['new'] if 'method' in
                    inspect.signature(np.percentile).parameters
                    else arg_map['old'])

        # First quartile (Q1)
        P1 = np.percentile(decision, 25, **{arg_name: 'midpoint'})

        # Third quartile (Q3)
        P3 = np.percentile(decision, 75, **{arg_name: 'midpoint'})

        # Calculate IQR and generate limit
        iqr = abs(P3-P1)
        limit = P3 + 1.5*iqr

        self.thresh_ = limit

        return cut(decision, limit)
