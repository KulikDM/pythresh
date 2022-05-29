from __future__ import division
from __future__ import print_function

import warnings

import abc
import six

@six.add_metaclass(abc.ABCMeta)
class BaseThresholder(object):
    """Abstract class for all outlier detection thresholding algorithms.

    .. warning::
    pyod would stop supporting Python 2 in the future. Consider move to
    Python 3.5+.

       
       Paramaters
       ----------

       Attributes
       ----------

       eval_: numpy array of binary labels of the training data. 0 stands
           for inliers and 1 for outliers/anomalies.

    """

    @abc.abstractmethod
    def __init__(self):

        self.thresh_ = None

    @abc.abstractmethod
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

        pass

