import abc


class BaseThresholder(metaclass=abc.ABCMeta):
    """Abstract class for all outlier detection thresholding algorithms.

       Parameters
       ----------

       Attributes
       ----------

       thresh_ : threshold value that separates inliers from outliers

       confidence_interval_ : lower and upper confidence interval of the contamination level

       dscores_ : 1D array of decomposed decision scores
    """

    @abc.abstractmethod
    def __init__(self):

        self.thresh_ = None
        self.confidence_interval_ = None
        self.dscores_ = None

    @abc.abstractmethod
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
