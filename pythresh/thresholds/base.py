import abc

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from .thresh_utility import check_scores, cut, get_min_max, normalize


class BaseThresholder(BaseEstimator, metaclass=abc.ABCMeta):
    """Abstract class for all outlier detection thresholding algorithms.

       Parameters
       ----------

       Attributes
       ----------

       thresh_ : threshold value that separates inliers from outliers

       labels_ : binary array of labels for the fitted thresholder

       confidence_interval_ : lower and upper confidence interval of the contamination level

       dscores_ : 1D array of decomposed decision scores
    """

    @abc.abstractmethod
    def __init__(self):

        self.thresh_ = None
        self.confidence_interval_ = None
        self.dscores_ = None

        self._base_attrs = ['_prenorm', '_postnorm',
                            '_decomp', '_is_fitted',
                            'labels_']
        self._set_attributes(self._base_attrs, None)

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

    def fit(self, X, y=None):
        """Outlier/inlier fit process for decision scores.

        Parameters
        ----------
        decision : np.array or list of shape (n_samples)
                   or np.array of shape (n_samples, n_detectors)
                   which are the decision scores from a
                   outlier detection.
        """

        self._set_attributes(self._base_attrs, None)

        self.labels_ = self.eval(X)
        self._is_fitted = True

        return self

    def predict(self, X):
        """Outlier/inlier predict process for decision scores.

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

        check_is_fitted(self)

        if self.thresh_ is None:

            return self.eval(X)

        else:

            X = self._data_setup(X)
            return cut(X, self.thresh_)

    def __sklearn_is_fitted__(self):
        """Check fitted status and return a Boolean value."""
        return hasattr(self, '_is_fitted') and self._is_fitted

    def _data_setup(self, data):
        """Data preprocessing step to normalize and decompose data."""
        if self._is_fitted is None:
            self._set_attributes(self._base_attrs, None)

        self._set_norm(data, '_prenorm', return_norm=False)

        data, self._decomp = check_scores(data,
                                          self._decomp,
                                          self._prenorm[0],
                                          self._prenorm[1],
                                          random_state=self.random_state)

        data = self._set_norm(data, '_postnorm')

        if self._is_fitted is None:
            self.dscores_ = data

        return data

    def _set_norm(self, data, norm_attr, return_norm=True):
        """MinMax normalize data based on fitted data."""
        norm = getattr(self, norm_attr)
        if norm is None:
            norm = get_min_max(data)
            setattr(self, norm_attr, norm)
        if return_norm:
            return normalize(data, norm[0], norm[1])

    def _set_attributes(self, attrs, values):
        """Setting attributes required for fit predict tracking."""
        if isinstance(values, list):
            if len(attrs) != len(values):
                raise ValueError(
                    'Length of attribute list and value list must be the same')
            for attr, val in zip(attrs, values):
                setattr(self, attr, val)
        else:
            for attr in attrs:
                setattr(self, attr, values)
