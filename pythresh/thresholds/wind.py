import numpy as np
from scipy import integrate, stats
from sklearn.utils import check_array
from .base import BaseThresholder
from .thresh_utility import normalize, cut, gen_kde


class WIND(BaseThresholder):
    r"""WIND class for topological Winding number thresholder.

       Use the topological winding number (with respect to the origin) to
       evaluate a non-parametric means to threshold scores generated by
       the decision_scores where outliers are set to any value beyond the
       mean intersection point calculated from the winding number.
       See :cite:`jacobson2013wind` for details.

       Parameters
       ----------

       random_state : int, optional (default=1234)
            Random seed for the normal distribution. Can also be set to None.

       Attributes
       ----------

       thresh_ : threshold value that separates inliers from outliers

       Notes
       -----

       The topological winding number or the degree of a continuous mapping. It is an
       integer sum of the number of completed/closed counterclockwise rotations in a plane
       around a point. And is given by,

       .. math::

           \mathrm{d}\theta = \frac{1}{r^2} \left(x\mathrm{d}y - y\mathrm{d}x \right) \mathrm{,}

       where :math:`r^2 = x^2 + y^2`

       .. math::

           wind(\gamma,0) = \frac{1}{2\pi} \oint_\gamma \mathrm{d}\theta

       The winding number intuitively captures self-intersections/contours, with a change in the
       distribution of the dataset or shift from inliers to outliers relating to these intersections.
       With this, it is assumed that if an intersection exists, then adjacent/incident regions
       must have different region labels. Since multiple intersection regions may exist. The
       threshold between inliers and outliers is taken as the mean intersection point.

    """

    def __init__(self, random_state=1234):
        super(WIND, self).__init__()
        self.random_state = random_state

    def eval(self, decision):
        """Outlier/inlier evaluation process for decision scores.

        Parameters
        ----------

        Returns
        -------
        outlier_labels : numpy array of shape (n_samples,)
            For each observation, tells whether or not
            it should be considered as an outlier according to the
            fitted model. 0 stands for inliers and 1 for outliers.
        """

        decision = check_array(decision, ensure_2d=False)

        decision = normalize(decision)

        # Create a normal distribution and normalize
        size = min(len(decision), 1500)
        norm = stats.norm.rvs(size=size, loc=0.0, scale=1.0,
                              random_state=self.random_state)
        norm = normalize(norm)

        # Create a KDE of the labels and the normal distribution
        # Generate KDE
        val_data, dat_range = gen_kde(decision, 0, 1, len(decision)*3)
        val_norm, _ = gen_kde(norm, 0, 1, len(decision)*3)

        # Get the rsquared value
        r2 = val_data**2 + val_norm**2

        val_data = val_data/np.max(val_data)
        val_norm = val_norm/np.max(val_norm)

        # Find the first derivatives of the decision and norm kdes
        # with respect to the decision scores
        deriv_data = np.gradient(val_data, dat_range[1]-dat_range[0])
        deriv_norm = np.gradient(val_norm, dat_range[1]-dat_range[0])

        # Compute integrand
        integrand = self._dtheta(
            val_data, val_norm, deriv_data, deriv_norm, r2)

        # Integrate to find winding numbers mean intersection point
        limit = integrate.simpson(integrand)/np.sum((val_data+val_norm)/2)

        self.thresh_ = limit

        return cut(decision, limit)

    def _dtheta(self, x, y, dx, dy, r2):
        """Calculate dtheta for the integrand"""
        return (x*dy - y*dx)/r2
