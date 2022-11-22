import numpy as np
from geomstats.learning.frechet_mean import FrechetMean
from geomstats.geometry.euclidean import Euclidean
from sklearn.utils import check_array
from .base import BaseThresholder
from .thresh_utility import normalize, cut, gen_kde


class KARCH(BaseThresholder):
    r"""KARCH class for Riemannian Center of Mass thresholder.

       Use the Karcher mean (Riemannian Center of Mass) to evaluate a
       non-parametric means to threshold scores generated by the
       decision_scores where outliers are set to any value beyond the
       Karcher mean + one standard deviation of the decision_scores.
       See :cite:`afsari2011karch` for details.

       Parameters
       ----------

       ndim : int, optional (default=2)
            Number of dimensions to construct the Euclidean manifold

       method : {'simple', 'complex'}, optional (default='complex')
            Method for computing the Karcher mean

            - 'simple':  Compute the Karcher mean using the 1D array of scores
            - 'complex': Compute the Karcher mean between a 2D array dot product of the scores and the sorted scores arrays

       Attributes
       ----------

       thresh_ : threshold value that separates inliers from outliers

       Notes
       -----

       The non-weighted Karcher mean which is also the Riemannian center of
       mass or the Riemannian geometric mean is defined to be a minimizer of:

       .. math::

           f(x) = \sum_{i=1}^n \delta^2(A,x) \mathrm{,}

       where :math:`A` is a member of a special orthoganal group where the group qualities are
       :math:`\left(X \in \mathbb{R}^{n \times n} \vert X^{\top}X=I \text{,} \mathrm{det}X=1 \right)`
       such that the group is a Lie group.

    """

    def __init__(self, ndim=2, method='complex'):

        self.ndim = ndim
        self.method = method

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

        # Create euclidean manifold and find Karcher mean
        manifold = Euclidean(dim=self.ndim)
        estimator = FrechetMean(metric=manifold.metric)

        if self.method == 'complex':

            # Create kde of scores
            val_data, _ = gen_kde(decision, 0, 1, len(decision))
            val_data = val_data.reshape(-1, 1)
            val_norm = np.sort(decision).reshape(1, -1)

            try:
                # find kde and score dot product and solve the
                vals = np.dot(val_data, val_norm)
                estimator.fit(vals)
                kmean = np.mean(estimator.estimate_)+np.std(decision)

            except ValueError:
                kmean = 1.0
        else:
            estimator.fit(decision.reshape(1, -1))
            kmean = np.mean(estimator.estimate_) + np.std(decision)

        # Get the mean of each dimension's Karcher mean
        limit = kmean

        self.thresh_ = limit

        return cut(decision, limit)
