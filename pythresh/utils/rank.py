import numpy as np
import scipy.spatial.distance as distance
import scipy.special as special
import scipy.stats as stats
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score
)
from sklearn.utils import check_array


class RANK():
    """RANK class for ranking outlier detection methods.

       Use the RANK class to rank outlier detection methods' capabilities
       to provide the best matthews correlation with respect to the
       selected threshold method

       Parameters
       ----------

       od_models : list, optional (default=None)

       thresh : {pythresh.threshold class, float, int}, optional (default=None)

       weights : list of shape 3, optional (default=None)
             These weights are applied to the combined rank score. The first
             is for the cdf rankings, the second for the clust rankings, and
             the third for the mode rankings. Default applies equal rankings
             to all criterion.

       Attributes
       ----------

       cdf_rank_ : list of shape (n_od_models) of cdf based rankings

       clust_rank_ : list of shape (n_od_models) of cluster based rankings

       mode_rank_ : list of shape (n_od_models) of mode based rankings

       Notes
       -----

       The RANK class ranks the outlier detection methods by evaluating
       three distinct criterion. The first criterion looks at the outlier
       likelihood scores by class and measures the cumulative distribution
       separation using the Jensen-Shannon distance, the Wasserstein
       distance, and the Lukaszyk-Karmowski metric for normal distributions.
       The second criterion looks at the relationship between the fitted
       features (X) and the evaluated classes (y) using the Silhouette,
       Davies-Bouldin, and the Calinski-Harabasz scores. The third criterion
       evaluates the class difference for each outlier detection method with
       respect to the mode of all the evaluated outlier detection class labels.

       Each criterion is ranked separately and a final ranking is applied
       using all three criterion to get a single ranked result of each
       outlier detection method

       Examples
       --------

       .. code:: python

            # Import libraries
            from pyod.models.knn import KNN
            from pyod.models.iforest import IForest
            from pyod.models.pca import PCA
            from pyod.models.mcd import MCD
            from pyod.models.qmcd import QMCD
            from pythresh.thresholds.filter import FILTER
            from pythresh.utils.ranking import RANK

            # Initalize models
            clfs = [KNN(), IForest(), PCA(), MCD(), QMCD()
            thres = FILTER()

            # Get rankings
            ranker = RANK(clfs, thres)
            rankings = ranker.eval(X)
    """

    def __init__(self, od_models=None, thresh=None, weights=None):

        self.od_models = od_models
        self.thresh = thresh

        no_weights = [1, 1, 1]
        self.weights = weights if weights is not None else no_weights

    def eval(self, X):
        """Outlier detection method ranking.

        Parameters
        ----------
        X : np.array or list of input data of shape
            (n_samples, 1) or (n_samples, n_features)

        Returns
        -------
        ranked_models : list of shape (n_od_models)
            For each outlier detection model ranked from
            best to worst in terms of performance with
            respect to the selected threshold method
        """

        X = check_array(X, ensure_2d=True)

        cdf_scores = []
        clust_scores = []
        all_labels = []

        # Apply outlier detection and threshold
        for clf in self.od_models:

            clf.fit(X)
            scores = clf.decision_scores_

            if not (isinstance(self.thresh, (float, int))):

                labels = self.thresh.eval(scores)

            else:

                threshold = np.percentile(scores, 100 * (1 - self.thresh))

                labels = (scores > threshold).astype('int').ravel()

            # Normalize scores between 0 and 1
            scores = (scores - scores.min())/(scores.max() - scores.min())

            # Calculate metrics
            cdf_scores.append(self._cdf_metric(scores, labels))
            clust_scores.append(self._clust_metric(X, labels))
            all_labels.append(labels)

        # Get sum of the difference from the mode
        mode = stats.mode(all_labels,
                          axis=0)[0].squeeze()

        mode_diff = np.sum(np.abs(np.vstack(all_labels) - mode), axis=1)

        # Equally rank metrics
        cdf_rank = self._equi_rank(np.vstack(cdf_scores),
                                   [True, True, True])

        clust_rank = self._equi_rank(np.vstack(clust_scores),
                                     [True, False, True])

        mode_rank = self._equi_rank(mode_diff.reshape(-1, 1), [False])

        # Get combined metric rank
        comb = [cdf_rank, clust_rank, mode_rank]
        combined_rank = self._rank_sort(comb, self.weights)

        # Map od models to rankings
        od_names = [od.__class__.__name__ for od in self.od_models]
        ranked_models = [od_names[rank] for rank in combined_rank]

        self.cdf_rank_ = [od_names[rank] for rank in cdf_rank]
        self.clust_rank_ = [od_names[rank] for rank in clust_rank]
        self.mode_rank_ = [od_names[rank] for rank in mode_rank]

        return ranked_models

    def _cdf_metric(self, scores, labels):
        """Calculate CDF based metrics."""

        if len(np.unique(labels)) == 1:
            return [-1e6, -1e6, -1e6]

        # Sanity check on highly repetitive scores
        scores1 = scores[labels == 0]
        if np.all(scores1 == scores1[0]):
            scores1 = scores1 + np.linspace(1e-30, 2e-30, len(scores1))

        scores2 = scores[labels == 1]
        if np.all(scores2 == scores2[0]):
            scores2 = scores2 - np.linspace(1e-30, 2e-30, len(scores2))

        # Generate KDEs of scores for both classed
        kde1 = stats.gaussian_kde(scores1)
        kde2 = stats.gaussian_kde(scores2)

        dat_range = np.linspace(0, 1, 5000)

        # Integrate KDEs to get CDFs
        cdf1 = np.array([kde1.integrate_box_1d(-1e-30, x)
                         for x in dat_range])

        cdf2 = np.array([kde2.integrate_box_1d(-1e-30, x)
                         for x in dat_range])

        # Calculate metrics
        lk = self._LK_metric(cdf1, cdf2)

        was = stats.wasserstein_distance(cdf1, cdf2)

        js = distance.jensenshannon(cdf1, cdf2)

        return [lk, was, js]

    def _clust_metric(self, X, labels):
        """Calculate clustering based metrics."""

        if len(np.unique(labels)) == 1:
            return [-1e6, 1e6, -1e6]

        sil = silhouette_score(X, labels)
        db = davies_bouldin_score(X, labels)
        ch = calinski_harabasz_score(X, labels)

        return [sil, db, ch]

    def _equi_rank(self, data, order):
        """Get equally weighted rankings from metrics."""

        # Get indexes of best to worst for data
        sortings = []

        for i in range(data.shape[1]):

            check = np.argsort(data[:, i].squeeze())

            if order[i]:
                check = check[::-1]

            sortings.append(check.tolist())

        sorted_scores = self._rank_sort(sortings, [1, 1, 1])

        return sorted_scores

    def _rank_sort(self, sortings, weights):
        """Sort weighted rankings."""

        # Get unique index values for ranking
        unique_values = {value for ls in sortings for value in ls}
        scores = {value: 0 for value in unique_values}

        # Get equally weighted rank
        for value in unique_values:
            for j, ls in enumerate(sortings):
                if value in ls:
                    scores[value] += weights[j] * ls.index(value)

        # Get best to worst performing indexes
        sorted_scores = sorted(scores.keys(),
                               key=lambda x: scores[x])

        return sorted_scores

    def _LK_metric(self, cdf1, cdf2):
        """Calculate the Lukaszyk-Karmowski metric for normal distributions."""

        # Get expected values for both distributions
        rng = np.linspace(0, 1, len(cdf1))
        exp_dist1 = (rng*cdf1).sum()/cdf1.sum()
        exp_dist2 = (rng*cdf2).sum()/cdf2.sum()

        nu_xy = np.abs(exp_dist1-exp_dist2)

        # STD is same for both distributions
        std = np.std(rng)

        # Get the LK distance
        return (nu_xy + 2*std/np.sqrt(np.pi)*np.exp(-nu_xy**2/(4*std**2))
                - nu_xy*special.erfc(nu_xy/(2*std)))
