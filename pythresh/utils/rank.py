import os
from os.path import dirname as up

import numpy as np
import scipy.stats as stats
import xgboost as xgb
from pyod.utils.utility import standardizer
from sklearn.metrics import calinski_harabasz_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import check_array

from pythresh.utils.rank_utility import (
    BREG_metric,
    Contam_score,
    GNB_score,
    mclain_rao_index
)


class RANK():
    """RANK class for ranking outlier detection and thresholding methods.

       Use the RANK class to rank outlier detection and thresholding methods' capabilities
       to provide the best matthews correlation with respect to the
       selected threshold method

       Parameters
       ----------

       od_models : {list of pyod.model classes}

       thresh : {pythresh.threshold class, float, int, list of pythresh.threshold classes, list of floats, list of ints}

       method : {'model', 'native'}, optional (default='model')

       weights : list of shape 3, optional (default=None)
             These weights are applied to the combined rank score. The first
             is for the cdf rankings, the second for the clust rankings, and
             the third for the mode rankings. Default applies equal weightings
             to all proxy-metrics. Only applies when method = 'native'.

       Attributes
       ----------

       cdf_rank_ : list of tuples shape (2, n_od_models) of cdf based rankings

       clust_rank_ : list of tuples shape (2, n_od_models) of cluster based rankings

       consensus_rank_ : list of tuples shape (2, n_od_models) of consensus based rankings

       Notes
       -----

       The RANK class ranks the outlier detection methods by evaluating
       three distinct proxy-metrics. The first proxy-metric looks at the outlier
       likelihood scores by class and measures the cumulative distribution
       separation using the the Wasserstein distance, and the Exponential Euclidean
       Bregman distance. The second proxy-metric looks at the relationship between the
       fitted features (X) and the evaluated classes (y) using the Calinski-Harabasz scores
       and between the outlier likihood score and the evaluated classes using the
       Mclain Rao Index. The third proxy-metric evaluates the class difference for each outlier
       detection and thresholding method with respect to consensus based metrics of all the evaluated
       outlier detection class labels. This is done using the mean contamination deviation based on
       TruncatedSVD decomposed scores and Gaussian Naive-Bayes trained consensus score

       Each proxy-metric is ranked separately and a final ranking is applied
       using all three proxy-metric to get a single ranked result of each
       outlier detection and thresholding method using the 'native' method. The model method uses
       a trained LambdaMART ranking model using all the proxy-metrics as input.

       Please note that the data is standardized using
       ``from pyod.utils.utility import standardizer`` during this ranking process

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

            # Initialize models
            clfs = [KNN(), IForest(), PCA(), MCD(), QMCD()]
            thres = FILTER()

            # Get rankings
            ranker = RANK(clfs, thres)
            rankings = ranker.eval(X)
    """

    def __init__(self, od_models, thresh, method='model', weights=None):

        self.od_models = od_models if isinstance(
            od_models, list) else [od_models]
        self.thr_models = thresh if isinstance(thresh, list) else [thresh]
        self.method = method

        no_weights = [1, 1, 1]
        self.weights = weights if weights is not None else no_weights

    def eval(self, X):
        """Outlier detection and thresholding method ranking.

        Parameters
        ----------
        X : np.array or list of input data of shape
            (n_samples, 1) or (n_samples, n_features)

        Returns
        -------
        rankings : list of tuples shape (2, n_od_models)
            For each combination of outlier detection model and
            thresholder ranked from best to worst in terms of
            performance
        """

        X = check_array(X, ensure_2d=True)
        X = standardizer(X)

        cdf_scores = []
        clust_scores = []
        all_scores = []
        all_labels = []
        models = []
        contam = []

        od_names = [od.__class__.__name__ for od in self.od_models]
        thr_names = [thr.__class__.__name__ for thr in self.thr_models]

        # Apply outlier detection and threshold
        for i, clf in enumerate(self.od_models):
            for j, thr in enumerate(self.thr_models):

                clf.fit(X)
                scores = clf.decision_scores_

                if not (isinstance(thr, (float, int))):

                    thr.fit(scores)
                    labels = thr.labels_

                else:

                    threshold = np.percentile(scores, 100 * (1 - thr))

                    labels = (scores > threshold).astype('int').ravel()

                # Normalize scores between 0 and 1
                scores = (scores - scores.min())/(scores.max() - scores.min())

                # Calculate metrics
                cdf_scores.append(self._cdf_metric(scores, labels))
                clust_scores.append(self._clust_metric(X, scores, labels))

                all_scores.append(scores)
                all_labels.append(labels)

                contam.append(labels.sum()/len(labels))
                models.append((od_names[i], thr_names[j]))

        # Get consensus based scores
        consensus_scores = self._consensus_metric(X, all_scores,
                                                  all_labels, contam)

        # Equally rank metrics
        cdf_rank = self._equi_rank(np.vstack(cdf_scores),
                                   [True, True])

        clust_rank = self._equi_rank(np.vstack(clust_scores),
                                     [True, True])

        consensus_rank = self._equi_rank(np.vstack(consensus_scores),
                                         [False, False])

        # Get combined metric rank
        comb = [cdf_rank, clust_rank, consensus_rank]
        combined_rank = self._rank_sort(comb, self.weights)

        # Map models to rankings
        ranked_models = [models[rank] for rank in combined_rank]

        self.cdf_rank_ = [models[rank] for rank in cdf_rank]
        self.clust_rank_ = [models[rank] for rank in clust_rank]
        self.consensus_rank_ = [models[rank] for rank in consensus_rank]

        if self.method == 'model':

            # Load trained ranking model
            clf = 'rank_model_XGB.json'
            parent = up(up(__file__))
            ranker = xgb.XGBRanker()
            ranker.load_model(os.path.join(parent, 'models', clf))

            # Transform data
            scaler = MinMaxScaler()

            model_data = np.concatenate([np.vstack(consensus_scores),
                                         np.vstack(cdf_scores),
                                         np.vstack(clust_scores)], axis=1)

            model_data = scaler.fit_transform(model_data)
            model_data[:, -1] = np.vstack(clust_scores)[:, -1]

            # Predict, rank, and map rankings
            pred = ranker.predict(model_data)
            pred = np.argsort(pred)

            ranked_models = [models[rank] for rank in pred]

        return ranked_models

    def _cdf_metric(self, scores, labels):
        """Calculate CDF based metrics."""

        if len(np.unique(labels)) == 1:
            return [-1e6, -1e6]

        # Sanity check on highly repetitive scores
        scores1 = scores[labels == 0]
        if np.all(scores1 == scores1[0]):
            scores1 = scores1 + np.linspace(1e-30, 2e-30, len(scores1))

        scores2 = scores[labels == 1]
        if len(scores2) < 2:
            return [-1e6, -1e6]

        if np.all(scores2 == scores2[0]):
            scores2 = scores2 - np.linspace(1e-30, 2e-30, len(scores2))

        # Generate KDEs of scores for both classes
        kde1 = stats.gaussian_kde(scores1)
        kde2 = stats.gaussian_kde(scores2)

        dat_range = np.linspace(0, 1, 5000)

        # Integrate KDEs to get CDFs
        cdf1 = np.array([kde1.integrate_box_1d(-1e-30, x)
                         for x in dat_range])

        cdf2 = np.array([kde2.integrate_box_1d(-1e-30, x)
                         for x in dat_range])

        # Calculate metrics
        was = stats.wasserstein_distance(cdf1, cdf2)
        breg = BREG_metric(cdf1, cdf2)

        return [was, breg]

    def _clust_metric(self, X, scores, labels):
        """Calculate clustering based metrics."""

        if len(np.unique(labels)) == 1:
            return [-1e6, -1e6]

        ch = calinski_harabasz_score(X, labels)
        mr = mclain_rao_index(scores, labels)

        return [ch, mr]

    def _consensus_metric(self, X, scores, labels, contam):
        """Calculate consensus based metrics."""

        gnb = GNB_score(X, labels)
        contam = Contam_score(scores, labels, contam)

        return np.vstack([gnb, contam]).T.tolist()

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
