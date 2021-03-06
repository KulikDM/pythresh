import numpy as np
import scipy.stats as stats
import scipy.special as special
from scipy.integrate import simpson
import scipy.spatial.distance as distance
from sklearn.covariance import MinCovDet
from itertools import combinations
from sklearn.utils import check_array
from .base import BaseThresholder
from .thresh_utility import normalize, cut, gen_kde, gen_cdf

class DSN(BaseThresholder):
    """DSN class for Distance Shift from Normal thresholder.

       Use the distance shift from normal to evaluate a non-parametric means
       to threshold scores generated by the decision_scores where outliers
       are set to any value beyond the distance calculated by the selected
       metric. See :cite:`amagata2021dsn` for details.
       
       Paramaters
       ----------

       metric : {'JS', 'WS', 'ENG', 'BHT', 'HLL', 'HI', 'LK', 'LP', 'MAH', 'TMT', 'RES', 'KS'}, optional (default='JS')
            Metric to use for distance computation
        
            - 'JS':  Jensen-Shannon distance
            - 'WS':  Wasserstein or Earth Movers distance
            - 'ENG': Energy distance
            - 'BHT': Bhattacharyya distance
            - 'HLL': Hellinger distance
            - 'HI':  Histogram intersection distance
            - 'LK':  Lukaszyk-Karmowski metric for normal distributions
            - 'LP':  Levy-Prokhorov metric
            - 'MAH': Mahalanobis distance
            - 'TMT': Tanimoto distance
            - 'RES': Studentized residual distance
            - 'KS':  Kolmogorov-Smirnov distance

       Attributes
       ----------

       thres_ : threshold value that seperates inliers from outliers

    """

    def __init__(self, metric='JS'):

        super(DSN, self).__init__()
        self.metric = metric
        self.metric_funcs = {'JS': self._JS_metric, 'WS': self._WS_metric,
                             'ENG': self._ENG_metric, 'BHT': self._BHT_metric,
                             'HLL': self._HLL_metric, 'HI': self._HI_metric,
                             'LK': self._LK_metric, 'LP': self._LP_metric,
                             'MAH': self._MAH_metric, 'TMT': self._TMT_metric,
                             'RES': self._RES_metric, 'KS': self._KS_metric}

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

        decision = np.sort(normalize(decision))

        #Create a normal distribution and normalize
        size = min(len(decision),1500)
        norm = stats.norm.rvs(size=size, loc=0.0, scale=1.0, random_state=1234)
        norm = np.sort(normalize(norm))

        n = 3

        if self.metric in ['JS','BHT']:
            # Create a KDE of the decision scores and the normal distribution
            # Generate KDE

            if self.metric=='LP':
                n=1

            val_data, _ = gen_kde(decision,0,1,len(decision)*n)
            val_norm, _ = gen_kde(norm,0,1,len(decision)*n)
            
        else:
            # Create a KDE of the decision scores and the normal distribution
            # Generate CDF
            val_data, _ = gen_cdf(decision,0,1,len(decision)*n)
            val_norm, _ = gen_cdf(norm,0,1,len(decision)*n)

        limit = self.metric_funcs[str(self.metric)](val_data, val_norm)

        self.thresh_ = limit

        return cut(decision, limit)
    
    def _JS_metric(self, val_data, val_norm):
        """Calculate the Jensen-Shannon distance"""

        return 1-distance.jensenshannon(val_data, val_norm)

    def _WS_metric(self, val_data, val_norm):
        """Calculate the Wasserstein or Earth Movers distance"""

        return stats.wasserstein_distance(val_data, val_norm)

    def _ENG_metric(self, val_data, val_norm):
        """Calculate the Energy distance"""

        return stats.energy_distance(val_data, val_norm)

    def _BHT_metric(self, val_data, val_norm):
        """Calculate the Bhattacharyya distance"""

        bht = simpson(np.sqrt(val_data*val_norm), dx=1/len(val_data))
        
        return np.log1p(bht)

    def _HLL_metric(self, val_data, val_norm):
        """Calculate the Hellinger distance"""

        val_data = val_data/np.sum(val_data)
        val_norm = val_norm/np.sum(val_norm)

        return (distance.euclidean(np.sqrt(val_data), np.sqrt(val_norm))
                        /np.sqrt(2))

    def _HI_metric(self, val_data, val_norm):
        """Calculate the Histogram intersection distance"""

        val_data = val_data/np.sum(val_data)
        val_norm = val_norm/np.sum(val_norm)
        
        return 1-np.sum(np.minimum(val_data,val_norm))

    def _LK_metric(self, val_data, val_norm):
        """Calculate the Lukaszyk-Karmowski metric for normal distributions"""
                
        # Get expected values for both distributions
        rng = np.linspace(0,1,len(val_data))
        exp_data = (rng*val_data).sum()/val_data.sum()
        exp_norm = (rng*val_norm).sum()/val_norm.sum()
                
        nu_xy = np.abs(exp_data-exp_norm)

        # STD is same for both distributions
        std = np.std(rng)

        # Get the LK distance
        return (nu_xy + 2*std/np.sqrt(np.pi)*np.exp(-nu_xy**2/(4*std**2))
                        - nu_xy*special.erfc(nu_xy/(2*std)))

    def _LP_metric(self, val_data, val_norm):
        """Calculate the Levy-Prokhorov metric"""

        # Get the edges for the complete graphs of the datasets
        f1 = np.array(list(combinations(val_data.tolist(),2)))
        f2 = np.array(list(combinations(val_norm.tolist(),2)))

        return (distance.directed_hausdorff(f1,f2)[0]/
                distance.correlation(val_data,val_norm))

    def _MAH_metric(self, val_data, val_norm):
        """Calculate the Mahalanobis distance"""

        # fit a Minimum Covariance Determinant (MCD) robust estimator to data 
        robust_cov = MinCovDet().fit(np.array([val_norm]).T)
    
        # Get the Mahalanobis distance
        dist = robust_cov.mahalanobis(np.array([val_data]).T)
        
        return 1-np.mean(dist)/np.max(dist)

    def _TMT_metric(self, val_data, val_norm):
        """Calculate the Tanimoto distance"""

        val_data = val_data/np.sum(val_data)
        val_norm = val_norm/np.sum(val_norm)
        
        p = np.sum(val_data)
        q = np.sum(val_norm)
        m = np.sum(np.minimum(val_data,val_norm))

        return (p+q-2*m)/(p+q-m)

    def _RES_metric(self, val_data, val_norm):
        """Calculate the studentized residual distance"""

        mean_X = np.mean(val_data)
        mean_Y = np.mean(val_norm)
        n = len(val_data)
        
        diff_mean_sqr = np.dot((val_data - mean_X), (val_data - mean_X))
        beta1 = np.dot((val_data - mean_X), (val_norm - mean_Y)) / diff_mean_sqr
        beta0 = mean_Y - beta1 * mean_X
        
        y_hat = beta0 + beta1 * val_data
        residuals = val_norm - y_hat
        
        h_ii = (val_data - mean_X) ** 2 / diff_mean_sqr + (1 / n)
        Var_e = np.sqrt(np.sum((val_norm - y_hat) ** 2)/(n-2))
        
        SE_regression = Var_e*((1-h_ii) ** 0.5)
        studentized_residuals = residuals/SE_regression
        
        return np.abs(np.sum(studentized_residuals))

    def _KS_metric(self, val_data, val_norm):
        """Calculate the Kolmogorov-Smirnov distance"""

        return np.max(np.abs(val_data-val_norm))
