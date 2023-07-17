import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.utils import check_array
from scipy.stats import beta,wishart,multivariate_normal,dirichlet
from sklearn.mixture import BayesianGaussianMixture
from scipy.optimize import least_squares

from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.iforest import IForest
from pyod.models.cblof import CBLOF
from pyod.models.copod import COPOD
from pyod.models.loda import LODA
from pyod.models.hbos import HBOS
from pyod.models.ocsvm import OCSVM

from .base import BaseThresholder
from .thresh_utility import normalize


class gammaGMM(BaseThresholder):
    r"""gammaGMM class for gammaGMM thresholder.
    
       Use a Bayesian method for estimating the posterior distribution
       of the contamination factor (i.e., the proportion of anomalies) 
       for a given unlabeled dataset. Then, it sets the threshold such
       that the proportion of predicted anomalies equals the
       contamination factor. See :cite:`perini2023estimating` for details.
       
       Two options are available:
       
       1. It samples n contamination factors from the posterior and uses
       each of them to obtain a threshold. This returns n arrays
       of predictions, one for each threshold.
       
       2. It takes the posterior mean and uses it to get predictions.
       This returns a unique array with predictions.

       Parameters
       ----------
       sample_n_thresholds : whether sampling the contamination (option 1) or using the mean (option 2)
       list_detectors      : list of PyOD detectors to be used for augmenting the score space
       n_contaminations    : number of samples to draw from the contamination posterior distribution
       ndraws              : number of samples simultaneously drawn from each DPGMM component
       p0                  : probability that no anomalies are in the data
       phigh               : probability that there are more than high_gamma anomalies
       high_gamma          : sensibly high number of anomalies that has low probability to occur
       lim_gamma           : there must not be more than this proportion of anomalies
       K                   : number of components for DPGMM used to approximate the Dirichlet Process
       seed                : seed for replicability
       cpu                 : number of cpus to parallelize the DPGMM
       verbose             : whether you want to check the iteration of the DPGMM (every 20 iterations a print will occur)

       Attributes
       ----------

       thresh_ : threshold value that separates inliers from outliers

       Notes
       -----
       The detector list must be in a PyOD format. Alternatively, one can produce the scores and skip the fitting of the detectors in the code.
       Check :cite:`perini2023estimating` to have further details on the code and the algorithm.
       

    """

    def __init__(self,
                 sample_n_thresholds = True, #Option 1
                 list_detectors = [KNN(), IForest(), LOF(), OCSVM(), HBOS(), COPOD(), LODA(), CBLOF()],
                 n_contaminations=1000,
                 ndraws=50, 
                 p0=0.01, 
                 phigh=0.01, 
                 high_gamma=0.15, 
                 gamma_lim = 0.25, 
                 K=100, 
                 seed=331, 
                 cpu = 1, 
                 verbose = True):

        super().__init__()
        self.sample_n_thresholds = sample_n_thresholds
        self.list_detectors = list_detectors
        self.n_contaminations = n_contaminations
        self.ndraws = ndraws
        self.p0 = p0
        self.phigh = phigh
        self.high_gamma = high_gamma
        self.gamma_lim = gamma_lim
        self.K = K
        self.seed = seed
        self.cpu = cpu
        self.verbose = verbose
        
    def eval(self, decision, X):
        """Outlier/inlier evaluation process for decision scores.

        Parameters
        ----------
        decision : np.array or list of shape (n_samples)
                   which are the decision scores from a
                   outlier detection.

        Returns
        -------
        outlier_labels : numpy array of shape (n_samples,n_contaminations)
            For each observation, tells whether or not
            it should be considered as an outlier according to the
            fitted model. 0 stands for inliers and 1 for outliers.
        """

        decision = check_array(decision, ensure_2d=False)
        
        #preprocess_scores
        decision = np.log(decision-np.min(decision)+0.01)
        mean_dec, std_dec = np.mean(decision), np.std(decision)
        if std_dec ==0:
            print("Detector", detector, "assigns constant scores (unique value). Removing this detector is suggested.")
        else:
            decision = (decision - mean_dec)/std_dec
        
        
        score_space = self.augment_space_with_detectors(decision, X)
        
        gamma_posterior_sample = self.compute_gamma_posterior(score_space)

        if self.sample_n_thresholds:
            self.thresh_ = np.array([np.percentile(decision, 100 * (1 - gamma_s)) for gamma_s in gamma_posterior_sample])
            labels = np.array([(decision > thr).astype('int').ravel() for thr in self.thresh_])
            
        else:
            gamma_mean = np.mean(gamma_posterior_sample)
            self.thresh_ = np.percentile(decision, 100 * (1 - gamma_mean))
            labels = (decision > self.thresh_).astype('int').ravel()
            print(gamma_mean)
        return labels
        
        
    def compute_gamma_posterior(self, decision):
    
        # we cannot use more components than samples.
        if np.shape(decision)[0]<self.K:
            self.K = np.shape(decision)[0]-1

        itv = 0
        # Repeat the loop until a valid gamma sample is found. If it cannot, then return gamma = 0.
        while True:
            whileseed = self.seed + 100*itv
            np.random.seed(whileseed)
            # Fit the DPGMM
            bgm = BayesianGaussianMixture(weight_concentration_prior_type = "dirichlet_process", n_components = self.K,
                                          weight_concentration_prior = 0.01, max_iter = 1500, random_state = whileseed,
                                          verbose = self.verbose, verbose_interval = 20).fit(decision)
            # Drop components with less than 2 instances assigned
            filter_idx = np.where(bgm.weight_concentration_[0]>=2)[0]
            tot_concentration = np.sum(bgm.weight_concentration_[0])
            partial_concentration = np.sum(bgm.weight_concentration_[0][filter_idx])
            means = bgm.means_[filter_idx]
            covariances = bgm.covariances_[filter_idx,:,:]
            alphas = bgm.weight_concentration_[0][filter_idx]
            mean_precs = bgm.mean_precision_[filter_idx]
            dgf = bgm.degrees_of_freedom_[filter_idx]
            idx_sortcomponents, meanstd = self.order_components(means, mean_precs, covariances, dgf)
            # Redistribute the lost mass (after cutting off some components)
            alphas = alphas[idx_sortcomponents] + (tot_concentration-partial_concentration)/len(filter_idx)
            # Solve the optimization problem to find the hyperparameters delta and tau
            res = least_squares(self.find_delta_tau, x0=(-2,1), args=(meanstd,alphas),
                                bounds=((-50,-1),(0,50)))
            delta,tau = res.x
            # Check that delta and tau are properly found, allowing 10% error top
            p0Est, phighEst = self.check_delta_tau(delta,tau,meanstd,alphas)
            if p0Est<self.p0*1.10 and p0Est>self.p0*0.90 and phighEst<self.phigh*1.10 and phighEst>self.phigh*0.90:
                # If hyperparameters are OK, break the loop. Otherwise repeat it with different seeds
                break;
            elif self.verbose:
                print(p0Est, phighEst)
                print("It cannot find the best hyperparameters. Let's run the model again.")
            itv+=1
            if itv>100:
                print("No solution found. It will return all zeros.")
                return np.zeros(self.n_contaminations, np.float)
        # Sort the components values and extract the parameters posteriors
        means = means[idx_sortcomponents]
        mean_precs = bgm.mean_precision_[idx_sortcomponents]
        covariances = covariances[idx_sortcomponents,:,:]
        dgf = bgm.degrees_of_freedom_[idx_sortcomponents]
        # Compute the cumulative sum of the mixing proportion (GMM weights)
        gmm_weights = np.cumsum(dirichlet(alphas).rvs(self.n_contaminations), axis = 1)
        w = {}
        for k in range(len(filter_idx)):
            w[k+1] = gmm_weights[:,k]
        # Sample from gamma's posterior by computing the probabilities
        gamma = self.sample_withexactprobs(means, mean_precs, covariances, dgf, delta, tau, w)

        #if gamma has not enough samples, just do oversampling
        gamma = gamma[gamma<self.gamma_lim]
        if len(gamma)<self.n_contaminations:
            gamma = np.concatenate((gamma,np.random.choice(gamma[gamma>0.0], self.n_contaminations-len(gamma), replace = True)))
        # return the sample from gamma's posterior
        return gamma
    
    def order_components(self, means, mean_precs, covariances, dgf):
        K, M = np.shape(means)
        meanstd = np.zeros(K, np.float)
        mean_std = np.sqrt(1/mean_precs)
        for k in range(K):
            sample_mean_component = multivariate_normal.rvs(mean = means[k,:], cov = mean_std[k]**2,
                                                            size = 1000,random_state=self.seed)
            sample_covariance = wishart.rvs(df=dgf[k],scale=covariances[k]/dgf[k],size=1000,random_state=self.seed)
            var = np.array([np.diag(sample_covariance[i]) for i in range(1000)])
            meanstd[k] = np.mean([np.mean(sample_mean_component[:,m].reshape(-1)/(1+np.sqrt(var[:,m].reshape(-1)))) \
                                  for m in range(M)])
        idx_components = np.argsort(-meanstd)
        meanstd = meanstd[idx_components]
        return idx_components, np.array(meanstd)
    
    
    def find_delta_tau(self, params, *args):
    
        delta, tau = params
        meanstd,alphas = args
        first_eq = delta - (np.log(self.p0/(1-self.p0)) - tau)/meanstd[0]

        prob_ck = self.sigmoid(delta, tau, meanstd)
        prob_c1ck = self.derive_jointprobs(prob_ck)

        a = np.cumsum(alphas)
        b = sum(alphas) - np.cumsum(alphas)
        probBetaGreaterT = np.nan_to_num(beta.sf(self.high_gamma, a, b), nan = 1.0)

        second_eq = np.sum(probBetaGreaterT*prob_c1ck)-self.phigh

        return (first_eq, second_eq)
    
    def check_delta_tau(self,delta,tau,meanstd,alphas):
    ### Check that delta and tau are properly set. Return the p_0 and p_high estimated using the given delta and tau. ###
    
        prob_ck = self.sigmoid(delta, tau, meanstd)
        p0Est = 1-prob_ck[0]

        prob_c1ck = self.derive_jointprobs(prob_ck)
        a = np.cumsum(alphas)
        b = sum(alphas) - np.cumsum(alphas)
        probBetaGreaterT = np.nan_to_num(beta.sf(self.high_gamma, a, b), nan = 1.0)
        phighEst = np.sum(probBetaGreaterT*prob_c1ck)
        return p0Est, phighEst
    

    def sigmoid(self,delta, tau, x):
        ### Transforms scores into probabilities using a sigmoid function. ###

        return 1/(1+np.exp(tau+delta*x))

    def derive_jointprobs(self,prob_ck):
        ### Obtain the joint probabilities given the conditional probabilities ###

        cumprobs = np.cumprod(prob_ck)
        negprobs = np.roll(1-prob_ck,-1)
        negprobs[-1] = 1
        prob_c1ck = cumprobs*negprobs
        return prob_c1ck

    def sample_withexactprobs(self, means, mean_precs, covariances, dgf, delta, tau, w):
        ### This function computes the joint probabilities and use them to get a sample from gamma's posterior. ###

        K = np.shape(means)[0]
        mean_std = np.sqrt(1/mean_precs)
        samples = np.array([])
        i = 0
        while len(samples)<self.n_contaminations*(1-self.p0):
            prob_ck = np.zeros(K, np.float)
            for k in range(K):
                rnd = (i+1)*(k+1)
                sample_mean_component = multivariate_normal.rvs(mean = means[k,:], cov = mean_std[k]**2,
                                                                size = 1,random_state=10*self.seed+rnd)
                sample_covariance = wishart.rvs(df=dgf[k],scale=covariances[k]/dgf[k],
                                                size=1,random_state=10*self.seed+rnd)
                var = np.diag(sample_covariance)
                meanstd = np.mean((sample_mean_component)/(1+np.sqrt(var)))

                prob_ck[k] = self.sigmoid(delta, tau, meanstd)

            prob_c1ck = self.derive_jointprobs(prob_ck)
            for k in range(K):
                ns = int(np.round(self.ndraws*prob_c1ck[k],0))
                if ns>0:
                    samples = np.concatenate((samples,np.random.choice(w[k+1], ns, replace = False)))
            i+=1
        if len(samples)> self.n_contaminations*(1-self.p0):
            samples = np.random.choice(samples, int(self.n_contaminations*(1-self.p0)), replace = False)
        samples = np.concatenate((samples, np.zeros(self.n_contaminations - len(samples), np.float)))
        return samples
        
        
    def augment_space_with_detectors(self, decision, X):
        ### Obtain the anomaly scores, map them into the positive axis, take the log and normalize the final values. ###
        df = pd.DataFrame(data = [])
        df['0'] = decision
        for i,detector in enumerate(self.list_detectors):
            detector.fit(X)
            x = detector.decision_function(X)
            minx = np.min(x)
            x = np.log(x-minx+0.01)
            meanx, stdx = np.mean(x), np.std(x)
            if stdx >0:
                x = (x - meanx)/stdx
            df[str(i+1)] = x
        return df.values
