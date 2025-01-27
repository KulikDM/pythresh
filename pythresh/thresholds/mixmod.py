from itertools import combinations

import numpy as np
import scipy.optimize as opt
import scipy.stats as stats
from scipy.special import digamma

from .base import BaseThresholder
from .thresh_utility import cut


class MIXMOD(BaseThresholder):
    r"""MIXMOD class for the Normal & Non-Normal Mixture Models thresholder.

       Use normal & non-normal mixture models to find a non-parametric means
       to threshold scores generated by the decision_scores, where outliers
       are set to any value beyond the posterior probability threshold
       for equal posteriors of a two distribution mixture model.
       See :cite:`veluw2023mixmod` for details

       Parameters
       ----------

       method : str, optional (default='mean')
            Method to evaluate selecting the best fit mixture model. Default
            'mean' sets this as the closest mixture models to the mean of the posterior
            probability threshold for equal posteriors of a two distribution mixture model
            for all fits. Setting 'ks' uses the two-sample Kolmogorov-Smirnov test for
            goodness of fit.

       tol : float, optional (default=1e-5)
            Tolerance for convergence of the EM fit

       max_iter : int, optional (default=250)
            Max number of iterations to run EM during fit

       random_state : int, optional (default=1234)
            Random seed for the random number generators of the thresholders. Can also
            be set to None.

       Attributes
       ----------

       thresh_ : threshold value that separates inliers from outliers

       dscores_ : 1D array of decomposed decision scores

       mixture_ : fitted mixture model class of the selected model used for thresholding

       Notes
       -----

       The Normal & Non-Normal Mixture Models thresholder is constructed by searching
       all possible two component combinations of the following distributions (normal,
       lognormal, uniform, student's t, pareto, laplace, gamma, fisk, and exponential).
       Each two component combination mixture is is fit to the data using
       expectation-maximization (EM) using the corresponding maximum likelihood estimation
       functions (MLE) for each distribution. From this the posterior probability threshold
       is obtained as the point where equal posteriors of a two distribution mixture model
       exists.

    """

    def __init__(self, method='mean', tol=1e-5, max_iter=250, random_state=1234):

        super().__init__()
        dists = [stats.expon, stats.fisk, stats.gamma, stats.laplace, stats.t,
                 stats.lognorm, stats.norm, stats.uniform, stats.pareto]

        self.combs = list(combinations(dists, 2))

        self.method = method
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        np.random.seed(random_state)

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

        decision = self._data_setup(decision)

        mix_scores = decision + 1

        # Create a KDE of the decision scores
        points = max(len(decision)*3, 1000)
        x = np.linspace(1, 2, points)

        if self.method == 'ks':
            kde = stats.gaussian_kde(mix_scores, bw_method=0.1)

        mixtures = []
        scores = []
        crossing = []

        # Fit all possible combinations of dists to the scores
        for comb in self.combs:

            mix = MixtureModel([comb[0], comb[1]], self.tol, self.max_iter)
            try:
                mix.fit(mix_scores)
            except Exception:
                continue

            # Get the posterior probability threshold for equal posteriors
            y = mix.posterior(x)

            y_diff = np.sign(y[1] - y[0])
            crossings = np.where(np.diff(y_diff) != 0)[0]

            if len(crossings) == 0:
                continue

            # Evaluate the fit
            if self.method == 'ks':
                stat, _ = stats.ks_2samp(kde(x), mix.pdf(x))
            else:
                stat = x[crossings[-1]]

            mixtures.append(mix)
            scores.append(stat)
            crossing.append(crossings[-1])

        # Use the highest fit score
        if self.method == 'ks':
            max_stat = np.argmax(scores)
        else:
            diff = np.mean(scores) - np.array(scores)
            max_stat = np.argmin(np.abs(diff))

        mixture = mixtures[max_stat]
        cross = crossing[max_stat]

        limit = x[cross] if len(crossing) > 0 else 2

        self.thresh_ = limit - 1
        self.mixture_ = mixture

        return cut(mix_scores, limit)

# This portion of code is derived from the GitHub repository marcsingleton/mixmod
# which is licensed under the MIT License. Copyright (c) 2022 Marc Singleton


class MixtureModel:
    """Class for performing calculations with mixture models."""

    def __init__(self, components, tol, max_iter):

        params = [{} for _ in components]
        weights = [1 / len(components) for _ in components]

        self.components = components
        self.params = params
        self.weights = weights
        self.tol = tol
        self.max_iter = max_iter
        self.converged = False

    def fit(self, data):
        """Fit the free parameters of the mixture model with EM algorithm."""

        weights_opt = self.weights.copy()
        params_opt = []

        # Get closed-form estimator for initial estimation
        for component, param in zip(self.components, self.params):

            cfe = MLES().cfes[component.name]
            param_init = {**cfe(data), **param}
            params_opt.append(param_init)

        ll0 = self._get_loglikelihood(data, self.components,
                                      params_opt, weights_opt)

        # Apply Expectation-Maximization
        for numiter in range(1, self.max_iter + 1):

            expts = self._get_posterior(
                data, self.components, params_opt, weights_opt)
            weights_opt = expts.sum(axis=1) / expts.sum()

            for component, param_opt, expt in zip(self.components, params_opt, expts):

                # Get MLE function and update parameters
                mle = MLES().mles[component.name]
                opt = mle(data, expt=expt, initial=param_opt)
                param_opt.update(opt)

            ll = self._get_loglikelihood(
                data, self.components, params_opt, weights_opt)

            # Test numerical exception then convergence
            if np.isnan(ll) or np.isinf(ll):
                break
            if abs(ll - ll0) < self.tol:
                self.converged = True
                break

            ll0 = ll

        self.params = params_opt
        self.weights = weights_opt.tolist()

        return numiter, ll

    def loglikelihood(self, data):
        """Return log-likelihood of data according to mixture model."""

        return self._get_loglikelihood(data, self.components, self.params, self.weights)

    def posterior(self, data):
        """Return array of posterior probabilities of data for each component of mixture model."""

        return self._get_posterior(data, self.components, self.params, self.weights)

    def pdf(self, x, component='sum'):
        """Return pdf evaluated at x."""

        ps = self._get_pdfstack(x, self.components, self.params, self.weights)
        return ps.sum(axis=0)

    def _get_loglikelihood(self, data, components, params, weights):
        """Return log-likelihood of data according to mixture model."""

        p = 0
        model_params = zip(components, params, weights)
        for component, param, weight in model_params:
            pf = getattr(component, 'pdf')
            p += weight * pf(data, **param)
        return np.log(p).sum()

    def _get_posterior(self, data, components, params, weights):
        """Return array of posterior probabilities of data for each component of mixture model."""

        ps = self._get_pdfstack(data, components, params, weights)

        return ps / ps.sum(axis=0)

    def _get_pdfstack(self, data, components, params, weights):
        """Return array of pdfs evaluated at data for each component of mixture model."""

        model_params = zip(components, params, weights)
        ps = [weight * component.pdf(data, **param)
              for component, param, weight in model_params]

        return np.stack(ps, axis=0)


class MLES:
    """Class of maximum likelihood estimation functions."""

    def __init__(self):

        pass

    def create_fisk_scale(data, expt=None):
        expt = np.full(len(data), 1) if expt is None else expt

        def fisk_scale(scale):
            # Compute sums
            e = expt.sum()
            q = ((expt * data) / (scale + data)).sum()

            return 2 * q - e

        return fisk_scale

    def create_fisk_shape(data, expt=None, scale=1):
        expt = np.full(len(data), 1) if expt is None else expt

        def fisk_shape(c):
            # Compute summands
            r = data / scale
            s = 1 / c + np.log(r) - 2 * np.log(r) * r ** c / (1 + r ** c)

            return (expt * s).sum()

        return fisk_shape

    def create_gamma_shape(data, expt=None):
        expt = np.full(len(data), 1) if expt is None else expt

        def gamma_shape(a):
            # Compute sums
            e = expt.sum()
            ed = (expt * data).sum()
            elogd = (expt * np.log(data)).sum()

            return elogd - e * np.log(ed / e) + e * (np.log(a) - digamma(a))

        return gamma_shape

    def mm_fisk(data, expt=None, **kwargs):
        """Method of moment estimator for a fisk distribution."""

        expt = np.full(len(data), 1) if expt is None else expt
        ests = {}

        # Moments
        logdata = np.log(data)
        m1 = (logdata * expt).sum() / expt.sum()
        m2 = (logdata ** 2 * expt).sum() / expt.sum()

        # Estimators
        ests['c'] = np.pi / np.sqrt(3 * (m2 - m1 ** 2))
        ests['scale'] = np.exp(m1)

        return ests

    def mm_gamma(data, expt=None, **kwargs):
        """Method of moment estimator for a gamma distribution."""

        expt = np.full(len(data), 1) if expt is None else expt
        ests = {}

        # Moments
        m1 = (data * expt).sum() / expt.sum()
        m2 = (data ** 2 * expt).sum() / expt.sum()

        # Estimators
        ests['a'] = m1 ** 2 / (m2 - m1 ** 2)
        ests['scale'] = (m2 - m1 ** 2) / m1

        return ests

    def mle_expon(data, expt=None, **kwargs):
        """MLE for an exponential distribution."""

        expt = np.full(len(data), 1) if expt is None else expt
        ests = {}

        # Scale parameter estimation
        e = expt.sum()
        ed = (expt * data).sum()
        scale = ed / e
        ests['scale'] = scale

        return ests

    def mle_fisk(data, expt=None, initial=None):
        """MLE for a fisk distribution."""

        expt = np.full(len(data), 1) if expt is None else expt
        initial = MLES.mm_fisk(data) if initial is None else initial
        ests = {}

        # Scale parameter estimation
        fisk_scale = MLES.create_fisk_scale(data, expt)
        scale = opt.newton(fisk_scale, initial['scale'])
        ests['scale'] = scale

        # Shape parameter estimation
        fisk_shape = MLES.create_fisk_shape(data, expt, scale)
        c = opt.newton(fisk_shape, initial['c'])
        ests['c'] = c

        return ests

    def mle_gamma(data, expt=None, initial=None):
        """MLE for a gamma distribution."""

        expt = np.full(len(data), 1) if expt is None else expt
        initial = MLES.mm_gamma(data) if initial is None else initial
        ests = {}

        # Shape parameter estimation
        gamma_shape = MLES.create_gamma_shape(data, expt)
        try:
            a = opt.newton(gamma_shape, initial['a'])
        except ValueError:
            lower = initial['a'] / 2
            upper = initial['a'] * 2
            while np.sign(gamma_shape(lower)) == np.sign(gamma_shape(upper)):
                lower /= 2
                upper *= 2
            a = opt.brentq(gamma_shape, lower, upper)
        ests['a'] = a

        # Scale parameter estimation
        scale = (expt * data).sum() / (a * expt.sum())
        ests['scale'] = scale

        return ests

    def mle_laplace(data, expt=None, **kwargs):
        """MLE for a laplace distribution."""

        expt = np.full(len(data), 1) if expt is None else expt[data.argsort()]
        data = np.sort(data)
        ests = {}

        # Location parameter estimation
        cm = expt.sum() / 2
        e_cum = expt.cumsum()
        idx = np.argmax(e_cum > cm)

        if data[idx] == data[idx - 1]:
            loc = data[idx]
        else:
            m = (e_cum[idx] - e_cum[idx - 1]) / (data[idx] - data[idx - 1])
            b = e_cum[idx] - m * data[idx]
            loc = (cm - b) / m
        ests['loc'] = loc

        # Scale parameter estimation
        e = expt.sum()
        d_abserr = abs(data - loc)
        scale = (expt * d_abserr).sum() / e
        ests['scale'] = scale

        return ests

    def mle_lognorm(data, expt=None, **kwargs):
        """MLE for a log-normal distribution."""

        expt = np.full(len(data), 1) if expt is None else expt
        ests = {}

        # Scale parameter estimation
        e = expt.sum()
        elogd = (expt * np.log(data)).sum()
        scale = np.exp(elogd / e)
        ests['scale'] = scale

        # Shape parameter estimation
        e = expt.sum()
        logd_sqerr = (np.log(data) - np.log(scale)) ** 2
        s = np.sqrt((expt * logd_sqerr).sum() / e)
        ests['s'] = s

        return ests

    def mle_norm(data, expt=None, **kwargs):
        """MLE for a normal distribution."""

        expt = np.full(len(data), 1) if expt is None else expt
        ests = {}

        # Location parameter estimation
        e = expt.sum()
        ed = (expt * data).sum()
        loc = ed / e
        ests['loc'] = loc

        # Scale parameter estimation
        e = expt.sum()
        d_sqerr = (data - loc) ** 2
        scale = np.sqrt((expt * d_sqerr).sum() / e)
        ests['scale'] = scale

        return ests

    def mle_pareto(data, expt=None, **kwargs):
        """MLE for a pareto distribution."""

        expt = np.full(len(data), 1) if expt is None else expt
        ests = {}

        # Scale parameter estimation
        scale = min(data)
        ests['scale'] = scale

        # Shape parameter estimation
        e = expt.sum()
        elogd = (expt * np.log(data)).sum()
        b = e / (elogd - e * np.log(scale))
        ests['b'] = b

        return ests

    def mle_uniform(data, **kwargs):
        """MLE for a uniform distribution."""

        ests = {}

        # Location parameter estimation
        loc = min(data)
        ests['loc'] = loc

        # Scale parameter estimation
        scale = max(data) - loc
        ests['scale'] = scale

        return ests

    def mle_t(data, expt=None, **kwargs):
        """MLE for an student-t distribution."""

        expt = np.full(len(data), 1) if expt is None else expt
        ests = {}

        # Location parameter estimation
        e = expt.sum()
        ed = (expt * data).sum()
        loc = ed / e
        ests['loc'] = loc

        # Scale parameter estimation
        e = expt.sum()
        w_data = data - loc
        scale = np.sqrt((expt * w_data**2).sum() / e)
        ests['scale'] = scale

        # Effective degrees of freedom estimation
        w_sum_squares = (expt**2).sum()
        w_sum = expt.sum()
        df = w_sum**2 / w_sum_squares
        ests['df'] = df

        return ests

    mles = {'expon': mle_expon,
            'fisk': mle_fisk,
            'gamma': mle_gamma,
            'laplace': mle_laplace,
            'lognorm': mle_lognorm,
            'norm': mle_norm,
            'pareto': mle_pareto,
            'uniform': mle_uniform,
            't': mle_t}

    cfes = {**mles,
            'fisk': mm_fisk,
            'gamma': mm_gamma}
