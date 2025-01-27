import os
from os.path import dirname as up

import joblib
import numpy as np
import pandas as pd
import scipy.stats as stats
from numba import njit, prange
from sklearn.preprocessing import MinMaxScaler

from .base import BaseThresholder


class META(BaseThresholder):
    r"""META class for Meta-modelling thresholder.

       Use a trained meta-model to evaluate a non-parametric means
       to threshold scores generated by the decision_scores where outliers
       are set based on the trained meta-model classifier.
       See :cite:`zhao2022meta` for details.

       Parameters
       ----------

       method : {'LIN', 'GNB', 'GNBC', 'GNBM'}, optional (default='GNBM')
           select

           - 'LIN':  RidgeCV trained linear classifier meta-model on true labels
           - 'GNB':  Gaussian Naive Bayes trained classifier meta-model on true labels
           - 'GNBC': Gaussian Naive Bayes trained classifier meta-model on best contamination
           - 'GNBM': Gaussian Naive Bayes multivariate trained classifier meta-model

       random_state : int, optional (default=1234)
            Random seed for the random number generators of the thresholders. Can also
            be set to None.

       Attributes
       ----------

       thresh_ : threshold value that separates inliers from outliers

       dscores_ : 1D array of decomposed decision scores

       Notes
       -----

       Meta-modelling is the creation of a model of models. If a dataset
       that contains only the explanatory variables (X), yet no response
       variable (y), it can still be predicted by using a meta-model. This
       is done by modelling datasets with known response variables that
       are similar to the dataset that is missing the response variable.

       The META thresholder was trained using the ``PyOD`` outlier
       detection methods ``LODA, QMCD, CD, MCD, GMM, KNN, KDE, PCA, Sampling`` and ``IForest``
       on the AD benchmark datasets: ``ALOI, annthyroid, breastw, campaign, cardio,
       Cardiotocography, fault, glass, Hepatitis, Ionosphere, landsat, letter, Lymphography,
       magic.gamma, mammography, mnist, musk, optdigits, PageBlocks, pendigits, Pima,
       satellite, satimage-2, shuttle, smtp, SpamBase, speech, Stamps, thyroid, vertebral,
       vowels, Waveform,  WBC, WDBC, Wilt, wine, WPBC, yeast`` available at
       `ADBench dataset <https://github.com/Minqi824/ADBench/tree/main/adbench/datasets/Classical>`_.
       META uses a majority vote of all the trained models to determine the
       inlier/outlier labels.

       Update: the latest GNBC model was further trained on the ``backdoor, celeba, census,
       cover, donors, fraud, http, InternetAds,`` and ``skin`` datasets and additionally using
       the ``AutoEncoder, LUNAR, OCSVM, HBOS, KPCA,`` and ``DIF`` outlier detection methods.

    """

    def __init__(self, method='GNBM', random_state=1234):

        super().__init__()
        self.method = method
        self.random_state = random_state
        np.random.seed(random_state)

        self._attrs = ['_kde', '_scaler', '_knorm', '_pnorm',
                       '_qnorm', '_is_flipped']

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

        if self._is_fitted is None:
            self._set_attributes(self._attrs, None)

        decision = self._data_setup(decision)

        if self.method == 'LIN':
            clf = 'meta_model_LIN.pkl'
        elif self.method == 'GNB':
            clf = 'meta_model_GNB.pkl'
        elif self.method == 'GNBC':
            clf = 'meta_model_GNBC.pkl'
        else:
            clf = 'meta_model_GNBM.pkl'

        contam = []
        counts = len(decision)
        parent = up(up(__file__))
        model = joblib.load(os.path.join(parent, 'models', clf))

        if self.method == 'GNBM':

            if self._scaler is None:
                scaler = MinMaxScaler()
                scaler.fit(decision.reshape(-1, 1))
                self._scaler = scaler
                self._norm = scaler.transform(decision.reshape(-1, 1))

            norm = self._scaler.transform(decision.reshape(-1, 1))

            qmcd = self._wrap_around_discrepancy(self._norm, norm)

            qmcd = self._set_norm(qmcd, '_qnorm')

            # Get criterion for inverting scores
            if self._is_flipped is None:
                skew = stats.skew(qmcd)
                kurt = stats.kurtosis(qmcd)

                # Invert score order based on criterion
                if (skew < 0) or ((skew >= 0) & (kurt < 0)):
                    self._is_flipped = True

            if self._is_flipped:
                qmcd = qmcd.max() + qmcd.min() - qmcd

            if self._kde is None:
                kde = stats.gaussian_kde(decision)
                self._kde = kde

            pdf = self._kde.pdf(decision)
            pdf = self._set_norm(pdf, '_knorm')

        for i in range(len(model.groups_)):

            df = pd.DataFrame()
            df['scores'] = decision
            df['groups'] = i

            if self.method == 'GNBM':

                df['qmcd'] = qmcd
                df['kdes'] = pdf**(1/10)

            labels = model.predict(df)
            outlier_ratio = np.sum(labels)/counts

            if (outlier_ratio < 0.5) & (outlier_ratio > 0):

                contam.append(labels)

        contam = np.array(contam)
        lbls = stats.mode(contam, axis=0)
        lbls = np.squeeze(lbls[0])

        self.thresh_ = None

        return lbls

    @staticmethod
    @njit(fastmath=True, parallel=True)
    def _wrap_around_discrepancy(data, check):
        """Wrap-around Quasi-Monte Carlo discrepancy method."""

        n = data.shape[0]
        d = data.shape[1]
        p = check.shape[0]

        disc = np.zeros(p)

        for i in prange(p):
            dc = 0.0
            for j in prange(n):
                prod = 1.0
                for k in prange(d):
                    x_kikj = abs(check[i, k] - data[j, k])
                    prod *= 3.0 / 2.0 - x_kikj + x_kikj ** 2

                dc += prod
            disc[i] = dc

        return - (4.0 / 3.0) ** d + 1.0 / (n ** 2) * disc
