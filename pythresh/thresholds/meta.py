import os
from os.path import dirname as up

import joblib
import numpy as np
import pandas as pd
import scipy.stats as stats
from numba import njit, prange
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import check_array

from .base import BaseThresholder
from .thresh_utility import normalize


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

       Attributes
       ----------

       thresh_ : threshold value that separates inliers from outliers

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
       `ADBench dataset <https://github.com/Minqi824/ADBench/tree/main/datasets/Classical>`_.
       META uses a majority vote of all the trained models to determine the
       inlier/outlier labels.

    """

    def __init__(self, method='GNBM'):

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

            scaler = MinMaxScaler()
            norm = scaler.fit_transform(decision)
            norm = (norm/(norm.max(axis=0, keepdims=True)
                            + np.spacing(0)))

            qmcd = self._wrap_around_discrepancy(norm)
            
            qmcd = normalize(qmcd)
            if len(qmcd[qmcd>0.5]) > 0.5*len(qmcd):
                qmcd = 1 - qmcd

            kde = stats.gaussian_kde(decision)
            pdf = normalize(kde.pdf(decision))

        for i in range(380):

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
    def _wrap_around_discrepancy(data):

        n = data.shape[0]
        d = data.shape[1]

        disc = np.zeros(n)

        for i in prange(n):
            dc = 0.0
            for j in prange(n):
                prod = 1.0
                for k in prange(d):
                    x_kikj = abs(data[i, k] - data[j, k])
                    prod *= 3.0 / 2.0 - x_kikj + x_kikj ** 2
                        
                dc += prod
            disc[i] = dc

        return - (4.0 / 3.0) ** d + 1.0 / (n ** 2) * disc
