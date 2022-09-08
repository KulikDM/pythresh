import numpy as np
import scipy.stats as stats
from sklearn.utils import check_array
from .base import BaseThresholder
from .thresh_utility import normalize, cut

from .iqr import IQR
from .mad import MAD
from .fwfm import FWFM
from .yj import YJ
from .zscore import ZSCORE
from .aucp import AUCP
from .qmcd import QMCD
from .fgd import FGD
from .dsn import DSN
from .clf import CLF
from .filter import FILTER
from .wind import WIND
from .eb import EB
from .regr import REGR
from .boot import BOOT
from .mcst import MCST
from .hist import HIST
from .moll import MOLL
from .chau import CHAU
from .gesd import GESD
from .mtt import MTT
from .karch import KARCH
from .ocsvm import OCSVM
from .clust import CLUST
from .decomp import DECOMP


class ALL(BaseThresholder):
    """ALL class for Combined thresholder.

       Use the multiple thresholders as a non-parametric means
       to threshold scores generated by the decision_scores where outliers
       are set to any value beyond the (mean, median, or mode) of the
       contamination from all the combined thresholders.
       
       Paramaters
       ----------

       thresholders : list, optional (default='all')
            List of instantiated thresholders, e.g. [DSN()]
       
       max_contam : float, optional (default=0.5)
            Maximum contamination allowed for each threshold output. Thresholded scores
            above the maximum contamination will not be included in the final combined
            threshold

       method : {'mean', 'median', 'mode'}, optional (default='mean')
           statistic to apply to contamination levels
           
           - 'mean':   calculate the mean combined threshold
           - 'median': calculate the median combined threshold
           - 'mode':  calculate the majority vote or mode of the thresholded labels
           

       Attributes
       ----------

       thres_ : threshold value that seperates inliers from outliers

    """

    def __init__(self, thresholders='all', max_contam=0.5, method='mean'):

        self.thresholders = thresholders
        self.max_contam = max_contam
        stat = {'mean':np.mean, 'median':np.median, 'mode':stats.mode}
        self.method = method
        self.method_func = stat[method]

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

        # Initialize thresholders
        if self.thresholders=='all':
            self.thresholders = [IQR(), MAD(), FWFM(), YJ(), ZSCORE(),
                                 AUCP(), QMCD(), FGD(), DSN(), CLF(),
                                 FILTER(), WIND(), EB(), REGR(), BOOT(),
                                 MCST(), HIST(), MOLL(), CHAU(), GESD(),
                                 MTT(), KARCH(), OCSVM(), CLUST(), DECOMP()]

        # Apply each thresholder
        contam = []
        counts = len(decision)
        
        for thresholder in self.thresholders:
            
            labels = thresholder.eval(decision)
            outlier_ratio = np.sum(labels)/counts
            
            if outlier_ratio<self.max_contam:

                contam.append(labels)

        contam = np.array(contam)
        
        # Get [mean, median, or mode] of inliers
        if self.method=='mode':

            self.thresh_ = None
            lbls = self.method_func(contam, axis=0)
            
            return np.squeeze(lbls[0])
            
        else:

            contam = np.sum(contam, axis=1)/contam.shape[1]
            inlier_ratio = 1-self.method_func(contam)
        
            idx = int(counts*inlier_ratio)
            if idx==counts:
                limit=1.0
            else:    
                limit = decision[idx]
        
            self.thresh_ = limit
        
            return cut(decision, limit)
