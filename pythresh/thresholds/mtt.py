import numpy as np
from sklearn.utils import check_array
from .base import BaseThresholder
from .thresh_utility import normalize, cut, gen_kde

#https://github.com/vvaezian/modified_thompson_tau_test/blob/main/src/Modified_Thompson_Tau_Test/modified_thompson_tau_test.py

def get_T_critical_value(n, strictness):
    """Get the value from the t-Student distribution for the given n"""
    
    mapping = {
      1:[ { 1:3.078,2:1.886,3:1.638,4:1.533,5:1.476,6:1.440,7:1.415,8:1.397,9:1.383,10:1.372,11:1.363,12:1.356,
            13:1.350,14:1.345,15:1.341,16:1.337,17:1.333,18:1.330,19:1.328,20:1.325,21:1.323,22:1.321,23:1.319,
            24:1.318,25:1.316,26:1.315,27:1.314,28:1.313,29:1.311,30:1.310,40:1.303,50:1.299,60:1.296,80:1.292,
            100:1.290,120:1.289}, 1.282 ],
      2:[ { 1:6.314,2:2.920,3:2.353,4:2.132,5:2.015,6:1.943,7:1.895,8:1.860,9:1.833,10:1.812,11:1.796,12:1.782,
            13:1.771,14:1.761,15:1.753,16:1.746,17:1.740,18:1.734,19:1.729,20:1.725,21:1.721,22:1.717,23:1.714,
            24:1.711,25:1.708,26:1.706,27:1.703,28:1.701,29:1.699,30:1.697,40:1.684,50:1.676,60:1.671,80:1.664,
            100:1.660,120:1.658}, 1.645 ],
      3:[ { 1:12.71, 2:4.303, 3:3.182, 4:2.776, 5:2.571, 6:2.447, 7:2.365, 8:2.306, 9:2.262, 10:2.228, 11:2.201, 
            12:2.179, 13:2.160, 14:2.145, 15:2.131, 16:2.120, 17:2.110, 18:2.101, 19:2.093, 20:2.086, 21:2.080, 
            22:2.074, 23:2.069, 24:2.064, 25:2.060, 26:2.056, 27:2.052, 28:2.048, 29:2.045, 30:2.042, 40:2.021, 
            50:2.009, 60:2.000, 80:1.990, 100:1.984, 120:1.980}, 1.96 ],
      4:[ { 1:31.82 ,2:6.965,3:4.541,4:3.747,5:3.365,6:3.143,7:2.998,8:2.896,9:2.821,10:2.764,11:2.718,12:2.681,
            13:2.650,14:2.624,15:2.602,16:2.583,17:2.567,18:2.552,19:2.539,20:2.528,21:2.518,22:2.508,23:2.500,
            24:2.492,25:2.485,26:2.479,27:2.473,28:2.467,29:2.462,30:2.457,40:2.423,50:2.403,60:2.390,80:2.374,
            100:2.364,120:2.358}, 2.326 ],
      5:[ { 1:63.66,2:9.925,3:5.841,4:4.604,5:4.032,6:3.707,7:3.499,8:3.355,9:3.250,10:3.169,11:3.106,12:3.055,
            13:3.012,14:2.977,15:2.947,16:2.921,17:2.898,18:2.878,19:2.861,20:2.845,21:2.831,22:2.819,23:2.807,
            24:2.797,25:2.787,26:2.779,27:2.771,28:2.763,29:2.756,30:2.750,40:2.704,50:2.678,60:2.660,80:2.639,
            100:2.626,120:2.617}, 2.576 ]
    }
    
    t, inf_val = mapping[strictness]
    for key, val in t.items():
      if n <= key:
        return val
    return inf_val



class MTT(BaseThresholder):
    """MTT class for Modified Thompson Tau test thresholder.

       Use the modified Thompson Tau test to evaluate a non-parametric means
       to threshold scores generated by the decision_scores where outliers
       are set to any value beyond the smallest outlier detected by the test
       
       Paramaters
       ----------

       strictness : int, optional (default=4)
        [1,2,3,4,5]

       Attributes
       ----------

       eval_: numpy array of binary labels of the training data. 0 stands
           for inliers and 1 for outliers/anomalies.

    """

    def __init__(self, strictness=4):

        self.strictness = strictness

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

        arr = np.sort(decision.copy())

        limit = 1.1

        while True:

            # Calculate the rejection threshold
            n = len(arr)
            t = get_T_critical_value(n - 2, strictness=self.strictness)
            thres = (t * (n - 1))/(np.sqrt(n) * np.sqrt(n - 2 + t**2))
            delta = np.abs(arr[-1] - arr.mean())/arr.std()

            if delta>thres: 
                limit = arr[-1]
                arr = np.delete(arr, n-1)

            else:
                break

        self.thresh_ = limit
        
        return cut(decision, limit)
