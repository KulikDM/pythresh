################
 API CheatSheet
################

The following APIs are applicable for all detector models for ease of
use.

-  :func:`pythresh.thresholders.base.BaseDetector.eval`: evaluate a
   single outlier or multiple outlier detection likelihood score set
   (Legacy method)

-  :func:`pythresh.thresholders.base.BaseDetector.fit`: fit a
   thresholder for a single outlier or multiple outlier detection
   likelihood score set

-  :func:`pythresh.thresholders.base.BaseDetector.predict`: predict the
   binary labels using the fitted thresholder on a single outlier or
   multiple outlier detection likelihood score set

Key Attributes of a fitted model:

-  :attr:`pythresh.thresholds.base.BaseThresholder.thresh_`: threshold
   value from scores normalize between 0 and 1

-  :attr:`pythresh.thresholds.base.BaseThresholder.labels_`: A binary
   array of labels for the fitted thresholder on the fitted dataset

-  :attr:`pythresh.thresholders.base.BaseDetector.confidence_interval_`:
   Return the lower and upper confidence interval of the contamination
   level. Only applies to the COMB thresholder

-  :attr:`pythresh.thresholders.base.BaseDetector.dscores_`: 1D array of
   the TruncatedSVD decomposed decision scores if multiple outlier
   detector score sets are passed

-  :attr:`pythresh.thresholders.mixmod.MIXMOD.mixture_`: fitted mixture
   model class of the selected model used for thresholding. Only applies
   to MIXMOD. Attributes include: components, weights, params. Functions
   include: fit, loglikelihood, pdf, and posterior.

See base class definition below:

*********************************
 pythresh.thresholds.base module
*********************************

.. automodule:: pythresh.thresholds.base
   :members:
   :exclude-members: _data_setup, _set_norm, _set_attributes
   :undoc-members:
   :show-inheritance:
   :inherited-members:
