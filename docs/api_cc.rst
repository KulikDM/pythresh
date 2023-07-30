################
 API CheatSheet
################

The following APIs are applicable for all detector models for ease of use.

-  :func:`pythresh.thresholders.base.BaseDetector.eval`: evaluate a single
   outlier or multiple outlier detection likelihood score sets

Key Attributes of a fitted model:

-  :attr:`pythresh.thresholds.base.BaseThresholder.thresh_`: threshold
   value from scores normalize between 0 and 1

-  :attr:`pythresh.thresholders.base.BaseDetector.confidence_interval_`:
   Return the lower and upper confidence interval of the contamination level.
   Only applies to the COMB thresholder

-  :attr:`pythresh.thresholders.base.BaseDetector.dscores_`: 1D array of the
   TruncatedSVD decomposed decision scores if multiple outlier detector score
   sets are passed

See base class definition below:

*********************************
 pythresh.thresholds.base module
*********************************

.. automodule:: pythresh.thresholds.base
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
