#########
 Ranking
#########

**************
 Introduction
**************

Outlier detection for unsupervised tasks is difficult. The difficulty
arises with how to evaluate whether the selected methods for outlier
detection are correct or the best for the dataset. Often, additional
post-evaluation involves visual or domain knowledge based decisions to
make a final call. This process can be tedious and is often just as
complex as the methods used to do the outlier detection in the first
place.

However, there are some robust statistical methods that can be used to
indicate or at least provide a guide of which outlier detection methods
perform better than others. This process is that of a ranking problem
and can be used to list in order the evaluated outlier detection methods
in terms of their respective performance with other methods.

----

In order to rank the multiple outlier detection methods' capabilities of
effectively labeling a given dataset, it is important to note what is
defined by "capabilities". In this case, this refers to how well the
computed labels solved by the outlier detection method compares to the
true labels. This can be done using the F1 score or Matthews correlation
coefficient (MCC). But, since unsupervised tasks means there is lack of
true labels, the best option is to use other robust statistical metrics
that have a strong correlation to the above mentioned scores.

To find these meta-metrics a good starting point is to know what data
can be used to compute them. In the case of unsupervised outlier
detection, there are essentially three main components: the exploratory
variables (X), the computed outlier likelihood scores, and the
thresholded binary labels. With these, criteria can be set to apply
meta-metrics that can then be ranked to provide an list of best to worst
performing outlier detection methods with respect to these metrics.

**************
 Meta-Metrics
**************

High correlation with the MMC and F1 scores has been seen for the
following meta-metrics. Correlation tests were done on the ``arrhythmia,
cardio, glass, ionosphere, letter, lympho, mnist, musk, optdigits,
pendigits, pima, satellite, satimage-2, vertebral, vowels,``and ``wbc``
datasets using the ``PCA, MCD, KNN, IForest, GMM,`` and ``COPOD``
outlier detection methods and the ``FILTER, OCSVM, DSN,`` and ``CLF``
thresholding methods on each dataset. While high correlation was noted
overall, there were significant low or even inverse correlation
relationships indicating that the meta-metrics are general or
sub-optimal indicators to the best performance.

The ``RANK`` method in ``PyThresh`` uses these meta-metrics to rank the
performance of the outlier detection methods against each other with
respect to the selected threshold or thresholding method.

Statistical Distances
=====================

Statistical distances quantify the measure of distances between
probability based measures. These distances can be used to compute a
single value difference between two probability distributions. This
hints to what data from the unsupervised outlier detection method task
should be used. Two distinct probability distributions can be computed
for the outlier likelihood scores with respected to their labeled class.
Three statistical distances have been selected:

-  The Jensen-Shannon distance
-  The Wasserstein distance
-  The Lukaszyk-Karmowski metric for normal distributions

Clustering Based
================

Since the dataset is considered to contain outliers, a clear distinction
should exist between the inliers and outliers. By using clustering based
metrics, the measure of similarity or distance between centroids or each
datapoints can provide a single score of outlier detection method's to
effectively distinguish and inliers from outliers. This should also
indicate greater distinction between the inlier and outlier clusters.
For these clustering based metrics the quality of the labeled data will
be evaluated using the exploratory variables and their assigned labels.
Three clustering based metrics have been selected:

-  The Silhouette score
-  The Davies-Bouldin score
-  The Calinski-Harabasz score

Mode Deviation
==============

Since the other two meta-metrics evaluate each outlier detection method
individually, a comparator meta-metric should also be added with which
to compare all outlier detection method results against some baseline.
This baseline can be set by taking the element-wise mode between all the
outlier detection method labels. The absolute difference between each
outlier detection's labels and the baseline can be used as a metric of
deviation from the mode of all outlier detection methods

*****************
 Rank OD Methods
*****************

The ranking process involves ordering the meta-metric scores with
respect to their performance. The meta-metrics are ordered
highest-to-lowest or lowest-to-highest based on their performance
criterion. The meta-metrics are combined as follows: the statistical
based distances are combined using equal weighting on their ordered
ranks to compute a single ranked list. The same method is done to the
clustering based metrics. Finally, an overall combined rank is computed
using the combined statistical based ranking, the combined clustering
based ranking, and the mode baseline deviation ranking. This final
combined ranking can either be computed using equal weightings for each
three meta-metric classed rankings or a weight list can be parsed based
on preference.

*********
 Example
*********

Below is a simple example of how to apply the ``RANK`` method:

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

#############
 Final Notes
#############

While the ``RANK`` method is a useful tool to assist in selecting the
possible best outlier detection method to use with respect to the
applied thresholder or threshold level, it is not infallible. It has
been noted from the tests above, that in general the ranked results
often returned the best-to-worst performing outlier detection methods
in the correct order. However, they were not perfect. They at times
exhibited slight incorrect orders and often the best performing OD
method was in the top three rather than being the top of the list.
Additionally, some times well performing OD methods was ranked poorly.

The ``RANK`` method should be used with discretion but hopefully provide
more clarity on which OD method to select.
