############################
 Frequently Asked Questions
############################

***********************************************
 What outlier thresholding method should I use
***********************************************

Since there are many methods to choose from, when selecting an outlier
thresholding method it is important to note: what type of data you are
working with, the selected outlier detection method, and the resultant
distribution of the outlier detection likelihood scores. The last
mentioned factor is of particular importance as it is the only
information that is directly provided to the thresholding method. A good
rule of thumb is to use the best benchmarked methods. However, these
methods may not necessarily be the best choice. Another option is to
combine multiple methods using the ``COMB`` thresholder. These options
should provide a general, if not best threshold, for the dataset, but at
the least give a good initial point for further investigation. Another
best practice is to use a thresholding method that is similar to the
outlier detection method. This may ensure a increased compatibility
between the two methods providing better results. A simple example of
this is using the ``PCA`` outlier detection method followed by the
``DECOMP`` thersholder. However, this in general works but is not always
the case. Finally, the distribution of the outlier detection likelihood
scores can be the most helpful for selecting the best thresholding
method. Understanding the profile and complexity of of the distribution
is an important factor (e.g. is the distribution Gaussian-like?). The
API reference page has added notes on most methods and can be very
helpful with matching them with the distributions of scores.

Remember these are unsupervised methods, and sometimes interpretability
may be more important than the best result. Hence, the range of level in
complexity and variety between all the available thresholding methods.
Another factor to consider is whether the importance is to get all the
outliers, our rather get the best balance between missing outliers while
removing the least amount of inliers.

----

*******************************************************************************************************
 Why are inliers and outliers incorrectly labeled even when the correct contamination level is applied
*******************************************************************************************************

Most thresholding methods follow the assumption that *the higher a
likelihood score is, the more likely it is to be an outlier*. Therefore,
if the outlier detection method incorrectly gives a high/low score to
the wrong class, this misclassification will be carried over to the
thresholding method. This makes selecting the best outlier detection
method even more important than the thresholding method. With that being
said, the ``META`` thresholder was explicitly constructed to provide the
best possible threshold given the use of any outlier detection method.
But even so, there are obvious limitations to this.

----

***********************************************************************************************
 How do I accurately threshold outliers from a test dataset with respect to a training dataset
***********************************************************************************************

So there are a few ways to threshold test data with respect to the
training dataset. A good method involves the outlier likelihood scores
of the test data being computed with regards to the training data. This
can be done with many of the outlier methods (e.g. using the
``decision_function`` function of a fitted PyOD model). It is important
to note that not all outlier detection methods genuinely implement this
functionality correctly so best to check. The threshold method can be
independently called for both datasets with reasonable confidence that
the new data is getting thresholded with respected to the training
dataset simply based on the likelihood scores.

However, if this is not sufficient and you would like more control over
the thresholding you can try the above mentioned method with a few
extra steps.

-  Fit an outlier detection model to a training dataset.

-  MinMax normalize the likelihood scores.

-  Evaluate the normalized likelihood scores with a thresholding method.

-  Get the threshold point from the normalized scores using the fitted
   thresholder from the ``.thresh_`` attribute as done in `Examples
   <https://pythresh.readthedocs.io/en/latest/example.html>`_

-  Apply the decision function of the fitted outlier detection method to
   the new incoming data and get the likelihood scores.

-  Normalize the new likelihood scores with the fitted MinMax from the
   training dataset.

-  Threshold these new scores using the ``thresh_`` value that you
   obtained earlier like this: ``new_labels = cut(normalized_new_scores,
   thresh_value)`` where the function ``cut`` can be imported from
   ``pythresh.thresholds.thresh_utility``

**Note** that if the training dataset was not meant to have outliers but
rather serve as a reference or baseline for the test data the first
mentioned method is probably the better option. If the datasets,
training and test, both are suspected of having outliers and the data
drift between the two datasets it small, the second option should work
well.

----

*********************************
 How can I visualize the results
*********************************

There are a few ways to visualize the labeled classes. One method
involves applying a 2D or 3D PCA transformation to the dataset and
scatter plotting the transformed variables while setting the colors to
the binary label output of the thresholder. Please note that a PCA
transformation will introduce its own bias of the dataset when
visualizing the results and sometimes it may look like the outlier
detection and thresholding have not worked well at all (this is
especially true for data that has a high non-linear relationship between
the classes). In this case perhaps a non-linear or more robust
decomposition method should be used for visualizing the results. Another
way to visualize the labeled classes is to generate a kernel density
estimation of the outlier likelihood scores and plot a vertical line on
the threshold point. This point can be obtained using the ``thresh_``
attribute after evaluating the likelihood scores.

----

*********************************************
 Can thresholders do multiclass thresholding
*********************************************

The short answer is *no* they cannot. PyThresh thresholding involves
only binary classification. However, if you wish for some reason to have
multiclass outlier classification (e.g. inliers, uncertains, outliers),
then perhaps clustering methods may be a good option.

----

**************
 Contributing
**************

Anyone is welcome to contribute to PyThresh:

-  Please share your ideas and ask questions by opening an issue.

-  To contribute, first check the Issue list for the "help wanted" tag
   and comment on the one that you are interested in. The issue will
   then be assigned to you.

-  If the bug, feature, or documentation change is novel (not in the
   Issue list), you can either log a new issue or create a pull request
   for the new changes.

-  To start, fork the main branch and add your
   improvement/modification/fix.

-  To make sure the code has the same style and standard, please refer
   to qmcd.py for example.

-  Create a pull request to the **main branch** and follow the pull
   request template `PR template
   <https://github.com/KulikDM/pythresh/blob/main/.github/PULL_REQUEST_TEMPLATE.md>`_

-  Please make sure that all code changes are accompanied with proper
   new/updated test functions. Automatic tests will be triggered. Before
   the pull request can be merged, make sure that all the tests pass.
