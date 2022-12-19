Benchmarks
==========

Introduction
------------

Selecting the correct outlier detection and thresholding method can be a difficult task. Especially 
with all the different methods available in both stages. Quantifying how well each method performs 
over a variety of datasets may help when selecting based on either accuracy or robustness or both. 
PyOD provides a highly detailed analysis on the performance of all the available methods, with great 
insight and interpetability `anomaly detection benchmark paper <https://www.andrew.cmu.edu/user/yuezhao2/papers/22-neurips-adbench.pdf>`_.

Since the thresholding methods are dependant on both the dataset and the outlier detection scores, 
in order to quantify how well a threshold method works, it must be tested against multiple datasets 
applying multiple outlier detection methods to each dataset. All the benchmark datasets can be found 
at `ODDS <http://odds.cs.stonybrook.edu/#table1>`_.

----

To quantify how well the threshold method is able to correctly set inlier/outlier labels for a 
dataset, a well-defined metric must be used. The Matthews correlation coefficient (MCC) will be 
used as it provides a balanced measure when assessing class labels from a binary setup for an 
imbalanced dataset. This coefficient is given as,

.. math::

   MCC = \frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP + FP) \cdot (TP + FN) \cdot (TN + FP) \cdot (TN + FN)}} \mathrm{,}
   
where :math:`TP, TN, FP, FN` represent the true positive, true negative, false positive, and the 
false negative respectively. The MMC ranges from -1 to 1 where 1 represents a perfect prediction, 
0 an average random prediction, and -1 an inverse prediction. Since the thresholding method is heavily 
dependant on the outlier detection scores, and therefore the selected outlier detection method, 
simplifying calculating the MMC for each dataset would yield varying results that would have more 
dependance on the selected outlier method than the thresholding method. To correctly evaluate and 
eliminate the effects of the selected outlier detection method, the MMC deterioration will be used. 
This deterioration score is the difference between the MMC of the thresholded labels and the MMC for 
the labels produced by setting the true contamination level for the selected outlier detection 
method (e.g. KNN(contamination=true_contam)).

For consistency, the benchmark results below used the unit-normalized MCC, which is given by,

.. math::

   MMC_{norm} = \frac{MMC + 1}/{2} \mathrm{.}
   
Benchmarking
------------

All the thresholders using default parameters were tested on the ``arrhythmia, cardio, glass, 
ionosphere, letter, lympho, mnist, musk, optdigits, pendigits, pima, satellite, satimage-2, 
vertebral, vowels,`` and ``wbc`` datasets using the ``PCA, MCD, KNN, IForest, GMM,`` and 
``COPOD`` outlier methods on each dataset. The MMC deterioration was calculated for each instance 
and the mean and standard deviation of all the scores were calculated.

To interpret the plot below, the best to worst performing thresholders have been plotted from 
left to right with their respective uncertainty. The closer the mean value is to zero, the closer 
the thresholder performed with regards to the MMC for the labels produced by setting the true 
contamination level for the selected outlier detection method. However, the uncertainty for many goes 
beyond zero indicating that in some instances the thresholder performed better than true contamination 
level for a particuar dataset and outlier detection method. Along with the thresholders, a random 
contamination prediction was tested as well. This was done by setting :math:`MMC_{norm} = 1` and 
represents what may be expected when a random contamination level is selected without prior knowledge. 

Overall, all the thresholders performed better than selecting a random contamination level. 
The `FILTER` thresholder performed best while the `META` thresholder provided the smallest 
uncertainty about its mean.  

.. figure:: figs/Benchmark1.png
    :alt: Benchmark defaults
    
----

For a deeper look at the different user input parameters for each thresholder, the benchmarking 
was repeated for the same outlier detection methods as above. However, due to time constraints, 
only the ``arrhythmia, cardio, glass, ionosphere, letter, lympho, pima, vertebral, vowels,`` and 
``wbc`` dataset was used. The table below indicates the legend labels seen in the plot and the 
thresholding method that it corresponds to. It can be noticed that the best performing thresholder 
differs from the first plot. This is due to a smaller dataset that with fewer examples and a greater 
bias.


===============  =======================================
Label            Method
===============  =======================================
AUCP             AUCP()
BOOT             BOOT()
CHAU             CHAU()
CLF              CLF()
CLUST1           CLUST(method='agg')
CLUST2           CLUST(method='birch')
CLUST3           CLUST(method='bang')
CLUST4           CLUST(method='bgm')
CLUST5           CLUST(method='bsas')
CLUST6           CLUST(method='dbscan')
CLUST7           CLUST(method='ema')
CLUST8           CLUST(method='kmeans')
CLUST9           CLUST(method='mbsas')
CLUST10          CLUST(method='mshift')
CLUST11          CLUST(method='optics')
CLUST12          CLUST(method='somsc')
CLUST13          CLUST(method='spec')
CLUST14          CLUST(method='xmeans')
CPD1             CPD(method='Dynp')
CPD2             CPD(method='KernelCPD')
CPD3             CPD(method='Binseg')
CPD4             CPD(method='BottomUp')
DECOMP1          DECOMP(method='NMF')
DECOMP2          DECOMP(method='PCA')
DSN1             DSN(metric='JS')
DSN2             DSN(metric='WS')
DSN3             DSN(metric='ENG') 
DSN4             DSN(metric='BHT')
DSN5             DSN(metric='HLL')
DSN6             DSN(metric='HI')
DSN7             DSN(metric='LK')
DSN8             DSN(metric='MAH')
DSN9             DSN(metric='TMT')
DSN10            DSN(metric='RES')
DSN11            DSN(metric='KS')
DSN12            DSN(metric='INT')
DSN13            DSN(metric='MMD')
EB               EB()
FGD              FGD()
FILTER1          FILTER(method='gaussian')
FILTER2          FILTER(method='savgol')
FILTER3          FILTER(method='hilbert')
FILTER4          FILTER(method='wiener')
FILTER5          FILTER(method='medfilt')
FILTER6          FILTER(method='decimate')
FILTER7          FILTER(method='detrend')
FILTER8          FILTER(method='resample')
FWFM             FWFM()
GESD             GESD()
HIST1            HIST(method='otsu')
HIST2            HIST(method='yen')
HIST3            HIST(method='isodata')
HIST4            HIST(method='li')
HIST5            HIST(method='triangle')
IQR              IQR()
KARCH            KARCH()
MAD              MAD()
MCST             MCST()
META1            META(method='LIN')
META2            META(method='GNB')
MOLL             MOLL()
MTT              MTT() 
OCSVM1           OCSVM(model='poly')
OCSVM2           OCSVM(model='sgd')
QMCD1            QMCD(method='CD')
QMCD2            QMCD(method='WD')
QMCD3            QMCD(method='MD')
QMCD4            QMCD(method='L2-star')
REGR1            REGR(method='siegel')
REGR2            REGR(method='theil')
VAE              VAE()
WIND             WIND()
YJ               YJ()
ZSCORE           ZSCORE()
===============  =======================================

.. figure:: figs/Benchmark2.png
    :alt: Benchmark all
    
----
    

External Benchmarking
---------------------

An external benchmark test of all the default thresholders is available in 
`Estimating the Contamination Factor's Distribution in Unsupervised Anomaly Detection <https://arxiv.org/abs/2210.10487>`_. 
However it is important to note that a different evaluation metric was used (F1 deterioration) was 
used, and also since the publishing of this article some default parameters for some thresholders 
have been changed. Still this article provides a thorough analysis of the performance of the 
thresholders in ``PyThresh`` with many insightful results and detailed analysis of thresholding 
outlier decision scores.

----


Effects of Randomness
---------------------

Some thresholders use randomness in their methods and the random seed can be set using the 
parameter ``random_state``. To investigate the effect of randomness on the resulting labels 
the MMC deterioration was calculated for each thresholder using the random states 
(1234, 42, 9685, and 111222). The same outlier detection methods as well as datasets from the 
first benchmarking test were applied. The means of the MMC deterioration were normalized to zero 
showing the extent of the effect of randomness of each thresholder's ability to evaluate labels for 
the outlier decision scores in the uncertainty. 

From the plot below, ``VAE`` performed the worst and was highly affected by the choice of the 
selected random state. ``DSN`` which is thresholder that overall performed well during the benchmark 
tests is also sensitive to randomness. To alleviate the effects of randomness on for the thresholders, 
it is recommended that a combined method be used by setting different random states 
(e.g. ``ALL(thresholders = [DSN(random_state=1234), DSN(random_state=42), DSN(random_state=9685), DSN(random_state=111222)])``). 
This should provide a more robust and reliable result.

.. figure:: figs/Randomness.png
    :alt: Effects of Randomness 