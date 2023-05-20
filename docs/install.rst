##############
 Installation
##############

It is recommended to use **pip** or **conda** installation. Please make
sure **the latest version** is installed, as PyThresh is updated
frequently:

.. code:: bash

   pip install pythresh            # normal install
   pip install --upgrade pythresh  # or update if needed

.. code:: bash

   conda install -c conda-forge pythresh

Alternatively, you can get the version with the latest updates by
cloning the repo and run setup.py file:

.. code:: bash

   git clone https://github.com/KulikDM/pythresh.git
   cd pythresh
   pip install .

Or with **pip**:

.. code:: bash

   pip install https://github.com/KulikDM/pythresh/archive/main.zip

**Required Dependencies**:

-  matplotlib
-  numpy>=1.13
-  pyod
-  scipy>=1.3.1
-  scikit_learn>=0.20.0

**Optional Dependencies**:

-  pyclustering (used in the CLUST thresholder)
-  ruptures (used in the CPD thresholder)
-  geomstats (used in the KARCH thresholder)
-  scikit-lego (used in the META thresholder)
-  joblib>=0.14.1 (used in the META thresholder)
-  pandas (used in the META thresholder)
-  torch (used in the VAE thresholder)
-  tqdm (used in the VAE thresholder)
