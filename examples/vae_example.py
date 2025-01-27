"""Example of using the area under the curve percentage.

   for outlier thresholding
"""
# Author: D Kulik
# License: BSD 2 clause


import os
import sys

from pyod.models.knn import KNN
from pyod.utils.data import evaluate_print, generate_data
from pyod.utils.example import visualize

from pythresh.thresholds.vae import VAE

# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname('__file__'), '..')))


if __name__ == '__main__':
    contamination = 0.1  # percentage of outliers
    n_train = 200  # number of training points
    n_test = 100  # number of testing points

    # Generate sample data
    X_train, X_test, y_train, y_test =\
        generate_data(n_train=n_train,
                      n_test=n_test,
                      n_features=2,
                      contamination=contamination,
                      random_state=42)

    # train KNN detector
    clf_name = 'KNN'
    clf = KNN()
    clf.fit(X_train)
    thres = VAE()

    # get the prediction labels and outlier scores of the training data
    y_train_scores = clf.decision_scores_  # raw outlier scores

    # (Legacy method)
    # y_train_pred = thres.eval(y_train_scores)

    # binary labels (0: inliers, 1: outliers)
    thres.fit(y_train_scores)
    y_train_pred = thres.labels_
    # or
    y_train_pred = thres.predict(y_train_scores)

    # get the prediction on the test data
    y_test_scores = clf.decision_function(X_test)  # outlier scores

    # (Legacy method)
    # y_test_pred = thres.eval(y_test_scores)  # outlier labels (0 or 1)
    y_test_pred = thres.predict(y_test_scores)

    # evaluate and print the results
    print('\nOn Training Data:')
    evaluate_print(clf_name, y_train, y_train_scores)
    print('\nOn Test Data:')
    evaluate_print(clf_name, y_test, y_test_scores)

    # visualize the results
    visualize(clf_name, X_train, X_test, y_train, y_test, y_train_pred,
              y_test_pred, show_figure=True, save_figure=False)
