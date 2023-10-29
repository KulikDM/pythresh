import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.naive_bayes import GaussianNB


def BREG_metric(x1, x2):
    """Calculate the Exponential Euclidean Bregman distance."""

    gradient_x = np.exp(x1) - 1
    gradient_y = np.exp(x2) - 1

    distance = np.sum((x1 - x2) * (gradient_x - gradient_y))

    return distance


def mclain_rao_index(data, labels):
    """Calculate the Mclain Rao index."""

    unique_labels = np.unique(labels)
    centroids = []

    # Calculate the centroids of each cluster
    for label in unique_labels:
        cluster_data = data[labels == label]
        centroid = np.mean(cluster_data)
        centroids.append(centroid)

    num_clusters = len(centroids)
    mri = 0.0

    # Calculate the MRI
    for i in range(num_clusters):
        for j in range(i + 1, num_clusters):
            distance = (centroids[i] - centroids[j]) ** 2
            mri += distance

    # Normalize the MRI by the number of cluster pairs
    num_cluster_pairs = num_clusters * (num_clusters - 1) / 2
    mri /= num_cluster_pairs

    return mri


def GNB_score(data, labels):
    """Calculate the Gaussian Naive-Bayes trained consensus score."""

    # Setup data for training
    X = np.tile(data, (len(labels), 1))
    y = np.hstack(labels)

    # Fit model and predict
    model = GaussianNB()
    model.fit(X, y)

    pred = model.predict(data)

    # Find the deviation of each model from fitted GNB model
    dev = np.sum(np.abs(np.vstack(labels) - pred), axis=1)

    return dev.squeeze()


def Contam_score(data, labels, contam):
    """Calculate the mean contamination deviation based on TruncatedSVD decomposed scores."""

    # Fit model and transform data
    decomp = TruncatedSVD(n_components=1, random_state=1234)
    dat = decomp.fit_transform(np.vstack(data).T).squeeze()

    # Find the deviation of the contamination of each model from the decomposed model
    thr = np.zeros(len(labels[0]))
    thr[dat > np.percentile(dat, (1-np.mean(contam))*100)] = 1

    dev = np.sum(np.abs(np.vstack(labels) - thr), axis=1)

    return dev.squeeze()
