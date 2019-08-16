# -----------------------------------------------------------
# 2019 Nicolas Lazcani, Chile
# email nlazcani@gmail.com
# -----------------------------------------------------------

from math import sqrt
import numpy as np


def distance(a, b):
    """utility method for calculating Euclidean distance"""
    squared_distance = sum([(x - y) ** 2 for x, y in zip(a, b)])
    return sqrt(squared_distance)


class KMeans(object):
    """
    Class generated with an interface similar to sklearn.
    Implementing only the parameters of n_clusters, max_iter, tol
    """

    def __init__(self, n_clusters=8, max_iter=300, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, df):
        # definition of the labels for the output depending on the input length
        labels = [0] * len(df)
        # definition of the centroids selecting the first n_clusters points
        centers = df[:self.n_clusters]
        for iter in range(self.max_iter):
            # creates an array of size of the number of clusters
            clusters = [[] for _ in range(self.n_clusters)]
            for i, point in enumerate(df):
                # obtains the distance from the point to all centroids
                distances = [distance(point, centroid) for centroid in centers]
                # get the index of the nearest
                near_cluster = distances.index(min(distances))
                # append the point to the near cluster
                clusters[near_cluster].append(point)
                # set the label
                labels[i] = near_cluster
            # record the last centroids
            centers_old = centers.copy()
            # obtain the mean point of each cluster
            for index, cluster in enumerate(clusters):
                centers[index] = np.average(cluster, axis=0)
            # calculate the difference between new and old centres and see if it is tolerable
            tolerance_per_center = [np.sum((x - y) / y * 100.0) < self.tol for x, y in zip(centers, centers_old)]
            if all(tolerance_per_center):
                break
        # set values to sklearn format
        self.cluster_centers_, self.labels_, self.n_iter_ = centers, labels, iter
        return self
