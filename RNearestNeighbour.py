from dataclasses import dataclass

import numpy as np
from numpy import ndarray
from sklearn.neighbors import NearestNeighbors


@dataclass
class MahalanobisData:
    """
    """
    Vt: ndarray = None  # Truncated right singular matrix transposed of the corpus
    mu: ndarray = None  # Mean of the corpus
    S: ndarray = None  # Truncated singular values of the corpus
    subspace_thres: float = 1e-3  # Threshold to decide whether a point is in the data subspace
    svd_thres: float = 1e-12  # Threshold to decide numerical rank of the data matrix
    numerical_rank: int = None  # Numerical rank
    corpus: ndarray = None  # Corpus used to compute the Data
    corpus_distances: ndarray = None  # Distances of data in the corpus


class RNearestNeighbour:
    """
    Nearest neighbours (possibly randomised)

    """

    def __init__(self, thres_quantile, distance="Euclidean"):
        """
        :param thres_quantile: a number between 0-1 to decide about the threshold distance for outliers
        """
        self.n_neighbours = 1
        self.distance = distance
        self.estimator = None
        self.thres_quantile = thres_quantile
        self.thres_distance = None
        self.mahalanobis_data = MahalanobisData()

    def fit(self, corpus) -> None:
        """

        :param corpus: numpy array of numpy array, corpus[i] is the feature of the ith data point
        :return:
        """
        if self.distance == "Euclidean":
            self.estimator = NearestNeighbors(n_neighbors=1)
            self.estimator.fit(corpus)
            dist, ind = self.estimator.kneighbors()
            self.thres_distance = np.quantile(dist, self.thres_quantile)
        elif self.distance == "Mahalanobis":
            self._mahalanobis(corpus)
        else:
            raise NotImplementedError

    def predict(self, x) -> bool:
        """
        :param x: data point
        :return: True if x is classified as an outlier, False if inlier
        """
        if self.distance == "Euclidean":
            return (self.estimator.kneighbors(np.reshape(x, (1, -1))))[0] >= self.thres_distance
        elif self.distance == "Mahalanobis":
            # 1. decide if x is in the subspace, mu + row_span(X - mu)
            y = x - self.mahalanobis_data.mu
            rho = np.linalg.norm(y - y @ self.mahalanobis_data.Vt.T @ self.mahalanobis_data.Vt) \
                  / np.linalg.norm(y)
            if rho > self.mahalanobis_data.subspace_thres:
                return True
            # 2. compute the minimal distance of x to all data point in the corpus using Mahalanobis distance
            n = len(self.mahalanobis_data.corpus)
            dists = [np.inf] * n
            for i in range(0, n):
                dists[i] = self._mahalanobis_distance(x, self.mahalanobis_data.corpus[i])
            if min(dists) > self.thres_distance:
                return True
            else:
                return False
        else:
            raise NotImplementedError

    def _mahalanobis(self, corpus) -> None:
        """
        Prepare to use nearest neighbours with Mahalanobis distance as a classifier
        :return:
        """
        X = corpus
        mean_X = np.mean(X, axis=0)
        U, S, Vt = np.linalg.svd(X - mean_X)
        k = np.sum(S >= self.mahalanobis_data.svd_thres)  # detected numerical rank
        self.mahalanobis_data.numerical_rank = k
        self.mahalanobis_data.Vt = Vt[:k]
        self.mahalanobis_data.corpus = corpus
        self.mahalanobis_data.S = S[:k]
        self.mahalanobis_data.mu = mean_X

        # Compute the distribution of Mahalanobis distances of the data in the corpus
        n = len(corpus)
        dists = [[np.inf] * n for i in range(0, n)]
        for i in range(0, n):
            for j in range(i + 1, n):
                dists[i][j] = self._mahalanobis_distance(corpus[i], corpus[j])
        for i in range(0, n):
            for j in range(0, i):
                dists[i][j] = dists[j][i]
        distances = np.min(dists, axis=1)
        self.mahalanobis_data.corpus_distances = distances
        self.thres_distance = np.quantile(distances, self.thres_quantile)

    def _mahalanobis_distance(self, x: ndarray, y: ndarray) -> float:
        """
        Compute Mahalanobis distance between two points x, y,
        Assume x-y-mahalanobis_data.mu in the column subspace of mahalanobis_data.Vt.T
        :param x: ndarray, dimension 1*n
        :param y: ndarray, dimension 1*n
        :return: The mahalanobis distance
        """
        assert (len(x) == len(y))
        assert (x.shape == y.shape)
        assert (len(x.shape) == 1)
        diff = x - y
        return diff @ self.mahalanobis_data.Vt.T @ np.diag(self.mahalanobis_data.S ** (-2)) \
            @ self.mahalanobis_data.Vt @ diff.T
