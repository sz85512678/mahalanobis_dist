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
    transformed_data = None  # Transformed data so that Euclidean distances will be used


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
        self.thres_distance: float = np.inf
        self.mahalanobis_data = MahalanobisData()
        self.corpus_mean = None

    def fit(self, corpus) -> None:
        """
        :param corpus: numpy array of numpy array, corpus[i] is the feature of the ith data point
        :return:
        """
        self.corpus_mean = np.mean(corpus, axis=0)
        corpus = corpus - self.corpus_mean
        if self.distance == "Euclidean":
            self.estimator = NearestNeighbors(n_neighbors=1)
            self.estimator.fit(corpus)
            dist, ind = self.estimator.kneighbors()
            self.thres_distance = np.quantile(dist, self.thres_quantile)
        elif self.distance == "Mahalanobis":
            self.estimator = NearestNeighbors(n_neighbors=1)
            self._mahalanobis(corpus)
            self.estimator.fit(self.mahalanobis_data.transformed_data)
            dist, ind = self.estimator.kneighbors()
            self.thres_distance = np.quantile(dist, self.thres_quantile)
        else:
            raise NotImplementedError

    def compute_distances(self, x) -> ndarray:
        """
        Compute the distances to the corpus
        :param x:
        :return:
        """
        x = x - self.corpus_mean
        if self.distance == "Euclidean":
            if len(x.shape) == 1:
                return self.estimator.kneighbors(np.reshape(x, (1, -1)))[0]
            else:
                return self.estimator.kneighbors(x)[0]
        elif self.distance == "Mahalanobis":
            rho = np.linalg.norm(x - x @ self.mahalanobis_data.Vt.T @ self.mahalanobis_data.Vt, axis=1) \
                  / np.linalg.norm(x, axis=1)
            x = x @ self.mahalanobis_data.Vt.T @ np.diag(self.mahalanobis_data.S ** (-1))
            if len(x.shape) == 1:
                if rho > self.mahalanobis_data.subspace_thres:
                    return np.array([np.inf])
                else:
                    return self.estimator.kneighbors(np.reshape(x, (1, -1)))[0]
            else:
                return np.where(rho > self.mahalanobis_data.subspace_thres,
                                np.inf,
                                (self.estimator.kneighbors(x)[0])[:, 0])
        else:
            raise NotImplementedError

    def predict(self, x) -> ndarray:
        """
        :param x: data point
        :return: True if x is classified as an outlier, False if inlier
        """
        return self.compute_distances(x) >= self.thres_distance

    def _mahalanobis(self, corpus) -> None:
        """
        Prepare to use nearest neighbours with Mahalanobis distance as a classifier
        :return:
        """
        X = corpus
        U, S, Vt = np.linalg.svd(X)
        k = np.sum(S >= self.mahalanobis_data.svd_thres)  # detected numerical rank
        self.mahalanobis_data.numerical_rank = k
        self.mahalanobis_data.Vt = Vt[:k]
        self.mahalanobis_data.corpus = corpus
        self.mahalanobis_data.S = S[:k]
        self.mahalanobis_data.mu = self.corpus_mean
        self.mahalanobis_data.transformed_data = (
                X @
                self.mahalanobis_data.Vt.T @
                np.diag(self.mahalanobis_data.S ** (-1))
        )

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
