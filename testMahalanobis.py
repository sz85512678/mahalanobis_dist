import unittest

import numpy as np

from RNearestNeighbour import RNearestNeighbour


class MyTestCase(unittest.TestCase):
    def test_mahalanobis(self):
        model = RNearestNeighbour(0.1, "Mahalanobis")
        corpus = np.array([[1, 2, 3, 4, 5], [4, 5, 6, 7, -1], [3, 2, 4, 5, 10]])
        model.fit(corpus)
        y0 = np.array([1, 2, 3, 4, 5])
        # y = X_c(1,:) - 0.5*X_c(2,:) + mu; y1 = y - X(1,:);
        y1 = np.array([0.3333,    1.0000,    2.1667,    3.1667,    7.8333])
        dist = model._mahalanobis_distance(y1, y0)
        self.assertAlmostEqual(dist, 0.1667, 3)
        self.assertFalse(model.predict(y1)) # Not an outlier, corpus distance is 2,2,2

        y2 = np.array([0.7398  , -0.4918 ,  -0.2999 ,  -0.2999  , -0.1759]) # orthogonal to the subspace
        dist1 = model._mahalanobis_distance(y2 + y1, y0)
        self.assertAlmostEqual(dist1, 0.1667, 3)
        # But it is an outlier, so the above number does not mean anything as it's not in the corpus subspace
        self.assertTrue(model.predict(y2+y1))

    def test_invariance(self):
        """
        Mahalanobis distance is invariant under invertible linear transformation of the features
        """
        model = RNearestNeighbour(0.1, "Mahalanobis")
        corpus = np.array([[1, 2, 3, 4, 5], [4, 5, 6, 7, -1], [3, 2, 4, 5, 10]])
        model.fit(corpus)
        y0 = np.array([1, 2, 3, 4, 5])
        y1 = np.array([0.3333, 1.0000, 2.1667, 3.1667, 7.8333])
        dist = model._mahalanobis_distance(y1, y0)

        L = np.random.randn(5, 5)
        c = np.random.randn(5, )
        while np.linalg.det(L)<1e-3:
            L = np.random.randn(5, 5)
        model.fit(corpus@L + c)
        y0 = y0@L + c; y1 = y1@L + c
        dist1 = model._mahalanobis_distance(y1, y0)
        self.assertAlmostEqual(dist, dist1, 3)

if __name__ == '__main__':
    unittest.main()
