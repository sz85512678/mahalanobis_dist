import unittest

import numpy as np

from RNearestNeighbour import RNearestNeighbour


class MyTestCase(unittest.TestCase):
    def test_mahalanobis(self):
        model = RNearestNeighbour(0.1, "Mahalanobis")
        corpus = np.array([[1, 2, 3, 4, 5], [4, 5, 6, 7, -1], [3, 2, 4, 5, 10]])
        model.fit(corpus)
        y0 = np.array([0,0,0,0,0])
        y1 = np.array([-1.0000,-0.5000,0,0.5000,5.5000]) # In the subspace, linear combo of first and second row of corpus
        dist = model._mahalanobis_distance(y1, y0)
        self.assertAlmostEqual(dist, 0.6180, 3)  # add assertion here

        y2 = np.array([0.7398  , -0.4918 ,  -0.2999 ,  -0.2999  , -0.1759]) # orthogonal to the subspace
        dist1 = model._mahalanobis_distance(y2 + y1, y0)
        self.assertAlmostEqual(dist1, 0.6180, 3)
        # But it is an outlier, so the above number does not mean anything as it's not in the corpus subspace
        self.assertTrue(model.predict(y2+y1))

if __name__ == '__main__':
    unittest.main()
