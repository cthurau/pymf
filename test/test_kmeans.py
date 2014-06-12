from pymf.kmeans import Kmeans
import numpy as np
from numpy.testing import *

class TestKMeans():

    data = np.array([[0.2, 0.1, 0.8, 0.9, 0.5], 
                     [0.2, 0.1, 0.8, 0.9, 0.5]])

    W = np.array([[0.15, 0.85, 0.5], 
                  [0.15, 0.85, 0.5]])

    H = np.array([[1.0, 1.0, 0.0, 0.0, 0.0], 
                  [0.0, 0.0, 1.0, 1.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 1.0]])


    def test_compute_w(self):
        mdl = Kmeans(self.data, num_bases=3)
        mdl.H = self.H
        mdl.factorize(niter=10, compute_h=False)
        assert_almost_equal(mdl.W, self.W, decimal=2)

    def test_compute_h(self):
        mdl = Kmeans(self.data, num_bases=3)
        mdl.W = self.W
        mdl.factorize(niter=10, compute_w=False)
        assert_almost_equal(mdl.H, self.H, decimal=2)

    def test_compute_wh(self):
        mdl = Kmeans(self.data, num_bases=3)

        # init W to some reasonable values, otherwise the random init can
        # lead to a changed order of basis vectors.
        mdl.W = np.array([[0.1, 0.7, 0.4],
                          [0.3, 0.6, 0.5]]) 
        mdl.factorize(niter=10)
        assert_almost_equal(mdl.H, self.H, decimal=2)
        assert_almost_equal(mdl.W, self.W, decimal=2)
