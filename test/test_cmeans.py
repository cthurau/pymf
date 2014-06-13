from pymf.cmeans import Cmeans
import numpy as np
from numpy.testing import *

class TestCMeans():

    data = np.array([[0.5, 0.1, 0.9], 
                     [0.5, 0.9, 0.1]])

    W = np.array([[0.757, 0.242], 
                  [0.242, 0.757]])

    H = np.array([[0.5, 0.02, 0.98], 
                  [0.5, 0.98, 0.02]])

    def test_compute_w(self):
        mdl = Cmeans(self.data, num_bases=2)
        mdl.H = self.H
        mdl.factorize(niter=10, compute_h=False)
        assert_almost_equal(mdl.W, self.W, decimal=2)

    def test_compute_h(self):
        mdl = Cmeans(self.data, num_bases=2)
        mdl.W = self.W
        mdl.factorize(niter=10, compute_w=False)
        assert_almost_equal(mdl.H, self.H, decimal=2)
