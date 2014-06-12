import numpy as np
from numpy.testing import *
from pymf.chnmf import CHNMF
from base import *

class TestCHNMF():

    data = np.array([[1.0, 0.0, 0.0, 0.5], 
                     [0.0, 1.0, 0.0, 0.0]])

    W = np.array([[1.0, 0.0, 0.0], 
                  [0.0, 1.0, 0.0]])

    H = np.array([[1.0, 0.0, 0.0, 0.5], 
                  [0.0, 1.0, 0.0, 0.0], 
                  [0.0, 0.0, 1.0, 0.5]])

    def test_compute_w(self):
        """ Computing W without computing H doesn't make much sense for chnmf..
        """
        mdl = CHNMF(self.data, num_bases=3)
        mdl.H = self.H
        mdl.factorize(niter=10, compute_h=False)
        assert_set_equal(mdl.W, self.W, decimal=2)

    def test_compute_h(self):
        mdl = CHNMF(self.data, num_bases=3)
        mdl.W = self.W
        mdl.factorize(niter=10, compute_w=False)
        assert_set_equal(mdl.H, self.H, decimal=2)
