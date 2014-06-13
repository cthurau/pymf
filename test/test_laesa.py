import numpy as np
from numpy.testing import *
from pymf.laesa import LAESA
from base import *

class TestLAESA():

    data = np.array([[1.0, 0.0, 0.0, 0.5], 
                     [0.0, 1.0, 0.0, 0.0]])

    W = np.array([[1.0, 0.0, 0.0], 
                  [0.0, 1.0, 0.0]])

    H = np.array([[1.0, 0.0, 0.0, 0.5], 
                  [0.0, 1.0, 0.0, 0.0], 
                  [0.0, 0.0, 1.0, 0.5]])

    def test_compute_w(self):
        mdl = LAESA(self.data, num_bases=3)
        mdl.H = self.H
        mdl.factorize(niter=10, compute_h=False)
        assert_set_equal(mdl.W, self.W, decimal=2)

    def test_compute_h(self):
        mdl = LAESA(self.data, num_bases=3)
        mdl.W = self.W
        mdl.factorize(niter=10, compute_w=False)
        assert_set_equal(mdl.H, self.H, decimal=2)
