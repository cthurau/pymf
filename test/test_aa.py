import pymf.aa
import numpy as np
from numpy.testing import *

class TestAA():

    data = np.array([[1.0, 0.0, 0.0, 0.5], 
                     [0.0, 1.0, 0.0, 0.0]])

    W = np.array([[1.0, 0.0, 0.0], 
                  [0.0, 1.0, 0.0]])

    H = np.array([[1.0, 0.0, 0.0, 0.5], 
                  [0.0, 1.0, 0.0, 0.0], 
                  [0.0, 0.0, 1.0, 0.5]])

    def test_compute_w(self):
        aa_mdl = pymf.aa.AA(self.data, num_bases=3)
        aa_mdl.H = self.H
        aa_mdl.factorize(niter=10, compute_h=False)
        assert_almost_equal(aa_mdl.W, self.W, decimal=2)

    def test_compute_h(self):
        aa_mdl = pymf.aa.AA(self.data, num_bases=3)
        aa_mdl.W = self.W
        aa_mdl.factorize(niter=10, compute_w=False)
        assert_almost_equal(aa_mdl.H, self.H, decimal=2)
