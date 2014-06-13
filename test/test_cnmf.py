from pymf.cnmf import *
import numpy as np
from numpy.testing import *
from base import *

class TestCNMF():

    data = np.array([[0, 1.0, 0.5], 
                     [1.0, 0, 0.5]])

    W = np.array([[0.0, 0.9], 
                  [0.9, 0.0]])

    H = np.array([[1.0, 0.0],
                [0.0, 1.0],
                [0.5, 0.5]])

    def test_cnmf(self):
        mdl = CNMF(self.data, num_bases=2)

        # nmf forms a cone in the input space, but it is unlikely to hit the
        # cone exactly.
        mdl.factorize(niter=100)
        W = mdl.W/np.sum(mdl.W, axis=0)
        assert_set_equal(W, self.W, decimal=1)

        # since the basis vector will be close to 1/0, the coefficients should
        # converge to the original data.
        H = mdl.H/np.sum(mdl.H, axis=0)
        assert_set_equal(H.T, self.H, decimal=1)

        # the reconstruction quality should still be close to perfect
        rec = mdl.frobenius_norm()
        assert(rec <= 0.1)



