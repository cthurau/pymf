from pymf.nmf import *
import numpy as np
from numpy.testing import *
from base import *

class TestNMF():

    data = np.array([[0.1, 0.1, 0.8, 0.4, 0.5, 1.0, 0.0], 
                     [0.5, 0.3, 0.4, 0.1, 0.5, 0.0, 1.0]])

    W = np.array([[1.0, 0.0], 
                  [0.0, 1.0]])

    def test_nmf(self):
        mdl = NMF(self.data, num_bases=2)

        # nmf forms a cone in the input space, but it is unlikely to hit the
        # cone exactly.
        mdl.factorize(niter=50)
        assert_set_equal(mdl.W, self.W, decimal=1)

        # since the basis vector will be close to 1/0, the coefficients should
        # converge to the original data.
        assert_set_equal(mdl.H.T, self.data.T, decimal=1)

        # the reconstruction quality should still be close to perfect
        rec = mdl.frobenius_norm()
        assert_almost_equal(0.0, rec, decimal=1)

    def test_rnmf(self):
        mdl = RNMF(self.data, num_bases=2)

        # nmf forms a cone in the input space, but it is unlikely to hit the
        # cone exactly.
        mdl.factorize(niter=50)
        assert_set_equal(mdl.W.T/np.sum(mdl.W, axis=1), self.W, decimal=1)

        # the reconstruction quality should still be close to perfect
        rec = mdl.frobenius_norm()
        assert_almost_equal(0.0, rec, decimal=1)

    def test_nmfals(self):
        # (todo) based on the initialization this test can fail
        mdl = NMFALS(self.data, num_bases=2)

        # nmf forms a cone in the input space, but it is unlikely to hit the
        # cone exactly.
        mdl.factorize(niter=50)
        assert_set_equal(mdl.W/np.sum(mdl.W, axis=0), self.W, decimal=1)

        # the reconstruction quality should still be close to perfect
        rec = mdl.frobenius_norm()
        assert_almost_equal(0.0, rec, decimal=1)


    def test_nmfnnls(self):
        mdl = NMFNNLS(self.data, num_bases=2)

        # nmf forms a cone in the input space, but it is unlikely to hit the
        # cone exactly.
        mdl.factorize(niter=100)
        assert_set_equal(mdl.W/np.sum(mdl.W, axis=0), self.W, decimal=1)

        # the reconstruction quality should still be close to perfect
        rec = mdl.frobenius_norm()
        assert_almost_equal(0.0, rec, decimal=6)

