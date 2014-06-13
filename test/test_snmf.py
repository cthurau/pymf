from pymf.snmf import *
import numpy as np
from numpy.testing import *
from base import *

class TestNMF():

    data = np.array([[1.0, 0.0, 0.2], 
                     [0.0, -1.0, 0.3]])

    def test_snmf(self):
        mdl = SNMF(self.data, num_bases=2)

        # nmf forms a cone in the input space, but it is unlikely to hit the
        # cone exactly.
        mdl.factorize(niter=1000)

        # the reconstruction quality should be close to perfect
        rec = mdl.frobenius_norm()
        assert_almost_equal(0.0, rec, decimal=1)

        # and H is not allowed to have <0 values
        l = np.where(mdl.H < 0)[0]
        assert(len(l) == 0)
