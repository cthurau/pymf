from pymf.svd import SVD
import numpy as np
from numpy.testing import *

class TestSVD():

    data = np.array([[1.0, 0.2, 1.0], 
                     [0.3, 1.0, 1.0],
                     [0.1, 0.6, 0.4]])

    U = np.array([[-0.64407866, 0.75716718, 0.10890606],
            [-0.69296703, -0.51722384, -0.50227103],
            [-0.32397433, -0.39897036,  0.85782474]])
    S = np.array([[ 1.99141135, 0., 0.],
            [ 0., 0.82984977,  0.],
            [ 0., 0., 0.07503454]])
    V = np.array([[-0.44409017, -0.51027497, -0.7364804],
            [ 0.67735513, -0.72925564, 0.09683102],
            [ 0.58649293, 0.45585707, -0.66949263]])

    def test_compute_wh(self):
        mdl = SVD(self.data)

        # init W to some reasonable values, otherwise the random init can
        # lead to a changed order of basis vectors.
        mdl.factorize()

        np_svdres = np.linalg.svd(self.data)

        print mdl.U
        print mdl.S
        print mdl.V

        # eigenvectors can be inverted, thus, take the absolute values
        assert_almost_equal(np.abs(self.U), np.abs(mdl.U), decimal=2)
        assert_almost_equal(np.abs(self.S), np.abs(mdl.S), decimal=2)
        assert_almost_equal(np.abs(self.V), np.abs(mdl.V), decimal=2)

        # and check if the reconstruction is perfect
        assert_almost_equal(mdl.frobenius_norm(), 0.0, decimal=5)
