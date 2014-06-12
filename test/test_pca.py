from pymf.pca import PCA
import numpy as np
from numpy.testing import *

class TestPCA():

    data = np.array([[1.0, 0.2, 1.0, 0.4], 
                     [0.3, 1.0, 1.0, 0.5],
                     [0.1, 0.6, 0.4, 0.5]])

    W = np.array([[0.76181603,  0.5583999,   0.32836851],
            [-0.48471704,  0.82765541, -0.2829062 ],
            [-0.42975077,  0.05635667,  0.90118711]])

    H = np.array([[0.58944766, -0.57418248, 0.1212205, -0.13648568],
            [-0.1525292, 0.008288, 0.44373659, -0.29949539],
            [-0.04226467, -0.05240027, 0.03005712,  0.06460782]])

    def test_compute_wh(self):
        
        mdl = PCA(self.data, num_bases=3)

        # init W to some reasonable values, otherwise the random init can
        # lead to a changed order of basis vectors.
        mdl.factorize(niter=10)

        # eigenvectors and coefficients should only differ in signs
        assert_almost_equal(np.abs(mdl.H), np.abs(self.H), decimal=4)
        assert_almost_equal(np.abs(mdl.W), np.abs(self.W), decimal=4)

        # reconsruction should be perfect
        rec = np.dot(TestPCA.W, TestPCA.H) + np.mean(self.data, axis=1).reshape(-1,1)
        assert_almost_equal(rec, self.data, decimal=5)
