# Authors: Christian Thurau
# License: BSD 3 Clause
"""
PyMF Non-negative Matrix Factorization.

    NMF: Class for Non-negative Matrix Factorization

[1] Lee, D. D. and Seung, H. S. (1999), Learning the Parts of Objects by Non-negative
Matrix Factorization, Nature 401(6755), 788-799.
"""
import numpy as np
import logging
import logging.config
import scipy.sparse
from base import PyMFBase

__all__ = ["NMF"]

class NMF(PyMFBase):
    """
    NMF(data, num_bases=4)

    Non-negative Matrix Factorization. Factorize a data matrix into two matrices
    s.t. F = | data - W*H | = | is minimal. H, and W are restricted to non-negative
    data. Uses the classicial multiplicative update rule.

    Parameters
    ----------
    data : array_like, shape (_data_dimension, _num_samples)
        the input data
    num_bases: int, optional
        Number of bases to compute (column rank of W and row rank of H).
        4 (default)        

    Attributes
    ----------
    W : "data_dimension x num_bases" matrix of basis vectors
    H : "num bases x num_samples" matrix of coefficients
    ferr : frobenius norm (after calling .factorize()) 

    Example
    -------
    Applying NMF to some rather stupid data set:

    >>> import numpy as np
    >>> data = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]])
    >>> nmf_mdl = NMF(data, num_bases=2, niter=10)
    >>> nmf_mdl.factorize()

    The basis vectors are now stored in nmf_mdl.W, the coefficients in nmf_mdl.H.
    To compute coefficients for an existing set of basis vectors simply    copy W
    to nmf_mdl.W, and set compute_w to False:

    >>> data = np.array([[1.5], [1.2]])
    >>> W = np.array([[1.0, 0.0], [0.0, 1.0]])
    >>> nmf_mdl = NMF(data, num_bases=2)
    >>> nmf_mdl.W = W
    >>> nmf_mdl.factorize(niter=20, compute_w=False)

    The result is a set of coefficients nmf_mdl.H, s.t. data = W * nmf_mdl.H.
    """
       
    def update_h(self):
            # pre init H1, and H2 (necessary for storing matrices on disk)
            H2 = np.dot(np.dot(self.W.T, self.W), self.H) + 10**-9
            self.H *= np.dot(self.W.T, self.data[:,:])
            self.H /= H2

    def update_w(self):
            # pre init W1, and W2 (necessary for storing matrices on disk)
            W2 = np.dot(np.dot(self.W, self.H), self.H.T) + 10**-9
            self.W *= np.dot(self.data[:,:], self.H.T)
            self.W /= W2
            self.W /= np.sqrt(np.sum(self.W**2.0, axis=0))

    def converged(self, i):
        derr = np.abs(self.ferr[i] - self.ferr[i-1])/self._num_samples
        if derr < self._EPS:
            return True
        else:
            return False

def _test():
    import doctest
    doctest.testmod()
 
if __name__ == "__main__":
    _test()
