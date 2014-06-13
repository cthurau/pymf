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
import scipy.optimize
from cvxopt import solvers, base
from base import PyMFBase
from svd import pinv

__all__ = ["NMF", "NMFALS", "NMFNNLS"]


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
       
    def _update_h(self):
        # pre init H1, and H2 (necessary for storing matrices on disk)
        H2 = np.dot(np.dot(self.W.T, self.W), self.H) + 10**-9
        self.H *= np.dot(self.W.T, self.data[:,:])
        self.H /= H2

    def _update_w(self):
        # pre init W1, and W2 (necessary for storing matrices on disk)
        W2 = np.dot(np.dot(self.W, self.H), self.H.T) + 10**-9
        self.W *= np.dot(self.data[:,:], self.H.T)
        self.W /= W2
        self.W /= np.sqrt(np.sum(self.W**2.0, axis=0))


class NMFALS(PyMFBase):
    """      
    NMFALS(data, num_bases=4)
    
    
    Non-negative Matrix Factorization. Factorize a data matrix into two matrices 
    s.t. F = | data - W*H | = | is minimal. H, and W are restricted to non-negative
    data. Uses the an alternating least squares procedure (quite slow for larger
    data sets) and cvxopt, similar to aa.
    
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
    >>> nmf_mdl = NMFALS(data, num_bases=2)
    >>> nmf_mdl.factorize(niter=10)
    
    The basis vectors are now stored in nmf_mdl.W, the coefficients in nmf_mdl.H. 
    To compute coefficients for an existing set of basis vectors simply    copy W 
    to nmf_mdl.W, and set compute_w to False:
    
    >>> data = np.array([[1.5], [1.2]])
    >>> W = np.array([[1.0, 0.0], [0.0, 1.0]])
    >>> nmf_mdl = NMFALS(data, num_bases=2)
    >>> nmf_mdl.W = W
    >>> nmf_mdl.factorize(niter=1, compute_w=False)
    
    The result is a set of coefficients nmf_mdl.H, s.t. data = W * nmf_mdl.H.
    """
 
    def _update_h(self):
        def updatesingleH(i):
            # optimize alpha using qp solver from cvxopt
            FA = base.matrix(np.float64(np.dot(-self.W.T, self.data[:,i])))
            al = solvers.qp(HA, FA, INQa, INQb)
            self.H[:,i] = np.array(al['x']).reshape((1,-1))
                                                                
        # float64 required for cvxopt
        HA = base.matrix(np.float64(np.dot(self.W.T, self.W)))            
        INQa = base.matrix(-np.eye(self._num_bases))
        INQb = base.matrix(0.0, (self._num_bases,1))            
    
        map(updatesingleH, xrange(self._num_samples))                        
            
                
    def _update_w(self):
        def updatesingleW(i):
        # optimize alpha using qp solver from cvxopt
            FA = base.matrix(np.float64(np.dot(-self.H, self.data[i,:].T)))
            al = solvers.qp(HA, FA, INQa, INQb)                
            self.W[i,:] = np.array(al['x']).reshape((1,-1))            
                                
        # float64 required for cvxopt
        HA = base.matrix(np.float64(np.dot(self.H, self.H.T)))                    
        INQa = base.matrix(-np.eye(self._num_bases))
        INQb = base.matrix(0.0, (self._num_bases,1))            

        map(updatesingleW, xrange(self._data_dimension))

        self.W = self.W/np.sum(self.W, axis=1)


class NMFNNLS(PyMFBase):
    """      
    NMFNNLS(data, num_bases=4)
    
    
    Non-negative Matrix Factorization. Factorize a data matrix into two matrices 
    s.t. F = | data - W*H | = | is minimal. H, and W are restricted to non-negative
    data. Uses the Lawsons and Hanson's algorithm for non negative constrained
    least squares (-> also see scipy.optimize.nnls)
    
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
    >>> nmf_mdl = NMFNNLS(data, num_bases=2)
    >>> nmf_mdl.factorize(niter=10)
    
    The basis vectors are now stored in nmf_mdl.W, the coefficients in nmf_mdl.H. 
    To compute coefficients for an existing set of basis vectors simply copy W 
    to nmf_mdl.W, and set compute_w to False:
    
    >>> data = np.array([[1.5], [1.2]])
    >>> W = np.array([[1.0, 0.0], [0.0, 1.0]])
    >>> nmf_mdl = NMFNNLS(data, num_bases=2)
    >>> nmf_mdl.W = W
    >>> nmf_mdl.factorize(niter=1, compute_w=False)
    
    The result is a set of coefficients nmf_mdl.H, s.t. data = W * nmf_mdl.H.
    """

    def _update_h(self):
        def updatesingleH(i):        
            self.H[:,i] = scipy.optimize.nnls(self.W, self.data[:,i])[0]
                                                                            
        map(updatesingleH, xrange(self._num_samples))                        
            
                
    def _update_w(self):
        def updatesingleW(i):            
            self.W[i,:] = scipy.optimize.nnls(self.H.T, self.data[i,:].T)[0]

        map(updatesingleW, xrange(self._data_dimension))


def _test():
    import doctest
    doctest.testmod()
 
if __name__ == "__main__":
    _test()
