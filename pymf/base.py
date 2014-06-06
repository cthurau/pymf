# Authors: Christian Thurau
# License: BSD 3 Clause
"""
PyMF base class used in (almost) all matrix factorization methods

"""
import numpy as np
import logging
import logging.config
import scipy.sparse

__all__ = ["PyMFBase"]

class PyMFBase():
    """
    Base Class ...


    """
    
    # some small value
    _EPS = np.finfo(float).eps
    
    def __init__(self, data, num_bases=4, **kwargs):
        
        def setup_logging():
            # create logger       
            self._logger = logging.getLogger("pymf")
       
            # add ch to logger
            if len(self._logger.handlers) < 1:
                # create console handler and set level to debug
                ch = logging.StreamHandler()
                ch.setLevel(logging.DEBUG)
                # create formatter
                formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        
                # add formatter to ch
                ch.setFormatter(formatter)

                self._logger.addHandler(ch)

        setup_logging()
        
        # set variables
        self.data = data       
        self._num_bases = num_bases             
      
        # initialize H and W to random values
        self._data_dimension, self._num_samples = self.data.shape
        

    def residual(self):
        """ Returns the residual in % of the total amount of data """
        res = np.sum(np.abs(self.data - np.dot(self.W, self.H)))
        total = 100.0*res/np.sum(np.abs(self.data))
        return total
        
    def frobenius_norm(self):
        """ Frobenius norm (||data - WH||) of a data matrix and a low rank
        approximation given by WH

        Returns:
            frobenius norm: F = ||data - WH||
        """
        # check if W and H exist
        if hasattr(self,'H') and hasattr(self,'W') and not scipy.sparse.issparse(self.data):
            err = np.sqrt( np.sum((self.data[:,:] - np.dot(self.W, self.H))**2 ))            
        elif hasattr(self,'H') and hasattr(self,'W') and scipy.sparse.issparse(self.data):
            tmp = self.data[:,:] - (self.W * self.H)
            tmp = tmp.multiply(tmp).sum()
            err = np.sqrt(tmp)
        else:
            err = -123456

        return err
        
    def init_w(self):
        self.W = np.random.random((self._data_dimension, self._num_bases)) 
        
    def init_h(self):
        self.H = np.random.random((self._num_bases, self._num_samples)) 
        
    def update_h(self):
        pass

    def update_w(self):
        pass

    def converged(self, i):
        derr = np.abs(self.ferr[i] - self.ferr[i-1])/self._num_samples
        if derr < self._EPS:
            return True
        else:
            return False

    def factorize(self, niter=100, show_progress=False, 
                  compute_w=True, compute_h=True, compute_err=True):
        """ Factorize s.t. WH = data
            
            Parameters
            ----------
            niter : int
                    number of iterations.
            show_progress : bool
                    print some extra information to stdout.
            compute_h : bool
                    iteratively update values for H.
            compute_w : bool
                    iteratively update values for W.
            compute_err : bool
                    compute Frobenius norm |data-WH| after each update and store
                    it to .ferr[k].
            
            Updated Values
            --------------
            .W : updated values for W.
            .H : updated values for H.
            .ferr : Frobenius norm |data-WH| for each iteration.
        """
        
        if show_progress:
            self._logger.setLevel(logging.INFO)
        else:
            self._logger.setLevel(logging.ERROR)        
        
        # create W and H if they don't already exist
        # -> any custom initialization to W,H should be done before
        if not hasattr(self,'W') and compute_w:
               self.init_w()
               
        if not hasattr(self,'H') and compute_h:
                self.init_h()                   

        if compute_err:
            self.ferr = np.zeros(niter)
             
        for i in xrange(niter):
            if compute_w:
                self.update_w()

            if compute_h:
                self.update_h()                                        
         
            if compute_err:                 
                self.ferr[i] = self.frobenius_norm()                
                self._logger.info('Iteration ' + str(i+1) + '/' + str(niter) + 
                ' FN:' + str(self.ferr[i]))
            else:                
                self._logger.info('Iteration ' + str(i+1) + '/' + str(niter))
           

            # check if the err is not changing anymore
            if i > 1 and compute_err:
                if self.converged(i):
                    # adjust the error measure
                    self.ferr = self.ferr[:i]
                    break

def _test():
    import doctest
    doctest.testmod()
 
if __name__ == "__main__":
    _test()
