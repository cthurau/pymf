# Authors: Christian Thurau
# License: BSD 3 Clause
"""
PyMF functions for computing matrix/simplex volumes

    cmdet(): Cayley-Menger determinant
    simplex_volume(): Ordinary simplex volume
        
"""
import numpy as np
from scipy.misc import factorial

__all__ = ["cmdet", "simplex"]

def cmdet(d):
    """ Returns the Volume of a simplex computed via the Cayley-Menger
    determinant.

    Arguments
    ---------
    d - euclidean distance matrix (shouldn't be squared)

    Returns
    -------
    V - volume of the simplex given by d
    """
    D = np.ones((d.shape[0]+1,d.shape[0]+1))
    D[0,0] = 0.0
    D[1:,1:] = d**2
    j = np.float32(D.shape[0]-2)
    f1 = (-1.0)**(j+1) / ( (2**j) * ((factorial(j))**2))
    cmd = f1 * np.linalg.det(D)

    # sometimes, for very small values, "cmd" might be negative, thus we take
    # the absolute value
    return np.sqrt(np.abs(cmd))

def simplex(d):
    """ Computed the volume of a simplex S given by a coordinate matrix D.

    Arguments
    ---------
    d - coordinate matrix (k x n, n samples in k dimensions)

    Returns
    -------
    V - volume of the Simplex spanned by d
    """
    # compute the simplex volume using coordinates
    D = np.ones((d.shape[0]+1, d.shape[1]))
    D[1:,:] = d
    V = np.abs(np.linalg.det(D)) / factorial(d.shape[1] - 1)
    return V

def _test():
    import doctest
    doctest.testmod()
 
if __name__ == "__main__":
    _test()
