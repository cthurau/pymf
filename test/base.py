import numpy as np
from numpy.testing import *

def assert_set_equal(m1, m2, decimal=2):
    """
    Tests if two matrices are equal s.t. each element of m1 can be found in 
    m2 and vice versa. An element means a column of m1/m2. Don't use this for
    large matrices as it will be very slow.

    Arguments
    ---------
    m1 - matrix 1
    m2 - matrix 2
    decimal - tolerance for testing for vector equality

    Returns
    -------
    asserts True/False based on the test result
    """
    test1 = np.zeros(m1.shape[1])
    test2 = np.zeros(m2.shape[1])

    for j in range(m1.shape[1]):
        for k in range(m2.shape[1]):
            if np.allclose(m1[:,j], m2[:,k], atol=10**-decimal):
                test1[j] = 1
         
    for j in range(m2.shape[1]):
        for k in range(m1.shape[1]):
            if np.allclose(m2[:,j], m1[:,k], atol=10**-decimal):
                test2[j] = 1

    if np.sum(test1) + np.sum(test2) == m1.shape[1] + m2.shape[1]:
        assert True
    else:
        print "%s not eq %s" %(m1,m2)
        assert False
