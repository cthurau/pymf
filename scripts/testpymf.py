#!/bin/python
##    pymf - Python Matrix Factorization library
##    Copyright (C) 2010 Christian Thurau
##
##    This library is free software; you can redistribute it and/or
##    modify it under the terms of the GNU Library General Public
##    License as published by the Free Software Foundation; either
##    version 2 of the License, or (at your option) any later version.
##
##    This library is distributed in the hope that it will be useful,
##    but WITHOUT ANY WARRANTY; without even the implied warranty of
##    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
##    Library General Public License for more details.
##
##    You should have received a copy of the GNU Library General Public
##    License along with this library; if not, write to the Free
##    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
##
##    Christian Thurau
##    cthurau@gmail.com
"""  

"""

import pymf
import time
import numpy as np
import scipy.sparse
    

def test_svd(A, func, desc, marker):
    stime = time.time()
    m = func(A, rrank=2, crank=2)    
    m.factorize()
    print desc + ': Fro.:', m.frobenius_norm()/(A.shape[0] + A.shape[1]) , ' elapsed:' , time.time() - stime
    return m


def test(A, func, desc, marker, niter=10, num_bases=3):
    stime = time.time()
    m = func(A, num_bases=num_bases)
    m.factorize(show_progress=False, niter=niter)   
    
    print desc + ': Fro.:', m.ferr[-1]/(A.shape[0] + A.shape[1]) , ' elapsed:' , time.time() - stime

    stime = time.time()
    m.factorize(show_progress=False, compute_h=False, niter=niter) 
    m.factorize(show_progress=False, compute_w=False, niter=niter)
    m.factorize(show_progress=False, compute_err=False, niter=niter)
    m.factorize(show_progress=False, niter=20)
    print desc + ' additional tests - elapsed:' , time.time() - stime
    
print "test all methods on boring random data..."
np.random.seed(400401) # VS for repeatability of experiments/tests
A = np.random.random((4,50)) + 2.0
B = scipy.sparse.csc_matrix(A)
# test pseudoinverse
pymf.pinv(A)
pymf.pinv(B)

m = test(A, pymf.SIVM_SEARCH, 'SIVM_SEARCH', 'c<', num_bases=2)
m = test(A, pymf.SIVM_GSAT, 'SIVM_GSAT ', 'c<', niter=20)
m = test(A, pymf.SIVM_SGREEDY, 'SIVM Greedy ', 'c<')
m = test(A, pymf.GMAP, 'GMAP ', 'c<')

svdm = test_svd(A, pymf.SVD, 'Singula Value Decomposition (SVD)', 'c<')
svdm = test_svd(A.T, pymf.SVD, 'Singula Value Decomposition (SVD)', 'c<')
svdm = test_svd(B, pymf.SVD, 'svd sparse', 'c<')
curm = test_svd(A, pymf.CUR, 'CUR Matrix Decomposition', 'b<')
curm = test_svd(B, pymf.CUR, 'CUR Matrix Decomposition (sparse data)', 'b<')
curm = test_svd(A, pymf.CURSL, 'CUR SL Matrix Decomposition', 'b<')
curm = test_svd(B, pymf.CURSL, 'CUR SL Matrix Decomposition (sparse data)', 'b<')
cmdm = test_svd(A, pymf.CMD, 'Compact Matrix Decomposition (CMD)', 'm<')
cmdm = test_svd(B, pymf.CMD, 'Compact Matrix Decomposition (CMD - sparse data)', 'm<')
cmdm = test_svd(A, pymf.GREEDYCUR, 'Greedy CUR (GREEDYCUR)', 'm<')
cmdm = test_svd(B, pymf.GREEDYCUR, 'Greedy CUR (GREEDYCUR - sparse data)', 'm<')
sparse_svmcur = test_svd(A, pymf.SIVM_CUR, 'Simplex Volume Maximization f. CUR (SIVMCUR)', 'm<')
svmcur = test_svd(A, pymf.SIVM_CUR, 'Simplex Volume Maximization f. CUR (SIVMCUR)', 'm<')

m = test(A, pymf.PCA, 'Principal Component Analysis (PCA)', 'c<')
m = test(A, pymf.NMF, 'Non-negative Matrix Factorization (NMF)', 'rs')
m = test(A, pymf.NMFALS, 'NMF u. alternating least squares (NMFALS)', 'rs', niter=10)
m = test(A, pymf.NMFNNLS, 'NMF u. non-neg. least squares (NMFNNLS)', 'rs', niter=10)
m = test(A, pymf.LAESA, 'Linear Approximating Eliminating Search Algorithm (LAESA)', 'rs')
m = test(A, pymf.SIVM, 'Simplex Volume Maximization (SIVM)', 'bs')
m = test(A, pymf.GREEDY, 'Greedy Volume Max. (GREEDY)', 'bs')
m = test(B, pymf.GREEDY, 'Greedy Volume Max. (GREEDY)', 'bs')
m = test(A, pymf.Kmeans, 'K-means clustering (Kmeans)', 'b*')
m = test(A, pymf.Cmeans, 'C-means clustering (Cmeans)', 'b*')
m = test(A, pymf.AA, 'Archetypal Analysis (AA)', 'bs')
m = test(A, pymf.SNMF, 'Semi Non-negative Matrix Factorization (SNMF)', 'bo')
m = test(A, pymf.CNMF, 'Convex non-negative Matrix Factorization (CNMF)', 'c<')
m = test(A, pymf.CHNMF, 'Convex-hull non-negative Matrix Factorization (CHNMF)', 'm*')
m = test(np.round(A-2.0), pymf.BNMF, 'Binary Matrix Factorization (BNMF)', 'b>')
