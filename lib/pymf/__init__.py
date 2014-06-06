#!/usr/bin/python
#
# Copyright (C) Christian Thurau, 2010. 
# Licensed under the GNU General Public License (GPL). 
# http://www.gnu.org/licenses/gpl.txt
'''pymf is a package for several Matrix Factorization variants.-
Detailed documentation is available at http://pymf.googlecode.com
Copyright (C) Christian Thurau, 2010. GNU General Public License (GPL)
'''

__version__ = "$Revision: 64 $"
# $HeadURL: http://pymf.googlecode.com/svn/trunk/lib/pymf/__init__.py $

# Non-negative matrix factorization (using the standard multiplicative update rule)
from .nmf import *
from .rnmf import *
#from .nmfimsparse import *
# NMF with an alternating least squares optimization
from .nmfals import *
# NMF with non-negative least squares optimization
from .nmfnnls import *
# Convex-NMF
from .cnmf import *
# Convex-hull-NMF
from .chnmf import *
# Semi-NMF
from .snmf import *
# Archetypal Analysis
from .aa import *
# LAESA algorithm
from .laesa import *
# Binary matrix fcatorization
from .bnmf import *
# Singular value decomposition
from .svd import *
# Non-negative Double Singular Value Decompositions
from .nndsvd import *
# Principal component analysis
from .pca import *
# CUR decomposition (using norm-based sampling)
from .cur import *
# CUR decomposition (using statistical leverage based sampling)
from .cursl import *
# Compact-Matrix-Decomposition (slightly enhanced CUR)
from .cmd import *
# K-means clustering
from .kmeans import *
# C-means clustering
from .cmeans import *
# Greedy-volume maximization 
from .greedy import *
# Greedy used for CUR-like decompositions
from .greedycur import *
# Simplex-volume maximization (SIVM)
from .sivm import *
# SIVM using greedy-search
from .sivm_sgreedy import *
# SIVM using basic a-star like search
from .sivm_search import *
# SIVM using random gsat selection
from .sivm_gsat import *
# SIVM-CUR decomposition
from .sivm_cur import *
# Geometric volume maximization
from .gmap import *

# Simple module for incorporating subsampling ....
from .sub import *
