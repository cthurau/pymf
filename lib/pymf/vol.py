#!/usr/bin/python
#
# Copyright (C) Christian Thurau, 2010. 
# Licensed under the GNU General Public License (GPL). 
# http://www.gnu.org/licenses/gpl.txt
#$Id: vol.py 62 2011-07-25 15:28:07Z cthurau $
"""
PyMF functions for computing matrix/simplex volumes

    cmdet(): Cayley-Menger determinant
    simplex_volume(): Ordinary simplex volume
        
"""

__version__ = "$Revision: 62 $"
# $HeadURL: http://pymf.googlecode.com/svn/trunk/lib/pymf/vol.py $

import numpy as np
from scipy import factorial

__all__ = ["cmdet", "simplex"]

def cmdet(d):
    # compute the CMD determinant of the euclidean distance matrix d
    # -> d should not be squared!
    D = np.ones((d.shape[0]+1,d.shape[0]+1))
    D[0,0] = 0.0
    D[1:,1:] = d**2
    j = np.float32(D.shape[0]-2)
    f1 = (-1.0)**(j+1) / ( (2**j) * ((factorial(j))**2))
    cmd = f1 * np.linalg.det(D)
    # sometimes, for very small values "cmd" might be negative ...
    return np.sqrt(np.abs(cmd))

def simplex(d):
    # compute the simplex volume using coordinates
    D = np.ones((d.shape[0]+1, d.shape[1]))
    D[1:,:] = d
    vol = np.abs(np.linalg.det(D)) / factorial(d.shape[1] - 1)
    return vol
